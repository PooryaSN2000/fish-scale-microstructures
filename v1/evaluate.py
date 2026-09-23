import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torchvision.transforms import v2

import config
from dataset import FishScaleDataset, EpisodicBatchSampler
from model import ProtoNet, compute_prototypes_and_logits, compute_protonet_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_gpu_val_transforms():
    return v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

@torch.no_grad()
def evaluate_test(model, dataloader, device, class_names):
    model.eval()
    gpu_transforms = get_gpu_val_transforms().to(device)
    
    episodic_accuracies = []
    all_global_targets = []
    all_global_preds = []
    
    tsne_embeddings = []
    tsne_targets = []
    tsne_prototypes = []
    
    # For prediction confidence grid
    grid_images = []
    grid_preds = []
    grid_targets = []
    grid_confs = []
    
    print("Running test episodes...")
    for batch_idx, batch in enumerate(dataloader):
        if config.USE_CLASSICAL:
            images, labels, classical_feats = batch
            images = images.to(device)
            # Save some un-normalized images for the grid
            if batch_idx == 0:
                grid_images_raw = images.cpu().clone()
            
            images = gpu_transforms(images)
            classical_feats = classical_feats.to(device)
            embeddings = model(images, classical_feats)
        else:
            images, labels = batch
            images = images.to(device)
            if batch_idx == 0:
                grid_images_raw = images.cpu().clone()
            images = gpu_transforms(images)
            embeddings = model(images)
            
        labels_reshaped = labels.view(config.N_WAY, config.K_SHOT + config.Q_QUERY)
        local_to_global = labels_reshaped[:, 0].cpu().numpy()
        
        logits, targets = compute_prototypes_and_logits(
            embeddings,
            n_way=config.N_WAY,
            k_shot=config.K_SHOT,
            q_query=config.Q_QUERY,
            temperature=model.temperature
        )
        
        # Softmax probabilities for confidence
        probs = F.softmax(logits, dim=-1)
        confidences, preds = torch.max(probs, dim=-1)
        
        acc = (preds == targets).float().mean().item()
        episodic_accuracies.append(acc)
        
        targets_np = targets.cpu().numpy()
        preds_np = preds.cpu().numpy()
        confs_np = confidences.cpu().numpy()
        
        for t, p in zip(targets_np, preds_np):
            all_global_targets.append(local_to_global[t])
            all_global_preds.append(local_to_global[p])
            
        # Collect data for visualizations from the first episode
        if batch_idx == 0:
            D = embeddings.size(-1)
            embs_reshaped = embeddings.view(config.N_WAY, config.K_SHOT + config.Q_QUERY, D)
            support_set = embs_reshaped[:, :config.K_SHOT, :]
            query_set = embs_reshaped[:, config.K_SHOT:, :]
            
            prototypes = support_set.mean(dim=1).cpu().numpy()
            queries = query_set.reshape(config.N_WAY * config.Q_QUERY, D).cpu().numpy()
            
            tsne_embeddings.extend(queries)
            tsne_targets.extend(targets_np)
            tsne_prototypes.extend(prototypes)
            
            # Save top 16 query images for confidence grid
            # Query images start after support images in the flat batch
            query_start_idx = config.N_WAY * config.K_SHOT
            for i in range(min(16, config.N_WAY * config.Q_QUERY)):
                # Flattened index of query image
                # The batch is sorted by class: C0_S0..C0_Sk, C0_Q0..C0_Qq, C1_S0...
                # Actually, the batch from sampler is exactly that.
                # To easily map, we just take the first few from the flat queries
                c_idx = i // config.Q_QUERY
                q_idx = i % config.Q_QUERY
                flat_idx = c_idx * (config.K_SHOT + config.Q_QUERY) + config.K_SHOT + q_idx
                
                grid_images.append(grid_images_raw[flat_idx])
                grid_preds.append(local_to_global[preds_np[i]])
                grid_targets.append(local_to_global[targets_np[i]])
                grid_confs.append(confs_np[i])
            
    avg_acc = np.mean(episodic_accuracies)
    std_acc = np.std(episodic_accuracies)
    ci95 = 1.96 * (std_acc / np.sqrt(len(episodic_accuracies)))
    
    f1_macro = f1_score(all_global_targets, all_global_preds, average="macro")
    f1_weighted = f1_score(all_global_targets, all_global_preds, average="weighted")
    
    print("\n" + "="*40)
    print("EVALUATION RESULTS ON TEST SPLIT")
    print("="*40)
    print(f"Test Accuracy: {avg_acc:.4f} ± {ci95:.4f} (95% CI)")
    print(f"Macro F1-Score: {f1_macro:.4f}")
    print(f"Weighted F1-Score: {f1_weighted:.4f}")
    print("="*40)
    
    unique_labels = sorted(list(set(all_global_targets)))
    target_names = [class_names[l] for l in unique_labels]
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_global_targets, all_global_preds, target_names=target_names))
    
    # Generates plots
    plot_confusion_matrix(all_global_targets, all_global_preds, class_names)
    plot_class_performance(all_global_targets, all_global_preds, class_names)
    
    if len(grid_images) > 0:
        plot_prediction_confidence_grid(grid_images, grid_targets, grid_preds, grid_confs, class_names)
    
    if len(tsne_embeddings) > 0:
        plot_tsne(
            np.array(tsne_embeddings),
            np.array(tsne_targets),
            np.array(tsne_prototypes),
            [class_names[local_to_global[i]] for i in range(config.N_WAY)]
        )


def plot_class_performance(targets, preds, class_names):
    """
    Bar chart of F1-scores per class.
    """
    unique_labels = sorted(list(set(targets)))
    labels_text = [class_names[l] for l in unique_labels]
    
    f1_scores = f1_score(targets, preds, average=None, labels=unique_labels)
    
    plt.figure(figsize=(20, 8))
    bars = plt.bar(labels_text, f1_scores, color='#3498db')
    
    plt.axhline(y=np.mean(f1_scores), color='r', linestyle='--', label=f'Mean F1 ({np.mean(f1_scores):.2f})')
    
    plt.title("Class-wise F1-Score Performance", fontsize=16, fontweight='bold')
    plt.xlabel("Fish Species", fontsize=10)
    plt.ylabel("F1 Score", fontsize=10)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.legend()
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=6, rotation=90)
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, "class_performance.png"), dpi=150)
    plt.close()
    print(f"Saved class performance chart to: {os.path.join(config.CHECKPOINT_DIR, 'class_performance.png')}")


def plot_prediction_confidence_grid(images, targets, preds, confs, class_names):
    """
    Plots a grid of images with predicted label, true label, and confidence.
    """
    n_images = len(images)
    cols = 4
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i in range(n_images):
        ax = axes[i]
        img = images[i].numpy().transpose(1, 2, 0) # C, H, W -> H, W, C
        ax.imshow(img)
        ax.axis('off')
        
        true_name = class_names[targets[i]]
        pred_name = class_names[preds[i]]
        conf = confs[i] * 100
        
        color = 'green' if true_name == pred_name else 'red'
        title = f"True: {true_name}\nPred: {pred_name}\nConf: {conf:.1f}%"
        ax.set_title(title, color=color, fontsize=10)
        
    # Hide empty subplots
    for j in range(n_images, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, "prediction_confidence.png"), dpi=150)
    plt.close()
    print(f"Saved prediction confidence grid to: {os.path.join(config.CHECKPOINT_DIR, 'prediction_confidence.png')}")


def plot_confusion_matrix(targets, preds, class_names):
    cm = confusion_matrix(targets, preds)
    unique_labels = sorted(list(set(targets)))
    labels_text = [class_names[l] for l in unique_labels]
    
    fig, ax = plt.subplots(figsize=(24, 22))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)
    
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels_text,
        yticklabels=labels_text,
        title="Normalized Confusion Matrix (Test Split)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm_normalized[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_normalized[i, j] > thresh else "black",
                fontsize=6
            )
            
    fig.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_tsne(query_embeddings, query_targets, prototypes, class_names):
    n_queries = len(query_embeddings)
    n_protos = len(prototypes)
    
    combined = np.concatenate([query_embeddings, prototypes], axis=0)
    
    perp = min(30, len(combined) - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=config.RANDOM_SEED)
    projected = tsne.fit_transform(combined)
    
    proj_queries = projected[:n_queries]
    proj_protos = projected[n_queries:]
    
    plt.figure(figsize=(14, 12))
    colors = plt.get_cmap("tab20")
    
    for c_idx in range(len(class_names)):
        indices = np.where(query_targets == c_idx)[0]
        plt.scatter(
            proj_queries[indices, 0],
            proj_queries[indices, 1],
            color=colors(c_idx),
            label=f"{class_names[c_idx]} (Query)",
            alpha=0.6,
            edgecolors='k',
            s=60
        )
        
    for c_idx in range(len(class_names)):
        plt.scatter(
            proj_protos[c_idx, 0],
            proj_protos[c_idx, 1],
            color=colors(c_idx),
            marker='X',
            s=300,
            edgecolors='black',
            linewidths=1.5,
            label=f"{class_names[c_idx]} (Proto)"
        )
        
    plt.title("t-SNE Visualization of Feature Space (Cosine / Metric Distance)", fontsize=14, fontweight='bold')
    # Use a smaller font size and multiple columns for the legend if many classes
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=7)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, "tsne_embeddings.png"), dpi=150)
    plt.close()
    print(f"Saved t-SNE visualization to: {os.path.join(config.CHECKPOINT_DIR, 'tsne_embeddings.png')}")


def main():
    set_seed(config.RANDOM_SEED)
    print(f"Using Device: {config.DEVICE}")
    
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Model checkpoint not found at: {checkpoint_path}")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    print("Loading test dataset...")
    test_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="test", seed=config.RANDOM_SEED)
    
    # Optionally print dataset distribution
    print(f"\nTest Split contains {len(test_dataset.classes)} classes and {len(test_dataset)} images.")
    
    test_sampler = EpisodicBatchSampler(
        labels=test_dataset.labels,
        n_way=config.N_WAY,
        k_shot=config.K_SHOT,
        q_query=config.Q_QUERY,
        num_episodes=config.TEST_EPISODES
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=0,
        pin_memory=True
    )
    
    model_cfg = checkpoint["config"]
    # Handle backward compatibility for newer config additions
    backbone = model_cfg.get("backbone", "resnet18")
    
    model = ProtoNet(
        backbone_name=backbone,
        use_pretrained=False,
        embedding_dim=model_cfg["embedding_dim"],
        use_classical=model_cfg["use_classical"],
        classical_in_dim=model_cfg.get("classical_in_dim", 6110),
        classical_proj_dim=model_cfg["classical_proj_dim"]
    )
    # If the model uses Cosine Distance, the checkpoint might contain temperature
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(config.DEVICE)
    
    evaluate_test(model, test_loader, config.DEVICE, test_dataset.classes)


if __name__ == "__main__":
    main()
