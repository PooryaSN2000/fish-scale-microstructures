import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import shutil
from PIL import Image
import config
from dataset import FishScaleDataset
from inference import FishClassifier


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_confusion_matrix(targets, preds, class_names):
    cm = confusion_matrix(targets, preds)
    n_classes = len(class_names)
    
    fig, ax = plt.subplots(figsize=(24, 20))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=f'Test Confusion Matrix - 40 Fish Species (ArcFace + ConvNeXt-Tiny)',
        ylabel='True Species',
        xlabel='Predicted Species'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=6)
                
    fig.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Confusion Matrix to: {save_path}")


def plot_roc_curves(targets, probs_list, class_names):
    probs = np.array(probs_list)
    n_classes = len(class_names)
    
    targets_bin = label_binarize(targets, classes=range(n_classes))
    if targets_bin.shape[1] == 1:
        targets_bin = np.hstack([1 - targets_bin, targets_bin])
        
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(targets_bin.ravel(), probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], color='#94a3b8', lw=0.8, alpha=0.35)
        
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC (AUC = {roc_auc["micro"]:.2f})',
             color='#ec4899', linestyle=':', linewidth=3)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC (AUC = {roc_auc["macro"]:.2f})',
             color='#2563eb', linestyle='-', linewidth=3)
                  
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'Multi-class ROC Curves ({n_classes} Species, One-vs-Rest)', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, "roc_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curves to: {save_path}")


def plot_pr_curves(targets, probs_list, class_names):
    probs = np.array(probs_list)
    n_classes = len(class_names)
    
    targets_bin = label_binarize(targets, classes=range(n_classes))
    if targets_bin.shape[1] == 1:
        targets_bin = np.hstack([1 - targets_bin, targets_bin])
        
    precision = dict()
    recall = dict()
    avg_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(targets_bin[:, i], probs[:, i])
        avg_precision[i] = auc(recall[i], precision[i])
        
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets_bin.ravel(), probs.ravel())
    avg_precision["micro"] = auc(recall["micro"], precision["micro"])
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(recall[i], precision[i], color='#cbd5e1', lw=0.7, alpha=0.4)
        
    plt.plot(recall["micro"], precision["micro"],
             label=f'Micro-average PR (AP = {avg_precision["micro"]:.2f})',
             color='#059669', linewidth=3)
             
    plt.xlim([0.0, 1.02])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curves Across {n_classes} Marine Species', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, "pr_curves.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved PR curves to: {save_path}")


def plot_class_performance(targets, preds, class_names):
    unique_labels = sorted(list(set(targets)))
    labels_text = [class_names[l] for l in unique_labels]
    
    f1_scores = f1_score(targets, preds, average=None, labels=unique_labels)
    
    plt.figure(figsize=(20, 8))
    bars = plt.bar(labels_text, f1_scores, color='#3b82f6')
    
    mean_f1 = np.mean(f1_scores)
    plt.axhline(y=mean_f1, color='#ef4444', linestyle='--', linewidth=2, label=f'Mean F1 ({mean_f1:.2f})')
    
    plt.title("Per-Class F1-Score (ArcFace Metric Learning)", fontsize=16, fontweight='bold')
    plt.xlabel("Fish Species", fontsize=11)
    plt.ylabel("F1 Score", fontsize=11)
    plt.ylim(0, 1.1)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, "class_performance.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Class Performance chart to: {save_path}")


def plot_tsne(embeddings, targets, prototypes, class_names):
    print("Computing t-SNE projection of test embeddings and prototype centroids...")
    n_samples = len(embeddings)
    n_protos = len(prototypes)
    
    combined = np.vstack([embeddings, prototypes])
    perplexity = min(30, max(5, n_samples // 5))
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)
    
    sample_reduced = reduced[:n_samples]
    proto_reduced = reduced[n_samples:]
    
    plt.figure(figsize=(14, 11))
    cmap = plt.get_cmap("tab20")
    
    # Plot test sample embeddings
    scatter = plt.scatter(
        sample_reduced[:, 0],
        sample_reduced[:, 1],
        c=targets,
        cmap=cmap,
        alpha=0.65,
        s=45,
        edgecolors='none',
        label='Test Scale Embeddings'
    )
    
    # Plot prototype centroids
    plt.scatter(
        proto_reduced[:, 0],
        proto_reduced[:, 1],
        c=range(n_protos),
        cmap=cmap,
        marker='X',
        s=160,
        edgecolors='black',
        linewidth=1.5,
        label='Species Prototype Centroids'
    )
    
    plt.title("t-SNE Metric Space: 40 Marine Fish Species Clusters & Prototypes", fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(loc="upper right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, "metric_tsne.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot to: {save_path}")


def plot_gradcam_gallery(classifier, test_dataset):
    print("Generating Grad-CAM Explainability Gallery for publication...")
    sample_indices = [0, min(50, len(test_dataset)-1), min(120, len(test_dataset)-1)]
    
    fig, axs = plt.subplots(len(sample_indices), 3, figsize=(12, 4 * len(sample_indices)))
    
    for row_idx, idx in enumerate(sample_indices):
        img_path = test_dataset.image_paths[idx]
        true_name = test_dataset.classes[test_dataset.labels[idx]]
        pil_img = Image.open(img_path).convert("RGB")
        
        cam_norm, overlay_arr, overlay_pil, pred_idx = classifier.explainer.generate_cam(pil_img)
        pred_name = classifier.class_names[pred_idx]
        
        img_resized = np.array(pil_img.resize(config.IMAGE_SIZE))
        
        axs[row_idx, 0].imshow(img_resized)
        axs[row_idx, 0].set_title(f"Input: {true_name[:24]}", fontsize=9, fontweight='bold')
        axs[row_idx, 0].axis("off")
        
        axs[row_idx, 1].imshow(cam_norm, cmap='jet')
        axs[row_idx, 1].set_title("Grad-CAM Heatmap", fontsize=9, fontweight='bold')
        axs[row_idx, 1].axis("off")
        
        axs[row_idx, 2].imshow(overlay_arr)
        axs[row_idx, 2].set_title(f"Focus: {pred_name[:24]}", fontsize=9, fontweight='bold')
        axs[row_idx, 2].axis("off")
        
    plt.tight_layout()
    save_path = os.path.join(config.CHECKPOINT_DIR, "gradcam_explainability.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Grad-CAM Gallery to: {save_path}")


def main():
    set_seed(42)
    print("=" * 60)
    print("EVALUATING 40-SPECIES ARCFACE MODEL ON TEST SET (185 IMAGES)")
    print("=" * 60)
    
    classifier = FishClassifier(
        model_path=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"),
        prototypes_path=os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")
    )
    
    test_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="test", seed=config.RANDOM_SEED)
    class_names = test_dataset.classes
    num_samples = len(test_dataset)
    
    targets = []
    preds = []
    probs_list = []
    embeddings_list = []
    
    print(f"Running inference on {num_samples} test samples with TTA...")
    for idx in range(num_samples):
        img_path = test_dataset.image_paths[idx]
        true_lbl = test_dataset.labels[idx]
        
        pred_class, conf, all_probs, max_sim, all_sims = classifier.predict(img_path, use_tta=True)
        pred_lbl = test_dataset.class_to_idx[pred_class]
        
        targets.append(true_lbl)
        preds.append(pred_lbl)
        probs_list.append([all_probs[c] for c in class_names])
        
        # Extract single embedding for t-SNE
        with torch.no_grad():
            raw_img = test_dataset.image_cache[idx]
            emb = classifier._extract_single_embedding(raw_img).squeeze(0).cpu().numpy()
            embeddings_list.append(emb)
            
    # Metrics
    acc = accuracy_score(targets, preds) * 100
    macro_f1 = f1_score(targets, preds, average="macro") * 100
    weighted_f1 = f1_score(targets, preds, average="weighted") * 100
    
    print("\n" + "=" * 60)
    print(f"REAL TEST SET EVALUATION RESULTS:")
    print(f"  Test Accuracy:     {acc:.2f}% ({sum(np.array(targets) == np.array(preds))}/{num_samples})")
    print(f"  Macro F1-Score:    {macro_f1:.2f}%")
    print(f"  Weighted F1-Score: {weighted_f1:.2f}%")
    print("=" * 60 + "\n")
    
    # Save classification report
    report = classification_report(targets, preds, target_names=class_names, digits=4)
    print(report)
    with open(os.path.join(config.CHECKPOINT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)
        f.write(f"\nTest Accuracy: {acc:.4f}%\nMacro F1: {macro_f1:.4f}%\n")
        
    # Generate 300 DPI Publication Plots
    plot_confusion_matrix(targets, preds, class_names)
    plot_roc_curves(targets, probs_list, class_names)
    plot_pr_curves(targets, probs_list, class_names)
    plot_class_performance(targets, preds, class_names)
    plot_tsne(np.array(embeddings_list), np.array(targets), classifier.prototypes.cpu().numpy(), class_names)
    plot_gradcam_gallery(classifier, test_dataset)
    
    # Copy all generated plots to paper/img/
    paper_img_dir = os.path.join(os.path.dirname(config.BASE_DIR), "paper", "img")
    if os.path.exists(paper_img_dir):
        plot_files = [
            "confusion_matrix.png", "roc_curves.png", "pr_curves.png",
            "class_performance.png", "metric_tsne.png", "training_history.png",
            "gradcam_explainability.png"
        ]
        for pf in plot_files:
            src = os.path.join(config.CHECKPOINT_DIR, pf)
            dst = os.path.join(paper_img_dir, pf)
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"Copied updated publication figures to {paper_img_dir}")
    
    print("\nAll evaluation artifacts generated successfully!")


if __name__ == "__main__":
    main()
