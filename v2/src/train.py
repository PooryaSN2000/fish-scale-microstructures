import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import matplotlib.pyplot as plt

import config
import dataset_analytics
import evaluate
from dataset import FishScaleDataset, EpisodicBatchSampler
from model import ProtoNet, compute_prototypes_and_logits, compute_protonet_loss

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_gpu_transforms():
    return v2.Compose([
        v2.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0), antialias=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_gpu_val_transforms():
    return v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_mixup_cutmix():
    return v2.RandomChoice([
        v2.CutMix(num_classes=config.NUM_CLASSES, alpha=1.0),
        v2.MixUp(num_classes=config.NUM_CLASSES, alpha=0.2)
    ])

# ==============================================================================
# STAGE 1: Standard Supervised Pre-training Functions
# ==============================================================================

def train_epoch_stage1(model, dataloader, optimizer, scaler, device, gpu_transforms, mixup_cutmix):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        if config.USE_CLASSICAL:
            images, labels, classical_feats = batch
            classical_feats = classical_feats.to(device)
        else:
            images, labels = batch
            
        images = images.to(device)
        labels = labels.to(device)
        
        images = gpu_transforms(images)
        images, mixed_labels = mixup_cutmix(images, labels)
        
        with torch.autocast(device_type=device.type if device.type != 'mac' else 'cpu', enabled=device.type=='cuda'):
            if config.USE_CLASSICAL:
                logits = model(images, classical_feats, stage=1)
            else:
                logits = model(images, stage=1)
                
            loss = F.cross_entropy(logits, mixed_labels)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        preds = torch.argmax(logits, dim=-1)
        true_labels = torch.argmax(mixed_labels, dim=-1)
        acc = (preds == true_labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / num_batches, total_acc / num_batches

@torch.no_grad()
def evaluate_stage1(model, dataloader, device, gpu_transforms):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(dataloader)
    
    for batch in dataloader:
        if config.USE_CLASSICAL:
            images, labels, classical_feats = batch
            images = images.to(device)
            images = gpu_transforms(images)
            classical_feats = classical_feats.to(device)
            logits = model(images, classical_feats, stage=1)
        else:
            images, labels = batch
            images = images.to(device)
            images = gpu_transforms(images)
            logits = model(images, stage=1)
            
        labels = labels.to(device)
        loss = F.cross_entropy(logits, labels)
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / num_batches, total_acc / num_batches

# ==============================================================================
# STAGE 2: Episodic Few-Shot Fine-Tuning Functions
# ==============================================================================

def train_epoch_stage2(model, dataloader, optimizer, scaler, device, gpu_transforms):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = len(dataloader)
    
    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        if config.USE_CLASSICAL:
            images, _, classical_feats = batch
            images = images.to(device)
            images = gpu_transforms(images)
            classical_feats = classical_feats.to(device)
            
            with torch.autocast(device_type=device.type if device.type != 'mac' else 'cpu', enabled=device.type=='cuda'):
                embeddings = model(images, classical_feats, stage=2)
                logits, targets = compute_prototypes_and_logits(
                    embeddings, n_way=config.N_WAY, k_shot=config.K_SHOT,
                    q_query=config.Q_QUERY, temperature=model.temperature
                )
                loss, acc = compute_protonet_loss(logits, targets)
        else:
            images, _ = batch
            images = images.to(device)
            images = gpu_transforms(images)
                
            with torch.autocast(device_type=device.type if device.type != 'mac' else 'cpu', enabled=device.type=='cuda'):
                embeddings = model(images, stage=2)
                logits, targets = compute_prototypes_and_logits(
                    embeddings, n_way=config.N_WAY, k_shot=config.K_SHOT,
                    q_query=config.Q_QUERY, temperature=model.temperature
                )
                loss, acc = compute_protonet_loss(logits, targets)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / num_episodes, total_acc / num_episodes

@torch.no_grad()
def evaluate_stage2(model, dataloader, device, gpu_transforms):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_episodes = len(dataloader)
    
    for batch in dataloader:
        if config.USE_CLASSICAL:
            images, _, classical_feats = batch
            images = images.to(device)
            images = gpu_transforms(images)
            classical_feats = classical_feats.to(device)
            embeddings = model(images, classical_feats, stage=2)
        else:
            images, _ = batch
            images = images.to(device)
            images = gpu_transforms(images)
            embeddings = model(images, stage=2)
            
        logits, targets = compute_prototypes_and_logits(
            embeddings, n_way=config.N_WAY, k_shot=config.K_SHOT,
            q_query=config.Q_QUERY, temperature=model.temperature
        )
        
        loss, acc = compute_protonet_loss(logits, targets)
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / num_episodes, total_acc / num_episodes


def save_plots(history, save_dir, prefix=""):
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, val_losses, label="Val Loss", color="#ff7f0e", linewidth=2)
    plt.title(f"{prefix} Loss History", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc", color="#2ca02c", linewidth=2)
    plt.plot(epochs, val_accs, label="Val Acc", color="#d62728", linewidth=2)
    plt.title(f"{prefix} Accuracy History", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{prefix.lower()}_loss_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    set_seed(config.RANDOM_SEED)
    print(f"Using Device: {config.DEVICE}")
    print(f"Data root: {config.ROOT_DIR}")
    
    print("\n" + "="*50)
    print("Pre-training Analytics: Generating Dataset Distribution...")
    dataset_analytics.analyze_dataset(config.ROOT_DIR, config.CHECKPOINT_DIR)
    print("="*50 + "\n")
    
    print("Loading datasets...")
    train_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    val_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="val", seed=config.RANDOM_SEED)
    
    classical_in_dim = 0
    if config.USE_CLASSICAL:
        _, _, sample_classical = train_dataset[0]
        classical_in_dim = sample_classical.shape[0]
        
    model = ProtoNet(
        backbone_name=config.BACKBONE,
        use_pretrained=True,
        embedding_dim=config.EMBEDDING_DIM,
        use_classical=config.USE_CLASSICAL,
        classical_in_dim=classical_in_dim,
        classical_proj_dim=config.CLASSICAL_PROJ_DIM
    )
    model = model.to(config.DEVICE)
    
    gpu_train_transforms = get_gpu_transforms().to(config.DEVICE)
    gpu_val_transforms = get_gpu_val_transforms().to(config.DEVICE)
    mixup_cutmix = get_mixup_cutmix()
    
    scaler = torch.amp.GradScaler(device='cuda') if config.DEVICE.type == 'cuda' else torch.amp.GradScaler(device='cpu', enabled=False)

    # ==========================================================================
    # STAGE 1 EXECUTION
    # ==========================================================================
    print("\n" + "="*50)
    print(f"STAGE 1: Supervised Pre-training ({config.EPOCHS_STAGE1} Epochs)")
    print("="*50)
    
    train_loader_stage1 = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_STAGE1, shuffle=True, num_workers=0, pin_memory=True)
    val_loader_stage1 = DataLoader(val_dataset, batch_size=config.BATCH_SIZE_STAGE1, shuffle=False, num_workers=0, pin_memory=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_STAGE1, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
    
    best_val_acc_stage1 = 0.0
    history_stage1 = []
    
    for epoch in range(1, config.EPOCHS_STAGE1 + 1):
        train_loss, train_acc = train_epoch_stage1(model, train_loader_stage1, optimizer, scaler, config.DEVICE, gpu_train_transforms, mixup_cutmix)
        val_loss, val_acc = evaluate_stage1(model, val_loader_stage1, config.DEVICE, gpu_val_transforms)
        scheduler.step()
        
        print(f"S1-Epoch {epoch:02d}/{config.EPOCHS_STAGE1} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        history_stage1.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        save_plots(history_stage1, config.CHECKPOINT_DIR, prefix="Stage1")
        
        if val_acc > best_val_acc_stage1:
            best_val_acc_stage1 = val_acc
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_stage1_model.pth"))
            print(f"==> Saved new best Stage 1 model! Val Acc: {val_acc:.4f}")

    # ==========================================================================
    # STAGE 2 EXECUTION
    # ==========================================================================
    print("\n" + "="*50)
    print(f"STAGE 2: Episodic Few-Shot Fine-Tuning ({config.EPOCHS_STAGE2} Epochs)")
    print("="*50)
    
    # Load best Stage 1 weights
    model.load_state_dict(torch.load(os.path.join(config.CHECKPOINT_DIR, "best_stage1_model.pth")))
    
    # Freeze the heavy CNN backbone for Stage 2 to prevent OOM and overfitting
    # We only fine-tune the metric embedding projections
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    train_sampler = EpisodicBatchSampler(labels=train_dataset.labels, n_way=config.N_WAY, k_shot=config.K_SHOT, q_query=config.Q_QUERY, num_episodes=config.EPISODES_PER_EPOCH)
    val_sampler = EpisodicBatchSampler(labels=val_dataset.labels, n_way=config.N_WAY, k_shot=config.K_SHOT, q_query=config.Q_QUERY, num_episodes=config.VAL_EPISODES)
    
    train_loader_stage2 = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    val_loader_stage2 = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE_STAGE2, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.SCHEDULER_STEP_SIZE, gamma=config.SCHEDULER_GAMMA)
    
    best_val_acc_stage2 = 0.0
    history_stage2 = []
    
    for epoch in range(1, config.EPOCHS_STAGE2 + 1):
        train_loss, train_acc = train_epoch_stage2(model, train_loader_stage2, optimizer, scaler, config.DEVICE, gpu_train_transforms)
        val_loss, val_acc = evaluate_stage2(model, val_loader_stage2, config.DEVICE, gpu_val_transforms)
        scheduler.step()
        
        print(f"S2-Epoch {epoch:02d}/{config.EPOCHS_STAGE2} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        history_stage2.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        save_plots(history_stage2, config.CHECKPOINT_DIR, prefix="Stage2")
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_acc": val_acc,
            "config": {
                "backbone": config.BACKBONE,
                "embedding_dim": config.EMBEDDING_DIM,
                "use_classical": config.USE_CLASSICAL,
                "classical_in_dim": classical_in_dim,
                "classical_proj_dim": config.CLASSICAL_PROJ_DIM,
                "n_way": config.N_WAY,
                "k_shot": config.K_SHOT,
            }
        }
        
        if val_acc > best_val_acc_stage2:
            best_val_acc_stage2 = val_acc
            torch.save(checkpoint, os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"==> Saved new best Stage 2 model! Val Acc: {val_acc:.4f}")
            
    print("\nTwo-Stage Training Completed!")
    print(f"Final Stage 2 Best Val Acc: {best_val_acc_stage2:.4f}")
    
    print("\n" + "="*50)
    print("Post-training Analytics: Generating Final Evaluation Charts...")
    evaluate.main()
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
