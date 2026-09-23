import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
import matplotlib.pyplot as plt
from collections import Counter

import config
import evaluate
import save_prototypes
from dataset import FishScaleDataset
from model import FishArcNet


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_gpu_transforms():
    return v2.Compose([
        v2.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.85, 1.0), antialias=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.2), value='random')
    ])


def get_gpu_val_transforms():
    return v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def train_epoch(model, dataloader, optimizer, scaler, device, gpu_transforms, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = len(dataloader)
    accum_steps = getattr(config, "ACCUMULATION_STEPS", 1)
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch[0].to(device), batch[1].to(device)
        images = gpu_transforms(images)
        
        with torch.autocast(device_type=device.type if device.type != 'mac' else 'cpu', enabled=device.type == 'cuda'):
            logits, embs = model(images, labels)
            loss = criterion(logits, labels)
            loss_scaled = loss / accum_steps
            
        scaler.scale(loss_scaled).backward()
        
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == num_batches:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        preds = torch.argmax(logits, dim=-1)
        acc = (preds == labels).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        
    return total_loss / num_batches, total_acc / num_batches


@torch.no_grad()
def evaluate_validation(model, dataloader, device, gpu_transforms):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    # Class weights from ArcFace serve as normalized prototypes
    prototypes = F.normalize(model.arcface.weight.data, p=2, dim=-1)
    
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        images = gpu_transforms(images)
        
        embs = model.extract_embedding(images)
        # Cosine similarity to prototypes
        similarities = torch.matmul(embs, prototypes.t())
        preds = torch.argmax(similarities, dim=-1)
        
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        
    val_acc = (total_correct / total_samples) if total_samples > 0 else 0.0
    return val_acc


def save_training_plots(history, save_dir):
    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    train_accs = [h["train_acc"] for h in history]
    val_accs = [h["val_acc"] for h in history]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss (ArcFace)", color="#ef4444", linewidth=2)
    plt.title("ArcFace Angular Margin Loss", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label="Train Acc", color="#3b82f6", linewidth=2)
    plt.plot(epochs, val_accs, label="Val Prototype Acc", color="#10b981", linewidth=2)
    plt.title("Classification & Metric Accuracy", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves to {plot_path}")


def main():
    set_seed(config.RANDOM_SEED)
    print("=" * 60)
    print(f"TRAINING 40-SPECIES CONVNEXT + ARCFACE METRIC MODEL")
    print(f"Device: {config.DEVICE}")
    print(f"Resolution: {config.IMAGE_SIZE}")
    print(f"Epochs: {config.EPOCHS} | Batch Size: {config.BATCH_SIZE}")
    print("=" * 60)
    
    # 1. Datasets
    train_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    val_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="val", seed=config.RANDOM_SEED)
    test_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="test", seed=config.RANDOM_SEED)
    
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
    
    # 2. Class-balanced sampling for training
    class_counts = Counter(train_dataset.labels)
    class_weights = {c: 1.0 / count for c, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_dataset.labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Model Architecture
    model = FishArcNet(
        backbone_name=config.BACKBONE,
        use_pretrained=True,
        num_classes=config.NUM_CLASSES,
        embedding_dim=config.EMBEDDING_DIM,
        scale=config.ARCFACE_SCALE,
        margin=config.ARCFACE_MARGIN
    ).to(config.DEVICE)
    
    # 4. Differential Learning Rates
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config.LR_BACKBONE, 'weight_decay': config.WEIGHT_DECAY},
        {'params': model.projector.parameters(), 'lr': config.LR_PROJECTOR, 'weight_decay': config.WEIGHT_DECAY},
        {'params': model.arcface.parameters(), 'lr': config.LR_ARCFACE, 'weight_decay': config.WEIGHT_DECAY}
    ])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=1e-6)
    scaler = torch.amp.GradScaler(device='cuda') if config.DEVICE.type == 'cuda' else torch.amp.GradScaler(device='cpu', enabled=False)
    criterion = nn.CrossEntropyLoss()
    
    gpu_train_transforms = get_gpu_transforms().to(config.DEVICE)
    gpu_val_transforms = get_gpu_val_transforms().to(config.DEVICE)
    
    best_val_acc = 0.0
    history = []
    
    print("\nStarting Training...")
    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scaler, config.DEVICE, gpu_train_transforms, criterion)
        val_acc = evaluate_validation(model, val_loader, config.DEVICE, gpu_val_transforms)
        scheduler.step()
        
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_acc": val_acc
        })
        
        print(f"Epoch [{epoch:02d}/{config.EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "config": {
                    "backbone": config.BACKBONE,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "num_classes": config.NUM_CLASSES,
                    "image_size": config.IMAGE_SIZE,
                    "scale": config.ARCFACE_SCALE,
                    "margin": config.ARCFACE_MARGIN
                }
            }
            save_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(checkpoint, save_path)
            print(f"   ==> New best model saved (Val Acc: {val_acc*100:.2f}%)")
            
    print("\n" + "=" * 60)
    print(f"Training Complete! Peak Val Accuracy: {best_val_acc*100:.2f}%")
    print("=" * 60)
    
    # Save training curves
    save_training_plots(history, config.CHECKPOINT_DIR)
    
    # Generate updated prototypes and evaluation metrics
    print("\n[Step 2/3] Extracting Augmented Prototypes...")
    save_prototypes.main()
    
    print("\n[Step 3/3] Generating Full Evaluation Analytics & Charts...")
    evaluate.main()


if __name__ == "__main__":
    main()
