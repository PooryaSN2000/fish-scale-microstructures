import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import time
import math
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
try:
    from . import config
    from .model import FishScaleConvNeXtV4
    from .loss import ClassBalancedArcFace
    from .dataset import get_dataloaders
    from .augmentations import fourier_amplitude_swap
    from .save_prototypes import compute_and_save_prototypes
except ImportError:
    import config
    from model import FishScaleConvNeXtV4
    from loss import ClassBalancedArcFace
    from dataset import get_dataloaders
    from augmentations import fourier_amplitude_swap
    from save_prototypes import compute_and_save_prototypes

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch(model, criterion, dataloader, optimizer, scheduler, device, is_stage2=True):
    model.train()
    criterion.train()
    
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()
    
    for batch_idx, (images, labels, _) in enumerate(dataloader):
        images = images.to(device)
        labels = labels.to(device)
        B = images.size(0)

        # Fourier Domain Augmentation (FDA) across batch pairs
        if is_stage2 and config.USE_FDA and random.random() < 0.35 and B > 1:
            perm = torch.randperm(B)
            # Apply FDA to half the batch
            half = B // 2
            augmented_half = torch.stack([
                fourier_amplitude_swap(images[i], images[perm[i]], beta=config.FDA_BETA)
                for i in range(half)
            ])
            images[:half] = augmented_half

        embeddings = model(images)
        logits = criterion(embeddings, labels)
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Gradient accumulation
        loss = loss / config.ACCUMULATION_STEPS
        loss.backward()

        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * config.ACCUMULATION_STEPS * B
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, criterion, dataloader, device):
    model.eval()
    criterion.eval()

    running_loss = 0.0
    correct = 0
    top3_correct = 0
    total = 0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            embeddings = model(images)
            logits = criterion(embeddings, labels)
            loss = nn.CrossEntropyLoss()(logits, labels)

            running_loss += loss.item() * images.size(0)
            
            # Top-1 Accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-3 Accuracy
            _, top3_pred = logits.topk(min(3, logits.size(1)), dim=1)
            top3_correct += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()

    val_loss = running_loss / total
    val_acc = 100.0 * correct / total
    val_top3 = 100.0 * top3_correct / total
    return val_loss, val_acc, val_top3

def run_training():
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    print(f"=== Starting AquaLens AI v4 Training Pipeline on {device} ===")

    # 1. Load Data
    train_loader, val_loader, test_loader, train_counts, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"[Data] Loaded {num_classes} species. Train samples: {sum(train_counts)}. Head min/max: {min(train_counts)}/{max(train_counts)}")

    # 2. Instantiate Model and Class-Balanced ArcFace
    model = FishScaleConvNeXtV4(
        embedding_dim=config.EMBEDDING_DIM,
        use_mixstyle=config.USE_MIXSTYLE,
        mixstyle_p=config.MIXSTYLE_P,
        mixstyle_alpha=config.MIXSTYLE_ALPHA
    ).to(device)

    criterion = ClassBalancedArcFace(
        in_features=config.EMBEDDING_DIM,
        num_classes=num_classes,
        s=config.ARCFACE_SCALE,
        m_base=config.ARCFACE_MARGIN_BASE,
        delta_m=config.ARCFACE_MARGIN_DELTA,
        gamma=config.ARCFACE_GAMMA,
        class_counts=train_counts
    ).to(device)

    # -------------------------------------------------------------------------
    # STAGE 1: Warmup & Projector Alignment (Frozen Backbone)
    # -------------------------------------------------------------------------
    print("\n--- Stage 1: Warmup & Projector Alignment (5 Epochs) ---")
    for param in model.backbone.parameters():
        param.requires_grad = False

    stage1_params = [
        {"params": model.projector.parameters(), "lr": config.LR_PROJECTOR},
        {"params": criterion.parameters(), "lr": config.LR_ARCFACE}
    ]
    optimizer1 = optim.AdamW(stage1_params, weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, 6):
        t0 = time.time()
        loss, acc = train_epoch(model, criterion, train_loader, optimizer1, None, device, is_stage2=False)
        val_loss, val_acc, val_top3 = evaluate(model, criterion, val_loader, device)
        t_epoch = time.time() - t0
        print(f"Stage 1 Epoch [{epoch}/5] - Train Loss: {loss:.4f}, Train Acc: {acc:.1f}% | Val Loss: {val_loss:.4f}, Val Top-1: {val_acc:.1f}%, Val Top-3: {val_top3:.1f}% ({t_epoch:.1f}s)")

    # -------------------------------------------------------------------------
    # STAGE 2: End-to-End Fine-tuning with MixStyle & FDA
    # -------------------------------------------------------------------------
    print("\n--- Stage 2: Full End-to-End Training with MixStyle & FDA (30 Epochs) ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    stage2_params = [
        {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
        {"params": model.projector.parameters(), "lr": config.LR_PROJECTOR},
        {"params": criterion.parameters(), "lr": config.LR_ARCFACE}
    ]
    optimizer2 = optim.AdamW(stage2_params, weight_decay=config.WEIGHT_DECAY)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=30, eta_min=1e-6)

    best_val_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_top3": []}

    for epoch in range(1, 31):
        t0 = time.time()
        loss, acc = train_epoch(model, criterion, train_loader, optimizer2, scheduler2, device, is_stage2=True)
        val_loss, val_acc, val_top3 = evaluate(model, criterion, val_loader, device)
        t_epoch = time.time() - t0

        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(val_acc)
        history["val_top3"].append(val_top3)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "val_acc": val_acc,
                "classes": classes
            }, best_path)

        star = " ★ (Best)" if is_best else ""
        print(f"Stage 2 Epoch [{epoch:02d}/30] - Train Loss: {loss:.4f}, Train Acc: {acc:.1f}% | Val Loss: {val_loss:.4f}, Val Top-1: {val_acc:.1f}%, Val Top-3: {val_top3:.1f}% ({t_epoch:.1f}s){star}")

    print(f"\n[Training Complete] Peak Validation Top-1 Accuracy: {best_val_acc:.2f}%")

    # 3. Compute and Save Prototypes
    print("\n--- Computing and Saving Calibrated Hyperspherical Prototypes ---")
    compute_and_save_prototypes(model, classes)

    # 4. Plot and Save Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(range(1, 31), history["train_loss"], label="Train Loss (CB-ArcFace)", color="#0284c7", lw=2)
    ax1.plot(range(1, 31), history["val_loss"], label="Val Loss", color="#e11d48", lw=2, linestyle="--")
    ax1.set_title("AquaLens v4 Loss Progression", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch (Stage 2)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(range(1, 31), history["val_acc"], label="Val Top-1 Accuracy", color="#059669", lw=2)
    ax2.plot(range(1, 31), history["val_top3"], label="Val Top-3 Accuracy", color="#6366f1", lw=2, linestyle=":")
    ax2.set_title("Validation Accuracy (57 Species)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch (Stage 2)")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(config.CHECKPOINT_DIR, "loss_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Report] Training progression plot saved to: {plot_path}")

    return history

if __name__ == "__main__":
    run_training()
