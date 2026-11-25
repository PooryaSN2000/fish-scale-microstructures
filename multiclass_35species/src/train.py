import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import time
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

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


def train_epoch(model, criterion, ce_loss_fn, dataloader, optimizer, scheduler, device, is_stage2=True):
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
        if is_stage2 and config.USE_FDA and random.random() < 0.25 and B > 1:
            perm = torch.randperm(B, device=device)
            half = B // 2
            augmented_half = torch.stack([
                fourier_amplitude_swap(images[i], images[perm[i]], beta=config.FDA_BETA)
                for i in range(half)
            ])
            images[:half] = augmented_half

        embeddings = model(images)
        logits = criterion(embeddings, labels)
        cos_logits = F.linear(embeddings, F.normalize(criterion.weight, p=2, dim=1)) * criterion.s
        loss = ce_loss_fn(logits, labels) / config.ACCUMULATION_STEPS
        loss.backward()

        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * config.ACCUMULATION_STEPS * B
        _, predicted = cos_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    if scheduler is not None:
        scheduler.step()

    return running_loss / max(1, total), 100.0 * correct / max(1, total)


def evaluate(model, criterion, ce_loss_fn, dataloader, device):
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
            w_norm = F.normalize(criterion.weight, p=2, dim=1)
            cos_logits = F.linear(embeddings, w_norm) * criterion.s
            loss = ce_loss_fn(cos_logits, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = cos_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            _, top3_pred = cos_logits.topk(min(3, cos_logits.size(1)), dim=1)
            top3_correct += top3_pred.eq(labels.view(-1, 1).expand_as(top3_pred)).sum().item()

    return running_loss / max(1, total), 100.0 * correct / max(1, total), 100.0 * top3_correct / max(1, total)


def run_training():
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    s1_epochs = getattr(config, "EPOCHS_STAGE1", 6)
    s2_epochs = getattr(config, "EPOCHS_STAGE2", 32)
    print(f"=== Starting AquaLens AI v4.1 (ROI + MixStyle + FDA + CB-ArcFace) on {device} ===")

    train_loader, val_loader, test_loader, train_counts, classes = get_dataloaders()
    num_classes = len(classes)
    print(f"[Data] Loaded {num_classes} species (ROI_Crop={getattr(config, 'USE_ROI_CROP', True)}, CLAHE={config.USE_CLAHE}).")
    print(f"[Data] Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}, counts range=[{min(train_counts)}, {max(train_counts)}]")

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

    ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Pre-warm dataset RAM cache cleanly before timing epochs
    print("[Cache] Pre-localizing ROI crops + CLAHE for train & val...")
    for idx in range(len(train_loader.dataset)):
        train_loader.dataset.load_processed_pil(train_loader.dataset.image_paths[idx])
    for idx in range(len(val_loader.dataset)):
        val_loader.dataset.load_processed_pil(val_loader.dataset.image_paths[idx])

    # STAGE 1: Warmup & Projector Alignment
    print(f"\n--- Stage 1: Warmup & Projector Alignment ({s1_epochs} Epochs) ---")
    for param in model.backbone.parameters():
        param.requires_grad = False

    optimizer1 = optim.AdamW([
        {"params": model.projector.parameters(), "lr": config.LR_PROJECTOR},
        {"params": criterion.parameters(), "lr": config.LR_ARCFACE},
    ], weight_decay=config.WEIGHT_DECAY)

    for epoch in range(1, s1_epochs + 1):
        t0 = time.time()
        loss, acc = train_epoch(model, criterion, ce_loss_fn, train_loader, optimizer1, None, device, is_stage2=False)
        val_loss, val_acc, val_top3 = evaluate(model, criterion, ce_loss_fn, val_loader, device)
        print(f"Stage 1 [{epoch}/{s1_epochs}] Loss: {loss:.4f} Acc: {acc:.1f}% | Val Loss: {val_loss:.4f} Val Top-1: {val_acc:.1f}% Top-3: {val_top3:.1f}% ({time.time() - t0:.1f}s)")

    # STAGE 2: End-to-End Fine-tuning with MixStyle & FDA
    print(f"\n--- Stage 2: Full End-to-End Training with MixStyle & FDA ({s2_epochs} Epochs) ---")
    for param in model.backbone.parameters():
        param.requires_grad = True

    optimizer2 = optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
        {"params": model.projector.parameters(), "lr": config.LR_PROJECTOR},
        {"params": criterion.parameters(), "lr": config.LR_ARCFACE},
    ], weight_decay=config.WEIGHT_DECAY)
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=s2_epochs, eta_min=1e-6)

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "val_top3": []}

    for epoch in range(1, s2_epochs + 1):
        t0 = time.time()
        loss, acc = train_epoch(model, criterion, ce_loss_fn, train_loader, optimizer2, scheduler2, device, is_stage2=True)
        val_loss, val_acc, val_top3 = evaluate(model, criterion, ce_loss_fn, val_loader, device)

        history["train_loss"].append(loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(acc)
        history["val_acc"].append(val_acc)
        history["val_top3"].append(val_top3)

        is_best = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if is_best:
            best_val_acc = val_acc
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
                "config": {
                    "backbone": config.BACKBONE,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "image_size": config.IMAGE_SIZE,
                    "use_roi_crop": getattr(config, "USE_ROI_CROP", True),
                    "roi_padding_ratio": getattr(config, "ROI_PADDING_RATIO", 0.12),
                    "use_clahe": config.USE_CLAHE,
                    "use_tta": config.USE_SPHERICAL_TTA,
                    "use_mixstyle": config.USE_MIXSTYLE,
                },
            }, best_path)

        star = " ★ (Best)" if is_best else ""
        print(f"Stage 2 [{epoch:02d}/{s2_epochs}] Loss: {loss:.4f} Acc: {acc:.1f}% | Val Loss: {val_loss:.4f} Val Top-1: {val_acc:.1f}% Top-3: {val_top3:.1f}% ({time.time() - t0:.1f}s){star}")

    print(f"\n[Training Complete] Peak Validation Top-1 Accuracy: {best_val_acc:.2f}%")

    # Reload strictly best checkpoint weights before generating leakage-free train prototypes
    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    print("\n--- Computing Strictly Leakage-Free Prototypes (split='train', 7-View Spherical-TTA) ---")
    compute_and_save_prototypes(model, classes, split="train")

    # Save training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ep_range = range(1, s2_epochs + 1)
    ax1.plot(ep_range, history["train_loss"], label="Train Loss (CB-ArcFace)", color="#0284c7", lw=2)
    ax1.plot(ep_range, history["val_loss"], label="Val Loss", color="#e11d48", lw=2, linestyle="--")
    ax1.set_title("AquaLens v4.1 Loss Progression (ROI + MixStyle + FDA)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch (Stage 2)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(ep_range, history["val_acc"], label="Val Top-1 Accuracy", color="#059669", lw=2)
    ax2.plot(ep_range, history["val_top3"], label="Val Top-3 Accuracy", color="#6366f1", lw=2, linestyle=":")
    ax2.set_title(f"Validation Accuracy ({num_classes} Species)", fontsize=11, fontweight="bold")
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
