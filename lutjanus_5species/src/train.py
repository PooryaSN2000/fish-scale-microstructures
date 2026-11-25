import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.transforms import v2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter

import config
import save_prototypes
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
        v2.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.82, 1.0), antialias=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_gpu_val_transforms():
    return v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def train_epoch_stage1(model, dataloader, optimizer, device, gpu_transforms):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        images, labels = batch[0].to(device), batch[1].to(device)
        images = gpu_transforms(images)
        logits = model(images, stage=1)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (logits.argmax(-1) == labels).float().mean().item()
    n = len(dataloader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_stage1(model, dataloader, device, gpu_transforms):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for batch in dataloader:
        images, labels = gpu_transforms(batch[0].to(device)), batch[1].to(device)
        logits = model(images, stage=1)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def train_epoch_stage2(model, dataloader, optimizer, device, gpu_transforms):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        images = gpu_transforms(batch[0].to(device))
        embeddings = model(images, stage=2)
        logits, targets = compute_prototypes_and_logits(
            embeddings, n_way=config.N_WAY, k_shot=config.K_SHOT,
            q_query=config.Q_QUERY, temperature=model.temperature
        )
        loss, acc = compute_protonet_loss(logits, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_acc += acc.item()
    n = len(dataloader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate_proto_tta(model, train_dataset, eval_dataset, device, gpu_val_transforms):
    """Evaluates val/test accuracy using Train-split-only 4-View Spherical-TTA prototypes."""
    model.eval()

    def _tta_emb(x_batch):
        views = [
            x_batch,
            v2.functional.horizontal_flip(x_batch),
            v2.functional.vertical_flip(x_batch),
            torch.rot90(x_batch, 1, [-2, -1]),
        ]
        es = [F.normalize(model(gpu_val_transforms(v), stage=2), p=2, dim=-1) for v in views]
        return F.normalize(torch.stack(es).mean(0), p=2, dim=-1)

    tr_embs, tr_lbls = [], []
    for batch in DataLoader(train_dataset, batch_size=32, shuffle=False):
        tr_embs.append(_tta_emb(batch[0].to(device)))
        tr_lbls.append(batch[1].to(device))
    tr_embs = torch.cat(tr_embs, dim=0)
    tr_lbls = torch.cat(tr_lbls, dim=0)
    protos = torch.stack([F.normalize(tr_embs[tr_lbls == c].mean(0), p=2, dim=-1) for c in range(config.NUM_CLASSES)])

    correct, total, total_loss = 0, 0, 0.0
    for batch in DataLoader(eval_dataset, batch_size=32, shuffle=False):
        emb = _tta_emb(batch[0].to(device))
        lbls = batch[1].to(device)
        sims = torch.matmul(emb, protos.t())
        logits = sims * (model.temperature if model.temperature is not None else 1.0)
        total_loss += F.cross_entropy(logits, lbls, reduction="sum").item()
        correct += (sims.argmax(-1) == lbls).sum().item()
        total += lbls.size(0)
    return total_loss / total, correct / total


def save_narrow_stage_plot(history, paper_img_dir, filename, title_prefix):
    epochs = [h["epoch"] for h in history]
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    tr_acc = [h["train_acc"] * 100.0 for h in history]
    va_acc = [h["val_acc"] * 100.0 for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.8), dpi=300, sharex=True)

    ax1.plot(epochs, tr_loss, label="Train Loss", color="#1f77b4", lw=2.3, marker="o", markersize=4)
    ax1.plot(epochs, va_loss, label="Validation Loss", color="#ff7f0e", lw=2.3, marker="s", markersize=4)
    ax1.set_title(f"{title_prefix} Metric Loss Progression", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10.5, framealpha=0.95)
    ax1.tick_params(labelsize=11)

    ax2.plot(epochs, tr_acc, label="Train Episodic Accuracy", color="#2ca02c", lw=2.3, marker="o", markersize=4)
    ax2.plot(epochs, va_acc, label="Validation Prototype Accuracy", color="#d62728", lw=2.3, marker="s", markersize=4)
    ax2.set_title(f"{title_prefix} Classification Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Training Epoch", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_ylim([50, 104])
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right", fontsize=10.5, framealpha=0.95)
    ax2.tick_params(labelsize=11)

    fig.tight_layout()
    os.makedirs(paper_img_dir, exist_ok=True)
    plt.savefig(os.path.join(paper_img_dir, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    set_seed(config.RANDOM_SEED)
    device = config.DEVICE
    paper_img_dir = os.path.join(os.path.dirname(config.CHECKPOINT_DIR), "paper", "img")

    train_ds = FishScaleDataset(root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    val_ds = FishScaleDataset(root_dir=config.ROOT_DIR, split="val", seed=config.RANDOM_SEED)

    model = ProtoNet(
        backbone_name=config.BACKBONE,
        use_pretrained=True,
        embedding_dim=config.EMBEDDING_DIM,
        use_classical=config.USE_CLASSICAL,
    ).to(device)

    gpu_train_tf = get_gpu_transforms().to(device)
    gpu_val_tf = get_gpu_val_transforms().to(device)

    # STAGE 1: Class-Balanced Pretraining
    print("\n" + "=" * 55)
    print(f"STAGE 1: Class-Balanced Pre-training ({config.EPOCHS_STAGE1} Epochs)")
    print("=" * 55)
    counts = Counter(train_ds.labels)
    sampler_s1 = WeightedRandomSampler([1.0 / counts[y] for y in train_ds.labels], num_samples=len(train_ds), replacement=True)
    loader_s1 = DataLoader(train_ds, batch_size=config.BATCH_SIZE_STAGE1, sampler=sampler_s1)
    val_loader_s1 = DataLoader(val_ds, batch_size=32, shuffle=False)

    opt1 = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": config.LEARNING_RATE_STAGE1},
        {"params": model.cnn_projection.parameters(), "lr": 4e-4},
        {"params": model.classifier_head.parameters(), "lr": 4e-4},
    ], weight_decay=config.WEIGHT_DECAY)

    best_s1_acc, best_s1_state, hist_s1 = -1.0, None, []
    for ep in range(1, config.EPOCHS_STAGE1 + 1):
        tr_loss, tr_acc = train_epoch_stage1(model, loader_s1, opt1, device, gpu_train_tf)
        va_loss, va_acc = evaluate_stage1(model, val_loader_s1, device, gpu_val_tf)
        hist_s1.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        print(f"S1-Epoch {ep:02d}/{config.EPOCHS_STAGE1} | Loss: {tr_loss:.4f} | Val Acc: {va_acc*100:.2f}%")
        if va_acc >= best_s1_acc:
            best_s1_acc = va_acc
            best_s1_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    save_narrow_stage_plot(hist_s1, paper_img_dir, "stage1_loss_curves.png", "Stage 1 Pre-training")

    # STAGE 2: Episodic Few-Shot Fine-Tuning (unfreeze upper ConvNeXt blocks 4..7)
    print("\n" + "=" * 55)
    print(f"STAGE 2: Episodic Few-Shot Fine-Tuning ({config.EPOCHS_STAGE2} Epochs)")
    print("=" * 55)
    model.load_state_dict({k: v.to(device) for k, v in best_s1_state.items()})
    for name, param in model.backbone.named_parameters():
        param.requires_grad = any(name.startswith(f"{blk}.") for blk in [4, 5, 6, 7])

    tr_ep = DataLoader(train_ds, batch_sampler=EpisodicBatchSampler(train_ds.labels, config.N_WAY, config.K_SHOT, config.Q_QUERY, config.EPISODES_PER_EPOCH))
    opt2 = torch.optim.AdamW([
        {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 3e-5},
        {"params": model.cnn_projection.parameters(), "lr": config.LEARNING_RATE_STAGE2},
        {"params": [model.temperature], "lr": 1e-2},
    ], weight_decay=config.WEIGHT_DECAY)

    best_s2_acc, hist_s2 = -1.0, []
    for ep in range(1, config.EPOCHS_STAGE2 + 1):
        tr_loss, tr_acc = train_epoch_stage2(model, tr_ep, opt2, device, gpu_train_tf)
        va_loss, va_acc = evaluate_proto_tta(model, train_ds, val_ds, device, gpu_val_tf)
        hist_s2.append({"epoch": ep, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": va_loss, "val_acc": va_acc})
        print(f"S2-Epoch {ep:02d}/{config.EPOCHS_STAGE2} | EpLoss: {tr_loss:.4f} | EpAcc: {tr_acc*100:.2f}% | ValProtoAcc: {va_acc*100:.2f}%")
        if va_acc >= best_s2_acc:
            best_s2_acc = va_acc
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "val_acc": va_acc,
                "stage1_best_val_acc": best_s1_acc,
                "config": {
                    "backbone": config.BACKBONE,
                    "embedding_dim": config.EMBEDDING_DIM,
                    "use_classical": config.USE_CLASSICAL,
                    "classical_in_dim": 0,
                    "classical_proj_dim": config.CLASSICAL_PROJ_DIM,
                    "image_size": config.IMAGE_SIZE,
                    "use_clahe": config.USE_CLAHE,
                    "use_tta": config.USE_TTA,
                    "n_way": config.N_WAY,
                    "k_shot": config.K_SHOT,
                },
            }, os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
            print(f"  ==> Saved best_model.pth (Val Prototype Acc: {va_acc*100:.2f}%)")

    save_narrow_stage_plot(hist_s2, paper_img_dir, "stage2_loss_curves.png", "Stage 2 Episodic Fine-Tuning")
    print(f"\n[Training Complete] Stage 1 Val: {best_s1_acc*100:.2f}% | Stage 2 Val: {best_s2_acc*100:.2f}%")

    # Recompute train-only prototypes
    save_prototypes.main()


if __name__ == "__main__":
    main()
