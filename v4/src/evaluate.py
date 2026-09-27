import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn.functional as F
try:
    from . import config
    from .model import FishScaleConvNeXtV4
    from .dataset import get_dataloaders
    from .inference import FishClassifierV4
except ImportError:
    import config
    from model import FishScaleConvNeXtV4
    from dataset import get_dataloaders
    from inference import FishClassifierV4

def evaluate_v4_system():
    device = config.DEVICE
    print(f"=== AquaLens v4 Comprehensive Evaluation & Cross-Camera Robustness ===")

    # 1. Load test data
    train_loader, val_loader, test_loader, train_counts, classes = get_dataloaders()
    num_classes = len(classes)

    # Load inference classifier with Spherical-TTA
    classifier = FishClassifierV4()

    # Identify class brackets: Head (>=20), Medium (10-19), Tail (<10)
    head_indices = [i for i, c in enumerate(train_counts) if c >= 20]
    med_indices = [i for i, c in enumerate(train_counts) if 10 <= c < 20]
    tail_indices = [i for i, c in enumerate(train_counts) if c < 10]

    print(f"[Cohort Distribution] Head (>=20): {len(head_indices)} species, Med (10-19): {len(med_indices)} species, Tail/Few-shot (<10): {len(tail_indices)} species")

    # Evaluation loop
    all_targets = []
    all_preds = []
    all_sims = []
    top3_matches = 0
    top5_matches = 0
    total = 0

    cohort_correct = {"head": 0, "head_total": 0, "med": 0, "med_total": 0, "tail": 0, "tail_total": 0}

    print("\n[Evaluation] Running Spherical-TTA metric evaluation on test split...")
    with torch.no_grad():
        for images, labels, names in test_loader:
            for b in range(images.size(0)):
                img_tensor = images[b]
                lbl = labels[b].item()

                # Denormalize to PIL for real inference pipeline
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_np = img_tensor.cpu().numpy() * std + mean
                img_np = (np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
                from PIL import Image
                pil_img = Image.fromarray(img_np)

                # Extract embedding with Spherical-TTA
                emb = classifier.extract_embedding(pil_img, use_tta=True)
                sims = torch.mm(emb, classifier.prototypes.t()).squeeze(0).cpu().numpy()

                ranked = np.argsort(sims)[::-1]
                pred_top1 = ranked[0]

                all_targets.append(lbl)
                all_preds.append(pred_top1)
                all_sims.append(sims[pred_top1])

                is_correct = (pred_top1 == lbl)
                if is_correct:
                    if lbl in head_indices:
                        cohort_correct["head"] += 1
                    elif lbl in med_indices:
                        cohort_correct["med"] += 1
                    elif lbl in tail_indices:
                        cohort_correct["tail"] += 1

                if lbl in head_indices:
                    cohort_correct["head_total"] += 1
                elif lbl in med_indices:
                    cohort_correct["med_total"] += 1
                elif lbl in tail_indices:
                    cohort_correct["tail_total"] += 1

                if lbl in ranked[:min(3, len(ranked))]:
                    top3_matches += 1
                if lbl in ranked[:min(5, len(ranked))]:
                    top5_matches += 1

                total += 1

    top1_acc = 100.0 * np.mean(np.array(all_preds) == np.array(all_targets))
    top3_acc = 100.0 * top3_matches / total
    top5_acc = 100.0 * top5_matches / total

    head_acc = 100.0 * cohort_correct["head"] / max(1, cohort_correct["head_total"])
    med_acc = 100.0 * cohort_correct["med"] / max(1, cohort_correct["med_total"])
    tail_acc = 100.0 * cohort_correct["tail"] / max(1, cohort_correct["tail_total"])

    print("\n==================================================")
    print(f"      AquaLens v4 Benchmark Results ({total} Test Micrographs)")
    print("==================================================")
    print(f"Overall Top-1 Accuracy:       {top1_acc:.2f}%")
    print(f"Overall Top-3 Accuracy:       {top3_acc:.2f}%")
    print(f"Overall Top-5 Accuracy:       {top5_acc:.2f}%")
    print("--------------------------------------------------")
    print(f"Head Cohort (>=20 samples):   {head_acc:.2f}% ({cohort_correct['head']}/{cohort_correct['head_total']})")
    print(f"Med Cohort (10-19 samples):   {med_acc:.2f}% ({cohort_correct['med']}/{cohort_correct['med_total']})")
    print(f"Tail / Few-Shot (<10 samples):{tail_acc:.2f}% ({cohort_correct['tail']}/{cohort_correct['tail_total']})")
    print("==================================================")

    # Save Classification Report
    report = classification_report(all_targets, all_preds, target_names=classes, zero_division=0)
    rep_path = os.path.join(config.CHECKPOINT_DIR, "classification_report.txt")
    with open(rep_path, "w") as f:
        f.write("AquaLens v4 Final Evaluation Report\n")
        f.write(f"Top-1: {top1_acc:.2f}%, Top-3: {top3_acc:.2f}%, Top-5: {top5_acc:.2f}%\n")
        f.write(f"Head Accuracy: {head_acc:.2f}%, Med Accuracy: {med_acc:.2f}%, Tail (Few-Shot): {tail_acc:.2f}%\n\n")
        f.write(report)
    print(f"[Saved] Classification report saved to: {rep_path}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"AquaLens v4 Confusion Matrix ({num_classes} Species) - Top-1: {top1_acc:.1f}%", fontsize=12, fontweight='bold')
    plt.colorbar()
    plt.xlabel("Predicted Species Index")
    plt.ylabel("True Species Index")
    plt.tight_layout()
    cm_path = os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Saved] Confusion matrix plot saved to: {cm_path}")

if __name__ == "__main__":
    evaluate_v4_system()
