import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from dataset import FishScaleDatasetV4
from inference import FishClassifierV4


def evaluate_v4_system():
    print("=== AquaLens v4.1 Strictly Leakage-Free Evaluation ===")

    test_ds = FishScaleDatasetV4(
        config.ROOT_DIR,
        split="test",
        seed=config.RANDOM_SEED,
        use_roi_crop=False,
        use_clahe=False,
    )
    classes = test_ds.classes
    full_counts = test_ds.full_counts
    num_classes = len(classes)

    classifier = FishClassifierV4()

    # Use actual species sample counts (NOT alphabetical indices!) for cohorts
    head_indices = {i for i, cnt in enumerate(full_counts) if cnt >= 20}
    med_indices = {i for i, cnt in enumerate(full_counts) if 10 <= cnt < 20}
    tail_indices = {i for i, cnt in enumerate(full_counts) if cnt < 10}

    print(f"[Cohorts] Head (>=20): {len(head_indices)} species | Med (10-19): {len(med_indices)} species | Tail (<10): {len(tail_indices)} species")

    all_targets, all_preds, all_sims = [], [], []
    top3_matches, top5_matches = 0, 0
    cohort_stats = {"head": [0, 0], "med": [0, 0], "tail": [0, 0]}

    # Use direct inference path matching WebApp production flow (auto_crop=True)
    for img_path, lbl in zip(test_ds.image_paths, test_ds.labels):
        pred_cls, _, _, max_sim, all_sims_dict, _ = classifier.predict(
            img_path, auto_crop=classifier.use_roi_crop, return_metadata=True
        )
        ranked_names = [k for k, _ in sorted(all_sims_dict.items(), key=lambda x: x[1], reverse=True)]
        gt_name = classes[lbl]
        pred_idx = test_ds.class_to_idx.get(pred_cls, -1)

        all_targets.append(lbl)
        all_preds.append(pred_idx)
        all_sims.append(float(max_sim))

        if lbl in head_indices:
            key = "head"
        elif lbl in med_indices:
            key = "med"
        else:
            key = "tail"

        cohort_stats[key][1] += 1
        if pred_idx == lbl:
            cohort_stats[key][0] += 1

        if gt_name in ranked_names[:3]:
            top3_matches += 1
        if gt_name in ranked_names[:5]:
            top5_matches += 1

    total = len(all_targets)
    top1_acc = 100.0 * float(np.mean(np.array(all_preds) == np.array(all_targets)))
    top3_acc = 100.0 * top3_matches / total
    top5_acc = 100.0 * top5_matches / total

    head_acc = 100.0 * cohort_stats["head"][0] / max(1, cohort_stats["head"][1])
    med_acc = 100.0 * cohort_stats["med"][0] / max(1, cohort_stats["med"][1])
    tail_acc = 100.0 * cohort_stats["tail"][0] / max(1, cohort_stats["tail"][1])

    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
    m_prec, m_rec, m_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    print("\n==================================================")
    print(f" AquaLens v4.1 Benchmark Results ({total} Held-Out Test Micrographs)")
    print("==================================================")
    print(f"Top-1 Accuracy (Train-Only Prototypes): {top1_acc:.2f}% ({sum(np.array(all_preds)==np.array(all_targets))}/{total})")
    print(f"Top-3 / Top-5 Accuracy:                 {top3_acc:.2f}% / {top5_acc:.2f}%")
    print(f"Weighted / Macro F1:                    {w_f1:.4f} / {m_f1:.4f}")
    print(f"Head (>=20): {head_acc:.2f}% ({cohort_stats['head'][0]}/{cohort_stats['head'][1]})")
    print(f"Med (10-19): {med_acc:.2f}% ({cohort_stats['med'][0]}/{cohort_stats['med'][1]})")
    print(f"Tail (<10):  {tail_acc:.2f}% ({cohort_stats['tail'][0]}/{cohort_stats['tail'][1]})")
    print("==================================================")

    report = classification_report(all_targets, all_preds, labels=range(num_classes), target_names=classes, zero_division=0)
    rep_path = os.path.join(config.CHECKPOINT_DIR, "classification_report.txt")
    with open(rep_path, "w") as f:
        f.write("AquaLens v4.1 Strictly Leakage-Free Evaluation Report (Train-Only Prototypes + ScaleDetector ROI)\n")
        f.write(f"Top-1: {top1_acc:.2f}%, Top-3: {top3_acc:.2f}%, Top-5: {top5_acc:.2f}%\n")
        f.write(f"Weighted F1: {w_f1:.4f}, Macro F1: {m_f1:.4f}\n")
        f.write(f"Head Accuracy: {head_acc:.2f}%, Med Accuracy: {med_acc:.2f}%, Tail (Few-Shot): {tail_acc:.2f}%\n\n")
        f.write(report)

    summary = {
        "test_total": total,
        "test_correct": int(sum(np.array(all_preds) == np.array(all_targets))),
        "top1_accuracy_pct": round(top1_acc, 2),
        "top3_accuracy_pct": round(top3_acc, 2),
        "top5_accuracy_pct": round(top5_acc, 2),
        "weighted_precision": round(float(w_prec), 4),
        "weighted_recall": round(float(w_rec), 4),
        "weighted_f1": round(float(w_f1), 4),
        "macro_precision": round(float(m_prec), 4),
        "macro_recall": round(float(m_rec), 4),
        "macro_f1": round(float(m_f1), 4),
        "head_accuracy_pct": round(head_acc, 2),
        "head_counts": cohort_stats["head"],
        "med_accuracy_pct": round(med_acc, 2),
        "med_counts": cohort_stats["med"],
        "tail_accuracy_pct": round(tail_acc, 2),
        "tail_counts": cohort_stats["tail"],
        "mean_top1_sim": round(float(np.mean(all_sims)), 4),
    }
    with open(os.path.join(config.CHECKPOINT_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    cm = confusion_matrix(all_targets, all_preds, labels=range(num_classes))
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"AquaLens v4.1 Confusion Matrix ({num_classes} Species) - Top-1: {top1_acc:.1f}%", fontsize=12, fontweight='bold')
    plt.colorbar()
    plt.xlabel("Predicted Species Index")
    plt.ylabel("True Species Index")
    plt.tight_layout()
    plt.savefig(os.path.join(config.CHECKPOINT_DIR, "confusion_matrix.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    evaluate_v4_system()
