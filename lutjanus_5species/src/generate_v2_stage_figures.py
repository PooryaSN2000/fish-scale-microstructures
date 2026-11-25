import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.manifold import TSNE
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from torchvision.transforms import v2

# Setup Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.abspath(os.path.join(SRC_DIR, ".."))
PAPER_IMG_DIR = os.path.join(V2_DIR, "paper", "img")
CHECKPOINTS_DIR = os.path.join(V2_DIR, "checkpoints")
PROJECT_ROOT = os.path.abspath(os.path.join(V2_DIR, ".."))
ROOT_DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")

os.makedirs(PAPER_IMG_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from dataset import FishScaleDataset, apply_clahe_preprocessing
from inference import FishClassifier
from gradcam import GradCAMExplainer5Class
from scale_detector import ScaleDetector
from augmentations import fourier_amplitude_swap

# Publication Styling: clean, large bold fonts, narrow IEEE-ready aspect ratios
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["xtick.labelsize"] = 11
plt.rcParams["ytick.labelsize"] = 11
plt.rcParams["legend.fontsize"] = 10

STAGES_INFO = [
    {
        "stage": "Stage 1",
        "code": "S1",
        "species": "Lutjanus johni Bloch 1792",
        "latin": "L. johni",
        "full_latin": "Lutjanus johni",
        "author": "Bloch, 1792",
        "samples": 127,
        "folder": "Lutjanus johni Bloch 1792",
        "color": "#1f77b4",
    },
    {
        "stage": "Stage 2",
        "code": "S2",
        "species": "Lutjanus lutjanus Bloch 1790",
        "latin": "L. lutjanus",
        "full_latin": "Lutjanus lutjanus",
        "author": "Bloch, 1790",
        "samples": 66,
        "folder": "Lutjanus lutjanus Bloch 1790",
        "color": "#2ca02c",
    },
    {
        "stage": "Stage 3",
        "code": "S3",
        "species": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "latin": "L. lemniscatus",
        "full_latin": "Lutjanus lemniscatus",
        "author": "(Valenciennes, 1828)",
        "samples": 21,
        "folder": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "color": "#ff7f0e",
    },
    {
        "stage": "Stage 4",
        "code": "S4",
        "species": "Lutjanus malabaricus (Bloch & Schneider, 1801)",
        "latin": "L. malabaricus",
        "full_latin": "Lutjanus malabaricus",
        "author": "(Bloch & Schneider, 1801)",
        "samples": 14,
        "folder": "Lutjanus malabaricus (Bloch & Schneider, 1801)",
        "color": "#9467bd",
    },
    {
        "stage": "Stage 5",
        "code": "S5",
        "species": "Lutjanus fulviflammus(Forsskal 1775)",
        "latin": "L. fulviflammus",
        "full_latin": "Lutjanus fulviflammus",
        "author": "(Forsskal, 1775)",
        "samples": 5,
        "folder": "Lutjanus fulviflammus(Forsskal 1775)",
        "color": "#d62728",
    },
]


def save_paper_fig(fig, filename):
    p = os.path.join(PAPER_IMG_DIR, filename)
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    print(f"[Saved] {filename} -> paper/img/")


def draw_legend_card(ax, title="Species Legend"):
    ax.axis("off")
    handles = [
        Patch(facecolor=item["color"], edgecolor="#0f172a", linewidth=0.8,
              label=f"{item['code']} ({item['stage']}): {item['latin']} (N={item['samples']})")
        for item in STAGES_INFO
    ]
    leg = ax.legend(
        handles=handles,
        loc="center",
        frameon=True,
        facecolor="#f8fafc",
        edgecolor="#94a3b8",
        framealpha=1.0,
        fontsize=10.2,
        title=title,
        title_fontsize=11.2,
        borderpad=1.0,
        labelspacing=0.85,
    )
    leg.get_title().set_fontweight("bold")
    for text in leg.get_texts():
        text.set_fontweight("bold")


def _stage_index_for_name(cname: str) -> int:
    c_low = cname.lower()
    if "johni" in c_low:
        return 0
    if "lutjanus lutjanus" in c_low:
        return 1
    if "lemniscatus" in c_low:
        return 2
    if "malabar" in c_low:
        return 3
    if "fulviflammus" in c_low:
        return 4
    raise ValueError(f"Unknown species name: {cname}")


# 1. Dataset Distribution
def plot_dataset_distribution_stage():
    print("[1/12] Generating Stage-labeled Dataset Distribution...")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    stages = [item["stage"] for item in STAGES_INFO]
    counts = [item["samples"] for item in STAGES_INFO]
    colors = [item["color"] for item in STAGES_INFO]

    bars = ax.bar(stages, counts, color=colors, width=0.58, edgecolor="black", linewidth=0.9, alpha=0.92)
    ax.set_ylabel("Number of Scale Micrographs", fontsize=12, fontweight="bold")
    ax.set_xlabel("Taxonomic Stage (Lutjanidae Family)", fontsize=12, fontweight="bold")
    ax.set_title("Sample Distribution Across 5 Lutjanus Stages", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim([0, 148])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=12, fontweight="bold")

    legend_patches = [
        Patch(facecolor=item["color"], edgecolor="black", linewidth=0.6,
              label=f"{item['stage']} ({item['code']}): {item['full_latin']}")
        for item in STAGES_INFO
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.96, fontsize=9.8,
              title="Species Legend", title_fontsize=10.5)
    fig.tight_layout()
    save_paper_fig(fig, "dataset_distribution.png")


# 2. Dataset Samples Grid (Localized Scale ROIs + Legend card)
def plot_dataset_samples_stage():
    print("[2/12] Generating Narrow Localized Dataset Samples Grid...")
    det = ScaleDetector(padding_ratio=0.12)
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.2))
    axes_flat = axes.flatten()

    for idx, item in enumerate(STAGES_INFO):
        ax = axes_flat[idx]
        folder = os.path.join(ROOT_DATA_DIR, item["folder"])
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        chosen = files[0] if idx == 0 else files[len(files) // 2]
        img = det.crop(Image.open(os.path.join(folder, chosen)).convert("RGB"), square=True).resize((336, 336), Image.Resampling.BILINEAR)
        ax.imshow(img)
        ax.set_title(f"{item['stage']} ({item['code']})", fontsize=12.5, fontweight="bold", color=item["color"], pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(item["color"])
            spine.set_linewidth(2.2)
        ax.set_xticks([])
        ax.set_yticks([])

    draw_legend_card(axes_flat[5], title="Stage Legend (Lutjanus Taxa)")
    fig.suptitle("Localized Scale ROIs Across the 5 Stages", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    save_paper_fig(fig, "dataset_samples.png")


# 3. Augmentation Showcase (Narrow 3x2 showing HUD ROI, CLAHE, Geometric, FDA+Color, CutMix)
def plot_augmentation_samples_stage():
    print("[3/12] Generating Narrow Domain Augmentation Showcase...")
    det = ScaleDetector(padding_ratio=0.12)
    f1_dir = os.path.join(ROOT_DATA_DIR, STAGES_INFO[0]["folder"])
    f4_dir = os.path.join(ROOT_DATA_DIR, STAGES_INFO[3]["folder"])
    raw1 = Image.open(os.path.join(f1_dir, sorted(os.listdir(f1_dir))[0])).convert("RGB")
    raw4 = Image.open(os.path.join(f4_dir, sorted(os.listdir(f4_dir))[0])).convert("RGB")

    hud1 = det.visualize_bbox(raw1).resize((336, 336), Image.Resampling.BILINEAR)
    roi1 = det.crop(raw1, square=True).resize((336, 336), Image.Resampling.BILINEAR)
    roi4 = det.crop(raw4, square=True).resize((336, 336), Image.Resampling.BILINEAR)
    clahe1 = apply_clahe_preprocessing(roi1)
    clahe4 = apply_clahe_preprocessing(roi4)

    to_t = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    t1, t4 = to_t(clahe1), to_t(clahe4)

    torch.manual_seed(7)
    spatial_aug = v2.Compose([
        v2.RandomResizedCrop((336, 336), scale=(0.82, 1.0), antialias=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(p=1.0),
    ])
    fda_t = fourier_amplitude_swap(t1, t4, beta=0.08)
    cutmix_t, _ = v2.CutMix(num_classes=2, alpha=1.0)(torch.stack([t1, t4]), torch.tensor([0, 1]))

    def _to_pil(t):
        return Image.fromarray((t.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(np.uint8))

    panels = [
        (hud1, "1. Slide + HUD Box (Stage 1)"),
        (clahe1, "2. Scale ROI + CLAHE (Stage 1)"),
        (_to_pil(spatial_aug(t1)), "3. Rot/Flip Invariance (Stage 1)"),
        (ImageEnhance.Color(_to_pil(fda_t)).enhance(1.3), "4. FDA Style + Color Shift (S1+S4)"),
        (clahe4, "5. Scale ROI + CLAHE (Stage 4)"),
        (_to_pil(cutmix_t[0]), "6. CutMix Regularization (S1+S4)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(6.8, 8.0))
    for ax, (im, title) in zip(axes.flatten(), panels):
        ax.imshow(im)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

    fig.suptitle("Scale Localization & Domain Generalization Pipeline", fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout()
    save_paper_fig(fig, "augmentation_samples.png")


def evaluate_splits():
    classifier = FishClassifier(
        model_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
        prototypes_path=os.path.join(CHECKPOINTS_DIR, "prototypes.pth"),
    )
    proto_to_stage = {cname: _stage_index_for_name(cname) for cname in classifier.class_names}
    ordered_proto_names = [next(c for c, s in proto_to_stage.items() if s == i) for i in range(5)]

    results = {}
    for split_name in ["test", "all"]:
        ds = FishScaleDataset(
            root_dir=config.ROOT_DIR, split=split_name, seed=config.RANDOM_SEED,
            use_clahe=False, use_roi_crop=False,
        )
        targets, preds, probs, paths, raw_sims = [], [], [], [], []
        ds_lbl_to_stage = {ds.class_to_idx[cname]: proto_to_stage[cname] for cname in ds.classes}

        for img_path, lbl in zip(ds.image_paths, ds.labels):
            pred_name, conf, prob_dict, max_sim, sim_dict = classifier.predict(img_path)
            targets.append(ds_lbl_to_stage[lbl])
            preds.append(proto_to_stage[pred_name])
            probs.append([prob_dict[ordered_proto_names[s]] for s in range(5)])
            raw_sims.append([sim_dict[ordered_proto_names[s]] for s in range(5)])
            paths.append(img_path)

        results[split_name] = {
            "targets": np.array(targets),
            "preds": np.array(preds),
            "probs": np.array(probs),
            "sims": np.array(raw_sims),
            "paths": paths,
        }
    return classifier, results


# 4. Confusion Matrix (Test Split)
def plot_confusion_matrix_stage(targets, preds, acc_pct):
    print("[4/12] Generating Stage-labeled Test Confusion Matrix...")
    cm = confusion_matrix(targets, preds, labels=range(5))
    cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(6.3, 5.5))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Recall", rotation=270, labelpad=16, fontsize=11.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=10.5)

    stage_labels = [item["stage"] for item in STAGES_INFO]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(stage_labels, fontsize=11.5, fontweight="bold")
    ax.set_yticklabels(stage_labels, fontsize=11.5, fontweight="bold")
    ax.set_xlabel("Predicted Stage", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Stage", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_title(f"Test Confusion Matrix (Acc: {acc_pct:.2f}%)", fontsize=13, fontweight="bold", pad=12)

    thresh = cm_norm.max() / 2.0
    for i in range(5):
        for j in range(5):
            val, cnt = cm_norm[i, j], cm[i, j]
            txt = f"{val:.2f}\n({cnt})" if cnt > 0 else "0.00"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=11, fontweight="bold")

    fig.tight_layout()
    save_paper_fig(fig, "confusion_matrix.png")


# 5. Multi-Class ROC Curves
def plot_roc_curves_stage(targets, probs):
    print("[5/12] Generating Stage-labeled ROC Curves...")
    targets_bin = label_binarize(targets, classes=range(5))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    aucs = []
    for i in range(5):
        fpr, tpr, _ = roc_curve(targets_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        item = STAGES_INFO[i]
        ax.plot(fpr, tpr, color=item["color"], lw=2.5,
                label=f"{item['stage']} ({item['latin']}): AUC={roc_auc:.2f}")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_title("One-vs-Rest ROC Curves (5 Stages)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.96, fontsize=9.6)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    save_paper_fig(fig, "roc_curves.png")
    return aucs


# 6. Precision-Recall Curves (Crucial for Imbalanced 25:1 Data)
def plot_pr_curves_stage(targets, probs):
    print("[6/12] Generating Stage-labeled Precision-Recall Curves...")
    targets_bin = label_binarize(targets, classes=range(5))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    aps = []
    for i in range(5):
        prec, rec, _ = precision_recall_curve(targets_bin[:, i], probs[:, i])
        ap = average_precision_score(targets_bin[:, i], probs[:, i])
        aps.append(ap)
        item = STAGES_INFO[i]
        ax.plot(rec, prec, color=item["color"], lw=2.5,
                label=f"{item['stage']} ({item['latin']}): AP={ap:.2f}")

    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=12, fontweight="bold")
    ax.set_title("Precision-Recall Curves Across 5 Stages", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower left", framealpha=0.96, fontsize=9.6)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    save_paper_fig(fig, "pr_curves.png")
    return aps


# 7. Per-Stage F1-Score Bar Chart
def plot_class_performance_stage(targets, preds):
    print("[7/12] Generating Stage-labeled F1-Score Bar Chart...")
    f1s = f1_score(targets, preds, average=None, labels=range(5))
    mean_f1 = float(np.mean(f1s))

    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    stages = [item["stage"] for item in STAGES_INFO]
    colors = [item["color"] for item in STAGES_INFO]
    bars = ax.bar(stages, f1s, color=colors, width=0.58, edgecolor="black", linewidth=0.9, alpha=0.92)
    ax.axhline(y=mean_f1, color="#e11d48", linestyle="--", lw=2.2)

    ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Taxonomic Stage (Lutjanidae Family)", fontsize=12, fontweight="bold")
    ax.set_title("Test F1-Score Performance by Stage", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim([0, 1.18])
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                    textcoords="offset points", ha="center", va="bottom", fontsize=12, fontweight="bold")

    legend_patches = [
        Patch(facecolor=item["color"], edgecolor="black", linewidth=0.6, label=f"{item['stage']}: {item['latin']}")
        for item in STAGES_INFO
    ]
    legend_patches.append(plt.Line2D([0], [0], color="#e11d48", linestyle="--", lw=2.2, label=f"Mean F1 ({mean_f1:.2f})"))
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.96, fontsize=9.6)
    fig.tight_layout()
    save_paper_fig(fig, "class_performance.png")
    return f1s.tolist()


# 8. t-SNE Metric Manifold
def plot_tsne_stage(classifier, paths, targets):
    print("[8/12] Generating Stage-labeled t-SNE Projection...")
    all_embs, all_stages = [], []
    for s_idx in range(5):
        idx_list = [k for k, t in enumerate(targets) if t == s_idx][:30]
        for k in idx_list:
            img = Image.open(paths[k]).convert("RGB")
            emb = classifier.extract_embedding(img).cpu().numpy().squeeze(0)
            all_embs.append(emb)
            all_stages.append(s_idx)

    sample_embs = np.array(all_embs)
    proto_to_stage = {_stage_index_for_name(cname): p_idx for p_idx, cname in enumerate(classifier.class_names)}
    ordered_protos = torch.stack([classifier.prototypes[proto_to_stage[s]] for s in range(5)]).cpu().numpy()

    combined = np.vstack([sample_embs, ordered_protos])
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="pca", learning_rate="auto")
    reduced = tsne.fit_transform(combined)
    samples_2d, protos_2d = reduced[:len(sample_embs)], reduced[len(sample_embs):]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for s_idx in range(5):
        mask = [s == s_idx for s in all_stages]
        item = STAGES_INFO[s_idx]
        ax.scatter(samples_2d[mask, 0], samples_2d[mask, 1], color=item["color"], s=52,
                   alpha=0.78, edgecolors="white", linewidth=0.5)
        ax.scatter(protos_2d[s_idx, 0], protos_2d[s_idx, 1], color=item["color"],
                   marker="X", s=220, edgecolors="black", linewidth=1.6, zorder=5)

    ax.set_title("t-SNE Metric Space (5 Lutjanus Stages)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12, fontweight="bold")
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    legend_handles = [
        Patch(facecolor=item["color"], edgecolor="black", linewidth=0.6, label=f"{item['stage']}: {item['latin']}")
        for item in STAGES_INFO
    ]
    legend_handles.append(plt.Line2D([0], [0], marker="X", color="w", markeredgecolor="black",
                                     markerfacecolor="gray", markersize=11, label="Prototype Centroid (X)"))
    ax.legend(handles=legend_handles, loc="best", framealpha=0.95, fontsize=9.2)
    fig.tight_layout()
    save_paper_fig(fig, "tsne_embeddings.png")


# 9. Congeneric Statistical Analysis (Intra/Inter Density + 5x5 Pairwise Similarity Matrix)
def plot_similarity_statistics(test_res, gen_stats):
    print("[9/12] Generating Congeneric Statistical Separation & Centroid Matrix...")
    cs = gen_stats["congeneric_statistics"]
    targets = test_res["targets"]
    sims = test_res["sims"]

    intra_vals = [sims[i, targets[i]] for i in range(len(targets))]
    inter_vals = [sims[i, j] for i in range(len(targets)) for j in range(5) if j != targets[i]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 8.0), gridspec_kw={"height_ratios": [1.0, 1.15]})

    bins = np.linspace(-0.35, 1.02, 26)
    ax1.hist(inter_vals, bins=bins, density=True, alpha=0.68, color="#dc2626", edgecolor="black", linewidth=0.7,
             label=f"Inter-Stage ({cs['inter_mean']:.2f} ± {cs['inter_std']:.2f})")
    ax1.hist(intra_vals, bins=bins, density=True, alpha=0.72, color="#059669", edgecolor="black", linewidth=0.7,
             label=f"Intra-Stage ({cs['intra_mean']:.2f} ± {cs['intra_std']:.2f})")
    ax1.axvline(cs["inter_mean"], color="#991b1b", linestyle="--", lw=2)
    ax1.axvline(cs["intra_mean"], color="#065f46", linestyle="--", lw=2)
    ax1.set_title(f"Intra- vs. Inter-Stage Similarity (t={cs['welch_t']}, d={cs['cohen_d']})",
                  fontsize=12.5, fontweight="bold", pad=10)
    ax1.set_xlabel("Cosine Similarity to Stage Prototype", fontsize=11.5, fontweight="bold")
    ax1.set_ylabel("Probability Density", fontsize=11.5, fontweight="bold")
    ax1.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
    ax1.grid(True, linestyle="--", alpha=0.45)

    mat = np.array(cs["pairwise_centroid_similarity"])
    im = ax2.imshow(mat, cmap="YlGnBu", vmin=-0.3, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label("Centroid Cosine Similarity", rotation=270, labelpad=15, fontsize=11, fontweight="bold")
    stage_codes = [item["code"] for item in STAGES_INFO]
    ax2.set_xticks(range(5))
    ax2.set_yticks(range(5))
    ax2.set_xticklabels(stage_codes, fontsize=11.5, fontweight="bold")
    ax2.set_yticklabels(stage_codes, fontsize=11.5, fontweight="bold")
    ax2.set_title("5x5 Congeneric Prototype Affinity Matrix", fontsize=12.5, fontweight="bold", pad=10)

    for i in range(5):
        for j in range(5):
            val = mat[i, j]
            ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                     color="white" if val > 0.50 else "black", fontsize=11, fontweight="bold")

    fig.tight_layout()
    save_paper_fig(fig, "similarity_statistics.png")


# 10. Robustness Stress Test & Background-Masking Proof
def plot_robustness_benchmark(gen_stats):
    print("[10/12] Generating Domain-Shift Robustness & Background Ablation Chart...")
    stress = gen_stats["robustness_stress_tests"]
    cv = gen_stats["cross_validation_5fold"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 7.6))

    # Top panel: Camera / Lighting Domain Shifts
    dom_keys = [
        ("Clean Scale ROI (Standard)", "Clean ROI\n(Baseline)", "#059669"),
        ("Dark / Low-Gamma Sensor (g=1.45)", "Dark Sensor\n(Gamma 1.45)", "#0284c7"),
        ("Bright Warm CMOS Shift (g=0.72)", "Warm CMOS\n(Gamma 0.72)", "#0ea5e9"),
        ("Optical Defocus Blur (r=1.2)", "Optical Blur\n(r=1.2)", "#6366f1"),
        ("Combined Unseen Microscope", "Combined\nStress Shift", "#8b5cf6"),
    ]
    labels1 = [lbl for _, lbl, _ in dom_keys]
    vals1 = [stress[k]["accuracy_pct"] for k, _, _ in dom_keys]
    cols1 = [c for _, _, c in dom_keys]

    bars1 = ax1.bar(labels1, vals1, color=cols1, width=0.55, edgecolor="black", linewidth=0.9)
    ax1.axhline(cv["mean_accuracy_pct"], color="#e11d48", linestyle="--", lw=2.0,
                label=f"5-Fold OOF Mean ({cv['mean_accuracy_pct']:.1f}% ± {cv['std_accuracy_pct']:.1f}%)")
    ax1.set_ylim([0, 114])
    ax1.set_ylabel("Test Accuracy (%)", fontsize=11.5, fontweight="bold")
    ax1.set_title("Cross-Camera & Sensor Perturbation Robustness", fontsize=12.5, fontweight="bold", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", labelsize=9.5)
    ax1.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=10.5, fontweight="bold")

    # Bottom panel: Scale ROI vs Uncropped vs Background/Center Masking
    bg_keys = [
        ("Clean Scale ROI (Standard)", "Localized ROI\n(Scale Body)", "#059669"),
        ("Full Uncropped Slide", "Uncropped Slide\n(With Rulers)", "#d97706"),
        ("Scale-Masked Background Only", "Slide Scale-Masked\n(Margins Only)", "#ea580c"),
    ]
    if "ROI Interior Masked (Blank Center)" in stress:
        bg_keys.append(("ROI Interior Masked (Blank Center)", "ROI Blank Center\n(No Scale/Margins)", "#dc2626"))
    labels2 = [lbl for _, lbl, _ in bg_keys]
    vals2 = [stress[k]["accuracy_pct"] for k, _, _ in bg_keys]
    cols2 = [c for _, _, c in bg_keys]

    bars2 = ax2.bar(labels2, vals2, color=cols2, width=0.50, edgecolor="black", linewidth=0.9)
    ax2.axhline(20.0, color="#475569", linestyle=":", lw=1.8, label="Random Chance (5-Way = 20%)")
    ax2.set_ylim([0, 114])
    ax2.set_ylabel("Test Accuracy (%)", fontsize=11.5, fontweight="bold")
    ax2.set_title("Background-Shortcut Ablation Proof (Scale vs. Borders)", fontsize=12.5, fontweight="bold", pad=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.tick_params(axis="x", labelsize=9.2)
    ax2.legend(loc="upper right", fontsize=9.5, framealpha=0.95)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=11, fontweight="bold")

    fig.tight_layout()
    save_paper_fig(fig, "robustness_benchmark.png")


# 11. Prediction Confidence Showcase (3x2: 5 Localized Test Predictions + Legend Card)
def plot_prediction_confidence_stage(test_res):
    print("[11/12] Generating Narrow Stage-labeled Prediction Showcase...")
    det = ScaleDetector(padding_ratio=0.12)
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 8.4))
    axes_flat = axes.flatten()

    targets, preds, probs, paths = test_res["targets"], test_res["preds"], test_res["probs"], test_res["paths"]
    for s_idx in range(5):
        ax = axes_flat[s_idx]
        candidates = [k for k, t in enumerate(targets) if t == s_idx]
        correct_cands = [k for k in candidates if preds[k] == targets[k]]
        chosen = max(correct_cands or candidates, key=lambda k: probs[k][preds[k]])
        p_stage = int(preds[chosen])
        conf = probs[chosen][p_stage] * 100.0
        img = det.crop(Image.open(paths[chosen]).convert("RGB"), square=True).resize((336, 336), Image.Resampling.BILINEAR)
        ax.imshow(img)
        color = "#15803d" if p_stage == s_idx else "#b91c1c"
        ax.set_title(f"True: {STAGES_INFO[s_idx]['code']} | Pred: {STAGES_INFO[p_stage]['code']}\nConf: {conf:.1f}%",
                     fontsize=12, fontweight="bold", color=color, pad=6)
        for spine in ax.spines.values():
            spine.set_edgecolor(STAGES_INFO[s_idx]["color"])
            spine.set_linewidth(2.2)
        ax.set_xticks([])
        ax.set_yticks([])

    draw_legend_card(axes_flat[5], title="Stage Legend (Test Predictions)")
    fig.suptitle("Held-Out Test Predictions on Localized ROIs", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    save_paper_fig(fig, "prediction_confidence.png")


# 12. 4-Column Localized Grad-CAM Gallery (Slide+HUD | Localized ROI | Grad-CAM | Overlay)
def plot_gradcam_5stages(classifier):
    print("[12/12] Generating 4-Column Localized Grad-CAM Gallery Across 5 Stages...")
    det = ScaleDetector(padding_ratio=0.12)
    explainer = GradCAMExplainer5Class(classifier.model, classifier.device, image_size=classifier.image_size)
    proto_map = {_stage_index_for_name(cname): classifier.prototypes[p_idx] for p_idx, cname in enumerate(classifier.class_names)}

    fig, axes = plt.subplots(5, 4, figsize=(7.6, 9.8))
    col_titles = ["(a) Slide + HUD", "(b) Scale ROI", "(c) Grad-CAM", "(d) Focus Overlay"]

    for s_idx, item in enumerate(STAGES_INFO):
        folder = os.path.join(ROOT_DATA_DIR, item["folder"])
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
        chosen_f = files[0] if s_idx == 0 else files[len(files) // 2]
        raw_pil = Image.open(os.path.join(folder, chosen_f)).convert("RGB")

        hud_vis = np.array(det.visualize_bbox(raw_pil).resize(classifier.image_size, Image.Resampling.BILINEAR))
        roi_pil = det.crop(raw_pil, square=True)
        roi_proc = apply_clahe_preprocessing(roi_pil) if classifier.use_clahe else roi_pil
        roi_arr, hmap_arr, ov_arr = explainer.generate(roi_proc, proto_map[s_idx])

        for col_idx, arr in enumerate([hud_vis, roi_arr, hmap_arr, ov_arr]):
            ax = axes[s_idx, col_idx]
            ax.imshow(arr)
            if s_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=10.8, fontweight="bold", pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{item['stage']}\n({item['latin']})", fontsize=10, fontweight="bold", color=item["color"], labelpad=6)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("HUD Scale Localization & Grad-CAM Across 5 Stages", fontsize=13.5, fontweight="bold", y=0.995)
    fig.tight_layout()
    save_paper_fig(fig, "gradcam_stages.png")


def main():
    print("=== Generating Complete Publication Figures for 5-Species Lutjanidae Paper ===")
    gen_stats_path = os.path.join(CHECKPOINTS_DIR, "generalization_stats.json")
    with open(gen_stats_path) as f:
        gen_stats = json.load(f)

    plot_dataset_distribution_stage()
    plot_dataset_samples_stage()
    plot_augmentation_samples_stage()

    classifier, results = evaluate_splits()
    test_res, all_res = results["test"], results["all"]

    test_acc = float(np.mean(test_res["targets"] == test_res["preds"]) * 100.0)
    all_acc = float(np.mean(all_res["targets"] == all_res["preds"]) * 100.0)
    prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(test_res["targets"], test_res["preds"], average="weighted", zero_division=0)
    macro_prec_t, macro_rec_t, macro_f1_t, _ = precision_recall_fscore_support(test_res["targets"], test_res["preds"], average="macro", zero_division=0)

    plot_confusion_matrix_stage(test_res["targets"], test_res["preds"], test_acc)
    test_aucs = plot_roc_curves_stage(test_res["targets"], test_res["probs"])
    test_aps = plot_pr_curves_stage(test_res["targets"], test_res["probs"])
    test_per_f1 = plot_class_performance_stage(test_res["targets"], test_res["preds"])
    plot_tsne_stage(classifier, all_res["paths"], all_res["targets"])
    plot_similarity_statistics(test_res, gen_stats)
    plot_robustness_benchmark(gen_stats)
    plot_prediction_confidence_stage(test_res)
    plot_gradcam_5stages(classifier)

    summary = {
        "test_total": int(len(test_res["targets"])),
        "test_correct": int(np.sum(test_res["targets"] == test_res["preds"])),
        "test_accuracy_pct": round(test_acc, 2),
        "test_weighted_precision": round(float(prec_t), 4),
        "test_weighted_recall": round(float(rec_t), 4),
        "test_weighted_f1": round(float(f1_t), 4),
        "test_macro_precision": round(float(macro_prec_t), 4),
        "test_macro_recall": round(float(macro_rec_t), 4),
        "test_macro_f1": round(float(macro_f1_t), 4),
        "test_per_stage_f1": [round(float(x), 4) for x in test_per_f1],
        "test_per_stage_auc": [round(float(x), 4) for x in test_aucs],
        "test_per_stage_ap": [round(float(x), 4) for x in test_aps],
        "all_total": int(len(all_res["targets"])),
        "all_correct": int(np.sum(all_res["targets"] == all_res["preds"])),
        "all_accuracy_pct": round(all_acc, 2),
    }
    with open(os.path.join(CHECKPOINTS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Updated Metrics Summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
