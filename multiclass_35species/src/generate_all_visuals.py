import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import scipy.stats as stats
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, '../..'))
PAPER_IMG_DIR = os.path.join(PROJECT_ROOT, 'multiclass_35species/paper/img')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'multiclass_35species/checkpoints')
os.makedirs(PAPER_IMG_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from dataset import FishScaleDatasetV4, load_rgb_image
from inference import FishClassifierV4
from scale_detector import ScaleDetector
from gradcam import GradCAMExplainer

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8.5
plt.rcParams['ytick.labelsize'] = 8.5
plt.rcParams['legend.fontsize'] = 9


def _clean_binomial(full_name: str) -> str:
    """Extracts clean 'Genus species' binomial without attached authority/parens."""
    import re
    cleaned = re.split(r"[\(\d]", full_name)[0]
    for auth in ("Cuvier", "Bloch", "Forsskal", "Forsskål", "Day", "Hamilton", "Schneider", "Jordan"):
        cleaned = re.sub(rf"(?<=[a-z]){auth}.*$", "", cleaned)
    parts = [p.strip(",") for p in cleaned.split() if p.strip(",")]
    return " ".join(parts[:2]) if len(parts) >= 2 else full_name


def _cohort_of(cnt: int) -> str:
    if cnt >= 20:
        return "head"
    if cnt >= 10:
        return "med"
    return "tail"


def evaluate_test_once(classifier):
    test_ds = FishScaleDatasetV4(
        config.ROOT_DIR, split="test", seed=config.RANDOM_SEED, use_roi_crop=False, use_clahe=False
    )
    all_embs, all_targets, all_preds, all_sims_raw, all_probs = [], [], [], [], []
    top3_ok = 0

    for p, lbl in zip(test_ds.image_paths, test_ds.labels):
        emb = classifier.extract_embedding(load_rgb_image(p), use_tta=True)
        sims = classifier.score_embedding(emb)
        ranked = np.argsort(sims)[::-1]
        pred = int(ranked[0])
        if lbl in ranked[:3]:
            top3_ok += 1

        scaled = np.exp(sims * 20.0)
        probs = scaled / np.sum(scaled)

        all_embs.append(emb.cpu().numpy().squeeze(0))
        all_targets.append(int(lbl))
        all_preds.append(pred)
        all_sims_raw.append(sims)
        all_probs.append(probs)

    return {
        "classes": test_ds.classes,
        "full_counts": test_ds.full_counts,
        "embs": np.array(all_embs),
        "targets": np.array(all_targets),
        "preds": np.array(all_preds),
        "sims": np.array(all_sims_raw),
        "probs": np.array(all_probs),
        "top1_pct": 100.0 * float(np.mean(np.array(all_preds) == np.array(all_targets))),
        "top3_pct": 100.0 * top3_ok / max(1, len(all_targets)),
    }


def generate_dataset_distribution(eval_res):
    print("\n[Fig 1] Generating Verified 35-Species Dataset Distribution...")
    classes = eval_res["classes"]
    counts = eval_res["full_counts"]
    order = np.argsort(counts)[::-1]

    sorted_names = [_clean_binomial(classes[i]) for i in order]
    sorted_counts = [counts[i] for i in order]
    colors = [
        "#0284c7" if c >= 20 else ("#0d9488" if c >= 10 else "#f59e0b")
        for c in sorted_counts
    ]

    fig, ax = plt.subplots(figsize=(15, 6.2))
    bars = ax.bar(range(len(sorted_names)), sorted_counts, color=colors, edgecolor="black", lw=0.5, width=0.72)
    ax.set_xticks(range(len(sorted_names)))
    ax.set_xticklabels(sorted_names, rotation=90, ha="right", fontsize=8)
    ax.set_ylabel("Number of Micrographs", fontsize=11, fontweight="bold")
    ax.set_title(f"Taxonomic Distribution Across {len(classes)} Validated Marine Species (N={sum(counts)})", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for b in bars:
        h = int(b.get_height())
        ax.annotate(f"{h}", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha="center", va="bottom", fontsize=7, fontweight="bold")

    legend_elements = [
        patches.Patch(facecolor="#0284c7", label=f"Head Cohort (>= 20): {sum(1 for c in counts if c >= 20)} taxa"),
        patches.Patch(facecolor="#0d9488", label=f"Medium Cohort (10-19): {sum(1 for c in counts if 10 <= c < 20)} taxa"),
        patches.Patch(facecolor="#f59e0b", label=f"Tail Cohort (7-9): {sum(1 for c in counts if c < 10)} taxa"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.95, fontsize=9.5)
    plt.tight_layout()
    out = os.path.join(PAPER_IMG_DIR, "dataset_distribution.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def generate_dataset_samples(detector):
    print("\n[Fig 2] Generating Localized Dataset Micrographs Grid...")
    target_taxa = [
        "Lutjanus johni Bloch 1792",
        "Upeneus sulphureusCuvier, 1829",
        "Platycephalus indicus (linnaeus, 1758)",
        "Gerres filamentosus Cuvier,1829",
        "Scolopsis vosmeri (Bloch, 1792)",
        "Acanthopagrus Cuvieri(Day, 1878)",
    ]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.4))
    for idx, sp in enumerate(target_taxa):
        ax = axs[idx // 3, idx % 3]
        folder = os.path.join(config.ROOT_DIR, sp)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))])
        im = detector.crop(load_rgb_image(os.path.join(folder, files[0])), padding_ratio=0.12, square=True)
        short = _clean_binomial(sp)
        ax.imshow(im)
        ax.set_title(f"{short} (N={len(files)})", fontsize=10, fontweight="bold")
        ax.axis("off")
    plt.suptitle("Representative Localized Scale Regions (ScaleDetector ROI)", fontsize=12.5, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(PAPER_IMG_DIR, "dataset_samples.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()


def generate_gradcam_gallery(classifier, detector, explainer):
    print("\n[Fig 3] Generating Multi-Species Grad-CAM Explainability Gallery...")
    target_taxa = [
        ("Lutjanus johni Bloch 1792", "Ctenoid Scale (Lutjanidae)", "Circuli annuli & posterior ctenii"),
        ("Upeneus sulphureusCuvier, 1829", "Cycloid Scale (Mullidae)", "Concentric circuli & nuclear focus"),
        ("Platycephalus indicus (linnaeus, 1758)", "Flathead Scale (Platycephalidae)", "Radiating radii & focus"),
        ("Gerres filamentosus Cuvier,1829", "Deep-body Scale (Gerreidae)", "Lateral circuli ridges"),
    ]
    fig, axs = plt.subplots(4, 4, figsize=(14, 13))
    for row_idx, (sp_name, scale_type, bio_note) in enumerate(target_taxa):
        sp_folder = os.path.join(config.ROOT_DIR, sp_name)
        files = sorted([f for f in os.listdir(sp_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))])
        raw_pil = load_rgb_image(os.path.join(sp_folder, files[len(files) // 2]))
        short = _clean_binomial(sp_name)

        axs[row_idx, 0].imshow(raw_pil)
        axs[row_idx, 0].set_title(f"(a) Raw: {short}\n[{scale_type}]", fontsize=9, fontweight="bold", pad=4)
        axs[row_idx, 0].axis("off")

        hud_pil = detector.visualize_detection(raw_pil)
        axs[row_idx, 1].imshow(hud_pil)
        axs[row_idx, 1].set_title("(b) Scale Saliency HUD\n[Border & Ruler Rejection]", fontsize=9, fontweight="bold", pad=4)
        axs[row_idx, 1].axis("off")

        cropped_pil = detector.crop(raw_pil, padding_ratio=0.12, square=True)
        cam_norm, overlay_arr, _, _ = explainer.generate_cam(cropped_pil, classifier.prototypes)

        axs[row_idx, 2].imshow(cam_norm, cmap="jet")
        axs[row_idx, 2].set_title("(c) Grad-CAM Activation\n[Micro-Texture Saliency]", fontsize=9, fontweight="bold", pad=4)
        axs[row_idx, 2].axis("off")

        axs[row_idx, 3].imshow(overlay_arr)
        axs[row_idx, 3].set_title(f"(d) Anatomical Focus\n[{bio_note}]", fontsize=9, fontweight="bold", color="#065f46", pad=4)
        axs[row_idx, 3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "gradcam_explainability.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_metric_tsne(classifier, eval_res):
    print("\n[Fig 4] Generating Metric Space t-SNE Projection...")
    test_embs = eval_res["embs"]
    proto_embs = classifier.prototypes[:classifier.base_class_count].cpu().numpy()
    targets = eval_res["targets"]
    classes = eval_res["classes"]

    combined = np.vstack([test_embs, proto_embs])
    reduced = TSNE(n_components=2, perplexity=18, random_state=42, init="pca", learning_rate="auto").fit_transform(combined)
    test_2d, proto_2d = reduced[: len(test_embs)], reduced[len(test_embs) :]

    plt.figure(figsize=(11.5, 9.5))
    cmap = plt.get_cmap("tab20")
    plt.scatter(test_2d[:, 0], test_2d[:, 1], c=[cmap(l % 20) for l in targets], s=55, alpha=0.78, edgecolors="white", linewidth=0.6, label=f"Held-Out Test Embeddings (N={len(test_embs)})")
    plt.scatter(proto_2d[:, 0], proto_2d[:, 1], c=[cmap(i % 20) for i in range(len(proto_2d))], marker="X", s=160, edgecolors="black", linewidth=1.4, label=f"Train-Only Prototypes (C={len(classes)})")

    for p_idx in [0, 4, 10, 15, 20, 25, 30]:
        short = _clean_binomial(classes[p_idx])
        plt.annotate(short, (proto_2d[p_idx, 0], proto_2d[p_idx, 1]), xytext=(5, 5), textcoords="offset points",
                     fontsize=7.5, fontweight="bold", color="#1e293b",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.85))

    plt.title("AquaLens v4.1 Metric Space: 35-Species Hyperspherical Manifold (t-SNE)", fontsize=12.5, fontweight="bold", pad=10)
    plt.xlabel("t-SNE Dimension 1", fontsize=10, fontweight="bold")
    plt.ylabel("t-SNE Dimension 2", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", framealpha=0.92, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "metric_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_roc_pr_curves(eval_res):
    print("\n[Fig 5] Generating Multi-Class ROC & Precision-Recall Curves...")
    classes = eval_res["classes"]
    n_classes = len(classes)
    targets_bin = label_binarize(eval_res["targets"], classes=range(n_classes))
    probs_arr = eval_res["probs"]

    fpr, tpr, roc_auc = {}, {}, {}
    precision, recall, pr_auc = {}, {}, {}
    for i in range(n_classes):
        if np.sum(targets_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probs_arr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            precision[i], recall[i], _ = precision_recall_curve(targets_bin[:, i], probs_arr[:, i])
            pr_auc[i] = auc(recall[i], precision[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(targets_bin.ravel(), probs_arr.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if i in fpr:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    roc_auc["macro"] = auc(all_fpr, mean_tpr)

    precision["micro"], recall["micro"], _ = precision_recall_curve(targets_bin.ravel(), probs_arr.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))
    for i in range(n_classes):
        if i in fpr:
            ax1.plot(fpr[i], tpr[i], color="#94a3b8", lw=0.7, alpha=0.35)
    ax1.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})', color="#ec4899", lw=2.5, linestyle=":")
    ax1.plot(all_fpr, mean_tpr, label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})', color="#0284c7", lw=2.5)
    ax1.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax1.set_xlim([-0.01, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel("False Positive Rate", fontsize=10, fontweight="bold")
    ax1.set_ylabel("True Positive Rate", fontsize=10, fontweight="bold")
    ax1.set_title("Multi-Class One-vs-Rest ROC Curves (35 Taxa)", fontsize=11, fontweight="bold")
    ax1.legend(loc="lower right", framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.4)

    for i in range(n_classes):
        if i in precision:
            ax2.plot(recall[i], precision[i], color="#94a3b8", lw=0.7, alpha=0.35)
    ax2.plot(recall["micro"], precision["micro"], label=f'Micro-average PR (AP = {pr_auc["micro"]:.4f})', color="#059669", lw=2.8)
    ax2.set_xlim([0.0, 1.01])
    ax2.set_ylim([0.0, 1.03])
    ax2.set_xlabel("Recall (Sensitivity)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Precision (PPV)", fontsize=10, fontweight="bold")
    ax2.set_title("Precision-Recall Curves Across 35 Species", fontsize=11, fontweight="bold")
    ax2.legend(loc="lower left", framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "roc_pr_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_similarity_distribution(eval_res):
    print("\n[Fig 6] Generating Verified Similarity Separation & Cohort Boxplot...")
    targets = eval_res["targets"]
    sims = eval_res["sims"]
    full_counts = eval_res["full_counts"]

    intra_vals, inter_vals = [], []
    cohort_intra = {"head": [], "med": [], "tail": []}

    for i, lbl in enumerate(targets):
        s_true = float(sims[i, lbl])
        intra_vals.append(s_true)
        cohort_intra[_cohort_of(full_counts[lbl])].append(s_true)
        for j in range(sims.shape[1]):
            if j != lbl:
                inter_vals.append(float(sims[i, j]))

    intra_arr, inter_arr = np.array(intra_vals), np.array(inter_vals)
    mean_intra, std_intra = float(np.mean(intra_arr)), float(np.std(intra_arr, ddof=1))
    mean_inter, std_inter = float(np.mean(inter_arr)), float(np.std(inter_arr, ddof=1))
    margin = mean_intra - mean_inter
    pooled = np.sqrt(((len(intra_arr) - 1) * std_intra**2 + (len(inter_arr) - 1) * std_inter**2) / (len(intra_arr) + len(inter_arr) - 2))
    cohens_d = margin / (pooled + 1e-8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5), gridspec_kw={"width_ratios": [1.3, 1.0]})
    bins = np.linspace(-0.25, 1.0, 50)
    ax1.hist(inter_arr, bins=bins, density=True, alpha=0.58, color="#94a3b8", edgecolor="black", lw=0.4,
             label=f"Inter-Class ($\\mu={mean_inter:.3f}, \\sigma={std_inter:.3f}$)")
    ax1.hist(intra_arr, bins=bins, density=True, alpha=0.75, color="#0284c7", edgecolor="black", lw=0.4,
             label=f"Intra-Class ($\\mu={mean_intra:.3f}, \\sigma={std_intra:.3f}$)")
    ax1.axvline(x=0.78, color="#e11d48", linestyle="--", lw=2, label="Calibrated OOD Cutoff (0.78)")
    ax1.axvline(x=0.85, color="#059669", linestyle="-.", lw=2, label="Definite Species Match (0.85)")
    ax1.set_title(f"Hyperspherical Margin ($\\Delta\\mu = {margin:.3f}$, Cohen's $d = {cohens_d:.2f}$)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Cosine Similarity ($\\cos\\theta$)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Probability Density", fontsize=10, fontweight="bold")
    ax1.legend(loc="upper left", framealpha=0.92, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.35)

    bplot = ax2.boxplot(
        [cohort_intra["head"], cohort_intra["med"], cohort_intra["tail"]],
        patch_artist=True,
        tick_labels=["Head (>=20)", "Medium (10-19)", "Tail (7-9)"],
        boxprops=dict(color="#1e293b"),
        medianprops=dict(color="#e11d48", lw=2),
    )
    for patch, col in zip(bplot["boxes"], ["#0284c7", "#0d9488", "#f59e0b"]):
        patch.set_facecolor(col)
        patch.set_alpha(0.70)
    ax2.set_title("Intra-Class Similarity by Actual Cohort", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cosine Similarity to Train Prototype", fontsize=10, fontweight="bold")
    ax2.set_ylim([0.35, 1.02])
    ax2.grid(True, linestyle="--", alpha=0.35)

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "similarity_distribution.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_measured_ablation_and_robustness(gen_stats):
    print("\n[Fig 7 & 8] Generating Measured Component Ablation & Robustness Charts...")
    stress = gen_stats["robustness_stress_tests"]
    cv = gen_stats.get("cross_validation_kfold")

    # Fig 7: Real Measured Component Ablation
    abl_keys = [
        ("Full Uncropped Slide (With Rulers)", "Uncropped Slide\n(Borders Included)"),
        ("Single-View ROI (No Spherical-TTA)", "Localized ROI\n(Single-View, No TTA)"),
        ("Mean Prototype Only (No Subcenters)", "ROI + 7-View TTA\n(Mean Proto Only)"),
        ("Clean Scale ROI (7-View TTA + Subcenters)", "Full Pipeline v4.1\n(ROI + TTA + Subcenters)"),
    ]
    labels = [lbl for _, lbl in abl_keys]
    top1s = [stress[k]["accuracy_pct"] for k, _ in abl_keys]
    tails = [stress[k]["tail_accuracy_pct"] for k, _ in abl_keys]

    x = np.arange(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.5))
    r1 = ax.bar(x - w / 2, top1s, w, label="Overall Top-1 Accuracy (%)", color="#0284c7", edgecolor="black", lw=0.7)
    r2 = ax.bar(x + w / 2, tails, w, label="Tail Cohort Accuracy (<10 samples) (%)", color="#f59e0b", edgecolor="black", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylim([40, 108])
    ax.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("Measured Ablation Contributions (Zero-Leakage Train Prototypes, N=142)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95)
    for b in list(r1) + list(r2):
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "ablation_study.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Fig 8: Cross-Camera Robustness + Background Masking Proof
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    dom_keys = [
        ("Clean Scale ROI (7-View TTA + Subcenters)", "Clean ROI", "#059669"),
        ("Dark / Low-Gamma Sensor (g=1.45)", "Dark g=1.45", "#0284c7"),
        ("Bright Warm CMOS Shift (g=0.72)", "Warm g=0.72", "#0ea5e9"),
        ("Optical Defocus Blur (r=1.2)", "Blur r=1.2", "#6366f1"),
        ("Combined Unseen Microscope", "Combined Shift", "#8b5cf6"),
    ]
    bars1 = ax1.bar([l for _, l, _ in dom_keys], [stress[k]["accuracy_pct"] for k, _, _ in dom_keys],
                    color=[c for _, _, c in dom_keys], edgecolor="black", lw=0.7, width=0.56)
    if cv:
        ax1.axhline(cv["mean_top1_pct"], color="#e11d48", linestyle="--", lw=1.8,
                    label=f"5-Fold CV Mean ({cv['mean_top1_pct']:.1f}% ± {cv['std_top1_pct']:.1f}%)")
        ax1.legend(loc="lower right", fontsize=8.5)
    ax1.set_ylim([0, 112])
    ax1.set_ylabel("Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
    ax1.set_title("Cross-Camera & Sensor Stress Robustness", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for b in bars1:
        h = b.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    bg_keys = [
        ("Clean Scale ROI (7-View TTA + Subcenters)", "Localized ROI\n(Scale Body)", "#059669"),
        ("Full Uncropped Slide (With Rulers)", "Uncropped Slide\n(With Rulers)", "#d97706"),
        ("Scale-Masked Background Only", "Slide Scale-Masked\n(Margins Only)", "#ea580c"),
        ("ROI Interior Masked (Blank Center)", "ROI Blank Center\n(No Scale/Margins)", "#dc2626"),
    ]
    bars2 = ax2.bar([l for _, l, _ in bg_keys], [stress[k]["accuracy_pct"] for k, _, _ in bg_keys],
                    color=[c for _, _, c in bg_keys], edgecolor="black", lw=0.7, width=0.52)
    ax2.axhline(100.0 / 35.0, color="#475569", linestyle=":", lw=1.8, label="35-Way Random Chance (2.9%)")
    ax2.set_ylim([0, 112])
    ax2.set_ylabel("Top-1 Accuracy (%)", fontsize=10, fontweight="bold")
    ax2.set_title("Background-Shortcut Ablation Proof", fontsize=11, fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", fontsize=8.5)
    for b in bars2:
        h = b.get_height()
        ax2.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "robustness_benchmark.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_novel_species_chart(gen_stats):
    print("\n[Fig 9] Generating Zero-Retraining Novel Species Few-Shot Benchmark Chart...")
    ns = gen_stats["novel_species_fewshot_benchmark"]
    shots = ns["shot_benchmarks"]
    keys = [k for k in ["1-shot", "2-shot", "3-shot"] if k in shots]

    x = np.arange(len(keys))
    w = 0.25
    open_t1 = [shots[k]["open_set_top1_pct"] for k in keys]
    open_t3 = [shots[k]["open_set_top3_pct"] for k in keys]
    novel_t1 = [shots[k]["novel_only_top1_pct"] for k in keys]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    b1 = ax.bar(x - w, novel_t1, w, label="Novel-Taxa Closed Accuracy (%)", color="#7c3aed", edgecolor="black", lw=0.7)
    b2 = ax.bar(x, open_t1, w, label="Open-Set Combined Top-1 (35 Base + Novel) (%)", color="#0284c7", edgecolor="black", lw=0.7)
    b3 = ax.bar(x + w, open_t3, w, label="Open-Set Combined Top-3 (%)", color="#059669", edgecolor="black", lw=0.7)

    x_labels = [f"{k}\n({shots[k]['novel_species_tested']} unseen taxa, {shots[k]['query_specimens']} queries)" for k in keys]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=9, fontweight="bold")
    ax.set_ylim([0, 114])
    ax.set_ylabel("Accuracy (%)", fontsize=10.5, fontweight="bold")
    ax.set_title("Zero-Retraining Registration of 21 Unseen Marine Species (<7 Samples)", fontsize=11.5, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    for b in list(b1) + list(b2) + list(b3):
        h = b.get_height()
        ax.annotate(f"{h:.1f}%", xy=(b.get_x() + b.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "novel_species_fewshot.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_class_performance(eval_res):
    print("\n[Fig 10] Generating Cohort-Coded Per-Species F1 Chart...")
    classes = eval_res["classes"]
    full_counts = eval_res["full_counts"]
    f1s = f1_score(eval_res["targets"], eval_res["preds"], average=None, labels=range(len(classes)), zero_division=0)

    colors = ["#0284c7" if c >= 20 else ("#0d9488" if c >= 10 else "#f59e0b") for c in full_counts]
    fig, ax = plt.subplots(figsize=(16, 6.8))
    ax.bar(range(len(classes)), f1s, color=colors, width=0.72, edgecolor="black", lw=0.5)
    mean_f1 = float(np.mean(f1s))
    ax.axhline(y=mean_f1, color="#e11d48", linestyle="--", lw=2)

    legend_elements = [
        patches.Patch(facecolor="#0284c7", label="Head Cohort (>= 20 samples)"),
        patches.Patch(facecolor="#0d9488", label="Medium Cohort (10-19 samples)"),
        patches.Patch(facecolor="#f59e0b", label="Tail Cohort (< 10 samples)"),
        plt.Line2D([0], [0], color="#e11d48", linestyle="--", lw=2, label=f"Macro-Average F1 ({mean_f1:.3f})"),
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=4, framealpha=0.95, fontsize=9.2)

    short_labels = [_clean_binomial(c) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short_labels, rotation=90, ha="right", fontsize=8)
    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Per-Species Taxonomic F1-Score (Zero-Leakage Train Prototypes, N=142 Held-Out)", fontsize=12, fontweight="bold", pad=16)
    ax.set_ylim([0.0, 1.18])
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "class_performance.png"), dpi=300, bbox_inches="tight")
    plt.close()


def generate_confusion_matrix(eval_res):
    print("\n[Fig 11] Generating Normalized Confusion Matrix...")
    classes = eval_res["classes"]
    cm = confusion_matrix(eval_res["targets"], eval_res["preds"], labels=range(len(classes)))
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Recall Probability", rotation=270, labelpad=15, fontsize=10, fontweight="bold")

    short_labels = [_clean_binomial(c) for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short_labels, rotation=90, ha="right", fontsize=7.5)
    ax.set_yticklabels(short_labels, fontsize=7.5)
    ax.set_xlabel("Predicted Species", fontsize=11, fontweight="bold")
    ax.set_ylabel("Ground-Truth Species", fontsize=11, fontweight="bold")
    ax.set_title(f"AquaLens v4.1 Confusion Matrix (35 Species — Top-1: {eval_res['top1_pct']:.2f}%, Top-3: {eval_res['top3_pct']:.2f}%)",
                 fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PAPER_IMG_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()


def main():
    classifier = FishClassifierV4()
    detector = ScaleDetector(padding_ratio=getattr(config, "ROI_PADDING_RATIO", 0.12))
    explainer = GradCAMExplainer(classifier.model, device=config.DEVICE)

    eval_res = evaluate_test_once(classifier)
    gen_stats_path = os.path.join(CHECKPOINTS_DIR, "generalization_stats.json")
    with open(gen_stats_path, "r") as f:
        gen_stats = json.load(f)

    generate_dataset_distribution(eval_res)
    generate_dataset_samples(detector)
    generate_gradcam_gallery(classifier, detector, explainer)
    generate_metric_tsne(classifier, eval_res)
    generate_roc_pr_curves(eval_res)
    generate_similarity_distribution(eval_res)
    generate_measured_ablation_and_robustness(gen_stats)
    generate_novel_species_chart(gen_stats)
    generate_class_performance(eval_res)
    generate_confusion_matrix(eval_res)
    print("\n[Complete] All verified publication figures saved to multiclass_35species/paper/img/!")


if __name__ == "__main__":
    main()
