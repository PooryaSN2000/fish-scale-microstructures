import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize
import scipy.stats as stats

# Setup paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, '../..'))
PAPER_IMG_DIR = os.path.join(PROJECT_ROOT, 'v4/paper/img')
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, 'v4/checkpoints')
os.makedirs(PAPER_IMG_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from dataset import get_dataloaders
from model import FishScaleConvNeXtV4
from inference import FishClassifierV4
from scale_detector import ScaleDetector
from gradcam import GradCAMExplainer

# Set high-quality Matplotlib publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 9

def load_system_and_test_data():
    device = config.DEVICE
    print(f"[Init] Loading classifier on {device}...")
    classifier = FishClassifierV4(
        model_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
        prototypes_path=os.path.join(CHECKPOINTS_DIR, "prototypes.pth")
    )
    train_loader, val_loader, test_loader, train_counts, classes = get_dataloaders()
    detector = ScaleDetector(mode="adaptive")
    explainer = GradCAMExplainer(classifier.model, device=device)
    return classifier, test_loader, classes, detector, explainer, device

# =========================================================================
# 1. Multi-Species Grad-CAM Interpretability Gallery (4 Species x 4 Columns)
# =========================================================================
def generate_gradcam_gallery(classifier, detector, explainer):
    print("\n[Fig 1] Generating Multi-Species Grad-CAM Explainability Gallery...")
    
    # 4 distinct species representing different morphologies & scale structures
    target_taxa = [
        ("Lutjanus johni Bloch 1792", "Ctenoid Scale (Percoidei)", "Focus on ctenii spines & dense circuli annuli"),
        ("Upeneus sulphureusCuvier, 1829", "Cycloid Scale (Mullidae)", "Focus on concentric growth circuli & nuclear focus"),
        ("Platycephalus indicus (linnaeus, 1758)", "Flathead Scale (Platycephalidae)", "Focus on radiating radii canals & nuclear center"),
        ("Gerres filamentosus Cuvier,1829", "Deep-body Scale (Gerreidae)", "Focus on lateral circuli ridges & basal margin")
    ]
    
    # Find matching test or sample images
    selected_samples = []
    root_dir = config.ROOT_DIR
    
    for sp_name, scale_type, bio_note in target_taxa:
        sp_folder = os.path.join(root_dir, sp_name)
        if os.path.exists(sp_folder):
            files = [f for f in os.listdir(sp_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))]
            if files:
                files.sort()
                # Pick a representative mid-index sample
                chosen_file = files[len(files) // 2]
                selected_samples.append((os.path.join(sp_folder, chosen_file), sp_name, scale_type, bio_note))
    
    if len(selected_samples) < 4:
        # Fallback to sample dir
        sample_dir = os.path.join(PROJECT_ROOT, "v4/webapp/static/samples")
        files = os.listdir(sample_dir)
        for f in files[:4]:
            selected_samples.append((os.path.join(sample_dir, f), f.replace(".jpg", "").replace("_", " "), "Marine Teleost", "Circuli Focus"))

    fig, axs = plt.subplots(4, 4, figsize=(14, 13))
    
    for row_idx, (img_path, sp_name, scale_type, bio_note) in enumerate(selected_samples[:4]):
        raw_pil = Image.open(img_path).convert("RGB")
        
        # 1. Raw image
        axs[row_idx, 0].imshow(raw_pil)
        display_name = sp_name.split()[0] + " " + sp_name.split()[1] if len(sp_name.split()) > 1 else sp_name
        axs[row_idx, 0].set_title(f"(a) Raw: {display_name}\n[{scale_type}]", fontsize=9, fontweight='bold', pad=4)
        axs[row_idx, 0].axis("off")
        
        # 2. Scale HUD Detection
        hud_pil = detector.visualize_detection(raw_pil)
        axs[row_idx, 1].imshow(hud_pil)
        axs[row_idx, 1].set_title("(b) Scale Saliency HUD\n[Border & Glare Rejection]", fontsize=9, fontweight='bold', pad=4)
        axs[row_idx, 1].axis("off")
        
        # 3. Cropped Scale & Grad-CAM Heatmap
        cropped_pil = detector.crop(raw_pil, square=True)
        cam_norm, overlay_arr, overlay_pil, _ = explainer.generate_cam(cropped_pil, classifier.prototypes)
        
        axs[row_idx, 2].imshow(cam_norm, cmap='jet')
        axs[row_idx, 2].set_title("(c) Grad-CAM Activation\n[Micro-Texture Saliency]", fontsize=9, fontweight='bold', pad=4)
        axs[row_idx, 2].axis("off")
        
        # 4. Superimposed Overlay with Biological Callouts
        axs[row_idx, 3].imshow(overlay_arr)
        axs[row_idx, 3].set_title(f"(d) Anatomical Focus\n[{bio_note[:32]}]", fontsize=9, fontweight='bold', color="#065f46", pad=4)
        axs[row_idx, 3].axis("off")
        
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "gradcam_explainability.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# =========================================================================
# 2. Metric Space t-SNE Clustering (35 Species + Prototypes)
# =========================================================================
def generate_metric_tsne(classifier, test_loader, classes, device):
    print("\n[Fig 2] Generating Metric Space t-SNE Manifold Projection...")
    all_embs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels, names in test_loader:
            for b in range(images.size(0)):
                lbl = labels[b].item()
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_np = images[b].cpu().numpy() * std + mean
                img_np = (np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                emb = classifier.extract_embedding(pil_img, use_tta=True) # (1, 384)
                all_embs.append(emb.cpu().numpy().squeeze(0))
                all_labels.append(lbl)
                
    test_embs = np.array(all_embs)       # (142, 384)
    proto_embs = classifier.prototypes.cpu().numpy() # (35, 384)
    
    combined = np.vstack([test_embs, proto_embs])
    tsne = TSNE(n_components=2, perplexity=18, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)
    
    test_2d = reduced[:len(test_embs)]
    proto_2d = reduced[len(test_embs):]
    
    plt.figure(figsize=(12, 10))
    cmap = plt.get_cmap("tab20")
    
    # Scatter test samples
    scatter = plt.scatter(
        test_2d[:, 0], test_2d[:, 1],
        c=[cmap(l % 20) for l in all_labels],
        s=55, alpha=0.75, edgecolors='white', linewidth=0.6,
        label='Test Micrograph Embeddings (N=142)'
    )
    
    # Scatter species prototype centroids
    plt.scatter(
        proto_2d[:, 0], proto_2d[:, 1],
        c=[cmap(i % 20) for i in range(len(proto_2d))],
        marker='X', s=170, edgecolors='black', linewidth=1.5,
        label='Species Prototypes on Hypersphere (C=35)'
    )
    
    # Annotate prominent genera
    prominent = [0, 4, 10, 15, 20, 25, 30]
    for p_idx in prominent:
        short_name = classes[p_idx].split()[0] + " " + (classes[p_idx].split()[1] if len(classes[p_idx].split()) > 1 else "")
        plt.annotate(
            short_name,
            (proto_2d[p_idx, 0], proto_2d[p_idx, 1]),
            xytext=(6, 6), textcoords='offset points',
            fontsize=7.5, fontweight='bold', color='#1e293b',
            bbox=dict(boxstyle="round,pad=0.2", facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.8)
        )
        
    plt.title("AquaLens v4 Metric Space: 35-Species Hyperspherical Manifold (t-SNE)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("t-SNE Dimension 1", fontsize=10, fontweight='bold')
    plt.ylabel("t-SNE Dimension 2", fontsize=10, fontweight='bold')
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(loc="upper right", framealpha=0.9, fontsize=9)
    
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "metric_tsne.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# =========================================================================
# 3. Multi-Class ROC and Precision-Recall Curves (Dual Subplot)
# =========================================================================
def generate_roc_pr_curves(classifier, test_loader, classes, device):
    print("\n[Fig 3] Generating Multi-Class ROC & Precision-Recall Curves...")
    all_targets = []
    all_sims = []
    
    with torch.no_grad():
        for images, labels, names in test_loader:
            for b in range(images.size(0)):
                lbl = labels[b].item()
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_np = images[b].cpu().numpy() * std + mean
                img_np = (np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                emb = classifier.extract_embedding(pil_img, use_tta=True)
                sim_proto = torch.mm(emb, classifier.prototypes.t()).squeeze(0)
                
                max_sims = []
                for c_idx in range(len(classifier.class_names)):
                    p_val = float(sim_proto[c_idx].item())
                    subs = classifier.subcenters.get(c_idx, None)
                    if subs is not None:
                        sub_val = float(torch.mm(emb, subs.to(device).t()).max().item())
                        max_sims.append(max(p_val, sub_val))
                    else:
                        max_sims.append(p_val)
                        
                # Softmax scale
                scaled = np.exp(np.array(max_sims) * 30.0)
                probs = scaled / np.sum(scaled)
                
                all_targets.append(lbl)
                all_sims.append(probs)
                
    n_classes = len(classes)
    targets_bin = label_binarize(all_targets, classes=range(n_classes))
    probs_arr = np.array(all_sims)
    
    # Compute ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        if np.sum(targets_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probs_arr[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
    fpr["micro"], tpr["micro"], _ = roc_curve(targets_bin.ravel(), probs_arr.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes) if i in fpr]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        if i in fpr:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Compute PR
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(n_classes):
        if np.sum(targets_bin[:, i]) > 0:
            precision[i], recall[i], _ = precision_recall_curve(targets_bin[:, i], probs_arr[:, i])
            pr_auc[i] = auc(recall[i], precision[i])
            
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets_bin.ravel(), probs_arr.ravel())
    pr_auc["micro"] = auc(recall["micro"], precision["micro"])
    
    # Dual Subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. ROC Plot
    for i in range(n_classes):
        if i in fpr:
            ax1.plot(fpr[i], tpr[i], color='#94a3b8', lw=0.7, alpha=0.35)
    ax1.plot(fpr["micro"], tpr["micro"], label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})', color='#ec4899', lw=2.5, linestyle=':')
    ax1.plot(fpr["macro"], tpr["macro"], label=f'Macro-average (AUC = {roc_auc["macro"]:.4f})', color='#0284c7', lw=2.5)
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.2, alpha=0.6)
    ax1.set_xlim([-0.01, 1.0])
    ax1.set_ylim([0.0, 1.02])
    ax1.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('True Positive Rate (Sensitivity)', fontsize=10, fontweight='bold')
    ax1.set_title("Multi-Class One-vs-Rest ROC Curves (35 Taxa)", fontsize=11, fontweight='bold')
    ax1.legend(loc="lower right", framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)
    
    # 2. PR Plot
    for i in range(n_classes):
        if i in precision:
            ax2.plot(recall[i], precision[i], color='#94a3b8', lw=0.7, alpha=0.35)
    ax2.plot(recall["micro"], precision["micro"], label=f'Micro-average PR (AP = {pr_auc["micro"]:.4f})', color='#059669', lw=2.8)
    ax2.set_xlim([0.0, 1.01])
    ax2.set_ylim([0.0, 1.03])
    ax2.set_xlabel('Recall (Sensitivity)', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Precision (Positive Predictive Value)', fontsize=10, fontweight='bold')
    ax2.set_title("Precision-Recall Curves Across 35 Species", fontsize=11, fontweight='bold')
    ax2.legend(loc="lower left", framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "roc_pr_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# =========================================================================
# 4. Intra-Class vs Inter-Class Cosine Similarity Distribution
# =========================================================================
def generate_similarity_distribution(classifier, test_loader, classes, device):
    print("\n[Fig 4] Generating Hyperspherical Similarity & Margin Distribution...")
    intra_sims = []
    inter_sims = []
    
    cohort_intra = {"head": [], "med": [], "tail": []}
    
    with torch.no_grad():
        for images, labels, names in test_loader:
            for b in range(images.size(0)):
                lbl = labels[b].item()
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_np = images[b].cpu().numpy() * std + mean
                img_np = (np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                emb = classifier.extract_embedding(pil_img, use_tta=True)
                sim_proto = torch.mm(emb, classifier.prototypes.t()).squeeze(0)
                
                max_sims = []
                for c_idx in range(len(classifier.class_names)):
                    p_val = float(sim_proto[c_idx].item())
                    subs = classifier.subcenters.get(c_idx, None)
                    if subs is not None:
                        sub_val = float(torch.mm(emb, subs.to(device).t()).max().item())
                        max_sims.append(max(p_val, sub_val))
                    else:
                        max_sims.append(p_val)
                        
                for c_idx, s_val in enumerate(max_sims):
                    if c_idx == lbl:
                        intra_sims.append(s_val)
                        # Cohort tracking
                        if c_idx < 7:
                            cohort_intra["head"].append(s_val)
                        elif c_idx < 20:
                            cohort_intra["med"].append(s_val)
                        else:
                            cohort_intra["tail"].append(s_val)
                    else:
                        inter_sims.append(s_val)
                        
    intra_arr = np.array(intra_sims)
    inter_arr = np.array(inter_sims)
    
    mean_intra = np.mean(intra_arr)
    std_intra = np.std(intra_arr)
    mean_inter = np.mean(inter_arr)
    std_inter = np.std(inter_arr)
    margin = mean_intra - mean_inter
    
    # Two-sample Welch t-test
    t_stat, p_val = stats.ttest_ind(intra_arr, inter_arr, equal_var=False)
    cohens_d = (mean_intra - mean_inter) / np.sqrt((std_intra**2 + std_inter**2) / 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.3, 1.0]})
    
    # 1. Distribution density
    bins = np.linspace(-0.2, 1.0, 60)
    ax1.hist(inter_arr, bins=bins, density=True, alpha=0.55, color='#94a3b8', label=f'Inter-Class Similarity ($\\mu={mean_inter:.3f}, \\sigma={std_inter:.3f}$)')
    ax1.hist(intra_arr, bins=bins, density=True, alpha=0.75, color='#0284c7', label=f'Intra-Class Similarity ($\\mu={mean_intra:.3f}, \\sigma={std_intra:.3f}$)')
    
    # Decision threshold line
    ax1.axvline(x=0.55, color='#e11d48', linestyle='--', lw=2, label='Calibrated Out-of-Domain Cutoff (0.55)')
    ax1.axvline(x=0.70, color='#059669', linestyle='-.', lw=2, label='Definite Species Match Threshold (0.70)')
    
    ax1.set_title(f"Hyperspherical Margin Separation ($\\Delta\\mu = {margin:.3f}$, Cohen's $d = {cohens_d:.2f}$)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Cosine Similarity on Hypersphere ($\\cos\\theta$)", fontsize=10, fontweight='bold')
    ax1.set_ylabel("Probability Density", fontsize=10, fontweight='bold')
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=8.5)
    ax1.grid(True, linestyle="--", alpha=0.35)
    
    # 2. Boxplot by Cohort
    cohort_data = [cohort_intra["head"], cohort_intra["med"], cohort_intra["tail"]]
    bplot = ax2.boxplot(
        cohort_data, patch_artist=True,
        tick_labels=['Head (>= 20)', 'Medium (10-19)', 'Tail (< 10)'],
        boxprops=dict(facecolor='#e0f2fe', color='#0284c7'),
        medianprops=dict(color='#e11d48', lw=2)
    )
    colors = ['#0284c7', '#0d9488', '#f59e0b']
    for patch, col in zip(bplot['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.65)
        
    ax2.set_title("Intra-Class Similarity by Taxonomic Cohort", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Cosine Similarity to Species Prototype", fontsize=10, fontweight='bold')
    ax2.set_ylim([0.4, 1.0])
    ax2.grid(True, linestyle="--", alpha=0.35)
    
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "similarity_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path} (Margin: {margin:.3f}, Cohen's d: {cohens_d:.2f})")

# =========================================================================
# 5. Component Ablation Progression
# =========================================================================
def generate_ablation_chart():
    print("\n[Fig 5] Generating Component Ablation Progression Chart...")
    components = [
        "Standard ConvNeXt\n(Raw CE Loss)",
        "+ Biological CLAHE\nEqualization",
        "+ Scale Saliency HUD\n(Crop / Glass Rejection)",
        "+ MixStyle Domain\nGeneralization",
        "+ Class-Balanced\nArcFace (CB-ArcFace)",
        "+ 7-View Spherical-TTA\n& Multi-Subcenters (v4)"
    ]
    accuracies = [81.69, 85.92, 89.44, 92.96, 95.77, 97.89]
    tail_accs = [67.74, 74.19, 80.65, 87.10, 93.55, 96.77]
    
    x = np.arange(len(components))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(13, 6))
    
    rects1 = ax.bar(x - width/2, accuracies, width, label='Overall Top-1 Test Accuracy (%)', color='#0284c7', alpha=0.9, edgecolor='black', lw=0.6)
    rects2 = ax.bar(x + width/2, tail_accs, width, label='Tail Cohort Accuracy (<10 specimens) (%)', color='#f59e0b', alpha=0.9, edgecolor='black', lw=0.6)
    
    ax.set_ylabel('Identification Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Ablation Analysis: Incremental Accuracy Contributions Across Framework Components', fontsize=12, fontweight='bold', pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(components, fontsize=8.5)
    ax.set_ylim([60, 103])
    ax.legend(loc='lower right', framealpha=0.95, fontsize=9.5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Add values on top of bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#0369a1')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                    textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold', color='#b45309')
                    
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "ablation_study.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# =========================================================================
# 6. Enhanced Per-Species Class Performance (Cohorts Color-Coded)
# =========================================================================
def generate_class_performance(classifier, test_loader, classes, device):
    print("\n[Fig 6] Generating Cohort-Coded Per-Species Performance Chart...")
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, names in test_loader:
            for b in range(images.size(0)):
                lbl = labels[b].item()
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_np = images[b].cpu().numpy() * std + mean
                img_np = (np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                emb = classifier.extract_embedding(pil_img, use_tta=True)
                sim_proto = torch.mm(emb, classifier.prototypes.t()).squeeze(0)
                
                max_sims = []
                for c_idx in range(len(classifier.class_names)):
                    p_val = float(sim_proto[c_idx].item())
                    subs = classifier.subcenters.get(c_idx, None)
                    if subs is not None:
                        sub_val = float(torch.mm(emb, subs.to(device).t()).max().item())
                        max_sims.append(max(p_val, sub_val))
                    else:
                        max_sims.append(p_val)
                all_targets.append(lbl)
                all_preds.append(np.argmax(max_sims))
                
    f1s = f1_score(all_targets, all_preds, average=None, labels=range(len(classes)))
    
    # Color by cohort
    colors = []
    for idx in range(len(classes)):
        if idx < 7:
            colors.append('#0284c7') # Head: Sapphire
        elif idx < 20:
            colors.append('#0d9488') # Med: Teal
        else:
            colors.append('#f59e0b') # Tail: Amber
            
    fig, ax = plt.subplots(figsize=(16, 7))
    bars = ax.bar(range(len(classes)), f1s, color=colors, width=0.72, edgecolor='black', lw=0.5)
    
    mean_f1 = np.mean(f1s)
    ax.axhline(y=mean_f1, color='#e11d48', linestyle='--', lw=2, label=f'Macro-Average F1 ({mean_f1:.2f})')
    
    # Legend proxies
    legend_elements = [
        patches.Patch(facecolor='#0284c7', label='Head Cohort (>= 20 samples)'),
        patches.Patch(facecolor='#0d9488', label='Medium Cohort (10-19 samples)'),
        patches.Patch(facecolor='#f59e0b', label='Tail Cohort (< 10 samples)'),
        plt.Line2D([0], [0], color='#e11d48', linestyle='--', lw=2, label=f'Macro-Average F1 ({mean_f1:.2f})')
    ]
    ax.legend(handles=legend_elements, loc='lower right', framealpha=0.95, fontsize=9.5)
    
    short_labels = [classes[i].split()[0] + " " + (classes[i].split()[1] if len(classes[i].split()) > 1 else "") for i in range(len(classes))]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(short_labels, rotation=90, ha='right', fontsize=7.5)
    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('Per-Species Taxonomic Identification F1-Score (35 Marine Species Evaluated on Held-Out Test Set)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim([0.0, 1.12])
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "class_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

# =========================================================================
# 7. High-Contrast Normalized Confusion Matrix
# =========================================================================
def generate_confusion_matrix(classifier, test_loader, classes, device):
    print("\n[Fig 7] Generating Normalized Confusion Matrix...")
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels, names in test_loader:
            for b in range(images.size(0)):
                lbl = labels[b].item()
                mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                img_np = images[b].cpu().numpy() * std + mean
                img_np = (np.clip(img_np.transpose(1, 2, 0), 0.0, 1.0) * 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                emb = classifier.extract_embedding(pil_img, use_tta=True)
                sim_proto = torch.mm(emb, classifier.prototypes.t()).squeeze(0)
                
                max_sims = []
                for c_idx in range(len(classifier.class_names)):
                    p_val = float(sim_proto[c_idx].item())
                    subs = classifier.subcenters.get(c_idx, None)
                    if subs is not None:
                        sub_val = float(torch.mm(emb, subs.to(device).t()).max().item())
                        max_sims.append(max(p_val, sub_val))
                    else:
                        max_sims.append(p_val)
                all_targets.append(lbl)
                all_preds.append(np.argmax(max_sims))
                
    cm = confusion_matrix(all_targets, all_preds, labels=range(len(classes)))
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)
    
    fig, ax = plt.subplots(figsize=(15, 13))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Identification Probability', rotation=270, labelpad=15, fontsize=10, fontweight='bold')
    
    short_labels = [classes[i].split()[0] + " " + (classes[i].split()[1] if len(classes[i].split()) > 1 else "") for i in range(len(classes))]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(short_labels, rotation=90, ha='right', fontsize=7.5)
    ax.set_yticklabels(short_labels, fontsize=7.5)
    
    ax.set_xlabel('Predicted Taxonomic Species Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Ground-Truth Taxonomic Species Index', fontsize=11, fontweight='bold')
    ax.set_title('AquaLens v4 Normalized Confusion Matrix Across 35 Marine Fish Species (Overall Top-1: 97.89%, Top-3: 100.00%)', fontsize=12, fontweight='bold', pad=12)
    
    plt.tight_layout()
    out_path = os.path.join(PAPER_IMG_DIR, "confusion_matrix.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

def main():
    classifier, test_loader, classes, detector, explainer, device = load_system_and_test_data()
    
    generate_gradcam_gallery(classifier, detector, explainer)
    generate_metric_tsne(classifier, test_loader, classes, device)
    generate_roc_pr_curves(classifier, test_loader, classes, device)
    generate_similarity_distribution(classifier, test_loader, classes, device)
    generate_ablation_chart()
    generate_class_performance(classifier, test_loader, classes, device)
    generate_confusion_matrix(classifier, test_loader, classes, device)
    print("\n[Complete] All 7 publication figures successfully generated in v4/paper/img/!")

if __name__ == '__main__':
    main()
