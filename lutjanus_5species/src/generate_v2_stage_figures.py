import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from sklearn.preprocessing import label_binarize

# Setup Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
V2_DIR = os.path.abspath(os.path.join(SRC_DIR, '..'))
PAPER_IMG_DIR = os.path.join(V2_DIR, 'paper/img')
CHECKPOINTS_DIR = os.path.join(V2_DIR, 'checkpoints')
PROJECT_ROOT = os.path.abspath(os.path.join(V2_DIR, '..'))
ROOT_DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')

os.makedirs(PAPER_IMG_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from inference import FishClassifier

# Publication Styling: clean, large fonts, narrow figures
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10

# Definition of the 5 Stages (sorted by sample abundance as in Table I)
STAGES_INFO = [
    {
        "stage": "Stage 1",
        "code": "S1",
        "species": "Lutjanus johni Bloch 1792",
        "latin": "Lutjanus johni",
        "author": "Bloch, 1792",
        "samples": 127,
        "folder": "Lutjanus johni Bloch 1792",
        "proto_name": "Lutjanus johni Bloch 1792",
        "color": "#1f77b4"
    },
    {
        "stage": "Stage 2",
        "code": "S2",
        "species": "Lutjanus lutjanus Bloch 1790",
        "latin": "Lutjanus lutjanus",
        "author": "Bloch, 1790",
        "samples": 66,
        "folder": "Lutjanus lutjanus Bloch 1790",
        "proto_name": "Lutjanus lutjanus Bloch 1790",
        "color": "#2ca02c"
    },
    {
        "stage": "Stage 3",
        "code": "S3",
        "species": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "latin": "Lutjanus lemniscatus",
        "author": "(Valenciennes, 1828)",
        "samples": 21,
        "folder": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "proto_name": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "color": "#ff7f0e"
    },
    {
        "stage": "Stage 4",
        "code": "S4",
        "species": "Lutjanus malabaricus (Bloch & Schneider, 1801)",
        "latin": "Lutjanus malabaricus",
        "author": "(Bloch & Schneider, 1801)",
        "samples": 14,
        "folder": "Lutjanus malabaricus (Bloch & Schneider, 1801)",
        "proto_name": "lutjanus malabarius(Bloch and Schneider,1801)",
        "color": "#9467bd"
    },
    {
        "stage": "Stage 5",
        "code": "S5",
        "species": "Lutjanus fulviflammus(Forsskal 1775)",
        "latin": "Lutjanus fulviflammus",
        "author": "(Forsskal, 1775)",
        "samples": 5,
        "folder": "Lutjanus fulviflammus(Forsskal 1775)",
        "proto_name": "Lutjanus fulviflammus(Forsskal 1775)",
        "color": "#d62728"
    }
]

def save_fig_dual(fig, filename):
    p1 = os.path.join(PAPER_IMG_DIR, filename)
    p2 = os.path.join(CHECKPOINTS_DIR, filename)
    fig.savefig(p1, dpi=300, bbox_inches='tight')
    fig.savefig(p2, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {filename} -> paper/img/ and checkpoints/")

# 1. Dataset Distribution: Narrow bar chart with Stage labels and clear legend box
def plot_dataset_distribution_stage():
    print("[1/5] Generating Stage-labeled Dataset Distribution...")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))  # Narrow, compact
    
    stages = [item["stage"] for item in STAGES_INFO]
    counts = [item["samples"] for item in STAGES_INFO]
    colors = [item["color"] for item in STAGES_INFO]
    
    bars = ax.bar(stages, counts, color=colors, width=0.6, edgecolor='black', linewidth=0.8, alpha=0.9)
    
    ax.set_ylabel("Number of Scale Micrographs", fontsize=12, fontweight='bold')
    ax.set_xlabel("Taxonomic Stage (Lutjanidae Family)", fontsize=12, fontweight='bold')
    ax.set_title("Sample Distribution Across the 5 Lutjanus Stages", fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim([0, 145])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate numbers on top of bars
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    # Legend mapping Stage to Species
    legend_patches = [
        Patch(facecolor=item["color"], edgecolor='black', linewidth=0.5,
              label=f'{item["stage"]} ({item["code"]}): {item["latin"]}')
        for item in STAGES_INFO
    ]
    ax.legend(handles=legend_patches, loc="upper right", framealpha=0.95, fontsize=9.5, title="Species Legend", title_fontsize=10)
    
    fig.tight_layout()
    save_fig_dual(fig, "dataset_distribution.png")

# 2. Extract predictions for the 233 images to generate exact Confusion Matrix, ROC, and F1
def evaluate_all_stages():
    print("[Inference] Evaluating all 233 samples across the 5 Stages...")
    classifier = FishClassifier(
        model_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
        prototypes_path=os.path.join(CHECKPOINTS_DIR, "prototypes.pth")
    )
    
    all_targets = []
    all_preds = []
    all_probs = []
    
    # Map classifier class names to Stage indices (0..4)
    # Class names in classifier:
    # 0: Lutjanus fulviflammus (Stage 5)
    # 1: Lutjanus johni (Stage 1)
    # 2: Lutjanus lemniscatus (Stage 3)
    # 3: Lutjanus lutjanus (Stage 2)
    # 4: lutjanus malabarius (Stage 4)
    proto_to_stage_idx = {}
    for stage_idx, item in enumerate(STAGES_INFO):
        proto_to_stage_idx[item["proto_name"]] = stage_idx
        
    for stage_idx, item in enumerate(STAGES_INFO):
        c_folder = os.path.join(ROOT_DATA_DIR, item["folder"])
        if not os.path.exists(c_folder):
            print(f"Warning: Folder not found: {c_folder}")
            continue
        files = [f for f in os.listdir(c_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        for f in files:
            f_path = os.path.join(c_folder, f)
            pred_name, conf, prob_dict, max_sim, sim_dict = classifier.predict(f_path)
            
            # Target is stage_idx (0..4)
            all_targets.append(stage_idx)
            pred_stage = proto_to_stage_idx[pred_name]
            all_preds.append(pred_stage)
            
            # Probability array aligned with Stage 1..5
            stage_probs = [prob_dict[STAGES_INFO[s]["proto_name"]] for s in range(5)]
            all_probs.append(stage_probs)
            
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)

# 3. Normalized Confusion Matrix: Narrow, clean square, large text, Stage labels
def plot_confusion_matrix_stage(targets, preds):
    print("[2/5] Generating Stage-labeled Confusion Matrix...")
    cm = confusion_matrix(targets, preds, labels=range(5))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    
    fig, ax = plt.subplots(figsize=(6.2, 5.5))  # Compact & square
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Accuracy', rotation=270, labelpad=16, fontsize=11, fontweight='bold')
    
    stage_labels = [item["stage"] for item in STAGES_INFO]
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(stage_labels, fontsize=11, fontweight='bold')
    ax.set_yticklabels(stage_labels, fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Predicted Stage', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('True Stage', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_title('Normalized Confusion Matrix (5 Lutjanus Stages)', fontsize=13, fontweight='bold', pad=12)
    
    # Values inside cells: Large bold font!
    thresh = cm_norm.max() / 2.0
    for i in range(5):
        for j in range(5):
            val = cm_norm[i, j]
            txt = f"{val:.2f}" if val > 0 else "0.00"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    color="white" if val > thresh else "black",
                    fontsize=12, fontweight='bold')
            
    fig.tight_layout()
    save_fig_dual(fig, "confusion_matrix.png")

# 4. Multi-class ROC Curves: Narrow, large fonts, Stage legend
def plot_roc_curves_stage(targets, probs):
    print("[3/5] Generating Stage-labeled ROC Curves...")
    n_classes = 5
    targets_bin = label_binarize(targets, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(targets_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    fig, ax = plt.subplots(figsize=(6.5, 5.2))  # Narrow publication format
    
    for i in range(n_classes):
        item = STAGES_INFO[i]
        ax.plot(fpr[i], tpr[i], color=item["color"], lw=2.4,
                label=f'{item["stage"]} ({item["latin"]}): AUC = {roc_auc[i]:.2f}')
        
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7)
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.03])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=11, fontweight='bold')
    ax.set_title('One-vs-Rest ROC Curves Across 5 Stages', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9.2)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    save_fig_dual(fig, "roc_curves.png")

# 5. Class Performance F1-Score: Narrow bar chart, large fonts, Stage labels
def plot_class_performance_stage(targets, preds):
    print("[4/5] Generating Stage-labeled F1-Score Bar Chart...")
    f1_scores = f1_score(targets, preds, average=None, labels=range(5))
    
    fig, ax = plt.subplots(figsize=(6.5, 4.8))  # Narrow format
    
    stages = [item["stage"] for item in STAGES_INFO]
    colors = [item["color"] for item in STAGES_INFO]
    
    bars = ax.bar(stages, f1_scores, color=colors, width=0.6, edgecolor='black', linewidth=0.8, alpha=0.9)
    mean_f1 = np.mean(f1_scores)
    ax.axhline(y=mean_f1, color='#e11d48', linestyle='--', lw=2, label=f'Mean F1 ({mean_f1:.2f})')
    
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_xlabel('Taxonomic Stage (Lutjanidae Family)', fontsize=12, fontweight='bold')
    ax.set_title('Classification F1-Score by Stage', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylim([0, 1.15])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Text on top of bars: Large bold font
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
        
    legend_patches = [
        Patch(facecolor=item["color"], edgecolor='black', linewidth=0.5,
              label=f'{item["stage"]}: {item["latin"]}')
        for item in STAGES_INFO
    ]
    legend_patches.append(plt.Line2D([0], [0], color='#e11d48', linestyle='--', lw=2, label=f'Mean F1 ({mean_f1:.2f})'))
    ax.legend(handles=legend_patches, loc="lower right", framealpha=0.95, fontsize=9.2)
    
    fig.tight_layout()
    save_fig_dual(fig, "class_performance.png")

# 6. t-SNE Embeddings: Narrow scatter plot with Stage markers
def plot_tsne_stage():
    print("[5/5] Generating Stage-labeled t-SNE Embedding Projection...")
    classifier = FishClassifier(
        model_path=os.path.join(CHECKPOINTS_DIR, "best_model.pth"),
        prototypes_path=os.path.join(CHECKPOINTS_DIR, "prototypes.pth")
    )
    
    all_embs = []
    all_stages = []
    
    # Sample up to 25 images per stage to keep plot balanced and sharp
    for stage_idx, item in enumerate(STAGES_INFO):
        c_folder = os.path.join(ROOT_DATA_DIR, item["folder"])
        if not os.path.exists(c_folder): continue
        files = [f for f in os.listdir(c_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        sampled_files = files[:min(30, len(files))]
        for f in sampled_files:
            f_path = os.path.join(c_folder, f)
            image = Image.open(f_path).convert("RGB")
            img_tensor = classifier.transform(image).unsqueeze(0).to(classifier.device)
            from features import extract_classical_features
            cf = extract_classical_features(image)
            cf_tensor = torch.tensor(cf, dtype=torch.float32).unsqueeze(0).to(classifier.device)
            with torch.no_grad():
                emb = classifier.model(img_tensor, cf_tensor, stage=2)
                emb_norm = F.normalize(emb, p=2, dim=-1).cpu().numpy().squeeze(0)
            all_embs.append(emb_norm)
            all_stages.append(stage_idx)
            
    sample_embs = np.array(all_embs)
    proto_embs = F.normalize(classifier.prototypes, p=2, dim=-1).cpu().numpy()
    
    # Order prototypes to match STAGES_INFO (0..4)
    # proto 0: fulviflammus (Stage 5 -> index 4)
    # proto 1: johni (Stage 1 -> index 0)
    # proto 2: lemniscatus (Stage 3 -> index 2)
    # proto 3: lutjanus (Stage 2 -> index 1)
    # proto 4: malabaricus (Stage 4 -> index 3)
    proto_stage_map = [4, 0, 2, 1, 3] # maps proto_idx to stage_idx
    ordered_proto_embs = np.zeros_like(proto_embs)
    for p_i, s_i in enumerate(proto_stage_map):
        ordered_proto_embs[s_i] = proto_embs[p_i]
        
    combined = np.vstack([sample_embs, ordered_proto_embs])
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init='pca', learning_rate='auto')
    reduced = tsne.fit_transform(combined)
    
    samples_2d = reduced[:len(sample_embs)]
    protos_2d = reduced[len(sample_embs):]
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5))  # Narrow format
    
    for stage_idx in range(5):
        mask = [s == stage_idx for s in all_stages]
        item = STAGES_INFO[stage_idx]
        ax.scatter(samples_2d[mask, 0], samples_2d[mask, 1],
                   color=item["color"], s=45, alpha=0.75, edgecolors='white', linewidth=0.5,
                   label=f'{item["stage"]} ({item["code"]})')
        # Plot Prototype
        ax.scatter(protos_2d[stage_idx, 0], protos_2d[stage_idx, 1],
                   color=item["color"], marker='X', s=200, edgecolors='black', linewidth=1.5)
        
    ax.set_title("t-SNE Metric Space for 5 Lutjanus Stages", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle="--", alpha=0.4)
    
    legend_handles = [
        Patch(facecolor=item["color"], edgecolor='black', linewidth=0.5,
              label=f'{item["stage"]}: {item["latin"]}')
        for item in STAGES_INFO
    ]
    legend_handles.append(plt.Line2D([0], [0], marker='X', color='w', markeredgecolor='black', markerfacecolor='gray', markersize=12, label='Prototype Centroid (X)'))
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.95, fontsize=8.8)
    
    fig.tight_layout()
    save_fig_dual(fig, "tsne_embeddings.png")

def main():
    print("=== Generating High-Quality Narrow Figures for Lutjanidae 5-Species Paper ===")
    plot_dataset_distribution_stage()
    targets, preds, probs = evaluate_all_stages()
    plot_confusion_matrix_stage(targets, preds)
    plot_roc_curves_stage(targets, probs)
    plot_class_performance_stage(targets, preds)
    plot_tsne_stage()
    print("\n[SUCCESS] All 5 Stage-labeled, narrow, large-font figures generated successfully!")

if __name__ == '__main__':
    main()
