#!/usr/bin/env python3
"""
AquaLens AI: Few-Shot Biometric Classification Demo
Designed for Defense Presentation & Supervisor Verification
Authors: Najmeh Sabbah, Poorya Saneei, et al.
University of Guilan (Marine Biology) & Shahid Bahonar University of Kerman
"""

import os
import sys
import time
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Auto-detect script directory and add v2/src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "v2":
    V2_DIR = SCRIPT_DIR
else:
    V2_DIR = os.path.join(SCRIPT_DIR, "v2") if os.path.exists(os.path.join(SCRIPT_DIR, "v2")) else SCRIPT_DIR

SRC_DIR = os.path.join(V2_DIR, "src")
CHECKPOINTS_DIR = os.path.join(V2_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(V2_DIR, "webapp/static/samples")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F
import config
from inference import FishClassifier

# Definition of the 5 Lutjanus Stages
STAGES_INFO = [
    {
        "stage": "Stage 1",
        "code": "S1",
        "species": "Lutjanus johni Bloch 1792",
        "latin": "Lutjanus johni",
        "persian": "ماهی سرخو معمولی (John's Snapper)",
        "sample_file": "Lutjanus johni Bloch 1792.jpg",
        "samples_count": 127,
        "color": "#1f77b4"
    },
    {
        "stage": "Stage 2",
        "code": "S2",
        "species": "Lutjanus lutjanus Bloch 1790",
        "latin": "Lutjanus lutjanus",
        "persian": "ماهی سرخو زردباله (Bigeye Snapper)",
        "sample_file": "Lutjanus lutjanus Bloch 1790.jpg",
        "samples_count": 66,
        "color": "#2ca02c"
    },
    {
        "stage": "Stage 3",
        "code": "S3",
        "species": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "latin": "Lutjanus lemniscatus",
        "persian": "ماهی سرخو نواری (Yellow-streaked Snapper)",
        "sample_file": "Lutjanus lemniscatus (Valenciennes, 1828).jpg",
        "samples_count": 21,
        "color": "#ff7f0e"
    },
    {
        "stage": "Stage 4",
        "code": "S4",
        "species": "lutjanus malabarius(Bloch and Schneider,1801)",
        "latin": "Lutjanus malabaricus",
        "persian": "ماهی سرخو مالابار (Malabar Blood Snapper)",
        "sample_file": "lutjanus malabarius(Bloch and Schneider,1801).jpg",
        "samples_count": 14,
        "color": "#9467bd"
    },
    {
        "stage": "Stage 5",
        "code": "S5",
        "species": "Lutjanus fulviflammus(Forsskal 1775)",
        "latin": "Lutjanus fulviflammus",
        "persian": "ماهی سرخو خط سیاه (Dory Snapper - Few-Shot Class)",
        "sample_file": "Lutjanus fulviflammus(Forsskal 1775).jpg",
        "samples_count": 5,
        "color": "#d62728"
    }
]

def load_engine():
    model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    protos_path = os.path.join(CHECKPOINTS_DIR, "prototypes.pth")
    if not os.path.exists(model_path) or not os.path.exists(protos_path):
        raise FileNotFoundError(f"Checkpoint files not found in: {CHECKPOINTS_DIR}")
    
    classifier = FishClassifier(model_path=model_path, prototypes_path=protos_path)
    return classifier

def run_classification_demo(image_path: str, output_plot_path: str = "defense_result.png"):
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        return
    
    classifier = load_engine()
    
    t0 = time.time()
    pred_species, confidence, all_probs, max_sim, all_sims = classifier.predict(image_path)
    elapsed = time.time() - t0
    
    # Identify Stage
    matched_stage = None
    for item in STAGES_INFO:
        if item["species"] == pred_species or item["latin"].lower() in pred_species.lower():
            matched_stage = item
            break
            
    if matched_stage is None:
        matched_stage = STAGES_INFO[0]
        
    print("\n" + "=" * 65)
    print("      AquaLens AI: Few-Shot Biometric Classification Demo")
    print("      Lutjanidae Family (Marine Fish Scales Biometrics)")
    print("=" * 65)
    print(f" [1] Input Image:      {os.path.basename(image_path)}")
    print(f" [2] Predicted Stage:  {matched_stage['stage']} ({matched_stage['code']})")
    print(f" [3] Scientific Name:  {matched_stage['latin']} ({matched_stage['persian']})")
    print(f" [4] Softmax Conf:     {confidence * 100:.2f}%")
    print(f" [5] Cosine Metric:    {max_sim * 100:.2f}% (Metric Similarity to Prototype)")
    print(f" [6] Inference Speed:  {elapsed:.3f} seconds ({1.0 / max(1e-4, elapsed):.1f} FPS)")
    print("=" * 65)
    print(" Stage-by-Stage Similarity Affinities:")
    
    # Print bar chart in terminal
    for item in STAGES_INFO:
        sim = all_sims.get(item["species"], 0.0)
        sim_pct = max(0.0, sim) * 100.0
        bar_len = int(sim_pct / 5.0)
        bar_str = "█" * bar_len + "░" * (20 - bar_len)
        marker = " <=== [TOP-1 MATCH]" if item["stage"] == matched_stage["stage"] else ""
        print(f"   - {item['stage']} ({item['code']}): {sim_pct:5.1f}% [{bar_str}] {item['latin']:<22}{marker}")
        
    print("=" * 65)
    
    # Generate Visual Result Plot
    try:
        raw_img = Image.open(image_path).convert("RGB")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={'width_ratios': [1.1, 1.3]})
        
        # Left: Input Image
        ax1.imshow(raw_img)
        ax1.set_title(f"Input Micrograph\nPredicted: {matched_stage['stage']} ({matched_stage['latin']})", fontsize=11, fontweight='bold', pad=8)
        ax1.axis('off')
        
        # Right: Bar Chart
        stages_labels = [f"{item['stage']}\n({item['code']})" for item in STAGES_INFO]
        similarities = [max(0.0, all_sims.get(item["species"], 0.0)) * 100.0 for item in STAGES_INFO]
        colors = ['#10b981' if item['stage'] == matched_stage['stage'] else '#94a3b8' for item in STAGES_INFO]
        
        bars = ax2.bar(stages_labels, similarities, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
        ax2.set_ylabel("Cosine Similarity (%)", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Taxonomic Stage (Lutjanus Family)", fontsize=11, fontweight='bold')
        ax2.set_title(f"Few-Shot Metric Prototype Matching (Conf: {confidence*100:.1f}%)", fontsize=11, fontweight='bold', pad=10)
        ax2.set_ylim([0, 110])
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold')
            
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[✓] Visual explanation figure saved to: {output_plot_path}")
    except Exception as e:
        print(f"[Warning] Could not save visualization: {e}")
        
    print("=" * 65 + "\n")

def main():
    parser = argparse.ArgumentParser(description="AquaLens Few-Shot Demo for Defense Presentation")
    parser.add_argument("image", nargs="?", default=None, help="Path to input scale image")
    parser.add_argument("--test", action="store_true", help="Run automated test on Stage 1 sample")
    parser.add_argument("--all", action="store_true", help="Run batch test on all 5 stages")
    args = parser.parse_args()
    
    if args.all:
        print("Running verification across all 5 Lutjanus Stages...")
        for item in STAGES_INFO:
            s_path = os.path.join(SAMPLES_DIR, item["sample_file"])
            run_classification_demo(s_path, f"result_{item['code']}.png")
        return
        
    if args.test or args.image is not None:
        img_path = args.image or os.path.join(SAMPLES_DIR, STAGES_INFO[0]["sample_file"])
        run_classification_demo(img_path)
        return
        
    # Interactive Mode
    print("\n" + "=" * 65)
    print("   AquaLens AI: Interactive Few-Shot Demo for Defense")
    print("=" * 65)
    print(" Select a test micrograph from the 5 Lutjanus stages:")
    for idx, item in enumerate(STAGES_INFO, 1):
        print(f"   [{idx}] {item['stage']} ({item['code']}): {item['latin']:<24} - {item['persian']}")
    print("=" * 65)
    
    try:
        user_input = input("Enter choice [1-5] (or press Enter for Stage 1): ").strip()
    except (EOFError, KeyboardInterrupt):
        user_input = "1"
        
    if user_input in ["1", "2", "3", "4", "5"]:
        chosen_idx = int(user_input) - 1
    else:
        chosen_idx = 0
        
    chosen_stage = STAGES_INFO[chosen_idx]
    chosen_path = os.path.join(SAMPLES_DIR, chosen_stage["sample_file"])
    print(f"\n[Selected] {chosen_stage['stage']} ({chosen_stage['latin']})")
    run_classification_demo(chosen_path)

if __name__ == '__main__':
    main()
