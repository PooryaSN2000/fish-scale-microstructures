#!/usr/bin/env python3
"""
AquaLens AI: Interactive Biometric Defense Presentation & Supervisor Verification
Authors: Najmeh Sabbah, Poorya Saneei, et al.
University of Guilan (Marine Biology) & Shahid Bahonar University of Kerman

Supports both:
1. Dr. Eftekhari's 5-Class / 5-Stage Lutjanidae Snapper Evaluation (Class 1..5 / C1..C5)
2. Guilan University Supervisor's Multi-Class Few-Shot & Novel / Unseen Species Evaluation
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Auto-detect script directory and set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(SCRIPT_DIR) == "v2":
    V2_DIR = SCRIPT_DIR
    ROOT_DIR = os.path.dirname(SCRIPT_DIR)
else:
    V2_DIR = os.path.join(SCRIPT_DIR, "v2") if os.path.exists(os.path.join(SCRIPT_DIR, "v2")) else SCRIPT_DIR
    ROOT_DIR = SCRIPT_DIR

SRC_DIR = os.path.join(V2_DIR, "src")
CHECKPOINTS_DIR = os.path.join(V2_DIR, "checkpoints")
SAMPLES_DIR = os.path.join(V2_DIR, "webapp/static/samples")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F
import config
from inference import FishClassifier

# Definition of the 5 Lutjanus Classes / Stages for Dr. Eftekhari
CLASSES_5_INFO = [
    {
        "id": 1,
        "class_label": "Class 1",
        "stage_label": "Stage 1",
        "code": "C1",
        "stage_code": "S1",
        "species": "Lutjanus johni Bloch 1792",
        "latin": "Lutjanus johni",
        "persian": "ماهی سرخو معمولی (John's Snapper)",
        "sample_file": "Lutjanus johni Bloch 1792.jpg",
        "samples_count": 127,
        "color": "#1f77b4"
    },
    {
        "id": 2,
        "class_label": "Class 2",
        "stage_label": "Stage 2",
        "code": "C2",
        "stage_code": "S2",
        "species": "Lutjanus lutjanus Bloch 1790",
        "latin": "Lutjanus lutjanus",
        "persian": "ماهی سرخو زردباله (Bigeye Snapper)",
        "sample_file": "Lutjanus lutjanus Bloch 1790.jpg",
        "samples_count": 66,
        "color": "#2ca02c"
    },
    {
        "id": 3,
        "class_label": "Class 3",
        "stage_label": "Stage 3",
        "code": "C3",
        "stage_code": "S3",
        "species": "Lutjanus lemniscatus (Valenciennes, 1828)",
        "latin": "Lutjanus lemniscatus",
        "persian": "ماهی سرخو نواری (Yellow-streaked Snapper)",
        "sample_file": "Lutjanus lemniscatus (Valenciennes, 1828).jpg",
        "samples_count": 21,
        "color": "#ff7f0e"
    },
    {
        "id": 4,
        "class_label": "Class 4",
        "stage_label": "Stage 4",
        "code": "C4",
        "stage_code": "S4",
        "species": "lutjanus malabarius(Bloch and Schneider,1801)",
        "latin": "Lutjanus malabaricus",
        "persian": "ماهی سرخو مالابار (Malabar Blood Snapper)",
        "sample_file": "lutjanus malabarius(Bloch and Schneider,1801).jpg",
        "samples_count": 14,
        "color": "#9467bd"
    },
    {
        "id": 5,
        "class_label": "Class 5",
        "stage_label": "Stage 5",
        "code": "C5",
        "stage_code": "S5",
        "species": "Lutjanus fulviflammus(Forsskal 1775)",
        "latin": "Lutjanus fulviflammus",
        "persian": "ماهی سرخو خط سیاه (Dory Snapper - Few-Shot Class)",
        "sample_file": "Lutjanus fulviflammus(Forsskal 1775).jpg",
        "samples_count": 5,
        "color": "#d62728"
    }
]


def load_v2_engine():
    model_path = os.path.join(CHECKPOINTS_DIR, "best_model.pth")
    protos_path = os.path.join(CHECKPOINTS_DIR, "prototypes.pth")
    if not os.path.exists(model_path) or not os.path.exists(protos_path):
        raise FileNotFoundError(f"Checkpoint files not found in: {CHECKPOINTS_DIR}")
    return FishClassifier(model_path=model_path, prototypes_path=protos_path)


def run_5class_demo(image_path: str, output_plot_path: str = "defense_result.png"):
    """
    Executes the 5-Class / 5-Stage Lutjanidae identification demo for Dr. Eftekhari.
    """
    if not os.path.exists(image_path):
        print(f"[Error] Image file not found: {image_path}")
        return
    
    classifier = load_v2_engine()
    
    t0 = time.time()
    pred_species, confidence, all_probs, max_sim, all_sims = classifier.predict(image_path)
    elapsed = time.time() - t0
    
    # Identify Class / Stage
    matched_item = None
    for item in CLASSES_5_INFO:
        if item["species"] == pred_species or item["latin"].lower() in pred_species.lower():
            matched_item = item
            break
            
    if matched_item is None:
        matched_item = CLASSES_5_INFO[0]
        
    print("\n" + "=" * 68)
    print("      AquaLens AI: Few-Shot Biometric Classification Demo")
    print("      5-Class Lutjanidae Snapper Evaluation (Dr. Eftekhari)")
    print("=" * 68)
    print(f" [1] Input Image:      {os.path.basename(image_path)}")
    print(f" [2] Predicted Class:  {matched_item['class_label']} / {matched_item['stage_label']} ({matched_item['code']}/{matched_item['stage_code']})")
    print(f" [3] Scientific Name:  {matched_item['latin']} ({matched_item['persian']})")
    print(f" [4] Softmax Conf:     {confidence * 100:.2f}%")
    print(f" [5] Cosine Metric:    {max_sim * 100:.2f}% (Metric Similarity to Prototype)")
    print(f" [6] Inference Speed:  {elapsed:.3f} seconds ({1.0 / max(1e-4, elapsed):.1f} FPS)")
    print("=" * 68)
    print(" Class-by-Class Similarity Affinities (با نام‌گذاری کلاس‌ها):")
    
    # Print bar chart in terminal
    for item in CLASSES_5_INFO:
        sim = all_sims.get(item["species"], 0.0)
        sim_pct = max(0.0, sim) * 100.0
        bar_len = int(sim_pct / 5.0)
        bar_str = "█" * bar_len + "░" * (20 - bar_len)
        marker = " <=== [TOP-1 MATCH]" if item["id"] == matched_item["id"] else ""
        print(f"   - {item['class_label']} ({item['code']}): {sim_pct:5.1f}% [{bar_str}] {item['latin']:<22}{marker}")
        
    print("=" * 68)
    
    # Generate Visual Result Plot
    try:
        raw_img = Image.open(image_path).convert("RGB")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=180, gridspec_kw={'width_ratios': [1.1, 1.3]})
        fig.patch.set_facecolor('#ffffff')
        
        # Left: Input Image
        ax1.imshow(raw_img)
        title_text = f"Input: {matched_item['class_label']} ({matched_item['code']})\n{matched_item['latin']}"
        ax1.set_title(title_text, fontsize=11, fontweight='bold', pad=10, color='#1e293b')
        ax1.axis("off")
        
        # Right: Bar Chart of Prototype Similarities
        labels = [f"{item['code']}\n({item['stage_code']})" for item in CLASSES_5_INFO]
        sim_values = [max(0.0, all_sims.get(item["species"], 0.0)) * 100.0 for item in CLASSES_5_INFO]
        colors = [item["color"] for item in CLASSES_5_INFO]
        
        bars = ax2.bar(labels, sim_values, color=colors, width=0.55, edgecolor='#0f172a', linewidth=0.8)
        ax2.set_ylabel("Cosine Similarity (%)", fontsize=11, fontweight='bold', color='#1e293b')
        ax2.set_xlabel("Taxonomic Class / Stage (Lutjanidae Family)", fontsize=11, fontweight='bold', color='#1e293b')
        ax2.set_title(f"Prototype Metric Matching: {matched_item['class_label']} (Conf: {confidence*100:.1f}%)", fontsize=11, fontweight='bold', pad=10, color='#0f766e')
        ax2.set_ylim([0, 110])
        ax2.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Add legend for Dr. Eftekhari
        legend_handles = [
            plt.Rectangle((0, 0), 1, 1, color=item["color"],
                          label=f"{item['code']}: {item['latin']} (N={item['samples_count']})")
            for item in CLASSES_5_INFO
        ]
        ax2.legend(handles=legend_handles, loc='upper right', fontsize=8.2, framealpha=0.92)
        
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f"{h:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#0f172a')
            
        plt.tight_layout()
        plt.savefig(output_plot_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
        plt.close()
        print(f"[✓] Visual explanation figure saved to: {output_plot_path}")
    except Exception as e:
        print(f"[Warning] Could not save visualization: {e}")
        
    print("=" * 68 + "\n")


def run_multiclass_demo(image_path: str = None, sample_id: int = None):
    """
    Delegates to the multi-class & novel species demo script for Guilan University supervisor.
    """
    multi_script = os.path.join(ROOT_DIR, "demo_defense_multiclass.py")
    if not os.path.exists(multi_script):
        print(f"[ERROR] Script not found: {multi_script}")
        return
    
    cmd = [sys.executable, multi_script]
    if image_path:
        cmd.extend(["--image", image_path])
    if sample_id:
        cmd.extend(["--sample", str(sample_id)])
        
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="AquaLens Defense Presentation Launcher")
    parser.add_argument("image", nargs="?", default=None, help="Path to input scale image")
    parser.add_argument("--mode", choices=["5", "40", "multi"], default=None,
                        help="Select evaluation mode: '5' for Dr. Eftekhari (5 classes), '40'/'multi' for Supervisor (Multi-class & Novel Species)")
    parser.add_argument("--sample", type=int, default=None, help="Preset sample number for multi-class mode (1-7)")
    parser.add_argument("--test", action="store_true", help="Run automated test on Class 1 sample")
    parser.add_argument("--all", action="store_true", help="Run batch test on all 5 Lutjanus classes")
    args = parser.parse_args()
    
    # Direct flag execution
    if args.mode in ["40", "multi"] or args.sample is not None:
        run_multiclass_demo(args.image, args.sample)
        return
    elif args.mode == "5":
        img_path = args.image or os.path.join(SAMPLES_DIR, CLASSES_5_INFO[0]["sample_file"])
        run_5class_demo(img_path)
        return

    if args.all:
        print("Running verification across all 5 Lutjanus Classes / Stages...")
        for item in CLASSES_5_INFO:
            s_path = os.path.join(SAMPLES_DIR, item["sample_file"])
            run_5class_demo(s_path, f"result_{item['code']}.png")
        return
        
    if args.test or args.image is not None:
        img_path = args.image or os.path.join(SAMPLES_DIR, CLASSES_5_INFO[0]["sample_file"])
        run_5class_demo(img_path)
        return
        
    # Interactive Mode
    is_interactive = sys.stdin.isatty()
    if not is_interactive:
        img_path = os.path.join(SAMPLES_DIR, CLASSES_5_INFO[0]["sample_file"])
        run_5class_demo(img_path)
        return

    print("\n" + "=" * 70)
    print("      AquaLens AI: Interactive Defense Presentation Console")
    print("      سامانه جامع ارائه دفاعیه پروپوزال و تست مدل‌های یادگیری بیومتریک")
    print("=" * 70)
    print(" لطفاً مدلی که می‌خواهید برای استاد تست بگیرید را انتخاب کنید:\n")
    print("  [1] مدل ۵ کلاسه سرخوماهیان (دکتر افتخاری)")
    print("      - نام‌گذاری با کلاس/استیج (Class 1..5 / Stage 1..5)")
    print("      - نمودارهای باریک با فونت درشت و لجند مجزا\n")
    print("  [2] مدل جامع چندگونه‌ای و تست گونه جدید (استاد راهنما - دانشگاه گیلان)")
    print("      - سنجش تعمیم و تست گونه‌های ناشناخته (Unseen / Novel Species)")
    print("      - شبیه‌سازی یادگیری در لحظه با چند نمونه (Few-Shot / One-Shot Learning)")
    print("      - نمایش نقشه حرارتی توجه بیولوژیکی (Grad-CAM)")
    print("=" * 70)
    
    mode_choice = input("شماره مدل را انتخاب کنید [۱ یا ۲] (پیش‌فرض: 1): ").strip()
    if mode_choice == "2":
        run_multiclass_demo()
        return

    # Sub-menu for 5-Class evaluation
    print("\n" + "-" * 70)
    print(" انتخاب نمونه میکروسکوپی از ۵ کلاس سرخوماهیان (Lutjanidae):")
    for item in CLASSES_5_INFO:
        print(f"   [{item['id']}] {item['class_label']} ({item['code']}): {item['latin']:<24} - {item['persian']}")
    print("-" * 70)
    
    sub_choice = input("شماره کلاس را وارد کنید [۱-۵] (پیش‌فرض: 1): ").strip()
    try:
        chosen_idx = int(sub_choice) - 1
        if not (0 <= chosen_idx < len(CLASSES_5_INFO)):
            chosen_idx = 0
    except ValueError:
        chosen_idx = 0
        
    chosen_item = CLASSES_5_INFO[chosen_idx]
    chosen_path = os.path.join(SAMPLES_DIR, chosen_item["sample_file"])
    print(f"\n[انتخاب شد] {chosen_item['class_label']} ({chosen_item['latin']})")
    run_5class_demo(chosen_path)


if __name__ == '__main__':
    main()
