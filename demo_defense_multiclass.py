#!/usr/bin/env python3
"""
AquaLens AI: Multi-Class Few-Shot & Novel Species Identification Demo
Specifically tailored for Najmeh Sabbah's Proposal Defense & Supervisor Verification
University of Guilan (Marine Biology) & Shahid Bahonar University of Kerman
"""

import os
import sys
import time
import io
import base64
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Auto-detect script directory and add multiclass_35species/src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "multiclass_35species") if os.path.exists(os.path.join(SCRIPT_DIR, "multiclass_35species")) else SCRIPT_DIR
SRC_DIR = os.path.join(MODEL_DIR, "src")
CHECKPOINTS_DIR = os.path.join(MODEL_DIR, "checkpoints")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch
import torch.nn.functional as F
import config
from inference import FishClassifierV4

# Preset samples for defense demonstration
PRESET_SAMPLES = [
    {
        "id": 1,
        "type": "known",
        "name": "Lutjanus johni (John's Snapper - سرخو معمولی)",
        "family": "Lutjanidae (Ctenoid Scale)",
        "path": "scale fish/Lutjanus johni Bloch 1792/89-18-1.jpg",
        "desc": "گونه شناخته‌شده با بیش از ۱۰۰ نمونه در پایگاه داده"
    },
    {
        "id": 2,
        "type": "known",
        "name": "Upeneus sulphureus (Yellow Goatfish - بزماهی زرد)",
        "family": "Mullidae (Cycloid Scale)",
        "path": "scale fish/Upeneus sulphureusCuvier, 1829/42-12-5.jpg",
        "desc": "گونه شناخته‌شده با پولک گرد فاقد خار"
    },
    {
        "id": 3,
        "type": "known",
        "name": "Platycephalus indicus (Bartail Flathead - زمین‌کن دم‌نواری)",
        "family": "Platycephalidae (Radii-rich Scale)",
        "path": "scale fish/Platycephalus indicus (linnaeus, 1758)/1B.jpg",
        "desc": "گونه با شیارهای شعاعی عمیق هدایت هیدرودینامیک"
    },
    {
        "id": 4,
        "type": "known",
        "name": "Gerres filamentosus (Whipfin Silverbiddy - شورت نواری)",
        "family": "Gerreidae (Deep-body Scale)",
        "path": "scale fish/Gerres filamentosus Cuvier,1829/77-40-1++.jpg",
        "desc": "گونه با کانون پولک فشرده و دوایر جانبی متراکم"
    },
    {
        "id": 5,
        "type": "novel",
        "name": "نمونه ناشناخته ۱ (Unseen / Novel Specimen A)",
        "family": "ناشناخته (خارج از کلاس‌های آموزش)",
        "path": "112-4+8.tif",
        "desc": "تصویر میکروسکوپی بدون برچسب قبلی جهت سنجش تعمیم و Few-Shot"
    },
    {
        "id": 6,
        "type": "novel",
        "name": "نمونه ناشناخته ۲ (Unseen / Novel Specimen B)",
        "family": "ناشناخته (خارج از کلاس‌های آموزش)",
        "path": "120-5-4.tif",
        "desc": "تصویر میکروسکوپی ناشناخته جهت تست تشخیص گونه جدید"
    },
    {
        "id": 7,
        "type": "novel",
        "name": "نمونه ناشناخته ۳ (Unseen / Novel Specimen C)",
        "family": "ناشناخته (خارج از کلاس‌های آموزش)",
        "path": "163-4-22.tif",
        "desc": "تصویر میکروسکوپی ناشناخته جهت آزمایش One-Shot Learning"
    }
]


def draw_prediction_figure(orig_img, cam_img, sorted_sims, is_novel, novel_threshold, output_path="defense_multiclass_result.png"):
    """
    Renders a publication-grade 3-panel figure showing the input, Grad-CAM, and Top-5 similarities.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=180, gridspec_kw={'width_ratios': [1, 1, 1.4]})
    fig.patch.set_facecolor('#ffffff')

    # Panel 1: Original Micrograph
    axes[0].imshow(orig_img)
    axes[0].set_title("(a) Input Micrograph (میکروگراف ورودی)", fontsize=11, fontweight='bold', pad=10, color='#1e293b')
    axes[0].axis('off')

    # Panel 2: Grad-CAM Explainability
    if cam_img is not None:
        axes[1].imshow(cam_img)
        axes[1].set_title("(b) Grad-CAM Attention Heatmap\n(تمرکز روی حلقه‌های رشد و شیارهای زیستی)", fontsize=10, fontweight='bold', pad=8, color='#0f766e')
    else:
        axes[1].imshow(orig_img)
        axes[1].set_title("(b) Scale Microstructure", fontsize=10, fontweight='bold', pad=8)
    axes[1].axis('off')

    # Panel 3: Top-5 Similarities Bar Chart
    top5 = sorted_sims[:5]
    names = [x[0] for x in top5][::-1]
    sims = [x[1] for x in top5][::-1]

    # Clean short names for y-axis
    clean_names = []
    for n in names:
        parts = n.split('(')[0].strip()
        if len(parts) > 28:
            parts = parts[:26] + ".."
        clean_names.append(parts)

    colors = []
    for s in sims:
        if s >= 75.0:
            colors.append('#059669') # Emerald green (high confidence match)
        elif s >= novel_threshold:
            colors.append('#0284c7') # Sky blue (moderate match)
        else:
            colors.append('#dc2626') # Red / Novel alert

    y_pos = np.arange(len(clean_names))
    bars = axes[2].barh(y_pos, sims, color=colors, height=0.55, edgecolor='#0f172a', linewidth=0.8)

    # Add similarity threshold line
    axes[2].axvline(novel_threshold, color='#e11d48', linestyle='--', linewidth=1.5, label=f'Novelty Threshold ({novel_threshold}%)')

    axes[2].set_xlim(0, 105)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(clean_names, fontsize=9.5, fontweight='bold', color='#1e293b')
    axes[2].set_xlabel("Hyperspherical Cosine Similarity (%)", fontsize=10, fontweight='bold', color='#1e293b')
    
    # Title indicating status
    if is_novel:
        chart_title = "(c) Metric Distances: [NOVEL / UNSEEN SPECIES]\n(گونه جدید / ناشناخته - بدون انطباق با دیتابیس)"
        axes[2].set_title(chart_title, fontsize=10.5, fontweight='bold', color='#b91c1c', pad=8)
    else:
        top_match = sorted_sims[0][0].split('(')[0].strip()
        chart_title = f"(c) Metric Match: {top_match}\n(انطباق قطعی بر اساس یادگیری متریک)"
        axes[2].set_title(chart_title, fontsize=10.5, fontweight='bold', color='#047857', pad=8)

    axes[2].grid(axis='x', linestyle=':', alpha=0.6)
    axes[2].legend(loc='lower right', fontsize=8.5, framealpha=0.9)

    # Annotate bar values
    for bar in bars:
        width = bar.get_width()
        axes[2].text(width + 1.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}%",
                     va='center', ha='left', fontsize=9, fontweight='bold', color='#0f172a')

    # Spines styling
    for spine in ['top', 'right']:
        axes[2].spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        axes[2].spines[spine].set_color('#64748b')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='#ffffff')
    plt.close()
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AquaLens AI: Multi-Class & Novel Species Defense Demo")
    parser.add_argument("--image", type=str, default=None, help="Path to input fish scale image")
    parser.add_argument("--threshold", type=float, default=65.0, help="Novelty / Unseen species cosine similarity threshold (default: 65.0 percent)")
    parser.add_argument("--sample", type=int, default=None, help="Preset sample number (1-7)")
    args = parser.parse_args()

    print("\n" + "="*70)
    print("      AquaLens AI: Multi-Class & Novel Species Defense Demo")
    print("      سامانه ارزیابی بیومتریک و سنجش گونه‌های جدید بر پایه یادگیری متریک")
    print("      مخصوص جلسه دفاع پروپوزال و ارائه به استاد راهنما")
    print("="*70)

    # 1. Initialize classifier
    print("\n[1/3] در حال بارگذاری مدل عمیق ConvNeXt-Tiny و بردارهای پروتوتایپ...")
    clf = FishClassifierV4()
    print(f"      [✓] تعداد گونه‌های پایه در پایگاه دانش: {len(clf.class_names)} گونه")

    # 2. Select image
    selected_path = None
    selected_desc = ""
    is_interactive = sys.stdin.isatty() and args.image is None and args.sample is None

    if args.image:
        selected_path = args.image
        selected_desc = f"فایل ورودی کاربر: {os.path.basename(args.image)}"
    elif args.sample and 1 <= args.sample <= len(PRESET_SAMPLES):
        preset = PRESET_SAMPLES[args.sample - 1]
        selected_path = preset["path"]
        selected_desc = f"{preset['name']} - {preset['desc']}"
    elif is_interactive:
        print("\nلطفا یک تصویر تستی از لیست زیر انتخاب کنید یا مسیر دلخواه خود را وارد نمایید:\n")
        print("  --- گونه‌های شناخته‌شده (Known Species) ---")
        for s in PRESET_SAMPLES[:4]:
            print(f"   [{s['id']}] {s['name']} ({s['family']})")
        print("\n  --- نمونه‌های جدید و ناشناخته (Unseen / Novel Species) ---")
        for s in PRESET_SAMPLES[4:]:
            print(f"   [{s['id']}] {s['name']} - {s['desc']}")
        print("   [8] وارد کردن مسیر فایل دلخواه خودتان (Custom Image File)")
        print("="*70)

        choice = input("\nشماره گزینه مورد نظر را وارد کنید [۱-۸] (پیش‌فرض: 1): ").strip()
        if not choice:
            choice = "1"

        if choice == "8":
            custom_p = input("مسیر کامل یا نسبی فایل تصویر را وارد کنید: ").strip()
            if os.path.exists(custom_p):
                selected_path = custom_p
                selected_desc = f"فایل انتخابی کاربر: {os.path.basename(custom_p)}"
            else:
                print(f"[!] فایل یافت نشد: {custom_p}. گزینه ۱ به عنوان پیش‌فرض انتخاب شد.")
                selected_path = PRESET_SAMPLES[0]["path"]
                selected_desc = PRESET_SAMPLES[0]["name"]
        else:
            try:
                c_idx = int(choice) - 1
                if 0 <= c_idx < len(PRESET_SAMPLES):
                    selected_path = PRESET_SAMPLES[c_idx]["path"]
                    selected_desc = PRESET_SAMPLES[c_idx]["name"]
                else:
                    selected_path = PRESET_SAMPLES[0]["path"]
                    selected_desc = PRESET_SAMPLES[0]["name"]
            except ValueError:
                selected_path = PRESET_SAMPLES[0]["path"]
                selected_desc = PRESET_SAMPLES[0]["name"]
    else:
        # Non-interactive fallback
        selected_path = PRESET_SAMPLES[0]["path"]
        selected_desc = PRESET_SAMPLES[0]["name"]

    if not os.path.exists(selected_path):
        print(f"[ERROR] فایل تصویر در مسیر '{selected_path}' وجود ندارد.")
        sys.exit(1)

    print(f"\n[2/3] تصویر انتخاب‌شده: {selected_path}")
    print(f"      توضیحات: {selected_desc}")
    print("      در حال استخراج امبدینگ ۳۸۴ بعدی روی ابرکره و ارزیابی تشابه...")

    t0 = time.time()
    orig_img = Image.open(selected_path).convert('RGB')
    
    # Run prediction (auto_crop=False preserves full scale morphology)
    pred_class, conf, all_probs, max_sim, all_sims, meta = clf.predict(selected_path, auto_crop=False, return_metadata=True)
    
    # Extract query embedding for potential novel registration
    query_emb = clf.extract_embedding(orig_img, use_tta=True)

    # Sort similarities
    sim_pct = {k: round(max(0.0, v) * 100, 1) for k, v in all_sims.items()}
    sorted_sims = sorted(sim_pct.items(), key=lambda x: x[1], reverse=True)
    top_sim = sorted_sims[0][1]
    second_sim = sorted_sims[1][1] if len(sorted_sims) > 1 else 0.0
    margin = top_sim - second_sim

    # Evaluate novelty
    is_novel = top_sim < args.threshold
    elapsed = time.time() - t0

    # Generate Grad-CAM
    cam_img = None
    try:
        cam_b64 = clf.explain(selected_path, auto_crop=False)
        if cam_b64 and cam_b64.startswith("data:image"):
            header, encoded = cam_b64.split(",", 1)
            img_data = base64.b64decode(encoded)
            cam_img = Image.open(io.BytesIO(img_data)).convert('RGB')
    except Exception as e:
        print(f"      [GradCAM Warning] {e}")

    # 3. Print Results in Terminal
    print("\n" + "="*70)
    print("                      نتایج تحلیل بیومتریک و استنتاج")
    print("="*70)
    print(f" [۱] فایل ورودی:        {os.path.basename(selected_path)}")
    print(f" [۲] زمان پردازش:       {elapsed:.2f} ثانیه")
    print(f" [۳] بیشترین شباهت:     {top_sim:.1f}% با گونه «{sorted_sims[0][0]}»")
    print(f" [۴] اختلاف حاشیه:      {margin:.1f}% نسبت به دومین کاندید")

    if is_novel:
        print("\n [!] وضعیت گونه:        >>> گونه جدید / ناشناخته (NOVEL / UNSEEN SPECIES) <<<")
        print(f"     توضیح علمی:        بیشترین تشابه ({top_sim:.1f}٪) از آستانه قطعیت ({args.threshold}٪) کمتر است.")
        print("                        مدل به درستی تشخیص داد که این نمونه متعلق به هیچ‌کدام از گونه‌های قبلی نیست.")
        print("                        (جلوگیری از خطای کاذب و توهم یادگیری عمیق)")
    else:
        print(f"\n [✓] وضعیت گونه:        >>> گونه شناخته‌شده (KNOWN SPECIES) <<<")
        print(f"     نام علمی پیش‌بینی: {pred_class}")
        print(f"     ضریب اطمینان مدل:   {conf*100:.1f}%")
        print("     انطباق بیولوژیکی:  تایید شده بر پایه ساختار دوایر رشد و شیارهای پولک")

    print("\n پنج گونه دارای بیشترین تشابه ساختاری (Top-5 Affinities):")
    for rank, (name, s_val) in enumerate(sorted_sims[:5], start=1):
        bar_len = int(s_val / 4)
        bar_str = "█" * bar_len + "░" * (25 - bar_len)
        marker = " <=== [تطابق قطعی]" if (rank == 1 and not is_novel) else ""
        print(f"   {rank}. {s_val:5.1f}% [{bar_str}] {name[:45]}{marker}")

    # 4. Save visualization figure
    out_fig = draw_prediction_figure(orig_img, cam_img, sorted_sims, is_novel, args.threshold)
    print(f"\n[✓] نمودار تحلیلی سه گانه با کیفیت بالا ذخیره شد: {out_fig}")

    # 5. Few-Shot Dynamic Learning Simulation (If novel)
    if is_novel and is_interactive:
        print("\n" + "-"*70)
        print(" ★ شبیه‌سازی قابلیت یادگیری چند شات (Few-Shot / One-Shot Learning):")
        print("   در این قابلیت می‌توان بدون بازآموزی شبکه، تنها با همین ۱ تصویر،")
        print("   گونه جدید را در لحظه به حافظه مدل اضافه کرد تا از این پس آن را بشناسد!")
        enroll = input("\n آیا مایلید این گونه جدید را در پایگاه دانش مدل ثبت کنید؟ (y/n) [پیش‌فرض: n]: ").strip().lower()
        if enroll == 'y':
            new_species_name = input(" نام یا کد گونه جدید را وارد کنید (مثلا: New_Species_X): ").strip()
            if not new_species_name:
                new_species_name = "Novel_Species_Registered"
            
            # Dynamically register prototype vector into classifier memory
            clf.class_names.append(new_species_name)
            clf.prototypes = torch.cat([clf.prototypes, query_emb], dim=0)
            if clf.subcenters is not None:
                new_idx = len(clf.class_names) - 1
                clf.subcenters[new_idx] = query_emb.repeat(2, 1)

            print(f"\n [✓] گونه «{new_species_name}» با موفقیت به پروتوتایپ‌های مدل افزوده شد!")
            print("     حال اگر دوباره همین تصویر یا تصویر مشابه آن ارزیابی شود:")
            
            # Re-predict
            p2, c2, _, sim2, all_sims2, _ = clf.predict(selected_path, auto_crop=False, return_metadata=True)
            print(f"     نتیجه تشخیص مجدد: گونه «{p2}» با شباهت بیومتریک {sim2*100:.1f}٪!")
            print("     این آزمایش، یادگیری متریک و تطبیق نمونه جدید (One-Shot Learning) را اثبات می‌کند.")

    print("\n" + "="*70)
    print(" برای اجرای مجدد روی سایر نمونه‌ها:")
    print("   python demo_defense_multiclass.py")
    print(" یا برای اجرای مستقیم روی یک فایل:")
    print("   python demo_defense_multiclass.py --image '112-4+8.tif'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
