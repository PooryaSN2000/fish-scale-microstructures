# AquaLens AI: 5-Species *Lutjanus* Few-Shot Scale Classification

Specialized, domain-generalizable metric-learning framework for taxonomic identification of five congeneric snapper species (*Lutjanus* genus, 233 optical scale micrographs) under severe 25:1 class imbalance.

## Architecture & Pipeline
- **Biological Localization (`ScaleDetector`):** Unsupervised Sobel + corner-contrast morphological ROI detector that isolates the elasmoid scale body and strips slide margins, timestamps, neighbor fragments, and calibration rulers prior to encoding.
- **Domain Generalization & Preprocessing:** $336 \times 336$ CIE-LAB CLAHE enhancement, batch-level Fourier Domain Adaptation (`FDA`, $\beta=0.08$), and internal channel-statistics `MixStyle` ($p=0.5, \alpha=0.3$) interleaved inside ConvNeXt-Tiny (`blocks 1 & 3`).
- **Metric Space, OOD Gate & WebApp:** Two-stage training (22-epoch balanced Stage 1 + 20-epoch unfrozen episodic 5-way 5-shot Stage 2) producing 256-d $L_2$-normalized hyperspherical prototypes, 4-View Spherical-TTA, validation-calibrated 768-d Ledoit-Wolf Mahalanobis + cosine Dual-Manifold OOD rejection, Top-3 Shortlist Consensus cards, native `.tif`/`.tiff` support, and 4-column Grad-CAM visual attribution across Stages $S_1$–$S_5$.
- **Target Stages:**
  - Stage 1 ($S_1$): *Lutjanus johni* ($N=127$)
  - Stage 2 ($S_2$): *Lutjanus lutjanus* ($N=66$)
  - Stage 3 ($S_3$): *Lutjanus lemniscatus* ($N=21$)
  - Stage 4 ($S_4$): *Lutjanus malabaricus* ($N=14$)
  - Stage 5 ($S_5$): *Lutjanus fulviflammus* ($N=5$, Few-Shot)

## Verified Benchmark & Statistical Results
- **Held-Out Test Split ($N=49$):** **97.96% Accuracy** (48/49), **0.9807 Weighted F1**, **0.9640 Macro F1**, **100%** recall on $S_1, S_3, S_4, S_5$ (**97.85%** across all 233 images).
- **5-Fold Stratified Cross-Validation ($N=233$):** **94.40% $\pm$ 2.96%** mean accuracy (**100%** OOF recall on $S_1$ and $S_5$).
- **Congeneric Statistical Separation:** Intra-stage cosine $0.8634 \pm 0.1848$ vs. Inter-stage $-0.2068 \pm 0.2399$ (Welch's $t=34.00$, $p=3.29\times 10^{-54}$, Cohen's $d=4.65$).
- **Dual-Manifold OOD Detection (49 ID vs. 52 Non-*Lutjanus*/External Slides):** **0.9653 Fused OOD AUROC** (`0.9572` Mahalanobis-768 vs. `0.9109` raw cosine), halving `FPR@90TPR` from `32.69%` to **15.38%** and flagging **98.08%** of out-of-group inputs.
- **Cross-Camera Stress & Attribution Proof:** Preserves **93.88%–97.96%** accuracy under unseen gamma/CMOS/blur perturbations; removing or masking the scale interior drops mean cosine similarity from **0.8814** down to **0.6104** (`59.18%` accuracy, background only) and **0.5956** (`71.43%` accuracy, ROI center masked).

## Directory Structure
- `src/` — `scale_detector.py`, `mixstyle.py`, `augmentations.py`, `dataset.py`, `model.py`, `train.py`, `inference.py`, `benchmark_generalization.py`, `generate_v2_stage_figures.py`
- `webapp/` — Flask web application (`app.py`, port `5000`)
- `checkpoints/` — Saved `prototypes.pth`, `metrics_summary.json`, `generalization_stats.json`

## Quick Start

### Launch Web App
```bash
python3 lutjanus_5species/webapp/app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Train, Benchmark & Regenerate Figures
```bash
PYTHONPATH=lutjanus_5species/src python3 lutjanus_5species/src/train.py
PYTHONPATH=lutjanus_5species/src python3 lutjanus_5species/src/save_prototypes.py
PYTHONPATH=lutjanus_5species/src python3 lutjanus_5species/src/benchmark_generalization.py
PYTHONPATH=lutjanus_5species/src python3 lutjanus_5species/src/generate_v2_stage_figures.py
```
