# AquaLens AI v4: 35-Species Marine Fish Scale Classification

Domain-invariant Deep Metric Learning pipeline for automated taxonomic identification across **35 validated marine fish species** ($\ge 7$ micrographs per species, 804 images total).

## Key Components
- **MixStyle Domain Generalization:** Perturbs early ConvNeXt-Tiny channel statistics to improve cross-camera and lighting robustness.
- **Class-Balanced ArcFace (CB-ArcFace):** Adaptive angular margins weighted inversely by species frequency to protect minority (tail) taxa.
- **7-View Spherical-TTA & Sub-Centers:** Multi-view hyperspherical pooling matched against $K=2$ species sub-centers.
- **Scale HUD & Grad-CAM:** Saliency localization and gradient heatmaps highlighting biological circuli, radii, and ctenii.
- **Online Active Learning:** Rapid prototype EMA updates triggered from expert feedback in the web UI.

## Verified Test Results (142 Held-Out Micrographs)

| Cohort / Metric | Score |
| :--- | :---: |
| **Overall Top-1 Accuracy** | **97.89%** (139/142) |
| **Overall Top-3 / Top-5 Accuracy** | **100.00%** (142/142) |
| **Head Cohort ($\ge 20$ samples)** | **98.51%** |
| **Medium Cohort (10–19 samples)** | **97.73%** |
| **Tail Cohort ($< 10$ samples)** | **96.77%** |

## Directory Structure
- `src/` — Dataset, augmentations (FDA), MixStyle, CB-ArcFace loss, training, evaluation, Grad-CAM, HUD detector, and inference engine
- `webapp/` — Flask production server (`app.py`, port `5004`)
- `checkpoints/` — Verified prototype vectors (`prototypes.pth`), `classification_report.txt`, `confusion_matrix.png`, and `loss_curves.png`

## Quick Start

### Launch Web App
```bash
python3 multiclass_35species/webapp/app.py
```
Open `http://127.0.0.1:5004` in your browser.

### Run Evaluation or Training
```bash
python3 multiclass_35species/src/evaluate.py
PYTHONPATH=. python3 -m multiclass_35species.src.train
```
