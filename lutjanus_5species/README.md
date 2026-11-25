# AquaLens AI: 5-Species *Lutjanus* Few-Shot Scale Classification

Specialized case study for taxonomic identification of 5 congeneric snapper species (*Lutjanus* family, 233 microscopic scale images) under severe class imbalance (25:1 ratio).

## Architecture & Pipeline
- **Preprocessing:** $336 \times 336$ spatial resolution with Biological CLAHE contrast enhancement (`apply_clahe_preprocessing`).
- **Backbone & Metric Space:** `convnext_tiny` projected into a 256-d L2-normalized hyperspherical embedding space with Episodic Prototypical fine-tuning (upper ConvNeXt blocks unfrozen) and scaled cosine similarity.
- **Inference & Explainability:** 4-View Spherical Test-Time Augmentation (Spherical-TTA) paired with localized Grad-CAM visual attention heatmaps across all 5 Stages ($S_1$–$S_5$).
- **Target Stages:**
  - Stage 1 ($S_1$): *Lutjanus johni* ($N=127$)
  - Stage 2 ($S_2$): *Lutjanus lutjanus* ($N=66$)
  - Stage 3 ($S_3$): *Lutjanus lemniscatus* ($N=21$)
  - Stage 4 ($S_4$): *Lutjanus malabaricus* ($N=14$)
  - Stage 5 ($S_5$): *Lutjanus fulviflammus* ($N=5$, Few-Shot)

## Verified Test Results
- **Held-Out Test Split (49 micrographs):** **97.96% Accuracy** (48/49), **0.9807 Weighted F1**, **0.9640 Macro F1** (Stage 4 Recall: **100%**, Stage 5 Recall: **100%**).
- **Complete Repository (233 micrographs):** **99.14% Accuracy** (231/233).

## Directory Structure
- `src/` — Dataset loader, CLAHE preprocessor, `ProtoNet`, two-stage training, Grad-CAM explainer, evaluation, and Stage figure generation scripts
- `webapp/` — Flask web application (`app.py`, port `5000`)
- `checkpoints/` — Saved train-only prototype vectors (`prototypes.pth`) and `metrics_summary.json`

## Quick Start

### Launch Web App
```bash
python3 lutjanus_5species/webapp/app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Train & Regenerate Paper Figures
```bash
PYTHONPATH=lutjanus_5species/src python3 lutjanus_5species/src/train.py
PYTHONPATH=lutjanus_5species/src python3 lutjanus_5species/src/generate_v2_stage_figures.py
```
