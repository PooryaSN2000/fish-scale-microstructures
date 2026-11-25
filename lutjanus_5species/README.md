# AquaLens AI: 5-Species *Lutjanus* Few-Shot Scale Classification

Specialized case study for taxonomic identification of 5 congeneric snapper species (*Lutjanus* family, 233 microscopic scale images) under extreme class imbalance.

## Architecture
- **Backbone:** `convnext_tiny` (256-d deep projection) fused with classical HOG + LBP descriptors (128-d projection)
- **Training:** Two-stage transfer learning (Supervised pre-training + Episodic Few-Shot Prototypical fine-tuning with scaled cosine distance)
- **Target Classes:** *Lutjanus johni*, *Lutjanus lutjanus*, *Lutjanus lemniscatus*, *Lutjanus malabaricus*, and *Lutjanus fulviflammus*

## Directory Structure
- `src/` — Dataset loader, feature extractor, `ProtoNet` model, training, evaluation, and prototype generation scripts
- `webapp/` — Flask web application (`app.py`, port `5000`)
- `checkpoints/` — Saved prototype vectors (`prototypes.pth`)

## Quick Start

### Launch Web App
```bash
python3 lutjanus_5species/webapp/app.py
```
Open `http://127.0.0.1:5000` in your browser.

### Train & Evaluate
```bash
cd lutjanus_5species/src
python3 train.py
python3 evaluate.py
```
