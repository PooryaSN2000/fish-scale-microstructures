# 🐟 Marine Fish Scale Microstructure Identification

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Backbone](https://img.shields.io/badge/Backbone-ConvNeXt--Tiny-00599C.svg)](https://arxiv.org/abs/2201.03545)
[![Metric-Learning](https://img.shields.io/badge/Head-ArcFace%20Margin-4B0082.svg)](https://arxiv.org/abs/1801.07698)
[![Accuracy](https://img.shields.io/badge/Top--1%20Test%20Accuracy-91.89%25-brightgreen.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-Grad--CAM-blue.svg)]()

An advanced, end-to-end Deep Metric Learning system for automated taxonomic classification of **40 marine fish species** based on microscopic scale patterns (circuli, radii, and ctenii).

---

## 🌟 Key Features

* **High-Accuracy Metric Learning:** Combines **ConvNeXt-Tiny** with **ArcFace (Additive Angular Margin Loss)** ($s=30, m=0.35$) on a 384-dimensional unit hypersphere.
* **Peak Test Performance:** Achieves **91.89% Top-1 Test Accuracy**, **91.32% Weighted F1**, and **93.46% Weighted Precision** across 185 held-out test scale images.
* **Biological Preprocessing:** Applies **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to normalize microscope illumination and sharpen fine growth circuli.
* **Sub-Center Prototypes ($K=2$):** Accounts for intra-species anatomical variations across the fish body (dorsal, ventral, and lateral line scales).
* **Explainable AI (Grad-CAM):** Generates live activation heatmaps showing that the AI focuses on biological structures (focus, radii, ctenii) rather than background ruler markings.
* **Active Learning Feedback Loop:** Interactive web dashboard allows fishery experts to validate or correct predictions, continuously expanding the verified dataset for retraining.
* **Automated Scale Localization:** Morphological scale segmenter automatically crops and centers the scale, stripping stage borders and measurement grids.

---

## 📊 Evaluation Summary (40 Species)

| Metric | Baseline | ConvNeXt + ArcFace (v3.1) | **Full Pipeline (v3.2)** |
| :--- | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | 66.49% | 88.11% | **91.89% (170/185)** |
| **Weighted F1-Score** | 63.80% | 86.89% | **91.32%** |
| **Weighted Precision** | 68.20% | 88.35% | **93.46%** |
| **Macro Recall** | 64.10% | 87.09% | **89.42%** |
| **Input Resolution** | $224 \times 224$ | $288 \times 288$ | **$336 \times 336$** |
| **Contrast Normalization** | None | None | **Adaptive CLAHE** |
| **Visual Explainability** | None | None | **Integrated Grad-CAM** |

---

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── v3/
│   ├── src/
│   │   ├── config.py                 # Configuration and hyperparameters
│   │   ├── dataset.py                # Dataset loader with CLAHE and caching
│   │   ├── model.py                  # FishArcNet architecture & ArcFace head
│   │   ├── train.py                  # Training loop with gradient accumulation
│   │   ├── save_prototypes.py        # Multi-angle & sub-center prototype builder
│   │   ├── inference.py              # Inference engine with 5-pass TTA
│   │   ├── gradcam.py                # Grad-CAM visual explainability module
│   │   ├── scale_detector.py         # Automated scale contour & crop engine
│   │   └── evaluate.py               # Comprehensive 40-class metric evaluation
│   ├── webapp/
│   │   ├── app.py                    # Flask server with /predict & /feedback routes
│   │   ├── templates/index.html      # Responsive dashboard with Grad-CAM & Active Learning
│   │   ├── static/                   # CSS styles and sample scale previews
│   │   └── active_learning_data/     # Expert-validated feedback storage
│   └── checkpoints/                  # 300 DPI evaluation figures and artifacts
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/PooryaSN2000/fish-scale-microstructures.git
cd fish-scale-microstructures

# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Interactive Web Application

```bash
cd v3/webapp
python app.py
```
Open your browser and navigate to:
👉 **`http://localhost:5001`**

### 3. Training & Evaluation Pipeline

```bash
cd v3/src

# Train ConvNeXt-Tiny with ArcFace Metric Learning
python train.py

# Extract augmented prototype centroids and sub-centers
python save_prototypes.py

# Run comprehensive test evaluation and generate publication figures
python evaluate.py
```

---

## 🔬 Explainable AI (Grad-CAM)

The system computes gradients on the final stage feature maps of ConvNeXt-Tiny to produce high-resolution spatial attention overlays. As demonstrated in biological evaluations:
- **Ctenoid Scales:** Focus strictly concentrates on apical ctenial teeth and marginal spines.
- **Cycloid Scales:** Focus concentrates on the central scale nucleus (focus) and concentric circuli rings.
- **Background Invariance:** Measurement grids, ruler marks, and microscope glass blemishes are consistently ignored.

---

## 👥 Authors & Contributors

* **Najmeh Sabbah** — Department of Biology, University of Guilan
* **Poorya Saneei** — Department of Computer Engineering, Iran University of Science and Technology
* **Nader Shabanipour** — Department of Biology, University of Guilan
* **Majid Askari Hesni** — Department of Biology, Shahid Bahonar University of Kerman
* **Mahdi Eftekhari** — Department of Computer Engineering, Shahid Bahonar University of Kerman
