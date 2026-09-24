# 🔬 AquaLens AI - Version 3 Architecture & Source Modules

This directory contains the core implementation of **AquaLens AI v3**, an end-to-end Deep Metric Learning framework for **40-species marine fish scale identification**.

---

## 🛠️ Sub-Module Overview

```text
v3/
├── src/                          # Core Machine Learning pipeline
│   ├── config.py                 # System hyperparameters (336x336 input, d=384 embedding, ArcFace s=30, m=0.35)
│   ├── dataset.py                # 40-species dataset loader with biological CLAHE enhancement
│   ├── model.py                  # FishArcNet (ConvNeXt-Tiny + LayerNorm + Linear Projector + ArcFace margin)
│   ├── train.py                  # Training engine with WeightedRandomSampler, AMP & gradient accumulation
│   ├── save_prototypes.py        # Offline prototype generator (Multi-angle TTA + Sub-Center K-Means, K=2)
│   ├── inference.py              # 5-pass TTA inference engine, similarity mapping & OOD rejection
│   ├── gradcam.py                # Grad-CAM visual explainability module for ConvNeXt stage 3 features
│   ├── scale_detector.py         # Morphological scale localization and bounding-box segmentation
│   ├── retrain_active_learning.py# Automated Active Learning engine (EMA sub-center prototype sync)
│   ├── dataset_analytics.py      # Statistical analytics and class sample distribution visualizer
│   ├── visualize_augmentations.py# Visual demonstration of geometric and biological augmentations
│   └── evaluate.py               # Comprehensive 40-class scientific evaluation (Confusion matrix, t-SNE, ROC, PR)
├── webapp/                       # Interactive Flask Web Application
│   ├── app.py                    # REST API server (/predict, /feedback, and /active_learning/* endpoints)
│   ├── templates/index.html      # Glassmorphic UI with Grad-CAM inspection & Active Learning interface
│   ├── static/                   # CSS styles and sample scale micrographs
│   └── active_learning_data/     # Storage for expert-verified feedback & retraining logs
├── checkpoints/                  # 300 DPI evaluation figures and inference prototypes
│   ├── prototypes.pth            # 40-species prototype & sub-center vectors (~184 KB)
│   ├── classification_report.txt # Detailed precision, recall, and F1 metrics per species
│   ├── confusion_matrix.png      # 300 DPI confusion matrix
│   ├── metric_tsne.png           # 300 DPI t-SNE hyperspherical embedding manifold
│   ├── gradcam_explainability.png# 300 DPI Grad-CAM attention comparison
│   ├── roc_curves.png            # 300 DPI multi-class ROC curves
│   ├── pr_curves.png             # 300 DPI Precision-Recall curves
│   └── class_performance.png     # 300 DPI per-class bar performance
└── requirements.txt              # Python dependencies
```

---

## 🚀 Execution Guide

### 1. Web Application with Grad-CAM & Active Learning
```bash
cd webapp
python app.py
```
Navigate to `http://localhost:5001`.

### 2. Active Learning Synchronization
To synchronize prototypes with verified expert samples from the CLI:
```bash
python src/retrain_active_learning.py --mode sync
```
To inspect feedback bank statistics:
```bash
python src/retrain_active_learning.py --mode stats
```

### 3. Prototype Extraction
```bash
python src/save_prototypes.py
```

### 4. Full Benchmark Evaluation
```bash
python src/evaluate.py
```

For the complete technical breakdown and mathematical formulations, please refer to the primary [Root README.md](../README.md).
