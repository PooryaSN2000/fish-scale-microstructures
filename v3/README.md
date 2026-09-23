# AquaLens AI - Version 3 (40-Species Marine Fish Scale Classifier)

## 📌 Project Overview
**AquaLens AI v3** extends the advanced hybrid deep metric-learning architecture to a broad **40-species marine biodiversity scale classification** system. It combines deep hierarchical representations from **ConvNeXt-Tiny** with classical morphological texture descriptors (**HOG + LBP**) under an **Episodic Few-Shot Prototypical Network** with scaled Cosine Similarity and **Open-Set / Out-of-Distribution (OOD)** detection.

---

## 📂 Directory Structure
```text
v3/
├── src/                          # Core Machine Learning pipeline
│   ├── config.py                 # System hyperparameters (40 classes, N_WAY=40, K_SHOT=5)
│   ├── dataset.py                # Top-40 dynamic loader and stratified Train/Val/Test splitter
│   ├── features.py               # Handcrafted texture extraction (HOG + LBP)
│   ├── model.py                  # Hybrid ProtoNet architecture with VRAM chunking
│   ├── train.py                  # Two-stage training regime (MixUp/CutMix + Episodic FSL)
│   ├── evaluate.py               # Comprehensive scientific evaluation (ROC-AUC, Confusion Matrix, t-SNE)
│   ├── dataset_analytics.py      # Dataset analytics and distribution visualization
│   ├── visualize_augmentations.py# Visual demonstration of CutMix, MixUp, and spatial jitter
│   ├── save_prototypes.py        # Prototype vector computation across 40 classes
│   └── inference.py              # 3-tier OOD detection & 40-species similarity engine
├── webapp/                       # Interactive Web Application
│   ├── app.py                    # Flask server with REST API
│   ├── templates/index.html      # Glassmorphism UI with Top-5 matches & full 40 breakdown
│   └── static/                   # Styles and microscopic sample scales
├── paper/                        # IEEE Conference Paper
│   ├── research_paper.tex        # LaTeX source
│   ├── research_paper.pdf        # Compiled publication-ready PDF (7 pages)
│   └── img/                      # High-resolution 300 DPI figures
├── checkpoints/                  # Trained model weights & 40-class prototype vectors
│   ├── best_model.pth            # Trained ConvNeXt + Classical weights (40 classes)
│   └── prototypes.pth           # 40-species prototype centroid vectors
├── requirements.txt              # Environment dependencies
└── README.md                     # Project documentation
```

---

## 🚀 Quick Start for Reviewers & Users

### 1. Activating the Environment
```bash
# From repository root
source myenv/bin/activate
```

### 2. Running the 40-Species Web Application
```bash
cd v3/webapp
python app.py
```
Open your browser and navigate to:
```
http://localhost:5001
```
Features available in the web interface:
- **Drag & Drop / File Browser:** Upload any microscopic fish scale or arbitrary image.
- **One-Click Demo:** Click *Try a Sample Image* to test instantly with a pre-loaded sample.
- **3-Tier Membership Detection:**
  - 🟢 **Inside Target Group ($\ge 75\%$ similarity):** Confirmed match with species identification.
  - 🟡 **Borderline / Moderate Similarity ($60\% - 75\%$):** Warning for degraded image or closely related species.
  - 🔴 **Not in Group ($< 60\%$):** Clear Out-of-Domain alert to prevent false classification.
- **Top-5 Closest Species:** Prominent visual bar breakdown of the top 5 nearest species.
- **Full 40 Species Breakdown:** Expandable drawer showing exact similarity percentages across all 40 species.

### 3. Re-generating Prototype Centroids
```bash
cd v3
python src/save_prototypes.py
```

### 4. Running Dataset Analytics & Evaluation
```bash
cd v3
# Generate 40-species dataset distribution chart
python src/dataset_analytics.py

# Run comprehensive test evaluation and generate 300 DPI charts
python src/evaluate.py
```

### 5. Compiling the Academic Paper
```bash
cd v3/paper
pdflatex -interaction=nonstopmode research_paper.tex
```

---

## 👥 Authors
- **Najmeh Sabbah** (University of Guilan)
- **Poorya Saneei** (Iran University of Science and Technology)
- **Nader Shabanipour** (University of Guilan)
- **Majid Askari Hesni** (Shahid Bahonar University of Kerman)
- **Mahdi Eftekhari** (Shahid Bahonar University of Kerman)
