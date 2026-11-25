# 🐟 Marine Fish Scale Microstructure Classification

Automated taxonomic identification of marine teleosts from optical fish scale micrographs using PyTorch Metric Learning and Flask.

## 📁 Repository Overview

This repository contains two complementary pipelines:

| Directory | Scope | Backbone & Method | Web Port | Key Result |
| :--- | :--- | :--- | :---: | :--- |
| **`multiclass_35species/`** | **35 Marine Species + Open-Set** (804 micrographs, $\ge 7$ samples/species; 21 unseen reserved) | `ScaleDetector` + ConvNeXt-Tiny + `MixStyle` + `FDA` + `CB-ArcFace` + 7-View Spherical-TTA | `5004` | **92.25% Top-1** (**95.07% Top-3**, **91.42% 5-Fold CV**) |
| **`lutjanus_5species/`** | **5 *Lutjanus* Species** (233 micrographs, *Lutjanidae* case study) | `ScaleDetector` + ConvNeXt-Tiny + `MixStyle` + `FDA` + Episodic `ProtoNet` + 4-View Spherical-TTA | `5000` | **97.96% Test** (**94.40% 5-Fold CV**) |

## 📂 Structure

```
.
├── multiclass_35species/       # Main 35-species metric learning system & active learning UI
│   ├── src/                    # ScaleDetector, MixStyle, FDA, CB-ArcFace, benchmark, open-set AL, Grad-CAM
│   ├── webapp/                 # Bilingual Flask web app + zero-retraining novel species UI + TIF support
│   └── checkpoints/            # Train-only prototypes, generalization_stats.json & reports (best_model.pth excluded)
├── lutjanus_5species/          # Specialized 5-species Lutjanus case study
│   ├── src/                    # ScaleDetector, MixStyle, FDA, ProtoNet, benchmark & Stage figures
│   ├── webapp/                 # Flask demo web app
│   └── checkpoints/            # Prototypes (best_model.pth >100MB excluded)
├── requirements.txt
└── README.md
```

> **Note:** Raw datasets (`dataset/`, `test_dataset/`, `.tif` slides), local environments (`myenv/`), manuscripts (`paper/`), and large checkpoint weights (`*.pth` > 100 MB) are excluded from Git tracking via `.gitignore`.

## 🚀 Quick Start

### 1. Environment Setup
```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2. Run the 35-Species Web App (Recommended)
```bash
python3 multiclass_35species/webapp/app.py
```
Open `http://localhost:5004` in your browser.

### 3. Run the 5-Species *Lutjanus* Web App
```bash
python3 lutjanus_5species/webapp/app.py
```
Open `http://localhost:5000` in your browser.

### 4. Run Evaluation
```bash
# 35-species comprehensive evaluation
python3 multiclass_35species/src/evaluate.py
```

## 👥 Authors
* **Najmeh Sabbah** — University of Guilan (Marine Biology)
* **Poorya Saneei** — Iran University of Science and Technology (Computer Engineering)
* **Nader Shabanipour** — University of Guilan
* **Majid Askari Hesni** — Shahid Bahonar University of Kerman
* **Mahdi Eftekhari** — Shahid Bahonar University of Kerman
