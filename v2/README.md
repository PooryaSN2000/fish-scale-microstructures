# AquaLens AI: 5-Way Few-Shot Classification of Fish Species

<div align="center">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status"/>
  <img src="https://img.shields.io/badge/Python-3.11+-blue" alt="Python"/>
  <img src="https://img.shields.io/badge/Framework-PyTorch_&_Flask-orange" alt="Tech"/>
</div>

## 📌 Project Overview
This repository contains the complete implementation for the research paper: **"Hybrid Deep Metric Learning for 5-Way Few-Shot Classification of Fish Species: A Lutjanus Family Case Study"**. 

It addresses the severe data imbalance in microscopic fish scale datasets using a Two-Stage training regime involving a frozen **ConvNeXt backbone**, classical **HOG/LBP features**, and an **Episodic Prototypical Network**.

این پروژه شامل یک سیستم کامل یادگیری هوش مصنوعی (Few-Shot Metric Learning) برای تشخیص ۵ گونه از خانواده‌ی ماهی‌های `Lutjanus` بر اساس تصاویر میکروسکوپی از فلس آن‌ها است.

---

## 👨‍🏫 Quick Start for Reviewers (اجرای سریع برای داوران و اساتید)

To test the final Web Application and verify the model's accuracy, simply run the pre-configured Flask server.

1. **Install Requirements / فعال‌سازی محیط:**
```bash
pip install -r requirements.txt
```

2. **Run the Web App / اجرای سایت:**
```bash
cd webapp
python app.py
```

3. **Open in Browser / مشاهده خروجی:**
Open your browser and navigate to `http://127.0.0.1:5000`. You can use the **"Try a Sample Image"** button for a one-click demonstration of the AI's capabilities!

---

## 📂 Directory Structure
- **`src/`**: Source code for data processing, the neural network (ProtoNet), and training/evaluation scripts.
- **`webapp/`**: Source code for the interactive Flask web application featuring a Glassmorphism UI.
- **`paper/`**: LaTeX source code and compiled PDF of the research paper (`research_paper.pdf`).
- **`checkpoints/`**: Directory storing the trained model (`best_model.pth`), metric prototypes, and evaluation charts (ROC, t-SNE, Confusion Matrix).

---

## 🧪 Scientific Evaluation (اجرای کدهای هوش مصنوعی)
All scientific scripts must be run inside the `src/` directory.

### 1. Train the Model (Two-Stage Pipeline)
```bash
cd src
python train.py
```
*This will automatically apply spatial/color augmentations, MixUp, and CutMix, followed by the Episodic Fine-tuning.*

### 2. Generate Evaluation Metrics
```bash
python evaluate.py
```
*This generates the `roc_curves.png`, `confusion_matrix.png`, `tsne_embeddings.png`, and more inside the `checkpoints` directory.*

### 3. Generate Augmentation Visuals
```bash
python visualize_augmentations.py
```

---
**Developed by Najmeh Sabbah, Poorya Saneei, et al.** 
*University of Guilan & Iran University of Science and Technology*


cd "/run/media/pooryasn/Personal/Code Projects/NajmehSabaah/v2/webapp"
source ../../myenv/bin/activate