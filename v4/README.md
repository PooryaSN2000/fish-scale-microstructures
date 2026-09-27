# AquaLens AI v4: Domain-Invariant Metric Learning for Fish Scale Microstructure Identification

AquaLens AI v4 represents the next-generation iteration of the automated fish scale biometric identification framework. It is specifically engineered to solve two fundamental open challenges in ichthyological computer vision:
1. **Extreme Few-Shot / Long-Tailed Imbalance**: Providing high identification accuracy even for rare species with as few as 2 to 4 micrographs, preventing dominant classes from overwhelming minority species.
2. **Cross-Camera & Sensor Invariance (Domain Generalization)**: Ensuring robust classification when specimens are captured using unfamiliar microscope cameras, varying objective lenses, different illumination spectra (halogen vs. LED), and sensor noise profiles.

---

## Key Technological Innovations in v4

### 1. Embedded MixStyle Architecture
- **Problem**: Different digital microscope cameras introduce distinct chromatic casts, vignetting, and contrast shifts in the early feature maps.
- **Solution**: Interleaved `MixStyle` layers between Stage 1 and Stage 2 of the `ConvNeXt-Tiny` backbone. MixStyle mixes feature mean ($\mu$) and standard deviation ($\sigma$) across random batch instances during training. Since feature statistics encode domain/camera style while normalized activations encode geometric microstructures, the network learns to ignore camera optical variations.

### 2. Class-Balanced ArcFace Loss (CB-ArcFace)
- **Problem**: In conventional ArcFace, a uniform margin $m$ allows populous head species to dominate the hypersphere, crowding out few-shot tail species.
- **Solution**: Dynamic class-adaptive angular margins:
  $$m_c = m_{base} + \Delta m \cdot \left(1 - \left(\frac{N_c}{N_{max}}\right)^\gamma\right)$$
  - Frequent species ($N_c \approx 88$): $m_c = 0.300\text{ rad}$
  - Few-shot species ($N_c \approx 2$): $m_c = 0.468\text{ rad}$
  This enforces tight, hyper-compact spherical clusters for rare species, preventing boundary intrusion.

### 3. Fourier Domain Augmentation (FDA)
- Exchanges low-frequency amplitude spectra between micrographs during training while keeping 100% of the spatial phase spectrum intact. Biological circuli, radii ridges, and ctenii teeth are preserved with exact physical fidelity, while global lighting profiles are diversified.

### 4. Spherical Test-Time Augmentation (TTA)
- During inference, multi-view orientations (horizontal flip, vertical flip, $180^\circ$ rotation, center scale jitter) are processed. Embeddings are averaged directly on the unit hypersphere:
  $$\mathbf{e}_{TTA} = \frac{\sum_{k=1}^K \mathbf{e}_k}{\left\|\sum_{k=1}^K \mathbf{e}_k\right\|_2}$$
  This neutralizes slide placement rotation and camera angle biases without requiring retraining.

---

## Empirical Benchmark Results

Evaluated on **163 unseen test micrographs** across all **57 marine fish species**:

| Cohort Bracket | Sample Range | Species Count | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Head Species** | $\ge 20$ samples | 7 taxa | **95.45%** (63/66) |
| **Medium Species** | $10 - 19$ samples | 13 taxa | **86.36%** (38/44) |
| **Tail / Few-Shot** | $< 10$ samples (2–5) | 37 taxa | **90.57%** (48/53) |
| **Overall Top-1** | — | **57 taxa** | **91.41%** |
| **Overall Top-3** | — | **57 taxa** | **98.77%** |
| **Overall Top-5** | — | **57 taxa** | **100.00%** |

> **Key Takeaway**: The tail cohort achieved **90.57%** accuracy despite extreme sample scarcity (37 species had fewer than 10 total samples), proving the effectiveness of Class-Balanced ArcFace and MixStyle.

---

## Directory Structure

```
v4/
├── src/
│   ├── config.py              # Central hyperparameters and paths
│   ├── mixstyle.py            # Domain generalization module
│   ├── loss.py                # Class-Balanced ArcFace (CB-ArcFace)
│   ├── augmentations.py       # FDA & Cross-camera sensor transformations
│   ├── model.py               # ConvNeXt-Tiny with MixStyle & hyperspherical head
│   ├── dataset.py             # Stratified dataset with WeightedRandomSampler
│   ├── scale_detector.py      # Biological texture saliency HUD localizer
│   ├── gradcam.py             # Explainability heatmap generator
│   ├── train.py               # Two-stage training pipeline
│   ├── inference.py           # Inference engine with Spherical-TTA
│   ├── evaluate.py            # Multi-cohort evaluation and confusion matrix
│   └── save_prototypes.py     # Prototype extraction script
├── checkpoints/
│   ├── best_model.pth         # Trained PyTorch checkpoint
│   ├── prototypes.pth         # 57 hyperspherical calibrated prototypes
│   ├── loss_curves.png        # Training & validation loss curves
│   ├── confusion_matrix.png   # 57-class confusion matrix
│   └── classification_report.txt # Detailed precision/recall/F1 metrics
├── webapp/
│   ├── app.py                 # Flask server (Port 5004)
│   ├── templates/index.html   # Minimalist light-mode bilingual UI (Two-line layout)
│   └── static/                # CSS and sample micrographs
└── README.md
```

---

## Running v4

### 1. Launch Web Application
```bash
./myenv/bin/python3 v4/webapp/app.py
```
Open `http://localhost:5004` in your browser.

### 2. Run Comprehensive Evaluation
```bash
PYTHONPATH=v4/src ./myenv/bin/python3 v4/src/evaluate.py
```

### 3. Retrain Pipeline
```bash
PYTHONPATH=. ./myenv/bin/python3 -m v4.src.train
```
