# 🐟 AquaLens AI v4: Domain-Invariant Deep Metric Learning for 57-Species Marine Fish Scale Biometrics

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Backbone](https://img.shields.io/badge/Backbone-ConvNeXt--Tiny-00599C.svg)](https://arxiv.org/abs/2201.03545)
[![Domain-Generalization](https://img.shields.io/badge/Domain%20Generalization-MixStyle-22c55e.svg)](https://arxiv.org/abs/2104.02008)
[![Metric-Learning](https://img.shields.io/badge/Loss-Class--Balanced%20ArcFace-7c3aed.svg)]()
[![Species](https://img.shields.io/badge/Species%20Coverage-57%20Taxa-0284c7.svg)]()
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1%20Test%20Accuracy-91.41%25-brightgreen.svg)]()
[![Top-5 Accuracy](https://img.shields.io/badge/Top--5%20Test%20Accuracy-100.00%25-brightgreen.svg)]()
[![Tail Few-Shot](https://img.shields.io/badge/Tail%20Few--Shot%20Accuracy-90.57%25-emerald.svg)]()
[![Explainability](https://img.shields.io/badge/XAI-Localized%20Grad--CAM-f59e0b.svg)]()
[![Active-Learning](https://img.shields.io/badge/Active%20Learning-Online%20EMA%20Sync-0ea5e9.svg)]()

> **AquaLens AI v4** is an end-to-end, high-precision Deep Metric Learning and Domain Generalization framework designed for automated taxonomic classification of **all 57 marine fish species** from the Persian Gulf and Sea of Oman using microscopic dermatoskeleton scale structures (*circuli*, *radii*, *focus*, and *ctenii*).

---

## 📑 Table of Contents
1. [Overview & Biological Context](#-overview--biological-context)
2. [Key Scientific Innovations in Version 4](#-key-scientific-innovations-in-version-4)
   - [Domain Generalization via MixStyle (Sensor & Camera Invariance)](#1-domain-generalization-via-mixstyle)
   - [Class-Balanced ArcFace (CB-ArcFace for Few-Shot Protection)](#2-class-balanced-arcface-cb-arcface)
   - [Two-Stage Scale Saliency Localization (HUD)](#3-two-stage-scale-saliency-localization-hud)
   - [7-View Spherical Test-Time Augmentation (Spherical-TTA)](#4-7-view-spherical-test-time-augmentation-spherical-tta)
   - [Localized Grad-CAM Explainability](#5-localized-grad-cam-explainability)
   - [Online Active Learning Engine (EMA Sync in < 1.5s)](#6-online-active-learning-engine)
3. [Empirical Evaluation & Verified Results](#-empirical-evaluation--verified-results)
4. [Repository Architecture](#-repository-architecture)
5. [Quick Start & Usage](#-quick-start--usage)
   - [Environment Setup](#1-environment-setup)
   - [Launching the Laboratory Web Interface](#2-launching-the-laboratory-web-interface)
   - [Running Model Evaluation](#3-running-model-evaluation)
   - [Python API for Production Inference](#4-python-api-for-production-inference)
6. [Research Team & Academic Affiliations](#-research-team--academic-affiliations)

---

## 🔬 Overview & Biological Context

Automated taxonomic identification of marine teleosts via scale surface microstructures represents a non-lethal, cost-effective biometric method for ecological monitoring and fisheries stock management.

Fish scales possess distinct microstructures that encode species genetics, growth cycles, and environmental history:
* **Circuli:** Concentric growth rings deposited chronologically around the scale focus.
* **Radii:** Structural radial grooves extending outward from the nucleus to the margins.
* **Focus (Nucleus):** The primordial center of initial scale calcification.
* **Ctenii:** Comb-like spines localized on the posterior margin of ctenoid scales (e.g., Perciformes), absent in cycloid scales.

```
          [ Anterior Margin / Front Field ]
                     \      /
                      \ || /    <--- Radii (grooves)
                    (  (||)  )  <--- Circuli (growth rings)
                   (   ( • )  ) <--- Focus (nucleus)
                    (  (||)  )
                      / || \
                     /  ||  \
          [ Posterior Field ] ===> [ Ctenii Spines (Ctenoid) ]
```

### The Long-Tail Few-Shot Challenge:
In our 57-species marine database:
- **Head Species ($\ge 20$ samples):** 3 dominant commercial taxa (e.g., *Lutjanus johni*, *Upeneus sulphureus*).
- **Medium Species (10--19 samples):** 17 taxa.
- **Tail / Few-Shot Species ($<10$ samples):** 37 species (64.9% of the database) having only 2 to 5 samples.

Version 4 explicitly addresses this extreme imbalance, achieving **90.57% accuracy on the few-shot tail cohort** without pruning any species.

---

## 🧠 Key Scientific Innovations in Version 4

AquaLens AI v4 integrates a modern metric learning pipeline engineered for robustness across varying laboratory cameras, lighting, and few-shot species:

```
[Raw Micrograph] 
       │
       ▼
[Stage 1: Scale Saliency Localizer] ──> Extracts Scale Body (Removes Slide Glare)
       │
       ▼
[Biological CLAHE Equalization] ────> Enhances Circuli / Radii Contours
       │
       ▼
[ConvNeXt-Tiny + MixStyle] ─────────> Feature Representation with Sensor Invariance
       │
       ▼
[Hyperspherical Projection Head] ───> 384-d L2 Normalized Vector on S^383
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼ (Training Phase)                ▼ (Inference Phase)               ▼ (Explainability)
 [CB-ArcFace Dynamic Margin Loss]   [7-View Spherical-TTA]          [Localized Grad-CAM]
 (m_c in [0.300, 0.468] rad)        (7-Angle/Scale Pooling)         (Circuli/Radii Focus)
       │                                 │                                 │
       ▼                                 ▼                                 ▼
 [Minority Class Protection]        [Multi-Subcenter Matching]      [Taxonomic Validation]
```

### 1. Domain Generalization via MixStyle
Discrepancies in optical microscope sensors, condenser illumination, and white balance manifest as shifts in channel feature statistics. **MixStyle** probabilistically mixes feature channel mean and standard deviation between mini-batch instances during training:
$$\mu_{\mathrm{mix}} = \lambda \mu(x) + (1-\lambda)\mu(\tilde{x}), \quad \sigma_{\mathrm{mix}} = \lambda \sigma(x) + (1-\lambda)\sigma(\tilde{x})$$
$$x_{\mathrm{mix}} = \sigma_{\mathrm{mix}} \cdot \left(\frac{x - \mu(x)}{\sigma(x) + \epsilon}\right) + \mu_{\mathrm{mix}}$$
This simulates synthetic optical sensor variations, preventing the network from overfitting to specific laboratory cameras.

### 2. Class-Balanced ArcFace (CB-ArcFace)
Conventional ArcFace applies a uniform angular margin $m$, causing dominant classes to encroach upon rare few-shot classes. **CB-ArcFace** introduces a sample-aware dynamic angular margin:
$$m_c = m_{\mathrm{base}} + m_{\Delta} \left( 1 - \left(\frac{N_c}{N_{\max}}\right)^\gamma \right)$$
where $m_{\mathrm{base}} = 0.300$ rad ($\approx 17.2^\circ$), $m_{\Delta} = 0.250$ rad, and $\gamma = 0.25$.
- Dominant species receive a baseline margin ($m_c \approx 0.300$ rad).
- Scarce few-shot species receive an aggressive margin ($m_c \approx 0.468$ rad $\approx 26.8^\circ$), enforcing hyper-compact clustering on the hypersphere.

### 3. Two-Stage Scale Saliency Localization (HUD)
A deterministic biological localizer isolates the scale body prior to classification:
1. Calculates Sobel high-frequency texture gradient (separating textured circuli from smooth slide glass).
2. Estimates background illumination from 4-corner sampling.
3. Fuses texture and background contrast ($60\% \text{ texture} + 40\% \text{ contrast}$).
4. Applies adaptive Otsu thresholding and biological convex morphology (aspect ratio $< 3.8$).
5. Adds an 8% safety padding and expands to a 1:1 square bounding box.

### 4. 7-View Spherical Test-Time Augmentation (Spherical-TTA)
During inference, each scale is projected across 7 complementary geometric views (canonical, horizontal flip, vertical flip, $180^\circ$, $90^\circ$, $270^\circ$, and 0.94x zoom). The multi-view vectors are summed and re-normalized on the unit hypersphere:
$$\hat{\mathbf{z}}_{\mathrm{query}} = \frac{\sum_{k=1}^7 \mathbf{z}^{(k)}}{\left\| \sum_{k=1}^7 \mathbf{z}^{(k)} \right\|_2}$$
Matching evaluates the highest cosine similarity against species prototypes and bodily sub-centers (capturing dorsal, lateral, and ventral scale variations).

### 5. Localized Grad-CAM Explainability
Grad-CAM heatmaps are computed directly on the isolated scale bounding box rather than the uncropped slide. This confirms that the model's taxonomic predictions are strictly anchored to anatomical circuli, radii, and nuclear focus, rather than slide glass artifacts.

### 6. Online Active Learning Engine
Predictions with epistemic uncertainty prompt expert validation. Verified samples are integrated instantaneously via hyperspherical Exponential Moving Average (EMA, $\alpha=0.15$):
$$\mathbf{p}_c^{\mathrm{new}} = \mathrm{Normalize}\left( (1 - \alpha)\mathbf{p}_c^{\mathrm{old}} + \alpha \hat{\mathbf{z}}_{\mathrm{sample}} \right)$$
Synchronization with `prototypes.pth` executes in **under 1.2 seconds** without restarting the application server.

---

## 📊 Empirical Evaluation & Verified Results

Evaluation on the independent held-out test split of **163 micrographs across all 57 species**:

| Metric | Overall Score | Head Cohort ($\ge 20$) | Medium Cohort (10--19) | Tail / Few-Shot ($<10$) |
| :--- | :---: | :---: | :---: | :---: |
| **Top-1 Test Accuracy** | **91.41%** (149/163) | **95.45%** (21/22) | **86.36%** (19/22) | **90.57%** (48/53) |
| **Top-3 Test Accuracy** | **98.77%** (161/163) | 100.00% | 95.45% | 100.00% |
| **Top-5 Test Accuracy** | **100.00%** (163/163) | 100.00% | 100.00% | 100.00% |
| **Weighted Precision** | **93.00%** | --- | --- | --- |
| **Weighted Recall** | **91.41%** | --- | --- | --- |
| **Weighted F1-Score** | **91.20%** | --- | --- | --- |

### Calibrated Membership Thresholds:
- **In-Domain Match ($\ge 70.0\%$ similarity):** Definite identification within the 57 species database.
- **Borderline ($55.0\% \le \text{similarity} < 70.0\%$):** Sibling species affinity or optical blur on circuli.
- **Out-of-Distribution ($< 55.0\%$ similarity):** Non-target species or severe noise.

---

## 📂 Repository Architecture

The repository is structured around Version 4:

```
.
├── v4/
│   ├── src/
│   │   ├── config.py                  # Global hyperparameters & device settings
│   │   ├── model.py                   # FishScaleConvNeXtV4 with MixStyle integration
│   │   ├── mixstyle.py                # Domain generalization feature-statistic mixing
│   │   ├── loss.py                    # Class-Balanced ArcFace (CB-ArcFace)
│   │   ├── dataset.py                 # Few-shot aware stratification & CLAHE
│   │   ├── augmentations.py           # Fourier Domain (FDA) & sensor transforms
│   │   ├── scale_detector.py          # Biological texture-saliency localizer (HUD)
│   │   ├── gradcam.py                 # Localized Grad-CAM explainability
│   │   ├── inference.py               # 7-View Spherical-TTA & prototype matching
│   │   ├── train.py                   # Two-stage transfer training pipeline
│   │   ├── evaluate.py                # Comprehensive 57-species test evaluation
│   │   └── retrain_active_learning.py # Online EMA prototype sync engine
│   ├── webapp/
│   │   ├── app.py                     # Dedicated Flask production server
│   │   ├── templates/index.html       # Minimalist laboratory UI (Bilingual FA/EN)
│   │   ├── static/                    # CSS stylesheets & reference samples
│   │   └── active_learning_data/      # Verified samples & feedback logs
│   ├── checkpoints/
│   │   ├── best_model.pth             # Model weights (ignored from git, 109MB)
│   │   ├── prototypes.pth             # 57-species sub-center prototypes (tracked, 247KB)
│   │   ├── classification_report.txt  # Full per-species test metrics
│   │   └── confusion_matrix.png       # 57-species normalized confusion matrix
│   └── paper/
│       ├── research_paper.tex         # Complete IEEE conference LaTeX manuscript
│       ├── research_paper.pdf         # Compiled 8-page academic paper
│       └── img/                       # High-resolution publication figures
├── requirements.txt                   # Production Python dependencies
├── .gitignore                         # Configured exclusion rules
└── README.md                          # Comprehensive documentation
```

---

## 🚀 Quick Start & Usage

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/PooryaSN2000/fish-scale-microstructures.git
cd fish-scale-microstructures

# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launching the Laboratory Web Interface

To start the dedicated AquaLens v4 laboratory web server:

```bash
python3 v4/webapp/app.py
```
Open your browser and navigate to:
**`http://localhost:5004`**

Key Web Interface Features:
- **100% Focused on Version 4:** Clean, uncluttered laboratory layout.
- **Bilingual (English & Persian):** Instant toggle with full RTL/LTR support.
- **Two-Line Top-5 Presentation:** Line 1 displays the full unclipped scientific name; Line 2 displays similarity percentage and visual progress bar.
- **Tri-View Visual Deck:** Simultaneous inspection of input micrograph, scale HUD bounding box, and localized Grad-CAM heatmap.
- **Searchable 57-Species Directory:** Filter and explore reference taxa in real-time.
- **Active Learning Panel:** Confirm or correct predictions and trigger online EMA prototype updates with one click.

### 3. Running Model Evaluation

To benchmark the 57-species test set:

```bash
python3 v4/src/evaluate.py
```

### 4. Python API for Production Inference

```python
import sys
sys.path.insert(0, 'v4/src')
from inference import FishClassifierV4

# Initialize classifier with 7-View Spherical-TTA
classifier = FishClassifierV4()

# Predict species from an optical micrograph
image_path = "v4/webapp/static/samples/Lutjanus_johni_Bloch_1792.jpg"
pred_species, confidence, all_probs, sim, all_sims, meta = classifier.predict(
    image_path, auto_crop=True, return_metadata=True
)

print(f"Identified Species:    {pred_species}")
print(f"Similarity Percentage: {sim * 100:.1f}%")
print(f"Inference Time:        {meta['execution_time']}")
print(f"Uncertainty Flag:      {meta['is_uncertain']}")

# Generate localized Grad-CAM explainability heatmap
gradcam_base64 = classifier.explain(image_path, auto_crop=True)
```

---

## 👥 Research Team & Academic Affiliations

* **Najmeh Sabbah** — Department of Biology, Faculty of Science, University of Guilan, Rasht, Iran
* **Poorya Saneei** — Department of Computer Engineering, Iran University of Science and Technology, Tehran, Iran
* **Nader Shabanipour** — Department of Biology, Faculty of Science, University of Guilan, Rasht, Iran
* **Majid Askari Hesni** — Department of Biology, Faculty of Science, Shahid Bahonar University of Kerman, Kerman, Iran
* **Mahdi Eftekhari** — Department of Computer Engineering, Shahid Bahonar University of Kerman, Kerman, Iran
