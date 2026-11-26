# AquaLens AI v4.1: 35-Species Marine Fish Scale Classification

Unified, domain-invariant metric-learning pipeline for taxonomic identification across **35 validated marine fish species** ($\ge 7$ micrographs per taxon, $N=804$) with **zero-retraining open-set few-shot registration** across newly encountered species.

## Key Upgrades (v4.1)
- **Unified `ScaleDetector` Localization:** Accelerated percentile-clipped saliency ROI detector (`padding_ratio=0.12`, `min_area=0.007`) synchronized identically across training, prototype extraction, Grad-CAM, and WebApp prediction.
- **Zero-Leakage Prototype Construction:** Reference centroids and head subcenters are extracted strictly from `split="train"` ($N=545$) via **7-View Spherical-TTA** — never touching validation or held-out test specimens.
- **Domain Generalization & Tail Protection:** `ConvNeXt-Tiny` interleaved with `MixStyle` ($p=0.25, \alpha=0.3$), batch Fourier Domain Adaptation (`FDA`, $\beta=0.06$), LayerNorm-stabilized 384-d hyperspherical projection, and `ClassBalancedArcFace`.
- **Dual-Manifold OOD Gate & Top-3 Shortlist WebApp:** Fits a validation-calibrated 384-d Ledoit-Wolf Mahalanobis precision matrix alongside cosine similarity (`0.8942` OOD AUROC) to flag out-of-distribution inputs, while displaying interactive Top-3 Shortlist Consensus cards (cumulative Top-3 probability, genus consensus, and 1-click validation).
- **Zero-Retraining Novel Species Expansion & `.tif` Support:** Instantly register 1-to-$K$ images of an unseen species (JPEG, PNG, BMP, or 8/16-bit TIF/TIFF) directly into `prototypes.pth` via the Web UI without retraining weights.

## Verified Zero-Leakage Benchmarks
- **Held-Out Test ($N=142$):** **92.25% Top-1** (131/142), **95.07% Top-3**, **96.48% Top-5**, **0.9118 Weighted F1**, **0.8717 Macro F1** (**97.78%** Head, **86.11%** Medium, **75.00%** Tail).
- **5-Fold Stratified Cross-Validation ($N=804$):** **91.42% $\pm$ 2.43% Top-1**, **95.89% Top-3**, **0.8740 Macro F1**.
- **Congeneric Separation & OOD Detection (142 ID vs. 98 OOD):** Intra-class cosine $0.9106 \pm 0.1059$ vs. Inter-class $0.4507 \pm 0.0915$ (Welch's $t=51.18$, $p=1.08\times 10^{-95}$, Cohen's $d=5.00$); 384-d Mahalanobis OOD detector achieves **0.8942 AUROC** (`0.8915` fused vs. `0.8600` raw cosine), reducing `FPR@90TPR` from `42.86%` to **29.59%** and flagging **82.65%** of unseen species/slides (`100%` of external `.tif` slides rejected as `out_of_group`).
- **Cross-Camera Stress & Attribution Proof:** Preserves **88.03%–92.96%** Top-1 accuracy under unseen sensor gamma, warm CMOS drift, optical blur, and combined perturbations; masking the scale body (`Scale-Masked Background Only`) drops accuracy to **5.63%**.
- **Open-Set Few-Shot Novel Species (21 Unseen Taxa):** Without retraining weights, 3-shot registration achieves **78.79% Top-3** combined retrieval across 52 classes.

## Quick Start

### Launch Web App
```bash
python3 multiclass_35species/webapp/app.py
```
Open `http://127.0.0.1:5004` in your browser.

### Train, Evaluate & Benchmark
```bash
python3 multiclass_35species/src/train.py
python3 multiclass_35species/src/evaluate.py
python3 multiclass_35species/src/benchmark_generalization.py
python3 multiclass_35species/src/generate_all_visuals.py
```
