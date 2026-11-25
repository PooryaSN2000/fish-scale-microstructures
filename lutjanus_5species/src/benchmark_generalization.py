import os
import sys
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.transforms import v2
from collections import Counter

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from dataset import FishScaleDataset, EpisodicBatchSampler, apply_clahe_preprocessing
from model import ProtoNet, compute_prototypes_and_logits, compute_protonet_loss
from inference import FishClassifier
from scale_detector import ScaleDetector
from train import apply_batch_fda, get_gpu_transforms, get_gpu_val_transforms


STAGES_INFO = [
    ("Stage 1", "S1", "Lutjanus johni", "Lutjanus johni Bloch 1792"),
    ("Stage 2", "S2", "Lutjanus lutjanus", "Lutjanus lutjanus Bloch 1790"),
    ("Stage 3", "S3", "Lutjanus lemniscatus", "Lutjanus lemniscatus (Valenciennes, 1828)"),
    ("Stage 4", "S4", "Lutjanus malabaricus", "Lutjanus malabaricus (Bloch & Schneider, 1801)"),
    ("Stage 5", "S5", "Lutjanus fulviflammus", "Lutjanus fulviflammus(Forsskal 1775)"),
]


def _stage_index_for_name(cname: str) -> int:
    c_low = cname.lower()
    if "johni" in c_low:
        return 0
    if "lutjanus lutjanus" in c_low:
        return 1
    if "lemniscatus" in c_low:
        return 2
    if "malabar" in c_low:
        return 3
    if "fulviflammus" in c_low:
        return 4
    raise ValueError(f"Unknown species name: {cname}")


def run_5fold_cv():
    """
    Runs strict 5-Fold Stratified Cross-Validation across all 233 specimens
    so every single specimen (including all 5 samples of S5 and 14 of S4) is tested out-of-fold.
    """
    print("\n" + "=" * 60)
    print("RUNNING STRICT 5-FOLD STRATIFIED CROSS-VALIDATION (N=233)")
    print("=" * 60)
    device = config.DEVICE
    full_ds = FishScaleDataset(config.ROOT_DIR, split="all", seed=config.RANDOM_SEED, use_clahe=True, use_roi_crop=True)
    labels_np = np.array([_stage_index_for_name(full_ds.classes[y]) for y in full_ds.labels])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_SEED)
    gpu_train_tf = get_gpu_transforms().to(device)
    gpu_val_tf = get_gpu_val_transforms().to(device)

    fold_metrics = []
    oof_targets, oof_preds = [], []

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels_np)), labels_np), start=1):
        random.seed(42 + fold)
        np.random.seed(42 + fold)
        torch.manual_seed(42 + fold)
        torch.cuda.manual_seed_all(42 + fold)

        # Build fold tensors (mapped directly to 0..4 stage indices)
        tr_imgs = torch.stack([full_ds[i][0] for i in train_idx])
        tr_lbls = torch.tensor(labels_np[train_idx], dtype=torch.long)
        te_imgs = torch.stack([full_ds[i][0] for i in test_idx])
        te_lbls = torch.tensor(labels_np[test_idx], dtype=torch.long)

        tr_tensor_ds = torch.utils.data.TensorDataset(tr_imgs, tr_lbls)
        te_tensor_ds = torch.utils.data.TensorDataset(te_imgs, te_lbls)

        model = ProtoNet(
            backbone_name=config.BACKBONE,
            use_pretrained=True,
            embedding_dim=config.EMBEDDING_DIM,
            use_classical=False,
            use_mixstyle=True,
            mixstyle_p=config.MIXSTYLE_P,
            mixstyle_alpha=config.MIXSTYLE_ALPHA,
        ).to(device)

        # Stage 1 balanced pretraining (14 epochs)
        counts = Counter(tr_lbls.tolist())
        sampler = WeightedRandomSampler([1.0 / counts[y] for y in tr_lbls.tolist()], len(tr_lbls), replacement=True)
        opt1 = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": 6e-5},
            {"params": model.cnn_projection.parameters(), "lr": 5e-4},
            {"params": model.classifier_head.parameters(), "lr": 5e-4},
        ], weight_decay=1e-4)

        for _ in range(14):
            model.train()
            for x, y in DataLoader(tr_tensor_ds, batch_size=16, sampler=sampler):
                x, y = gpu_train_tf(apply_batch_fda(x.to(device))), y.to(device)
                opt1.zero_grad()
                F.cross_entropy(model(x, stage=1), y).backward()
                opt1.step()

        # Stage 2 episodic fine-tuning (8 epochs x 12 episodes, 5-way 3-shot 3-query, upper ConvNeXt unfrozen)
        for name, param in model.backbone.named_parameters():
            param.requires_grad = any(name.startswith(f"{blk}.") for blk in [4, 5, 6, 7])
        ep_sampler = EpisodicBatchSampler(tr_lbls.tolist(), 5, 3, 3, 12)
        opt2 = torch.optim.AdamW([
            {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 4e-5},
            {"params": model.cnn_projection.parameters(), "lr": 3e-4},
            {"params": [model.temperature], "lr": 1e-2},
        ], weight_decay=1e-4)

        for _ in range(8):
            model.train()
            for x, _ in DataLoader(tr_tensor_ds, batch_sampler=ep_sampler):
                x = gpu_train_tf(apply_batch_fda(x.to(device)))
                opt2.zero_grad()
                embs = model(x, stage=2)
                logits, targets = compute_prototypes_and_logits(embs, 5, 3, 3, model.temperature)
                loss, _ = compute_protonet_loss(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()

        # Out-of-fold 4-View Spherical-TTA evaluation
        model.eval()

        def _tta_emb(xb):
            views = [xb, v2.functional.horizontal_flip(xb), v2.functional.vertical_flip(xb), torch.rot90(xb, 1, [-2, -1])]
            with torch.no_grad():
                es = [F.normalize(model(gpu_val_tf(v), stage=2), p=2, dim=-1) for v in views]
            return F.normalize(torch.stack(es).mean(0), p=2, dim=-1)

        tr_e = torch.cat([_tta_emb(x.to(device)) for x, _ in DataLoader(tr_tensor_ds, batch_size=32)])
        protos = torch.stack([F.normalize(tr_e[tr_lbls.to(device) == c].mean(0), p=2, dim=-1) for c in range(5)])
        te_e = torch.cat([_tta_emb(x.to(device)) for x, _ in DataLoader(te_tensor_ds, batch_size=32)])
        preds = torch.matmul(te_e, protos.t()).argmax(-1).cpu().numpy()
        tgts = te_lbls.numpy()

        acc = float(accuracy_score(tgts, preds) * 100.0)
        mf1 = float(f1_score(tgts, preds, average="macro", zero_division=0))
        wf1 = float(f1_score(tgts, preds, average="weighted", zero_division=0))
        fold_metrics.append({"fold": fold, "n_test": int(len(tgts)), "accuracy_pct": round(acc, 2), "macro_f1": round(mf1, 4), "weighted_f1": round(wf1, 4)})
        oof_targets.extend(tgts.tolist())
        oof_preds.extend(preds.tolist())
        print(f"  Fold {fold}/5 (N={len(tgts)}): Acc={acc:.2f}% | Macro F1={mf1:.4f} | Weighted F1={wf1:.4f}")

    accs = [f["accuracy_pct"] for f in fold_metrics]
    mf1s = [f["macro_f1"] for f in fold_metrics]
    wf1s = [f["weighted_f1"] for f in fold_metrics]
    oof_tgts_np = np.array(oof_targets)
    oof_preds_np = np.array(oof_preds)
    per_stage_oof_acc = [
        round(float(np.mean(oof_preds_np[oof_tgts_np == s] == s) * 100.0), 2)
        for s in range(5)
    ]
    cv_summary = {
        "folds": fold_metrics,
        "mean_accuracy_pct": round(float(np.mean(accs)), 2),
        "std_accuracy_pct": round(float(np.std(accs)), 2),
        "mean_macro_f1": round(float(np.mean(mf1s)), 4),
        "std_macro_f1": round(float(np.std(mf1s)), 4),
        "mean_weighted_f1": round(float(np.mean(wf1s)), 4),
        "per_stage_oof_recall_pct": per_stage_oof_acc,
        "overall_oof_accuracy_pct": round(float(np.mean(oof_preds_np == oof_tgts_np) * 100.0), 2),
    }
    print(f"5-Fold Summary: Acc = {cv_summary['mean_accuracy_pct']:.2f}% +- {cv_summary['std_accuracy_pct']:.2f}% | Stage Recalls = {per_stage_oof_acc}")
    return cv_summary


def run_stress_and_background_tests():
    """
    Tests held-out generalization under unseen microscope shifts & background ablation.
    """
    print("\n" + "=" * 60)
    print("RUNNING DOMAIN-SHIFT STRESS TESTS & BACKGROUND ABLATION")
    print("=" * 60)
    clf = FishClassifier()
    detector = ScaleDetector(padding_ratio=0.12)
    test_ds = FishScaleDataset(config.ROOT_DIR, split="test", seed=config.RANDOM_SEED, use_clahe=False, use_roi_crop=False)

    proto_map = {_stage_index_for_name(cname): p_idx for p_idx, cname in enumerate(clf.class_names)}
    ordered_protos = torch.stack([clf.prototypes[proto_map[s]] for s in range(5)])
    true_stages = [_stage_index_for_name(test_ds.classes[y]) for y in test_ds.labels]

    def _apply_gamma(pil_img, g):
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), g)
        return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    def _mask_roi_center(pil_crop):
        """Masks the central biological scale inside an already-cropped ROI (no margins/rulers)."""
        arr = np.array(pil_crop.convert("RGB")).copy()
        h, w, _ = arr.shape
        border = np.concatenate([
            arr[: max(4, h // 12), :, :].reshape(-1, 3),
            arr[-max(4, h // 12) :, :, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0).astype(np.uint8)
        y1, y2 = int(h * 0.12), int(h * 0.88)
        x1, x2 = int(w * 0.12), int(w * 0.88)
        arr[y1:y2, x1:x2] = bg
        return Image.fromarray(arr)

    scenarios = {
        "Clean Scale ROI (Standard)": lambda im: (detector.crop(im, square=True), False),
        "Full Uncropped Slide": lambda im: (im, False),
        "Dark / Low-Gamma Sensor (g=1.45)": lambda im: (_apply_gamma(ImageEnhance.Brightness(detector.crop(im, square=True)).enhance(0.75), 1.45), False),
        "Bright Warm CMOS Shift (g=0.72)": lambda im: (_apply_gamma(ImageEnhance.Color(ImageEnhance.Brightness(detector.crop(im, square=True)).enhance(1.20)).enhance(1.35), 0.72), False),
        "Optical Defocus Blur (r=1.2)": lambda im: (detector.crop(im, square=True).filter(ImageFilter.GaussianBlur(radius=1.2)), False),
        "Combined Unseen Microscope": lambda im: (_apply_gamma(ImageEnhance.Color(detector.crop(im, square=True).filter(ImageFilter.GaussianBlur(radius=1.0))).enhance(1.25), 0.80), False),
        "Scale-Masked Background Only": lambda im: (detector.mask_scale_out(im), False),
        "ROI Interior Masked (Blank Center)": lambda im: (_mask_roi_center(detector.crop(im, square=True)), False),
    }

    stress_results = {}
    clean_sims_intra, clean_sims_inter = [], []

    for sc_name, fn in scenarios.items():
        correct = 0
        sims_list = []
        for idx, img_path in enumerate(test_ds.image_paths):
            raw = Image.open(img_path).convert("RGB")
            mod_img, _ = fn(raw)
            emb = clf.extract_embedding(mod_img, apply_roi_crop=False)
            sims = torch.matmul(emb, ordered_protos.t()).squeeze(0).cpu().numpy()
            pred = int(np.argmax(sims))
            tgt = true_stages[idx]
            if pred == tgt:
                correct += 1
            sims_list.append(float(sims[pred]))
            if sc_name == "Clean Scale ROI (Standard)":
                clean_sims_intra.append(float(sims[tgt]))
                for j in range(5):
                    if j != tgt:
                        clean_sims_inter.append(float(sims[j]))

        acc = round(100.0 * correct / len(true_stages), 2)
        stress_results[sc_name] = {
            "correct": int(correct),
            "total": int(len(true_stages)),
            "accuracy_pct": acc,
            "mean_top1_sim": round(float(np.mean(sims_list)), 4),
        }
        print(f"  {sc_name:36s}: {acc:6.2f}% ({correct}/{len(true_stages)}) | Mean Sim: {np.mean(sims_list):.3f}")

    # Intra- vs Inter-Stage Statistical Separation (Welch t-test, Mann-Whitney U, Cohen's d)
    intra_np = np.array(clean_sims_intra)
    inter_np = np.array(clean_sims_inter)
    t_stat, p_val = stats.ttest_ind(intra_np, inter_np, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(intra_np, inter_np, alternative="greater")
    pooled_sd = np.sqrt(((len(intra_np) - 1) * np.var(intra_np, ddof=1) + (len(inter_np) - 1) * np.var(inter_np, ddof=1)) / (len(intra_np) + len(inter_np) - 2))
    cohen_d = float((np.mean(intra_np) - np.mean(inter_np)) / (pooled_sd + 1e-8))

    # Pairwise 5x5 Stage Centroid Similarity Matrix
    pairwise_mat = torch.matmul(ordered_protos, ordered_protos.t()).cpu().numpy().tolist()

    # Out-of-Group Non-Lutjanus Check (test_dataset + 4 other marine families)
    ood_paths = []
    test_dir = os.path.join(os.path.dirname(config.ROOT_DIR), "test_dataset")
    if os.path.exists(test_dir):
        for f in sorted(os.listdir(test_dir)):
            if f.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
                ood_paths.append(("External TIF", os.path.join(test_dir, f)))
    for fam in ["Upeneus sulphureusCuvier, 1829", "Platycephalus indicus (linnaeus, 1758)", "Gerres filamentosus Cuvier,1829", "Scolopsis vosmeri (Bloch, 1792)"]:
        fd = os.path.join(config.ROOT_DIR, fam)
        if os.path.exists(fd):
            fs = sorted([f for f in os.listdir(fd) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
            for f in fs[:3]:
                ood_paths.append((fam.split()[0], os.path.join(fd, f)))

    ood_max_sims = []
    for label, p in ood_paths:
        emb = clf.extract_embedding(Image.open(p).convert("RGB"))
        ms = float(torch.matmul(emb, ordered_protos.t()).max().item())
        ood_max_sims.append({"source": label, "file": os.path.basename(p), "max_sim": round(ms, 4)})

    stats_summary = {
        "intra_mean": round(float(np.mean(intra_np)), 4),
        "intra_std": round(float(np.std(intra_np, ddof=1)), 4),
        "inter_mean": round(float(np.mean(inter_np)), 4),
        "inter_std": round(float(np.std(inter_np, ddof=1)), 4),
        "welch_t": round(float(t_stat), 2),
        "welch_p": float(p_val),
        "mannwhitney_p": float(u_pval),
        "cohen_d": round(cohen_d, 2),
        "pairwise_centroid_similarity": [[round(v, 3) for v in row] for row in pairwise_mat],
        "ood_mean_max_sim": round(float(np.mean([x["max_sim"] for x in ood_max_sims])), 4) if ood_max_sims else None,
        "ood_samples": ood_max_sims,
    }
    print(f"Statistical Separation: Intra={stats_summary['intra_mean']:.3f}+-{stats_summary['intra_std']:.3f} vs Inter={stats_summary['inter_mean']:.3f}+-{stats_summary['inter_std']:.3f} | t={stats_summary['welch_t']} | Cohen d={stats_summary['cohen_d']}")
    return stress_results, stats_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-only", action="store_true", help="Reuse saved 5-fold CV results and recompute stress/ablation/statistical tests.")
    args = parser.parse_args()

    out_path = os.path.join(config.CHECKPOINT_DIR, "generalization_stats.json")
    if args.stress_only and os.path.exists(out_path):
        with open(out_path, "r") as f:
            prev = json.load(f)
        cv_summary = prev["cross_validation_5fold"]
        print(f"[Reusing saved 5-Fold CV] Acc={cv_summary['mean_accuracy_pct']}% +- {cv_summary['std_accuracy_pct']}%")
    else:
        cv_summary = run_5fold_cv()

    stress_results, stats_summary = run_stress_and_background_tests()
    out = {
        "cross_validation_5fold": cv_summary,
        "robustness_stress_tests": stress_results,
        "congeneric_statistics": stats_summary,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] Full generalization & statistical benchmark -> {out_path}")


if __name__ == "__main__":
    main()
