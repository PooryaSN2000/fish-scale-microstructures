import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import sys
import json
import random
import argparse
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from dataset import FishScaleDatasetV4, get_unseen_novel_species, load_rgb_image, apply_clahe_preprocessing
from model import FishScaleConvNeXtV4
from loss import ClassBalancedArcFace
from augmentations import fourier_amplitude_swap, get_v4_transforms
from inference import FishClassifierV4
from scale_detector import ScaleDetector


class _InMemoryCachedDataset(Dataset):
    """Lightweight Dataset over pre-localized PIL ROI crops so K-fold loops avoid re-cropping."""
    def __init__(self, pil_images, labels, transform):
        self.pil_images = pil_images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.pil_images)

    def __getitem__(self, idx):
        return self.transform(self.pil_images[idx]), self.labels[idx]


def run_stress_ablation_and_stats(clf: FishClassifierV4):
    """
    Runs real ablations (TTA, Subcenters, Uncropped), cross-camera stress perturbations,
    background-masking attribution proofs, statistical separation tests, and OOD checks.
    """
    print("\n" + "=" * 65)
    print("RUNNING REAL ABLATIONS, DOMAIN-SHIFT STRESS TESTS & STATS")
    print("=" * 65)

    test_ds = FishScaleDatasetV4(
        config.ROOT_DIR, split="test", seed=config.RANDOM_SEED, use_roi_crop=False, use_clahe=False
    )
    detector = ScaleDetector(padding_ratio=getattr(config, "ROI_PADDING_RATIO", 0.12))
    classes = test_ds.classes
    full_counts = test_ds.full_counts

    # Precompute raw + ROI crops once so 10 scenarios execute fast
    raw_imgs = [load_rgb_image(p) for p in test_ds.image_paths]
    roi_imgs = [detector.crop(im, padding_ratio=clf.roi_padding_ratio, square=True) for im in raw_imgs]
    targets = list(test_ds.labels)

    def _apply_gamma(pil_img, g):
        arr = np.array(pil_img, dtype=np.float32) / 255.0
        arr = np.power(np.clip(arr, 0.0, 1.0), g)
        return Image.fromarray((arr * 255.0).clip(0, 255).astype(np.uint8))

    def _mask_roi_center(pil_crop):
        arr = np.array(pil_crop.convert("RGB")).copy()
        h, w, _ = arr.shape
        border = np.concatenate([
            arr[: max(4, h // 12), :, :].reshape(-1, 3),
            arr[-max(4, h // 12) :, :, :].reshape(-1, 3),
        ], axis=0)
        bg = np.median(border, axis=0).astype(np.uint8)
        arr[int(h * 0.12) : int(h * 0.88), int(w * 0.12) : int(w * 0.88)] = bg
        return Image.fromarray(arr)

    scenarios = [
        ("Clean Scale ROI (7-View TTA + Subcenters)", lambda i: roi_imgs[i], True, True),
        ("Single-View ROI (No Spherical-TTA)", lambda i: roi_imgs[i], False, True),
        ("Mean Prototype Only (No Subcenters)", lambda i: roi_imgs[i], True, False),
        ("Full Uncropped Slide (With Rulers)", lambda i: raw_imgs[i], True, True),
        ("Dark / Low-Gamma Sensor (g=1.45)", lambda i: _apply_gamma(ImageEnhance.Brightness(roi_imgs[i]).enhance(0.75), 1.45), True, True),
        ("Bright Warm CMOS Shift (g=0.72)", lambda i: _apply_gamma(ImageEnhance.Color(ImageEnhance.Brightness(roi_imgs[i]).enhance(1.20)).enhance(1.35), 0.72), True, True),
        ("Optical Defocus Blur (r=1.2)", lambda i: roi_imgs[i].filter(ImageFilter.GaussianBlur(radius=1.2)), True, True),
        ("Combined Unseen Microscope", lambda i: _apply_gamma(ImageEnhance.Color(roi_imgs[i].filter(ImageFilter.GaussianBlur(radius=1.0))).enhance(1.25), 0.80), True, True),
        ("Scale-Masked Background Only", lambda i: detector.mask_scale_out(raw_imgs[i]), True, True),
        ("ROI Interior Masked (Blank Center)", lambda i: _mask_roi_center(roi_imgs[i]), True, True),
    ]

    stress_results = {}
    intra_vals, inter_vals = [], []
    cohort_stats_map = {"head": [], "med": [], "tail": []}

    for sc_name, img_fn, use_tta_flag, use_sub_flag in scenarios:
        top1_ok, top3_ok = 0, 0
        sims_top1 = []
        cohort_ok = {"head": [0, 0], "med": [0, 0], "tail": [0, 0]}

        for idx in range(len(targets)):
            lbl = targets[idx]
            mod_img = img_fn(idx)
            emb = clf.extract_embedding(mod_img, use_tta=use_tta_flag, apply_roi_crop=False, apply_clahe=clf.use_clahe)
            if use_sub_flag:
                sims = clf.score_embedding(emb)
            else:
                sims = torch.mm(emb, clf.prototypes.t()).squeeze(0).cpu().numpy()

            ranked = np.argsort(sims)[::-1]
            pred = int(ranked[0])
            sims_top1.append(float(sims[pred]))

            cnt = full_counts[lbl]
            ck = "head" if cnt >= 20 else ("med" if cnt >= 10 else "tail")
            cohort_ok[ck][1] += 1
            if pred == lbl:
                top1_ok += 1
                cohort_ok[ck][0] += 1
            if lbl in ranked[:3]:
                top3_ok += 1

            if sc_name == "Clean Scale ROI (7-View TTA + Subcenters)":
                intra_vals.append(float(sims[lbl]))
                cohort_stats_map[ck].append(float(sims[lbl]))
                for j in range(len(classes)):
                    if j != lbl:
                        inter_vals.append(float(sims[j]))

        total = len(targets)
        acc = round(100.0 * top1_ok / total, 2)
        t3_acc = round(100.0 * top3_ok / total, 2)
        tail_acc = round(100.0 * cohort_ok["tail"][0] / max(1, cohort_ok["tail"][1]), 2)
        stress_results[sc_name] = {
            "correct": int(top1_ok),
            "total": int(total),
            "accuracy_pct": acc,
            "top3_accuracy_pct": t3_acc,
            "tail_accuracy_pct": tail_acc,
            "mean_top1_sim": round(float(np.mean(sims_top1)), 4),
        }
        print(f"  {sc_name:44s}: Top-1={acc:6.2f}% ({top1_ok}/{total}) | Tail={tail_acc:5.1f}% | Sim={np.mean(sims_top1):.3f}")

    intra_np = np.array(intra_vals)
    inter_np = np.array(inter_vals)
    t_stat, p_val = stats.ttest_ind(intra_np, inter_np, equal_var=False)
    u_stat, u_pval = stats.mannwhitneyu(intra_np, inter_np, alternative="greater")
    pooled_sd = np.sqrt(((len(intra_np) - 1) * np.var(intra_np, ddof=1) + (len(inter_np) - 1) * np.var(inter_np, ddof=1)) / (len(intra_np) + len(inter_np) - 2))
    cohen_d = float((np.mean(intra_np) - np.mean(inter_np)) / (pooled_sd + 1e-8))

    # OOD External TIF check
    ood_tifs = []
    test_dir = os.path.join(os.path.dirname(config.ROOT_DIR), "test_dataset")
    if os.path.exists(test_dir):
        for f in sorted(os.listdir(test_dir)):
            if f.lower().endswith((".tif", ".tiff", ".jpg", ".png")):
                emb = clf.extract_embedding(load_rgb_image(os.path.join(test_dir, f)), use_tta=True)
                sims = clf.score_embedding(emb)
                top_i = int(np.argmax(sims))
                ood_tifs.append({
                    "file": f,
                    "max_sim": round(float(sims[top_i]), 4),
                    "closest_species": classes[top_i],
                })

    stats_summary = {
        "intra_mean": round(float(np.mean(intra_np)), 4),
        "intra_std": round(float(np.std(intra_np, ddof=1)), 4),
        "inter_mean": round(float(np.mean(inter_np)), 4),
        "inter_std": round(float(np.std(inter_np, ddof=1)), 4),
        "margin": round(float(np.mean(intra_np) - np.mean(inter_np)), 4),
        "welch_t": round(float(t_stat), 2),
        "welch_p": float(p_val),
        "mannwhitney_p": float(u_pval),
        "cohen_d": round(cohen_d, 2),
        "cohort_intra_means": {
            k: round(float(np.mean(v)), 4) for k, v in cohort_stats_map.items() if v
        },
        "ood_tif_samples": ood_tifs,
        "ood_tif_mean_max_sim": round(float(np.mean([x["max_sim"] for x in ood_tifs])), 4) if ood_tifs else None,
    }
    print(f"\n[Stats] Intra={stats_summary['intra_mean']:.3f}+-{stats_summary['intra_std']:.3f} vs Inter={stats_summary['inter_mean']:.3f}+-{stats_summary['inter_std']:.3f} | t={stats_summary['welch_t']} | Cohen d={stats_summary['cohen_d']}")
    return stress_results, stats_summary


def run_novel_species_fewshot_benchmark(clf: FishClassifierV4):
    """
    Evaluates zero-retraining Open-Set Few-Shot Novel Species Registration on all
    completely unseen marine species (< 7 specimens) excluded from training.
    Tests 1-Shot, 2-Shot, and 3-Shot dynamic prototype registration!
    """
    print("\n" + "=" * 65)
    print("RUNNING ZERO-RETRAINING NOVEL SPECIES FEW-SHOT BENCHMARK (21 UNSEEN TAXA)")
    print("=" * 65)

    novel_dict = get_unseen_novel_species(config.ROOT_DIR, clf.class_names[:clf.base_class_count])
    novel_names = sorted(novel_dict.keys())
    print(f"Found {len(novel_names)} completely unseen species ({sum(len(v) for v in novel_dict.values())} specimens total).")

    # Pre-extract ROI + 7-View Spherical-TTA embeddings once for all unseen species specimens
    novel_embs = {}
    pre_reg_max_sims = []
    base_protos = clf.prototypes[:clf.base_class_count]

    for sp in novel_names:
        embs = []
        for p in novel_dict[sp]:
            emb = clf.extract_embedding(load_rgb_image(p), use_tta=True)
            embs.append(emb.squeeze(0))
            base_max = float(torch.mm(emb, base_protos.t()).max().item())
            pre_reg_max_sims.append(base_max)
        novel_embs[sp] = embs

    pre_ood_mean = round(float(np.mean(pre_reg_max_sims)), 4)
    print(f"  Pre-Registration Unseen Species Max Sim to 35 Base Classes: {pre_ood_mean:.4f}")

    shot_results = {}
    for k_shot in [1, 2, 3]:
        # Build novel prototypes for all species having at least k_shot + 1 specimens
        eligible = [sp for sp in novel_names if len(novel_embs[sp]) >= k_shot + 1]
        if not eligible:
            continue

        novel_proto_list = []
        for sp in eligible:
            support = torch.stack(novel_embs[sp][:k_shot], dim=0)
            novel_proto_list.append(F.normalize(support.mean(dim=0, keepdim=True), p=2, dim=1))
        novel_protos = torch.cat(novel_proto_list, dim=0)  # (U, D)

        # Open-Set Bank = 35 Base + U Newly Registered Novel Taxa (ZERO network retraining!)
        open_bank = torch.cat([base_protos, novel_protos], dim=0)
        offset = base_protos.size(0)

        novel_only_ok, open_top1_ok, open_top3_ok, total_q = 0, 0, 0, 0
        query_sims = []

        for u_idx, sp in enumerate(eligible):
            target_open_idx = offset + u_idx
            for q_emb in novel_embs[sp][k_shot:]:
                qv = q_emb.unsqueeze(0)
                sims_open = torch.mm(qv, open_bank.t()).squeeze(0).cpu().numpy()
                sims_novel = torch.mm(qv, novel_protos.t()).squeeze(0).cpu().numpy()

                r_open = np.argsort(sims_open)[::-1]
                r_novel = np.argsort(sims_novel)[::-1]

                if int(r_novel[0]) == u_idx:
                    novel_only_ok += 1
                if int(r_open[0]) == target_open_idx:
                    open_top1_ok += 1
                if target_open_idx in r_open[:3]:
                    open_top3_ok += 1
                query_sims.append(float(sims_open[target_open_idx]))
                total_q += 1

        res_k = {
            "k_shot": k_shot,
            "novel_species_tested": len(eligible),
            "query_specimens": int(total_q),
            "open_set_top1_pct": round(100.0 * open_top1_ok / max(1, total_q), 2),
            "open_set_top3_pct": round(100.0 * open_top3_ok / max(1, total_q), 2),
            "novel_only_top1_pct": round(100.0 * novel_only_ok / max(1, total_q), 2),
            "mean_target_sim": round(float(np.mean(query_sims)), 4),
        }
        shot_results[f"{k_shot}-shot"] = res_k
        print(
            f"  {k_shot}-Shot Registration ({len(eligible)} unseen taxa, {total_q} queries): "
            f"Open-Set Top-1 ({offset + len(eligible)}-way)={res_k['open_set_top1_pct']}% | "
            f"Top-3={res_k['open_set_top3_pct']}% | Novel-Only={res_k['novel_only_top1_pct']}% | Sim={res_k['mean_target_sim']:.3f}"
        )

    return {
        "total_unseen_species_in_dataset": len(novel_names),
        "total_unseen_specimens": int(len(pre_reg_max_sims)),
        "pre_registration_base_max_sim": pre_ood_mean,
        "shot_benchmarks": shot_results,
    }


def run_kfold_cv(n_splits=5, s2_epochs=24):
    """
    Runs strict K-Fold Stratified Cross-Validation across all 804 images (35 species),
    pre-caching ROI+CLAHE once so folds evaluate quickly and honestly without leakage.
    """
    print("\n" + "=" * 65)
    print(f"RUNNING {n_splits}-FOLD STRATIFIED CROSS-VALIDATION (35 SPECIES, N=804)")
    print("=" * 65)
    device = config.DEVICE

    full_ds = FishScaleDatasetV4(config.ROOT_DIR, split="all", seed=config.RANDOM_SEED)
    paths = full_ds.image_paths
    labels = np.array(full_ds.labels)
    classes = full_ds.classes
    num_classes = len(classes)

    print(f"[Cache] Pre-localizing ROI+CLAHE for all {len(paths)} specimens once...")
    cached_pils = [full_ds.load_processed_pil(p) for p in paths]

    train_tfm = get_v4_transforms(config.IMAGE_SIZE, is_train=True)
    eval_tfm = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _tta_emb(model, pil_img):
        w, h = pil_img.size
        zoomed = pil_img.crop((int(0.03 * w), int(0.03 * h), int(0.97 * w), int(0.97 * h))).resize(pil_img.size, Image.Resampling.BILINEAR)
        views = [
            pil_img,
            pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            pil_img.rotate(180),
            pil_img.rotate(90),
            pil_img.rotate(270),
            zoomed,
        ]
        xb = torch.stack([eval_tfm(v) for v in views]).to(device)
        with torch.no_grad():
            return F.normalize(model(xb).sum(dim=0, keepdim=True), p=2, dim=1)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
    fold_metrics = []
    oof_targets, oof_preds = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(paths, labels), start=1):
        torch.manual_seed(config.RANDOM_SEED + fold_idx)
        random.seed(config.RANDOM_SEED + fold_idx)
        np.random.seed(config.RANDOM_SEED + fold_idx)

        tr_pils = [cached_pils[i] for i in tr_idx]
        tr_lbls = labels[tr_idx].tolist()
        te_pils = [cached_pils[i] for i in te_idx]
        te_lbls = labels[te_idx].tolist()

        counts = np.bincount(tr_lbls, minlength=num_classes)
        w = 1.0 / np.power(np.maximum(counts, 1), 0.5)
        sampler = WeightedRandomSampler(torch.DoubleTensor(w[tr_lbls]), num_samples=len(tr_lbls), replacement=True)

        tr_loader = DataLoader(_InMemoryCachedDataset(tr_pils, tr_lbls, train_tfm), batch_size=8, sampler=sampler, num_workers=2, pin_memory=True, drop_last=True)

        model = FishScaleConvNeXtV4(
            embedding_dim=config.EMBEDDING_DIM,
            use_mixstyle=config.USE_MIXSTYLE,
            mixstyle_p=config.MIXSTYLE_P,
            mixstyle_alpha=config.MIXSTYLE_ALPHA,
        ).to(device)
        criterion = ClassBalancedArcFace(
            in_features=config.EMBEDDING_DIM,
            num_classes=num_classes,
            s=config.ARCFACE_SCALE,
            m_base=config.ARCFACE_MARGIN_BASE,
            delta_m=config.ARCFACE_MARGIN_DELTA,
            gamma=config.ARCFACE_GAMMA,
            class_counts=counts,
        ).to(device)
        ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Stage 1: 3 epochs projector warmup
        for p in model.backbone.parameters():
            p.requires_grad = False
        opt1 = optim.AdamW([
            {"params": model.projector.parameters(), "lr": config.LR_PROJECTOR},
            {"params": criterion.parameters(), "lr": config.LR_ARCFACE},
        ], weight_decay=config.WEIGHT_DECAY)
        for _ in range(3):
            model.train()
            criterion.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt1.zero_grad()
                loss = ce_loss_fn(criterion(model(xb), yb), yb)
                loss.backward()
                opt1.step()

        # Stage 2: Full fine-tuning
        for p in model.backbone.parameters():
            p.requires_grad = True
        opt2 = optim.AdamW([
            {"params": model.backbone.parameters(), "lr": config.LR_BACKBONE},
            {"params": model.projector.parameters(), "lr": config.LR_PROJECTOR},
            {"params": criterion.parameters(), "lr": config.LR_ARCFACE},
        ], weight_decay=config.WEIGHT_DECAY)
        sched2 = CosineAnnealingLR(opt2, T_max=s2_epochs, eta_min=1e-6)

        for _ in range(s2_epochs):
            model.train()
            criterion.train()
            opt2.zero_grad()
            for b_i, (xb, yb) in enumerate(tr_loader):
                xb, yb = xb.to(device), yb.to(device)
                B = xb.size(0)
                if config.USE_FDA and random.random() < 0.25 and B > 1:
                    perm = torch.randperm(B, device=device)
                    half = B // 2
                    xb[:half] = torch.stack([fourier_amplitude_swap(xb[i], xb[perm[i]], beta=config.FDA_BETA) for i in range(half)])
                loss = ce_loss_fn(criterion(model(xb), yb), yb) / 2.0
                loss.backward()
                if (b_i + 1) % 2 == 0 or (b_i + 1) == len(tr_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt2.step()
                    opt2.zero_grad()
            sched2.step()

        # Strict Fold Train Prototypes + Test Evaluation
        model.eval()
        tr_buckets = {c: [] for c in range(num_classes)}
        for pil_im, lbl in zip(tr_pils, tr_lbls):
            tr_buckets[lbl].append(_tta_emb(model, pil_im))
        fold_protos = torch.cat([F.normalize(torch.cat(tr_buckets[c], dim=0).mean(dim=0, keepdim=True), p=2, dim=1) for c in range(num_classes)], dim=0)

        preds, top3_hits = [], 0
        for pil_im, lbl in zip(te_pils, te_lbls):
            emb = _tta_emb(model, pil_im)
            sims = torch.mm(emb, fold_protos.t()).squeeze(0).cpu().numpy()
            ranked = np.argsort(sims)[::-1]
            preds.append(int(ranked[0]))
            if lbl in ranked[:3]:
                top3_hits += 1

        acc = 100.0 * accuracy_score(te_lbls, preds)
        t3 = 100.0 * top3_hits / len(te_lbls)
        mf1 = f1_score(te_lbls, preds, average="macro", zero_division=0)
        wf1 = f1_score(te_lbls, preds, average="weighted", zero_division=0)

        oof_targets.extend(te_lbls)
        oof_preds.extend(preds)
        fold_metrics.append({
            "fold": fold_idx,
            "n_test": len(te_lbls),
            "top1_pct": round(acc, 2),
            "top3_pct": round(t3, 2),
            "macro_f1": round(mf1, 4),
            "weighted_f1": round(wf1, 4),
        })
        print(f"  Fold {fold_idx}/{n_splits} (n={len(te_lbls)}): Top-1={acc:.2f}% | Top-3={t3:.2f}% | Macro-F1={mf1:.4f} | Weighted-F1={wf1:.4f}")
        del model, criterion, opt1, opt2, sched2, tr_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    accs = [m["top1_pct"] for m in fold_metrics]
    t3s = [m["top3_pct"] for m in fold_metrics]
    mf1s = [m["macro_f1"] for m in fold_metrics]
    summary = {
        "n_splits": n_splits,
        "folds": fold_metrics,
        "mean_top1_pct": round(float(np.mean(accs)), 2),
        "std_top1_pct": round(float(np.std(accs)), 2),
        "mean_top3_pct": round(float(np.mean(t3s)), 2),
        "mean_macro_f1": round(float(np.mean(mf1s)), 4),
        "std_macro_f1": round(float(np.std(mf1s)), 4),
        "overall_oof_top1_pct": round(100.0 * accuracy_score(oof_targets, oof_preds), 2),
    }
    print(f"5-Fold Summary: Top-1={summary['mean_top1_pct']}% +- {summary['std_top1_pct']}% | Top-3={summary['mean_top3_pct']}% | Macro-F1={summary['mean_macro_f1']}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-cv", action="store_true", help="Skip 5-fold CV and run stress/stats/few-shot novel benchmarks only.")
    parser.add_argument("--cv-only", action="store_true", help="Reuse saved stress/stats/few-shot benchmarks and run 5-fold CV only.")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    out_path = os.path.join(config.CHECKPOINT_DIR, "generalization_stats.json")
    if args.cv_only and os.path.exists(out_path):
        with open(out_path, "r") as f:
            prev = json.load(f)
        stress_results = prev["robustness_stress_tests"]
        stats_summary = prev["congeneric_and_cohort_statistics"]
        novel_fewshot = prev["novel_species_fewshot_benchmark"]
        print("[Reusing saved stress, stats & novel-species few-shot benchmarks]")
    else:
        clf = FishClassifierV4()
        stress_results, stats_summary = run_stress_ablation_and_stats(clf)
        novel_fewshot = run_novel_species_fewshot_benchmark(clf)
        del clf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cv_summary = None
    if not args.skip_cv:
        cv_summary = run_kfold_cv(n_splits=args.folds)
    elif os.path.exists(out_path):
        with open(out_path, "r") as f:
            cv_summary = json.load(f).get("cross_validation_kfold")

    out = {
        "robustness_stress_tests": stress_results,
        "congeneric_and_cohort_statistics": stats_summary,
        "novel_species_fewshot_benchmark": novel_fewshot,
        "cross_validation_kfold": cv_summary,
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] Complete v4.1 generalization benchmark -> {out_path}")


if __name__ == "__main__":
    main()
