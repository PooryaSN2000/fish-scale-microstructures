import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.covariance import LedoitWolf

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import config
from model import FishScaleConvNeXtV4
from dataset import FishScaleDatasetV4


def _tta_views(pil_img: Image.Image, tfm) -> list[torch.Tensor]:
    w, h = pil_img.size
    zoomed = pil_img.crop((int(0.03 * w), int(0.03 * h), int(0.97 * w), int(0.97 * h))).resize(pil_img.size, Image.Resampling.BILINEAR)
    imgs = [
        pil_img,
        pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
        pil_img.rotate(180),
        pil_img.rotate(90),
        pil_img.rotate(270),
        zoomed,
    ]
    return [tfm(im) for im in imgs]


def compute_and_save_prototypes(model, class_names, output_path=None, split="train"):
    """
    Computes strictly leakage-free class prototypes (and K=2 head subcenters) from `split='train'`,
    fits a Ledoit-Wolf shrinkage Mahalanobis precision matrix on train metric residuals,
    and calibrates OOD manifold cutoffs strictly on unseen validation specimens (`split='val'`).
    """
    device = config.DEVICE
    model.eval()

    if output_path is None:
        output_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")

    ds = FishScaleDatasetV4(config.ROOT_DIR, split=split, seed=config.RANDOM_SEED)
    val_ds = FishScaleDatasetV4(config.ROOT_DIR, split="val", seed=config.RANDOM_SEED)
    class_to_embeddings = {i: [] for i in range(len(class_names))}
    all_tr_embs, all_tr_labels = [], []

    eval_tfm = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def _emb_for_pil(pil_img):
        if getattr(config, "USE_SPHERICAL_TTA", True):
            batch = torch.stack(_tta_views(pil_img, eval_tfm)).to(device)
            return F.normalize(model(batch).sum(dim=0, keepdim=True), p=2, dim=1)
        batch = eval_tfm(pil_img).unsqueeze(0).to(device)
        return F.normalize(model(batch), p=2, dim=1)

    print(f"[Prototypes] Extracting 7-View Spherical-TTA embeddings from split='{split}' ({len(ds)} specimens)...")
    with torch.no_grad():
        for i in range(len(ds)):
            pil_img = ds.load_processed_pil(ds.image_paths[i])
            emb = _emb_for_pil(pil_img)
            lbl = ds.labels[i]
            class_to_embeddings[lbl].append(emb.cpu())
            all_tr_embs.append(emb.cpu().numpy().squeeze(0))
            all_tr_labels.append(lbl)

    prototype_vectors = []
    subcenter_dict = {}

    for c in range(len(class_names)):
        embs = class_to_embeddings[c]
        if len(embs) == 0:
            print(f"[Warning] Class {c} ({class_names[c]}) has 0 embeddings! Initializing randomly.")
            proto = F.normalize(torch.randn(1, config.EMBEDDING_DIM), p=2, dim=1)
            subcenter_dict[c] = proto
        else:
            embs_tensor = torch.cat(embs, dim=0)
            proto = F.normalize(embs_tensor.mean(dim=0, keepdim=True), p=2, dim=1)

            if embs_tensor.size(0) >= 15:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(embs_tensor.numpy())
                sub_centers = F.normalize(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32), p=2, dim=1)
                subcenter_dict[c] = sub_centers
            else:
                subcenter_dict[c] = proto

        prototype_vectors.append(proto)

    prototypes = torch.cat(prototype_vectors, dim=0)
    protos_np = prototypes.numpy()

    # Fit shrinkage Mahalanobis precision strictly on train residuals
    tr_np = np.array(all_tr_embs, dtype=np.float32)
    tr_res = tr_np - protos_np[np.array(all_tr_labels)]
    prec_384 = LedoitWolf().fit(tr_res).precision_.astype(np.float32)

    # Calibrate OOD manifold distribution strictly on unseen validation set (zero test leakage!)
    val_mahas, val_sims, val_margins = [], [], []
    protos_dev = prototypes.to(device)
    with torch.no_grad():
        for i in range(len(val_ds)):
            pil_img = val_ds.load_processed_pil(val_ds.image_paths[i])
            emb = _emb_for_pil(pil_img)
            sim_proto = torch.mm(emb, protos_dev.t()).squeeze(0).cpu().numpy()
            max_sims = []
            for c_idx in range(len(class_names)):
                pv = float(sim_proto[c_idx])
                subs = subcenter_dict.get(c_idx)
                if subs is not None:
                    sv = float(torch.mm(emb, subs.to(device).t()).max().item())
                    max_sims.append(max(pv, sv))
                else:
                    max_sims.append(pv)
            ss = np.sort(np.array(max_sims))[::-1]
            val_sims.append(float(ss[0]))
            val_margins.append(float(ss[0] - ss[1]))

            e_np = emb.cpu().numpy()
            diffs = e_np - protos_np
            maha = float(np.sqrt(np.min(np.sum((diffs @ prec_384) * diffs, axis=1))))
            val_mahas.append(maha)

    val_m = np.array(val_mahas)
    val_s = np.array(val_sims)
    val_r = np.array(val_margins)

    m_maha, s_maha = float(val_m.mean()), float(val_m.std() + 1e-6)
    m_sim, s_sim = float(val_s.mean()), float(val_s.std() + 1e-6)
    m_mar, s_mar = float(val_r.mean()), float(val_r.std() + 1e-6)
    val_fused = 0.60 * (-(val_m - m_maha) / s_maha) + 0.30 * ((val_s - m_sim) / s_sim) + 0.10 * ((val_r - m_mar) / s_mar)

    ood_stats = {
        "maha_mean": m_maha,
        "maha_std": s_maha,
        "maha_p85": float(np.percentile(val_m, 85)),
        "maha_p92": float(np.percentile(val_m, 92)),
        "maha_p95": float(np.percentile(val_m, 95)),
        "sim_mean": m_sim,
        "sim_std": s_sim,
        "sim_p10": float(np.percentile(val_s, 10)),
        "margin_mean": m_mar,
        "margin_std": s_mar,
        "fused_mean": float(val_fused.mean()),
        "fused_std": float(val_fused.std() + 1e-6),
        "fused_p15": float(np.percentile(val_fused, 15)),
        "fused_p10": float(np.percentile(val_fused, 10)),
        "fused_p05": float(np.percentile(val_fused, 5)),
    }
    print(f"[OOD Calibration (Val N={len(val_ds)})] Maha384 mean={m_maha:.2f} (p85={ood_stats['maha_p85']:.2f}, p95={ood_stats['maha_p95']:.2f}) | Fused p15={ood_stats['fused_p15']:.2f}")

    checkpoint = {
        "prototypes": prototypes,
        "prec_384": torch.from_numpy(prec_384),
        "ood_stats": ood_stats,
        "subcenters": subcenter_dict,
        "class_names": class_names,
        "full_counts": ds.full_counts,
        "train_counts": ds.train_counts,
        "embedding_dim": config.EMBEDDING_DIM,
        "image_size": config.IMAGE_SIZE,
        "use_roi_crop": getattr(config, "USE_ROI_CROP", True),
        "roi_padding_ratio": getattr(config, "ROI_PADDING_RATIO", 0.12),
        "use_clahe": config.USE_CLAHE,
        "use_tta": getattr(config, "USE_SPHERICAL_TTA", True),
        "use_mixstyle": getattr(config, "USE_MIXSTYLE", True),
        "prototype_split": split,
        "base_class_count": len(class_names),
        "custom_species": [],
    }

    torch.save(checkpoint, output_path)
    print(f"[Prototypes] Saved {len(class_names)} strictly leakage-free ('{split}') prototypes + val-calibrated OOD precision -> {output_path}")
    return checkpoint


if __name__ == "__main__":
    ds = FishScaleDatasetV4(config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    model = FishScaleConvNeXtV4(embedding_dim=config.EMBEDDING_DIM, use_mixstyle=False).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=config.DEVICE)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    compute_and_save_prototypes(model, ds.classes, split="train")
