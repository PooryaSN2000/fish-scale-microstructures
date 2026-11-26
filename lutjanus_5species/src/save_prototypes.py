import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from sklearn.covariance import LedoitWolf

import config
from dataset import FishScaleDataset
from model import ProtoNet
from features import extract_classical_features


@torch.no_grad()
def extract_4view_tta_features(model, pil_img: Image.Image, base_transform, device, use_tta: bool = True, use_classical: bool = False):
    """
    Computes both L2-normalized metric embeddings (256-d) and pre-projection backbone
    features (768-d) averaged over 4 geometric views on the unit hypersphere.
    """
    if use_tta and not use_classical:
        views = [
            pil_img,
            pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            pil_img.rotate(90),
        ]
    else:
        views = [pil_img]

    batch_tensors = torch.stack([base_transform(v) for v in views]).to(device)
    cf_tensor = None
    if use_classical:
        cfs = [extract_classical_features(v) for v in views]
        cf_tensor = torch.tensor(cfs, dtype=torch.float32).to(device)

    feats_768 = F.normalize(model.extract_backbone_features(batch_tensors), p=2, dim=-1)
    mean_feat_768 = F.normalize(feats_768.mean(dim=0, keepdim=True), p=2, dim=-1)

    embs = model(batch_tensors, cf_tensor, stage=2)
    embs_norm = F.normalize(embs, p=2, dim=-1)
    mean_emb = F.normalize(embs_norm.mean(dim=0, keepdim=True), p=2, dim=-1)
    return mean_emb, mean_feat_768


def main():
    print(f"Using Device: {config.DEVICE}")
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    mcfg = checkpoint["config"]

    use_clahe = mcfg.get("use_clahe", config.USE_CLAHE)
    use_roi_crop = mcfg.get("use_roi_crop", getattr(config, "USE_ROI_CROP", True))
    use_tta = mcfg.get("use_tta", config.USE_TTA)
    use_classical = mcfg.get("use_classical", config.USE_CLASSICAL)
    use_mixstyle = mcfg.get("use_mixstyle", False)
    img_size = tuple(mcfg.get("image_size", config.IMAGE_SIZE))

    train_dataset = FishScaleDataset(
        root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED,
        use_clahe=use_clahe, use_roi_crop=use_roi_crop
    )
    val_dataset = FishScaleDataset(
        root_dir=config.ROOT_DIR, split="val", seed=config.RANDOM_SEED,
        use_clahe=use_clahe, use_roi_crop=use_roi_crop
    )
    class_names = train_dataset.classes

    model = ProtoNet(
        backbone_name=mcfg["backbone"],
        use_pretrained=False,
        embedding_dim=mcfg["embedding_dim"],
        use_classical=use_classical,
        classical_in_dim=mcfg.get("classical_in_dim", 0),
        classical_proj_dim=mcfg.get("classical_proj_dim", config.CLASSICAL_PROJ_DIM),
        use_mixstyle=use_mixstyle,
    ).to(config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    base_transform = v2.Compose([
        v2.Resize(img_size),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    class_to_embs = {c: [] for c in range(len(class_names))}
    class_to_feats768 = {c: [] for c in range(len(class_names))}
    tr_f768_all, tr_labels_all = [], []

    for idx in range(len(train_dataset)):
        pil_img = train_dataset.image_cache[idx]
        lbl = train_dataset.labels[idx]
        emb, feat768 = extract_4view_tta_features(
            model, pil_img, base_transform, config.DEVICE, use_tta=use_tta, use_classical=use_classical
        )
        class_to_embs[lbl].append(emb.cpu())
        class_to_feats768[lbl].append(feat768.cpu())
        tr_f768_all.append(feat768.cpu().numpy().squeeze(0))
        tr_labels_all.append(lbl)

    prototypes, means_768 = [], []
    for c in range(len(class_names)):
        embs_c = torch.cat(class_to_embs[c], dim=0)
        proto = F.normalize(embs_c.mean(dim=0, keepdim=True), p=2, dim=-1)
        prototypes.append(proto)

        f768_c = torch.cat(class_to_feats768[c], dim=0)
        m768 = F.normalize(f768_c.mean(dim=0, keepdim=True), p=2, dim=-1)
        means_768.append(m768)
        print(f"Class '{class_names[c]}' prototype computed from {embs_c.size(0)} train samples.")

    prototypes_tensor = torch.cat(prototypes, dim=0)
    means_768_tensor = torch.cat(means_768, dim=0)
    means_768_np = means_768_tensor.numpy()

    # Fit Ledoit-Wolf shrinkage Mahalanobis precision strictly on train residuals (no test leakage!)
    tr_f768_np = np.array(tr_f768_all)
    tr_res = tr_f768_np - means_768_np[np.array(tr_labels_all)]
    prec_768_np = LedoitWolf().fit(tr_res).precision_.astype(np.float32)

    # Calibrate OOD manifold distribution strictly on unseen validation samples (zero test leakage!)
    val_mahas, val_sims, val_margins = [], [], []
    protos_dev = prototypes_tensor.to(config.DEVICE)
    for idx in range(len(val_dataset)):
        pil_img = val_dataset.image_cache[idx]
        emb, feat768 = extract_4view_tta_features(
            model, pil_img, base_transform, config.DEVICE, use_tta=use_tta, use_classical=use_classical
        )
        sims = torch.matmul(emb, protos_dev.t()).squeeze(0).cpu().numpy()
        ss = np.sort(sims)[::-1]
        val_sims.append(float(ss[0]))
        val_margins.append(float(ss[0] - ss[1]))
        f_np = feat768.cpu().numpy()
        diffs = f_np - means_768_np
        maha = float(np.sqrt(np.min(np.sum((diffs @ prec_768_np) * diffs, axis=1))))
        val_mahas.append(maha)

    val_mahas_np = np.array(val_mahas)
    val_sims_np = np.array(val_sims)
    val_margins_np = np.array(val_margins)

    m_maha, s_maha = float(val_mahas_np.mean()), float(val_mahas_np.std() + 1e-6)
    m_sim, s_sim = float(val_sims_np.mean()), float(val_sims_np.std() + 1e-6)
    m_mar, s_mar = float(val_margins_np.mean()), float(val_margins_np.std() + 1e-6)
    val_fused = 0.55 * (-(val_mahas_np - m_maha) / s_maha) + 0.35 * ((val_sims_np - m_sim) / s_sim) + 0.10 * ((val_margins_np - m_mar) / s_mar)

    ood_stats = {
        "maha_mean": m_maha,
        "maha_std": s_maha,
        "maha_p85": float(np.percentile(val_mahas_np, 85)),
        "maha_p95": float(np.percentile(val_mahas_np, 95)),
        "sim_mean": m_sim,
        "sim_std": s_sim,
        "sim_p10": float(np.percentile(val_sims_np, 10)),
        "margin_mean": m_mar,
        "margin_std": s_mar,
        "fused_mean": float(val_fused.mean()),
        "fused_std": float(val_fused.std() + 1e-6),
        "fused_p10": float(np.percentile(val_fused, 10)),
        "fused_p05": float(np.percentile(val_fused, 5)),
    }
    print(f"[OOD Calibration (Val N={len(val_dataset)})] Maha768 mean={m_maha:.2f} (p85={ood_stats['maha_p85']:.2f}, p95={ood_stats['maha_p95']:.2f}) | Fused p10={ood_stats['fused_p10']:.2f}")

    save_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")
    torch.save({
        "prototypes": prototypes_tensor.to(config.DEVICE),
        "means_768": means_768_tensor,
        "prec_768": torch.from_numpy(prec_768_np),
        "ood_stats": ood_stats,
        "class_names": class_names,
        "use_clahe": use_clahe,
        "use_roi_crop": use_roi_crop,
        "use_tta": use_tta,
        "image_size": img_size,
    }, save_path)
    print(f"Saved {len(class_names)} prototypes + val-calibrated OOD detector to {save_path}")


if __name__ == "__main__":
    main()
