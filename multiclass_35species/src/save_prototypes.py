import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

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
    Computes strictly leakage-free class prototypes (and K=2 subcenters) from `split`
    (default='train') using the exact 7-View Spherical-TTA embedding space used at test time.
    """
    device = config.DEVICE
    model.eval()

    if output_path is None:
        output_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")

    ds = FishScaleDatasetV4(config.ROOT_DIR, split=split, seed=config.RANDOM_SEED)
    class_to_embeddings = {i: [] for i in range(len(class_names))}

    eval_tfm = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[Prototypes] Extracting 7-View Spherical-TTA embeddings from split='{split}' ({len(ds)} specimens)...")
    with torch.no_grad():
        for i in range(len(ds)):
            path = ds.image_paths[i]
            label = ds.labels[i]
            pil_img = ds.load_processed_pil(path)
            if getattr(config, "USE_SPHERICAL_TTA", True):
                batch = torch.stack(_tta_views(pil_img, eval_tfm)).to(device)
                emb = F.normalize(model(batch).sum(dim=0, keepdim=True), p=2, dim=1)
            else:
                batch = eval_tfm(pil_img).unsqueeze(0).to(device)
                emb = F.normalize(model(batch), p=2, dim=1)
            class_to_embeddings[label].append(emb.cpu())

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

            # Use K=2 subcenters only for populous Head classes (>= 15 train shots) so minority classes never split on outlier noise
            if embs_tensor.size(0) >= 15:
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(embs_tensor.numpy())
                sub_centers = F.normalize(torch.tensor(kmeans.cluster_centers_, dtype=torch.float32), p=2, dim=1)
                subcenter_dict[c] = sub_centers
            else:
                subcenter_dict[c] = proto

        prototype_vectors.append(proto)

    prototypes = torch.cat(prototype_vectors, dim=0)

    checkpoint = {
        "prototypes": prototypes,
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
    print(f"[Prototypes] Successfully saved {len(class_names)} strictly leakage-free ('{split}') prototypes -> {output_path}")
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
