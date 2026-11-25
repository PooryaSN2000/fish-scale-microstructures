import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

import config
from dataset import FishScaleDataset, apply_clahe_preprocessing
from model import ProtoNet
from features import extract_classical_features


def extract_4view_tta_embedding(model, pil_img: Image.Image, base_transform, device, use_tta: bool = True, use_classical: bool = False):
    """
    Computes L2-normalized embedding averaged over 4 geometric views on the unit hypersphere.
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

    with torch.no_grad():
        embs = model(batch_tensors, cf_tensor, stage=2)
        embs_norm = F.normalize(embs, p=2, dim=-1)
        mean_emb = F.normalize(embs_norm.mean(dim=0, keepdim=True), p=2, dim=-1)
    return mean_emb


def main():
    print(f"Using Device: {config.DEVICE}")
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    mcfg = checkpoint["config"]

    use_clahe = mcfg.get("use_clahe", config.USE_CLAHE)
    use_tta = mcfg.get("use_tta", config.USE_TTA)
    use_classical = mcfg.get("use_classical", config.USE_CLASSICAL)
    img_size = tuple(mcfg.get("image_size", config.IMAGE_SIZE))

    # Extract prototypes ONLY from training split (no data leakage)
    train_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED, use_clahe=use_clahe)
    class_names = train_dataset.classes

    model = ProtoNet(
        backbone_name=mcfg["backbone"],
        use_pretrained=False,
        embedding_dim=mcfg["embedding_dim"],
        use_classical=use_classical,
        classical_in_dim=mcfg.get("classical_in_dim", 0),
        classical_proj_dim=mcfg.get("classical_proj_dim", config.CLASSICAL_PROJ_DIM),
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
    for idx in range(len(train_dataset)):
        pil_img = train_dataset.image_cache[idx]
        lbl = train_dataset.labels[idx]
        emb = extract_4view_tta_embedding(model, pil_img, base_transform, config.DEVICE, use_tta=use_tta, use_classical=use_classical)
        class_to_embs[lbl].append(emb.cpu())

    prototypes = []
    for c in range(len(class_names)):
        embs_c = torch.cat(class_to_embs[c], dim=0)
        proto = F.normalize(embs_c.mean(dim=0, keepdim=True), p=2, dim=-1)
        prototypes.append(proto)
        print(f"Class '{class_names[c]}' prototype computed from {embs_c.size(0)} train samples.")

    prototypes_tensor = torch.cat(prototypes, dim=0)
    save_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")
    torch.save({
        "prototypes": prototypes_tensor.to(config.DEVICE),
        "class_names": class_names,
        "use_clahe": use_clahe,
        "use_tta": use_tta,
        "image_size": img_size,
    }, save_path)
    print(f"Saved {len(class_names)} prototypes to {save_path}")


if __name__ == "__main__":
    main()
