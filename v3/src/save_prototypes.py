import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from PIL import Image

import config
from dataset import FishScaleDataset
from model import FishArcNet


def get_prototype_augmentations():
    """
    Returns an augmentation pipeline that generates 12 diverse spatial views per image
    to ensure the prototype centroid is completely rotation and scale invariant.
    """
    angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
    transforms_list = []
    
    for ang in angles:
        transforms_list.append(v2.Compose([
            v2.Resize(config.IMAGE_SIZE),
            v2.RandomRotation(degrees=(ang, ang)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
        
    return transforms_list


def compute_subcenters(embeddings: torch.Tensor, k: int = 2) -> torch.Tensor:
    """
    Computes k unit-normalized subcenters from a collection of unit embeddings.
    If N < k, replicates the mean centroid.
    """
    if embeddings.size(0) < k:
        centroid = F.normalize(embeddings.mean(dim=0, keepdim=True), p=2, dim=-1)
        return centroid.repeat(k, 1)
    
    # Initialize with top-2 farthest points for maximum coverage
    c1 = embeddings[0:1]
    dists = 1.0 - torch.matmul(embeddings, c1.t()).squeeze(-1)
    idx2 = torch.argmax(dists).item()
    c2 = embeddings[idx2:idx2+1]
    centers = torch.cat([c1, c2], dim=0)
    
    # 5 iterations of Spherical K-Means
    for _ in range(5):
        sims = torch.matmul(embeddings, centers.t())
        assign = torch.argmax(sims, dim=-1)
        new_centers = []
        for j in range(k):
            mask = (assign == j)
            if mask.sum() > 0:
                new_c = F.normalize(embeddings[mask].mean(dim=0, keepdim=True), p=2, dim=-1)
            else:
                new_c = centers[j:j+1]
            new_centers.append(new_c)
        centers = torch.cat(new_centers, dim=0)
        
    return centers


def main():
    device = config.DEVICE
    print(f"Generating Multi-View Augmented Prototypes and Sub-Centers on {device}...")
    
    # 1. Load train dataset
    train_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # 2. Load trained model
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train the model first.")
        
    checkpoint = torch.load(model_path, map_location=device)
    model = FishArcNet(
        backbone_name=config.BACKBONE,
        use_pretrained=False,
        num_classes=num_classes,
        embedding_dim=config.EMBEDDING_DIM,
        scale=config.ARCFACE_SCALE,
        margin=config.ARCFACE_MARGIN
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Get the learned ArcFace class weight prototypes
    arcface_weights = F.normalize(model.arcface.weight.data, p=2, dim=-1)
    
    aug_transforms = get_prototype_augmentations()
    prototypes = torch.zeros((num_classes, config.EMBEDDING_DIM), device=device)
    subcenters = torch.zeros((num_classes, 2, config.EMBEDDING_DIM), device=device)
    
    print(f"Extracting multi-angle embeddings across all {num_classes} species...")
    with torch.no_grad():
        for class_idx, class_name in enumerate(class_names):
            class_indices = [i for i, lbl in enumerate(train_dataset.labels) if lbl == class_idx]
            class_embs = []
            
            for idx in class_indices:
                raw_img = train_dataset.image_cache[idx]
                
                # Extract embeddings for each rotation angle
                for t_fn in aug_transforms:
                    tensor_img = t_fn(raw_img).unsqueeze(0).to(device)
                    emb = model.extract_embedding(tensor_img)
                    class_embs.append(emb.squeeze(0))
                    
            # Compute empirical augmented centroid
            class_embs_tensor = torch.stack(class_embs, dim=0)
            empirical_centroid = F.normalize(class_embs_tensor.mean(dim=0), p=2, dim=-1)
            
            # Blend empirical centroid (70%) with ArcFace learned metric vector (30%)
            blended = 0.7 * empirical_centroid + 0.3 * arcface_weights[class_idx]
            prototypes[class_idx] = F.normalize(blended, p=2, dim=-1)
            
            # Compute K=2 subcenters
            subcenters[class_idx] = compute_subcenters(class_embs_tensor, k=2)
            
            print(f"  [{class_idx+1:02d}/{num_classes}] '{class_name}': {len(class_indices)} images -> {len(class_embs)} views -> 2 subcenters.")
            
    # 3. Save Prototypes and Sub-Centers
    save_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")
    torch.save({
        "prototypes": prototypes.cpu(),
        "subcenters": subcenters.cpu(),
        "class_names": class_names,
        "embedding_dim": config.EMBEDDING_DIM,
        "image_size": config.IMAGE_SIZE,
        "use_clahe": getattr(config, "USE_CLAHE", True)
    }, save_path)
    
    print(f"\n[OK] Successfully saved {num_classes} unit prototypes and sub-centers to {save_path}!")


if __name__ == "__main__":
    main()
