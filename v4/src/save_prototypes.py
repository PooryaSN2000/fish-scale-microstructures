import os
import torch
import torch.nn.functional as F
import numpy as np
try:
    from . import config
    from .model import FishScaleConvNeXtV4
    from .dataset import FishScaleDatasetV4
except ImportError:
    import config
    from model import FishScaleConvNeXtV4
    from dataset import FishScaleDatasetV4

def compute_and_save_prototypes(model, class_names, output_path=None):
    """
    Computes class prototypes on the unit hypersphere for all species.
    """
    device = config.DEVICE
    model.eval()

    if output_path is None:
        output_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")

    # Load complete dataset without heavy train augmentations
    full_dataset = FishScaleDatasetV4(config.ROOT_DIR, split="all", seed=config.RANDOM_SEED)
    class_to_embeddings = {i: [] for i in range(len(class_names))}

    print(f"[Prototypes] Extracting hyperspherical embeddings for {len(full_dataset)} biological specimens...")
    with torch.no_grad():
        for i in range(len(full_dataset)):
            img, label, _ = full_dataset[i]
            img = img.unsqueeze(0).to(device)
            emb = model(img)  # (1, embedding_dim)
            class_to_embeddings[label].append(emb.cpu())

    prototype_vectors = []
    subcenter_dict = {}

    for c in range(len(class_names)):
        embs = class_to_embeddings[c]
        if len(embs) == 0:
            print(f"[Warning] Class {c} ({class_names[c]}) has 0 embeddings! Initializing randomly.")
            proto = torch.randn(1, config.EMBEDDING_DIM)
            proto = F.normalize(proto, p=2, dim=1)
        else:
            embs_tensor = torch.cat(embs, dim=0)  # (N_c, embedding_dim)
            # Hyperspherical Mean Centroid
            mean_vector = embs_tensor.mean(dim=0, keepdim=True)
            proto = F.normalize(mean_vector, p=2, dim=1)

            # Sub-center clustering for classes with sufficient sample count
            if embs_tensor.size(0) >= 6:
                # 2 sub-centers (e.g. lateral vs dorsal scales)
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=2, random_state=42, n_init=5).fit(embs_tensor.numpy())
                sub_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
                sub_centers = F.normalize(sub_centers, p=2, dim=1)
                subcenter_dict[c] = sub_centers
            else:
                subcenter_dict[c] = proto

        prototype_vectors.append(proto)

    prototypes = torch.cat(prototype_vectors, dim=0)  # (num_classes, embedding_dim)

    checkpoint = {
        "prototypes": prototypes,
        "subcenters": subcenter_dict,
        "class_names": class_names,
        "embedding_dim": config.EMBEDDING_DIM,
        "image_size": config.IMAGE_SIZE,
        "use_clahe": config.USE_CLAHE
    }

    torch.save(checkpoint, output_path)
    print(f"[Prototypes] Successfully saved {len(class_names)} prototypes to {output_path}")
    return checkpoint

if __name__ == "__main__":
    from .dataset import FishScaleDatasetV4
    ds = FishScaleDatasetV4(config.ROOT_DIR, split="train")
    model = FishScaleConvNeXtV4(embedding_dim=config.EMBEDDING_DIM, use_mixstyle=False).to(config.DEVICE)
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=config.DEVICE)
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    compute_and_save_prototypes(model, ds.classes)
