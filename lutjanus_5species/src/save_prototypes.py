import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import v2

import config
from dataset import FishScaleDataset
from model import ProtoNet

def main():
    print(f"Using Device: {config.DEVICE}")
    
    # 1. Load dataset (Train Split only for prototypes)
    train_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="train", seed=config.RANDOM_SEED)
    class_names = train_dataset.classes
    print(f"Target classes: {class_names}")
    
    # 2. Load Model
    classical_in_dim = 0
    if config.USE_CLASSICAL:
        _, _, sample_classical = train_dataset[0]
        classical_in_dim = sample_classical.shape[0]
        
    model = ProtoNet(
        backbone_name=config.BACKBONE,
        use_pretrained=False,
        embedding_dim=config.EMBEDDING_DIM,
        use_classical=config.USE_CLASSICAL,
        classical_in_dim=classical_in_dim,
        classical_proj_dim=config.CLASSICAL_PROJ_DIM
    )
    
    model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    checkpoint = torch.load(model_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(config.DEVICE)
    model.eval()
    
    # Transforms
    gpu_transforms = v2.Compose([
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Extract all train embeddings
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings for all training images...")
    with torch.no_grad():
        for batch in loader:
            if config.USE_CLASSICAL:
                images, labels, classical_feats = batch
                images = images.to(config.DEVICE)
                classical_feats = classical_feats.to(config.DEVICE)
                images = gpu_transforms(images)
                
                with torch.autocast(device_type=config.DEVICE.type if config.DEVICE.type != 'mac' else 'cpu', enabled=config.DEVICE.type=='cuda'):
                    embs = model(images, classical_feats, stage=2)
            else:
                images, labels = batch
                images = images.to(config.DEVICE)
                images = gpu_transforms(images)
                with torch.autocast(device_type=config.DEVICE.type if config.DEVICE.type != 'mac' else 'cpu', enabled=config.DEVICE.type=='cuda'):
                    embs = model(images, stage=2)
                    
            all_embeddings.append(embs.cpu())
            all_labels.append(labels)
            
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # 4. Compute class prototypes
    num_classes = len(class_names)
    prototypes = torch.zeros(num_classes, all_embeddings.shape[-1])
    
    print("Computing prototypes...")
    for class_idx in range(num_classes):
        mask = (all_labels == class_idx)
        class_embs = all_embeddings[mask]
        prototypes[class_idx] = class_embs.mean(dim=0)
        print(f"Class '{class_names[class_idx]}' prototype generated from {class_embs.size(0)} images.")
        
    # 5. Save prototypes
    save_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")
    torch.save({
        "prototypes": prototypes.to(config.DEVICE),
        "class_names": class_names
    }, save_path)
    
    print(f"Successfully saved {num_classes} prototypes to {save_path}")

if __name__ == "__main__":
    main()
