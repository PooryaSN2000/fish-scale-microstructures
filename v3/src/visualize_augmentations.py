import os
import random
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import v2
import numpy as np

import config

def main():
    # We will pick two random images from the dataset to show original vs augmented
    # Specifically, let's pick one from Lutjanus johni and one from Lutjanus lutjanus
    class1 = "Lutjanus johni Bloch 1792"
    class2 = "Lutjanus lutjanus Bloch 1790"
    
    dir1 = os.path.join(config.ROOT_DIR, class1)
    dir2 = os.path.join(config.ROOT_DIR, class2)
    
    img_name1 = [f for f in os.listdir(dir1) if f.endswith(('.jpg', '.png'))][0]
    img_name2 = [f for f in os.listdir(dir2) if f.endswith(('.jpg', '.png'))][0]
    
    path1 = os.path.join(dir1, img_name1)
    path2 = os.path.join(dir2, img_name2)
    
    # Base transform to tensor
    base_transform = v2.Compose([
        v2.Resize(config.IMAGE_SIZE),
        v2.ToTensor()
    ])
    
    img1_pil = Image.open(path1).convert("RGB")
    img2_pil = Image.open(path2).convert("RGB")
    
    t1 = base_transform(img1_pil)
    t2 = base_transform(img2_pil)
    
    # Let's define the individual augmentations
    spatial_aug = v2.Compose([
        v2.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.8, 1.0), antialias=True),
        v2.RandomRotation(180),
        v2.RandomHorizontalFlip(p=1.0)
    ])
    
    color_aug = v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3) # Slightly exaggerated for visualization
    
    # Apply standard augs
    t1_spatial = spatial_aug(t1)
    t1_color = color_aug(t1)
    
    # For CutMix/MixUp, we need a batch
    batch_imgs = torch.stack([t1, t2])
    # Dummy one-hot labels
    batch_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    
    cutmix = v2.CutMix(num_classes=2, alpha=1.0)
    mixup = v2.MixUp(num_classes=2, alpha=0.5) # alpha=0.5 for a clear mix
    
    # Fix seed for reproducible visual
    torch.manual_seed(42)
    mixed_imgs_cutmix, _ = cutmix(batch_imgs.clone(), batch_labels.clone())
    mixed_imgs_mixup, _ = mixup(batch_imgs.clone(), batch_labels.clone())
    
    # We will display:
    # Row 1: Original 1, Spatial Aug, Color Aug
    # Row 2: Original 2, CutMix (1+2), MixUp (1+2)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    def imshow(ax, tensor, title):
        img = tensor.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        
    imshow(axes[0, 0], t1, "Original (Species A)")
    imshow(axes[0, 1], t1_spatial, "Spatial Aug (Crop/Rot/Flip)")
    imshow(axes[0, 2], t1_color, "Color Jitter")
    
    imshow(axes[1, 0], t2, "Original (Species B)")
    imshow(axes[1, 1], mixed_imgs_cutmix[0], "CutMix (A + B)")
    imshow(axes[1, 2], mixed_imgs_mixup[0], "MixUp (A + B)")
    
    plt.tight_layout()
    out_path = os.path.join(config.CHECKPOINT_DIR, "augmentation_samples.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved augmentation visualization to {out_path}")

if __name__ == "__main__":
    main()
