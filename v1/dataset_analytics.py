import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import config

def plot_dataset_samples(root_dir, top_classes, save_dir):
    """Plots a 4x4 grid of random images from the top selected classes."""
    plt.figure(figsize=(16, 16))
    sns.set_style("white")
    
    # Pick up to 16 classes
    sample_classes = random.sample(top_classes, min(16, len(top_classes)))
    
    for i, class_name in enumerate(sample_classes):
        class_path = os.path.join(root_dir, class_name)
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not files: continue
        
        img_file = random.choice(files)
        img_path = os.path.join(class_path, img_file)
        
        try:
            img = Image.open(img_path).convert("RGB")
            plt.subplot(4, 4, i + 1)
            plt.imshow(img)
            # Shorten class name if it's too long
            short_name = class_name[:30] + "..." if len(class_name) > 30 else class_name
            plt.title(short_name, fontsize=11, fontweight='bold')
            plt.axis('off')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset_samples.png"), dpi=200)
    plt.close()
    print(f"Saved dataset samples grid to {os.path.join(save_dir, 'dataset_samples.png')}")

def analyze_dataset(root_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    if not os.path.exists(root_dir):
        print(f"Root dir {root_dir} not found. Ensure the dataset is extracted.")
        return
        
    raw_classes = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    
    class_counts = []
    for class_name in raw_classes:
        class_path = os.path.join(root_dir, class_name)
        files = os.listdir(class_path)
        valid_files = [f for f in files if os.path.splitext(f.lower())[1] in supported_exts]
        if len(valid_files) > 0:
            class_counts.append((class_name, len(valid_files)))
            
    # Sort by count descending
    class_counts.sort(key=lambda x: x[1], reverse=True)
    
    names = [x[0] for x in class_counts]
    counts = [x[1] for x in class_counts]
    
    # --- Plot 1: Dataset Distribution Bar Chart ---
    plt.figure(figsize=(20, 8))
    sns.set_style("whitegrid")
    
    colors = ['#1f77b4' if i < config.NUM_CLASSES else '#d3d3d3' for i in range(len(names))]
    bars = plt.bar(names, counts, color=colors)
    
    if len(names) > config.NUM_CLASSES:
        plt.axvline(x=config.NUM_CLASSES - 0.5, color='red', linestyle='--', linewidth=2, label=f'Top {config.NUM_CLASSES} Selection Cutoff')
        
    plt.title("Fish Species Sample Distribution (Long-Tail Data Imbalance)", fontsize=18, fontweight='bold')
    plt.xlabel("Fish Species", fontsize=14)
    plt.ylabel("Number of Images", fontsize=14)
    plt.xticks(rotation=90, ha='right', fontsize=8)
    
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='#1f77b4', label=f'Selected (Top {config.NUM_CLASSES})'),
        Patch(facecolor='#d3d3d3', label='Discarded (Minority Classes)')
    ]
    if len(names) > config.NUM_CLASSES:
        legend_elements.append(Line2D([0], [0], color='red', lw=2, linestyle='--', label='Cutoff Threshold'))
    plt.legend(handles=legend_elements, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "dataset_distribution.png"), dpi=200)
    plt.close()
    
    print("=== Dataset Statistics ===")
    print(f"Total Classes Found: {len(names)}")
    print(f"Total Images: {sum(counts)}")
    print(f"Mean images per class: {np.mean(counts):.2f}")
    print(f"Median images per class: {np.median(counts):.2f}")
    print(f"Max images in a class: {np.max(counts)}")
    print(f"Min images in a class: {np.min(counts)}")
    print(f"\nTop {config.NUM_CLASSES} Selected Total Images: {sum(counts[:config.NUM_CLASSES])}")
    print(f"Chart saved to {os.path.join(save_dir, 'dataset_distribution.png')}")
    
    # --- Plot 2: Dataset Samples Grid ---
    print("Generating dataset samples grid...")
    top_classes = names[:config.NUM_CLASSES]
    plot_dataset_samples(root_dir, top_classes, save_dir)
    
if __name__ == "__main__":
    analyze_dataset(config.ROOT_DIR, config.CHECKPOINT_DIR)
