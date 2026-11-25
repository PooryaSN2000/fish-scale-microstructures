import os
import sys
import torch
from torch.utils.data import DataLoader

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from dataset import FishScaleDataset, EpisodicBatchSampler
from model import ProtoNet, compute_prototypes_and_distances, compute_protonet_loss


def run_diagnostics():
    print("="*50)
    print("RUNNING PIPELINE DIAGNOSTIC CHECKS")
    print("="*50)
    
    # 1. Test Config Settings
    print(f"Device: {config.DEVICE}")
    print(f"Dataset path: {config.ROOT_DIR}")
    print(f"FSL Setting: {config.N_WAY}-way, {config.K_SHOT}-shot, {config.Q_QUERY}-query")
    print(f"CNN Backbone: {config.BACKBONE}")
    print(f"Classical Features Toggle: {config.USE_CLASSICAL}")
    print("--> Config loaded successfully.")
    
    # 2. Test Dataset Loading
    try:
        train_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="train")
        val_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="val")
        test_dataset = FishScaleDataset(root_dir=config.ROOT_DIR, split="test")
        
        print(f"\nTrain Classes: {len(train_dataset.classes)}")
        print(f"Val Classes: {len(val_dataset.classes)}")
        print(f"Test Classes: {len(test_dataset.classes)}")
        
        # Verify disjointness
        assert set(train_dataset.classes).isdisjoint(set(val_dataset.classes)), "Train and Val classes overlap!"
        assert set(train_dataset.classes).isdisjoint(set(test_dataset.classes)), "Train and Test classes overlap!"
        assert set(val_dataset.classes).isdisjoint(set(test_dataset.classes)), "Val and Test classes overlap!"
        print("--> Class disjointness assertion passed.")
        
    except Exception as e:
        print(f"\n[FAIL] Dataset loading failed: {e}")
        return False
        
    # 3. Test Sampler and Batch Generation
    try:
        n_episodes = 2
        sampler = EpisodicBatchSampler(
            labels=train_dataset.labels,
            n_way=config.N_WAY,
            k_shot=config.K_SHOT,
            q_query=config.Q_QUERY,
            num_episodes=n_episodes
        )
        
        loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0)
        
        # Fetch one batch
        batch = next(iter(loader))
        if config.USE_CLASSICAL:
            images, labels, classical_feats = batch
            print(f"\nBatch loaded:")
            print(f"  - Images shape: {images.shape}")
            print(f"  - Labels shape: {labels.shape}")
            print(f"  - Classical features shape: {classical_feats.shape}")
            
            # Dimensions validation
            expected_batch_size = config.N_WAY * (config.K_SHOT + config.Q_QUERY)
            assert images.shape[0] == expected_batch_size, "Batch size mismatch!"
            assert classical_feats.shape[0] == expected_batch_size, "Classical features batch size mismatch!"
        else:
            images, labels = batch
            print(f"\nBatch loaded:")
            print(f"  - Images shape: {images.shape}")
            print(f"  - Labels shape: {labels.shape}")
            
            expected_batch_size = config.N_WAY * (config.K_SHOT + config.Q_QUERY)
            assert images.shape[0] == expected_batch_size, "Batch size mismatch!"
            
        print("--> Sampler and DataLoader tests passed.")
        
    except Exception as e:
        print(f"\n[FAIL] DataLoader batch loading failed: {e}")
        return False
        
    # 4. Test Model Forward/Backward Pass
    try:
        # Determine classical feature size
        classical_in_dim = 0
        if config.USE_CLASSICAL:
            classical_in_dim = classical_feats.shape[1]
            
        # Instantiate model (run on CPU/GPU based on device)
        device = torch.device("cpu")  # Use CPU for fast local verification
        
        model = ProtoNet(
            backbone_name=config.BACKBONE,
            use_pretrained=False,  # Set false to speed up model creation for test
            embedding_dim=config.EMBEDDING_DIM,
            use_classical=config.USE_CLASSICAL,
            classical_in_dim=classical_in_dim,
            classical_proj_dim=config.CLASSICAL_PROJ_DIM
        ).to(device)
        
        model.train()
        
        # Prepare inputs
        images_dev = images.to(device)
        if config.USE_CLASSICAL:
            classical_dev = classical_feats.to(device)
            embeddings = model(images_dev, classical_dev)
        else:
            embeddings = model(images_dev)
            
        print(f"\nEmbeddings extracted shape: {embeddings.shape}")
        
        # Compute distances
        distances, targets = compute_prototypes_and_distances(
            embeddings,
            n_way=config.N_WAY,
            k_shot=config.K_SHOT,
            q_query=config.Q_QUERY
        )
        print(f"Distances shape: {distances.shape}")
        print(f"Targets shape: {targets.shape}")
        
        # Compute loss
        loss, acc = compute_protonet_loss(distances, targets)
        print(f"Episodic loss: {loss.item():.4f} | Episodic accuracy: {acc.item():.4f}")
        
        # Backward pass
        loss.backward()
        print("--> Forward and Backward pass successful.")
        
    except Exception as e:
        print(f"\n[FAIL] Model execution failed: {e}")
        return False
        
    # 5. Test Checkpoint Saving
    try:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "verify_checkpoint.pth")
        checkpoint = {
            "epoch": 0,
            "model_state_dict": model.state_dict(),
            "val_acc": acc.item(),
            "config": {
                "backbone": config.BACKBONE,
                "embedding_dim": config.EMBEDDING_DIM,
                "use_classical": config.USE_CLASSICAL,
                "classical_in_dim": classical_in_dim,
                "classical_proj_dim": config.CLASSICAL_PROJ_DIM,
                "n_way": config.N_WAY,
                "k_shot": config.K_SHOT,
            }
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"\nSaved test checkpoint to: {checkpoint_path}")
        
        # Load and verify
        loaded = torch.load(checkpoint_path, map_location="cpu")
        assert loaded["config"]["backbone"] == config.BACKBONE
        print("--> Checkpoint saving and loading verified.")
        os.remove(checkpoint_path)
        
    except Exception as e:
        print(f"\n[FAIL] Checkpoint operations failed: {e}")
        return False
        
    print("\n" + "="*50)
    print("ALL PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("="*50)
    return True


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
