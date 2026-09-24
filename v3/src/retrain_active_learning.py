import os
import sys
import json
import csv
import glob
import shutil
import argparse
from datetime import datetime
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

# Local imports
import config
from model import FishArcNet
from dataset import apply_clahe_preprocessing


class ActiveLearningEngine:
    """
    Automated Active Learning Engine for AquaLens AI v3.
    
    Provides:
    1. Inspection of human-in-the-loop expert feedback data.
    2. Online adaptive prototype & sub-center synchronization (Instant EMA re-centering).
    3. Continual fine-tuning pipeline with experience replay.
    """
    def __init__(
        self,
        active_learning_dir: str = None,
        checkpoints_dir: str = None,
        device: torch.device = None
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.active_learning_dir = active_learning_dir or os.path.join(
            os.path.dirname(base_dir), "webapp", "active_learning_data"
        )
        self.checkpoints_dir = checkpoints_dir or os.path.join(
            os.path.dirname(base_dir), "checkpoints"
        )
        self.device = device or config.DEVICE
        self.verified_samples_dir = os.path.join(self.active_learning_dir, "verified_samples")
        os.makedirs(self.verified_samples_dir, exist_ok=True)

        self.transform = v2.Compose([
            v2.Resize(config.IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_stats(self) -> dict:
        """
        Aggregates statistics on verified feedback images and log entries.
        """
        log_jsonl = os.path.join(self.active_learning_dir, "feedback_log.jsonl")
        entries = []
        confirmed_count = 0
        corrected_count = 0
        
        if os.path.exists(log_jsonl):
            with open(log_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            entries.append(item)
                            if item.get("status") == "confirmed":
                                confirmed_count += 1
                            elif item.get("status") == "corrected":
                                corrected_count += 1
                        except Exception:
                            continue

        # Count actual image files per species folder
        species_counts = {}
        total_images = 0
        if os.path.exists(self.verified_samples_dir):
            for sp_folder in sorted(os.listdir(self.verified_samples_dir)):
                full_sp_path = os.path.join(self.verified_samples_dir, sp_folder)
                if os.path.isdir(full_sp_path):
                    imgs = glob.glob(os.path.join(full_sp_path, "*.jpg")) + glob.glob(os.path.join(full_sp_path, "*.png"))
                    if len(imgs) > 0:
                        species_counts[sp_folder] = len(imgs)
                        total_images += len(imgs)

        # Check last sync record
        sync_history_path = os.path.join(self.active_learning_dir, "sync_history.json")
        last_sync = None
        if os.path.exists(sync_history_path):
            try:
                with open(sync_history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    if history and len(history) > 0:
                        last_sync = history[-1]
            except Exception:
                pass

        return {
            "total_verified_images": total_images,
            "total_feedback_entries": len(entries),
            "confirmed_count": confirmed_count,
            "corrected_count": corrected_count,
            "species_counts": species_counts,
            "num_active_species": len(species_counts),
            "last_sync": last_sync,
            "recent_entries": entries[-5:] if entries else []
        }

    def _extract_multi_view_embedding(self, model: torch.nn.Module, pil_image: Image.Image) -> torch.Tensor:
        """
        Extracts rotation-invariant 5-pass TTA embedding for a verified image.
        """
        views = [
            pil_image,
            pil_image.transpose(Image.ROTATE_90),
            pil_image.transpose(Image.ROTATE_180),
            pil_image.transpose(Image.ROTATE_270),
            pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        ]
        embs = []
        for v in views:
            proc_img = v.convert("RGB").resize(config.IMAGE_SIZE)
            if getattr(config, "USE_CLAHE", True):
                proc_img = apply_clahe_preprocessing(proc_img)
            tensor_img = self.transform(proc_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = model.extract_embedding(tensor_img)
                embs.append(emb.squeeze(0))
                
        avg_emb = torch.stack(embs, dim=0).mean(dim=0)
        return F.normalize(avg_emb, p=2, dim=-1)

    def sync_prototypes(self, eta: float = 0.20) -> dict:
        """
        Performs Instant Prototype Synchronization.
        
        Reads all verified samples from active_learning_data/verified_samples/,
        extracts their feature embeddings with the trained model, and shifts the
        corresponding class prototypes and sub-centers via adaptive exponential smoothing.
        
        Args:
            eta: Adaptive update weight (0.0 to 1.0). Higher eta places more weight on new samples.
            
        Returns:
            Dict containing sync status, number of updated classes, and execution time.
        """
        import time
        start_time = time.time()
        
        prototypes_path = os.path.join(self.checkpoints_dir, "prototypes.pth")
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Base prototypes not found at {prototypes_path}")

        model_path = os.path.join(self.checkpoints_dir, "best_model.pth")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")

        # 1. Load Prototypes
        proto_data = torch.load(prototypes_path, map_location="cpu")
        prototypes = proto_data["prototypes"].to(self.device)          # (40, 384)
        subcenters = proto_data["subcenters"].to(self.device)          # (40, 2, 384)
        class_names = proto_data["class_names"]
        name_to_idx = {name: i for i, name in enumerate(class_names)}

        # 2. Load Model for embedding extraction
        checkpoint = torch.load(model_path, map_location=self.device)
        model = FishArcNet(
            backbone_name=config.BACKBONE,
            use_pretrained=False,
            num_classes=len(class_names),
            embedding_dim=config.EMBEDDING_DIM,
            scale=config.ARCFACE_SCALE,
            margin=config.ARCFACE_MARGIN
        ).to(self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # 3. Process verified samples per species
        updated_species = {}
        total_samples_processed = 0

        for sp_folder in os.listdir(self.verified_samples_dir):
            full_sp_path = os.path.join(self.verified_samples_dir, sp_folder)
            if not os.path.isdir(full_sp_path):
                continue
                
            # Match folder name to class name
            matched_idx = None
            if sp_folder in name_to_idx:
                matched_idx = name_to_idx[sp_folder]
            else:
                # Fuzzy normalized match
                clean_target = sp_folder.replace("_", " ").lower().strip()
                for c_name, idx in name_to_idx.items():
                    if c_name.replace("_", " ").lower().strip() == clean_target:
                        matched_idx = idx
                        break

            if matched_idx is None:
                continue

            img_files = glob.glob(os.path.join(full_sp_path, "*.jpg")) + glob.glob(os.path.join(full_sp_path, "*.png"))
            if not img_files:
                continue

            # Accumulate embeddings for this species
            new_embs = []
            for img_p in img_files:
                try:
                    img = Image.open(img_p)
                    emb = self._extract_multi_view_embedding(model, img)
                    new_embs.append(emb)
                    total_samples_processed += 1
                except Exception as e:
                    print(f"Warning: Could not process {img_p}: {e}")

            if not new_embs:
                continue

            # 4. Adaptive Sub-Center & Centroid Update
            for z_new in new_embs:
                # Subcenters for this class: (2, 384)
                sc = subcenters[matched_idx]
                sims = torch.matmul(sc, z_new)  # (2,)
                best_sub_idx = torch.argmax(sims).item()

                # Update the closest subcenter towards z_new
                updated_sub = (1.0 - eta) * sc[best_sub_idx] + eta * z_new
                subcenters[matched_idx, best_sub_idx] = F.normalize(updated_sub, p=2, dim=-1)

                # Update the global prototype centroid
                curr_proto = prototypes[matched_idx]
                updated_proto = (1.0 - (eta * 0.5)) * curr_proto + (eta * 0.5) * z_new
                prototypes[matched_idx] = F.normalize(updated_proto, p=2, dim=-1)

            updated_species[class_names[matched_idx]] = len(new_embs)

        if total_samples_processed == 0:
            return {
                "success": True,
                "message": "No new verified images found to sync. Prototypes are up-to-date.",
                "samples_processed": 0,
                "updated_classes": 0,
                "elapsed_time": f"{time.time() - start_time:.2f}s"
            }

        # 5. Backup existing prototypes & save updated
        backup_path = os.path.join(self.checkpoints_dir, "prototypes_backup.pth")
        shutil.copyfile(prototypes_path, backup_path)

        proto_data["prototypes"] = prototypes.cpu()
        proto_data["subcenters"] = subcenters.cpu()
        proto_data["last_al_sync"] = datetime.utcnow().isoformat() + "Z"
        proto_data["al_samples_count"] = total_samples_processed

        torch.save(proto_data, prototypes_path)

        # 6. Record sync history
        sync_record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "samples_synced": total_samples_processed,
            "classes_updated": list(updated_species.keys()),
            "elapsed_time_seconds": round(time.time() - start_time, 3)
        }
        sync_history_path = os.path.join(self.active_learning_dir, "sync_history.json")
        history = []
        if os.path.exists(sync_history_path):
            try:
                with open(sync_history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(sync_record)
        with open(sync_history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)

        return {
            "success": True,
            "message": f"Successfully updated prototypes for {len(updated_species)} species using {total_samples_processed} verified samples.",
            "samples_processed": total_samples_processed,
            "updated_species": updated_species,
            "backup_created": backup_path,
            "elapsed_time": f"{time.time() - start_time:.2f}s"
        }


def main():
    parser = argparse.ArgumentParser(description="AquaLens AI v3 - Active Learning Synchronization")
    parser.add_argument("--mode", choices=["stats", "sync"], default="stats", help="Mode: stats or sync")
    parser.add_argument("--eta", type=float, default=0.20, help="Adaptive update weight (default: 0.20)")
    args = parser.parse_args()

    engine = ActiveLearningEngine()

    if args.mode == "stats":
        stats = engine.get_stats()
        print("\n=== AquaLens AI v3 - Active Learning Statistics ===")
        print(f"Total Verified Images:     {stats['total_verified_images']}")
        print(f"Total Feedback Entries:    {stats['total_feedback_entries']}")
        print(f"  - Confirmed by Expert:   {stats['confirmed_count']}")
        print(f"  - Corrected by Expert:   {stats['corrected_count']}")
        print(f"Active Species in Bank:    {stats['num_active_species']}")
        if stats['species_counts']:
            print("Species Breakdown:")
            for sp, cnt in stats['species_counts'].items():
                print(f"  • {sp}: {cnt} sample(s)")
        if stats['last_sync']:
            print(f"Last Synchronization:      {stats['last_sync'].get('timestamp')}")
        else:
            print("Last Synchronization:      None (Never synced)")
        print("===================================================\n")

    elif args.mode == "sync":
        print("Starting Active Learning Prototype Synchronization...")
        res = engine.sync_prototypes(eta=args.eta)
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
