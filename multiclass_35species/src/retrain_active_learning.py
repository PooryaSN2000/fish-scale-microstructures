import os
import sys
import json
import shutil
from datetime import datetime
import torch
import torch.nn.functional as F

try:
    from . import config
    from .dataset import load_rgb_image
except ImportError:
    import config
    from dataset import load_rgb_image


class ActiveLearningEngineV4:
    """
    Real-Time Active Learning & Zero-Retraining Open-Set Species Registration Engine.
    Supports:
    1. Expert confirmations and label corrections (EMA prototype refinement).
    2. Instant 1-to-K Shot Open-Set Novel Species Registration (adds brand-new fish species
       directly to prototypes.pth without retraining the ConvNeXt weights).
    3. Reversion/removal of custom test species after classroom/defense demos.
    """
    def __init__(self, active_learning_dir=None, checkpoints_dir=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.active_learning_dir = active_learning_dir or os.path.join(
            os.path.dirname(base_dir), "webapp", "active_learning_data"
        )
        self.checkpoints_dir = checkpoints_dir or os.path.join(
            os.path.dirname(base_dir), "checkpoints"
        )
        self.device = config.DEVICE
        self.verified_samples_dir = os.path.join(self.active_learning_dir, "verified_samples")
        self.custom_species_dir = os.path.join(self.active_learning_dir, "custom_species")
        os.makedirs(self.verified_samples_dir, exist_ok=True)
        os.makedirs(self.custom_species_dir, exist_ok=True)

    def _proto_path(self):
        return os.path.join(self.checkpoints_dir, "prototypes.pth")

    def get_stats(self) -> dict:
        log_jsonl = os.path.join(self.active_learning_dir, "feedback_log.jsonl")
        entries = []
        confirmed = 0
        corrected = 0
        novel_added = 0
        species_counts = {}

        if os.path.exists(log_jsonl):
            with open(log_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        entries.append(item)
                        sp = item.get("confirmed_species")
                        if sp:
                            species_counts[sp] = species_counts.get(sp, 0) + 1
                        st = item.get("status")
                        if st == "confirmed":
                            confirmed += 1
                        elif st == "corrected":
                            corrected += 1
                        elif st == "new_species":
                            novel_added += 1
                    except Exception:
                        pass

        custom_species = []
        proto_path = self._proto_path()
        if os.path.exists(proto_path):
            try:
                ckpt = torch.load(proto_path, map_location="cpu")
                custom_species = list(ckpt.get("custom_species", []))
            except Exception:
                pass

        sync_state_file = os.path.join(self.active_learning_dir, "sync_state.json")
        last_sync = "Never"
        if os.path.exists(sync_state_file):
            try:
                with open(sync_state_file, "r") as sf:
                    last_sync = json.load(sf).get("last_synced_at", "Never")
            except Exception:
                pass

        return {
            "total_verified": len(entries),
            "confirmed_count": confirmed,
            "corrected_count": corrected,
            "novel_registrations": novel_added,
            "custom_species": custom_species,
            "custom_species_count": len(custom_species),
            "active_species_count": len(species_counts),
            "species_distribution": species_counts,
            "last_synced_at": last_sync,
        }

    def register_new_species(self, classifier, species_name: str, image_paths: list[str]) -> dict:
        """
        Registers a brand-new unseen fish species using 1 or more micrograph files
        without retraining ConvNeXt weights, updating prototypes.pth and memory immediately.
        """
        species_name = (species_name or "").strip()
        if not species_name:
            return {"success": False, "error": "Species name cannot be empty."}
        if not image_paths:
            return {"success": False, "error": "At least one micrograph is required."}

        proto_path = self._proto_path()
        if not os.path.exists(proto_path):
            return {"success": False, "error": "prototypes.pth not found."}

        ckpt = torch.load(proto_path, map_location=self.device)
        prototypes = ckpt["prototypes"].to(self.device)
        subcenters = dict(ckpt.get("subcenters", {}))
        class_names = list(ckpt["class_names"])
        custom_species = list(ckpt.get("custom_species", []))

        embs = []
        saved_files = []
        sp_slug = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in species_name)[:48]
        sp_dir = os.path.join(self.custom_species_dir, sp_slug)
        os.makedirs(sp_dir, exist_ok=True)

        for idx, p in enumerate(image_paths):
            if not os.path.exists(p):
                continue
            pil_img = load_rgb_image(p)
            emb = classifier.extract_embedding(pil_img, use_tta=True)  # Applies ROI crop + CLAHE + 7-View TTA
            embs.append(emb)

            dest_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{idx}_{os.path.basename(p)}"
            dest_path = os.path.join(sp_dir, dest_name)
            shutil.copy2(p, dest_path)
            saved_files.append(dest_name)

        if not embs:
            return {"success": False, "error": "Could not extract valid embeddings from uploaded micrographs."}

        embs_tensor = torch.cat(embs, dim=0)  # (K, D)
        new_proto = F.normalize(embs_tensor.mean(dim=0, keepdim=True), p=2, dim=1)

        if species_name in class_names:
            c_idx = class_names.index(species_name)
            old_proto = prototypes[c_idx : c_idx + 1]
            merged_proto = F.normalize(0.5 * old_proto + 0.5 * new_proto, p=2, dim=1)
            prototypes[c_idx : c_idx + 1] = merged_proto
            old_sub = subcenters.get(c_idx, old_proto).to(self.device)
            subcenters[c_idx] = F.normalize(torch.cat([old_sub, embs_tensor], dim=0), p=2, dim=1).cpu()
            action = "updated"
        else:
            c_idx = len(class_names)
            class_names.append(species_name)
            prototypes = torch.cat([prototypes, new_proto], dim=0)
            subcenters[c_idx] = F.normalize(embs_tensor, p=2, dim=1).cpu()
            if species_name not in custom_species:
                custom_species.append(species_name)
            action = "created"

        ckpt["prototypes"] = prototypes.cpu()
        ckpt["subcenters"] = subcenters
        ckpt["class_names"] = class_names
        ckpt["custom_species"] = custom_species
        torch.save(ckpt, proto_path)
        classifier.reload_prototypes()

        # Log registration event
        log_file = os.path.join(self.active_learning_dir, "feedback_log.jsonl")
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "status": "new_species",
            "action": action,
            "confirmed_species": species_name,
            "shots": len(embs),
            "sample_filename": saved_files[0] if saved_files else "",
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.active_learning_dir, "sync_state.json"), "w") as sf:
            json.dump({"last_synced_at": now_str, "custom_species": custom_species}, sf)

        return {
            "success": True,
            "action": action,
            "species_name": species_name,
            "shots": len(embs),
            "total_species": len(class_names),
            "custom_species": custom_species,
            "message": f"Species '{species_name}' ({len(embs)} shot(s)) {action} on hypersphere without retraining.",
        }

    def remove_custom_species(self, classifier, species_name: str = None) -> dict:
        """
        Removes a custom-registered species (or all custom species if species_name is None)
        so demo sessions can be cleanly reset back to the base 35 species.
        """
        proto_path = self._proto_path()
        if not os.path.exists(proto_path):
            return {"success": False, "error": "prototypes.pth not found"}

        ckpt = torch.load(proto_path, map_location=self.device)
        base_n = int(ckpt.get("base_class_count", 35))
        class_names = list(ckpt["class_names"])
        prototypes = ckpt["prototypes"].to(self.device)
        subcenters = dict(ckpt.get("subcenters", {}))
        custom_species = list(ckpt.get("custom_species", []))

        if species_name is None:
            keep_indices = list(range(min(base_n, len(class_names))))
            removed = class_names[base_n:]
            custom_species = []
        else:
            if species_name not in custom_species:
                return {"success": False, "error": f"'{species_name}' is not a custom-added species."}
            keep_indices = [i for i, n in enumerate(class_names) if n != species_name]
            removed = [species_name]
            custom_species = [n for n in custom_species if n != species_name]

        new_class_names = [class_names[i] for i in keep_indices]
        new_prototypes = prototypes[keep_indices].cpu()
        new_subcenters = {new_i: subcenters[old_i] for new_i, old_i in enumerate(keep_indices) if old_i in subcenters}

        ckpt["class_names"] = new_class_names
        ckpt["prototypes"] = new_prototypes
        ckpt["subcenters"] = new_subcenters
        ckpt["custom_species"] = custom_species
        torch.save(ckpt, proto_path)
        classifier.reload_prototypes()

        return {
            "success": True,
            "removed": removed,
            "total_species": len(new_class_names),
            "custom_species": custom_species,
        }

    def sync_prototypes_online(self, classifier, ema_alpha: float = 0.15) -> dict:
        """
        Adapts existing species prototypes on the hypersphere using verified expert feedback.
        """
        proto_path = self._proto_path()
        if not os.path.exists(proto_path):
            return {"success": False, "error": "prototypes.pth not found"}

        log_jsonl = os.path.join(self.active_learning_dir, "feedback_log.jsonl")
        if not os.path.exists(log_jsonl):
            return {"success": False, "error": "No verified feedback recorded yet."}

        ckpt = torch.load(proto_path, map_location=self.device)
        prototypes = ckpt["prototypes"].to(self.device)
        class_names = list(ckpt["class_names"])
        class_to_idx = {name: i for i, name in enumerate(class_names)}

        updated_species = set()
        count = 0

        with open(log_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("status") == "new_species":
                        continue
                    sp = entry.get("confirmed_species")
                    img_filename = entry.get("sample_filename")
                    if not sp or not img_filename or sp not in class_to_idx:
                        continue

                    img_path = os.path.join(self.verified_samples_dir, img_filename)
                    if not os.path.exists(img_path):
                        continue

                    pil_img = load_rgb_image(img_path)
                    emb = classifier.extract_embedding(pil_img, use_tta=True)

                    c_idx = class_to_idx[sp]
                    old_proto = prototypes[c_idx : c_idx + 1]
                    new_proto = F.normalize((1.0 - ema_alpha) * old_proto + ema_alpha * emb, p=2, dim=1)

                    prototypes[c_idx : c_idx + 1] = new_proto
                    updated_species.add(sp)
                    count += 1
                except Exception as e:
                    print(f"[Sync Warning] {e}")

        ckpt["prototypes"] = prototypes.cpu()
        torch.save(ckpt, proto_path)
        classifier.reload_prototypes()

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(os.path.join(self.active_learning_dir, "sync_state.json"), "w") as sf:
            json.dump({"last_synced_at": now_str, "synced_samples": count}, sf)

        return {
            "success": True,
            "synced_samples": count,
            "updated_species_count": len(updated_species),
            "message": f"Synchronized {count} verified samples across {len(updated_species)} species.",
        }
