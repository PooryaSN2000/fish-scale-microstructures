import os
import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import v2

import config
from model import ProtoNet
from features import extract_classical_features
from dataset import apply_clahe_preprocessing
from scale_detector import ScaleDetector
from gradcam import GradCAMExplainer5Class


STAGE_LABELS = [
    ("Lutjanus johni", "Stage 1 (S1)"),
    ("Lutjanus lutjanus", "Stage 2 (S2)"),
    ("Lutjanus lemniscatus", "Stage 3 (S3)"),
    ("Lutjanus malabaricus", "Stage 4 (S4)"),
    ("Lutjanus fulviflammus", "Stage 5 (S5)"),
]


def load_rgb_image(path: str) -> Image.Image:
    """Loads RGB micrographs safely, handling 16-bit/grayscale TIFFs cleanly."""
    with Image.open(path) as raw_img:
        if raw_img.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
            arr = np.array(raw_img, dtype=np.float32)
            mx = float(arr.max()) if arr.size else 1.0
            arr = (arr / mx * 255.0).clip(0, 255).astype(np.uint8) if mx > 0 else np.zeros_like(arr, dtype=np.uint8)
            return Image.fromarray(arr).convert("RGB")
        return raw_img.convert("RGB")


def _stage_label_for(cname: str) -> str:
    low = cname.lower()
    if "johni" in low:
        return "Stage 1 (S1)"
    if "lutjanus lutjanus" in low:
        return "Stage 2 (S2)"
    if "lemniscatus" in low:
        return "Stage 3 (S3)"
    if "malabari" in low:
        return "Stage 4 (S4)"
    if "fulviflammus" in low:
        return "Stage 5 (S5)"
    return "Unknown"


def _arr_to_b64(img_pil: Image.Image) -> str:
    tmp = img_pil.copy()
    tmp.thumbnail((800, 800))
    buf = io.BytesIO()
    tmp.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


class FishClassifier:
    def __init__(self, model_path="checkpoints/best_model.pth", prototypes_path="checkpoints/prototypes.pth"):
        self.device = config.DEVICE

        base_ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
        if not os.path.exists(model_path) and os.path.exists(os.path.join(base_ckpt_dir, os.path.basename(model_path))):
            model_path = os.path.join(base_ckpt_dir, os.path.basename(model_path))
        if not os.path.exists(prototypes_path) and os.path.exists(os.path.join(base_ckpt_dir, os.path.basename(prototypes_path))):
            prototypes_path = os.path.join(base_ckpt_dir, os.path.basename(prototypes_path))

        checkpoint = torch.load(model_path, map_location=self.device)
        model_config = checkpoint["config"]

        self.use_clahe = model_config.get("use_clahe", config.USE_CLAHE)
        self.use_roi_crop = model_config.get("use_roi_crop", getattr(config, "USE_ROI_CROP", True))
        self.use_tta = model_config.get("use_tta", config.USE_TTA)
        self.use_classical = model_config.get("use_classical", False)
        self.use_mixstyle = model_config.get("use_mixstyle", False)
        self.image_size = tuple(model_config.get("image_size", config.IMAGE_SIZE))
        self.detector = ScaleDetector(padding_ratio=0.12) if self.use_roi_crop else None

        self.model = ProtoNet(
            backbone_name=model_config["backbone"],
            use_pretrained=False,
            embedding_dim=model_config["embedding_dim"],
            use_classical=self.use_classical,
            classical_in_dim=model_config.get("classical_in_dim", 0),
            classical_proj_dim=model_config.get("classical_proj_dim", config.CLASSICAL_PROJ_DIM),
            use_mixstyle=self.use_mixstyle,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes not found at {prototypes_path}. Please run save_prototypes.py first.")

        proto_data = torch.load(prototypes_path, map_location=self.device)
        self.prototypes = F.normalize(proto_data["prototypes"].to(self.device), p=2, dim=-1)
        self.class_names = proto_data["class_names"]
        self.means_768 = proto_data["means_768"].cpu().numpy() if "means_768" in proto_data else None
        self.prec_768 = proto_data["prec_768"].cpu().numpy() if "prec_768" in proto_data else None
        self.ood_stats = proto_data.get("ood_stats", None)
        self.gradcam = GradCAMExplainer5Class(self.model, self.device, image_size=self.image_size)

        self.transform = v2.Compose([
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_embedding(self, pil_img: Image.Image, apply_roi_crop: bool = None, return_backbone: bool = False):
        crop_flag = self.use_roi_crop if apply_roi_crop is None else apply_roi_crop
        if crop_flag and self.detector is not None:
            pil_img = self.detector.crop(pil_img, padding_ratio=0.12, square=True)
        if self.use_clahe:
            pil_img = apply_clahe_preprocessing(pil_img)

        if self.use_tta and not self.use_classical:
            views = [
                pil_img,
                pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
                pil_img.rotate(90),
            ]
        else:
            views = [pil_img]

        batch = torch.stack([self.transform(v) for v in views]).to(self.device)
        cf_tensor = None
        if self.use_classical:
            cfs = [extract_classical_features(v) for v in views]
            cf_tensor = torch.tensor(cfs, dtype=torch.float32).to(self.device)

        feats_768 = F.normalize(self.model.extract_backbone_features(batch), p=2, dim=-1)
        mean_768 = F.normalize(feats_768.mean(dim=0, keepdim=True), p=2, dim=-1)

        emb = self.model(batch, cf_tensor, stage=2)
        emb_norm = F.normalize(emb, p=2, dim=-1)
        mean_emb = F.normalize(emb_norm.mean(dim=0, keepdim=True), p=2, dim=-1)
        if return_backbone:
            return mean_emb, mean_768
        return mean_emb

    def compute_ood_metrics(self, max_sim: float, margin: float, feat_768: torch.Tensor) -> dict:
        if self.means_768 is None or self.prec_768 is None or self.ood_stats is None:
            score = max(0.0, min(100.0, max_sim * 100.0))
            membership = "in_group" if score >= 75.0 else ("borderline" if score >= 60.0 else "out_of_group")
            return {"maha_dist": None, "in_domain_score": round(score, 1), "membership": membership}

        f_np = feat_768.cpu().numpy()
        diffs = f_np - self.means_768
        maha = float(np.sqrt(np.min(np.sum((diffs @ self.prec_768) * diffs, axis=1))))

        st = self.ood_stats
        z_maha = -(maha - st["maha_mean"]) / st["maha_std"]
        z_sim = (max_sim - st["sim_mean"]) / st["sim_std"]
        z_mar = (margin - st["margin_mean"]) / st["margin_std"]
        fused = 0.55 * z_maha + 0.35 * z_sim + 0.10 * z_mar

        # Map fused z-score smoothly onto calibrated 0..100% in-domain scale
        z_rel = (fused - st["fused_p05"]) / max(st["fused_std"], 0.5)
        in_domain_score = float(100.0 / (1.0 + np.exp(-0.9 * (z_rel + 0.8))))

        if max_sim >= 0.70 and maha <= st["maha_p95"] and fused >= st["fused_p10"]:
            membership = "in_group"
        elif max_sim >= 0.55 and maha <= st["maha_p95"] * 1.18 and fused >= st["fused_p05"] - 1.5:
            membership = "borderline"
        else:
            membership = "out_of_group"

        return {
            "maha_dist": round(maha, 2),
            "maha_p95_cutoff": round(st["maha_p95"], 2),
            "fused_ood_z": round(float(fused), 3),
            "in_domain_score": round(in_domain_score, 1),
            "membership": membership,
        }

    def predict(self, image_path: str):
        image = load_rgb_image(image_path)
        emb_norm = self.extract_embedding(image)

        similarities = torch.matmul(emb_norm, self.prototypes.t())
        max_raw_sim = torch.max(similarities).item()
        all_similarities = dict(zip(self.class_names, similarities.squeeze(0).tolist()))

        logits = similarities * self.model.temperature if self.model.temperature is not None else similarities
        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        return self.class_names[pred_idx], probs[pred_idx].item(), dict(zip(self.class_names, probs.tolist())), max_raw_sim, all_similarities

    def predict_detailed(self, image_path: str, return_visuals: bool = True) -> dict:
        orig_img = load_rgb_image(image_path)
        proc_img = orig_img
        hud_b64, gradcam_b64 = None, None
        preview_b64 = _arr_to_b64(orig_img) if return_visuals else None

        if self.use_roi_crop and self.detector is not None:
            x1, y1, x2, y2 = self.detector.detect_bbox(orig_img, padding_ratio=0.12, square=True)
            proc_img = orig_img.crop((x1, y1, x2, y2))
            if return_visuals:
                hud_img = orig_img.copy()
                draw = ImageDraw.Draw(hud_img)
                lw = max(4, int(min(hud_img.size) * 0.008))
                draw.rectangle([x1, y1, x2, y2], outline=(6, 182, 212), width=lw)
                hud_b64 = _arr_to_b64(hud_img)

        with torch.no_grad():
            emb_norm, feat_768 = self.extract_embedding(proc_img, apply_roi_crop=False, return_backbone=True)
            similarities = torch.matmul(emb_norm, self.prototypes.t()).squeeze(0)
            logits = similarities * self.model.temperature if self.model.temperature is not None else similarities
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            sims_np = similarities.cpu().numpy()

        ranked = np.argsort(sims_np)[::-1]
        pred_idx = int(ranked[0])
        max_sim = float(sims_np[pred_idx])
        second_sim = float(sims_np[ranked[1]]) if len(ranked) > 1 else 0.0
        margin = max_sim - second_sim

        ood = self.compute_ood_metrics(max_sim, margin, feat_768)

        if return_visuals:
            try:
                clahe_roi = apply_clahe_preprocessing(proc_img) if self.use_clahe else proc_img
                _, _, overlay_arr = self.gradcam.generate(clahe_roi, self.prototypes[pred_idx])
                gradcam_b64 = _arr_to_b64(Image.fromarray(overlay_arr))
            except Exception as e:
                print(f"[GradCAM Warning] {e}")

        top3_candidates = []
        top3_cum_prob = 0.0
        for rank_i, c_idx in enumerate(ranked[:3], start=1):
            sp = self.class_names[int(c_idx)]
            pr = round(float(probs[int(c_idx)]) * 100.0, 2)
            top3_cum_prob += float(probs[int(c_idx)]) * 100.0
            top3_candidates.append({
                "rank": rank_i,
                "stage": _stage_label_for(sp),
                "species_name": sp,
                "similarity": round(max(0.0, float(sims_np[int(c_idx)])) * 100.0, 1),
                "probability": pr,
            })

        return {
            "prediction": self.class_names[pred_idx],
            "prediction_stage": _stage_label_for(self.class_names[pred_idx]),
            "confidence": round(float(probs[pred_idx]) * 100.0, 2),
            "overall_similarity": round(max(0.0, max_sim) * 100.0, 1),
            "margin_pct": round(margin * 100.0, 1),
            "top3_candidates": top3_candidates,
            "top3_cumulative_confidence": round(min(100.0, top3_cum_prob), 1),
            "similarities": {self.class_names[i]: round(max(0.0, float(sims_np[i])) * 100.0, 1) for i in range(len(self.class_names))},
            "all_probabilities": {self.class_names[i]: round(float(probs[i]) * 100.0, 2) for i in range(len(self.class_names))},
            "preview_image": preview_b64,
            "crop_overlay_image": hud_b64,
            "gradcam_image": gradcam_b64,
            **ood,
        }
