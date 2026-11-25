import os
import io
import time
import base64
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from . import config
    from .model import FishScaleConvNeXtV4
    from .scale_detector import ScaleDetector
    from .gradcam import GradCAMExplainer
    from .dataset import apply_clahe_preprocessing, load_rgb_image
except ImportError:
    import config
    from model import FishScaleConvNeXtV4
    from scale_detector import ScaleDetector
    from gradcam import GradCAMExplainer
    from dataset import apply_clahe_preprocessing, load_rgb_image


class FishClassifierV4:
    """
    Production Inference & Open-Set Registry Engine for AquaLens AI v4.1.
    Unifies ScaleDetector ROI isolation, CLAHE contrast enhancement, 7-View Spherical-TTA,
    multi-subcenter prototype matching, Grad-CAM explainability, and runtime novel species registration.
    """
    def __init__(self, model_path: str = None, prototypes_path: str = None):
        self.device = config.DEVICE

        if model_path is None:
            model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        if prototypes_path is None:
            prototypes_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")

        self.model_path = model_path
        self.prototypes_path = prototypes_path

        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes checkpoint not found at: {prototypes_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        self.reload_prototypes()

        self.detector = ScaleDetector(
            mode="adaptive",
            padding_ratio=getattr(config, "ROI_PADDING_RATIO", 0.12),
        )

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.CenterCrop(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.model = FishScaleConvNeXtV4(
            embedding_dim=self.embedding_dim,
            use_mixstyle=False,
        ).to(self.device)

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        target_layer = self.model.backbone.features[-1]
        self.grad_cam = GradCAMExplainer(self.model, target_layer)

        print(f"[FishClassifierV4] Loaded {len(self.class_names)} species prototypes (ROI_Crop={self.use_roi_crop}, CLAHE={self.use_clahe}, TTA={self.use_tta}).")

    def reload_prototypes(self):
        proto_dict = torch.load(self.prototypes_path, map_location=self.device)
        self.class_names = list(proto_dict["class_names"])
        self.prototypes = F.normalize(proto_dict["prototypes"].to(self.device), p=2, dim=1)
        self.subcenters = proto_dict.get("subcenters", {})
        self.custom_species = set(proto_dict.get("custom_species", []))
        self.base_class_count = int(proto_dict.get("base_class_count", len(self.class_names)))
        self.image_size = tuple(proto_dict.get("image_size", config.IMAGE_SIZE))
        self.embedding_dim = int(proto_dict.get("embedding_dim", config.EMBEDDING_DIM))
        self.use_roi_crop = bool(proto_dict.get("use_roi_crop", getattr(config, "USE_ROI_CROP", True)))
        self.roi_padding_ratio = float(proto_dict.get("roi_padding_ratio", getattr(config, "ROI_PADDING_RATIO", 0.12)))
        self.use_clahe = bool(proto_dict.get("use_clahe", config.USE_CLAHE))
        self.use_tta = bool(proto_dict.get("use_tta", getattr(config, "USE_SPHERICAL_TTA", True)))

    def _generate_tta_views(self, pil_img: Image.Image) -> list:
        w, h = pil_img.size
        crop_box = (int(0.03 * w), int(0.03 * h), int(0.97 * w), int(0.97 * h))
        zoomed = pil_img.crop(crop_box).resize(pil_img.size, Image.Resampling.BILINEAR)
        return [
            self.transform(pil_img),
            self.transform(pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)),
            self.transform(pil_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)),
            self.transform(pil_img.rotate(180)),
            self.transform(pil_img.rotate(90)),
            self.transform(pil_img.rotate(270)),
            self.transform(zoomed),
        ]

    @torch.no_grad()
    def extract_embedding(
        self,
        pil_img: Image.Image,
        use_tta: bool = True,
        apply_roi_crop: bool = None,
        apply_clahe: bool = None,
    ) -> torch.Tensor:
        """
        Preprocesses an arbitrary PIL micrograph through optional ScaleDetector ROI crop
        and CLAHE contrast enhancement, then extracts an L2-normalized embedding.
        """
        crop_flag = self.use_roi_crop if apply_roi_crop is None else apply_roi_crop
        clahe_flag = self.use_clahe if apply_clahe is None else apply_clahe

        proc = pil_img.convert("RGB")
        if crop_flag and self.detector is not None:
            proc = self.detector.crop(proc, padding_ratio=self.roi_padding_ratio, square=True)
        if clahe_flag:
            proc = apply_clahe_preprocessing(proc)

        if use_tta and self.use_tta:
            views = self._generate_tta_views(proc)
            batch = torch.stack(views).to(self.device)
            emb_views = self.model(batch)
            mean_emb = emb_views.sum(dim=0, keepdim=True)
            return F.normalize(mean_emb, p=2, dim=1)
        inp = self.transform(proc).unsqueeze(0).to(self.device)
        return F.normalize(self.model(inp), p=2, dim=1)

    def score_embedding(self, query_emb: torch.Tensor) -> np.ndarray:
        sim_proto = torch.mm(query_emb, self.prototypes.t()).squeeze(0)
        if self.subcenters:
            max_sims = []
            for c in range(len(self.class_names)):
                proto_val = float(sim_proto[c].item())
                subs = self.subcenters.get(c, None)
                if subs is not None:
                    sub_val = float(torch.mm(query_emb, subs.to(self.device).t()).max().item())
                    max_sims.append(max(proto_val, sub_val))
                else:
                    max_sims.append(proto_val)
            return np.array(max_sims, dtype=np.float32)
        return sim_proto.cpu().numpy()

    def predict(self, image_path: str, auto_crop: bool = None, return_metadata: bool = True):
        """
        Runs biometric identification on an image path (.jpg/.png/.bmp/.tif/.tiff).
        """
        start_time = time.time()
        orig_img = load_rgb_image(image_path)
        crop_flag = self.use_roi_crop if auto_crop is None else auto_crop

        crop_overlay_b64 = None
        processed_img = orig_img

        if crop_flag:
            try:
                x1, y1, x2, y2 = self.detector.detect_bbox(
                    orig_img, padding_ratio=self.roi_padding_ratio, square=True
                )
                cropped = orig_img.crop((x1, y1, x2, y2))
                if cropped.size[0] > 40 and cropped.size[1] > 40:
                    processed_img = cropped
                    crop_overlay_b64 = self.detector.visualize_detection_base64(
                        orig_img, bbox=(x1, y1, x2, y2)
                    )
            except Exception as e:
                print(f"[Warning] Scale detector fallback: {e}")
                processed_img = orig_img

        # We already cropped above if crop_flag is True, so skip double cropping in extract_embedding
        query_emb = self.extract_embedding(
            processed_img, use_tta=True, apply_roi_crop=False, apply_clahe=self.use_clahe
        )

        sims_np = self.score_embedding(query_emb)
        sorted_indices = np.argsort(sims_np)[::-1]
        top_idx = int(sorted_indices[0])
        pred_class = self.class_names[top_idx]
        max_sim = float(sims_np[top_idx])

        second_sim = float(sims_np[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = max_sim - second_sim

        all_sims = {self.class_names[i]: float(sims_np[i]) for i in range(len(self.class_names))}
        scaled_sims = torch.tensor(sims_np, device=self.device) * 18.0
        probs = F.softmax(scaled_sims, dim=0).cpu().numpy()
        all_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        confidence = float(probs[top_idx])
        exec_time = time.time() - start_time
        n_total = len(self.class_names)

        is_uncertain = (max_sim < 0.78) or (margin < 0.04)
        uncertainty_reason = ""
        if max_sim < 0.78:
            uncertainty_reason = (
                f"حداکثر تشابه ساختاری پایین است ({max_sim*100:.1f}٪ < آستانه ۷۸٪). "
                f"این نمونه احتمالاً متعلق به گونه‌ای جدید خارج از {n_total} گونه ثبت‌شده است "
                f"(می‌توانید همین نمونه را مستقیماً به عنوان گونه جدید ثبت کنید)."
            )
        elif margin < 0.04:
            second_class = self.class_names[int(sorted_indices[1])]
            uncertainty_reason = (
                f"ابهام نزدیک بین دو گونه: «{pred_class}» و «{second_class}» "
                f"(اختلاف تشابه {margin*100:.1f}٪)."
            )

        meta = {
            "execution_time": f"{exec_time:.2f}s",
            "margin": round(margin * 100, 1),
            "is_uncertain": is_uncertain,
            "is_custom_species": pred_class in self.custom_species,
            "uncertainty_reason": uncertainty_reason,
            "crop_overlay_image": crop_overlay_b64,
        }

        if return_metadata:
            return pred_class, confidence, all_probs, max_sim, all_sims, meta
        return pred_class, confidence

    def explain(self, image_input, auto_crop: bool = None) -> str:
        try:
            if isinstance(image_input, str):
                orig_img = load_rgb_image(image_input)
            else:
                orig_img = image_input

            crop_flag = self.use_roi_crop if auto_crop is None else auto_crop
            target_img = orig_img
            if crop_flag:
                try:
                    x1, y1, x2, y2 = self.detector.detect_bbox(
                        orig_img, padding_ratio=self.roi_padding_ratio, square=True
                    )
                    cropped = orig_img.crop((x1, y1, x2, y2))
                    if cropped.size[0] > 40 and cropped.size[1] > 40:
                        target_img = cropped
                except Exception as e:
                    print(f"[GradCAM Fallback] Auto-crop fallback: {e}")
                    target_img = orig_img

            return self.grad_cam.get_base64_overlay(target_img, prototypes=self.prototypes)
        except Exception as e:
            print(f"[GradCAM Warning] {e}")
            return None


FishClassifier = FishClassifierV4
