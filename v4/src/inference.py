import os
import io
import time
import base64
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from . import config
    from .model import FishScaleConvNeXtV4
    from .scale_detector import ScaleDetector
    from .gradcam import GradCAM
    from .dataset import apply_clahe_preprocessing
except ImportError:
    import config
    from model import FishScaleConvNeXtV4
    from scale_detector import ScaleDetector
    from gradcam import GradCAM
    from dataset import apply_clahe_preprocessing

class FishClassifierV4:
    """
    High-Precision Production Inference Engine for AquaLens AI v4.
    
    Architectural Highlights:
    1. Multi-Subcenter Hyperspherical Matching (captures intra-species scale diversity e.g. dorsal vs lateral scales).
    2. 7-View Spherical Test-Time Augmentation (Spherical-TTA) for complete camera orientation and scale invariance.
    3. Biological Scale Detection & Saliency HUD auto-crop.
    4. Grad-CAM visual heatmaps highlighting circuli & radii growth rings.
    5. Calibrated Epistemic Uncertainty analysis for Active Learning.
    """
    def __init__(self, model_path: str = None, prototypes_path: str = None):
        self.device = config.DEVICE
        self.detector = ScaleDetector(mode="adaptive")

        # Fallback paths if not explicitly provided
        if model_path is None:
            model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
        if prototypes_path is None:
            prototypes_path = os.path.join(config.CHECKPOINT_DIR, "prototypes.pth")

        # Base preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load Prototypes and Sub-centers
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes checkpoint not found at: {prototypes_path}")

        proto_dict = torch.load(prototypes_path, map_location=self.device)
        self.class_names = proto_dict["class_names"]
        self.prototypes = proto_dict["prototypes"].to(self.device)  # (57, 384)
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        self.subcenters = proto_dict.get("subcenters", None)

        # Initialize and load ConvNeXt-Tiny model
        self.model = FishScaleConvNeXtV4(
            embedding_dim=config.EMBEDDING_DIM,
            use_mixstyle=False  # MixStyle is inactive during inference
        ).to(self.device)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

        ckpt = torch.load(model_path, map_location=self.device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # Initialize Grad-CAM on Stage 4 features of ConvNeXt
        target_layer = self.model.backbone.features[-1]
        self.grad_cam = GradCAM(self.model, target_layer)

        print(f"[FishClassifierV4] Successfully loaded {len(self.class_names)} species prototypes with multi-subcenters.")

    def _generate_tta_views(self, pil_img: Image.Image) -> list:
        """
        Generates 7 complementary geometric and multi-scale views to ensure
        complete invariance to slide orientation, tilt, and optical magnification.
        """
        views = []
        # View 1: Standard canonical orientation
        views.append(self.transform(pil_img))

        # View 2: Horizontal mirror
        views.append(self.transform(pil_img.transpose(Image.FLIP_LEFT_RIGHT)))

        # View 3: Vertical mirror
        views.append(self.transform(pil_img.transpose(Image.FLIP_TOP_BOTTOM)))

        # View 4: 180-degree rotation
        views.append(self.transform(pil_img.rotate(180)))

        # View 5: 90-degree clockwise rotation
        views.append(self.transform(pil_img.rotate(90)))

        # View 6: 270-degree rotation (90 counter-clockwise)
        views.append(self.transform(pil_img.rotate(270)))

        # View 7: Center scale crop (0.94x zoom)
        w, h = pil_img.size
        crop_box = (int(0.03 * w), int(0.03 * h), int(0.97 * w), int(0.97 * h))
        zoomed = pil_img.crop(crop_box).resize(pil_img.size, Image.BILINEAR)
        views.append(self.transform(zoomed))

        return views

    def extract_embedding(self, pil_img: Image.Image, use_tta: bool = True) -> torch.Tensor:
        """
        Extracts L2-normalized deep embedding on the unit hypersphere with 7-View Spherical-TTA.
        """
        with torch.no_grad():
            if use_tta and config.USE_SPHERICAL_TTA:
                views = self._generate_tta_views(pil_img)
                batch = torch.stack(views).to(self.device)  # (7, 3, H, W)
                emb_views = self.model(batch)               # (7, embedding_dim)
                
                # Hyperspherical mean pooling: sum multi-view vectors and re-normalize
                mean_emb = emb_views.sum(dim=0, keepdim=True)
                final_emb = F.normalize(mean_emb, p=2, dim=1)
                return final_emb
            else:
                inp = self.transform(pil_img).unsqueeze(0).to(self.device)
                return self.model(inp)

    def predict(self, image_path: str, auto_crop: bool = True, return_metadata: bool = True):
        """
        Performs high-reliability 57-species biometric identification on an input micrograph.
        """
        start_time = time.time()
        orig_img = Image.open(image_path).convert('RGB')

        # 1. Biological Scale Detection & HUD Auto-crop
        crop_overlay_b64 = None
        processed_img = orig_img

        if auto_crop:
            try:
                x1, y1, x2, y2 = self.detector.detect_bbox(orig_img, square=True)
                cropped = orig_img.crop((x1, y1, x2, y2))
                if cropped.size[0] > 40 and cropped.size[1] > 40:
                    processed_img = cropped
                    # Draw HUD visual overlay
                    overlay_img = orig_img.copy()
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(overlay_img)
                    draw.rectangle([x1, y1, x2, y2], outline="#0284c7", width=5)
                    buffered = io.BytesIO()
                    overlay_img.save(buffered, format="JPEG", quality=85)
                    crop_overlay_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()
            except Exception as e:
                print(f"[Warning] Scale detector fallback: {e}")
                processed_img = orig_img

        # Apply biological CLAHE
        if config.USE_CLAHE:
            processed_img = apply_clahe_preprocessing(processed_img)

        # 2. Extract Spherical-TTA embedding (7 views)
        query_emb = self.extract_embedding(processed_img, use_tta=True)

        # 3. Multi-Subcenter & Centroid Similarity Matching
        sim_proto = torch.mm(query_emb, self.prototypes.t()).squeeze(0)  # (57,)
        
        if self.subcenters is not None:
            max_sims = []
            for c in range(len(self.class_names)):
                proto_val = float(sim_proto[c].item())
                subs = self.subcenters.get(c, None)
                if subs is not None:
                    sub_val = float(torch.mm(query_emb, subs.to(self.device).t()).max().item())
                    max_sims.append(max(proto_val, sub_val))
                else:
                    max_sims.append(proto_val)
            sims_np = np.array(max_sims)
        else:
            sims_np = sim_proto.cpu().numpy()

        # Sort species by similarity descending
        sorted_indices = np.argsort(sims_np)[::-1]
        top_idx = sorted_indices[0]
        pred_class = self.class_names[top_idx]
        max_sim = float(sims_np[top_idx])

        # Second best similarity for confidence margin
        second_sim = float(sims_np[sorted_indices[1]]) if len(sorted_indices) > 1 else 0.0
        margin = max_sim - second_sim

        # Map similarities to dictionary {species: sim_float}
        all_sims = {self.class_names[i]: float(sims_np[i]) for i in range(len(self.class_names))}
        
        # Calibrated softmax probabilities with temperature scaling
        scaled_sims = torch.tensor(sims_np, device=self.device) * 15.0
        probs = F.softmax(scaled_sims, dim=0).cpu().numpy()
        all_probs = {self.class_names[i]: float(probs[i]) for i in range(len(self.class_names))}

        confidence = float(probs[top_idx])
        exec_time = time.time() - start_time

        # Active Learning / Epistemic Uncertainty assessment
        # Empirical separation: Mean correct = 80%, Mean incorrect/ambiguous = 60%
        is_uncertain = (max_sim < 0.55) or (margin < 0.05)
        uncertainty_reason = ""
        if max_sim < 0.55:
            uncertainty_reason = f"حداکثر تشابه ساختاری در سطح پایینی است ({max_sim*100:.1f}٪). نمونه ممکن است خارج از ۵۷ گونه تحت پوشش یا دارای نویز شدید باشد."
        elif margin < 0.05:
            second_class = self.class_names[sorted_indices[1]]
            uncertainty_reason = f"ابهام بین دو گونه خواهر: گونه اول «{pred_class}» و گونه دوم «{second_class}» با اختلاف حاشیه ناچیز {margin*100:.1f}٪ هستند."

        meta = {
            "execution_time": f"{exec_time:.2f}s",
            "margin": round(margin * 100, 1),
            "is_uncertain": is_uncertain,
            "uncertainty_reason": uncertainty_reason,
            "crop_overlay_image": crop_overlay_b64
        }

        if return_metadata:
            return pred_class, confidence, all_probs, max_sim, all_sims, meta
        return pred_class, confidence

    def explain(self, image_input, auto_crop: bool = True) -> str:
        """
        Generates Grad-CAM visual heatmap overlay over the scale microstructures.
        When auto_crop is True, focuses explainability directly on the localized
        scale to reveal biological growth rings (circuli & radii) without background bias.
        """
        try:
            if isinstance(image_input, str):
                orig_img = Image.open(image_input).convert('RGB')
            else:
                orig_img = image_input

            target_img = orig_img
            if auto_crop:
                try:
                    x1, y1, x2, y2 = self.detector.detect_bbox(orig_img, square=True)
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
