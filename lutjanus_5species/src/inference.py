import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

import config
from model import ProtoNet
from features import extract_classical_features
from dataset import apply_clahe_preprocessing
from scale_detector import ScaleDetector


class FishClassifier:
    def __init__(self, model_path="checkpoints/best_model.pth", prototypes_path="checkpoints/prototypes.pth"):
        self.device = config.DEVICE

        # Auto-resolve relative paths if not found in current working directory
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

        self.transform = v2.Compose([
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def extract_embedding(self, pil_img: Image.Image, apply_roi_crop: bool = None) -> torch.Tensor:
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

        emb = self.model(batch, cf_tensor, stage=2)
        emb_norm = F.normalize(emb, p=2, dim=-1)
        return F.normalize(emb_norm.mean(dim=0, keepdim=True), p=2, dim=-1)

    @torch.no_grad()
    def predict(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        emb_norm = self.extract_embedding(image)

        similarities = torch.matmul(emb_norm, self.prototypes.t())
        max_raw_sim = torch.max(similarities).item()
        all_similarities = dict(zip(self.class_names, similarities.squeeze(0).tolist()))

        if self.model.temperature is not None:
            logits = similarities * self.model.temperature
        else:
            logits = similarities

        probs = F.softmax(logits, dim=-1).squeeze(0)
        pred_idx = torch.argmax(probs).item()
        return self.class_names[pred_idx], probs[pred_idx].item(), dict(zip(self.class_names, probs.tolist())), max_raw_sim, all_similarities
