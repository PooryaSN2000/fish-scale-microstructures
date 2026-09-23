import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

import config
from model import FishArcNet
from dataset import apply_clahe_preprocessing
from gradcam import GradCAMExplainer
from scale_detector import ScaleDetector


class FishClassifier:
    """
    High-precision Inference Engine for 40 Marine Fish Species using
    ConvNeXt-Tiny ArcFace Metric Learning, Sub-Center Prototypes,
    Test-Time Augmentation (TTA), and Grad-CAM Visual Explainability.
    """
    def __init__(self, model_path="checkpoints/best_model.pth", prototypes_path="checkpoints/prototypes.pth"):
        self.device = config.DEVICE
        
        # 1. Load Prototypes and Sub-Centers
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes not found at {prototypes_path}. Please run save_prototypes.py first.")
            
        proto_data = torch.load(prototypes_path, map_location=self.device)
        self.prototypes = proto_data["prototypes"].to(self.device)
        self.prototypes_norm = F.normalize(self.prototypes, p=2, dim=-1)
        
        if "subcenters" in proto_data:
            self.subcenters = proto_data["subcenters"].to(self.device)
            self.subcenters_norm = F.normalize(self.subcenters, p=2, dim=-1)
        else:
            self.subcenters = None
            self.subcenters_norm = None
            
        self.class_names = proto_data["class_names"]
        self.embedding_dim = proto_data.get("embedding_dim", config.EMBEDDING_DIM)
        self.image_size = proto_data.get("image_size", config.IMAGE_SIZE)
        
        # 2. Load Model Checkpoint
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = FishArcNet(
            backbone_name=config.BACKBONE,
            use_pretrained=False,
            num_classes=len(self.class_names),
            embedding_dim=self.embedding_dim,
            scale=config.ARCFACE_SCALE,
            margin=config.ARCFACE_MARGIN
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        
        # 3. Explainability Hook
        self.explainer = GradCAMExplainer(self.model, device=self.device)
        
        # 4. Scale Detector
        self.detector = ScaleDetector()
        
        # Base Transform
        self.base_transform = v2.Compose([
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _extract_single_embedding(self, pil_image: Image.Image) -> torch.Tensor:
        proc_img = pil_image.convert("RGB").resize(self.image_size)
        if getattr(config, "USE_CLAHE", False):
            proc_img = apply_clahe_preprocessing(proc_img)
        tensor_img = self.base_transform(proc_img).unsqueeze(0).to(self.device)
        return self.model.extract_embedding(tensor_img)

    @torch.no_grad()
    def predict(self, image_path: str, use_tta: bool = True, auto_crop: bool = False):
        image = Image.open(image_path).convert("RGB")
        if auto_crop:
            image = self.detector.crop(image)
        
        if use_tta:
            # 5-pass TTA: 4 rotation angles + 1 horizontal flip
            views = [
                image,
                image.transpose(Image.ROTATE_90),
                image.transpose(Image.ROTATE_180),
                image.transpose(Image.ROTATE_270),
                image.transpose(Image.FLIP_LEFT_RIGHT)
            ]
            embs = [self._extract_single_embedding(v) for v in views]
            avg_emb = torch.stack(embs, dim=0).mean(dim=0)
            embedding = F.normalize(avg_emb, p=2, dim=-1)
        else:
            embedding = self._extract_single_embedding(image)
            
        # Cosine Similarity against prototypes and sub-centers
        sim_proto = torch.matmul(embedding, self.prototypes_norm.t()).squeeze(0)  # (40,)
        if self.subcenters_norm is not None:
            sim_sub = torch.einsum('d,ckd->ck', embedding.squeeze(0), self.subcenters_norm)  # (40, 2)
            max_sub = torch.max(sim_sub, dim=-1).values  # (40,)
            similarities = torch.maximum(sim_proto, max_sub)
        else:
            similarities = sim_proto
        
        max_raw_sim = torch.max(similarities).item()
        all_similarities = dict(zip(self.class_names, similarities.tolist()))
        
        # Softmax probability distribution calibrated by temperature
        temperature = 15.0
        logits = similarities * temperature
        probs = F.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs).item()
        pred_class = self.class_names[pred_idx]
        confidence = probs[pred_idx].item()
        all_probs = dict(zip(self.class_names, probs.tolist()))
        
        return pred_class, confidence, all_probs, max_raw_sim, all_similarities

    def explain(self, image_path: str, target_class_idx: int = None) -> str:
        """
        Generates Grad-CAM base64 data URI for the uploaded image.
        """
        image = Image.open(image_path).convert("RGB")
        return self.explainer.get_base64_overlay(image, target_class_idx=target_class_idx)


if __name__ == "__main__":
    print("Inference module ready.")
