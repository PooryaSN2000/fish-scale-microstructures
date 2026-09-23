import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

import config
from dataset import apply_clahe_preprocessing


class GradCAMExplainer:
    """
    Grad-CAM visual explanation generator for FishArcNet (ConvNeXt-Tiny).
    Highlights the anatomical scale regions (circuli, radii, ctenii, focus)
    responsible for the model's species classification.
    """
    def __init__(self, model: torch.nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or config.DEVICE
        self.model.eval()
        
        self.features = []
        self.gradients = []
        
        # Hook target: the last stage block of ConvNeXt-Tiny
        target_layer = self.model.backbone[-1]
        target_layer.register_forward_hook(self._hook_features)
        target_layer.register_full_backward_hook(self._hook_gradients)

        self.transform = v2.Compose([
            v2.Resize(config.IMAGE_SIZE),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _hook_features(self, module, input, output):
        self.features.append(output)

    def _hook_gradients(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def generate_cam(self, pil_image: Image.Image, target_class_idx: int = None) -> tuple[np.ndarray, np.ndarray, Image.Image, int]:
        """
        Generate Grad-CAM heatmap and blended overlay.
        
        Returns:
            cam_norm: 2D numpy array [0, 1] of heatmap values
            overlay: 3D numpy array [0, 1] RGB of blended image
            overlay_pil: PIL.Image of the blended visualization
            target_class_idx: The class index explained
        """
        self.features.clear()
        self.gradients.clear()
        self.model.zero_grad()

        # Preprocess with CLAHE if enabled
        proc_img = pil_image.convert("RGB").resize(config.IMAGE_SIZE)
        if getattr(config, "USE_CLAHE", False):
            proc_img = apply_clahe_preprocessing(proc_img)

        tensor_img = self.transform(proc_img).unsqueeze(0).to(self.device)
        tensor_img.requires_grad = True

        logits, _ = self.model(tensor_img)
        
        if target_class_idx is None:
            target_class_idx = torch.argmax(logits, dim=-1).item()

        score = logits[0, target_class_idx]
        score.backward()

        if len(self.features) == 0 or len(self.gradients) == 0:
            raise RuntimeError("Failed to capture activations or gradients for Grad-CAM.")

        feats = self.features[0]   # Shape: (1, C, H', W')
        grads = self.gradients[0]  # Shape: (1, C, H', W')

        # Global average pooling over spatial dimensions
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feats, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=config.IMAGE_SIZE, mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().detach().numpy()

        # Min-max normalize
        cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        # Colormap mapping
        cmap = plt.get_cmap('jet')
        heatmap_rgb = cmap(cam_norm)[:, :, :3]

        # Blend with input image
        base_arr = np.array(proc_img, dtype=np.float32) / 255.0
        overlay_arr = 0.55 * base_arr + 0.45 * heatmap_rgb
        overlay_arr = np.clip(overlay_arr, 0.0, 1.0)
        overlay_uint8 = (overlay_arr * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_uint8)

        return cam_norm, overlay_arr, overlay_pil, target_class_idx

    def get_base64_overlay(self, pil_image: Image.Image, target_class_idx: int = None) -> str:
        """
        Returns the Grad-CAM blended overlay as a base64-encoded PNG data URI.
        """
        _, _, overlay_pil, _ = self.generate_cam(pil_image, target_class_idx)
        buffer = io.BytesIO()
        overlay_pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"
