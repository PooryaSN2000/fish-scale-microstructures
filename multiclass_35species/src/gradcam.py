import base64
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms

try:
    from . import config
    from .dataset import apply_clahe_preprocessing
except ImportError:
    import config
    from dataset import apply_clahe_preprocessing


class GradCAMExplainer:
    """
    Grad-CAM visual explanation generator for FishScaleConvNeXtV4.
    Highlights biological microstructure regions (circuli, radii, ctenii, focus)
    responsible for model's species similarity identification.
    """
    def __init__(self, model: torch.nn.Module, target_layer=None, device: torch.device = None):
        self.model = model
        self.device = device or config.DEVICE
        self.model.eval()
        
        self.features = []
        self.gradients = []
        
        # Target layer: Stage 4 of ConvNeXt-Tiny
        if target_layer is None:
            if hasattr(self.model, 'backbone') and hasattr(self.model.backbone, 'features'):
                target_layer = self.model.backbone.features[-1]
            else:
                target_layer = list(self.model.children())[0]

        target_layer.register_forward_hook(self._hook_features)
        target_layer.register_full_backward_hook(self._hook_gradients)

        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _hook_features(self, module, input, output):
        self.features.append(output)

    def _hook_gradients(self, module, grad_in, grad_out):
        self.gradients.append(grad_out[0])

    def generate_cam(self, pil_image: Image.Image, prototypes: torch.Tensor = None, target_class_idx: int = None):
        """
        Generates Grad-CAM heatmap and blended overlay.
        """
        self.features.clear()
        self.gradients.clear()
        self.model.zero_grad()

        # Preprocess with CLAHE
        proc_img = pil_image.convert("RGB").resize(config.IMAGE_SIZE)
        if getattr(config, "USE_CLAHE", False):
            proc_img = apply_clahe_preprocessing(proc_img)

        tensor_img = self.transform(proc_img).unsqueeze(0).to(self.device)
        tensor_img.requires_grad = True

        emb = self.model(tensor_img)  # (1, 384)

        if prototypes is not None:
            sims = torch.mm(emb, prototypes.t()).squeeze(0)  # (num_classes,)
            if target_class_idx is None:
                target_class_idx = int(torch.argmax(sims).item())
            score = sims[target_class_idx]
        else:
            # Score is embedding magnitude
            score = emb.sum()

        score.backward()

        if len(self.features) == 0 or len(self.gradients) == 0:
            return None, None, None, None

        feats = self.features[0]   # (1, C, H', W')
        grads = self.gradients[0]  # (1, C, H', W')

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feats, dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=config.IMAGE_SIZE, mode='bilinear', align_corners=False)
        cam_np = cam.squeeze().cpu().detach().numpy()

        # Normalize [0, 1]
        cam_norm = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        # Colormap mapping (Jet / Turbo)
        cmap = plt.get_cmap('jet')
        heatmap_rgb = cmap(cam_norm)[:, :, :3]

        # Blend with input image
        base_arr = np.array(proc_img, dtype=np.float32) / 255.0
        overlay_arr = 0.55 * base_arr + 0.45 * heatmap_rgb
        overlay_arr = np.clip(overlay_arr, 0.0, 1.0)
        overlay_uint8 = (overlay_arr * 255).astype(np.uint8)
        overlay_pil = Image.fromarray(overlay_uint8)

        return cam_norm, overlay_arr, overlay_pil, target_class_idx

    def get_base64_overlay(self, pil_image: Image.Image, prototypes: torch.Tensor = None, target_class_idx: int = None) -> str:
        """
        Returns the Grad-CAM blended overlay as a base64-encoded JPEG data URI.
        """
        try:
            _, _, overlay_pil, _ = self.generate_cam(pil_image, prototypes, target_class_idx)
            if overlay_pil is None:
                return None
            buffer = io.BytesIO()
            overlay_pil.save(buffer, format="JPEG", quality=85)
            encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            print(f"[GradCAM Error] {e}")
            return None

# Alias for backwards compatibility
GradCAM = GradCAMExplainer
