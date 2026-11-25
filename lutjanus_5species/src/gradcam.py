import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
import matplotlib.cm as cm

import config


class GradCAMExplainer5Class:
    """
    Grad-CAM visual interpretability for the ConvNeXt-Tiny Prototypical Network.
    Computes spatial activation heatmaps showing which morphological scale structures
    (circuli, radii, ctenii) drive cosine similarity to the target Stage prototype.
    """
    def __init__(self, model, device, image_size=(336, 336)):
        self.model = model
        self.device = device
        self.image_size = image_size
        self.activations = None
        self.gradients = None
        self.target_layer = self.model.backbone[-1]

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

        self.transform = v2.Compose([
            v2.Resize(self.image_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, pil_img: Image.Image, target_proto: torch.Tensor):
        """
        Generates (raw_rgb, heatmap_rgb, overlay_rgb) numpy uint8 arrays at image_size.
        """
        self.model.eval()
        self.model.zero_grad()

        inp = self.transform(pil_img).unsqueeze(0).to(self.device)
        inp.requires_grad_(True)

        emb = self.model(inp, stage=2)
        emb_norm = F.normalize(emb, p=2, dim=-1)
        proto_norm = F.normalize(target_proto.view(1, -1).to(self.device), p=2, dim=-1)
        score = torch.sum(emb_norm * proto_norm)
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=self.image_size, mode="bilinear", align_corners=False)
        cam_np = cam.squeeze().cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)

        heatmap = (cm.jet(cam_np)[:, :, :3] * 255).astype(np.uint8)
        raw_resized = np.array(pil_img.resize(self.image_size, Image.Resampling.BILINEAR))
        overlay = np.clip(0.55 * raw_resized + 0.45 * heatmap, 0, 255).astype(np.uint8)
        return raw_resized, heatmap, overlay
