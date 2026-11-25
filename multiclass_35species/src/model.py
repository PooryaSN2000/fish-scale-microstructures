import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ConvNeXt_Tiny_Weights
try:
    from .mixstyle import MixStyle
except ImportError:
    from mixstyle import MixStyle

class FishScaleConvNeXtV4(nn.Module):
    """
    Next-Generation Metric Learning Model for Fish Scale Microstructure Identification (v4).
    
    Features:
    1. ConvNeXt-Tiny modern convolutional architecture with 7x7 depthwise separable kernels.
    2. Embedded MixStyle layers after Stage 1 and Stage 2 for cross-camera and sensor invariance.
    3. Multi-layer projection bottleneck with GELU activation and L2 hyperspherical normalization.
    """
    def __init__(self, embedding_dim=384, use_mixstyle=True, mixstyle_p=0.5, mixstyle_alpha=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_mixstyle = use_mixstyle

        # Load ImageNet-pretrained ConvNeXt-Tiny
        weights = ConvNeXt_Tiny_Weights.DEFAULT
        self.backbone = models.convnext_tiny(weights=weights)

        # MixStyle domain generalization modules
        if self.use_mixstyle:
            self.mixstyle1 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
            self.mixstyle2 = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        else:
            self.mixstyle1 = nn.Identity()
            self.mixstyle2 = nn.Identity()

        # In ConvNeXt-Tiny, backbone output channels = 768
        in_features = 768

        # Remove default classification head
        self.backbone.classifier = nn.Identity()

        # Hyperspherical Projection Head
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

    def extract_features(self, x):
        """
        Forward pass through ConvNeXt stages with interleaved MixStyle.
        """
        # Stem
        x = self.backbone.features[0](x)
        # Stage 1
        x = self.backbone.features[1](x)
        if self.use_mixstyle:
            x = self.mixstyle1(x)

        # Downsample 1 + Stage 2
        x = self.backbone.features[2](x)
        x = self.backbone.features[3](x)
        if self.use_mixstyle:
            x = self.mixstyle2(x)

        # Downsample 2 + Stage 3 (Deeps feature representations)
        x = self.backbone.features[4](x)
        x = self.backbone.features[5](x)

        # Downsample 3 + Stage 4
        x = self.backbone.features[6](x)
        x = self.backbone.features[7](x)

        # Global average pooling (B, 768, 1, 1) -> (B, 768)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x, return_unnormalized=False):
        """
        Args:
            x: Input images (B, 3, H, W)
            return_unnormalized (bool): Return raw features before L2 normalization
        Returns:
            torch.Tensor: Hyperspherical embeddings on unit sphere (B, embedding_dim)
        """
        feats = self.extract_features(x)
        proj = self.projector(feats)

        if return_unnormalized:
            return proj

        # Project onto unit hypersphere: ||z||_2 = 1.0
        normalized_embeddings = F.normalize(proj, p=2, dim=1)
        return normalized_embeddings
