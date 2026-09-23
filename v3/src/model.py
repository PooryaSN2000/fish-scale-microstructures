import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import config


class ArcMarginProduct(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace).
    Forces intra-class compactness and inter-class angular separation.
    
    Formula: cos(theta + m) for target class, scaled by factor s.
    """
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35, easy_margin: bool = False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input_features: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        # Normalize features and class weight vectors to unit sphere
        cosine = F.linear(F.normalize(input_features, p=2, dim=-1), F.normalize(self.weight, p=2, dim=-1))
        
        # During inference without labels, return scaled cosine similarities directly
        if labels is None:
            return cosine * self.s
            
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        # One-hot target selection
        one_hot = torch.zeros(cosine.size(), device=input_features.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class FishArcNet(nn.Module):
    """
    High-accuracy ConvNeXt-Tiny with ArcFace Metric Learning Head for Marine Fish Scales.
    """
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        use_pretrained: bool = True,
        num_classes: int = 40,
        embedding_dim: int = 384,
        scale: float = 30.0,
        margin: float = 0.35
    ):
        super(FishArcNet, self).__init__()
        self.backbone_name = backbone_name
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        if backbone_name == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained else None
            convnext = models.convnext_tiny(weights=weights)
            self.backbone = convnext.features
            self.avgpool = convnext.avgpool
            in_features = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
            
        # Deep metric projection head
        self.projector = nn.Sequential(
            nn.Linear(in_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # ArcFace margin classification head
        self.arcface = ArcMarginProduct(
            in_features=embedding_dim,
            out_features=num_classes,
            s=scale,
            m=margin
        )

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized L2 unit embedding (B, embedding_dim) from input images.
        """
        feat = self.backbone(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        emb = self.projector(feat)
        return F.normalize(emb, p=2, dim=-1)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, normalized_embeddings).
        """
        emb = self.extract_embedding(x)
        logits = self.arcface(emb, labels)
        return logits, emb


# Compatibility alias
ProtoNet = FishArcNet
