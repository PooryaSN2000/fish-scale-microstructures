import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedArcFace(nn.Module):
    """
    Class-Balanced Additive Angular Margin Loss (CB-ArcFace).
    
    Addresses the extreme long-tail / few-shot distribution in biological specimen datasets.
    Classes with fewer samples receive an adaptively larger angular margin m_c:
        m_c = m_base + delta_m * (1 - (N_c / N_max)^gamma)
        
    This forces minority classes to form tighter, hyper-compact clusters on the 
    hypersphere, preventing them from being swallowed by dominant species.
    """
    def __init__(self, in_features, num_classes, s=30.0, m_base=0.30, delta_m=0.25, gamma=0.25, class_counts=None):
        """
        Args:
            in_features (int): Dimensionality of input feature vectors (e.g. 384).
            num_classes (int): Number of target species (e.g. 57 or 40).
            s (float): Hypersphere radius / logit scale factor.
            m_base (float): Baseline angular margin for most populated species.
            delta_m (float): Maximum extra angular margin added to few-shot species.
            gamma (float): Non-linear curvature for class count weighting.
            class_counts (list or torch.Tensor, optional): Number of training samples per class.
        """
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = float(s)
        self.m_base = float(m_base)
        self.delta_m = float(delta_m)
        self.gamma = float(gamma)

        # Centroid weight vectors on the hypersphere: (num_classes, in_features)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)

        # Compute per-class adaptive margins
        self.register_buffer('margins', torch.full((num_classes,), self.m_base))
        if class_counts is not None:
            self.set_class_counts(class_counts)

    def set_class_counts(self, class_counts):
        """Calculates adaptive margin vector based on empirical sample frequencies."""
        if not isinstance(class_counts, torch.Tensor):
            class_counts = torch.tensor(class_counts, dtype=torch.float32)
        else:
            class_counts = class_counts.float()

        N_max = class_counts.max()
        # Normalized sample frequency ratio: (N_c / N_max)^gamma
        freq_ratio = (class_counts / (N_max + 1e-6)).clamp(min=1e-4, max=1.0).pow(self.gamma)
        
        # Adaptive margin formula: m_c = m_base + delta_m * (1 - freq_ratio)
        adaptive_margins = self.m_base + self.delta_m * (1.0 - freq_ratio)
        self.margins.copy_(adaptive_margins)
        print(f"[CB-ArcFace] Adaptive margins configured: Min={adaptive_margins.min():.3f} rad, Max={adaptive_margins.max():.3f} rad")

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: Normalized deep feature embeddings (B, in_features)
            labels: Ground truth class indices (B,)
        Returns:
            Cosine logits scaled by s with adaptive angular margin applied (B, num_classes)
        """
        # Normalize weights to lie on unit hypersphere
        w_norm = F.normalize(self.weight, p=2, dim=1)
        # Cosine similarity between embeddings and class centroids: cos(theta)
        cosine = F.linear(embeddings, w_norm).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Compute sine: sin(theta) = sqrt(1 - cos^2(theta))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(min=1e-7))

        # Retrieve adaptive margins for target classes in batch
        target_m = self.margins[labels].unsqueeze(1)  # (B, 1)
        cos_m = torch.cos(target_m)
        sin_m = torch.sin(target_m)

        # Addition formula: cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        phi = cosine * cos_m - sine * sin_m

        # Threshold to ensure monotonic decreasing behavior for theta + m > pi
        th = torch.cos(math.pi - target_m)
        mm = torch.sin(math.pi - target_m) * target_m
        phi = torch.where(cosine > th, phi, cosine - mm)

        # Construct one-hot mask for ground truth labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        # Apply margin only to target class logit
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output
