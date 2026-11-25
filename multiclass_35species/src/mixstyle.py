import random
import torch
import torch.nn as nn

class MixStyle(nn.Module):
    """
    MixStyle module for Domain Generalization (Cross-Camera & Sensor Invariance).
    
    Reference:
        Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
        
    Concept:
        In deep feature representations, channel-wise mean and variance capture 
        style/domain information (such as microscope sensor color cast, illumination, 
        and camera optical variations). In contrast, the normalized features represent 
        content and semantic geometry (circuli, radii, and scale microstructures).
        
        MixStyle mixes the feature statistics of different images in the training batch, 
        forcing the network to learn camera-agnostic biological features.
    """
    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
            p (float): Probability of applying MixStyle on a given forward pass.
            alpha (float): Parameter of the symmetric Beta distribution Beta(alpha, alpha).
            eps (float): Epsilon for numerical stability during variance calculation.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x

        B = x.size(0)
        if B <= 1:
            return x

        # Compute instance statistics across spatial dimensions
        # Supports both 4D (B, C, H, W) and 3D (B, N, C) feature representations
        if x.dim() == 4:
            mu = x.mean(dim=[2, 3], keepdim=True)
            var = x.var(dim=[2, 3], keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            # Randomly permute the batch
            perm = torch.randperm(B, device=x.device)
            lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
            
            # Mix feature statistics
            mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
            sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)

            return x_normed * sig_mix + mu_mix

        elif x.dim() == 3:
            # (B, H*W, C) as in some Vision Transformers or channels-last formats
            mu = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True)
            sig = (var + self.eps).sqrt()
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig

            perm = torch.randperm(B, device=x.device)
            lmda = self.beta.sample((B, 1, 1)).to(x.device)

            mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
            sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)

            return x_normed * sig_mix + mu_mix

        return x

    def __repr__(self):
        return f"MixStyle(p={self.p}, beta_alpha={self.beta.concentration1}, eps={self.eps})"
