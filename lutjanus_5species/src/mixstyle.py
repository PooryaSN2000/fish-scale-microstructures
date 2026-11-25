import random
import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """
    Probabilistic MixStyle domain generalization module (Zhou et al., ICLR 2021).
    Perturbs early channel-wise feature statistics (mean/std) across mini-batch samples
    so the backbone ignores camera illumination, sensor cast, and background contrast
    while preserving microstructural scale geometry.
    """
    def __init__(self, p: float = 0.5, alpha: float = 0.3, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or random.random() > self.p or x.size(0) <= 1:
            return x

        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_norm = (x - mu) / sig

        perm = torch.randperm(B, device=x.device)
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        mu_mix = mu * lmda + mu[perm] * (1.0 - lmda)
        sig_mix = sig * lmda + sig[perm] * (1.0 - lmda)
        return x_norm * sig_mix + mu_mix
