import numpy as np
import torch


def fourier_amplitude_swap(source_tensor: torch.Tensor, target_tensor: torch.Tensor, beta: float = 0.08) -> torch.Tensor:
    """
    Fourier Domain Augmentation (FDA): swaps the low-frequency amplitude spectrum
    between images while keeping 100% of the high-frequency phase spectrum
    (circuli, radii, and ctenii geometry intact).
    """
    fft_src = torch.fft.fftshift(torch.fft.fft2(source_tensor, dim=(-2, -1)), dim=(-2, -1))
    fft_tgt = torch.fft.fftshift(torch.fft.fft2(target_tensor, dim=(-2, -1)), dim=(-2, -1))

    amp_src = torch.abs(fft_src)
    amp_tgt = torch.abs(fft_tgt)
    phase_src = torch.angle(fft_src)

    _, H, W = source_tensor.shape
    bh, bw = max(1, int(np.floor(H * beta))), max(1, int(np.floor(W * beta)))
    ch, cw = H // 2, W // 2
    amp_src[:, ch - bh:ch + bh, cw - bw:cw + bw] = amp_tgt[:, ch - bh:ch + bh, cw - bw:cw + bw]

    recombined = amp_src * torch.exp(1j * phase_src)
    out = torch.fft.ifft2(torch.fft.ifftshift(recombined, dim=(-2, -1)), dim=(-2, -1)).real
    return out.clamp(0.0, 1.0)
