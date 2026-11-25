import random
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageFilter
from skimage import exposure

def fourier_amplitude_swap(source_tensor, target_tensor, beta=0.08):
    """
    Fourier Domain Augmentation (FDA).
    Swaps the low-frequency amplitude spectrum of source_tensor with target_tensor,
    while completely preserving the phase spectrum (which contains the biological scale microstructures).
    
    Args:
        source_tensor (torch.Tensor): (C, H, W) normalized tensor
        target_tensor (torch.Tensor): (C, H, W) normalized tensor
        beta (float): Radius fraction of the central low-frequency window [0.01, 0.15]
    Returns:
        torch.Tensor: Augmented image tensor with target's camera/sensor style
    """
    # Compute 2D Fast Fourier Transform
    fft_src = torch.fft.fft2(source_tensor, dim=(-2, -1))
    fft_tgt = torch.fft.fft2(target_tensor, dim=(-2, -1))

    # Shift zero-frequency components to center
    fft_src_shift = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_tgt_shift = torch.fft.fftshift(fft_tgt, dim=(-2, -1))

    # Decompose into Amplitude and Phase
    amp_src = torch.abs(fft_src_shift)
    amp_tgt = torch.abs(fft_tgt_shift)
    phase_src = torch.angle(fft_src_shift)

    _, H, W = source_tensor.shape
    b_h = int(np.floor(H * beta))
    b_w = int(np.floor(W * beta))

    c_h, c_w = H // 2, W // 2
    h1, h2 = max(0, c_h - b_h), min(H, c_h + b_h)
    w1, w2 = max(0, c_w - b_w), min(W, c_w + b_w)

    # Swap low-frequency amplitude window
    amp_src[:, h1:h2, w1:w2] = amp_tgt[:, h1:h2, w1:w2]

    # Reconstruct complex spectrum with original phase
    fft_recombined = amp_src * torch.exp(1j * phase_src)
    fft_unshift = torch.fft.ifftshift(fft_recombined, dim=(-2, -1))
    reconstructed = torch.fft.ifft2(fft_unshift, dim=(-2, -1)).real

    return reconstructed.clamp(0.0, 1.0)


class CrossCameraAugmentation:
    """
    Comprehensive sensor and optical augmentation suite simulating diverse 
    microscopes, objective lenses, and digital camera sensors.
    """
    def __init__(self, p_blur=0.3, p_color=0.5, p_gamma=0.3):
        self.p_blur = p_blur
        self.p_color = p_color
        self.p_gamma = p_gamma

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Input scale image
        Returns:
            PIL.Image: Stylistically diversified image
        """
        # 1. Random optical defocus / slight blur
        if random.random() < self.p_blur:
            radius = random.uniform(0.3, 1.2)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        # 2. Random color temperature & white-balance shift (warm vs cold lighting)
        if random.random() < self.p_color:
            # Color jittering
            color_factor = random.uniform(0.75, 1.25)
            contrast_factor = random.uniform(0.85, 1.25)
            brightness_factor = random.uniform(0.85, 1.15)
            
            img = ImageEnhance.Color(img).enhance(color_factor)
            img = ImageEnhance.Contrast(img).enhance(contrast_factor)
            img = ImageEnhance.Brightness(img).enhance(brightness_factor)

        # 3. Random Gamma curve (simulates CMOS vs CCD sensor non-linear dynamic range)
        if random.random() < self.p_gamma:
            gamma = random.uniform(0.8, 1.25)
            arr = np.array(img, dtype=np.float32) / 255.0
            arr = np.power(arr, gamma)
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(arr)

        return img


def get_v4_transforms(image_size=(336, 336), is_train=True):
    """
    Constructs PyTorch transforms for v4, featuring spatial invariance 
    and sensor-agnostic representations.
    """
    if is_train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            CrossCameraAugmentation(p_blur=0.25, p_color=0.5, p_gamma=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
