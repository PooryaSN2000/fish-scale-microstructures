import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import filters, morphology, measure


class ScaleDetector:
    """
    Biological Microstructure Scale Localizer & Border-Artifact Rejector.
    Isolates the elasmoid scale body from calibration rulers, microscope
    timestamp banners, neighboring scale edges, and empty glass background.
    """
    def __init__(self, padding_ratio: float = 0.12):
        self.padding_ratio = padding_ratio

    def detect_bbox(self, pil_image: Image.Image, padding_ratio: float = None, square: bool = True) -> tuple[int, int, int, int]:
        if padding_ratio is None:
            padding_ratio = self.padding_ratio

        w, h = pil_image.size
        gray = np.array(pil_image.convert("L"), dtype=float)

        # 1. Sobel high-frequency texture saliency (circuli & radii vs smooth slide)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_smooth = ndimage.gaussian_filter(np.hypot(grad_x, grad_y), sigma=3.5)

        # 2. Corner-background luminance baseline
        ps = min(30, max(5, h // 15), max(5, w // 15))
        corners = np.concatenate([
            gray[:ps, :ps].ravel(),
            gray[:ps, -ps:].ravel(),
            gray[-ps:, :ps].ravel(),
            gray[-ps:, -ps:].ravel(),
        ])
        bg_val = np.median(corners)
        diff_smooth = ndimage.gaussian_filter(np.abs(gray - bg_val), sigma=3.5)

        # 3. Fused saliency + Otsu threshold
        sal = 0.6 * (grad_smooth / (grad_smooth.max() + 1e-6)) + 0.4 * (diff_smooth / (diff_smooth.max() + 1e-6))
        th = filters.threshold_otsu(sal)
        binary = sal > (th * 0.72)

        # 4. Suppress microscope metadata banners & edge lines (4% top/bottom, 2% left/right)
        by, bx = max(6, int(h * 0.04)), max(6, int(w * 0.02))
        binary[:by, :] = False
        binary[-by:, :] = False
        binary[:, :bx] = False
        binary[:, -bx:] = False

        # 5. Morphological hole filling & small noise removal (0.7% min area handles small S1 scales)
        binary = morphology.closing(binary, morphology.disk(12))
        binary = ndimage.binary_fill_holes(binary)
        min_area = int(h * w * 0.007)
        try:
            binary = morphology.remove_small_objects(binary, max_size=min_area)
        except TypeError:
            binary = morphology.remove_small_objects(binary, min_size=min_area)

        props = measure.regionprops(measure.label(binary))
        if not props:
            return (int(w * 0.075), int(h * 0.075), int(w * 0.925), int(h * 0.925))

        # 6. Biological prior: elliptical aspect ratio < 3.2 (rejects vertical rulers) & optical centrality
        valid = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            rh, rw = maxr - minr, maxc - minc
            aspect = max(rh / (rw + 1e-5), rw / (rh + 1e-5))
            if aspect < 3.2:
                cy, cx = p.centroid
                dist_c = np.hypot(cy - h / 2.0, cx - w / 2.0) / np.hypot(h / 2.0, w / 2.0)
                score = p.area * (1.0 - 0.55 * dist_c)
                valid.append((score, p))

        target = max(valid, key=lambda x: x[0])[1] if valid else max(props, key=lambda x: x.area)
        minr, minc, maxr, maxc = target.bbox

        ph, pw = int((maxr - minr) * padding_ratio), int((maxc - minc) * padding_ratio)
        x1 = max(0, minc - pw)
        y1 = max(0, minr - ph)
        x2 = min(w, maxc + pw)
        y2 = min(h, maxr + ph)

        if square:
            side = min(max(x2 - x1, y2 - y1), min(w, h))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, min(w - side, cx - side // 2))
            y1 = max(0, min(h - side, cy - side // 2))
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

        return (int(x1), int(y1), int(x2), int(y2))

    def crop(self, pil_image: Image.Image, padding_ratio: float = None, square: bool = True) -> Image.Image:
        bbox = self.detect_bbox(pil_image, padding_ratio=padding_ratio, square=square)
        return pil_image.crop(bbox)

    def mask_scale_out(self, pil_image: Image.Image) -> Image.Image:
        """
        Replaces ONLY the detected scale ROI with median corner background intensity
        for background-shortcut ablation testing (keeps rulers/borders/slide intact).
        """
        x1, y1, x2, y2 = self.detect_bbox(pil_image, padding_ratio=0.05, square=False)
        arr = np.array(pil_image.convert("RGB")).copy()
        h, w, _ = arr.shape
        ps = min(25, max(5, h // 15), max(5, w // 15))
        bg = np.median(arr[:ps, :ps], axis=(0, 1)).astype(np.uint8)
        arr[y1:y2, x1:x2] = bg
        return Image.fromarray(arr)

    def visualize_bbox(self, pil_image: Image.Image, color=(6, 182, 212)) -> Image.Image:
        vis = pil_image.copy().convert("RGB")
        x1, y1, x2, y2 = self.detect_bbox(vis)
        draw = ImageDraw.Draw(vis)
        lw = max(4, int(min(vis.size) * 0.008))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
        return vis
