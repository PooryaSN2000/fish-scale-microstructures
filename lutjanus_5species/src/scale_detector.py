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
        max_side = 768
        scale_down = min(1.0, max_side / max(w, h))
        if scale_down < 1.0:
            dw, dh = max(32, int(round(w * scale_down))), max(32, int(round(h * scale_down)))
            gray = np.array(pil_image.convert("L").resize((dw, dh), Image.Resampling.BILINEAR), dtype=float)
        else:
            dw, dh = w, h
            gray = np.array(pil_image.convert("L"), dtype=float)

        # 1. Sobel high-frequency texture saliency (circuli & radii vs smooth slide)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_smooth = ndimage.gaussian_filter(np.hypot(grad_x, grad_y), sigma=3.0)

        # 2. Corner-background luminance baseline (immune to bright rulers in 1 corner)
        ps = min(25, max(5, dh // 15), max(5, dw // 15))
        corner_meds = [
            float(np.median(gray[:ps, :ps])),
            float(np.median(gray[:ps, -ps:])),
            float(np.median(gray[-ps:, :ps])),
            float(np.median(gray[-ps:, -ps:])),
        ]
        bg_val = float(np.percentile(corner_meds, 25))
        diff_smooth = ndimage.gaussian_filter(np.abs(gray - bg_val), sigma=3.0)

        # 3. Percentile-clipped fused saliency + Otsu threshold
        p_grad = max(float(np.percentile(grad_smooth, 98.5)), 1e-6)
        p_diff = max(float(np.percentile(diff_smooth, 98.5)), 1e-6)
        sal = 0.55 * np.clip(grad_smooth / p_grad, 0.0, 1.0) + 0.45 * np.clip(diff_smooth / p_diff, 0.0, 1.0)
        th = filters.threshold_otsu(sal)
        binary = sal > (th * 0.65)

        # 4. Suppress microscope metadata banners & edge lines (4% top/bottom, 2% left/right)
        by, bx = max(4, int(dh * 0.04)), max(4, int(dw * 0.02))
        binary[:by, :] = False
        binary[-by:, :] = False
        binary[:, :bx] = False
        binary[:, -bx:] = False

        # 5. Morphological hole filling & small noise removal (0.7% min area handles small S1 scales)
        disk_r = max(4, int(round(12 * min(dw, dh) / 1024.0)))
        binary = morphology.closing(binary, morphology.disk(disk_r))
        binary = ndimage.binary_fill_holes(binary)
        min_area = int(dh * dw * 0.007)
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
                dist_c = np.hypot(cy - dh / 2.0, cx - dw / 2.0) / np.hypot(dh / 2.0, dw / 2.0)
                score = p.area * (1.0 - 0.55 * dist_c)
                valid.append((score, p))

        target = max(valid, key=lambda x: x[0])[1] if valid else max(props, key=lambda x: x.area)
        minr, minc, maxr, maxc = target.bbox
        sx, sy = w / float(dw), h / float(dh)
        minr, minc, maxr, maxc = int(round(minr * sy)), int(round(minc * sx)), int(round(maxr * sy)), int(round(maxc * sx))

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
