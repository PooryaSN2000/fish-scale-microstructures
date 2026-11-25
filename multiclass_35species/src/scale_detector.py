import io
import base64
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import filters, morphology, measure


class ScaleDetector:
    """
    Fast Biological Microstructure Scale Localizer & Border-Artifact Rejector.
    Detects the elasmoid scale body on a downscaled proxy (max_side=768) for ~40x speedup,
    maps bounding boxes accurately back to native micrograph resolution, and suppresses
    microscope timestamp banners, slide glass edges, and calibration rulers.
    """
    def __init__(self, mode: str = "adaptive", yolo_weights: str = None, padding_ratio: float = 0.12):
        self.mode = mode
        self.padding_ratio = padding_ratio
        self.yolo_model = None

        if mode == "yolo" and yolo_weights is not None:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_weights)
            except Exception as e:
                print(f"Warning: Failed to load YOLO weights ({e}). Falling back to adaptive biological mode.")
                self.mode = "adaptive"

    def detect_bbox(
        self,
        pil_image: Image.Image,
        padding_ratio: float = None,
        square: bool = True,
    ) -> tuple[int, int, int, int]:
        if padding_ratio is None:
            padding_ratio = self.padding_ratio

        if self.mode == "yolo" and self.yolo_model is not None:
            results = self.yolo_model(pil_image, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                b = boxes[0].xyxy[0].cpu().numpy().astype(int)
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        orig_w, orig_h = pil_image.size
        max_side = 768
        scale_down = min(1.0, max_side / max(orig_w, orig_h))
        if scale_down < 1.0:
            dw, dh = max(32, int(round(orig_w * scale_down))), max(32, int(round(orig_h * scale_down)))
            gray = np.array(pil_image.convert("L").resize((dw, dh), Image.Resampling.BILINEAR), dtype=float)
        else:
            dw, dh = orig_w, orig_h
            gray = np.array(pil_image.convert("L"), dtype=float)

        # 1. Sobel gradient texture saliency (circuli & radii vs smooth glass)
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_smooth = ndimage.gaussian_filter(np.hypot(grad_x, grad_y), sigma=3.0)

        # 2. Robust darkfield background luminance baseline (immune to bright ruler in 1 corner)
        ps = min(25, max(5, dh // 15), max(5, dw // 15))
        corner_meds = [
            float(np.median(gray[:ps, :ps])),
            float(np.median(gray[:ps, -ps:])),
            float(np.median(gray[-ps:, :ps])),
            float(np.median(gray[-ps:, -ps:])),
        ]
        bg_val = float(np.percentile(corner_meds, 25))
        diff_smooth = ndimage.gaussian_filter(np.abs(gray - bg_val), sigma=3.0)

        # 3. Robustly percentile-clipped fused saliency + relaxed Otsu threshold
        # Percentile clipping prevents bright ctenii specular glare from starving translucent lobes
        p_grad = max(float(np.percentile(grad_smooth, 98.5)), 1e-6)
        p_diff = max(float(np.percentile(diff_smooth, 98.5)), 1e-6)
        sal = 0.45 * np.clip(grad_smooth / p_grad, 0.0, 1.0) + 0.55 * np.clip(diff_smooth / p_diff, 0.0, 1.0)
        th = filters.threshold_otsu(sal)
        binary = sal > (th * 0.45)

        # 4. Suppress microscope metadata banners & edge lines (4% vertical, 2% horizontal)
        by, bx = max(4, int(dh * 0.04)), max(4, int(dw * 0.02))
        binary[:by, :] = False
        binary[-by:, :] = False
        binary[:, :bx] = False
        binary[:, -bx:] = False

        # 5. Morphological consolidation (0.7% min area handles small S1/minority scales)
        disk_r = max(5, int(round(14 * min(dw, dh) / 1024.0)))
        binary = morphology.closing(binary, morphology.disk(disk_r))
        binary = ndimage.binary_fill_holes(binary)
        min_area = int(dh * dw * 0.007)
        try:
            binary = morphology.remove_small_objects(binary, max_size=min_area)
        except TypeError:
            binary = morphology.remove_small_objects(binary, min_size=min_area)

        props = measure.regionprops(measure.label(binary))
        if not props:
            return (int(orig_w * 0.075), int(orig_h * 0.075), int(orig_w * 0.925), int(orig_h * 0.925))

        # 6. Biological prior: elliptical aspect < 3.2, reject edge-pinned full-height rulers
        valid = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            rh, rw = maxr - minr, maxc - minc
            aspect = max(rh / (rw + 1e-5), rw / (rh + 1e-5))
            touches_lr_edge = (minc <= bx + 4) or (maxc >= dw - bx - 4)
            is_edge_ruler = touches_lr_edge and (rh >= dh * 0.72) and (rw < dw * 0.35)
            if aspect < 3.2 and not is_edge_ruler:
                cy, cx = p.centroid
                dist_c = np.hypot(cy - dh / 2.0, cx - dw / 2.0) / np.hypot(dh / 2.0, dw / 2.0)
                score = p.area * (1.0 - 0.55 * dist_c)
                valid.append((score, p))

        target = max(valid, key=lambda x: x[0])[1] if valid else max(props, key=lambda x: x.area)
        minr, minc, maxr, maxc = target.bbox

        # Map detected box back to native image coordinates
        sx = orig_w / float(dw)
        sy = orig_h / float(dh)
        x1_raw, y1_raw = int(round(minc * sx)), int(round(minr * sy))
        x2_raw, y2_raw = int(round(maxc * sx)), int(round(maxr * sy))

        ph = int((y2_raw - y1_raw) * padding_ratio)
        pw = int((x2_raw - x1_raw) * padding_ratio)
        x1 = max(0, x1_raw - pw)
        y1 = max(0, y1_raw - ph)
        x2 = min(orig_w, x2_raw + pw)
        y2 = min(orig_h, y2_raw + ph)

        if square:
            side = min(max(x2 - x1, y2 - y1), min(orig_w, orig_h))
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            x1 = max(0, min(orig_w - side, cx - side // 2))
            y1 = max(0, min(orig_h - side, cy - side // 2))
            x2 = min(orig_w, x1 + side)
            y2 = min(orig_h, y1 + side)

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

    def visualize_detection(self, pil_image: Image.Image, bbox: tuple[int, int, int, int] = None, padding_ratio: float = None) -> Image.Image:
        if bbox is None:
            bbox = self.detect_bbox(pil_image, padding_ratio=padding_ratio, square=True)

        vis_img = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(vis_img)
        x1, y1, x2, y2 = bbox

        outline_color = (6, 182, 212)  # Cyan #06b6d4
        accent_color = (16, 185, 129)  # Emerald #10b981
        line_width = max(3, int(min(vis_img.size) * 0.006))
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=line_width)

        corner_len = max(15, int(min(x2 - x1, y2 - y1) * 0.12))
        c_width = line_width + 2
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=accent_color, width=c_width)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=accent_color, width=c_width)
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=accent_color, width=c_width)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=accent_color, width=c_width)
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=accent_color, width=c_width)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=accent_color, width=c_width)
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=accent_color, width=c_width)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=accent_color, width=c_width)

        return vis_img

    def visualize_detection_base64(self, pil_image: Image.Image, bbox: tuple[int, int, int, int] = None, padding_ratio: float = None) -> str:
        vis_img = self.visualize_detection(pil_image, bbox=bbox, padding_ratio=padding_ratio)
        vis_img.thumbnail((800, 800))
        buffer = io.BytesIO()
        vis_img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
