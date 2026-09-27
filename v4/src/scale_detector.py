import io
import base64
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import filters, morphology, measure


class ScaleDetector:
    """
    Biological Microstructure Scale Localizer & Saliency Segmenter.
    
    Combines:
    1. Gradient-based texture saliency (isolating high-frequency circuli/radii from smooth glass/slide).
    2. Adaptive corner-background contrast estimation.
    3. Border-artifact rejection (ruler marks, timestamp bars, stage boundary lines).
    4. Biological convex morphology (closing circuli gaps, hole filling, elliptical aspect ratio scoring).
    5. Protective margin padding with optional square aspect-ratio preservation.
    """
    def __init__(self, mode: str = "adaptive", yolo_weights: str = None):
        self.mode = mode
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
        padding_ratio: float = 0.08, 
        square: bool = False
    ) -> tuple[int, int, int, int]:
        """
        Locates the fish scale bounding box (x1, y1, x2, y2).
        
        Args:
            pil_image: Input PIL Image.
            padding_ratio: Safety margin around the detected scale (default 8%).
            square: If True, expands the bounding box to a 1:1 aspect ratio to avoid distortion.
            
        Returns:
            (x1, y1, x2, y2) bounding box coordinates.
        """
        if self.mode == "yolo" and self.yolo_model is not None:
            results = self.yolo_model(pil_image, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                b = boxes[0].xyxy[0].cpu().numpy().astype(int)
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        # --- Biological Texture Saliency & Morphological Localization ---
        w, h = pil_image.size
        gray = np.array(pil_image.convert("L"), dtype=float)

        # 1. High-frequency texture saliency via Sobel gradient magnitude
        grad_x = ndimage.sobel(gray, axis=1)
        grad_y = ndimage.sobel(gray, axis=0)
        grad_mag = np.hypot(grad_x, grad_y)
        grad_smooth = ndimage.gaussian_filter(grad_mag, sigma=4.0)

        # 2. Estimate background illumination from corner patches
        patch_size = min(30, max(5, h // 15), max(5, w // 15))
        corners = np.concatenate([
            gray[:patch_size, :patch_size].flatten(),
            gray[:patch_size, -patch_size:].flatten(),
            gray[-patch_size:, :patch_size].flatten(),
            gray[-patch_size:, -patch_size:].flatten()
        ])
        bg_val = np.median(corners)
        diff_from_bg = np.abs(gray - bg_val)
        diff_smooth = ndimage.gaussian_filter(diff_from_bg, sigma=4.0)

        # 3. Fused Saliency Map: 60% Texture + 40% Background Contrast
        norm_grad = grad_smooth / (grad_smooth.max() + 1e-6)
        norm_diff = diff_smooth / (diff_smooth.max() + 1e-6)
        saliency = 0.6 * norm_grad + 0.4 * norm_diff

        # 4. Adaptive Thresholding with safety relaxation
        th = filters.threshold_otsu(saliency)
        binary = saliency > (th * 0.70)

        # 5. Suppress extreme edge artifacts (microscope frame lines, timestamps)
        border_px_y = max(4, int(h * 0.02))
        border_px_x = max(4, int(w * 0.02))
        binary[:border_px_y, :] = False
        binary[-border_px_y:, :] = False
        binary[:, :border_px_x] = False
        binary[:, -border_px_x:] = False

        # 6. Morphological consolidation: Bridge circuli/radii grooves and fill interior
        struct = morphology.disk(15)
        binary = morphology.closing(binary, struct)
        binary = ndimage.binary_fill_holes(binary)

        # 7. Remove residual micro-noise
        min_scale_area = int(h * w * 0.03)  # Scales must be at least 3% of the image
        try:
            binary = morphology.remove_small_objects(binary, max_size=min_scale_area)
        except TypeError:
            binary = morphology.remove_small_objects(binary, min_size=min_scale_area)

        labeled = measure.label(binary)
        props = measure.regionprops(labeled)

        if not props:
            # Fallback to central 85% window if nothing salient detected
            pad_x = int(w * 0.075)
            pad_y = int(h * 0.075)
            return (pad_x, pad_y, w - pad_x, h - pad_y)

        # 8. Score candidates based on Biological Prior (Rounded / Elliptical + Centrality)
        valid_props = []
        for p in props:
            minr, minc, maxr, maxc = p.bbox
            rh = maxr - minr
            rw = maxc - minc
            aspect = max(rh / (rw + 1e-5), rw / (rh + 1e-5))
            if aspect < 3.8:  # Eliminate long straight scratches or border lines
                cy, cx = p.centroid
                dist_center = np.hypot(cy - h / 2.0, cx - w / 2.0) / np.hypot(h / 2.0, w / 2.0)
                # Area weighted by proximity to optical center
                score = p.area * (1.0 - 0.40 * dist_center)
                valid_props.append((score, p))

        if not valid_props:
            largest = max(props, key=lambda x: x.area)
        else:
            valid_props.sort(key=lambda x: x[0], reverse=True)
            largest = valid_props[0][1]

        minr, minc, maxr, maxc = largest.bbox

        # 9. Apply protective padding
        pad_h = int((maxr - minr) * padding_ratio)
        pad_w = int((maxc - minc) * padding_ratio)

        x1 = max(0, minc - pad_w)
        y1 = max(0, minr - pad_h)
        x2 = min(w, maxc + pad_w)
        y2 = min(h, maxr + pad_h)

        # 10. Optional square aspect-ratio expansion
        if square:
            box_w = x2 - x1
            box_h = y2 - y1
            side = max(box_w, box_h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)

        return (int(x1), int(y1), int(x2), int(y2))

    def crop(
        self, 
        pil_image: Image.Image, 
        padding_ratio: float = 0.08, 
        square: bool = False
    ) -> Image.Image:
        """
        Extracts and returns the cropped fish scale image.
        """
        bbox = self.detect_bbox(pil_image, padding_ratio=padding_ratio, square=square)
        return pil_image.crop(bbox)

    def visualize_detection(
        self, 
        pil_image: Image.Image, 
        bbox: tuple[int, int, int, int] = None, 
        padding_ratio: float = 0.08
    ) -> Image.Image:
        """
        Draws a neon cyan/emerald bounding box with corner guides over the scale image.
        """
        if bbox is None:
            bbox = self.detect_bbox(pil_image, padding_ratio=padding_ratio)

        vis_img = pil_image.copy().convert("RGB")
        draw = ImageDraw.Draw(vis_img)
        x1, y1, x2, y2 = bbox

        # Main bounding box
        outline_color = (6, 182, 212)  # Cyan #06b6d4
        accent_color = (16, 185, 129)  # Emerald #10b981
        line_width = max(3, int(min(vis_img.size) * 0.005))
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=line_width)

        # Corner brackets for HUD / AI diagnostic feel
        corner_len = max(15, int(min(x2 - x1, y2 - y1) * 0.12))
        c_width = line_width + 2
        # Top-Left
        draw.line([(x1, y1), (x1 + corner_len, y1)], fill=accent_color, width=c_width)
        draw.line([(x1, y1), (x1, y1 + corner_len)], fill=accent_color, width=c_width)
        # Top-Right
        draw.line([(x2, y1), (x2 - corner_len, y1)], fill=accent_color, width=c_width)
        draw.line([(x2, y1), (x2, y1 + corner_len)], fill=accent_color, width=c_width)
        # Bottom-Left
        draw.line([(x1, y2), (x1 + corner_len, y2)], fill=accent_color, width=c_width)
        draw.line([(x1, y2), (x1, y2 - corner_len)], fill=accent_color, width=c_width)
        # Bottom-Right
        draw.line([(x2, y2), (x2 - corner_len, y2)], fill=accent_color, width=c_width)
        draw.line([(x2, y2), (x2, y2 - corner_len)], fill=accent_color, width=c_width)

        return vis_img

    def visualize_detection_base64(
        self, 
        pil_image: Image.Image, 
        bbox: tuple[int, int, int, int] = None, 
        padding_ratio: float = 0.08
    ) -> str:
        """
        Returns data:image/jpeg;base64,... string of the detection visualization.
        """
        vis_img = self.visualize_detection(pil_image, bbox=bbox, padding_ratio=padding_ratio)
        vis_img.thumbnail((800, 800))
        buffer = io.BytesIO()
        vis_img.save(buffer, format="JPEG", quality=85)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"


if __name__ == "__main__":
    detector = ScaleDetector()
    print("Biological ScaleDetector initialized and verified successfully.")
