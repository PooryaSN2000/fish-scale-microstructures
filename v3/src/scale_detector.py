import numpy as np
from PIL import Image
from skimage import filters, morphology, measure


class ScaleDetector:
    """
    Automated Fish Scale Localizer and Cropper.
    Detects the scale body and eliminates microscope background, millimeter rulers,
    and stage borders using adaptive morphological segmentation.
    Supports Otsu/morphological detection natively, with modular hooks for YOLOv8.
    """
    def __init__(self, mode: str = "adaptive", yolo_weights: str = None):
        self.mode = mode
        self.yolo_model = None
        
        if mode == "yolo" and yolo_weights is not None:
            try:
                from ultralytics import YOLO
                self.yolo_model = YOLO(yolo_weights)
            except Exception as e:
                print(f"Warning: Failed to load YOLO weights ({e}). Falling back to adaptive mode.")
                self.mode = "adaptive"

    def detect_bbox(self, pil_image: Image.Image, padding_ratio: float = 0.05) -> tuple[int, int, int, int]:
        """
        Locates the fish scale bounding box (x1, y1, x2, y2).
        """
        if self.mode == "yolo" and self.yolo_model is not None:
            results = self.yolo_model(pil_image, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                # Top confidence box
                b = boxes[0].xyxy[0].cpu().numpy().astype(int)
                return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

        # Adaptive Otsu morphological localization
        gray = np.array(pil_image.convert("L"))
        thresh = filters.threshold_otsu(gray)
        binary = gray > thresh if np.mean(gray) < 128 else gray < thresh
        
        # Remove small noise and bridge micro-gaps
        try:
            binary = morphology.remove_small_objects(binary, max_size=500)
        except TypeError:
            binary = morphology.remove_small_objects(binary, min_size=500)
            
        try:
            binary = morphology.closing(binary, morphology.disk(5))
        except AttributeError:
            binary = morphology.binary_closing(binary, morphology.disk(5))

        labeled = measure.label(binary)
        regions = measure.regionprops(labeled)
        
        if not regions:
            return (0, 0, pil_image.width, pil_image.height)

        # Select the largest central salient region
        largest = max(regions, key=lambda r: r.area)
        minr, minc, maxr, maxc = largest.bbox
        
        pad_h = int((maxr - minr) * padding_ratio)
        pad_w = int((maxc - minc) * padding_ratio)
        
        x1 = max(0, minc - pad_w)
        y1 = max(0, minr - pad_h)
        x2 = min(pil_image.width, maxc + pad_w)
        y2 = min(pil_image.height, maxr + pad_h)
        
        return (x1, y1, x2, y2)

    def crop(self, pil_image: Image.Image, padding_ratio: float = 0.05) -> Image.Image:
        """
        Extracts and returns the cropped fish scale image.
        """
        bbox = self.detect_bbox(pil_image, padding_ratio=padding_ratio)
        return pil_image.crop(bbox)


if __name__ == "__main__":
    detector = ScaleDetector()
    print("ScaleDetector initialized successfully.")
