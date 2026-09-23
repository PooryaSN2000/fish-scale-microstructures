import numpy as np
from PIL import Image
import config

try:
    from skimage.feature import hog, local_binary_pattern
except ImportError:
    # Fallback placeholders in case skimage is not imported
    hog = None
    local_binary_pattern = None


def extract_classical_features(image: Image.Image) -> np.ndarray:
    """
    Extract combined HOG and LBP features from a PIL Image.
    
    Args:
        image: A PIL Image (can be RGB or L).
        
    Returns:
        A 1D numpy array containing concatenated HOG and LBP features.
    """
    # Convert image to grayscale and resize to a standard size for classical features
    # Standardizing size ensures consistent feature vector lengths.
    gray_img = image.convert("L").resize(config.IMAGE_SIZE)
    img_array = np.array(gray_img)

    # 1. HOG Feature Extraction
    if hog is not None:
        try:
            hog_feats = hog(
                img_array,
                orientations=config.HOG_ORIENTATIONS,
                pixels_per_cell=config.HOG_PIXELS_PER_CELL,
                cells_per_block=config.HOG_CELLS_PER_BLOCK,
                visualize=False,
            )
        except Exception as e:
            # Fallback if HOG fails
            hog_dim = (
                (config.IMAGE_SIZE[0] // config.HOG_PIXELS_PER_CELL[0] - config.HOG_CELLS_PER_BLOCK[0] + 1)
                * (config.IMAGE_SIZE[1] // config.HOG_PIXELS_PER_CELL[1] - config.HOG_CELLS_PER_BLOCK[1] + 1)
                * (config.HOG_CELLS_PER_BLOCK[0] * config.HOG_CELLS_PER_BLOCK[1])
                * config.HOG_ORIENTATIONS
            )
            hog_feats = np.zeros(hog_dim, dtype=np.float32)
    else:
        # Compute default size
        hog_dim = 6084  # Default for 224x224, cell=16x16, block=2x2, orient=9
        hog_feats = np.zeros(hog_dim, dtype=np.float32)

    # 2. LBP Feature Extraction (using 'uniform' method for rotation invariance)
    if local_binary_pattern is not None:
        try:
            lbp = local_binary_pattern(
                img_array,
                P=config.LBP_POINTS,
                R=config.LBP_RADIUS,
                method="uniform"
            )
            # Uniform method yields P + 2 bins
            n_bins = config.LBP_POINTS + 2
            lbp_hist, _ = np.histogram(
                lbp.ravel(),
                bins=np.arange(0, n_bins + 1),
                range=(0, n_bins),
                density=True
            )
        except Exception as e:
            lbp_hist = np.zeros(config.LBP_POINTS + 2, dtype=np.float32)
    else:
        lbp_hist = np.zeros(config.LBP_POINTS + 2, dtype=np.float32)

    # Concatenate features
    combined_features = np.concatenate([hog_feats, lbp_hist]).astype(np.float32)
    
    # L2 normalize the combined feature vector
    norm = np.linalg.norm(combined_features)
    if norm > 0:
        combined_features = combined_features / norm
        
    return combined_features


if __name__ == "__main__":
    # Test feature extraction with a dummy image
    dummy_img = Image.fromarray(np.uint8(np.random.rand(300, 300, 3) * 255))
    feats = extract_classical_features(dummy_img)
    print(f"Extracted features shape: {feats.shape}")
    print(f"First few values: {feats[:10]}")
