import math
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

def compute_mse(original: np.ndarray, stego: np.ndarray) -> float:
    diff = original.astype(np.float64) - stego.astype(np.float64)
    return float(np.mean(diff ** 2))

def compute_psnr(mse: float, max_val: float = 255.0) -> float:
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(max_val ** 2 / mse)

def compute_ssim(original: np.ndarray, stego: np.ndarray, **kwargs) -> float:
    return float(ssim_metric(original, stego, data_range=255, **kwargs))
