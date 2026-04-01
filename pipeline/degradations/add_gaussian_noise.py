from pathlib import Path
import numpy as np
from PIL import Image

def add_gaussian_noise(img_path, sigma=25, seed=None):
    img = Image.open(img_path)
    img_np = np.array(img).astype(np.float32) / 255.0

    noise = np.random.normal(0, sigma / 255.0, img_np.shape).astype(np.float32)
    noisy = img_np + noise
    noisy = np.clip(noisy, 0.0, 1.0)

    noisy_img = Image.fromarray((noisy * 255.0).astype(np.uint8))
    return noisy_img