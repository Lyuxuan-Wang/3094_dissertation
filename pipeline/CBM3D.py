from bm3d import bm3d

def denoise(noisy):
    noisy = noisy / 255.0
    denoised = bm3d(noisy, sigma_psd=10/255)