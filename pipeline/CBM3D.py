from bm3d import bm3d

def denoise(noisy, sigma):
    noisy = noisy / 255.0
    denoised = bm3d(noisy, sigma_psd=sigma/255)
    return denoised