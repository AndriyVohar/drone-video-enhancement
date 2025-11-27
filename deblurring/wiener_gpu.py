"""
GPU-accelerated Wiener deconvolution using CuPy
"""
import numpy as np
from utils.gpu_utils import is_gpu_available, to_gpu, get_array_module

if is_gpu_available():
    import cupy as cp

def wiener_deconvolution_gpu(blurred_image, psf, K=0.01):
    """
    Wiener deconvolution using GPU acceleration.

    Args:
        blurred_image: Blurred input image (CuPy array)
        psf: Point spread function (CuPy array)
        K: Noise-to-signal ratio (regularization parameter)

    Returns:
        Deconvolved image (CuPy array)
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available for Wiener deconvolution")

    xp = get_array_module(blurred_image)

    # Ensure PSF is normalized
    psf = psf / xp.sum(psf)

    # Pad PSF to image size
    psf_padded = xp.zeros_like(blurred_image)
    psf_h, psf_w = psf.shape
    psf_padded[:psf_h, :psf_w] = psf

    # Circular shift PSF
    psf_padded = xp.roll(psf_padded, -psf_h // 2, axis=0)
    psf_padded = xp.roll(psf_padded, -psf_w // 2, axis=1)

    # FFT
    G = xp.fft.fft2(blurred_image)
    H = xp.fft.fft2(psf_padded)

    # Wiener filter: F = (H* * G) / (|H|^2 + K)
    H_conj = xp.conj(H)
    H_abs_sq = xp.abs(H) ** 2

    F = (H_conj * G) / (H_abs_sq + K)

    # Inverse FFT
    restored = xp.fft.ifft2(F)
    restored = xp.real(restored)

    # Clip to valid range
    restored = xp.clip(restored, 0, 255)

    return restored
