"""
GPU-accelerated Richardson-Lucy deconvolution using CuPy
"""
import numpy as np
from utils.gpu_utils import is_gpu_available, get_array_module

if is_gpu_available():
    import cupy as cp

def richardson_lucy_gpu(blurred_image, psf, iterations=10):
    """
    Richardson-Lucy deconvolution using GPU acceleration.

    Args:
        blurred_image: Blurred input image (CuPy array)
        psf: Point spread function (CuPy array)
        iterations: Number of iterations

    Returns:
        Deconvolved image (CuPy array)
    """
    if not is_gpu_available():
        raise RuntimeError("GPU not available for Richardson-Lucy deconvolution")

    xp = get_array_module(blurred_image)

    # Ensure PSF is normalized
    psf = psf / xp.sum(psf)

    # Flip PSF for correlation
    psf_flipped = xp.flip(xp.flip(psf, axis=0), axis=1)

    # Initialize estimate
    estimate = xp.copy(blurred_image)

    for i in range(iterations):
        # Convolve estimate with PSF
        conv_estimate = _convolve_gpu(estimate, psf)

        # Avoid division by zero
        conv_estimate = xp.maximum(conv_estimate, 1e-10)

        # Compute ratio
        ratio = blurred_image / conv_estimate

        # Convolve ratio with flipped PSF
        correction = _convolve_gpu(ratio, psf_flipped)

        # Update estimate
        estimate = estimate * correction

        # Clip to valid range
        estimate = xp.clip(estimate, 0, 255)

    return estimate

def _convolve_gpu(image, kernel):
    """Helper function for 2D convolution using FFT on GPU"""
    xp = get_array_module(image)

    # Pad kernel to image size
    kernel_padded = xp.zeros_like(image)
    k_h, k_w = kernel.shape
    kernel_padded[:k_h, :k_w] = kernel

    # Circular shift
    kernel_padded = xp.roll(kernel_padded, -k_h // 2, axis=0)
    kernel_padded = xp.roll(kernel_padded, -k_w // 2, axis=1)

    # FFT convolution
    image_fft = xp.fft.fft2(image)
    kernel_fft = xp.fft.fft2(kernel_padded)
    result_fft = image_fft * kernel_fft
    result = xp.fft.ifft2(result_fft)

    return xp.real(result)
