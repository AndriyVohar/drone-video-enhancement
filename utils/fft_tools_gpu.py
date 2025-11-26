"""
GPU-accelerated FFT utility functions using CuPy.
"""
import numpy as np
from .gpu_utils import GPU_AVAILABLE

if GPU_AVAILABLE:
    import cupy as cp


def psf2otf_gpu(psf, output_shape):
    """
    Convert point-spread function to optical transfer function (GPU version).

    Args:
        psf: Point spread function (2D CuPy array)
        output_shape: Desired output shape (tuple)

    Returns:
        Optical transfer function in frequency domain (CuPy array)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    # Ensure PSF is normalized
    psf = psf / cp.sum(psf)

    # Pad PSF to output shape
    psf_padded = cp.zeros(output_shape, dtype=psf.dtype)

    # Place PSF at origin
    psf_h, psf_w = psf.shape
    psf_padded[:psf_h, :psf_w] = psf

    # Circular shift to center
    psf_padded = cp.roll(psf_padded, -psf_h // 2, axis=0)
    psf_padded = cp.roll(psf_padded, -psf_w // 2, axis=1)

    # Compute OTF using CuPy FFT
    otf = cp.fft.fft2(psf_padded)

    return otf


def convolve_fft_gpu(image, kernel):
    """
    Convolve image with kernel using FFT (GPU version).

    Args:
        image: Input image (CuPy array)
        kernel: Convolution kernel (CuPy array)

    Returns:
        Convolved image (CuPy array)
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available")

    # Get OTF
    otf = psf2otf_gpu(kernel, image.shape)

    # FFT of image
    image_fft = cp.fft.fft2(image)

    # Multiply in frequency domain
    result_fft = image_fft * otf

    # Inverse FFT
    result = cp.fft.ifft2(result_fft)

    # Return real part
    return cp.real(result)

