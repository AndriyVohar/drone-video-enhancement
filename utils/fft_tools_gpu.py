"""
GPU-accelerated FFT tools using CuPy
"""
import numpy as np
from utils.gpu_utils import get_array_module, is_gpu_available

if is_gpu_available():
    import cupy as cp


def fft2_gpu(image):
    """2D FFT on GPU"""
    xp = get_array_module(image)
    return xp.fft.fft2(image)


def ifft2_gpu(spectrum):
    """2D inverse FFT on GPU"""
    xp = get_array_module(spectrum)
    return xp.fft.ifft2(spectrum)


def fftshift_gpu(array):
    """Shift zero-frequency component to center"""
    xp = get_array_module(array)
    return xp.fft.fftshift(array)


def ifftshift_gpu(array):
    """Inverse FFT shift"""
    xp = get_array_module(array)
    return xp.fft.ifftshift(array)


def psf2otf_gpu(psf, output_shape):
    """
    Convert point-spread function to optical transfer function (GPU version).

    Args:
        psf: Point spread function (2D CuPy array)
        output_shape: Desired output shape (tuple)

    Returns:
        Optical transfer function in frequency domain (CuPy array)
    """
    if not is_gpu_available():
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
    if not is_gpu_available():
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
