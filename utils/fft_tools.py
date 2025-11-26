"""
CPU-based FFT utility functions using NumPy.
"""
import numpy as np


def psf2otf(psf, output_shape):
    """
    Convert point-spread function to optical transfer function (CPU version).

    Args:
        psf: Point spread function (2D array)
        output_shape: Desired output shape (tuple)

    Returns:
        Optical transfer function in frequency domain
    """
    # Ensure PSF is normalized
    psf = psf / np.sum(psf)

    # Pad PSF to output shape
    psf_padded = np.zeros(output_shape, dtype=psf.dtype)

    # Place PSF at origin
    psf_h, psf_w = psf.shape
    psf_padded[:psf_h, :psf_w] = psf

    # Circular shift to center
    psf_padded = np.roll(psf_padded, -psf_h // 2, axis=0)
    psf_padded = np.roll(psf_padded, -psf_w // 2, axis=1)

    # Compute OTF
    otf = np.fft.fft2(psf_padded)

    return otf


def convolve_fft(image, kernel):
    """
    Convolve image with kernel using FFT (CPU version).

    Args:
        image: Input image
        kernel: Convolution kernel

    Returns:
        Convolved image
    """
    # Get OTF
    otf = psf2otf(kernel, image.shape)

    # FFT of image
    image_fft = np.fft.fft2(image)

    # Multiply in frequency domain
    result_fft = image_fft * otf

    # Inverse FFT
    result = np.fft.ifft2(result_fft)

    # Return real part
    return np.real(result)

