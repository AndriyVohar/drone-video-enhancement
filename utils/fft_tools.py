"""
FFT utilities for frequency domain processing.
"""
import numpy as np
from typing import Tuple


def fft2_img(img: np.ndarray) -> np.ndarray:
    """
    Compute 2D FFT of an image.

    Args:
        img: Input image (real-valued)

    Returns:
        2D FFT (complex-valued)
    """
    return np.fft.fft2(img)


def ifft2_img(freq: np.ndarray) -> np.ndarray:
    """
    Compute inverse 2D FFT.

    Args:
        freq: Frequency domain representation

    Returns:
        Real-valued image
    """
    return np.real(np.fft.ifft2(freq))


def psf_to_otf(psf: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert PSF to Optical Transfer Function (OTF) by padding and FFT.

    The PSF is assumed to be centered. This function:
    1. Pads the PSF to the target shape
    2. Circularly shifts it so the center is at (0,0)
    3. Computes the FFT

    Args:
        psf: Point Spread Function kernel
        shape: Target shape (height, width)

    Returns:
        OTF in frequency domain
    """
    # Pad PSF to target shape
    psf_padded = np.zeros(shape, dtype=np.float64)

    # Calculate where to place the PSF (centered)
    psf_h, psf_w = psf.shape
    start_h = (shape[0] - psf_h) // 2
    start_w = (shape[1] - psf_w) // 2

    psf_padded[start_h:start_h + psf_h, start_w:start_w + psf_w] = psf

    # Circular shift to move center to origin (required for FFT convolution)
    psf_padded = np.fft.ifftshift(psf_padded)

    # Compute OTF
    otf = np.fft.fft2(psf_padded)

    return otf


def fftshift(img: np.ndarray) -> np.ndarray:
    """
    Shift zero-frequency component to center.

    Args:
        img: Input array

    Returns:
        Shifted array
    """
    return np.fft.fftshift(img)


def ifftshift(img: np.ndarray) -> np.ndarray:
    """
    Inverse FFT shift.

    Args:
        img: Input array

    Returns:
        Shifted array
    """
    return np.fft.ifftshift(img)

