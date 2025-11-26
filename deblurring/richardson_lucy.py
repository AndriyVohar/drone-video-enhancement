"""
Richardson-Lucy deconvolution algorithm.
"""
import numpy as np
from scipy import signal


def richardson_lucy(img: np.ndarray, psf: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Apply Richardson-Lucy deconvolution algorithm.

    This is an iterative algorithm that maximizes likelihood assuming Poisson noise.

    Algorithm:
    f^(k+1) = f^(k) · [(g / (h * f^(k))) * h_flipped]

    where:
    - f^(k) is the estimate at iteration k
    - g is the observed blurred image
    - h is the PSF
    - * denotes convolution
    - h_flipped is the PSF flipped (for correlation)

    Args:
        img: Blurred input image (grayscale, float, non-negative)
        psf: Point Spread Function kernel
        iterations: Number of iterations

    Returns:
        Deblurred image
    """
    # Ensure image is float and non-negative
    img = img.astype(np.float64)
    img = np.maximum(img, 0) + 1e-10  # Add small epsilon to avoid division by zero

    # Normalize PSF
    psf = psf.astype(np.float64)
    psf = psf / (np.sum(psf) + 1e-10)

    # Flip PSF for correlation
    psf_flipped = np.flip(psf)

    # Initialize estimate with the observed image
    f_estimate = np.copy(img)

    # Iterate
    for i in range(iterations):
        # Convolve current estimate with PSF
        # Using 'same' mode to keep same size
        convolved = signal.convolve2d(f_estimate, psf, mode='same', boundary='symm')

        # Avoid division by zero
        convolved = np.maximum(convolved, 1e-10)

        # Compute ratio
        ratio = img / convolved

        # Correlate ratio with flipped PSF
        correlation = signal.convolve2d(ratio, psf_flipped, mode='same', boundary='symm')

        # Update estimate
        f_estimate = f_estimate * correlation

        # Ensure non-negative
        f_estimate = np.maximum(f_estimate, 0)

    return f_estimate


def richardson_lucy_fft(img: np.ndarray, psf: np.ndarray, iterations: int = 10) -> np.ndarray:
    """
    Richardson-Lucy deconvolution using FFT for faster convolution.

    Same algorithm as richardson_lucy but uses FFT-based convolution
    for better performance on larger images.

    Args:
        img: Blurred input image (grayscale, float, non-negative)
        psf: Point Spread Function kernel
        iterations: Number of iterations

    Returns:
        Deblurred image
    """
    from utils.fft_tools import psf_to_otf, fft2_img, ifft2_img

    # Ensure image is float and non-negative
    img = img.astype(np.float64)
    img = np.maximum(img, 0) + 1e-10

    # Normalize PSF
    psf = psf.astype(np.float64)
    psf = psf / (np.sum(psf) + 1e-10)

    # Get OTF for PSF and flipped PSF
    img_shape = img.shape
    H = psf_to_otf(psf, img_shape)

    # Flip PSF for correlation
    psf_flipped = np.flip(psf)
    H_flipped = psf_to_otf(psf_flipped, img_shape)

    # Initialize estimate
    f_estimate = np.copy(img)

    # Iterate
    for i in range(iterations):
        # Convolve with PSF using FFT
        F = fft2_img(f_estimate)
        convolved = ifft2_img(H * F)
        convolved = np.real(convolved)
        convolved = np.maximum(convolved, 1e-10)

        # Compute ratio
        ratio = img / convolved

        # Correlate with flipped PSF
        R = fft2_img(ratio)
        correlation = ifft2_img(H_flipped * R)
        correlation = np.real(correlation)

        # Update estimate
        f_estimate = f_estimate * correlation
        f_estimate = np.maximum(f_estimate, 0)

    return f_estimate
"""Deblurring algorithms using classical techniques."""

