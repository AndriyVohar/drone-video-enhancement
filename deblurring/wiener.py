"""
Wiener deconvolution filter implementation.
"""
import numpy as np
from utils.fft_tools import fft2_img, ifft2_img, psf_to_otf


def wiener_deconvolution(img: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """
    Apply Wiener filtering for image deblurring.

    Wiener filter in frequency domain:
    F_hat = (H* · G) / (|H|^2 + K)

    where:
    - F_hat is the estimated original image (frequency domain)
    - H is the PSF in frequency domain (OTF)
    - G is the blurred image in frequency domain
    - K is the noise-to-signal power ratio (regularization parameter)
    - H* is the complex conjugate of H

    Args:
        img: Blurred input image (grayscale, float)
        psf: Point Spread Function kernel
        K: Regularization parameter (noise-to-signal ratio)

    Returns:
        Deblurred image
    """
    # Ensure image is float
    img = img.astype(np.float64)

    # Get image dimensions
    img_shape = img.shape

    # Convert PSF to OTF (frequency domain)
    H = psf_to_otf(psf, img_shape)

    # Compute FFT of blurred image
    G = fft2_img(img)

    # Wiener filter formula
    # F_hat = conj(H) * G / (|H|^2 + K)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2

    # Apply Wiener filter
    F_hat = (H_conj * G) / (H_mag_sq + K)

    # Inverse FFT to get restored image
    restored = ifft2_img(F_hat)

    # Ensure real and non-negative
    restored = np.real(restored)
    restored = np.maximum(restored, 0)

    return restored

