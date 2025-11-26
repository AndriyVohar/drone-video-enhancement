"""
Tikhonov regularization deconvolution.
"""
import numpy as np
from utils.fft_tools import fft2_img, ifft2_img, psf_to_otf


def tikhonov_deconvolution(img: np.ndarray, psf: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Apply zero-order Tikhonov regularization for image deblurring.

    Zero-order Tikhonov in frequency domain:
    F_hat = (H* · G) / (|H|^2 + alpha)

    This is equivalent to Wiener filter when alpha represents regularization strength.

    Args:
        img: Blurred input image (grayscale, float)
        psf: Point Spread Function kernel
        alpha: Regularization parameter (controls smoothness)

    Returns:
        Deblurred image
    """
    # Ensure image is float
    img = img.astype(np.float64)

    # Get image dimensions
    img_shape = img.shape

    # Convert PSF to OTF
    H = psf_to_otf(psf, img_shape)

    # Compute FFT of blurred image
    G = fft2_img(img)

    # Tikhonov filter (zero-order)
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2

    F_hat = (H_conj * G) / (H_mag_sq + alpha)

    # Inverse FFT
    restored = ifft2_img(F_hat)

    # Ensure real and non-negative
    restored = np.real(restored)
    restored = np.maximum(restored, 0)

    return restored


def tikhonov_gradient(img: np.ndarray, psf: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Apply gradient-based Tikhonov regularization (first-order).

    This minimizes: ||Hf - g||^2 + alpha * ||∇f||^2

    In frequency domain:
    F_hat = (H* · G) / (|H|^2 + alpha * |L|^2)

    where L is the Laplacian operator in frequency domain.

    Args:
        img: Blurred input image (grayscale, float)
        psf: Point Spread Function kernel
        alpha: Regularization parameter

    Returns:
        Deblurred image
    """
    # Ensure image is float
    img = img.astype(np.float64)

    # Get image dimensions
    img_shape = img.shape
    h, w = img_shape

    # Convert PSF to OTF
    H = psf_to_otf(psf, img_shape)

    # Compute FFT of blurred image
    G = fft2_img(img)

    # Create Laplacian operator in frequency domain
    # Laplacian kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
    laplacian = np.array([[0, -1, 0],
                          [-1, 4, -1],
                          [0, -1, 0]], dtype=np.float64)

    L = psf_to_otf(laplacian, img_shape)
    L_mag_sq = np.abs(L) ** 2

    # First-order Tikhonov filter
    H_conj = np.conj(H)
    H_mag_sq = np.abs(H) ** 2

    F_hat = (H_conj * G) / (H_mag_sq + alpha * L_mag_sq)

    # Inverse FFT
    restored = ifft2_img(F_hat)

    # Ensure real and non-negative
    restored = np.real(restored)
    restored = np.maximum(restored, 0)

    return restored

