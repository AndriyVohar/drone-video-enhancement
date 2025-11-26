"""
Tikhonov regularization deconvolution (CPU version).
"""
import numpy as np
from utils.fft_tools import psf2otf


def tikhonov_deconvolution(image, psf, alpha=0.01):
    """
    Tikhonov regularization deconvolution for image deblurring (CPU version).

    Args:
        image: Blurred input image (2D or 3D numpy array, float)
        psf: Point spread function (2D array)
        alpha: Regularization parameter

    Returns:
        Deblurred image (numpy array, float)
    """
    # Handle color images by processing each channel separately
    if len(image.shape) == 3:
        # Color image - process each channel
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            result[:, :, channel] = tikhonov_deconvolution(image[:, :, channel], psf, alpha)
        return result

    # Grayscale image processing
    # Convert PSF to OTF
    otf = psf2otf(psf, image.shape)

    # FFT of blurred image
    image_fft = np.fft.fft2(image)

    # Tikhonov filter: H* / (|H|^2 + alpha)
    otf_conj = np.conj(otf)
    otf_abs_sq = np.abs(otf) ** 2

    tikhonov_filter = otf_conj / (otf_abs_sq + alpha)

    # Apply filter in frequency domain
    result_fft = image_fft * tikhonov_filter

    # Inverse FFT
    result = np.fft.ifft2(result_fft)

    # Return real part
    return np.real(result)

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
