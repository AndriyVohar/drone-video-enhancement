"""
Wiener deconvolution implementation (CPU version).
"""
import numpy as np
from utils.fft_tools import psf2otf


def wiener_deconvolution(image, psf, K=0.01):
    """
    Wiener deconvolution for image deblurring (CPU version).

    Args:
        image: Blurred input image (2D or 3D numpy array, float)
        psf: Point spread function (2D array)
        K: Noise-to-signal power ratio (regularization parameter)

    Returns:
        Deblurred image (numpy array, float)
    """
    # Handle color images by processing each channel separately
    if len(image.shape) == 3:
        # Color image - process each channel
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            result[:, :, channel] = wiener_deconvolution(image[:, :, channel], psf, K)
        return result

    # Grayscale image processing
    # Convert PSF to OTF
    otf = psf2otf(psf, image.shape)

    # FFT of blurred image
    image_fft = np.fft.fft2(image)

    # Wiener filter
    otf_conj = np.conj(otf)
    otf_abs_sq = np.abs(otf) ** 2

    # H* / (|H|^2 + K)
    wiener_filter = otf_conj / (otf_abs_sq + K)

    # Apply filter in frequency domain
    result_fft = image_fft * wiener_filter

    # Inverse FFT
    result = np.fft.ifft2(result_fft)

    # Return real part
    return np.real(result)
