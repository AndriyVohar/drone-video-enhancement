"""
Richardson-Lucy deconvolution implementation (CPU version).
"""
import numpy as np
from utils.fft_tools import psf2otf


def richardson_lucy_fft(image, psf, iterations=10):
    """
    Richardson-Lucy deconvolution using FFT for speed (CPU version).

    Args:
        image: Blurred input image (2D or 3D numpy array, float)
        psf: Point spread function (2D array)
        iterations: Number of iterations

    Returns:
        Deblurred image (numpy array, float)
    """
    # Handle color images by processing each channel separately
    if len(image.shape) == 3:
        # Color image - process each channel
        result = np.zeros_like(image)
        for channel in range(image.shape[2]):
            result[:, :, channel] = richardson_lucy_fft(image[:, :, channel], psf, iterations)
        return result

    # Grayscale image processing
    # Initialize estimate with blurred image
    estimate = np.copy(image)

    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    estimate = np.maximum(estimate, epsilon)

    # Precompute PSF and flipped PSF in frequency domain
    otf = psf2otf(psf, image.shape)
    psf_flipped = np.flip(psf)
    otf_flipped = psf2otf(psf_flipped, image.shape)

    # Richardson-Lucy iterations
    for i in range(iterations):
        # Convolve estimate with PSF (using FFT)
        estimate_fft = np.fft.fft2(estimate)
        blurred_estimate_fft = estimate_fft * otf
        blurred_estimate = np.real(np.fft.ifft2(blurred_estimate_fft))
        blurred_estimate = np.maximum(blurred_estimate, epsilon)

        # Compute ratio
        ratio = image / blurred_estimate

        # Convolve ratio with flipped PSF (using FFT)
        ratio_fft = np.fft.fft2(ratio)
        correction_fft = ratio_fft * otf_flipped
        correction = np.real(np.fft.ifft2(correction_fft))

        # Update estimate
        estimate = estimate * correction
        estimate = np.maximum(estimate, epsilon)

    return estimate
"""
Wiener deconvolution implementation (CPU version).
"""
import numpy as np
from utils.fft_tools import psf2otf


def wiener_deconvolution(image, psf, K=0.01):
    """
    Wiener deconvolution for image deblurring (CPU version).

    Args:
        image: Blurred input image (2D numpy array, float)
        psf: Point spread function
        K: Noise-to-signal power ratio (regularization parameter)

    Returns:
        Deblurred image (2D numpy array, float)
    """
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
