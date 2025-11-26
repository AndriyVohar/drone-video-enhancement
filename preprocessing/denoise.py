"""
Classical denoising methods using OpenCV.
"""
import numpy as np
import cv2


def denoise_gaussian(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur denoising.

    Simple Gaussian smoothing to reduce noise.

    Args:
        img: Input image (grayscale or color)
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Denoised image
    """
    # Calculate kernel size from sigma (6*sigma rule)
    ksize = int(6 * sigma + 1)
    if ksize % 2 == 0:
        ksize += 1

    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def denoise_bilateral(img: np.ndarray, d: int = 9,
                      sigma_color: float = 75,
                      sigma_space: float = 75) -> np.ndarray:
    """
    Apply bilateral filtering for edge-preserving denoising.

    Bilateral filter smooths images while preserving edges by considering
    both spatial and intensity differences.

    Args:
        img: Input image (grayscale or color)
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color/intensity space
        sigma_space: Filter sigma in the coordinate space

    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)


def denoise_nlm(img: np.ndarray, h: float = 10,
                patch_size: int = 7,
                search_size: int = 21) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising.

    NLM is a powerful denoising algorithm that replaces each pixel with
    a weighted average of pixels with similar neighborhoods.

    Args:
        img: Input image (grayscale or color)
        h: Filter strength (higher h removes more noise but also removes detail)
        patch_size: Size of patches used for denoising (should be odd)
        search_size: Size of area where search is performed (should be odd)

    Returns:
        Denoised image
    """
    # Check if grayscale or color
    if len(img.shape) == 2 or img.shape[2] == 1:
        # Grayscale
        return cv2.fastNlMeansDenoising(img, None, h, patch_size, search_size)
    else:
        # Color
        return cv2.fastNlMeansDenoisingColored(img, None, h, h, patch_size, search_size)


def denoise_median(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply median filtering for salt-and-pepper noise removal.

    Args:
        img: Input image
        ksize: Kernel size (must be odd)

    Returns:
        Denoised image
    """
    if ksize % 2 == 0:
        ksize += 1

    return cv2.medianBlur(img, ksize)


def denoise_morphological(img: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Apply morphological opening for noise removal.

    Useful for removing small noise artifacts.

    Args:
        img: Input image
        ksize: Kernel size

    Returns:
        Denoised image
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

