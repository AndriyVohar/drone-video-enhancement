"""
Denoising functions for preprocessing.
"""
import cv2
import numpy as np


def denoise_gaussian(image, sigma=1.5):
    """
    Apply Gaussian blur denoising.

    Args:
        image: Input image
        sigma: Standard deviation for Gaussian kernel

    Returns:
        Denoised image
    """
    return cv2.GaussianBlur(image, (0, 0), sigma)


def denoise_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """
    Apply bilateral filter denoising.

    Args:
        image: Input image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space

    Returns:
        Denoised image
    """
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_nlm(image, h=10, patch_size=7, search_size=21):
    """
    Apply Non-Local Means denoising.

    Args:
        image: Input image
        h: Filter strength
        patch_size: Size of patches used for denoising
        search_size: Size of area where search is performed

    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        # Color image
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, patch_size, search_size)
    else:
        # Grayscale image
        return cv2.fastNlMeansDenoising(image, None, h, patch_size, search_size)

