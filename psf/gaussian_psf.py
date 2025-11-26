"""
Gaussian PSF (Point Spread Function) generation.
"""
import numpy as np
import cv2


def gaussian_psf(size=(15, 15), sigma=5.0):
    """
    Generate a Gaussian point spread function.

    Args:
        size: PSF size (height, width)
        sigma: Standard deviation of Gaussian

    Returns:
        2D Gaussian PSF (normalized)
    """
    # Create coordinate grids
    h, w = size
    center_y, center_x = h // 2, w // 2

    y, x = np.ogrid[:h, :w]

    # Gaussian formula
    psf = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # Normalize so sum equals 1
    psf = psf / np.sum(psf)

    return psf.astype(np.float32)

