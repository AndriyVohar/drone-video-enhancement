"""
Gaussian PSF (Point Spread Function) generation.
"""
import numpy as np


def gaussian_psf(size=(15, 15), sigma=5.0):
    """
    Generate a Gaussian point spread function.

    Args:
        size: PSF size (height, width)
        sigma: Standard deviation of Gaussian

    Returns:
        2D Gaussian PSF (normalized)
    """
    if isinstance(size, int):
        size = (size, size)

    height, width = size
    center_y, center_x = height // 2, width // 2

    # Create coordinate grids
    y, x = np.ogrid[:height, :width]

    # Gaussian formula
    psf = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # Normalize so sum equals 1
    psf /= np.sum(psf)

    return psf.astype(np.float32)
