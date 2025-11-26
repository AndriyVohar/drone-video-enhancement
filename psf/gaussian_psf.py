"""
Gaussian PSF generation.
"""
import numpy as np


def gaussian_psf(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian Point Spread Function.

    The Gaussian is defined as:
    G(x, y) = (1 / (2π σ²)) * exp(-(x² + y²) / (2σ²))

    Args:
        size: Size of the kernel (should be odd)
        sigma: Standard deviation of the Gaussian

    Returns:
        Normalized Gaussian PSF kernel
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size

    # Create coordinate grid centered at 0
    ax = np.arange(-size // 2 + 1, size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)

    # Compute Gaussian
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    # Normalize so sum equals 1
    kernel = kernel / np.sum(kernel)

    return kernel

