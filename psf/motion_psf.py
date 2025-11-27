"""
Motion blur PSF generation
"""
import numpy as np


def create_motion_psf(length, angle, size=None):
    """
    Create motion blur point spread function (PSF).

    Args:
        length: Length of motion blur in pixels
        angle: Angle of motion in degrees
        size: Size of PSF kernel (height, width). If None, auto-calculated

    Returns:
        Motion blur PSF (normalized 2D array)
    """
    if size is None:
        size = (max(15, int(length) + 4), max(15, int(length) + 4))

    psf = np.zeros(size)
    center = (size[0] // 2, size[1] // 2)

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate motion trajectory
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Draw line representing motion
    for i in range(int(length)):
        offset = i - length / 2
        y = int(center[0] + offset * sin_angle)
        x = int(center[1] + offset * cos_angle)

        if 0 <= y < size[0] and 0 <= x < size[1]:
            psf[y, x] = 1

    # Normalize
    psf_sum = np.sum(psf)
    if psf_sum > 0:
        psf /= psf_sum
    else:
        # Fallback: single pixel at center
        psf[center] = 1.0

    return psf
