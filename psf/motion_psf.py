"""
Motion blur PSF generation.
"""
import numpy as np
import cv2


def motion_psf(length: int, angle: float, size: int) -> np.ndarray:
    """
    Generate a linear motion blur PSF kernel.

    Creates a kernel representing linear motion blur with specified
    length and direction angle.

    Args:
        length: Length of motion blur in pixels
        angle: Angle of motion in degrees (0 = horizontal right)
        size: Size of the output kernel (should be odd and >= length)

    Returns:
        Normalized motion blur PSF kernel
    """
    if size % 2 == 0:
        size += 1  # Ensure odd size

    # Create blank kernel
    kernel = np.zeros((size, size), dtype=np.float64)

    # Center of the kernel
    center = size // 2

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate line endpoints
    dx = np.cos(angle_rad) * length / 2
    dy = np.sin(angle_rad) * length / 2

    # Start and end points
    x1 = int(center - dx)
    y1 = int(center - dy)
    x2 = int(center + dx)
    y2 = int(center + dy)

    # Draw the motion line
    cv2.line(kernel, (x1, y1), (x2, y2), color=1.0, thickness=1)

    # Normalize so sum equals 1
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel = kernel / kernel_sum

    return kernel

