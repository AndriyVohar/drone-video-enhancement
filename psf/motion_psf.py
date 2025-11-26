"""
Motion blur PSF (Point Spread Function) generation.
"""
import numpy as np
import cv2


def motion_psf(length, angle, size=(15, 15)):
    """
    Generate a motion blur point spread function.

    Args:
        length: Length of motion blur in pixels
        angle: Angle of motion in degrees
        size: PSF size (height, width)

    Returns:
        2D motion blur PSF (normalized)
    """
    # Create PSF
    psf = np.zeros(size, dtype=np.float32)

    # Center of PSF
    center_y, center_x = size[0] // 2, size[1] // 2

    # Convert angle to radians
    angle_rad = np.deg2rad(angle)

    # Calculate line endpoints
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Draw line representing motion
    half_length = length / 2.0

    x1 = int(center_x - half_length * cos_angle)
    y1 = int(center_y - half_length * sin_angle)
    x2 = int(center_x + half_length * cos_angle)
    y2 = int(center_y + half_length * sin_angle)

    # Draw the line
    cv2.line(psf, (x1, y1), (x2, y2), 1.0, 1)

    # Normalize so sum equals 1
    psf = psf / np.sum(psf)

    return psf

