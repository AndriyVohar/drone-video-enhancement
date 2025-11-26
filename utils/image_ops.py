"""
Image operation utilities.
"""
import numpy as np
import cv2


def to_gray(image):
    """Convert image to grayscale if it's not already."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def to_float(image):
    """Convert image to float [0, 1] range."""
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def from_float(image):
    """Convert float image back to uint8 [0, 255] range."""
    # Clip to valid range
    image = np.clip(image, 0, 1)
    return (image * 255).astype(np.uint8)

