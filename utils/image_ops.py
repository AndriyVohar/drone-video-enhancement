"""
Basic image operations and utilities.
"""
import numpy as np
import cv2


def normalize(img: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Args:
        img: Input image array

    Returns:
        Normalized image in [0, 1] range
    """
    img = img.astype(np.float64)
    img_min = np.min(img)
    img_max = np.max(img)

    if img_max - img_min < 1e-10:
        return np.zeros_like(img)

    return (img - img_min) / (img_max - img_min)


def clip_uint8(img: np.ndarray) -> np.ndarray:
    """
    Clip image values to [0, 255] and convert to uint8.

    Args:
        img: Input image array

    Returns:
        Clipped uint8 image
    """
    return np.clip(img, 0, 255).astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    """
    Convert image to grayscale if it's in color.

    Args:
        img: Input image (can be grayscale or color)

    Returns:
        Grayscale image
    """
    if len(img.shape) == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def to_float(img: np.ndarray) -> np.ndarray:
    """
    Convert image to float64 in [0, 1] range.

    Args:
        img: Input image

    Returns:
        Float64 image in [0, 1]
    """
    return img.astype(np.float64) / 255.0 if img.dtype == np.uint8 else img.astype(np.float64)


def from_float(img: np.ndarray) -> np.ndarray:
    """
    Convert float image [0, 1] back to uint8 [0, 255].

    Args:
        img: Float image in [0, 1]

    Returns:
        Uint8 image
    """
    return clip_uint8(img * 255.0)
