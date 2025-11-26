"""
PSF estimation from blurred images using classical techniques.
"""
import numpy as np
import cv2


def estimate_blur_spectrum(img: np.ndarray) -> np.ndarray:
    """
    Estimate blur characteristics from Fourier spectrum.

    Args:
        img: Grayscale blurred image

    Returns:
        Power spectrum of the image
    """
    # Convert to float
    img_float = img.astype(np.float64)

    # Compute FFT
    f_transform = np.fft.fft2(img_float)
    f_shift = np.fft.fftshift(f_transform)

    # Compute power spectrum
    power_spectrum = np.abs(f_shift) ** 2

    return power_spectrum


def estimate_motion_angle(img: np.ndarray) -> float:
    """
    Estimate motion blur angle from image using spectral analysis.

    Uses Fourier transform to detect dominant direction of blur.

    Args:
        img: Grayscale blurred image

    Returns:
        Estimated angle in degrees
    """
    # Get power spectrum
    power_spectrum = estimate_blur_spectrum(img)

    # Apply log transform for better visibility
    power_spectrum_log = np.log1p(power_spectrum)

    # Find the orientation of the spectrum
    # Motion blur creates dark lines perpendicular to motion direction
    h, w = power_spectrum_log.shape
    center_h, center_w = h // 2, w // 2

    # Extract central region
    region_size = min(h, w) // 4
    y1 = center_h - region_size
    y2 = center_h + region_size
    x1 = center_w - region_size
    x2 = center_w + region_size

    central_region = power_spectrum_log[y1:y2, x1:x2]

    # Use Hough transform to detect lines
    central_region_norm = cv2.normalize(central_region, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.Canny(central_region_norm, 50, 150)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=30)

    if lines is not None and len(lines) > 0:
        # Get the most prominent angle (perpendicular to blur)
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle_deg = np.rad2deg(theta)
            # Convert to motion direction (perpendicular)
            motion_angle = (angle_deg + 90) % 180
            angles.append(motion_angle)

        # Return median angle
        return float(np.median(angles))

    # Default angle if estimation fails
    return 0.0


def estimate_motion_length(img: np.ndarray) -> int:
    """
    Estimate motion blur length from image.

    Args:
        img: Grayscale blurred image

    Returns:
        Estimated blur length in pixels
    """
    # Get power spectrum
    power_spectrum = estimate_blur_spectrum(img)

    # Apply log transform
    power_spectrum_log = np.log1p(power_spectrum)

    # Analyze spectrum decay to estimate blur size
    h, w = power_spectrum_log.shape
    center_h, center_w = h // 2, w // 2

    # Sample along horizontal and vertical axes
    horizontal_profile = power_spectrum_log[center_h, :]
    vertical_profile = power_spectrum_log[:, center_w]

    # Find the width of the central peak (inversely related to blur extent)
    # Use autocorrelation to estimate
    def estimate_from_profile(profile):
        # Normalize
        profile = profile - np.min(profile)
        profile = profile / (np.max(profile) + 1e-10)

        # Find first minimum from center
        center = len(profile) // 2
        threshold = 0.1

        for i in range(1, center):
            if profile[center + i] < threshold:
                return i
        return 10  # Default

    h_length = estimate_from_profile(horizontal_profile)
    v_length = estimate_from_profile(vertical_profile)

    # Return average, scaled appropriately
    estimated_length = int((h_length + v_length) / 2)

    # Clamp to reasonable range
    return max(3, min(estimated_length, 50))


def estimate_gaussian_sigma(img: np.ndarray) -> float:
    """
    Estimate Gaussian blur sigma from image.

    Uses Laplacian variance method.

    Args:
        img: Grayscale image

    Returns:
        Estimated sigma value
    """
    # Compute Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Variance of Laplacian (lower = more blurred)
    variance = laplacian.var()

    # Map variance to sigma (empirical relationship)
    # Higher blur -> lower variance -> higher sigma
    if variance < 10:
        sigma = 5.0
    elif variance < 50:
        sigma = 3.0
    elif variance < 100:
        sigma = 2.0
    else:
        sigma = 1.0

    return sigma


def estimate_psf_auto(img: np.ndarray, psf_type: str = "motion") -> np.ndarray:
    """
    Automatically estimate PSF from image.

    Args:
        img: Grayscale blurred image
        psf_type: Type of PSF to estimate ("motion" or "gaussian")

    Returns:
        Estimated PSF kernel
    """
    from .motion_psf import motion_psf
    from .gaussian_psf import gaussian_psf

    if psf_type == "motion":
        angle = estimate_motion_angle(img)
        length = estimate_motion_length(img)
        size = max(31, length * 2 + 1)
        return motion_psf(length, angle, size)

    elif psf_type == "gaussian":
        sigma = estimate_gaussian_sigma(img)
        size = int(6 * sigma + 1)
        if size % 2 == 0:
            size += 1
        return gaussian_psf(size, sigma)

    else:
        raise ValueError(f"Unknown PSF type: {psf_type}")
