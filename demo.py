"""
Demo script to test individual components and create example outputs.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from psf.gaussian_psf import gaussian_psf
from psf.motion_psf import motion_psf
from deblurring.wiener import wiener_deconvolution
from deblurring.tikhonov import tikhonov_deconvolution
from deblurring.richardson_lucy import richardson_lucy_fft
from preprocessing.denoise import denoise_nlm
from utils.image_ops import to_float, from_float


def create_synthetic_blur(img: np.ndarray, psf_type: str = "motion") -> tuple:
    """
    Create synthetic blurred image for testing.

    Args:
        img: Original sharp image
        psf_type: Type of blur to apply

    Returns:
        Tuple of (blurred_image, psf_used)
    """
    from scipy import signal

    img_float = to_float(img)

    if psf_type == "motion":
        psf = motion_psf(length=20, angle=30, size=41)
    else:
        psf = gaussian_psf(size=15, sigma=3.0)

    # Apply blur using convolution
    blurred = signal.convolve2d(img_float, psf, mode='same', boundary='symm')

    # Add small amount of noise
    noise = np.random.normal(0, 0.01, blurred.shape)
    blurred_noisy = blurred + noise

    # Clip to valid range
    blurred_noisy = np.clip(blurred_noisy, 0, 1)

    return blurred_noisy, psf


def demo_deblurring_comparison(image_path: str = None):
    """
    Compare different deblurring methods on a test image.

    Args:
        image_path: Path to test image, or None to create synthetic
    """
    print("=== Deblurring Methods Comparison ===\n")

    # Load or create test image
    if image_path and Path(image_path).exists():
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"Loaded image: {image_path}")

        # Create synthetic blur
        blurred, psf = create_synthetic_blur(img, "motion")
        original = to_float(img)
    else:
        print("Creating synthetic test image...")
        # Create simple test pattern
        img = np.zeros((256, 256), dtype=np.uint8)
        cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
        cv2.circle(img, (128, 128), 40, 128, -1)

        blurred, psf = create_synthetic_blur(img, "motion")
        original = to_float(img)

    print(f"Image size: {blurred.shape}")
    print(f"PSF size: {psf.shape}\n")

    # Denoise
    blurred_uint8 = from_float(blurred)
    denoised = denoise_nlm(blurred_uint8, h=10)
    blurred_denoised = to_float(denoised)

    # Apply different deblurring methods
    print("Applying Wiener deconvolution...")
    result_wiener = wiener_deconvolution(blurred_denoised, psf, K=0.01)

    print("Applying Tikhonov regularization...")
    result_tikhonov = tikhonov_deconvolution(blurred_denoised, psf, alpha=0.01)

    print("Applying Richardson-Lucy (10 iterations)...")
    result_rl = richardson_lucy_fft(blurred_denoised, psf, iterations=10)

    print("\nResults ready!")

    # Save results
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)

    cv2.imwrite(str(output_dir / "original.png"), from_float(original))
    cv2.imwrite(str(output_dir / "blurred.png"), from_float(blurred))
    cv2.imwrite(str(output_dir / "denoised.png"), denoised)
    cv2.imwrite(str(output_dir / "wiener.png"), from_float(result_wiener))
    cv2.imwrite(str(output_dir / "tikhonov.png"), from_float(result_tikhonov))
    cv2.imwrite(str(output_dir / "richardson_lucy.png"), from_float(result_rl))

    # Save PSF visualization
    psf_vis = (psf - psf.min()) / (psf.max() - psf.min() + 1e-10)
    cv2.imwrite(str(output_dir / "psf.png"), (psf_vis * 255).astype(np.uint8))

    print(f"\nResults saved to: {output_dir}/")
    print("Files:")
    print("  - original.png")
    print("  - blurred.png")
    print("  - denoised.png")
    print("  - wiener.png")
    print("  - tikhonov.png")
    print("  - richardson_lucy.png")
    print("  - psf.png")


def demo_psf_visualization():
    """Create visualizations of different PSF types."""
    print("\n=== PSF Visualization ===\n")

    output_dir = Path("demo_output/psf_types")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Motion PSFs with different angles
    print("Generating motion PSFs...")
    for angle in [0, 45, 90, 135]:
        psf = motion_psf(length=25, angle=angle, size=51)
        psf_vis = (psf - psf.min()) / (psf.max() - psf.min() + 1e-10)
        cv2.imwrite(str(output_dir / f"motion_angle_{angle}.png"),
                   (psf_vis * 255).astype(np.uint8))

    # Gaussian PSFs with different sigmas
    print("Generating Gaussian PSFs...")
    for sigma in [1.0, 3.0, 5.0, 7.0]:
        psf = gaussian_psf(size=51, sigma=sigma)
        psf_vis = (psf - psf.min()) / (psf.max() - psf.min() + 1e-10)
        cv2.imwrite(str(output_dir / f"gaussian_sigma_{sigma:.1f}.png"),
                   (psf_vis * 255).astype(np.uint8))

    print(f"PSF visualizations saved to: {output_dir}/\n")


def demo_parameter_sensitivity():
    """
    Demonstrate sensitivity to regularization parameters.
    """
    print("\n=== Parameter Sensitivity Demo ===\n")

    # Create test image
    img = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (200, 200), 255, -1)
    cv2.circle(img, (128, 128), 40, 128, -1)

    # Create blur
    blurred, psf = create_synthetic_blur(img, "motion")

    output_dir = Path("demo_output/parameters")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test different K values for Wiener
    print("Testing Wiener filter with different K values...")
    k_values = [0.001, 0.01, 0.1, 1.0]

    for k in k_values:
        result = wiener_deconvolution(blurred, psf, K=k)
        cv2.imwrite(str(output_dir / f"wiener_K_{k:.3f}.png"),
                   from_float(result))

    # Test different iterations for Richardson-Lucy
    print("Testing Richardson-Lucy with different iterations...")
    iterations = [5, 10, 20, 50]

    for iters in iterations:
        result = richardson_lucy_fft(blurred, psf, iterations=iters)
        cv2.imwrite(str(output_dir / f"rl_iter_{iters}.png"),
                   from_float(result))

    print(f"Parameter comparison saved to: {output_dir}/\n")


def main():
    """Run all demos."""
    print("=" * 60)
    print("CLASSICAL IMAGE DEBLURRING - DEMO SCRIPT")
    print("=" * 60)
    print()

    # Demo 1: Comparison of methods
    demo_deblurring_comparison()

    # Demo 2: PSF visualization
    demo_psf_visualization()

    # Demo 3: Parameter sensitivity
    demo_parameter_sensitivity()

    print("=" * 60)
    print("All demos complete!")
    print("Check the 'demo_output' directory for results.")

