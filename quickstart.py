"""
Quick start script for testing the deblurring pipeline without a video.

This script creates a synthetic blurred image and demonstrates the deblurring.
No video file required - perfect for testing the installation.
"""
import cv2
import numpy as np
from pathlib import Path

from psf.motion_psf import motion_psf
from psf.gaussian_psf import gaussian_psf
from deblurring.wiener import wiener_deconvolution
from utils.image_ops import to_float, from_float


def create_test_image():
    """Create a simple test image with text and shapes."""
    img = np.ones((512, 512), dtype=np.uint8) * 255

    # Add some shapes
    cv2.rectangle(img, (50, 50), (250, 250), 0, 3)
    cv2.circle(img, (400, 150), 80, 0, 3)
    cv2.line(img, (50, 350), (450, 350), 0, 3)

    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'DEBLUR TEST', (80, 450), font, 1.5, 0, 3)

    return img


def apply_blur(img, blur_type='motion'):
    """Apply synthetic blur to image."""
    from scipy import signal

    img_float = to_float(img)

    if blur_type == 'motion':
        psf = motion_psf(length=25, angle=45, size=51)
    else:
        psf = gaussian_psf(size=21, sigma=4.0)

    # Convolve
    blurred = signal.convolve2d(img_float, psf, mode='same', boundary='symm')

    # Add noise
    noise = np.random.normal(0, 0.02, blurred.shape)
    blurred = np.clip(blurred + noise, 0, 1)

    return blurred, psf


def main():
    print("=" * 60)
    print("QUICK START - Classical Image Deblurring")
    print("=" * 60)
    print()

    # Create output directory
    output_dir = Path("quickstart_output")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Create test image
    print("Step 1: Creating test image...")
    original = create_test_image()
    cv2.imwrite(str(output_dir / "1_original.png"), original)
    print(f"  ✓ Saved: {output_dir}/1_original.png")

    # Step 2: Apply motion blur
    print("\nStep 2: Applying motion blur...")
    blurred, psf = apply_blur(original, 'motion')
    cv2.imwrite(str(output_dir / "2_blurred.png"), from_float(blurred))
    print(f"  ✓ Saved: {output_dir}/2_blurred.png")

    # Step 3: Deblur with Wiener filter
    print("\nStep 3: Deblurring with Wiener filter...")
    deblurred = wiener_deconvolution(blurred, psf, K=0.01)
    cv2.imwrite(str(output_dir / "3_deblurred.png"), from_float(deblurred))
    print(f"  ✓ Saved: {output_dir}/3_deblurred.png")

    # Visualize PSF
    psf_vis = (psf / psf.max() * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / "psf_used.png"), psf_vis)

    print("\n" + "=" * 60)
    print("SUCCESS! Quick start complete.")
    print("=" * 60)
    print(f"\nCheck the '{output_dir}' directory to see:")
    print("  1. Original sharp image")
    print("  2. Synthetically blurred image")
    print("  3. Deblurred result using Wiener filter")
    print("\nThe deblurring works! You can now try it on real videos.")
    print("=" * 60)


if __name__ == "__main__":
    main()
