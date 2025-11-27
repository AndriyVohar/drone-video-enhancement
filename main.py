"""
Main video deblurring and enhancement pipeline.

This script processes drone videos using classical image processing techniques:
- Video stabilization
- Denoising
- PSF-based deblurring (Wiener, Tikhonov, Richardson-Lucy)
- Contrast enhancement

No AI/ML methods are used - only classical DSP and computer vision.
"""
import os
import cv2
import numpy as np
from typing import Optional

import config
from utils.video_io import iterate_frames, get_video_info, VideoWriter
from utils.image_ops import to_gray, to_float, from_float
from preprocessing.denoise import denoise_gaussian, denoise_bilateral, denoise_nlm
from preprocessing.stabilization import stabilize_frame
from psf.gaussian_psf import gaussian_psf
from psf.motion_psf import motion_psf
from psf.estimate_psf import estimate_psf_auto
from deblurring.wiener import wiener_deconvolution
from deblurring.tikhonov import tikhonov_deconvolution, tikhonov_gradient
from deblurring.richardson_lucy import richardson_lucy_fft


def apply_denoising(frame: np.ndarray, method: str) -> np.ndarray:
    """
    Apply denoising to frame based on selected method.

    Args:
        frame: Input frame
        method: Denoising method name

    Returns:
        Denoised frame
    """
    if method == "gaussian":
        return denoise_gaussian(frame, config.GAUSSIAN_SIGMA)
    elif method == "bilateral":
        return denoise_bilateral(frame, config.BILATERAL_D,
                                config.BILATERAL_SIGMA_COLOR,
                                config.BILATERAL_SIGMA_SPACE)
    elif method == "nlm":
        return denoise_nlm(frame, config.NLM_H,
                          config.NLM_PATCH_SIZE,
                          config.NLM_SEARCH_SIZE)
    else:
        return frame


def get_psf(frame: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate or estimate PSF based on configuration.

    Args:
        frame: Optional frame for PSF estimation

    Returns:
        PSF kernel
    """
    if config.PSF_TYPE == "motion":
        return motion_psf(config.MOTION_LENGTH, config.MOTION_ANGLE, config.PSF_SIZE)
    elif config.PSF_TYPE == "gaussian":
        return gaussian_psf(config.GAUSSIAN_PSF_SIZE, config.GAUSSIAN_PSF_SIGMA)
    elif config.PSF_TYPE == "estimate" and frame is not None:
        # Estimate PSF from the frame
        return estimate_psf_auto(frame, "motion")
    else:
        # Default to motion PSF
        return motion_psf(15, 45, 31)


def apply_deblurring(frame: np.ndarray, psf: np.ndarray, method: str) -> np.ndarray:
    """
    Apply deblurring based on selected method.

    Args:
        frame: Blurred frame (float, grayscale)
        psf: Point Spread Function
        method: Deblurring method name

    Returns:
        Deblurred frame
    """
    if method == "wiener":
        return wiener_deconvolution(frame, psf, config.WIENER_K)
    elif method == "tikhonov":
        return tikhonov_deconvolution(frame, psf, config.TIKHONOV_ALPHA)
    elif method == "tikhonov_gradient":
        return tikhonov_gradient(frame, psf, config.TIKHONOV_ALPHA)
    elif method == "richardson_lucy":
        return richardson_lucy_fft(frame, psf, config.RL_ITERATIONS)
    else:
        return frame


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization.

    Args:
        frame: Input frame (uint8)

    Returns:
        Enhanced frame
    """
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                           tileGridSize=config.CLAHE_TILE_GRID_SIZE)
    
    if len(frame.shape) == 2:
        # Grayscale
        return clahe.apply(frame)
    else:
        # Color - apply to each channel
        result = np.zeros_like(frame)
        for i in range(frame.shape[2]):
            result[:, :, i] = clahe.apply(frame[:, :, i])
        return result


def process_video(input_path: str, output_path: str) -> None:
    """
    Process video with deblurring and enhancement pipeline.
    Memory-efficient streaming implementation.

    Pipeline:
    1. Read frame
    2. Convert to grayscale (if configured)
    3. Stabilize (if enabled)
    4. Denoise
    5. Estimate or use predefined PSF
    6. Deblur using classical algorithm
    7. Enhance with CLAHE (if enabled)
    8. Save frame
    
    Args:
        input_path: Path to input video
        output_path: Path to output video
    """
    print("=" * 60)
    print("CLASSICAL VIDEO DEBLURRING & ENHANCEMENT")
    print("=" * 60)
    
    # Get video info
    try:
        video_info = get_video_info(input_path)
        print(f"Input video: {input_path}")
        print(f"Resolution: {video_info['width']}x{video_info['height']}")
        print(f"FPS: {video_info['fps']}")
        print(f"Total frames: {video_info['frame_count']}")
    except Exception as e:
        print(f"Error reading video info: {e}")
        return
    
    print("\nProcessing settings:")
    print(f"  Grayscale: {config.CONVERT_TO_GRAYSCALE}")
    print(f"  Stabilization: {config.ENABLE_STABILIZATION}")
    print(f"  Denoising: {config.DENOISE_METHOD}")
    print(f"  PSF type: {config.PSF_TYPE}")
    print(f"  Deblur method: {config.DEBLUR_METHOD}")
    print(f"  CLAHE: {config.APPLY_CLAHE}")
    print(f"  Memory-efficient streaming: ✓ ENABLED")
    print()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    # Initialize variables
    prev_frame = None
    psf = None
    video_writer = None

    try:
        for frame_num, frame in iterate_frames(input_path):
            print(f"Processing frame {frame_num + 1}/{video_info['frame_count']}...", end='\r')

            # Convert to grayscale if configured
            if config.CONVERT_TO_GRAYSCALE:
                frame_gray = to_gray(frame)
            else:
                frame_gray = frame

            # Stabilization
            if config.ENABLE_STABILIZATION and prev_frame is not None:
                frame_stabilized = stabilize_frame(prev_frame, frame_gray)
            else:
                frame_stabilized = frame_gray

            # Denoising
            frame_denoised = apply_denoising(frame_stabilized, config.DENOISE_METHOD)

            # Convert to float for deblurring
            frame_float = to_float(frame_denoised)

            # Get PSF (generate once or estimate per frame)
            if psf is None or config.PSF_TYPE == "estimate":
                psf = get_psf(frame_float)

            # Deblurring
            frame_deblurred = apply_deblurring(frame_float, psf, config.DEBLUR_METHOD)

            # Convert back to uint8
            frame_restored = from_float(frame_deblurred)

            # CLAHE enhancement
            if config.APPLY_CLAHE:
                frame_enhanced = apply_clahe(frame_restored)
            else:
                frame_enhanced = frame_restored

            # Initialize video writer on first frame
            if video_writer is None:
                height, width = frame_enhanced.shape[:2]
                is_color = len(frame_enhanced.shape) == 3
                video_writer = VideoWriter(
                    output_path,
                    video_info['fps'],
                    (width, height),
                    is_color
                )

            # Write frame immediately (no buffering!)
            video_writer.write_frame(frame_enhanced)

            # Update previous frame for stabilization (minimal memory)
            if config.ENABLE_STABILIZATION:
                prev_frame = frame_gray.copy()

            # Clean up intermediate arrays
            del frame_denoised, frame_float, frame_deblurred, frame_restored

        print()
        print(f"\nProcessed {video_info['frame_count']} frames")

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_path}")

    print("Processing complete!")
    print("=" * 60)


def process_single_frame(frame: np.ndarray, 
                        psf_type: str = "motion",
                        deblur_method: str = "wiener") -> np.ndarray:
    """
    Process a single frame with the deblurring pipeline.
    
    Useful for testing and experimentation.
    
    Args:
        frame: Input frame
        psf_type: PSF type to use
        deblur_method: Deblurring method
        
    Returns:
        Processed frame
    """
    # Convert to grayscale
    frame_gray = to_gray(frame)
    
    # Denoise
    frame_denoised = denoise_nlm(frame_gray)
    
    # Convert to float
    frame_float = to_float(frame_denoised)
    
    # Get PSF
    if psf_type == "motion":
        psf = motion_psf(15, 45, 31)
    elif psf_type == "gaussian":
        psf = gaussian_psf(15, 3.0)
    else:
        psf = estimate_psf_auto(frame_float, "motion")
    
    # Deblur
    if deblur_method == "wiener":
        frame_deblurred = wiener_deconvolution(frame_float, psf, 0.01)
    elif deblur_method == "tikhonov":
        frame_deblurred = tikhonov_deconvolution(frame_float, psf, 0.01)
    else:
        frame_deblurred = richardson_lucy_fft(frame_float, psf, 10)
    
    # Convert back to uint8
    frame_restored = from_float(frame_deblurred)
    
    # Apply CLAHE
    frame_enhanced = apply_clahe(frame_restored)
    
    return frame_enhanced


def main():
    """Main entry point."""
    input_path = config.INPUT_VIDEO_PATH
    output_path = config.OUTPUT_VIDEO_PATH
    
    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}")
        print("\nPlease update the INPUT_VIDEO_PATH in config.py")
        print("or place your video at the specified location.")
        return
    
    process_video(input_path, output_path)


if __name__ == "__main__":
    main()
