"""
Main video deblurring with GPU acceleration support.

GPU-accelerated classical image processing - NO AI/ML!
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

# Try to import GPU versions
try:
    from utils.gpu_utils import GPU_AVAILABLE, get_device_info
    from deblurring.wiener_gpu import wiener_deconvolution_gpu
    from deblurring.richardson_lucy_gpu import richardson_lucy_fft_gpu
    GPU_SUPPORT = True
except ImportError:
    GPU_SUPPORT = False
    GPU_AVAILABLE = False
    print("⚠ GPU modules not available, using CPU only")

# Fallback to CPU versions
from deblurring.wiener import wiener_deconvolution
from deblurring.tikhonov import tikhonov_deconvolution, tikhonov_gradient
from deblurring.richardson_lucy import richardson_lucy_fft


def apply_denoising(frame: np.ndarray, method: str) -> np.ndarray:
    """Apply denoising to frame based on selected method."""
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
    """Generate or estimate PSF based on configuration."""
    if config.PSF_TYPE == "motion":
        return motion_psf(config.MOTION_LENGTH, config.MOTION_ANGLE, config.PSF_SIZE)
    elif config.PSF_TYPE == "gaussian":
        return gaussian_psf(config.GAUSSIAN_PSF_SIZE, config.GAUSSIAN_PSF_SIGMA)
    elif config.PSF_TYPE == "estimate" and frame is not None:
        return estimate_psf_auto(frame, "motion")
    else:
        return motion_psf(15, 45, 31)


def apply_deblurring(frame: np.ndarray, psf: np.ndarray, method: str,
                     use_gpu: bool = False) -> np.ndarray:
    """
    Apply deblurring with GPU acceleration if available.

    Args:
        frame: Blurred frame (float, grayscale)
        psf: Point Spread Function
        method: Deblurring method name
        use_gpu: Whether to use GPU acceleration

    Returns:
        Deblurred frame
    """
    # GPU-accelerated methods
    if use_gpu and GPU_AVAILABLE and GPU_SUPPORT:
        if method == "wiener":
            return wiener_deconvolution_gpu(frame, psf, config.WIENER_K, use_gpu=True)
        elif method == "richardson_lucy":
            return richardson_lucy_fft_gpu(frame, psf, config.RL_ITERATIONS, use_gpu=True)

    # CPU fallback or non-GPU methods
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
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                           tileGridSize=config.CLAHE_TILE_GRID_SIZE)

    if len(frame.shape) == 2:
        return clahe.apply(frame)
    else:
        result = np.zeros_like(frame)
        for i in range(frame.shape[2]):
            result[:, :, i] = clahe.apply(frame[:, :, i])
        return result


def process_video(input_path: str, output_path: str, use_gpu: bool = None) -> None:
    """
    Process video with GPU-accelerated deblurring pipeline.
    Memory-efficient streaming implementation.

    Args:
        input_path: Path to input video
        output_path: Path to output video
        use_gpu: Whether to use GPU (None = auto-detect from config)
    """
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = config.USE_GPU and GPU_AVAILABLE

    print("=" * 60)
    print("CLASSICAL VIDEO DEBLURRING & ENHANCEMENT")
    if use_gpu and GPU_AVAILABLE:
        print("GPU ACCELERATION: ✓ ENABLED")
        device_info = get_device_info()
        if device_info['device_name']:
            print(f"GPU Device: {device_info['device_name']}")
    else:
        print("GPU ACCELERATION: ✗ DISABLED (using CPU)")
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

    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Initialize variables
    prev_frame = None
    psf = None
    video_writer = None

    try:
        # Process frames with streaming (no buffering in memory)
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

            # Get PSF (only once, or per frame if estimating)
            if psf is None or config.PSF_TYPE == "estimate":
                psf = get_psf(frame_float)

            # Deblurring (GPU-accelerated if available)
            frame_deblurred = apply_deblurring(frame_float, psf, config.DEBLUR_METHOD,
                                              use_gpu=use_gpu)

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
                print(f"Initialized streaming video writer: {output_path}")

            # Write frame immediately (no buffering!)
            video_writer.write_frame(frame_enhanced)

            # Store only previous frame for stabilization (minimal memory)
            if config.ENABLE_STABILIZATION:
                prev_frame = frame_gray.copy()

            # Clean up intermediate arrays to free memory
            del frame_denoised, frame_float, frame_deblurred, frame_restored

        print()
        print(f"\nProcessed {video_info['frame_count']} frames")

    finally:
        # Release video writer
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved to: {output_path}")

    print("Processing complete!")
    print("=" * 60)


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
