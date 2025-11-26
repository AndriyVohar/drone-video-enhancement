"""
Simple main script WITH GPU SUPPORT.
Automatically uses GPU if available, falls back to CPU.
"""
import os
import cv2
import numpy as np

import config
from utils.video_io import iterate_frames, save_video, get_video_info
from utils.image_ops import to_gray, to_float, from_float
from preprocessing.denoise import denoise_gaussian, denoise_bilateral, denoise_nlm
from psf.gaussian_psf import gaussian_psf
from psf.motion_psf import motion_psf

# Try to import GPU versions
try:
    from utils.gpu_utils import GPU_AVAILABLE, get_device_info
    from deblurring.wiener_gpu import wiener_deconvolution_gpu
    from deblurring.richardson_lucy_gpu import richardson_lucy_fft_gpu
    GPU_SUPPORT = True
    print("✓ GPU modules loaded")
except ImportError as e:
    GPU_SUPPORT = False
    GPU_AVAILABLE = False
    print(f"⚠ GPU modules not available: {e}")

# Always import CPU versions as fallback
from deblurring.wiener import wiener_deconvolution
from deblurring.tikhonov import tikhonov_deconvolution
from deblurring.richardson_lucy import richardson_lucy_fft


def apply_denoising(frame: np.ndarray, method: str) -> np.ndarray:
    """Apply denoising to frame based on selected method."""
    if method == "none":
        return frame
    elif method == "gaussian":
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


def apply_deblurring(frame: np.ndarray, psf: np.ndarray, method: str, use_gpu: bool = False) -> np.ndarray:
    """Apply deblurring with optional GPU acceleration."""
    if method == "none":
        return frame

    # Try GPU first if available and requested
    if use_gpu and GPU_AVAILABLE and GPU_SUPPORT:
        try:
            if method == "wiener":
                return wiener_deconvolution_gpu(frame, psf, config.WIENER_K, use_gpu=True)
            elif method == "richardson_lucy":
                return richardson_lucy_fft_gpu(frame, psf, config.RL_ITERATIONS, use_gpu=True)
        except Exception as e:
            print(f"\n⚠ GPU failed for {method}: {str(e)[:60]}, using CPU...")

    # CPU fallback
    if method == "wiener":
        return wiener_deconvolution(frame, psf, config.WIENER_K)
    elif method == "tikhonov":
        return tikhonov_deconvolution(frame, psf, config.TIKHONOV_ALPHA)
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
        # Apply to each channel separately
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        result[:,:,0] = clahe.apply(result[:,:,0])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)


def apply_sharpening(frame: np.ndarray, amount: float = 0.5) -> np.ndarray:
    """
    Apply unsharp masking for extra sharpness.
    
    Args:
        frame: Input frame
        amount: Sharpening strength (0.0 to 1.0)
    
    Returns:
        Sharpened frame
    """
    # Gaussian blur
    blurred = cv2.GaussianBlur(frame, (0, 0), 3.0)
    
    # Unsharp mask: sharp = original + amount * (original - blurred)
    sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)
    
    return sharpened


def process_video_simple(input_path: str, output_path: str, use_gpu: bool = None) -> None:
    """
    Video processing with automatic GPU/CPU selection.
    """
    # Determine GPU usage
    if use_gpu is None:
        use_gpu = config.USE_GPU and GPU_AVAILABLE and GPU_SUPPORT

    print("=" * 60)
    print("DRONE VIDEO ENHANCEMENT")
    if use_gpu and GPU_AVAILABLE:
        print("GPU ACCELERATION: ✓ ENABLED")
        try:
            device_info = get_device_info()
            if device_info.get('device_name'):
                print(f"GPU Device: {device_info['device_name']}")
        except:
            pass
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
    print(f"  Color: {'COLOR ✓' if not config.CONVERT_TO_GRAYSCALE else 'GRAYSCALE'}")
    print(f"  Denoising: {config.DENOISE_METHOD}")
    print(f"  Deblurring: {config.DEBLUR_METHOD}", end='')
    if config.DEBLUR_METHOD == "wiener":
        print(f" (K={config.WIENER_K})")
    elif config.DEBLUR_METHOD == "richardson_lucy":
        print(f" ({config.RL_ITERATIONS} iterations)")
    else:
        print()
    print(f"  CLAHE contrast: {'ON ✓' if config.APPLY_CLAHE else 'OFF'}")
    print(f"  PSF: {config.PSF_TYPE} (len={config.MOTION_LENGTH}, angle={config.MOTION_ANGLE}°)")
    print()

    # Create output directory
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    # Generate PSF once (consistent for all frames)
    print("Generating PSF...")
    if config.PSF_TYPE == "motion":
        psf = motion_psf(config.MOTION_LENGTH, config.MOTION_ANGLE, config.PSF_SIZE)
    else:
        psf = gaussian_psf(config.GAUSSIAN_PSF_SIZE, config.GAUSSIAN_PSF_SIGMA)
    print(f"PSF size: {psf.shape}")
    print()
    print("Processing frames...")

    # Process frames
    processed_frames = []
    frame_count = 0

    for frame_num, frame in iterate_frames(input_path):
        print(f"Frame {frame_num + 1}/{video_info['frame_count']} - ", end='')

        # Keep color
        frame_work = frame

        # Step 1: Denoising
        print("denoise", end='')
        frame_denoised = apply_denoising(frame_work, config.DENOISE_METHOD)

        # Step 2: Deblurring (per channel for color) WITH GPU SUPPORT
        print("→deblur", end='')
        if use_gpu:
            print("(GPU)", end='')

        if len(frame_denoised.shape) == 3:
            # Color image - process each channel
            channels = cv2.split(frame_denoised)
            deblurred_channels = []
            for ch in channels:
                ch_float = to_float(ch)
                ch_deblurred = apply_deblurring(ch_float, psf, config.DEBLUR_METHOD, use_gpu=use_gpu)
                ch_restored = from_float(ch_deblurred)
                deblurred_channels.append(ch_restored)
            frame_restored = cv2.merge(deblurred_channels)
        else:
            # Grayscale
            frame_float = to_float(frame_denoised)
            frame_deblurred = apply_deblurring(frame_float, psf, config.DEBLUR_METHOD, use_gpu=use_gpu)
            frame_restored = from_float(frame_deblurred)

        # Step 3: CLAHE contrast enhancement
        if config.APPLY_CLAHE:
            print("→clahe", end='')
            frame_enhanced = apply_clahe(frame_restored)
        else:
            frame_enhanced = frame_restored

        # Step 4: Optional light sharpening
        if hasattr(config, 'APPLY_SHARPENING') and config.APPLY_SHARPENING:
            print("→sharp", end='')
            frame_enhanced = apply_sharpening(frame_enhanced, amount=0.3)

        print(" ✓")
        
        # Store frame
        processed_frames.append(frame_enhanced)
        frame_count += 1

    print()
    print(f"Processed {frame_count} frames")
    print(f"\nSaving video to: {output_path}")

    # Save processed video
    save_video(processed_frames, output_path, video_info['fps'])

    print("\n" + "=" * 60)
    print("✓ Processing complete!")
    print("=" * 60)
    print(f"\nCompare:")
    print(f"  Original: {input_path}")
    print(f"  Enhanced: {output_path}")
    print()


def main():
    """Main entry point."""
    input_path = config.INPUT_VIDEO_PATH
    output_path = config.OUTPUT_VIDEO_PATH

    if not os.path.exists(input_path):
        print(f"Error: Input video not found: {input_path}")
        print("\nPlease update the INPUT_VIDEO_PATH in config.py")
        print("or place your video at the specified location.")
        return

    process_video_simple(input_path, output_path)


if __name__ == "__main__":
    main()
