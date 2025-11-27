"""
GPU-accelerated drone video enhancement with real-time side-by-side display
"""
import cv2
import numpy as np
import time
from pathlib import Path

# Import configuration
import config

# Import GPU utilities
from utils.gpu_utils import is_gpu_available, to_gpu, to_cpu

# Import PSF generators
from psf.motion_psf import create_motion_psf
from psf.gaussian_psf import gaussian_psf

# Import deblurring methods
from deblurring.wiener_gpu import wiener_deconvolution_gpu
from deblurring.richardson_lucy_gpu import richardson_lucy_gpu

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: CuPy not available. GPU acceleration disabled.")


def process_frame_gpu(frame, psf_gpu, method='wiener'):
    """
    Process a single frame using GPU acceleration.

    Args:
        frame: Input frame (numpy array, BGR)
        psf_gpu: Point spread function on GPU (CuPy array)
        method: Deblurring method ('wiener' or 'richardson_lucy')

    Returns:
        Processed frame (numpy array, BGR)
    """
    # Convert to float32
    frame_float = frame.astype(np.float32)

    # Split channels
    b, g, r = cv2.split(frame_float)

    # Transfer to GPU
    b_gpu = to_gpu(b)
    g_gpu = to_gpu(g)
    r_gpu = to_gpu(r)

    # Process each channel
    if method == 'wiener':
        b_proc = wiener_deconvolution_gpu(b_gpu, psf_gpu, K=config.WIENER_K)
        g_proc = wiener_deconvolution_gpu(g_gpu, psf_gpu, K=config.WIENER_K)
        r_proc = wiener_deconvolution_gpu(r_gpu, psf_gpu, K=config.WIENER_K)
    elif method == 'richardson_lucy':
        b_proc = richardson_lucy_gpu(b_gpu, psf_gpu, iterations=config.RL_ITERATIONS)
        g_proc = richardson_lucy_gpu(g_gpu, psf_gpu, iterations=config.RL_ITERATIONS)
        r_proc = richardson_lucy_gpu(r_gpu, psf_gpu, iterations=config.RL_ITERATIONS)
    else:
        raise ValueError(f"Unknown deblurring method: {method}")

    # Transfer back to CPU
    b_cpu = to_cpu(b_proc)
    g_cpu = to_cpu(g_proc)
    r_cpu = to_cpu(r_proc)

    # Merge channels
    processed = cv2.merge([b_cpu, g_cpu, r_cpu])

    # Convert back to uint8
    processed = np.clip(processed, 0, 255).astype(np.uint8)

    return processed


def create_side_by_side(original, processed, scale=0.5):
    """
    Create side-by-side comparison of original and processed frames.

    Args:
        original: Original frame
        processed: Processed frame
        scale: Display scale factor

    Returns:
        Combined frame with both images side-by-side
    """
    # Resize for display
    if scale != 1.0:
        height, width = original.shape[:2]
        new_height = int(height * scale)
        new_width = int(width * scale)

        original_resized = cv2.resize(original, (new_width, new_height))
        processed_resized = cv2.resize(processed, (new_width, new_height))
    else:
        original_resized = original
        processed_resized = processed

    # Add labels
    original_labeled = original_resized.copy()
    processed_labeled = processed_resized.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2

    cv2.putText(original_labeled, "Original", (10, 30), font, font_scale, (0, 255, 0), thickness)
    cv2.putText(processed_labeled, "Enhanced", (10, 30), font, font_scale, (0, 255, 0), thickness)

    # Combine horizontally
    combined = np.hstack([original_labeled, processed_labeled])

    return combined


def main():
    """Main processing function with real-time side-by-side display"""

    print("=" * 60)
    print("GPU-Accelerated Drone Video Enhancement")
    print("=" * 60)

    # Check GPU availability
    if not is_gpu_available():
        print("\nERROR: GPU (CuPy) not available!")
        print("Please install CuPy for GPU acceleration.")
        return

    print(f"\n✓ GPU Available: {GPU_AVAILABLE}")
    print(f"✓ Input video: {config.INPUT_VIDEO_PATH}")
    print(f"✓ Deblur method: {config.DEBLUR_METHOD}")
    print(f"✓ Display mode: Side-by-side comparison\n")

    # Open video
    cap = cv2.VideoCapture(config.INPUT_VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video file: {config.INPUT_VIDEO_PATH}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Create PSF
    print(f"\nCreating PSF ({config.PSF_TYPE})...")
    if config.PSF_TYPE == "motion":
        psf = create_motion_psf(config.MOTION_LENGTH, config.MOTION_ANGLE)
    else:
        psf = gaussian_psf(config.GAUSSIAN_PSF_SIZE, config.GAUSSIAN_PSF_SIGMA)

    # Transfer PSF to GPU
    psf_gpu = to_gpu(psf.astype(np.float32))
    print(f"✓ PSF shape: {psf.shape}")

    # Setup output video writer
    output_path = Path(config.OUTPUT_VIDEO_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Processing loop
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    print("\nProcessing video...")
    print("Press 'q' to quit, 'p' to pause/resume")
    print("-" * 60)

    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()

                if not ret:
                    break

                frame_count += 1

                # Skip frames if configured
                if config.SKIP_FRAMES > 1 and frame_count % config.SKIP_FRAMES != 0:
                    continue

                # Check max frames limit
                if config.MAX_FRAMES and processed_count >= config.MAX_FRAMES:
                    break

                # Process frame
                frame_start = time.time()
                processed_frame = process_frame_gpu(frame, psf_gpu, method=config.DEBLUR_METHOD)
                frame_time = time.time() - frame_start

                # Write to output video
                out.write(processed_frame)
                processed_count += 1

                # Display side-by-side comparison
                if config.DISPLAY_REALTIME and config.DISPLAY_COMPARISON:
                    combined = create_side_by_side(frame, processed_frame, scale=config.DISPLAY_SCALE)

                    # Add processing info
                    info_text = f"Frame: {processed_count}/{total_frames} | FPS: {1.0/frame_time:.1f} | Time: {frame_time*1000:.1f}ms"
                    cv2.putText(combined, info_text, (10, combined.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    cv2.imshow('Video Enhancement - Original vs Enhanced', combined)
                elif config.DISPLAY_REALTIME:
                    cv2.imshow('Enhanced Video', processed_frame)

                # Progress update
                if processed_count % 10 == 0:
                    elapsed = time.time() - start_time
                    fps_avg = processed_count / elapsed if elapsed > 0 else 0
                    eta = (total_frames - processed_count) / fps_avg if fps_avg > 0 else 0

                    print(f"Frame {processed_count}/{total_frames} | "
                          f"Avg FPS: {fps_avg:.2f} | "
                          f"ETA: {eta:.1f}s")

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nStopping...")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"\n{'Paused' if paused else 'Resumed'}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # Cleanup
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Final statistics
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("Processing complete!")
        print(f"  Frames processed: {processed_count}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average FPS: {processed_count/total_time:.2f}")
        print(f"  Output saved to: {output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()

