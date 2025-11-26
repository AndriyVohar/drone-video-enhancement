"""
Video input/output utilities.
"""
import cv2
import numpy as np
from typing import List, Tuple, Generator


def load_video(path: str) -> Tuple[List[np.ndarray], float]:
    """
    Load all frames from a video file.
    
    Args:
        path: Path to video file
        
    Returns:
        Tuple of (list of frames, fps)
    """
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps


def save_video(frames: List[np.ndarray], path: str, fps: float = 30.0) -> None:
    """
    Save frames to a video file.
    
    Args:
        frames: List of frames (numpy arrays)
        path: Output video path
        fps: Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Determine if grayscale or color
    is_color = len(frames[0].shape) == 3
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height), isColor=is_color)
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def iterate_frames(path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Iterate over video frames without loading all into memory.
    
    Args:
        path: Path to video file
        
    Yields:
        Tuple of (frame_number, frame)
    """
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame_num, frame
        frame_num += 1
    
    cap.release()


def get_video_info(path: str) -> dict:
    """
    Get video metadata.
    
    Args:
        path: Path to video file
        
    Returns:
        Dictionary with video information
    """
    cap = cv2.VideoCapture(path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {path}")
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    
    cap.release()
    return info
"""
Configuration file for video deblurring and enhancement.
All parameters for classical image processing pipeline.
"""

# Video I/O
INPUT_VIDEO_PATH = "input/drone_video.mp4"
OUTPUT_VIDEO_PATH = "output/enhanced_video.mp4"

# Denoising parameters
DENOISE_METHOD = "nlm"  # Options: "gaussian", "bilateral", "nlm"
GAUSSIAN_SIGMA = 1.0
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75
NLM_H = 10
NLM_PATCH_SIZE = 7
NLM_SEARCH_SIZE = 21

# PSF parameters
PSF_TYPE = "motion"  # Options: "motion", "gaussian", "estimate"
MOTION_LENGTH = 15
MOTION_ANGLE = 45
GAUSSIAN_PSF_SIZE = 15
GAUSSIAN_PSF_SIGMA = 5.0
PSF_SIZE = 31

# Deblurring parameters
DEBLUR_METHOD = "wiener"  # Options: "wiener", "tikhonov", "richardson_lucy"
WIENER_K = 0.01
TIKHONOV_ALPHA = 0.01
RL_ITERATIONS = 10

# Enhancement parameters
APPLY_CLAHE = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Stabilization
ENABLE_STABILIZATION = True

# Processing
CONVERT_TO_GRAYSCALE = True
