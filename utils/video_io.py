"""
Video input/output utilities.
"""
import cv2
import numpy as np


def get_video_info(video_path):
    """
    Get video information.

    Args:
        video_path: Path to video file

    Returns:
        Dictionary with video info (width, height, fps, frame_count)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        'width': width,
        'height': height,
        'fps': fps,
        'frame_count': frame_count
    }


def iterate_frames(video_path):
    """
    Iterate through video frames.

    Args:
        video_path: Path to video file

    Yields:
        Tuple of (frame_number, frame)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_num = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        yield frame_num, frame
        frame_num += 1

    cap.release()


def save_video(frames, output_path, fps):
    """
    Save frames as video.

    Args:
        frames: List of frames (numpy arrays)
        output_path: Output video path
        fps: Frames per second
    """
    if len(frames) == 0:
        raise ValueError("No frames to save")

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Determine if color or grayscale
    is_color = len(frames[0].shape) == 3

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()


class VideoWriter:
    """
    Streaming video writer for memory-efficient processing.
    Writes frames immediately without storing in memory.
    """
    def __init__(self, output_path: str, fps: float, frame_size: tuple, is_color: bool = True):
        """
        Initialize video writer.

        Args:
            output_path: Output video path
            fps: Frames per second
            frame_size: (width, height) tuple
            is_color: Whether frames are in color
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.is_color = is_color

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size, is_color)

        if not self.writer.isOpened():
            raise ValueError(f"Failed to open video writer for: {output_path}")

        self.frame_count = 0

    def write_frame(self, frame: np.ndarray):
        """Write a single frame to video."""
        self.writer.write(frame)
        self.frame_count += 1

    def release(self):
        """Release video writer."""
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False
