"""
Video stabilization using optical flow.
"""
import numpy as np
import cv2


def stabilize_frame(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """
    Stabilize current frame based on optical flow from previous frame.

    Uses Farneback optical flow to estimate motion between frames,
    then compensates for camera movement.

    Args:
        prev_frame: Previous frame (grayscale)
        curr_frame: Current frame to stabilize (grayscale)

    Returns:
        Stabilized current frame
    """
    # Ensure grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    # Compute dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Calculate average motion (global translation)
    avg_dx = np.median(flow[:, :, 0])
    avg_dy = np.median(flow[:, :, 1])

    # Create transformation matrix for compensation
    M = np.float32([[1, 0, -avg_dx],
                    [0, 1, -avg_dy]])

    # Warp current frame to compensate for motion
    h, w = curr_frame.shape[:2]
    stabilized = cv2.warpAffine(curr_frame, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT)

    return stabilized


def stabilize_frame_features(prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
    """
    Stabilize frame using feature matching.

    Uses ORB features to detect keypoints and match between frames,
    then estimates affine transformation for stabilization.

    Args:
        prev_frame: Previous frame (grayscale)
        curr_frame: Current frame to stabilize (grayscale)

    Returns:
        Stabilized current frame
    """
    # Ensure grayscale
    if len(prev_frame.shape) == 3:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    else:
        prev_gray = prev_frame

    if len(curr_frame.shape) == 3:
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    else:
        curr_gray = curr_frame

    # Detect ORB features
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(prev_gray, None)
    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None or len(kp1) < 3 or len(kp2) < 3:
        # Not enough features, return original
        return curr_frame

    # Match features
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)

    if len(matches) < 3:
        return curr_frame

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Use top matches
    num_matches = min(50, len(matches))
    matches = matches[:num_matches]

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Estimate affine transformation
    M, inliers = cv2.estimateAffinePartial2D(pts2, pts1, method=cv2.RANSAC)

    if M is None:
        return curr_frame

    # Apply transformation
    h, w = curr_frame.shape[:2]
    stabilized = cv2.warpAffine(curr_frame, M, (w, h),
                                borderMode=cv2.BORDER_REFLECT)

    return stabilized


def compute_motion_trajectory(frames: list) -> np.ndarray:
    """
    Compute cumulative motion trajectory across frames.

    Args:
        frames: List of frames (grayscale)

    Returns:
        Array of shape (n_frames, 3) with dx, dy, da for each frame
    """
    n_frames = len(frames)
    trajectory = np.zeros((n_frames, 3))  # dx, dy, angle

    for i in range(1, n_frames):
        prev_gray = frames[i-1]
        curr_gray = frames[i]

        if len(prev_gray.shape) == 3:
            prev_gray = cv2.cvtColor(prev_gray, cv2.COLOR_BGR2GRAY)
        if len(curr_gray.shape) == 3:
            curr_gray = cv2.cvtColor(curr_gray, cv2.COLOR_BGR2GRAY)

        # Detect features
        orb = cv2.ORB_create(nfeatures=200)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(curr_gray, None)

        if des1 is not None and des2 is not None and len(kp1) >= 3 and len(kp2) >= 3:
            # Match
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des1, des2)

            if len(matches) >= 3:
                pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
                pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

                # Estimate transformation
                M, _ = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)

                if M is not None:
                    dx = M[0, 2]
                    dy = M[1, 2]
                    da = np.arctan2(M[1, 0], M[0, 0])

                    trajectory[i] = [dx, dy, da]

    # Cumulative sum
    trajectory = np.cumsum(trajectory, axis=0)

    return trajectory
