"""
Configuration file for video deblurring and enhancement.
All parameters for classical image processing pipeline.
"""

# GPU Acceleration
USE_GPU = True  # Set to False to force CPU processing
GPU_BATCH_SIZE = 10  # Number of frames to process on GPU before clearing memory

# Video I/O
INPUT_VIDEO_PATH = "input/drone_auto.mp4"
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
PSF_TYPE = "estimate"  # Options: "motion", "gaussian", "estimate"
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
