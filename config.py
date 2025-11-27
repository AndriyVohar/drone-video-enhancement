"""
Configuration file for drone video enhancement
"""

# ========================================
# GPU SETTINGS
# ========================================
USE_GPU = True  # Set to False to use CPU only

# ========================================
# INPUT/OUTPUT
# ========================================
INPUT_VIDEO_PATH = "input/YTDown.com_YouTube_Media_GHCPvJU_wFY_001_1080p.mp4"
OUTPUT_VIDEO_PATH = "output/enhanced_video.mp4"

# ========================================
# PREPROCESSING
# ========================================
CONVERT_TO_GRAYSCALE = False  # Keep color by default

# Denoising method: "none", "gaussian", "bilateral", "nlm"
DENOISE_METHOD = "none"

# Gaussian denoising parameters
GAUSSIAN_SIGMA = 1.5

# Bilateral filter parameters
BILATERAL_D = 9
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# Non-local means denoising parameters
NLM_H = 10
NLM_PATCH_SIZE = 7
NLM_SEARCH_SIZE = 21

# ========================================
# POINT SPREAD FUNCTION (PSF)
# ========================================
# PSF type: "motion" or "gaussian"
PSF_TYPE = "motion"
PSF_SIZE = (15, 15)

# Motion blur PSF parameters
MOTION_LENGTH = 1.5
MOTION_ANGLE = 0

# Gaussian PSF parameters
GAUSSIAN_PSF_SIZE = (15, 15)
GAUSSIAN_PSF_SIGMA = 5.0

# ========================================
# DECONVOLUTION/DEBLURRING
# ========================================
# Deblurring method: "none", "wiener", "tikhonov", "richardson_lucy"
DEBLUR_METHOD = "wiener"

# Wiener filter parameter (noise-to-signal ratio)
WIENER_K = 0.03

# Tikhonov regularization parameter
TIKHONOV_ALPHA = 0.01

# Richardson-Lucy iterations
RL_ITERATIONS = 10

# ========================================
# POST-PROCESSING
# ========================================
# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
APPLY_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

# Apply additional sharpening
APPLY_SHARPENING = False
SHARPENING_AMOUNT = 0.3

ENABLE_STABILIZATION = False