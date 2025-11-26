### 1. Wiener Deconvolution

Optimal linear filter in frequency domain:

```
F_hat = (H* · G) / (|H|^2 + K)
```

Where:
- `F_hat`: Estimated original image (frequency domain)
- `H`: PSF in frequency domain (OTF)
- `G`: Blurred image (frequency domain)
- `K`: Noise-to-signal ratio
- `H*`: Complex conjugate of H

### 2. Tikhonov Regularization

Minimizes:
```
||Hf - g||^2 + α||Lf||^2
```

- Zero-order: L is identity (smoothness)
- First-order: L is gradient operator (edge preservation)

### 3. Richardson-Lucy

Iterative maximum likelihood:
```
f^(k+1) = f^(k) · [(g / (h * f^(k))) * h_flipped]
```

Assumes Poisson noise model.

## 📊 Parameter Tuning Guide

### Denoising Parameters

- **NLM_H**: 10 (default)
  - Lower (3-7): Preserve more detail, less noise removal
  - Higher (15-20): Remove more noise, may blur details

### PSF Parameters

- **MOTION_LENGTH**: 15 (pixels of blur)
  - Measure blur extent in your video
  - Typical range: 5-30 pixels

- **MOTION_ANGLE**: 45 (degrees)
  - 0° = horizontal
  - 90° = vertical
  - Observe blur direction in your video

### Deblurring Parameters

- **WIENER_K**: 0.01
  - Lower (0.001): Less regularization, sharper but noisier
  - Higher (0.1): More regularization, smoother but less sharp

- **TIKHONOV_ALPHA**: 0.01
  - Similar to Wiener K
  - Balance between fidelity and smoothness

- **RL_ITERATIONS**: 10
  - More iterations: Sharper results but risk of over-sharpening
  - Fewer iterations: Conservative deblurring
  - Typical range: 5-20

## 🧪 Testing Different Methods

Compare deblurring methods by running with different configurations:

```python
# Edit config.py
DEBLUR_METHOD = "wiener"     # Fast, good for Gaussian noise
# or
DEBLUR_METHOD = "tikhonov"   # Similar to Wiener, adjustable regularization
# or
DEBLUR_METHOD = "richardson_lucy"  # Best for Poisson noise, iterative
```

## 🎓 Mathematical Background

All algorithms are based on classical signal processing theory:

1. **Convolution Model**: `g = h * f + n`
   - g: observed blurred image
   - h: PSF (blur kernel)
   - f: original sharp image
   - n: noise

2. **Frequency Domain**: FFT used for efficient convolution
   - `G = H · F + N` (in frequency domain)
   - Deconvolution estimates F from G and H

3. **Regularization**: Prevents noise amplification
   - Wiener/Tikhonov add regularization term
   - Balances deblurring vs noise suppression

## ⚠️ Important Notes

- **No AI/ML**: This project uses only classical DSP techniques
- **Grayscale**: Processing is done on grayscale for efficiency (configurable)
- **Memory**: Large videos may require significant RAM
- **Speed**: FFT-based methods are faster than spatial methods

## 🔧 Troubleshooting

### Video not found
- Check `INPUT_VIDEO_PATH` in config.py
- Ensure the input directory and video file exist

### Out of memory
- Process fewer frames at a time
- Reduce video resolution
- Use spatial convolution instead of FFT for small PSFs

### Results too noisy
- Increase regularization parameter (WIENER_K, TIKHONOV_ALPHA)
- Apply stronger denoising (increase NLM_H)
- Use bilateral or NLM denoising

### Results too blurry
- Decrease regularization parameter
- Reduce denoising strength
- Check PSF parameters (length, angle)
- Try Richardson-Lucy with fewer iterations

## 📚 References

- Wiener Filter: Classical optimal linear filter theory
- Tikhonov Regularization: Inverse problem regularization
- Richardson-Lucy: Maximum likelihood estimation (1972, 1974)
- Non-Local Means: Buades et al. (2005)

## 📝 License

This project implements classical algorithms from signal processing literature.

---

**Built with classical signal processing - No neural networks, No AI models, Just mathematics! 📐**
# 🎥 Drone Video Enhancement - Classical Image Processing

A video deblurring and enhancement system using **only classical image processing techniques**. No AI, deep learning, or neural networks - just pure signal processing theory!

## 🎯 Features

- **Classical Deblurring Algorithms**:
  - Wiener Deconvolution (frequency domain)
  - Tikhonov Regularization (zero-order and gradient-based)
  - Richardson-Lucy Deconvolution (iterative)

- **PSF Modeling**:
  - Motion blur PSF (linear motion)
  - Gaussian blur PSF
  - Automatic PSF estimation from spectral analysis

- **Preprocessing**:
  - Video stabilization using optical flow
  - Multiple denoising methods (Gaussian, Bilateral, Non-Local Means)
  - CLAHE contrast enhancement

- **100% Deterministic**: No machine learning, no pretrained models, completely reproducible

## 📁 Project Structure

```
project/
├── main.py                      # Main pipeline
├── config.py                    # Configuration parameters
├── requirements.txt             # Python dependencies
├── utils/                       # Utility modules
│   ├── fft_tools.py            # FFT operations
│   ├── video_io.py             # Video I/O
│   └── image_ops.py            # Image utilities
├── psf/                        # Point Spread Functions
│   ├── gaussian_psf.py         # Gaussian PSF
│   ├── motion_psf.py           # Motion blur PSF
│   └── estimate_psf.py         # PSF estimation
├── deblurring/                 # Deblurring algorithms
│   ├── wiener.py               # Wiener filter
│   ├── tikhonov.py             # Tikhonov regularization
│   └── richardson_lucy.py      # Richardson-Lucy
└── preprocessing/              # Preprocessing
    ├── denoise.py              # Denoising methods
    └── stabilization.py        # Video stabilization
```

## 🚀 Installation

1. **Clone or navigate to the project directory**:
```bash
cd /home/thinkpad/Documents/drone-video-enhancement
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or if using the virtual environment:
```bash
source .venv/bin/activate  # Activate your venv
pip install -r requirements.txt
```

## ⚙️ Configuration

Edit `config.py` to customize processing parameters:

```python
# Video paths
INPUT_VIDEO_PATH = "input/drone_video.mp4"
OUTPUT_VIDEO_PATH = "output/enhanced_video.mp4"

# Denoising
DENOISE_METHOD = "nlm"  # "gaussian", "bilateral", "nlm"

# PSF type
PSF_TYPE = "motion"  # "motion", "gaussian", "estimate"
MOTION_LENGTH = 15
MOTION_ANGLE = 45

# Deblurring algorithm
DEBLUR_METHOD = "wiener"  # "wiener", "tikhonov", "richardson_lucy"
WIENER_K = 0.01  # Regularization parameter

# Enhancement
APPLY_CLAHE = True
ENABLE_STABILIZATION = True
```

## 🎬 Usage

### Basic Usage

1. **Prepare your video**:
   - Create an `input/` directory
   - Place your drone video as `input/drone_video.mp4`
   - Or update `INPUT_VIDEO_PATH` in `config.py`

2. **Run the pipeline**:
```bash
python main.py
```

3. **Output**:
   - Enhanced video will be saved to `output/enhanced_video.mp4`

### Processing Single Frames

You can also process individual frames programmatically:

```python
import cv2
from main import process_single_frame

# Read frame
frame = cv2.imread("blurred_frame.jpg")

# Process
enhanced = process_single_frame(
    frame, 
    psf_type="motion",
    deblur_method="wiener"
)

# Save
cv2.imwrite("enhanced_frame.jpg", enhanced)
```

## 🔬 Algorithms Explained

numpy>=1.21.0
opencv-python>=4.5.0
scipy>=1.7.0

