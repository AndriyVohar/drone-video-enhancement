# Drone Video Enhancement

GPU-accelerated system for enhancing drone video quality using deconvolution and deblurring methods.

**Based on**: Lecture 11 (Restoration of Blurred Images) from Igor Aizenberg's CMPG 767 - Image Processing and Analysis course  
[Course Link](https://www.igoraizenberg.com/my-classes/cmpg-767-image-processing-and-analysis)

## Features

- **GPU Acceleration** - utilizing CuPy for processing on NVIDIA GPUs
- **Real-time Display** - view original and processed video side-by-side in real-time
- **Multiple Deblurring Methods**:
    - Wiener deconvolution (fast, optimal for Gaussian noise)
    - Richardson-Lucy (iterative, better for Poisson noise)
    - Tikhonov regularization
- **Configurable PSF**:
    - Motion blur PSF (motion simulation)
    - Gaussian blur PSF
- **High Performance** - Full HD video processing in real-time on GPU

## Requirements

### System Requirements

- Python 3.8+
- NVIDIA GPU with CUDA support (for GPU acceleration)
- Linux / Windows / macOS

### Dependencies

```bash
numpy
fastrlock
scipy
opencv-python

opencv-contrib-python # if GPU support is needed
cupy-cuda11x  # or cupy-cuda12x depending on CUDA version
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AndriyVohar/drone-video-enhancement.git
cd drone-video-enhancement
```

### 2. Install Dependencies

#### Automatic Installation (Linux/macOS)

```bash
chmod +x install.sh
./install.sh
```

#### Manual Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install packages
pip install -r requirements.txt

# Install CuPy (choose appropriate CUDA version)
pip install cupy-cuda11x  # for CUDA 11.x
# or
pip install cupy-cuda12x  # for CUDA 12.x
```

### 3. Verify GPU

```bash
python diagnose_cupy.py
```

## Quick Start

### Basic Usage

```bash
python main_gpu.py
```

This will run video processing with default settings and show side-by-side comparison in real-time.

### Configure Parameters

Edit `config.py`:

```python
# Input/Output video
INPUT_VIDEO_PATH = "input/your_video.mp4"
OUTPUT_VIDEO_PATH = "output/enhanced_video.mp4"

# Deblurring method
DEBLUR_METHOD = "wiener"  # "wiener", "tikhonov", or "richardson_lucy"

# PSF settings
PSF_TYPE = "motion"  # "motion" or "gaussian"
MOTION_LENGTH = 5  # motion blur length in pixels
MOTION_ANGLE = 0  # motion angle in degrees

# Deblurring parameters
WIENER_K = 0.03  # Wiener regularization
RL_ITERATIONS = 10  # Richardson-Lucy iterations

# Display settings
DISPLAY_REALTIME = True  # show during processing
DISPLAY_COMPARISON = True  # side-by-side comparison
DISPLAY_SCALE = 0.5  # display scale (0.5 = 50%)
```

## Controls

When video is running:

- **`q`** - quit the program
- **`p`** - pause/resume processing

## Deblurring Methods

### 1. Wiener Deconvolution

**When to use**: Fast processing, Gaussian noise

**Formula**:

```
F_hat = (H* · G) / (|H|^2 + K)
```

**Parameters**:

- `WIENER_K` (0.001 - 0.1): balance between sharpness and noise
    - Lower value → more sharpness, more noise
    - Higher value → less noise, less sharpness

**Example configuration**:

```python
DEBLUR_METHOD = "wiener"
WIENER_K = 0.01  # standard value
```

### 2. Richardson-Lucy

**When to use**: Better quality, Poisson noise, when processing time is available

**Parameters**:

- `RL_ITERATIONS` (5-20): number of iterations
    - More iterations → better detail, longer processing
    - Fewer iterations → faster, but less pronounced effect

**Example configuration**:

```python
DEBLUR_METHOD = "richardson_lucy"
RL_ITERATIONS = 10
```

### 3. Tikhonov Regularization

**When to use**: Edge preservation, controlled regularization

**Parameters**:

- `TIKHONOV_ALPHA` (0.001 - 0.1): regularization parameter

## PSF (Point Spread Function)

### Motion Blur PSF

Used for videos with motion blur from camera or object movement.

```python
PSF_TYPE = "motion"
MOTION_LENGTH = 3  # blur length in pixels (1-30)
MOTION_ANGLE = 0  # blur direction in degrees (0-360)
```

**How to determine parameters**:

1. Open a frame from the video
2. Measure blur length on a high-contrast object
3. Determine blur direction (0° = horizontal, 90° = vertical)

### Gaussian Blur PSF

Used for defocus blur.

```python
PSF_TYPE = "gaussian"
GAUSSIAN_PSF_SIZE = (15, 15)
GAUSSIAN_PSF_SIGMA = 5.0
```

## Project Structure

```
drone-video-enhancement/
├── main_gpu.py              # Main script with GPU acceleration
├── config.py                # Configuration file
├── requirements.txt         # Python dependencies
├── diagnose_cupy.py        # GPU diagnostics
│
├── deblurring/             # Deblurring algorithms
│   ├── wiener_gpu.py       # Wiener deconvolution (GPU)
│   ├── richardson_lucy_gpu.py  # Richardson-Lucy (GPU)
│   └── tikhonov.py         # Tikhonov regularization
│
├── psf/                    # PSF generation
│   ├── motion_psf.py       # Motion blur PSF
│   └── gaussian_psf.py     # Gaussian blur PSF
│
├── utils/                  # Utilities
│   ├── gpu_utils.py        # GPU helper functions
│   ├── fft_tools_gpu.py    # FFT operations on GPU
│   └── video_io.py         # Video I/O
│
├── input/                  # Input videos
└── output/                 # Output videos
```

## Mathematical Background

All algorithms are based on classical signal processing theory:

### Convolution Model

```
g = h * f + n
```

where:

- `g` - observed (blurred) image
- `h` - PSF (blur kernel)
- `f` - original (sharp) image
- `n` - noise

### Frequency Domain

FFT is used for efficient convolution:

```
G = H · F + N  (in frequency domain)
```

Deconvolution estimates `F` from `G` and `H`.

### Regularization

Prevents noise amplification:

- Wiener/Tikhonov add regularization term
- Balances between deblurring and noise suppression

## Performance Optimization

### GPU Memory

If out of memory error occurs:

```python
SKIP_FRAMES = 2  # process every 2nd frame
DISPLAY_SCALE = 0.3  # reduce display size
```

### Processing Speed

- **Wiener**: ~30-50 FPS (Full HD, RTX 3050)
- **Richardson-Lucy**: ~5-15 FPS (depends on iterations)

### Recommendations

1. Start with `WIENER_K = 0.03` and `wiener` method
2. Process a few frames for testing
3. Adjust parameters
4. Run full processing

## Troubleshooting

### GPU Not Detected

```bash
# Check CUDA
nvidia-smi

# Check CuPy
python -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"

# Reinstall CuPy with correct CUDA version
pip uninstall cupy
pip install cupy-cuda11x  # or cupy-cuda12x
```

### Video Won't Open

- Check path in `INPUT_VIDEO_PATH`
- Ensure format is supported (mp4, avi, mov)
- Try converting video to mp4

### Slow Processing

- Reduce `RL_ITERATIONS` (if using Richardson-Lucy)
- Enable `SKIP_FRAMES = 2` or more
- Use `wiener` method instead of `richardson_lucy`

## Usage Examples

### Example 1: Fast Processing with Wiener

```python
# config.py
DEBLUR_METHOD = "wiener"
WIENER_K = 0.01
SKIP_FRAMES = 1
DISPLAY_COMPARISON = True
```

### Example 2: High Quality with Richardson-Lucy

```python
# config.py
DEBLUR_METHOD = "richardson_lucy"
RL_ITERATIONS = 15
SKIP_FRAMES = 1
DISPLAY_COMPARISON = True
```

### Example 3: Batch Processing without Display

```python
# config.py
DISPLAY_REALTIME = False
DISPLAY_COMPARISON = False
MAX_FRAMES = None  # all frames
```

## Testing
```bash
# Test on single image
python test_image.py

# GPU diagnostics
python diagnose_cupy.py
```

## Author

Andrii Vohar - [GitHub](https://github.com/AndriyVohar)

---

**Note**: For best results, NVIDIA GPU with at least 4GB VRAM and CUDA 11.0+ is recommended.

---

## Support Ukraine

Ukraine is currently defending itself against russian aggression. Thousands of civilians have been killed, millions
displaced, and critical infrastructure has been destroyed. Despite these challenges, Ukraine continues to fight for its
freedom and sovereignty. If you find this project useful, consider supporting Ukraine in its fight for freedom.

- **Donate**: Consider supporting humanitarian organizations like the [Red Cross Ukraine Crisis Appeal](https://redcross.org.ua/en/donate/), [United24](https://u24.gov.ua/), or [Come Back Alive](https://savelife.in.ua/en/donate-en/)
- **Stay Informed**: Follow reliable news sources about the situation
