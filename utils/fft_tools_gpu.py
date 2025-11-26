"""
GPU-accelerated FFT operations using CuPy.

These are the same classical algorithms, just running on GPU for speed.
No AI/ML - pure signal processing.
"""
import numpy as np
from utils.gpu_utils import get_array_module, to_gpu, to_cpu, GPU_AVAILABLE


def fft2_img_gpu(img: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Compute 2D FFT on GPU if available.
    
    Args:
        img: Input image (real-valued)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        2D FFT (complex-valued)
    """
    if not use_gpu or not GPU_AVAILABLE:
        return np.fft.fft2(img)

    try:
        xp = get_array_module(use_gpu)
        img_gpu = to_gpu(img)
        result = xp.fft.fft2(img_gpu)
        return to_cpu(result)
    except Exception:
        # Fallback to CPU
        return np.fft.fft2(img)


def ifft2_img_gpu(freq: np.ndarray, use_gpu: bool = True) -> np.ndarray:
    """
    Compute inverse 2D FFT on GPU if available.
    
    Args:
        freq: Frequency domain representation
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Real-valued image
    """
    if not use_gpu or not GPU_AVAILABLE:
        return np.real(np.fft.ifft2(freq))

    try:
        xp = get_array_module(use_gpu)
        freq_gpu = to_gpu(freq)
        result = xp.real(xp.fft.ifft2(freq_gpu))
        return to_cpu(result)
    except Exception:
        # Fallback to CPU
        return np.real(np.fft.ifft2(freq))


def psf_to_otf_gpu(psf: np.ndarray, shape: tuple, use_gpu: bool = True) -> np.ndarray:
    """
    Convert PSF to OTF on GPU if available.
    
    Args:
        psf: Point Spread Function kernel
        shape: Target shape (height, width)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        OTF in frequency domain
    """
    if not use_gpu or not GPU_AVAILABLE:
        # CPU fallback
        psf_padded = np.zeros(shape, dtype=np.float64)
        psf_h, psf_w = psf.shape
        start_h = (shape[0] - psf_h) // 2
        start_w = (shape[1] - psf_w) // 2
        psf_padded[start_h:start_h + psf_h, start_w:start_w + psf_w] = psf
        psf_padded = np.fft.ifftshift(psf_padded)
        otf = np.fft.fft2(psf_padded)
        return otf

    try:
        xp = get_array_module(use_gpu)

        # Calculate where to place the PSF (centered)
        psf_h, psf_w = psf.shape
        start_h = (shape[0] - psf_h) // 2
        start_w = (shape[1] - psf_w) // 2

        # Move to GPU
        psf_gpu = to_gpu(psf)
        psf_padded = xp.zeros(shape, dtype=xp.float64)
        psf_padded[start_h:start_h + psf_h, start_w:start_w + psf_w] = psf_gpu
        # Circular shift to move center to origin
        psf_padded = xp.fft.ifftshift(psf_padded)
        # Compute OTF
        otf = xp.fft.fft2(psf_padded)
        return to_cpu(otf)

    except Exception:
        # Fallback to CPU
        psf_padded = np.zeros(shape, dtype=np.float64)
        psf_h, psf_w = psf.shape
        start_h = (shape[0] - psf_h) // 2
        start_w = (shape[1] - psf_w) // 2
        psf_padded[start_h:start_h + psf_h, start_w:start_w + psf_w] = psf
        psf_padded = np.fft.ifftshift(psf_padded)
        otf = np.fft.fft2(psf_padded)
        return otf
