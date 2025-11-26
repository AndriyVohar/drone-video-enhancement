"""
GPU-accelerated Wiener deconvolution.

Same classical Wiener filter formula, but computed on GPU for speed.
Formula: F_hat = (H* · G) / (|H|^2 + K)
"""
import numpy as np
from utils.gpu_utils import get_array_module, to_gpu, to_cpu, GPU_AVAILABLE
from utils.fft_tools_gpu import fft2_img_gpu, ifft2_img_gpu, psf_to_otf_gpu


def wiener_deconvolution_gpu(img: np.ndarray, psf: np.ndarray, K: float = 0.01, 
                              use_gpu: bool = True) -> np.ndarray:
    """
    Apply Wiener filtering on GPU for faster processing.
    
    Same classical algorithm as CPU version:
    F_hat = (H* · G) / (|H|^2 + K)
    
    Args:
        img: Blurred input image (grayscale, float)
        psf: Point Spread Function kernel
        K: Regularization parameter (noise-to-signal ratio)
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Deblurred image
    """
    if not use_gpu or not GPU_AVAILABLE:
        # Fallback to CPU version
        from deblurring.wiener import wiener_deconvolution
        return wiener_deconvolution(img, psf, K)
    
    try:
        xp = get_array_module(use_gpu)

        # Move to GPU
        img_gpu = to_gpu(img.astype(np.float64))

        # Get image dimensions
        img_shape = img.shape

        # Convert PSF to OTF (on GPU)
        H = psf_to_otf_gpu(psf, img_shape, use_gpu=True)
        H_gpu = to_gpu(H)

        # Compute FFT of blurred image (on GPU)
        G_gpu = xp.fft.fft2(img_gpu)

        # Wiener filter formula
        H_conj = xp.conj(H_gpu)
        H_mag_sq = xp.abs(H_gpu) ** 2

        # Apply Wiener filter
        F_hat = (H_conj * G_gpu) / (H_mag_sq + K)

        # Inverse FFT to get restored image
        restored = xp.real(xp.fft.ifft2(F_hat))

        # Ensure non-negative
        restored = xp.maximum(restored, 0)

        # Move back to CPU
        return to_cpu(restored)

    except Exception as e:
        # If GPU fails, fall back to CPU
        print(f"\n⚠ GPU processing failed: {str(e)[:80]}")
        print("  Falling back to CPU for this frame...")
        from deblurring.wiener import wiener_deconvolution
        return wiener_deconvolution(img, psf, K)
