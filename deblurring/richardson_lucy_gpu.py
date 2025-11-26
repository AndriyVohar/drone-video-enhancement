"""
GPU-accelerated Richardson-Lucy deconvolution.

Same classical iterative algorithm, but with GPU acceleration for speed.
"""
import numpy as np
from utils.gpu_utils import get_array_module, to_gpu, to_cpu, GPU_AVAILABLE
from utils.fft_tools_gpu import psf_to_otf_gpu


def richardson_lucy_fft_gpu(img: np.ndarray, psf: np.ndarray, iterations: int = 10,
                             use_gpu: bool = True) -> np.ndarray:
    """
    Richardson-Lucy deconvolution using FFT on GPU.
    
    Same classical algorithm as CPU version, accelerated with GPU:
    f^(k+1) = f^(k) · [(g / (h * f^(k))) * h_flipped]
    
    Args:
        img: Blurred input image (grayscale, float, non-negative)
        psf: Point Spread Function kernel
        iterations: Number of iterations
        use_gpu: Whether to use GPU acceleration
        
    Returns:
        Deblurred image
    """
    if not use_gpu or not GPU_AVAILABLE:
        # Fallback to CPU version
        from deblurring.richardson_lucy import richardson_lucy_fft
        return richardson_lucy_fft(img, psf, iterations)
    
    try:
        xp = get_array_module(use_gpu)
        
        # Move to GPU
        img_gpu = to_gpu(img.astype(np.float64))
        img_gpu = xp.maximum(img_gpu, 0) + 1e-10
        
        # Normalize PSF
        psf_norm = psf.astype(np.float64) / (np.sum(psf) + 1e-10)
        
        # Get OTF for PSF and flipped PSF
        img_shape = img.shape
        H = psf_to_otf_gpu(psf_norm, img_shape, use_gpu=True)
        H_gpu = to_gpu(H)
        
        # Flip PSF for correlation
        psf_flipped = np.flip(psf_norm)
        H_flipped = psf_to_otf_gpu(psf_flipped, img_shape, use_gpu=True)
        H_flipped_gpu = to_gpu(H_flipped)
        
        # Initialize estimate
        f_estimate = xp.copy(img_gpu)
        
        # Iterate
        for i in range(iterations):
            # Convolve with PSF using FFT
            F = xp.fft.fft2(f_estimate)
            convolved = xp.real(xp.fft.ifft2(H_gpu * F))
            convolved = xp.maximum(convolved, 1e-10)
            
            # Compute ratio
            ratio = img_gpu / convolved
            
            # Correlate with flipped PSF
            R = xp.fft.fft2(ratio)
            correlation = xp.real(xp.fft.ifft2(H_flipped_gpu * R))
            
            # Update estimate
            f_estimate = f_estimate * correlation
            f_estimate = xp.maximum(f_estimate, 0)
        
        # Move back to CPU
        return to_cpu(f_estimate)
        
    except Exception as e:
        # If GPU fails, fall back to CPU
        print(f"\n⚠ GPU processing failed: {str(e)[:80]}")
        print("  Falling back to CPU for this frame...")
        from deblurring.richardson_lucy import richardson_lucy_fft
        return richardson_lucy_fft(img, psf, iterations)
