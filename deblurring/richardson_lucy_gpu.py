"""
GPU-accelerated Richardson-Lucy deconvolution using CuPy.
"""
import numpy as np
from utils.gpu_utils import GPU_AVAILABLE

if GPU_AVAILABLE:
    try:
        import cupy as cp
        from utils.gpu_utils import to_gpu, to_cpu
        from utils.fft_tools_gpu import psf2otf_gpu
    except Exception:
        GPU_AVAILABLE = False


def richardson_lucy_fft_gpu(image, psf, iterations=10, use_gpu=True):
    """
    Richardson-Lucy deconvolution using FFT for speed (GPU version).

    Args:
        image: Blurred input image (2D or 3D numpy array, float)
        psf: Point spread function (2D numpy array)
        iterations: Number of iterations
        use_gpu: Whether to use GPU acceleration

    Returns:
        Deblurred image (numpy array, float)
    """
    if not GPU_AVAILABLE or not use_gpu:
        # Fallback to CPU version
        from .richardson_lucy import richardson_lucy_fft
        return richardson_lucy_fft(image, psf, iterations)

    try:
        # Handle color images by processing each channel separately
        if len(image.shape) == 3:
            # Color image - process each channel
            result = np.zeros_like(image)
            for channel in range(image.shape[2]):
                result[:, :, channel] = richardson_lucy_fft_gpu(image[:, :, channel], psf, iterations, use_gpu=True)
            return result

        # Grayscale image processing on GPU
        # Transfer to GPU
        image_gpu = to_gpu(image)
        psf_gpu = to_gpu(psf)

        # Initialize estimate with blurred image
        estimate = cp.copy(image_gpu)

        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        estimate = cp.maximum(estimate, epsilon)

        # Precompute PSF and flipped PSF in frequency domain
        otf = psf2otf_gpu(psf_gpu, image_gpu.shape)
        psf_flipped = cp.flip(psf_gpu)
        otf_flipped = psf2otf_gpu(psf_flipped, image_gpu.shape)

        # Richardson-Lucy iterations on GPU
        for i in range(iterations):
            # Convolve estimate with PSF (using FFT)
            estimate_fft = cp.fft.fft2(estimate)
            blurred_estimate_fft = estimate_fft * otf
            blurred_estimate = cp.real(cp.fft.ifft2(blurred_estimate_fft))
            blurred_estimate = cp.maximum(blurred_estimate, epsilon)

            # Compute ratio
            ratio = image_gpu / blurred_estimate

            # Convolve ratio with flipped PSF (using FFT)
            ratio_fft = cp.fft.fft2(ratio)
            correction_fft = ratio_fft * otf_flipped
            correction = cp.real(cp.fft.ifft2(correction_fft))

            # Update estimate
            estimate = estimate * correction
            estimate = cp.maximum(estimate, epsilon)

        # Transfer back to CPU
        return to_cpu(estimate)
    except Exception as e:
        # If GPU processing fails, fallback to CPU
        print(f"Warning: GPU processing failed ({e}), falling back to CPU")
        from .richardson_lucy import richardson_lucy_fft
        return richardson_lucy_fft(image, psf, iterations)
