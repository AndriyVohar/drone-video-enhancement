"""
GPU-accelerated Wiener deconvolution using CuPy.
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


def wiener_deconvolution_gpu(image, psf, K=0.01, use_gpu=True):
    """
    Wiener deconvolution for image deblurring (GPU version).

    Args:
        image: Blurred input image (2D or 3D numpy array, float)
        psf: Point spread function (2D numpy array)
        K: Noise-to-signal power ratio (regularization parameter)
        use_gpu: Whether to use GPU acceleration

    Returns:
        Deblurred image (numpy array, float)
    """
    if not GPU_AVAILABLE or not use_gpu:
        # Fallback to CPU version
        from .wiener import wiener_deconvolution
        return wiener_deconvolution(image, psf, K)

    try:
        # Handle color images by processing each channel separately
        if len(image.shape) == 3:
            # Color image - process each channel
            result = np.zeros_like(image)
            for channel in range(image.shape[2]):
                result[:, :, channel] = wiener_deconvolution_gpu(image[:, :, channel], psf, K, use_gpu=True)
            return result

        # Grayscale image processing on GPU
        # Transfer to GPU
        image_gpu = to_gpu(image)
        psf_gpu = to_gpu(psf)

        # Convert PSF to OTF on GPU
        otf = psf2otf_gpu(psf_gpu, image_gpu.shape)

        # FFT of blurred image on GPU
        image_fft = cp.fft.fft2(image_gpu)

        # Wiener filter
        otf_conj = cp.conj(otf)
        otf_abs_sq = cp.abs(otf) ** 2

        # H* / (|H|^2 + K)
        wiener_filter = otf_conj / (otf_abs_sq + K)

        # Apply filter in frequency domain
        result_fft = image_fft * wiener_filter

        # Inverse FFT on GPU
        result = cp.fft.ifft2(result_fft)

        # Get real part
        result_real = cp.real(result)

        # Transfer back to CPU
        return to_cpu(result_real)
    except Exception as e:
        # If GPU processing fails, fallback to CPU
        print(f"Warning: GPU processing failed ({e}), falling back to CPU")
        from .wiener import wiener_deconvolution
        return wiener_deconvolution(image, psf, K)
