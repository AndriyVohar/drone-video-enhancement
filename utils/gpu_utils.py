"""
GPU Configuration and Detection

Manages GPU availability and fallback to CPU.
Uses CuPy for GPU-accelerated NumPy operations (classical algorithms only).
"""
import os
import sys

# Try to import GPU libraries
GPU_AVAILABLE = False
CUDA_OPENCV_AVAILABLE = False
cp = None

try:
    import cupy as cp
    # Test if CuPy actually works with a simple operation
    try:
        test_array = cp.array([1, 2, 3])
        _ = test_array + 1
        GPU_AVAILABLE = True
        print("✓ CuPy detected - GPU acceleration enabled")
    except Exception as e:
        print(f"⚠ CuPy detected but GPU test failed: {str(e)[:100]}")
        print("  Falling back to CPU. This is usually due to:")
        print("  - Missing CUDA libraries (libnvrtc.so)")
        print("  - CUDA version mismatch")
        print("  Fix: Install matching CUDA Toolkit or use CPU mode")
        cp = None
        GPU_AVAILABLE = False
except ImportError:
    print("⚠ CuPy not found - using CPU (install: pip install cupy-cuda11x or cupy-cuda12x)")
    cp = None

try:
    import cv2
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        CUDA_OPENCV_AVAILABLE = True
        print(f"✓ OpenCV CUDA enabled - {cv2.cuda.getCudaEnabledDeviceCount()} GPU(s) detected")
    else:
        print("⚠ OpenCV compiled without CUDA support")
except:
    print("⚠ OpenCV CUDA not available")


def get_array_module(gpu=True):
    """
    Get numpy or cupy based on availability and preference.

    Args:
        gpu: Whether to use GPU if available

    Returns:
        Module (numpy or cupy)
    """
    if gpu and GPU_AVAILABLE and cp is not None:
        return cp
    else:
        import numpy as np
        return np


def to_gpu(array):
    """
    Move numpy array to GPU if available.

    Args:
        array: NumPy array

    Returns:
        CuPy array if GPU available, otherwise original array
    """
    if GPU_AVAILABLE and cp is not None:
        try:
            return cp.asarray(array)
        except Exception as e:
            print(f"⚠ Warning: GPU transfer failed ({str(e)[:50]}), using CPU")
            return array
    return array


def to_cpu(array):
    """
    Move array from GPU to CPU.

    Args:
        array: CuPy or NumPy array

    Returns:
        NumPy array
    """
    if GPU_AVAILABLE and cp is not None:
        try:
            if isinstance(array, cp.ndarray):
                return cp.asnumpy(array)
        except:
            pass
    return array


def get_device_info():
    """Get information about available GPU devices."""
    info = {
        'gpu_available': GPU_AVAILABLE,
        'cuda_opencv': CUDA_OPENCV_AVAILABLE,
        'device_count': 0,
        'device_name': None
    }

    if GPU_AVAILABLE and cp is not None:
        try:
            info['device_count'] = cp.cuda.runtime.getDeviceCount()
            device_id = cp.cuda.Device().id
            info['device_name'] = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()
        except:
            pass

    return info


# Print device info on import
if GPU_AVAILABLE or CUDA_OPENCV_AVAILABLE:
    device_info = get_device_info()
    print(f"\n{'='*60}")
    print("GPU ACCELERATION STATUS")
    print(f"{'='*60}")
    print(f"CuPy (NumPy on GPU):     {'✓ Enabled' if GPU_AVAILABLE else '✗ Disabled'}")
    print(f"OpenCV CUDA:             {'✓ Enabled' if CUDA_OPENCV_AVAILABLE else '✗ Disabled'}")
    if device_info['device_name']:
        print(f"GPU Device:              {device_info['device_name']}")
    print(f"{'='*60}\n")
elif not GPU_AVAILABLE:
    print("\n⚠ GPU acceleration disabled - using CPU mode")
    print("=" * 60 + "\n")
