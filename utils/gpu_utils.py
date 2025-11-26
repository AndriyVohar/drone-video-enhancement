"""
GPU utility functions for CuPy-based acceleration.
"""
import numpy as np

# Try to import CuPy and test GPU functionality
GPU_AVAILABLE = False
cp = None

try:
    import cupy as cp
    # Test if CuPy can actually use the GPU with a simple operation
    try:
        test = cp.array([1, 2, 3])
        result = cp.sum(test)
        _ = cp.asnumpy(result)
        GPU_AVAILABLE = True
    except Exception as e:
        # CuPy is installed but can't initialize GPU
        print(f"Warning: CuPy installed but GPU unavailable: {e}")
        GPU_AVAILABLE = False
        cp = None
except ImportError:
    GPU_AVAILABLE = False
    cp = None


def get_device_info():
    """Get information about the GPU device."""
    if not GPU_AVAILABLE or cp is None:
        return {
            'available': False,
            'device_name': None,
            'memory_total': None,
            'memory_free': None
        }

    try:
        device = cp.cuda.Device()
        mem_info = device.mem_info

        # Get device name safely
        device_name = "GPU Device"
        try:
            device_name = device.compute_capability
        except:
            pass

        return {
            'available': True,
            'device_name': f"CUDA Device {device.id}",
            'device_id': device.id,
            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}" if hasattr(device, 'compute_capability') else "Unknown",
            'memory_total': mem_info[1] / (1024**3),  # GB
            'memory_free': mem_info[0] / (1024**3)    # GB
        }
    except Exception as e:
        return {
            'available': False,
            'error': str(e)
        }


def to_gpu(array):
    """Transfer numpy array to GPU (CuPy array)."""
    if not GPU_AVAILABLE or cp is None:
        raise RuntimeError("GPU is not available")
    try:
        return cp.asarray(array)
    except Exception as e:
        raise RuntimeError(f"Failed to transfer to GPU: {e}")


def to_cpu(array):
    """Transfer CuPy array to CPU (numpy array)."""
    if GPU_AVAILABLE and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if GPU_AVAILABLE and cp is not None:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
