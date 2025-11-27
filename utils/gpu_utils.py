"""
GPU utility functions for CuPy
"""
import numpy as np

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

def is_gpu_available():
    """Check if GPU (CuPy) is available"""
    return GPU_AVAILABLE

def to_gpu(array):
    """Transfer numpy array to GPU"""
    if GPU_AVAILABLE and cp is not None:
        return cp.asarray(array)
    return array

def to_cpu(array):
    """Transfer CuPy array to CPU"""
    if GPU_AVAILABLE and cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return array

def get_array_module(array):
    """Get appropriate array module (numpy or cupy)"""
    if GPU_AVAILABLE and cp is not None:
        return cp.get_array_module(array)
    return np
