#!/usr/bin/env python3
"""
Diagnose and fix CuPy installation issues.
"""
import sys
import subprocess
import os

print("=" * 60)
print("CUPY DIAGNOSTIC TOOL")
print("=" * 60)

# Check if CuPy is installed
print("\n1. Checking CuPy installation...")
try:
    import cupy as cp
    print(f"   ✓ CuPy is installed")
    print(f"   Version: {cp.__version__}")
except ImportError:
    print("   ✗ CuPy is not installed")
    sys.exit(1)

# Check CUDA libraries
print("\n2. Checking CUDA libraries...")
result = subprocess.run(['find', '/usr', '-name', 'libnvrtc.so*'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
libs = result.stdout.strip().split('\n')
for lib in libs:
    if lib:
        print(f"   Found: {lib}")

# Try to create a simple CuPy array
print("\n3. Testing CuPy functionality...")
try:
    test_array = cp.array([1, 2, 3])
    print(f"   ✓ Successfully created CuPy array: {test_array}")
except Exception as e:
    print(f"   ✗ Failed to create CuPy array")
    print(f"   Error: {e}")
    print("\n" + "=" * 60)
    print("DIAGNOSIS:")
    print("=" * 60)

    error_str = str(e)
    if 'libnvrtc.so.13' in error_str:
        print("\n❌ CuPy 13.x is looking for CUDA 13 runtime libraries")
        print("   But you have CUDA 12.0 installed")
        print("\n✅ SOLUTION:")
        print("   Uninstall current CuPy and install correct version:")
        print("\n   pip uninstall cupy cupy-cuda11x cupy-cuda12x cupy-cuda13x -y")
        print("   pip install cupy-cuda12x")
    elif 'libnvrtc.so.12' in error_str:
        print("\n❌ CUDA runtime library issue")
        print("\n✅ SOLUTION:")
        print("   Create symlink or install CUDA 12.x toolkit properly")
    else:
        print(f"\n❌ Unknown error: {e}")

    sys.exit(1)

print("\n" + "=" * 60)
print("✅ CuPy is working correctly!")
print("=" * 60)

