"""
Test script to verify and benchmark the vectorized SLdshear function.

This script:
1. Compares the vectorized version with a reference loop-based implementation
2. Benchmarks the performance improvement
"""

import torch
import time
import sys
sys.path.insert(0, '/home/suxh/mount/code/pycode/pyshearlab')


def SLdshear_loop_reference(inputArray: torch.Tensor, k: int, axis: int) -> torch.Tensor:
    """
    Original loop-based SLdshear for reference.
    """
    # Convert from MATLAB-style to 0-indexed
    axis = axis - 1
    
    if k == 0:
        return inputArray.clone()
    
    rows = inputArray.shape[0]
    cols = inputArray.shape[1]
    
    shearedArray = torch.zeros((rows, cols), dtype=inputArray.dtype, device=inputArray.device)
    
    if axis == 0:
        for col in range(cols):
            shift = int(k * (cols // 2 - col))
            shearedArray[:, col] = torch.roll(inputArray[:, col], shifts=shift, dims=0)
    else:
        for row in range(rows):
            shift = int(k * (rows // 2 - row))
            shearedArray[row, :] = torch.roll(inputArray[row, :], shifts=shift, dims=0)
    
    return shearedArray


def test_correctness():
    """Test that vectorized version matches loop-based version."""
    from pyshearlab.pySLUtilities_torch import SLdshear
    
    print("=" * 60)
    print("Testing Correctness")
    print("=" * 60)
    
    # Test various configurations
    test_cases = [
        # (rows, cols, k, axis)
        (64, 64, 0, 1),
        (64, 64, 1, 1),
        (64, 64, -1, 1),
        (64, 64, 2, 1),
        (64, 64, -2, 1),
        (64, 64, 0, 2),
        (64, 64, 1, 2),
        (64, 64, -1, 2),
        (64, 64, 2, 2),
        (64, 64, -2, 2),
        # Non-square
        (128, 64, 1, 1),
        (128, 64, 1, 2),
        (64, 128, -1, 1),
        (64, 128, -1, 2),
        # Larger k values
        (64, 64, 4, 1),
        (64, 64, -4, 2),
        # Complex dtype
        (64, 64, 2, 1),  # Will test complex below
    ]
    
    all_passed = True
    
    for rows, cols, k, axis in test_cases:
        # Real tensor
        x = torch.randn(rows, cols, dtype=torch.float64)
        
        result_vec = SLdshear(x, k, axis)
        result_loop = SLdshear_loop_reference(x, k, axis)
        
        if torch.allclose(result_vec, result_loop, atol=1e-14):
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False
            max_diff = (result_vec - result_loop).abs().max().item()
            print(f"  Max diff: {max_diff}")
        
        print(f"  {status}: shape=({rows}, {cols}), k={k}, axis={axis}")
    
    # Test complex input
    print("\n  Testing complex input:")
    x_complex = torch.randn(64, 64, dtype=torch.complex128)
    result_vec = SLdshear(x_complex, 2, 1)
    result_loop = SLdshear_loop_reference(x_complex, 2, 1)
    
    if torch.allclose(result_vec, result_loop, atol=1e-14):
        print("  ✓ PASS: complex input")
    else:
        print("  ✗ FAIL: complex input")
        all_passed = False
    
    print()
    if all_passed:
        print("All correctness tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    
    return all_passed


def test_performance():
    """Benchmark vectorized vs loop-based implementation."""
    from pyshearlab.pySLUtilities_torch import SLdshear
    
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    k_values = [1, 2, 4]
    n_warmup = 5
    n_runs = 20
    
    for rows, cols in sizes:
        print(f"\nSize: {rows} x {cols}")
        print("-" * 40)
        
        x = torch.randn(rows, cols, dtype=torch.float64)
        
        for k in k_values:
            for axis in [1, 2]:
                # Warmup
                for _ in range(n_warmup):
                    _ = SLdshear(x, k, axis)
                    _ = SLdshear_loop_reference(x, k, axis)
                
                # Benchmark vectorized
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    _ = SLdshear(x, k, axis)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                time_vec = (time.perf_counter() - t0) / n_runs * 1000  # ms
                
                # Benchmark loop
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    _ = SLdshear_loop_reference(x, k, axis)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                time_loop = (time.perf_counter() - t0) / n_runs * 1000  # ms
                
                speedup = time_loop / time_vec
                print(f"  k={k:2d}, axis={axis}: vec={time_vec:6.3f}ms, loop={time_loop:6.3f}ms, speedup={speedup:5.2f}x")


def test_integration():
    """Test that the full shearlet transform still works correctly."""
    from pyshearlab import pyShearLab2D_torch as shearlab
    
    print("\n" + "=" * 60)
    print("Integration Test (Full Shearlet Transform)")
    print("=" * 60)
    
    # Create test image
    rows, cols = 256, 256
    X = torch.randn(rows, cols, dtype=torch.float64)
    
    # Build shearlet system
    print("  Building shearlet system...")
    shearletSystem = shearlab.SLgetShearletSystem2D(rows, cols, nScales=3, device='cpu')
    
    # Decompose
    print("  Decomposing...")
    coeffs = shearlab.SLsheardec2D(X, shearletSystem)
    
    # Reconstruct
    print("  Reconstructing...")
    X_rec = shearlab.SLshearrec2D(coeffs, shearletSystem)
    
    # Check reconstruction error
    rec_error = (X - X_rec).abs().max().item()
    print(f"  Max reconstruction error: {rec_error:.2e}")
    
    if rec_error < 1e-10:
        print("  ✓ PASS: Integration test passed!")
        return True
    else:
        print("  ✗ FAIL: Reconstruction error too large!")
        return False


if __name__ == "__main__":
    print("SLdshear Vectorization Test")
    print("=" * 60)
    
    correctness_ok = test_correctness()
    test_performance()
    integration_ok = test_integration()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    if correctness_ok and integration_ok:
        print("All tests PASSED! The vectorized SLdshear is working correctly.")
    else:
        print("Some tests FAILED. Please review the results above.")
