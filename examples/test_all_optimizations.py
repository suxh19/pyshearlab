"""
Test script to verify and benchmark all vectorized optimizations.

This script tests:
1. SLupsample vectorization
2. SLsheardec2D/SLshearrec2D batch FFT
3. SLshearadjoint2D/SLshearrecadjoint2D batch FFT
4. Full integration test
"""

import torch
import time
import sys
sys.path.insert(0, '/home/suxh/mount/code/pycode/pyshearlab')

from pyshearlab import pyShearLab2D_torch as shearlab
from pyshearlab.pySLUtilities_torch import SLupsample


def test_slupsample():
    """Test SLupsample correctness."""
    print("=" * 60)
    print("Testing SLupsample Vectorization")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1D
    x1d = torch.randn(10, dtype=torch.float64)
    result_1d = SLupsample(x1d, 1, 1)
    expected_size = 2 * 10 - 1
    if result_1d.shape[0] == expected_size and torch.allclose(result_1d[::2], x1d):
        print("  ✓ PASS: 1D upsample")
    else:
        print("  ✗ FAIL: 1D upsample")
        all_passed = False
    
    # Test 2D dim=1
    x2d = torch.randn(8, 12, dtype=torch.float64)
    for nZeros in [1, 2, 3]:
        result = SLupsample(x2d, 1, nZeros)
        step = nZeros + 1
        expected_rows = (8 - 1) * step + 1
        if result.shape[0] == expected_rows and result.shape[1] == 12:
            if torch.allclose(result[::step, :], x2d):
                print(f"  ✓ PASS: 2D dim=1, nZeros={nZeros}")
            else:
                print(f"  ✗ FAIL: 2D dim=1, nZeros={nZeros} (values mismatch)")
                all_passed = False
        else:
            print(f"  ✗ FAIL: 2D dim=1, nZeros={nZeros} (shape mismatch)")
            all_passed = False
    
    # Test 2D dim=2
    for nZeros in [1, 2, 3]:
        result = SLupsample(x2d, 2, nZeros)
        step = nZeros + 1
        expected_cols = (12 - 1) * step + 1
        if result.shape[0] == 8 and result.shape[1] == expected_cols:
            if torch.allclose(result[:, ::step], x2d):
                print(f"  ✓ PASS: 2D dim=2, nZeros={nZeros}")
            else:
                print(f"  ✗ FAIL: 2D dim=2, nZeros={nZeros} (values mismatch)")
                all_passed = False
        else:
            print(f"  ✗ FAIL: 2D dim=2, nZeros={nZeros} (shape mismatch)")
            all_passed = False
    
    return all_passed


def test_shearlet_transforms():
    """Test shearlet transform functions."""
    print("\n" + "=" * 60)
    print("Testing Shearlet Transform Functions")
    print("=" * 60)
    
    all_passed = True
    
    # Build shearlet system
    rows, cols = 256, 256
    nScales = 3
    print(f"  Building shearlet system ({rows}x{cols}, {nScales} scales)...")
    shearletSystem = shearlab.SLgetShearletSystem2D(rows, cols, nScales=nScales, device='cpu')
    
    # Test data
    X = torch.randn(rows, cols, dtype=torch.float64)
    
    # Test decomposition
    print("  Testing SLsheardec2D...")
    coeffs = shearlab.SLsheardec2D(X, shearletSystem)
    print(f"    Coefficients shape: {coeffs.shape}")
    
    # Test reconstruction
    print("  Testing SLshearrec2D...")
    X_rec = shearlab.SLshearrec2D(coeffs, shearletSystem)
    rec_error = (X - X_rec).abs().max().item()
    print(f"    Max reconstruction error: {rec_error:.2e}")
    
    if rec_error < 1e-10:
        print("    ✓ PASS: Reconstruction")
    else:
        print("    ✗ FAIL: Reconstruction error too large")
        all_passed = False
    
    # Test adjoint
    print("  Testing SLshearadjoint2D...")
    X_adj = shearlab.SLshearadjoint2D(coeffs, shearletSystem)
    print(f"    Adjoint output shape: {X_adj.shape}")
    
    # Test adjoint property: <Ax, y> = <x, A*y>
    Y = torch.randn(rows, cols, dtype=torch.float64)
    coeffs_Y = shearlab.SLsheardec2D(Y, shearletSystem)
    
    lhs = torch.sum(coeffs * coeffs_Y)  # <Ax, Ay> (approx)
    coeffs_X = shearlab.SLsheardec2D(X, shearletSystem)
    rhs = torch.sum(coeffs_X * coeffs_Y)
    
    # This is a rough check - actual adjoint test
    X_adj_from_Y = shearlab.SLshearadjoint2D(coeffs_Y, shearletSystem)
    lhs_adj = torch.sum(coeffs * coeffs_Y)
    rhs_adj = torch.sum(X * X_adj_from_Y)
    
    # Check <dec(X), Y_coeffs> ≈ <X, adj(Y_coeffs)>
    adj_diff = abs(lhs_adj.item() - rhs_adj.item()) / (abs(lhs_adj.item()) + 1e-10)
    if adj_diff < 1e-5:  # Allow small relative error
        print(f"    ✓ PASS: Adjoint property (rel diff: {adj_diff:.2e})")
    else:
        print(f"    ✗ FAIL: Adjoint property (rel diff: {adj_diff:.2e})")
        all_passed = False
    
    # Test recadjoint
    print("  Testing SLshearrecadjoint2D...")
    coeffs_recadj = shearlab.SLshearrecadjoint2D(X, shearletSystem)
    print(f"    RecAdjoint output shape: {coeffs_recadj.shape}")
    
    return all_passed


def benchmark_transforms():
    """Benchmark shearlet transform performance."""
    print("\n" + "=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    sizes = [(256, 256), (512, 512)]
    n_warmup = 3
    n_runs = 10
    
    for rows, cols in sizes:
        print(f"\nSize: {rows} x {cols}")
        print("-" * 40)
        
        # Build system
        shearletSystem = shearlab.SLgetShearletSystem2D(rows, cols, nScales=3, device='cpu')
        nShearlets = shearletSystem['nShearlets']
        
        X = torch.randn(rows, cols, dtype=torch.float64)
        
        # Warmup
        for _ in range(n_warmup):
            coeffs = shearlab.SLsheardec2D(X, shearletSystem)
            _ = shearlab.SLshearrec2D(coeffs, shearletSystem)
        
        # Benchmark decomposition
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(n_runs):
            coeffs = shearlab.SLsheardec2D(X, shearletSystem)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_dec = (time.perf_counter() - t0) / n_runs * 1000  # ms
        
        # Benchmark reconstruction
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = shearlab.SLshearrec2D(coeffs, shearletSystem)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_rec = (time.perf_counter() - t0) / n_runs * 1000  # ms
        
        # Benchmark adjoint
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = shearlab.SLshearadjoint2D(coeffs, shearletSystem)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        time_adj = (time.perf_counter() - t0) / n_runs * 1000  # ms
        
        print(f"  nShearlets: {nShearlets}")
        print(f"  SLsheardec2D:     {time_dec:8.2f} ms")
        print(f"  SLshearrec2D:     {time_rec:8.2f} ms")
        print(f"  SLshearadjoint2D: {time_adj:8.2f} ms")


def test_padded_api():
    """Test the padded API for non-square images."""
    print("\n" + "=" * 60)
    print("Testing Padded API (Non-Square Images)")
    print("=" * 60)
    
    # Non-square image
    rows, cols = 256, 128
    X = torch.randn(rows, cols, dtype=torch.float64)
    
    print(f"  Input shape: {X.shape}")
    
    # Decompose with padding
    coeffs, context = shearlab.SLsheardecPadded2D(X, nScales=3)
    print(f"  Coeffs shape: {coeffs.shape}")
    
    # Reconstruct
    X_rec = shearlab.SLshearrecPadded2D(coeffs, context)
    print(f"  Reconstructed shape: {X_rec.shape}")
    
    rec_error = (X - X_rec).abs().max().item()
    print(f"  Max reconstruction error: {rec_error:.2e}")
    
    if rec_error < 1e-10:
        print("  ✓ PASS: Padded API")
        return True
    else:
        print("  ✗ FAIL: Padded API")
        return False


if __name__ == "__main__":
    print("PyShearLab Torch Optimization Tests")
    print("=" * 60)
    
    upsample_ok = test_slupsample()
    transforms_ok = test_shearlet_transforms()
    benchmark_transforms()
    padded_ok = test_padded_api()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_ok = upsample_ok and transforms_ok and padded_ok
    if all_ok:
        print("All tests PASSED! ✓")
        print("\nOptimizations applied:")
        print("  1. SLdshear: vectorized with torch.gather")
        print("  2. SLupsample: vectorized with slice assignment")
        print("  3. SLsheardec2D: batch FFT operations")
        print("  4. SLshearrec2D: batch FFT operations")
        print("  5. SLshearadjoint2D: batch FFT operations")
        print("  6. SLshearrecadjoint2D: batch FFT operations")
    else:
        print("Some tests FAILED! ✗")
        print(f"  SLupsample: {'PASS' if upsample_ok else 'FAIL'}")
        print(f"  Transforms: {'PASS' if transforms_ok else 'FAIL'}")
        print(f"  Padded API: {'PASS' if padded_ok else 'FAIL'}")
