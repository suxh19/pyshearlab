"""
Comprehensive benchmark for shearlet system construction and transforms.

This script benchmarks:
1. Shearlet system construction (filter generation)
2. Forward transform (decomposition)
3. Inverse transform (reconstruction)
"""

import torch
import time
import sys
sys.path.insert(0, '/home/suxh/mount/code/pycode/pyshearlab')

from pyshearlab import pyShearLab2D_torch as shearlab


def benchmark_system_construction():
    """Benchmark shearlet system construction time."""
    print("=" * 60)
    print("Shearlet System Construction Benchmark")
    print("=" * 60)
    
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    scales_list = [2, 3, 4]
    n_warmup = 1
    n_runs = 3
    
    for rows, cols in sizes:
        print(f"\nImage size: {rows} x {cols}")
        print("-" * 40)
        
        for nScales in scales_list:
            # Warmup
            for _ in range(n_warmup):
                _ = shearlab.SLgetShearletSystem2D(rows, cols, nScales=nScales, device='cpu')
            
            # Benchmark
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                system = shearlab.SLgetShearletSystem2D(rows, cols, nScales=nScales, device='cpu')
                times.append(time.perf_counter() - t0)
            
            avg_time = sum(times) / len(times) * 1000  # ms
            nShearlets = system['nShearlets']
            print(f"  nScales={nScales}, nShearlets={nShearlets:3d}: {avg_time:8.1f} ms")


def benchmark_transforms():
    """Benchmark forward and inverse transforms."""
    print("\n" + "=" * 60)
    print("Transform Benchmark")
    print("=" * 60)
    
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    n_warmup = 3
    n_runs = 10
    
    for rows, cols in sizes:
        print(f"\nImage size: {rows} x {cols}")
        print("-" * 40)
        
        # Build system once
        system = shearlab.SLgetShearletSystem2D(rows, cols, nScales=3, device='cpu')
        nShearlets = system['nShearlets']
        
        X = torch.randn(rows, cols, dtype=torch.float64)
        
        # Warmup
        for _ in range(n_warmup):
            coeffs = shearlab.SLsheardec2D(X, system)
            _ = shearlab.SLshearrec2D(coeffs, system)
        
        # Benchmark decomposition
        t0 = time.perf_counter()
        for _ in range(n_runs):
            coeffs = shearlab.SLsheardec2D(X, system)
        time_dec = (time.perf_counter() - t0) / n_runs * 1000
        
        # Benchmark reconstruction
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = shearlab.SLshearrec2D(coeffs, system)
        time_rec = (time.perf_counter() - t0) / n_runs * 1000
        
        # Throughput
        pixels = rows * cols
        throughput_dec = pixels / (time_dec / 1000) / 1e6  # MPixels/s
        throughput_rec = pixels / (time_rec / 1000) / 1e6  # MPixels/s
        
        print(f"  nShearlets: {nShearlets}")
        print(f"  Decomposition: {time_dec:7.2f} ms ({throughput_dec:5.2f} MPixels/s)")
        print(f"  Reconstruction: {time_rec:7.2f} ms ({throughput_rec:5.2f} MPixels/s)")


def benchmark_gpu_if_available():
    """Benchmark with GPU if available."""
    if not torch.cuda.is_available():
        print("\n" + "=" * 60)
        print("GPU not available, skipping GPU benchmarks")
        print("=" * 60)
        return
    
    print("\n" + "=" * 60)
    print("GPU Benchmark (CUDA)")
    print("=" * 60)
    
    device = 'cuda'
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    n_warmup = 5
    n_runs = 20
    
    for rows, cols in sizes:
        print(f"\nImage size: {rows} x {cols}")
        print("-" * 40)
        
        # Build system on GPU
        t0 = time.perf_counter()
        system = shearlab.SLgetShearletSystem2D(rows, cols, nScales=3, device=device)
        torch.cuda.synchronize()
        time_build = (time.perf_counter() - t0) * 1000
        print(f"  System build: {time_build:7.2f} ms")
        
        nShearlets = system['nShearlets']
        X = torch.randn(rows, cols, dtype=torch.float64, device=device)
        
        # Warmup
        for _ in range(n_warmup):
            coeffs = shearlab.SLsheardec2D(X, system)
            _ = shearlab.SLshearrec2D(coeffs, system)
        torch.cuda.synchronize()
        
        # Benchmark decomposition
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            coeffs = shearlab.SLsheardec2D(X, system)
        torch.cuda.synchronize()
        time_dec = (time.perf_counter() - t0) / n_runs * 1000
        
        # Benchmark reconstruction
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = shearlab.SLshearrec2D(coeffs, system)
        torch.cuda.synchronize()
        time_rec = (time.perf_counter() - t0) / n_runs * 1000
        
        print(f"  nShearlets: {nShearlets}")
        print(f"  Decomposition: {time_dec:7.2f} ms")
        print(f"  Reconstruction: {time_rec:7.2f} ms")


def verify_correctness():
    """Verify that results are correct."""
    print("\n" + "=" * 60)
    print("Correctness Verification")
    print("=" * 60)
    
    rows, cols = 256, 256
    X = torch.randn(rows, cols, dtype=torch.float64)
    
    system = shearlab.SLgetShearletSystem2D(rows, cols, nScales=3, device='cpu')
    
    # Forward-inverse
    coeffs = shearlab.SLsheardec2D(X, system)
    X_rec = shearlab.SLshearrec2D(coeffs, system)
    
    error = (X - X_rec).abs().max().item()
    print(f"  Max reconstruction error: {error:.2e}")
    
    if error < 1e-10:
        print("  ✓ PASS")
    else:
        print("  ✗ FAIL")


if __name__ == "__main__":
    print("PyShearLab Torch Comprehensive Benchmark")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()
    
    verify_correctness()
    benchmark_system_construction()
    benchmark_transforms()
    benchmark_gpu_if_available()
    
    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)
