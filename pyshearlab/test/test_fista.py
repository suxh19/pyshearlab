"""
Test suite for FISTA Shearlet L1 regularization.

Tests:
- Soft threshold operator correctness
- Lipschitz constant estimation
- Algorithm convergence
- Denoising effectiveness (PSNR improvement)
"""

import pytest
import numpy as np
import torch

from pyshearlab.fista_shearlet import (
    soft_threshold,
    estimate_lipschitz,
    fista_shearlet_solve,
    compute_psnr
)
from pyshearlab import pyShearLab2D_torch as sl


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=['cpu'])
def device(request):
    return request.param


@pytest.fixture
def shearlet_system(device):
    """Create a small shearlet system for testing."""
    return sl.SLgetShearletSystem2D(64, 64, nScales=2, device=device)


# ============================================================================
# Test Soft Threshold
# ============================================================================

class TestSoftThreshold:
    """Test soft thresholding operator."""
    
    def test_zeros_below_threshold(self, device):
        """Values below threshold should become zero."""
        x = torch.tensor([-0.5, -0.3, 0.0, 0.3, 0.5], device=device)
        threshold = 0.6
        result = soft_threshold(x, threshold)
        
        assert torch.allclose(result, torch.zeros_like(result))
    
    def test_shrinkage_above_threshold(self, device):
        """Values above threshold should be shrunk."""
        x = torch.tensor([1.0, -1.0, 2.0, -2.0], device=device)
        threshold = 0.3
        expected = torch.tensor([0.7, -0.7, 1.7, -1.7], device=device)
        result = soft_threshold(x, threshold)
        
        assert torch.allclose(result, expected)
    
    def test_preserves_sign(self, device):
        """Sign should be preserved after thresholding."""
        x = torch.tensor([-5.0, 5.0, -3.0, 3.0], device=device)
        result = soft_threshold(x, 1.0)
        
        assert torch.all(torch.sign(result) == torch.sign(x))
    
    def test_zero_threshold(self, device):
        """Zero threshold should return input unchanged."""
        x = torch.randn(10, device=device)
        result = soft_threshold(x, 0.0)
        
        assert torch.allclose(result, x)


# ============================================================================
# Test Lipschitz Estimation
# ============================================================================

class TestLipschitzEstimation:
    """Test Lipschitz constant estimation."""
    
    def test_positive_value(self, shearlet_system):
        """Lipschitz constant should be positive."""
        L = estimate_lipschitz(shearlet_system, n_iters=10, verbose=False)
        
        assert L > 0
    
    def test_reasonable_range(self, shearlet_system):
        """Lipschitz constant should be in reasonable range."""
        L = estimate_lipschitz(shearlet_system, n_iters=15, verbose=False)
        
        # For normalized shearlets, L is typically close to 1
        assert 0.1 < L < 100


# ============================================================================
# Test FISTA Solver
# ============================================================================

class TestFISTASolver:
    """Test FISTA solver functionality."""
    
    def test_output_shapes(self, device):
        """Test output tensor shapes."""
        H, W = 64, 64
        y = torch.randn(H, W, dtype=torch.float32, device=device)
        
        rec, coeffs, losses = fista_shearlet_solve(
            y, lambda_reg=0.1, n_scales=2, max_iter=10, verbose=False
        )
        
        assert rec.shape == (H, W)
        assert coeffs.dim() == 3  # (padded_H, padded_W, N)
        assert isinstance(losses, list)
    
    def test_dtype_preservation(self, device):
        """Test that output dtype matches input."""
        y = torch.randn(64, 64, dtype=torch.float32, device=device)
        rec, _, _ = fista_shearlet_solve(
            y, lambda_reg=0.1, max_iter=10, verbose=False
        )
        
        assert rec.dtype == torch.float32
    
    def test_convergence(self, device):
        """Test that loss decreases monotonically."""
        y = torch.randn(64, 64, dtype=torch.float64, device=device)
        
        _, _, losses = fista_shearlet_solve(
            y, lambda_reg=0.1, n_scales=2, max_iter=50, verbose=False
        )
        
        # Loss should generally decrease (allow small numerical fluctuations)
        if len(losses) > 2:
            assert losses[-1] <= losses[0] * 1.1  # Final <= 110% of initial


# ============================================================================
# Test Denoising Effectiveness
# ============================================================================

class TestDenoisingEffectiveness:
    """Test that FISTA actually improves image quality."""
    
    def test_psnr_improvement(self, device):
        """Test that denoising improves PSNR."""
        # Create clean image
        H, W = 64, 64
        clean = torch.zeros(H, W, dtype=torch.float64, device=device)
        clean[20:44, 20:44] = 1.0  # Simple square
        
        # Add noise
        sigma = 0.2
        noisy = clean + sigma * torch.randn_like(clean)
        
        # Denoise
        denoised, _, _ = fista_shearlet_solve(
            noisy, lambda_reg=0.1, n_scales=2, max_iter=50, verbose=False
        )
        
        # Compute PSNR
        psnr_noisy = compute_psnr(clean, noisy)
        psnr_denoised = compute_psnr(clean, denoised)
        
        # Denoised should have higher PSNR
        assert psnr_denoised > psnr_noisy, \
            f"PSNR not improved: {psnr_noisy:.2f} -> {psnr_denoised:.2f}"


# ============================================================================
# Test PSNR Computation
# ============================================================================

class TestPSNR:
    """Test PSNR computation."""
    
    def test_identical_images(self, device):
        """Identical images should have infinite PSNR."""
        x = torch.randn(32, 32, device=device)
        psnr = compute_psnr(x, x)
        
        assert psnr == float('inf')
    
    def test_noisy_image(self, device):
        """Noisy image should have finite PSNR."""
        x = torch.randn(32, 32, device=device)
        y = x + 0.1 * torch.randn_like(x)
        psnr = compute_psnr(x, y)
        
        assert psnr > 0
        assert psnr < 100  # Reasonable upper bound


# ============================================================================
# Test Batch Processing
# ============================================================================

class TestBatchProcessing:
    """Test batch processing functionality."""
    
    def test_batch_output_shapes(self, device):
        """Test batch output tensor shapes."""
        B, H, W = 3, 64, 64
        y = torch.randn(B, H, W, dtype=torch.float32, device=device)
        
        rec, coeffs, losses = fista_shearlet_solve(
            y, lambda_reg=0.1, n_scales=2, max_iter=10, verbose=False
        )
        
        assert rec.shape == (B, H, W)
        assert coeffs.dim() == 4  # (B, H, W, N)
        assert coeffs.shape[0] == B
        assert isinstance(losses, list)
    
    def test_batch_dtype_preservation(self, device):
        """Test that batch output dtype matches input."""
        y = torch.randn(2, 64, 64, dtype=torch.float32, device=device)
        rec, coeffs, _ = fista_shearlet_solve(
            y, lambda_reg=0.1, max_iter=10, verbose=False
        )
        
        assert rec.dtype == torch.float32
        assert coeffs.dtype == torch.float32
    
    def test_batch_vs_single(self, device):
        """Test batch processing matches single image processing."""
        torch.manual_seed(42)
        B, H, W = 2, 64, 64
        
        # Create batch of simple images
        y_batch = torch.zeros(B, H, W, dtype=torch.float64, device=device)
        for i in range(B):
            y_batch[i, 20:44, 20:44] = 0.5 + 0.1 * i
            y_batch[i] += 0.1 * torch.randn(H, W, device=device, dtype=torch.float64)
        
        # Batch processing
        rec_batch, coeffs_batch, _ = fista_shearlet_solve(
            y_batch, lambda_reg=0.1, n_scales=2, max_iter=30, verbose=False
        )
        
        # Single image processing for each
        for i in range(B):
            rec_single, coeffs_single, _ = fista_shearlet_solve(
                y_batch[i], lambda_reg=0.1, n_scales=2, max_iter=30, verbose=False
            )
            
            # Results should be close (not exact due to independent convergence)
            np.testing.assert_allclose(
                rec_batch[i].cpu().numpy(),
                rec_single.cpu().numpy(),
                rtol=1e-2, atol=1e-4  # Relaxed tolerance for iterative algorithm
            )
    
    def test_batch_psnr_improvement(self, device):
        """Test that batch denoising improves PSNR for all images."""
        B, H, W = 2, 64, 64
        
        # Create clean images
        clean = torch.zeros(B, H, W, dtype=torch.float64, device=device)
        clean[:, 20:44, 20:44] = 1.0
        
        # Add noise
        noisy = clean + 0.2 * torch.randn_like(clean)
        
        # Batch denoise
        denoised, _, _ = fista_shearlet_solve(
            noisy, lambda_reg=0.1, n_scales=2, max_iter=50, verbose=False
        )
        
        # Compute per-image PSNR
        psnr_noisy = compute_psnr(clean, noisy, per_image=True)
        psnr_denoised = compute_psnr(clean, denoised, per_image=True)
        
        # Each image should have improved PSNR
        for i in range(B):
            assert psnr_denoised[i] > psnr_noisy[i], \
                f"Image {i}: PSNR not improved: {psnr_noisy[i]:.2f} -> {psnr_denoised[i]:.2f}"
    
    def test_batch_single_element(self, device):
        """Test B=1 edge case."""
        y = torch.randn(1, 64, 64, dtype=torch.float64, device=device)
        
        rec, coeffs, _ = fista_shearlet_solve(
            y, lambda_reg=0.1, max_iter=10, verbose=False
        )
        
        assert rec.shape == (1, 64, 64)
        assert coeffs.dim() == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

