"""
Test suite for pyShearLab2D_torch.py

Validates that PyTorch implementations produce identical results to NumPy versions
and satisfy mathematical properties (perfect reconstruction, adjoint).
"""

import pytest
import numpy as np
import torch

from pyshearlab import pyShearLab2D as np_shearlab
from pyshearlab import pyShearLab2D_torch as torch_shearlab


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=['cpu'])
def device(request):
    return request.param


@pytest.fixture
def test_image():
    """Generate a simple test image."""
    np.random.seed(42)
    return np.random.randn(128, 128).astype(np.float64)


@pytest.fixture
def test_image_rect():
    """Generate a rectangular test image."""
    np.random.seed(42)
    return np.random.randn(96, 128).astype(np.float64)


# ============================================================================
# Test SLgetShearletSystem2D
# ============================================================================

class TestSLgetShearletSystem2D:
    """Test shearlet system creation."""
    
    def test_basic_creation(self, device):
        """Test basic shearlet system creation."""
        rows, cols, nScales = 128, 128, 2
        
        result = torch_shearlab.SLgetShearletSystem2D(
            rows, cols, nScales, device=device)
        
        assert 'shearlets' in result
        assert 'RMS' in result
        assert 'dualFrameWeights' in result
        assert result['nShearlets'] > 0
        assert result['shearlets'].shape[0] == rows
        assert result['shearlets'].shape[1] == cols
    
    def test_consistency_with_numpy(self, device, test_image):
        """Test that shearlet system matches NumPy version."""
        rows, cols = test_image.shape
        nScales = 2
        
        # NumPy version
        np_system = np_shearlab.SLgetShearletSystem2D(0, rows, cols, nScales)
        
        # PyTorch version
        torch_system = torch_shearlab.SLgetShearletSystem2D(
            rows, cols, nScales, device=device)
        
        # Compare shapes
        assert torch_system['nShearlets'] == np_system['nShearlets']
        assert torch_system['shearlets'].shape == np_system['shearlets'].shape
        
        # Compare dualFrameWeights
        np.testing.assert_allclose(
            np_system['dualFrameWeights'],
            torch_system['dualFrameWeights'].cpu().numpy(),
            rtol=1e-6, atol=1e-8
        )


# ============================================================================
# Test Decomposition and Reconstruction
# ============================================================================

class TestPerfectReconstruction:
    """Test perfect reconstruction property."""
    
    def test_dec_rec_roundtrip(self, device, test_image):
        """Test that decomposition + reconstruction recovers original."""
        X = torch.from_numpy(test_image).to(device)
        
        # Create system
        rows, cols = X.shape
        shearletSystem = torch_shearlab.SLgetShearletSystem2D(
            rows, cols, nScales=2, device=device)
        
        # Decompose and reconstruct
        coeffs = torch_shearlab.SLsheardec2D(X, shearletSystem)
        X_rec = torch_shearlab.SLshearrec2D(coeffs, shearletSystem)
        
        # Check reconstruction error
        error = torch.max(torch.abs(X - X_rec)).item()
        assert error < 1e-10, f"Reconstruction error too large: {error}"
    
    def test_consistency_with_numpy(self, device, test_image):
        """Test that decomposition matches NumPy version."""
        X_np = test_image
        X_torch = torch.from_numpy(X_np).to(device)
        
        rows, cols = X_np.shape
        nScales = 2
        
        # NumPy decomposition
        np_system = np_shearlab.SLgetShearletSystem2D(0, rows, cols, nScales)
        np_coeffs = np_shearlab.SLsheardec2D(X_np, np_system)
        
        # PyTorch decomposition
        torch_system = torch_shearlab.SLgetShearletSystem2D(
            rows, cols, nScales, device=device)
        torch_coeffs = torch_shearlab.SLsheardec2D(X_torch, torch_system)
        
        # Compare coefficients
        np.testing.assert_allclose(
            np_coeffs,
            torch_coeffs.cpu().numpy(),
            rtol=1e-6, atol=1e-8
        )


# ============================================================================
# Test Adjoint Property
# ============================================================================

class TestAdjointProperty:
    """Test that adjoint satisfies <Ax, y> = <x, A*y>."""
    
    def test_adjoint_equation(self, device, test_image):
        """Test adjoint property."""
        X = torch.from_numpy(test_image).to(device)
        
        rows, cols = X.shape
        shearletSystem = torch_shearlab.SLgetShearletSystem2D(
            rows, cols, nScales=2, device=device)
        
        # Forward: coeffs = A * X
        coeffs = torch_shearlab.SLsheardec2D(X, shearletSystem)
        
        # Adjoint: X_adj = A^* * coeffs  
        X_adj = torch_shearlab.SLshearadjoint2D(coeffs, shearletSystem)
        
        # Check adjoint equation: <X, X_adj> ≈ <coeffs, coeffs>
        lhs = torch.sum(X * X_adj).item()
        rhs = torch.sum(coeffs * coeffs).item()
        
        np.testing.assert_allclose(lhs, rhs, rtol=1e-8)


# ============================================================================
# Test Padded API
# ============================================================================

class TestPaddedAPI:
    """Test padded decomposition/reconstruction."""
    
    def test_padded_roundtrip(self, device, test_image_rect):
        """Test padded decomposition + reconstruction."""
        X = torch.from_numpy(test_image_rect).to(device)
        
        # Decompose with padding
        coeffs, context = torch_shearlab.SLsheardecPadded2D(X, nScales=2)
        
        # Check coefficients shape matches original image
        assert coeffs.shape[0] == X.shape[0]
        assert coeffs.shape[1] == X.shape[1]
        
        # Reconstruct
        X_rec = torch_shearlab.SLshearrecPadded2D(coeffs, context)
        
        # Check reconstruction
        assert X_rec.shape == X.shape
        error = torch.max(torch.abs(X - X_rec)).item()
        assert error < 1e-10, f"Padded reconstruction error: {error}"
    
    def test_dtype_preservation(self, device):
        """Test that data types are preserved."""
        X = torch.randn(64, 96, dtype=torch.float32, device=device)
        
        coeffs, context = torch_shearlab.SLsheardecPadded2D(X, nScales=2)
        X_rec = torch_shearlab.SLshearrecPadded2D(coeffs, context)
        
        assert coeffs.dtype == torch.float32
        assert X_rec.dtype == torch.float32


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
