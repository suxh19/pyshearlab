"""
Test suite for pySLFilters_torch.py

Validates that PyTorch implementations produce identical results to NumPy versions.
"""

import pytest
import numpy as np
import torch

from pyshearlab import pySLFilters as np_filters
from pyshearlab import pySLFilters_torch as torch_filters


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request):
    return request.param


@pytest.fixture(params=['cpu'])
def device(request):
    # Add 'cuda' if GPU testing is desired: params=['cpu', 'cuda']
    return request.param


# ============================================================================
# Test MakeONFilter
# ============================================================================

class TestMakeONFilter:
    """Test orthonormal filter generation."""
    
    @pytest.mark.parametrize("filter_type,par", [
        ('Haar', 1),
        ('Beylkin', 1),
        ('Coiflet', 1), ('Coiflet', 2), ('Coiflet', 3), ('Coiflet', 4), ('Coiflet', 5),
        ('Daubechies', 4), ('Daubechies', 6), ('Daubechies', 8), ('Daubechies', 10),
        ('Daubechies', 12), ('Daubechies', 14), ('Daubechies', 16), ('Daubechies', 18), ('Daubechies', 20),
        ('Symmlet', 4), ('Symmlet', 5), ('Symmlet', 6), ('Symmlet', 7),
        ('Symmlet', 8), ('Symmlet', 9), ('Symmlet', 10),
        ('Vaidyanathan', 1),
        ('Battle', 1), ('Battle', 3), ('Battle', 5),
    ])
    def test_consistency(self, filter_type, par, device):
        """Test that PyTorch output matches NumPy output."""
        np_result = np_filters.MakeONFilter(filter_type, par)
        torch_result = torch_filters.MakeONFilter(filter_type, par, 
                                                   dtype=torch.float64, device=device)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )
    
    def test_filter_normalization(self, device):
        """Test that filters are properly normalized."""
        torch_result = torch_filters.MakeONFilter('Daubechies', 8, device=device)
        norm = torch.linalg.norm(torch_result)
        assert torch.abs(norm - 1.0) < 1e-10


# ============================================================================
# Test MirrorFilt
# ============================================================================

class TestMirrorFilt:
    """Test mirror filter modulation."""
    
    @pytest.mark.parametrize("length", [4, 8, 16, 32])
    def test_consistency(self, length, device):
        """Test that PyTorch output matches NumPy output."""
        x_np = np.random.randn(length)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_filters.MirrorFilt(x_np)
        torch_result = torch_filters.MirrorFilt(x_torch)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )
    
    def test_alternating_sign(self, device):
        """Test that alternating sign pattern is correct."""
        x = torch.ones(4, device=device)
        result = torch_filters.MirrorFilt(x)
        expected = torch.tensor([1, -1, 1, -1], dtype=x.dtype, device=device)
        assert torch.allclose(result, expected)


# ============================================================================
# Test modulate2
# ============================================================================

class TestModulate2:
    """Test 2D modulation."""
    
    @pytest.mark.parametrize("mod_type", ['r', 'c', 'b'])
    def test_consistency_2d(self, mod_type, device):
        """Test 2D modulation matches NumPy."""
        x_np = np.random.randn(8, 8)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_filters.modulate2(x_np, mod_type)
        torch_result = torch_filters.modulate2(x_torch, mod_type)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )
    
    @pytest.mark.parametrize("mod_type", ['r', 'c', 'b'])
    def test_consistency_1d(self, mod_type, device):
        """Test 1D modulation matches NumPy."""
        x_np = np.random.randn(8)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_filters.modulate2(x_np, mod_type)
        torch_result = torch_filters.modulate2(x_torch, mod_type)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )
    
    @pytest.mark.parametrize("shape", [(4, 4), (8, 16), (16, 8)])
    def test_different_shapes(self, shape, device):
        """Test modulation with different shapes."""
        x_np = np.random.randn(*shape)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_filters.modulate2(x_np, 'b')
        torch_result = torch_filters.modulate2(x_torch, 'b')
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )


# ============================================================================
# Test dmaxflat
# ============================================================================

class TestDmaxflat:
    """Test diamond maxflat filter generation."""
    
    @pytest.mark.parametrize("N", [1, 2, 3, 4, 5, 6, 7])
    @pytest.mark.parametrize("d", [0, 1])
    def test_consistency(self, N, d, device):
        """Test that PyTorch output matches NumPy output."""
        np_result = np_filters.dmaxflat(N, d)
        torch_result = torch_filters.dmaxflat(N, d, dtype=torch.float64, device=device)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )
    
    def test_symmetry(self, device):
        """Test that filters are symmetric."""
        h = torch_filters.dmaxflat(4, 0, device=device)
        # Check left-right symmetry
        assert torch.allclose(h, torch.flip(h, [1]))
        # Check up-down symmetry
        assert torch.allclose(h, torch.flip(h, [0]))
    
    def test_invalid_N(self, device):
        """Test error handling for invalid N values."""
        with pytest.raises(ValueError):
            torch_filters.dmaxflat(0, 0, device=device)
        with pytest.raises(ValueError):
            torch_filters.dmaxflat(8, 0, device=device)


# ============================================================================
# Test mctrans
# ============================================================================

class TestMctrans:
    """Test McClellan transformation."""
    
    def test_consistency_simple(self, device):
        """Test simple McClellan transform matches NumPy."""
        b_np = np.array([1, 2, 1]) / 4
        t_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
        
        b_torch = torch.from_numpy(b_np).to(device)
        t_torch = torch.from_numpy(t_np).to(device)
        
        np_result = np_filters.mctrans(b_np, t_np)
        torch_result = torch_filters.mctrans(b_torch, t_torch)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-8, atol=1e-10
        )
    
    def test_consistency_longer(self, device):
        """Test McClellan transform with longer filter."""
        b_np = np.array([0.026748757411, -0.016864118443, -0.078223266529,
                    0.266864118443, 0.602949018236, 0.266864118443,
                    -0.078223266529, -0.016864118443, 0.026748757411])
        t_np = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4
        
        b_torch = torch.from_numpy(b_np).to(device)
        t_torch = torch.from_numpy(t_np).to(device)
        
        np_result = np_filters.mctrans(b_np, t_np)
        torch_result = torch_filters.mctrans(b_torch, t_torch)
        
        np.testing.assert_allclose(
            np_result, 
            torch_result.cpu().numpy(), 
            rtol=1e-8, atol=1e-10
        )


# ============================================================================
# Test dfilters
# ============================================================================

class TestDfilters:
    """Test directional filter generation."""
    
    @pytest.mark.parametrize("fname,type_", [
        ('haar', 'd'), ('haar', 'r'),
        ('vk', 'd'), ('vk', 'r'),
        ('ko', 'd'), ('ko', 'r'),
        ('kos', 'd'), ('kos', 'r'),
        ('cd', 'd'), ('cd', 'r'),
        ('oqf_362', 'd'), ('oqf_362', 'r'),
        ('dmaxflat4', 'd'), ('dmaxflat4', 'r'),
        ('dmaxflat5', 'd'), ('dmaxflat5', 'r'),
        ('dmaxflat6', 'd'), ('dmaxflat6', 'r'),
        ('dmaxflat7', 'd'), ('dmaxflat7', 'r'),
    ])
    def test_consistency(self, fname, type_, device):
        """Test that PyTorch output matches NumPy output."""
        np_h0, np_h1 = np_filters.dfilters(fname, type_)
        torch_h0, torch_h1 = torch_filters.dfilters(fname, type_, 
                                                     dtype=torch.float64, device=device)
        
        np.testing.assert_allclose(
            np_h0, 
            torch_h0.cpu().numpy(), 
            rtol=1e-8, atol=1e-10,
            err_msg=f"h0 mismatch for {fname} type={type_}"
        )
        np.testing.assert_allclose(
            np_h1, 
            torch_h1.cpu().numpy(), 
            rtol=1e-8, atol=1e-10,
            err_msg=f"h1 mismatch for {fname} type={type_}"
        )
    
    def test_unknown_filter(self, device):
        """Test error handling for unknown filter names."""
        with pytest.raises(ValueError):
            torch_filters.dfilters('unknown_filter', 'd', device=device)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests using typical filter combinations."""
    
    def test_typical_shearlet_filters(self, device):
        """Test the typical filter setup used in ShearLab."""
        # This is the default filter setup in SLprepareFilters2D
        np_h0, np_h1 = np_filters.dfilters('dmaxflat4', 'd')
        np_h0 = np_h0 / np.sqrt(2)
        np_directional = np_filters.modulate2(np_h0, 'c')
        
        torch_h0, torch_h1 = torch_filters.dfilters('dmaxflat4', 'd', device=device)
        torch_h0 = torch_h0 / torch.sqrt(torch.tensor(2.0, device=device))
        torch_directional = torch_filters.modulate2(torch_h0, 'c')
        
        np.testing.assert_allclose(
            np_directional, 
            torch_directional.cpu().numpy(), 
            rtol=1e-6, atol=1e-8  # Relaxed tolerance for numerical precision differences
        )
    
    def test_scaling_filter_with_mirror(self, device):
        """Test scaling filter + mirror filter combination."""
        np_scaling = np.array([0.0104933261758410, -0.0263483047033631,
                        -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                        0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                        0.0104933261758408])
        np_wavelet = np_filters.MirrorFilt(np_scaling)
        
        torch_scaling = torch.tensor(np_scaling, dtype=torch.float64, device=device)
        torch_wavelet = torch_filters.MirrorFilt(torch_scaling)
        
        np.testing.assert_allclose(
            np_wavelet, 
            torch_wavelet.cpu().numpy(), 
            rtol=1e-10, atol=1e-12
        )


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
