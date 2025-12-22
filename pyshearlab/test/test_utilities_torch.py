"""
Test suite for pySLUtilities_torch.py

Validates that PyTorch implementations produce identical results to NumPy versions.
"""

import pytest
import numpy as np
import torch

from pyshearlab import pySLUtilities as np_utils
from pyshearlab import pySLUtilities_torch as torch_utils


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(params=['cpu'])
def device(request):
    return request.param


# ============================================================================
# Test SLpadArray
# ============================================================================

class TestSLpadArray:
    """Test array padding."""
    
    def test_1d_padding(self, device):
        """Test 1D array padding."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLpadArray(x_np, 8)
        torch_result = torch_utils.SLpadArray(x_torch, 8)
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)
    
    def test_2d_padding(self, device):
        """Test 2D array padding."""
        x_np = np.random.randn(4, 4)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLpadArray(x_np, np.array([8, 8]))
        torch_result = torch_utils.SLpadArray(x_torch, [8, 8])
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)
    
    def test_asymmetric_padding(self, device):
        """Test asymmetric 2D padding."""
        x_np = np.random.randn(3, 5)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLpadArray(x_np, np.array([7, 9]))
        torch_result = torch_utils.SLpadArray(x_torch, [7, 9])
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)


# ============================================================================
# Test SLupsample
# ============================================================================

class TestSLupsample:
    """Test upsampling."""
    
    def test_1d_upsample(self, device):
        """Test 1D array upsampling."""
        x_np = np.array([1.0, 2.0, 3.0, 4.0])
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLupsample(x_np, 1, 2)
        torch_result = torch_utils.SLupsample(x_torch, 1, 2)
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)
    
    def test_2d_upsample_rows(self, device):
        """Test 2D upsampling along rows."""
        x_np = np.random.randn(4, 4)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLupsample(x_np, 1, 3)
        torch_result = torch_utils.SLupsample(x_torch, 1, 3)
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)
    
    def test_2d_upsample_cols(self, device):
        """Test 2D upsampling along columns."""
        x_np = np.random.randn(4, 4)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLupsample(x_np, 2, 3)
        torch_result = torch_utils.SLupsample(x_torch, 2, 3)
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)


# ============================================================================
# Test SLdshear
# ============================================================================

class TestSLdshear:
    """Test discrete shearing operator."""
    
    @pytest.mark.parametrize("k", [-2, -1, 0, 1, 2])
    @pytest.mark.parametrize("axis", [1, 2])
    def test_shearing(self, k, axis, device):
        """Test shearing with different k and axis values."""
        x_np = np.random.randn(8, 8)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_result = np_utils.SLdshear(x_np, k, axis)
        torch_result = torch_utils.SLdshear(x_torch, k, axis)
        
        np.testing.assert_allclose(
            np_result, torch_result.cpu().numpy(), rtol=1e-10)
    
    def test_zero_shear(self, device):
        """Test that k=0 returns input unchanged."""
        x_np = np.random.randn(8, 8)
        x_torch = torch.from_numpy(x_np).to(device)
        
        result = torch_utils.SLdshear(x_torch, 0, 1)
        np.testing.assert_allclose(x_np, result.cpu().numpy(), rtol=1e-10)


# ============================================================================
# Test SLsymmetricPad2D and SLcrop2D
# ============================================================================

class TestSymmetricPadCrop:
    """Test symmetric padding and cropping."""
    
    def test_pad_and_crop_roundtrip(self, device):
        """Test that pad then crop returns original."""
        x_np = np.random.randn(32, 64)
        x_torch = torch.from_numpy(x_np).to(device)
        
        padded, pad_info = torch_utils.SLsymmetricPad2D(x_torch)
        cropped = torch_utils.SLcrop2D(padded, pad_info)
        
        np.testing.assert_allclose(x_np, cropped.cpu().numpy(), rtol=1e-10)
    
    def test_consistency_with_numpy(self, device):
        """Test padding matches NumPy version."""
        x_np = np.random.randn(32, 48)
        x_torch = torch.from_numpy(x_np).to(device)
        
        np_padded, np_info = np_utils.SLsymmetricPad2D(x_np, pad_mode='reflect')
        torch_padded, torch_info = torch_utils.SLsymmetricPad2D(x_torch, pad_mode='reflect')
        
        np.testing.assert_allclose(
            np_padded, torch_padded.cpu().numpy(), rtol=1e-10)
        
        assert np_info['original_shape'] == torch_info['original_shape']
        assert np_info['padded_shape'] == torch_info['padded_shape']


# ============================================================================
# Test SLgetShearletIdxs2D
# ============================================================================

class TestSLgetShearletIdxs2D:
    """Test shearlet index generation."""
    
    def test_consistency(self, device):
        """Test that indices match NumPy version."""
        shearLevels = np.array([1, 1, 2])
        
        np_result = np_utils.SLgetShearletIdxs2D(shearLevels, 0)
        torch_result = torch_utils.SLgetShearletIdxs2D(shearLevels, 0)
        
        np.testing.assert_array_equal(np_result, torch_result)
    
    def test_full_system(self, device):
        """Test full shearlet system."""
        shearLevels = np.array([1, 2])
        
        np_result = np_utils.SLgetShearletIdxs2D(shearLevels, 1)
        torch_result = torch_utils.SLgetShearletIdxs2D(shearLevels, 1)
        
        np.testing.assert_array_equal(np_result, torch_result)


# ============================================================================
# Test Core Frequency Domain Functions
# ============================================================================

class TestSLgetWedgeBandpassAndLowpassFilters2D:
    """Test wedge, bandpass and lowpass filter generation."""
    
    def test_basic_consistency(self, device):
        """Test basic filter generation matches NumPy."""
        rows, cols = 128, 128  # Larger size needed for filters
        shearLevels = np.array([1, 1])
        
        # NumPy version
        np_wedge, np_bandpass, np_lowpass = np_utils.SLgetWedgeBandpassAndLowpassFilters2D(
            rows, cols, shearLevels)
        
        # PyTorch version
        torch_wedge, torch_bandpass, torch_lowpass = torch_utils.SLgetWedgeBandpassAndLowpassFilters2D(
            rows, cols, shearLevels, device=device)
        
        # Compare lowpass
        np.testing.assert_allclose(
            np_lowpass, torch_lowpass.cpu().numpy(), rtol=1e-6, atol=1e-8)
        
        # Compare bandpass
        np.testing.assert_allclose(
            np_bandpass, torch_bandpass.cpu().numpy(), rtol=1e-6, atol=1e-8)


class TestSLprepareFilters2D:
    """Test filter preparation."""
    
    def test_basic(self, device):
        """Test basic filter preparation."""
        rows, cols, nScales = 128, 128, 2  # Larger size
        
        result = torch_utils.SLprepareFilters2D(rows, cols, nScales, device=device)
        
        assert result['rows'] == rows
        assert result['cols'] == cols
        assert 'cone1' in result
        assert 'cone2' in result
        assert 'wedge' in result['cone1']
        assert 'bandpass' in result['cone1']
        assert 'lowpass' in result['cone1']


class TestSLgetShearlets2D:
    """Test shearlet computation."""
    
    def test_basic(self, device):
        """Test basic shearlet computation."""
        rows, cols, nScales = 128, 128, 2  # Larger size
        
        preparedFilters = torch_utils.SLprepareFilters2D(rows, cols, nScales, device=device)
        shearlets, RMS, dualFrameWeights = torch_utils.SLgetShearlets2D(preparedFilters)
        
        assert shearlets.shape[0] == rows
        assert shearlets.shape[1] == cols
        assert shearlets.shape[2] > 0
        assert RMS.numel() == shearlets.shape[2]
        assert dualFrameWeights.shape == (rows, cols)


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
