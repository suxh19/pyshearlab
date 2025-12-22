"""
Test cases for the square padding API functions.

These tests validate:
1. SLsymmetricPad2D and SLcrop2D utility functions
2. SLsheardecPadded2D and SLshearrecPadded2D high-level API
3. Filter wrapping prevention for non-square images
"""

import numpy as np
import pyshearlab
import pytest


class TestPaddingUtilities:
    """Tests for SLsymmetricPad2D and SLcrop2D functions."""
    
    def test_symmetric_pad_square_output(self):
        """Verify that padding creates a square output."""
        img = np.random.randn(256, 1024).astype(np.float64)
        padded, pad_info = pyshearlab.SLsymmetricPad2D(img)
        
        assert padded.shape[0] == padded.shape[1], "Output should be square"
        assert padded.shape[0] == 1024, "Square size should match larger dimension"
    
    def test_symmetric_pad_info_correct(self):
        """Verify pad_info contains correct values."""
        img = np.random.randn(256, 1024).astype(np.float64)
        padded, pad_info = pyshearlab.SLsymmetricPad2D(img)
        
        assert pad_info['original_shape'] == (256, 1024)
        assert pad_info['padded_shape'] == (1024, 1024)
        assert pad_info['pad_top'] == 384
        assert pad_info['pad_bottom'] == 384
        assert pad_info['pad_left'] == 0
        assert pad_info['pad_right'] == 0
    
    def test_crop_restores_original_shape_2d(self):
        """SLcrop2D should restore original 2D shape."""
        img = np.random.randn(256, 1024).astype(np.float64)
        padded, pad_info = pyshearlab.SLsymmetricPad2D(img)
        cropped = pyshearlab.SLcrop2D(padded, pad_info)
        
        assert cropped.shape == img.shape
    
    def test_crop_restores_original_shape_3d(self):
        """SLcrop2D should work with 3D coefficient arrays."""
        img = np.random.randn(256, 1024).astype(np.float64)
        padded, pad_info = pyshearlab.SLsymmetricPad2D(img)
        
        # Simulate 3D coefficient array
        coeffs_3d = np.random.randn(1024, 1024, 17).astype(np.float64)
        cropped = pyshearlab.SLcrop2D(coeffs_3d, pad_info)
        
        assert cropped.shape == (256, 1024, 17)
    
    def test_pad_crop_roundtrip_preserves_center(self):
        """Center region should be preserved after pad->crop."""
        img = np.random.randn(256, 1024).astype(np.float64)
        padded, pad_info = pyshearlab.SLsymmetricPad2D(img)
        cropped = pyshearlab.SLcrop2D(padded, pad_info)
        
        np.testing.assert_array_equal(cropped, img)


class TestPaddedAPI:
    """Tests for SLsheardecPadded2D and SLshearrecPadded2D functions."""
    
    @pytest.fixture
    def non_square_image(self):
        """Generate a non-square test image."""
        np.random.seed(42)
        return np.random.randn(256, 1024).astype(np.float64)
    
    @pytest.fixture
    def square_image(self):
        """Generate a square test image."""
        np.random.seed(42)
        return np.random.randn(256, 256).astype(np.float64)
    
    def test_decomposition_returns_original_size(self, non_square_image):
        """Coefficients should have the original image size."""
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(non_square_image, nScales=2)
        
        assert coeffs.shape[0] == 256
        assert coeffs.shape[1] == 1024
        assert coeffs.ndim == 3
    
    def test_context_contains_required_keys(self, non_square_image):
        """Context should contain shearletSystem, pad_info, original_dtype."""
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(non_square_image, nScales=2)
        
        assert 'shearletSystem' in ctx
        assert 'pad_info' in ctx
        assert 'original_dtype' in ctx
        assert ctx['original_dtype'] == np.float64
    
    def test_reconstruction_shape_matches_input(self, non_square_image):
        """Reconstruction should have the same shape as input."""
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(non_square_image, nScales=2)
        reconstructed = pyshearlab.SLshearrecPadded2D(coeffs, ctx)
        
        assert reconstructed.shape == non_square_image.shape
    
    def test_reconstruction_dtype_matches_input(self, non_square_image):
        """Reconstruction should preserve original dtype."""
        img_f32 = non_square_image.astype(np.float32)
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(img_f32, nScales=2)
        reconstructed = pyshearlab.SLshearrecPadded2D(coeffs, ctx)
        
        assert reconstructed.dtype == np.float32
    
    def test_reconstruction_error_is_small(self, non_square_image):
        """Reconstruction error should be small (within numerical precision)."""
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(non_square_image, nScales=2)
        reconstructed = pyshearlab.SLshearrecPadded2D(coeffs, ctx)
        
        # For non-Parseval frames, reconstruction error is typically ~0.5-1%
        rel_error = np.linalg.norm(non_square_image - reconstructed) / np.linalg.norm(non_square_image)
        assert rel_error < 0.02, f"Relative reconstruction error too large: {rel_error}"
    
    def test_square_image_also_works(self, square_image):
        """API should also work correctly for square images."""
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(square_image, nScales=2)
        reconstructed = pyshearlab.SLshearrecPadded2D(coeffs, ctx)
        
        assert reconstructed.shape == square_image.shape
        rel_error = np.linalg.norm(square_image - reconstructed) / np.linalg.norm(square_image)
        assert rel_error < 0.02


class TestFilterWrappingPrevention:
    """Tests specifically for filter wrapping prevention."""
    
    def test_no_edge_artifacts_in_coefficients(self):
        """Coefficients should not have wrapping artifacts at edges."""
        # Create an image with clear horizontal structure
        img = np.zeros((256, 1024), dtype=np.float64)
        img[100:156, :] = 1.0  # Horizontal band in the middle
        
        coeffs, ctx = pyshearlab.SLsheardecPadded2D(img, nScales=2)
        
        # The top and bottom edges should not have significant energy
        # (wrapping would cause the band to appear at edges)
        top_edge_energy = np.sum(coeffs[:10, :, :] ** 2)
        bottom_edge_energy = np.sum(coeffs[-10:, :, :] ** 2)
        middle_energy = np.sum(coeffs[100:156, :, :] ** 2)
        
        # Edge energy should be much smaller than middle energy
        edge_ratio = (top_edge_energy + bottom_edge_energy) / middle_energy
        assert edge_ratio < 0.1, f"Edge energy ratio too high: {edge_ratio}"


if __name__ == '__main__':
    pytest.main([str(__file__.replace('\\', '/')), '-v'])
