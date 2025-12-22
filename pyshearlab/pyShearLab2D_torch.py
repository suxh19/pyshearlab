"""
PyTorch implementation of pyShearLab2D core transform functions.
Translated from pyShearLab2D.py (NumPy version).

This module provides the main 2D shearlet transform functions using
PyTorch tensors instead of NumPy arrays, enabling GPU acceleration.

Stefan Loock (original NumPy), PyTorch translation 2024
"""

from __future__ import division
from typing import Dict, Optional, Union, Tuple, Any, List
import math
import torch

from pyshearlab import pySLFilters_torch as filters_torch
from pyshearlab import pySLUtilities_torch as utils_torch


def SLgetShearletSystem2D(
    rows: int, cols: int, nScales: int,
    shearLevels: Optional[Union[torch.Tensor, List]] = None,
    full: int = 0,
    directionalFilter: Optional[torch.Tensor] = None,
    quadratureMirrorFilter: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu',
    dtype: torch.dtype = torch.float64
) -> Dict[str, Any]:
    """
    Compute a 2D shearlet system (PyTorch version).
    
    Args:
        rows: Number of rows
        cols: Number of columns
        nScales: Number of scales (>= 1)
        shearLevels: Shear levels per scale (default: ceil(1:nScales)/2)
        full: 0 for reduced system, 1 for full system
        directionalFilter: Optional 2D directional filter
        quadratureMirrorFilter: Optional 1D QMF filter
        device: Computation device ('cpu' or 'cuda')
        dtype: Data type (torch.float32 or torch.float64)
    
    Returns:
        Dictionary containing the shearlet system
    """
    # Set defaults
    if shearLevels is None:
        # Equivalent to: np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
        shearLevels = [int(math.ceil(i / 2)) for i in range(1, nScales + 1)]
    elif isinstance(shearLevels, torch.Tensor):
        shearLevels = shearLevels.cpu().tolist()
    
    if directionalFilter is None:
        h0, _ = filters_torch.dfilters('dmaxflat4', 'd', dtype=dtype, device=device)
        h0 = h0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        directionalFilter = filters_torch.modulate2(h0, 'c')
    
    if quadratureMirrorFilter is None:
        quadratureMirrorFilter = torch.tensor([
            0.0104933261758410, -0.0263483047033631, -0.0517766952966370,
            0.276348304703363, 0.582566738241592, 0.276348304703363,
            -0.0517766952966369, -0.0263483047033631, 0.0104933261758408
        ], dtype=dtype, device=device)
    
    # Prepare filters
    preparedFilters = utils_torch.SLprepareFilters2D(
        rows, cols, nScales, shearLevels,
        directionalFilter, quadratureMirrorFilter,
        device=device
    )
    
    # Get shearlet indices
    shearletIdxs = utils_torch.SLgetShearletIdxs2D(shearLevels, full)
    
    # Get shearlets
    shearlets, RMS, dualFrameWeights = utils_torch.SLgetShearlets2D(
        preparedFilters, shearletIdxs)
    
    # Create system dictionary
    shearletSystem = {
        "shearlets": shearlets,
        "size": (rows, cols),
        "shearLevels": shearLevels,
        "full": full,
        "nShearlets": len(shearletIdxs),
        "shearletIdxs": shearletIdxs,
        "dualFrameWeights": dualFrameWeights,
        "RMS": RMS,
        "device": device,
        "dtype": dtype,
        "isComplex": 0
    }
    
    return shearletSystem


def SLsheardec2D(X: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    Shearlet decomposition of 2D data (PyTorch version, vectorized).
    
    This is an optimized version using batch FFT operations instead of loops.
    
    Args:
        X: 2D input data tensor (real-valued)
        shearletSystem: Shearlet system dictionary from SLgetShearletSystem2D
    
    Returns:
        coeffs: X x Y x N tensor of shearlet coefficients
    """
    shearlets = shearletSystem["shearlets"]  # (rows, cols, nShearlets)
    
    # Get data in frequency domain
    Xfreq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(X)))
    
    # Batch multiply: Xfreq (rows, cols) * conj(shearlets) (rows, cols, N)
    # Broadcasting Xfreq to (rows, cols, 1) then multiply
    products = Xfreq.unsqueeze(-1) * torch.conj(shearlets)  # (rows, cols, N)
    
    # Batch ifftshift, ifft2, fftshift on dims (0,1)
    # Move N dimension to front for batch processing: (N, rows, cols)
    products_t = products.permute(2, 0, 1)  # (N, rows, cols)
    products_t = torch.fft.ifftshift(products_t, dim=(1, 2))
    coeffs_t = torch.fft.ifft2(products_t, dim=(1, 2))
    coeffs_t = torch.fft.fftshift(coeffs_t, dim=(1, 2))
    
    # Move back to (rows, cols, N)
    coeffs = coeffs_t.permute(1, 2, 0)
    
    # Return real part (imaginary part should be negligible for real input)
    return coeffs.real.to(X.dtype)


def SLshearrec2D(coeffs: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    2D reconstruction from shearlet coefficients (PyTorch version, vectorized).
    
    This is an optimized version using batch FFT operations instead of loops.
    
    Args:
        coeffs: X x Y x N tensor of shearlet coefficients
        shearletSystem: Shearlet system dictionary
    
    Returns:
        X: Reconstructed 2D data tensor
    """
    shearlets = shearletSystem["shearlets"]  # (rows, cols, N)
    dualFrameWeights = shearletSystem["dualFrameWeights"]  # (rows, cols)
    
    # Batch fft2 on all coefficients
    # Move N to batch dimension: (N, rows, cols)
    coeffs_t = coeffs.permute(2, 0, 1)
    coeffs_t = torch.fft.ifftshift(coeffs_t, dim=(1, 2))
    coeffs_freq_t = torch.fft.fft2(coeffs_t, dim=(1, 2))
    coeffs_freq_t = torch.fft.fftshift(coeffs_freq_t, dim=(1, 2))
    
    # Move back to (rows, cols, N)
    coeffs_freq = coeffs_freq_t.permute(1, 2, 0)
    
    # Multiply with shearlets and sum over N dimension
    X = torch.sum(coeffs_freq * shearlets, dim=2)
    
    # Normalize by dual frame weights and inverse FFT
    X = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(X / dualFrameWeights))
    )
    
    return X.real.to(coeffs.dtype)


def SLshearadjoint2D(coeffs: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    2D adjoint from shearlet coefficients (PyTorch version, vectorized).
    
    The adjoint satisfies: <Ax, y> = <x, A*y>
    
    This is an optimized version using batch FFT operations instead of loops.
    
    Args:
        coeffs: X x Y x N tensor of shearlet coefficients
        shearletSystem: Shearlet system dictionary
    
    Returns:
        X: Adjoint result, 2D data tensor
    """
    shearlets = shearletSystem["shearlets"]  # (rows, cols, N)
    
    # Batch fft2 on all coefficients
    # Move N to batch dimension: (N, rows, cols)
    coeffs_t = coeffs.permute(2, 0, 1)
    coeffs_t = torch.fft.ifftshift(coeffs_t, dim=(1, 2))
    coeffs_freq_t = torch.fft.fft2(coeffs_t, dim=(1, 2))
    coeffs_freq_t = torch.fft.fftshift(coeffs_freq_t, dim=(1, 2))
    
    # Move back to (rows, cols, N)
    coeffs_freq = coeffs_freq_t.permute(1, 2, 0)
    
    # Multiply with shearlets and sum over N dimension
    X = torch.sum(shearlets * coeffs_freq, dim=2)
    
    # Inverse FFT (no normalization by dual frame weights)
    X = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))
    
    return X.real.to(coeffs.dtype)


def SLshearrecadjoint2D(X: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    Adjoint of (pseudo-)inverse of 2D data (PyTorch version, vectorized).
    
    Note: This is also the (pseudo-)inverse of the adjoint.
    
    This is an optimized version using batch FFT operations instead of loops.
    
    Args:
        X: 2D input data tensor
        shearletSystem: Shearlet system dictionary
    
    Returns:
        coeffs: X x Y x N tensor of shearlet coefficients
    """
    shearlets = shearletSystem["shearlets"]  # (rows, cols, N)
    dualFrameWeights = shearletSystem["dualFrameWeights"]  # (rows, cols)
    
    # Get data in frequency domain, normalized by dual frame weights
    Xfreq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(X)))
    Xfreq = Xfreq / dualFrameWeights
    
    # Batch multiply: Xfreq (rows, cols) * conj(shearlets) (rows, cols, N)
    products = Xfreq.unsqueeze(-1) * torch.conj(shearlets)  # (rows, cols, N)
    
    # Batch ifftshift, ifft2, fftshift on dims (0,1)
    # Move N dimension to front for batch processing: (N, rows, cols)
    products_t = products.permute(2, 0, 1)  # (N, rows, cols)
    products_t = torch.fft.ifftshift(products_t, dim=(1, 2))
    coeffs_t = torch.fft.ifft2(products_t, dim=(1, 2))
    coeffs_t = torch.fft.fftshift(coeffs_t, dim=(1, 2))
    
    # Move back to (rows, cols, N)
    coeffs = coeffs_t.permute(1, 2, 0)
    
    return coeffs.real.to(X.dtype)


# ============================================================================
# High-level API with Automatic Square Padding
# ============================================================================

def SLsheardecPadded2D(
    X: torch.Tensor,
    nScales: int,
    shearLevels: Optional[Union[torch.Tensor, List]] = None,
    pad_mode: str = 'reflect',
    full: int = 0,
    directionalFilter: Optional[torch.Tensor] = None,
    quadratureMirrorFilter: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Shearlet decomposition with automatic square padding (PyTorch version).
    
    This function solves filter wrapping issues for non-square images.
    
    Args:
        X: 2D input tensor
        nScales: Number of scales
        shearLevels: Optional shear levels per scale
        pad_mode: Padding mode ('reflect' or 'replicate')  
        full: 0 for reduced system, 1 for full
        directionalFilter: Optional directional filter
        quadratureMirrorFilter: Optional QMF filter
    
    Returns:
        Tuple of (coeffs, context) where:
            - coeffs: Cropped coefficients matching original input size
            - context: Dictionary for use in SLshearrecPadded2D
    """
    device = X.device
    dtype = X.dtype
    original_dtype = X.dtype
    
    # Convert to float64 for computation if needed
    if dtype == torch.float32:
        X_compute = X.to(torch.float64)
    else:
        X_compute = X
    
    # Pad to square
    X_padded, pad_info = utils_torch.SLsymmetricPad2D(X_compute, make_square=True, pad_mode=pad_mode)
    padded_rows, padded_cols = X_padded.shape
    
    # Create shearlet system for padded size
    shearletSystem = SLgetShearletSystem2D(
        padded_rows, padded_cols, nScales,
        shearLevels=shearLevels,
        full=full,
        directionalFilter=directionalFilter,
        quadratureMirrorFilter=quadratureMirrorFilter,
        device=device
    )
    
    # Decompose
    coeffs_padded = SLsheardec2D(X_padded, shearletSystem)
    
    # Crop coefficients to original size
    coeffs = utils_torch.SLcrop2D(coeffs_padded, pad_info)
    
    # Build context for reconstruction
    context = {
        'shearletSystem': shearletSystem,
        'pad_info': pad_info,
        'original_dtype': original_dtype,
        'coeffs_padded': coeffs_padded
    }
    
    return coeffs.to(original_dtype), context


def SLshearrecPadded2D(coeffs: torch.Tensor, context: Dict[str, Any]) -> torch.Tensor:
    """
    Reconstruct from shearlet coefficients obtained via SLsheardecPadded2D.
    
    Args:
        coeffs: X x Y x N tensor of shearlet coefficients (original image size)
        context: Dictionary returned by SLsheardecPadded2D
    
    Returns:
        X: Reconstructed 2D data with original input size
    """
    shearletSystem = context['shearletSystem']
    pad_info = context['pad_info']
    original_dtype = context['original_dtype']
    coeffs_padded_original = context['coeffs_padded']
    
    pad_top = pad_info['pad_top']
    pad_left = pad_info['pad_left']
    original_rows, original_cols = pad_info['original_shape']
    
    # Update central region with potentially modified coefficients
    coeffs_padded = coeffs_padded_original.clone()
    coeffs_padded[pad_top : pad_top + original_rows,
                  pad_left : pad_left + original_cols, :] = coeffs.to(coeffs_padded.dtype)
    
    # Reconstruct on padded coefficients
    X_padded = SLshearrec2D(coeffs_padded, shearletSystem)
    
    # Crop to original size
    X_rec = utils_torch.SLcrop2D(X_padded, pad_info)
    
    return X_rec.to(original_dtype)
