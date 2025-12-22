"""
PyTorch implementation of pyShearLab2D core transform functions.
Translated from pyShearLab2D.py (NumPy version).

This module provides the main 2D shearlet transform functions using
PyTorch tensors instead of NumPy arrays, enabling GPU acceleration.

Stefan Loock (original NumPy), PyTorch translation 2024
"""

from __future__ import division
from typing import Dict, Optional, Union, Tuple, Any
import torch
import numpy as np

from pyshearlab import pySLFilters_torch as filters_torch
from pyshearlab import pySLUtilities_torch as utils_torch


def SLgetShearletSystem2D(
    rows: int, cols: int, nScales: int,
    shearLevels: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
        shearLevels = np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
    elif isinstance(shearLevels, torch.Tensor):
        shearLevels = shearLevels.cpu().numpy()
    
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
    Shearlet decomposition of 2D data (PyTorch version).
    
    Args:
        X: 2D input data tensor (real-valued)
        shearletSystem: Shearlet system dictionary from SLgetShearletSystem2D
    
    Returns:
        coeffs: X x Y x N tensor of shearlet coefficients
    """
    shearlets = shearletSystem["shearlets"]
    nShearlets = shearletSystem["nShearlets"]
    device = X.device
    
    # Determine complex dtype
    if X.dtype == torch.float32:
        ctype = torch.complex64
    else:
        ctype = torch.complex128
    
    # Allocate output
    coeffs = torch.zeros(shearlets.shape, dtype=ctype, device=device)
    
    # Get data in frequency domain
    Xfreq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(X)))
    
    # Compute shearlet coefficients
    for j in range(nShearlets):
        coeffs[:, :, j] = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(Xfreq * torch.conj(shearlets[:, :, j]))
            )
        )
    
    # Return real part (imaginary part should be negligible for real input)
    return coeffs.real.to(X.dtype)


def SLshearrec2D(coeffs: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    2D reconstruction from shearlet coefficients (PyTorch version).
    
    Args:
        coeffs: X x Y x N tensor of shearlet coefficients
        shearletSystem: Shearlet system dictionary
    
    Returns:
        X: Reconstructed 2D data tensor
    """
    shearlets = shearletSystem["shearlets"]
    dualFrameWeights = shearletSystem["dualFrameWeights"]
    nShearlets = shearletSystem["nShearlets"]
    device = coeffs.device
    
    # Determine complex dtype
    if coeffs.dtype == torch.float32:
        ctype = torch.complex64
    else:
        ctype = torch.complex128
    
    # Accumulate in frequency domain
    X = torch.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=ctype, device=device)
    
    for j in range(nShearlets):
        coeff_freq = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(coeffs[:, :, j]))
        )
        X = X + coeff_freq * shearlets[:, :, j]
    
    # Normalize by dual frame weights and inverse FFT
    X = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(X / dualFrameWeights))
    )
    
    return X.real.to(coeffs.dtype)


def SLshearadjoint2D(coeffs: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    2D adjoint from shearlet coefficients (PyTorch version).
    
    The adjoint satisfies: <Ax, y> = <x, A*y>
    
    Args:
        coeffs: X x Y x N tensor of shearlet coefficients
        shearletSystem: Shearlet system dictionary
    
    Returns:
        X: Adjoint result, 2D data tensor
    """
    shearlets = shearletSystem["shearlets"]
    nShearlets = shearletSystem["nShearlets"]
    device = coeffs.device
    
    # Determine complex dtype
    if coeffs.dtype == torch.float32:
        ctype = torch.complex64
    else:
        ctype = torch.complex128
    
    # Accumulate in frequency domain
    X = torch.zeros((coeffs.shape[0], coeffs.shape[1]), dtype=ctype, device=device)
    
    for j in range(nShearlets):
        coeff_freq = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(coeffs[:, :, j]))
        )
        X = X + shearlets[:, :, j] * coeff_freq
    
    # Inverse FFT (no normalization by dual frame weights)
    X = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(X)))
    
    return X.real.to(coeffs.dtype)


def SLshearrecadjoint2D(X: torch.Tensor, shearletSystem: Dict[str, Any]) -> torch.Tensor:
    """
    Adjoint of (pseudo-)inverse of 2D data (PyTorch version).
    
    Note: This is also the (pseudo-)inverse of the adjoint.
    
    Args:
        X: 2D input data tensor
        shearletSystem: Shearlet system dictionary
    
    Returns:
        coeffs: X x Y x N tensor of shearlet coefficients
    """
    shearlets = shearletSystem["shearlets"]
    dualFrameWeights = shearletSystem["dualFrameWeights"]
    nShearlets = shearletSystem["nShearlets"]
    device = X.device
    
    # Determine complex dtype
    if X.dtype == torch.float32:
        ctype = torch.complex64
    else:
        ctype = torch.complex128
    
    # Allocate output
    coeffs = torch.zeros(shearlets.shape, dtype=ctype, device=device)
    
    # Get data in frequency domain, normalized by dual frame weights
    Xfreq = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(X)))
    Xfreq = Xfreq / dualFrameWeights
    
    # Compute coefficients
    for j in range(nShearlets):
        coeffs[:, :, j] = torch.fft.fftshift(
            torch.fft.ifft2(
                torch.fft.ifftshift(Xfreq * torch.conj(shearlets[:, :, j]))
            )
        )
    
    return coeffs.real.to(X.dtype)


# ============================================================================
# High-level API with Automatic Square Padding
# ============================================================================

def SLsheardecPadded2D(
    X: torch.Tensor,
    nScales: int,
    shearLevels: Optional[Union[np.ndarray, torch.Tensor]] = None,
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
