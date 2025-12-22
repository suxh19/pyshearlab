"""
PyTorch implementation of pyShearLab2D utilities.
Translated from pySLUtilities.py (NumPy version).

This module provides utility functions for the ShearLab2D toolbox using
PyTorch tensors instead of NumPy arrays, enabling GPU acceleration.

Stefan Loock (original NumPy), PyTorch translation 2024
"""

from __future__ import division
import sys
from typing import Tuple, Optional, Union, Dict, List, Any
import torch
import torch.nn.functional as F
import numpy as np

from pyshearlab import pySLFilters_torch as filters_torch


# ============================================================================
# Phase 2a: Simple Utility Functions
# ============================================================================

def SLpadArray(array: torch.Tensor, newSize: Union[int, torch.Tensor, np.ndarray, List, Tuple],
               device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Implements the padding of an array as performed by the Matlab variant.
    
    Centers the input array in a larger array of zeros.
    
    Args:
        array: 1D or 2D input tensor
        newSize: Target size (scalar for 1D, or (rows, cols) for 2D)
        device: Device for output tensor (default: same as input)
    
    Returns:
        Padded tensor with zeros around the original content
    """
    if device is None:
        device = array.device
    
    if isinstance(newSize, (int, float)):
        # 1D case
        newSize = int(newSize)
        currSize = array.numel()
        paddedArray = torch.zeros(newSize, dtype=array.dtype, device=device)
        sizeDiff = newSize - currSize
        
        if sizeDiff < 0:
            raise ValueError("newSize is smaller than actual array size.")
        
        if sizeDiff % 2 == 0:
            padSizes = sizeDiff // 2
            idxModifier = 0
        else:
            padSizes = (sizeDiff + 1) // 2
            if currSize % 2 == 0:
                idxModifier = 1
            else:
                idxModifier = 0
        
        paddedArray[padSizes - idxModifier : padSizes + currSize - idxModifier] = array.flatten()
    else:
        # 2D case
        if isinstance(newSize, (np.ndarray, list, tuple)):
            newSize_0 = int(newSize[0])
            newSize_1 = int(newSize[1])
        else:
            newSize_0 = int(newSize[0].item())
            newSize_1 = int(newSize[1].item())
        
        paddedArray = torch.zeros((newSize_0, newSize_1), dtype=array.dtype, device=device)
        
        if array.dim() == 1:
            currSize = torch.tensor([array.numel(), 0], device=device)
        else:
            currSize = torch.tensor(array.shape, device=device)
        
        padSizes = torch.zeros(2, dtype=torch.int64, device=device)
        idxModifier = torch.zeros(2, dtype=torch.int64, device=device)
        
        for k in range(2):
            newSz = newSize_0 if k == 0 else newSize_1
            sizeDiff = newSz - int(currSize[k].item())
            
            if sizeDiff < 0:
                raise ValueError(f"newSize is smaller than actual array size in dimension {k}.")
            
            if sizeDiff % 2 == 0:
                padSizes[k] = sizeDiff // 2
            else:
                padSizes[k] = (sizeDiff + 1) // 2
                if int(currSize[k].item()) % 2 == 0:
                    idxModifier[k] = 1
                else:
                    idxModifier[k] = 0
        
        ps0 = int(padSizes[0].item())
        ps1 = int(padSizes[1].item())
        im0 = int(idxModifier[0].item())
        im1 = int(idxModifier[1].item())
        cs0 = int(currSize[0].item())
        cs1 = int(currSize[1].item())
        
        if array.dim() == 1:
            # 1D array in 2D output - place as row in middle
            paddedArray[ps0, ps1 : ps1 + cs0] = array
        else:
            paddedArray[ps0 - im0 : ps0 + cs0 - im0,
                        ps1 : ps1 + cs1 - im1] = array
    
    return paddedArray


def SLupsample(array: torch.Tensor, dims: int, nZeros: int) -> torch.Tensor:
    """
    Performs upsampling by inserting zeros along specified dimension(s).
    
    Note: dims uses MATLAB-style indexing (1 or 2, not 0 or 1).
    
    Args:
        array: Input tensor (1D or 2D)
        dims: Dimension for upsampling (1=rows, 2=cols in MATLAB style)
        nZeros: Number of zeros to insert between elements
    
    Returns:
        Upsampled tensor
    """
    if array.dim() == 1:
        # For 1D: NumPy just inserts one 0 between each element (ignores nZeros)
        sz = array.numel()
        new_size = 2 * sz - 1
        arrayUpsampled = torch.zeros(new_size, dtype=array.dtype, device=array.device)
        for i in range(sz):
            arrayUpsampled[2 * i] = array[i]
    else:
        sz = torch.tensor(array.shape, device=array.device)
        sz0 = int(sz[0].item())
        sz1 = int(sz[1].item())
        
        if dims == 0:
            raise ValueError("SLupsample behaves like MATLAB, use dims=1 or dims=2.")
        
        if dims == 1:
            # Upsample along rows
            new_rows = (sz0 - 1) * (nZeros + 1) + 1
            arrayUpsampled = torch.zeros((new_rows, sz1), dtype=array.dtype, device=array.device)
            for col in range(sz0):
                arrayUpsampled[col * (nZeros + 1), :] = array[col, :]
        elif dims == 2:
            # Upsample along columns
            new_cols = (sz1 - 1) * (nZeros + 1) + 1
            arrayUpsampled = torch.zeros((sz0, new_cols), dtype=array.dtype, device=array.device)
            for row in range(sz1):
                arrayUpsampled[:, row * (nZeros + 1)] = array[:, row]
        else:
            raise ValueError(f"Invalid dims={dims}, must be 1 or 2.")
    
    return arrayUpsampled


def SLdshear(inputArray: torch.Tensor, k: int, axis: int) -> torch.Tensor:
    """
    Computes the discretized shearing operator.
    
    Args:
        inputArray: 2D input tensor
        k: Shear number
        axis: Axis for shearing (1 or 2 in MATLAB style)
    
    Returns:
        Sheared tensor
    """
    # Convert from MATLAB-style to 0-indexed
    axis = axis - 1
    
    if k == 0:
        return inputArray
    
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


def SLsymmetricPad2D(array: torch.Tensor, make_square: bool = True, 
                     pad_mode: str = 'reflect') -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Pad a 2D tensor using symmetric (mirror) padding.
    
    This function solves filter wrapping issues in ShearLab when processing
    images with extreme aspect ratios.
    
    Args:
        array: 2D input tensor
        make_square: If True, pad to square shape (default: True)
        pad_mode: Padding mode ('reflect' or 'replicate')
    
    Returns:
        Tuple of (padded_tensor, pad_info dict)
    """
    if array.dim() != 2:
        raise ValueError("SLsymmetricPad2D only supports 2D tensors.")
    
    rows, cols = array.shape
    
    if make_square:
        target_size = max(rows, cols)
        target_rows = target_size
        target_cols = target_size
    else:
        target_rows = rows
        target_cols = cols
    
    # Calculate padding amounts
    pad_top = (target_rows - rows) // 2
    pad_bottom = target_rows - rows - pad_top
    pad_left = (target_cols - cols) // 2
    pad_right = target_cols - cols - pad_left
    
    # Store padding info for later cropping
    pad_info = {
        'original_shape': (rows, cols),
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'padded_shape': (target_rows, target_cols)
    }
    
    # F.pad expects 4D input for 2D padding
    array_4d = array.unsqueeze(0).unsqueeze(0)
    
    # PyTorch F.pad reflect mode requires padding < input dimension
    # So we need to apply padding iteratively if it exceeds the limit
    remaining_pad = [pad_left, pad_right, pad_top, pad_bottom]
    current = array_4d
    
    while any(r > 0 for r in remaining_pad):
        h, w = current.shape[2], current.shape[3]
        # Calculate how much we can pad this iteration
        # Max padding is (dimension - 1) for reflect mode
        max_pad_h = h - 1
        max_pad_w = w - 1
        
        this_pad = [
            min(remaining_pad[0], max_pad_w),  # left
            min(remaining_pad[1], max_pad_w),  # right
            min(remaining_pad[2], max_pad_h),  # top
            min(remaining_pad[3], max_pad_h),  # bottom
        ]
        
        if all(p == 0 for p in this_pad):
            break
            
        current = F.pad(current, tuple(this_pad), mode=pad_mode)
        
        # Update remaining padding
        remaining_pad = [
            remaining_pad[0] - this_pad[0],
            remaining_pad[1] - this_pad[1],
            remaining_pad[2] - this_pad[2],
            remaining_pad[3] - this_pad[3],
        ]
    
    padded_array = current.squeeze(0).squeeze(0)
    
    return padded_array, pad_info


def SLcrop2D(array: torch.Tensor, pad_info: Dict[str, Any]) -> torch.Tensor:
    """
    Crop a 2D or 3D tensor back to its original size.
    
    This function reverses the padding applied by SLsymmetricPad2D.
    
    Args:
        array: 2D or 3D tensor to crop
        pad_info: Dictionary returned by SLsymmetricPad2D
    
    Returns:
        Cropped tensor
    """
    original_rows, original_cols = pad_info['original_shape']
    pad_top = pad_info['pad_top']
    pad_left = pad_info['pad_left']
    
    if array.dim() == 2:
        return array[pad_top : pad_top + original_rows,
                     pad_left : pad_left + original_cols]
    elif array.dim() == 3:
        return array[pad_top : pad_top + original_rows,
                     pad_left : pad_left + original_cols, :]
    else:
        raise ValueError(f"SLcrop2D supports 2D and 3D tensors, got {array.dim()}D.")


# ============================================================================
# Phase 2b: Index Computation Functions
# ============================================================================

def SLgetShearletIdxs2D(shearLevels: Union[np.ndarray, torch.Tensor, List], 
                         full: int = 0, *args) -> np.ndarray:
    """
    Computes an index set describing a 2D shearlet system.
    
    This function returns a numpy array as it's primarily used for indexing.
    Matches the original NumPy version exactly.
    
    Args:
        shearLevels: Array specifying shear levels on each scale
        full: 0 for reduced system, 1 for full system
        *args: Optional restriction parameters (pairs of name, value)
    
    Returns:
        shearletIdxs: Nx3 numpy array with columns [cone, scale, shearing]
    """
    # Convert to numpy if needed
    if isinstance(shearLevels, torch.Tensor):
        shearLevels = shearLevels.cpu().numpy()
    elif isinstance(shearLevels, list):
        shearLevels = np.array(shearLevels)
    
    # If scalar, treat as array
    if not hasattr(shearLevels, "__len__"):
        shearLevels = np.array([shearLevels])
    
    shearletIdxs = []
    includeLowpass = 1
    
    scales = np.asarray(range(1, len(shearLevels) + 1))
    shearings = np.asarray(range(-int(np.power(2, np.max(shearLevels))),
                                  int(np.power(2, np.max(shearLevels))) + 1))
    cones = np.array([1, 2])
    
    # Parse restriction arguments
    for j in range(0, len(args), 2):
        includeLowpass = 0
        if args[j] == "scales":
            scales = args[j + 1]
        elif args[j] == "shearings":
            shearings = args[j + 1]
        elif args[j] == "cones":
            cones = args[j + 1]
    
    # Build shearlet indices
    for cone in np.intersect1d(np.array([1, 2]), cones):
        for scale in np.intersect1d(np.asarray(range(1, len(shearLevels) + 1)), scales):
            shearLevel = shearLevels[scale - 1]
            shear_range = np.asarray(range(-int(np.power(2, shearLevel)),
                                           int(np.power(2, shearLevel)) + 1))
            for shearing in np.intersect1d(shear_range, shearings):
                if (full == 1) or (cone == 1) or (np.abs(shearing) < np.power(2, shearLevel)):
                    shearletIdxs.append(np.array([cone, scale, shearing]))
    
    # Add lowpass at the end (matching NumPy version)
    if includeLowpass or 0 in scales or 0 in cones:
        shearletIdxs.append(np.array([0, 0, 0]))
    
    return np.asarray(shearletIdxs)


# ============================================================================
# Helper Functions
# ============================================================================

def _convolve2d_full(input: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with 'full' mode (like scipy.signal.convolve2d mode='full').
    """
    kernel_flipped = torch.flip(kernel, [0, 1])
    pad_h = kernel.shape[0] - 1
    pad_w = kernel.shape[1] - 1
    input_4d = input.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel_flipped.unsqueeze(0).unsqueeze(0)
    output = F.conv2d(input_4d, kernel_4d, padding=(pad_h, pad_w))
    return output.squeeze(0).squeeze(0)


def _convolve1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    1D convolution with 'full' mode.
    """
    # Flip for convolution (not correlation)
    b_flipped = torch.flip(b, [0])
    
    # Pad for 'full' output
    pad_size = b.numel() - 1
    a_padded = F.pad(a.unsqueeze(0).unsqueeze(0), (pad_size, pad_size))
    b_kernel = b_flipped.unsqueeze(0).unsqueeze(0)
    
    result = F.conv1d(a_padded, b_kernel)
    return result.squeeze(0).squeeze(0)


# ============================================================================
# Phase 2c: Core Frequency Domain Functions
# ============================================================================

def SLgetWedgeBandpassAndLowpassFilters2D(
    rows: int, cols: int, 
    shearLevels: Union[np.ndarray, torch.Tensor],
    directionalFilter: Optional[torch.Tensor] = None,
    scalingFilter: Optional[torch.Tensor] = None,
    waveletFilter: Optional[torch.Tensor] = None,
    scalingFilter2: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[List[Optional[torch.Tensor]], torch.Tensor, torch.Tensor]:
    """
    Computes wedge, bandpass and lowpass filters for 2D shearlets.
    
    Args:
        rows, cols: Image dimensions
        shearLevels: Array specifying shear levels on each scale
        directionalFilter: Optional directional filter
        scalingFilter: Optional scaling filter
        waveletFilter: Optional wavelet filter
        scalingFilter2: Optional secondary scaling filter
        device: Computation device
    
    Returns:
        Tuple of (wedge filters, bandpass filters, lowpass filter)
    """
    dtype = torch.float64
    ctype = torch.complex128
    
    # Convert shearLevels
    if isinstance(shearLevels, np.ndarray):
        shearLevels_np = shearLevels
    elif isinstance(shearLevels, torch.Tensor):
        shearLevels_np = shearLevels.cpu().numpy()
    else:
        shearLevels_np = np.array(shearLevels)
    
    # Default filters
    if scalingFilter is None:
        scalingFilter = torch.tensor([0.0104933261758410, -0.0263483047033631,
                    -0.0517766952966370, 0.276348304703363,
                    0.582566738241592, 0.276348304703363,
                    -0.0517766952966369, -0.0263483047033631,
                    0.0104933261758408], dtype=dtype, device=device)
    
    if scalingFilter2 is None:
        scalingFilter2 = scalingFilter.clone()
    
    if waveletFilter is None:
        waveletFilter = filters_torch.MirrorFilt(scalingFilter)
    
    if directionalFilter is None:
        h0, _ = filters_torch.dfilters('dmaxflat4', 'd', dtype=dtype, device=device)
        h0 = h0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        directionalFilter = filters_torch.modulate2(h0, 'c')
    
    # Initialize
    NScales = len(shearLevels_np)
    max_shearLevel = int(np.max(shearLevels_np))
    
    bandpass = torch.zeros((rows, cols, NScales), dtype=ctype, device=device)
    wedge: List[Optional[torch.Tensor]] = [None] * (max_shearLevel + 1)
    
    # Normalize directional filter
    directionalFilter = directionalFilter / torch.sum(torch.abs(directionalFilter))
    
    # Compute 1D high and low pass filters at different scales
    filterHigh: List[Optional[torch.Tensor]] = [None] * NScales
    filterLow: List[Optional[torch.Tensor]] = [None] * NScales
    filterLow2: List[Optional[torch.Tensor]] = [None] * (max_shearLevel + 1)
    
    filterHigh[-1] = waveletFilter
    filterLow[-1] = scalingFilter
    filterLow2[-1] = scalingFilter2
    
    # Compute wavelet filters at all scales
    for j in range(len(filterHigh) - 2, -1, -1):
        prev_low = filterLow[j + 1]
        prev_high = filterHigh[j + 1]
        last_low = filterLow[-1]
        assert prev_low is not None and prev_high is not None and last_low is not None
        filterLow[j] = _convolve1d(last_low, SLupsample(prev_low, 2, 1))
        filterHigh[j] = _convolve1d(last_low, SLupsample(prev_high, 2, 1))
    
    for j in range(len(filterLow2) - 2, -1, -1):
        prev_low2 = filterLow2[j + 1]
        last_low2 = filterLow2[-1]
        assert prev_low2 is not None and last_low2 is not None
        filterLow2[j] = _convolve1d(last_low2, SLupsample(prev_low2, 2, 1))
    
    # Construct bandpass filters
    for j in range(len(filterHigh)):
        fh_opt = filterHigh[j]
        assert fh_opt is not None
        fh = fh_opt.flatten()
        padded = SLpadArray(fh, torch.tensor([rows, cols], device=device))
        bandpass[:, :, j] = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(padded)))
    
    # Construct wedge filters
    f_low2_last_opt = filterLow2[-1]
    assert f_low2_last_opt is not None
    filterLow2_last = f_low2_last_opt.flatten()
    
    for shearLevel in np.unique(shearLevels_np):
        shearLevel = int(shearLevel)
        nWedges = int(np.floor(np.power(2, shearLevel + 1) + 1))
        wedge[shearLevel] = torch.zeros((rows, cols, nWedges), dtype=ctype, device=device)
        
        # Upsample directional filter
        nZeros = int(np.power(2, shearLevel + 1) - 1)
        directionalFilterUpsampled = SLupsample(directionalFilter, 1, nZeros)
        
        # Convolve with lowpass filter
        idx = len(filterLow2) - shearLevel - 1
        f_low2_idx_opt = filterLow2[idx]
        assert f_low2_idx_opt is not None
        filterLow2_reshaped = f_low2_idx_opt.reshape(-1, 1)
        wedgeHelp = _convolve2d_full(directionalFilterUpsampled, filterLow2_reshaped)
        wedgeHelp = SLpadArray(wedgeHelp, torch.tensor([rows, cols], device=device))
        
        # Upsample wedge filter
        nZeros2 = int(np.power(2, shearLevel) - 1)
        wedgeUpsampled = SLupsample(wedgeHelp, 2, nZeros2)
        
        # Convolve with lowpass if shearLevel >= 1
        idx2 = len(filterLow2) - max(shearLevel - 1, 0) - 1
        f_low2_idx2_opt = filterLow2[idx2]
        assert f_low2_idx2_opt is not None
        lowpassHelp = SLpadArray(f_low2_idx2_opt.flatten(), 
                                  torch.tensor(wedgeUpsampled.shape, device=device))
        
        if shearLevel >= 1:
            # Frequency domain multiplication
            lowpass_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(lowpassHelp)))
            wedge_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(wedgeUpsampled)))
            wedgeUpsampled = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(lowpass_fft * wedge_fft)))
        
        lowpassHelpFlip = torch.flip(lowpassHelp, [1])
        
        # Traverse all directions
        nShears = int(np.power(2, shearLevel))
        for k in range(-nShears, nShears + 1):
            # Apply shearing
            wedgeUpsampledSheared = SLdshear(wedgeUpsampled, k, 2)
            
            if shearLevel >= 1:
                lowflip_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(lowpassHelpFlip)))
                sheared_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(wedgeUpsampledSheared)))
                wedgeUpsampledSheared = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(lowflip_fft * sheared_fft)))
            
            # Downsample and transform to frequency domain
            step = int(np.power(2, shearLevel))
            downsampled = (np.power(2, shearLevel) * 
                          wedgeUpsampledSheared[:, 0 : step * cols : step])
            
            wedge_idx = int(np.fix(np.power(2, shearLevel)) - k)
            wedge_sl = wedge[shearLevel]
            assert wedge_sl is not None
            wedge_sl[:, :, wedge_idx] = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(downsampled)))
    
    # Compute lowpass filter
    fl0_opt = filterLow[0]
    assert fl0_opt is not None
    fl0 = fl0_opt.flatten()
    lowpass_2d = torch.outer(fl0, fl0)
    lowpass_padded = SLpadArray(lowpass_2d, torch.tensor([rows, cols], device=device))
    lowpass = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(lowpass_padded)))
    
    return wedge, bandpass, lowpass


def SLprepareFilters2D(
    rows: int, cols: int, nScales: int,
    shearLevels: Optional[Union[np.ndarray, torch.Tensor, List]] = None,
    directionalFilter: Optional[torch.Tensor] = None,
    scalingFilter: Optional[torch.Tensor] = None,
    waveletFilter: Optional[torch.Tensor] = None,
    scalingFilter2: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = 'cpu'
) -> Dict[str, Any]:
    """
    Prepares filters for a 2D shearlet system.
    
    Args:
        rows, cols: Image dimensions
        nScales: Number of scales
        shearLevels: Optional array of shear levels per scale
        directionalFilter, scalingFilter, waveletFilter, scalingFilter2: Optional filters
        device: Computation device
    
    Returns:
        Dictionary containing prepared filters
    """
    dtype = torch.float64
    
    # Default shear levels
    if shearLevels is None:
        shearLevels = np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
    elif isinstance(shearLevels, torch.Tensor):
        shearLevels = shearLevels.cpu().numpy()
    elif isinstance(shearLevels, list):
        shearLevels = np.array(shearLevels)
    
    # Default filters
    if directionalFilter is None:
        h0, _ = filters_torch.dfilters('dmaxflat4', 'd', dtype=dtype, device=device)
        h0 = h0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        directionalFilter = filters_torch.modulate2(h0, 'c')
    
    if scalingFilter is None:
        scalingFilter = torch.tensor([0.0104933261758410, -0.0263483047033631,
                    -0.0517766952966370, 0.276348304703363,
                    0.582566738241592, 0.276348304703363,
                    -0.0517766952966369, -0.0263483047033631,
                    0.0104933261758408], dtype=dtype, device=device)
    
    if waveletFilter is None:
        waveletFilter = filters_torch.MirrorFilt(scalingFilter)
    
    if scalingFilter2 is None:
        scalingFilter2 = scalingFilter.clone()
    
    # Get wedge, bandpass and lowpass filters for cone1 (horizontal)
    wedge1, bandpass1, lowpass1 = SLgetWedgeBandpassAndLowpassFilters2D(
        rows, cols, shearLevels,
        directionalFilter, scalingFilter, waveletFilter, scalingFilter2,
        device=device
    )
    
    # Build cone1 structure
    cone1 = {'wedge': wedge1, 'bandpass': bandpass1, 'lowpass': lowpass1}
    
    # For cone2: if square, share filters; otherwise compute with swapped dimensions
    if rows == cols:
        cone2 = cone1
    else:
        wedge2, bandpass2, lowpass2 = SLgetWedgeBandpassAndLowpassFilters2D(
            cols, rows, shearLevels,  # Note: swapped dimensions
            directionalFilter, scalingFilter, waveletFilter, scalingFilter2,
            device=device
        )
        cone2 = {'wedge': wedge2, 'bandpass': bandpass2, 'lowpass': lowpass2}
    
    # Build result dictionary (matching NumPy structure)
    preparedFilters = {
        'size': (rows, cols),
        'rows': rows,
        'cols': cols,
        'shearLevels': shearLevels,
        'cone1': cone1,
        'cone2': cone2,
        'directionalFilter': directionalFilter,
        'scalingFilter': scalingFilter,
        'waveletFilter': waveletFilter,
        'scalingFilter2': scalingFilter2,
        'device': device
    }
    
    return preparedFilters


def SLgetShearlets2D(
    preparedFilters: Dict[str, Any],
    shearletIdxs: Optional[np.ndarray] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D shearlets in the frequency domain.
    
    Args:
        preparedFilters: Dictionary from SLprepareFilters2D
        shearletIdxs: Optional custom shearlet indices
    
    Returns:
        Tuple of (shearlets, RMS values, dualFrameWeights)
    """
    rows = preparedFilters['rows']
    cols = preparedFilters['cols']
    shearLevels = preparedFilters['shearLevels']
    cone1 = preparedFilters['cone1']
    cone2 = preparedFilters['cone2']
    device = preparedFilters.get('device', 'cpu')
    
    # Get shearlet indices
    if shearletIdxs is None:
        shearletIdxs = SLgetShearletIdxs2D(shearLevels, full=0)
    
    nShearlets = len(shearletIdxs)
    ctype = torch.complex128
    
    # Allocate output
    shearlets = torch.zeros((rows, cols, nShearlets), dtype=ctype, device=device)
    
    # Compute each shearlet
    for j in range(nShearlets):
        cone = shearletIdxs[j, 0]
        scale = shearletIdxs[j, 1]
        shearing = shearletIdxs[j, 2]
        
        if cone == 0:
            # Lowpass shearlet (use cone1's lowpass)
            shearlets[:, :, j] = cone1['lowpass']
        elif cone == 1:
            # Horizontal cone - use -shearing (matching NumPy)
            shearLevel = int(shearLevels[scale - 1])
            wedgeIdx = int(-shearing + np.power(2, shearLevel))
            wedge = cone1['wedge'][shearLevel]
            bandpass = cone1['bandpass']
            shearlets[:, :, j] = wedge[:, :, wedgeIdx] * torch.conj(bandpass[:, :, scale - 1])
        else:
            # Vertical cone - use +shearing and transpose (matching NumPy)
            shearLevel = int(shearLevels[scale - 1])
            wedgeIdx = int(shearing + np.power(2, shearLevel))
            wedge = cone2['wedge'][shearLevel]
            bandpass = cone2['bandpass']
            # cone2 filters are computed with swapped dimensions, then transposed
            shearlets[:, :, j] = (wedge[:, :, wedgeIdx] * torch.conj(bandpass[:, :, scale - 1])).T
    
    # Compute RMS and dual frame weights
    shearlets_abs = torch.abs(shearlets)
    RMS = torch.sqrt(torch.sum(shearlets_abs ** 2, dim=(0, 1)) / (rows * cols))
    
    dualFrameWeights = torch.sum(shearlets_abs ** 2, dim=2)
    
    return shearlets, RMS, dualFrameWeights
