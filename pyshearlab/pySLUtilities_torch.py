"""
PyTorch implementation of pyShearLab2D utilities.
Translated from pySLUtilities.py (NumPy version).

This module provides utility functions for the ShearLab2D toolbox using
PyTorch tensors instead of NumPy arrays, enabling GPU acceleration.

Stefan Loock (original NumPy), PyTorch translation 2024
"""

from __future__ import division
import sys
import math
from typing import Tuple, Optional, Union, Dict, List, Any
import torch
import torch.nn.functional as F

from pyshearlab import pySLFilters_torch as filters_torch


# ============================================================================
# Phase 2a: Simple Utility Functions
# ============================================================================

def SLpadArray(array: torch.Tensor, newSize: Union[int, torch.Tensor, List, Tuple],
               device: Optional[Union[str, torch.device]] = None) -> torch.Tensor:
    """
    Implements the padding of an array as performed by the Matlab variant.
    
    Centers the input array in a larger array of zeros.
    Optimized to avoid GPU synchronization by using pure Python arithmetic.
    
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
            idxModifier = 1 if currSize % 2 == 0 else 0
        
        paddedArray[padSizes - idxModifier : padSizes + currSize - idxModifier] = array.flatten()
    else:
        # 2D case - extract sizes as pure Python ints to avoid .item() calls
        if isinstance(newSize, torch.Tensor):
            newSize_0 = int(newSize[0].item())
            newSize_1 = int(newSize[1].item())
        else:  # list or tuple
            newSize_0 = int(newSize[0])
            newSize_1 = int(newSize[1])
        
        # Get current size as pure Python ints
        # For 1D array: match NumPy behavior where currSize = [len(array), 0]
        if array.dim() == 1:
            array_len = array.numel()
            # NumPy sets currSize = [len(array), 0] for 1D arrays
            # This means for row padding: sizeDiff0 = newSize_0 - array_len
            # And for col padding: sizeDiff1 = newSize_1 - 0 = newSize_1 (data goes at padSizes[1])
            cs0 = array_len  # For row position calculation
            cs1 = 0  # NumPy uses 0 for column size of 1D array
        else:
            array_len = 0  # Not used for 2D
            cs0 = array.shape[0]
            cs1 = array.shape[1]
        
        paddedArray = torch.zeros((newSize_0, newSize_1), dtype=array.dtype, device=device)
        
        # Compute padding for dimension 0 (rows)
        sizeDiff0 = newSize_0 - cs0
        if sizeDiff0 < 0:
            raise ValueError("newSize is smaller than actual array size in dimension 0.")
        if sizeDiff0 % 2 == 0:
            ps0 = sizeDiff0 // 2
            im0 = 0
        else:
            ps0 = (sizeDiff0 + 1) // 2
            im0 = 1 if cs0 % 2 == 0 else 0
        
        # Compute padding for dimension 1 (cols)
        # For 1D array: NumPy uses cs1=0, so sizeDiff1 = newSize_1
        sizeDiff1 = newSize_1 - cs1
        if sizeDiff1 < 0:
            raise ValueError("newSize is smaller than actual array size in dimension 1.")
        if sizeDiff1 % 2 == 0:
            ps1 = sizeDiff1 // 2
            im1 = 0
        else:
            ps1 = (sizeDiff1 + 1) // 2
            im1 = 1 if cs1 % 2 == 0 else 0
        
        if array.dim() == 1:
            # 1D array in 2D output - place as row
            # Match NumPy: paddedArray[ps0, ps1:ps1+len(array)] = array
            # Note: ps1 is based on cs1=0, so ps1 = newSize_1 // 2 (data starts from middle-right)
            paddedArray[ps0, ps1 : ps1 + array_len] = array
        else:
            paddedArray[ps0 - im0 : ps0 + cs0 - im0,
                        ps1 : ps1 + cs1 - im1] = array
    
    return paddedArray


def SLupsample(array: torch.Tensor, dims: int, nZeros: int) -> torch.Tensor:
    """
    Performs upsampling by inserting zeros along specified dimension(s).
    
    This is an optimized version using slice assignment instead of loops.
    
    Note: dims uses MATLAB-style indexing (1 or 2, not 0 or 1).
    
    Args:
        array: Input tensor (1D or 2D)
        dims: Dimension for upsampling (1=rows, 2=cols in MATLAB style)
        nZeros: Number of zeros to insert between elements
    
    Returns:
        Upsampled tensor
    """
    if array.dim() == 1:
        # For 1D: insert one 0 between each element (ignores nZeros per original behavior)
        sz = array.numel()
        new_size = 2 * sz - 1
        arrayUpsampled = torch.zeros(new_size, dtype=array.dtype, device=array.device)
        # Use slice with step=2 to assign all elements at once
        arrayUpsampled[::2] = array
    else:
        sz0, sz1 = array.shape
        
        if dims == 0:
            raise ValueError("SLupsample behaves like MATLAB, use dims=1 or dims=2.")
        
        step = nZeros + 1
        
        if dims == 1:
            # Upsample along rows: insert nZeros zeros between each row
            new_rows = (sz0 - 1) * step + 1
            arrayUpsampled = torch.zeros((new_rows, sz1), dtype=array.dtype, device=array.device)
            # Use slice with step to assign all rows at once
            arrayUpsampled[::step, :] = array
        elif dims == 2:
            # Upsample along columns: insert nZeros zeros between each column
            new_cols = (sz1 - 1) * step + 1
            arrayUpsampled = torch.zeros((sz0, new_cols), dtype=array.dtype, device=array.device)
            # Use slice with step to assign all columns at once
            arrayUpsampled[:, ::step] = array
        else:
            raise ValueError(f"Invalid dims={dims}, must be 1 or 2.")
    
    return arrayUpsampled


def SLdshear(inputArray: torch.Tensor, k: int, axis: int) -> torch.Tensor:
    """
    Computes the discretized shearing operator (vectorized version).
    
    This is an optimized implementation that uses index-based gathering
    instead of Python loops, enabling full GPU parallelization.
    
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
        return inputArray.clone()
    
    rows = inputArray.shape[0]
    cols = inputArray.shape[1]
    device = inputArray.device
    
    if axis == 0:
        # Shearing along axis 0: each column gets shifted by k * (cols//2 - col)
        # shifts[col] = k * (cols // 2 - col)
        col_indices = torch.arange(cols, device=device)
        shifts = k * (cols // 2 - col_indices)  # shape: (cols,)
        
        # For each column c, row r: new_row[r] = (r - shifts[c]) % rows
        # Create row indices: shape (rows, 1)
        row_indices = torch.arange(rows, device=device).unsqueeze(1)  # (rows, 1)
        
        # Compute source row for each position: (r - shift) mod rows
        # shifts has shape (cols,), broadcast to (rows, cols)
        source_rows = (row_indices - shifts.unsqueeze(0)) % rows  # (rows, cols)
        source_rows = source_rows.long()
        
        # Gather along dimension 0
        shearedArray = torch.gather(inputArray, 0, source_rows)
    else:
        # Shearing along axis 1: each row gets shifted by k * (rows//2 - row)
        # shifts[row] = k * (rows // 2 - row)
        row_indices = torch.arange(rows, device=device)
        shifts = k * (rows // 2 - row_indices)  # shape: (rows,)
        
        # For each row r, col c: new_col[c] = (c - shifts[r]) % cols
        # Create col indices: shape (1, cols)
        col_indices = torch.arange(cols, device=device).unsqueeze(0)  # (1, cols)
        
        # Compute source column for each position: (c - shift) mod cols
        # shifts has shape (rows,), expand to (rows, 1) then broadcast to (rows, cols)
        source_cols = (col_indices - shifts.unsqueeze(1)) % cols  # (rows, cols)
        source_cols = source_cols.long()
        
        # Gather along dimension 1
        shearedArray = torch.gather(inputArray, 1, source_cols)
    
    return shearedArray


def SLdshear_batch(inputArray: torch.Tensor, k_values: torch.Tensor, axis: int) -> torch.Tensor:
    """
    Computes the discretized shearing operator for multiple k values (batch version).
    
    This is an optimized implementation that processes all k values simultaneously,
    leveraging GPU parallelization for significant speedup.
    
    Args:
        inputArray: 2D input tensor (rows, cols)
        k_values: 1D tensor of shear numbers, shape (batch,)
        axis: Axis for shearing (1 or 2 in MATLAB style)
    
    Returns:
        Sheared tensors, shape (batch, rows, cols)
    """
    # Convert from MATLAB-style to 0-indexed
    axis = axis - 1
    
    rows = inputArray.shape[0]
    cols = inputArray.shape[1]
    device = inputArray.device
    batch_size = k_values.shape[0]
    
    # Handle k=0 case: just replicate input
    if torch.all(k_values == 0):
        return inputArray.unsqueeze(0).expand(batch_size, -1, -1).clone()
    
    if axis == 0:
        # Shearing along axis 0: each column gets shifted by k * (cols//2 - col)
        col_indices = torch.arange(cols, device=device)  # (cols,)
        # shifts: (batch, cols)
        shifts = k_values.unsqueeze(1) * (cols // 2 - col_indices.unsqueeze(0))
        
        # row_indices: (1, rows, 1) for broadcasting
        row_indices = torch.arange(rows, device=device).view(1, rows, 1)
        
        # source_rows: (batch, rows, cols)
        source_rows = (row_indices - shifts.unsqueeze(1)) % rows
        source_rows = source_rows.long()
        
        # Expand input to (batch, rows, cols)
        expanded_input = inputArray.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Gather along dimension 1 (rows)
        shearedArray = torch.gather(expanded_input, 1, source_rows)
    else:
        # Shearing along axis 1: each row gets shifted by k * (rows//2 - row)
        row_indices = torch.arange(rows, device=device)  # (rows,)
        # shifts: (batch, rows)
        shifts = k_values.unsqueeze(1) * (rows // 2 - row_indices.unsqueeze(0))
        
        # col_indices: (1, 1, cols) for broadcasting
        col_indices = torch.arange(cols, device=device).view(1, 1, cols)
        
        # source_cols: (batch, rows, cols)
        source_cols = (col_indices - shifts.unsqueeze(2)) % cols
        source_cols = source_cols.long()
        
        # Expand input to (batch, rows, cols)
        expanded_input = inputArray.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Gather along dimension 2 (cols)
        shearedArray = torch.gather(expanded_input, 2, source_cols)
    
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


def SLcheckFilterSizes(
    rows: int, cols: int,
    shearLevels: Union[torch.Tensor, List],
    directionalFilter: torch.Tensor,
    scalingFilter: torch.Tensor,
    waveletFilter: torch.Tensor,
    scalingFilter2: torch.Tensor,
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Check and adjust filter sizes for a given image size.
    
    If the specified filters are too large for the image dimensions,
    this function automatically selects a smaller filter configuration.
    
    Args:
        rows, cols: Image dimensions
        shearLevels: Array of shear levels
        directionalFilter, scalingFilter, waveletFilter, scalingFilter2: Initial filters
        device: Computation device
    
    Returns:
        Tuple of (directionalFilter, scalingFilter, waveletFilter, scalingFilter2)
    """
    dtype = torch.float64
    
    # Build filter configurations
    filterSetup = []
    
    # Configuration 1: Original filters
    filterSetup.append({
        "directionalFilter": directionalFilter,
        "scalingFilter": scalingFilter,
        "waveletFilter": waveletFilter,
        "scalingFilter2": scalingFilter2
    })
    
    # Configuration 2: dmaxflat4 + default scaling
    h0, _ = filters_torch.dfilters('dmaxflat4', 'd', dtype=dtype, device=device)
    h0 = h0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    df2 = filters_torch.modulate2(h0, 'c')
    sf2 = torch.tensor([0.0104933261758410, -0.0263483047033631,
                -0.0517766952966370, 0.276348304703363, 0.582566738241592,
                0.276348304703363, -0.0517766952966369, -0.0263483047033631,
                0.0104933261758408], dtype=dtype, device=device)
    wf2 = filters_torch.MirrorFilt(sf2)
    filterSetup.append({"directionalFilter": df2, "scalingFilter": sf2, 
                        "waveletFilter": wf2, "scalingFilter2": sf2})
    
    # Configuration 3-4: cd filter + default scaling
    h0, _ = filters_torch.dfilters('cd', 'd', dtype=dtype, device=device)
    h0 = h0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    df3 = filters_torch.modulate2(h0, 'c')
    filterSetup.append({"directionalFilter": df3, "scalingFilter": sf2.clone(), 
                        "waveletFilter": wf2.clone(), "scalingFilter2": sf2.clone()})
    filterSetup.append({"directionalFilter": df3.clone(), "scalingFilter": sf2.clone(), 
                        "waveletFilter": wf2.clone(), "scalingFilter2": sf2.clone()})
    
    # Configuration 5: cd + Coiflet 1
    sf5 = filters_torch.MakeONFilter('Coiflet', 1, dtype=dtype, device=device)
    wf5 = filters_torch.MirrorFilt(sf5)
    filterSetup.append({"directionalFilter": df3.clone(), "scalingFilter": sf5, 
                        "waveletFilter": wf5, "scalingFilter2": sf5.clone()})
    
    # Configuration 6: cd + Daubechies 4
    sf6 = filters_torch.MakeONFilter('Daubechies', 4, dtype=dtype, device=device)
    wf6 = filters_torch.MirrorFilt(sf6)
    filterSetup.append({"directionalFilter": df3.clone(), "scalingFilter": sf6, 
                        "waveletFilter": wf6, "scalingFilter2": sf6.clone()})
    
    # Configuration 7: oqf_362 + Daubechies 4
    h0, _ = filters_torch.dfilters('oqf_362', 'd', dtype=dtype, device=device)
    h0 = h0 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
    df7 = filters_torch.modulate2(h0, 'c')
    filterSetup.append({"directionalFilter": df7, "scalingFilter": sf6.clone(), 
                        "waveletFilter": wf6.clone(), "scalingFilter2": sf6.clone()})
    
    # Configuration 8: oqf_362 + Haar
    sf8 = filters_torch.MakeONFilter('Haar', 1, dtype=dtype, device=device)
    wf8 = filters_torch.MirrorFilt(sf8)
    filterSetup.append({"directionalFilter": df7.clone(), "scalingFilter": sf8, 
                        "waveletFilter": wf8, "scalingFilter2": sf8.clone()})
    
    success = False
    selected_k = 0
    
    for k in range(len(filterSetup)):
        # Check 1: wavelet/scaling filter size
        lwfilter = filterSetup[k]["waveletFilter"].numel()
        lsfilter = filterSetup[k]["scalingFilter"].numel()
        lcheck1 = lwfilter
        for j in range(len(shearLevels)):
            lcheck1 = lsfilter + 2 * lcheck1 - 2
        if lcheck1 > cols or lcheck1 > rows:
            continue
        
        # Check 2: directional filter size
        df = filterSetup[k]["directionalFilter"]
        rowsdirfilter = df.shape[0]
        colsdirfilter = df.shape[1]
        lcheck2 = (rowsdirfilter - 1) * (1 << (int(max(shearLevels)) + 1)) + 1
        
        lsfilter2 = filterSetup[k]["scalingFilter2"].numel()
        lcheck2help = lsfilter2
        for j in range(1, int(max(shearLevels)) + 1):
            lcheck2help = lsfilter2 + 2 * lcheck2help - 2
        lcheck2 = lcheck2help + lcheck2 - 1
        
        if lcheck2 > cols or lcheck2 > rows or colsdirfilter > cols or colsdirfilter > rows:
            continue
        
        success = True
        selected_k = k
        break
    
    if not success:
        raise ValueError(f"The specified Shearlet system is not available for data of size "
                        f"{rows}x{cols}. Try decreasing the number of scales and shearings.")
    
    if success and selected_k > 0:
        print(f"Warning: The specified Shearlet system was not available for data of size "
              f"{rows}x{cols}. Filters were automatically set to configuration {selected_k + 1} "
              f"(see SLcheckFilterSizes).")
    
    return (filterSetup[selected_k]["directionalFilter"],
            filterSetup[selected_k]["scalingFilter"],
            filterSetup[selected_k]["waveletFilter"],
            filterSetup[selected_k]["scalingFilter2"])


# ============================================================================
# Phase 2b: Index Computation Functions
# ============================================================================

def SLgetShearletIdxs2D(shearLevels: Union[torch.Tensor, List], 
                         full: int = 0, *args) -> torch.Tensor:
    """
    Computes an index set describing a 2D shearlet system (pure PyTorch version).
    
    Args:
        shearLevels: Tensor or list specifying shear levels on each scale
        full: 0 for reduced system, 1 for full system
        *args: Optional restriction parameters (pairs of name, value)
    
    Returns:
        shearletIdxs: Nx3 tensor with columns [cone, scale, shearing]
    """
    # Convert to list for easier processing
    shearLevels_list: List[int]
    if isinstance(shearLevels, torch.Tensor):
        shearLevels_list = [int(x) for x in shearLevels.cpu().tolist()]
    elif isinstance(shearLevels, list):
        shearLevels_list = [int(x) for x in shearLevels]
    else:
        shearLevels_list = [int(shearLevels)]  # scalar case
    
    # If scalar, treat as list
    if not hasattr(shearLevels_list, "__len__"):
        shearLevels_list = [int(shearLevels_list)]  # type: ignore
    
    shearletIdxs: List[List[int]] = []
    includeLowpass = 1
    
    max_shear: int = max(shearLevels_list)
    scales_set = set(range(1, len(shearLevels_list) + 1))
    shearings_set = set(range(-(1 << max_shear), (1 << max_shear) + 1))
    cones_set = {1, 2}
    
    # Parse restriction arguments
    for j in range(0, len(args), 2):
        includeLowpass = 0
        if args[j] == "scales":
            scales_set = set(args[j + 1]) if hasattr(args[j + 1], '__iter__') else {args[j + 1]}
        elif args[j] == "shearings":
            shearings_set = set(args[j + 1]) if hasattr(args[j + 1], '__iter__') else {args[j + 1]}
        elif args[j] == "cones":
            cones_set = set(args[j + 1]) if hasattr(args[j + 1], '__iter__') else {args[j + 1]}
    
    # Intersect with valid cones
    valid_cones = sorted(cones_set & {1, 2})
    valid_scales = sorted(scales_set & set(range(1, len(shearLevels_list) + 1)))
    
    # Build shearlet indices
    for cone in valid_cones:
        for scale in valid_scales:
            shearLevel = shearLevels_list[scale - 1]
            shear_bound = 1 << shearLevel  # 2^shearLevel
            shear_range = set(range(-shear_bound, shear_bound + 1))
            valid_shearings = sorted(shear_range & shearings_set)
            
            for shearing in valid_shearings:
                if (full == 1) or (cone == 1) or (abs(shearing) < shear_bound):
                    shearletIdxs.append([cone, scale, shearing])
    
    # Add lowpass at the end (matching original version)
    if includeLowpass or 0 in scales_set or 0 in cones_set:
        shearletIdxs.append([0, 0, 0])
    
    return torch.tensor(shearletIdxs, dtype=torch.int64)


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
    shearLevels: Union[torch.Tensor, List],
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
    
    # Convert shearLevels to list
    if isinstance(shearLevels, torch.Tensor):
        shearLevels_list = shearLevels.cpu().tolist()
    elif isinstance(shearLevels, list):
        shearLevels_list = shearLevels
    else:
        shearLevels_list = list(shearLevels)
    
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
    NScales = len(shearLevels_list)
    max_shearLevel = int(max(shearLevels_list))
    
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
    
    for shearLevel in sorted(set(shearLevels_list)):
        shearLevel = int(shearLevel)
        nWedges = (1 << (shearLevel + 1)) + 1  # 2^(shearLevel+1) + 1
        wedge[shearLevel] = torch.zeros((rows, cols, nWedges), dtype=ctype, device=device)
        
        # Upsample directional filter
        nZeros = (1 << (shearLevel + 1)) - 1  # 2^(shearLevel+1) - 1
        directionalFilterUpsampled = SLupsample(directionalFilter, 1, nZeros)
        
        # Convolve with lowpass filter
        idx = len(filterLow2) - shearLevel - 1
        f_low2_idx_opt = filterLow2[idx]
        assert f_low2_idx_opt is not None
        filterLow2_reshaped = f_low2_idx_opt.reshape(-1, 1)
        wedgeHelp = _convolve2d_full(directionalFilterUpsampled, filterLow2_reshaped)
        wedgeHelp = SLpadArray(wedgeHelp, torch.tensor([rows, cols], device=device))
        
        # Upsample wedge filter
        nZeros2 = (1 << shearLevel) - 1  # 2^shearLevel - 1
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
        
        # Precompute lowflip_fft ONCE outside the loop (it's constant for all k)
        lowpassHelpFlip = torch.flip(lowpassHelp, [1])
        if shearLevel >= 1:
            lowflip_fft = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(lowpassHelpFlip)))
        
        # Precompute step value  
        step = 2 ** shearLevel
        scale_factor = float(step)
        nShears = step  # 2^shearLevel
        
        # Process all shearing directions in batch (optimized)
        wedge_sl = wedge[shearLevel]
        assert wedge_sl is not None
        
        # Create k_values tensor for batch processing
        k_values = torch.arange(-nShears, nShears + 1, device=device)
        
        # Batch shearing: (batch, rows, cols_upsampled)
        all_sheared = SLdshear_batch(wedgeUpsampled, k_values, 2)
        
        if shearLevel >= 1:
            # Batch FFT operations
            # all_sheared: (batch, rows, cols_upsampled)
            sheared_fft = torch.fft.fftshift(
                torch.fft.fft2(torch.fft.ifftshift(all_sheared, dim=(1, 2)), dim=(1, 2)),
                dim=(1, 2)
            )
            # Multiply with lowflip_fft (broadcast: (rows, cols_up) * (batch, rows, cols_up))
            all_sheared = torch.fft.fftshift(
                torch.fft.ifft2(
                    torch.fft.ifftshift(lowflip_fft.unsqueeze(0) * sheared_fft, dim=(1, 2)),
                    dim=(1, 2)
                ),
                dim=(1, 2)
            )
        
        # Batch downsample: select every 'step' column
        # all_sheared: (batch, rows, cols_upsampled) -> (batch, rows, cols)
        downsampled = scale_factor * all_sheared[:, :, 0 : step * cols : step]
        
        # Batch final FFT
        # downsampled: (batch, rows, cols)
        wedge_result = torch.fft.fftshift(
            torch.fft.fft2(torch.fft.ifftshift(downsampled, dim=(1, 2)), dim=(1, 2)),
            dim=(1, 2)
        )
        
        # Assign to wedge: need to reverse order (k from -nShears to +nShears maps to indices 2*nShears to 0)
        # wedge_idx = nShears - k, so for k in [-nShears, ..., nShears], idx in [2*nShears, ..., 0]
        # We computed in order k=[-nShears,...,nShears], so result[0] -> idx=2*nShears, result[-1] -> idx=0
        # Need to flip the batch dimension
        wedge_sl[:, :, :] = wedge_result.flip(0).permute(1, 2, 0)
    
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
    shearLevels: Optional[Union[torch.Tensor, List]] = None,
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
        # Equivalent to: np.ceil(np.arange(1, nScales + 1) / 2).astype(int)
        shearLevels = [int(math.ceil(i / 2)) for i in range(1, nScales + 1)]
    elif isinstance(shearLevels, torch.Tensor):
        shearLevels = shearLevels.cpu().tolist()
    # If already a list, keep as is
    
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
    
    # Check and adjust filter sizes if needed
    directionalFilter, scalingFilter, waveletFilter, scalingFilter2 = SLcheckFilterSizes(
        rows, cols, shearLevels,
        directionalFilter, scalingFilter, waveletFilter, scalingFilter2,
        device=device
    )
    
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
    shearletIdxs: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute 2D shearlets in the frequency domain (optimized version).
    
    This version groups shearlets by cone type to reduce branching overhead.
    
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
    
    # Group indices by cone for more efficient processing
    cone0_indices = []
    cone1_indices = []
    cone2_indices = []
    
    for j in range(nShearlets):
        cone = shearletIdxs[j, 0]
        if cone == 0:
            cone0_indices.append(j)
        elif cone == 1:
            cone1_indices.append(j)
        else:
            cone2_indices.append(j)
    
    # Process cone 0 (lowpass) - all get the same filter
    if cone0_indices:
        lowpass = cone1['lowpass']
        for j in cone0_indices:
            shearlets[:, :, j] = lowpass
    
    # Process cone 1 (horizontal) - group by scale for efficiency
    if cone1_indices:
        wedge_dict = cone1['wedge']
        bandpass = cone1['bandpass']
        
        # Precompute conjugate bandpass for each scale (avoid repeated conj calls)
        bandpass_conj = torch.conj(bandpass)
        
        for j in cone1_indices:
            scale = shearletIdxs[j, 1]
            shearing = shearletIdxs[j, 2]
            shearLevel = int(shearLevels[scale - 1])
            wedgeIdx = -shearing + (1 << shearLevel)  # 2^shearLevel using bit shift
            wedge = wedge_dict[shearLevel]
            shearlets[:, :, j] = wedge[:, :, wedgeIdx] * bandpass_conj[:, :, scale - 1]
    
    # Process cone 2 (vertical) - need transpose
    if cone2_indices:
        wedge_dict = cone2['wedge']
        bandpass = cone2['bandpass']
        
        # Precompute conjugate bandpass for each scale
        bandpass_conj = torch.conj(bandpass)
        
        for j in cone2_indices:
            scale = shearletIdxs[j, 1]
            shearing = shearletIdxs[j, 2]
            shearLevel = int(shearLevels[scale - 1])
            wedgeIdx = shearing + (1 << shearLevel)  # 2^shearLevel using bit shift
            wedge = wedge_dict[shearLevel]
            # cone2 filters are computed with swapped dimensions, then transposed
            shearlets[:, :, j] = (wedge[:, :, wedgeIdx] * bandpass_conj[:, :, scale - 1]).T
    
    # Compute RMS and dual frame weights (already vectorized)
    shearlets_abs_sq = shearlets.real ** 2 + shearlets.imag ** 2  # Faster than abs()**2
    RMS = torch.sqrt(torch.sum(shearlets_abs_sq, dim=(0, 1)) / (rows * cols))
    
    dualFrameWeights = torch.sum(shearlets_abs_sq, dim=2)
    
    return shearlets, RMS, dualFrameWeights

