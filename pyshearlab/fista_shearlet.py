"""
FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) for Shearlet L1 Regularization.

This module implements sparse reconstruction based on the Synthesis Model:
    ĉ = argmin_c  1/2 ||y - Ψ* c||² + λ||c||₁

Where:
    - y: Observed (noisy) image
    - Ψ* (Synthesis): SLshearadjoint2D - maps coefficients to image
    - Ψ  (Analysis):  SLsheardec2D    - maps image to coefficients
    - c: Sparse shearlet coefficients

Supports both single image (H, W) and batch processing (B, H, W).
"""

from __future__ import annotations

import math
import time
from typing import Tuple, List, Optional, Dict, Any, Union

import torch
import torch.nn.functional as F

from pyshearlab import pyShearLab2D_torch as sl
from pyshearlab import pySLUtilities_torch as utils


def soft_threshold(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """
    Soft thresholding (proximal operator for L1 norm).
    
    Prox_{λ||·||₁}(x) = sign(x) * max(|x| - λ, 0)
    
    Args:
        x: Input tensor (any shape)
        threshold: Threshold value (λ)
    
    Returns:
        Thresholded tensor
    """
    return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


def _symmetric_pad_batch(
    array: torch.Tensor, 
    make_square: bool = True,
    pad_mode: str = 'reflect'
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Symmetric padding for 2D or 3D (batch) tensors.
    
    Args:
        array: 2D (H, W) or 3D (B, H, W) tensor
        make_square: Pad to square shape
        pad_mode: Padding mode
    
    Returns:
        Tuple of (padded_tensor, pad_info)
    """
    is_batch = array.dim() == 3
    
    if is_batch:
        B, rows, cols = array.shape
    else:
        rows, cols = array.shape
        B = None
    
    if make_square:
        target_size = max(rows, cols)
        target_rows = target_size
        target_cols = target_size
    else:
        target_rows = rows
        target_cols = cols
    
    pad_top = (target_rows - rows) // 2
    pad_bottom = target_rows - rows - pad_top
    pad_left = (target_cols - cols) // 2
    pad_right = target_cols - cols - pad_left
    
    pad_info = {
        'original_shape': (rows, cols),
        'pad_top': pad_top,
        'pad_bottom': pad_bottom,
        'pad_left': pad_left,
        'pad_right': pad_right,
        'padded_shape': (target_rows, target_cols),
        'is_batch': is_batch,
        'batch_size': B
    }
    
    # F.pad expects (N, C, H, W) for 2D spatial padding
    if is_batch:
        array_4d = array.unsqueeze(1)  # (B, 1, H, W)
    else:
        array_4d = array.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    # Apply padding iteratively if needed (reflect mode has size limit)
    remaining_pad = [pad_left, pad_right, pad_top, pad_bottom]
    current = array_4d
    
    while any(r > 0 for r in remaining_pad):
        h, w = current.shape[2], current.shape[3]
        max_pad_h = h - 1
        max_pad_w = w - 1
        
        this_pad = [
            min(remaining_pad[0], max_pad_w),
            min(remaining_pad[1], max_pad_w),
            min(remaining_pad[2], max_pad_h),
            min(remaining_pad[3], max_pad_h),
        ]
        
        if all(p == 0 for p in this_pad):
            break
            
        current = F.pad(current, tuple(this_pad), mode=pad_mode)
        
        remaining_pad = [
            remaining_pad[0] - this_pad[0],
            remaining_pad[1] - this_pad[1],
            remaining_pad[2] - this_pad[2],
            remaining_pad[3] - this_pad[3],
        ]
    
    if is_batch:
        padded_array = current.squeeze(1)  # (B, H', W')
    else:
        padded_array = current.squeeze(0).squeeze(0)  # (H', W')
    
    return padded_array, pad_info


def _crop_batch(
    array: torch.Tensor, 
    pad_info: Dict[str, Any]
) -> torch.Tensor:
    """
    Crop 2D/3D/4D tensor back to original size.
    
    Args:
        array: Tensor to crop
        pad_info: Padding info from _symmetric_pad_batch
    
    Returns:
        Cropped tensor
    """
    original_rows, original_cols = pad_info['original_shape']
    pad_top = pad_info['pad_top']
    pad_left = pad_info['pad_left']
    
    if array.dim() == 2:
        return array[pad_top:pad_top + original_rows,
                     pad_left:pad_left + original_cols]
    elif array.dim() == 3:
        # Could be (B, H, W) or (H, W, N)
        if pad_info.get('is_batch', False):
            # (B, H, W)
            return array[:, pad_top:pad_top + original_rows,
                         pad_left:pad_left + original_cols]
        else:
            # (H, W, N) - shearlet coefficients
            return array[pad_top:pad_top + original_rows,
                         pad_left:pad_left + original_cols, :]
    elif array.dim() == 4:
        # (B, H, W, N) - batch coefficients
        return array[:, pad_top:pad_top + original_rows,
                     pad_left:pad_left + original_cols, :]
    else:
        raise ValueError(f"Unsupported tensor dimension: {array.dim()}")


def estimate_lipschitz(
    shearletSystem: Dict[str, Any], 
    n_iters: int = 15, 
    verbose: bool = True
) -> float:
    """
    Estimate Lipschitz constant L using power iteration.
    
    L is the spectral norm (largest eigenvalue) of A^T A.
    For synthesis model: A = Synthesis (adjoint2D), A^T = Analysis (dec2D)
    
    Args:
        shearletSystem: Shearlet system dictionary
        n_iters: Number of power iterations
        verbose: Print progress
    
    Returns:
        Estimated Lipschitz constant (with safety margin)
    """
    if verbose:
        print("正在估算 Lipschitz 常数...", end="", flush=True)
    
    rows, cols = shearletSystem['size']
    n_shearlets = shearletSystem['nShearlets']
    device = shearletSystem['device']
    dtype = shearletSystem['dtype']
    
    # Random initialize coefficient tensor (H, W, N)
    b = torch.randn((rows, cols, n_shearlets), device=device, dtype=dtype)
    b = b / torch.norm(b)
    
    b_norm = torch.tensor(1.0)
    
    for _ in range(n_iters):
        img = sl.SLshearadjoint2D(b, shearletSystem)
        b_new = sl.SLsheardec2D(img, shearletSystem)
        
        b_norm = torch.norm(b_new)
        if b_norm < 1e-10:
            break
        b = b_new / b_norm
    
    L = b_norm.item()
    L_safe = L * 1.05
    
    if verbose:
        print(f" 完成。L ≈ {L:.4f}, 安全步长 1/L ≈ {1.0/L_safe:.6f}")
    
    return L_safe


def fista_shearlet_solve(
    y: torch.Tensor,
    lambda_reg: float,
    n_scales: int = 3,
    max_iter: int = 100,
    tol: float = 1e-4,
    verbose: bool = True,
    return_loss: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Solve Shearlet L1 regularization problem using FISTA.
    
    Minimizes: 1/2 ||y - Ψ* c||² + λ||c||₁
    
    Supports both single image and batch processing.
    
    Args:
        y: Input tensor, 2D (H, W) or 3D batch (B, H, W)
        lambda_reg: L1 regularization weight
        n_scales: Number of shearlet decomposition scales
        max_iter: Maximum number of iterations
        tol: Relative error tolerance for convergence
        verbose: Print progress
        return_loss: Whether to compute and return loss history
    
    Returns:
        rec_image: Reconstructed image(s), same shape as input
        coeffs: Final sparse coefficients, (H, W, N) or (B, H, W, N)
        loss_history: Loss values (empty if return_loss=False)
    """
    device = y.device
    original_dtype = y.dtype
    is_batch = y.dim() == 3
    
    if is_batch:
        batch_size, orig_h, orig_w = y.shape
    else:
        orig_h, orig_w = y.shape
        batch_size = None
    
    # --- 1. Preprocessing: Symmetric Padding ---
    y_double = y.to(dtype=torch.float64)
    y_padded, pad_info = _symmetric_pad_batch(y_double, make_square=True)
    
    if is_batch:
        _, rows_pad, cols_pad = y_padded.shape
    else:
        rows_pad, cols_pad = y_padded.shape
    
    if verbose:
        shape_str = f"({batch_size}, {orig_h}, {orig_w})" if is_batch else f"({orig_h}, {orig_w})"
        print(f"输入尺寸: {shape_str}, Padding 后: ({rows_pad}, {cols_pad})")
        print("构建 Shearlet 系统...")
    
    # --- 2. Build Shearlet System ---
    shearletSystem = sl.SLgetShearletSystem2D(
        rows_pad, cols_pad, n_scales, 
        device=device, dtype=torch.float64
    )
    
    # --- 3. Estimate Step Size ---
    L = estimate_lipschitz(shearletSystem, verbose=verbose)
    step_size = 1.0 / L
    threshold = lambda_reg * step_size
    
    # --- 4. Initialize Variables ---
    # Warm start: use analysis of noisy image(s) as initial coefficients
    x = sl.SLsheardec2D(y_padded, shearletSystem)
    z = x.clone()  # Momentum variable
    t = 1.0        # FISTA step size
    
    loss_history: List[float] = []
    
    if verbose:
        batch_str = f", Batch={batch_size}" if is_batch else ""
        print(f"开始 FISTA 迭代 (Max Iter={max_iter}, λ={lambda_reg}{batch_str})")
    
    start_time = time.time()
    
    for k in range(max_iter):
        x_old = x.clone()
        
        # === Gradient Step ===
        # f(c) = 0.5 * ||y - Ψ*(c)||²
        # ∇f(c) = Ψ(Ψ*(c) - y)
        
        # 1. Synthesis: z -> image_est
        rec_z = sl.SLshearadjoint2D(z, shearletSystem)
        
        # 2. Residual
        residual = rec_z - y_padded
        
        # 3. Analysis of residual -> gradient
        grad = sl.SLsheardec2D(residual, shearletSystem)
        
        # === Proximal Step (Gradient Descent + Soft Threshold) ===
        z_grad = z - step_size * grad
        x = soft_threshold(z_grad, threshold)
        
        # === FISTA Momentum Update ===
        t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = x + ((t - 1.0) / t_next) * (x - x_old)
        t = t_next
        
        # === Convergence Check ===
        diff_norm = torch.norm(x - x_old)
        x_norm = torch.norm(x)
        rel_error = (diff_norm / (x_norm + 1e-9)).item()
        
        # Compute loss (optional)
        if return_loss and (k % 10 == 0 or k == max_iter - 1):
            with torch.no_grad():
                l1_val = torch.sum(torch.abs(x))
                data_term = 0.5 * torch.norm(residual) ** 2
                loss_val = (data_term + lambda_reg * l1_val).item()
                loss_history.append(loss_val)
            
            if verbose:
                print(f"  Iter {k:3d} | Loss: {loss_val:.4e} | Rel Err: {rel_error:.2e}")
        
        if rel_error < tol:
            if verbose:
                print(f"收敛于第 {k} 次迭代。")
            break
    
    total_time = time.time() - start_time
    if verbose:
        print(f"求解完成，耗时: {total_time:.2f}s")
    
    # --- 5. Final Reconstruction ---
    final_rec_padded = sl.SLshearadjoint2D(x, shearletSystem)
    
    # --- 6. Crop to Original Size ---
    final_rec = _crop_batch(final_rec_padded, pad_info)
    final_coeffs = _crop_batch(x, pad_info)
    
    return final_rec.to(original_dtype), final_coeffs.to(original_dtype), loss_history


from typing import overload, Literal

@overload
def compute_psnr(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    per_image: Literal[False] = False
) -> float: ...

@overload
def compute_psnr(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    per_image: Literal[True]
) -> torch.Tensor: ...

def compute_psnr(
    original: torch.Tensor, 
    reconstructed: torch.Tensor,
    per_image: bool = False
) -> Union[float, torch.Tensor]:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Ground truth image(s), 2D (H, W) or 3D (B, H, W)
        reconstructed: Reconstructed/denoised image(s)
        per_image: If True and batch input, return PSNR for each image
    
    Returns:
        PSNR in dB (float for single, tensor for batch with per_image=True)
    """
    is_batch = original.dim() == 3
    
    if is_batch and per_image:
        # Compute per-image PSNR
        B = original.shape[0]
        psnrs = []
        for i in range(B):
            mse = torch.mean((original[i] - reconstructed[i]) ** 2)
            if mse == 0:
                psnrs.append(float('inf'))
            else:
                max_val = torch.max(original[i]) - torch.min(original[i])
                psnr = 10 * torch.log10(max_val ** 2 / mse)
                psnrs.append(psnr.item())
        return torch.tensor(psnrs)
    else:
        # Compute global PSNR
        mse = torch.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        max_val = torch.max(original) - torch.min(original)
        psnr = 10 * torch.log10(max_val ** 2 / mse)
        return psnr.item()

