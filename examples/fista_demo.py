"""
FISTA Shearlet L1 Regularization Demo with Batch Processing.

This script demonstrates image denoising using the FISTA algorithm
with Shearlet sparsity prior (Synthesis Model).

Supports both single image and batch processing modes.
"""

import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import torch
import matplotlib.pyplot as plt

from pyshearlab.fista_shearlet import fista_shearlet_solve, compute_psnr

# Ensure output directory exists
os.makedirs("results", exist_ok=True)


def create_test_batch(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    """Create a batch of synthetic test images with varying patterns."""
    batch = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    
    for i in range(B):
        # Base rectangle with varying position
        offset = i * 20
        r_start, r_end = 40 + offset, 180 + offset // 2
        c_start, c_end = 40, 110 + offset
        batch[i, max(0, r_start):min(H, r_end), max(0, c_start):min(W, c_end)] = 0.7 + 0.1 * i
        
        # Circle with varying position/size
        cx, cy = 160 + i * 15, 160 + i * 10
        radius = 35 + i * 5
        mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) < radius ** 2
        batch[i, mask] = 1.0
        
        # Diagonal line
        for j in range(60 + i * 10):
            r, c = 30 + j, 140 + j + i * 10
            if 0 <= r < H and 0 <= c < W:
                batch[i, r, c] = 0.5 + 0.1 * i
    
    return batch


def main():
    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # === Create Batch of Test Images ===
    B = 4  # Batch size
    H, W = 256, 256
    batch_clean = create_test_batch(B, H, W, device)
    
    # === Add Gaussian Noise ===
    sigma = 0.1
    noise = torch.randn_like(batch_clean) * sigma
    batch_noisy = batch_clean + noise
    
    print(f"批量大小: {B}")
    print(f"图像尺寸: {H}x{W}")
    print(f"噪声水平 σ = {sigma}")
    
    # === Run FISTA (Batch Mode) ===
    lambda_reg = 0.08
    n_scales = 3
    
    print("\n--- 开始 FISTA L1 正则化 (批处理模式) ---")
    batch_denoised, coeffs, losses = fista_shearlet_solve(
        batch_noisy,
        lambda_reg=lambda_reg,
        n_scales=n_scales,
        max_iter=100,
        verbose=True
    )
    
    # === Compute PSNR (per image) ===
    psnr_noisy = compute_psnr(batch_clean, batch_noisy, per_image=True)
    psnr_denoised = compute_psnr(batch_clean, batch_denoised, per_image=True)
    
    print(f"\n=== 结果 ===")
    print(f"{'Image':<8} {'Noisy PSNR':>12} {'Denoised PSNR':>15} {'Improvement':>12}")
    print("-" * 50)
    for i in range(B):
        improvement = psnr_denoised[i] - psnr_noisy[i]
        print(f"{i:<8} {psnr_noisy[i]:>10.2f} dB {psnr_denoised[i]:>13.2f} dB {improvement:>10.2f} dB")
    
    avg_noisy = psnr_noisy.mean()
    avg_denoised = psnr_denoised.mean()
    print("-" * 50)
    print(f"{'Average':<8} {avg_noisy:>10.2f} dB {avg_denoised:>13.2f} dB {avg_denoised - avg_noisy:>10.2f} dB")
    
    # === Visualization (show all batch images) ===
    fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
    
    for i in range(B):
        clean_np = batch_clean[i].cpu().numpy()
        noisy_np = batch_noisy[i].cpu().numpy()
        denoised_np = batch_denoised[i].cpu().numpy()
        
        axes[i, 0].imshow(clean_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f"Image {i}: Original")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(noisy_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f"Noisy (PSNR={psnr_noisy[i]:.2f} dB)")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(denoised_np, cmap='gray', vmin=0, vmax=1)
        axes[i, 2].set_title(f"Denoised (PSNR={psnr_denoised[i]:.2f} dB)")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/fista_batch_denoising.png", dpi=150, bbox_inches='tight')
    print(f"\n批处理结果已保存到 results/fista_batch_denoising.png")
    
    # === Convergence Plot ===
    if len(losses) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(range(0, len(losses) * 10, 10), losses, 'b-o', markersize=4)
        plt.title(f"FISTA Convergence (Batch Size={B})")
        plt.xlabel("Iteration")
        plt.ylabel("Objective Value")
        plt.grid(True, alpha=0.3)
        plt.savefig("results/fista_batch_convergence.png", dpi=150, bbox_inches='tight')
        print("收敛曲线已保存到 results/fista_batch_convergence.png")


if __name__ == "__main__":
    main()
