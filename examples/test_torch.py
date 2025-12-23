"""
PyTorch 版 Shearlet 分解与重建测试脚本
参考: examples/test.py (NumPy 版本)
"""

import torch
import numpy as np
import os
from PIL import Image
from pyshearlab import pyShearLab2D_torch as shearlab_torch

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results_torch")

# 确保结果目录存在
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 设置设备 (GPU 可用时自动使用)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

# 1. 加载 0000.tif 图像
image_path = os.path.join(base_dir, "0001.tif")
image_np = np.array(Image.open(image_path), dtype=np.float64)

# 归一化到 [0, 1]
if image_np.max() > 1:
    image_np = image_np / 255.0

# 转为 PyTorch tensor
image = torch.from_numpy(image_np).to(device)

print(f"图像已加载，尺寸: {image.shape}, 设备: {device}")

# 2. 设置参数
rows, cols = image.shape[:2]
scales = 2

print("正在使用带填充的剪切波分解 (PyTorch)...")
print(f"原始图像尺寸: {rows}x{cols}")

# 3. 使用 Padded API 自动处理非方形图像
# 这会自动将图像填充为方形，避免滤波器环绕问题
coeffs, context = shearlab_torch.SLsheardecPadded2D(image, nScales=scales)

print(f"填充后尺寸: {context['pad_info']['padded_shape']}")
print(f"系数尺寸: {coeffs.shape}")

# 4. 正在进行剪切波重构
print("正在进行剪切波重构...")
X_rec = shearlab_torch.SLshearrecPadded2D(coeffs, context)

# 获取系统信息用于后续分析
shearletSystem = context['shearletSystem']

# 计算重建误差
error = torch.max(torch.abs(image - X_rec)).item()
print(f"最大重建误差: {error:.2e}")

# 6. 可视化结果
import matplotlib.pyplot as plt

# 将结果移回 CPU 用于绘图
image_cpu = image.cpu().numpy()
X_rec_cpu = X_rec.cpu().numpy()
coeffs_cpu = coeffs.cpu().numpy()

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.title(f"Original Image from 0000.tif ({rows}x{cols}) - PyTorch")
plt.imshow(image_cpu, cmap='gray', aspect='auto')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title(f"Reconstructed Image (Max Error: {error:.2e})")
plt.imshow(np.real(X_rec_cpu), cmap='gray', aspect='auto')
plt.colorbar()

plt.tight_layout()
save_path = os.path.join(results_dir, "0000_result_torch.png")
plt.savefig(save_path)
print(f"结果已保存至 {save_path}")

# 7. 分析系数并可视化分解层级
print(f"总共生成了 {shearletSystem['nShearlets']} 个剪切波系数通道。")

# 获取尺度信息
idxs = shearletSystem['shearletIdxs']
unique_scales = np.unique(idxs[:, 1])

plt.figure(figsize=(15, 12))
n_rows = len(unique_scales)

for i, scale in enumerate(unique_scales):
    # 找到属于当前尺度的所有系数索引
    scale_mask = (idxs[:, 1] == scale)
    scale_coeffs = coeffs_cpu[:, :, scale_mask]
    
    # 计算该尺度下的能量分布（所有方向的平方和开根号）
    energy = np.sqrt(np.sum(scale_coeffs**2, axis=2))
    
    plt.subplot(n_rows, 1, i + 1)
    if scale == 0:
        plt.title("Scale 0: Low-pass Coefficients")
    else:
        plt.title(f"Scale {int(scale)}: Shearlet Coefficients Energy (Combined Directions)")
    
    plt.imshow(energy, cmap='hot', aspect='auto')
    plt.colorbar(label='Energy')

plt.tight_layout()
save_path = os.path.join(results_dir, "0000_coefficients_analysis_torch.png")
plt.savefig(save_path)
print(f"系数分析结果已保存至 {save_path}")

# 8. 深入分析：可视化特定尺度下的不同方向
target_scale = 1
scale_mask = (idxs[:, 1] == target_scale)
scale_coeffs = coeffs_cpu[:, :, scale_mask]
n_directions = scale_coeffs.shape[2]

plt.figure(figsize=(15, 10))
plt.suptitle(f"Detailed Analysis: Scale {target_scale} Coefficients for Different Directions (PyTorch)")

# 计算网格布局
cols_plot = 4
rows_plot = int(np.ceil(n_directions / cols_plot))

for d in range(n_directions):
    plt.subplot(rows_plot, cols_plot, d + 1)
    info = idxs[scale_mask][d]
    cone_str = "Vertical" if info[0] == 1 else "Horizontal"
    plt.title(f"Dir {d}: {cone_str}, Shear {int(info[2])}")
    plt.imshow(scale_coeffs[:, :, d], cmap='RdBu', aspect='auto')
    plt.axis('off')

plt.tight_layout()
save_path = os.path.join(results_dir, "0000_scale_directions_analysis_torch.png")
plt.savefig(save_path)
print(f"尺度 {target_scale} 的方向分析已保存至 {save_path}")

# 9. 稀疏性与统计分析
print("\n--- 剪切波系数稀疏性与统计分析 (PyTorch) ---")

all_coeffs = coeffs_cpu.flatten()
total_elements = all_coeffs.size
magnitudes = np.abs(all_coeffs)
sorted_magnitudes = np.sort(magnitudes)[::-1]
cumulative_energy = np.cumsum(sorted_magnitudes**2)
total_energy = cumulative_energy[-1]

energy_thresholds = [0.5, 0.9, 0.95, 0.99]

print(f"总系数数量: {total_elements}")
for threshold in energy_thresholds:
    count = np.searchsorted(cumulative_energy, threshold * total_energy) + 1
    percentage = (count / total_elements) * 100
    print(f"包含 {threshold*100:.0f}% 能量所需的系数比例: {percentage:.4f}% ({count} 个)")

# 各尺度统计
print("\n各尺度统计信息:")
for scale in unique_scales:
    scale_mask = (idxs[:, 1] == scale)
    scale_coeffs = coeffs_cpu[:, :, scale_mask]
    scale_mag = np.abs(scale_coeffs)
    
    print(f"Scale {int(scale)}:")
    print(f"  - 最大幅值: {np.max(scale_mag):.6f}")
    print(f"  - 平均幅值: {np.mean(scale_mag):.6f}")
    print(f"  - 标准差:   {np.std(scale_mag):.6f}")
    print(f"  - 能量占比: {(np.sum(scale_mag**2) / total_energy)*100:.2f}%")

# 可视化稀疏性
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(magnitudes + 1e-10, bins=100, log=True, color='skyblue', edgecolor='black')
plt.title("Coefficient Magnitude Distribution (Log Scale)")
plt.xlabel("Magnitude")
plt.ylabel("Frequency (Log)")
plt.grid(True, which='both', linestyle='--', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(np.linspace(0, 100, len(cumulative_energy)), cumulative_energy / total_energy * 100, 'r-', linewidth=2)
plt.title("Cumulative Energy vs. Percentage of Coefficients")
plt.xlabel("Percentage of Coefficients (%)")
plt.ylabel("Cumulative Energy (%)")
plt.grid(True)
plt.xlim(0, 10)
plt.axhline(y=95, color='g', linestyle='--', label='95% Energy')
plt.legend()

plt.tight_layout()
save_path = os.path.join(results_dir, "0000_sparsity_analysis_torch.png")
plt.savefig(save_path)
print(f"\n稀疏性分析图表已保存至 {save_path}")

plt.show()
