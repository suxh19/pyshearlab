"""
修正版剪切波分解脚本 - 使用正方形对称填充

解决长宽比悬殊图像（如 256x1024）导致的:
1. 滤波器卷绕（Filter Wrapping）: np.roll 的循环移位在短边上卷绕多次
2. 振铃效应（Gibbs Phenomenon）: 零填充导致的边缘突变
"""

import numpy as np
import pyshearlab
import os
from PIL import Image
import matplotlib.pyplot as plt

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")

# 确保结果目录存在
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 1. 加载图像
image_path = os.path.join(base_dir, "0000.tif")
image = np.array(Image.open(image_path), dtype=np.float64)

# 归一化到 [0, 1]
if image.max() > 1:
    image = image / 255.0

print(f"原始图像尺寸: {image.shape}")
rows, cols = image.shape[:2]

# ============================================================================
# 【关键步骤 1】将图像填充为正方形，使用对称镜像填充
# ============================================================================
target_size = max(rows, cols)  # 目标尺寸取长边

# 计算填充量：将图像填充为正方形
pad_top = (target_size - rows) // 2
pad_bottom = target_size - rows - pad_top
pad_left = (target_size - cols) // 2
pad_right = target_size - cols - pad_left

print(f"填充量: 上={pad_top}, 下={pad_bottom}, 左={pad_left}, 右={pad_right}")

# 使用 'symmetric' 对称填充（镜像边界）
# 这会像镜子一样延展边界，消除突变，欺骗 FFT 认为信号是连续的
img_padded = np.pad(image, 
                    ((pad_top, pad_bottom), (pad_left, pad_right)), 
                    mode='symmetric')

print(f"填充后尺寸: {img_padded.shape}")  # 应该是 (target_size, target_size)
assert img_padded.shape == (target_size, target_size), "填充后必须是正方形！"

# ============================================================================
# 【关键步骤 2】基于【填充后的尺寸】生成 Shearlet 系统
# ============================================================================
# 必须用正方形尺寸生成系统，防止滤波器卷绕
scales = 2
print(f"正在生成 {target_size}x{target_size} 的剪切波系统...")
shearletSystem = pyshearlab.SLgetShearletSystem2D(0, target_size, target_size, scales)
print("系统生成成功！")

# ============================================================================
# 【关键步骤 3】对填充后的图像进行分解
# ============================================================================
print("正在进行剪切波分解...")
coeffs_padded = pyshearlab.SLsheardec2D(img_padded, shearletSystem)
print(f"填充图像系数尺寸: {coeffs_padded.shape}")

# ============================================================================
# 【关键步骤 4】裁剪回原始尺寸
# ============================================================================
# 分解后的系数包含了填充区域，必须切掉
coeffs = coeffs_padded[pad_top : pad_top + rows, 
                       pad_left : pad_left + cols, 
                       :]
print(f"裁剪后系数尺寸: {coeffs.shape}")

# ============================================================================
# 【关键步骤 5】重建也需要同样的流程
# ============================================================================
print("正在进行剪切波重构...")
X_rec_padded = pyshearlab.SLshearrec2D(coeffs_padded, shearletSystem)

# 裁剪回原始尺寸
X_rec = X_rec_padded[pad_top : pad_top + rows, 
                     pad_left : pad_left + cols]
print(f"裁剪后重建图像尺寸: {X_rec.shape}")

# ============================================================================
# 可视化对比
# ============================================================================

# 对比图 1: 原图 vs 重建
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.title(f"Original Image ({rows}x{cols})")
plt.imshow(image, cmap='gray', aspect='auto')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.title(f"Padded Image ({target_size}x{target_size})")
plt.imshow(img_padded, cmap='gray', aspect='auto')
plt.colorbar()
# 标记原图区域
rect = plt.Rectangle((pad_left, pad_top), cols, rows, 
                      linewidth=2, edgecolor='red', facecolor='none')
plt.gca().add_patch(rect)

plt.subplot(2, 2, 3)
plt.title("Reconstructed (Cropped Back)")
plt.imshow(np.real(X_rec), cmap='gray', aspect='auto')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.title("Reconstruction Error (×1000)")
error = np.abs(image - np.real(X_rec)) * 1000
plt.imshow(error, cmap='hot', aspect='auto')
plt.colorbar()

plt.tight_layout()
save_path = os.path.join(results_dir, "square_padding_reconstruction.png")
plt.savefig(save_path, dpi=150)
print(f"重建对比图已保存至 {save_path}")

# 对比图 2: 系数可视化（检查边缘是否干净）
idxs = shearletSystem['shearletIdxs']
unique_scales = np.unique(idxs[:, 1])

plt.figure(figsize=(15, 12))
n_rows = len(unique_scales)

for i, scale in enumerate(unique_scales):
    scale_mask = (idxs[:, 1] == scale)
    scale_coeffs = coeffs[:, :, scale_mask]
    
    # 计算该尺度下的能量分布
    energy = np.sqrt(np.sum(scale_coeffs**2, axis=2))
    
    plt.subplot(n_rows, 1, i + 1)
    if scale == 0:
        plt.title("Scale 0: Low-pass Coefficients (Cropped)")
    else:
        plt.title(f"Scale {int(scale)}: Shearlet Coefficients Energy (Cropped, No Edge Wrapping)")
    
    plt.imshow(energy, cmap='hot', aspect='auto')
    plt.colorbar(label='Energy')

plt.tight_layout()
save_path = os.path.join(results_dir, "square_padding_coefficients.png")
plt.savefig(save_path, dpi=150)
print(f"系数分析图已保存至 {save_path}")

# 对比图 3: 特定尺度下的不同方向（检查方向一致性）
target_scale = 1
scale_mask = (idxs[:, 1] == target_scale)
scale_coeffs = coeffs[:, :, scale_mask]
n_directions = scale_coeffs.shape[2]

plt.figure(figsize=(15, 10))
plt.suptitle(f"Scale {target_scale}: Direction Coefficients (Square Padding Method)")

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
save_path = os.path.join(results_dir, "square_padding_directions.png")
plt.savefig(save_path, dpi=150)
print(f"方向分解图已保存至 {save_path}")

# 打印重建误差
max_error = np.max(np.abs(image - np.real(X_rec)))
mean_error = np.mean(np.abs(image - np.real(X_rec)))
print(f"\n重建误差: 最大={max_error:.2e}, 平均={mean_error:.2e}")

plt.show()
print("\n✅ 正方形对称填充方法处理完成！边缘卷绕问题应该已解决。")
