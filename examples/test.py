import numpy as np
import pyshearlab
import os
from PIL import Image

# 获取当前脚本所在目录
base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")

# 确保结果目录存在
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# 1. 加载 0000.tif 图像
image_path = os.path.join(base_dir, "0000.tif")
image = np.array(Image.open(image_path), dtype=np.float64)

# 归一化到 [0, 1]
if image.max() > 1:
    image = image / 255.0

print(f"图像已加载，尺寸: {image.shape}")

# 2. 设置参数
rows, cols = image.shape[:2]
scales = 3

print("正在生成剪切波系统...")

# 3. 调用函数生成剪切波系统 (使用库内部默认滤波器)
shearletSystem = pyshearlab.SLgetShearletSystem2D(0, rows, cols, scales)

print("系统生成成功！")

# 4. 正在进行剪切波分解
print("正在进行剪切波分解...")
coeffs = pyshearlab.SLsheardec2D(image, shearletSystem)

# 5. 正在进行剪切波重构
print("正在进行剪切波重构...")
X_rec = pyshearlab.SLshearrec2D(coeffs, shearletSystem)

# 6. 可视化结果
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.title(f"Original Image from 0000.tif ({rows}x{cols})")
plt.imshow(image, cmap='gray', aspect='auto')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title("Reconstructed Image")
plt.imshow(np.real(X_rec), cmap='gray', aspect='auto')
plt.colorbar()

plt.tight_layout()
save_path = os.path.join(results_dir, "0000_result.png")
plt.savefig(save_path)
print(f"结果已保存至 {save_path}")

# 7. 分析系数并可视化分解层级
print(f"总共生成了 {shearletSystem['nShearlets']} 个剪切波系数通道。")

# 获取尺度信息
# shearletIdxs 格式为 [cone, scale, shearing]
# scale=0 是低频部分
idxs = shearletSystem['shearletIdxs']
unique_scales = np.unique(idxs[:, 1])

plt.figure(figsize=(15, 12))
n_rows = len(unique_scales)

for i, scale in enumerate(unique_scales):
    # 找到属于当前尺度的所有系数索引
    scale_mask = (idxs[:, 1] == scale)
    scale_coeffs = coeffs[:, :, scale_mask]
    
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
save_path = os.path.join(results_dir, "0000_coefficients_analysis.png")
plt.savefig(save_path)
print(f"系数分析结果已保存至 {save_path}")

# 8. 深入分析：可视化特定尺度下的不同方向
target_scale = 1
scale_mask = (idxs[:, 1] == target_scale)
scale_coeffs = coeffs[:, :, scale_mask]
n_directions = scale_coeffs.shape[2]

plt.figure(figsize=(15, 10))
plt.suptitle(f"Detailed Analysis: Scale {target_scale} Coefficients for Different Directions")

# 计算网格布局
cols_plot = 4
rows_plot = int(np.ceil(n_directions / cols_plot))

for d in range(n_directions):
    plt.subplot(rows_plot, cols_plot, d + 1)
    # 获取该方向的索引信息 [cone, scale, shearing]
    info = idxs[scale_mask][d]
    cone_str = "Vertical" if info[0] == 1 else "Horizontal"
    plt.title(f"Dir {d}: {cone_str}, Shear {int(info[2])}")
    plt.imshow(scale_coeffs[:, :, d], cmap='RdBu', aspect='auto')
    plt.axis('off')

plt.tight_layout()
save_path = os.path.join(results_dir, "0000_scale_directions_analysis.png")
plt.savefig(save_path)
print(f"尺度 {target_scale} 的方向分析已保存至 {save_path}")
plt.show()
