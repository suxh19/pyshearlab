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

# 1. 加载 0000.tif 图像
image_path = os.path.join(base_dir, "0000.tif")
image = np.array(Image.open(image_path), dtype=np.float64)

# 归一化到 [0, 1]
if image.max() > 1:
    image = image / 255.0

print(f"图像已加载，尺寸: {image.shape}")

# 2. 设置参数
# 使用新的高层级 API：SLsheardecPadded2D
# 它会自动处理非正方形图像的填充问题（解决滤波器卷绕）
nScales = 3

print("正在进行剪切波分解 (使用自动填充 API)...")

# 3. 直接调用 SLsheardecPadded2D
# 无需手动生成 shearletSystem，API 内部会处理
coeffs, context = pyshearlab.SLsheardecPadded2D(image, nScales=nScales)

print(f"分解完成。系数尺寸: {coeffs.shape} (已自动裁剪回原始尺寸)")
print(f"总共生成了 {context['shearletSystem']['nShearlets']} 个剪切波系数通道。")

# 4. 正在进行剪切波重构
print("正在进行剪切波重构 (使用自动处理 API)...")
# 使用配套的 SLshearrecPadded2D 函数
X_rec = pyshearlab.SLshearrecPadded2D(coeffs, context)

# 5. 可视化结果
plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
plt.title(f"Original Image from 0000.tif ({image.shape[0]}x{image.shape[1]})")
plt.imshow(image, cmap='gray', aspect='auto')
plt.colorbar()

plt.subplot(2, 1, 2)
plt.title("Reconstructed Image (using SLshearrecPadded2D)")
plt.imshow(np.real(X_rec), cmap='gray', aspect='auto')
plt.colorbar()

plt.tight_layout()
save_path = os.path.join(results_dir, "new_api_result.png")
plt.savefig(save_path)
print(f"结果已保存至 {save_path}")

# 6. 分析系数并可视化能量分布
idxs = context['shearletSystem']['shearletIdxs']
unique_scales = np.unique(idxs[:, 1])

plt.figure(figsize=(15, 12))
n_rows = len(unique_scales)

for i, scale in enumerate(unique_scales):
    scale_mask = (idxs[:, 1] == scale)
    scale_coeffs = coeffs[:, :, scale_mask]
    
    # 计算能量分布
    energy = np.sqrt(np.sum(scale_coeffs**2, axis=2))
    
    plt.subplot(n_rows, 1, i + 1)
    if scale == 0:
        plt.title("Scale 0: Low-pass Coefficients")
    else:
        plt.title(f"Scale {int(scale)}: Shearlet Coefficients Energy (No Wrapping Artifacts)")
    
    plt.imshow(energy, cmap='hot', aspect='auto')
    plt.colorbar(label='Energy')

plt.tight_layout()
save_path = os.path.join(results_dir, "new_api_coefficients_analysis.png")
plt.savefig(save_path)
print(f"系数分析结果已保存至 {save_path}")

# 7. 重建误差分析
error = np.abs(image - np.real(X_rec))
print(f"\n重建误差统计:")
print(f"  - 最大误差: {np.max(error):.2e}")
print(f"  - 平均误差: {np.mean(error):.2e}")

plt.show()
print("\n✅ 新 API 测试完成！")
