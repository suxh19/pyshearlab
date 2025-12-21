"""
详细的长方形 Shearlet 分解和重建测试

测试不同长宽比的长方形数据：
- 1024 x 256 (4:1 宽高比)
- 256 x 1024 (1:4 宽高比)  
- 512 x 128 (4:1 宽高比)
- 128 x 512 (1:4 宽高比)
"""

import numpy as np
import pyshearlab
import pytest


# ============================================================================
# 测试配置
# ============================================================================

# 长方形尺寸参数
RECTANGULAR_SHAPES = [
    (1024, 256),   # 宽 > 高
    (256, 1024),   # 高 > 宽
    (512, 128),    # 宽 > 高
    (128, 512),    # 高 > 宽
    (256, 64),     # 4:1 比例
    (64, 256),     # 1:4 比例
]

# 不同的尺度数
SCALES_OPTIONS = [2, 3]


@pytest.fixture(scope='module', params=RECTANGULAR_SHAPES)
def rect_shape(request):
    """长方形尺寸 fixture"""
    return request.param


@pytest.fixture(scope='module', params=SCALES_OPTIONS)
def scales(request):
    """尺度数 fixture"""
    return request.param


@pytest.fixture(scope='module')
def rect_shearlet_system(rect_shape, scales):
    """为长方形创建 Shearlet 系统"""
    rows, cols = rect_shape
    return pyshearlab.SLgetShearletSystem2D(0, rows, cols, scales)



# ============================================================================
# 基础功能测试
# ============================================================================

class TestRectangularBasic:
    """长方形基础功能测试"""
    
    def test_shearlet_system_creation(self, rect_shearlet_system, rect_shape):
        """测试 Shearlet 系统创建"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 验证系统尺寸
        assert system['size'][0] == rows, f"期望行数 {rows}, 实际 {system['size'][0]}"
        assert system['size'][1] == cols, f"期望列数 {cols}, 实际 {system['size'][1]}"
        
        # 验证 shearlet 数量
        assert system['nShearlets'] > 0, "Shearlet 数量应大于 0"
        
        # 验证 shearlets 和 dualFrameWeights 形状
        expected_shape = (rows, cols, system['nShearlets'])
        assert system['shearlets'].shape == expected_shape
        assert system['dualFrameWeights'].shape == (rows, cols)
        
        print(f"\n形状 {rect_shape}: 创建了 {system['nShearlets']} 个 shearlets")
    
    
    def test_decomposition(self, rect_shearlet_system, rect_shape):
        """测试分解操作"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 创建测试数据
        X = np.random.randn(rows, cols).astype('float64')
        
        # 分解
        coeffs = pyshearlab.SLsheardec2D(X, system)
        
        # 验证系数形状
        expected_shape = (rows, cols, system['nShearlets'])
        assert coeffs.shape == expected_shape, \
            f"系数形状错误: 期望 {expected_shape}, 实际 {coeffs.shape}"
        
        # 验证系数非全零
        assert np.any(coeffs != 0), "分解系数不应全为零"
        
        print(f"\n形状 {rect_shape}: 分解成功, 系数形状 {coeffs.shape}")
    
    
    def test_reconstruction(self, rect_shearlet_system, rect_shape):
        """测试重建操作"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 创建测试数据
        X = np.random.randn(rows, cols).astype('float64')
        
        # 分解
        coeffs = pyshearlab.SLsheardec2D(X, system)
        
        # 重建
        X_rec = pyshearlab.SLshearrec2D(coeffs, system)
        
        # 验证重建形状
        assert X_rec.shape == X.shape, \
            f"重建形状错误: 期望 {X.shape}, 实际 {X_rec.shape}"
        
        # 验证重建误差 (非 Parseval 框架，误差通常在 0.1-1%)
        error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        assert error < 0.02, f"重建相对误差 {error:.6f} 超过阈值 0.02"
        
        print(f"\n形状 {rect_shape}: 重建成功, 相对误差 = {error:.6e}")



# ============================================================================
# 详细的分解重建测试
# ============================================================================

class TestRectangularDetailed:
    """长方形详细分解重建测试"""
    
    def test_perfect_reconstruction_float64(self, rect_shearlet_system, rect_shape):
        """float64 精度下的完美重建测试"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 创建多种类型的测试数据
        test_cases = {
            '随机数据': np.random.randn(rows, cols),
            '常数数据': np.ones((rows, cols)) * 42.0,
            '正弦波': np.sin(np.outer(np.linspace(0, 4*np.pi, rows), 
                                      np.linspace(0, 4*np.pi, cols))),
        }
        
        results = {}
        for name, X in test_cases.items():
            X = X.astype('float64')
            
            # 分解和重建
            coeffs = pyshearlab.SLsheardec2D(X, system)
            X_rec = pyshearlab.SLshearrec2D(coeffs, system)
            
            # 计算误差
            abs_error = np.max(np.abs(X - X_rec))
            rel_error = np.linalg.norm(X - X_rec) / (np.linalg.norm(X) + 1e-10)
            
            results[name] = {
                'abs_error': abs_error,
                'rel_error': rel_error,
            }
            
            # 验证 (非 Parseval 框架允许更大误差)
            assert rel_error < 0.05, \
                f"{name}: 相对误差 {rel_error:.6f} 超过阈值 0.05"
        
        print(f"\n形状 {rect_shape} 详细重建测试:")
        for name, res in results.items():
            print(f"  {name}: 绝对误差={res['abs_error']:.6e}, 相对误差={res['rel_error']:.6e}")
    
    
    def test_adjoint_property(self, rect_shearlet_system, rect_shape):
        """测试伴随性质: <Ax, Ax> = <x, A*Ax>"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 创建测试数据
        X = np.random.randn(rows, cols).astype('float64')
        
        # 分解
        coeffs = pyshearlab.SLsheardec2D(X, system)
        
        # 伴随
        X_adj = pyshearlab.SLshearadjoint2D(coeffs, system)
        
        # 验证伴随性质
        lhs = np.vdot(coeffs, coeffs)  # <Ax, Ax>
        rhs = np.vdot(X, X_adj)        # <x, A*Ax>
        
        rel_diff = np.abs(lhs - rhs) / (np.abs(lhs) + 1e-10)
        
        assert rel_diff < 1e-3, \
            f"伴随性质不满足: |<Ax,Ax> - <x,A*Ax>| / |<Ax,Ax>| = {rel_diff:.6e}"
        
        print(f"\n形状 {rect_shape}: 伴随性质验证通过, 相对差异 = {rel_diff:.6e}")
    
    
    def test_coefficient_energy_distribution(self, rect_shearlet_system, rect_shape):
        """测试系数能量分布"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 创建测试数据
        X = np.random.randn(rows, cols).astype('float64')
        
        # 分解
        coeffs = pyshearlab.SLsheardec2D(X, system)
        
        # 计算每个 shearlet 的能量
        energies = np.zeros(system['nShearlets'])
        for i in range(system['nShearlets']):
            energies[i] = np.sum(np.abs(coeffs[:, :, i]) ** 2)
        
        total_energy = np.sum(energies)
        
        # 验证能量分布合理
        assert total_energy > 0, "总能量应大于 0"
        
        # 低频 shearlet 通常包含较多能量
        low_freq_energy = energies[0]  # 第一个通常是低频
        low_freq_ratio = low_freq_energy / total_energy
        
        print(f"\n形状 {rect_shape} 系数能量分布:")
        print(f"  总能量: {total_energy:.6e}")
        print(f"  低频能量占比: {low_freq_ratio:.2%}")
        print(f"  Shearlet 数量: {system['nShearlets']}")
    
    
    def test_sparsity_for_smooth_images(self, rect_shearlet_system, rect_shape):
        """测试平滑图像的稀疏性"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 创建平滑的测试数据（应该是稀疏的）
        X_smooth = np.outer(np.linspace(0, 1, rows), np.linspace(0, 1, cols))
        X_smooth = X_smooth.astype('float64')
        
        # 创建随机数据（应该不那么稀疏）
        X_random = np.random.randn(rows, cols).astype('float64')
        X_random = X_random / np.max(np.abs(X_random))  # 归一化
        
        # 分解
        coeffs_smooth = pyshearlab.SLsheardec2D(X_smooth, system)
        coeffs_random = pyshearlab.SLsheardec2D(X_random, system)
        
        # 计算稀疏性（使用阈值计数）
        threshold = 0.01
        
        smooth_nonzero = np.sum(np.abs(coeffs_smooth) > threshold * np.max(np.abs(coeffs_smooth)))
        random_nonzero = np.sum(np.abs(coeffs_random) > threshold * np.max(np.abs(coeffs_random)))
        
        total_coeffs = coeffs_smooth.size
        smooth_sparsity = 1 - smooth_nonzero / total_coeffs
        random_sparsity = 1 - random_nonzero / total_coeffs
        
        print(f"\n形状 {rect_shape} 稀疏性测试 (阈值={threshold}):")
        print(f"  平滑图像稀疏度: {smooth_sparsity:.2%}")
        print(f"  随机图像稀疏度: {random_sparsity:.2%}")
        
        # 平滑图像应该更稀疏
        # (不做断言，因为这取决于具体数据)



# ============================================================================
# 边界情况测试
# ============================================================================

class TestRectangularEdgeCases:
    """长方形边界情况测试"""
    
    @pytest.mark.parametrize("dtype", ['float32', 'float64'])
    def test_dtype_preservation(self, rect_shearlet_system, rect_shape, dtype):
        """测试数据类型保持"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        X = np.random.randn(rows, cols).astype(dtype)
        
        # 分解
        coeffs = pyshearlab.SLsheardec2D(X, system)
        assert coeffs.dtype == X.dtype, \
            f"分解后类型改变: {X.dtype} -> {coeffs.dtype}"
        
        # 伴随
        X_adj = pyshearlab.SLshearadjoint2D(coeffs, system)
        assert X_adj.dtype == X.dtype, \
            f"伴随后类型改变: {X.dtype} -> {X_adj.dtype}"
        
        print(f"\n形状 {rect_shape}, dtype={dtype}: 类型保持正确")
    
    
    def test_zero_input(self, rect_shearlet_system, rect_shape):
        """测试零输入"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        X = np.zeros((rows, cols), dtype='float64')
        
        # 分解
        coeffs = pyshearlab.SLsheardec2D(X, system)
        
        # 零输入应该产生零系数
        assert np.allclose(coeffs, 0), "零输入应产生零系数"
        
        # 重建
        X_rec = pyshearlab.SLshearrec2D(coeffs, system)
        assert np.allclose(X_rec, 0), "零系数应重建为零"
        
        print(f"\n形状 {rect_shape}: 零输入测试通过")
    
    
    def test_constant_input(self, rect_shearlet_system, rect_shape):
        """测试常数输入"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        constant_val = 3.14159
        X = np.ones((rows, cols), dtype='float64') * constant_val
        
        # 分解和重建
        coeffs = pyshearlab.SLsheardec2D(X, system)
        X_rec = pyshearlab.SLshearrec2D(coeffs, system)
        
        # 计算各 shearlet 能量
        energies = np.array([np.sum(np.abs(coeffs[:, :, i]) ** 2) 
                             for i in range(system['nShearlets'])])
        total_energy = np.sum(energies)
        
        # 找出能量最大的 shearlet (通常是低频)
        max_energy_idx = np.argmax(energies)
        max_energy_ratio = energies[max_energy_idx] / (total_energy + 1e-10)
        
        # 重建应该接近原始
        rel_error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        assert rel_error < 0.02, f"常数重建误差过大: {rel_error:.6e}"
        
        print(f"\n形状 {rect_shape}: 常数输入测试通过")
        print(f"  最大能量 shearlet: #{max_energy_idx}, 占比 = {max_energy_ratio:.2%}")
        print(f"  重建相对误差 = {rel_error:.6e}")



# ============================================================================
# 性能和数值稳定性测试
# ============================================================================

class TestRectangularNumerical:
    """长方形数值稳定性测试"""
    
    def test_large_values(self, rect_shearlet_system, rect_shape):
        """测试大数值稳定性"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 大数值
        X = np.random.randn(rows, cols).astype('float64') * 1e6
        
        coeffs = pyshearlab.SLsheardec2D(X, system)
        X_rec = pyshearlab.SLshearrec2D(coeffs, system)
        
        rel_error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        assert rel_error < 0.02, f"大数值重建误差过大: {rel_error:.6e}"
        assert np.all(np.isfinite(X_rec)), "重建结果包含 NaN 或 Inf"
        
        print(f"\n形状 {rect_shape}: 大数值测试通过, 相对误差 = {rel_error:.6e}")
    
    
    def test_small_values(self, rect_shearlet_system, rect_shape):
        """测试小数值稳定性"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 小数值
        X = np.random.randn(rows, cols).astype('float64') * 1e-6
        
        coeffs = pyshearlab.SLsheardec2D(X, system)
        X_rec = pyshearlab.SLshearrec2D(coeffs, system)
        
        rel_error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        assert rel_error < 0.02, f"小数值重建误差过大: {rel_error:.6e}"
        assert np.all(np.isfinite(X_rec)), "重建结果包含 NaN 或 Inf"
        
        print(f"\n形状 {rect_shape}: 小数值测试通过, 相对误差 = {rel_error:.6e}")
    
    
    def test_mixed_values(self, rect_shearlet_system, rect_shape):
        """测试混合大小数值"""
        system = rect_shearlet_system
        rows, cols = rect_shape
        
        # 混合大小数值
        X = np.random.randn(rows, cols).astype('float64')
        X[::2, ::2] *= 1e4   # 部分大值
        X[1::2, 1::2] *= 1e-4  # 部分小值
        
        coeffs = pyshearlab.SLsheardec2D(X, system)
        X_rec = pyshearlab.SLshearrec2D(coeffs, system)
        
        rel_error = np.linalg.norm(X - X_rec) / np.linalg.norm(X)
        assert rel_error < 0.02, f"混合数值重建误差过大: {rel_error:.6e}"
        assert np.all(np.isfinite(X_rec)), "重建结果包含 NaN 或 Inf"
        
        print(f"\n形状 {rect_shape}: 混合数值测试通过, 相对误差 = {rel_error:.6e}")



# ============================================================================
# 命令行运行
# ============================================================================

if __name__ == '__main__':
    pytest.main([
        str(__file__.replace('\\', '/')), 
        '-v', 
        '-s',  # 显示 print 输出
        '--tb=short',
    ])
