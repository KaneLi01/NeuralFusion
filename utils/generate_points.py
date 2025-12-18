import numpy as np

# --- 具体的形状生成函数 ---

def _gen_f_block(num_points):
    """生成 F 形状 """
    spine = np.random.uniform([-1.0, -0.8], [1.0, -0.4], (num_points//2, 2))
    top = np.random.uniform([0.6, -0.4], [1.0, 0.6], (num_points//4, 2))
    mid = np.random.uniform([-0.2, -0.4], [0.2, 0.2], (num_points//4, 2))
    yz = np.vstack([spine, top, mid])
    x = np.random.uniform(-0.5, 0.5, len(yz))
    points = np.column_stack([x, yz[:,0], yz[:,1]])
    return points

def _gen_sphere(num_points):
    """生成简单的球体"""
    vec = np.random.randn(num_points, 3)
    vec /= np.linalg.norm(vec, axis=1)[:, np.newaxis]
    return vec

def _gen_plane(num_points):
    """生成平面用于测试最简单的拟合"""
    xy = np.random.uniform(-1, 1, (num_points, 2))
    z = np.zeros(num_points)
    return np.column_stack([xy, z])

# --- 统一入口 ---

def get_mock_cloud(shape_type='f_block', num_points=4000, noise=0.01):
    """
    测试数据的统一工厂函数
    """
    generators = {
        'f_block': _gen_f_block,
        'sphere': _gen_sphere,
        'plane': _gen_plane
    }
    
    if shape_type not in generators:
        raise ValueError(f"未知形状: {shape_type}. 可选: {list(generators.keys())}")
        
    # 生成基础点
    points = generators[shape_type](num_points)
    
    # 统一添加噪声 (测试鲁棒性时很有用)
    if noise > 0:
        points += np.random.normal(0, noise, points.shape)
        
    return points