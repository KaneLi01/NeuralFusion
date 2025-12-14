import numpy as np
from typing import Tuple

def make_cube_grid(lim: float, res: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray]:
    """
    在 [-lim, lim]^3 上生成采样点网格。
    返回:
      grid: (res^3, 3)
      x: (res,) 轴坐标（立方体三轴相同）
    """
    x = np.linspace(-lim, lim, res, dtype=dtype)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return grid

def make_grid_from_bounds(
    min_xyz: np.ndarray,
    max_xyz: np.ndarray,
    res: int,
    dtype=np.float32
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    在轴对齐包围盒 [min_xyz, max_xyz] 上生成 3D 采样网格。

    Args:
        min_xyz: (3,) 最小点
        max_xyz: (3,) 最大点
        res:     每个维度的采样分辨率
        dtype:   数据类型

    Returns:
        grid: (res^3, 3) 采样点坐标
        axes: (x, y, z)  各轴采样坐标，各为 shape (res,)
    """
    min_xyz = np.asarray(min_xyz, dtype=dtype)
    max_xyz = np.asarray(max_xyz, dtype=dtype)

    if min_xyz.shape != (3,) or max_xyz.shape != (3,):
        raise ValueError("min_xyz and max_xyz must have shape (3,)")

    x = np.linspace(min_xyz[0], max_xyz[0], res, dtype=dtype)
    y = np.linspace(min_xyz[1], max_xyz[1], res, dtype=dtype)
    z = np.linspace(min_xyz[2], max_xyz[2], res, dtype=dtype)

    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    return grid

def compute_points_cm(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    给定点云 points，计算：
    - center: 点云的几何中心（均值）
    - min_xyz: 每个维度上的最小值
    - max_xyz: 每个维度上的最大值

    Args:
        points: (N, 3) 点云

    Returns:
        center: (3,)  点云中心
        min_xyz: (3,) 包围盒最小点
        max_xyz: (3,) 包围盒最大点
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    points = np.asarray(points, dtype=np.float32)

    center = np.mean(points, axis=0)
    min_xyz = np.min(points, axis=0)
    max_xyz = np.max(points, axis=0)

    return center, min_xyz, max_xyz