# utils/visualization/objects.py
import numpy as np

class VisObject:
    """所有可视化对象的基类，只定义接口，不做实现。"""
    pass


class PointCloudObject(VisObject):
    def __init__(self, points, colors=None, normals=None):
        self.points = np.asarray(points, dtype=np.float32)
        self.colors = None if colors is None else np.asarray(colors, dtype=np.float32)
        self.normals = None if normals is None else np.asarray(normals, dtype=np.float32)

    @property
    def num_points(self):
        """返回点云数量"""
        return len(self.points)

    @property
    def min_point(self):
        """返回点云每个维度的最小值 (x_min, y_min, z_min)"""
        return np.min(self.points, axis=0)

    @property
    def max_point(self):
        """返回点云每个维度的最大值 (x_max, y_max, z_max)"""
        return np.max(self.points, axis=0)

    @property
    def bounds(self):
        """返回点云的包围盒 (min_point, max_point)"""
        return self.min_point, self.max_point

    def summary(self):
        """打印点云基本信息"""
        mn, mx = self.bounds
        print("PointCloud Summary:")
        print(f"  Points: {self.num_points}")
        print(f"  Min: {mn}")
        print(f"  Max: {mx}")


class MeshObject(VisObject):
    def __init__(self, vertices, faces, colors=None, normals=None, uv=None, texture=None):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.colors = colors
        self.normals = normals
        self.uv = uv
        self.texture = texture   # 可以先留空，将来你要搞贴图再用


class ImageObject(VisObject):
    def __init__(self, image):
        # H x W x 3 or 1
        self.image = np.asarray(image)
