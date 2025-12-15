import numpy as np
from .base import BaseRenderer
from ..objects import PointCloudObject, MeshObject

try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("[Open3DRenderer] Open3D not installed, visualization disabled.")


class Open3DRenderer(BaseRenderer):
    def __init__(self):
        if not HAS_OPEN3D:
            raise RuntimeError("Open3D is not available. Please install it first.")

    def render(self, obj, window_name="Open3D Viewer"):
        """
        obj 可以是：
          - 单个 PointCloudObject
          - 单个 MeshObject
          - 任意个对象组成的 list / tuple / set：
                [pc1, mesh1, pc2, ...]
        """
        geoms = self._build_geometry_list(obj)
        if len(geoms) == 0:
            raise ValueError("No valid geometry to render.")
        o3d.visualization.draw_geometries(geoms, window_name=window_name)

    # --------- 内部工具：把任意输入转成 o3d geometry 列表 ---------
    def _build_geometry_list(self, obj):
        geoms = []

        # 情况 1：输入是容器（list / tuple / set 等）
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                geoms.extend(self._build_geometry_list(item))  # 递归展开
            return geoms

        # 情况 2：单个点云
        if isinstance(obj, PointCloudObject):
            geoms.append(self._to_o3d_point_cloud(obj))
            return geoms

        # 情况 3：单个 mesh
        if isinstance(obj, MeshObject):
            geoms.append(obj.to_o3d_mesh())
            return geoms

        # 其他类型：直接忽略或抛错，你可以按需要改
        raise TypeError(f"Open3DRenderer does not support object type: {type(obj)}")

    # --------- 将 PointCloudObject 转成 open3d 点云 ---------
    def _to_o3d_point_cloud(self, pc: PointCloudObject):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc.points)

        if pc.colors is not None:
            colors = np.asarray(pc.colors, dtype=np.float32)

            if colors.ndim == 1 and colors.shape[0] == 3:
                # 情况 1：你传的是一个单色 RGB，比如 [0.55, 0.10, 0.07]
                pcd.paint_uniform_color(colors.tolist())

            elif colors.ndim == 2 and colors.shape[1] == 3:
                # 情况 2：你传的是 per-point 颜色，shape = (N,3)
                if colors.shape[0] != pc.points.shape[0]:
                    raise ValueError(
                        f"colors 点数 {colors.shape[0]} 和 points 点数 {pc.points.shape[0]} 不一致"
                    )
                pcd.colors = o3d.utility.Vector3dVector(colors)

            else:
                raise ValueError(f"不支持的 colors 形状: {colors.shape}")

        else:
            # 没有颜色就给一个默认颜色
            pcd.paint_uniform_color([0.2, 0.6, 1.0])

        return pcd

    # --------- 将 MeshObject 转成 open3d Mesh ---------
    def _to_o3d_mesh(self, mesh: MeshObject):
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        m.triangles = o3d.utility.Vector3iVector(mesh.faces)
        m.compute_vertex_normals()

        # 如果你以后给 MeshObject 加颜色/纹理，这里可以统一设置
        return m
