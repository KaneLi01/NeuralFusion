import os
import imageio.v2 as imageio
import numpy as np
import datasets.datasets_loader.shapenetv2 as snv2

from .objects import PointCloudObject, MeshObject, ImageObject
from .open3d_renderer import Open3DRenderer

class Viewer:
    """
    对外的统一接口：
    - from_file(...)
    - from_data(...)
    内部根据类型选择合适的 renderer。
    """

    def __init__(self, pc_backend="open3d", mesh_backend="open3d"):
        self.pc_backend = pc_backend
        self.mesh_backend = mesh_backend

        self.renderer = Open3DRenderer()

    # ======== from file ========
    def show_point_cloud_file(self, path):
        
        loader = snv2.PointCloudLoader()
        points = loader.load(path)
        pc_obj = PointCloudObject(points)
        self.renderer.render(pc_obj, window_name=os.path.basename(path))

    def show_mesh_file(self, vertices, faces):
        mesh_obj = MeshObject(vertices, faces)
        self.renderer.render(mesh_obj)

    # ======== from data ========
    def show_point_cloud(self, points, colors=None, normals=None):
        pc_obj = PointCloudObject(points, colors, normals)
        self.renderer.render(pc_obj)

    def show_mesh(self, vertices, faces, colors=None):
        mesh_obj = MeshObject(vertices, faces, colors)
        self.renderer.render(mesh_obj)

    def show_point_mesh(self, points, vertices, faces, colors=None, normals=None):
        pc_obj = PointCloudObject(points, colors, normals)
        mesh_obj = MeshObject(vertices, faces, colors)
        self.renderer.render(obj=[pc_obj, mesh_obj])

    # 通用接口
    def show_objects(self, items):
        """
        items: 列表，每个元素可以是：
            1) PointCloudObject / MeshObject       （已经包装好的）
            2) np.ndarray                          （认为是点云：points）
            3) (points,)                           （点云）
            4) (points, colors)                    （点云 + 颜色）
            5) (points, colors, normals)           （点云 + 颜色 + 法线）
            6) (vertices, faces)                   （mesh）
            7) (vertices, faces, colors)           （带顶点颜色的 mesh）

        最终会统一转成 [VisObject, VisObject, ...] 传给 renderer.render(...)
        """
        if not isinstance(items, (list, tuple)):
            items = [items]

        vis_objs = []

        for item in items:
            # 已经是 VisObject，直接用
            if isinstance(item, (PointCloudObject, MeshObject)):
                vis_objs.append(item)
                continue

            # 纯点云 array
            if isinstance(item, np.ndarray):
                vis_objs.append(PointCloudObject(item))
                continue

            # tuple/list：根据长度和类型猜测是点云还是 mesh
            if isinstance(item, (list, tuple)):
                if len(item) == 1:
                    # (points,)
                    pts = item[0]
                    vis_objs.append(PointCloudObject(pts))

                elif len(item) == 2:
                    a, b = item
                    # 判断是 (points, colors) 还是 (vertices, faces)
                    a = np.asarray(a)
                    b = np.asarray(b)

                    # 简单判断：faces 通常是 int，colors 通常是 float
                    if b.dtype.kind in ('i', 'u'):   # int / uint → 认为是 faces
                        verts, faces = a, b
                        vis_objs.append(MeshObject(verts, faces))
                    else:
                        points, colors = a, b
                        vis_objs.append(PointCloudObject(points, colors))

                elif len(item) == 3:
                    a, b, c = item
                    a = np.asarray(a)
                    b = np.asarray(b)

                    if b.dtype.kind in ('i', 'u'):
                        # (vertices, faces, colors)
                        verts, faces, colors = a, b, c
                        vis_objs.append(MeshObject(verts, faces, colors))
                    else:
                        # (points, colors, normals)
                        points, colors, normals = a, b, c
                        vis_objs.append(PointCloudObject(points, colors, normals))

                else:
                    raise ValueError(f"Unsupported item tuple length: {len(item)}")

            else:
                raise TypeError(f"Unsupported item type: {type(item)}")

        # 统一丢给 renderer（之前你已经支持 list 了）
        self.renderer.render(vis_objs)