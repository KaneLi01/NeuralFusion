# data_loader.py
import os
import numpy as np

# Try Open3D
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("(!) Warning: Open3D not installed, mesh sampling and visualization disabled.")


# =========================================================
# 1. 目录扫描类：只负责“知道有哪些样本”
# =========================================================
class DatasetIndex:
    """
    负责扫描数据集总目录，记录每个样本的信息：
    root/
      category_1/
        inst_a/
          models/*.ply
          screenshots/*.png
      category_2/
        ...

    用法：
        index = DatasetIndex(root_dir)
        len(index)              -> 样本数量
        entry = index[i]        -> {'category', 'instance', 'model_path', 'image_paths'}
    """

    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Root directory does not exist: {self.root_dir}")

        self.entries = self._scan_root()
        print(f"[DatasetIndex] Scanned root: {self.root_dir}")
        print(f"[DatasetIndex] Found {len(self.entries)} samples.")

    def _scan_root(self):
        entries = []
        model_exts = (".ply", ".obj", ".off", ".stl")

        root_dir = self.root_dir

        # 遍历一级目录（category）
        for cat_name in sorted(os.listdir(root_dir)):
            cat_dir = os.path.join(root_dir, cat_name)
            if not os.path.isdir(cat_dir):
                continue

            # 遍历二级目录（instance）
            for inst_name in sorted(os.listdir(cat_dir)):
                inst_dir = os.path.join(cat_dir, inst_name)
                if not os.path.isdir(inst_dir):
                    continue

                models_dir = os.path.join(inst_dir, "models")
                screenshots_dir = os.path.join(inst_dir, "screenshots")

                # 寻找一个模型文件
                model_path = None
                if os.path.isdir(models_dir):
                    for fname in sorted(os.listdir(models_dir)):
                        if fname.lower().endswith(model_exts):
                            model_path = os.path.join(models_dir, fname)
                            break

                if model_path is None:
                    continue  # 没模型就跳过

                # 收集所有 png 截图路径
                image_paths = []
                if os.path.isdir(screenshots_dir):
                    for fname in sorted(os.listdir(screenshots_dir)):
                        if fname.lower().endswith(".png"):
                            image_paths.append(os.path.join(screenshots_dir, fname))

                entries.append(
                    {
                        "category": cat_name,
                        "instance": inst_name,
                        "model_path": model_path,
                        "image_paths": image_paths,
                    }
                )

        return entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]


# =========================================================
# 2. 点云加载类：只负责“根据文件路径读点云”
# =========================================================
class GeometryLoader:
    """
    负责根据单个模型路径加载几何数据（默认 Open3D 可用）：
    - 支持 obj/ply/off/stl 等 Open3D 支持的格式
    - 可返回：
        1) 点云 points (N, 3)
        2) 网格 faces (M, 3) 以及 vertices (V, 3)
        3) 二者都返回（点云 + 网格）
    - 最后可选做中心化 + 单位球缩放（对 points/vertices 一致应用）
    """

    def __init__(self, samples: int = 5000, normalize: bool = True):
        """
        samples: 当输入是 mesh 且需要点云时，采样点数（Poisson disk）
        normalize: 是否做中心化 + 单位球缩放
        """
        self.samples = samples
        self.do_normalize = normalize

    def load(
        self,
        path: str,
        return_points: bool = True,
        return_faces: bool = False,
        sample_if_mesh: bool = True,
    ):
        """
        加载指定路径的模型数据。

        参数:
        - return_points: 是否返回点云 points (N,3)
        - return_faces: 是否返回网格 faces (M,3) 以及 vertices (V,3)
        - sample_if_mesh: 若点云读取为空且能读到 mesh，是否从 mesh 采样点云

        返回:
        - 只要 points: points
        - 只要 faces: (vertices, faces)
        - 两者都要: (points, vertices, faces)
        - 失败: None
        """
        if not (return_points or return_faces):
            # 没有任何输出需求
            return None

        points = None
        vertices = None
        faces = None

        try:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            # 1) 尝试读取点云
            if return_points:
                pcd = o3d.io.read_point_cloud(path)
                if len(pcd.points) > 0:
                    points = np.asarray(pcd.points, dtype=np.float32)

            # 2) 若需要 faces 或 点云为空但允许从 mesh 采样，则读取 mesh
            need_mesh = return_faces or (return_points and points is None and sample_if_mesh)
            mesh = None
            if need_mesh:
                mesh = o3d.io.read_triangle_mesh(path)
                if mesh is not None and len(mesh.vertices) > 0:
                    vertices = np.asarray(mesh.vertices, dtype=np.float32)
                    faces = np.asarray(mesh.triangles, dtype=np.int64)  # (M,3) 三角面

                    # 从 mesh 采样点云（仅当点云还没读到）
                    if return_points and points is None and sample_if_mesh:
                        pcd2 = mesh.sample_points_poisson_disk(self.samples)
                        if len(pcd2.points) > 0:
                            points = np.asarray(pcd2.points, dtype=np.float32)

            # 3) 若读取失败
            if (return_points and points is None) and (not return_faces):
                return None
            if return_faces and (vertices is None or faces is None or len(faces) == 0):
                # 注意：某些文件可能只有点云没有 mesh
                return None

            # 4) 归一化（让输出坐标在同一规范下）
            if self.do_normalize:
                points, vertices = self._normalize_consistently(points, vertices)

            # 5) 按需求返回
            if return_points and return_faces:
                return points, vertices, faces
            elif return_points:
                return points
            else:
                return vertices, faces

        except Exception:
            return None

    # ----------------- 内部方法 -----------------

    def _normalize_consistently(self, points: np.ndarray | None, vertices: np.ndarray | None):
        """
        对 points / vertices 做一致的中心化 + 单位球缩放：
        - 如果有 vertices：用 vertices 估计中心与尺度（mesh 更“完整”）
        - 否则用 points
        """
        ref = None
        if vertices is not None and len(vertices) > 10:
            ref = vertices
        elif points is not None and len(points) > 10:
            ref = points
        else:
            return points, vertices

        ref = ref.astype(np.float32)
        c = np.mean(ref, axis=0)
        ref_centered = ref - c
        scale = float(np.max(np.linalg.norm(ref_centered, axis=1)))
        if scale <= 0:
            return points, vertices

        def apply(arr):
            if arr is None:
                return None
            arr = arr.astype(np.float32)
            arr = (arr - c) / scale
            return arr

        return apply(points), apply(vertices)
