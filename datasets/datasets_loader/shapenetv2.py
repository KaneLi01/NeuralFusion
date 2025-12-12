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
class PointCloudLoader:
    """
    负责根据单个模型路径加载点云：
    - 支持 obj/ply/off/stl
    - 优先用 Open3D 加载点云或网格并采样
    - 无 Open3D 时，回退到手动解析 OBJ 顶点
    - 最后做中心化 + 单位球缩放
    """

    def __init__(self, samples=5000):
        """
        samples: 用 mesh 采样点云时的点数（仅在有 Open3D 且输入是 mesh 时使用）
        """
        self.samples = samples

    def load(self, path: str):
        """
        加载指定路径的模型，返回归一化后的点云 (N, 3) 或 None
        """
        points = None

        # 1. 有 Open3D，则优先用 Open3D
        if HAS_OPEN3D:
            points = self._load_with_open3d(path)

        # 2. 如果 Open3D 失败 / 不可用，则尝试 OBJ fallback
        if points is None:
            points = self._load_obj_fallback(path)

        # 3. 归一化
        if points is not None and len(points) > 10:
            return self._normalize(points)

        return None

    # ----------------- 内部方法 -----------------

    def _load_with_open3d(self, path):
        try:
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

            # 先尝试读取为点云
            pcd = o3d.io.read_point_cloud(path)

            # 如果点云为空，再尝试读 mesh，并采样出点云
            if len(pcd.points) == 0:
                mesh = o3d.io.read_triangle_mesh(path)
                if len(mesh.vertices) > 0:
                    pcd = mesh.sample_points_poisson_disk(self.samples)
                else:
                    return None

            points = np.asarray(pcd.points)
            return points

        except Exception:
            return None

    def _load_obj_fallback(self, path):
        """
        无 Open3D 或 Open3D 加载失败时，简单解析 .obj 的 'v x y z'
        """
        if not path.lower().endswith(".obj"):
            return None

        verts = []
        try:
            with open(path, "r", encoding="latin-1") as f:
                for line in f:
                    if line.startswith("v "):
                        parts = line.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])

            if len(verts) == 0:
                return None

            return np.array(verts, dtype=np.float32)

        except Exception:
            return None

    def _normalize(self, points: np.ndarray):
        """
        中心化 + 缩放到单位球
        """
        points = points.astype(np.float32)
        c = np.mean(points, axis=0)
        points = points - c
        scale = np.max(np.linalg.norm(points, axis=1))
        if scale > 0:
            points = points / scale
        return points


# =========================================================
# 3. 小测试（可选）
# =========================================================
if __name__ == "__main__":
    # 例 1：测试目录扫描
    root = "/path/to/your/dataset_root"
    if os.path.isdir(root):
        index = DatasetIndex(root)
        print("Total samples:", len(index))
        if len(index) > 0:
            e0 = index[0]
            print("Sample[0]:", e0)

            # 例 2：用 PointCloudLoader 加载第 0 个样本的点云
            loader = PointCloudLoader(samples=8000)
            pts = loader.load(e0["model_path"])
            print("Points shape:", None if pts is None else pts.shape)
    else:
        print("请先把 root 换成真实的数据集根目录")
