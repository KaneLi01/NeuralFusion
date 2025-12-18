import numpy as np

class SDFBlender:
    """
    SDF 平滑混合策略管理器。
    提供两种模式：
    1. 'iq' (Inigo Quilez): 计算极快，C1 连续。适合实时预览。
    2. 'c2' (Curvature Continuous): 计算较重，C2 连续。适合高精度网格提取和物理模拟。
    """
    
    # 当前默认策略
    _strategy = 'iq'

    @classmethod
    def set_strategy(cls, mode: str):
        if mode not in ['iq', 'c2']:
            raise ValueError("Mode must be 'iq' or 'c2'")
        cls._strategy = mode
        print(f"[SDFBlender] Switched to strategy: {mode.upper()}")

    @staticmethod
    def union(d1, d2, k=0.05):
        """ 平滑并集 (Smooth Union) -> 用于融合 """
        if SDFBlender._strategy == 'iq':
            # Inigo Quilez method (Fast, C1)
            h = np.maximum(k - np.abs(d1 - d2), 0.0) / k
            return np.minimum(d1, d2) - h * h * k * 0.25
        else:
            # C2 Continuous method (High Quality)
            return 0.5 * (d1 + d2 - SDFBlender._soft_abs_c2(d1 - d2, k))

    @staticmethod
    def intersection(d1, d2, k=0.05):
        """ 平滑交集 (Smooth Intersection) -> 用于切削/布尔运算 """
        if SDFBlender._strategy == 'iq':
            # Inigo Quilez method (Fast, C1)
            # Intersection(A, B) = -Union(-A, -B)
            h = np.maximum(k - np.abs(d1 - d2), 0.0) / k
            return np.maximum(d1, d2) + h * h * k * 0.25
        else:
            # C2 Continuous method (High Quality)
            # max(a, b) = 0.5 * (a + b + |a - b|)
            return 0.5 * (d1 + d2 + SDFBlender._soft_abs_c2(d1 - d2, k))

    @staticmethod
    def difference(d1, d2, k=0.05):
        """ 平滑差集 (Smooth Difference) -> d1 减去 d2 """
        # Difference(A, B) = Intersection(A, -B)
        return SDFBlender.intersection(d1, -d2, k)

    @staticmethod
    def _soft_abs_c2(x, k):
        """ 内部辅助函数：C2 连续的绝对值近似 """
        k = np.maximum(k, 1e-6)
        xx = 2.0 * x / k
        abs_xx = np.abs(xx)
        
        # 当 |x| < k/2 时使用多项式平滑过渡，否则使用标准 abs
        # 这种写法利用 np.where 避免了分支预测失败，保持向量化性能
        res = np.where(
            abs_xx < 2.0,
            0.5 * xx**2 * (1.0 - abs_xx / 6.0) + 2.0 / 3.0,
            abs_xx
        )
        return res * k / 2.0

class OrientedBounds:
    """
    Utility class to fit and evaluate Oriented Bounding Boxes (OBB).
    Uses PCA (Principal Component Analysis) to determine the box orientation.
    """

    @staticmethod
    def fit(points: np.ndarray) -> dict:
        """
        Computes the tightest axis-aligned bounding box in the PCA frame.
        
        Args:
            points: (N, 3) input point cloud.
            
        Returns:
            dict: Parameters of the OBB {'c': center, 'R': rotation, 'min': vec3, 'max': vec3}
        """
        # 1. Centering the data
        center = np.mean(points, axis=0)
        centered_pts = points - center

        # 2. PCA Analysis (Covariance Matrix)
        # Compute the covariance matrix to find principal axes
        covariance = centered_pts.T @ centered_pts / len(points)
        
        # Eigen decomposition:
        # Eigenvectors (columns of R) represent the rotation matrix (principal axes)
        _, eigenvectors = np.linalg.eigh(covariance)
        rotation_matrix = eigenvectors

        # 3. Project points to local PCA coordinate system
        # shape: (N, 3)
        local_pts = centered_pts @ rotation_matrix

        # 4. Find extents in the local frame
        min_bound = np.min(local_pts, axis=0)
        max_bound = np.max(local_pts, axis=0)

        return {
            'c': center,            # World space center of mass
            'R': rotation_matrix,   # Rotation from Local to World
            'min': min_bound,       # Local min corner
            'max': max_bound        # Local max corner
        }

    @staticmethod
    def eval(query_pts: np.ndarray, box_params: dict) -> np.ndarray:
        """
        Evaluates the Signed Distance Function (SDF) of the OBB.
        
        Args:
            query_pts: (N, 3) query points in world space.
            box_params: dict returned by `fit`.
            
        Returns:
            (N,) array of signed distances. Negative = inside, Positive = outside.
        """
        c = box_params['c']
        R = box_params['R']
        b_min = box_params['min']
        b_max = box_params['max']

        # 1. Transform World -> Local
        # (p - c) @ R transforms points into the box's aligned local space
        local_pts = (query_pts - c) @ R

        # 2. Calculate Box Geometry
        # Center of the box in local space (usually close to 0,0,0 but not always)
        local_center = (b_min + b_max) / 2.0
        # Half-size (extents) of the box
        half_size = (b_max - b_min) / 2.0

        # 3. Calculate Distance Vector 'd'
        # d is the distance from the query point to the box surface in each axis
        # abs(p - center) - extent
        d = np.abs(local_pts - local_center) - half_size

        # 4. Combine into SDF (Inigo Quilez's Box SDF formula)
        # Part A: Outside distance (Euclidean distance to the nearest corner/edge)
        # Clamp d to 0 (ignore negative components), then take length
        outside_dist = np.linalg.norm(np.maximum(d, 0.0), axis=1)

        # Part B: Inside distance (Negative distance to the nearest face)
        # Take the max component (closest face), clamp to 0 (ignore positive components)
        inside_dist = np.minimum(np.maximum(d[:, 0], np.maximum(d[:, 1], d[:, 2])), 0.0)

        return outside_dist + inside_dist