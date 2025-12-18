import numpy as np

class NaturalFeatureSegmenter:
    """
    基于几何特征（法线一致性）的区域生长分割。
    适用于：提取平滑表面（如墙面、桌面、规则几何体）。
    """
    def __init__(self, similarity_thresh=0.85, min_cluster_size=50):
        """
        :param similarity_thresh: 法线点积阈值 (cos theta)，越接近 1 要求越平。
        :param min_cluster_size: 最小簇大小，小于此数量的簇会被标记为噪声 (-1)。
        """
        self.t = similarity_thresh
        self.s = min_cluster_size

    def segment(self, points):
        n_points = len(points)
        if n_points < self.s:
            return np.full(n_points, -1, dtype=int)

        # 1. 估计法线 (Estimate Normals)
        # 使用 KDTree 找最近邻 (k=10)
        tree = cKDTree(points)
        _, indices = tree.query(points, k=10)
        
        normals = np.zeros((n_points, 3))
        for i in range(n_points):
            # PCA: 对局部邻域做协方差矩阵特征分解
            # 特征值最小对应的特征向量即为法线
            local_patch = points[indices[i]]
            cov = np.cov(local_patch, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            normals[i] = eigenvectors[:, 0] # 假设最小特征值在第一个 (取决于具体实现，通常eigh是升序)

        # 2. 区域生长 (Region Growing)
        labels = np.full(n_points, -1, dtype=int)
        visited = np.zeros(n_points, dtype=bool)
        cluster_id = 0

        for i in range(n_points):
            if visited[i]:
                continue

            # 开始一个新的簇
            queue = [i]
            visited[i] = True
            labels[i] = cluster_id
            count = 0
            
            # 取当前种子的法线作为参考
            current_normal = normals[i]

            while queue:
                u = queue.pop(0)
                count += 1
                
                # 检查邻居
                for v in indices[u]:
                    if not visited[v]:
                        # 检查法线夹角是否足够小 (点积 > 阈值)
                        if np.abs(np.dot(current_normal, normals[v])) > self.t:
                            visited[v] = True
                            labels[v] = cluster_id
                            queue.append(v)
            
            # 3. 过滤过小的簇
            if count >= self.s:
                cluster_id += 1
            else:
                # 回退：将该簇标记回噪声 (-1)
                labels[labels == cluster_id] = -1

        return labels


class PatchSegmenter:
    """ Divides the surface into working patches (Step 2 of prompt) """
    def __init__(self, n_patches=32): self.k = n_patches
    
    def segment(self, points):
        return self._numpy_kmeans(points, self.k)

    def _numpy_kmeans(self, X, k, max_iter=15):
        '''numpy 的 kmeans实现'''
        n_points = len(X)
        if n_points < k: return np.zeros(n_points)

        indices = np.random.choice(n_points, k, replace=False)
        centroids = X[indices]
        labels = np.zeros(n_points, dtype=int)
        for _ in range(max_iter):
            dists = np.linalg.norm(X[:, None] - centroids[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.all(labels == new_labels): break
            labels = new_labels
            for i in range(k):
                mask = (labels == i)
                if np.any(mask): centroids[i] = X[mask].mean(axis=0)
        return labels