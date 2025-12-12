import numpy as np

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