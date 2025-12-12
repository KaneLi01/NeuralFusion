import numpy as np

class GeometricAugmentor:
    def augment(self, points, thickness=0.05):
        '''将点云沿整体法向平移一点'''
        c = np.mean(points, 0); centered = points - c
        cov = centered.T @ centered / len(points)
        val, vec = np.linalg.eigh(cov)
        normal = vec[:, 0]
        if np.dot(normal, -c) < 0: normal = -normal 
        return points + normal * thickness