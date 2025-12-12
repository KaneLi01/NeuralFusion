import numpy as np

class BoundingSphere:
    """ The Raw Stone Block """
    def __init__(self, points):
        self.center = np.mean(points, axis=0)
        self.radius = np.max(np.linalg.norm(points - self.center, axis=1)) + 0.05 # padding
        
    def eval(self, pts):
        # SDF of a sphere: dist - radius
        # Negative inside, Positive outside
        dists = np.linalg.norm(pts - self.center, axis=1)
        return dists - self.radius