import numpy as np
from skimage.measure import marching_cubes

from neural_fusion.geometry.fitting import QuadricFitter
from neural_fusion.geometry.primitives import BoundingSphere
from neural_fusion.geometry.augmentor import GeometricAugmentor
from neural_fusion.segmentation.patch import PatchSegmenter
from neural_fusion.segmentation.refiner import HomogeneityRefiner
from neural_fusion.modeling.blender import SculptingBlender
from utils.geo3d import make_cube_grid
from utils.vis.viewer import Viewer

class PartQuadricVisHook:
    def __init__(self, lim=1.2, res=100):
        self.viewer = Viewer()
        self.fitter = QuadricFitter(regularization=3000)
        self.lim = lim
        self.res = res
        self.raw_points = None
        self.raw_vertices = None
        self.raw_faces = None
    
    def register(self, **kwargs):
        for key, value in kwargs.items():
                    # setattr 允许你动态地设置属性
                    setattr(self, key, value)
                    # print(f"[Debug] Registered self.{key}")    

    def patch_check(
        self,
        *,
        part_pts,
        aug_pts,
        params,
        center,
    ):
        if self.viewer is None:
            return

        raw_stone = BoundingSphere(self.raw_points)
        blender = SculptingBlender(raw_stone)
        blender.upgrade(self.fitter, params, center)

        grid = make_cube_grid(lim=self.lim, res=self.res)
        vals = blender.eval(grid).reshape(self.res, self.res, self.res)

        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not (vmin <= 0.0 <= vmax):
            # 0 等值面不在范围内就跳过
            return

        verts, faces, _, _ = marching_cubes(vals, 0.0)
        verts = verts / (self.res - 1) * (2 * self.lim) - self.lim

        self.viewer.show_objects(items=[
            (part_pts, [1.0, 0.0, 0.0]),
            (aug_pts,  [0.0, 1.0, 0.0]),
            (self.raw_vertices, self.raw_faces),
            (verts, faces, [0.2, 0.2, 0.2]),
        ])
