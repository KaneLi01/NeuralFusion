import numpy as np
from skimage.measure import marching_cubes

from utils.geo3d import make_cube_grid
from utils.vis.viewer import Viewer

class PartQuadricVisHook:
    def __init__(self, lim=1.2, res=100, max_parts=None):
        self.viewer = Viewer()
        self.lim = lim
        self.res = res
        self.max_parts = max_parts

    def on_part_fitted(
        self,
        *,
        part_id,
        part_pts,
        aug_pts,
        vertices,
        faces,
        fitter,
        params,
        center,
        raw_stone,
        blender_cls,
        model_name=None,
    ):
        if self.viewer is None:
            return
        if self.max_parts is not None and part_id >= self.max_parts:
            return

        # 只为当前 part 构造一个临时 blender（你的 blender0）
        blender0 = blender_cls(raw_stone)
        blender0.upgrade(fitter, params, center)

        grid = make_cube_grid(lim=self.lim, res=self.res)
        vals = blender0.eval(grid).reshape(self.res, self.res, self.res)

        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if not (vmin <= 0.0 <= vmax):
            # 0 等值面不在范围内就跳过（避免你之前的报错）
            return

        verts0, faces0, _, _ = marching_cubes(vals, 0.0)
        verts0 = verts0 / (self.res - 1) * (2 * self.lim) - self.lim

        # 统一通过 viewer 展示（相当于你现在的 self._dbg）
        self.viewer.show_objects(items=[
            (part_pts, [1.0, 0.0, 0.0]),
            (aug_pts,  [0.0, 1.0, 0.0]),
            (vertices, faces),
            (verts0, faces0, [0.2, 0.2, 0.2]),
        ])
