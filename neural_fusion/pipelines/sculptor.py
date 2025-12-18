import os
import traceback
import numpy as np
from skimage.measure import marching_cubes # 记得安装 scikit-image
import open3d as o3d

from datasets.datasets_loader.shapenetv2 import DatasetIndex, GeometryLoader
from neural_fusion.geometry.fitting import QuadricFitter
from neural_fusion.geometry.primitives import BoundingSphere
from neural_fusion.geometry.augmentor import GeometricAugmentor
from neural_fusion.segmentation.patch import PatchSegmenter
from neural_fusion.segmentation.refiner import HomogeneityRefiner
from neural_fusion.modeling.blender import SculptingBlender, SculptingBlender_
from neural_fusion.metrics.rec_mesh import MeshEvaluator
from neural_fusion.ops.math_utils import OrientedBounds
from utils.vis.viewer import Viewer
from utils.geo3d import make_cube_grid, make_grid_from_bounds, compute_points_cm
from utils.objects import MeshObject

class SculptingPipeline:
    def __init__(self, hooks=None):
        self.debug = Viewer()   # 可以是 None
        self.hooks = hooks or []

    def _dbg(self, rupt=True, **kwargs):
        if self.debug is None:
            return
        # 统一在一个地方决定怎么用这些中间结果
        if "objectives" in kwargs:
            self.debug.show_objects(items=kwargs["objectives"])  
        if rupt:
            raise Exception("debug over")

    def _emit(self, name: str, **kwargs):
        for h in self.hooks:
            fn = getattr(h, name, None)
            if fn is not None:
                fn(**kwargs)

    def _random_color(self):
        color = np.random.rand(3).tolist()
        return color

    @staticmethod
    def get_model_paths(dataset_path):
        """Automatic loader that accepts both:
           1) Single file (.ply / .obj / .stl ...)
        """

        if os.path.isfile(dataset_path):
            if dataset_path.lower().endswith(('.ply', '.obj', '.stl', '.off')):
                print(f"[Loader] Single model detected: {dataset_path}")
                return [dataset_path]
            else:
                raise ValueError(f"File type not supported: {dataset_path}")

        # Case 3: 无效路径
        raise ValueError(f"Invalid dataset_path: {dataset_path}")

    def run(self, cfg, data_path):

        # 1. 解析参数 (使用 .get 设置默认值，防止配置文件漏写报错)
        fitter_k = cfg['quadric_fitter'].get('k', 2000.0)
        n_patches = cfg['patch_segmenter'].get('n_patches', 30)
        err_tol = cfg['refiner'].get('error_tolerance', 0.005)
        aug_thick = cfg['augmentor'].get('thickness', 0.04)

        # 2. 初始化组件
        data_loader = GeometryLoader(samples=5000)
        fitter = QuadricFitter(regularization=fitter_k)
        segmenter = PatchSegmenter(n_patches=n_patches) 
        refiner = HomogeneityRefiner(fitter=fitter)
        augmentor = GeometricAugmentor()
        model_paths = SculptingPipeline.get_model_paths(data_path)
        viewer = Viewer()
        mesh_eval = MeshEvaluator()
        print(f"--- Running sculpting on {len(model_paths)} model(s) ---")

        print(f"--- SCULPTOR PIPELINE (Config Loaded) ---")
        print(f"    K={fitter_k}, Patches={n_patches}, Tol={err_tol}")

        # 源代码
        if os.name == 'nt' and not data_path.startswith('\\\\?\\'):
            data_path = '\\\\?\\' + os.path.abspath(data_path)

        print(f"--- SCULPTOR PIPELINE (Carving Stone) ---")

        for i, fpath in enumerate(model_paths):
            fname = os.path.basename(fpath)
            print(f"\n[{i+1}] Sculpting: {fname}")
            
            # 1. Load Data
            points, vertices, faces = data_loader.load(fpath, return_points=True, return_faces=True)
            self._emit("register", raw_points=points, raw_vertices=vertices, raw_faces=faces)
            if points is None: continue

            # 2. Define Raw Stone (Bounding Sphere) [Step 1 of Prompt]
            raw_stone = BoundingSphere(points)
            print(f"    -> Raw Stone Prepared (Radius: {raw_stone.radius:.2f})")

            # 3. Segment & Refine (Dividing Surface) [Step 2]
            labels_initial = segmenter.segment(points)  # 利用kmeans得到点云分类的标签
            labels = refiner.refine(points, labels_initial, error_tolerance=0.005)  # 合并patch
            unique_parts = np.unique(labels)
            print(f"    -> Surface divided into {len(unique_parts)} working regions.")

            # 4. Augment & Fit (The Chisel) [Step 3 & 4]
            # We initialize the blender with the Raw Stone
            blender = SculptingBlender(raw_stone)
            blender_ = SculptingBlender_()
            
            for part_id in unique_parts:
                part_pts = points[labels == part_id]
                if len(part_pts) < 10: continue
                
                # Augment the region with surface points (Step 3 of prompt)
                # Instead of adding sphere points (which causes artifacts),
                # we augment locally to ensure the quadric fits the surface patch tightly.
                aug = augmentor.augment(part_pts, 0.04)
                patch_bbox = OrientedBounds.fit(part_pts)
                # Fit Ellipsoid (Step 4)
                p, c = fitter.fit(np.vstack([part_pts, aug]))
                # p, c = fitter.fit(part_pts)  # 没有augment

                # 可视化中间结果
                # self._emit("patch_check", part_pts=part_pts, aug_pts=aug, params=p, center=c)

                # Add to sculpting tool
                blender.add(fitter, p, c)
                blender_.add(fitter, p, c, patch_bbox)

            # 5. Carve (Intersection of Stone and Fits) [Step 5]
            # This happens inside blender.eval()
            
            # 6. Meshing
            res = 100; lim = 1.2
            grid = make_cube_grid(lim=lim, res=res)
            vals = blender.eval(grid).reshape(res, res, res)
            vals_ = blender_.eval(grid).reshape(res, res, res)
            
            try:
                verts, faces, _, _ = marching_cubes(vals, 0.0)
                verts = verts / (res-1) * (2*lim) - lim              
                print(f"    -> Sculpture Finished: {len(verts)} vertices.")
                result_mesh = MeshObject(verts, faces).to_o3d_mesh()
                result_metric = mesh_eval.eval_mesh(mesh=result_mesh, pointcloud_tgt=points)
                viewer.show_point_mesh(points=points, vertices=verts, faces=faces, colors=None, normals=None)
                
                verts, faces, _, _ = marching_cubes(vals_, 0.0)
                verts = verts / (res-1) * (2*lim) - lim              
                viewer.show_point_mesh(points=points, vertices=verts, faces=faces, colors=None, normals=None)



                return result_metric                 

                
            except Exception as e:
                print(f"    -> Carving failed (Mesh error): {e}")
                traceback.print_exc()
                try:
                    # Fallback if iso-surface 0 is missed
                    verts, faces, _, _ = marching_cubes(vals, vals.min() + 0.1)
                    verts = verts / (res-1) * (2*lim) - lim
                except: pass