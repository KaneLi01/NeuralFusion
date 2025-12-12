import os
import traceback
import numpy as np
from skimage.measure import marching_cubes # 记得安装 scikit-image
import open3d as o3d

from datasets.datasets_loader.shapenetv2 import DatasetIndex, PointCloudLoader
from neural_fusion.geometry.fitting import QuadricFitter
from neural_fusion.geometry.primitives import BoundingSphere
from neural_fusion.geometry.augmentor import GeometricAugmentor
from neural_fusion.segmentation.patch import PatchSegmenter
from neural_fusion.segmentation.refiner import HomogeneityRefiner
from neural_fusion.modeling.blender import SculptingBlender
from utils.vis.viewer import Viewer

class SculptingPipeline:
    def __init__(self):
        self.debug = Viewer()   # 可以是 None

    def _dbg(self, **kwargs):
        if self.debug is None:
            return
        # 统一在一个地方决定怎么用这些中间结果
        if "objectives" in kwargs:
            self.debug.show_objects(items=kwargs["objectives"])  
        raise Exception("debug over")
    
    def _random_color(self):
        pcd = o3d.geometry.PointCloud()
        color = np.random.rand(3).tolist()
        # return pcd.paint_uniform_color(color)
        return color

    @staticmethod
    def get_model_paths(dataset_path):
        """Automatic loader that accepts both:
           1) Single file (.ply / .obj / .stl ...)
           2) Dataset root directory (ShapeNet style)
        """
        # Case 1: 单模型路径
        if os.path.isfile(dataset_path):
            if dataset_path.lower().endswith(('.ply', '.obj', '.stl', '.off')):
                print(f"[Loader] Single model detected: {dataset_path}")
                return [dataset_path]
            else:
                raise ValueError(f"File type not supported: {dataset_path}")

        # Case 2: 根目录 → 使用 DatasetIndex 扫描
        if os.path.isdir(dataset_path):
            print(f"[Loader] Dataset directory detected, scanning with DatasetIndex...")
            index = DatasetIndex(dataset_path)
            model_paths = [entry["model_path"] for entry in index.entries]
            print(f"[Loader] Found {len(model_paths)} models.")
            return model_paths

        # Case 3: 无效路径
        raise ValueError(f"Invalid dataset_path: {dataset_path}")

    def run(self, cfg):

        # 1. 解析参数 (使用 .get 设置默认值，防止配置文件漏写报错)
        fitter_k = cfg['quadric_fitter'].get('k', 2000.0)
        n_patches = cfg['patch_segmenter'].get('n_patches', 30)
        err_tol = cfg['refiner'].get('error_tolerance', 0.005)
        aug_thick = cfg['augmentor'].get('thickness', 0.04)
        dataset_path = cfg['dataset_path']

        # 2. 初始化组件
        data_loader = PointCloudLoader(samples=5000)
        fitter = QuadricFitter(regularization=fitter_k)
        segmenter = PatchSegmenter(n_patches=n_patches) 
        refiner = HomogeneityRefiner(fitter=fitter)
        augmentor = GeometricAugmentor()
        model_paths = SculptingPipeline.get_model_paths(dataset_path)
        viewer = Viewer()
        print(f"--- Running sculpting on {len(model_paths)} model(s) ---")

        print(f"--- SCULPTOR PIPELINE (Config Loaded) ---")
        print(f"    K={fitter_k}, Patches={n_patches}, Tol={err_tol}")

        # 源代码
        if os.name == 'nt' and not dataset_path.startswith('\\\\?\\'):
            dataset_path = '\\\\?\\' + os.path.abspath(dataset_path)

        print(f"--- SCULPTOR PIPELINE (Carving Stone) ---")

        for i, fpath in enumerate(model_paths):
            fname = os.path.basename(fpath)
            print(f"\n[{i+1}] Sculpting: {fname}")
            
            # 1. Load Data
            points = data_loader.load(fpath)
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
            
            _dbg_list = []
            for part_id in unique_parts:
                part_pts = points[labels == part_id]
                _dbg_list.append((part_pts, self._random_color()))
                if len(part_pts) < 10: continue
                
                # Augment the region with surface points (Step 3 of prompt)
                # Instead of adding sphere points (which causes artifacts),
                # we augment locally to ensure the quadric fits the surface patch tightly.
                aug = augmentor.augment(part_pts, 0.04)
                
                # Fit Ellipsoid (Step 4)
                p, c = fitter.fit(np.vstack([part_pts, aug]))
                # p, c = fitter.fit(part_pts)  # 没有augment
                
                # Add to sculpting tool
                blender.add(fitter, p, c)
            # self._dbg(objectives=_dbg_list)
            # 5. Carve (Intersection of Stone and Fits) [Step 5]
            # This happens inside blender.eval()
            
            # 6. Meshing
            res = 100; lim = 1.2
            x = np.linspace(-lim, lim, res)
            grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
            vals = blender.eval(grid).reshape(res, res, res)
            
            try:
                verts, faces, _, _ = marching_cubes(vals, 0.0)
                verts = verts / (res-1) * (2*lim) - lim
                
                print(f"    -> Sculpture Finished: {len(verts)} vertices.")
                viewer.show_point_mesh(points=points, vertices=verts, faces=faces, colors=None, normals=None)

                
            except Exception as e:
                print(f"    -> Carving failed (Mesh error): {e}")
                traceback.print_exc()
                try:
                    # Fallback if iso-surface 0 is missed
                    verts, faces, _, _ = marching_cubes(vals, vals.min() + 0.1)
                    verts = verts / (res-1) * (2*lim) - lim
                    # SculptingPipeline.visualize_overlay(points, verts, faces, f"Sculpture (Fallback): {fname}")
                except: pass