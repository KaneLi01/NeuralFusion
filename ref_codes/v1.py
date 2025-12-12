import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree, ConvexHull
from scipy.linalg import eig, inv, pinv
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

# Handle optional Open3D dependency
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("(!) Open3D not found. Running in NumPy-only mode.")

# ==========================================
# PART 1: Extended Data Generator (Man-made + Natural)
# ==========================================

class SyntheticShapeNet:
    """
    Generates synthetic point clouds for both Man-Made and Natural objects.
    """
    @staticmethod
    def _create_ellipsoid(n, radii, center):
        pts = np.random.randn(n, 3)
        pts /= np.linalg.norm(pts, axis=1)[:, None] # Unit sphere
        pts *= radii
        pts += center
        return pts

    @staticmethod
    def get_airplane(n=10000):
        parts = []
        labels = []
        # Fuselage
        parts.append(SyntheticShapeNet._create_ellipsoid(int(n*0.4), [0.4, 0.4, 2.5], [0,0,0]))
        labels.append(np.zeros(int(n*0.4)))
        # Wings
        w_l = (np.random.rand(int(n*0.1), 3)-0.5) * [3.0, 0.1, 1.5] + [1.6, 0, 0.2]
        w_r = w_l.copy(); w_r[:,0] *= -1
        parts.extend([w_l, w_r])
        labels.extend([np.ones(len(w_l))*1, np.ones(len(w_r))*2])
        # Tail
        t_v = (np.random.rand(int(n*0.1), 3)-0.5) * [0.1, 1.5, 1.0] + [0, 0.8, -2.0]
        parts.append(t_v); labels.append(np.ones(len(t_v))*3)

        points = np.vstack(parts)
        all_labels = np.concatenate(labels)
        return SyntheticShapeNet._normalize(points), all_labels

    @staticmethod
    def get_chair(n=10000):
        parts = []
        labels = []
        # Seat
        parts.append((np.random.rand(int(n*0.3), 3)-0.5) * [1.0, 1.0, 0.1])
        labels.append(np.zeros(int(n*0.3)))
        # Back
        back = (np.random.rand(int(n*0.3), 3)-0.5) * [1.0, 0.1, 1.2] + [0, -0.45, 0.6]
        parts.append(back); labels.append(np.ones(len(back))*1)
        # Legs
        for i, (x,y) in enumerate([(-0.4,-0.4), (0.4,-0.4), (-0.4,0.4), (0.4,0.4)]):
            leg = (np.random.rand(int(n*0.1), 3)-0.5) * [0.1, 0.1, 1.0] + [x, y, -0.5]
            parts.append(leg); labels.append(np.full(len(leg), 2+i))

        points = np.vstack(parts)
        return SyntheticShapeNet._normalize(points), np.concatenate(labels)

    @staticmethod
    def get_dog(n=10000):
        # Organic Shape: Approximated with metaball-like ellipsoids
        parts = []
        labels = []
        # Body
        parts.append(SyntheticShapeNet._create_ellipsoid(int(n*0.3), [0.5, 1.0, 0.6], [0,0,0]))
        labels.append(np.zeros(int(n*0.3)))
        # Head
        parts.append(SyntheticShapeNet._create_ellipsoid(int(n*0.1), [0.35, 0.35, 0.35], [0, 1.1, 0.4]))
        labels.append(np.ones(int(n*0.1)))
        # Legs (4)
        for i, (x,y) in enumerate([(-0.3, 0.6), (0.3, 0.6), (-0.3, -0.6), (0.3, -0.6)]):
            leg = SyntheticShapeNet._create_ellipsoid(int(n*0.1), [0.12, 0.12, 0.6], [x, y, -0.6])
            parts.append(leg); labels.append(np.full(len(leg), 2+i))
        # Tail
        tail = SyntheticShapeNet._create_ellipsoid(int(n*0.05), [0.08, 0.4, 0.08], [0, -1.1, 0.2])
        parts.append(tail); labels.append(np.full(len(tail), 6))

        points = np.vstack(parts)
        return SyntheticShapeNet._normalize(points), np.concatenate(labels)

    @staticmethod
    def get_horse(n=10000):
        parts = []
        labels = []
        # Body (Larger, longer)
        parts.append(SyntheticShapeNet._create_ellipsoid(int(n*0.3), [0.6, 1.4, 0.7], [0,0,0]))
        labels.append(np.zeros(int(n*0.3)))
        # Neck
        neck = SyntheticShapeNet._create_ellipsoid(int(n*0.1), [0.25, 0.6, 0.3], [0, 1.2, 0.6])
        # Rotate neck approx (simple rotation logic omitted, using position)
        parts.append(neck); labels.append(np.ones(len(neck)))
        # Head
        head = SyntheticShapeNet._create_ellipsoid(int(n*0.05), [0.25, 0.4, 0.25], [0, 1.5, 1.1])
        parts.append(head); labels.append(np.full(len(head), 2))
        # Legs (Longer)
        for i, (x,y) in enumerate([(-0.35, 0.8), (0.35, 0.8), (-0.35, -0.8), (0.35, -0.8)]):
            leg = SyntheticShapeNet._create_ellipsoid(int(n*0.1), [0.12, 0.15, 1.0], [x, y, -0.9])
            parts.append(leg); labels.append(np.full(len(leg), 3+i))

        points = np.vstack(parts)
        return SyntheticShapeNet._normalize(points), np.concatenate(labels)

    @staticmethod
    def get_chicken(n=10000):
        parts = []
        labels = []
        # Body (Rounder)
        parts.append(SyntheticShapeNet._create_ellipsoid(int(n*0.5), [0.5, 0.6, 0.5], [0,0,0]))
        labels.append(np.zeros(int(n*0.5)))
        # Head/Neck
        head = SyntheticShapeNet._create_ellipsoid(int(n*0.15), [0.2, 0.2, 0.3], [0, 0.5, 0.5])
        parts.append(head); labels.append(np.ones(len(head)))
        # Legs (2)
        for i, x in enumerate([-0.2, 0.2]):
            leg = SyntheticShapeNet._create_ellipsoid(int(n*0.1), [0.05, 0.05, 0.4], [x, 0.1, -0.5])
            parts.append(leg); labels.append(np.full(len(leg), 2+i))

        points = np.vstack(parts)
        return SyntheticShapeNet._normalize(points), np.concatenate(labels)

    @staticmethod
    def _normalize(points):
        points -= np.mean(points, axis=0)
        scale = np.max(np.linalg.norm(points, axis=1))
        return points / scale

# ==========================================
# PART 2: ALGORITHMIC CORE (Augment + Fit + Blend)
# ==========================================

class GeometricAugmentor:
    def augment(self, points, thickness=0.05):
        c = np.mean(points, 0); centered = points - c
        cov = centered.T @ centered / len(points)
        val, vec = np.linalg.eigh(cov)
        normal = vec[:, 0] # Normal is eigenvector of smallest eigenvalue
        if np.dot(normal, -c) < 0: normal = -normal # Point inward
        return points + normal * thickness

class QuadricFitter:
    def __init__(self, k=2000.0): self.k = k
    def fit(self, points):
        c = np.mean(points, 0); p = points - c; x,y,z = p.T
        D1 = np.stack([x**2, y**2, z**2, x*y, x*z, y*z], 1)
        D2 = np.stack([x, y, z, np.ones_like(x)], 1)
        try:
            M = D1.T@D1 - D1.T@D2 @ np.linalg.pinv(D2.T@D2) @ D2.T@D1
            C = np.zeros((6,6)); C[0,0]=C[1,1]=C[2,2]=-1; off=(self.k/2)-1; cr=-self.k/4
            C[0,1]=C[1,0]=C[0,2]=C[2,0]=C[1,2]=C[2,1]=off; C[3,3]=C[4,4]=C[5,5]=cr
            val, vec = eig(M, C)
            idx = np.where(val.real > 1e-9)[0]
            v1 = vec[:, idx[np.argmax(val.real[idx])]].real if len(idx)>0 else np.array([1.,1.,1.,0.,0.,0.])
        except: v1 = np.array([1.,1.,1.,0.,0.,0.])
        v2 = -np.linalg.pinv(D2.T@D2) @ D2.T@D1 @ v1
        params = np.concatenate([v1, v2])
        params /= (np.linalg.norm(params) + 1e-9)
        if self.eval(c[None,:], params, c) > 0: params = -params
        # SDF Normalization
        A,B,C,D,E,F,G,H,I,J = params
        dx=2*A*x+D*y+E*z+G; dy=2*B*y+D*x+F*z+H; dz=2*C*z+E*x+F*y+I
        params /= (np.mean(np.sqrt(dx**2+dy**2+dz**2)) + 1e-9)
        return params, c
    def eval(self, pts, params, c):
        p=pts-c; x,y,z=p.T; A,B,C,D,E,F,G,H,I,J=params
        return A*x**2 + B*y**2 + C*z**2 + D*x*y + E*x*z + F*y*z + G*x + H*y + I*z + J

class OrientedBounds:
    @staticmethod
    def fit(pts):
        c = np.mean(pts,0); p = pts-c; cov = p.T@p/len(pts)
        R = np.linalg.eigh(cov)[1]; loc = p@R
        return {'R':R, 'min':np.min(loc,0)-0.01, 'max':np.max(loc,0)+0.01, 'c':c}
    @staticmethod
    def eval(pts, b):
        local = (pts - b['c']) @ b['R']
        d = np.abs(local - (b['min']+b['max'])/2) - (b['max']-b['min'])/2
        return np.minimum(np.maximum(d[:,0], np.maximum(d[:,1], d[:,2])), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=1)

def soft_min(x, y, a):
    a = np.maximum(a, 1e-6)
    h = np.clip(0.5 + 0.5*(y-x)/a, 0.0, 1.0)
    return (1-h)*y + h*x - a*h*(1-h) + a*0.1 # Bias corrected

class ImplicitBlender:
    def __init__(self): self.prims = []
    def add(self, f, p, c, s, b): self.prims.append({'f':f, 'p':p, 'c':c, 's':s, 'b':b})
    def eval(self, pts):
        if not self.prims: return np.zeros(len(pts))
        fields = []
        sigmas = []
        for x in self.prims:
            d = x['f'].eval(pts, x['p'], x['c'])
            b = OrientedBounds.eval(pts, x['b'])
            fields.append(np.maximum(d, b))
            sigmas.append(x['s'])

        curr, curr_s = np.array(fields), np.array(sigmas)
        while len(curr) > 1:
            half = len(curr)//2
            new_f = []
            for i in range(half):
                a = 0.005 #max(curr_s[i], curr_s[half+i])
                new_f.append(soft_min(curr[i], curr[half+i], a))
            if len(curr)%2: new_f.append(curr[-1])
            curr = np.array(new_f)
            curr_s = curr_s[:len(curr)]
        return curr[0]

# ==========================================
# PART 3: SOTA BASELINES & METRICS
# ==========================================

class BaselineMethods:
    @staticmethod
    def run_poisson(points, depth=8):
        if not HAS_OPEN3D: return None, None # Return tuple to avoid unpacking error
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                mesh, dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
            mask = dens < np.quantile(dens, 0.02)
            mesh.remove_vertices_by_mask(mask)
            return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        except: return None, None

    @staticmethod
    def run_bpa(points, radii=[0.02, 0.04, 0.08]):
        if not HAS_OPEN3D: return None, None # Return tuple to avoid unpacking error
        try:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.estimate_normals()
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd, o3d.utility.DoubleVector(radii))
            return np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        except: return None, None

    @staticmethod
    def run_convex_hull(points):
        try:
            hull = ConvexHull(points)
            return points[hull.vertices], None
        except: return None, None

class EvaluationMetrics:
    @staticmethod
    def compute(gt, verts):
        if verts is None or len(verts) == 0: return 999.0, 999.0, 0.0
        stree = cKDTree(gt); ttree = cKDTree(verts)
        d1, _ = ttree.query(gt); d2, _ = stree.query(verts)
        cd = np.mean(d1**2) + np.mean(d2**2)
        hd = max(np.max(d1), np.max(d2))
        prec = np.mean(d2 < 0.05)
        rec = np.mean(d1 < 0.05)
        f1 = 2*prec*rec/(prec+rec+1e-9)
        return cd, hd, f1

class ResultVisualizer:
    @staticmethod
    def show(gt_points, verts, faces, title):
        print(f"[Visualizer] Showing: {title}")
        if HAS_OPEN3D:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gt_points)
                pcd.paint_uniform_color([1, 0, 0])
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.8, 0.8, 0.8])
                mesh.translate([2.0, 0, 0])
                o3d.visualization.draw_geometries([pcd, mesh], window_name=title)
                return
            except: pass

        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        step = max(1, len(gt_points)//500)
        ax1.scatter(gt_points[::step,0], gt_points[::step,1], gt_points[::step,2], c='r', s=1)
        ax1.set_title("Input")
        ax2 = fig.add_subplot(122, projection='3d')
        if len(verts)>0:
            ax2.plot_trisurf(verts[:,0], verts[:,1], verts[:,2], triangles=faces, color='gray', alpha=0.5)
        ax2.set_title("Reconstruction")
        plt.show()

def run_full_benchmark():
    print("--- 3D RECONSTRUCTION BENCHMARK: MAN-MADE vs NATURAL ---")

    datasets = {
        'Airplane': SyntheticShapeNet.get_airplane(),
        'Chair':    SyntheticShapeNet.get_chair(),
        'Dog':      SyntheticShapeNet.get_dog(),
        'Horse':    SyntheticShapeNet.get_horse(),
        'Chicken':  SyntheticShapeNet.get_chicken()
    }

    all_results = []
    augmentor = GeometricAugmentor()
    fitter = QuadricFitter(k=2000.0)

    for cat, (points, labels) in datasets.items():
        print(f"\nProcessing {cat} ({len(points)} pts)...")

        # 1. Run Ours
        blender = ImplicitBlender()
        for l in np.unique(labels):
            patch = points[np.where(labels==l)[0]]
            if len(patch) < 10: continue
            aug = augmentor.augment(patch, 0.04)
            p, c = fitter.fit(np.vstack([patch, aug]))
            b = OrientedBounds.fit(patch)
            d = np.linalg.norm(np.max(patch,0)-np.min(patch,0))
            blender.add(fitter, p, c, max(0.04, d*0.15), b)

        res = 100; lim = 1.5
        x = np.linspace(-lim, lim, res)
        grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
        vals = blender.eval(grid).reshape(res, res, res)
        try:
            v_ours, f_ours, _, _ = marching_cubes(vals, 0.0)
            v_ours = v_ours/(res-1)*(2*lim) - lim
            cd, hd, f1 = EvaluationMetrics.compute(points, v_ours)
            all_results.append({'Category': cat, 'Method': 'Ours', 'CD': cd, 'HD': hd, 'F1': f1})
            ResultVisualizer.show(points, v_ours, f_ours, f"{cat} (Ours)")
        except:
            all_results.append({'Category': cat, 'Method': 'Ours', 'CD': 99, 'HD': 99, 'F1': 0})

        # 2. Run Baselines
        v_p, f_p = BaselineMethods.run_poisson(points)
        if v_p is not None:
            cd, hd, f1 = EvaluationMetrics.compute(points, v_p)
            all_results.append({'Category': cat, 'Method': 'Poisson', 'CD': cd, 'HD': hd, 'F1': f1})
        else:
            all_results.append({'Category': cat, 'Method': 'Poisson', 'CD': 99, 'HD': 99, 'F1': 0})

        v_b, f_b = BaselineMethods.run_bpa(points)
        if v_b is not None:
            cd, hd, f1 = EvaluationMetrics.compute(points, v_b)
            all_results.append({'Category': cat, 'Method': 'BPA', 'CD': cd, 'HD': hd, 'F1': f1})
        else:
             all_results.append({'Category': cat, 'Method': 'BPA', 'CD': 99, 'HD': 99, 'F1': 0})

        v_h, _ = BaselineMethods.run_convex_hull(points)
        if v_h is not None:
            cd, hd, f1 = EvaluationMetrics.compute(points, v_h)
            all_results.append({'Category': cat, 'Method': 'Hull', 'CD': cd, 'HD': hd, 'F1': f1})
        else:
            all_results.append({'Category': cat, 'Method': 'Hull', 'CD': 99, 'HD': 99, 'F1': 0})

    df = pd.DataFrame(all_results)
    print("\n=== BENCHMARK RESULTS ===")
    print("F-Score Comparison (Higher is better):")
    print(df.pivot(index='Category', columns='Method', values='F1'))
    print("\nChamfer Distance (Lower is better):")
    print(df.pivot(index='Category', columns='Method', values='CD'))

if __name__ == "__main__":
    run_full_benchmark()