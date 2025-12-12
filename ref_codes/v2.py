import numpy as np
import os
import warnings
from scipy.spatial import cKDTree
from scipy.linalg import eig, svd
from skimage.measure import marching_cubes
import time

# Suppress warnings
warnings.filterwarnings("ignore")

# Open3D Check
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    print("(!) CRITICAL: Open3D needed for interactive visualization.")

# ==========================================
# PART 1: CORE ALGORITHMS (Primitives & Blending)
# ==========================================

class GeometricAugmentor:
    def augment(self, points, thickness=0.05):
        c = np.mean(points, 0); centered = points - c
        cov = centered.T @ centered / len(points)
        val, vec = np.linalg.eigh(cov)
        normal = vec[:, 0]
        if np.dot(normal, -c) < 0: normal = -normal 
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
        return {'R':R, 'min':np.min(loc,0)-0.05, 'max':np.max(loc,0)+0.05, 'c':c}
    @staticmethod
    def eval(pts, b):
        local = (pts - b['c']) @ b['R']
        d = np.abs(local - (b['min']+b['max'])/2) - (b['max']-b['min'])/2
        return np.minimum(np.maximum(d[:,0], np.maximum(d[:,1], d[:,2])), 0.0) + np.linalg.norm(np.maximum(d, 0.0), axis=1)

def soft_min(x, y, a):
    """C2 continuous smooth minimum blend."""
    k = np.maximum(a, 1e-6)
    XX = 2.0 * (x - y) / k
    Abs2 = np.abs(XX)
    abs_result = np.where(Abs2 < 2.0, 0.5 * XX**2 * (1.0 - Abs2 / 6.0) + 2.0 / 3.0, Abs2)
    return 0.5 * (x + y - abs_result * k / 2.0)

class ImplicitBlender:
    def __init__(self): self.prims = []
    def add(self, f, p, c, s, b): self.prims.append({'f':f, 'p':p, 'c':c, 's':s, 'b':b})
    def eval(self, pts):
        if not self.prims: return np.zeros(len(pts))
        fields = []
        for x in self.prims:
            d = x['f'].eval(pts, x['p'], x['c'])
            b = OrientedBounds.eval(pts, x['b'])
            fields.append(np.maximum(d, b)) 

        curr = np.array(fields)
        while len(curr) > 1:
            half = len(curr)//2
            new_f = []
            for i in range(half):
                a = 0.2 # Smoothing factor
                new_f.append(soft_min(curr[i], curr[half+i], a))
            if len(curr)%2: new_f.append(curr[-1])
            curr = np.array(new_f)
        return curr[0]

# ==========================================
# PART 2: GEOMETRIC PATCH SEGMENTATION
# ==========================================

class PatchSegmenter:
    def __init__(self, n_patches=30): self.k = n_patches
    def segment(self, points):
        if len(points) < self.k: return np.zeros(len(points), dtype=int)
        
        indices = np.random.choice(len(points), self.k, replace=False)
        centroids = points[indices]
        labels = np.zeros(len(points), dtype=int)
        
        for _ in range(15):
            dists = np.linalg.norm(points[:, None] - centroids[None, :], axis=2)
            new_labels = np.argmin(dists, axis=1)
            if np.all(labels == new_labels): break
            labels = new_labels
            for i in range(self.k):
                mask = (labels == i)
                if np.any(mask): centroids[i] = points[mask].mean(axis=0)
        return labels

class HomogeneityRefiner:
    def __init__(self, fitter):
        self.fitter = fitter

    def _calculate_quadric_error(self, pts, params, center):
        if len(pts) < 10: return np.inf
        vals = self.fitter.eval(pts, params, center)
        return np.mean(np.abs(vals))

    def refine(self, points, labels, error_tolerance=0.005):
        unique_labels = np.unique(labels)
        
        patch_metrics = {}
        patch_points = {}
        for l in unique_labels:
            pts = points[labels == l]
            if len(pts) < 10: continue
            
            params, center = self.fitter.fit(pts)
            error = self._calculate_quadric_error(pts, params, center)
            patch_metrics[l] = {'error': error, 'size': len(pts), 'centroid': np.mean(pts, 0)}
            patch_points[l] = pts

        merges_made = True
        
        while merges_made:
            merges_made = False
            best_score = np.inf
            best_merge = None
            
            current_labels_active = list(patch_metrics.keys())
            if len(current_labels_active) <= 1: break

            centroids_active = np.array([patch_metrics[l]['centroid'] for l in current_labels_active])
            centroid_tree_active = cKDTree(centroids_active)
            active_idx_to_label = {i: l for i, l in enumerate(current_labels_active)}

            
            for i in range(len(current_labels_active)):
                l_i = current_labels_active[i]
                pts_i = patch_points[l_i]
                
                _, neighbors_idx = centroid_tree_active.query(patch_metrics[l_i]['centroid'], k=5)
                
                for n_idx in neighbors_idx:
                    l_j = active_idx_to_label[n_idx]
                    if l_i == l_j: continue
                    if l_j not in patch_metrics: continue

                    pts_j = patch_points[l_j]
                    pts_combined = np.vstack([pts_i, pts_j])
                    
                    params_combined, center_combined = self.fitter.fit(pts_combined)
                    error_combined = self._calculate_quadric_error(pts_combined, params_combined, center_combined)
                    
                    if error_combined < error_tolerance:
                        score = error_combined * len(pts_combined) 
                        
                        if score < best_score:
                            best_score = score
                            best_merge = (l_i, l_j, pts_combined)
            
            if best_merge:
                l_winner, l_loser, pts_new = best_merge
                
                labels[labels == l_loser] = l_winner
                
                patch_points[l_winner] = pts_new
                patch_metrics[l_winner]['error'] = best_score / len(pts_new)
                patch_metrics[l_winner]['size'] = len(pts_new)
                
                del patch_metrics[l_loser]
                del patch_points[l_loser]
                merges_made = True
                
        final_labels = np.copy(labels)
        current_map = {old: new for new, old in enumerate(np.unique(final_labels))}
        for old, new in current_map.items():
            final_labels[final_labels == old] = new
            
        return final_labels

# ==========================================
# PART 3: PIPELINE EXECUTION
# ==========================================

class ReconstructionPipeline:
    @staticmethod
    def load_data(path, samples=5000):
        points = None
        if HAS_OPEN3D:
            try:
                o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
                pcd = o3d.io.read_point_cloud(path)
                if len(pcd.points) == 0:
                     mesh = o3d.io.read_triangle_mesh(path)
                     if len(mesh.vertices) > 0: pcd = mesh.sample_points_poisson_disk(samples)
                points = np.asarray(pcd.points)
            except: pass
        if points is not None and len(points) > 10:
            c = np.mean(points, axis=0)
            points -= c
            scale = np.max(np.linalg.norm(points, axis=1))
            if scale > 0: points /= scale
            return points
        return None

    @staticmethod
    def visualize(gt_points, verts, faces, title):
        """
        Modified visualization: Overlaps Input Cloud and Reconstructed Mesh.
        [cite_start][cite: 38-40]
        """
        print(f"    [Display] Opening interactive window for: {title}")
        
        if HAS_OPEN3D:
            try:
                # 1. Input (Red)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gt_points)
                pcd.paint_uniform_color([1, 0, 0])
                
                # 2. Result (Gray) - OVERLAPPED
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.7, 0.7, 0.7])
                
                # Draw both in the same space
                o3d.visualization.draw_geometries([pcd, mesh], window_name=title)
            except Exception as e:
                print(f"    [Error] Visualization failed: {e}")

    @staticmethod
    def run(dataset_path):
        fitter = QuadricFitter(k=2000.0)
        
        # Tools
        segmenter = PatchSegmenter(n_patches=30) 
        refiner = HomogeneityRefiner(fitter=fitter)
        augmentor = GeometricAugmentor()

        # Update resolution to 128x128x128
        MESH_RESOLUTION = 128 
        LIM_SCALE = 1.2

        if os.name == 'nt' and not dataset_path.startswith('\\\\?\\'):
            dataset_path = '\\\\?\\' + os.path.abspath(dataset_path)

        print(f"--- HOMOGENEITY-GUIDED RECONSTRUCTION PIPELINE (Res: {MESH_RESOLUTION}) ---")
        files = []
        for root, _, fnames in os.walk(dataset_path):
            for f in fnames:
                if f.lower().endswith(('.obj', '.ply', '.off', '.stl')):
                    files.append(os.path.join(root, f))
        
        if not files: print("[!] No models found."); return

        for i, fpath in enumerate(files[:3]):
            fname = os.path.basename(fpath)
            print(f"\n[{i+1}] Processing: {fname}")
            
            points = ReconstructionPipeline.load_data(fpath)
            if points is None: continue

            # 1. Initial Segmentation
            labels_initial = segmenter.segment(points)

            # 2. Homogeneity Refinement (Merges slices based on fit quality)
            labels = refiner.refine(points, labels_initial, error_tolerance=0.005)
            unique_parts = np.unique(labels)
            print(f"    -> Refined to {len(unique_parts)} structurally homogeneous patches.")

            # 3. Fit & Blend
            blender = ImplicitBlender()
            for part_id in unique_parts:
                part_pts = points[labels == part_id]
                if len(part_pts) < 10: continue
                
                aug = augmentor.augment(part_pts, 0.04)
                p, c = fitter.fit(np.vstack([part_pts, aug]))
                b = OrientedBounds.fit(part_pts)
                diag = np.linalg.norm(np.max(part_pts,0)-np.min(part_pts,0))
                blender.add(fitter, p, c, max(0.04, diag*0.2), b)

            # 4. Marching Cubes
            res = MESH_RESOLUTION
            lim = LIM_SCALE
            x = np.linspace(-lim, lim, res)
            grid = np.stack(np.meshgrid(x, x, x, indexing='ij'), axis=-1).reshape(-1, 3)
            vals = blender.eval(grid).reshape(res, res, res)
            
            try:
                verts, faces, _, _ = marching_cubes(vals, 0.0)
                verts = verts / (res-1) * (2*lim) - lim
                
                print(f"    -> Mesh generated: {len(verts)} vertices.")
                ReconstructionPipeline.visualize(points, verts, faces, f"Result: {fname}")
                
            except Exception as e:
                print(f"    -> Meshing failed: {e}")

if __name__ == "__main__":
    DATASET_PATH = r"C:\Users\cssql\.cache\kagglehub\datasets\hajareddagni\shapenetcorev2\versions\1\ShapeNetCore.v2\ShapeNetCore.v2"
    ReconstructionPipeline.run(DATASET_PATH)