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
# PART 1: C2-CONTINUOUS BLENDING LOGIC
# ==========================================

def softAbs2(x, a):
    """ C2 continuous approximation of |x| """
    k = np.maximum(a, 1e-6)
    XX = 2.0 * x / k
    Abs2 = np.abs(XX)
    abs_result = np.where(
        Abs2 < 2.0,
        0.5 * XX**2 * (1.0 - Abs2 / 6.0) + 2.0 / 3.0,
        Abs2 
    )
    return abs_result * k / 2.0

def softMin2(x, y, a):
    """ Smooth Minimum (Union for SDF) """
    return 0.5 * (x + y - softAbs2(x - y, a))

def softMax2(x, y, a):
    """ Smooth Maximum (Intersection for SDF) """
    return 0.5 * (x + y + softAbs2(x - y, a))

# ==========================================
# PART 2: THE "SCULPTOR'S TOOLS"
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

class SculptingBlender:
    """ 
    Implements the Sculpting Operation:
    Result = Intersection(RawStone, Union(FittedParts))
    """
    def __init__(self, raw_stone): 
        self.raw_stone = raw_stone
        self.prims = []
        
    def add(self, f, p, c): 
        self.prims.append({'f':f, 'p':p, 'c':c})
        
    def eval(self, pts):
        # 1. Evaluate the Raw Stone (Bounding Sphere)
        sdf_stone = self.raw_stone.eval(pts)
        
        if not self.prims: return sdf_stone

        # 2. Evaluate the Fitted Parts (Chisel Cuts)
        fields = []
        for x in self.prims:
            d = x['f'].eval(pts, x['p'], x['c'])
            fields.append(d)

        # 3. Union of Parts (Combine cuts into the target shape)
        curr = np.array(fields)
        while len(curr) > 1:
            half = len(curr)//2
            new_f = []
            for i in range(half):
                a = 0.15 # Smoothing for union
                new_f.append(softMin2(curr[i], curr[half+i], a))
            if len(curr)%2: new_f.append(curr[-1])
            curr = np.array(new_f)
        
        sdf_parts = curr[0]
        
        # 4. THE SCULPTING STEP: Intersection (Stone AND Parts)
        # We ensure the final shape is carved OUT of the sphere
        # SoftMax(Stone, Parts) = Intersection
        
        return softMax2(sdf_stone, sdf_parts, 0.01)

# ==========================================
# PART 3: SCULPTOR'S PROCESS (Patch Segmentation)
# ==========================================

class PatchSegmenter:
    """ Divides the surface into working patches (Step 2 of prompt) """
    def __init__(self, n_patches=32): self.k = n_patches
    
    def segment(self, points):
        return self._numpy_kmeans(points, self.k)

    def _numpy_kmeans(self, X, k, max_iter=15):
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

class HomogeneityRefiner:
    """ Merges patches that belong to the same 'Chisel Stroke' """
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
        
        # Pre-calculate
        for l in unique_labels:
            pts = points[labels == l]
            if len(pts) < 10: continue
            params, center = self.fitter.fit(pts)
            error = self._calculate_quadric_error(pts, params, center)
            patch_metrics[l] = {'error': error, 'size': len(pts), 'centroid': np.mean(pts, 0)}
            patch_points[l] = pts

        # Merge Loop
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
        for old, new in current_map.items(): final_labels[final_labels == old] = new
        return final_labels

# ==========================================
# PART 4: PIPELINE EXECUTION
# ==========================================

class SculptingPipeline:
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
        
        if points is None: # Manual Fallback
            try:
                v = []
                with open(path, 'r', encoding='latin-1') as f:
                    for l in f:
                        if l.startswith('v '): v.append([float(x) for x in l.split()[1:4]])
                if len(v)>0: points = np.array(v)
            except: pass

        if points is not None and len(points) > 10:
            c = np.mean(points, axis=0)
            points -= c
            scale = np.max(np.linalg.norm(points, axis=1))
            if scale > 0: points /= scale
            return points
        return None

    @staticmethod
    def visualize_overlay(gt_points, verts, faces, title):
        print(f"    [Display] Opening interactive window for: {title}")
        if HAS_OPEN3D:
            try:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gt_points)
                pcd.paint_uniform_color([1, 0, 0]) # Red Input
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                mesh.compute_vertex_normals()
                mesh.paint_uniform_color([0.7, 0.7, 0.7]) # Gray Result
                
                o3d.visualization.draw_geometries([pcd, mesh], window_name=title)
            except Exception as e:
                print(f"    [Error] Visualization failed: {e}")

    @staticmethod
    def run(dataset_path):
        fitter = QuadricFitter(k=2000.0)
        segmenter = PatchSegmenter(n_patches=30) 
        refiner = HomogeneityRefiner(fitter=fitter)
        augmentor = GeometricAugmentor()

        if os.name == 'nt' and not dataset_path.startswith('\\\\?\\'):
            dataset_path = '\\\\?\\' + os.path.abspath(dataset_path)

        print(f"--- SCULPTOR PIPELINE (Carving Stone) ---")
        
        # Robust File Search
        files = []
        for root, dirs, fnames in os.walk(dataset_path):
            for f in fnames:
                if f.lower().endswith(('.obj', '.ply', '.off', '.stl')):
                    files.append(os.path.join(root, f))
        
        if not files: print("[!] No models found."); return

        for i, fpath in enumerate(files[:3]):
            fname = os.path.basename(fpath)
            print(f"\n[{i+1}] Sculpting: {fname}")
            
            # 1. Load Data
            points = SculptingPipeline.load_data(fpath)
            if points is None: continue

            # 2. Define Raw Stone (Bounding Sphere) [Step 1 of Prompt]
            raw_stone = BoundingSphere(points)
            print(f"    -> Raw Stone Prepared (Radius: {raw_stone.radius:.2f})")

            # 3. Segment & Refine (Dividing Surface) [Step 2]
            labels_initial = segmenter.segment(points)
            labels = refiner.refine(points, labels_initial, error_tolerance=0.005)
            unique_parts = np.unique(labels)
            print(f"    -> Surface divided into {len(unique_parts)} working regions.")

            # 4. Augment & Fit (The Chisel) [Step 3 & 4]
            # We initialize the blender with the Raw Stone
            blender = SculptingBlender(raw_stone)
            
            for part_id in unique_parts:
                part_pts = points[labels == part_id]
                if len(part_pts) < 10: continue
                
                # Augment the region with surface points (Step 3 of prompt)
                # Instead of adding sphere points (which causes artifacts),
                # we augment locally to ensure the quadric fits the surface patch tightly.
                aug = augmentor.augment(part_pts, 0.04)
                
                # Fit Ellipsoid (Step 4)
                p, c = fitter.fit(np.vstack([part_pts, aug]))
                
                # Add to sculpting tool
                blender.add(fitter, p, c)

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
                SculptingPipeline.visualize_overlay(points, verts, faces, f"Sculpture: {fname}")
                
            except Exception as e:
                print(f"    -> Carving failed (Mesh error): {e}")
                try:
                    # Fallback if iso-surface 0 is missed
                    verts, faces, _, _ = marching_cubes(vals, vals.min() + 0.1)
                    verts = verts / (res-1) * (2*lim) - lim
                    SculptingPipeline.visualize_overlay(points, verts, faces, f"Sculpture (Fallback): {fname}")
                except: pass

if __name__ == "__main__":
    DATASET_PATH = "/home/lkh/data/point_cloud/ShapeNetCore.v2/"
    SculptingPipeline.run(DATASET_PATH)