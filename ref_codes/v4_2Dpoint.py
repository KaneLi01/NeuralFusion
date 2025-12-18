import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes, find_contours
from scipy.spatial import cKDTree
from scipy.linalg import eig
from scipy.ndimage import gaussian_filter
import time

# =========================================================================
# 1. CORE MATH: ALGEBRAIC SPLINE 2D
# =========================================================================
class AlgebraicSpline2D:
    @staticmethod
    def H0(t): return 0.5 * (np.sign(t) + 1.0)
    @staticmethod
    def H2(x):
        val = np.ones_like(x); mask_le0 = (x <= 0); val[mask_le0] = 0.5 * (1.0 + 0.5 * x[mask_le0])**2
        mask_gt0 = (x > 0); val[mask_gt0] = 1.0 - 0.5 * (1.0 - 0.5 * x[mask_gt0])**2
        val[x < -2.0] = 0.0; val[x > 2.0] = 1.0
        return val
    @staticmethod
    def genH(t, delta): return AlgebraicSpline2D.H2(2.0 * t / (delta + 1e-9))
    @staticmethod
    def impPoint(xy, x0, slope, delta):
        dx = xy[:, 0] - x0[0]; dy = xy[:, 1] - x0[1]
        if np.abs(slope) > 1e8: return np.zeros_like(dx)
        elif np.abs(slope) < 1e-9: return AlgebraicSpline2D.genH(-dx, 2*delta) * AlgebraicSpline2D.genH(-dy, 2*delta)
        
        def impAngle(x, y, m, d):
            def L(kx, ky, km):
                v = np.zeros_like(kx); am = np.abs(km)
                msk = (ky < 0) & (ky < am * kx)
                if np.any(msk):
                    xm=kx[msk]; ym=ky[msk]; r=np.zeros_like(xm)
                    le=(xm<=0); gt=~le
                    if np.any(le): z=am*xm[le]-ym[le]; r[le]=0.5*(np.sign(z)+1)*(z**4)/(24*am**2+1e-12)
                    if np.any(gt): t=xm[gt]; y=ym[gt]; r[gt]=(-t*y**3/(6*am))+(y**4/(24*am**2))
                    v[msk]=r
                return v
            def U(ux, uy): return L(ux+2*d, uy, m)+L(ux-2*d, uy, m)-2*L(ux, uy, m)
            return (U(x, y-2*d)+U(x, y+2*d)-2*U(x, y))/(16*d**4+1e-12)

        if slope > 0:
            if slope > 1: return impAngle(dx, dy, slope, delta)
            return AlgebraicSpline2D.genH(-dx, 2*delta)*AlgebraicSpline2D.genH(-dy, 2*delta) - impAngle(dy, dx, 1/slope, delta)
        else:
            if -slope > 1: return impAngle(-dx, dy, -slope, delta)
            return AlgebraicSpline2D.genH(-(-dx), 2*delta)*AlgebraicSpline2D.genH(-dy, 2*delta) - impAngle(dy, -dx, 1/(-slope), delta)

    @staticmethod
    def eval_boundary_field(query_pts, vertices, delta=0.05):
        total_field = np.zeros(len(query_pts)); M = len(vertices)
        for i in range(M):
            p_curr = vertices[i]; p_next = vertices[(i + 1) % M]
            xDir = p_curr[0] - p_next[0]
            if np.abs(xDir) < 1e-9: continue
            y01 = p_next[1] - p_curr[1]; x01 = p_next[0] - p_curr[0]
            slope = 1e8 if np.abs(x01) < 1e-9 else y01 / x01
            t1 = AlgebraicSpline2D.impPoint(query_pts, p_next, slope, delta)
            t2 = AlgebraicSpline2D.impPoint(query_pts, p_curr, slope, delta)
            val = (t1 - t2) if (np.abs(slope)<1e-9 and x01>0) or (np.abs(slope)>=1e-9 and y01>0) else (t2 - t1)
            total_field += np.sign(xDir) * val
        return total_field

# =========================================================================
# 2. ROBUST DENSITY-BASED BOUNDARY EXTRACTION
# =========================================================================
class DensityBoundaryExtractor:
    @staticmethod
    def get_boundary(points, grid_res=64, smooth_sigma=1.0, simplify_eps=0.04):
        """
        Extracts boundary by rasterizing points into a grid, smoothing, and contouring.
        Robust to noise and non-convexity.
        """
        if len(points) < 4: return points
        
        # 1. Setup Grid
        min_p = np.min(points, axis=0); max_p = np.max(points, axis=0)
        padding = (max_p - min_p) * 0.2
        min_p -= padding; max_p += padding
        range_p = max_p - min_p
        
        # Avoid division by zero for lines
        if range_p[0] < 1e-9: range_p[0] = 1.0
        if range_p[1] < 1e-9: range_p[1] = 1.0
        
        # 2. Histogram (Density)
        # Normalize points to integer grid indices
        norm_pts = (points - min_p) / range_p * (grid_res - 1)
        grid_indices = norm_pts.astype(int)
        
        # Clip to safe range
        grid_indices[:,0] = np.clip(grid_indices[:,0], 0, grid_res-1)
        grid_indices[:,1] = np.clip(grid_indices[:,1], 0, grid_res-1)
        
        density = np.zeros((grid_res, grid_res))
        for x, y in grid_indices:
            density[x, y] += 1
            
        # 3. Gaussian Smooth
        density = gaussian_filter(density, sigma=smooth_sigma)
        
        # 4. Marching Squares (Find Contour at 10% of max density)
        thresh = density.max() * 0.1
        contours = find_contours(density, thresh)
        
        if not contours:
            # Fallback if density is too weak
            return points[ConvexHull(points).vertices]
            
        # Take largest contour
        contour = max(contours, key=len)
        
        # 5. Transform back to world space
        # contour comes out as (row, col) -> (y, x) indices
        # We need to map back to x, y physical coordinates
        # Contour vertices are [row (x_idx), col (y_idx)] based on how we filled 'density[x,y]'
        
        boundary = np.zeros_like(contour)
        boundary[:, 0] = contour[:, 0] / (grid_res - 1) * range_p[0] + min_p[0]
        boundary[:, 1] = contour[:, 1] / (grid_res - 1) * range_p[1] + min_p[1]
        
        # 6. Simplify
        simplified = PolygonSimplifier.simplify(boundary, epsilon=simplify_eps)
        return DensityBoundaryExtractor.enforce_ccw(simplified)

    @staticmethod
    def enforce_ccw(p):
        area = 0.5 * np.sum(p[:,0]*np.roll(p[:,1],-1) - np.roll(p[:,0],-1)*p[:,1])
        return p[::-1] if area < 0 else p

class PolygonSimplifier:
    @staticmethod
    def simplify(points, epsilon=0.04):
        if len(points) < 3: return points
        dists_sq = np.sum((points - points[0])**2, axis=1); farthest_idx = np.argmax(dists_sq)
        path1 = PolygonSimplifier._rdp(points[:farthest_idx+1], epsilon)
        path2 = PolygonSimplifier._rdp(np.vstack([points[farthest_idx:], points[0]]), epsilon)
        return np.vstack([path1[:-1], path2[:-1]])
    @staticmethod
    def _rdp(points, epsilon):
        if len(points) < 3: return points
        start, end = points[0], points[-1]; vec = end - start; norm_vec = np.linalg.norm(vec)
        if norm_vec < 1e-9: dists = np.linalg.norm(points - start, axis=1)
        else:
            vec_pts = points - start; cross = vec_pts[:, 0] * vec[1] - vec_pts[:, 1] * vec[0]
            dists = np.abs(cross) / norm_vec
        index = np.argmax(dists)
        if dists[index] > epsilon:
            return np.vstack([PolygonSimplifier._rdp(points[:index+1], epsilon)[:-1], PolygonSimplifier._rdp(points[index:], epsilon)])
        else: return np.array([start, end])

# =========================================================================
# 3. PIPELINE HELPERS
# =========================================================================
class NaturalFeatureSegmenter:
    def __init__(self, t=0.85, s=50): self.t=t; self.s=s
    def segment(self, p):
        tree=cKDTree(p); _,idx=tree.query(p,10); n=np.zeros((len(p),3))
        for i in range(len(p)): n[i]=np.linalg.eigh(np.cov(p[idx[i]],rowvar=False))[1][:,0]
        l=np.full(len(p),-1,int); vis=np.zeros(len(p),bool); sid=0
        for i in range(len(p)):
            if not vis[i]:
                q=[i]; vis[i]=True; l[i]=sid; cnt=0; curr=n[i]
                while q:
                    u=q.pop(0); cnt+=1
                    for v in idx[u]:
                        if not vis[v] and np.abs(np.dot(curr,n[v]))>self.t: vis[v]=True; l[v]=sid; q.append(v)
                if cnt>=self.s: sid+=1
                else: l[l==sid]=-1
        return l

class OrientedBounds:
    @staticmethod
    def fit(pts):
        c=np.mean(pts,0); p=pts-c; R=np.linalg.eigh(p.T@p/len(pts))[1]
        return {'R':R, 'min':np.min(p@R,0), 'max':np.max(p@R,0), 'c':c}
    @staticmethod
    def eval(pts, b):
        local=(pts-b['c'])@b['R']; d=np.abs(local-(b['min']+b['max'])/2)-(b['max']-b['min'])/2
        return np.minimum(np.maximum(d[:,0],np.maximum(d[:,1],d[:,2])),0)+np.linalg.norm(np.maximum(d,0),axis=1)

def soft_min(x, y, a=0.05): k=np.maximum(a,1e-6); h=np.maximum(k-np.abs(x-y),0)/k; return np.minimum(x,y)-h*h*k*0.25
def soft_max(x, y, a=0.05): return -soft_min(-x,-y,a)

class ImplicitBlender:
    def __init__(self): self.prims = []
    def add_planar(self, m, b): self.prims.append({'type':'planar', 'm':m, 'b':b})
    def add_quadric(self, f, p, c, b): self.prims.append({'type':'quadric', 'f':f, 'p':p, 'c':c, 'b':b})
    def eval(self, pts):
        if not self.prims: return np.ones(len(pts))
        res = np.full(len(pts), 0.5, dtype=np.float32)
        for x in self.prims:
            c=x['b']['c']; r=np.linalg.norm((x['b']['max']-x['b']['min'])/2)+0.2
            mask=np.sum((pts-c)**2,1)<r**2
            if not np.any(mask): continue
            p_act=pts[mask]
            
            if x['type']=='planar':
                loc=(p_act-x['m']['c'])@x['m']['rot']
                d_th=np.abs(loc[:,0])-x['m']['thickness']/2
                uv=loc[:,1:3]
                b_min=np.min(x['m']['boundary'],0); b_max=np.max(x['m']['boundary'],0); s=np.max(b_max-b_min)+1e-9
                cov=AlgebraicSpline2D.eval_boundary_field(uv/s, x['m']['boundary']/s, 0.03)
                val=soft_max(d_th, 0.5-cov, 0.01)
                val=soft_max(val, OrientedBounds.eval(p_act, x['b']), 0.01)
            else:
                val=x['f'].eval(p_act, x['p'], x['c'])
                val=soft_max(val, OrientedBounds.eval(p_act, x['b']), 0.02)
            res[mask]=soft_min(res[mask], val, 0.05)
        return res

# =========================================================================
# 4. MAIN PIPELINE
# =========================================================================
def generate_F_block(num_points=4000):
    # F shape in YZ Plane (Y=vertical, Z=horizontal)
    # Extrude along X
    spine = np.random.uniform([-1.0, -0.8], [1.0, -0.4], (num_points//2, 2))
    top = np.random.uniform([0.6, -0.4], [1.0, 0.6], (num_points//4, 2))
    mid = np.random.uniform([-0.2, -0.4], [0.2, 0.2], (num_points//4, 2))
    yz = np.vstack([spine, top, mid])
    x = np.random.uniform(-0.5, 0.5, len(yz))
    points = np.column_stack([x, yz[:,0], yz[:,1]])
    return points + np.random.normal(0, 0.01, points.shape)

def run_pipeline():
    points = generate_F_block()
    
    # Simple segmentation (treat as one block for F test)
    labels = np.zeros(len(points), int)
    
    blender = ImplicitBlender()
    class SimpleQuadric:
        def eval(self,p,par,c): return np.linalg.norm(p-c,1)-par[0]
            
    boundary_list = [] # Store local 2D boundaries for viz

    for lid in [0]:
        mask=(labels==lid); patch=points[mask]
        c=np.mean(patch,0); cov=(patch-c).T@(patch-c)/len(patch)
        vals,vecs=np.linalg.eigh(cov); idx=np.argsort(vals); vals=vals[idx]; vecs=vecs[:,idx]
        flat=np.sqrt(vals[0])/(np.sqrt(vals[1])+1e-9)
        bbox=OrientedBounds.fit(patch)
        
        # Force PLANAR logic for F-block
        uv=(patch-c)@vecs[:,1:3]
        
        # USE DENSITY EXTRACTOR
        boundary=DensityBoundaryExtractor.get_boundary(uv, grid_res=64, smooth_sigma=1.0, simplify_eps=0.05)
        
        # Store for 2D Viz
        boundary_list.append(boundary)
        
        blender.add_planar({'c':c,'rot':vecs,'thickness':2*np.sqrt(vals[0]),'boundary':boundary}, bbox)

    res=64; lim=2.0
    x=np.linspace(-lim,lim,res)
    X,Y,Z=np.meshgrid(x,x,x,indexing='ij')
    grid=np.vstack([X.ravel(),Y.ravel(),Z.ravel()]).T
    
    vals=blender.eval(grid).reshape(res,res,res)
    
    # --- VISUALIZATION ---
    fig = plt.figure(figsize=(14, 7))
    
    # Plot 1: The Local 2D Shape (What the Spline sees)
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(uv[:,0], uv[:,1], s=1, c='gray', alpha=0.5, label='Projected Points (UV)')
    # Plot Boundary
    b_plot = np.vstack([boundary_list[0], boundary_list[0][0]])
    ax1.plot(b_plot[:,0], b_plot[:,1], 'r-', linewidth=3, label='Extracted Boundary')
    ax1.scatter(boundary_list[0][:,0], boundary_list[0][:,1], c='red', s=40, zorder=5)
    ax1.set_title(f"Local 2D Shape Extraction\n({len(boundary_list[0])} Vertices)")
    ax1.legend()
    ax1.axis('equal')
    
    # Plot 2: The Reconstructed 3D Object
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    try:
        verts,faces,_,_=marching_cubes(vals,0.0)
        verts=verts*(2*lim/(res-1))-lim
        mesh = Poly3DCollection(verts[faces], alpha=0.6, edgecolor='k', linewidth=0.1)
        mesh.set_facecolor([0.2, 0.8, 0.2])
        ax2.add_collection3d(mesh)
        ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_zlim(-lim, lim)
        ax2.set_title("3D Implicit Reconstruction")
    except:
        print("Meshing failed (likely empty volume).")
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline()