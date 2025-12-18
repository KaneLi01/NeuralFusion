from utils.generate_points import get_mock_cloud

class ProjectivePipeline:
    def _get_data(self):
        points = get_mock_cloud()
        return points

def run_pipeline():
    points = generate_F_block()
    
    # Simple segmentation (treat as one block for F test)
    labels = np.zeros(len(points), int)
    
    blender = ImplicitBlender()
            
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