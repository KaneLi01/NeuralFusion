import numpy as np
from neural_fusion.ops.math_utils import SDFBlender, OrientedBounds

class SculptingBlender:
    """ 
    Implements the Sculpting Operation:
    Result = Intersection(RawStone, Union(FittedParts))
    将两个SDF进行合并
    """
    def __init__(self, raw_stone): 
        self.raw_stone = raw_stone
        self.prims = []
        
    def add(self, f, p, c): 
        self.prims.append({'f':f, 'p':p, 'c':c})
    
    def upgrade(self, f, p, c): 
        self.prims = [{'f':f, 'p':p, 'c':c}]
        
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
                new_f.append(SDFBlender.union(curr[i], curr[half+i], a))
            if len(curr)%2: new_f.append(curr[-1])
            curr = np.array(new_f)
        
        sdf_parts = curr[0]
        
        # 4. THE SCULPTING STEP: Intersection (Stone AND Parts)
        # We ensure the final shape is carved OUT of the sphere
        # SoftMax(Stone, Parts) = Intersection
        
        return SDFBlender.intersection(sdf_stone, sdf_parts, 0.01)


class SculptingBlender_:
    """
    Implements the Sculpting Operation with optimized primitive blending.
    
    Logic:
    1. Initialize a base field.
    2. For each primitive (Planar or Quadric):
       - Apply spatial masking (Optimization).
       - Compute local SDF (Geometric shapes + 2D Spline boundaries).
       - Intersect with Oriented Bounding Box (OBB).
    3. Blend into the global field using Soft Union (Constructive Solid Geometry).
    """
    def __init__(self):
        self.prims = []

    def add_planar(self, m, b):
        """
        Register a planar primitive (e.g., a flat surface with a complex 2D boundary).
        :param m: Mesh/Geometry dict {'c': center, 'rot': rotation_matrix, 'thickness': float, 'boundary': 2D_points}
        :param b: Oriented Bounding Box {'c': center, 'min': vector, 'max': vector}
        """
        self.prims.append({'type': 'planar', 'm': m, 'b': b})

    def add_quadric(self, f, p, c, b):
        """
        Register a quadric primitive (e.g., sphere, ellipsoid).
        :param f: Quadric function object
        :param p: Parameters for the function
        :param c: Center
        :param b: Oriented Bounding Box
        """
        self.prims.append({'type': 'quadric', 'f': f, 'p': p, 'c': c, 'b': b})

    def eval(self, pts):
        """
        Evaluate the Signed Distance Field (SDF) for the entire sculpted object at given points.
        :param pts: (N, 3) array of query points.
        :return: (N,) array of SDF values.
        """
        # 0. Handle empty case
        if not self.prims: 
            return np.ones(len(pts))

        # 1. Initialize Base Field
        # distinct positive value (0.5) represents empty space "outside" the surface
        res = np.full(len(pts), 0.5, dtype=np.float32)

        # 2. Iterate through all registered primitives
        for x in self.prims:
            # --- Optimization: Bounding Sphere Check ---
            # Only evaluate complex SDFs for points close to this primitive
            c = x['b']['c']
            # Radius = half-diagonal of bbox + margin (0.2)
            r = np.linalg.norm((x['b']['max'] - x['b']['min']) / 2) + 0.2
            
            # Create a mask for points within influence radius
            dist_sq = np.sum((pts - c)**2, axis=1)
            mask = dist_sq < r**2
            
            if not np.any(mask): 
                continue
            
            p_act = pts[mask]
            val = None

            # 3. Compute Local SDF based on Type
            if x['type'] == 'planar':
                # --- A. Planar Logic ---
                # Transform points to local 2D plane coordinates
                loc = (p_act - x['m']['c']) @ x['m']['rot']
                
                # a. Thickness distance (SDF of an infinite slab)
                d_th = np.abs(loc[:, 0]) - x['m']['thickness'] / 2
                
                # b. 2D Boundary Field (Spline-based shape within the plane)
                uv = loc[:, 1:3]
                b_min = np.min(x['m']['boundary'], 0)
                b_max = np.max(x['m']['boundary'], 0)
                s = np.max(b_max - b_min) + 1e-9 # Scale factor
                
                # Evaluate the algebraic spline field for the contour
                cov = AlgebraicSpline2D.eval_boundary_field(uv / s, x['m']['boundary'] / s, 0.03)
                
                # c. Intersection: Slab AND 2D_Shape
                # We use soft_max (Intersection) to cut the shape out of the slab
                # Note: 0.5 - cov converts the field range to match SDF conventions roughly
                val = SDFBlender.intersection(d_th, 0.5 - cov, 0.01)
                
                # d. Intersection with OBB (Global bounding cut)
                val = SDFBlender.intersection(val, OrientedBounds.eval(p_act, x['b']), 0.01)

            else:
                # --- B. Quadric Logic ---
                # Evaluate mathematical primitive (Sphere/Ellipsoid/etc)
                val = x['f'].eval(p_act, x['p'], x['c'])
                
                # Intersection with OBB
                val = SDFBlender.intersection(val, OrientedBounds.eval(p_act, x['b']), 0.02)

            # 4. Blend into Global Field (Union)
            # Use soft_min to smoothly merge this primitive with previous results
            res[mask] = SDFBlender.union(res[mask], val, 0.05)

        return res