import numpy as np
from neural_fusion.ops.math_utils import softAbs2, softMin2, softMax2

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