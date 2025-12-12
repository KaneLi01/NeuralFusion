import numpy as np

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