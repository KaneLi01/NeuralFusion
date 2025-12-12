import numpy as np
from scipy.linalg import eig

class QuadricFitter:
    """
    Fits a quadric surface to a set of 3D points using Generalized Eigenvalue fitting.
    The surface is defined by:
    Ax^2 + By^2 + Cz^2 + Dxy + Exz + Fyz + Gx + Hy + Iz + J = 0
    根据输入的点，最小二乘得到最近似的参数
    """

    def __init__(self, regularization: float = 2000.0):
        """
        Args:
            regularization (float): Parameter 'k' controlling the constraint matrix.
                                    Higher values enforce strictly bounded shapes (ellipsoids).
        """
        self.regularization = regularization

    def fit(self, points: np.ndarray):
        """
        Fits a quadric surface to the input point cloud.

        Args:
            points (np.ndarray): (N, 3) array of points.

        Returns:
            tuple: (params, center)
                - params: Normalized coefficients [A, B, C, D, E, F, G, H, I, J]
                - center: The mean of the input points (used for centering).
        """
        # 1. Center the data
        center = np.mean(points, axis=0)
        centered_points = points - center
        x, y, z = centered_points.T

        # 2. Construct Design Matrices
        # D1: Quadratic terms [x^2, y^2, z^2, xy, xz, yz]
        mat_quadratic = np.stack([x**2, y**2, z**2, x*y, x*z, y*z], axis=1)
        # D2: Linear terms + constant [x, y, z, 1]
        mat_linear = np.stack([x, y, z, np.ones_like(x)], axis=1)

        # 3. Construct Scatter Matrix (M)
        # M = D1'D1 - D1'D2 * (D2'D2)^-1 * D2'D1
        # This projects the problem into the null space of the linear terms
        d2_t_d2_inv = np.linalg.pinv(mat_linear.T @ mat_linear)
        scatter_matrix = mat_quadratic.T @ mat_quadratic - \
                         mat_quadratic.T @ mat_linear @ d2_t_d2_inv @ mat_linear.T @ mat_quadratic

        # 4. Construct Constraint Matrix (C) for the generalized eigenvalue problem
        # The constraint prevents the trivial solution (all zeros) and controls shape type.
        constraint_matrix = np.zeros((6, 6))
        
        # Diagonal elements for x^2, y^2, z^2
        np.fill_diagonal(constraint_matrix[:3, :3], -1.0)
        
        # Off-diagonal and cross-term weights
        offset = (self.regularization / 2.0) - 1.0
        const_r = -self.regularization / 4.0
        
        # Symmetric indices for cross terms (xy, xz, yz)
        indices = [(0, 1), (0, 2), (1, 2)]
        for i, j in indices:
            constraint_matrix[i, j] = offset
            constraint_matrix[j, i] = offset
            
        # Diagonal for cross terms (indices 3, 4, 5 correspond to xy, xz, yz)
        constraint_matrix[3, 3] = const_r
        constraint_matrix[4, 4] = const_r
        constraint_matrix[5, 5] = const_r

        # 5. Solve Generalized Eigenproblem: M * v = lambda * C * v
        try:
            evals, evecs = eig(scatter_matrix, constraint_matrix)
            # Find the best positive eigenvalue
            valid_idx = np.where(evals.real > 1e-9)[0]
            
            if len(valid_idx) > 0:
                best_idx = valid_idx[np.argmax(evals.real[valid_idx])]
                quad_params = evecs[:, best_idx].real
            else:
                # Fallback default: Sphere-like
                quad_params = np.array([1., 1., 1., 0., 0., 0.])
                
        except np.linalg.LinAlgError:
            # Fallback if solution fails
            quad_params = np.array([1., 1., 1., 0., 0., 0.])

        # 6. Recover Linear Parameters
        # v2 = -(D2'D2)^-1 * D2'D1 * v1
        linear_params = -d2_t_d2_inv @ mat_linear.T @ mat_quadratic @ quad_params

        # Combine into full parameter vector [A..J]
        params = np.concatenate([quad_params, linear_params])
        params /= (np.linalg.norm(params) + 1e-9)

        # 7. Orient the surface
        # Ensure the center of the points evaluates to < 0 (Inside)
        if self.eval(center[None, :], params, center) > 0:
            params = -params

        # 8. Normalize by Gradient Magnitude
        # This approximates Euclidean distance for SDF rendering
        A, B, C, D, E, F, G, H, I, J = params
        dx = 2*A*x + D*y + E*z + G
        dy = 2*B*y + D*x + F*z + H
        dz = 2*C*z + E*x + F*y + I
        
        avg_grad_norm = np.mean(np.sqrt(dx**2 + dy**2 + dz**2))
        params /= (avg_grad_norm + 1e-9)

        return params, center

    def eval(self, pts: np.ndarray, params: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Evaluates the quadric function at given points.
        
        Args:
            pts: Points to evaluate (N, 3)
            params: Quadric parameters [A..J]
            center: Center used during fitting
            
        Returns:
            Scalar values (algebraic distance). < 0 is inside, > 0 is outside.
        """
        p = pts - center
        x, y, z = p.T
        A, B, C, D, E, F, G, H, I, J = params
        
        return (A*x**2 + B*y**2 + C*z**2 + 
                D*x*y + E*x*z + F*y*z + 
                G*x + H*y + I*z + J)