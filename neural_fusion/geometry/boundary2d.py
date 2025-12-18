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
    """
    Implements the Ramer-Douglas-Peucker (RDP) algorithm for polygon simplification.
    Reduces the number of vertices in a curve while preserving its overall shape.
    """

    @staticmethod
    def simplify(points: np.ndarray, epsilon: float = 0.04) -> np.ndarray:
        """
        Main entry point. Simplifies a closed or open polygon.
        
        For closed polygons (loops), it intelligently splits the loop at the 
        farthest point to avoid artifacts at the start/end index.
        
        Args:
            points: (N, 2) array of XY coordinates.
            epsilon: Maximum distance tolerance. Higher value = fewer points (coarser).
            
        Returns:
            (M, 2) array of simplified points.
        """
        if len(points) < 3:
            return points

        # Strategy for closed loops:
        # Instead of arbitrarily breaking the loop at index 0, we find the point
        # farthest from index 0. This splits the polygon into two "open" paths 
        # that are structurally significant.
        dists_sq = np.sum((points - points[0]) ** 2, axis=1)
        farthest_idx = np.argmax(dists_sq)

        # Split into two paths: Start -> Farthest, and Farthest -> Start
        # We simplify them separately and then merge.
        path1 = PolygonSimplifier._rdp_recursive(points[:farthest_idx + 1], epsilon)
        
        # Construct path2 ensuring continuity (Farthest -> ... -> Start)
        # We stack [points[farthest:], points[0]] to close the loop logic
        path2_input = np.vstack([points[farthest_idx:], points[0]])
        path2 = PolygonSimplifier._rdp_recursive(path2_input, epsilon)

        # Merge (path1's end is path2's start, so slice [:-1])
        return np.vstack([path1[:-1], path2[:-1]])

    @staticmethod
    def _rdp_recursive(points: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Recursive implementation of the RDP algorithm.
        """
        if len(points) < 3:
            return points

        start_pt = points[0]
        end_pt = points[-1]

        # Find the point with the maximum perpendicular distance to the line (start->end)
        dists = PolygonSimplifier._point_line_distance(points, start_pt, end_pt)
        index = np.argmax(dists)
        max_dist = dists[index]

        if max_dist > epsilon:
            # If the point is too far, it's significant. Keep it and recurse.
            left_segment = PolygonSimplifier._rdp_recursive(points[:index + 1], epsilon)
            right_segment = PolygonSimplifier._rdp_recursive(points[index:], epsilon)
            
            # Merge results (left's last point is right's first point)
            return np.vstack([left_segment[:-1], right_segment])
        else:
            # All points are close enough to the line. Discard intermediate points.
            return np.array([start_pt, end_pt])

    @staticmethod
    def _point_line_distance(points: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """
        Vectorized calculation of perpendicular distance from a set of points 
        to the line segment defined by 'start' and 'end'.
        """
        line_vec = end - start
        point_vec = points - start
        
        line_len = np.linalg.norm(line_vec)

        # Handle degenerate case where start == end (distance is just point-to-point)
        if line_len < 1e-9:
            return np.linalg.norm(point_vec, axis=1)

        # In 2D, the magnitude of the cross product gives the area of the parallelogram.
        # Area = Base * Height  =>  Height = Area / Base
        # Cross product: x1*y2 - x2*y1
        cross_product = point_vec[:, 0] * line_vec[1] - point_vec[:, 1] * line_vec[0]
        
        return np.abs(cross_product) / line_len


import numpy as np

class AlgebraicSpline2D:
    """
    Implements algebraic spline mathematics for 2D implicit field evaluation.
    
    This class constructs C2-continuous implicit fields from polygon boundaries
    using smoothed Heaviside functions and spline integration.
    """

    @staticmethod
    def smooth_step(x: np.ndarray) -> np.ndarray:
        """
        Calculates the H2 smooth step function (C2 continuous).
        Maps [-2, 2] to a smooth transition from 0 to 1.
        """
        val = np.ones_like(x)
        
        # Region: -2 <= x <= 0
        mask_le0 = (x <= 0)
        # Formula: 0.5 * (1 + x/2)^2
        val[mask_le0] = 0.5 * (1.0 + 0.5 * x[mask_le0])**2
        
        # Region: 0 < x <= 2
        mask_gt0 = (x > 0)
        # Formula: 1 - 0.5 * (1 - x/2)^2
        val[mask_gt0] = 1.0 - 0.5 * (1.0 - 0.5 * x[mask_gt0])**2
        
        # Clamp outside regions
        val[x < -2.0] = 0.0
        val[x > 2.0] = 1.0
        
        return val

    @staticmethod
    def generalized_step(t: np.ndarray, delta: float) -> np.ndarray:
        """Scaled smooth step function controlled by delta."""
        return AlgebraicSpline2D.smooth_step(2.0 * t / (delta + 1e-9))

    @staticmethod
    def _sector_poly(kx: np.ndarray, ky: np.ndarray, m_abs: float) -> np.ndarray:
        """
        Internal Helper: Computes the polynomial 'L' for a specific angular sector.
        """
        v = np.zeros_like(kx)
        mask = (ky < 0) & (ky < m_abs * kx)
        
        if np.any(mask):
            x_m = kx[mask]
            y_m = ky[mask]
            res = np.zeros_like(x_m)
            
            # Sub-region 1: x <= 0
            le = (x_m <= 0)
            if np.any(le):
                z = m_abs * x_m[le] - y_m[le]
                # Fourth-order polynomial term
                term = (z ** 4) / (24 * m_abs**2 + 1e-12)
                res[le] = 0.5 * (np.sign(z) + 1) * term
            
            # Sub-region 2: x > 0
            gt = ~le
            if np.any(gt):
                t_val = x_m[gt]
                y_val = y_m[gt]
                term1 = -t_val * (y_val ** 3) / (6 * m_abs)
                term2 = (y_val ** 4) / (24 * m_abs ** 2)
                res[gt] = term1 + term2
            
            v[mask] = res
        return v

    @staticmethod
    def _finite_diff_u(ux: np.ndarray, uy: np.ndarray, m: float, d: float) -> np.ndarray:
        """
        Internal Helper: Computes 'U' using finite differences of the sector polynomial L.
        U approx = L(x+2d) + L(x-2d) - 2L(x)
        """
        l_plus  = AlgebraicSpline2D._sector_poly(ux + 2 * d, uy, m)
        l_minus = AlgebraicSpline2D._sector_poly(ux - 2 * d, uy, m)
        l_center = AlgebraicSpline2D._sector_poly(ux, uy, m)
        return l_plus + l_minus - 2 * l_center

    @staticmethod
    def _implicit_angle_field(x: np.ndarray, y: np.ndarray, m: float, d: float) -> np.ndarray:
        """
        Computes the implicit field value for a wedge defined by slope m.
        Represents the integration of the smooth indicator over the angular sector.
        """
        # Second order finite difference in Y direction
        num = (AlgebraicSpline2D._finite_diff_u(x, y - 2 * d, m, d) + 
               AlgebraicSpline2D._finite_diff_u(x, y + 2 * d, m, d) - 
               2 * AlgebraicSpline2D._finite_diff_u(x, y, m, d))
        
        denom = 16 * (d ** 4) + 1e-12
        return num / denom

    @staticmethod
    def evaluate_corner_field(xy: np.ndarray, origin: np.ndarray, slope: float, delta: float) -> np.ndarray:
        """
        Calculates the field contribution from a single vertex (corner context).
        Handles different slope cases (vertical, horizontal, general).
        """
        dx = xy[:, 0] - origin[0]
        dy = xy[:, 1] - origin[1]

        # Case 1: Vertical line (Infinite slope)
        if np.abs(slope) > 1e8:
            return np.zeros_like(dx)

        # Case 2: Horizontal line (Zero slope)
        # Use separable approximation: H(x) * H(y)
        if np.abs(slope) < 1e-9:
            val_x = AlgebraicSpline2D.generalized_step(-dx, 2 * delta)
            val_y = AlgebraicSpline2D.generalized_step(-dy, 2 * delta)
            return val_x * val_y

        # Case 3: General slopes
        # Depending on slope steepness and direction, combine generalized steps and angular fields
        gen_h_dx = AlgebraicSpline2D.generalized_step
        
        if slope > 0:
            if slope > 1:
                return AlgebraicSpline2D._implicit_angle_field(dx, dy, slope, delta)
            else:
                # Symmetry transformation for shallow positive slopes
                term1 = gen_h_dx(-dx, 2 * delta) * gen_h_dx(-dy, 2 * delta)
                term2 = AlgebraicSpline2D._implicit_angle_field(dy, dx, 1.0 / slope, delta)
                return term1 - term2
        else:
            # Negative slopes
            if -slope > 1:
                return AlgebraicSpline2D._implicit_angle_field(-dx, dy, -slope, delta)
            else:
                # Symmetry transformation for shallow negative slopes
                term1 = gen_h_dx(-(-dx), 2 * delta) * gen_h_dx(-dy, 2 * delta)
                term2 = AlgebraicSpline2D._implicit_angle_field(dy, -dx, 1.0 / (-slope), delta)
                return term1 - term2

    @staticmethod
    def eval_boundary_field(query_pts: np.ndarray, vertices: np.ndarray, delta: float = 0.05) -> np.ndarray:
        """
        Main Entry Point: Computes the total implicit field for a 2D polygon.
        
        Args:
            query_pts: (N, 2) array of query coordinates.
            vertices: (M, 2) array of ordered polygon vertices.
            delta: Smoothing width parameter.
            
        Returns:
            (N,) array of field values (approximate SDF).
        """
        total_field = np.zeros(len(query_pts))
        num_verts = len(vertices)

        for i in range(num_verts):
            p_curr = vertices[i]
            p_next = vertices[(i + 1) % num_verts]

            # Vector along the edge
            edge_vec = p_next - p_curr
            dx_edge = edge_vec[0]
            dy_edge = edge_vec[1]

            # Skip degenerate vertical edges (handled by integration logic)
            if np.abs(dx_edge) < 1e-9:
                continue

            # Calculate edge slope
            slope = 1e8 if np.abs(dx_edge) < 1e-9 else dy_edge / dx_edge

            # Evaluate contributions from both endpoints (Green's theorem integration)
            val_next = AlgebraicSpline2D.evaluate_corner_field(query_pts, p_next, slope, delta)
            val_curr = AlgebraicSpline2D.evaluate_corner_field(query_pts, p_curr, slope, delta)
            
            # Determine contribution sign based on edge direction
            is_horizontal_right = (np.abs(slope) < 1e-9) and (dx_edge > 0)
            is_sloped_up = (np.abs(slope) >= 1e-9) and (dy_edge > 0)
            
            if is_horizontal_right or is_sloped_up:
                edge_val = val_next - val_curr
            else:
                edge_val = val_curr - val_next

            # Accumulate signed contribution
            total_field += np.sign(dx_edge) * edge_val

        return total_field