import numpy as np
from scipy.spatial import cKDTree

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
        
        # Pre-calculate 为每个patch 拟合
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

            # 考虑空间上相邻质心的patch
            centroids_active = np.array([patch_metrics[l]['centroid'] for l in current_labels_active])
            centroid_tree_active = cKDTree(centroids_active)
            active_idx_to_label = {i: l for i, l in enumerate(current_labels_active)}
            
            # 找到可合并的patch
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

            # 执行合并
            if best_merge:
                l_winner, l_loser, pts_new = best_merge
                labels[labels == l_loser] = l_winner
                patch_points[l_winner] = pts_new
                patch_metrics[l_winner]['error'] = best_score / len(pts_new)
                patch_metrics[l_winner]['size'] = len(pts_new)
                del patch_metrics[l_loser]
                del patch_points[l_loser]
                merges_made = True
                
        # 压缩重编号
        final_labels = np.copy(labels)
        current_map = {old: new for new, old in enumerate(np.unique(final_labels))}
        for old, new in current_map.items(): final_labels[final_labels == old] = new
        return final_labels