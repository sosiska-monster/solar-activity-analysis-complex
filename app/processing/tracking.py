import numpy as np
from scipy.spatial import KDTree
from collections import defaultdict

class SolarPhysics:
    @staticmethod
    def predict_position(x, y, cx, cy, R, delta_t_days):
        if delta_t_days == 0: return x, y
        X, Y = (x - cx) / float(R), (cy - y) / float(R) 
        r_sq = X**2 + Y**2
        if r_sq >= 0.95: return x, y 
        Z = np.sqrt(1.0 - r_sq)
        phi, theta = np.arcsin(Y), np.arctan2(X, Z)
        omega_deg = 14.713 - 2.396 * (np.sin(phi)**2) - 1.787 * (np.sin(phi)**4)
        theta_new = theta + np.radians(omega_deg) * delta_t_days
        if np.cos(theta_new) < -0.15: return None 
        return int(cx + np.cos(phi) * np.sin(theta_new) * R), int(cy - Y * R)

class SpotTracker:
    def __init__(self, sun_radius, center, max_dist_factor=0.15):
        self.R = sun_radius
        self.cx, self.cy = center
        self.max_dist = sun_radius * max_dist_factor

    def match_frames(self, prev_spots, curr_spots, delta_t_days):
        if not prev_spots:
            for s in curr_spots: s['is_new'] = True
            return curr_spots
        predictions, valid_prev_indices = [], []
        for i, s in enumerate(prev_spots):
            pos = SolarPhysics.predict_position(s['x'], s['y'], self.cx, self.cy, self.R, delta_t_days)
            predictions.append(pos if pos else (s['x'], s['y']))
            valid_prev_indices.append(i)
        
        tree = KDTree(np.array(predictions))
        dists, idxs = tree.query(np.array([[s['x'], s['y']] for s in curr_spots]))
        
        prev_to_curr, curr_to_prev = defaultdict(list), defaultdict(list)
        for i_curr, (dist, p_idx) in enumerate(zip(dists, idxs)):
            if dist <= self.max_dist:
                i_prev = valid_prev_indices[p_idx]
                prev_to_curr[i_prev].append(i_curr); curr_to_prev[i_curr].append(i_prev)

        used_prev, used_curr = set(), set()
        for i_prev, curr_indices in prev_to_curr.items():
            if len(curr_indices) == 1:
                i_curr = curr_indices[0]
                curr_spots[i_curr]['track_id'] = prev_spots[i_prev]['track_id']
                used_prev.add(i_prev); used_curr.add(i_curr)
            else:
                for i_curr in curr_indices:
                    curr_spots[i_curr]['parent_track'] = prev_spots[i_prev]['track_id']
                    used_curr.add(i_curr)
                used_prev.add(i_prev)

        for i_curr, prev_indices in curr_to_prev.items():
            if len(prev_indices) > 1 and i_curr not in used_curr:
                curr_spots[i_curr]['parent_tracks'] = [prev_spots[i]['track_id'] for i in prev_indices]
                used_curr.add(i_curr)
                for i_prev in prev_indices: used_prev.add(i_prev)

        for i, spot in enumerate(curr_spots):
            if i not in used_curr: spot['is_new'] = True
        return curr_spots