"""
2D polygon geometry and 3D physical voxel-spacing helpers.
"""

from medimg_annotator.imports import np

def resample_points(raw_pts, n_pts=25):
    """Uniformly resample a 2D polyline to n_pts vertices along arc length."""
    if len(raw_pts) < n_pts:
        return list(raw_pts)
    lengths = []
    total_len = 0.0
    for i in range(len(raw_pts) - 1):
        p1, p2 = raw_pts[i], raw_pts[i + 1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        dist = (dx * dx + dy * dy) ** 0.5
        lengths.append(dist)
        total_len += dist

    if total_len == 0:
        return list(raw_pts)

    step = total_len / n_pts
    new_pts = [raw_pts[0]]
    curr_dist, target = 0.0, step

    for i in range(len(raw_pts) - 1):
        seg_len, p1, p2 = lengths[i], raw_pts[i], raw_pts[i + 1]
        while curr_dist + seg_len >= target and len(new_pts) < n_pts:
            overshoot = target - curr_dist
            t = overshoot / seg_len if seg_len > 0 else 0
            nx = p1[0] + t * (p2[0] - p1[0])
            ny = p1[1] + t * (p2[1] - p1[1])
            new_pts.append((nx, ny))
            target += step
        curr_dist += seg_len

    while len(new_pts) < n_pts:
        new_pts.append(raw_pts[-1])
    return new_pts

def chaikins_smooth(pts, iterations=4):
    """Apply closed-loop Chaikin corner-cutting subdivision to smooth polygon vertices."""
    if len(pts) < 3:
        return list(pts)
    curr = list(pts)
    for _ in range(iterations):
        nxt = []
        n = len(curr)
        for i in range(n):
            p0, p1 = curr[i], curr[(i + 1) % n]
            qx, qy = p0[0] * 0.75 + p1[0] * 0.25, p0[1] * 0.75 + p1[1] * 0.25
            rx, ry = p0[0] * 0.25 + p1[0] * 0.75, p0[1] * 0.25 + p1[1] * 0.75
            nxt.extend([(qx, qy), (rx, ry)])
        curr = nxt
    return curr

def get_voxel_spacing_mm(header=None, affine=None):
    """Extract physical voxel spacing (dx, dy, dz) in millimeters from header or affine."""
    dx, dy, dz = 1.0, 1.0, 1.0
    if header is not None:
        try:
            z = header.get_zooms()
            if len(z) >= 1 and z[0] > 0 and not np.isnan(z[0]):
                dx = float(abs(z[0]))
            if len(z) >= 2 and z[1] > 0 and not np.isnan(z[1]):
                dy = float(abs(z[1]))
            if len(z) >= 3 and z[2] > 0 and not np.isnan(z[2]):
                dz = float(abs(z[2]))
            return dx, dy, dz
        except Exception:
            pass
    if affine is not None:
        try:
            norms = np.linalg.norm(affine[:3, :3], axis=0)
            if len(norms) >= 3:
                if norms[0] > 0:
                    dx = float(norms[0])
                if norms[1] > 0:
                    dy = float(norms[1])
                if norms[2] > 0:
                    dz = float(norms[2])
        except Exception:
            pass
    return dx, dy, dz
