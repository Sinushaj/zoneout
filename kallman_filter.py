import numpy as np

def clean_ball_trajectory(points, deviation_threshold=2.0, angle_threshold_deg=40.0,
                          min_segment_length=6, max_outlier_skip=3):
    """
    Clean a volleyball trajectory by fitting parabolic arcs, tolerating short outlier runs.

    Parameters
    ----------
    points : list of (float, float, float) or None
        Input trajectory. None indicates missing detection.
    deviation_threshold : float, default=2.0
        Maximum allowed Euclidean distance between an actual point and the
        parabola prediction before classifying as outlier.
    angle_threshold_deg : float, default=30.0
        Maximum allowed angle (degrees) between smoothed actual motion direction
        and the parabola's tangent direction before classifying as outlier.
    min_segment_length : int, default=6
        Minimum number of inlier points needed to keep a fitted parabola segment.
    max_outlier_skip : int, default=2
        Maximum consecutive outlier frames tolerated before splitting the segment.

    Returns
    -------
    list of (float, float, float)
        Cleaned trajectory of same length as input, one 3D point per frame.
    """
    N = len(points)
    data = [p if p is not None else None for p in points]

    # Identify contiguous blocks of valid points
    valid = [p is not None for p in data]
    blocks = []
    start = None
    for i in range(N):
        if valid[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                blocks.append((start, i-1))
                start = None
    if start is not None:
        blocks.append((start, N-1))

    # List to store accepted segments: (segment_start, segment_end, list_of_(global_index, point))
    segments = []

    for blk_start, blk_end in blocks:
        # Gather all points in this block (no None inside)
        blk_indices = list(range(blk_start, blk_end+1))
        blk_points = [data[i] for i in blk_indices]   # all are tuples

        # Greedy segmentation with outlier skipping
        seg_indices = []      # global indices of inliers in current segment
        seg_points = []       # corresponding points
        skip_counter = 0

        # Start with first point as first inlier
        seg_indices.append(blk_indices[0])
        seg_points.append(blk_points[0])

        for idx in range(1, len(blk_indices)):
            global_idx = blk_indices[idx]
            point = blk_points[idx]
            seg_len = len(seg_indices)

            # ---------- Fit parabola to current inliers ----------
            fit_available = seg_len >= 3
            if fit_available:
                # Use global indices as time (frame numbers)
                t_vals = np.array(seg_indices)
                pos_vals = np.array(seg_points)
                try:
                    coeffs_x = np.polyfit(t_vals, pos_vals[:,0], 2)
                    coeffs_y = np.polyfit(t_vals, pos_vals[:,1], 2)
                    coeffs_z = np.polyfit(t_vals, pos_vals[:,2], 2)
                    # Predict at current point's global index
                    pred_x = np.polyval(coeffs_x, global_idx)
                    pred_y = np.polyval(coeffs_y, global_idx)
                    pred_z = np.polyval(coeffs_z, global_idx)
                    pred = np.array([pred_x, pred_y, pred_z])

                    # Compute parabola tangent at last inlier point
                    t_last = seg_indices[-1]
                    # Derivative: 2*a*t + b
                    tangent_x = 2 * coeffs_x[0] * t_last + coeffs_x[1]
                    tangent_y = 2 * coeffs_y[0] * t_last + coeffs_y[1]
                    tangent_z = 2 * coeffs_z[0] * t_last + coeffs_z[1]
                    tangent = np.array([tangent_x, tangent_y, tangent_z])
                except np.linalg.LinAlgError:
                    # Fallback: treat as inlier (conservative)
                    fit_available = False
                    pred = seg_points[-1]
                    tangent = None
            else:
                # Not enough points for parabola – use linear extrapolation from last two inliers
                if seg_len >= 2:
                    p0, p1 = seg_points[-2], seg_points[-1]
                    t0, t1 = seg_indices[-2], seg_indices[-1]
                    if t1 != t0:
                        vel = (p1 - p0) / (t1 - t0)
                        pred = p1 + vel * (global_idx - t1)
                    else:
                        pred = p1
                else:
                    pred = seg_points[-1]
                tangent = None

            # ---------- Evaluate criteria ----------
            is_inlier = False
            if fit_available:
                distance = np.linalg.norm(point - pred)

                # Smoothed actual direction using last two velocities (if available)
                if seg_len >= 4:
                    v1 = seg_points[-1] - seg_points[-2]
                    v2 = seg_points[-2] - seg_points[-3]
                    actual_dir = (v1 + v2) / 2.0
                elif seg_len >= 3:
                    actual_dir = seg_points[-1] - seg_points[-2]
                else:
                    actual_dir = None

                angle = 0.0
                if actual_dir is not None and tangent is not None and np.linalg.norm(tangent) > 1e-6:
                    if np.linalg.norm(actual_dir) > 1e-6:
                        cos_angle = np.dot(actual_dir, tangent) / (np.linalg.norm(actual_dir) * np.linalg.norm(tangent))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle = np.degrees(np.arccos(cos_angle))

                if distance <= deviation_threshold and angle <= angle_threshold_deg:
                    is_inlier = True
            else:
                # Without a reliable fit, we cannot judge; treat as inlier (avoid splitting)
                is_inlier = True

            if is_inlier:
                # Good point: add to segment, reset skip counter
                seg_indices.append(global_idx)
                seg_points.append(point)
                skip_counter = 0
            else:
                # Outlier: increment skip counter
                skip_counter += 1
                if skip_counter > max_outlier_skip:
                    # Too many consecutive outliers – finalize current segment
                    if len(seg_indices) >= min_segment_length:
                        segments.append((blk_start, blk_end, list(zip(seg_indices, seg_points))))
                    # Start new segment with this point as first inlier
                    seg_indices = [global_idx]
                    seg_points = [point]
                    skip_counter = 0

        # End of block: finalize segment if enough inliers
        if len(seg_indices) >= min_segment_length:
            segments.append((blk_start, blk_end, list(zip(seg_indices, seg_points))))

    # ---------- Fit parabola to each accepted segment ----------
    cleaned = np.full((N, 3), np.nan)

    for _, _, seg_data in segments:
        # seg_data: list of (global_index, point)
        indices = [idx for idx, _ in seg_data]
        points_arr = np.array([pt for _, pt in seg_data])
        if len(indices) < 3:
            # Fallback: keep original points
            for idx, pt in seg_data:
                cleaned[idx] = pt
            continue

        try:
            coeffs_x = np.polyfit(indices, points_arr[:,0], 2)
            coeffs_y = np.polyfit(indices, points_arr[:,1], 2)
            coeffs_z = np.polyfit(indices, points_arr[:,2], 2)
        except np.linalg.LinAlgError:
            # Fallback: keep original points
            for idx, pt in seg_data:
                cleaned[idx] = pt
            continue

        # Evaluate parabola for every frame from segment_start to segment_end
        # We need segment_start and segment_end; we can compute from min/max of indices
        seg_start = min(indices)
        seg_end = max(indices)
        for t in range(seg_start, seg_end + 1):
            x_fit = np.polyval(coeffs_x, t)
            y_fit = np.polyval(coeffs_y, t)
            z_fit = np.polyval(coeffs_z, t)
            cleaned[t] = [x_fit, y_fit, z_fit]

    # ---------- Fill remaining gaps (frames not covered by any segment) ----------
    valid_fitted = ~np.isnan(cleaned).any(axis=1)
    if not np.any(valid_fitted):
        return points

    gap_start = None
    for i in range(N):
        if valid_fitted[i]:
            if gap_start is not None:
                gap_end = i - 1
                # Find previous valid point
                prev_idx = gap_start - 1
                while prev_idx >= 0 and not valid_fitted[prev_idx]:
                    prev_idx -= 1
                # Find next valid point
                next_idx = gap_end + 1
                while next_idx < N and not valid_fitted[next_idx]:
                    next_idx += 1

                if prev_idx >= 0 and next_idx < N:
                    t0, t1 = prev_idx, next_idx
                    p0, p1 = cleaned[t0], cleaned[t1]
                    for j in range(gap_start, gap_end+1):
                        alpha = (j - t0) / (t1 - t0)
                        cleaned[j] = p0 + alpha * (p1 - p0)
                elif prev_idx >= 0:
                    for j in range(gap_start, gap_end+1):
                        cleaned[j] = cleaned[prev_idx]
                elif next_idx < N:
                    for j in range(gap_start, gap_end+1):
                        cleaned[j] = cleaned[next_idx]
                gap_start = None
        else:
            if gap_start is None:
                gap_start = i

    if gap_start is not None:
        prev_idx = gap_start - 1
        while prev_idx >= 0 and not valid_fitted[prev_idx]:
            prev_idx -= 1
        if prev_idx >= 0:
            for j in range(gap_start, N):
                cleaned[j] = cleaned[prev_idx]

    # ---------- Convert back to list of tuples ----------
    result = [tuple(row) for row in cleaned]
    return result



def moving_average(points):
    new_points = [points[0]]
    # ta snitt av 3 cons som är nära nog varandra
    # Ersätt först none med närmaste värde
    for i in range(len(points) - 3):
        if points[i + 1] is None:
            new_points.append(None)
            continue
        neighborhood = points[i:i+3]
        # hantera vänsterkant
        if neighborhood[0] is None:
            neighborhood[0] = neighborhood[1]
        if np.linalg.norm(neighborhood[0] - neighborhood[1]) > 1.5:
            neighborhood[0] = neighborhood[1]
        # hantera högerkant
        if neighborhood[2] is None:
            neighborhood[2] = neighborhood[1]
        if np.linalg.norm(neighborhood[2] - neighborhood[1]) > 1:
            neighborhood[2] = neighborhood[1]
        new_points.append(np.mean(neighborhood, axis = 0))
    new_points.append(points[-1])
    return new_points