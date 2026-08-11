"""Trajectory cleaning and smoothing.

Cleaning happens at two distinct stages of the pipeline, and the functions are
grouped below accordingly:

1. **2D, per camera, before triangulation** — operating on (x, y) pixel
   midpoints straight out of the detector, where `None` means "no detection in
   this frame". `remove_bad_points` and `interpolate_nones`.
2. **3D, after triangulation** — operating on (x, y, z) world points in meters.
   `smooth_trajectory`.

`moving_average` is a general helper used by the serve/reception detector in
`events.py`. Note it is NOT interchangeable with `smooth_trajectory`: it takes a
dense array with no `None` values and returns a plain windowed mean, whereas
`smooth_trajectory` preserves gaps and rejects neighbours that are too far
away. The two used to share the name `moving_average` in different modules,
which made it easy to import the wrong one.
"""

from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: 2D pixel-space cleaning, applied per camera before triangulation
# ---------------------------------------------------------------------------

def remove_bad_points(points):
    cleaned_points = points.copy()
    max_frame_distance = 100 # kanske får justera detta beroende på hur snabbt bollen rör sig i videon
    last_valid_point = None
    last_valid_index = -1

    for i, point in enumerate(points):
        if point is not None:
            if last_valid_point is not None:
                distance = np.linalg.norm(np.array(point) - np.array(last_valid_point))
                if distance > max_frame_distance * (i - last_valid_index):
                    cleaned_points[i] = None
                    continue
            last_valid_point = point
            last_valid_index = i

    return cleaned_points


def interpolate_nones(points: List[Optional[Tuple[int, int]]]) -> List[Optional[Tuple[int, int]]]:
    result = points[:]
    n = len(points)
    i = 0

    while i < n:
        if result[i] is None:
            start = i

            # Find end of None sequence
            while i < n and result[i] is None:
                i += 1
            end = i  # first non-None after sequence

            length = end - start

            # Check constraints: ≤10 and bounded by valid points
            if length <= 60 and start > 0 and end < n:
                p0 = result[start - 1]
                p1 = result[end]

                if p0 is not None and p1 is not None:
                    x0, y0 = p0
                    x1, y1 = p1

                    # Interpolate
                    for k in range(1, length + 1):
                        t = k / (length + 1)
                        x = round(x0 + t * (x1 - x0))
                        y = round(y0 + t * (y1 - y0))
                        result[start + k - 1] = (x, y)
        else:
            i += 1

    return result


# ---------------------------------------------------------------------------
# Stage 2: 3D world-space smoothing, applied after triangulation
# ---------------------------------------------------------------------------

def smooth_trajectory(points):
    """Average each point with its two neighbours, keeping None gaps intact.

    Neighbours further away than the thresholds below are treated as outliers
    and replaced by the centre point before averaging. Was `moving_average` in
    the old `detection_stuff` module.
    """
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


# ---------------------------------------------------------------------------
# General helper
# ---------------------------------------------------------------------------

def moving_average(points, window_size=5):
    """Plain centred windowed mean over a dense array of points (no Nones)."""
    points = np.array(points)
    smoothed = []

    for i in range(len(points)):
        start = max(0, i - window_size // 2)
        end = min(len(points), i + window_size // 2 + 1)
        smoothed.append(points[start:end].mean(axis=0))

    return np.array(smoothed)
