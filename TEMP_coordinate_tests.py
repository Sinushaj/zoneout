import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def write_coords(path, coords):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(coords)

def read_coords(path):
    with open(path) as f:
        return [tuple(map(float, row)) for row in csv.reader(f)]

my_list = read_coords('temp.csv')


'''
def plot_points(my_list):
    x = [p[0] for p in my_list]
    y = [p[1] for p in my_list]
    z = [p[2] for p in my_list]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Rectangle in XY plane (z=0) ---
    rect_x = [0, 9, 9, 0, 0]
    rect_y = [-9, -9, 9, 9, -9]
    rect_z = [0, 0, 0, 0, 0]

    all_x = np.concatenate([x, rect_x])
    all_y = np.concatenate([y, rect_y])
    all_z = np.concatenate([z, rect_z])

    ax.set_box_aspect([
        np.ptp(all_x),
        np.ptp(all_y),
        np.ptp(all_z)
    ])  
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    # Scatter points
    ax.scatter(x, y, z)

    ax.plot(rect_x, rect_y, rect_z, color='black')
    # --- Vertical lines ---
    lines = [
        [(0, 0, 0), (9, 0, 0)],
        [(0, 3, 0), (9, 3, 0)],
        [(0,-3, 0), (9,-3, 0)]
    ]

    for line in lines:
        lx = [line[0][0], line[1][0]]
        ly = [line[0][1], line[1][1]]
        lz = [line[0][2], line[1][2]]
        ax.plot(lx, ly, lz, color = 'black')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
'''

'''
def ransac_piecewise_fit(points, residual_threshold=0.8, min_samples=5):
    """
    Cleans 2D trajectory data using iterative RANSAC line fitting.

    Args:
        points: list of (x, y)
        residual_threshold: max distance to be considered an inlier
        min_samples: minimum samples for RANSAC

    Returns:
        cleaned_points: list of (x, y) reconstructed from fitted segments
    """

    points = np.array(points)
    remaining = points.copy()
    segments = []

    while len(remaining) >= min_samples:
        X = remaining[:, 0].reshape(-1, 1)
        y = remaining[:, 1]

        # Fit line y = ax + b using RANSAC
        model = RANSACRegressor(
            min_samples=min_samples,
            residual_threshold=residual_threshold
        )
        model.fit(X, y)

        inlier_mask = model.inlier_mask_
        outlier_mask = ~inlier_mask

        if np.sum(inlier_mask) < min_samples:
            break

        inliers = remaining[inlier_mask]
        segments.append((model, inliers))

        # Remove inliers and continue
        remaining = remaining[outlier_mask]

    # Reconstruct cleaned trajectory
    cleaned_points = []

    for model, segment in segments:
        X_seg = segment[:, 0].reshape(-1, 1)
        y_pred = model.predict(X_seg)

        for x, y in zip(X_seg.flatten(), y_pred):
            cleaned_points.append((float(x), float(y)))

    # Sort by x (or time if implicit)
    cleaned_points.sort(key=lambda p: p[0])

    return cleaned_points
'''








'''
def fit_line_pca(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]
    return mean, direction

def point_line_distance(points, mean, direction):
    diff = points - mean
    proj = diff @ direction
    proj_point = np.outer(proj, direction)
    return np.linalg.norm(diff - proj_point, axis=1)

def ransac_line(points, threshold=0.5, iterations=100):
    best_inliers = []

    for _ in range(iterations):
        sample_idx = np.random.choice(len(points), 2, replace=False)
        sample = points[sample_idx]

        mean, direction = fit_line_pca(sample)
        distances = point_line_distance(points, mean, direction)

        inliers = points[distances < threshold]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    return np.array(best_inliers)

def ransac_piecewise_general(points, threshold=0.1, min_points=5):
    points = np.array(points)
    remaining = points.copy()
    segments = []

    while len(remaining) >= min_points:
        inliers = ransac_line(remaining, threshold=threshold)

        if len(inliers) < min_points:
            break

        segments.append(inliers)

        mask = np.ones(len(remaining), dtype=bool)
        for p in inliers:
            idx = np.where((remaining == p).all(axis=1))[0]
            mask[idx] = False

        remaining = remaining[mask]

    return segments


list_2d = [elem[:2] for elem in my_list]
def filter_coords(coords):
    return [[x, y] for x, y in coords if 0 <= x <= 9 and -9 <= y <= 9]
list_2d = filter_coords(list_2d)

x1 = [elem[0] for elem in list_2d]
y1 = [elem[1] for elem in list_2d]

plt.scatter(x1, y1)
plt.show()

clean_2d_segments = ransac_piecewise_general(list_2d)
for clean_2d in clean_2d_segments:
    x2 = [elem[0] for elem in clean_2d]
    y2 = [elem[1] for elem in clean_2d]

    print(len(clean_2d))
    plt.scatter(x2, y2)
    plt.xlim(0, 9)
    plt.ylim(-9, 9)
    plt.show()
'''

# import numpy as np
# from sklearn.linear_model import RANSACRegressor

def fit_line_pca(points):
    mean = np.mean(points, axis=0)
    centered = points - mean
    _, _, vh = np.linalg.svd(centered)
    direction = vh[0]
    return mean, direction

def point_line_distance(points, mean, direction):
    diff = points - mean
    proj = diff @ direction
    proj_point = np.outer(proj, direction)
    return np.linalg.norm(diff - proj_point, axis=1)

def ransac_segment(points, threshold=0.5, min_samples=5, max_trials=100):
    best_inliers = []
    n = len(points)

    if n < min_samples:
        return None

    for _ in range(max_trials):
        idx = np.random.choice(n, 2, replace=False)
        sample = points[idx]

        mean, direction = fit_line_pca(sample)
        distances = point_line_distance(points, mean, direction)

        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < min_samples:
        return None

    # Refit using all inliers
    inlier_points = points[best_inliers]
    mean, direction = fit_line_pca(inlier_points)

    return best_inliers, mean, direction


def clean_trajectory_ransac(
    points,
    window_size=50,
    step=10,
    threshold=0.3,
    min_segment_size=25
):
    """
    Cleans trajectory using sliding-window RANSAC with segment validation.

    Args:
        points: list of (x, y) ordered in time
        window_size: size of sliding window
        step: step size for window movement
        threshold: distance threshold for inliers
        min_segment_size: minimum accepted segment length (>= 20)

    Returns:
        cleaned_points: list of (x, y)
    """

    points = np.array(points)
    n = len(points)

    segments = []
    used = np.zeros(n, dtype=bool)

    i = 0
    while i < n - window_size:
        window = points[i:i + window_size]

        result = ransac_segment(window, threshold=threshold)

        if result is None:
            i += step
            continue

        inlier_idx_local, mean, direction = result

        # Convert to global indices
        inlier_idx_global = i + inlier_idx_local

        # Keep only contiguous runs (important!)
        inlier_idx_global = np.sort(inlier_idx_global)

        splits = np.where(np.diff(inlier_idx_global) > 1)[0] + 1
        groups = np.split(inlier_idx_global, splits)

        for group in groups:
            if len(group) >= min_segment_size:
                segments.append((group, mean, direction))
                used[group] = True

        i += step

    # ---- Reconstruct cleaned trajectory ----
    cleaned_points = []

    for group, mean, direction in segments:
        pts = points[group]

        # Project points onto fitted line
        diff = pts - mean
        t = diff @ direction
        projected = mean + np.outer(t, direction)

        for p in projected:
            cleaned_points.append(tuple(p))

    # Preserve original order
    cleaned_points = sorted(cleaned_points, key=lambda p: (p[0], p[1]))

    return cleaned_points



list_2d = [elem[:2] for elem in my_list]
def filter_coords(coords):
    return [[x, y] for x, y in coords if 0 <= x <= 9 and -9 <= y <= 9]
list_2d = filter_coords(list_2d)

x1 = [elem[0] for elem in list_2d]
y1 = [elem[1] for elem in list_2d]

plt.scatter(x1, y1)
plt.show()


clean_2d = clean_trajectory_ransac(list_2d)

x2 = [elem[0] for elem in clean_2d]
y2 = [elem[1] for elem in clean_2d]

plt.scatter(x2, y2)
plt.xlim(0, 9)
plt.ylim(-9, 9)
plt.show()

#import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_coords(coords):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 9)
    ax.set_ylim(-9, 9)

    point, = ax.plot([], [], 'o')  # current point
    path, = ax.plot([], [], '-')   # trail

    xs, ys = [], []

    def update(frame):
        x, y = coords[frame]  # works with [x, y] lists
        xs.append(x)
        ys.append(y)
        point.set_data([x], [y])
        path.set_data(xs, ys)
        return point, path

    ani = FuncAnimation(fig, update, frames=len(coords), interval=10)
    plt.show()

animate_coords(clean_2d)