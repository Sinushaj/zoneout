"""3D reconstruction: two camera views of the ball -> one 3D world point.

Each camera's 2D pixel detection is turned into a world-space ray via solvePnP
(using that camera's intrinsics plus the clicked court reference points), and
the two rays are intersected by least squares.
"""

import numpy as np
import cv2

from .config import read_csv_to_tuples_np
from .court import CALIBRATION_POINTS


# Camera intrinsics, per camera.
GOPRO_CAMERA_MATRIX = np.array([
    [960.0,   0.0, 960.0],
    [  0.0, 960.0, 540.0],
    [  0.0,   0.0,   1.0]
])

ZVE10_CAMERA_MATRIX = np.array([
    [2371,    0, 960],
    [   0, 2008, 540],
    [   0,    0,   1]
], dtype=float)


def screen_point_to_world_ray(image_points, world_points, K, query_pixel):
    """
    Compute 3D ray from a screen pixel.

    Parameters:
        image_points: list of (x, y)
        world_points: list of (x, y, z)
        K: camera matrix (3x3)
        query_pixel: (u, v)

    Returns:
        origin: (3,) camera position in world coords
        direction: (3,) normalized direction vector in world coords
    """

    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)

    dist_coeffs = np.zeros((4, 1))

    # --- Solve PnP ---
    success, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    # --- Convert rotation ---
    R, _ = cv2.Rodrigues(rvec)

    # --- Camera center in world ---
    C = -R.T @ tvec

    # --- Pixel to normalized camera coordinates ---
    u, v = query_pixel
    pixel_h = np.array([u, v, 1.0])

    K_inv = np.linalg.inv(K)
    ray_cam = K_inv @ pixel_h

    # --- Convert to world direction ---
    ray_world = R.T @ ray_cam

    # Normalize direction
    ray_world = ray_world / np.linalg.norm(ray_world)

    return C.flatten(), ray_world.flatten()




def intersect_rays(rays):
    """
    Compute least-squares intersection point of multiple 3D rays.

    Parameters:
        rays: list of (origin, direction)
              origin: (3,)
              direction: (3,)

    Returns:
        point: (3,) numpy array (best intersection estimate)
    """

    A = np.zeros((3, 3))
    b = np.zeros(3)

    for origin, direction in rays:
        origin = np.array(origin, dtype=float)
        direction = np.array(direction, dtype=float)

        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Projection matrix
        I = np.eye(3)
        P = I - np.outer(direction, direction)

        A += P
        b += P @ origin

    # Solve A x = b
    point = np.linalg.solve(A, b)

    return point



def point_from_camera_coordinates(gopro_coordinate, zve10_coordinate):
    """Triangulate one 3D world point from a pixel in each camera."""
    court_points = CALIBRATION_POINTS

    gopro_court_points = read_csv_to_tuples_np('gopro_points.csv')
    gopro_origin, gopro_ray = screen_point_to_world_ray(gopro_court_points, court_points, GOPRO_CAMERA_MATRIX, gopro_coordinate)

    zve10_court_points = read_csv_to_tuples_np('zve10_points.csv')
    zve10_origin, zve10_ray = screen_point_to_world_ray(zve10_court_points, court_points, ZVE10_CAMERA_MATRIX, zve10_coordinate)

    ray_list = [
        (gopro_origin, gopro_ray),
        (zve10_origin, zve10_ray),
    ]

    intersection_estimate = intersect_rays(ray_list)
    return intersection_estimate