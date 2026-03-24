import numpy as np
import cv2
from point_finder import *

def estimate_camera_position(image_points, world_points, K):
    """
    Estimate camera position from 2D-3D correspondences.

    Parameters:
        image_points: list of (x, y) tuples (pixel coordinates)
        world_points: list of (x, y, z) tuples (real-world coordinates)
        K: 3x3 camera intrinsic matrix (numpy array)

    Returns:
        camera_position: (3,) numpy array in world coordinates
    """

    # Convert to numpy arrays
    image_points = np.array(image_points, dtype=np.float32)
    world_points = np.array(world_points, dtype=np.float32)

    # Assume no lens distortion (reasonable for GoPro Linear mode)
    dist_coeffs = np.zeros((4, 1))

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP failed")

    # Convert rotation vector to matrix
    R, _ = cv2.Rodrigues(rvec)

    # Compute camera position in world coordinates
    camera_position = -R.T @ tvec

    return camera_position.flatten()


video_name = 'GH011611.MP4'

court_points = [
    (0,-3, 0),
    (9,-3, 0),
    (9,3, 0),
    (0,3, 0),
    (0,-9, 0),
    (0,9, 0),
]

gopro_cameramatrix = np.array([
    [960.0,   0.0, 960.0],
    [  0.0, 960.0, 540.0],
    [  0.0,   0.0,   1.0]
])

screen_points = get_screen_coordinates(video_name, 6)
print(screen_points)

gopro_location = estimate_camera_position(screen_points, court_points, gopro_cameramatrix)

print('gopro position:', gopro_location)