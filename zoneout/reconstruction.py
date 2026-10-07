"""3D reconstruction: two camera views of the ball -> one 3D world point.

Each camera's 2D pixel detection is turned into a world-space ray via solvePnP
(using that camera's intrinsics plus the clicked court reference points), and
the two rays are intersected by least squares.

Two rays in 3D essentially never meet exactly, so the least-squares
"intersection" always leaves a residual. How badly they missed is the most
useful signal in the whole pipeline and is returned alongside the point rather
than discarded: if the two rays disagree, at least one of the two detections is
not the ball, and no amount of single-camera reasoning could have told you that.

The disagreement is measured as **reprojection error in pixels** rather than as
a distance in meters — see MAX_REPROJECTION_ERROR for why. The metric residual
is still reported because it is the one a human can interpret.

`match_detections` builds on this. Given several candidate detections per frame
per camera (see `detection.process_video`), it tries every pairing and keeps
the one that triangulates to a plausible point in space, or nothing at all if
no pairing does.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, List, Optional, Sequence

import numpy as np
import cv2

from .cameras import GOPRO_CAMERA_MATRIX, ZVE10_CAMERA_MATRIX, matrix_for
from .config import read_csv_to_tuples_np
from .court import BALL_VOLUME_X, BALL_VOLUME_Y, BALL_VOLUME_Z, CALIBRATION_POINTS

if TYPE_CHECKING:
    # Only needed for the annotations below. Importing `detection` for real
    # would pull in ultralytics and torch, and this module is pure geometry —
    # it stays usable for debugging calibration without paying for that.
    from .detection import Detection


# Camera intrinsics now come from `zoneout.cameras`, which builds them from the
# camera model recorded by `update_parameters.py` and falls back to the two
# matrices that used to live here when nothing has been recorded. They are
# re-exported above so that `reconstruction.GOPRO_CAMERA_MATRIX` still names
# what it always did.

# Per-camera calibration: which side of the court the camera stands on, which
# decides the intrinsics, and the file holding the clicked pixels that
# correspond, in order, to court.CALIBRATION_POINTS.
#
# The keys are historical: "gopro" and "zve10" are the bodies that happened to
# be filming when this was written, and they now mean nothing more than the
# sideline and baseline positions - either position can hold any of the models
# in `cameras.MODELS`.
CAMERAS = {
    "gopro": ("sideline", "gopro_points.csv"),
    "zve10": ("baseline", "zve10_points.csv"),
}

# How far, in pixels, the triangulated point may reproject from the detections
# that produced it. The gate is in pixels rather than meters on purpose: the
# metric distance between the rays grows with how far the ball is from the
# cameras, so any fixed metric threshold is simultaneously too tight for a ball
# at the far baseline and too loose for one at the net. Reprojection error does
# not have that problem.
#
# This is deliberately loose, and it must stay loose. An earlier version gated
# at 15 px, chosen to sit in the valley of the measured error distribution, and
# it was a disaster on real footage: across receptions 1-3 it matched only
# 64-165 frames of 299, the median confidence of the boxes it selected was
# 0.37-0.55, and in 31-83% of frames it passed over a camera's top box. The ball
# was frequently detected confidently in *both* views and still lost, because
# some obscure low-confidence pairing happened to be geometrically tidier.
#
# The lesson: the geometric gate's job is to reject the impossible, not to
# adjudicate between plausible candidates. Calibration error alone — hand-clicked
# reference points, unmodelled GoPro lens distortion — puts correct pairs tens of
# pixels out, so a gate tight enough to be selective is tight enough to throw
# away the truth. At 40 px the same three receptions match 201-268 frames with
# median selected confidence 0.77-0.78 and under 3% physically impossible
# frame-to-frame jumps.
#
# Tightening this is only justified once the GoPro's lens distortion is actually
# calibrated, which would pull correct pairs back toward zero error.
MAX_REPROJECTION_ERROR = 40.0

# Pairings are ranked by confidence weighted by geometric fit, with this as the
# scale (in pixels) of "fit". **Confidence is the primary signal**: this model
# scores the ball high and other objects low, so its own opinion is the best
# evidence available about which box is the ball, and geometry should not be
# allowed to overrule it.
#
# The scale is therefore set wide relative to the errors correct pairs actually
# show, which makes the geometric term a gentle tiebreak rather than a veto.
# It is not zero, because it still settles the one case confidence cannot: both
# cameras looking at the *same* wrong object. A player's head is visible from
# both angles and triangulates with near-zero error, so if two pairings are
# similarly confident the tidier one is the better bet.
#
# Measured on receptions 1-3, varying this between 15 px, 25 px and "off"
# changes the result by at most one percentage point of bad frames — which is
# the point. Confidence is doing the work.
GEOMETRIC_SCORE_SCALE = 25.0


@dataclass(frozen=True)
class Match:
    """One frame's chosen pair of detections and the point they triangulate to."""

    point: np.ndarray        # (3,) world coordinates in meters
    reprojection_error: float  # pixels; what the gate is applied to
    residual: float          # meters, ray-to-point distance; for human reading
    gopro_index: int         # index into that frame's gopro candidate list
    zve10_index: int


@lru_cache(maxsize=None)
def _camera_pose(camera: str):
    """Solve, once per camera, for where that camera sits and how it is oriented.

    The pose depends only on the config files — the clicked points and the
    camera model behind the intrinsics — which do not change during a run, so
    this is cached; it used to be re-solved (and the CSV re-read) for every
    single frame. Restart the process after re-running the calibration tools.
    The cache is also why each camera announces its intrinsics exactly once
    per run rather than once per frame.

    Returns:
        (R, C, K, K_inv): world<-camera rotation, camera center in world
        coordinates, and the intrinsics both ways round.
    """
    position, points_file = CAMERAS[camera]
    K = matrix_for(position)

    image_points = np.array(read_csv_to_tuples_np(points_file), dtype=np.float32)
    world_points = np.array(CALIBRATION_POINTS, dtype=np.float32)

    # Lens distortion is not modelled. This is a real error source, worst at the
    # frame edges — where the serve contact and the reception tend to be.
    dist_coeffs = np.zeros((4, 1))

    success, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError(f"solvePnP failed for camera {camera!r}")

    R, _ = cv2.Rodrigues(rvec)
    C = (-R.T @ tvec).flatten()

    return R, C, K, np.linalg.inv(K)


def world_to_pixel(camera: str, point):
    """Project a world point back into one camera.

    Returns None if the point is behind the camera, which doubles as the
    "is this reconstruction even in front of the lens" check — the
    least-squares intersection is perfectly happy to put a point behind a
    camera for a badly mismatched pair of detections.
    """
    R, C, K, _ = _camera_pose(camera)

    camera_coords = R @ (np.asarray(point, dtype=float) - C)
    if camera_coords[2] <= 0:
        return None

    homogeneous = K @ camera_coords
    return homogeneous[:2] / homogeneous[2]


def cameras_near_top_edge(point, margin):
    """The cameras that show `point` within `margin` pixels of the top of the picture.

    Above the picture counts too. Pixels are the 1920x1080 space the calibration
    is written in, so the top edge is row 0 whatever the source video is. A
    point behind a camera is not near its top edge.

    Returns a set of camera names, the keys of `CAMERAS`.
    """
    near = set()
    for camera in CAMERAS:
        pixel = world_to_pixel(camera, point)
        if pixel is not None and pixel[1] < margin:
            near.add(camera)
    return near


def reprojection_error(point, gopro_pixel, zve10_pixel) -> float:
    """How far, in pixels, `point` reprojects from the detections it came from.

    The worse of the two cameras is what counts, so one badly disagreeing view
    cannot be averaged away by the other. Returns infinity if the point falls
    behind either camera.
    """
    errors = []
    for camera, pixel in (("gopro", gopro_pixel), ("zve10", zve10_pixel)):
        reprojected = world_to_pixel(camera, point)
        if reprojected is None:
            return float("inf")
        errors.append(float(np.linalg.norm(reprojected - np.asarray(pixel, dtype=float))))

    return max(errors)


def pixel_to_world_ray(camera: str, query_pixel):
    """Compute the 3D world-space ray through a pixel of one camera.

    Parameters:
        camera: key into CAMERAS, i.e. "gopro" or "zve10"
        query_pixel: (u, v)

    Returns:
        origin: (3,) camera position in world coords
        direction: (3,) normalized direction vector in world coords
    """
    R, C, _, K_inv = _camera_pose(camera)

    u, v = query_pixel
    ray_cam = K_inv @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam

    return C, ray_world / np.linalg.norm(ray_world)


def intersect_rays(rays):
    """
    Compute least-squares intersection point of multiple 3D rays.

    Parameters:
        rays: list of (origin, direction)
              origin: (3,)
              direction: (3,)

    Returns:
        point: (3,) numpy array (best intersection estimate)
        residual: float, the RMS perpendicular distance from that point to the
            rays. Zero would mean they met exactly. For the two-camera case this
            is half the closest-approach distance between the two rays.
    """

    A = np.zeros((3, 3))
    b = np.zeros(3)
    projections = []

    for origin, direction in rays:
        origin = np.array(origin, dtype=float)
        direction = np.array(direction, dtype=float)

        # Normalize direction
        direction = direction / np.linalg.norm(direction)

        # Projection matrix onto the plane perpendicular to the ray
        I = np.eye(3)
        P = I - np.outer(direction, direction)

        A += P
        b += P @ origin
        projections.append((P, origin))

    # lstsq rather than solve: near-parallel rays make A singular, and returning
    # a garbage point that the plausibility gate then rejects beats raising.
    point = np.linalg.lstsq(A, b, rcond=None)[0]

    distances = [np.linalg.norm(P @ (point - origin)) for P, origin in projections]
    residual = float(np.sqrt(np.mean(np.square(distances))))

    return point, residual


def point_from_camera_coordinates(gopro_coordinate, zve10_coordinate):
    """Triangulate one 3D world point from a pixel in each camera.

    Returns the point only. Use `triangulate` when you also want the residual.
    """
    point, _ = triangulate(gopro_coordinate, zve10_coordinate)
    return point


def triangulate(gopro_coordinate, zve10_coordinate):
    """Triangulate a pixel pair, returning both the world point and the residual."""
    gopro_origin, gopro_ray = pixel_to_world_ray("gopro", gopro_coordinate)
    zve10_origin, zve10_ray = pixel_to_world_ray("zve10", zve10_coordinate)

    return intersect_rays([
        (gopro_origin, gopro_ray),
        (zve10_origin, zve10_ray),
    ])


def _is_plausible(point) -> bool:
    """Is this world point somewhere a volleyball could actually be?"""
    x, y, z = point
    return (BALL_VOLUME_X[0] <= x <= BALL_VOLUME_X[1]
            and BALL_VOLUME_Y[0] <= y <= BALL_VOLUME_Y[1]
            and BALL_VOLUME_Z[0] <= z <= BALL_VOLUME_Z[1])


def match_frame(
    gopro_candidates: Sequence["Detection"],
    zve10_candidates: Sequence["Detection"],
    max_error: float = MAX_REPROJECTION_ERROR,
) -> Optional[Match]:
    """Pick the one pairing of candidates that is geometrically consistent.

    Every gopro candidate is tried against every zve10 candidate. A pairing has
    to clear two hard gates: the triangulated point must reproject close to both
    detections, and it must lie somewhere a volleyball could plausibly be.

    Among the survivors the winner is the one with the highest product of
    confidences weighted by geometric fit — see GEOMETRIC_SCORE_SCALE for why
    ranking on either signal by itself gets beaten.

    Returns None if no pairing survives, meaning the frame stays a gap.
    """
    best = None
    best_score = -1.0

    for g_index, gopro in enumerate(gopro_candidates):
        gopro_ray = pixel_to_world_ray("gopro", gopro.midpoint)

        for z_index, zve10 in enumerate(zve10_candidates):
            zve10_ray = pixel_to_world_ray("zve10", zve10.midpoint)

            point, residual = intersect_rays([gopro_ray, zve10_ray])

            if not _is_plausible(point):
                continue

            error = reprojection_error(point, gopro.midpoint, zve10.midpoint)
            if error > max_error:
                continue

            fit = np.exp(-0.5 * (error / GEOMETRIC_SCORE_SCALE) ** 2)
            score = gopro.confidence * zve10.confidence * fit
            if score > best_score:
                best_score = score
                best = Match(point=point, reprojection_error=error,
                             residual=residual,
                             gopro_index=g_index, zve10_index=z_index)

    return best


def match_detections(
    gopro_frames: Sequence[Sequence["Detection"]],
    zve10_frames: Sequence[Sequence["Detection"]],
    max_error: float = MAX_REPROJECTION_ERROR,
) -> List[Optional[Match]]:
    """Run `match_frame` over a whole clip.

    The two cameras' clips are assumed to be frame-aligned already (see
    `scout.get_reception_start_frame`); the result is truncated to the shorter
    of the two. Each frame is matched independently — there is no temporal
    continuity term yet, so a frame where the ball is genuinely hidden in one
    view simply produces None.
    """
    return [
        match_frame(gopro_frames[i], zve10_frames[i], max_error)
        for i in range(min(len(gopro_frames), len(zve10_frames)))
    ]


def match_summary(matches: Sequence[Optional[Match]]) -> str:
    """One line describing how the geometric gate did over a whole clip.

    Lives here rather than in a pipeline because every pipeline wants the same
    reading of the same thing, and because it is the number that says where to
    look when a reconstruction is wrong: if the reprojection errors reported
    regularly sit near MAX_REPROJECTION_ERROR, the calibration is what needs
    fixing, not the gate.
    """
    matched = [m for m in matches if m is not None]
    if not matched:
        return f"matched 0/{len(matches)} frames — nothing survived the geometric gate"

    errors = [m.reprojection_error for m in matched]
    residuals = [m.residual for m in matched]
    return (f"matched {len(matched)}/{len(matches)} frames,"
            f" reprojection median {np.median(errors):.1f} px /"
            f" 90th pct {np.percentile(errors, 90):.1f} px"
            f" (median {np.median(residuals):.2f} m apart in space)")
