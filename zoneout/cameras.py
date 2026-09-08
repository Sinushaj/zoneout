"""Which camera is on each side of the court, and the intrinsics it implies.

`update_parameters.py` records the camera model - and, for an interchangeable
lens, the focal length it was shot at - into `camera_models.csv`, and this
module turns that into the matrix `reconstruction` hands to solvePnP.

**These are approximations from the specification sheet, not calibrations.**
Nothing here measures a lens; it computes what a pinhole camera with that
sensor and that focal length would do. The real lens differs - distortion is
not modelled at all (`reconstruction._camera_pose` passes zero coefficients), a
zoom's marked focal length is nominal, it shifts with focus distance, and
Sony's in-camera lens compensation narrows the field slightly when it is on.
That error is part of why `reconstruction.MAX_REPROJECTION_ERROR` has to stay
loose at 40 px; read the note there before tightening anything on the strength
of these numbers.

The approximation for a Sony body, which is the whole of the arithmetic:

    fx = fy = focal_length_mm * IMAGE_WIDTH / sensor_width_mm

The ZV-E10 and the a6000 carry the same 23.5 x 15.6 mm APS-C sensor, and at
1080p60 with stabilisation off both read out its full width - the ZV-E10 crops
only at 1080p120 (1.14x), at 4K30 (1.23x), and with Active SteadyShot. So the
sensor width in play is the full 23.5 mm, the 16:9 frame is a vertical crop of
the 3:2 sensor rather than a horizontal one, the pixels stay square, and fy
therefore equals fx. The principal point is assumed to be the middle of the
frame, which is the best guess available without calibrating.

Sanity check on that formula: the ZV-E10 matrix that was hardcoded here before
any of this was selectable has fx = 2371, which back-solves to 29.0 mm - a
plausible setting on the 16-50 kit zoom. Its fy = 2008 does not fit any focal
length and is the part that looks fitted rather than measured, since square
pixels make fy = fx.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import read_camera_models

# The frame size the intrinsics are written for. Every frame is resized to it
# before a court point is clicked (`calibration.points.FRAME_SIZE`) and before
# the ball is detected, so a principal point at the middle of this is the
# middle of the pixels that are actually measured.
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# Both bodies use the same APS-C sensor, and at 1080p60 without stabilisation
# crop the whole width of it is read out.
APS_C_SENSOR_WIDTH = 23.5

# --------------------------------------------------------------------------
# The matrices that were hardcoded before the camera became selectable. They
# stay exactly as they were, as the per-position fallback: a run with no
# `camera_models.csv`, or with an unreadable one, reconstructs precisely as it
# did before this module existed rather than failing or guessing.
# --------------------------------------------------------------------------

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

FALLBACK_MATRICES = {
    'sideline': GOPRO_CAMERA_MATRIX,
    'baseline': ZVE10_CAMERA_MATRIX,
}


@dataclass(frozen=True)
class CameraModel:
    """One selectable camera.

    A body with an interchangeable lens has a `sensor_width` and needs a focal
    length to say anything; one with a fixed lens carries its `matrix` instead.
    """

    name: str
    sensor_width: Optional[float] = None
    matrix: Optional[np.ndarray] = None

    @property
    def needs_focal_length(self):
        return self.sensor_width is not None


# The selectable cameras, keyed by what goes in the CSV. Adding a body with the
# same sensor is one line; adding one with a different sensor is one number.
MODELS = {
    'zve10': CameraModel('Sony ZV-E10', sensor_width=APS_C_SENSOR_WIDTH),
    'a6000': CameraModel('Sony a6000', sensor_width=APS_C_SENSOR_WIDTH),
    # A GoPro's lens is fixed, but its field of view is not: the digital lens
    # setting (Wide, Linear, SuperView) changes it, and the footage carries no
    # note of which was used. This is the 90-degree horizontal approximation
    # that has been in the pipeline all along, kept rather than re-derived.
    'gopro': CameraModel('GoPro', matrix=GOPRO_CAMERA_MATRIX),
}


def pinhole_matrix(focal_length_pixels):
    """A camera matrix for a pinhole with square pixels, centred in the frame."""
    return np.array([
        [focal_length_pixels, 0.0, IMAGE_WIDTH / 2],
        [0.0, focal_length_pixels, IMAGE_HEIGHT / 2],
        [0.0, 0.0, 1.0],
    ])


def camera_matrix(model_key, focal_length_mm=None):
    """The intrinsics for one selected camera.

    Args:
        model_key: a key of `MODELS`.
        focal_length_mm: the lens setting, required for a model with a sensor
            width and meaningless for one with a fixed matrix.

    Raises:
        KeyError: no such model.
        ValueError: a model that needs a focal length was given none, or one
            that cannot be a focal length.
    """
    model = MODELS[model_key]

    if not model.needs_focal_length:
        return model.matrix

    if focal_length_mm is None:
        raise ValueError(f"{model.name} needs a focal length in mm")
    if focal_length_mm <= 0:
        raise ValueError(f"focal length must be positive, got {focal_length_mm}")

    return pinhole_matrix(focal_length_mm * IMAGE_WIDTH / model.sensor_width)


def describe(model_key, focal_length_mm=None):
    """How a selection reads in a prompt or a run's output."""
    model = MODELS[model_key]
    if model.needs_focal_length and focal_length_mm:
        return f"{model.name} at {focal_length_mm:g} mm"
    return model.name


def matrix_for(position, csv_path='camera_models.csv'):
    """The intrinsics for the 'sideline' or 'baseline' camera.

    Falls back to the matrix that was hardcoded for that position, and says so,
    whenever the selection cannot be used: no file, nothing recorded for this
    camera, a model name that is not one of ours, or a missing focal length.
    Falling back rather than raising is deliberate - a batch run that takes
    hours should not die on a config file, and the fallback is exactly the
    behaviour the pipeline had before the camera was selectable - but it is
    always announced, because silently reconstructing a match with the wrong
    intrinsics is the one outcome worth being loud about.
    """
    fallback = FALLBACK_MATRICES[position]

    try:
        selected = read_camera_models(csv_path)
    except FileNotFoundError:
        # The normal state until the camera step has been run: not a fault.
        print(f"{position} camera: no {csv_path} yet; using the built-in matrix.")
        return fallback
    except (OSError, ValueError) as error:
        print(f"{position} camera: {csv_path} unreadable ({error});"
              f" using the built-in matrix.")
        return fallback

    if position not in selected:
        print(f"{position} camera: not recorded in {csv_path};"
              f" using the built-in matrix.")
        return fallback

    model_key, focal_length = selected[position]

    try:
        matrix = camera_matrix(model_key, focal_length)
    except KeyError:
        print(f"{position} camera: {model_key!r} is not one of"
              f" {', '.join(MODELS)}; using the built-in matrix.")
        return fallback
    except ValueError as error:
        print(f"{position} camera: cannot use {model_key!r} ({error});"
              f" using the built-in matrix.")
        return fallback

    print(f"{position} camera: {describe(model_key, focal_length)},"
          f" fx = {matrix[0, 0]:.0f} px")
    return matrix
