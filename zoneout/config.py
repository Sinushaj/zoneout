"""Reading and writing the small CSV files that configure a run.

- `video_filepaths.csv`        - sideline / baseline source video paths
- `first_reception_frames.csv` - the manual sync anchor: one clicked frame per
  camera, and the `video_time` those frames show. The name is historical: the
  anchor used to be fixed at the match's first reception, and can now be any
  set's first serve or first reception (see `update_parameters.py`). A file
  written before that change has no `video_time` column, and the pipeline reads
  it as the match's first reception, which is what it was.
- `gopro_points.csv`, `zve10_points.csv` - clicked court calibration pixels
- `camera_models.csv`          - which camera is on each side, and the focal
  length it was shot at where the body has an interchangeable lens. Read by
  `zoneout.cameras`, which turns it into the intrinsics for solvePnP.

The interactive tools that *produce* these files live in `zoneout.calibration`.
This module only persists and reads them, so the batch pipeline never has to
pull in GUI code.
"""

import csv

import numpy as np


def write_tuples_to_csv(data, filepath):
    """
    Write a collection of integer tuples to a CSV file.

    Parameters
    ----------
    data : array_like
        A list, tuple, or NumPy array of integer tuples. Common examples:
        - List of tuples: [(1,2), (38,22)]
        - 2D NumPy array: np.array([[1,2],[38,22]])
    filepath : str
        Path to the output CSV file. The file will be overwritten if it
        already exists, or created if it does not.

    Returns
    -------
    None

    Notes
    -----
    - The function uses `numpy.savetxt` with integer formatting ('%d').
    - If `data` is empty, an empty file is created.
    """
    # Convert input to a NumPy array (handles list of tuples, lists, etc.)
    arr = np.asarray(data)

    # Ensure the array is at least 2D (e.g., if input is a single tuple)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Write to CSV with integer formatting, overwriting any existing file
    np.savetxt(filepath, arr, delimiter=',', fmt='%d')


def read_csv_to_tuples_np(filepath):
    try:
        data = np.loadtxt(filepath, delimiter=',', dtype=int)
    except ValueError:
        return []  # empty file
    if data.ndim == 1:
        return [tuple(data)]
    else:
        return [tuple(row) for row in data]


def extract_dict_from_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        rows = list(reader)

        if len(rows) < 2:
            raise ValueError("CSV must have at least two rows")

        keys = rows[0]
        values = rows[1]

        return {key: value for key, value in zip(keys, values)}


def write_to_csv(csv_path, side, base):
    write_dict_to_csv(csv_path, {"sideline": side, "baseline": base})


def write_dict_to_csv(csv_path, values):
    """Write a mapping as the two-row header/values CSV the config files use."""
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # First row: headers
        writer.writerow(list(values.keys()))

        # Second row: values
        writer.writerow(list(values.values()))


def read_sync_anchor(csv_path='first_reception_frames.csv'):
    """The manual sync anchor: the frame each camera shows one known action at.

    Returns a dict with the two `sideline`/`baseline` frames as ints, the
    `video_time` those frames correspond to, and the `set` and `skill` the
    anchor was taken from. `video_time` is None for a file written before the
    anchor became selectable; those files always meant the match's first
    reception, so the caller can look that up instead of demanding a re-sync.
    `set` and `skill` are only there to say what was clicked - nothing computes
    with them except the warning about running a set the sync was not taken in.
    """
    values = extract_dict_from_csv(csv_path)
    video_time = values.get('video_time')

    return {
        'sideline': int(values['sideline']),
        'baseline': int(values['baseline']),
        'video_time': int(video_time) if video_time else None,
        'set': values.get('set') or None,
        'skill': values.get('skill') or None,
    }


# The two camera positions, in the order they are written to the config file.
# They are positions rather than models: the same body can be on either side,
# and which model is there is exactly what the file records.
CAMERA_POSITIONS = ('sideline', 'baseline')


def read_camera_models(csv_path='camera_models.csv'):
    """Which camera is on each side, as {position: (model key, focal length)}.

    The focal length is a float in mm, or None for a body whose lens is fixed.
    A position with nothing recorded is simply absent, so a half-written file
    still says what it does know; `cameras.matrix_for` falls back for the rest.
    """
    values = extract_dict_from_csv(csv_path)
    models = {}

    for position in CAMERA_POSITIONS:
        model = (values.get(f'{position}_model') or '').strip()
        if not model:
            continue

        focal_length = (values.get(f'{position}_focal_length') or '').strip()
        models[position] = (model, float(focal_length) if focal_length else None)

    return models


def write_camera_models(models, csv_path='camera_models.csv'):
    """Write {position: (model key, focal length or None)} back out."""
    values = {}

    for position in CAMERA_POSITIONS:
        model, focal_length = models.get(position, ('', None))
        values[f'{position}_model'] = model
        values[f'{position}_focal_length'] = '' if focal_length is None else focal_length

    write_dict_to_csv(csv_path, values)
