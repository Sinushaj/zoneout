"""Reading and writing the small CSV files that configure a run.

- `video_filepaths.csv`        - sideline / baseline source video paths
- `first_reception_frames.csv` - manually synced start frame per camera
- `gopro_points.csv`, `zve10_points.csv` - clicked court calibration pixels

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
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # First row: headers
        writer.writerow(["sideline", "baseline"])

        # Second row: values
        writer.writerow([side, base])
