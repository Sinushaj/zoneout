import numpy as np
from point_finder import get_screen_coordinates

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


change_gopro = False
change_zve10 = False

# spara csv för gopro och cve10 film
if change_gopro:
    gopro_filepath = 'GH011611.MP4'
    gopro_court_points = get_screen_coordinates(gopro_filepath, 6)
    write_tuples_to_csv(gopro_court_points, 'gopro_points.csv')

if change_zve10:
    zve10_filepath = 'C0600.MP4'
    zve10_court_points = get_screen_coordinates(zve10_filepath, 6)
    write_tuples_to_csv(zve10_court_points, 'zve10_points.csv')

print(read_csv_to_tuples_np('gopro_points.csv'))