import numpy as np
import cv2
# from point_finder import get_screen_coordinates

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
    

def get_screen_coordinates(video_file, n):
    # --- Load video and grab first frame ---
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise IOError("Cannot open video file")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise IOError("Cannot read first frame")

    # Resize to 1920x1080 if needed (to match your camera matrix)
    frame = cv2.resize(frame, (1920, 1080))

    # --- Storage for clicked points ---
    points = []

    # --- Mouse callback ---
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < n:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

            # Draw a small circle where clicked
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Frame", frame)

    # --- Display and collect clicks ---
    cv2.namedWindow("Frame", cv2.WINDOW_NORMAL) # TODO kolla om detta hjälper
    cv2.imshow("Frame", frame)
    cv2.resizeWindow("Frame", 1920, 1080) # TODO: kolla om detta hjälper
    cv2.setMouseCallback("Frame", click_event)

    print(f"Click {n} points on the image...")

    while True:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # Exit when enough points are collected
        if len(points) >= n:
            break

        # Optional: press ESC to quit early
        if key == 27:
            break

    cv2.destroyAllWindows()

    # --- Output ---
    print("\nCollected points:")
    print(points)
    return points


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

# print(read_csv_to_tuples_np('gopro_points.csv'))