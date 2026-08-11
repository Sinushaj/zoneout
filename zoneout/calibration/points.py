"""Interactive collection of court reference points by clicking a video frame.

Produces the pixel coordinates stored in `gopro_points.csv` / `zve10_points.csv`.
The clicks must follow the order of `zoneout.court.CALIBRATION_POINTS`, since
the two lists form the 2D<->3D correspondence handed to solvePnP.
"""

import cv2


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
