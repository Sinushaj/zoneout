import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Optional, Tuple



def process_video(
    video_path: str,
    model_path: str,
    conf_threshold: float = 0.5,
    output_path: str = "output.mp4"
):
    """
    Processes a video using a YOLOv8 model.

    Args:
        video_path (str): Path to input video
        model_path (str): Path to YOLOv8 model (.pt)
        conf_threshold (float): Confidence threshold for detection
        output_path (str): Path to save annotated video

    Returns:
        list: List of (x, y) midpoints or None per frame
    """

    # Load model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    midpoints = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        best_box = None
        best_conf = 0

        # Iterate detections
        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf > best_conf:
                    best_conf = conf
                    best_box = box

        if best_box is not None:
            # Extract bounding box
            x1, y1, x2, y2 = map(int, best_box.xyxy[0])

            # Compute midpoint
            mx = int((x1 + x2) / 2)
            my = int((y1 + y2) / 2)

            midpoints.append((mx, my))

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (mx, my), 5, (0, 0, 255), -1)

            # Optional: confidence label
            label = f"{best_conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            midpoints.append(None)

        # Write frame
        out.write(frame)

    cap.release()
    out.release()

    return midpoints



def remove_bad_points(points):
    cleaned_points = points.copy()
    max_frame_distance = 100 # kanske får justera detta beroende på hur snabbt bollen rör sig i videon
    last_valid_point = None
    last_valid_index = -1

    for i, point in enumerate(points):
        if point is not None:
            if last_valid_point is not None:
                distance = np.linalg.norm(np.array(point) - np.array(last_valid_point))
                if distance > max_frame_distance * (i - last_valid_index):
                    cleaned_points[i] = None
                    continue
            last_valid_point = point
            last_valid_index = i
    
    return cleaned_points
    


def interpolate_nones(points: List[Optional[Tuple[int, int]]]) -> List[Optional[Tuple[int, int]]]:
    result = points[:]
    n = len(points)
    i = 0

    while i < n:
        if result[i] is None:
            start = i

            # Find end of None sequence
            while i < n and result[i] is None:
                i += 1
            end = i  # first non-None after sequence

            length = end - start

            # Check constraints: ≤10 and bounded by valid points
            if length <= 60 and start > 0 and end < n:
                p0 = result[start - 1]
                p1 = result[end]

                if p0 is not None and p1 is not None:
                    x0, y0 = p0
                    x1, y1 = p1

                    # Interpolate
                    for k in range(1, length + 1):
                        t = k / (length + 1)
                        x = round(x0 + t * (x1 - x0))
                        y = round(y0 + t * (y1 - y0))
                        result[start + k - 1] = (x, y)
        else:
            i += 1

    return result





'''
def temporal_median_filter(points, window=6):
    smoothed = points.copy()
    
    for i in range(len(points)):
        neighbors = []
        
        for j in range(max(0, i - window//2), min(len(points), i + window//2 + 1)):
            if points[j] is not None:
                neighbors.append(points[j])
        
        if len(neighbors) >= 3:
            median = np.median(np.array(neighbors), axis=0)
            smoothed[i] = tuple(median)
    
    return smoothed
'''


def moving_average(points):
    new_points = [points[0]]
    # ta snitt av 3 cons som är nära nog varandra
    # Ersätt först none med närmaste värde
    for i in range(len(points) - 3):
        if points[i + 1] is None:
            new_points.append(None)
            continue
        neighborhood = points[i:i+3]
        # hantera vänsterkant
        if neighborhood[0] is None:
            neighborhood[0] = neighborhood[1]
        if np.linalg.norm(neighborhood[0] - neighborhood[1]) > 1.5:
            neighborhood[0] = neighborhood[1]
        # hantera högerkant
        if neighborhood[2] is None:
            neighborhood[2] = neighborhood[1]
        if np.linalg.norm(neighborhood[2] - neighborhood[1]) > 1:
            neighborhood[2] = neighborhood[1]
        new_points.append(np.mean(neighborhood, axis = 0))
    new_points.append(points[-1])
    return new_points