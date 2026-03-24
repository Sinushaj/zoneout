import cv2
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
            if length <= 10 and start > 0 and end < n:
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