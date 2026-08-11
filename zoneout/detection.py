"""YOLO ball detection: video frames in, one 2D pixel midpoint per frame out.

Trajectory cleaning of those midpoints lives in `trajectory.py`.
"""

import cv2
from ultralytics import YOLO



def process_video(
    video_path: str,
    model_path: str,
    conf_threshold: float = 0.5,
    output_path: str = "output.mp4",
    device: str = "cpu"
):
    """
    Processes a video using a YOLOv8 model.

    Args:
        video_path (str): Path to input video
        model_path (str): Path to YOLOv8 model (.pt)
        conf_threshold (float): Confidence threshold for detection
        output_path (str): Path to save annotated video
        device (str): Torch device for inference. Forced to 'cpu' by default:
            torch.cuda.is_available() reports True on this machine, but the
            installed CUDA build ships no kernels for the GTX 980 (sm_52), so
            letting ultralytics auto-pick the GPU makes every conv layer fail
            with "GET was unable to find an engine to execute this computation".

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
        results = model(frame, conf=conf_threshold, verbose=False, device=device)[0]

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
