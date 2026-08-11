"""Extracting short clips from the full match videos.

Detection only ever needs a few seconds around each reception, so the pipeline
cuts a chunk out of the (very large) source video rather than decoding all of
it.
"""

import os
import cv2


def create_video_chunk(file_path: str, save_name: str, n: int):
    # Ensure output directory exists
    output_dir = "temporary_videos"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, save_name)

    # Open the input video
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print('fps =', fps) # ta bort eller använd sen

    # Validate FPS (expecting ~60)
    if fps <= 0:
        raise ValueError("Invalid FPS detected.")

    # Calculate frame range
    start_frame = n
    frames_to_capture = int(5 * fps)  # 5 seconds worth of frames
    end_frame = start_frame + frames_to_capture

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define video writer (overwrite if exists)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Move to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    current_frame = start_frame

    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()

    return output_path