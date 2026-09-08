"""Interactive video navigation and two-camera synchronization.

`navigate_video` is what `update_parameters.py` uses to record the frame each
camera shows the synced action at. `sync_two_videos` is a standalone
helper for working out the frame mapping between two cameras by eye.
"""

import cv2


def sync_two_videos(video1_path, video2_path, fps1=60.0, fps2=60.0):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: Could not open one of the video files.")
        return

    # Get total frame counts for boundary checks
    total_frames1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Current frame indices
    frame1_idx = 0
    frame2_idx = 0

    selected_pair = None

    cv2.namedWindow('Video 1 (25 fps)', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Video 2 (60 fps)', cv2.WINDOW_NORMAL)

    print("\nControls:")
    print("  Video 1: a (prev) / d (next)")
    print("  Video 2: z (prev) / c (next)")
    print("  s       – select current pair and finish")
    print("  q       – quit without selecting\n")

    while True:
        # Seek to current frame indices
        cap1.set(cv2.CAP_PROP_POS_FRAMES, frame1_idx)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, frame2_idx)

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Reached end of a video or cannot read frame.")
            break

        # Overlay frame numbers
        cv2.putText(frame1, f"Frame: {frame1_idx}/{total_frames1-1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame2, f"Frame: {frame2_idx}/{total_frames2-1}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video 1 (25 fps)', frame1)
        cv2.imshow('Video 2 (60 fps)', frame2)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('u'):          # Video 1 previous frame
            frame1_idx = max(0, frame1_idx - 1)
        elif key == ord('i'):          # Video 1 next frame
            frame1_idx = min(total_frames1 - 1, frame1_idx + 1)
        elif key == ord('o'):          # Video 2 previous frame
            frame2_idx = max(0, frame2_idx - 1)
        elif key == ord('p'):          # Video 2 next frame
            frame2_idx = min(total_frames2 - 1, frame2_idx + 1)

        elif key == ord('j'):          # Video 1 previous second
            frame1_idx = max(0, frame1_idx - 25)
        elif key == ord('k'):          # Video 1 next second
            frame1_idx = min(total_frames1 - 1, frame1_idx + 25)
        elif key == ord('l'):          # Video 2 previous second
            frame2_idx = max(0, frame2_idx - 60)
        elif key == ord('ö'):          # Video 2 next second
            frame2_idx = min(total_frames2 - 1, frame2_idx + 60)

        elif key == ord('n'):          # Video 1 previous minute
            frame1_idx = max(0, frame1_idx - 25 * 60)
        elif key == ord('m'):          # Video 1 next minute
            frame1_idx = min(total_frames1 - 1, frame1_idx + 25 * 60)
        elif key == ord(','):          # Video 2 previous minute
            frame2_idx = max(0, frame2_idx - 60 * 60)
        elif key == ord('.'):          # Video 2 next minute
            frame2_idx = min(total_frames2 - 1, frame2_idx + 60 * 60)

        elif key == ord('s'):
            selected_pair = (frame1_idx, frame2_idx)
            print(f"\nSelected pair: video1 frame {frame1_idx}, video2 frame {frame2_idx}")
            break


    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    if selected_pair is not None:
        f1, f2 = selected_pair
        a = fps2 / fps1          # = 2.4
        b = f2 - a * f1
        print("\nMapping from video1 frame to video2 frame:")
        print(f"f2 = {a} * f1 + {b}")
        print(f"Python function: lambda f1: int(round({a} * f1 + {b}))")
    else:
        print("No pair selected.")




# single video tester

def navigate_video(file_path: str, start_frame: int = 0):
    """Scrub a video by hand and return the frame stopped on.

    `start_frame` is only where the scrubbing starts, never an answer: on a
    full-match recording the action being synced can be an hour in, and stepping
    there a minute at a time is not a reasonable thing to ask of anyone.
    `update_parameters.py` seeds it from the previous anchor when there is one.
    """
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    current_frame = start_frame

    print("Controls:")
    print("  j / f  -> +1 / -1 frame")
    print("  k / d  -> +1 / -1 second")
    print("  l / s  -> +1 / -1 minute")
    print("  q      -> quit and return current frame")

    while True:
        # Clamp frame index
        current_frame = max(0, min(current_frame, total_frames - 1))

        # Jump to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        # Display overlay info
        display_frame = frame.copy()
        text = f"Frame: {current_frame} | Time: {current_frame / fps:.2f}s"
        cv2.putText(display_frame, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show frame in window
        cv2.imshow("Video Navigator", display_frame)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('j'):          # forward 1 frame
            current_frame += 1
        elif key == ord('f'):        # back 1 frame
            current_frame -= 1
        elif key == ord('k'):        # forward 1 second
            current_frame += int(fps)
        elif key == ord('d'):        # back 1 second
            current_frame -= int(fps)
        elif key == ord('l'):        # forward 1 minute
            current_frame += int(fps * 60)
        elif key == ord('s'):        # back 1 minute
            current_frame -= int(fps * 60)
        elif key == ord('q'):        # exit
            break

    cap.release()
    cv2.destroyAllWindows()

    return current_frame