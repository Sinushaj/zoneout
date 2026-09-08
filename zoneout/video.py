"""Extracting short clips from the full match videos.

Detection only ever needs a few seconds around each reception, so the pipeline
cuts a chunk out of the (very large) source video rather than decoding all of
it.
"""

import os
from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class Clip:
    """A cut clip, and where it came from in the source video.

    `start_frame` is what turns a frame number inside the clip back into one in
    the match video, which is the only way anything downstream can be related to
    the scout file or compared between the two cameras: the two clips are cut
    from different recordings at different rates, so clip frame *i* means the
    same instant in both only because they were cut to the same window.
    """

    path: str
    start_frame: int      # first frame of the clip, in the source video
    frames: int           # how many frames were actually written
    frame_rate: float     # the source video's own rate

    def source_frame(self, clip_frame: int) -> int:
        """The frame in the source video that clip frame `clip_frame` came from."""
        return self.start_frame + clip_frame


def get_frame_rate(file_path: str) -> float:
    """The video's own frame rate, read from the file rather than assumed.

    Cameras sold as 60 fps almost always record at 59.94 (the NTSC rate), and
    two of them rarely agree exactly - the sideline and baseline cameras here
    differ in the fifth decimal. Per second that is nothing; over an hour and a
    half of match it is frames, which is why the frame lookup takes each
    camera's real rate instead of one nominal number for both.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {file_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if fps <= 0:
        raise ValueError(f"{file_path} reports no usable frame rate ({fps}).")

    return fps


def create_video_chunk(file_path: str, save_name: str, n: int,
                       lead_seconds: float = 0.0,
                       duration_seconds: float = 5.0) -> Clip:
    """Cut a clip around frame `n`, starting `lead_seconds` before it.

    `n` is the frame the action is timed at. The clip runs from `lead_seconds`
    before it for `duration_seconds`, both measured in this video's own frames,
    so the same window in seconds comes out of two cameras recording at slightly
    different rates.

    The default is no lead at all, which is the reception case: a serve starts
    the rally, and the scout's `video_time` lands at or before it, so cutting
    straight forward covers the whole flight. An action scouted in the middle of
    a rally is entered *after* it happens and needs the window moved back over
    it - see `zoneout.attacks`.

    Raises rather than clamping when the lead would run before the start of the
    recording. Silently clamping would shift one camera's window and not the
    other's, and the two clips are paired frame by frame downstream, so a clip
    that quietly starts somewhere else is worse than one that fails.
    """
    # Ensure output directory exists
    output_dir = "temporary_videos"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, save_name)

    # Open the input video
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)

    # Validate FPS (expecting ~60)
    if fps <= 0:
        raise ValueError("Invalid FPS detected.")

    # Calculate frame range
    start_frame = n - int(round(lead_seconds * fps))
    frames_to_capture = int(duration_seconds * fps)
    end_frame = start_frame + frames_to_capture

    if start_frame < 0:
        cap.release()
        raise ValueError(
            f"{file_path}: a {lead_seconds} s lead before frame {n} starts at "
            f"frame {start_frame}, before the beginning of the recording.")

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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Release resources
    cap.release()
    out.release()

    # Asking for a frame the video does not reach writes an empty clip, which
    # then fails much later as an unopenable video. Say what actually went
    # wrong instead: the recording is shorter than the scout file, which is
    # what a batch run over a whole set or match will hit as soon as a camera
    # stopped early or the match spans more than one file per camera.
    if current_frame == start_frame:
        raise ValueError(
            f"{file_path} has {total_frames} frames, so nothing could be read "
            f"from frame {start_frame}: the video does not cover this action."
        )

    return Clip(path=output_path, start_frame=start_frame,
                frames=current_frame - start_frame, frame_rate=fps)