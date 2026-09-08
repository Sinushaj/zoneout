"""YOLO ball detection: video frames in, candidate 2D pixel midpoints per frame out.

The detector deliberately returns *several* candidates per frame rather than
only the most confident box. A single spurious detection (a head, a shoe, a
line judge's flag) is frequently more confident than the real ball for a few
frames, and picking top-1 here makes that unrecoverable. Instead the candidates
are handed to `reconstruction.match_detections`, which picks the one pair —
one candidate per camera — that actually triangulates to a plausible point in
space. A false positive in one camera almost never lines up geometrically with
a false positive in the other, so the choice is made where there is real
evidence to make it with.

Because the choice is made downstream, drawing the annotated debug video is a
separate pass (`annotate_video`) rather than something `process_video` can do
on the fly: at detection time we do not yet know which box won.

Trajectory cleaning of the chosen midpoints lives in `trajectory.py`.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import cv2
from ultralytics import YOLO


# The custom-trained ball detector checked into the repo.
DEFAULT_MODEL = 'gala_model.pt'

# This model scores the ball high and other objects low, so a low-confidence box
# is much more likely to be junk than to be a faint ball. Measured over
# receptions 1-3, lifting the floor from 0.25 to 0.40 cuts physically impossible
# frame-to-frame jumps (reception 1: 7% -> 3%) while giving up only a handful of
# frames, which the gap filling covers anyway.
DETECTION_CONF_THRESHOLD = 0.40

# At most this many candidate boxes per frame per camera. The ball is nearly
# always the top box or the runner-up; anything further down is another chance
# for a wrong pairing to win. Above the confidence floor there are rarely more
# than two candidates anyway, so this mostly just bounds the worst case.
MAX_CANDIDATES_PER_FRAME = 3

# These two live here rather than in a pipeline because every pipeline needs the
# same ones: they are properties of this detector on this footage, measured on
# real receptions, and the comments above record what was measured. A second
# copy in a second pipeline is how the two would end up different by accident.


@dataclass(frozen=True)
class Detection:
    """One candidate ball detection in one frame of one camera."""

    midpoint: Tuple[int, int]
    confidence: float
    box: Tuple[int, int, int, int]  # x1, y1, x2, y2


@lru_cache(maxsize=2)
def load_model(model_path: str) -> YOLO:
    """The detector, loaded once per path and reused.

    A batch run calls `process_video` twice per action - 584 times for a match's
    attacks - and reloading the weights each time is pure overhead. The model is
    only ever read from during inference, so sharing one is safe.
    """
    return YOLO(model_path)


def process_video(
    video_path: str,
    model_path: str = DEFAULT_MODEL,
    conf_threshold: float = DETECTION_CONF_THRESHOLD,
    device: str = "cpu",
    top_k: int = MAX_CANDIDATES_PER_FRAME,
) -> List[List[Detection]]:
    """
    Run the ball detector over every frame of a video.

    Args:
        video_path (str): Path to input video
        model_path (str): Path to YOLOv8 model (.pt)
        conf_threshold (float): Confidence threshold for detection. This can be
            set lower than you would dare with top-1 selection, because the
            geometric gate downstream is what actually rejects false
            positives — a low threshold here only widens the candidate pool.
        device (str): Torch device for inference. Forced to 'cpu' by default:
            torch.cuda.is_available() reports True on this machine, but the
            installed CUDA build ships no kernels for the GTX 980 (sm_52), so
            letting ultralytics auto-pick the GPU makes every conv layer fail
            with "GET was unable to find an engine to execute this computation".
        top_k (int): Maximum candidates to keep per frame, most confident first.
            Kept deliberately small. This model scores the ball high and other
            objects low, so the ball is nearly always in the top two or three;
            candidates past that are almost all junk, and every extra one is
            another chance for a wrong pairing to come out ahead.

    Returns:
        list: One list of Detection per frame, most confident first. The inner
            list is empty for frames with no detection above the threshold.
    """

    # Load model (cached, so a batch run pays for the weights once)
    model = load_model(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    candidates_per_frame = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run detection
        results = model(frame, conf=conf_threshold, verbose=False, device=device)[0]

        candidates = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                candidates.append(Detection(
                    midpoint=((x1 + x2) // 2, (y1 + y2) // 2),
                    confidence=float(box.conf[0]),
                    box=(x1, y1, x2, y2),
                ))

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        candidates_per_frame.append(candidates[:top_k])

    cap.release()

    return candidates_per_frame


def annotate_video(
    video_path: str,
    output_path: str,
    candidates_per_frame: Sequence[Sequence[Detection]],
    chosen_indices: Sequence[Optional[int]],
    frame_labels: Optional[Sequence[Optional[str]]] = None,
):
    """Write a debug video showing every candidate and which one was chosen.

    The chosen box is drawn in green with its midpoint marked; candidates that
    were considered and passed over are drawn thin and grey. Seeing the
    rejected candidates is the point — it is how you tell "the detector never
    saw the ball" apart from "the detector saw it and the gate threw it away".

    Args:
        chosen_indices: per frame, the index into that frame's candidate list
            that was selected, or None if the frame was left unmatched.
        frame_labels: optional per-frame caption, e.g. the ray residual.
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video.")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates = candidates_per_frame[frame_index] if frame_index < len(candidates_per_frame) else []
        chosen = chosen_indices[frame_index] if frame_index < len(chosen_indices) else None

        for i, detection in enumerate(candidates):
            x1, y1, x2, y2 = detection.box
            is_chosen = (i == chosen)
            color = (0, 255, 0) if is_chosen else (150, 150, 150)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_chosen else 1)
            cv2.putText(frame, f"{detection.confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2 if is_chosen else 1)
            if is_chosen:
                cv2.circle(frame, detection.midpoint, 5, (0, 0, 255), -1)

        label = frame_labels[frame_index] if frame_labels and frame_index < len(frame_labels) else None
        if label:
            cv2.putText(frame, label, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
