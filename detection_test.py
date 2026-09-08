"""Temporary test environment: run ball detection on any video file.

Standalone from the reception pipeline — it takes one video path, runs
`gala_model.pt` over it, and writes an annotated copy showing every ball the
detector found. Nothing here is imported by the pipeline, and it imports
`zoneout.detection` without changing it.

    python detection_test.py                       # opens a file picker
    python detection_test.py /path/to/match.mp4
    python detection_test.py match.mp4 --max-seconds 30 --conf 0.25

With no path on the command line it opens the same Tk file dialog the
calibration tools use. `--no-gui` types the path in at the terminal instead,
which is also what happens automatically when Tk cannot open a window (no
display, e.g. over a plain ssh session).

Only the first `--max-seconds` (default 60, i.e. 1 minute) of the video are
processed; a longer video is trimmed to that length first and the trimmed clip
is kept next to the output so what was annotated is always inspectable.

The annotation is `detection.annotate_video`, the same one the pipeline writes
per reception: the most confident box in each frame is drawn green with its
midpoint marked, and the other candidates thin and grey. There is no second
camera here, so nothing verifies the green box geometrically — it is the
detector's own top pick, which is exactly what you want to look at when the
question is "does the model see the ball in this footage".

Inference is CPU-only for the same reason as the rest of the project (see
"GPU is unusable" in NOTES.md), and runs at about 0.2 s per frame: a minute of
60 fps footage is ~3 600 frames, so around ten minutes of detection. Use
--max-seconds to test on a shorter window first.
"""

import argparse
import os
import time

import cv2

from zoneout.detection import annotate_video, process_video

DEFAULT_MODEL = "gala_model.pt"
DEFAULT_MAX_SECONDS = 60.0         # 1 minute
DEFAULT_CONF = 0.40                # same floor as pipeline.DETECTION_CONF_THRESHOLD
DEFAULT_TOP_K = 3                  # same as pipeline.MAX_CANDIDATES_PER_FRAME
OUTPUT_DIR = "detection_test_output"


def select_video_file() -> str:
    """Ask for the video with a Tk file dialog, falling back to typing a path.

    Same dialog as `update_parameters.select_file`, but that one escapes
    backslashes for writing into a CSV and lives in an entry point rather than
    a library, so it is not worth importing here for four lines.
    """

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.update()  # ensures the dialog appears
        path = filedialog.askopenfilename(
            title="Select a video to run ball detection on",
            filetypes=[("Video files", "*.mp4 *.MP4 *.mov *.MOV *.mts *.MTS "
                                       "*.m2ts *.M2TS *.avi *.AVI *.mkv *.MKV"),
                       ("All files", "*.*")],
        )
        root.destroy()
        return path
    except Exception as error:
        # No display, or no tkinter in this environment. Typing the path still
        # works, so say why the dialog did not appear and carry on.
        print(f"Could not open the file dialog ({error}); type the path instead.")
        return input("Video filepath: ").strip().strip('"').strip("'")


def trim_video(source_path: str, output_path: str, max_seconds: float) -> str:
    """Copy at most `max_seconds` of `source_path` into `output_path`.

    Returns the path that should actually be detected on: the source itself
    when it is already short enough, so a short clip is never needlessly
    re-encoded.
    """

    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {source_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        cap.release()
        raise ValueError(f"{source_path} reports no usable frame rate ({fps}).")

    frame_limit = int(max_seconds * fps)
    if total_frames > 0 and total_frames <= frame_limit:
        cap.release()
        print(f"Video is {total_frames / fps:.1f} s, under the {max_seconds:.0f} s cap — using it as is.")
        return source_path

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    written = 0
    while written < frame_limit:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        written += 1

    cap.release()
    out.release()

    if written == 0:
        raise ValueError(f"Read no frames at all from {source_path}.")

    print(f"Trimmed to the first {written} frames ({written / fps:.1f} s) -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", nargs="?", help="path to the video to run detection on")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"YOLO weights (default {DEFAULT_MODEL})")
    parser.add_argument("--max-seconds", type=float, default=DEFAULT_MAX_SECONDS,
                        help=f"cap on how much video is processed (default {DEFAULT_MAX_SECONDS:.0f})")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help=f"confidence floor for a detection (default {DEFAULT_CONF})")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"candidates kept per frame (default {DEFAULT_TOP_K})")
    parser.add_argument("--output", help="path for the annotated video "
                                         f"(default {OUTPUT_DIR}/<name>_detected.mp4)")
    parser.add_argument("--no-gui", action="store_true",
                        help="type the path at the terminal instead of opening the file dialog")
    args = parser.parse_args()

    if args.video:
        video_path = args.video
    elif args.no_gui:
        video_path = input("Video filepath: ").strip().strip('"').strip("'")
    else:
        video_path = select_video_file()
    if not video_path:
        raise SystemExit("No video selected.")
    if not os.path.isfile(video_path):
        raise SystemExit(f"No such file: {video_path}")
    if not os.path.isfile(args.model):
        raise SystemExit(f"No such model: {args.model}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    output_path = args.output or os.path.join(OUTPUT_DIR, f"{stem}_detected.mp4")
    trimmed_path = os.path.join(OUTPUT_DIR, f"{stem}_trimmed.mp4")

    detect_on = trim_video(video_path, trimmed_path, args.max_seconds)

    print(f"Detecting with {args.model} (conf >= {args.conf}, top {args.top_k} per frame) on CPU...")
    started = time.time()
    candidates_per_frame = process_video(detect_on, args.model,
                                         conf_threshold=args.conf, top_k=args.top_k)
    elapsed = time.time() - started

    # No second camera here, so there is no geometric gate to pick a winner:
    # the detector's own most confident box is the chosen one, and the rest are
    # drawn as the passed-over candidates they would be in the pipeline.
    chosen_indices = [0 if candidates else None for candidates in candidates_per_frame]
    labels = [f"{candidates[0].confidence:.2f}" if candidates else "no ball"
              for candidates in candidates_per_frame]

    annotate_video(detect_on, output_path, candidates_per_frame, chosen_indices, labels)

    frames = len(candidates_per_frame)
    hits = sum(1 for candidates in candidates_per_frame if candidates)
    confidences = [candidates[0].confidence for candidates in candidates_per_frame if candidates]
    print(f"\n{frames} frames in {elapsed:.0f} s ({elapsed / max(frames, 1):.2f} s/frame)")
    print(f"Ball found in {hits} frames ({100 * hits / max(frames, 1):.1f}%)")
    if confidences:
        print(f"Top-box confidence: min {min(confidences):.2f}, "
              f"mean {sum(confidences) / len(confidences):.2f}, max {max(confidences):.2f}")
    print(f"Annotated video: {output_path}")


if __name__ == "__main__":
    main()
