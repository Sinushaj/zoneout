"""Per-reception orchestration: video in, 3D coordinates written back to the .dvw.

This module holds the processing for a single reception. The match-specific
constants and the loop over receptions live in `run_pipeline.py` at the repo
root, so importing this module never kicks off a batch run.
"""

import os

import numpy as np

from . import ballistic
from .detection import (DEFAULT_MODEL, DETECTION_CONF_THRESHOLD,
                        MAX_CANDIDATES_PER_FRAME, annotate_video,
                        process_video)
from .events import find_reception, find_serve_and_receive
from .figures import save_trajectory_figure, save_trajectory_html
from .reconstruction import (match_detections, match_summary,
                             point_from_camera_coordinates)
from .scout import add_serve_direction
from .timing import camera_timing
from .trajectory import interpolate_nones, remove_bad_points, smooth_trajectory
from .video import create_video_chunk


# Set False to disable the ballistic correction entirely and fall back to the
# pixel-space cleanup (drop bad points, fill gaps linearly, smooth). Everything
# the correction does lives in `zoneout/ballistic.py`, and it reports what it
# changed on every reception — if those numbers ever look large, flip this off
# and compare.
USE_BALLISTIC_FIT = True

# Where the reception contact comes from. On, it is read off the ballistic
# segmentation (`events.find_reception`): the reception is the start of the
# first fitted parabola that begins once the ball is past the net, which is what
# a contact *is* — the frame where one free-flight model stops describing the
# ball and another starts. Off, it falls back to the older geometric reading,
# the first direction change of more than 45 degrees after the net crossing.
#
# The two are printed side by side on every run, so the difference is always
# visible. Over receptions 4-10 they land 0.15-1.11 m apart, heights within
# 0.20 m and the rest of it in y — and the parabola reading is the deeper one
# every time, which is the expected direction rather than a coincidence: the
# direction-change reading fires at the first frame whose turn exceeds 45
# degrees, a few frames after the contact, by which point the ball is already
# heading back toward the net. Since y is what picks the zone written to the
# .dvw, that bias is worth removing. Requires USE_BALLISTIC_FIT, since without
# it there are no segments to read.
USE_BALLISTIC_RECEPTION = True


def _report_matching(name, matches):
    """Print how the geometric gate did, so its threshold can be tuned."""
    print(f"{name}: {match_summary(matches)}")






def get_data_from_reception(dvw_filepath, reception_number, sideline_filepath, baseline_filepath):
    # os.makedirs(os.path.dirname(f'reception_data\\reception{reception_number}\\file_to_add'), exist_ok=True)
    path = os.path.join("reception_data", f"reception{reception_number}", "file_to_add")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_path = DEFAULT_MODEL
    gopro_path = 'temporary_videos/sideline_temp.mp4'
    zve10_path = 'temporary_videos/baseline_temp.mp4'


    # The manual sync: one action, its video_time, and the frame each camera
    # shows it at, plus each camera's own frame rate. Everything below is an
    # offset from that, so it is the one number in the run that nothing can
    # check. `zoneout.timing` owns it so that the attack pipeline resolves the
    # anchor by exactly the same rules.
    timing = camera_timing(dvw_filepath, sideline_filepath, baseline_filepath)
    sideline_rate, baseline_rate = timing.sideline_rate, timing.baseline_rate

    sideline_reception_frame, baseline_reception_frame, video_time = (
        timing.frames_for_action(dvw_filepath, 'Reception', reception_number))
    create_video_chunk(sideline_filepath, 'sideline_temp.mp4', sideline_reception_frame)
    create_video_chunk(baseline_filepath, 'baseline_temp.mp4', baseline_reception_frame)
    print(f"Done creating video chunks (frames {sideline_reception_frame} /"
          f" {baseline_reception_frame}, at {sideline_rate:.4f} /"
          f" {baseline_rate:.4f} frames per video_time second)")



    gopro_candidates = process_video(gopro_path, model_path, DETECTION_CONF_THRESHOLD,
                                          top_k=MAX_CANDIDATES_PER_FRAME)
    print("Done processing gopro video")

    zve10_candidates = process_video(zve10_path, model_path, DETECTION_CONF_THRESHOLD,
                                          top_k=MAX_CANDIDATES_PER_FRAME)
    print("Done processing zve10 video")

    # Choose, per frame, the one pair of candidates that triangulates to a
    # plausible point. A detection that only one camera believes in is dropped
    # here rather than being carried forward as a fabricated pixel.
    matches = match_detections(gopro_candidates, zve10_candidates)
    _report_matching(f"reception {reception_number}", matches)

    annotate_video(
        gopro_path,
        f'reception_data/reception{reception_number}/gopro_boxes.mp4',
        gopro_candidates,
        [m.gopro_index if m else None for m in matches],
        [f"{m.reprojection_error:.1f} px / {m.residual:.2f} m" if m else None for m in matches],
    )
    annotate_video(
        zve10_path,
        f'reception_data/reception{reception_number}/zve10_boxes.mp4',
        zve10_candidates,
        [m.zve10_index if m else None for m in matches],
        [f"{m.reprojection_error:.1f} px / {m.residual:.2f} m" if m else None for m in matches],
    )

    # The matcher's own triangulated points, with None where no pairing
    # survived. This is what the ballistic fit is fitted to, and it is kept
    # around unmodified so the interactive figure can draw it alongside the
    # result: a parabola is only trustworthy next to the measurements it came
    # from.
    matched_points = [m.point if m else None for m in matches]

    if USE_BALLISTIC_FIT:
        # Fit free-flight physics per segment to the matcher's own 3D points.
        # This deliberately bypasses the pixel-space cleanup below: those points
        # are linear inventions, and handing them to the fit as though they were
        # measurements would let a straight-line guess bend the very parabola it
        # is meant to be corrected by.
        real_coords, ballistic_report = ballistic.correct_trajectory(matched_points)
        print(" ", ballistic_report.summary())
    else:
        # The chosen pixels, per camera, with None where the gate found nothing.
        gopro_coordinate_list = [
            gopro_candidates[i][m.gopro_index].midpoint if m else None
            for i, m in enumerate(matches)
        ]
        zve10_coordinate_list = [
            zve10_candidates[i][m.zve10_index].midpoint if m else None
            for i, m in enumerate(matches)
        ]

        # Pixel-space cleanup: drop implausible jumps, then fill short gaps
        # linearly. Linear filling is wrong through a change of direction, which
        # is what the ballistic branch above exists to fix.
        gopro_coordinate_list = interpolate_nones(remove_bad_points(gopro_coordinate_list))
        zve10_coordinate_list = interpolate_nones(remove_bad_points(zve10_coordinate_list))

        real_coords = []
        for i in range(min(len(gopro_coordinate_list), len(zve10_coordinate_list))):
            gopro_coordinate = gopro_coordinate_list[i]
            zve10_coordinate = zve10_coordinate_list[i]
            if (gopro_coordinate is None) or (zve10_coordinate is None):
                real_coords.append(None)
                continue
            real_coordinate = point_from_camera_coordinates(gopro_coordinate, zve10_coordinate)
            real_coords.append(real_coordinate)
        real_coords = smooth_trajectory(real_coords)



    raw_start, raw_end = find_serve_and_receive([elem for elem in real_coords if not (elem is None)])
    print('returned sereve/reception:', raw_start, raw_end)

    # The serve stays as the geometric reading found it; only the reception is
    # re-read off the parabolas, since that is the one that has to be accurate —
    # it is what gets written back to the .dvw as where the ball was dug.
    if USE_BALLISTIC_FIT and USE_BALLISTIC_RECEPTION:
        found = find_reception(ballistic_report.segments)
        if found is None:
            print("  reception from parabolas: none found past the net —"
                  " keeping the direction-change reception")
        else:
            point, frame = found
            moved = float(np.linalg.norm(point - raw_end))
            print(f"  reception from parabolas: frame {frame},"
                  f" {np.array2string(point, precision=2)}"
                  f" ({moved:.2f} m from the direction-change reception)")
            raw_end = point

    # ----------------------
    # Add direction to file
    # ----------------------

    add_serve_direction(dvw_filepath, video_time, raw_start.copy(), raw_end.copy())

    # ---------------
    # Plotta punkter
    # ---------------

    save_trajectory_figure(
        f"reception_data/reception{reception_number}/raw_trajectory.png",
        real_coords,
        serve_point=raw_start,
        receive_point=raw_end,
        title=f"Reception {reception_number}",
    )

    # Interactive version: open in a browser and drag to rotate. It also gets
    # the pre-fit points, so the fitted curve can be checked against them;
    # clicking a legend entry hides either series.
    save_trajectory_html(
        f"reception_data/reception{reception_number}/raw_trajectory.html",
        real_coords,
        serve_point=raw_start,
        receive_point=raw_end,
        title=f"Reception {reception_number}",
        raw_points=matched_points,
    )
