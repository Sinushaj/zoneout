"""Entry point: interactive calibration.

Prompts for the things that change per match or per camera setup and writes
them to the CSV config files the pipeline reads:

    python update_parameters.py

The camera step records which body is filming from each side, and the focal
length for one with an interchangeable lens; `zoneout.cameras` turns that into
the intrinsics. Nothing is measured - these are specification-sheet
approximations - and a camera left unrecorded keeps the matrix that has always
been hardcoded for that position.

The court reference points are clicked one at a time in a window that shows a
court diagram with the point being asked for marked on it, because the clicks
are matched positionally with `zoneout.court.CALIBRATION_POINTS` and clicking
them in the wrong order solves quietly for the wrong camera pose.

The scout file step picks the `.dvw` the match is processed against, and stores
it in `scout_filepath.csv` for the pipelines to read. The sync step looks up its
action's video_time in that file, so the two belong together: changing the
scout file means redoing the sync, and this script says so when it does.

The sync step asks which set to sync in and whether to use that set's first
serve or its first reception, then has you find that action in each camera. Set
by set is the useful granularity because a match is often filmed as one video
per set, and a frame number only means anything within one recording — but an
anchor taken in any set also works for a whole match filmed continuously, since
the frame lookup works in signed elapsed time from wherever the anchor is.

Finding that action is done in a window (`calibration.sync`) with the picture
filling it and a scrubber under it, because the rally can be an hour into the
recording. Closing either camera's window writes nothing: the two frames only
mean anything as a pair.
"""

import os
import tkinter as tk
from tkinter import filedialog

from zoneout.cameras import MODELS, camera_matrix, describe
from zoneout.calibration import navigate_video, pick_court_points
from zoneout.config import (CAMERA_POSITIONS, extract_dict_from_csv,
                            read_camera_models, read_csv_to_tuples_np,
                            read_scout_filepath, read_sync_anchor,
                            write_camera_models, write_dict_to_csv,
                            write_scout_filepath, write_to_csv,
                            write_tuples_to_csv)
from zoneout.court import CALIBRATION_POINTS
from zoneout.scout import (describe_scout_file, get_sync_action,
                           get_sync_video_time, video_time_frame_rates)
from zoneout.timing import anchor_mismatch
from zoneout.video import get_frame_rate


def select_file(**dialog_options):
    root = tk.Tk()
    root.withdraw()
    root.update()  # ensures the dialog appears

    file_path = filedialog.askopenfilename(**dialog_options)

    root.destroy()  # clean up
    return file_path.replace("\\", "\\\\")


def update_scout_filepath():
    """Pick the `.dvw` scout file and store it for every entry point to read.

    Returns the path on file afterwards, which is the old one when the dialog is
    cancelled. Checks the sync anchor against the new file straight away, since
    an anchor taken against another scout file is exactly the mistake that
    goes unnoticed until the clips come out of the wrong minutes of the video.
    """
    try:
        current = read_scout_filepath()
    except (OSError, ValueError):
        current = None

    path = select_file(
        title='Choose the DataVolley scout file',
        initialdir=os.path.dirname(current) if current else None,
        filetypes=[('DataVolley scout file', '*.dvw'), ('All files', '*')])

    if not path:
        print('Cancelled; the scout file on record is unchanged.')
        return current

    write_scout_filepath(path)
    print(f'Scout file: {describe_scout_file(path)}')

    try:
        mismatch = anchor_mismatch(path)
    except (OSError, KeyError, ValueError):
        mismatch = None  # no anchor on file yet, so nothing to disagree with
    if mismatch:
        print(f'Note: {mismatch}')

    return path


def suggested_start_frames(dvw_filepath, video_time):
    """Where to start scrubbing each video, from the anchor already on file.

    Only a convenience: it applies the frame lookup's own arithmetic to the old
    anchor, which lands within a few seconds of the action when the previous
    sync was in the same recording, and somewhere useless when it was not.
    Either way the person doing the sync is looking at the video and decides.
    """
    try:
        previous = read_sync_anchor()
    except (FileNotFoundError, KeyError, ValueError):
        return 0, 0

    known_time = previous['video_time']
    if known_time is None:
        # Written before the anchor was selectable, so it is the match's first
        # reception, whatever set that was in.
        known_time = get_sync_video_time(dvw_filepath, skill='Reception')

    paths = extract_dict_from_csv('video_filepaths.csv')
    sideline_rate, baseline_rate = video_time_frame_rates(
        get_frame_rate(paths['sideline']), get_frame_rate(paths['baseline']))
    elapsed = video_time - known_time

    return (max(0, previous['sideline'] + int(round(sideline_rate * elapsed))),
            max(0, previous['baseline'] + int(round(baseline_rate * elapsed))))


# Which config file holds each camera's clicked pixels, and which video they
# are clicked in. The file names are historical - a GoPro filmed the sideline
# and a Sony ZV-E10 the baseline when they were named - and say nothing about
# which camera is there now; that is `camera_models.csv`, asked for separately.
CAMERA_POINT_FILES = (
    ('sideline', 'gopro_points.csv'),
    ('baseline', 'zve10_points.csv'),
)


def existing_pixels(csv_path):
    """The pixels already on file, if they still match the points being asked for.

    Handing them back to the picker means a single badly placed point can be
    redone on its own, rather than re-clicking all of them because one was off.
    A file from before a change to `CALIBRATION_POINTS` has the wrong number of
    rows and is ignored, since nothing can say which point each row was.
    """
    try:
        pixels = read_csv_to_tuples_np(csv_path)
    except (OSError, ValueError):
        return None

    return pixels if len(pixels) == len(CALIBRATION_POINTS) else None


def update_court_points():
    """Click each camera's court reference points and write them out.

    A camera whose window is closed without saving leaves its file alone, so
    re-clicking one camera does not mean re-clicking both.
    """
    paths = extract_dict_from_csv('video_filepaths.csv')

    for camera, csv_path in CAMERA_POINT_FILES:
        initial = existing_pixels(csv_path)
        loaded = f' - {csv_path} loaded, adjust or Clear all' if initial else ''
        points = pick_court_points(paths[camera], title=f'{camera}{loaded}',
                                   initial=initial)

        if points is None:
            print(f'Cancelled; {csv_path} left as it was.')
            continue

        write_tuples_to_csv(points, csv_path)
        print(f'Wrote {len(points)} points to {csv_path}.')


def ask_camera(position, current):
    """Which camera is on one side, and at what focal length where that matters.

    `current` is what is already on file for this position, offered as the
    default so that changing one camera does not mean re-entering the other.
    """
    keys = list(MODELS)

    print(f'\n{position} camera:')
    for index, key in enumerate(keys, 1):
        print(f'  {index}. {MODELS[key].name}')

    while True:
        answer = input(f'Which camera? {default_note(current)}').strip().lower()

        if not answer and current:
            return current

        if answer.isdigit() and 1 <= int(answer) <= len(keys):
            model_key = keys[int(answer) - 1]
        elif answer in MODELS:
            model_key = answer
        else:
            print('  Pick one of the numbers above.')
            continue

        if not MODELS[model_key].needs_focal_length:
            return model_key, None

        return model_key, ask_focal_length(model_key, current)


def ask_focal_length(model_key, current):
    """The lens setting in mm, which is what decides a Sony body's matrix.

    A zoom's marked focal length is nominal and shifts with focus distance, so
    this is an approximation whatever is typed - but it is the difference
    between a 16 mm and a 50 mm view, which is not a detail.
    """
    previous = current[1] if current and current[0] == model_key else None

    while True:
        answer = input('Focal length in mm?'
                       f' {default_note(previous)}').strip()

        if not answer and previous:
            return previous

        try:
            focal_length = float(answer.replace(',', '.'))
        except ValueError:
            print('  A number of millimeters, e.g. 29.')
            continue

        if focal_length <= 0:
            print('  A focal length is positive.')
            continue

        return focal_length


def default_note(current):
    """The '[current: ...]' part of a prompt, for a setting already on file."""
    if not current:
        return ''
    if isinstance(current, tuple):
        return f'[{describe(*current)}] '
    return f'[{current:g} mm] '


def update_camera_models():
    """Record which camera films from each side, and write the two out together.

    Both are asked for before anything is written, so abandoning the prompts
    half way leaves the file - and with it the intrinsics of the camera that
    was already answered - exactly as it was.
    """
    try:
        current = read_camera_models()
    except (OSError, ValueError):
        current = {}

    chosen = {position: ask_camera(position, current.get(position))
              for position in CAMERA_POSITIONS}

    write_camera_models(chosen)

    print()
    for position, (model_key, focal_length) in chosen.items():
        matrix = camera_matrix(model_key, focal_length)
        print(f'{position}: {describe(model_key, focal_length)},'
              f' fx = {matrix[0, 0]:.0f} px, fy = {matrix[1, 1]:.0f} px')
    print('These are approximations from the specification sheet, not'
          ' calibrations: no lens distortion is modelled.')


def _ordinal(n):
    """'1st', '2nd', '3rd', '4th', ... '11th', '12th', '13th', '21st'."""
    suffix = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{suffix}'


def sync_videos():
    """Record the frame each camera shows one known action at.

    The action is picked from the scout file rather than described by hand, so
    that its `video_time` is exact: everything the pipeline computes afterwards
    is an offset from this one pair of numbers, and an anchor off by a second is
    a second of error on every reception in the set.
    """
    try:
        dvw_filepath = read_scout_filepath()
    except (OSError, ValueError) as error:
        print(error)
        dvw_filepath = update_scout_filepath()
        if dvw_filepath is None:
            print('No scout file, so nothing to sync against; start frames unchanged.')
            return
    print(f'Syncing against {describe_scout_file(dvw_filepath)}')

    set_number = input('Which set to sync (blank for the whole match)? ').strip()
    set_number = set_number or None

    # The serve is the better default: a set that opens with a service error has
    # no reception in its first rally, and for every normal rally the serve and
    # the reception share a video_time anyway, so this is the same instant with
    # a more findable name.
    skill = 'Reception' if input('Sync on the first reception instead of the'
                                 ' first serve? ').capitalize() == 'Yes' else 'Serve'

    try:
        video_time, ordinal, code = get_sync_action(dvw_filepath, set_number, skill)
    except ValueError as error:
        print(f'{error} Start frames unchanged.')
        return
    where = 'the match' if set_number is None else f'set {set_number}'
    # Usually the first. When the scout started the video clock late the first
    # few have no video_time, and the anchor is the first that does - which has
    # to be said, or the rally found in the video is not the one the time is for.
    action = f'first {skill.lower()}'
    if ordinal > 1:
        action = (f'{_ordinal(ordinal)} {skill.lower()} ({code.split("~")[0]})')
        print(f'\nThe first {ordinal - 1} {skill.lower()}(s) of {where} have no'
              f' video_time in the scout file, so the sync is on the'
              f' {_ordinal(ordinal)}.')
    note = (f'Find the {action} of {where}, at video_time'
            f' {video_time}, and stop on the frame the serve is struck — or up'
            ' to a second before it.\n\n'
            'Each 5 s clip is cut *forward* from here, so a little lead is'
            ' insurance: the scout file timestamps rallies to the second and'
            ' lands up to ~1.5 s from the serve, and a serve before its clip'
            ' starts is a reception with no arc to fit.')
    print(f'\nSyncing on the {action} of {where},'
          f' at video_time {video_time}.')
    print(note.replace('\n\n', '\n'))

    paths = extract_dict_from_csv('video_filepaths.csv')
    sideline_start, baseline_start = suggested_start_frames(dvw_filepath, video_time)

    # Both cameras are found before anything is written, so closing either
    # window leaves the anchor on file alone. Half a sync is worse than none:
    # the two frames are only meaningful as a pair, and an anchor from two
    # different rallies cuts every clip of the run from the wrong minute.
    sideline_frame = navigate_video(paths['sideline'], sideline_start,
                                    title='sideline', note=note)
    if sideline_frame is None:
        print('Cancelled; the frames on record are unchanged.')
        return

    baseline_frame = navigate_video(paths['baseline'], baseline_start,
                                    title='baseline', note=note)
    if baseline_frame is None:
        print('Cancelled; the frames on record are unchanged.')
        return

    write_dict_to_csv('first_reception_frames.csv', {
        'set': set_number if set_number is not None else '',
        'skill': skill,
        'video_time': video_time,
        'sideline': sideline_frame,
        'baseline': baseline_frame,
    })
    print(f'Synced: video_time {video_time} is sideline frame {sideline_frame},'
          f' baseline frame {baseline_frame}.')


def main():
    # uppdatera court coordinates och videolänk:
    if input('Change video paths?').capitalize() == 'Yes':
        print('Choose the sideline video file:')
        sideline_path = select_file()
        print('Choose the baseline video file:')    
        baseline_path = select_file()
        write_to_csv('video_filepaths.csv', sideline_path, baseline_path)

    if input('Change scout file?').capitalize() == 'Yes':
        update_scout_filepath()

    if input('Change cameras?').capitalize() == 'Yes':
        update_camera_models()

    if input('Change reference court coordinates?').capitalize() == 'Yes':
        update_court_points()

    if input('Change start frames?').capitalize() == 'Yes':
        sync_videos()

    print('done!')


if __name__ == "__main__":
    main()
