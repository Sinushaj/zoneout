"""Reading from and writing back to DataVolley .dvw scout files.

Reading uses pydatavolley; writing is done by rewriting the relevant serve/
reception lines in place, since pydatavolley is read-only.

`get_plays()` gives one row per scouted action with, among much else, the
`skill` ("Serve", "Reception", "Attack", ...), the `video_time` in seconds on
the match video's clock, and the `set_number`. Everything the pipeline needs to
find and select an action comes from those three columns: which rows are of the
skill wanted, which set each belongs to, and where in the video each one is.

Selection is written once, for any skill, in `get_actions` / `action_numbers` /
`get_action_start_frame`. Receptions and attacks are thin wrappers over those,
so the two pipelines cannot end up numbering or timing their actions by
different rules.
"""

import numpy as np
import pandas as pd
from datavolley.read_dv import DataVolley
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
np.NaN = np.nan


# The name of the section holding the play-by-play, and the marker that starts
# it. Only needed to count its lines - see `_scout_line_count`.
SCOUT_SECTION = '[3SCOUT]'


def _scout_line_count(dvw_filepath):
    """How many play-by-play lines the file's `[3SCOUT]` section holds."""
    with open(dvw_filepath, 'r', encoding='cp1252') as scout_file:
        lines = scout_file.read().split('\n')

    try:
        start = lines.index(SCOUT_SECTION)
    except ValueError:
        raise ValueError(f"{dvw_filepath} has no {SCOUT_SECTION} section.")

    return len([line for line in lines[start + 1:] if line.strip()])


def describe_scout_file(dvw_filepath):
    """The path with the two team names after it, for printing.

    A scout file's name does not reliably say which match it is, and processing
    one match's videos against another match's scout file does not fail - it
    cuts every clip from the wrong minute. Seeing the teams is the cheap check.
    """
    try:
        with open(dvw_filepath, 'r', encoding='cp1252') as scout_file:
            lines = scout_file.read().split('\n')
        start = lines.index('[3TEAMS]')
        teams = [lines[start + i].split(';')[1] for i in (1, 2)]
    except (OSError, ValueError, IndexError):
        return dvw_filepath

    return f'{dvw_filepath} ({teams[0]} vs {teams[1]})'


def get_plays(dvw_filepath):
    """Every scouted line of the match, in file order, with its line number.

    A `scout_line` column is added: the 0-based position of the row's line
    within the `[3SCOUT]` section. `get_plays()` returns one row per line there
    - including the ones that are not skills at all, such as rotations and
    points, which come back with `skill` empty - so the two are in step, and
    that is checked here rather than assumed.

    The check is what makes `scout_line` safe to write against. Addressing a
    line by its number is the only unambiguous way to patch one: `video_time` is
    not unique (a rally's set, attack, block and dig routinely share a second,
    and 8 of this match's 292 attacks share one with another attack) and is not
    even monotonic, so a run that matched on it would sooner or later write an
    attack's coordinates onto some other action's line. If pydatavolley ever
    starts inserting or dropping rows, this raises instead of quietly shifting
    every line number by one.
    """
    plays = DataVolley(dvw_filepath).get_plays()

    lines = _scout_line_count(dvw_filepath)
    if len(plays) != lines:
        raise ValueError(
            f"{dvw_filepath}: pydatavolley returned {len(plays)} rows for"
            f" {lines} lines in {SCOUT_SECTION}, so a row's position no longer"
            f" identifies its line. Nothing can safely be written back until"
            f" that is understood.")

    return plays.reset_index(drop=True).rename_axis('scout_line').reset_index()


def get_actions(dvw_filepath, skill):
    """Every action of one skill in the scout file, in the order they were played.

    One row per action, so its position in this frame *is* its action number
    minus one. Everything that counts actions goes through here so that nothing
    can end up numbering them differently: `action_numbers` picks which ones to
    process, `get_action_start_frame` turns one of them into a video frame, and
    the per-action output directory is named after the same N.

    `skill` is the value pydatavolley puts in the `skill` column - "Reception",
    "Attack", "Serve", "Dig", "Block", "Set", "Freeball".
    """
    plays = get_plays(dvw_filepath)
    return plays[plays['skill'] == skill].reset_index(drop=True)


def action_numbers(dvw_filepath, skill, set_number=None):
    """The action numbers of one skill, optionally only those in one set.

    Action numbers are 1-based and count across the whole match, so they do not
    restart with each set: asking for set 3 of a four-set match might return
    85..121. That is deliberate - the number is what identifies an action
    everywhere else in the pipeline, so it has to mean the same thing whether a
    run was selected by set, by range, or not at all.

    They are also counted *per skill*: attack 7 and reception 7 are unrelated
    rallies. That is why the two pipelines keep their own selections rather than
    sharing one range.

    `set_number` is compared as text, since the scout file stores it that way,
    so both 3 and "3" work. An unknown set simply has no actions and returns an
    empty list rather than raising: the caller is a batch run that should say it
    has nothing to do, not crash.
    """
    actions = get_actions(dvw_filepath, skill)
    numbers = range(1, len(actions) + 1)
    if set_number is None:
        return list(numbers)
    wanted = str(set_number)
    return [n for n, s in zip(numbers, actions['set_number']) if str(s) == wanted]


def get_receptions(dvw_filepath):
    """Every reception in the scout file, in the order they were played."""
    return get_actions(dvw_filepath, 'Reception')


def reception_numbers(dvw_filepath, set_number=None):
    """The reception numbers in a scout file, optionally only those in one set."""
    return action_numbers(dvw_filepath, 'Reception', set_number)


def get_attacks(dvw_filepath):
    """Every attack in the scout file, in the order they were played."""
    return get_actions(dvw_filepath, 'Attack')


def attack_numbers(dvw_filepath, set_number=None):
    """The attack numbers in a scout file, optionally only those in one set."""
    return action_numbers(dvw_filepath, 'Attack', set_number)


# How many frames pass per second of `video_time`, for the camera the scout file
# was timed against. This is *not* necessarily a camera's own frame rate: it is
# the rate of the clock DataVolley wrote its seconds on, and the two agree only
# if that clock ran in true seconds.
#
# 60.0 is close but not exact for this file, and the residual is measured, not
# guessed. Cutting a 16 s window around a late reception and finding where the
# ball actually is puts the rally *earlier* than 60.0 predicts, by more the
# later the reception: 53 frames at reception 100 (4226 s in) and about 124-169
# at reception 120 (5033 s in). That is a true rate of roughly 59.96-59.99
# frames per second of video_time - so neither 60.0 nor either camera's own
# 59.94 is the right number, and the cameras' seconds are not DataVolley's.
#
# 60.0 is kept because it is right where it has been checked against real
# reconstructions (the first sets, where the error is under a frame per ten
# seconds and the ball is comfortably inside the 5 s clip). It is wrong enough
# by the third set to push the serve out of the clip entirely, which is what
# makes late receptions reconstruct into nonsense. The fix is not a better
# constant - fitting one to two noisy measurements of one match is how a
# number becomes match-specific - but a second sync anchor late in the file, so
# each camera's rate is measured as (last_frame - first_frame) / elapsed
# video_time instead of assumed. Until that exists, treat receptions late in a
# long match as suspect and check the annotated videos.
DVW_CLOCK_FPS = 60.0


def video_time_frame_rates(sideline_fps, baseline_fps, clock_fps=DVW_CLOCK_FPS):
    """Frames per second of `video_time`, for each camera.

    The sideline camera defines the clock: `video_time` is taken to advance at
    `clock_fps` of its frames per second, since that is the view a match is
    normally scouted from. The baseline camera is then scaled by the ratio of
    the two cameras' true rates, which is the part that has to come from the
    files rather than from a nominal 60.

    The ratio is what matters here, and only the ratio. Two cameras nominally
    at 59.94 still differ a little - these two by 0.0004 fps - and that
    difference accumulates against a *fixed* pair of anchor frames: by the end
    of the third set the baseline camera is ~2 frames further along than the
    same arithmetic with one shared rate would say. Two frames is 0.6 m of
    travel for a served ball, and the two views are triangulated frame by frame,
    so it is worth not throwing away.

    Returns `(sideline_rate, baseline_rate)`.
    """
    return clock_fps, clock_fps * baseline_fps / sideline_fps


def get_sync_action(dvw_filepath, set_number=None, skill='Serve'):
    """The action a sync anchor is pinned to: `(video_time, ordinal, code)`.

    It is the first serve (or reception) of the set - or of the match, for
    `set_number=None` - **that has a `video_time`**. `ordinal` is its position
    among that skill's actions there, 1 for the first, and `code` its scout
    code, so the person syncing can be told exactly which rally to find.

    The first one usually has a time, but not always: a scout can start the
    video clock a rally late. The Randaberg file's first rally carries no
    `video_time` on any of its lines, its second serve is at 0 and its third at
    24, so "the first serve of set 1" has no time to anchor on and the anchor
    has to be the second - and the person syncing has to be told to look for the
    second one, or the anchor lands a rally away from the time it is recorded
    at.

    Prefer the **serve**. A rally's serve and its reception carry the same
    `video_time` in the scout file, so for a normal rally the two choices are
    the same number — but a set that opens with a service error has no reception
    in its first rally at all, and then "the first reception" is some rally
    later, which is a needlessly awkward thing to go looking for in the video.
    The first serve of a set is always the start of play. Reception stays
    available for when the first serve is not on camera.
    """
    plays = DataVolley(dvw_filepath).get_plays()
    rows = plays[plays['skill'] == skill]
    if set_number is not None:
        rows = rows[rows['set_number'].astype(str) == str(set_number)]

    where = 'the match' if set_number is None else f'set {set_number}'
    if rows.empty:
        raise ValueError(f"{dvw_filepath} has no {skill} in {where}.")

    for ordinal, (video_time, code) in enumerate(
            zip(rows['video_time'], rows['code']), start=1):
        video_time = str(video_time).strip()
        if video_time and video_time.lower() not in ('nan', 'none', '<na>'):
            return int(video_time), ordinal, code

    raise ValueError(f"No {skill.lower()} in {where} of {dvw_filepath} has a"
                     f" video_time, so there is nothing to sync against.")


def get_sync_video_time(dvw_filepath, set_number=None, skill='Serve'):
    """The `video_time` of the sync action, `get_sync_action`'s first value.

    This is what a sync anchor is pinned to: the person calibrating finds that
    action in each camera and records the frame, and every reception afterwards
    is computed from the two. The sync step, the check that an anchor belongs
    to the scout file (`timing.anchor_mismatch`) and the start-frame suggestion
    all come through here, so they cannot disagree about which action it is.
    """
    return get_sync_action(dvw_filepath, set_number, skill)[0]


def frame_for_video_time(anchor_frame, anchor_video_time, video_time,
                         frames_per_second=DVW_CLOCK_FPS):
    """The frame one camera shows `video_time` at, given its sync anchor.

    `anchor_frame` is the frame this camera shows the synced action at, and
    `anchor_video_time` is that action's `video_time` — the pair produced by the
    sync step and stored in `first_reception_frames.csv`. Everything else is
    that anchor plus the elapsed `video_time`, converted at this camera's rate
    from `video_time_frame_rates`.

    The anchor does not have to come before the action being looked up: elapsed
    time is signed, so an anchor taken in the middle of a match still places the
    rallies before it, as long as the video is one continuous recording. What
    the anchor *must* be is in the same recording — which is the reason it is
    selectable per set at all, since a match filmed as one video per set has a
    different frame origin in each.
    """
    elapsed = int(video_time) - int(anchor_video_time)
    return anchor_frame + int(round(frames_per_second * elapsed))


def get_action_video_time(dvw_filepath, skill, action_number):
    """The `video_time` of action `action_number` of that skill, as an int.

    Raises rather than guessing when the scout file has no usable time for the
    action: everything downstream is an offset from this number, so a blank one
    would place the clip at the anchor and reconstruct a completely different
    rally without ever looking wrong.
    """
    actions = get_actions(dvw_filepath, skill)
    if not 1 <= action_number <= len(actions):
        raise ValueError(
            f"{dvw_filepath} has {len(actions)} {skill.lower()}(s), so there is"
            f" no {skill.lower()} {action_number}.")

    video_time = str(actions['video_time'].iloc[action_number - 1]).strip()
    if not video_time or video_time.lower() in ('nan', 'none'):
        raise ValueError(
            f"{skill} {action_number} of {dvw_filepath} has no video_time, so"
            f" it cannot be located in the video.")

    return int(video_time)


def get_action_start_frame(dvw_filepath, skill, anchor_frame, anchor_video_time,
                           action_number, frames_per_second=DVW_CLOCK_FPS):
    """The video frame action `action_number` is timed at, and its video_time."""
    video_time = get_action_video_time(dvw_filepath, skill, action_number)
    frame = frame_for_video_time(anchor_frame, anchor_video_time, video_time,
                                 frames_per_second)
    return frame, video_time


def get_reception_start_frame(dvw_filepath, anchor_frame, anchor_video_time,
                              reception_number, frames_per_second=DVW_CLOCK_FPS):
    """The video frame reception `reception_number` starts at, and its video_time."""
    return get_action_start_frame(dvw_filepath, 'Reception', anchor_frame,
                                  anchor_video_time, reception_number,
                                  frames_per_second)



def coords_to_dvindex(point):
    """A world point as the index a scout line's coordinate field holds.

    A thin wrapper over `dv_grid`, which owns the grid arithmetic for every
    writer, so the serve/reception write-back here and the attack write-back in
    `attacks` cannot end up using two different grids.

    **This is a corrected version and it does not agree with the one it
    replaced.** The old arithmetic laid 80 rows over the 18 m length where
    openvolley's grid has 81, and truncated toward zero instead of binning to
    the cell edges. Measured over 40 000 points spread across the court, the two
    disagree on the index *everywhere* - the corrected one sits a uniform
    +0.1125 m further along x, and 0 to +0.222 m further along y, the y error
    growing toward the far baseline. `dv_grid.index_to_xy` now round-trips
    exactly against `datavolley.helpers.dv_index2xy`, pydatavolley's own inverse
    of the same grid, so the correction is checked against the format's
    definition rather than argued for.
    """
    from .dv_grid import format_index, xy_to_index
    return format_index(xy_to_index(point[0], point[1]))


def add_serve_direction(dvw_file, time, start_point, end_point):
    # OBS!!! Kanske måste vrida runt punkter först då serve alltid börjar på mindre sidan

    # rotera planen 180 grader
    if 0 < start_point[1]:
        start_point[0], start_point[1] = 9-start_point[0], -start_point[1]
        end_point[0], end_point[1] = 9-end_point[0], -end_point[1]
    # Compute start and end coordinates in index format for datavolley
    start_point[1] = -9

    start_index = coords_to_dvindex(start_point)
    end_index = coords_to_dvindex(end_point)

    # Open datavolley file
    with open(dvw_file, 'r', encoding="cp1252") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line_data = line.split(';')
        if len(line_data) < 12: # isf är det inte en spelhandling
            continue
        if str(time) != line_data[12]:
            continue
        if (line_data[0][3] != 'S') and (line_data[0][3] != 'R'):
            continue
        line_data[4] = start_index
        line_data[5] = '-1-1'
        line_data[6] = end_index
        lines[i] = ';'.join(line_data)
        if line_data[0][3] == 'R':
            break
    
    with open(dvw_file, 'w', encoding="cp1252") as f: 
        f.writelines(lines)

# add_serve_direction("C:/Data Project/Data Volley 4/Seasons/Elit H 25-26/Scout/&svk-ork_test.dvw", 253, [8.9,8.9,0], [0,-6.5,0])
