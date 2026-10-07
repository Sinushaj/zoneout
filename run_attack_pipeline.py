"""Entry point: the attack pipeline, for one match.

Choose the scout file and sync the videos with `update_parameters.py`, pick
the attacks with the constants below, then run:

    python run_attack_pipeline.py

Separate from `run_pipeline.py` on purpose. The two pipelines share the scout
reading, the sync anchor and (later) the reconstruction, but nothing else: this
file can be left unrun, or deleted, and receptions carry on exactly as before.

The steps: work out which attacks the run covers and where in each video each of
them is, cut the clip each one needs out of both match videos into
`temporary_videos/`, run the ball detector over both clips, pair the two
cameras' candidates into 3D points, fit free-flight physics to them, read the
set, the attack and its end off the fitted flights, and write the attacker's
contact and where the attack finished back into the `.dvw`, adding a set in front
of every attack that has none. The first step reads
the scout file and the two calibration CSVs and nothing else, so it takes about
a second for a whole match and is worth eyeballing before any of it is handed to
the rest.

The scout file is written **once**, after every attack has been processed, and
the original is copied to `<name>.dvw.bak` first. Set `WRITE_TO_SCOUT_FILE` to
False to measure everything and leave the file alone.

Detection is the expensive part — budget roughly a minute per attack on CPU, so
a set is an hour and a half and a match is most of a day. Set `DETECT = False`
to stop after the clips.

The clips are named after the attack rather than overwritten, so the whole
selection survives to be watched. Each attack's output goes to
`attack_data/attackN/`: the two annotated detection videos, and
`trajectory.html` — rotatable in a browser, with the fitted flights as a joined
blue path and the triangulated points they were fitted to underneath in ochre,
as markers with no line joining them. Scattered ochre means the problem is
upstream in detection or matching; tight ochre under a wandering blue path means
it is the fit.
"""

import time

from zoneout.attacks import (ATTACK_CLIP_DURATION, ATTACK_CLIP_LEAD,
                             USE_BALLISTIC_FIT, annotate_attack_clips,
                             attack_output_dir, attack_timeline,
                             detect_attack_ball, fit_attack_trajectory,
                             attack_points_row, extract_attack_clips,
                             match_attack_detections, matched_points,
                             save_attack_figure, save_attack_points,
                             scouted_attacks, write_to_scout_file)
from zoneout.config import (extract_dict_from_csv, read_scout_filepath,
                            read_sync_anchor)
from zoneout.events import find_attack_points
from zoneout.reconstruction import match_summary
from zoneout.scout import describe_scout_file
from zoneout.timing import anchor_mismatch

# --- Edit these for the match being processed ------------------------------
# The scout file is not set here: it is chosen in update_parameters.py and read
# from scout_filepath.csv, the same one run_pipeline.py uses.

# Which attacks to process. Attacks are numbered from 1 across the whole match,
# in the order they were played, and that numbering is per skill: attack 7 and
# reception 7 are unrelated rallies, which is why this pipeline keeps its own
# selection rather than sharing `run_pipeline.py`'s.
#
#   SET_NUMBER = None,  no range   the whole match
#   SET_NUMBER = 2                 every attack in set 2
#   FIRST/LAST = 7, 16             attacks 7-16
#
# The range is applied on top of the set, and the numbers stay match-wide when a
# set is selected: set 3 might be attacks 147-210, not 1-64.
SET_NUMBER = 1         # e.g. 2 for a single set; None for every set
FIRST_ATTACK = 22          # match-wide number, inclusive; None for no lower bound
LAST_ATTACK = None          # match-wide number, inclusive; None for no upper bound

# Run the ball detector over the clips. This is the expensive step — roughly a
# minute per attack on CPU — so it is worth turning off while checking that the
# selection and the clip window are right.
DETECT = True

# Write each attack's clips back with every candidate box drawn on them, into
# `attack_data/attackN/`. Off, detection still runs and is still reported; this
# only controls whether there is something to watch afterwards.
ANNOTATE = True

# Patch the measured coordinates into the `.dvw` scout file: the attacker's
# contact as the start and where the attack finished as the end, on the attack's
# own line. Off, the run still measures everything and still writes
# `attack_data/attack_points.csv`, so the results can be read and checked
# without the scout file being touched at all.
#
# The same step adds a set line in front of every attack in the selection that
# has none directly above it - blocked and recycled attacks included, since
# their set happened all the same - with the setter on court as the player, the
# attack's tempo, a + grade, K1/K2/K7 on X1/X2/X7, the attack combination's
# target letter, the attack's own clock time and video_time, and the measured
# set location where there is one. PR, PP and P2 are never set. `zoneout/set_codes.py` has
# the rules and what they were checked against.
#
# Only with DETECT on: a run with detection off is a check of the selection and
# the clips, and should not edit the scout file.
#
# The file is written **once**, at the end of the run. A run that dies half way
# leaves it exactly as it was rather than partly patched, and the original is
# copied to `<name>.dvw.bak` before the first write.
WRITE_TO_SCOUT_FILE = True
# ---------------------------------------------------------------------------


def scout_file_or_stop():
    """The scout file on record, or None after saying why the run cannot start.

    Checked before the first clip is cut rather than left to fail per action:
    a scout file that does not match the sync anchor fails nothing at all, it
    just puts every clip in the wrong place, and a run that takes hours should
    not find that out one action at a time.
    """
    try:
        dvw_filepath = read_scout_filepath()
        mismatch = anchor_mismatch(dvw_filepath)
    except (OSError, KeyError, ValueError) as error:
        print(error)
        return None

    print(f'Scout file: {describe_scout_file(dvw_filepath)}')
    if mismatch:
        print(mismatch)
        return None
    return dvw_filepath


def warn_if_sync_is_from_another_set():
    """Say so when the run reaches outside the set the sync was taken in.

    Harmless when the match is one continuous recording per camera — the frame
    lookup works in signed elapsed time, so an anchor anywhere in the video
    places every rally in it. Not harmless at all when the match was filmed as
    one video per set, because then a frame number from set 2 means nothing in
    set 3's file. Nothing here can tell those two cases apart, so this warns
    rather than refuses.
    """
    try:
        anchor = read_sync_anchor()
    except (FileNotFoundError, KeyError, ValueError):
        return

    synced_set = anchor['set']
    if synced_set is None or str(SET_NUMBER) == synced_set:
        return

    scope = 'the whole match' if SET_NUMBER is None else f'set {SET_NUMBER}'
    print(f'Warning: the sync anchor was taken in set {synced_set}, but this run'
          f' covers {scope}. Fine if both cameras filmed the match as one video;'
          f' re-run update_parameters.py if there is a video per set.')


def main():
    dvw_filepath = scout_file_or_stop()
    if dvw_filepath is None:
        return

    paths = extract_dict_from_csv('video_filepaths.csv')

    timeline = attack_timeline(
        dvw_filepath,
        sideline_filepath=paths['sideline'],
        baseline_filepath=paths['baseline'],
        set_number=SET_NUMBER,
        first=FIRST_ATTACK,
        last=LAST_ATTACK,
    )

    if not timeline:
        print('Nothing to process: no attacks matched the selection.')
        return

    where = 'the whole match' if SET_NUMBER is None else f'set {SET_NUMBER}'
    print(f'{len(timeline)} attack(s) from {where}:'
          f' {timeline[0].number}-{timeline[-1].number}')
    warn_if_sync_is_from_another_set()
    print(f'Clips: {ATTACK_CLIP_DURATION} s starting {ATTACK_CLIP_LEAD} s'
          f' before each attack, into temporary_videos/')
    print()

    started = time.time()
    failures = []
    measured = []
    for position, attack in enumerate(timeline, start=1):
        print(f'[{position}/{len(timeline)}] {attack}')
        try:
            sideline, baseline = extract_attack_clips(
                attack,
                sideline_filepath=paths['sideline'],
                baseline_filepath=paths['baseline'],
            )
        # Deliberately broad, for the same reason `run_pipeline.py` is: an
        # attack the footage does not cover, or one too close to the start of a
        # recording for the lead, should cost that attack and not the rest of
        # the run. Every failure is reported here and again at the end.
        except Exception as error:
            failures.append((attack.number, error))
            print(f'    failed: {type(error).__name__}: {error}')
            continue

        print(f'    {sideline.path} frames'
              f' {sideline.start_frame}-{sideline.start_frame + sideline.frames - 1}'
              f' ({sideline.frames} at {sideline.frame_rate:.2f} fps)')
        print(f'    {baseline.path} frames'
              f' {baseline.start_frame}-{baseline.start_frame + baseline.frames - 1}'
              f' ({baseline.frames} at {baseline.frame_rate:.2f} fps)')

        if not DETECT:
            continue

        try:
            detections = detect_attack_ball(attack, sideline, baseline)
            print(detections.summary())

            # Pair the two cameras frame by frame and triangulate. A candidate
            # only one camera believes in is dropped here rather than carried
            # forward as a fabricated point.
            matches = match_attack_detections(detections)
            print(f'    {match_summary(matches)}')

            # Fit free-flight physics per flight. The spike is what this is for:
            # it is the shortest flight in the clip and the one occlusion hurts
            # most, so it is the one at risk of being discarded as too few
            # points to be evidence.
            raw = matched_points(matches)
            fitted, segments, found = None, (), None
            if USE_BALLISTIC_FIT:
                fitted, report = fit_attack_trajectory(raw)
                print(f'    {report.summary()}')

                # The set, the attack and where the attack finished, read off
                # the first three fitted flights. Collected for the whole run
                # and written once at the end.
                segments = report.segments
                found = find_attack_points(segments)
                if found is None:
                    print(f'    no attack points: only {len(report.segments)}'
                          f' flight(s) fitted, and the set, the attack and its'
                          f' end need three')
                else:
                    print(f'    {found}')
                    for warning in found.warnings:
                        print(f'    WARNING: {warning}')
                    measured.append((attack, found))

            output_dir = attack_output_dir(attack.number)
            print(f'    wrote {save_attack_figure(attack, raw, output_dir, fitted, segments, found)}')
            if ANNOTATE:
                for path in annotate_attack_clips(detections, output_dir, matches):
                    print(f'    wrote {path}')
        except Exception as error:
            failures.append((attack.number, error))
            print(f'    failed after detection: {type(error).__name__}: {error}')

    if measured:
        rows = [attack_points_row(attack, points) for attack, points in measured]
        path = save_attack_points(rows)
        flagged = sum(1 for row in rows if row['warnings'])
        print(f'\nWrote {len(rows)} attack(s) to {path}'
              + (f' — {flagged} carry a warning' if flagged else ''))

    # The scout file last, and in one save. Everything above is this pipeline's
    # own output and can be regenerated; the `.dvw` is hours of a human's work,
    # so it is touched once, at the end, with a backup. Not inside `if measured`:
    # an attack that could not be measured still gets its set added.
    if DETECT and WRITE_TO_SCOUT_FILE:
        attacks = scouted_attacks(dvw_filepath, SET_NUMBER, FIRST_ATTACK,
                                  LAST_ATTACK)
        report = write_to_scout_file(dvw_filepath, measured, attacks)
        print(f'\nScout file:\n  {report.summary()}')
    elif DETECT:
        print('\nThe scout file was not touched (WRITE_TO_SCOUT_FILE is off)')

    elapsed = (time.time() - started) / 60
    done = len(timeline) - len(failures)
    stage = 'Processed' if DETECT else 'Cut clips for'
    print(f'\n{stage} {done}/{len(timeline)} attack(s) in {elapsed:.1f} min')
    for number, error in failures:
        print(f'  attack {number} failed: {type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
