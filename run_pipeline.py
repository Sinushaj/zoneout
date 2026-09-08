"""Entry point: batch-process receptions for one match.

Edit the constants below for the match being processed, then run:

    python run_pipeline.py

Note this rewrites DVW_FILEPATH in place, and takes roughly a minute per
reception on CPU — so a whole set is the better part of an hour and a whole
match is a few hours. The run keeps going when a reception fails and lists the
failures at the end, since one bad reception in a hundred should not cost the
other ninety-nine.
"""

import time

from zoneout.config import extract_dict_from_csv, read_sync_anchor
from zoneout.pipeline import get_data_from_reception
from zoneout.scout import reception_numbers

# --- Edit these for the match being processed ------------------------------
DVW_FILEPATH = "/home/neo/Desktop/&svk-ork_q5.dvw"
# DVW_FILEPATH = "C:/Data Project/Data Volley 4/Seasons/Elit H 25-26/Scout/&svk-ork_test.dvw"

# Which receptions to process. Receptions are numbered from 1 across the whole
# match, in the order they were played, and that number is what names the
# output in `reception_data/receptionN/` — so it means the same thing however a
# run is selected here.
#
#   SET_NUMBER = None,  no range   the whole match
#   SET_NUMBER = 2                 every reception in set 2
#   FIRST/LAST = 7, 16             receptions 7-16, as before
#
# The range is applied on top of the set, so leaving it at None is what makes
# "all of set 2" mean all of it. The numbers stay match-wide when a set is
# selected: set 3 might be receptions 85-121, not 1-37.
SET_NUMBER = None         # e.g. 2 for a single set; None for every set
FIRST_RECEPTION = 1    # match-wide number, inclusive; None for no lower bound
LAST_RECEPTION = 5     # match-wide number, inclusive; None for no upper bound
# ---------------------------------------------------------------------------


def selected_receptions():
    """The reception numbers this run should process, in order."""
    numbers = reception_numbers(DVW_FILEPATH, SET_NUMBER)
    if FIRST_RECEPTION is not None:
        numbers = [n for n in numbers if n >= FIRST_RECEPTION]
    if LAST_RECEPTION is not None:
        numbers = [n for n in numbers if n <= LAST_RECEPTION]
    return numbers


def warn_if_sync_is_from_another_set():
    """Say so when the run reaches outside the set the sync was taken in.

    Harmless when the match is one continuous recording per camera — the frame
    lookup works in signed elapsed time, so an anchor anywhere in the video
    places every rally in it. Not harmless at all when the match was filmed as
    one video per set, because then a frame number from set 2 means nothing in
    set 3's file, and the run would quietly cut clips from the wrong minutes of
    the wrong video. Nothing here can tell those two cases apart, so this warns
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
    sideline_filepath = extract_dict_from_csv('video_filepaths.csv')['sideline']
    baseline_filepath = extract_dict_from_csv('video_filepaths.csv')['baseline']

    numbers = selected_receptions()
    if not numbers:
        print('Nothing to process: no receptions matched the selection.')
        return

    where = 'the whole match' if SET_NUMBER is None else f'set {SET_NUMBER}'
    print(f'Processing {len(numbers)} reception(s) from {where}:'
          f' {numbers[0]}-{numbers[-1]}')
    warn_if_sync_is_from_another_set()

    started = time.time()
    failures = []
    for position, i in enumerate(numbers, start=1):
        print(f'\n[{position}/{len(numbers)}] reception {i}')
        try:
            get_data_from_reception(
                dvw_filepath=DVW_FILEPATH,
                reception_number=i,
                sideline_filepath=sideline_filepath,
                baseline_filepath=baseline_filepath,
            )
        # Deliberately broad: a reception can fail for any number of reasons
        # (the video does not reach that far, nothing was detected, the fit
        # found no trajectory to read events off), and in a run of a hundred
        # none of them is a reason to abandon the rest. Every failure is
        # reported here and again at the end, and the receptions that did work
        # are already written to the .dvw. Ctrl-C still stops the run, since
        # KeyboardInterrupt is not an Exception.
        except Exception as error:
            failures.append((i, error))
            print(f'Reception {i} failed: {type(error).__name__}: {error}')
        else:
            print('Finished reception number', i)

    elapsed = (time.time() - started) / 60
    done = len(numbers) - len(failures)
    print(f'\nDone: {done}/{len(numbers)} reception(s) in {elapsed:.1f} min')
    for i, error in failures:
        print(f'  reception {i} failed: {type(error).__name__}: {error}')


if __name__ == "__main__":
    main()
