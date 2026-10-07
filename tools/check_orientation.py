"""Check the DataVolley coordinate convention against a human scout.

Standalone, not part of any pipeline run, and it reads only - it never writes to
a scout file. Run it on any `.dvw`:

    python tools/check_orientation.py "/path/to/&match.dvw"

It answers the two questions `ATTACK_PLAN.md` §5 raised, which had to be settled
before attack coordinates could be written to a real file:

1. **Which end does a coordinate get written from?** For every line carrying a
   coordinate, it reports which half of the grid the start and the end fall in,
   split by team. If every action of both teams starts at the bottom and ends at
   the top, the convention is action-relative and `dv_grid.orient_for_action` is
   right.

2. **Is the zone layout right, and is it mirrored?** It converts the scout's own
   clicked index back to a zone with `dv_grid.xy_to_zone` and compares it
   against the zone the scout typed on the same line. That is a comparison of a
   human against themselves, so it will never be 100%; what matters is the shape
   of the confusion matrix. A diagonal one with its misses in neighbouring zones
   means the layout is right. A mirror shows up unmistakably as 1<->5 or 4<->2.

**This is only meaningful on a file whose coordinates a human entered.** A file
this pipeline has written is the pipeline marking its own homework, and the
script says so rather than reporting a confident 100%.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from zoneout import dv_grid
from zoneout.scout import SCOUT_SECTION, describe_scout_file, get_plays

# The `;`-separated fields of a scout line that hold the three coordinates.
START_FIELD, MID_FIELD, END_FIELD = 4, 5, 6


def scouted_coordinates(dvw_filepath):
    """One row per scouted line that carries a coordinate, with its zones."""
    plays = get_plays(dvw_filepath)

    with open(dvw_filepath, 'r', encoding='cp1252') as scout_file:
        lines = scout_file.read().split('\n')
    start = lines.index(SCOUT_SECTION)
    fields = [line.split(';') for line in lines[start + 1:] if line.strip()]

    def index(row, field):
        if len(row) <= field:
            return None
        value = row[field].strip()
        if value in ('', dv_grid.NO_COORDINATE):
            return None
        return int(value)

    rows = []
    for _, play in plays.iterrows():
        row = fields[int(play['scout_line'])]
        start_index, end_index = index(row, START_FIELD), index(row, END_FIELD)
        if start_index is None and end_index is None:
            continue
        rows.append({
            'skill': play['skill'],
            'team': play['team'],
            'start_zone': play['start_zone'],
            'end_zone': play['end_zone'],
            'start_index': start_index,
            'end_index': end_index,
        })
    return pd.DataFrame(rows)


def report_halves(coordinates):
    """Which half of the grid each action starts and ends in, by team."""
    print('\n--- Which end are coordinates written from?')
    print('    (bottom = the y < 0 half of the grid, where `orient_for_action`'
          ' puts the acting team)\n')

    for label, column in (('start', 'start_index'), ('end', 'end_index')):
        rows = coordinates.dropna(subset=[column])
        if rows.empty:
            continue
        halves = ['bottom' if dv_grid.index_to_xy(int(i))[1] < 0 else 'top'
                  for i in rows[column]]
        table = pd.crosstab([rows['skill'], rows['team']], pd.Series(
            halves, index=rows.index, name=f'{label} half'))
        print(table.to_string(), '\n')

    starts = coordinates.dropna(subset=['start_index'])
    bottom = sum(dv_grid.index_to_xy(int(i))[1] < 0 for i in starts['start_index'])
    print(f'    {bottom}/{len(starts)} actions start in the bottom half.')
    if bottom == len(starts):
        print('    Unanimous, for both teams: the convention is action-relative,'
              ' which is what\n    `dv_grid.orient_for_action` implements.')
    else:
        print('    NOT unanimous. Either this file mixes conventions, or the'
              ' acting team is not\n    always drawn at the bottom - in which'
              ' case `orient_for_action` does not hold here.')


def report_zones(coordinates):
    """The scout's own index against the scout's own typed zone."""
    print('\n--- Does the zone layout agree with the scout?')
    print('    (their clicked coordinate, converted by `xy_to_zone`, against the'
          ' zone they typed)\n')

    for label, index_column, zone_column in (
            ('start', 'start_index', 'start_zone'),
            ('end', 'end_index', 'end_zone')):
        rows = coordinates.dropna(subset=[index_column, zone_column])
        if rows.empty:
            continue

        computed, scouted = [], []
        for _, row in rows.iterrows():
            x, y = dv_grid.index_to_xy(int(row[index_column]))
            computed.append(dv_grid.xy_to_zone(x, y)[0])
            scouted.append(int(row[zone_column]))

        agree = sum(a == b for a, b in zip(computed, scouted))
        print(f'  {label} zone: {agree}/{len(rows)} = {agree / len(rows):.0%}\n')
        print(pd.crosstab(pd.Series(scouted, name='  scouted'),
                          pd.Series(computed, name='computed')).to_string())

        misses = Counter((a, b) for a, b in zip(scouted, computed) if a != b)
        if misses:
            listed = ', '.join(f'{a}->{b} x{n}' for (a, b), n in misses.most_common())
            print(f'\n  misses: {listed}')
        print()


def looks_pipeline_written(coordinates):
    """Whether these coordinates look like this pipeline's output, not a scout's.

    The pipeline writes serves and receptions only, and forces a serve's start
    to the baseline. A file of nothing but `S`/`R` lines is its own output, and
    checking a convention against a file that was written by the code being
    checked proves nothing.
    """
    skills = set(coordinates['skill'].dropna())
    return skills and skills <= {'Serve', 'Reception'}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        print(f'usage: python {sys.argv[0]} <scout file.dvw>')
        return 1

    dvw_filepath = sys.argv[1]
    print(describe_scout_file(dvw_filepath))

    coordinates = scouted_coordinates(dvw_filepath)
    if coordinates.empty:
        print('\nNo line in this file carries a coordinate, so there is nothing'
              ' to check against.')
        return 1

    print(f'{len(coordinates)} line(s) carry a coordinate.')
    if looks_pipeline_written(coordinates):
        print('\nWARNING: every coordinate here is on a serve or a reception,'
              ' which is what this\npipeline writes and all it writes. These are'
              ' almost certainly its own output, so\nagreement below would only'
              ' show the code agreeing with itself. Run this on a file a\nhuman'
              ' scouted with coordinates.')

    report_halves(coordinates)
    report_zones(coordinates)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
