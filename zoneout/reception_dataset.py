"""The serve-and-reception dataset: one row per reception, for analysing the two together.

Each row puts what the scout file says about a rally's serve and reception next
to what the ballistic fit measured of it, in `datasets/receptions.csv`. It is a
different thing from `datasets/serves.csv` (`zoneout/serve_dataset.py`), which
holds each serve's whole parabola; this one is the rally from the server's hand
to the end of the pass, reduced to the numbers a reception analysis asks for.

From the scout file:

- `serving_team`, `receiving_team`, `serving_player`, `receiving_player` - the
  teams and names on the serve and reception lines;
- `serve_type`: `spin` for a `Q` serve, `float` for `M`, and any other letter
  as it stands;
- `reception_grade` (`#`, `+`, `!`, `-`, `/`, `=`);
- `receiving_team_won`: whether the receiving team won the rally;
- `receiving_rotation`: where the receiving team's setter stood, 1-6.

From the fitted flights, in metres, **oriented with the server at the `y < 0`
end** exactly as `serves.csv` is (`dv_grid.orient_for_action`), so the reception
and landing are at `y > 0`, and so is the end of the pass unless it went over:

- `serve_start_*` - the server's contact. Where the ball was first seen already
  in flight the serve's parabola is run back to
  `serve_dataset.SERVE_CONTACT_DEPTH`, the baseline, as `serves.csv` does;
- `reception_*` - `events.find_reception`'s reception, the pass flight's model
  at the contact;
- `serve_landing_x/y` - where the serve would have come down had it not been
  received: its last flight carried forward until the ball's centre is
  `court.BALL_RADIUS` off the floor;
- `serve_hit_net` - whether the serve changed flight at the net
  (`events.find_serve_path`). A touch that barely deflects the ball needs no
  second parabola and cannot be seen as one, so this can miss a graze;
- `serve_average_speed` (m/s) - path length from the contact to the reception
  over the time it took;
- `reception_end_*` - where the pass ended: the end of the reception's own
  trajectory, its model at the contact that played it next (`events.find_pass`).
  Normally the setter's hands, but written for every reception whose pass could
  be followed to its end, overpasses included;
- `reception_apex_z` - how high the pass went: the top of the pass's parabola,
  or, when the pass left the top of the picture, of one parabola fitted
  through both halves (`events.find_pass`).

A row is written for every reception the pipeline gets this far with, and a
measurement that could not be made is left **blank** rather than the row being
dropped: the scout-file half is still worth having, and in pandas a blank is
simply NaN. What was not measured, and why, is printed.

Rows are merged by `(match, reception_number)` like `serves.csv`, so a re-run
replaces its row.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import dv_grid
from .ballistic import FPS
from .court import BALL_RADIUS
from .events import find_pass, find_serve_path
from .scout import get_plays
from .serve_dataset import SERVE_CONTACT_DEPTH, merge_row, serve_for_reception

RECEPTIONS_CSV = os.path.join('datasets', 'receptions.csv')

# The serve-type letter (the fifth character of a serve code, `a12SQ-`) as the
# dataset names it. Any other letter is written as it stands.
SERVE_TYPES = {'Q': 'spin', 'M': 'float'}

CSV_COLUMNS = [
    'match', 'reception_number',
    'serving_team', 'receiving_team', 'serving_player', 'receiving_player',
    'serve_type',
    'serve_start_x', 'serve_start_y', 'serve_start_z',
    'reception_x', 'reception_y', 'reception_z',
    'serve_landing_x', 'serve_landing_y',
    'serve_hit_net',
    'serve_average_speed',
    'reception_end_x', 'reception_end_y', 'reception_end_z',
    'reception_apex_z',
    'reception_grade', 'receiving_team_won', 'receiving_rotation',
]


# --- The scout file ---------------------------------------------------------

def _setter_position(reception):
    """Where the receiving team's setter stood, 1-6, off the reception line."""
    home = reception['team'] == reception['home_team']
    position = reception['home_setter_position' if home else 'visiting_setter_position']
    try:
        return int(position)
    except (TypeError, ValueError):
        return ''


def scout_fields(dvw_filepath, reception_number, plays):
    """Everything the row takes from the scout file."""
    receptions = plays[plays['skill'] == 'Reception']
    reception = receptions.iloc[reception_number - 1]
    serve = serve_for_reception(dvw_filepath, reception_number, plays)

    serve_letter = serve['code'][4:5]
    won_by = reception['point_won_by']
    row = {
        'match': os.path.splitext(os.path.basename(dvw_filepath))[0],
        'reception_number': reception_number,
        'serving_team': serve['team'],
        'receiving_team': reception['team'],
        'serving_player': serve['player_name'],
        'receiving_player': reception['player_name'],
        'serve_type': SERVE_TYPES.get(serve_letter, serve_letter),
        'reception_grade': reception['evaluation_code'],
        'receiving_team_won': (won_by == reception['team']
                               if isinstance(won_by, str) else ''),
        'receiving_rotation': _setter_position(reception),
    }
    return row


# --- The fitted flights -----------------------------------------------------

def _frames_at(segment, axis, value):
    """The frames at which `segment`'s model has coordinate `axis` equal to `value`."""
    p0, v, a = (c[axis] for c in segment.coefficients)
    c = p0 - value
    if abs(a) < 1e-12:
        return [] if abs(v) < 1e-12 else [segment.start - c / v]
    disc = v * v - 2 * a * c
    if disc < 0:
        return []
    root = math.sqrt(disc)
    return sorted(segment.start + dt for dt in ((-v - root) / a, (-v + root) / a))


def _serve_contact_frame(path, serve_side):
    """The frame the server struck the ball, possibly fractional.

    Observed contacts are used as they are. Otherwise the serve's parabola is
    run back to `SERVE_CONTACT_DEPTH` behind the net, as `serve_dataset` does;
    a serve first seen already at or behind that depth is not moved.
    """
    first = path.flights[0]
    if path.contact_observed:
        return path.start_frame
    if serve_side * first.evaluate(path.start_frame)[1] >= SERVE_CONTACT_DEPTH:
        return path.start_frame
    earlier = [f for f in _frames_at(first, 1, serve_side * SERVE_CONTACT_DEPTH)
               if f <= path.start_frame]
    return max(earlier) if earlier else path.start_frame


def _landing(path):
    """Where the serve's last flight comes down to the floor, carried on past the reception."""
    last = path.flights[-1]
    frames = [f for f in _frames_at(last, 2, BALL_RADIUS) if f > path.start_frame]
    return last.evaluate(max(frames)) if frames else None


def _average_speed(path, start):
    """Path length over time, from the contact to the reception, sampled every frame."""
    duration = (path.end_frame - start) / FPS
    if duration <= 0:
        return None
    frames = np.linspace(start, path.end_frame, max(int(math.ceil(path.end_frame - start)), 1) + 1)
    points = np.array([path.evaluate(f) for f in frames])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) / duration


def _put(row, name, point, axes='xyz'):
    for axis, value in zip(axes, point):
        row[f'{name}_{axis}'] = round(float(value), 4)


@dataclass(frozen=True)
class DrawnPoints:
    """The serve start, the end of the pass and its top as they went into the row, in world coordinates.

    For the trajectory figure, which is drawn in the reconstruction's own frame
    rather than the row's server-at-the-bottom one. Each is None exactly when
    the row's columns for it are blank, so the figure shows what the dataset
    says and nothing else.
    """

    serve_start: Optional[np.ndarray] = None
    reception_end: Optional[np.ndarray] = None
    apex_point: Optional[np.ndarray] = None


def trajectory_fields(segments):
    """Everything the row takes from the fit, a note for each thing it could not, and `DrawnPoints`."""
    row, notes = {}, []
    found = find_pass(segments)
    if found is None:
        return row, ['no reception found on the parabolas, so nothing was measured'], DrawnPoints()

    # The serve came from the other side of the net to the reception, which
    # `find_pass` has put more than a metre past it - so this sign is sure.
    serve_side = -1.0 if found.reception_point[1] > 0 else 1.0

    def oriented(point):
        return dv_grid.orient_for_action([point], serve_side)[0]

    _put(row, 'reception', oriented(found.reception_point))

    serve_start = None
    path, reason = find_serve_path(segments)
    if path is None:
        notes.append(f'serve: {reason}')
    else:
        start = _serve_contact_frame(path, serve_side)
        serve_start = path.evaluate(start)
        _put(row, 'serve_start', oriented(serve_start))
        row['serve_hit_net'] = path.net_touch
        landing = _landing(path)
        if landing is None:
            notes.append('serve landing: the serve\'s parabola never comes down to the floor')
        else:
            _put(row, 'serve_landing', oriented(landing)[:2], axes='xy')
        speed = _average_speed(path, start)
        if speed is not None:
            row['serve_average_speed'] = round(speed, 3)

    if found.end_point is None:
        notes.append(f'reception end: {found.end_reason}')
    else:
        _put(row, 'reception_end', oriented(found.end_point))
    if found.apex is not None:
        row['reception_apex_z'] = round(found.apex, 3)
    notes.extend(f'pass: {note}' for note in found.notes)
    return row, notes, DrawnPoints(serve_start, found.end_point, found.apex_point)


def reception_row(dvw_filepath, reception_number, segments):
    """The dataset row for reception N, and notes on what was not measured.

    Returns `(row, notes, DrawnPoints)`, or `(None, [reason], DrawnPoints())`
    when the scout file cannot say whose serve and reception these were - then
    there is nothing to key the measurements to.
    """
    plays = get_plays(dvw_filepath)
    try:
        row = scout_fields(dvw_filepath, reception_number, plays)
    except (ValueError, IndexError) as error:
        return None, [str(error)], DrawnPoints()
    measured, notes, drawn = trajectory_fields(segments)
    row.update(measured)
    return row, notes, drawn


def save_reception(row, csv_path=RECEPTIONS_CSV):
    """Merge one row into the dataset, keyed by `(match, reception_number)`."""
    return merge_row(row, csv_path, CSV_COLUMNS)


def record_reception(dvw_filepath, reception_number, segments):
    """Measure the rally and add it to the dataset, printing what happened.

    Returns the `DrawnPoints` for the trajectory figure.
    """
    row, notes, drawn = reception_row(dvw_filepath, reception_number, segments)
    if row is None:
        print(f'  reception dataset: not recorded - {notes[0]}')
        return drawn
    save_reception(row)

    def show(name):
        if f'{name}_z' not in row:
            return '-'
        return f"({row[f'{name}_x']:.2f}, {row[f'{name}_y']:.2f}, {row[f'{name}_z']:.2f})"

    speed = row.get('serve_average_speed')
    apex = row.get('reception_apex_z')
    print(f"  reception dataset: {row['serving_player']} ({row['serve_type']},"
          f" {'-' if speed is None else f'{speed:.1f} m/s'}) -> {row['receiving_player']}"
          f" {row['reception_grade']}; serve {show('serve_start')},"
          f" reception {show('reception')},"
          f" apex {'-' if apex is None else f'{apex:.2f} m'}, end {show('reception_end')}")
    for note in notes:
        print(f'    not measured: {note}')
    return drawn
