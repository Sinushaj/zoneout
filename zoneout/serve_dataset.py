"""The serve dataset: one row per serve, written as the reception pipeline runs.

Every reception the pipeline processes has a serve in front of it, and the
ballistic fit has already modelled that serve's flight. This module turns the
flight into one row of `datasets/serves.csv`:

- who served: the match, set, `video_time`, team, player number and name, read
  off the serve line in the scout file;
- `start_*`, the ball at the server's contact, and `end_*`, at the reception
  contact, in metres; `contact_observed` says whether the contact was seen or
  extrapolated (below);
- `duration_s` and `average_speed`, the path length of the flight divided by
  its duration;
- `v0_*` (m/s) and `a_*` (m/s^2), which with `start_*` are the whole parabola:

      p(t) = start + v0 * t + 0.5 * a * t**2,   0 <= t <= duration_s

  so `p(duration_s)` is `end`, and anything else about the flight - speed at
  the net, height over it, launch angle - can be recomputed without re-running
  detection.

**Every row is oriented with the server at the bottom**: the serve is struck
from the `y < 0` end and travels toward `+y`, `x` running 0-9 across the court.
That is the convention the `.dvw` itself uses (`dv_grid.orient_for_action`), and
it is what makes a player's serves from both ends comparable without flipping.

**The contact is usually extrapolated.** The ball is mostly not tracked near
the server: of the 100 cached receptions that give a row, only 20 have a toss
fitted before the serve whose model hands over to the serve's, and only there
is the contact observed. Everywhere else the serve's
own parabola is run backwards to `SERVE_CONTACT_DEPTH`, the baseline, and the
row says `contact_observed = False`. See that constant for why the baseline.

Only serves that were received are here, since the pipeline runs over
receptions: service errors are missing, aces scouted as `R=` are not.

Rows are merged by `(match, reception_number)`, so re-running a reception
replaces its row and a batch that stops half-way keeps what it measured.
"""

import csv
import os

import numpy as np

from . import dv_grid
from .ballistic import FPS
from .events import find_serve_flight
from .scout import get_plays

SERVES_CSV = os.path.join('datasets', 'serves.csv')

# Where an unobserved serve contact is placed: this far behind the net, i.e. on
# the baseline. A parabola alone does not say where along it the ball was
# struck, so this is an assumption, and it is the one the observed contacts
# support. Over the 20 cached serves whose toss hands over to the serve, the
# contact sits at 8.2-9.8 m with a median of 9.0, and 2.7-3.3 m up. Running the
# unobserved serves back to 9.0 m puts them at 2.6-3.4 m up, the same heights,
# which the depth alone could not have forced - that agreement is the check that
# the backward extrapolation is sound. Its cost is that a server standing well
# behind the line is still placed on it; this footage has no such serve to
# measure.
#
# A serve first seen already behind this depth is not moved: going further back
# would need a depth nothing measured.
SERVE_CONTACT_DEPTH = 9.0

CSV_COLUMNS = [
    'match', 'set_number', 'reception_number', 'video_time',
    'team', 'player_number', 'player_name',
    'contact_observed',
    'start_x', 'start_y', 'start_z',
    'end_x', 'end_y', 'end_z',
    'duration_s', 'average_speed',
    'v0_x', 'v0_y', 'v0_z',
    'a_x', 'a_y', 'a_z',
]


def serve_for_reception(dvw_filepath, reception_number, plays=None):
    """The scout file's serve line for the rally of reception N, as a row.

    The serve is the last serve line above the reception. Checked rather than
    assumed: it must be in the same set and by the other team, or this raises
    instead of attributing the serve to the wrong player. `plays` is
    `get_plays(dvw_filepath)`, for a caller that has already parsed the file.
    """
    if plays is None:
        plays = get_plays(dvw_filepath)
    reception = plays[plays['skill'] == 'Reception'].iloc[reception_number - 1]
    serves = plays[(plays['skill'] == 'Serve')
                   & (plays['scout_line'] < reception['scout_line'])]
    if serves.empty:
        raise ValueError(f'reception {reception_number} has no serve line above it')

    serve = serves.iloc[-1]
    if serve['set_number'] != reception['set_number'] or serve['team'] == reception['team']:
        raise ValueError(
            f"reception {reception_number} (line {reception['scout_line']}) does not"
            f" follow a serve by the other team in its set; the nearest serve above"
            f" it is line {serve['scout_line']}, {serve['code']}")
    return serve


def _physical_flight(flight):
    """`(start, v0, a, duration)` in metres and seconds, from the contact frame.

    `Segment` stores its model per frame from its first observed frame; this
    moves the origin to the contact and converts to seconds.
    """
    p0, v, a = flight.segment.coefficients
    dt = flight.start_frame - flight.segment.start
    start = flight.segment.evaluate(flight.start_frame)
    velocity = (v + a * dt) * FPS
    acceleration = a * FPS * FPS
    duration = (flight.end_frame - flight.start_frame) / FPS
    return start, velocity, acceleration, duration


def _orient(start, velocity, acceleration):
    """Turn the flight so the server is at the `y < 0` end (`orient_for_action`)."""
    half = -1 if start[1] < 0 else 1
    start = np.array(dv_grid.orient_for_action([start], half)[0], dtype=float)
    if half > 0:
        flip = np.array([-1.0, -1.0, 1.0])
        velocity, acceleration = velocity * flip, acceleration * flip
    return start, velocity, acceleration


def _time_at_depth(start, velocity, acceleration, depth):
    """The latest time t <= 0 at which the flight was `depth` m behind the net.

    Solves `y(t) = -depth` for the oriented flight. None when it never was, or
    when the flight starts at or behind that depth already.
    """
    if start[1] <= -depth:
        return None
    a, b, c = 0.5 * acceleration[1], velocity[1], start[1] + depth
    if abs(a) < 1e-9:
        roots = [-c / b] if b else []
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        roots = [(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)]
    earlier = [t for t in roots if t <= 0]
    return max(earlier) if earlier else None


def _extrapolate_to_contact(start, velocity, acceleration, duration):
    """Run the flight back to `SERVE_CONTACT_DEPTH`; unchanged if it cannot be."""
    t = _time_at_depth(start, velocity, acceleration, SERVE_CONTACT_DEPTH)
    if t is None:
        return start, velocity, duration
    start = start + velocity * t + 0.5 * acceleration * t * t
    return start, velocity + acceleration * t, duration - t


def _average_speed(start, velocity, acceleration, duration):
    """Path length over duration, with the path sampled once per frame."""
    frames = max(int(round(duration * FPS)), 1)
    t = np.linspace(0.0, duration, frames + 1)[:, None]
    path = start + velocity * t + 0.5 * acceleration * t * t
    length = float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
    return length / duration if duration > 0 else float('nan')


def serve_row(dvw_filepath, reception_number, segments):
    """The dataset row for the serve before reception N, or `(None, reason)`.

    `segments` is the ballistic fit of that reception's clip. Returns
    `(row, None)` or `(None, reason)`; nothing here raises for a serve that
    simply could not be measured, since that is routine in a batch run.
    """
    flight, reason = find_serve_flight(segments)
    if flight is None:
        return None, reason

    try:
        serve = serve_for_reception(dvw_filepath, reception_number)
    except (ValueError, IndexError) as error:
        return None, str(error)

    start, velocity, acceleration, duration = _physical_flight(flight)
    start, velocity, acceleration = _orient(start, velocity, acceleration)
    if not flight.contact_observed:
        start, velocity, duration = _extrapolate_to_contact(
            start, velocity, acceleration, duration)
    end = start + velocity * duration + 0.5 * acceleration * duration * duration

    row = {
        'match': os.path.splitext(os.path.basename(dvw_filepath))[0],
        'set_number': serve['set_number'],
        'reception_number': reception_number,
        'video_time': serve['video_time'],
        'team': serve['team'],
        'player_number': serve['player_number'],
        'player_name': serve['player_name'],
        'contact_observed': flight.contact_observed,
        'duration_s': round(duration, 4),
        'average_speed': round(_average_speed(start, velocity, acceleration, duration), 3),
    }
    for name, vector in (('start', start), ('end', end),
                         ('v0', velocity), ('a', acceleration)):
        for axis, value in zip('xyz', vector):
            row[f'{name}_{axis}'] = round(float(value), 4)
    return row, None


def save_serve(row, csv_path=SERVES_CSV):
    """Merge one row into the dataset, keyed by `(match, reception_number)`."""
    return merge_row(row, csv_path, CSV_COLUMNS)


def merge_row(row, csv_path, columns):
    """Merge one row into a per-reception dataset, keyed by `(match, reception_number)`.

    Shared by every dataset the reception pipeline writes. Rows come back
    sorted by that key, and a row already there for the same reception is
    replaced, so re-running a reception after a fix never duplicates it.
    """
    rows = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as existing:
            for old in csv.DictReader(existing):
                rows[(old['match'], int(old['reception_number']))] = old
    rows[(row['match'], int(row['reception_number']))] = row

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow(rows[key])
    return csv_path


def record_serve(dvw_filepath, reception_number, segments):
    """Measure the serve and add it to the dataset, printing what happened."""
    row, reason = serve_row(dvw_filepath, reception_number, segments)
    if row is None:
        print(f'  serve dataset: not recorded - {reason}')
        return None
    save_serve(row)
    print(f"  serve dataset: {row['team']} #{row['player_number']},"
          f" {row['average_speed']:.1f} m/s over {row['duration_s']:.2f} s,"
          f" ({row['start_x']:.2f}, {row['start_y']:.2f}, {row['start_z']:.2f})"
          f" -> ({row['end_x']:.2f}, {row['end_y']:.2f}, {row['end_z']:.2f})")
    return row
