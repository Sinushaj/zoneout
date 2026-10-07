"""The DataVolley coordinate grid: world metres in, grid index or zone out.

Pure arithmetic - no I/O, no pandas, no scout file. This is to the DataVolley
coordinate format what `court.py` is to the world geometry: the one place that
owns the conversion, so the serve writer and the attack writer cannot drift
apart on it.

The grid is defined by openvolley's `datavolley` R package (`R/xy.R`), whose
convention `pydatavolley` reads back with `helpers.dv_index2xy`. It is a
100 x 101 grid of cells over an area rather larger than the court, addressed by

    index = column + (row - 1) * 100          1 <= index <= 10100

in openvolley's own plotting units, where the court is `x in [0.5, 3.5]`,
`y in [0.5, 6.5]` and the net is at `y = 3.5`. One of those units is 3 m, so
mapping from `zoneout.court`'s metres (`x in [0, 9]`, `y in [-9, 9]`, net at
`y = 0`) is

    dv_x = 0.5 + x / 3                        dv_y = 3.5 + y / 3

and the court occupies **columns 11-90 and rows 11-91** - 80 columns of
0.1125 m across the width, 81 rows of 0.2222 m along the length. Note the
asymmetry: 80 against 81, which is where the old `coords_to_dvindex` went
wrong. It laid 80 rows over the 18 m length instead of 81 and truncated toward
zero instead of binning to the cell edges, which put every index about 0.11 m
out in x and up to a full cell (0.22 m) out in y, the error growing toward
`y = +9`. Small, but an attack's end coordinate is usually near a line and
in-or-out is exactly the question a 0.22 m bias sits on.

**Coordinates are written from the acting team's end**, which is a measured
result rather than a convention assumed - see `orient_for_action`.
"""

# The grid, in cells. `COLUMNS` is the stride an index is built with; `ROWS` is
# only a bound. Both come from openvolley's bin definitions.
COLUMNS = 100
ROWS = 101
MAX_INDEX = COLUMNS * ROWS          # 10100

# Where the court sits in that grid, and how big it is in metres. The court's
# near-left corner (x = 0, y = -9) is the bottom-left corner of cell
# (FIRST_COURT_COLUMN, FIRST_COURT_ROW), and the far-right corner (x = 9,
# y = +9) is the *top-right* corner of the last court cell - so a point exactly
# on the far line bins into the first cell outside, as openvolley's own
# `right = FALSE` binning does.
FIRST_COURT_COLUMN = 11
COURT_COLUMNS = 80
FIRST_COURT_ROW = 11
COURT_ROWS = 81

COURT_WIDTH = 9.0                   # metres along x
COURT_LENGTH = 18.0                 # metres along y, baseline to baseline

# How far outside the court the grid still reaches, so a ball dug from behind
# the baseline or beyond the sideline still has an index to be written as.
# Columns 1-10 and 91-100 are 1.125 m of run-off either side; rows 1-10 and
# 92-101 are 2.22 m. Anything further out has no index at all and is clamped -
# see `xy_to_index`.


def _bin(value, span, cells, first_cell):
    """Which cell `value` falls in, binning to cell edges as openvolley does."""
    return first_cell + int((value * cells) // span)


def _clamp(value, low, high):
    return max(low, min(high, value))


def xy_to_index(x, y):
    """World metres -> the DataVolley grid index, as an int in 1..10100.

    Off-court points are fine and expected - a ball is dug from outside the
    sideline and lands beyond the baseline - and the grid has about a metre of
    run-off along x and two along y for exactly that. Further out than the grid
    reaches, the index is **clamped to the edge** rather than refused, because
    the caller is the only thing that can judge whether a point that far out is
    a real measurement or a bad reconstruction, and it already does: the
    plausibility gate is `court.BALL_VOLUME_*` and the warnings
    `events.find_attack_points` attaches. A clamp here would be the second place
    that decision got made, and the two would disagree.
    """
    column = _clamp(_bin(x, COURT_WIDTH, COURT_COLUMNS, FIRST_COURT_COLUMN),
                    1, COLUMNS)
    row = _clamp(_bin(y + COURT_LENGTH / 2, COURT_LENGTH, COURT_ROWS,
                      FIRST_COURT_ROW), 1, ROWS)
    return column + (row - 1) * COLUMNS


def index_to_xy(index):
    """The inverse of `xy_to_index`: the **centre** of that cell, in metres.

    Not a round trip - a cell is 0.11 x 0.22 m and the index is all that
    survives being written to the file, so this is the best the format can say
    about where the ball was. Used to report what was actually written.
    """
    if not 1 <= index <= MAX_INDEX:
        raise ValueError(f"{index} is not a DataVolley grid index (1-{MAX_INDEX}).")

    column = (index - 1) % COLUMNS + 1
    row = (index - 1) // COLUMNS + 1
    x = (column - FIRST_COURT_COLUMN + 0.5) * COURT_WIDTH / COURT_COLUMNS
    y = ((row - FIRST_COURT_ROW + 0.5) * COURT_LENGTH / COURT_ROWS
         - COURT_LENGTH / 2)
    return x, y


def flip_index(index):
    """The same point seen from the other end of the court."""
    return MAX_INDEX + 1 - index


def format_index(index):
    """The index as it is written into a scout line's coordinate field.

    Zero-padded to four characters, which is what every coordinate already in
    these files uses and what DataVolley writes. Four is enough for anywhere on
    the court - the highest in-court index is 9091 - and only the two rows above
    the far baseline run to five digits, which is left to widen rather than be
    truncated into a different cell.
    """
    return f'{index:04d}'


NO_COORDINATE = '-1-1'
"""What a scout line carries in a coordinate field it has no value for."""


# --- Zones -----------------------------------------------------------------
#
# DataVolley's zones are a 3x3 grid over one half-court, numbered from the
# attacking team's own point of view. Looking at a team's own half with the net
# at the top and their right hand at the right:
#
#       net        4  3  2
#                  7  8  9
#       baseline   5  6  1
#
# so zone 1 is the right-back corner a serve is struck from, zone 4 the
# left-front corner most attacks are hit from, and zone 6 the middle back.
#
# Because the numbering is relative to the team, the *far* half is this same
# picture turned 180 degrees, and `xy_to_zone` folds it that way.
#
# The layout above is not asserted from a rulebook, it is measured. Against the
# 201 coordinates a human scout entered by hand into
# `&32135954_Hom15 Örkelljunga vs Sollentuna .dvw` - the one file to hand whose
# coordinates this pipeline did not write - the zone computed from the scout's
# own clicked index agrees with the zone they typed on the same line 81% of the
# time for start zones and 93% for end zones. The residual is the scout
# disagreeing with themselves rather than a mirror: the confusion matrix is
# diagonal, and 13 of the 15 end-zone misses are one step along *depth* in the
# same direction (typed as a back-row zone, clicked just in front of the 6 m
# line). A mirrored layout would show as 1<->5 or 4<->2 confusions, and there
# are none.
_NEAR_HALF_ZONES = (
    (5, 6, 1),      # back third,   x low -> high
    (7, 8, 9),      # middle third
    (4, 3, 2),      # front third, at the net
)


def xy_to_zone(x, y):
    """World metres -> `(zone, half)`: the zone 1-9, and which half it is in.

    `half` is `-1` for the `y < 0` half and `+1` for `y > 0`, so it says which
    team's zone numbering was used without needing to know anything about teams,
    sets or ends. Reading the half off the coordinate rather than off the scout
    file is the general answer: it works for a file where the teams swapped ends
    mid-set, or where the scout's court orientation is unknown.

    A point beyond a line belongs to the nearest zone; there is no "out" zone,
    and the scout file has none either.
    """
    half = -1 if y < 0 else 1

    # Fold the far half onto the near one by turning it 180 degrees about the
    # centre of the court, which is exactly what the team-relative numbering
    # means. The near half is then always `x` rightward and `y` toward the net.
    if half > 0:
        x, y = COURT_WIDTH - x, -y

    across = _clamp(int(x * 3 // COURT_WIDTH), 0, 2)
    along = _clamp(int((y + COURT_WIDTH) * 3 // COURT_WIDTH), 0, 2)
    return _NEAR_HALF_ZONES[along][across], half


def orient_for_action(points, half):
    """Turn an action's world points so the acting team is at the bottom.

    This is the answer to the orientation question `ATTACK_PLAN.md` §5 posed,
    and it is a measurement, not a convention adopted for tidiness. openvolley
    warns that coordinates "might not appear in the dvw file in any particular
    orientation", and every coordinate in the two match files this pipeline has
    processed was written by the pipeline itself, so neither could settle it.

    `&32135954_Hom15 Örkelljunga vs Sollentuna .dvw` could. It is an Elitserien
    file scouted by hand, untouched by this pipeline, and it carries 201 lines
    with coordinates across attacks, serves, receptions and digs. **All 201
    start in the bottom half of the grid and end in the top, for both teams
    alike** - 104 attacks, 31 serves, 27 receptions and 39 digs, without one
    exception. So the convention is strictly action-relative: whoever is acting
    is drawn at the bottom and the ball always travels up the picture.

    That is what `scout.add_serve_direction` has always done for serves by
    rotating whenever the serve started at `y > 0`, on a guess that `NOTES.md`
    recorded as unverified. It was right, and it now applies to attacks for the
    same measured reason instead of by imitation.

    `half` is the side the acting team was on, `-1` or `+1`, as `xy_to_zone`
    returns it - read off the reconstruction rather than looked up from team and
    set, so it holds for a file where the teams swapped ends mid-set. Points may
    be any length; only `x` and `y` are touched and `z` is passed through, since
    the grid has no height.

    Returns a new list; the caller's points are not modified.
    """
    if half < 0:
        return [None if p is None else list(p) for p in points]
    return [None if p is None else [COURT_WIDTH - p[0], -p[1], *p[2:]]
            for p in points]


def zone_centre(zone, half):
    """The middle of a zone, in world metres. For reporting and figures."""
    for along, row in enumerate(_NEAR_HALF_ZONES):
        for across, number in enumerate(row):
            if number != zone:
                continue
            x = (across + 0.5) * COURT_WIDTH / 3
            y = (along + 0.5) * COURT_WIDTH / 3 - COURT_WIDTH
            return (COURT_WIDTH - x, -y) if half > 0 else (x, y)
    raise ValueError(f"{zone} is not a DataVolley zone (1-9).")
