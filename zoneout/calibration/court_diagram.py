"""The small top-down court diagram that says *which* point to click next.

The calibration clicks are matched positionally with `court.CALIBRATION_POINTS`
when they are handed to solvePnP, so clicking them in the wrong order does not
fail — it quietly produces a wrong camera pose. Until now the only statement of
that order was the source of `court.CALIBRATION_POINTS` itself. This module
draws it: a plan view of the court with every reference point on it, the one
being asked for highlighted, and the ones already clicked marked as done.

Everything here is derived from `zoneout.court` rather than written out again,
including the text describing each point, so a change to `CALIBRATION_POINTS`
is reflected without editing this file. It draws on a plain `tkinter.Canvas`
and knows nothing about video, clicks or files.

The colors deliberately mirror `figures.style` (the same orange court, the same
white lines) but are written out here instead of imported: importing anything
from `zoneout.figures` pulls in matplotlib and plotly, which a calibration tool
has no use for — the same reason `zoneout/__init__.py` is free of imports.
"""

from ..court import ANTENNA_HEIGHT, ATTACK_LINE, COURT_WIDTH, HALF_LENGTH

# Court surface and markings, matching zoneout.figures.style.
COURT_FILL = "#E0762F"
COURT_LINE = "#FFFFFF"

# Marker states. A point is never identified by color alone — every marker
# carries its number, and the pending/done/current states differ in fill,
# outline width and size as well as hue.
PENDING_FILL = "#FFFFFF"
PENDING_OUTLINE = "#374151"
DONE_FILL = "#059669"
CURRENT_FILL = "#1D4ED8"
LABEL_COLOR = "#111827"
PANEL_BG = "#F3F4F6"

# Clear space around the court, for the numbers sitting outside it.
MARGIN = 30


def describe_point(point):
    """A human name for one calibration point, derived from its coordinates.

    Written in terms of court features rather than left/right or near/far,
    which mean nothing until you know where the camera stands — the diagram is
    what says which of the four corners is meant, and this says what kind of
    thing to look for once you are there.
    """
    x, y, z = point

    along = ("net line" if y == 0 else
             "attack line" if abs(y) == ATTACK_LINE else
             "baseline" if abs(y) == HALF_LENGTH else
             f"y = {y:g} m")
    across = ("sideline" if x in (0, COURT_WIDTH) else f"x = {x:g} m")

    if along == "baseline" and across == "sideline":
        where = "court corner"
    else:
        where = f"{along} × {across}"

    if z == 0:
        return where
    if z == ANTENNA_HEIGHT:
        return f"top of the antenna, at the {where}"
    return f"{where}, {z:g} m above the floor"


def is_elevated(point):
    """Whether a point is off the floor, and so not where the plan view puts it."""
    return point[2] != 0


class CourtDiagram:
    """Draws the reference points on a plan view of the court.

    The court is drawn from `zoneout.court`'s own dimensions, so it is the same
    court the reconstruction solves against. `rotate()` turns the *picture*
    only — the coordinates are untouched — so that the diagram can be put the
    way round the camera sees the court, which is the whole difficulty in
    telling one corner from another.
    """

    def __init__(self, canvas, points):
        self.canvas = canvas
        self.points = list(points)
        self.rotation = 0

    def rotate(self):
        self.rotation = (self.rotation + 1) % 4

    def _projection(self):
        """A world (x, y) -> canvas (px, py) mapping that fits the current size."""
        width = max(self.canvas.winfo_width(), 1)
        height = max(self.canvas.winfo_height(), 1)

        # Half-extents of the court after rotation: 9 x 18 becomes 18 x 9 on the
        # quarter turns, and the fit has to follow it round.
        half_u, half_v = _rotate(COURT_WIDTH / 2, HALF_LENGTH, self.rotation)
        half_u, half_v = abs(half_u), abs(half_v)

        scale = min((width - 2 * MARGIN) / (2 * half_u),
                    (height - 2 * MARGIN) / (2 * half_v))
        centre_x, centre_y = width / 2, height / 2

        def project(x, y):
            u, v = _rotate(x - COURT_WIDTH / 2, y, self.rotation)
            # Canvas y grows downward, world y grows up the court.
            return centre_x + u * scale, centre_y - v * scale

        return project, scale, (centre_x, centre_y)

    def draw(self, placed, current):
        """Redraw for the given state.

        `placed` is one entry per point, falsy where it has not been clicked
        yet; `current` is the index being asked for, or None.
        """
        canvas = self.canvas
        canvas.delete("all")
        project, scale, centre = self._projection()

        canvas.create_polygon(
            *_flatten(project(x, y) for x, y in
                      ((0, -HALF_LENGTH), (COURT_WIDTH, -HALF_LENGTH),
                       (COURT_WIDTH, HALF_LENGTH), (0, HALF_LENGTH))),
            fill=COURT_FILL, outline=COURT_LINE, width=2)

        for y in (-ATTACK_LINE, ATTACK_LINE):
            canvas.create_line(*project(0, y), *project(COURT_WIDTH, y),
                               fill=COURT_LINE, width=1)

        canvas.create_line(*project(0, 0), *project(COURT_WIDTH, 0),
                           fill=COURT_LINE, width=3)
        # Set off the net line in world coordinates, so the label stays clear
        # of the line it names however the diagram is turned.
        canvas.create_text(
            *project(COURT_WIDTH / 2, 0.7), text="net",
            fill=COURT_LINE, font=("TkDefaultFont", 8),
            angle=0 if self.rotation % 2 == 0 else 90)

        for index, point in enumerate(self.points):
            self._draw_marker(index, point, project, centre,
                              bool(placed[index]), index == current)

    def _draw_marker(self, index, point, project, centre, done, current):
        canvas = self.canvas
        px, py = project(point[0], point[1])

        if current:
            fill, outline, width, radius = CURRENT_FILL, COURT_LINE, 3, 8
        elif done:
            fill, outline, width, radius = DONE_FILL, COURT_LINE, 2, 6
        else:
            fill, outline, width, radius = PENDING_FILL, PENDING_OUTLINE, 1, 5

        if current:
            # A ring outside the marker, so the point being asked for is
            # findable at a glance and not by hue alone.
            canvas.create_oval(px - radius - 5, py - radius - 5,
                               px + radius + 5, py + radius + 5,
                               outline=CURRENT_FILL, width=2, dash=(3, 2))

        if is_elevated(point):
            # Up in the air, so it is not really on the floor where the plan
            # view has to put it: a triangle rather than a disc says so.
            canvas.create_polygon(
                px, py - radius - 2, px + radius + 1, py + radius,
                px - radius - 1, py + radius,
                fill=fill, outline=outline, width=width)
        else:
            canvas.create_oval(px - radius, py - radius, px + radius, py + radius,
                               fill=fill, outline=outline, width=width)

        # Number pushed away from the middle of the court, so it never sits on
        # the court lines the point is defined by.
        dx, dy = px - centre[0], py - centre[1]
        length = (dx * dx + dy * dy) ** 0.5 or 1.0
        offset = radius + 11
        canvas.create_text(px + dx / length * offset, py + dy / length * offset,
                           text=str(index + 1), fill=LABEL_COLOR,
                           font=("TkDefaultFont", 9, "bold" if current else "normal"))


def _rotate(u, v, quarter_turns):
    for _ in range(quarter_turns % 4):
        u, v = -v, u
    return u, v


def _flatten(pairs):
    return [value for pair in pairs for value in pair]
