"""Canonical court geometry, in meters.

Single source of truth for the coordinate system used across the project:
x runs 0..9 across the court, y runs -9..9 along it with the net at y = 0,
and z is up. Both the 3D reconstruction (which needs the reference points for
solvePnP) and the figures (which draw the court) read their geometry from here,
so the two can never drift apart.
"""

COURT_WIDTH = 9.0     # x, sideline to sideline
HALF_LENGTH = 9.0     # |y|, net to baseline
ATTACK_LINE = 3.0     # |y| of the attack lines
NET_HEIGHT = 2.43
ANTENNA_HEIGHT = 3.23

# 3D reference points used as the world half of the 2D<->3D correspondence in
# solvePnP.
#
# ORDER IS SIGNIFICANT: it must match the order the corresponding pixels were
# clicked into gopro_points.csv / zve10_points.csv by
# calibration.points.pick_court_points, which asks for them one at a time in
# this order and marks each on a court diagram. Changing this list means
# re-clicking both calibration files; the picker also names and draws each
# point from its coordinates alone, so nothing else needs editing to match.
CALIBRATION_POINTS = [
    (0, -3, 0),
    (9, -3, 0),
    (9, 3, 0),
    (0, 3, 0),
    (0, -9, 0),
    (0, 9, 0),
    (0, 0, ANTENNA_HEIGHT),
    (9, 0, ANTENNA_HEIGHT),
]

# Generous bounds on where the ball can plausibly be during a serve and its
# reception, used by the reconstruction to reject triangulations that land
# somewhere physically impossible. Deliberately loose: a serve is struck from
# several meters behind the baseline and a ball can be played well outside the
# sidelines, so these are a sanity check, not a court boundary. The lower z
# bound dips below the floor to tolerate calibration error on a ball at ground
# level.
BALL_VOLUME_X = (-4.0, COURT_WIDTH + 4.0)
BALL_VOLUME_Y = (-HALF_LENGTH - 4.0, HALF_LENGTH + 4.0)
BALL_VOLUME_Z = (-0.5, 12.0)

# Court outline in the z = 0 plane, as a closed loop.
OUTLINE_X = [0, 9, 9, 0, 0]
OUTLINE_Y = [-9, -9, 9, 9, -9]
OUTLINE_Z = [0, 0, 0, 0, 0]

# Lines running across the court (x = 0..9): the net line and the two attack
# lines, 3 m either side of it.
CROSS_LINES = [
    ((0, 0, 0), (9, 0, 0)),
    ((0, 3, 0), (9, 3, 0)),
    ((0, -3, 0), (9, -3, 0)),
]
