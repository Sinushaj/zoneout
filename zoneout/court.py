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
# calibration.points.get_screen_coordinates. Changing this list means
# re-clicking both calibration files.
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
