"""Locating contacts within a reconstructed 3D trajectory.

A contact is where one free-flight model stops describing the ball and the next
starts, so everything here reads off `ballistic`'s segmentation. `find_reception`
does it for a serve reception, `find_attack_points` for the set, the attack and
where the attack finished; both locate a contact at the *seam* between two
flights via the shared `_contact_frame`.

Two ways of finding the *reception* live here:

- `find_reception` reads it off the ballistic segmentation: the reception is
  the start of the first new parabola once the ball is properly onto the far
  side of the net. This is the better answer whenever the segmentation exists,
  because a contact is *by definition* where one free-flight model stops
  explaining the ball and the next one starts, which is precisely what
  `ballistic.find_segments` already solved for.
- `find_serve_and_receive` is the older geometric reading, and still the one
  that finds the serve: the trajectory is split where y changes sign (the ball
  crossing the net), the serve is the pre-split point nearest the baseline, and
  the reception is the first sharp change of direction after the split. It
  remains the fallback for when there is no segmentation to read, and the thing
  to compare a new reception against.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ballistic import find_seam
from .court import BALL_VOLUME_X, BALL_VOLUME_Y, BALL_VOLUME_Z
from .trajectory import moving_average


# How far past the net, in meters, a contact has to be before it can be the
# reception. Without it the rule would fire on a serve that clips the net cord:
# that also ends one parabola and starts another, and does so a frame after the
# ball crosses, so "first new parabola on the far side" alone would call the net
# the reception and put the dig nowhere. A net touch happens at y ~ 0 by
# construction (the net is the plane y = 0 and the ball is 0.1 m across), so
# almost any margin excludes it; a metre is chosen to leave room for
# reconstruction error at the net while staying far short of where a serve is
# actually dug, which is 4-8 m in. The cost of the margin is a genuine contact
# made right at the net, which for a *serve* reception is not a thing that
# happens - if this is ever reused for rally play, revisit it.
#
# Not hypothetical: reception 10 of the svk-ork quarter final splits into a new
# parabola at frame 69 with the ball at y = -0.76 and the two arcs agreeing to
# 0.01 m, which is the serve clipping the cord. Without the margin that would be
# the reception; with it, the dig is found at y = -3.64 where it belongs.
NET_ENTRY_MARGIN = 1.0


def angle_between(v1, v2):
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)


def find_serve_and_receive(points):
    points = np.array(points)
    
    # --- 1. Find split index (y sign change) ---
    split_idx = None
    for i in range(1, len(points)):
        if points[i-1][1] * points[i][1] < 0:
            split_idx = i
            break
    
    if split_idx is None:
        # raise ValueError("No y sign change found.")
        split_idx = -1  # fallback to last index if no sign change found
    
    first_half = points[:split_idx]
    second_half = points[split_idx:]
    
    # --- 2. Serve location (closest to |y| = 9) ---
    serve_idx = np.argmin(np.abs(np.abs(first_half[:, 1]) - 9))
    serve_point = first_half[serve_idx]
    
    # --- 3. Smooth second half ---
    smoothed = moving_average(second_half, window_size=5)
    
    # --- 4. Detect direction change ---
    window = 3
    threshold = np.deg2rad(45)  # adjust if needed
    
    receive_point = None

    angles_temp_check = []
    for i in range(window, len(smoothed) - window):
        v1 = smoothed[i] - smoothed[i - window]
        v2 = smoothed[i + window] - smoothed[i]
        
        angle = angle_between(v1, v2)
        angles_temp_check.append(angle)
        
        if angle > threshold:
            receive_point = second_half[i]
            #plt.plot(range(len(angles_temp_check)), angles_temp_check)
            # plt.show()
            break
    
    # fallback if nothing detected
    if receive_point is None:
        print('no reception detected')
        receive_point = second_half[0] #fixa sen
    
    return serve_point, receive_point


def _is_on_court(point):
    """Is this somewhere a volleyball could actually be?

    The same generous bounds the reconstruction uses to reject impossible
    triangulations, reused here on a *fitted* point. A model evaluated outside
    the frames it observed is not bounded by anything the fit checked, so a
    point that lands off the court means an extrapolation went too far rather
    than that the ball did something surprising.
    """
    x, y, z = point
    return (BALL_VOLUME_X[0] <= x <= BALL_VOLUME_X[1]
            and BALL_VOLUME_Y[0] <= y <= BALL_VOLUME_Y[1]
            and BALL_VOLUME_Z[0] <= z <= BALL_VOLUME_Z[1])


def _contact_frame(before, segment):
    """The frame at which `segment`'s flight began, i.e. the contact that started it.

    With no gap before it, that is simply its first observed frame: segments are
    cut at the observation a single parabola cannot explain, and that point --
    the one most likely to be the contact itself -- is dropped, so the flight
    starts within a frame of where the observations resume.

    When the two segments are separated by a gap the answer is the seam instead,
    the frame at which `ballistic` handed the fill over from one model to the
    other. That frame is chosen as the one where the two flights most nearly
    agree about where the ball is, which is the same physical fact a contact is:
    position is continuous through it, velocity is not. Taking the segment start
    instead would be badly wrong here rather than a frame or two out -- real
    receptions leave gaps of 50-100 frames around the dig, so the observations
    resume with the ball already well back up in the air.

    A gap `find_seam` rejects gives no seam to use, and this falls back to the
    segment start. That is the honest answer and not a good one: the contact is
    somewhere inside a gap the fits could not agree across, so the reception will
    read late and high. It shows up in the `Report` as a skipped gap.
    """
    seam = find_seam(before, segment)
    if seam is None:
        return segment.start
    return seam[0]


def find_reception(segments, serve_side=None):
    """Find the reception contact from a trajectory's ballistic segmentation.

    The reception is the start of the first new parabola that begins once the
    ball is on the far side of the net, `NET_ENTRY_MARGIN` past it. Everything
    that rule steps over, it steps over for a reason: the first segment is the
    serve's own flight; a segment starting on the serving side is the serve or a
    mis-split of its flight, not a contact by the receiving team; and a segment
    starting at the net is the ball touching the cord on its way over.

    The point returned is the *new* segment's model evaluated at the contact
    frame, never the old one's and never anything in between. Where the contact
    falls inside a bridged gap the outgoing flight is the one that describes
    where the ball went, and the two models are known to agree there to within
    `ballistic.MAX_BRIDGE_SEAM` anyway; blending them would invent a position
    that is neither flight's.

    Args:
        segments: the fitted flights in frame order, as `ballistic.Report`
            carries them.
        serve_side: +1 or -1, the sign of y the serve was struck from. Inferred
            from where the first flight starts if not given.

    Returns:
        `(point, frame)` for the reception, or None if no segment qualifies --
        which means either that there is no segmentation to read, or that the
        ball was never seen to be contacted on the far side. The caller decides
        what to do about it; the pipeline falls back to `find_serve_and_receive`.
    """
    if len(segments) < 2:
        return None

    if serve_side is None:
        serve_side = np.sign(segments[0].evaluate(segments[0].start)[1])
    if serve_side == 0:
        return None

    for before, segment in zip(segments, segments[1:]):
        frame = _contact_frame(before, segment)
        point = segment.evaluate(frame)
        # Depth into the receiving half, so the test reads the same either way
        # round the court.
        if -serve_side * point[1] > NET_ENTRY_MARGIN:
            return point, frame

    return None


@dataclass(frozen=True)
class AttackPoints:
    """The three contacts an attack is made of, read off the fitted flights.

    Each point is a fitted parabola evaluated at the frame the contact happened,
    and nothing else - no blend of the two flights meeting there. Which of the
    two models is used follows the rule the reception already uses: a contact is
    described by the flight that *leaves* it, because that is the one that says
    where the ball went. The one exception is `end_point`, which is the end of
    the attack's own flight rather than the start of anything, so it is the
    attack's model that is evaluated there.

    `attack_index` is the position of the attack flight among the fitted
    segments; the figure uses it to draw the three flights an attack is made of
    - the defence or reception before the set, the set, and the attack - and
    nothing else.

    `set_point` is None when nothing was fitted before the attack, so there is
    no setting flight to locate a contact on. The attack and its end are still
    measured; a warning says which case it was.

    `warnings` is empty when the trajectory looked the way an attack should. It
    is not an error channel: a warning means something about the segmentation
    was not what an attack should look like, so that coordinate is worth less
    than the others and the clip is worth watching.
    """

    set_point: Optional[np.ndarray]
    set_frame: Optional[int]
    attack_point: np.ndarray
    attack_frame: int
    end_point: np.ndarray
    end_frame: int
    attack_index: int = 0        # which fitted flight the attack was
    crossed_net: bool = True
    warnings: Tuple[str, ...] = ()

    def __str__(self):
        def show(name, point, frame):
            if point is None:
                return f"{name} -"
            return f"{name} f{frame} {np.array2string(point, precision=2)}"
        line = (f"{show('set', self.set_point, self.set_frame)}, "
                f"{show('attack', self.attack_point, self.attack_frame)}, "
                f"{show('end', self.end_point, self.end_frame)}")
        return line + (f"  [{'; '.join(self.warnings)}]" if self.warnings else "")


def _flight_start(segments, index):
    """The frame flight `index` began at: its contact, or its own start."""
    if index == 0:
        return segments[0].start
    return _contact_frame(segments[index - 1], segments[index])


def _net_crossing_flights(segments):
    """The flights that carry the ball from one side of the net to the other.

    The side is read at the flight's **contact frame**, not at its first
    observed frame, and that distinction decides real cases. An attack is struck
    at the net, and the detector routinely loses the ball for the few frames it
    spends there - so the flight's observations often begin once it is already
    past. Attack 3 is the case: its spike is observed from y = +0.61 to +1.22
    and never spans zero, yet the flight before it ends at y = -0.59 and the
    fitted model puts the contact at y = -0.5. Testing observed frames alone
    calls that no crossing at all and loses the attack.

    Returns `[(index, contact_frame), ...]` in flight order.
    """
    crossing = []
    for index, segment in enumerate(segments):
        start = _flight_start(segments, index)
        if segment.evaluate(start)[1] * segment.evaluate(segment.end)[1] < 0:
            crossing.append((index, start))
    return crossing


def find_attack_points(segments):
    """The set, the attack contact and where the attack finished.

    The attack is identified by **what it does, not by where it sits**: it is
    the flight that carries the ball across the net, judged from its contact
    frame rather than its first observed one - see `_net_crossing_flights`. That is the defining property of an attack,
    and unlike counting flights it does not care what else the clip happened to
    catch.

    An earlier version took the first three flights on the assumption that a
    clip holds the previous rally's tail, the set and the attack in that order.
    That assumption is right more often than not but not reliably: measured over
    46 attacks it held 16 times, and in 12 more the attack was there but one or
    two flights earlier, because the clip started too late to catch the tail and
    every index shifted. The net-crossing rule recovers those without giving up
    any of the originals - across the same clips the crossing flight is at index
    2 fifteen times, index 1 twelve times and index 0 three times, and wherever
    index 2 crosses it *is* the crossing flight, so the rule is a strict
    superset of what counting already got right.

    It is also unambiguous in practice: **no clip measured had more than one
    net-crossing flight**, so there is nothing to disambiguate and no speed
    threshold is needed to do it. If more than one ever appears the earliest is
    taken and a warning says so.

    Around the attack flight, the other two points are the contacts either side
    of it:

    - the **set** is the contact that started the flight *before* the attack -
      the setter's touch;
    - the **attack** is the contact that started the attack flight;
    - the **end** is where the attack flight finished, the seam with whatever
      follows it, or its own last frame when nothing does.

    Contacts are **seams**, not segment starts, via the shared `_contact_frame`:
    the contact happens inside the gap neither flight observed, and by the time
    observations resume the ball has moved.

    When the attack is the first flight in the clip there is nothing before it
    to contact against, so no set is reported and its own start is the best that
    can be said about where it was struck. Both cases warn rather than guess.

    Returns None when no flight crosses the net. That is not the same failure as
    the old rule's "fewer than three flights": it means the attack was never
    reconstructed, which is a detection or matching problem, and the clip and the
    annotated videos are where to look.
    """
    crossing = _net_crossing_flights(segments)
    if not crossing:
        return None

    warnings = []
    if len(crossing) > 1:
        warnings.append(
            f"{len(crossing)} flights cross the net (at"
            f" {[i for i, _ in crossing]}); taking the first as the attack")

    index, attack_frame = crossing[0]
    attack_flight = segments[index]

    # The attack contact, and the set that fed it.
    if index == 0:
        warnings.append(
            f"the attack is the first flight in the clip, so its contact is the"
            f" first frame it was observed at ({attack_frame}) rather than a"
            f" seam with the flight before it")
    attack_point = attack_flight.evaluate(attack_frame)

    set_point = set_frame = None
    if index >= 2:
        set_frame = _contact_frame(segments[index - 2], segments[index - 1])
        set_point = segments[index - 1].evaluate(set_frame)
    elif index == 1:
        set_frame = segments[0].start
        set_point = segments[0].evaluate(set_frame)
        warnings.append(
            f"nothing was fitted before the set, so the set location is the"
            f" first frame the setting flight was observed at ({set_frame})"
            f" rather than the setter's contact")
    else:
        warnings.append(
            "no flight precedes the attack, so there is no set to locate")

    # Where the attack flight ended. `find_seam` directly rather than
    # `_contact_frame`, because that one falls back to the *next* segment's
    # first observed frame - right for asking where that flight began, a
    # disaster for evaluating the attack's own parabola there. Measured on
    # attack 1, whose attack ends at 142 and whose next flight is not seen until
    # 208, the fallback put the model 66 frames past its data, at y = 17.9 m and
    # 13 m below the floor.
    unbridged = False
    if index + 1 < len(segments):
        seam = find_seam(attack_flight, segments[index + 1])
        if seam is None:
            unbridged = True
            end_frame = attack_flight.end
        else:
            end_frame = seam[0]
    else:
        end_frame = attack_flight.end
    end_point = attack_flight.evaluate(end_frame)

    if unbridged:
        warnings.append(
            f"no seam between the attack flight and the one after it, so the end"
            f" is the attack's last observed frame ({end_frame}) rather than the"
            f" contact that stopped it - the ball ended somewhere in the gap")
    if end_frame <= attack_frame:
        warnings.append(
            f"the attack flight ends at frame {end_frame}, at or before its own"
            f" contact at {attack_frame}")
    for name, point in (('set', set_point), ('attack', attack_point),
                        ('end', end_point)):
        if point is not None and not _is_on_court(point):
            warnings.append(
                f"the {name} point {np.array2string(point, precision=2)} is"
                f" outside the volume a ball can be in")

    return AttackPoints(
        set_point=set_point, set_frame=set_frame,
        attack_point=attack_point, attack_frame=attack_frame,
        end_point=end_point, end_frame=end_frame,
        attack_index=index, crossed_net=True, warnings=tuple(warnings),
    )
