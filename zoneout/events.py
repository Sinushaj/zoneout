"""Locating contacts within a reconstructed 3D trajectory.

A contact is where one free-flight model stops describing the ball and the next
starts, so everything here reads off `ballistic`'s segmentation. `find_reception`
does it for a serve reception, `find_serve_flight` for the serve's own flight
from the server's contact to the reception (for the serve dataset),
`find_serve_path` and `find_pass` for the serve, the pass and where the pass
ended (for the serve-and-reception dataset), and `find_attack_points` for
the set, the attack and where the attack finished; all locate a contact at the
*seam* between two flights via the shared `_contact_frame`.

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

from .ballistic import FPS, INLIER_TOLERANCE, MAX_BRIDGE_SEAM, find_seam
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

# How far behind the net, in meters, an *observed* serve contact must be. A
# serve is struck behind the baseline, or a metre inside it at the top of a
# jump: the 20 contacts seen on cached receptions sit at 8.2-9.8 m. A flight
# handed over to nearer the net than this is not the serve leaving the server's
# hand but the serve itself split in two by the fit - reception 103's "contact"
# was 1.1 m from the net - so no single parabola covers the serve.
MIN_SERVE_CONTACT_DEPTH = 7.0


# A high set can leave the top of a camera's picture and come back down into it,
# and a set that does is fitted as two flights with the apex missing between
# them - the rising half and the falling half. `_set_cut_off_above` recognises
# that, so the set is located where the ball left the setter rather than where
# it came back into view. Attack 59 is the case it was built on: seen rising to
# 4.8 m in frames 80-101, gone for 64 frames, seen falling from 4.8 m in
# 166-186, and the set was being located up there, at the start of the falling
# half.
#
# How close to the top edge of a picture, in pixels of the 1920x1080 calibration
# space, the ball has to be where it disappears and where it reappears. The
# detector needs most of the ball in frame, so its last box sits a little below
# the edge: 44 and 48 px on attack 59's baseline camera. 100 leaves room for
# that without reaching the part of the picture a set is normally seen in.
TOP_EDGE_MARGIN = 100

# How far apart, horizontally, the rising half's model extended across the gap
# and the falling half's first observation may be, for the two to be one ball.
# Horizontal only: two short halves cannot pin the vertical acceleration, and
# after a second of extrapolation attack 59's disagree by 1.7 m in height while
# agreeing to 0.55 m across the floor. The value is `MAX_BRIDGE_SEAM`'s, the
# tolerance used everywhere else for "these two models are the same ball".
CUT_OFF_SET_MAX_DISTANCE = 1.0


def _vertical_velocity(segment, frame):
    """The model's vertical velocity at `frame`, in m/s."""
    dt = float(frame - segment.start)
    return float((segment.coefficients[1][2] + segment.coefficients[2][2] * dt) * FPS)


def _top_edge_camera(point, cameras_near_top_edge):
    """The cameras showing `point` at the top edge of their picture."""
    if cameras_near_top_edge is None:
        from .reconstruction import cameras_near_top_edge
    return cameras_near_top_edge(point, TOP_EDGE_MARGIN)


def _set_cut_off_above(rising, falling, cameras_near_top_edge=None):
    """The camera a set left and re-entered through the top of, or None.

    `rising` and `falling` are consecutive fitted flights. They are taken to be
    one set cut in two by the top of a picture when all three hold:

    1. **The apex was never seen**: `rising` is still going up at its last
       observed frame and `falling` is already coming down at its first.
    2. **The ball left through the top of a picture**: where `rising` was last
       seen and where `falling` was first seen are both within
       `TOP_EDGE_MARGIN` of the top edge of the *same* camera.
    3. **It is the same ball**: `rising`'s model carried across the gap lands
       within `CUT_OFF_SET_MAX_DISTANCE` horizontally of where `falling` is
       first seen.

    Condition 1 is what separates it from everything else: of 30 cached
    attacks with a flight before the set, attack 59 is the only one where that
    flight is rising at its end - in every other it is the pass or dig, falling.
    """
    if not (_vertical_velocity(rising, rising.end) > 0
            and _vertical_velocity(falling, falling.start) < 0):
        return None

    cameras = (_top_edge_camera(rising.evaluate(rising.end), cameras_near_top_edge)
               & _top_edge_camera(falling.evaluate(falling.start),
                                  cameras_near_top_edge))
    if not cameras:
        return None

    carried = rising.evaluate(falling.start)[:2]
    if np.linalg.norm(carried - falling.evaluate(falling.start)[:2]) > CUT_OFF_SET_MAX_DISTANCE:
        return None
    return sorted(cameras)[0]


def _set_entered_from_above(segments, set_index, cameras_near_top_edge=None):
    """The camera a set flight first appeared falling from the top of, or None.

    The case `_set_cut_off_above` cannot rescue: the set is first seen coming
    down from the top edge of a picture, so it went up out of view, but no
    rising half was fitted to locate the setter's touch on. Its contact would be
    read 5-6 m in the air - attacks 10, 38 and 44 of the Örkelljunga match, all
    seen again only after a 90-113 frame gap. A set location that wrong is worse
    than none.

    Requires frames before the set that no flight observed (a gap, or nothing
    fitted before it): a ball tracked continuously into the start of a flight
    did not come from above the picture, whatever its height.
    """
    flight = segments[set_index]
    if _vertical_velocity(flight, flight.start) >= 0:
        return None
    if set_index > 0 and flight.start - segments[set_index - 1].end <= 1:
        return None
    cameras = _top_edge_camera(flight.evaluate(flight.start), cameras_near_top_edge)
    return sorted(cameras)[0] if cameras else None


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
    found = _reception_contact(segments, serve_side)
    if found is None:
        return None
    _, point, frame = found
    return point, frame


def _serve_side(segments):
    """+1 or -1, the sign of y the first flight starts on; 0 if it is on the net."""
    return np.sign(segments[0].evaluate(segments[0].start)[1])


def _reception_contact(segments, serve_side=None):
    """`find_reception`, plus the index of the segment the reception starts.

    Returns `(index, point, frame)` or None. Kept separate so `find_serve_flight`
    reads the reception by exactly the same rule rather than a copy of it.
    """
    if len(segments) < 2:
        return None

    if serve_side is None:
        serve_side = _serve_side(segments)
    if serve_side == 0:
        return None

    for index in range(1, len(segments)):
        frame = _contact_frame(segments[index - 1], segments[index])
        point = segments[index].evaluate(frame)
        # Depth into the receiving half, so the test reads the same either way
        # round the court.
        if -serve_side * point[1] > NET_ENTRY_MARGIN:
            return index, point, frame

    return None


@dataclass(frozen=True)
class ServeFlight:
    """The serve's own fitted flight, from the server's contact to the reception.

    `start_frame` is the contact that started the flight and `end_frame` the
    reception contact, both clip frames; the ball is `segment.evaluate(f)` for
    every frame between. `segment` is the one parabola that covers the whole of
    it - `find_serve_flight` refuses a serve that needs more than one.

    `contact_observed` says whether `start_frame` is the server's contact or
    only where tracking picked the ball up. It is the contact when a flight
    before it - the toss - hands over to it there, the two models agreeing on
    where the ball is. Otherwise the ball was first seen already in flight: over
    the cached receptions that is 80 serves in 100, typically 5-15 frames after
    the contact, and in 14 the clip started after it.
    """

    segment: object         # ballistic.Segment
    start_frame: int
    end_frame: int
    contact_observed: bool  # a flight before it (the toss) hands over at start_frame


def find_serve_flight(segments):
    """The serve's flight, read off the segmentation, or the reason there is none.

    The serve is the last flight that begins more than `NET_ENTRY_MARGIN` back
    on the serving side before the reception. "Last" is what steps over the toss,
    which is sometimes fitted as a flight of its own (reception 5); the margin is
    what steps over a flight that begins at the net cord. It starts at its
    contact frame, the same seam-based reading every other contact uses, and it
    ends at the reception's contact frame, as `find_reception` reads it.

    Evaluated on the *serve's* model, not the reception's. The two agree at the
    reception to within `ballistic.MAX_BRIDGE_SEAM`, and using the serve's makes
    the start point, the end point and the parabola between them one model.

    Refused, with the reason:

    - no reception on the parabolas, or no flight before it on the serving side;
    - more than one flight between the serve and the reception. The serve then
      touched the net cord (reception 10) or was split mid-flight, and no single
      parabola goes from the server to the passer;
    - a flight handed over to nearer the net than `MIN_SERVE_CONTACT_DEPTH`,
      which is the serve split in two rather than the server's contact;
    - a gap before the reception that the fits could not bridge. The reception
      frame is then only where observation resumed, and running the serve's
      model to it would be extrapolating across a gap already judged too wide.

    Returns `(ServeFlight, None)` or `(None, reason)`.
    """
    located, reason = _locate_serve(segments)
    if located is None:
        return None, reason
    serve_index, start_frame, reception_index, end_frame = located
    serve_side = _serve_side(segments)

    between = reception_index - serve_index - 1
    if between:
        return None, (f'{between} more flight(s) between the serve and the reception'
                      f' (net touch or a split serve), so no single parabola covers it')

    serve, reception = segments[serve_index], segments[reception_index]
    if reception.start - serve.end > 1 and find_seam(serve, reception) is None:
        return None, (f'frames {serve.end + 1}-{reception.start - 1} before the'
                      f' reception could not be bridged, so its frame is unknown')

    contact_observed = serve_index > 0 and _hands_over(segments[serve_index - 1], serve)
    depth = serve_side * serve.evaluate(start_frame)[1]
    if contact_observed and depth < MIN_SERVE_CONTACT_DEPTH:
        return None, (f'the serve flight is handed over to {depth:.1f} m from the net,'
                      f' too near to be the server\'s contact (a split serve)')
    return ServeFlight(serve, start_frame, end_frame, contact_observed), None


def _locate_serve(segments):
    """Which flight the serve is, and where it starts and ends.

    The rule `find_serve_flight` documents: the serve is the last flight
    beginning more than `NET_ENTRY_MARGIN` back on the serving side before the
    reception. Shared with `find_serve_path` so the two cannot disagree on which
    flight the serve is.

    Returns `((serve_index, start_frame, reception_index, end_frame), None)` or
    `(None, reason)`.
    """
    found = _reception_contact(segments)
    if found is None:
        return None, 'no reception found on the parabolas'
    reception_index, _, end_frame = found
    serve_side = _serve_side(segments)

    serve_index, start_frame = None, None
    for index in range(reception_index):
        frame = (segments[index].start if index == 0
                 else _contact_frame(segments[index - 1], segments[index]))
        if serve_side * segments[index].evaluate(frame)[1] > NET_ENTRY_MARGIN:
            serve_index, start_frame = index, frame
    if serve_index is None:
        return None, 'no flight starts on the serving side before the reception'
    return (serve_index, start_frame, reception_index, end_frame), None


@dataclass(frozen=True)
class ServePath:
    """The serve from the server's contact to the reception, as a chain of flights.

    Usually one flight. More than one when the ball changed flight on the way -
    the net cord is the case that matters - and then `joins[i]` is the frame
    `flights[i]` hands over to `flights[i + 1]`, a seam like any other contact.
    `evaluate` walks the chain, so the path is always exactly one fitted model
    per frame and never a blend of two.

    `net_touch` is True when one of those hand-overs happened at the net, within
    `NET_ENTRY_MARGIN` of it. `contact_observed` is `ServeFlight`'s.
    """

    flights: Tuple[object, ...]   # ballistic.Segment, in order
    joins: Tuple[int, ...]        # len(flights) - 1 hand-over frames
    start_frame: int
    end_frame: int
    contact_observed: bool
    net_touch: bool

    def flight_at(self, frame):
        """The flight that describes the ball at `frame`."""
        for flight, join in zip(self.flights, self.joins):
            if frame <= join:
                return flight
        return self.flights[-1]

    def evaluate(self, frame):
        return self.flight_at(frame).evaluate(frame)


def find_serve_path(segments):
    """The serve as a chain of flights, or the reason there is none.

    `find_serve_flight`, except that flights between the serve and the reception
    are followed instead of refused, as long as each hands over to the next
    (`_hands_over`: the two models agree on where the ball is at the seam). That
    is what lets a serve that clipped the net cord be measured at all, and what
    makes "did it touch the net" answerable: the cord is a contact, so it shows
    up as a hand-over at `y ~ 0` - reception 10 of the quarter final splits at
    `y = -0.76` with the two arcs agreeing to 0.01 m. A hand-over anywhere else
    is the fit splitting one flight, which float serves, not being quite
    parabolas, can provoke; the chain is still the path the ball took.

    A touch that does not deflect the ball enough to need a second parabola
    cannot be seen this way.

    Refused, with the reason, in the same cases as `find_serve_flight` apart
    from the flights in between: no reception or serve flight, two neighbouring
    flights on the path that do not hand over, an unbridged gap before the
    reception, and an observed contact nearer the net than
    `MIN_SERVE_CONTACT_DEPTH`.

    Returns `(ServePath, None)` or `(None, reason)`.
    """
    located, reason = _locate_serve(segments)
    if located is None:
        return None, reason
    serve_index, start_frame, reception_index, end_frame = located
    serve_side = _serve_side(segments)

    flights = tuple(segments[serve_index:reception_index])
    joins = []
    for before, after in zip(flights, flights[1:]):
        if not _hands_over(before, after):
            return None, (f'the serve flights ending at frame {before.end} and'
                          f' starting at {after.start} do not hand over, so the'
                          f' path between them is unknown')
        joins.append(_contact_frame(before, after))

    last, reception = flights[-1], segments[reception_index]
    if reception.start - last.end > 1 and find_seam(last, reception) is None:
        return None, (f'frames {last.end + 1}-{reception.start - 1} before the'
                      f' reception could not be bridged, so its frame is unknown')

    contact_observed = serve_index > 0 and _hands_over(segments[serve_index - 1],
                                                       flights[0])
    depth = serve_side * flights[0].evaluate(start_frame)[1]
    if contact_observed and depth < MIN_SERVE_CONTACT_DEPTH:
        return None, (f'the serve flight is handed over to {depth:.1f} m from the net,'
                      f' too near to be the server\'s contact (a split serve)')

    net_touch = any(abs(after.evaluate(join)[1]) <= NET_ENTRY_MARGIN
                    for join, after in zip(joins, flights[1:]))
    return ServePath(flights, tuple(joins), start_frame, end_frame,
                     contact_observed, net_touch), None


# What the height of a pass whose top was out of view is allowed to be fitted
# with: a quadratic in z through both halves, accepted only if it fits them to
# `INLIER_TOLERANCE` and bends like a falling ball - the vertical gate
# `pipeline.RECEPTION_FIT` uses for every flight.
JOINED_APEX_ACCELERATION = (-25.0, -4.0)


@dataclass(frozen=True)
class Pass:
    """The reception's own flight, from the passer's contact to where it ended.

    `flights` is the pass as fitted: one flight, or two when it went above the
    top of a camera's picture and came back down into it (`_set_cut_off_above`,
    the same test the attack pipeline uses on high sets).

    `end_point` is where the pass ended - normally the setter's hands, but on an
    overpass wherever the ball was next played, on either side of the net. It is
    the last pass flight's own model at the contact that ended it, so it
    describes the reception trajectory and nothing after it.

    `apex` is the highest the pass went, the ball's centre in metres, and
    `apex_source` says what it rests on:

    - `'observed'` - the top of the pass flight's parabola, inside its own
      observed frames;
    - `'extrapolated'` - the top of the same parabola, but in frames it was
      extended into rather than observed (a bridged gap before the pass ended);
    - `'joined'` - the pass left the picture: one parabola in z fitted through
      both halves, and its top.

    `apex_point` is where that top is, for drawing it: the pass's model at
    `apex_frame`, or for a joined pass the joined height over the ground
    position of whichever half is nearer in frames to the top - one model's
    position, never an average of the two.

    `end_point` can be None, and `end_reason` then says why; `apex` can be None
    too, and a note says why. `notes` also records a pass that left the picture.
    """

    reception_point: np.ndarray
    reception_frame: int
    flights: Tuple[object, ...]
    end_point: Optional[np.ndarray] = None
    end_frame: Optional[int] = None
    apex: Optional[float] = None
    apex_frame: Optional[float] = None
    apex_source: Optional[str] = None
    apex_point: Optional[np.ndarray] = None
    end_reason: Optional[str] = None
    notes: Tuple[str, ...] = ()


def _highest_point(segment, first, last):
    """The top of `segment`'s model between frames `first` and `last`.

    Returns `(z, frame)`, or None when the ball is still rising at `last` - then
    the top was never reached inside the range and nothing says how high it went.
    """
    p0, v, a = segment.coefficients
    if a[2] >= 0:
        return None
    vertex = segment.start - v[2] / a[2]
    if vertex > last:
        return None
    frame = max(vertex, first)
    return float(segment.evaluate(frame)[2]), float(frame)


def _joined_apex(rising, falling):
    """The top of a pass seen only on its way up and on its way down.

    Neither half's own model can be trusted over the gap: each is short, and two
    short halves of attack 59's set disagree by 1.7 m in height a second out.
    Together they span the whole flight, so the curvature is well determined by
    one parabola in z through both. Horizontal motion is left out because the
    height is all that is asked.

    Returns `(z, frame)`, or None if one parabola does not fit both halves.
    """
    frames = np.array(rising.inlier_frames + falling.inlier_frames, dtype=float)
    heights = np.array([rising.evaluate(f)[2] for f in rising.inlier_frames]
                       + [falling.evaluate(f)[2] for f in falling.inlier_frames])
    curve = np.polyfit(frames, heights, 2)
    rms = float(np.sqrt(np.mean((np.polyval(curve, frames) - heights) ** 2)))
    acceleration = 2 * curve[0] * FPS * FPS
    low, high = JOINED_APEX_ACCELERATION
    if rms > INLIER_TOLERANCE or not low <= acceleration <= high:
        return None
    vertex = -curve[1] / (2 * curve[0])
    return float(np.polyval(curve, vertex)), float(vertex)


def find_pass(segments, cameras_near_top_edge=None):
    """The reception, how high the pass went, and where it ended.

    The reception is `find_reception`'s, and the pass is the flight that leaves
    it. The pass ends at the contact starting the flight after it, located the
    way every contact is (`_contact_frame`, a seam), and the end point is the
    **pass's** model there: the end of the reception trajectory, not the start
    of the next one. Where the two flights meet at a bridged seam they agree to
    within `MAX_BRIDGE_SEAM` anyway; what the choice settles is which flight
    the point belongs to. Which side of the net it is on is not tested, so an
    overpass ends wherever the ball was next played.

    Things that stand in the way, each measured on cached receptions:

    - **A high pass leaves the top of the picture.** It is then fitted as two
      flights with the apex missing between them, and the second is the pass
      coming back down, not what followed it - reception 56 is seen rising to
      frame 184, then again at 5.2 m from 226, and only played at 263.
      `_set_cut_off_above` pairs the halves, the end is on the falling half, and
      the height is `_joined_apex`.
    - **The flight after the pass came into view from above, already falling**,
      with no rising half it could be paired with (`_set_entered_from_above`).
      That is either an unpaired pass - which had not ended - or a later ball
      that went out of the picture. No end point.
    - **The clip ends before the pass does.**
    - **The pass and the next flight cannot be bridged** (`find_seam`): the
      ball ended somewhere in a gap neither model could be carried across -
      on reception 54, 108 frames.
    - **The pass is still rising where the next flight starts.** That is the fit
      splitting the pass in two, not the pass being played - reception 17 does
      it at 2.4 m on the way up.

    An end point outside the volume a ball can be in is an extrapolation gone
    wrong and is dropped too. Each refusal leaves a reason in `end_reason`.

    Returns a `Pass`, or None when there is no reception on the parabolas.
    """
    found = _reception_contact(segments)
    if found is None:
        return None
    index, reception_point, reception_frame = found
    notes = []

    flights = [segments[index]]
    after = index + 1
    if after < len(segments):
        cut_off = _set_cut_off_above(segments[index], segments[after],
                                     cameras_near_top_edge)
        if cut_off is not None:
            flights.append(segments[after])
            notes.append(f'the pass went above the top of the {cut_off} camera\'s'
                         f' picture between frames {segments[index].end} and'
                         f' {segments[after].start}')
            after += 1

    end_point = end_frame = end_reason = None
    if after >= len(segments):
        end_reason = 'the clip ends before the pass does'
    else:
        entered = _set_entered_from_above(segments, after, cameras_near_top_edge)
        if entered is not None:
            end_reason = (f'the flight after the pass came into the {entered}'
                          f' camera\'s picture from above, already falling, at'
                          f' frame {segments[after].start}, so the pass may not'
                          f' have ended there')
        else:
            pass_flight, following = segments[after - 1], segments[after]
            unbridged = (following.start - pass_flight.end > 1
                         and find_seam(pass_flight, following) is None)
            frame = _contact_frame(pass_flight, following)
            point = pass_flight.evaluate(frame)
            if unbridged:
                end_reason = (f'frames {pass_flight.end + 1}-{following.start - 1}'
                              f' after the pass could not be bridged, so where it'
                              f' ended is unknown')
            elif _vertical_velocity(pass_flight, frame) > 0:
                end_reason = (f'the pass is still rising where the next flight'
                              f' starts, at frame {frame}, so that flight is more'
                              f' of the pass rather than the next touch')
            elif not _is_on_court(point):
                end_reason = (f'the end point {np.array2string(point, precision=2)}'
                              f' is outside the volume a ball can be in')
            else:
                end_point, end_frame = point, frame

    # The pass ends where it was played, or where it was last seen.
    pass_end = end_frame if end_frame is not None else flights[-1].end
    if len(flights) == 2:
        top = _joined_apex(*flights)
        source = 'joined'
        if top is None:
            notes.append('one parabola does not fit both halves of the pass,'
                         ' so how high it went is unknown')
    else:
        pass_flight = flights[0]
        top = _highest_point(pass_flight, reception_frame, pass_end)
        source = (None if top is None else
                  'observed' if pass_flight.start <= top[1] <= pass_flight.end
                  else 'extrapolated')
        if top is None:
            notes.append(f'the pass is still rising at frame {pass_end}, the last'
                         f' it is known at, so how high it went is unknown')

    apex = apex_frame = apex_point = None
    if top is not None:
        apex, apex_frame = top
        if len(flights) == 2:
            rising, falling = flights
            nearer = (rising if apex_frame - rising.end <= falling.start - apex_frame
                      else falling)
            apex_point = np.append(nearer.evaluate(apex_frame)[:2], apex)
        else:
            apex_point = flights[0].evaluate(apex_frame)
    return Pass(reception_point, reception_frame, tuple(flights),
                end_point, end_frame, apex, apex_frame,
                source if top is not None else None, apex_point, end_reason,
                tuple(notes))


def _hands_over(before, segment):
    """Does `before`'s flight end where `segment`'s begins - one contact, seen from both sides?

    Across a gap that is `find_seam`'s test; with no gap, the two models have to
    agree at the first frame to within the same `MAX_BRIDGE_SEAM`. The agreement
    is what rules out an unrelated flight that merely comes first in the clip,
    such as the stationary object at the net some receptions pick up.
    """
    if segment.start - before.end > 1:
        return find_seam(before, segment) is not None
    miss = np.linalg.norm(before.evaluate(segment.start) - segment.evaluate(segment.start))
    return bool(miss <= MAX_BRIDGE_SEAM)


@dataclass(frozen=True)
class AttackPoints:
    """The three contacts an attack is made of, read off the fitted flights.

    Each point is a fitted parabola evaluated at the frame the contact happened,
    and nothing else - no blend of the two flights meeting there. Which model
    is evaluated differs per point:

    - `set_point` is the flight that *leaves* the setter's hands, the rule the
      reception uses, because that is the one that says where the ball went;
    - `attack_point` is where the **set** ended - see `find_attack_points` for
      why the set's model and not the attack's;
    - `end_point` is the end of the attack's own flight, so the attack's model.

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
    set_index: Optional[int] = None  # the flight the set was located on
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


def find_attack_points(segments, cameras_near_top_edge=None):
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
    - the **attack** is where the **set flight ended**: its model at the seam
      with the attack flight, or at its own last observed frame when the two
      were not bridged. Only when there is no set flight - the attack is the
      first flight in the clip - is it the attack flight's own start;
    - the **end** is where the attack flight finished, the seam with whatever
      follows it, or its own last frame when nothing does.

    Contacts are **seams**, not segment starts, via the shared `_contact_frame`:
    the contact happens inside the gap neither flight observed, and by the time
    observations resume the ball has moved.

    The attack is read off the end of the set rather than the start of the
    crossing flight because **the crossing flight does not always begin at the
    attacker's hand**. A block touch is a contact too: on attack 59 the spike is
    struck 2.1 m off the net, travels at ~18 m/s for five frames too few to be
    fitted as a flight of their own, and is deflected by the block at the net,
    and the flight that crosses starts there - at `y = -0.24`. The set's
    descent, on the other hand, is long and well observed and ends at the
    attacker's hand whatever happens after it. Over 42 cached attacks the change
    moves the point a median of 0.22 m (90th percentile 0.71 m); attack 59
    moves 1.97 m, back to 2.12 m off the net.

    When the attack is the first flight in the clip there is nothing before it
    to contact against, so no set is reported and its own start is the best that
    can be said about where it was struck. Both cases warn rather than guess.

    A **high set that went above the top of a camera's picture** is fitted as
    two flights with the apex missing between them. When `_set_cut_off_above`
    recognises that, the set is located at the start of the *rising* half - the
    flight before the set flight - rather than where the ball came back into
    view. When the set flight instead comes back into view falling and there is
    no rising half to use (`_set_entered_from_above`), there is no set location
    at all: `set_point` is None and a warning says why. Both need the camera
    calibration, through `cameras_near_top_edge` (default
    `reconstruction.cameras_near_top_edge`); the attack and end points do not.

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

    # The attack contact: where the set ended, or with no set flight to end,
    # where the attack flight began.
    if index == 0:
        warnings.append(
            f"the attack is the first flight in the clip, so its contact is the"
            f" first frame it was observed at ({attack_frame}) rather than the"
            f" end of a set")
        attack_point = attack_flight.evaluate(attack_frame)
    else:
        setting_flight = segments[index - 1]
        seam = find_seam(setting_flight, attack_flight)
        attack_frame = setting_flight.end if seam is None else seam[0]
        attack_point = setting_flight.evaluate(attack_frame)

    set_point = set_frame = set_index = None
    if index == 0:
        warnings.append(
            "no flight precedes the attack, so there is no set to locate")
    else:
        set_index = index - 1
        rising = segments[set_index - 1] if set_index >= 1 else None
        cut_off = (None if rising is None else
                   _set_cut_off_above(rising, segments[set_index],
                                      cameras_near_top_edge))
        if cut_off is not None:
            warnings.append(
                f"the set went above the top of the {cut_off} camera's picture"
                f" between frames {rising.end} and {segments[set_index].start},"
                f" so it is located at the start of its rising half")
            set_index -= 1
        else:
            entered = _set_entered_from_above(segments, set_index,
                                              cameras_near_top_edge)
            if entered is not None:
                warnings.append(
                    f"the set came back into the {entered} camera's picture from"
                    f" above, already falling, at frame"
                    f" {segments[set_index].start}, and its rising half was never"
                    f" fitted, so there is no set location")
                set_index = None

        if set_index is not None:
            setting = segments[set_index]
            if set_index >= 1:
                set_frame = _contact_frame(segments[set_index - 1], setting)
            else:
                set_frame = setting.start
                warnings.append(
                    f"nothing was fitted before the set, so the set location is"
                    f" the first frame the setting flight was observed at"
                    f" ({set_frame}) rather than the setter's contact")
            set_point = setting.evaluate(set_frame)

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
        attack_index=index, set_index=set_index, crossed_net=True,
        warnings=tuple(warnings),
    )
