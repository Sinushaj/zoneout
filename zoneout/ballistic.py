"""Segmented ballistic fitting: physics as the model for cleaning and gap filling.

This is an **optional correction step**, isolated here so it can be switched off
in one place (`pipeline.USE_BALLISTIC_FIT`) without disturbing anything else. If
it is off the pipeline behaves exactly as it did before this module existed.

The idea: between contacts the ball is in free flight, and free flight is exactly

    p(t) = p0 + v0 t + 1/2 a t^2

which is *linear* in the nine unknowns (p0, v0, a), so fitting it is one
least-squares solve. `a` is left free rather than pinned to gravity because
aerodynamic drag on a struck volleyball is not a small correction — at 25 m/s it
is comparable to gravity — so a fitted effective acceleration describes a real
serve far better than -9.81 alone. It is still checked against gravity
afterwards (see `PLAUSIBLE_VERTICAL_ACCELERATION`) as a sanity test that the
segment really is a ball in flight.

One model fits one flight. A contact — the serve, the reception, a touch on the
net — breaks it, so the trajectory is first cut into segments at the points where
a single parabola stops explaining the data (`find_segments`), and each segment
is fitted separately. This is the reason for doing any of this: a gap that spans
a change of direction cannot be filled by interpolating across it, but it can be
filled by two parabolas that each know where they end. `find_seam` and `_bridge`
do that last part, and it is where the real payoff is — on a real reception the
linear fill held the ball at a flat 2.1 m straight through the dig, while the two
fitted arcs bring it down to 1.4 m at the contact and back up.

Everything a bridge writes is a fitted parabola evaluated outside the frames it
observed — nothing else. No blend, no ramp, no correction is mixed in to make the
two sides meet. Where they do not meet on their own, the gap is left empty
instead: an honest hole is a better answer than an invented join, because a
filled frame is read downstream (by `events.find_serve_and_receive`, by the
figures) as if it were a measurement.

Careful with the data it is given, because the input trajectories are mostly good
already and a correction step that rewrites good data is worse than none:

* An accepted segment is guaranteed to sit within `INLIER_TOLERANCE` of the
  observations it kept, so the correction cannot silently move a point far.
* The only extrapolation beyond observed frames is into the gap *between* two
  fitted segments, so it is anchored at both ends and bounded in length
  (`MAX_BRIDGE_FRAMES`). Nothing is ever extrapolated off the open ends of the
  trajectory, where there would be no second anchor.
* A gap between two segments is filled only if the two models actually agree
  about where the ball is at the handover frame, to within `MAX_BRIDGE_SEAM`.
  They are never nudged into agreement; if they disagree, the whole gap stays
  empty.
* Observations no segment can explain are dropped rather than kept — see
  `DROP_UNEXPLAINED`, which records why that turned out to be the safer choice.
* `correct_trajectory` returns a `Report` of exactly what it changed, which the
  pipeline prints for every reception. If those numbers ever look large, this
  step is the first suspect.

Time is measured in *frames*, not seconds, so nothing here depends on the frame
rate. `FPS` is used only to express fitted accelerations in m/s^2 for the
plausibility check and for reporting.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


FPS = 60.0

# How far an observation may sit from its segment's model and still be counted
# part of that flight, in meters. This doubles as the guarantee on how much the
# correction can move any point it keeps.
INLIER_TOLERANCE = 0.30

# An accepted segment must explain at least this many observations. Three points
# determine the nine parameters exactly, so anything near that fits perfectly and
# proves nothing; a segment needs to be over-determined to be evidence.
MIN_SEGMENT_POINTS = 8

# --- Short flights, fitted with the acceleration pinned --------------------
#
# A served ball is in the air for a second or more and is seen in a hundred
# frames. A spike is not: it crosses the court in a quarter of a second at 25
# m/s, and after occlusion behind the block and the frames the detector loses,
# what survives triangulation can be a handful of points. Under
# `MIN_SEGMENT_POINTS` those points form no segment, and `DROP_UNEXPLAINED`
# then deletes them - so the very flight the attack pipeline exists to measure
# is the one most likely to vanish.
#
# Lowering `MIN_SEGMENT_POINTS` is the wrong fix, because the count is not
# really the problem: **the acceleration is**. Fitting p0, v0 and a costs nine
# parameters, and over a short span `a` is not measurable at all. Gravity moves
# the ball 0.02 m over 4 frames and 0.05 m over 6 - far below the 0.30 m the
# reconstruction is already allowed to be wrong by - so a free fit spends three
# parameters on curvature it cannot see, and pays for it in a wrong p0 and v0.
#
# So a short span is fitted with `a` **pinned** and only p0 and v0 solved for.
# That is six parameters instead of nine: three points now over-determine it
# and four give real evidence, and the position and velocity it reports are the
# ones actually supported by the observations. The pinned value is plain
# gravity, and the honesty of that is what `MAX_CONSTRAINED_SEGMENT_FRAMES`
# protects: across 20 frames the difference between a = -9.81 and a drag-heavy
# -15 is 0.29 m, still inside `INLIER_TOLERANCE`, while by 30 frames it is
# 0.65 m and the assumption would be doing real work. Beyond that cap there are
# normally enough points for a free fit anyway.
#
# The compensation for the weaker evidence is `MIN_SEGMENT_SPEED`. Four points
# that fit a line are not much of a claim, and a false detection both cameras
# agree on - a head, which triangulates cleanly - could make one. A ball in
# free flight during a rally is always moving, so requiring the fitted flight to
# actually go somewhere rejects the stationary impostor without adjudicating
# between plausible flights. It is a floor, not a discriminator: a set, a spike
# and a dig are all far above it.
GRAVITY = -9.81                        # m/s^2
MIN_CONSTRAINED_SEGMENT_POINTS = 4     # four points, six parameters
MAX_CONSTRAINED_SEGMENT_FRAMES = 20    # a third of a second; see above
MIN_SEGMENT_SPEED = 2.0                # m/s, only applied to pinned fits

# Set False to go back to free fits only, i.e. to the behaviour before short
# flights were handled at all. Everything the constrained path does is reported
# separately in the `Report`, so its effect is visible on every run.
USE_CONSTRAINED_SHORT_SEGMENTS = True

# While fitting one segment, at most this fraction of its points may be discarded
# as outliers before we conclude it is not a single flight and split it instead.
MAX_OUTLIER_FRACTION = 0.25

# Observations separated by more than this many frames are not assumed to belong
# to the same flight. Half a second is long enough for a contact to happen
# unseen, and bridging that on faith is how a fit invents a trajectory.
MAX_BRIDGED_GAP = 25

# A fitted segment's vertical acceleration must land in this range, in m/s^2, or
# the segment is rejected as not-a-ball-in-flight. Generously wide: gravity is
# -9.81, drag opposes motion so it pushes this either way depending on whether
# the ball is rising or falling, and the fit absorbs some reconstruction error
# too. The point is only to reject the wildly unphysical.
PLAUSIBLE_VERTICAL_ACCELERATION = (-30.0, 5.0)

# Whether to discard observations that no accepted segment explains.
#
# Passing them through untouched is the more cautious-looking option, but on
# real receptions it is the wrong one. Every large disagreement between this
# module and the old pixel-space cleanup traced back to such a frame: they are
# not ball positions the fit failed to cover, they are false detections that no
# physics can accommodate, and leaving them in creates spurious direction
# changes that mislead `events.find_serve_and_receive` about where the reception
# was. If a run of points is long enough to be real, it forms its own segment;
# if it cannot, it is not evidence of anything.
DROP_UNEXPLAINED = True

# Bridging the gap between two segments: the original motivating case. Neither
# adjacent segment observed those frames, so filling them means extrapolating
# both models into the gap and deciding where to hand over from one to the other.
#
# The handover frame is chosen on a physical fact: position is continuous through
# a contact even though velocity is not. So the seam is the frame where the two
# models most nearly agree on where the ball is. This is the sound version of
# "extend both ends until they meet" — two curves in 3D generically never
# intersect, but the frame minimising their separation always exists.
#
# The seam is *tested*, not *closed*. Each side of a bridged gap is its own
# fitted model evaluated there and nothing else, so if the two extrapolations do
# not already agree at the handover frame there is no honest fill to write, and
# the gap is left empty rather than papered over.
#
# An earlier version instead closed the seam, adding to each side a quadratic
# offset that grew to half the disagreement by the handover frame, so the two
# always met exactly. The arithmetic was sound — the correction has the shape of
# a free-flight model, so each side stayed a parabola — but it fabricates ball
# positions from a mismatch rather than from data, and every frame it writes is
# indistinguishable downstream from a measured one. Extending the real parabolas
# and stopping where they stop being credible is the behaviour that is wanted
# here.
#
# `MAX_BRIDGE_SEAM` is therefore a genuine gate, and it is what rejects a bridge
# in practice. It is set at the size of disagreement real receptions produce
# between two arcs that are plainly the same rally, measured after the search
# below has been given its best chance: over receptions 4-6 the honest gaps came
# out at 0.04-0.93 m, so anything under a metre is within the fits' own error and
# anything much over it is two arcs that are not continuing into each other.
#
# `MAX_BRIDGE_FRAMES` is only a ceiling on how far either model is trusted
# outside its own observations; length alone should not be what throws a gap
# away, since two arcs that still agree to within the seam tolerance after fifty
# frames are agreeing on more, not less.
MAX_BRIDGE_FRAMES = 90       # longest gap between two segments worth bridging
MAX_BRIDGE_SEAM = 1.0        # meters the two models may disagree at the seam

# How far past its own observations each model may be run *while searching* for
# the frame where the two agree best. This is not extra fill — no frame outside
# the gap is ever written — it only widens where the minimum is looked for.
#
# It matters because two arcs closing on each other are often still converging
# when they reach the edge of the gap, so the best frame inside the gap is not
# the best frame at all, and the separation measured there overstates how badly
# they miss. Reception 6's gap 97-117 is the case: 0.62 m at frame 117, the last
# frame of the gap, but 0.38 m at frame 119 just past it. Searching only inside
# the gap therefore rejected a pair of arcs that do meet. When the minimum does
# land outside, the handover clamps back to the gap boundary and one model fills
# the whole gap, which is the honest reading — the contact happened at or beyond
# where the next segment's observations start.
SEAM_SEARCH_MARGIN = 30


@dataclass(frozen=True)
class FitOptions:
    """How to cut a trajectory into flights, and what counts as one.

    The defaults are the behaviour the reception pipeline had until it moved to
    its own settings (`pipeline.RECEPTION_FIT`, with the old ones kept beside it
    as `pipeline.LEGACY_RECEPTION_FIT`), and `correct_trajectory` uses them when
    given nothing else, so a caller
    that passes no options is unaffected by anything tuned here. That matters:
    a serve and a spike are not the same fitting problem — a serve is slow, long
    and seen in a hundred frames, a spike is fast, short and seen in fifteen —
    and what makes one fit well is not what makes the other fit well. Tuning is
    therefore per-caller rather than global, and `zoneout.attacks` carries the
    attack set.

    `split` is how a span that is not one flight gets cut:

    - `'worst'` removes the single worst-fitting observation and recurses on
      what is left either side. Cheap, and right when the misfit is one bad
      point.
    - `'changepoint'` scans for the *time* at which one flight stops explaining
      the data and the next starts, and cuts there, keeping every observation.

    The difference is not academic. On a span holding a long slow flight and a
    short fast one, every one of the fast flight's points is a worse fit than
    every one of the slow flight's, so `'worst'` removes them one at a time,
    from the far end inwards, and never once offers the fast flight to the
    fitter as a span of its own — each peeled point becomes a fragment too small
    to fit and is dropped. Measured on attack 1: the recursion peeled frames
    145, 144, 142, 141, 140, 139, 138, 134 ... down to 127 in single steps, then
    fitted 0-49 and 53-126 and threw the spike away, even though a free fit of
    frames 126-145 on its own accepts 15 points at 20.8 m/s with an rms of
    0.037. The spike is the one flight an attack run exists to measure, so this
    is the failure that matters most.
    """

    split: str = 'worst'
    merge: bool = False
    plausible_vertical_acceleration: Tuple[float, float] = None
    max_cross_track_acceleration: Optional[float] = None
    short_segments: bool = None
    min_constrained_points: int = None
    max_constrained_frames: int = None
    min_speed: float = None

    def __post_init__(self):
        # Filled in from the module constants rather than defaulted inline, so
        # there is still one place those are written down.
        for field, default in (
                ('plausible_vertical_acceleration', PLAUSIBLE_VERTICAL_ACCELERATION),
                ('short_segments', USE_CONSTRAINED_SHORT_SEGMENTS),
                ('min_constrained_points', MIN_CONSTRAINED_SEGMENT_POINTS),
                ('max_constrained_frames', MAX_CONSTRAINED_SEGMENT_FRAMES),
                ('min_speed', MIN_SEGMENT_SPEED)):
            if getattr(self, field) is None:
                object.__setattr__(self, field, default)

    def minimum_points(self) -> int:
        """Fewest observations any accepted segment can be built from."""
        if self.short_segments:
            return min(MIN_SEGMENT_POINTS, self.min_constrained_points)
        return MIN_SEGMENT_POINTS


DEFAULT_OPTIONS = FitOptions()


@dataclass(frozen=True)
class Segment:
    """One free-flight arc: a fitted parabola plus the frames it explains.

    `constrained` says the acceleration was pinned to gravity rather than
    fitted, which is how a flight too short to measure curvature on is kept -
    see `MIN_CONSTRAINED_SEGMENT_POINTS`. It is carried rather than forgotten
    because it changes what the segment is evidence *of*: its position and
    velocity are measured, its curvature is assumed, so
    `vertical_acceleration` is an input for these and not the independent check
    on the calibration that it is for a freely fitted flight.
    """

    start: int                      # first observed frame, inclusive
    end: int                        # last observed frame, inclusive
    coefficients: np.ndarray        # (3, 3); rows are p0, v0, a; columns x, y, z
    inlier_frames: Tuple[int, ...]
    rms: float                      # meters
    constrained: bool = False       # acceleration pinned, not fitted

    def evaluate(self, frame: int) -> np.ndarray:
        """Where the model puts the ball at `frame`."""
        dt = float(frame - self.start)
        p0, v0, a = self.coefficients
        return p0 + v0 * dt + 0.5 * a * dt * dt

    @property
    def vertical_acceleration(self) -> float:
        """Fitted vertical acceleration in m/s^2 (gravity alone would be -9.81)."""
        return float(self.coefficients[2][2] * FPS * FPS)

    @property
    def speed(self) -> float:
        """Speed in m/s at the start of the segment."""
        return float(np.linalg.norm(self.coefficients[1]) * FPS)

    @property
    def cross_track_acceleration(self) -> float:
        """Sideways acceleration in m/s^2: how hard the fitted path bends.

        A ball in free flight does not bend. Drag is large — nearly 20 m/s^2 on
        a spike at 25 m/s — but it acts *against the direction of travel*, so it
        slows the ball without turning it. What is left across the direction of
        travel is only the Magnus force, which for a volleyball is small. So the
        horizontal acceleration is split into the part along the flight, which
        may legitimately be big, and the part across it, which may not.

        Measured over the 39 flights fitted across attacks 1-10: the median is
        0.4 m/s^2 and the 90th percentile 2.2, with one flight at 19.9 — the
        segment on attack 4 that is visibly impossible in the figure. Total
        horizontal acceleration cannot separate those, because a genuine spike
        shows 11.9 m/s^2 of it; only splitting off the along-track part can.
        """
        velocity = self.coefficients[1][:2]
        acceleration = self.coefficients[2][:2] * FPS * FPS
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-9:
            return 0.0
        unit = velocity / speed
        return float(abs(acceleration[0] * unit[1] - acceleration[1] * unit[0]))


@dataclass(frozen=True)
class Bridge:
    """One filled gap between two segments, and how it was put together.

    `before_frames` and `after_frames` say how much of the gap each side ended up
    covering. They are worth reading: when the two arcs are still converging at
    the edge of the gap rather than crossing inside it, the seam lands on the
    boundary and one side fills everything, which is the honest answer when the
    contact really did happen right where the next segment's observations start,
    and a sign the fill deserves a look when it did not.
    """

    seam: int                # frame the two models were handed over at
    separation: float        # meters they disagreed by where they agree best
    before_frames: int
    after_frames: int
    approach: int = 0        # frame where they agree best; == seam unless clamped

    def __str__(self) -> str:
        # Only worth saying where the best agreement was when it is not where the
        # handover happened, i.e. when it fell outside the gap and was clamped.
        outside = (f", closest at {self.approach}" if self.approach != self.seam else "")
        return (f"frame {self.seam} ({self.separation:.2f} m{outside}, "
                f"{self.before_frames}+{self.after_frames} frames)")


@dataclass(frozen=True)
class SkippedGap:
    """A gap between two segments that was left empty, and why.

    Worth reporting rather than dropping silently: this is now the only reason a
    stretch of trajectory goes missing between two accepted flights, so a hole in
    a plot should always be findable here. `separation` is how far apart the two
    models were at their closest approach — if that number is only just over
    `MAX_BRIDGE_SEAM` the fits are probably fine and the tolerance is the thing
    to look at; if it is metres, the two arcs are genuinely not the same ball.
    """

    start: int               # first empty frame
    frames: int              # how many frames were left empty
    separation: Optional[float]   # meters at closest approach, None if too long to check

    def __str__(self) -> str:
        if self.separation is None:
            return f"frames {self.start}-{self.start + self.frames - 1} (too long)"
        return (f"frames {self.start}-{self.start + self.frames - 1} "
                f"({self.separation:.2f} m apart)")


@dataclass
class Report:
    """What the correction actually did. Printed by the pipeline every run.

    The fitted `segments` are carried along rather than just counted, because
    where one flight ends and the next begins is where a contact happened, and
    that is worth more downstream than any smoothing this module does:
    `events.find_reception` reads the reception straight off them.
    """

    observations: int = 0        # frames that had a point going in
    segments: Tuple["Segment", ...] = ()  # the fitted flights, in frame order
    filled: int = 0              # frames with no observation that the model supplied
    adjusted: int = 0            # frames whose existing point the model moved
    rejected: int = 0            # observations a segment discarded as outliers
    unexplained: int = 0         # observations no segment covered
    bridged: int = 0             # frames filled in the gaps between segments
    seams: Tuple[Bridge, ...] = ()   # one per bridged gap; see Bridge
    skipped: Tuple[SkippedGap, ...] = ()  # one per gap left empty; see SkippedGap
    median_shift: float = 0.0    # meters, over adjusted frames
    max_shift: float = 0.0

    @property
    def constrained_segments(self) -> Tuple["Segment", ...]:
        """The flights too short to fit a curvature to, so gravity was assumed.

        Worth having separately: these are the ones whose acceleration is an
        input rather than a measurement, so they are excluded from the
        gravity-comes-out-right check that makes the fitted accelerations a
        test of the calibration.
        """
        return tuple(s for s in self.segments if s.constrained)

    def summary(self) -> str:
        if not self.segments:
            return (f"ballistic fit: no segments found in {self.observations} "
                    f"observations — trajectory passed through unchanged")
        fate = "dropped" if DROP_UNEXPLAINED else "kept"
        short = self.constrained_segments
        pinned = ("" if not short else
                  f" ({len(short)} of them short, fitted with gravity pinned:"
                  f" {', '.join(f'{s.start}-{s.end} on {len(s.inlier_frames)} points'
                                f' at {s.speed:.0f} m/s' for s in short)})")
        joins = ", ".join(str(bridge) for bridge in self.seams)
        bridge = (f", bridged {self.bridged} between segments at {joins}"
                  if self.bridged else "")
        misses = ", ".join(str(gap) for gap in self.skipped)
        skipped = f", left {misses} empty" if self.skipped else ""
        return (f"ballistic fit: {len(self.segments)} segment(s){pinned} over"
                f" {self.observations}"
                f" observations; filled {self.filled} missing frames{bridge}"
                f"{skipped}, adjusted"
                f" {self.adjusted} (median {self.median_shift:.2f} m, max"
                f" {self.max_shift:.2f} m), rejected {self.rejected} outliers,"
                f" {self.unexplained} unexplained ({fate})")


def _gravity_per_frame() -> np.ndarray:
    """The acceleration a pinned fit assumes, in meters per frame squared."""
    return np.array([0.0, 0.0, GRAVITY]) / (FPS * FPS)


def _fit(frames: np.ndarray, points: np.ndarray, origin: int,
         acceleration: Optional[np.ndarray] = None):
    """Least-squares fit of p(t) = p0 + v0 t + 1/2 a t^2, t in frames from `origin`.

    Returns the (3, 3) coefficient matrix and the per-point residual distances.
    Linear in the unknowns, so all three coordinates solve at once against one
    design matrix.

    With `acceleration` given, `a` is not a free parameter: its contribution is
    moved to the other side and only p0 and v0 are solved for. Six unknowns
    instead of nine, which is what makes a four-point flight worth fitting -
    see `MIN_CONSTRAINED_SEGMENT_POINTS`.
    """
    dt = (frames - origin).astype(float)

    if acceleration is None:
        design = np.column_stack([np.ones_like(dt), dt, 0.5 * dt * dt])
        coefficients, *_ = np.linalg.lstsq(design, points, rcond=None)
        residuals = np.linalg.norm(points - design @ coefficients, axis=1)
        return coefficients, residuals

    design = np.column_stack([np.ones_like(dt), dt])
    curvature = 0.5 * np.outer(dt * dt, acceleration)
    solved, *_ = np.linalg.lstsq(design, points - curvature, rcond=None)

    coefficients = np.vstack([solved, acceleration])
    residuals = np.linalg.norm(points - (design @ solved + curvature), axis=1)

    return coefficients, residuals


def _is_plausible_flight(segment: Segment, options: FitOptions) -> bool:
    """Is this fitted arc something a ball in free flight could actually do?

    Two tests, and they reject different things. The vertical one catches an arc
    that is not falling like a ball; the cross-track one catches an arc that
    bends sideways, which nothing in free flight does. A fit through mismatched
    points fails one or the other, and usually the second.

    A segment fitted with the acceleration pinned passes both by construction,
    so `MIN_SEGMENT_SPEED` is its test instead — see `_fit_robustly`.
    """
    low, high = options.plausible_vertical_acceleration
    if not (low <= segment.vertical_acceleration <= high):
        return False

    limit = options.max_cross_track_acceleration
    return limit is None or segment.cross_track_acceleration <= limit


def _fit_robustly(frames: np.ndarray, points: np.ndarray,
                  acceleration: Optional[np.ndarray] = None,
                  minimum_points: int = MIN_SEGMENT_POINTS,
                  options: FitOptions = DEFAULT_OPTIONS):
    """Fit one flight, discarding a few outliers, or give up.

    Repeatedly drops the single worst-fitting observation until everything left
    is within `INLIER_TOLERANCE`. Returns None if that would cost more than
    `MAX_OUTLIER_FRACTION` of the points, which is the signal that this span
    contains a contact and needs splitting rather than cleaning.

    With `acceleration` given the fit is the pinned one, and the checks change
    with it: the vertical acceleration is an input rather than a result, so
    testing it would prove nothing, and `MIN_SEGMENT_SPEED` takes its place as
    the evidence that this is a ball in flight and not a stationary object both
    cameras agreed on.
    """
    constrained = acceleration is not None
    keep = np.ones(len(frames), dtype=bool)
    floor = max(minimum_points, int(np.ceil((1 - MAX_OUTLIER_FRACTION) * len(frames))))

    while keep.sum() >= floor:
        origin = int(frames[keep][0])
        coefficients, residuals = _fit(frames[keep], points[keep], origin, acceleration)

        worst = int(np.argmax(residuals))
        if residuals[worst] <= INLIER_TOLERANCE:
            inliers = frames[keep]
            segment = Segment(
                start=int(inliers[0]),
                end=int(inliers[-1]),
                coefficients=coefficients,
                inlier_frames=tuple(int(f) for f in inliers),
                rms=float(np.sqrt(np.mean(residuals ** 2))),
                constrained=constrained,
            )
            if constrained:
                if segment.speed < options.min_speed:
                    return None
            elif not _is_plausible_flight(segment, options):
                return None
            return segment

        # Drop the worst point (index into the kept subset) and try again.
        kept_indices = np.flatnonzero(keep)
        keep[kept_indices[worst]] = False

    return None


def _changepoint_index(frames: np.ndarray, points: np.ndarray,
                       minimum_points: int) -> Optional[int]:
    """The moment one flight stops explaining the span and the next starts.

    Every cut is tried that leaves `minimum_points` observations on each side;
    each side is fitted, and the cut chosen is the one whose *worse* side fits
    best. Nothing is discarded — the cut is a time, and both halves keep all
    their observations, which is the whole difference from `_split_index`.

    Two choices here were measured rather than assumed, over the ten attacks in
    `attack_data/`:

    *Judge on the worse of the two sides, not their total.* A handful of points
    fits nine parameters almost exactly, so any criterion that adds the sides up
    is minimised by shaving a few points off one end — which is the peeling this
    exists to replace. The worse side cannot be made small by shaving.

    *Judge each side by its median residual, not its rms.* This is the
    difference between the fix working and half working. The scan fits without
    outlier rejection, so an rms cost is dominated by whichever few points are
    worst, and the cut it likes best is the one that isolates *those* rather
    than the one at the contact. Switching to the median took the observations
    no flight could explain from 90 to 38 across the ten attacks, and the
    observations covered by a flight from 1163 to 1217. (The unmodified
    worst-point split scores 1139 and 137.)
    """
    n = len(frames)
    if n < 2 * minimum_points:
        return None

    best = None
    best_cost = float('inf')
    for k in range(minimum_points, n - minimum_points + 1):
        _, left = _fit(frames[:k], points[:k], int(frames[0]))
        _, right = _fit(frames[k:], points[k:], int(frames[k]))
        cost = max(float(np.median(left)), float(np.median(right)))
        if cost < best_cost:
            best_cost, best = cost, k

    return best


def _short_segment(frames: np.ndarray, points: np.ndarray,
                   options: FitOptions = DEFAULT_OPTIONS) -> Optional[Segment]:
    """Try to explain a short span as one flight with the acceleration pinned.

    Returns None unless the span is genuinely short in *time* as well as in
    points. That is the condition under which pinning is nearly free: across
    `MAX_CONSTRAINED_SEGMENT_FRAMES` the assumed curvature differs from a
    drag-heavy one by less than the tolerance the fit already allows, so it is
    not deciding anything. Over a longer span it would be, and a longer span
    generally has the points for a free fit anyway.
    """
    if not options.short_segments:
        return None
    if len(frames) < options.min_constrained_points:
        return None
    if int(frames[-1] - frames[0]) + 1 > options.max_constrained_frames:
        return None

    return _fit_robustly(frames, points, _gravity_per_frame(),
                         minimum_points=options.min_constrained_points,
                         options=options)


def _merge_segments(segments: List[Segment],
                    points: Sequence[Optional[np.ndarray]],
                    options: FitOptions) -> List[Segment]:
    """Put back together adjacent segments that are really one flight.

    Splitting is greedy and top-down: it picks the best cut it can see at each
    level and never reconsiders, so a cut can land *inside* a genuine flight
    when the parent span's best cut was not at a contact. Nothing downstream
    then knows the difference — a spurious boundary reads as a contact that
    never happened, which for the attack pipeline is a fabricated event.

    Attack 5 is the case. Its attack runs from frame 127 to 167 and fits as one
    flight to an rms of 0.016 m, but the recursion cut it at 137 into two
    segments, one of which bends at 6.1 m/s^2 sideways — a fit that is only
    holding together because it is describing half an arc.

    The test for merging is the same one used to accept any segment, plus one
    condition: the merged fit has to explain **at least as many observations as
    the two separate fits did between them**. That is what stops a merge from
    quietly buying a tidy single parabola by discarding the points that
    disagreed with it — `_fit_robustly` is allowed to drop a quarter of its
    input as outliers, and without this condition a merge could use that budget
    to erase a real contact.
    """
    while len(segments) > 1:
        for i, (before, after) in enumerate(zip(segments, segments[1:])):
            observed = [(f, np.asarray(points[f], dtype=float))
                        for f in range(before.start, after.end + 1)
                        if f < len(points) and points[f] is not None]
            if len(observed) < options.minimum_points():
                continue

            frames = np.array([f for f, _ in observed])
            coords = np.array([p for _, p in observed])

            candidate = _fit_robustly(frames, coords, options=options)
            if candidate is None:
                candidate = _short_segment(frames, coords, options)

            explained = len(before.inlier_frames) + len(after.inlier_frames)
            if candidate is not None and len(candidate.inlier_frames) >= explained:
                segments = segments[:i] + [candidate] + segments[i + 2:]
                break
        else:
            break

    return segments


def _explains(segment: Segment, frames, points) -> bool:
    """Does `segment`'s model already describe these observations?"""
    errors = [float(np.linalg.norm(segment.evaluate(int(f)) - p))
              for f, p in zip(frames, points)]
    return bool(errors) and max(errors) <= INLIER_TOLERANCE


def _drop_continuations(segments: List[Segment],
                        points: Sequence[Optional[np.ndarray]]) -> List[Segment]:
    """Remove short segments that are really the neighbouring flight carrying on.

    A short pinned fit is cheap to satisfy, and a run of points left over at the
    edge of a long flight will satisfy it happily - they lie on a smooth arc,
    because they are *on that same arc*. Accepting them as a segment invents a
    contact where none happened, and `events` reads a contact off exactly that:
    the boundary between two segments.

    This is not hypothetical. It is what the constrained fit did to receptions 1
    and 10 before this check existed: it cut a new "flight" out of the serve's
    own descent, `find_reception` read the new boundary as the dig, and the
    reception moved 2.7-3.3 m up the court to a point the ball merely passed
    through. Reception 10 is the case `NET_ENTRY_MARGIN` was written for, and
    this put it back to reporting the net instead of the dig.

    The test is the one the module already uses for "this model describes this
    observation": if a neighbouring segment's model passes within
    `INLIER_TOLERANCE` of every one of these points, they are that flight's
    points and no contact separates them. Measured on the receptions above, the
    split is not marginal - a continuation sits 0.06-0.14 m from the neighbour's
    model, while a genuinely new flight sits 1.15-2.58 m away.

    Only *constrained* segments are tested. A freely fitted segment carries
    enough evidence to stand on its own, and rechecking those would change
    behaviour that is already validated.
    """
    kept: List[Segment] = []
    for i, segment in enumerate(segments):
        if not segment.constrained:
            kept.append(segment)
            continue

        frames = [f for f in segment.inlier_frames if points[f] is not None]
        observed = [np.asarray(points[f], dtype=float) for f in frames]
        neighbours = [segments[j] for j in (i - 1, i + 1)
                      if 0 <= j < len(segments)]

        if any(_explains(neighbour, frames, observed) for neighbour in neighbours):
            continue
        kept.append(segment)

    return kept


def _split_index(frames: np.ndarray, points: np.ndarray) -> int:
    """Where a span that isn't one flight most likely breaks: the worst residual."""
    _, residuals = _fit(frames, points, int(frames[0]))
    return int(np.argmax(residuals))


def _segment_span(frames: np.ndarray, points: np.ndarray,
                  options: FitOptions = DEFAULT_OPTIONS) -> List[Segment]:
    """Recursively cut a contiguous span into flights (split-and-merge)."""
    if len(frames) >= MIN_SEGMENT_POINTS:
        segment = _fit_robustly(frames, points, options=options)
        if segment is not None:
            return [segment]

    # No single free-flight parabola explains this span - either because it has
    # too few points to fit one, or because one was tried and failed. Before
    # splitting, see whether it is short enough in time to be one flight with
    # the curvature pinned. Trying this *before* the split matters: a spike is
    # over in a fifth of a second, and no rally fits two contacts into that, so
    # splitting a span that short would manufacture a contact that never
    # happened and hand the two halves to `find_seam` as if it had.
    short = _short_segment(frames, points, options)
    if short is not None:
        return [short]

    if len(frames) < MIN_SEGMENT_POINTS:
        return []

    if options.split == 'changepoint':
        # Cut at the moment the span stops being one flight, keeping every
        # observation on one side or the other. See `_changepoint_index`.
        k = _changepoint_index(frames, points, options.minimum_points())
        if k is not None:
            return (_segment_span(frames[:k], points[:k], options)
                    + _segment_span(frames[k:], points[k:], options))
        # Too short to cut in two. Fall through to dropping the worst point,
        # which can still salvage a fittable run out of what is left — giving up
        # here would throw the whole span away instead.

    # Not one flight: split at the worst-fitting observation and recurse. That
    # point is dropped rather than assigned to either side, since it is the one
    # most likely to be sitting in the middle of the contact.
    k = _split_index(frames, points)
    left = _segment_span(frames[:k], points[:k], options)
    right = _segment_span(frames[k + 1:], points[k + 1:], options)
    return left + right


def find_segments(points: Sequence[Optional[np.ndarray]],
                  options: FitOptions = DEFAULT_OPTIONS) -> List[Segment]:
    """Cut a trajectory into free-flight segments.

    Observations separated by more than `MAX_BRIDGED_GAP` frames are treated as
    belonging to different flights up front; each resulting block is then split
    further wherever one parabola cannot explain it.
    """
    minimum = options.minimum_points()
    observed = [(i, np.asarray(p, dtype=float))
                for i, p in enumerate(points) if p is not None]
    if len(observed) < minimum:
        return []

    segments: List[Segment] = []
    block_start = 0
    for i in range(1, len(observed) + 1):
        crosses_gap = (i < len(observed)
                       and observed[i][0] - observed[i - 1][0] > MAX_BRIDGED_GAP)
        if i == len(observed) or crosses_gap:
            block = observed[block_start:i]
            if len(block) >= minimum:
                frames = np.array([f for f, _ in block])
                coords = np.array([p for _, p in block])
                segments.extend(_segment_span(frames, coords, options))
            block_start = i

    # Both of these compare a segment against its neighbours, so they can only
    # run once the whole trajectory has been cut up. Merge first: it is undoing
    # cuts, and a merged segment is the one the continuation check should then
    # be judging against.
    if options.merge:
        segments = _merge_segments(segments, points, options)
    if options.short_segments:
        segments = _drop_continuations(segments, points)

    return segments


def closest_approach(before: Segment, after: Segment,
                     margin: int = SEAM_SEARCH_MARGIN) -> Optional[Tuple[int, float]]:
    """The frame at which two segments' models most nearly agree on the ball.

    Both models are run `margin` frames past their own observations — `before`
    forward beyond `after.start`, `after` backward beyond `before.end` — because
    the frame where they agree best is not always inside the gap. Two arcs
    converging on each other can still be closing when they reach the gap's edge,
    and measuring their separation there says they miss by more than they really
    do. Always giving the extension its chance is the point: the answer can only
    improve on searching the gap alone, never get worse, since the gap frames are
    still in the window.

    Returns `(frame, separation)`, or None if there is no gap between the two.
    The frame returned may lie outside the gap; `find_seam` is what decides where
    the fill actually hands over. No thresholds are applied here, so this is also
    what tells you *how badly* a rejected gap missed.

    The comparison is deliberately between the two models at the *same frame*,
    not between the two curves at whatever times bring them closest in space.
    A contact happens at one instant, and position is continuous through it, so
    same-frame agreement is the thing that means "these are the same ball". Two
    arcs that pass close in space at times fifteen frames apart are not
    continuing into each other; they only look like it in a plot, which has no
    time axis.
    """
    lo = before.end + 1 - margin
    hi = after.start - 1 + margin
    candidates = [(float(np.linalg.norm(before.evaluate(f) - after.evaluate(f))), f)
                  for f in range(lo, hi + 1)]
    if not candidates or after.start - before.end - 1 <= 0:
        return None

    separation, frame = min(candidates)
    return frame, separation


def find_seam(before: Segment, after: Segment) -> Optional[Tuple[int, float, int]]:
    """Find where to hand over from one segment to the next across a gap.

    Returns `(seam, separation, approach)`, or None if the gap is too long or the
    two arcs do not meet closely enough to be describing the same ball — in which
    case the gap is not filled at all, since the alternative is writing positions
    that come from the mismatch rather than from either flight.

    `approach` is the frame where the models agree best and `separation` is how
    well, both from `closest_approach`, which is allowed to look outside the gap.
    `seam` is that frame clamped back into the gap, because only gap frames are
    ever written: if the best agreement is past `after.start` then the contact
    happened at or beyond where the next segment's observations start, so
    `before` fills the whole gap and `after` is never extended into it (and the
    mirror image the other way). The `N+0` and `0+N` splits in the `Report` are
    exactly this case.
    """
    gap = after.start - before.end - 1
    if gap <= 0 or gap > MAX_BRIDGE_FRAMES:
        return None

    found = closest_approach(before, after)
    if found is None:
        return None

    approach, separation = found
    if separation > MAX_BRIDGE_SEAM:
        return None

    seam = min(max(approach, before.end), after.start - 1)
    return seam, separation, approach


def _bridge(before: Segment, after: Segment, seam: int):
    """Yield `(frame, point)` for every frame strictly between two segments.

    Each side of the seam is purely its own model extrapolated into the gap. No
    blend, ramp or offset is applied to bring the two together: a bridge is only
    reached at all once `find_seam` has established that they already agree at
    the handover frame to within `MAX_BRIDGE_SEAM`, so whatever disagreement is
    left shows up as a small step there and is reported as the seam separation.

    Keeping it this plain is the point. Every frame written here is a real fitted
    parabola evaluated a little past its own observations, which is a claim the
    physics supports; anything mixed in to make the ends meet would be a position
    derived from the mismatch rather than from the ball, and nothing downstream
    can tell the two apart afterwards.
    """
    for frame in range(before.end + 1, after.start):
        model = before if frame <= seam else after
        yield frame, model.evaluate(frame)


def correct_trajectory(points: Sequence[Optional[np.ndarray]],
                       drop_unexplained: bool = DROP_UNEXPLAINED,
                       options: FitOptions = DEFAULT_OPTIONS):
    """Clean and gap-fill a 3D trajectory using per-flight ballistic models.

    Returns `(corrected, report)`, where `corrected` is a new list of the same
    length. Frames inside an accepted segment are replaced by that segment's
    model; frames outside every segment are dropped or kept according to
    `drop_unexplained`.
    """
    corrected: List[Optional[np.ndarray]] = list(points)
    report = Report(observations=sum(1 for p in points if p is not None))

    segments = find_segments(points, options)
    report.segments = tuple(segments)

    covered = set()
    shifts = []

    for segment in segments:
        inliers = set(segment.inlier_frames)
        for frame in range(segment.start, min(segment.end, len(corrected) - 1) + 1):
            modelled = segment.evaluate(frame)
            previous = points[frame]

            if previous is None:
                report.filled += 1
            else:
                shift = float(np.linalg.norm(modelled - np.asarray(previous, dtype=float)))
                if frame in inliers:
                    report.adjusted += 1
                    shifts.append(shift)
                else:
                    report.rejected += 1

            corrected[frame] = modelled
            covered.add(frame)

    # Fill the gaps *between* segments, where neither side has observations —
    # including the ones spanning a contact, which is where linear filling has
    # nothing sensible to offer. Each half of the gap is covered by the model
    # that owns it, handing over at the seam frame, and a gap whose two models
    # do not meet there is left empty and recorded instead.
    seams = []
    skipped = []
    for before, after in zip(segments, segments[1:]):
        gap_start = before.end + 1
        gap_frames = after.start - gap_start
        found = find_seam(before, after)
        if found is None:
            if gap_frames > 0:
                closest = (closest_approach(before, after)
                           if gap_frames <= MAX_BRIDGE_FRAMES else None)
                skipped.append(SkippedGap(gap_start, gap_frames,
                                          closest[1] if closest else None))
            continue
        seam, separation, approach = found
        from_before = from_after = 0
        for frame, modelled in _bridge(before, after, seam):
            if frame >= len(corrected):
                break
            if points[frame] is None:
                report.bridged += 1
            if frame <= seam:
                from_before += 1
            else:
                from_after += 1
            corrected[frame] = modelled
            covered.add(frame)
        seams.append(Bridge(seam, separation, from_before, from_after, approach))
    report.seams = tuple(seams)
    report.skipped = tuple(skipped)

    unexplained = [i for i, p in enumerate(points) if p is not None and i not in covered]
    report.unexplained = len(unexplained)
    if drop_unexplained:
        for i in unexplained:
            corrected[i] = None

    if shifts:
        report.median_shift = float(np.median(shifts))
        report.max_shift = float(np.max(shifts))

    return corrected, report
