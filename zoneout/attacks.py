"""The attack pipeline, kept separate from the reception one.

Nothing in `zoneout.pipeline` imports this and nothing here imports that, so
attacks can be switched off - or this file deleted outright - without receptions
noticing. What the two share is the machinery underneath: `scout` for reading
the scout file, `timing` for turning a `video_time` into a frame in each camera,
and later `detection`, `reconstruction` and `ballistic` for the reconstruction
itself, which is the same problem for both.

So far it does five steps. First, work out **which** attacks a run covers and
**where** in each video each of them is (`attack_timeline`) - that reads the
scout file and the sync CSVs, touches no video frames, and takes under a second
for a whole match, so the selection and the timing can be checked by eye before
committing hours of CPU to detection. Second, cut the clip each attack needs out
of both match videos (`extract_attack_clips`). Third, run the ball detector over
both clips (`detect_attack_ball`). Fourth, pair the two cameras' candidates and
triangulate them into 3D points (`match_attack_detections`), and draw those
points (`save_attack_figure`). Fifth, fit free-flight physics to them
(`fit_attack_trajectory`) and read the set, the attack and its end off the
fitted flights, collecting them into `attack_data/attack_points.csv`
(`attack_points_row`, `save_attack_points`; the reading itself is
`events.find_attack_points`).

Every one of those steps is the *same* machinery the reception pipeline uses -
the same detector, confidence floor and candidate limit from `zoneout.detection`,
the same matcher and geometric gate from `zoneout.reconstruction`, the same
figure code from `zoneout.figures`, and the same `zoneout.ballistic` fit - though
that one is *tuned* per caller through `ballistic.FitOptions`, since a serve and
a spike are not the same fitting problem. What is still missing is the write-back
to the `.dvw`, which needs the coordinate orientation settled first.
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from . import ballistic
from .detection import (DEFAULT_MODEL, DETECTION_CONF_THRESHOLD,
                        MAX_CANDIDATES_PER_FRAME, Detection, annotate_video,
                        process_video)
from .figures import save_trajectory_html
from .figures.style import ATTACK_CONTACT, ATTACK_END, SET_CONTACT
from .reconstruction import match_detections
from .scout import action_numbers, get_actions
from .timing import camera_timing
from .video import Clip, create_video_chunk

SKILL = 'Attack'

# The clip window around an attack, in seconds of the camera's own frames.
#
# Unlike a serve, an attack is not the start of anything: it is coded in the
# middle of a rally, so the clip has to be moved back over it rather than cut
# forward from it. How far back depends on what the scout's `video_time`
# actually marks, which is a scouting convention rather than something the file
# says - here it is the moment the **setter touches the ball**, entered by hand
# on the video. The attack contact then follows within about a second, so half a
# second of lead is enough to hold the set, the attack and the flight after it,
# while keeping the clip short: detection is the expensive step, and at 292
# attacks per match every second of clip is real time.
#
# The lead was reviewed against real clips and left alone; the duration went
# from 3.5 s to 4.0 for margin at the end, where a long rally continuing after
# the attack is cheaper to have than a flight cut off before it lands. Judge
# these on whether the clips contain the whole flight from the attacker's
# contact to the floor or the dig: if attacks are being cut off at the end,
# lengthen the duration, and if the set is missing from the front, lengthen the
# lead. They are separate constants because they fail separately.
ATTACK_CLIP_LEAD = 0.5
ATTACK_CLIP_DURATION = 4.0

# Fit free-flight physics to the triangulated points, per flight, as the
# reception pipeline does. Separate from `pipeline.USE_BALLISTIC_FIT` on
# purpose: the two pipelines switch independently.
#
# The fit matters more here than it does for a reception, and for the opposite
# reason. A serve is slow and long and is seen in a hundred frames, so the fit
# mostly tidies points that were already right. A spike is over in a fifth of a
# second, and between the block, the net and the frames the detector loses,
# what survives triangulation can be a handful of points - too few for the fit
# to have treated as a flight at all until `ballistic` learned to pin the
# acceleration on short spans. Without that, the spike was the part of the
# trajectory most likely to be thrown away.
USE_BALLISTIC_FIT = True

# Attack evaluations to skip. The scout's grade is the fourth character of the
# code, and pydatavolley puts it in `evaluation_code`:
#
#   #  kill        /  blocked
#   +  positive    !  recycled (played again off the block)
#   -  poor        =  error
#
# `/` and `!` are the two where the ball did not complete the attack it was
# scouted as: a blocked ball is stopped at the net and a recycled one comes back
# off the block to the attacking team. Either way the flight after the
# attacker's contact is not the attack arriving anywhere, so the third parabola
# is measuring a rebound, and the end coordinate would be a fiction.
#
# They are skipped rather than measured-and-flagged because there is nothing to
# measure: 48 of this match's 292 attacks carry one of these grades, and each
# costs ~50 s of detection to produce a coordinate that should not be written.
# The numbering is unaffected - attacks keep their match-wide numbers, so
# skipping attack 12 leaves 11 and 13 exactly where they were.
SKIP_EVALUATIONS = ('/', '!')

# How the fit is tuned for attacks. Passed to `ballistic.correct_trajectory`
# rather than set on the module, so the reception pipeline — which calls it with
# no options — is not affected by anything here. A serve and a spike are not the
# same fitting problem and there is no reason to expect one set of settings to
# suit both.
#
# `split='changepoint'` is the important one. The default 'worst' strategy cuts a
# span by deleting its single worst-fitting observation, which works when the
# misfit is one bad point and fails badly when it is a whole short flight: every
# point of a spike fits worse than every point of the set before it, so the
# recursion peels the spike away one frame at a time and never offers it to the
# fitter as a span of its own. On attack 1 it peeled frames 145, 144, 142, 141,
# 140 ... down to 127 and then dropped them all, even though a free fit of frames
# 126-145 accepts 15 points at 20.8 m/s with an rms of 0.037. 'changepoint'
# instead looks for the *time* the flight changes and cuts there, keeping every
# observation.
#
# The two acceleration limits reject arcs that are not free flight at all:
#
# - `max_cross_track_acceleration` is the "a ball cannot bend" rule. Drag is
#   large on a spike - nearly 20 m/s^2 at 25 m/s - but it acts against the
#   direction of travel, so it slows the ball without turning it. Measured over
#   the flights fitted across attacks 1-10, sideways acceleration has a median of
#   0.4 m/s^2 and a 90th percentile of 2.2, with a single flight at 19.9: the
#   segment on attack 4 that is visibly impossible in the figure. 8.0 sits in the
#   wide empty space between the two, so it rejects the impossible without
#   adjudicating between plausible flights.
# - `plausible_vertical_acceleration` is tightened from the module's very wide
#   default. Every genuine flight measured across those ten attacks falls between
#   -7.5 and -13.9 m/s^2; the one exception is +2.4, on a fit whose horizontal
#   deceleration was also twice what drag can produce. A ball in flight is always
#   falling, so the upper end has no business being positive.
ATTACK_FIT = ballistic.FitOptions(
    split='changepoint',
    merge=True,
    max_cross_track_acceleration=8.0,
    plausible_vertical_acceleration=(-25.0, -4.0),
)


@dataclass(frozen=True)
class Attack:
    """One scouted attack, located in both videos.

    `number` is 1-based and counts across the whole match, exactly as reception
    numbers do and for the same reason: it names the output directory and is
    what the frame lookup takes, so it has to mean one thing however a run was
    selected. It is counted per skill, though - attack 7 and reception 7 are
    unrelated rallies.

    `scout_line` is the attack's 0-based line number within the file's `[3SCOUT]`
    section, and is how the write-back will find the line to patch. `video_time`
    cannot do that job: within a rally the set, the attack, the block and the
    dig routinely share one second, and it is not even monotonic down the file.

    `start_zone`, `end_zone` and `end_subzone` are the human scout's own reading
    of where the attack came from and where it finished. Nothing computes with
    them - they are carried because they are free ground truth to check a
    reconstruction against, which is the only independent check this pipeline
    has.
    """

    number: int
    set_number: str
    video_time: int
    sideline_frame: int
    baseline_frame: int
    scout_line: int
    code: str
    team: str
    player_number: Optional[str] = None
    player_name: Optional[str] = None
    attack_code: Optional[str] = None
    evaluation_code: Optional[str] = None
    start_zone: Optional[str] = None
    end_zone: Optional[str] = None
    end_subzone: Optional[str] = None

    def __str__(self):
        zones = f"{self.start_zone or '?'}->{self.end_zone or '?'}"
        return (f"attack {self.number} (set {self.set_number},"
                f" video_time {self.video_time}): {self.code}"
                f" {self.team} zone {zones}, sideline frame"
                f" {self.sideline_frame}, baseline frame {self.baseline_frame}")


def selected_attacks(dvw_filepath, set_number=None, first=None, last=None,
                     skip_evaluations=SKIP_EVALUATIONS):
    """The attack numbers a run should process, in order.

    `set_number` picks one set (None for the whole match) and `first`/`last`
    narrow whatever that selected to an inclusive range of match-wide numbers.
    The range is applied on top of the set, so leaving it at None is what makes
    "all of set 2" mean all of it, and the numbers stay match-wide when a set is
    selected: set 3 might be attacks 147-210, not 1-64.

    Attacks graded in `skip_evaluations` are dropped last, and dropping them
    changes no other attack's number - the numbering comes from
    `scout.action_numbers` over *every* attack in the file, so a filtered run is
    a subset of the same sequence rather than a renumbering of it. That is what
    keeps `attack_data/attack12/` meaning the same rally whatever a run selected.
    """
    numbers = action_numbers(dvw_filepath, SKILL, set_number)
    if first is not None:
        numbers = [n for n in numbers if n >= first]
    if last is not None:
        numbers = [n for n in numbers if n <= last]

    if skip_evaluations:
        grades = list(get_actions(dvw_filepath, SKILL)['evaluation_code'])
        numbers = [n for n in numbers
                   if _text(grades[n - 1]) not in skip_evaluations]
    return numbers


def _text(value):
    """A scout-file field as a stripped string, or None when it is empty."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in ('nan', 'none', '<na>'):
        return None
    return text


def attack_timeline(dvw_filepath, sideline_filepath, baseline_filepath,
                    set_number=None, first=None, last=None,
                    skip_evaluations=SKIP_EVALUATIONS):
    """Every selected attack, with the frame each camera shows it at.

    The frames come from the sync anchor already in `first_reception_frames.csv`
    - the set's first serve or first reception, its `video_time`, and the frame
    each camera was stopped on - plus each camera's own frame rate read from its
    file. Each attack is that anchor plus its elapsed `video_time`, converted at
    that camera's rate, so nothing here needs the videos to be aligned with each
    other or with the scout file beyond that one pair of numbers.

    The frame returned is the one the attack's `video_time` maps to, not the
    start of a clip. `video_time` is only accurate to the second and the scout
    enters an attack *after* it happens, so the clip around it will need to
    begin before this frame rather than at it - but how far before is a question
    about the detection window, and belongs to the step that cuts the clip.

    Returns a list of `Attack`, in play order.
    """
    numbers = selected_attacks(dvw_filepath, set_number, first, last,
                               skip_evaluations)
    if not numbers:
        return []

    attacks = get_actions(dvw_filepath, SKILL)
    timing = camera_timing(dvw_filepath, sideline_filepath, baseline_filepath)

    timeline = []
    for number in numbers:
        row = attacks.iloc[number - 1]

        video_time = _text(row['video_time'])
        if video_time is None:
            # Everything downstream is an offset from this number, so a blank
            # one would quietly cut the clip at the anchor and reconstruct a
            # different rally entirely. Skipping is the honest answer, and the
            # caller reports it.
            continue

        video_time = int(video_time)
        sideline_frame, baseline_frame = timing.frames_for(video_time)

        timeline.append(Attack(
            number=number,
            set_number=_text(row['set_number']),
            video_time=video_time,
            sideline_frame=sideline_frame,
            baseline_frame=baseline_frame,
            scout_line=int(row['scout_line']),
            code=_text(row['code']),
            team=_text(row['team']),
            player_number=_text(row['player_number']),
            player_name=_text(row['player_name']),
            attack_code=_text(row['attack_code']),
            evaluation_code=_text(row['evaluation_code']),
            start_zone=_text(row['start_zone']),
            end_zone=_text(row['end_zone']),
            end_subzone=_text(row['end_subzone']),
        ))

    return timeline


def extract_attack_clips(attack, sideline_filepath, baseline_filepath,
                         lead_seconds=ATTACK_CLIP_LEAD,
                         duration_seconds=ATTACK_CLIP_DURATION):
    """Cut one attack's clip out of both match videos.

    Both clips cover the same window in seconds — `lead_seconds` before the
    attack's scouted moment, running for `duration_seconds` — but each is cut in
    its own camera's frames, since the two do not record at quite the same rate.
    That is what lets the frames of the two clips be paired one for one
    downstream.

    The clips are named after the attack rather than overwritten, so a run's
    whole selection survives to be watched. They live in `temporary_videos/`,
    which is gitignored scratch: everything here is regenerable by re-running,
    and nothing downstream should treat a clip as input it cannot rebuild.

    Returns `(sideline_clip, baseline_clip)` as `video.Clip`, which carry the
    frame each clip starts at in its source video — the only way to turn a frame
    inside a clip back into one the scout file can be asked about.
    """
    sideline = create_video_chunk(
        sideline_filepath, f'attack{attack.number}_sideline.mp4',
        attack.sideline_frame, lead_seconds, duration_seconds)
    baseline = create_video_chunk(
        baseline_filepath, f'attack{attack.number}_baseline.mp4',
        attack.baseline_frame, lead_seconds, duration_seconds)

    return sideline, baseline


def attack_output_dir(attack_number, root='attack_data'):
    """The directory one attack's output goes in, created if it does not exist."""
    path = os.path.join(root, f'attack{attack_number}')
    os.makedirs(path, exist_ok=True)
    return path


@dataclass(frozen=True)
class CameraDetections:
    """What the detector found in one camera's clip of one attack.

    Held per camera rather than merged because the two are independent evidence:
    a frame the ball was seen in from the sideline and not from the baseline
    cannot be triangulated, and knowing which camera lost it is what says whether
    the problem is occlusion behind the block or the detector generally.
    """

    name: str
    clip: Clip
    candidates: Sequence[Sequence[Detection]]

    @property
    def frames(self) -> int:
        return len(self.candidates)

    @property
    def frames_with_a_candidate(self) -> int:
        return sum(1 for frame in self.candidates if frame)

    @property
    def top_confidences(self) -> List[float]:
        """The best confidence in each frame that had any candidate at all."""
        return [frame[0].confidence for frame in self.candidates if frame]

    @property
    def longest_blind_run(self) -> int:
        """The most consecutive frames with nothing detected.

        The number that matters more than the total: the ballistic fit bridges
        a gap between two flights it can both see, so scattered misses cost
        little, while one long blind stretch over the contact is what leaves an
        attack with nothing to read.
        """
        longest = run = 0
        for frame in self.candidates:
            run = 0 if frame else run + 1
            longest = max(longest, run)
        return longest

    def summary(self) -> str:
        seen = self.frames_with_a_candidate
        if not seen:
            return f"{self.name}: nothing detected in {self.frames} frames"
        confidences = self.top_confidences
        return (f"{self.name}: {seen}/{self.frames} frames with a candidate"
                f" (median top confidence {np.median(confidences):.2f},"
                f" longest blind run {self.longest_blind_run})")


@dataclass(frozen=True)
class AttackDetections:
    """Both cameras' detections for one attack, and where they came from."""

    attack: 'Attack'
    sideline: CameraDetections
    baseline: CameraDetections

    @property
    def frames_seen_by_both(self) -> int:
        """Frames where both cameras have at least one candidate.

        The ceiling on how much of the flight can be triangulated at all: a
        frame only one camera saw is a frame the matcher has to leave empty.
        """
        return sum(1 for s, b in zip(self.sideline.candidates,
                                     self.baseline.candidates) if s and b)

    def summary(self) -> str:
        frames = min(self.sideline.frames, self.baseline.frames)
        return (f"    {self.sideline.summary()}\n"
                f"    {self.baseline.summary()}\n"
                f"    both cameras: {self.frames_seen_by_both}/{frames} frames")


def detect_attack_ball(attack, sideline_clip, baseline_clip,
                       model_path=DEFAULT_MODEL,
                       conf_threshold=DETECTION_CONF_THRESHOLD,
                       top_k=MAX_CANDIDATES_PER_FRAME):
    """Run the ball detector over both of one attack's clips.

    Independently per camera, as it is for receptions: the detector sees one
    view at a time, and deciding which of its candidates is the ball needs both
    views at once, which is a later step. Every candidate above the floor is
    kept, not just the most confident, because a spurious box is often the most
    confident one for a few frames and picking top-1 here would make that
    unrecoverable.
    """
    return AttackDetections(
        attack=attack,
        sideline=CameraDetections(
            name='sideline',
            clip=sideline_clip,
            candidates=process_video(sideline_clip.path, model_path,
                                     conf_threshold, top_k=top_k),
        ),
        baseline=CameraDetections(
            name='baseline',
            clip=baseline_clip,
            candidates=process_video(baseline_clip.path, model_path,
                                     conf_threshold, top_k=top_k),
        ),
    )


def match_attack_detections(detections):
    """Pair the two cameras' candidates and triangulate, frame by frame.

    Straight to `reconstruction.match_detections`, which is the whole point:
    every pairing of one sideline candidate with one baseline candidate is
    triangulated, has to reproject close to both detections and land somewhere a
    volleyball could be, and the survivor with the best confidence-weighted
    geometric fit wins. A frame where no pairing survives stays a gap rather
    than being filled with a guess.

    Nothing about that is reception-specific, and nothing here should make it
    attack-specific: the gate is deliberately loose and confidence is the
    primary signal, and both of those were measured on real footage. If attacks
    match badly the thing to suspect is the calibration or the detection, not
    these thresholds.

    Returns a list of `reconstruction.Match` or None, one per frame, truncated
    to the shorter of the two clips.
    """
    return match_detections(detections.sideline.candidates,
                            detections.baseline.candidates)


def matched_points(matches):
    """The triangulated 3D point per frame, with None where nothing survived.

    Kept as a list with the gaps in it, rather than compacted, so a point's
    position in it is still its frame number in the clip.
    """
    return [m.point if m else None for m in matches]


def annotate_attack_clips(detections, output_dir, matches=None):
    """Write both clips back with every candidate drawn on them.

    With `matches`, the pairing each frame settled on is drawn as the chosen box
    and the rest as rejected candidates, captioned with how far the triangulated
    point reprojected and how far apart the two rays were. Without it nothing is
    marked chosen. Either way the *rejected* candidates are drawn, which is the
    point of the whole exercise: it is how you tell "the detector never saw the
    ball" from "it saw it and the gate threw it away".

    Each frame is captioned with its number in the clip and, in brackets, the
    frame it came from in the match video — the clip number is what every other
    part of the pipeline counts in, and the source number is what finds the
    moment in the full recording.

    Returns the two written paths.
    """
    written = []
    for camera, index_of in ((detections.sideline, lambda m: m.gopro_index),
                             (detections.baseline, lambda m: m.zve10_index)):
        path = os.path.join(output_dir, f'{camera.name}_boxes.mp4')

        chosen = [None] * camera.frames
        labels = []
        for i, frame in enumerate(camera.candidates):
            match = matches[i] if matches and i < len(matches) else None
            if match is not None:
                chosen[i] = index_of(match)
            caption = f"{i} (src {camera.clip.source_frame(i)})"
            if match is not None:
                caption += (f"  {match.reprojection_error:.1f} px /"
                            f" {match.residual:.2f} m")
            elif frame:
                caption += f"  {frame[0].confidence:.2f} unmatched"
            labels.append(caption)

        annotate_video(camera.clip.path, path, camera.candidates, chosen, labels)
        written.append(path)
    return written


def fit_attack_trajectory(points):
    """Fit free-flight physics to one attack's triangulated points.

    Straight to `ballistic.correct_trajectory`, which cuts the points into
    flights, replaces each flight's frames with its fitted model, and bridges
    the gaps between flights by extending both models to where they agree. The
    spike is the flight this exists for: it is the shortest one in the clip and
    the one occlusion damages most, and it is only kept at all because
    `ballistic` will fit a short span with the acceleration pinned rather than
    demanding enough points to measure curvature from.

    The tuning is `ATTACK_FIT` and is passed in rather than set globally, so
    nothing here reaches the reception pipeline.

    Returns `(fitted, report)`. The report is worth printing on every run — its
    segment list is where the contact and the landing will be read from once
    `events` learns about attacks, and its numbers are the first thing to
    suspect when an attack comes out wrong.
    """
    return ballistic.correct_trajectory(points, options=ATTACK_FIT)


def _only_frames(points, first, last):
    """The same list with everything outside `first`..`last` blanked out."""
    return [p if first <= i <= last else None for i, p in enumerate(points)]


def save_attack_figure(attack, raw_points, output_dir, fitted_points=None,
                       segments=(), contacts=None):
    """Write the interactive 3D page for one attack.

    Only the part of the clip the attack is made of is drawn: from the start of
    the first fitted flight to where the attack finished. A clip is four seconds
    of a rally and usually holds more than that - the dig that follows, the next
    team's set - and those flights are not what the figure is for. Both series
    are trimmed the same way, the fitted path and the raw points alike, since
    leaving the measurements in would draw the discarded flights back onto the
    plot as ochre markers.

    The three contacts are marked with labels and outline rings. `style.py`
    explains why the labels are not optional: two of the data colors sit under
    3:1 against the orange court, so identity can never rest on hue alone.

    The raw triangulated points stay `raw_points=`, drawn as **markers with no
    connecting line** - they are whatever survived matching, gaps and all, and
    joining them would draw straight segments across the frames no pairing
    survived. The fitted flights are the joined path on top, so a bad fit and
    bad input can be told apart: scattered ochre means the problem is upstream
    in detection or matching, while tight ochre under a wandering blue path
    means it is the fit.

    Only the HTML is written, not the PNG the receptions also get. A scatter of
    points in a fixed 3D projection is very hard to read; being able to rotate
    it is what makes it worth looking at.
    """
    markers = None
    if contacts is not None:
        markers = [
            ("Set", contacts.set_point, SET_CONTACT),
            ("Attack", contacts.attack_point, ATTACK_CONTACT),
            ("End", contacts.end_point, ATTACK_END),
        ]
        if segments:
            # The three flights an attack is made of: the defence or reception
            # that preceded the set, the set, and the attack. Everything before
            # those belongs to the previous rally and everything after the
            # attack's end is the defence of it.
            first = segments[max(0, contacts.attack_index - 2)].start
            last = contacts.end_frame
            raw_points = _only_frames(raw_points, first, last)
            if fitted_points is not None:
                fitted_points = _only_frames(fitted_points, first, last)

    return save_trajectory_html(
        os.path.join(output_dir, 'trajectory.html'),
        points=fitted_points,
        raw_points=raw_points,
        title=(f"Attack {attack.number} — set {attack.set_number},"
               f" {attack.team}, {attack.code}"),
        markers=markers,
    )


# Where the measured coordinates are collected, one row per attack.
ATTACK_POINTS_CSV = os.path.join('attack_data', 'attack_points.csv')

POINT_NAMES = ('set', 'attack', 'end')

CSV_COLUMNS = (
    ['attack', 'set_number', 'video_time', 'scout_line', 'code', 'team',
     'player_number', 'attack_code', 'evaluation_code',
     'scouted_start_zone', 'scouted_end_zone', 'scouted_end_subzone']
    + [f'{name}_{field}' for name in POINT_NAMES
       for field in ('frame', 'x', 'y', 'z')]
    + ['crossed_net', 'warnings']
)


def attack_points_row(attack, points):
    """One CSV row: which attack it was, and the three coordinates measured.

    The scouted zones ride along beside the measurements deliberately. They are
    the human scout's own reading of where the attack came from and landed, so
    having them in the same row is what makes the file checkable without going
    back to the `.dvw` - and settling the coordinate orientation against them is
    the step that has to happen before any of this is written back.
    """
    row = {
        'attack': attack.number,
        'set_number': attack.set_number,
        'video_time': attack.video_time,
        'scout_line': attack.scout_line,
        'code': attack.code,
        'team': attack.team,
        'player_number': attack.player_number,
        'attack_code': attack.attack_code,
        'evaluation_code': attack.evaluation_code,
        'scouted_start_zone': attack.start_zone,
        'scouted_end_zone': attack.end_zone,
        'scouted_end_subzone': attack.end_subzone,
        'crossed_net': int(points.crossed_net),
        'warnings': ' | '.join(points.warnings),
    }
    for name, point, frame in (
            ('set', points.set_point, points.set_frame),
            ('attack', points.attack_point, points.attack_frame),
            ('end', points.end_point, points.end_frame)):
        # A missing point leaves its columns empty rather than writing a zero,
        # which would read as a measurement of the corner of the court.
        row[f'{name}_frame'] = '' if frame is None else frame
        for axis, value in zip('xyz', point if point is not None else (None,) * 3):
            row[f'{name}_{axis}'] = '' if value is None else round(float(value), 3)
    return row


def save_attack_points(rows, csv_path=ATTACK_POINTS_CSV):
    """Merge measured attacks into the coordinates file, keyed by attack number.

    Merged rather than overwritten because a run covers a range: processing
    attacks 1-10 and later 11-20 should leave twenty rows, not ten, and re-running
    an attack after a fix should replace its row rather than duplicate it. That
    matters for the same reason the reception pipeline writes to the `.dvw` as it
    goes - a batch that takes hours is never all-or-nothing, and what already
    worked should survive the next run.

    Existing rows whose attack number is not in `rows` are left exactly as they
    were. Returns the path written.
    """
    by_number = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as existing:
            for old in csv.DictReader(existing):
                by_number[int(old['attack'])] = old

    for row in rows:
        by_number[int(row['attack'])] = row

    os.makedirs(os.path.dirname(csv_path) or '.', exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for number in sorted(by_number):
            writer.writerow(by_number[number])

    return csv_path
