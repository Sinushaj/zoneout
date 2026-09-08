"""Turning a scout file's `video_time` into a frame number in each camera.

The one piece of machinery every pipeline needs before it can look at any video,
and the one that nothing downstream can check: a frame number is only meaningful
relative to the manual sync anchor, and an anchor off by a second is a second of
error on every action derived from it.

Kept here rather than in either pipeline so that receptions and attacks resolve
the anchor, the frame rates and the arithmetic by exactly the same rules. The
reading of the anchor CSV lives in `config`, the scout file's side of it in
`scout`, and each camera's true frame rate in `video`; this module is what joins
the three.
"""

from dataclasses import dataclass

from .config import read_sync_anchor
from .scout import (get_action_video_time, get_sync_video_time,
                    frame_for_video_time, video_time_frame_rates)
from .video import get_frame_rate


@dataclass(frozen=True)
class CameraTiming:
    """How to convert a `video_time` into a frame, for both cameras at once.

    `sideline_rate` and `baseline_rate` are frames per second of `video_time`,
    not the cameras' own frame rates - see `scout.video_time_frame_rates`. They
    differ slightly from each other, and the difference accumulates with the
    distance from the anchor, so they are kept apart rather than averaged.
    """

    anchor_frames: tuple        # (sideline, baseline) frame the anchor is at
    anchor_video_time: int
    sideline_rate: float
    baseline_rate: float
    anchor_set: str = None      # only for reporting; nothing computes with it
    anchor_skill: str = None

    def frames_for(self, video_time):
        """`(sideline_frame, baseline_frame)` for one second of `video_time`."""
        sideline_anchor, baseline_anchor = self.anchor_frames
        return (
            frame_for_video_time(sideline_anchor, self.anchor_video_time,
                                 video_time, self.sideline_rate),
            frame_for_video_time(baseline_anchor, self.anchor_video_time,
                                 video_time, self.baseline_rate),
        )

    def frames_for_action(self, dvw_filepath, skill, action_number):
        """`(sideline_frame, baseline_frame, video_time)` for one scouted action."""
        video_time = get_action_video_time(dvw_filepath, skill, action_number)
        sideline_frame, baseline_frame = self.frames_for(video_time)
        return sideline_frame, baseline_frame, video_time


def camera_timing(dvw_filepath, sideline_filepath, baseline_filepath):
    """Read the sync anchor and both cameras' frame rates into a `CameraTiming`.

    The anchor is whatever the sync step recorded - a set's first serve or first
    reception, its `video_time`, and the frame each camera shows it at. A config
    file written before the anchor was selectable has no `video_time` column;
    those always meant the match's first reception, so they are read as exactly
    that rather than demanding a re-sync of a setup that already works.
    """
    anchor = read_sync_anchor()

    anchor_video_time = anchor['video_time']
    if anchor_video_time is None:
        anchor_video_time = get_sync_video_time(dvw_filepath, skill='Reception')

    sideline_rate, baseline_rate = video_time_frame_rates(
        get_frame_rate(sideline_filepath), get_frame_rate(baseline_filepath))

    return CameraTiming(
        anchor_frames=(anchor['sideline'], anchor['baseline']),
        anchor_video_time=int(anchor_video_time),
        sideline_rate=sideline_rate,
        baseline_rate=baseline_rate,
        anchor_set=anchor['set'],
        anchor_skill=anchor['skill'],
    )
