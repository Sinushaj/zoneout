"""Figure generation for the ball-tracking pipeline.

Keeps plotting code out of the processing loop in `zoneout.pipeline`. Add new
figure types as further modules here and re-export them below.
"""

from .court import court_traces, draw_court, net_traces
from .serves import plot_player_serves, save_player_serves_html
from .trajectory import (
    plot_trajectory,
    plot_trajectory_plotly,
    save_trajectory_figure,
    save_trajectory_html,
)

__all__ = [
    "court_traces",
    "draw_court",
    "net_traces",
    "plot_player_serves",
    "save_player_serves_html",
    "plot_trajectory",
    "plot_trajectory_plotly",
    "save_trajectory_figure",
    "save_trajectory_html",
]
