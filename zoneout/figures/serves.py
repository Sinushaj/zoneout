"""One player's serves, from the serve dataset, as a rotatable 3D figure.

Reads `datasets/serves.csv` (see `zoneout/serve_dataset.py`), picks one player's
rows and draws each serve as its fitted parabola - evaluated from the stored
start, velocity and acceleration, not joined up from points - over the court
and the net, with a cone near the end of each path for its direction.

Paths are colored by the serve's average speed on a **fixed scale**,
`SPEED_RANGE`, not the player's or the dataset's own range, so a color means
the same speed in every figure, from any dataset, and figures can be compared
side by side. The
contact and reception points are small, unlabelled, in neutral ink and told
apart by shape; the legend names them, and hovering any path or point gives
the serve's details.

Every serve in the dataset is already oriented with the server at the `y < 0`
end, so all of a player's serves arrive from the same end here without any
flipping.
"""

import os

import numpy as np
import pandas as pd

from .court import court_traces, net_traces
from .style import MARKER_OUTLINE, SERVE_POINT, SPEED_SCALE

# Samples per path. A serve is a single parabola, so this is only about the
# line looking smooth.
PATH_SAMPLES = 40

# The direction arrow sits this far before the end of the path, so the
# reception dot at the very end stays visible rather than hidden under it.
ARROW_LEAD_S = 0.08
ARROW_LENGTH = 0.55     # metres

# The speed color scale's bounds, in m/s of average speed over the flight: a
# really slow float serve at the bottom and a really hard spin serve at the top.
# Measured over 100 serves of the quarter-final replay, floats average 12-17 m/s
# and topspin jump serves (vertical acceleration -15 to -19 m/s^2, the spin
# pulling them down) 19.5-26.4, and the slowest serve in any dataset so far is
# 11.6. 10 sits under all of that and 28 over it, where elite jump serves are.
# A serve outside the range takes the end color.
SPEED_RANGE = (10.0, 28.0)


def _read(dataset):
    """The dataset as a DataFrame, from a DataFrame or a path to the CSV."""
    if isinstance(dataset, pd.DataFrame):
        return dataset
    return pd.read_csv(dataset)


def player_serves(dataset, player_number, team=None):
    """The rows of one player's serves.

    A player is a shirt number within a team. `team` may be left out when only
    one team in the dataset has that number; otherwise this raises, naming the
    candidates, rather than quietly merging two players.
    """
    serves = _read(dataset)
    rows = serves[serves['player_number'].astype(str) == str(player_number)]
    if team is not None:
        rows = rows[rows['team'] == team]

    teams = sorted(rows['team'].unique())
    if not teams:
        known = sorted({f"{t} #{n}" for t, n in zip(serves['team'], serves['player_number'])})
        raise ValueError(f"No serves by #{player_number}"
                         f"{'' if team is None else f' of {team}'} in the dataset."
                         f" Players in it: {', '.join(known)}")
    if len(teams) > 1:
        raise ValueError(f"#{player_number} serves for more than one team"
                         f" ({', '.join(teams)}); pass team= to choose.")
    return rows


def _flight(row, times):
    """Points of one serve's parabola at `times` (seconds from contact)."""
    start = row[['start_x', 'start_y', 'start_z']].to_numpy(float)
    v0 = row[['v0_x', 'v0_y', 'v0_z']].to_numpy(float)
    a = row[['a_x', 'a_y', 'a_z']].to_numpy(float)
    t = np.asarray(times, float)[:, None]
    return start + v0 * t + 0.5 * a * t * t, v0 + a * t


def _speed_color(speed, lo, hi):
    from plotly.colors import sample_colorscale
    fraction = 0.5 if hi <= lo else float(np.clip((speed - lo) / (hi - lo), 0, 1))
    return sample_colorscale(SPEED_SCALE, fraction)[0]


def plot_player_serves(dataset, player_number, team=None):
    """Build the plotly figure of one player's serves. Returns the Figure.

    `dataset` is the serve dataset, as a DataFrame or a path to the CSV. See
    `player_serves` for how the player is chosen.
    """
    import plotly.graph_objects as go

    rows = player_serves(_read(dataset), player_number, team)
    lo, hi = SPEED_RANGE

    fig = go.Figure()
    for trace in court_traces() + net_traces():
        fig.add_trace(trace)

    for _, row in rows.iterrows():
        color = _speed_color(row['average_speed'], lo, hi)
        contact = 'observed' if str(row['contact_observed']) == 'True' else 'extrapolated'
        hover = (f"<b>{row['average_speed']:.1f} m/s</b> over {row['duration_s']:.2f} s<br>"
                 f"set {row['set_number']}, reception {row['reception_number']}"
                 f" ({row['match']})<br>contact {contact}<extra></extra>")

        points, _ = _flight(row, np.linspace(0, row['duration_s'], PATH_SAMPLES))
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=points[:, 2],
            mode='lines',
            line=dict(color=color, width=5),
            hovertemplate=hover,
            showlegend=False,
        ))

        # The arrow: a cone whose tip sits a little before the reception,
        # pointing along the ball's velocity there.
        t = max(row['duration_s'] - ARROW_LEAD_S, 0.0)
        (tip,), (velocity,) = _flight(row, [t])
        direction = velocity / np.linalg.norm(velocity) * ARROW_LENGTH
        fig.add_trace(go.Cone(
            x=[tip[0]], y=[tip[1]], z=[tip[2]],
            u=[direction[0]], v=[direction[1]], w=[direction[2]],
            anchor='tip',
            sizemode='absolute',
            sizeref=ARROW_LENGTH,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            hovertemplate=hover,
        ))

    for label, prefix, symbol in (('Serve contact', 'start', 'circle'),
                                  ('Reception', 'end', 'diamond')):
        fig.add_trace(go.Scatter3d(
            x=rows[f'{prefix}_x'], y=rows[f'{prefix}_y'], z=rows[f'{prefix}_z'],
            mode='markers',
            marker=dict(size=3, color=SERVE_POINT, symbol=symbol,
                        line=dict(color=MARKER_OUTLINE, width=1)),
            name=label,
            hovertemplate=(f"<b>{label}</b><br>"
                           "x %{x:.2f}  y %{y:.2f}  z %{z:.2f}<extra></extra>"),
        ))

    # Carries the color bar only; plotly attaches one to a marker color scale,
    # and the paths are drawn as plain colored lines.
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(color=[lo], cmin=lo, cmax=hi, colorscale=SPEED_SCALE,
                    showscale=True,
                    colorbar=dict(title=dict(text='Speed', side='top'),
                                  ticksuffix=' m/s', thickness=14, len=0.55, y=0.5)),
        hoverinfo='skip',
        showlegend=False,
    ))

    name = rows['player_name'].mode().iloc[0]
    number = rows['player_number'].iloc[0]
    fig.update_layout(
        title=dict(
            text=(f"{name}  #{number} · {rows['team'].iloc[0]}"
                  f"<br><sup>Average serve speed {rows['average_speed'].mean():.1f} m/s"
                  f" across {len(rows)} serve{'s' if len(rows) != 1 else ''}</sup>"),
            x=0.02,
        ),
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
            # From above and to the side of the serving end. Straight from
            # behind, the paths stack on top of each other and the arrows are
            # seen end-on as blobs; from here both read.
            camera=dict(eye=dict(x=1.9, y=-1.0, z=0.8)),
        ),
        margin=dict(l=0, r=30, t=70, b=0),
        # The points are drawn small on purpose; 'constant' keeps their legend
        # swatches large enough to read.
        legend=dict(orientation='h', yanchor='bottom', y=0.01, itemsizing='constant'),
    )
    return fig


def save_player_serves_html(output_path, dataset, player_number, team=None,
                            include_plotlyjs=True):
    """Write one player's serve figure as a page. Returns the path.

    Open it in a browser to rotate, zoom and hover. By default the plotly
    javascript is embedded, so the file works offline on its own at ~5 MB.
    Writing many figures into one folder, pass `include_plotlyjs='directory'`:
    plotly then writes the library once as `plotly.min.js` beside them and each
    page stays a few tens of kB - the pages need that file next to them.
    """
    fig = plot_player_serves(dataset, player_number, team)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.write_html(output_path, include_plotlyjs=include_plotlyjs, full_html=True)
    return output_path
