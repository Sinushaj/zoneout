"""Generate the serve figure for every serving player on one team.

    python datasets/serve_figures.py

Asks for a serve dataset (a file dialog, opening in this directory), lists the
teams in it with how many serves each has, and asks which one. Then writes one
page per player who has served, via `zoneout.figures.save_player_serves_html`,
into

    serve_figures/<team>/<number>_<name>.html

next to the chosen dataset. The plotly library is written once into that folder
as `plotly.min.js` rather than embedded in every page, so the pages have to stay
next to it; move or share the folder as a whole.

Re-running for the same team overwrites its figures, so they always reflect the
dataset as it is now.
"""

import os
import re
import sys
import tkinter as tk
from tkinter import filedialog

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# The repo root, so `zoneout` imports however this script is started.
sys.path.insert(0, os.path.dirname(HERE))

from zoneout.figures import save_player_serves_html  # noqa: E402


def select_dataset():
    """The dataset's path from a file dialog, or '' if it was cancelled."""
    root = tk.Tk()
    root.withdraw()
    root.update()  # ensures the dialog appears
    path = filedialog.askopenfilename(
        title='Choose a serve dataset',
        initialdir=HERE,
        filetypes=[('CSV files', '*.csv'), ('All files', '*')],
    )
    root.destroy()
    return path


def select_team(serves):
    """Ask for one of the dataset's teams by number. None if the answer is blank."""
    counts = serves.groupby('team').size().sort_index()
    print('Teams in the dataset:')
    for i, (team, n) in enumerate(counts.items(), start=1):
        players = serves.loc[serves['team'] == team, 'player_number'].nunique()
        print(f'  {i}. {team}  ({n} serves, {players} players)')

    while True:
        answer = input('Which team (number, blank to quit)? ').strip()
        if not answer:
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(counts):
            return counts.index[int(answer) - 1]
        print(f'  Please enter a number from 1 to {len(counts)}.')


def safe_filename(text):
    """`text` with anything a file system might object to replaced by '_'."""
    return re.sub(r'[^\w\-]+', '_', str(text), flags=re.UNICODE).strip('_')


def main():
    path = select_dataset()
    if not path:
        print('No dataset chosen.')
        return

    serves = pd.read_csv(path)
    if serves.empty:
        print(f'{path} has no serves in it.')
        return

    team = select_team(serves)
    if team is None:
        return

    players = (serves[serves['team'] == team]
               .groupby('player_number')['player_name']
               .agg(lambda names: names.mode().iloc[0]))
    out_dir = os.path.join(os.path.dirname(path), 'serve_figures', safe_filename(team))

    print(f'Writing {len(players)} figure(s) for {team} to {out_dir}')
    for number, name in players.sort_index(key=lambda n: n.astype(int)).items():
        out = os.path.join(out_dir, f'{number}_{safe_filename(name)}.html')
        save_player_serves_html(out, serves, number, team=team,
                                include_plotlyjs='directory')
        print(f'  #{number} {name}: {os.path.basename(out)}')


if __name__ == '__main__':
    main()
