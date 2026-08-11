"""Entry point: interactive calibration.

Prompts for the things that change per match or per camera setup and writes
them to the CSV config files the pipeline reads:

    python update_parameters.py
"""

import tkinter as tk
from tkinter import filedialog

from zoneout.calibration import get_screen_coordinates, navigate_video
from zoneout.config import extract_dict_from_csv, write_to_csv, write_tuples_to_csv
from zoneout.court import CALIBRATION_POINTS


def select_file():
    root = tk.Tk()
    root.withdraw()
    root.update()  # ensures the dialog appears

    file_path = filedialog.askopenfilename()

    root.destroy()  # clean up
    return file_path.replace("\\", "\\\\")


def main():
    # uppdatera court coordinates och videolänk:
    if input('Change video paths?').capitalize() == 'Yes':
        sideline_path = select_file()
        baseline_path = select_file()
        write_to_csv('video_filepaths.csv', sideline_path, baseline_path)

    if input('Change reference court coordinates?').capitalize() == 'Yes':
        # Click the points in the order they are listed in
        # zoneout.court.CALIBRATION_POINTS - the two lists are matched up
        # positionally when handed to solvePnP.
        n_points = len(CALIBRATION_POINTS)

        gopro_filepath = extract_dict_from_csv('video_filepaths.csv')['sideline']
        gopro_court_points = get_screen_coordinates(gopro_filepath, n_points)
        write_tuples_to_csv(gopro_court_points, 'gopro_points.csv')

        zve10_filepath = extract_dict_from_csv('video_filepaths.csv')['baseline']
        zve10_court_points = get_screen_coordinates(zve10_filepath, n_points)
        write_tuples_to_csv(zve10_court_points, 'zve10_points.csv')

    if input('Change start frames?').capitalize() == 'Yes':
        sideline_frame = navigate_video(extract_dict_from_csv('video_filepaths.csv')['sideline'])
        baseline_frame = navigate_video(extract_dict_from_csv('video_filepaths.csv')['baseline'])
        write_to_csv('first_reception_frames.csv', sideline_frame, baseline_frame)
        # Inte testad ännu

    print('done!')


if __name__ == "__main__":
    main()
