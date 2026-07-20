import tkinter as tk
from tkinter import filedialog
from save_to_csv import write_to_csv, extract_dict_from_csv, get_screen_coordinates, write_tuples_to_csv
from video_syncer import navigate_video



def select_file():
    root = tk.Tk()
    root.withdraw()
    root.update()  # ensures the dialog appears

    file_path = filedialog.askopenfilename()

    root.destroy()  # clean up
    return file_path.replace("\\", "\\\\")



# uppdatera court coordinates och videolänk:
if input('Change video paths?').capitalize() == 'Yes':
    sideline_path =  select_file()
    baseline_path = select_file()
    write_to_csv('video_filepaths.csv', sideline_path, baseline_path)

if input('Change reference court coordinates?').capitalize() == 'Yes':
    gopro_filepath = extract_dict_from_csv('video_filepaths.csv')['sideline']
    gopro_court_points = get_screen_coordinates(gopro_filepath, 8) #temp 8 med antenner test
    write_tuples_to_csv(gopro_court_points, 'gopro_points.csv')

    zve10_filepath = extract_dict_from_csv('video_filepaths.csv')['baseline']
    zve10_court_points = get_screen_coordinates(zve10_filepath, 8) #temp 8 med antenner test
    write_tuples_to_csv(zve10_court_points, 'zve10_points.csv')

if input('Change start frames?').capitalize() == 'Yes':
    sideline_frame = navigate_video(extract_dict_from_csv('video_filepaths.csv')['sideline'])
    baseline_frame = navigate_video(extract_dict_from_csv('video_filepaths.csv')['baseline'])
    write_to_csv('first_reception_frames.csv', sideline_frame, baseline_frame)
    # Inte testad ännu

print('done!')



