"""Per-reception orchestration: video in, 3D coordinates written back to the .dvw.

This module holds the processing for a single reception. The match-specific
constants and the loop over receptions live in `run_pipeline.py` at the repo
root, so importing this module never kicks off a batch run.
"""

import os

from .config import extract_dict_from_csv
from .detection import process_video
from .events import find_serve_and_receive
from .figures import save_trajectory_figure
from .reconstruction import point_from_camera_coordinates
from .scout import get_reception_start_frame, add_serve_direction
from .trajectory import interpolate_nones, remove_bad_points, smooth_trajectory
from .video import create_video_chunk









def get_data_from_reception(dvw_filepath, reception_number, sideline_filepath, baseline_filepath):
    # os.makedirs(os.path.dirname(f'reception_data\\reception{reception_number}\\file_to_add'), exist_ok=True)
    path = os.path.join("reception_data", f"reception{reception_number}", "file_to_add")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    model_path = 'gala_model.pt'
    gopro_path = 'temporary_videos/sideline_temp.mp4'
    zve10_path = 'temporary_videos/baseline_temp.mp4'


    sideline_start_frame = int(extract_dict_from_csv('first_reception_frames.csv')['sideline'])
    baseline_start_frame = int(extract_dict_from_csv('first_reception_frames.csv')['baseline'])
    # print('first_frames:', sideline_start_frame, baseline_start_frame)

    sideline_reception_frame, video_time = get_reception_start_frame(dvw_filepath, sideline_start_frame, reception_number)
    baseline_reception_frame, video_time = get_reception_start_frame(dvw_filepath, baseline_start_frame, reception_number)
    create_video_chunk(sideline_filepath, 'sideline_temp.mp4', sideline_reception_frame)
    create_video_chunk(baseline_filepath, 'baseline_temp.mp4', baseline_reception_frame)
    print("Done creating video chunks")



    gopro_coordinate_list = process_video(gopro_path, model_path, 0.60, f'reception_data/reception{reception_number}/gopro_boxes.mp4')
    gopro_coordinate_list = remove_bad_points(gopro_coordinate_list)
    gopro_coordinate_list = interpolate_nones(gopro_coordinate_list)
    print("Done processing gopro video")


    zve10_coordinate_list = process_video(zve10_path, model_path, 0.60, f'reception_data/reception{reception_number}/zve10_boxes.mp4')
    zve10_coordinate_list = remove_bad_points(zve10_coordinate_list)
    zve10_coordinate_list = interpolate_nones(zve10_coordinate_list)
    print("Done processing zve10 video")


    real_coords = []
    for i in range(min(len(gopro_coordinate_list), len(zve10_coordinate_list))):
        gopro_coordinate = gopro_coordinate_list[i]
        zve10_coordinate = zve10_coordinate_list[i]
        if (gopro_coordinate is None) or (zve10_coordinate is None):
            real_coords.append(None)
            continue
        real_coordinate = point_from_camera_coordinates(gopro_coordinate, zve10_coordinate)
        real_coords.append(real_coordinate)
    real_coords = smooth_trajectory(real_coords)



    raw_start, raw_end = find_serve_and_receive([elem for elem in real_coords if not (elem is None)])
    print('returned sereve/reception:', raw_start, raw_end)

    # ----------------------
    # Add direction to file
    # ----------------------

    add_serve_direction(dvw_filepath, video_time, raw_start.copy(), raw_end.copy())

    # ---------------
    # Plotta punkter
    # ---------------

    save_trajectory_figure(
        f"reception_data/reception{reception_number}/raw_trajectory.png",
        real_coords,
        serve_point=raw_start,
        receive_point=raw_end,
    )