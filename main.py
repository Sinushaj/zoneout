import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import os
import numpy as np

from detection_stuff import process_video, interpolate_nones
from line_from_point import point_from_camera_coordinates
from scout_file_extraction import get_reception_start_frame
from video_extraction import create_video_chunk
from kallman_filter import clean_ball_trajectory
# from parabola_filter import parabola_filter

'''
# testa att fixa andra vägar
gopro_path = 'gopro_serve_clip.mp4'
zve10_path = 'zve10_serve_clip.mp4'
'''

# model_path = 'gala_model.pt'


# testa extraction 
dvw_filepath = "C:\\Data Project\\Data Volley 4\\Seasons\\Elit H 25-26\\Scout\\&svk-ork_kvart_2.dvw"
reception_number = 4 #temporary hardcoded, make into 

sideline_start_frame = 27354
sideline_filepath = 'C:\\Users\\neoda\\OneDrive\\Desktop\\GH011611.MP4'
# sideline_reception_frame = get_reception_start_frame(dvw_filepath, sideline_start_frame, reception_number)

baseline_start_frame = 124 #hitta
baseline_filepath = "C:\\Users\\neoda\\OneDrive\\Desktop\\baseline_test.mp4"
# baseline_reception_frame = get_reception_start_frame(dvw_filepath, baseline_start_frame, reception_number)

#create_video_chunk(sideline_filepath, 'sideline_temp.mp4', sideline_reception_frame)
#create_video_chunk(baseline_filepath, 'baseline_temp.mp4', baseline_reception_frame)

# gopro_path = 'temporary_videos\\sideline_temp.mp4'
# zve10_path = 'temporary_videos\\baseline_temp.mp4'


# slut på testa extraction



def get_data_from_reception(dvw_filepath, reception_number, sideline_start_frame, sideline_filepath, baseline_start_frame, baseline_filepath):
    os.makedirs(os.path.dirname(f'reception_data\\reception{reception_number}\\file_to_add'), exist_ok=True)

    model_path = 'gala_model.pt'
    gopro_path = 'temporary_videos\\sideline_temp.mp4'
    zve10_path = 'temporary_videos\\baseline_temp.mp4'

    sideline_reception_frame = get_reception_start_frame(dvw_filepath, sideline_start_frame, reception_number)
    baseline_reception_frame = get_reception_start_frame(dvw_filepath, baseline_start_frame, reception_number)
    create_video_chunk(sideline_filepath, 'sideline_temp.mp4', sideline_reception_frame)
    create_video_chunk(baseline_filepath, 'baseline_temp.mp4', baseline_reception_frame)
    print("Done creating video chunks")



    gopro_coordinate_list = process_video(gopro_path, model_path, 0.4, f'reception_data\\reception{reception_number}\\gopro_boxes.mp4')
    gopro_coordinate_list = interpolate_nones(gopro_coordinate_list)
    # print(gopro_coordinate_list)

    zve10_coordinate_list = process_video(zve10_path, model_path, 0.4, f'reception_data\\reception{reception_number}\\zve10_boxes.mp4')
    zve10_coordinate_list = interpolate_nones(zve10_coordinate_list)
    # print(zve10_coordinate_list)



    real_coords = []
    for i in range(min(len(gopro_coordinate_list), len(zve10_coordinate_list))):
        gopro_coordinate = gopro_coordinate_list[i]
        zve10_coordinate = zve10_coordinate_list[i]
        if (gopro_coordinate is None) or (zve10_coordinate is None):
            real_coords.append(None)
            continue
        #print(gopro_coordinate, zve10_coordinate)
        real_coordinate = point_from_camera_coordinates(gopro_coordinate, zve10_coordinate)
        #print('real coordinate:', real_coordinate)
        real_coords.append(real_coordinate)


    kalman_real_coords = clean_ball_trajectory(real_coords)


    # ---------------
    # Plotta punkter
    # ---------------


    my_list = [elem for elem in real_coords if not (elem is None)]

    x = [p[0] for p in my_list]
    y = [p[1] for p in my_list]
    z = [p[2] for p in my_list]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Rectangle in XY plane (z=0) ---
    rect_x = [0, 9, 9, 0, 0]
    rect_y = [-9, -9, 9, 9, -9]
    rect_z = [0, 0, 0, 0, 0]

    all_x = np.concatenate([x, rect_x])
    all_y = np.concatenate([y, rect_y])
    all_z = np.concatenate([z, rect_z])

    ax.set_box_aspect([
        np.ptp(all_x),
        np.ptp(all_y),
        np.ptp(all_z)
    ])  
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    # Scatter points
    ax.scatter(x, y, z)

    ax.plot(rect_x, rect_y, rect_z, color='black')
    # --- Vertical lines ---
    lines = [
        [(0, 0, 0), (9, 0, 0)],
        [(0, 3, 0), (9, 3, 0)],
        [(0,-3, 0), (9,-3, 0)]
    ]

    for line in lines:
        lx = [line[0][0], line[1][0]]
        ly = [line[0][1], line[1][1]]
        lz = [line[0][2], line[1][2]]
        ax.plot(lx, ly, lz, color = 'black')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.savefig(
        f"reception_data\\reception{reception_number}\\raw_trajectory.png",
        dpi=300,              # higher resolution
        bbox_inches="tight",  # trims whitespace
        transparent=True      # optional background
    )

    plt.show()



    # Print filtered
    my_list = kalman_real_coords


    x = [p[0] for p in my_list]
    y = [p[1] for p in my_list]
    z = [p[2] for p in my_list]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # --- Rectangle in XY plane (z=0) ---
    rect_x = [0, 9, 9, 0, 0]
    rect_y = [-9, -9, 9, 9, -9]
    rect_z = [0, 0, 0, 0, 0]

    all_x = np.concatenate([x, rect_x])
    all_y = np.concatenate([y, rect_y])
    all_z = np.concatenate([z, rect_z])

    ax.set_box_aspect([
        np.ptp(all_x),
        np.ptp(all_y),
        np.ptp(all_z)
    ])  
    ax.set_xlim(min(all_x), max(all_x))
    ax.set_ylim(min(all_y), max(all_y))
    ax.set_zlim(min(all_z), max(all_z))

    # Scatter points
    ax.scatter(x, y, z)


    ax.plot(rect_x, rect_y, rect_z, color='black')
    # --- Vertical lines ---
    lines = [
        [(0, 0, 0), (9, 0, 0)],
        [(0, 3, 0), (9, 3, 0)],
        [(0,-3, 0), (9,-3, 0)]
    ]

    for line in lines:
        lx = [line[0][0], line[1][0]]
        ly = [line[0][1], line[1][1]]
        lz = [line[0][2], line[1][2]]
        ax.plot(lx, ly, lz, color = 'black')

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    fig.savefig(
        f"reception_data\\reception{reception_number}\\cleaned_trajectory.png",
        dpi=300,              # higher resolution
        bbox_inches="tight",  # trims whitespace
        transparent=True      # optional background
    )

    plt.show()

for i in range(1,6):
    get_data_from_reception(dvw_filepath = dvw_filepath, reception_number = i, sideline_start_frame = sideline_start_frame, sideline_filepath=sideline_filepath, baseline_start_frame=baseline_start_frame, baseline_filepath=baseline_filepath)
    print('Finished reception number', i)