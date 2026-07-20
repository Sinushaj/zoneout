import numpy as np
import pandas as pd
from datavolley.read_dv import DataVolley
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
np.NaN = np.nan



def get_reception_start_frame(dvw_filepath, first_serve_frame, reception_number):
    # Assumes 60 fps at the moment. Maybe change to get exact frame rate later
    plays = DataVolley(dvw_filepath).get_plays()
    # print(plays.head(10))
    reception_times = np.array(plays[plays['skill'] == 'Reception']['video_time']).astype(int) # se om astype funkar
    return first_serve_frame + 60 * (reception_times[reception_number - 1] - reception_times[0]), reception_times[reception_number - 1] # second param temporary to test dvw file insertion



def coords_to_dvindex(point):
    px = point[0]
    py = point[1]

    py += 9
    py *= 80 / 18
    py += 10
    py = int(py)

    px *= 80 / 9
    px += 10
    px = int(px)

    # gör om till index
    index = str(px + 100 * py)
    if len(index) < 4:
        index = '0000'[:4-len(index)] + index
    return index


# TESTER:
# y = -9 motsvarar 1000 = 100 * 10
# y = 9 motsvarar 9000 = 100 * 90
# x = 0 motsvarar 10
# x = 9 motsvarar 89 / 90?





def add_serve_direction(dvw_file, time, start_point, end_point):
    # OBS!!! Kanske måste vrida runt punkter först då serve alltid börjar på mindre sidan

    # rotera planen 180 grader
    if 0 < start_point[1]:
        start_point[0], start_point[1] = 9-start_point[0], -start_point[1]
        end_point[0], end_point[1] = 9-end_point[0], -end_point[1]
    # Compute start and end coordinates in index format for datavolley
    start_point[1] = -9

    start_index = coords_to_dvindex(start_point)
    end_index = coords_to_dvindex(end_point)

    # Open datavolley file
    with open(dvw_file, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line_data = line.split(';')
        if len(line_data) < 12: # isf är det inte en spelhandling
            continue
        if str(time) != line_data[12]:
            continue
        if (line_data[0][3] != 'S') and (line_data[0][3] != 'R'):
            continue
        line_data[4] = start_index
        line_data[5] = '-1-1'
        line_data[6] = end_index
        lines[i] = ';'.join(line_data)
        if line_data[0][3] == 'R':
            break
    
    with open(dvw_file, 'w') as f:
        f.writelines(lines)

# add_serve_direction("C:/Data Project/Data Volley 4/Seasons/Elit H 25-26/Scout/&svk-ork_test.dvw", 253, [8.9,8.9,0], [0,-6.5,0])
