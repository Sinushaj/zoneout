import numpy as np
import pandas as pd
from datavolley.read_dv import DataVolley
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
np.NaN = np.nan

'''
dv = DataVolley("C:\\Data Project\\Data Volley 4\\Seasons\\Elit H 25-26\\Scout\\&svk-ork_kvart_2.dvw")
plays = dv.get_plays()
receptions = plays[plays['skill'] == 'Reception']['video_time']
# View result
print(receptions.head(20))
print(np.array(receptions))


#temporary test:
first_serve_frame = 27354
serve_to_get = 2

print(first_serve_frame + 60 * (int(np.array(receptions)[serve_to_get - 1]) - int(np.array(receptions)[0])))
'''

def get_reception_start_frame(dvw_filepath, first_serve_frame, reception_number):
    # Assumes 60 fps at the moment. Maybe change to get exact frame rate later
    plays = DataVolley(dvw_filepath).get_plays()
    reception_times = np.array(plays[plays['skill'] == 'Reception']['video_time']).astype(int) # se om astype funkar
    return first_serve_frame + 60 * (reception_times[reception_number - 1] - reception_times[0])