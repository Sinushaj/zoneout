"""Entry point: batch-process receptions for one match.

Edit the constants below for the match being processed, then run:

    python run_pipeline.py

Note this rewrites DVW_FILEPATH in place, and takes roughly a minute per
reception on CPU.
"""

from zoneout.config import extract_dict_from_csv
from zoneout.pipeline import get_data_from_reception

# --- Edit these for the match being processed ------------------------------
DVW_FILEPATH = "/home/neo/Desktop/&svk-ork_q5.dvw"
# DVW_FILEPATH = "C:/Data Project/Data Volley 4/Seasons/Elit H 25-26/Scout/&svk-ork_test.dvw"

FIRST_RECEPTION = 24
LAST_RECEPTION = 25      # inclusive
# ---------------------------------------------------------------------------


def main():
    sideline_filepath = extract_dict_from_csv('video_filepaths.csv')['sideline']
    baseline_filepath = extract_dict_from_csv('video_filepaths.csv')['baseline']

    for i in range(FIRST_RECEPTION, LAST_RECEPTION + 1):
        get_data_from_reception(
            dvw_filepath=DVW_FILEPATH,
            reception_number=i,
            sideline_filepath=sideline_filepath,
            baseline_filepath=baseline_filepath,
        )
        print('Finished reception number', i)


if __name__ == "__main__":
    main()
