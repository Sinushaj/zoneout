"""Ball-tracking pipeline for volleyball serve/reception coordinates.

Reconstructs 3D ball trajectories from two synchronized camera angles and
writes the resulting court coordinates back into a DataVolley .dvw scout file.

Modules
-------
court          canonical court geometry, shared by reconstruction and figures
config         read/write the CSV configuration files
scout          .dvw reading and writing
video          extracting short clips from the full match videos
detection      YOLO ball detection (2D pixel midpoints)
reconstruction two camera rays -> one 3D world point
trajectory     cleaning and smoothing, in 2D pixel space and in 3D
events         locating the serve and reception in a trajectory
pipeline       per-reception orchestration
figures        plotting
calibration    interactive tools for producing the config files

Deliberately empty of imports: pulling in `zoneout.detection` costs a torch and
ultralytics import, which the calibration tools have no use for. Import the
submodule you actually need.
"""
