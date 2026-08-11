# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Prototype pipeline that reconstructs 3D ball trajectories for volleyball serves and receptions from two
synchronized camera angles — a sideline GoPro and a baseline camera ("zve10") — then writes the computed
court coordinates back into a DataVolley `.dvw` scout file. This is a personal research/prototype codebase:
there is no test suite, linter config, build step, or packaging (no `requirements.txt`/`pyproject.toml`).
Some comments in the code are written in Swedish.

## Environment

- Python 3.12, dependencies installed into `.venv` via plain `venv`+`pip` (no lockfile — if the env is lost,
  reinstall by inspecting what's listed below). Activate with `source .venv/bin/activate`.
- Key dependencies: `ultralytics` (YOLO), `torch`/`torchvision`, `opencv-python`, `numpy`, `pandas`,
  `pydatavolley` (imported as `datavolley`), `matplotlib`, `plotly` (interactive figures).
- `gala_model.pt` is a custom-trained YOLO model checked into the repo for volleyball-ball detection.
- **Inference must stay on CPU.** See "GPU is unusable" in `NOTES.md` — `torch.cuda.is_available()` returns
  True but the installed CUDA build has no kernels for this machine's GPU, so letting ultralytics auto-select
  the device fails deep inside the forward pass with a misleading error.

## Layout

Two entry points at the repo root, everything else in the `zoneout` package:

```
run_pipeline.py         batch-process receptions for one match (edit constants at top)
update_parameters.py    interactive calibration
zoneout/
    court.py            canonical court geometry, shared by reconstruction and figures
    config.py           read/write the CSV config files
    scout.py            .dvw reading and writing
    video.py            extracting short clips from the full match videos
    detection.py        YOLO ball detection (2D pixel midpoints)
    reconstruction.py   two camera rays -> one 3D world point
    trajectory.py       cleaning and smoothing, in 2D pixel space and in 3D
    events.py           locating the serve and reception in a trajectory
    pipeline.py         per-reception orchestration
    figures/            plotting
    calibration/        interactive tools that produce the CSV config files
```

`zoneout/__init__.py` is deliberately free of imports: pulling in `zoneout.detection` costs a torch and
ultralytics import that the calibration tools have no use for. Import the submodule you need directly.

## Running

Both entry points take no CLI arguments — edit the constants at the top of the file first.

1. `python update_parameters.py` — one-time/per-match interactive calibration:
   - Prompts to pick the sideline & baseline video files (Tk file dialog) → `video_filepaths.csv`.
   - Prompts to click the court reference points in each camera's first frame (OpenCV window), **in the order
     they appear in `zoneout.court.CALIBRATION_POINTS`** → `gopro_points.csv` / `zve10_points.csv`.
   - Prompts to manually navigate each video to the first reception's start frame → `first_reception_frames.csv`.
2. `python run_pipeline.py` — batch-processes receptions using the CSVs from step 1. Edit `DVW_FILEPATH`,
   `FIRST_RECEPTION` and `LAST_RECEPTION` at the top. Writes per-reception output to
   `reception_data/receptionN/` (two annotated detection videos + a trajectory plot) and **patches the `.dvw`
   file in place**. Budget roughly a minute per reception on CPU.

## Architecture

Per-reception processing (`zoneout.pipeline.get_data_from_reception`) chains together:

1. **Frame lookup** (`scout.get_reception_start_frame`): reads the `.dvw` via `pydatavolley` to get each
   reception's `video_time`, then combines that with the manually-recorded first-reception frame numbers
   (`first_reception_frames.csv`) to compute the start frame for reception N in each camera. Assumes 60fps.
2. **Video chunking** (`video.create_video_chunk`): extracts a 5-second clip around that frame from each full
   match video into `temporary_videos/`, so detection only runs on a small window rather than the whole match.
3. **Ball detection** (`detection.process_video`): runs `gala_model.pt` frame-by-frame on each clip, keeping
   the highest-confidence box's midpoint (or `None`). Then `trajectory.remove_bad_points` (drops points that
   jump implausibly far given the elapsed frames) and `trajectory.interpolate_nones` (linearly fills gaps ≤60
   frames bounded by valid points). Run independently per camera.
4. **Triangulation** (`reconstruction.point_from_camera_coordinates`): per frame, turns each camera's 2D pixel
   detection into a 3D ray via `cv2.solvePnP` (that camera's intrinsics + the clicked court reference points
   as the 2D↔3D correspondence), then intersects the two rays by least squares (`intersect_rays`).
5. **Smoothing** (`trajectory.smooth_trajectory`) over the resulting 3D trajectory.
6. **Serve/reception detection** (`events.find_serve_and_receive`): splits the trajectory where `y` changes
   sign (ball crossing the net); the serve is the pre-split point nearest `|y| = 9` (the baseline), the
   reception is the first sharp direction change (>45°) after the split.
7. **Write-back** (`scout.add_serve_direction`): converts the 3D start/end coordinates into DataVolley's
   numeric court-index format (`coords_to_dvindex`) and rewrites the matching `S`/`R` skill lines in the
   `.dvw` in place, matched by `video_time`. Handles court-side rotation (coordinates flipped 180° when the
   action isn't already on the serving side).
8. **Figures** (`figures/`): a 3D plot of the trajectory with the serve/reception points highlighted, written
   twice per reception into `reception_data/receptionN/` — `raw_trajectory.png` for flicking through, and
   `raw_trajectory.html`, an interactive page you open in a browser and rotate/zoom/hover.

### Coordinate system and court geometry

The court is modeled in meters as `x ∈ [0,9]`, `y ∈ [-9,9]`, `z` up, with the net along `x` at `y=0`.
**All of this lives in `zoneout/court.py` and nowhere else** — both the reconstruction (which needs the 3D
reference points for solvePnP) and the figures (which draw the lines) read from it, so the two cannot drift
apart. `court.CALIBRATION_POINTS` is order-sensitive: it must match the order the corresponding pixels were
clicked into `gopro_points.csv` / `zve10_points.csv`. Changing it means re-clicking both calibration files.

### Two smoothing functions, deliberately distinct

`trajectory.smooth_trajectory` and `trajectory.moving_average` are **not** interchangeable, and both used to
be called `moving_average` in different modules — which made importing the wrong one easy.

- `smooth_trajectory(points)` — used on 3D world points by the pipeline. Preserves `None` gaps and rejects
  neighbours further than a threshold before averaging.
- `moving_average(points, window_size=5)` — a plain centred windowed mean over a dense array, used inside
  `events.find_serve_and_receive`.

### `figures/` package

- `style.py` — the palette. Read its docstring before changing any color: the three data colors were checked
  with the dataviz palette validator against the orange court, and the obvious choice of a **red** reception
  marker was rejected because red-on-orange measured ΔE 14.2 for normal vision, below the floor of 15.
- `court.py` — `draw_court(ax)` for matplotlib, `court_traces()` for plotly. Geometry comes from
  `zoneout.court`.
- `trajectory.py` — `save_trajectory_figure(path, ...)` writes the PNG, `save_trajectory_html(path, ...)`
  writes the interactive page. `plot_trajectory(...)` / `plot_trajectory_plotly(...)` return the figure
  objects if you want to tweak before saving. All accept point lists containing `None` and skip those frames.

Put new figure types here rather than inlining plotting back into the pipeline.

Three things not to undo:

- The matplotlib figures set **`ax.computed_zorder = False`** and draw with explicit zorder. Matplotlib's 3D
  renderer otherwise depth-sorts whole artists, which paints the court floor over the trajectory above it.
- The matplotlib court is **opaque** while the plotly court is translucent. Matplotlib composites a
  translucent plane over geometry it thinks is behind it, which tinted the blue path purple; plotly renders
  the same scene correctly.
- The serve/reception markers keep their **text labels and outline rings**. The validator raised a contrast
  warning for the serve green against the court, and a visible label is the required relief — identity must
  never rest on hue alone.

The matplotlib figures build on `matplotlib.figure.Figure` directly rather than `pyplot`. pyplot keeps every
figure in a global registry, so in a batch run figures accumulate in memory unless explicitly closed; going
through `Figure` sidesteps that and needs no GUI backend. Keep it that way — don't reintroduce
`plt.figure()` here.

### Test data

`&svk-ork_test.dvw` in the repo root is a small DataVolley scout file kept as a fixture for exercising
`add_serve_direction` without touching a real match file.
