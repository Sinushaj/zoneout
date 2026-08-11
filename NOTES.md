# Working notes

Living notes on the state of the pipeline, open issues, and things to check before/while picking up work in a
new session. See `CLAUDE.md` for the architecture reference; this file is for status and TODOs, which will
go stale faster.

## Current state (as of 2026-08-11)

- The full pipeline (`update_parameters.py` → `run_pipeline.py`) runs end-to-end on at least one real match
  (`reception_data/reception1` … `reception17` exist with detection videos + trajectory plots), using video
  files under `/home/neo/Videos/svk-ving_quarter_final/` (per `video_filepaths.csv`) and calibration points
  in `gopro_points.csv` / `zve10_points.csv` (currently 8 points each, including two "antenna" points — see
  open questions below).
- The code was reorganized into the `zoneout` package in Aug 2026 — see "Restructure" below.
- Match-specific constants now live at the top of `run_pipeline.py` (`DVW_FILEPATH`, `FIRST_RECEPTION`,
  `LAST_RECEPTION`) rather than buried in the module body. Still hand-edited per match; a config file or CLI
  args would be the next step if this needs to run across many matches.

## Gotcha: GPU is unusable, inference must stay on CPU

This machine has a GTX 980 (compute capability **sm_52**), but the installed `torch 2.13.0+cu130` ships
kernels only for sm_75/80/86/90/100/120, and cuDNN 9 refuses anything below sm_75. Critically,
`torch.cuda.is_available()` still returns **True**, so ultralytics will happily move the model to the GPU and
then fail on the first conv layer with the very unhelpful:

```
RuntimeError: GET was unable to find an engine to execute this computation
```

`zoneout.detection.process_video` therefore takes `device="cpu"` and defaults to it — do not remove that or
let ultralytics auto-select the device. Reinstalling a different CUDA build will not help; Maxwell-era cards
are below the floor of both modern PyTorch and cuDNN 9. Using the GPU would mean pinning a much older torch.

Cost of CPU inference: roughly **65 s per reception** (2 × ~300 frames).

## Known issues / open questions

- **`add_serve_direction` court rotation is flagged as unverified** (`zoneout/scout.py`, Swedish comment:
  "OBS!!! Kanske måste vrida runt punkter först då serve alltid börjar på mindre sidan" — "may need to rotate
  points first since serve always starts on the smaller side"). Worth checking the written `.dvw` coordinates
  against DataVolley's own display for a few known receptions before trusting this broadly.
- **Reception points can land off the court.** Reception 18 reconstructs to `x ≈ 11.1` (the court is only 0–9
  wide) at `z ≈ 2.9`. The serve point for the same rally looks sane, so this is likely the direction-change
  detection in `events.find_serve_and_receive` firing on the wrong frame, or triangulation drift late in the
  trajectory. Worth eyeballing a batch of `raw_trajectory.png` plots before trusting the write-back.
- **The antenna calibration points were experimental.** `court.CALIBRATION_POINTS` currently has 8 entries,
  the last two being the antenna tops at z = 3.23. Confirm whether that is the intended permanent set or
  should revert to the original 6 ground points. Changing it means re-clicking both calibration CSVs.
- **60fps is hardcoded** in `zoneout.scout.get_reception_start_frame` — will silently misalign frame numbers
  if fed a video recorded at a different frame rate.
- **Repo hygiene**: `.venv/`, `reception_data/`, and `temporary_videos/` are untracked. A `.gitignore`
  covering `__pycache__/`, `*.pyc`, `.venv/`, `temporary_videos/` would be worth adding — `reception_data/`
  is real pipeline output so it's a judgment call whether it belongs in git (it contains per-reception
  videos, which are large).

## Ideas for next steps

- Decide how the serve/reception write-back accuracy will be validated (compare against manual scouting, or
  visually against the saved `raw_trajectory.png` plots).
- If interpolation/smoothing quality is a problem, the deleted `clean_ball_trajectory` (parabola-fit segment
  cleaner) and `ransac_ball_trajectory` are recoverable from git history — see "Dead code cleanup" below.

## Restructure into the `zoneout` package (Aug 2026)

Flat modules with vague names were reorganized into a package by concern. Old → new:

| old                          | new                                              |
|------------------------------|--------------------------------------------------|
| `main.py`                    | `zoneout/pipeline.py` + `run_pipeline.py`        |
| `detection_stuff.py`         | `zoneout/detection.py` + `zoneout/trajectory.py` |
| `save_to_csv.py`             | `zoneout/config.py` + `zoneout/calibration/points.py` |
| `line_from_point.py`         | `zoneout/reconstruction.py`                      |
| `scout_file_extraction.py`   | `zoneout/scout.py`                               |
| `video_extraction.py`        | `zoneout/video.py`                               |
| `video_syncer.py`            | `zoneout/calibration/sync.py`                    |
| `detect_serve_and_reception.py` | `zoneout/events.py`                           |
| `figures/`                   | `zoneout/figures/`                               |
| (new)                        | `zoneout/court.py`                               |

Behaviour-affecting changes made along the way, all verified against a golden end-to-end run of reception 18
(identical 3D coordinates, identical figure, identical patched `.dvw`):

- The two different functions both named `moving_average` were given distinct names —
  `trajectory.smooth_trajectory` (3D, gap-preserving; was in `detection_stuff`) and
  `trajectory.moving_average` (dense windowed mean; was in `detect_serve_and_reception`). Bodies unchanged.
- Court geometry was centralized in `zoneout/court.py`; `reconstruction` and `figures` now read the same
  constants instead of each hardcoding their own copy.
- `point_from_camera_coordinates` lost its unused `plot_rays` parameter.
- `video_syncer.main()` became `calibration.sync.sync_two_videos(video1_path, video2_path)` — it had
  hardcoded Windows paths from the original dev machine.
- Both entry points gained `if __name__ == "__main__"` guards, so importing them no longer starts a batch run
  or an interactive prompt.
- `update_parameters.py` now derives the number of points to click from `len(court.CALIBRATION_POINTS)`
  instead of a hardcoded 8.

## Dead code cleanup (Aug 2026)

Removed everything unreachable from the two entry points. All of it is in git history:

- `parabola_filter.py`, `kallman_filter.py` — unused experimental trajectory cleaners.
- `deprecated/` (whole package) — superseded single-camera approach. `line_from_point.py` had a live
  `from deprecated.court_points import *`, but the imported `court_points` name was shadowed by a local
  variable of the same name inside `point_from_camera_coordinates`, so removing it was a no-op.
- `TEMP_coordinate_tests.py`, `temp.csv` — scratch work for the RANSAC filter.

`&svk-ork_test.dvw` was deliberately kept as a test fixture. `scikit-learn` is now unused by the tree (it
was only needed by the two deleted filters) though it remains installed in `.venv`.
