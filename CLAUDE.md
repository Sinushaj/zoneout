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
- `ball_n1280_v2.pt` is the default ball detector (YOLOv8n at imgsz 1280, trained in `~/Desktop/ball_detector`,
  run `n1280_v2`). `gala_model.pt`, the previous default, stays in the repo for comparison. The detection
  thresholds below were measured on `gala_model.pt` and have not been re-measured on the new model.
- **Inference must stay on CPU.** See "GPU is unusable" in `NOTES.md` — `torch.cuda.is_available()` returns
  True but the installed CUDA build has no kernels for this machine's GPU, so letting ultralytics auto-select
  the device fails deep inside the forward pass with a misleading error.

## Layout

Three entry points at the repo root, everything else in the `zoneout` package:

```
run_pipeline.py         batch-process receptions for one match (edit constants at top)
run_attack_pipeline.py  the attack pipeline (in progress — see ATTACK_PLAN.md)
update_parameters.py    interactive calibration
zoneout/
    court.py            canonical court geometry, shared by reconstruction and figures
    cameras.py          which camera films each side, and the intrinsics it implies
    config.py           read/write the CSV config files
    scout.py            .dvw reading and writing
    timing.py           video_time -> a frame number in each camera
    video.py            extracting short clips from the full match videos
    detection.py        YOLO ball detection (candidate 2D pixel midpoints) + debug annotation
    reconstruction.py   two camera rays -> one 3D world point, and candidate matching
    trajectory.py       cleaning and smoothing, in 2D pixel space and in 3D
    ballistic.py        optional: segmented free-flight fitting, cleaning + gap filling
    events.py           locating the serve and reception in a trajectory
    serve_dataset.py    the serve dataset, datasets/serves.csv
    reception_dataset.py  the serve-and-reception dataset, datasets/receptions.csv
    pipeline.py         per-reception orchestration
    attacks.py          per-attack orchestration (in progress)
    dv_grid.py          the DataVolley coordinate grid: metres <-> index and zone
    dvw_edit.py         the safe .dvw writer: coordinates, and inserted lines
    set_codes.py        which set line goes in front of an attack, and what it says
    figures/            plotting
    calibration/        interactive tools that produce the CSV config files
                        points.py: the court-point picker window; court_diagram.py: its plan view
                        sync.py: the window that finds the anchor frame in one camera
                        frames.py: decoding one frame at a time, shared by both windows
```

`ATTACK_PLAN.md` is the design plan for the attack work — what was measured about the `.dvw` format, the
structure being built toward, and the order of work. Fold the parts of it that land into this file.

`zoneout/__init__.py` is deliberately free of imports: pulling in `zoneout.detection` costs a torch and
ultralytics import that the calibration tools have no use for. Import the submodule you need directly.

## Running

Both entry points take no CLI arguments — edit the constants at the top of the file first.

1. `python update_parameters.py` — one-time/per-match interactive calibration.
   - Prompts to pick the sideline & baseline video files (Tk file dialog) → `video_filepaths.csv`.
   - Prompts to pick the `.dvw` scout file (Tk file dialog) → `scout_filepath.csv`, which **every entry
     point reads**; none of them has a scout path of its own. It is a separate file because
     `video_filepaths.csv` is rewritten whole when the videos are re-picked. The sync step below looks up
     its action's video_time in this file, so the scout file and the anchor belong together.

     **An anchor taken against a different scout file fails nothing — it cuts every clip from the wrong
     minute.** The first reception looks right (it is the anchor, found by eye) and every later one is off
     by the difference between the two files' rally spacing. This happened: a Södermalm match was synced and
     run against the Örkelljunga scout file, whose first reception is at video_time 221 against Södermalm's
     101. `timing.anchor_mismatch` guards it by asking the scout file for the anchor's action (set + skill)
     again and comparing video_times; both pipelines refuse to start on a mismatch, and choosing a new scout
     file prints the same message. The team names are printed alongside the path for the same reason.
   - Prompts for **which camera films each side**, and the focal length for a body with an interchangeable
     lens → `camera_models.csv`. See "Camera intrinsics" below.
   - Prompts to click the court reference points in each camera's first frame → `gopro_points.csv` /
     `zve10_points.csv`. They are clicked **in the order they appear in `zoneout.court.CALIBRATION_POINTS`**,
     which is load-bearing: the two lists are paired positionally in `solvePnP`, so a wrong order does not
     fail, it quietly solves for the wrong camera pose. The picker window (`calibration.points`) therefore
     asks for one point at a time and marks it on a plan view of the court (`calibration.court_diagram`)
     rather than expecting the order to be remembered — the video fills the window and the diagram sits in a
     fixed panel beside it, with a magnifier, the list of points, and Undo/Clear/Save/Cancel. A scrubber
     under the picture moves the video: the first frame is not always usable (a recording can open on a hand
     in front of the lens, and a corner can be stood on), and since a clicked pixel is a property of where
     the camera stands rather than of a frame, the marks stay put while the video moves and each point can
     be placed on whichever frame shows it. `VideoFrames` keeps one capture open for the window, so stepping
     one frame on is a plain read and only a jump pays for a seek (~0.3 s into a 33 GB file).

     Three things it does that are not decoration. Clicking a row in the list makes that point the one being
     placed, so a single bad point is redone on its own; the arrow keys nudge the selected point a pixel at a
     time, because at any window smaller than full size one screen pixel is more than one source pixel and
     these points are line intersections; and the pixels already on file are loaded in as a starting point,
     so re-clicking one camera does not mean re-clicking all sixteen points. Closing a window without saving
     leaves that camera's file exactly as it was.

     Everything the panel says about a point is **derived from its coordinates** — its name, whether it is
     drawn as a disc on the floor or a triangle up in the air — so adding or moving a point in
     `CALIBRATION_POINTS` needs no edit here. The diagram can be turned in quarter turns to sit the way the
     camera sees the court; that rotates the picture only, never the coordinates.

     Pixels are recorded in **1920x1080** coordinates whatever the source video is, because
     `reconstruction`'s camera matrices are written for that size. Any change to how the frame is displayed
     has to keep clicks in that space.
   - Prompts for **which set to sync in** and whether to use that set's **first serve** (default) or first
     reception, then has you find that rally in each camera → `first_reception_frames.csv`, which stores the
     two frames *and* the action's video_time.

     Prefer the serve. A rally's serve and reception share a `video_time` in the scout file, so for a normal
     rally the two choices are the same instant — but a set that opens with a service error has no reception
     in its first rally, and then "the first reception" is some later rally that has to be hunted for. This
     is not hypothetical: set 1 of the test match opens with a service error (serve at video_time 197, first
     reception at 221).

     **"First" means the first one with a `video_time`** (`scout.get_sync_action`). A scout can start the
     video clock a rally late: the Randaberg file's whole first rally has no `video_time`, its second serve
     is at 0. The anchor is then the second serve, and the prompt and the sync window both say so ("the 2nd
     serve (`a11SM-`)") — otherwise the rally found in the video would not be the one the time belongs to.
     `anchor_mismatch` and the start-frame suggestion go through the same function, so all three agree.

     Stop on the frame the serve is struck, **or up to a second before it**. Each clip is cut *forward* 5 s
     from the point the anchor maps to, and the scout file's `video_time` is only accurate to the second and
     sits anywhere up to ~1.5 s from the serve depending on when the scout entered the code, so a little
     lead is insurance — a serve that falls before its clip starts leaves the reception with no arc to fit.

     Sync per set because **a frame number only means something inside one recording**, and a match is often
     filmed as one video per set. An anchor taken in any one set still serves a whole match filmed
     continuously: the frame lookup works in *signed* elapsed video_time, so it places rallies before the
     anchor as happily as after.

     Each camera is scrubbed in a window (`calibration.sync.navigate_video`) built like the point picker:
     the picture fills it, a fixed panel beside it says which action is being looked for and what the keys
     do, and a scrubber under it covers the whole recording in one drag. **The keys this project has always
     used still work** — `j`/`f` a frame, `k`/`d` a second, `l`/`s` a minute, `q` to accept — with the
     arrow keys, the mouse wheel (shift for a second) and the scrubber as additions on top of them. The
     window opens where the previous anchor's arithmetic lands, which is near the rally when that sync was
     in this recording; the panel shows how far you have moved from it.

     The scrubber decodes only once the drag **settles** (`SCRUB_SETTLE_MS`, 120 ms), because a seek into a
     33 GB recording costs ~0.3 s and seeking per pixel of travel makes the scrubber fight back. A key
     pressed while a scrub is still pending cancels it rather than being overwritten by it a moment later.

     **Closing or cancelling either window writes nothing.** The two frames only mean anything as a pair, and
     an anchor is the one number in a run that nothing downstream can check — a wrong one fails nothing, it
     cuts every clip from the wrong minute. Accepting a frame is therefore deliberate (`q`, enter, or the
     button) and everything else leaves `first_reception_frames.csv` as it was.
2. `python run_pipeline.py` — batch-processes receptions using the CSVs from step 1. Edit the selection at the top
   (the scout file comes from `scout_filepath.csv`): `SET_NUMBER` picks one set (`None` = the whole match), and
   `FIRST_RECEPTION`/`LAST_RECEPTION` optionally narrow whatever that selected to a range. The selection is
   built by `scout.reception_numbers` from the scout file itself, so it is not tied to any particular match.

   **Receptions are numbered from 1 across the whole match and keep those numbers when a set is selected** —
   set 3 might be receptions 85-121, not 1-37. That number names the output directory and is what
   `get_reception_start_frame` looks up, so a per-set numbering would mean the same reception had two names.
   `scout.get_receptions` is the single place the numbering comes from, and everything that counts receptions
   goes through it.

   Writes per-reception output to `reception_data/receptionN/` (two annotated detection videos + a trajectory
   plot) and **patches the `.dvw` file in place**. Budget roughly a minute per reception on CPU, so a set is
   most of an hour and a match a few hours. A reception that fails — nothing detected, no trajectory to read
   events off, or a video that does not reach that far — is reported and skipped rather than ending the run,
   and every failure is listed again in the summary at the end. That matters at this scale: the receptions
   that did work are already written to the `.dvw`, so a run is never all-or-nothing.

   Asking for a reception the footage does not cover is worth recognising rather than debugging: a camera
   that stopped early, or a match split across more than one file per camera (`video_filepaths.csv` holds
   exactly one path per camera), leaves the scout file describing rallies the video does not have.
   `video.create_video_chunk` raises a clear error naming the video's length in that case, instead of
   silently writing an empty clip that fails later as an unopenable video.
3. `python run_attack_pipeline.py` — the attack pipeline, end to end: which attacks the run covers and
   where in each video each of them is, the clip each one needs cut out of both match videos, the ball
   detector run over both clips, the two cameras' candidates paired and triangulated into 3D points, the
   ballistic fit, the three coordinates read off the fitted flights, and the write-back to the `.dvw` —
   which also adds a set line in front of every attack the scout left without one.
   Selected the same way as receptions (`SET_NUMBER` plus an optional `FIRST_ATTACK`/`LAST_ATTACK` range
   on top of it), with `DETECT` and `ANNOTATE` to stop after the clips or skip the debug videos, and
   `WRITE_TO_SCOUT_FILE` to measure everything without touching the scout file. The selection step reads
   the scout file and the two config CSVs and no video frames, so it takes about a second for a whole
   match — worth eyeballing before any of it is handed to the rest.

   Measured at ~50 s per attack end to end (~15 s of it clip cutting, where seeking into a 33 GB source
   dominates), so a set is about an hour and a match around four hours.

   The clips are named after the attack (`temporary_videos/attack6_sideline.mp4`) rather than overwritten
   as the reception ones are, so a run's whole selection survives to be watched. That is what the window
   constants are judged on. They are ~33 MB per attack across the two cameras, in gitignored scratch.
   Per-attack output goes to `attack_data/attackN/`:

   - `{sideline,baseline}_boxes.mp4` — the annotated detection videos, with the chosen box in green and
     the rejected candidates in grey, captioned with the clip frame number, the source video's frame
     number in brackets, and either the match's reprojection error and ray separation or the top
     confidence of an unmatched frame.
   - the three measured coordinates, appended to `attack_data/attack_points.csv` (see below).
   - `trajectory.html` — rotatable in a browser: the fitted flights as a joined blue path, and underneath
     in ochre the triangulated points they were fitted to, as **markers with no line** (they are whatever
     survived matching, gaps and all, and joining them would draw straight segments across the frames no
     pairing survived). Scattered ochre means the problem is upstream in detection or matching; tight
     ochre under a wandering blue path means it is the fit. No PNG — a scatter in a fixed 3D projection is
     what being able to rotate exists for.

     The page is **trimmed to the attack**: only frames from the start of the first fitted flight to where
     the attack finished are drawn, both series alike. A clip is four seconds of a rally and usually holds
     the dig that follows and the next team's set as well, and those are not what the figure is for;
     leaving the raw points untrimmed would draw the discarded flights straight back on as ochre markers.
     The three contacts are marked and labelled **Set**, **Attack** and **End**.

   Quality on the first attacks, as a baseline to compare against. Detection: the sideline camera finds a
   candidate in ~90% of frames and the baseline in ~78%, both at median top confidence ~0.8, longest blind
   run 13-15 frames (a quarter of a second, comfortably inside what the ballistic bridging covers).
   **The baseline camera is consistently the weaker of the two on attacks**, which is expected — the
   contact is at the net with the block in front of it — and it is the pair that matters: ~76% of frames
   have a candidate in both, the ceiling on how much can be triangulated. Matching then keeps 134-168 of
   239 frames at a reprojection median of 15-22 px against the 40 px gate (0.15-0.19 m of ray separation),
   which is the same shape as receptions. The resulting points cross the net, sit inside the court in x,
   and step 0.10-0.14 m per frame at the median with ~4% of steps over 1 m — the usual few mismatched
   frames, and what the ballistic fit's outlier rejection exists for.

   The fit (`attacks.USE_BALLISTIC_FIT`, independent of `pipeline.USE_BALLISTIC_FIT`) then finds 3-8
   flights per clip. Over attacks 1-4, **pinning the acceleration on short spans gains 35 and 43 frames on
   two of them and changes the other two not at all**, and loses nothing anywhere; the points no flight
   explains drop from 8 and 7 to zero. The case it exists for is visible in attack 3: a six-point segment
   at frames 85-91 running from `y = 0.00` to `y = 1.22` at **13.1 m/s** — a ball crossing the net at
   spike speed, which every earlier version of the fit discarded for having too few points. One cost worth
   knowing: when a span of 8-20 frames fails the free fit, the pinned fit is now tried *before* splitting,
   which on attack 3 replaced a free 11-point fit (rms 0.026) with a pinned 12-point one (rms 0.104). Both
   are far inside `INLIER_TOLERANCE`, and the trade buys two frames and avoids splitting a short flight
   into a fake contact — but it is a real trade, not a free win.

   **This is a separate entry point and a separate module on purpose.** `zoneout.attacks` imports nothing
   from `zoneout.pipeline` and vice versa, so attacks can be left unrun, or the file deleted, and
   receptions carry on unchanged. What the two share is the machinery underneath — `scout`, `timing`, and
   later the reconstruction, which is the same problem for both.

## The serve dataset

`datasets/` holds datasets built while the pipelines run. The first is **`datasets/serves.csv`**, one row
per serve, written by the reception pipeline (`pipeline.WRITE_SERVE_DATASET`, needs the ballistic fit)
through `zoneout/serve_dataset.py`. It is one flat file on purpose: a player's serves are a
`groupby(['team', 'player_number'])` away, and nothing has to be flattened before analysis.

Columns: `match` (scout file name), `set_number`, `reception_number`, `video_time`, `team`,
`player_number`, `player_name` from the scout file's serve line; `contact_observed`; `start_*` and `end_*` in metres;
`duration_s`; `average_speed` (path length over duration); `v0_*` (m/s) and `a_*` (m/s²). The last three
groups are the whole parabola, `p(t) = start + v0·t + ½·a·t²` for `0 ≤ t ≤ duration_s`, with
`p(duration_s) = end` exactly.

- **Every row has the server at the `y < 0` end**, the ball travelling toward `+y` — the `.dvw`'s own
  action-relative convention (`dv_grid.orient_for_action`), so serves from both ends compare directly.
- **The flight is `events.find_serve_flight`**: the last flight starting more than `NET_ENTRY_MARGIN`
  back on the serving side before the reception (which steps over a separately fitted toss). It starts at
  its contact frame and ends at the reception's, evaluated on the *serve's* model so start, end and
  parameters are one parabola. It is refused, and the reason printed, when there is no reception, when
  more than one flight lies between serve and reception (a net-cord touch or a split serve), or when the
  gap before the reception could not be bridged.
- **`start_*` is the server's contact, and it is usually extrapolated.** The ball is rarely tracked
  near the server: replayed over the 113 cached receptions, 100 give a row and only 20 have a fitted toss
  whose model hands over to the serve's (`events._hands_over`) — the only case where the contact is
  observed. In the other 80 the serve was first seen 5-15 frames into its flight (in 14 the clip started
  after the contact), and `serve_dataset` runs the serve's parabola back to `SERVE_CONTACT_DEPTH`, the
  baseline (`contact_observed = False`). The observed contacts sit 8.2-9.8 m from the net, median 9.0.
  Checked by dropping the first 5-15 frames of each observed serve, refitting and extrapolating back:
  the extrapolation itself is good to ~0.05 m across and in height, and the error, median 0.28 m and
  at most 1.0 m, is almost all depth — the fixed baseline against where that server actually stood. The
  unobserved contacts come out 2.6-3.7 m up, the same heights as the observed ones, which the depth
  assumption could not have forced.
- An observed contact nearer than `events.MIN_SERVE_CONTACT_DEPTH` (7 m) to the net is refused: that is
  the serve split in two by the fit (reception 103, 1.1 m), not the server's hand.
- `python datasets/serve_figures.py` picks a dataset (file dialog) and a team (numbered prompt) and writes
  every serving player's figure (`figures.save_player_serves_html`) to `serve_figures/<team>/` beside the
  dataset, sharing one `plotly.min.js` there instead of embedding it in each page — so the folder moves as
  a whole. Gitignored; it is regenerable.
- Only received serves are in it, since the pipeline runs over receptions — no service errors.
- The serve is the last serve line above the reception in the scout file, checked to be in the same set
  and by the other team. Rows merge by `(match, reception_number)`, so a re-run replaces its row.
- Times use `ballistic.FPS` (60), so durations and speeds carry the same ~0.1 % as the fit itself.

## The serve-and-reception dataset

**`datasets/receptions.csv`**, one row per reception, written by the reception pipeline
(`pipeline.WRITE_RECEPTION_DATASET`, needs the ballistic fit) through `zoneout/reception_dataset.py`. Where
`serves.csv` keeps each serve's whole parabola, this one is the rally from the server's hand to the end of
the pass, reduced to what a serve/reception analysis asks for. Same orientation (server at `y < 0`), same merge key.

Columns: `match`, `reception_number` (the merge key); from the scout file `serving_team`,
`receiving_team`, `serving_player`, `receiving_player` (names), `serve_type` (`Q` → `spin`, `M` → `float`,
any other letter as it stands), `reception_grade`, `receiving_team_won` (`point_won_by` is the receiving
team), `receiving_rotation` (the receiving setter's position, 1-6); from the fit, in metres,
`serve_start_*`, `reception_*`, `serve_landing_x/y`, `serve_hit_net`, `serve_average_speed` (m/s),
`reception_end_*`, `reception_apex_z`. **These columns are the ones Neo specified — do not add others to this file**
(an earlier version carried observed/extrapolated flags and a net clearance, and they were removed).

- **A row is written for every reception that gets this far, with blanks for what could not be measured**,
  and each blank is explained in the run output. The scout-file half is always there.
- **The serve is `events.find_serve_path`**: `find_serve_flight`'s serve, but flights between it and the
  reception are *followed* when each hands over to the next, instead of refusing the serve. That is what
  makes `serve_hit_net` answerable: a net-cord touch is a contact, so it is a hand-over within
  `NET_ENTRY_MARGIN` of the net. A touch that barely deflects the ball needs no second parabola and cannot
  be seen that way. No serve in the Uppsala match changed flight at the net; the lowest the ball's centre
  crossed the tape was 0.05 m (reception 94, a jump serve), within reconstruction error of a graze.
  `serve_start_*` and `serve_average_speed` match `serves.csv` exactly on every reception both have a
  row for — `serve_start_*` is extrapolated back to the baseline the same way when the contact was not seen.
- **The landing** is the serve's last flight carried on past the reception until the ball's centre is
  `BALL_RADIUS` off the floor — where it would have come down unplayed.
- **The reception end and the pass height are `events.find_pass`.** `reception_end_*` is the **end of the
  reception trajectory**: the pass's own model at the contact that played it next (the seam with the
  following flight) — not the start of the next flight, which is what an earlier version called the set.
  The two agree to within the seam tolerance (median 0.14 m, max 0.65 m on the cached receptions); what
  the choice settles is that the point belongs to the pass. Normally that is the setter's hands, but it is
  written whatever came next and on either side of the net — **overpasses included**, and with no
  scout-file gate (an earlier version only located it when the scout file recorded a set). It is refused
  when the pass → next-flight gap cannot be bridged (reception 54: 108 frames, so where the pass ended is
  unknown), and when the pass is still *rising* where the next flight starts — the ball is played on its
  way down, so that is the fit splitting the pass (reception 17).
- **A high pass leaves the top of the picture**, is fitted as two flights with its apex missing, and the
  second one is the pass coming down, not the next touch. `_set_cut_off_above` — the attack pipeline's test
  for high sets — pairs the halves; the pass then ends on the falling half, and the height comes from one
  parabola in `z` fitted through both halves (3 of the Uppsala receptions, at 5.5-5.8 m; reception 89
  gave 5.47 m this way from its cached run and 5.49 m observed directly when re-detected). A single-flight
  pass takes the top of its own parabola (`Pass.apex_source` says which case, for debugging). A pass still
  rising at the last frame it is known at, or one that re-enters from above with no rising half to pair it
  with, has no height and no end.
- **The clip is 7 s instead of 5 when this is on** (`pipeline.RECEPTION_DATASET_CLIP_DURATION`), because
  the pass ends 1.4-2.0 s after the reception and the reception lands 1.7-4.7 s into a clip: a 5 s clip
  ended before the pass did on 45 of the 108 cached receptions. It costs ~40% more detection per reception.

Replayed over the 108 reproducible cached Uppsala receptions (with their 5 s clips, so pass ends are
under-counted — 50 clips end before the pass does): 109 serve paths, 84 pass heights (median 4.9 m), 54
reception ends.

## The three coordinates an attack is measured as

`events.find_attack_points` reads them off the first three fitted flights, which is what an attack clip
holds: the tail of the previous rally, the set, and the attack.

**The attack is found by what it does, not by where it sits: it is the flight that carries the ball across
the net.** Everything else is positioned relative to it — if the attack is flight *k*, the set is the contact
starting flight *k-1* and the end is where flight *k* finished.

| point | what it is | where it comes from |
|---|---|---|
| **set** | the setter's touch | the contact starting the flight before the attack, on that flight's model |
| **attack** | where the attacker struck it | where the set flight ended, on the set's model (the seam with the attack flight, or the set's last observed frame); the attack flight's own start only when there is no set flight |
| **end** | where the attack finished | the seam with the flight after it, on the attack's model |

The side of the net is read at each flight's **contact frame, not its first observed frame**, and that
decides real cases. An attack is struck at the net and the detector routinely loses the ball for the frames
it spends there, so the flight's observations often begin once it is already past. Attack 3's spike is
observed from `y = +0.61` to `+1.22` and never spans zero, while the flight before it ends at `y = -0.59`
and the fitted model puts the contact at `y = -0.5`. Testing observed frames alone called that no crossing
and lost the attack; testing from the contact frame recovers it, and four others.

An earlier version took the first three flights, assuming a clip holds the previous rally's tail, the set
and the attack in that order. That is right more often than not but not reliably — measured over the same
clips it held 16 times, while in 12 more the attack was there one or two flights earlier because the clip
started too late to catch the tail. The crossing rule is a **strict superset**: wherever flight 3 crosses,
it *is* the crossing flight (15 of 15), and it also finds the shifted ones. It is unambiguous in practice —
no clip measured had more than one net-crossing flight, so no speed threshold is needed to disambiguate;
if more than one ever appears the earliest is taken and a warning says so.

Contacts are **seams**, not segment starts — the contact happens inside the gap neither flight observed, and
by the time observations resume the ball has moved. `_contact_frame` is shared with `find_reception` for
exactly that reason. Each point is one fitted parabola evaluated there and nothing blended, following the
same rule the reception uses: a contact is described by the flight that *leaves* it, because that is the one
that says where the ball went. The end point is the exception, being the end of the attack's own flight
rather than the start of anything, so it is the attack's model that is evaluated.

**The attack point is the end of the set, not the start of the crossing flight**, because the crossing flight
does not always begin at the attacker's hand — a block touch is a contact too. On attack 59 the spike is
struck 2.1 m off the net and travels at ~18 m/s for five frames, too few to fit as a flight, before the block
deflects it at the net; the flight that crosses starts there, at `y = -0.24`. The set's descent is long, well
observed, and ends at the attacker's hand whatever follows. Over 42 cached attacks this moves the point a median
of 0.22 m (90th percentile 0.71 m), and attack 59 by 1.97 m. One case it cannot fix: if the spike itself were
ever fitted as a short flight that does not cross, *it* would be the flight before the crossing one, and its
end is the block touch again.

**A high set that goes above the top of a camera's picture is located where it started rising.** Such a set
is fitted as two flights with the apex missing between them, and the set's contact was being read where the
ball came back into view — attack 59's at 4.8 m. `events._set_cut_off_above` recognises it when all three of
these hold for the flight before the set flight (the rising half) and the set flight (the falling half):
the rising half is still going up at its last frame and the falling half already coming down at its first;
both of those positions project within `TOP_EDGE_MARGIN` (100 px) of the top of the *same* camera's picture
(44 and 48 px on attack 59's baseline camera); and the rising half's model carried across the gap lands within
`CUT_OFF_SET_MAX_DISTANCE` (1.0 m) horizontally of where the falling half is first seen (0.55 m), horizontal
only because two short halves disagree by 1.7 m in height after a second of extrapolation. The set is then
located at the contact that started the rising half, and `AttackPoints.set_index` says which flight that was
so the figure draws both halves. The first condition is what discriminates: of 30 cached attacks with a flight
before the set, 59 is the only one where that flight rises at its end.

**When the set comes back into view falling and there is no rising half, there is no set location**
(`_set_entered_from_above`): the set flight starts falling, within the top-edge margin of a camera, with
unobserved frames before it. Its contact would otherwise be written 4.75-6.04 m in the air. Attacks 10, 21, 31,
38/39 (one rally clipped twice) and 44 of the Örkelljunga match, and 60 of Södermalm (a `PR`, never set anyway).
`set_point` is None, a warning says why, and the set line is still added — just without a coordinate.

Both checks project through the calibration (`reconstruction.cameras_near_top_edge`); the attack and end points
do not. With each match's own calibration, enabling them changes exactly those eight attacks and leaves every
set, attack and end point of the other 36 identical.

**The end point does not use `_contact_frame`, and that difference is not cosmetic.** `_contact_frame` falls
back to the *next* segment's first observed frame when the two cannot be bridged, which is right when asking
where that flight began and a disaster when evaluating the attack's parabola there. On attack 1 the attack
ends at frame 142 and the next flight is not seen until 208; the fallback put the attack's model 66 frames
past its last observation, at `y = 17.9` and **13 m below the floor**. So `find_attack_points` calls
`find_seam` directly and, when there is no seam, ends the attack at its own last observation and says so in
a warning — the ball really did end somewhere inside that gap and nothing knows where.

`find_attack_points` returns **None when no flight crosses the net** — the attack was never reconstructed,
which is a detection or matching problem, so the clip and the annotated videos are where to look. That is a
different failure from a measurement with caveats, and it is reported as one rather than producing a row.

`warnings` is not an error channel: where a row exists the coordinates are returned. A warning means one of
them rests on less than the others. The conditions: the attack being the first flight in the clip (so its
contact is where observation began, not a seam), nothing fitted before the set (so the set location is the
setting flight's first observed frame — `set_point` is None entirely when the attack is flight 0), an
unbridged end, an end at or before its own contact, more than one crossing flight, any point landing
outside `court.BALL_VOLUME_*`, a set located on its rising half after going above a picture, and a set with
no location because it came back into view already falling.

Measured over the 67 cached attacks that survive the grade filter: **35 are measured and 31 have no
net-crossing flight at all**. Of the 35, **18 are completely clean**; 14 warn that the set is the setting
flight's first observed frame rather than a contact, and 3 that the attack is the clip's first flight and so
has no set to locate. The attack turns up at flight index 2 eighteen times, index 1 fourteen times and index
0 three times — which is exactly why counting flights was not enough. Attack-flight speeds run 4.0 to
25.5 m/s with a median of 12.5, the right shape for struck balls.

The 31 with no crossing flight are the honest remaining problem, and it is upstream: the attack was not
reconstructed, not mis-selected.

Rows go to **`attack_data/attack_points.csv`**, one per attack, carrying the attack's identity (number, set,
`video_time`, `scout_line`, code, team, player), the scout's own `start_zone`/`end_zone`/`end_subzone`, the
three points with their frames, `crossed_net`, and the warnings. The scouted zones ride alongside the
measurements on purpose: they are the human reading of the same event, so the file is checkable without
going back to the `.dvw`.

## Writing attacks back to the `.dvw`

`attacks.write_to_scout_file` patches the attacker's contact in as the start coordinate and where the
attack finished as the end, on the attack's own line, with the mid left as `-1-1`. **The scouted `code` is
never rewritten** — the zone and cone letters in it are the human's reading of the rally, and the computed
zones stay in the CSV for checking rather than being written over a scout's own work. The same save adds
the missing sets — see "Adding sets in front of attacks" below.

### Which end of the court a coordinate is written from — settled, by measurement

This was the open question that gated the whole step (`ATTACK_PLAN.md` §5), and it could not be answered
from either processed match: openvolley warns that coordinates "might not appear in the dvw file in any
particular orientation", and **every coordinate in both files was written by this pipeline**, so checking
against them would only have been the code agreeing with itself.

`&32135954_Hom15 Örkelljunga vs Sollentuna .dvw` answered it. It is an Elitserien file scouted by hand,
untouched by this pipeline, carrying 201 lines with coordinates. **All 201 start in the bottom half of the
grid and end in the top, for both teams alike** — 104 attacks, 31 serves, 27 receptions, 39 digs, without
one exception. So the convention is strictly action-relative: whoever is acting is drawn at the bottom and
the ball always travels up the picture. That is what `add_serve_direction` has always done for serves on a
guess `NOTES.md` recorded as unverified; it was right, and `dv_grid.orient_for_action` now holds it in one
documented place for both writers instead of as inline sign flips in each.

The same file settles the zone layout. Converting the scout's *own* clicked index back to a zone and
comparing it against the zone they typed on the same line agrees 81% on start zones and 93% on end zones —
a human against themselves, so never 100%. What matters is the shape: the confusion matrix is diagonal and
13 of the 15 end-zone misses are one step along *depth* in the same direction (typed as a back-row zone,
clicked just in front of the 6 m line). **A mirrored layout would show as 1↔5 or 4↔2 confusions and there
are none.** `tools/check_orientation.py` is that whole comparison, re-runnable on any scout file, and it
refuses to draw conclusions from a file whose only coordinates are on serves and receptions.

Replayed over the 35 cached attacks, the written start zone agrees with the scout 74% of the time — in line
with the 81% a human scores against their own clicks. The misses repeat the *same* pattern: 8→3 and 9→2 are
back-row attacks, where the scout types the zone the attacker took off from while the pipeline measures the
contact at the net. That agreement of failure modes is the strongest evidence the convention is right.

**The side is decided by whichever of the two points is further from the net**, not by the contact. An
attack is struck at the net, so the contact sits routinely within a quarter of a metre of `y = 0` — three of
the 35 do, the closest at 0.10 m — and at that range an error smaller than the pipeline's own tolerance
flips the sign. A flipped sign does not give a slightly wrong coordinate, it writes the whole attack in from
the opposite end of the court. The two points always disagree in sign (that is *why* `find_attack_points`
chose that flight), so taking the more distant one costs nothing and removes the only input the answer was
sensitive to.

### `dv_grid.py` — the grid, in one place

Pure arithmetic, no I/O. Both writers go through it, so they cannot drift apart on the format.
`index_to_xy` round-trips **exactly** against `datavolley.helpers.dv_index2xy`, pydatavolley's own inverse
of the same grid (max difference 1.8e-15 m in x, 1.1e-05 m in y, the latter purely pydatavolley's rounded
constants).

That round trip caught a real bug. **`scout.coords_to_dvindex` was up to one cell out**: it laid 80 rows
over the 18 m length where openvolley's grid has 81, and truncated toward zero instead of binning to the
cell edges. Measured over 40 000 points spread across the court, the old and corrected indices differ
*everywhere* — the corrected one sits a uniform **+0.1125 m further along x** and **0 to +0.222 m further
along y**, the y error growing toward the far baseline. `coords_to_dvindex` is now a thin wrapper over
`dv_grid`, so the serve/reception write-back is corrected too. Small, but an attack's end coordinate is
usually near a line and in-or-out is exactly the question a 0.22 m bias sits on.

### `dvw_edit.py` — the safe writer

A scout file is the only thing in this project that is not reproducible: videos can be re-cut and
trajectories re-fitted, but a `.dvw` is hours of a human's work and the pipeline edits it in place. What the
writer enforces, each rule earning its place:

- **Address by line number, verify by code.** `scout.get_plays` supplies `scout_line` and checks it against
  the section's own line count; `set_coordinates` and `insert_before` then re-read the `code` at that line
  and **raise** unless it is the one expected. A stale index is impossible to write through.
- **Line numbers mean the file as opened.** Every edit and insertion in a run is addressed against the file
  as read and all are applied together in `save`, since each insertion shifts every line below it.
- **Touch fields 4, 5 and 6, or insert whole lines — nothing else**, rejoining every other byte verbatim.
  No existing code is ever rewritten. An inserted line takes the line ending of the line below it.
- **Preserve encoding and line endings** — `cp1252`, opened with `newline=''` so a CRLF file stays CRLF.
  Text-mode round-tripping would silently rewrite every line and show up as a total diff.
- **One save per run.** Edits accumulate in memory; a crash mid-run leaves the file exactly as it was.
  That is the opposite trade from `attack_points.csv`, which merges per attack precisely so a long run is
  never all-or-nothing — the difference is that the CSV is this pipeline's output and the `.dvw` is not.
- **Back up before the first write, never overwrite the backup**, and **replace atomically** (temp file,
  fsync, `os.replace`).
- **Do not clobber a human's coordinates.** The backup is what makes this exact rather than a guess: it is
  the pristine file even on the tenth run, so a previous run's own output is distinguishable from a scout's
  clicks. Without it the choice would be between refusing to re-run an attack after a fix and quietly wiping
  a scout's work; neither is acceptable. `attacks.OVERWRITE_SCOUTED_COORDINATES` overrides. This is not
  hypothetical — the Elitserien file above carries 104 hand-clicked attack positions.

  **Lines are matched to the backup by content with the coordinates blanked, not by line number** — once a
  run has inserted sets, every line below each one has moved. `DvwFile._compare_with_backup` aligns the two
  with `difflib`: a matched line carries the backup's coordinate status, a line with no counterpart at all
  is one this pipeline inserted (`added_by_pipeline`), and a line standing where a *different* line stood —
  which only a human editing the file produces — is treated as the human's. Checked on the Elitserien file
  with its sets stripped and re-added: all 104 hand-clicked attacks had moved, and every one was still
  protected.
- **Verify after saving** by re-parsing with pydatavolley and asserting every original row is unchanged
  outside the coordinate columns, and every inserted row reads back as the code inserted. This caught
  something on its first run: `match_id` is a fresh `uuid4` pydatavolley mints per call, not read from the
  file at all, so it is in `UNSTABLE_COLUMNS` rather than the check being loosened. `scout_line` is there
  too, since an insertion moves it by design. One column is exempt on the row directly below an insertion
  and nowhere else: pydatavolley derives `attack_phase` from the rows *above* an attack, and only gives an
  attack one when a set precedes it — adding the set is exactly what changes it.

Verified on a copy of the real match: 35 attacks written, **exactly 35 lines differ, all of skill `Attack`,
all in fields 4/5/6 only**, file length unchanged, and re-parsing clean. Also checked: a stale line number
raises, a re-run replaces its own earlier output, and a human-scouted line is skipped and reported.

An attack whose contact or end falls outside `court.BALL_VOLUME_*` is **not written** and is reported
instead. That gate is needed because `dv_grid.xy_to_index` *clamps* an off-grid point rather than refusing
it, so a reconstruction that put the ball thirty metres away would otherwise be written as a confident
coordinate on the edge of the court. A `warnings` entry on `AttackPoints` is deliberately **not** a reason
to skip: it means one coordinate rests on less evidence than the others, which the scout file has no way to
say, but it is still the best measurement of that attack — and 17 of the 35 carry one.

**Attacks graded `/` (blocked) or `!` (recycled) are skipped** — `attacks.SKIP_EVALUATIONS`. Neither
completes the attack it was scouted as: a blocked ball is stopped at the net and a recycled one comes back
off the block, so the flight after the attacker's contact is a rebound and the end coordinate would be a
fiction. Skipping rather than measuring-and-flagging is the right trade because there is nothing to measure
and each costs ~50 s of detection — 48 of this match's 292 attacks carry one of the two grades. **The
numbering is untouched**: it comes from `scout.action_numbers` over every attack in the file, so a filtered
run is a subset of the same sequence, not a renumbering, and `attack_data/attack12/` keeps meaning the same
rally.

### Adding sets in front of attacks

A set is its own line, directly above the attack it fed — `*13ET+K1F;;;;;;;16.05.22;1;6;2;1;1078;;…` — and
Neo's files often leave it out: the Södermalm match has 8 sets for 154 attacks. `write_to_scout_file` adds
one in front of every attack in the run's selection where `set_codes.decide` says one belongs. The rules,
all in `set_codes.py` with what each was checked against:

- **Already there** means the line directly above is a set **by the same team**. Then nothing is added.
- **`PR`, `PP` and `P2` are never set** (attack on an overpass, setter tip, second-hand attack).
- **The setter is the player in the setter's zone**, read off the attack's own line: field 9 / 10 is the
  home / visiting setter's zone and fields 14-19 / 20-25 the players in zones 1-6. When that player *is*
  the attacker no set is added and the run reports it — someone else set the ball and the line cannot say
  who.
- **The tempo is the attack's** (`AT` → `ET`) and **the grade is always `+`**.
- **A setter call is only ever added on a sideout attack** — the receiving team's first swing of the rally.
  `set_codes.sideout_reception` walks back from the attack to the rally's serve and says so: the rally has
  a reception, it is by the attacking team, and no attack by either team came between. That is the scouts'
  own practice rather than a rule of thumb — over the 18 hand-scouted Elitserien files, 689 of the 712
  quick calls (97%) and **all 71 shifted calls** sit on an attack this test calls a sideout. Measured
  against those files, gating on it removes 99 of the 192 calls the pipeline would otherwise have put on a
  set the scout left uncalled, and costs 6 of the 283 it put on a set the scout did call.
- **`X1`/`X2`/`X7` then get the setter call `K1`/`K2`/`K7`** — only if the file's `[3SETTERCALL]` table
  defines it.
- **A set made within 2 m of a sideline gets the shifted call instead**, `KM` out by zone 2 or `KP` out by
  zone 4 (`attacks.SHIFTED_SET_MARGIN`, `set_codes.SHIFTED_SETTER_CALLS`), whatever the attack combination
  is — a shifted setter is a fact about where the setter stood, not about who they set to. It needs a
  measured set location, a sideout, and a reception graded `#`, `+` or `!`; all three are what the
  hand-scouted files show, where every one of the 71 shifted calls is on a sideout and 70 of them follow a
  pass graded that well. The distance is read off the set point turned into the setting team's own frame
  (`_oriented_set_point`), so `x` runs from their zone 4 sideline at 0 to their zone 2 sideline at 9 and
  nothing here needs to know about teams or ends. 2 m is comfortably outside a setter's station: the three
  hand-scouted sets on this disk that carry a coordinate sit at `x` = 5.46, 6.36 and 6.47, while attack 1
  of the quarter final measures 7.40 and was genuinely pulled wide. **The shifted call wins over a quick
  one** when both apply, because `X1` in the attack code already says "front quick" and nothing else on the
  line records that the setter was out at the pin.

  This is the one rule that needs the reconstruction, so an unmeasured attack can still get a quick call
  but never a shifted one. Note also that a set *this pipeline added on an earlier run* counts as already
  there, and only its location is refreshed — its code, call and all, is never rewritten, so a file already
  written under the old rules keeps the calls it was given unless it is restored from `.dvw.bak` first.
- **The target attack** (`F`/`C`/`B`/`P`/`S`) is the attack combination's own, from the ninth field of the
  file's `[3ATTACKCOMBINATION]` table — `X5` → `F`, `X1` → `C`, `XP` → `P`. With no call, `~~` holds the
  call's place (`*13ET+~~F`). An attack with no combination code, or a table entry of `-`, gets no letter.
  Across the hand-scouted files, 2,145 of the 2,153 sets carrying a target agree with the table.
- **Everything after the code is copied from the attack's line**, so **the set's clock time and `video_time`
  are always the attack's own**, as are the set, setter zones, video file and rotation. The attack's `video_time` marks the setter's touch in these files, and
  `&svk-ork_test.dvw` agrees — 232 of its 235 sets share their attack's `video_time` and 212 its clock time.
  Fields 1-3 stay empty, as on every scouted set.
- **The set location**, when `find_attack_points` found one, is written as `start;-1-1;-1-1` — how the three
  hand-scouted sets with coordinates carry it — with the setting team at the bottom. The half is the
  attack's (`_attacking_half`), not read off the set point, because a setter stands at the net where the sign
  of `y` is unreliable. No location when the attack itself is not written, or the set point is outside
  `court.BALL_VOLUME_*`.

**Blocked and recycled attacks get their set too.** They are skipped for measuring because the flight after
the contact is a rebound, but the set happened all the same, so `scouted_attacks` is the selection without
the grade filter. The scout file is only written with `DETECT` on; a detection-off run is a check of the
selection and should not edit it.

**Re-running never adds a second set** — the one added last time is directly above the attack, so it counts
as already there. A set *this pipeline* added (`added_by_pipeline`) does get its location refreshed from the
new measurement; a scout's own set is never relocated. A run with no measurement leaves an earlier location
alone.

Checked three ways. Stripping every same-team set in front of an attack out of the hand-scouted Örkelljunga
file and letting the pipeline put them back: all 192 compared came back with the scout's tempo and every
copied field identical except clock time and `video_time` (that scout timed sets a second or so before
the attack; this pipeline deliberately does not); the player matched 171 times (89%), the misses
being rallies where someone other than the rotation's setter set the ball — which the rule cannot know.
On a copy of `&svk-ork_q5.dvw` with the 35 cached measurements: 170 sets added (16 located), and the diff is
exactly those 170 inserted set lines plus 35 attack lines with changed coordinates. A second run leaves the
file byte-identical, and a run with moved set points changes exactly the 16 located set lines.

`save_attack_points` **merges by attack number** rather than overwriting: a run covering attacks 1-10 and a
later one covering 11-20 leave twenty rows, and re-running an attack after a fix replaces its row instead of
duplicating it. Same reason the reception pipeline patches the `.dvw` as it goes — a batch that takes hours
is never all-or-nothing.

## Selecting and timing a scouted action

Both pipelines pick their actions and locate them in the videos through the same three functions in
`scout`, so neither can end up numbering or timing things by its own rules:

- `get_actions(dvw, skill)` — one row per action of that skill, in play order. Its position in the frame
  *is* the action number minus one. `get_receptions` / `get_attacks` are wrappers.
- `action_numbers(dvw, skill, set_number=None)` — **1-based and counted across the whole match**, so they
  do not restart with each set: set 3 might be attacks 147-210. They are counted *per skill*, though —
  attack 7 and reception 7 are unrelated rallies, which is why the two pipelines keep separate selections
  rather than sharing one range.
- `get_action_start_frame(...)` / `timing.CameraTiming.frames_for_action(...)` — the anchor plus elapsed
  `video_time` at that camera's rate.

`scout.get_plays` adds a **`scout_line`** column: the 0-based line number within the file's `[3SCOUT]`
section. That is how the write-back will find the line to patch, and `video_time` cannot do the job —
within a rally the set, attack, block and dig routinely share one second, 8 of this match's 292 attacks
share one with another attack, and it is not even monotonic down the file. `get_plays` checks that
pydatavolley returned exactly as many rows as the section has lines and raises if not, so a row's position
identifying its line is verified rather than assumed. Checked over the whole match: all 292 attacks address
the right line.

`zoneout/timing.py` owns the anchor: reading `first_reception_frames.csv`, falling back to the match's
first reception for a config file written before the anchor was selectable, reading both cameras' true
frame rates, and converting a `video_time` to a frame in each. It exists so that both pipelines resolve the
anchor identically — it is the one number in a run that nothing downstream can check.

### The clip window is not the same for a serve and an attack

`video.create_video_chunk(path, name, frame, lead_seconds=0.0, duration_seconds=5.0)` cuts a window in the
video's **own** frames, so the same window in seconds comes out of two cameras recording at slightly
different rates — which is what lets the two clips be paired frame for frame afterwards. It returns a
`video.Clip` carrying `start_frame`, so a frame inside a clip can be turned back into one in the match
video; nothing else can relate a detection to the scout file.

The default of no lead at all is the **reception** case and is deliberate: a serve starts the rally and the
scout's `video_time` lands at or before it, so cutting straight forward covers the flight. Receptions still
get exactly the window they always did (frame *n*, `int(5 × fps)` = 299 frames), verified against the old
arithmetic.

An **attack** is coded in the middle of a rally, so the window has to be moved back over it.
`attacks.ATTACK_CLIP_LEAD` (0.5 s) and `ATTACK_CLIP_DURATION` (3.5 s) hold that, and the lead is only
correct because of a **scouting convention, not anything the file says**: these files are synced by hand
with the attack's `video_time` set at the moment the *setter* touches the ball, so the attack contact
follows within about a second. A file scouted to a different convention needs a different lead, and there
is nothing in the `.dvw` that would reveal that — the clips would just quietly miss the flight.

A lead that would start before frame 0 **raises** rather than clamping. Clamping would move one camera's
window and not the other's, and the two are paired frame by frame, so a clip that quietly starts somewhere
else is worse than one that fails.

## Architecture

Per-reception processing (`zoneout.pipeline.get_data_from_reception`) chains together:

1. **Frame lookup** (`scout.get_reception_start_frame`): reads the `.dvw` via `pydatavolley` to get each
   reception's `video_time`, then combines that with the manually-recorded first-reception frame numbers
   (`first_reception_frames.csv`) to compute the start frame for reception N in each camera.

   The anchor is whatever the sync step recorded — a set's first serve or first reception, its video_time,
   and the frame each camera shows it at (`config.read_sync_anchor`). A config file written before the
   anchor was selectable has no video_time column; those always meant the match's first reception, and the
   pipeline reads them as exactly that rather than demanding a re-sync.

   The conversion rate is **per camera** and comes from `scout.video_time_frame_rates`, fed by each video's
   own frame rate (`video.get_frame_rate`). Cameras sold as 60 fps record at 59.94 and no two agree exactly;
   since every reception is extrapolated from the one anchor, the difference between the two cameras
   accumulates with distance from it — about 2 frames across this match, which is 0.6 m of travel for a
   served ball being triangulated frame by frame between the two views. Moving the anchor changes nothing
   else: re-anchoring on set 3's first serve reproduces the old frames exactly for the sets around it, and
   differs by at most 1 frame anywhere in the match, which is rounding.

   **`scout.DVW_CLOCK_FPS` (60.0) is a separate thing from either camera's rate.** It is the rate
   DataVolley's `video_time` seconds advance at, which is not the cameras' 59.94 — the seconds appear to have
   been counted 60 frames at a time on 59.94 fps footage, so the errors largely cancel. Largely, not
   exactly: measuring where the ball really is inside a wide (16 s) window puts the rally *earlier* than 60.0
   predicts, by 53 frames at reception 100 and 124-169 at reception 120, i.e. a true rate around
   **59.96-59.99**. Neither 60.0 nor 59.94 is right, and 59.94 is the worse of the two here (it would be
   ~300 frames early at reception 120 where 60.0 is ~150 late).

   Syncing per set is what makes this survivable in practice — the error only accumulates from the anchor,
   so an anchor inside the set being processed keeps it to well under a second. It is still a **known open
   problem for a whole match run off one anchor**, and the reason receptions late in a long match
   reconstruct into nonsense:
   by the third set the error pushes the serve outside the 5 s clip entirely — at reception 120 the serve is
   at wide-window offset 356 while the clip starts at 480. Early sets are unaffected (under a frame per ten
   seconds). The fix is not a better constant, which would just be this match's number: it is a **second sync
   anchor late in the file**, so each camera's rate can be measured as
   `(last_frame - first_frame) / elapsed video_time` rather than assumed. A wider clip would also mask it, at
   proportionally more detection time per reception.
2. **Video chunking** (`video.create_video_chunk`): extracts a 5-second clip around that frame from each full
   match video into `temporary_videos/`, so detection only runs on a small window rather than the whole match.
3. **Ball detection** (`detection.process_video`): runs `DEFAULT_MODEL` (`ball_n1280_v2.pt`) frame-by-frame on each clip, keeping
   the top *k* boxes per frame above a confidence floor (k = 3, floor = 0.40; not just the most confident
   box — see below). Run independently per camera.

   `DEFAULT_MODEL`, `DETECTION_CONF_THRESHOLD` and `MAX_CANDIDATES_PER_FRAME` live in
   `zoneout/detection.py`, not in a pipeline, because every pipeline needs the same ones — they are
   properties of this detector on this footage, measured on real receptions, and a second copy in a second
   pipeline is how the two would end up different by accident. The comments on them record what was
   measured; do not change them without re-measuring. `detection.load_model` caches the loaded weights, so
   a batch run pays for them once rather than 584 times.
4. **Candidate matching and triangulation** (`reconstruction.match_detections`): per frame, tries every
   pairing of one gopro candidate with one zve10 candidate. Each pairing is turned into two 3D rays via
   `cv2.solvePnP` (that camera's intrinsics + the clicked court reference points as the 2D↔3D
   correspondence) and intersected by least squares (`intersect_rays`). A pairing survives only if the
   resulting point reprojects within `MAX_REPROJECTION_ERROR` of both detections and lands inside
   `court.BALL_VOLUME_*`; the survivor with the best confidence×geometric-fit score wins, and a frame with
   no survivor stays a gap. Then `trajectory.remove_bad_points` and `trajectory.interpolate_nones` (linear
   fill of gaps ≤60 frames) run on the chosen pixels.

   Why it works this way: two cameras can rule out a false positive that one cannot, because a false
   positive in one view rarely lines up geometrically with one in the other. But **the detector's own
   confidence is the primary signal, and the geometric gate must stay loose.** `gala_model.pt` scores the
   ball high and other objects low, so confidence is a strong discriminator; geometry's job is to reject
   the impossible, not to adjudicate between plausible candidates. An earlier version had this backwards —
   a 15 px gate, tuned to look statistically well-placed — and it wrecked the trajectories: it selected
   boxes of median confidence 0.37-0.55 and passed over a camera's top box in 31-83% of frames, because
   calibration error alone puts *correct* pairs tens of pixels out. Do not tighten `MAX_REPROJECTION_ERROR`
   or `DETECTION_CONF_THRESHOLD` without re-measuring on real receptions; the comments on those constants
   record what was measured. **The gate is deliberately in pixels, not meters** — metric ray distance grows
   with range, so a fixed metric threshold is at once too tight for a ball at the far baseline and too
   loose for one at the net.

   Because the winning candidate is only known after matching, `process_video` does not draw the annotated
   debug video; `detection.annotate_video` does, in a second pass. It draws the rejected candidates too,
   which is how you tell "the detector never saw the ball" apart from "the gate threw it away".
5. **Cleaning and gap filling**, one of two branches selected by `pipeline.USE_BALLISTIC_FIT`:
   - **On (default)** — `ballistic.correct_trajectory` takes the matcher's own 3D points and fits
     `p(t) = p0 + v0·t + ½a·t²` per free-flight segment, splitting wherever one parabola stops explaining
     the data (i.e. at contacts). Each segment's model then replaces its frames, filling interior gaps
     *along the actual curve* — which is the whole point, since a gap spanning a change of direction cannot
     be interpolated across but can be covered by two parabolas that each know where they end. It runs on
     the matcher's points rather than on the interpolated pixels below, because feeding linear inventions
     to the fit as though they were measurements would bend the parabola meant to correct them.

     A gap *between* two segments is bridged by extending both models into it and handing over at the frame
     where they most nearly agree on where the ball is (`find_seam`) — position is continuous through a
     contact even though velocity is not. **Every frame a bridge writes is one of the two fitted parabolas
     evaluated outside its own observations, and nothing else.** Nothing is blended, ramped or offset to
     make the two sides meet; where they do not already meet, within `MAX_BRIDGE_SEAM` at the handover
     frame, the whole gap is left empty and recorded in the `Report` as a `SkippedGap`. A filled frame is
     read downstream — by `events.find_serve_and_receive`, by the figures, by the `.dvw` write-back — as if
     it were a measurement, so a position derived from the mismatch between two models rather than from
     either flight is not an acceptable thing to write.

     **Always give the extension its best chance before rejecting a gap.** `closest_approach` runs both
     models `SEAM_SEARCH_MARGIN` frames past their own observations — `before` forward beyond `after.start`,
     `after` backward beyond `before.end` — because two arcs closing on each other are often still
     converging when they reach the edge of the gap, so the best frame *inside* the gap is not the best
     frame at all and the separation measured there overstates the miss. Reception 6's gap 97-117 is the
     case: 0.62 m at frame 117, the last frame of the gap, but 0.38 m at frame 119 just past it. Widening
     the window can only improve the answer, never worsen it, since the gap frames stay in it. When the
     minimum lands outside the gap the handover clamps back to the boundary, so one model fills the whole
     gap and the other is never extended into it — that is what an `N+0` or `0+N` split in the `Report`
     means, and the seam line then reads `frame 117 (0.38 m, closest at 119, 21+0 frames)`.

     The comparison is between the two models **at the same frame**, not between the two curves at whatever
     times bring them closest in space. That distinction is not cosmetic: on reception 4's first gap the
     same-frame separation is 0.76 m while the curves pass within 0.40 m at times fifteen frames apart. A
     contact happens at one instant, so same-frame agreement is what means "the same ball"; two arcs that
     pass near each other at different times only look continuous in a plot, which has no time axis.

     An earlier version did close the seam, adding to each side a quadratic offset growing to half the
     disagreement, which kept each side exactly a parabola (with a nudged acceleration) and left its fitted
     position and velocity untouched where the observations ended. That is sound arithmetic and still the
     right shape if seam-closing is ever wanted again — but it fabricates ball positions, which is the
     reason it is gone. Do not reintroduce it, and in particular do not reach for a linear ramp, which was
     the version before that and is worse on every count: it mixes a straight-line component into the fill
     and leaves the path departing its last real point at the wrong angle (11° of kink at the anchor on
     reception 4, against 1° for the quadratic).

     `MAX_BRIDGE_SEAM` (1.0 m) is therefore a real gate, and is what rejects a bridge in practice. It sits
     where it does because that is what real receptions measure once the search has been given its best
     chance: over receptions 4-6 the honest gaps come out at 0.04-0.93 m. At 0.5 m it threw away four gaps
     at 0.62-0.93 m, three of them the gap containing the dig, which left the reception detected in mid-air
     at 3.1-5.2 m; at 1.0 m all four bridge and the receptions land at 0.5-1.5 m off the floor.
     `MAX_BRIDGE_FRAMES` (90) is only a ceiling on how far either model is trusted past its observations;
     length alone should not throw a gap away, since two arcs still agreeing to within the seam tolerance
     after fifty frames are agreeing on *more*, not less. That distinction matters: an early version paired
     0.5 m with a 30-frame limit and dropped gaps whose arcs agreed to 0.04 m and 0.12 m on length alone.
   - **Off** — the older path: pixel-space `remove_bad_points` + `interpolate_nones` (linear fill), then
     `trajectory.smooth_trajectory`.

     **Short flights are fitted with the acceleration pinned** (`MIN_CONSTRAINED_SEGMENT_POINTS`,
     `USE_CONSTRAINED_SHORT_SEGMENTS`). A serve is in the air for a second and is seen in a hundred
     frames; a spike is over in a fifth of a second, and after the block, the net and the frames the
     detector loses, what survives triangulation can be under ten points. Below `MIN_SEGMENT_POINTS` those
     formed no segment and `DROP_UNEXPLAINED` then deleted them — the flight the attack pipeline exists to
     measure was the one most likely to vanish.

     Lowering `MIN_SEGMENT_POINTS` is the wrong fix, because the count is not the problem — **the
     acceleration is**. Nine free parameters, and over a short span `a` is not observable: gravity moves
     the ball 0.02 m over 4 frames and 0.05 m over 6, far under the 0.30 m `INLIER_TOLERANCE` already
     allows. A free fit spends three parameters on curvature it cannot see and pays in a wrong `p0` and
     `v0`. Pinning `a` to gravity leaves six parameters, so three points over-determine the fit and four
     carry real evidence. `MAX_CONSTRAINED_SEGMENT_FRAMES` (20) is what keeps the assumption honest: over
     20 frames the difference between `a = -9.81` and a drag-heavy `-15` is 0.29 m, still inside the
     tolerance, while by 30 frames it is 0.65 m and the pinned value would be deciding something. Longer
     spans have the points for a free fit anyway. `MIN_SEGMENT_SPEED` is the compensation for the thinner
     evidence — a ball in flight is always moving, so it rejects a stationary object both cameras agreed
     on without adjudicating between plausible flights.

     **`_drop_continuations` is the part that makes this safe, and it must not be removed.** A short pinned
     fit is cheap to satisfy, and the points left over at the edge of a long flight satisfy it happily —
     because they lie on that same arc. Accepting them invents a contact where none happened, and `events`
     reads contacts off exactly that boundary. Measured: without the check it cut a new "flight" out of the
     serve's own descent on receptions 1 and 10, `find_reception` read the new boundary as the dig, and the
     reception moved 2.7-3.3 m up the court — putting reception 10 back to reporting the net touch that
     `NET_ENTRY_MARGIN` exists to exclude. The test is the one the module already uses for "this model
     describes this observation": if a neighbouring segment's model passes within `INLIER_TOLERANCE` of
     every point, they are that flight's points. The split is not marginal — a continuation sits 0.06-0.14 m
     from the neighbour's model, a genuinely new flight 1.15-2.58 m. Only *constrained* segments are tested;
     re-checking free fits would change behaviour that is already validated.

     Measured over 19 cached reception trajectories: **14 come out bit-identical**, 151 frames are gained
     and 49 lost (all on reception 120, one of the known-broken late ones). The reception point moves on
     two: reception 4 from **0.93 m below the floor to 0.33 m above it** — the fit now has the two short
     flights either side of the dig instead of extrapolating one model across 47 frames — and reception 14
     by 0.40 m, downward, the direction the parabola reading is known to be more accurate in. Receptions 44
     and 120 previously found no reception at all and now find one; both have very poor input, and 44's
     answer is visibly implausible (`x = 10.14`, 4.24 m up), so a plausibility gate on the contact point is
     the obvious next thing if those start mattering.

     **Tuning is per-caller, via `ballistic.FitOptions`, not per-module.** The reception pipeline passes
     `pipeline.RECEPTION_FIT`, the attack pipeline `attacks.ATTACK_FIT`. A serve and a spike are not the
     same fitting problem — one is slow, long and seen in a hundred frames, the other is fast, short and
     seen in fifteen — and there is no reason a single set of settings should suit both. Anything tuned
     for one therefore cannot reach the other. A change to either is checked by replaying cached
     trajectories (the figures embed their raw points, so old runs can be re-fitted without re-running
     detection) and comparing segments and contact points before and after.

     **Receptions now split by `'changepoint'` too, with `merge` and the `(-25, -4)` vertical gate.**
     The settings they used before are kept, written out in full, as `pipeline.LEGACY_RECEPTION_FIT`, and
     are still exactly what `correct_trajectory(points)` does with no options; pointing `RECEPTION_FIT`
     at them restores the old behaviour. The reason is the same peeling failure described below for
     attacks, and it hit whole clean flights on receptions: reception 28 lost its entire serve
     (frames 30-89), and reception 36 the pass out of the dig, after which the rest fitted as one arc
     bending *upward* at +3.2 m/s², which the old `(-30, 5)` gate let through. Replayed over the 42
     reproducible cached receptions (1-41, 121): the reception point moves ~0.1 m at the median;
     13, 15, 28, 36 and 39 are fixed outright (four of them had read the set or a rolling ball as the
     dig); 7 and 21 move 1.05 and 0.49 m toward the low point of the dig. **One regression, reception
     23**: a near-stationary object at the net in frames 39-50, before the serve, which the old split
     happened to drop and `'changepoint'` isolates as a flight — both the serve and the reception
     readings then go wrong. (Receptions 44 and 120 were cached by an older pipeline version and do not
     reproduce, so they were left out.)

     Three things `ATTACK_FIT` changes, each measured on the ten attacks in `attack_data/`:

     - **`split='changepoint'`.** The default `'worst'` cuts a span by deleting its single worst-fitting
       observation. That is right when the misfit is one bad point and catastrophic when it is a whole
       short flight: every point of a spike fits worse than every point of the set before it, so the
       recursion peels the spike away *one frame at a time* — measured on attack 1, frames 145, 144, 142,
       141, 140 … down to 127 — and never once offers it to the fitter as a span of its own. Each peeled
       point becomes a fragment too small to fit and is dropped, even though a free fit of frames 126-145
       accepts 15 points at 20.8 m/s with an rms of 0.037. `'changepoint'` instead scans for the *time*
       one flight stops explaining the span, and cuts there keeping every observation.

       Its cost function matters as much as the idea. The scan fits without outlier rejection, so an rms
       cost is dominated by whichever few points are worst and the cut it prefers isolates *those* rather
       than the contact. Using each side's **median** residual instead took the unexplained observations
       from 90 to 38 and the covered ones from 1163 to 1217. Judging on the *worse* of the two sides rather
       than their sum is what keeps the cut away from the ends, since a handful of points fits nine
       parameters almost exactly.

     - **`merge=True`.** Splitting is greedy and top-down and never reconsiders, so a cut can land inside a
       genuine flight. `_merge_segments` puts adjacent segments back together when the union fits as one
       flight **and explains at least as many observations as the two did separately** — that second
       condition is what stops a merge from buying a tidy parabola by spending the 25% outlier budget on
       the points that disagreed with it. Attack 5's attack runs 127-167 and fits to an rms of 0.016, but
       the recursion cut it at 137 into two segments, one bending sideways at 6.1 m/s²; merging restores
       it exactly. Across the ten attacks the merge pass took coverage from 1217 to 1286.

     - **Two acceleration limits.** `max_cross_track_acceleration` is the "a ball cannot bend" rule.
       `Segment.cross_track_acceleration` splits horizontal acceleration into the part along the flight,
       which may legitimately be large — drag is nearly 20 m/s² on a spike at 25 m/s — and the part across
       it, which may not, since only the Magnus force acts there. Total horizontal acceleration cannot
       separate good from bad (a genuine spike shows 11.9 m/s² of it); the cross-track part can. Measured
       over the fitted flights: median 0.4 m/s², 90th percentile 2.2, and one at **19.9** — the segment on
       attack 4 that is visibly impossible in the figure. The limit of 8.0 sits in the empty space between.
       `plausible_vertical_acceleration` is tightened to (-25, -4) from the module's very wide default:
       every genuine flight measured falls between -7.5 and -13.9, and the one exception was +2.4 on a fit
       whose horizontal deceleration was also twice what drag can produce. A ball in flight is falling.

     Together, over attacks 1-10: observations no flight explains fall from **131 to 38**, observations
     covered by a flight rise from **1146 to 1286**, the worst cross-track acceleration anywhere drops from
     19.9 to 4.5 m/s², and the two attacks identified by eye are found — attack 1's at frames 127-142
     (against 126-145 by eye) and attack 5's at exactly 127-167.

   Validated across receptions 1-5: the two branches agree to a median of 0.01 m and a maximum of 0.30 m,
   with no frame differing by more than 0.5 m, so the correction is not rewriting already-good data. The
   fitted vertical accelerations come out at -10 to -11 m/s² — nothing in the pipeline knows about gravity,
   so that is an independent check that calibration and triangulation are sound, and a useful smoke test
   after any change to either. `correct_trajectory` returns a `Report` that the pipeline prints every run;
   large numbers there mean this step is the first thing to suspect. The `Report` lists each bridged gap as
   `frame S (D m, N+M frames)`: where the two models were handed over, how far apart they still were there,
   and how many frames each side ended up contributing. Those last two numbers are the diagnostic — `21+0`
   means one arc filled the whole gap and the other was never extended into it, which happens when the two
   are still converging at the edge of the gap instead of crossing inside it. That is the right answer when
   the contact really did happen where the next segment's observations start, and a reason to look at the
   clip when it did not. Each gap left empty is listed too, as `frames A-B (D m apart)`, so a hole in a
   plot is always findable in the run output rather than being a silent drop; if `D` sits just over
   `MAX_BRIDGE_SEAM` the fits are probably fine and the tolerance is the thing to look at.

   Measured on receptions 4-6 with the seam tested rather than closed, the search widened past the gap
   edges, and the tolerance at 1.0 m: bridges fill 85, 104 and 46 frames at seams of 0.04-0.93 m, and no
   gap is rejected. Every bridged frame was checked to be an exact evaluation of one of the two fitted
   parabolas — no blend survives anywhere. The receptions come out at 0.75, 0.51 and 1.53 m off the floor,
   against the 3.1-5.2 m they showed when those gaps were dropped and the serve's descent was truncated
   short of the dig. The unextended distance between the two segments' observed endpoints across those gaps
   is 2.9-7.5 m, so the extension is doing nearly all of the work: that ratio, not the absolute seam, is
   the sign that two arcs really are continuing into each other.
6. **Serve/reception detection** (`events`): the serve comes from `find_serve_and_receive`, which splits the
   trajectory where `y` changes sign (ball crossing the net) and takes the pre-split point nearest `|y| = 9`
   (the baseline).

   The **reception** is read off the ballistic segmentation instead, by `events.find_reception`
   (`pipeline.USE_BALLISTIC_RECEPTION`, on by default and requiring the fit): it is the start of the first
   fitted parabola that begins once the ball is more than `NET_ENTRY_MARGIN` (1 m) past the net on the
   receiving side. That is what a contact *is* — the frame where one free-flight model stops describing the
   ball and the next starts — so if the segmentation is right the reception is already located, and
   re-deriving it from turn angles only adds a way to get it wrong. Same reason the margin is there: a serve
   that clips the net cord also starts a new parabola, and does so at `y ≈ 0`, so without the margin the net
   would be read as the dig.

   The contact frame is the **seam** with the preceding segment, not the new segment's first observed frame
   — those differ by 50-100 frames on a real reception, because the dig sits in a gap neither flight
   observed, and by the time observations resume the ball is back up in the air. Where there is no gap the
   two coincide anyway. The point returned is the new segment's model at that frame and nothing else; the
   two models are known to agree there to within `MAX_BRIDGE_SEAM`, and blending them would invent a
   position that is neither flight's.

   Both readings are printed side by side every run. Measured over receptions 4-10 they land 0.15-1.11 m
   apart, with the heights within 0.20 m and nearly all of the difference in `y`, depth into the court —
   which matters, because `y` is what decides the zone written to the `.dvw`. The parabola reading is
   *deeper* on all seven, by 0.3-1.1 m, and that direction is expected rather than coincidental: the
   direction-change reading fires at the first frame whose ±3-frame turn exceeds 45°, which is a few frames
   after the contact, by which time the ball is already on its way back toward the net. So this is a
   refinement of an already-working step, not a rescue of a broken one — which is why the old reading is
   kept as the fallback for when `find_reception` finds nothing.

   Reception 10 is the case that justifies the margin, and it is real rather than hypothetical: its serve
   splits into a new parabola at frame 69 with the ball at `y = -0.76`, the two arcs agreeing to 0.01 m —
   a touch on the net cord. Without `NET_ENTRY_MARGIN` that segment would have been reported as the dig,
   half a metre past the net; with it, the reception comes out at `y = -3.64` where the ball was actually
   played. Reception 5 shows the other thing the rule steps over — its first two segments are the toss and
   the struck serve, both on the serving side.
7. **Write-back** (`scout.add_serve_direction`): converts the 3D start/end coordinates into DataVolley's
   numeric court-index format (`coords_to_dvindex`) and rewrites the matching `S`/`R` skill lines in the
   `.dvw` in place, matched by `video_time`. Handles court-side rotation (coordinates flipped 180° when the
   action isn't already on the serving side).
8. **Figures** (`figures/`): a 3D plot of the trajectory with the serve/reception points highlighted, written
   twice per reception into `reception_data/receptionN/` — `raw_trajectory.png` for flicking through, and
   `raw_trajectory.html`, an interactive page you open in a browser and rotate/zoom/hover. The HTML page also
   carries the matcher's own triangulated points as a second series in ochre (`raw_points=`), so the fitted
   parabolas can be judged against the measurements they were fitted to; clicking a legend entry hides either
   series. That overlay is the fastest way to tell a bad fit from bad input — if the ochre points scatter, the
   problem is upstream in detection or matching, and if they sit tight while the blue path wanders, it is the
   ballistic step. With the serve-and-reception dataset on, the HTML page's **Serve** marker is the
   dataset's `serve_start_*` (the ballistic contact, extrapolated to the baseline when unseen) instead of
   the geometric reading written to the `.dvw`, and it also marks the **Reception end** and the **Pass apex** — the
   point `reception_apex_z` was read at, drawn in neutral ink as a cross with a dashed drop line and its
   height in the label. All three are exactly as they went into `datasets/receptions.csv`
   (`reception_dataset.DrawnPoints`): a blank in the row is an absent marker here. The PNG, and the HTML
   with the dataset off, still mark the geometric serve. The PNG stays a single path deliberately: two
   overlaid trajectories in a fixed 3D projection
   are hard to separate without being able to rotate them.

### Camera intrinsics

`zoneout/cameras.py` turns a camera model plus a focal length into the matrix `solvePnP` is given. The
selectable models are the Sony ZV-E10, the Sony a6000 and a GoPro; the two Sony bodies need a focal length
in mm, the GoPro does not.

For a Sony body the whole of the arithmetic is `fx = fy = focal_length_mm × 1920 / 23.5`. Both carry the
same 23.5 × 15.6 mm APS-C sensor and at 1080p60 with stabilisation off both read out its full width — the
ZV-E10 crops only at 1080p120 (1.14×), at 4K30 (1.23×), and with Active SteadyShot — so the 16:9 frame is a
vertical crop of the 3:2 sensor, the pixels stay square, and **fy equals fx**. The principal point is
assumed to be the middle of the frame. The GoPro keeps the 90°-horizontal matrix that has always been in
the pipeline, because its digital lens setting (Wide/Linear/SuperView) changes the field of view and the
footage does not record which was used.

**These are specification-sheet approximations, not calibrations**, and no lens distortion is modelled
anywhere (`_camera_pose` passes zero coefficients). That is part of why `MAX_REPROJECTION_ERROR` has to
stay loose at 40 px.

**The two matrices that used to be hardcoded in `reconstruction` are still there, as the per-position
fallback**, and `cameras.matrix_for` returns them — announcing it — whenever the selection cannot be used:
no `camera_models.csv`, nothing recorded for that camera, an unknown model name, or a Sony with no focal
length. A run with no config file therefore reconstructs exactly as it did before any of this existed.
Falling back rather than raising is deliberate, since a batch run takes hours; being loud about it is
equally deliberate, since silently reconstructing a match with the wrong intrinsics is the bad outcome.
Each camera announces its intrinsics once per run, because `_camera_pose` is cached.

`reconstruction.CAMERAS` still keys on `"gopro"` and `"zve10"`. **Those names now mean only the sideline
and baseline positions** — either position can hold any of the models — and the same goes for the
`gopro_points.csv` / `zve10_points.csv` file names.

How good is the approximation? Fitting it against this match's own clicked points is a cheap check, since
solvePnP has 8 correspondences for 6 pose parameters and the leftover reprojection error is what the
intrinsics could not explain. On the test match: the baseline camera comes out at **33 mm, mean 1.43 px**,
against 23.65 px for the old fx=2371/fy=2008 matrix, and the sideline GoPro is best at **fx ≈ 910 (93°)**
at 2.29 px against 6.14 px at fx = 960. The minima are sharp, but note the fit can absorb distortion and
click error into the focal length, so "best fitting" is not the same as "true".

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

- `style.py` — the palette. Read its docstring before changing any color: the four data colors were checked
  with the dataviz palette validator against the orange court, and the obvious choice of a **red** reception
  marker was rejected because red-on-orange measured ΔE 14.2 for normal vision, below the floor of 15. The raw
  overlay's ochre is likewise not a free choice — every cool color tried for it landed within ΔE 13.7 of the
  blue fitted path, which is under the floor, so a warm one was the only room left.
- `court.py` — `draw_court(ax)` for matplotlib, `court_traces()` for plotly. Geometry comes from
  `zoneout.court`.
- `serves.py` — `save_player_serves_html(path, dataset, player_number, team=None)`: one player's serves from
  the serve dataset, each drawn as its stored parabola with a cone for direction, over the court and the
  net (`court.net_traces`), with small unlabelled contact/reception dots and the player's mean speed in
  the title. Paths are colored by average speed on `style.SPEED_SCALE`, a cyan-to-violet
  ramp, over the **fixed** `serves.SPEED_RANGE` of 10-28 m/s (a really slow float to a really hard spin
  serve; measured floats average 12-17, topspin jump serves 19.5-26.4), so a color means the same speed
  in every figure from any dataset. The ramp's light end is limited by staying dE ≥ 15 from both the court
  and the scene panels — see `style.py`. `team` is needed only when two teams share
  the number, and it raises rather than merge them.
- `trajectory.py` — `save_trajectory_figure(path, ...)` writes the PNG, `save_trajectory_html(path, ...)`
  writes the interactive page. `plot_trajectory(...)` / `plot_trajectory_plotly(...)` return the figure
  objects if you want to tweak before saving. All accept point lists containing `None` and skip those frames.
  The plotly ones additionally take `raw_points=`, drawn as **markers with no connecting line** — the raw
  series has gaps wherever no pairing survived, and joining across one would draw a straight segment the
  reconstruction never claimed. The plotly ones also take `heights=[(label, point)]` for a point whose
  height is the measurement (the top of a pass): a cross in `style.APEX` ink with a dashed drop line to
  the floor, labelled with the height. They also accept `points=None`, which draws no ball-path trace
  at all: that is for a pipeline stage that has triangulated points but has not fitted a trajectory through
  them yet, and it is how the attack pipeline draws its raw points today.

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
