# Plan: attack directions

Design plan for extending the pipeline from serves/receptions to **attacks**: pulling the attack timeline
out of the `.dvw`, reconstructing the ball, and writing the coordinates back safely.

Written before any code. Nothing here has been implemented. `CLAUDE.md` stays the architecture reference for
what exists today; this file is what is proposed, why, and in what order. Fold the parts that survive
contact with reality into `CLAUDE.md` as they land, and delete the rest.

The governing constraint from the request: **attacks must be switchable off while receptions keep running.**
That is a structural requirement, not a flag — it means shared machinery gets factored out and the
action-specific reasoning lives in separate modules, rather than a second branch threaded through
`pipeline.get_data_from_reception`.

---

## 1. What was checked, and what it means

Everything in this section was measured against `&svk-ork_q5.dvw` (the real match) and
`&svk-ork_test.dvw`, the installed `pydatavolley`, and openvolley's R sources. It is the factual basis for
the design; the design choices follow from it.

### 1.1 `get_plays()` rows are 1:1 with the `[3SCOUT]` lines

The scout section of `&svk-ork_q5.dvw` has 1387 non-empty lines, and `DataVolley(...).get_plays()` returns
exactly 1387 rows, in order — non-skill lines (`*z2`, `ap01:02`, `**4set`) are kept as rows with
`skill = NaN`. So **the dataframe row index is the scout-line index**, and an action can be addressed in the
file by position rather than by content.

This is the single most useful thing found, because:

### 1.2 `video_time` is ambiguous, and not even monotonic

`add_serve_direction` currently finds the line to patch by matching `video_time` and the skill letter. That
works for serve/reception because a rally has one of each and they share a `video_time`. It does not
generalise:

- 292 attacks in the match; 8 of them share a `video_time` with another attack.
- Within a rally, the set, the attack, the block and the dig routinely share one `video_time` (the scout
  enters the whole sequence in one second).
- `video_time` is not monotonic across the file: the action *after* an attack sometimes carries a
  `video_time` one second **smaller**.

So a `video_time` match is a coin flip for attacks. The write-back has to address lines by index.

### 1.3 The DataVolley coordinate grid, exactly

From openvolley's `datavolley` R package (`R/plot.R`), the grid is defined by its cell **left/bottom edges**:

```r
dv_xy_xbins() = 0.5 + (1:100 - 11)/80 * 3.0     # 100 columns
dv_xy_ybins() = 0.5 + (1:101 - 11)/81 * 6.0     # 101 rows
dv_xy2index(x, y) = xi + (yi - 1) * 100          # values are binned with right = FALSE
dv_flip_index(i)  = 10101 - i
```

In openvolley's plotting units the court is `x ∈ [0.5, 3.5]`, `y ∈ [0.5, 6.5]`, net at `y = 3.5`, so one
unit is 3 m. Mapping to `zoneout.court` (`x ∈ [0, 9]`, `y ∈ [-9, 9]`, net at `y = 0`):

```
dv_x = 0.5 + x/3            dv_y = 3.5 + y/3
```

The court therefore occupies columns 11–90 (80 cells over 9 m, 0.1125 m each) and rows 11–91 (81 cells over
18 m, 0.2222 m each). `pydatavolley` ships the inverse as `datavolley.helpers.dv_index2xy`, so a round-trip
test is available with no new dependency.

**`scout.coords_to_dvindex` is up to one cell out.** It lays 80 rows over the 18 m length where the grid has
81, and truncates instead of binning to the openvolley edges. Checked against `dv_index2xy`:

| world (x, y) | current index | reads back as | openvolley index | reads back as |
|---|---|---|---|---|
| (0, −9) | 1010 | (−0.06, −8.89) | 1011 | (0.06, −8.89) |
| (9, −9) | 1090 | (8.94, −8.89) | 1091 | (9.06, −8.89) |
| (0, +9) | 9010 | (−0.06, 8.89) | 9111 | (0.06, 9.11) |
| (9, +9) | 9090 | (8.94, 8.89) | 9191 | (9.06, 9.11) |

So: ~0.11 m of systematic shift in x everywhere, and up to one row (0.22 m) in y, growing toward `y = +9`.
Small, but free to fix and worth fixing before attacks make it visible — an attack's end coordinate is
often near a line, and in/out is exactly the question a 0.22 m bias sits on.

### 1.4 The file has no coordinate orientation to conform to

openvolley's own snippets say it plainly: *"Coordinates might not appear in the dvw file in any particular
orientation (i.e. starting consistently on one side of the court)."* Every coordinate currently in
`&svk-ork_q5.dvw` (55 lines) and `&svk-ork_kvart_5.dvw` (13) was written by this pipeline — the original
scout entered none — so there is no in-file convention to match, and no ground truth for the rotation
`add_serve_direction` performs. `NOTES.md` already flags that rotation as unverified. Attacks make the
question unavoidable, because unlike serves they happen from **both** halves.

### 1.5 The scout file already contains ground truth for validation

Of 292 attacks, **291 have a scouted `start_zone`** and **221 a scouted `end_zone`** (50 also an
`end_subzone`), entered by a human watching the match. Zones are a 3×3 grid over a half-court
(`dv_xy` centres at x ∈ {1,2,3}, y ∈ {1,2,3} in DV units), so a reconstructed 3D point converts to a zone
trivially. That gives a free, match-independent check on every attack the pipeline produces, and — because
zone numbering is relative to a team's own half — enough signal to settle the orientation question
empirically instead of by assertion.

Serves also carry zones (`as_for_serve` uses only 1/5/6/7/9), so the same harness retroactively validates
the serve/reception write-back that has never been checked.

### 1.6 openvolley's writer is not a shortcut

`dv_write` exists in the R package but is documented as *"really rather experimental"* and explicitly does
not update the scouted `code` column to match changed fields. It is R, and this project is Python. Keep the
current approach — surgically edit the coordinate fields of specific lines, leave every other byte alone —
but harden it (§4). `pydatavolley` is read-only and stays a reader.

### 1.7 Scale and cost

292 attacks against 153 receptions. At the current ~65 s per action on CPU, a full-match attack run is
**5–6 hours**, and that is with the *same* clip length. Anything that widens the window scales it directly.

---

## 2. What an "attack direction" is, and what gets written

A DataVolley scout line is `;`-separated; the pipeline already knows fields 0 (code), 4 (start coordinate),
5 (mid), 6 (end), 12 (`video_time`). For an attack:

- **start coordinate** — where the attacker contacted the ball, projected to the floor.
- **mid coordinate** — left as `-1-1`, as the serve write-back already does.
- **end coordinate** — where the ball finished: the floor if it landed, otherwise where it was dug or
  blocked.

**The scouted `code` string is never rewritten.** The zone and cone letters in it are the human scout's
reading, `dv_write`'s own limitation is exactly this, and silently disagreeing with a scout inside their
own file is not a thing this pipeline should do. Computed zones are for *validation and reporting*
(§5) — if they should ever be written, that is a separate, deliberate decision.

---

## 3. Proposed structure

```
run_pipeline.py                 two independent toggles, one selection block each
zoneout/
    dv_grid.py          NEW     world metres <-> DataVolley grid index, zone, subzone
    dvw_edit.py         NEW     safe, line-indexed patching of a .dvw
    scout.py            CHANGED reading generalised from receptions to "actions"
    video.py            CHANGED create_video_chunk gains a lead and a duration
    events.py           CHANGED gains find_attack alongside find_reception
    pipeline/           NEW package, replacing pipeline.py
        __init__.py             re-exports get_data_from_reception (import path unchanged)
        core.py                 window -> clips -> detection -> matching -> fit -> figures
        receptions.py           reception selection, event rules, write-back
        attacks.py              attack selection, event rules, write-back
    ballistic.py        unchanged
    reconstruction.py   unchanged
    detection.py        unchanged
    trajectory.py       unchanged
    court.py            unchanged
    figures/            unchanged
```

### 3.1 `zoneout/dv_grid.py` — the coordinate format, in one place

Pure arithmetic, no I/O, no pandas. Mirrors what `court.py` does for the world geometry: one place that owns
the DataVolley grid so the reception writer and the attack writer cannot drift apart.

```
xy_to_index(x, y)        world metres -> grid index, openvolley binning
index_to_xy(index)       the inverse, cell centre
flip_index(index)        10101 - index
xy_to_zone(x, y)         -> 1..9 for the half the point is in, plus which half
zone_to_xy(zone, half)   for reporting and figures
xy_to_subzone(x, y)      A/B/C/D within the zone
```

Two things this earns immediately: the §1.3 fix, and a round-trip test against
`datavolley.helpers.dv_index2xy` that proves the convention rather than assuming it.

`scout.coords_to_dvindex` becomes a thin wrapper over `xy_to_index` (or goes away). Because the index values
change slightly, the corrected version is measured against the current one over the receptions already
processed, and the shift reported — same discipline as the ballistic fit.

### 3.2 `zoneout/dvw_edit.py` — the safe writer

The current writer reads the whole file, string-matches on `video_time`, mutates, and writes the whole file
back, once per reception, with no backup. That is 292 more chances to corrupt a match file. Replace it with
an explicit, auditable editor:

```
class DvwFile:
    scout_lines()                     the [3SCOUT] block, split, in order
    set_coordinates(line_index, start=, mid=, end=, expect_code=)
    save(backup=True)                 atomic: write temp, fsync, os.replace
```

Rules it enforces:

- **Address by line index, verify by code.** `expect_code` is the `code` field as read; if the line at that
  index does not carry it, raise instead of writing. Cheap, and it makes a stale index impossible to act on.
- **Touch fields 4, 5 and 6 and nothing else.** Rejoin the untouched fields verbatim; never reformat, never
  rewrite the code.
- **Preserve encoding and line endings.** `cp1252` (the file has `ö`, `å`, `é`), opened with `newline=''` so
  a CRLF file stays CRLF. Today's files are LF, but a file straight out of DataVolley on Windows need not be,
  and text-mode round-tripping would silently rewrite every line in the file.
- **One save per run, not one per action.** Accumulate edits in memory; write once at the end and once more
  on interrupt, so a crash mid-run leaves the file as it was rather than half-patched.
- **Back up before the first write** (`<name>.dvw.bak`, refusing to overwrite an existing backup) and
  **replace atomically**, so an interrupted write cannot truncate a match file.
- **Do not clobber a human's coordinates.** If a line already has a start/end coordinate that this run did
  not write, skip it and report, unless `OVERWRITE_EXISTING` says otherwise. (Today every coordinate in the
  test files is pipeline output, so this costs nothing now and prevents a real accident later.)
- **Verify after saving** by re-reading with `pydatavolley` and asserting every column except the three
  coordinate columns is unchanged, and that the coordinates read back within one cell of what was asked for.

`add_serve_direction` is reimplemented on top of this with its behaviour preserved exactly (§6, phase B),
so receptions gain the safety without a behaviour change.

### 3.3 `zoneout/scout.py` — reading, generalised

`get_receptions` is the "single place the numbering comes from" and that property must survive. Generalise
rather than duplicate:

```
get_actions(dvw_filepath, skill)      one row per action of that skill, in play order,
                                      carrying the scout-line index as a column
action_numbers(dvw, skill, set_number=None)
get_action_start_frame(dvw, skill, anchor_frame, anchor_video_time, n, fps)
```

`get_receptions` / `reception_numbers` / `get_reception_start_frame` become one-line wrappers, so nothing
downstream changes and the reception numbering is provably the same function it was. `get_attacks` /
`attack_numbers` are the same wrappers with `skill='Attack'`.

**Attacks are numbered from 1 across the whole match**, exactly as receptions are, for exactly the same
reason: the number names the output directory and is what the frame lookup takes, so it must mean one thing
however a run is selected.

The new part is the scout-line index. `get_plays()` preserves file order (§1.1), so the index is
`plays.index` before filtering — captured with `reset_index()` rather than recomputed, and asserted against
the line count at read time so the day pydatavolley starts inserting synthetic rows is the day this raises
instead of the day it writes to the wrong line.

### 3.4 `zoneout/video.py` — a window, not a forward cut

`create_video_chunk` currently cuts a hardcoded 5 s **forward** from the anchor frame. That is a reception
assumption: the serve starts the rally, and the scout's `video_time` lands before or at it. An attack sits
mid-rally and the scout codes it *after* it happens, so the flight is behind the anchor as often as ahead.

```
create_video_chunk(path, name, frame, lead_seconds=0.0, duration_seconds=5.0)
```

Receptions keep `lead=0, duration=5` and are bit-identical. Attacks get `lead=0.5, duration=3.5`.

The small lead is not a guess about scouting in general — it rests on **how these files are scouted**: they
are synced by hand with an attack's `video_time` placed at the moment the *setter* touches the ball, so the
attacker's contact follows within about a second. That is a property of the input, not of DataVolley, and
nothing in the file would reveal a match scouted to a different convention; the clips would simply miss the
flight. Hence a named constant with the assumption written next to it, not a number in a call.

The existing "video does not reach that far" error stays, and gains a matching one for a lead that would run
before frame 0 — raising rather than clamping, because clamping would move one camera's window and not the
other's, and the two clips are paired frame for frame.

### 3.5 `zoneout/pipeline/core.py` — the shared reconstruction

Everything in `get_data_from_reception` that is not reception-specific, taking a window and returning what
was reconstructed:

```
reconstruct_window(sideline_path, baseline_path, sideline_frame, baseline_frame,
                   output_dir, label) -> Reconstruction
```

`Reconstruction` carries `points` (the corrected trajectory), `raw_points` (the matcher's own triangulated
points), `report` (the `ballistic.Report`, hence `segments`), and the two per-camera candidate lists for
the annotated debug videos. It cuts the clips, runs detection twice, matches, runs the ballistic branch (or
the pixel-space fallback), writes the debug videos and the PNG/HTML figures. It knows nothing about serves,
receptions or attacks.

`DETECTION_CONF_THRESHOLD`, `MAX_CANDIDATES_PER_FRAME`, `USE_BALLISTIC_FIT` move here unchanged.

### 3.6 `zoneout/pipeline/receptions.py` and `attacks.py`

Each is thin: pick the frames for action *N*, call `reconstruct_window`, read its events off the segments,
write back. `USE_BALLISTIC_RECEPTION` lives in `receptions.py`; the attack equivalent lives in `attacks.py`.
Neither imports the other. Deleting `attacks.py` outright would leave receptions working.

`pipeline/__init__.py` re-exports `get_data_from_reception`, so `run_pipeline.py` and any scratch script
keep working through the existing import path.

### 3.7 `run_pipeline.py`

Two independent, explicitly separate blocks:

```python
PROCESS_RECEPTIONS = True
PROCESS_ATTACKS    = False      # default off until §5 validation passes

SET_NUMBER = None
FIRST_RECEPTION, LAST_RECEPTION = 1, 10
FIRST_ATTACK,    LAST_ATTACK    = 1, 10
```

Separate ranges because reception 7 and attack 7 are unrelated rallies, and a debugging run wants one or the
other. The existing per-action failure handling, the summary of failures, and the
"sync anchor is from another set" warning are shared and stay as they are. The `.dvw` is saved once at the
end of the whole run, covering both kinds.

### 3.8 Output

`reception_data/receptionN/` stays as it is — it is referenced in `CLAUDE.md`, `NOTES.md` and a run's own
output, and renaming it buys nothing. Attacks get `attack_data/attackN/` alongside, with the same four
files (`gopro_boxes.mp4`, `zve10_boxes.mp4`, `raw_trajectory.png`, `raw_trajectory.html`).

The figures need one change: `save_trajectory_figure` / `save_trajectory_html` take `serve_point=` and
`receive_point=`, which for an attack should read "contact" and "landing". Generalise to a small list of
labelled, coloured markers rather than two named keywords, keeping the existing serve/reception colours
(`figures/style.py` — the palette was validated against the orange court and is not a free choice; an attack
pair should reuse `SERVE`/`RECEPTION` rather than introduce new hues without re-running the validator).

---

## 4. Finding the attack in the trajectory

This is the attack-specific reasoning, and the one place where receptions cannot simply be reused.

Reception logic is *"first new parabola once the ball is past the net"* — the serve anchors the rally, and
there are only ever two or three flights in the window. An attack's window contains a set, the attack, and
a dig or block; the segmentation will produce several flights, and the right one has to be identified rather
than counted to.

**`events.find_attack(segments, anchor_frame, ...) -> (contact, landing, frames)`**, with these rules, each
of which is a physical statement about what an attack is:

1. **The attack is a flight that crosses the net.** Its start and end lie on opposite sides of `y = 0`. A
   set does not cross the net; a dig usually does not.
2. **It starts high and near the net.** A spike contact is above net height and within a couple of metres of
   `y = 0`. This is the discriminator against a set (which starts low, or high but does not cross) and a
   freeball.
3. **It is fast.** `Segment.speed` already exists; an attack is many times a set's speed. Useful mainly as a
   tie-break and as a rejection criterion, not as a primary test — same lesson as the reprojection gate: the
   physical impossibility is what a threshold should reject, not the merely less likely.
4. **Among candidates, take the one whose contact frame is nearest the anchor**, since the anchor is
   `video_time` and is good to a second or so.

**The contact frame is the seam with the preceding segment, not the segment's first observed frame** —
identical reasoning to `events._contact_frame` for receptions, so that function is shared, not copied. The
point returned is the attack segment's own model evaluated there.

**The landing is the other end of the same flight**, and there are two cases:

- The ball hit the floor: the landing is where the attack's parabola crosses `z = 0`, evaluated from the
  model. This is the honest answer even when detection loses the ball on the way down, and it is the
  coordinate DataVolley means by the attack's end.
- The ball was dug or blocked: the flight ends at a contact, so the landing is the **seam with the following
  segment**, exactly as the reception is the seam after the serve.

Which case applies is decided by the segmentation itself — if there is a following segment whose seam is
accepted, the flight ended in a contact; otherwise the model is extrapolated to the floor. A floor crossing
that lands implausibly far outside the court, or a segment with no following segment and no floor crossing
inside the window, is a failed attack: reported and skipped, never guessed, in keeping with
`ballistic`'s rule that a written position must be a real model evaluated on its own terms.

`find_attack` returns None when nothing qualifies, and the caller reports and skips. There is deliberately
**no geometric fallback** the way `find_serve_and_receive` backs up `find_reception`: for receptions the
fallback exists because it predates the fit and is known to work; for attacks there is nothing to fall back
*to*, and inventing a heuristic to fill the gap would be a way to get a wrong answer confidently.

### 4.1 Which half, and hence how to write it

`find_attack` also returns the side (`sign(y)` of the contact). That is measured from the reconstruction
rather than looked up from team and set, which is the general answer — it works for a file where the teams
swapped ends mid-set, or where the scout's court orientation is unknown, and it needs no assumption about
how DataVolley numbers sides.

What the side is *used for* is §5.

---

## 5. Orientation, and validating the write-back

This is the part to settle before any attack is written to a real file, and it is also the overdue answer to
the `NOTES.md` question about `add_serve_direction`'s rotation.

The problem: the pipeline knows where the ball was in world coordinates; DataVolley wants a grid index; and
per §1.4 there is no in-file convention saying which world half maps to which grid half. Currently
`add_serve_direction` rotates 180° so the serve always starts at `y < 0`, and forces the serve's `y` to
exactly −9. Whether that matches what a scout sees in DataVolley has never been checked.

**The check, which costs nothing and needs no new data:** a standalone script (not part of a pipeline run)
that, for every attack in a processed match, converts the reconstructed contact and landing to zones with
`dv_grid.xy_to_zone` and compares them against the `start_zone` / `end_zone` the human scout entered — 291
and 221 attacks of ground truth in this match alone. Run it under each candidate convention:

- world → grid directly (`y = −9` at the bottom of the grid, no rotation)
- rotated 180° per action so the acting team is always at the bottom (today's serve behaviour)
- the above with x mirrored

The convention that agrees with the scout is the right one, and the disagreement rate is a standing quality
metric — a per-zone confusion matrix is exactly the kind of thing that shows a systematic mirror instantly,
where a handful of eyeballed cases would not. The same script run over serves (`as_for_serve` zones
1/5/6/7/9) settles the reception rotation retroactively.

The conclusion gets written into `dv_grid.py` as documented, named functions, not spread through two
writers as inline sign flips.

**Second check, structural rather than statistical:** after a run, re-read the file with `pydatavolley` and
assert that every column other than the three coordinate columns is byte-identical to the pre-run parse.
That catches a mangled delimiter, a lost `~`, an encoding slip, or a patched-wrong-line long before anyone
opens the file in DataVolley.

**Third check, human:** open one patched match in DataVolley itself and look at a handful of attacks whose
direction is unmistakable (a line shot, a sharp cross). Statistics agreeing with the scout's zones and the
file parsing cleanly still do not prove the numbers land where DataVolley draws them.

---

## 6. Order of work

Each phase is independently useful, and independently revertable. Nothing writes attacks to a real file
until phase E passes.

**A — `dv_grid.py`.** The grid conversion, with a round-trip test against `datavolley.helpers.dv_index2xy`
and the §1.3 correction. `coords_to_dvindex` becomes a wrapper. Measure the index shift against the
receptions already written and report it. *No pipeline behaviour change beyond the corrected index.*

**B — `dvw_edit.py`.** The safe writer, with `add_serve_direction` reimplemented on top of it. Verify by
patching a copy of `&svk-ork_test.dvw` — the fixture kept for exactly this — with the old and new writers
and diffing the results byte for byte. Receptions gain the backup, the atomicity and the verification with
no change in what they write.

**C — split `pipeline.py` into the package.** Pure refactor. Verify the way the last restructure was
verified: a golden end-to-end run of receptions 4–6 producing identical 3D coordinates, identical figures
and an identical patched `.dvw`.

**D — attacks.** `scout.get_actions`, the video window, `events.find_attack`, `pipeline/attacks.py`, the
figure labels, the `run_pipeline.py` toggles. Run over a handful of attacks with `PROCESS_ATTACKS` on and
the write-back **disabled**, and read the results off the interactive figures — the ochre raw-points overlay
is what tells a bad fit from bad input, and an attack is a harder detection problem than a serve (fast, and
the contact is behind the block).

*Done so far, in `zoneout/attacks.py` and `run_attack_pipeline.py`:* selection and timing, the clip window,
detection, matching and triangulation, the ballistic fit, and `events.find_attack_points` — the set, the
attack and where the attack finished, collected into `attack_data/attack_points.csv`. Still to do here: the
write-back, which phase E gates.

*One thing this phase turned up that the plan did not anticipate.* A spike is short — a fifth of a second,
and under ten surviving points after occlusion — and `ballistic` discarded anything below
`MIN_SEGMENT_POINTS`, so the flight the whole pipeline exists to measure was the one it threw away. The fix
was not to lower that threshold but to **pin the acceleration** on short spans: nine free parameters become
six, and over a tenth of a second the curvature was never observable anyway. See the ballistic section of
`CLAUDE.md` for the measurements, including `_drop_continuations`, which is what stops a cheap short fit
from cutting a fake contact out of a long flight's leftovers.

**E — validation, then enable.** The §5 harness over a full match. Fix what it finds. Only then does
`PROCESS_ATTACKS` default on, and only then does the attack path write.

---

## 7. Risks and open questions

**The clock drift becomes a blocker, not a nuisance.** `CLAUDE.md` records that `DVW_CLOCK_FPS = 60.0` is
wrong by enough to push a serve out of its 5 s clip by the third set, and that the fix is a second sync
anchor so each camera's rate can be *measured* as `(last_frame − first_frame) / elapsed video_time` rather
than assumed. Receptions survive this because a per-set anchor keeps the error small and a serve's arc is
long. An attack's whole flight is a fraction of a second, so the same drift is proportionally far more
likely to leave the flight outside the window — and attacks are spread across the entire match rather than
concentrated at rally starts. **Do the two-anchor fix before or alongside phase D**; widening the window to
mask it multiplies an already 5–6 hour run.

**Runtime.** 292 attacks at ~65 s is most of a working day on CPU. Caching detections per clip to disk (as
the ballistic work did) is what makes iterating on `find_attack` bearable — inference once, then seconds per
experiment. Worth building into phase D rather than retrofitting.

**Occlusion at the contact.** An attack contact is high and right at the net, which is where the block is
and where the net itself is. The baseline camera in particular may not see it. Where the contact frame is
inside a gap the segmentation covers it (the seam is a fitted model evaluated past its data, which is
precisely what `ballistic` is for), but a contact with no observed flight *before* it has no seam and falls
back to the segment start. Expect this to be the dominant failure mode and to show up as attacks reading
late.

**Attacks off the block, and joust balls.** A blocked ball that comes back over is a second net crossing in
the same window, and rule 1 in §4 does not distinguish it from the attack. Rule 4 (nearest the anchor) and
rule 2 (starts high, near the net, on the attacking side) should, but this is the case to look at first when
an attack reads wrong.

**Decisions taken here, worth disagreeing with:** attacks numbered match-wide like receptions rather than
per set; `attack_data/` alongside `reception_data/` rather than a unified `output/` tree; the scouted `code`
string left alone rather than having its zones rewritten to match; and no geometric fallback for
`find_attack`. Each is stated where it is made, and none is load-bearing for the rest of the plan.
