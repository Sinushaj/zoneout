"""Patching a `.dvw` scout file, safely: coordinates, and set lines before attacks.

A scout file is the only thing in this project that is not reproducible. Videos
can be re-cut and trajectories re-fitted, but a match file holds hours of a
human's work and the pipeline edits it in place. A batch run patches it 292
times, so it is 292 chances to corrupt it.

So this module does two narrow things and refuses to do anything else: it
replaces **fields 4, 5 and 6** - the start, mid and end coordinates - on scout
lines addressed **by line number**, and it **inserts new lines** in front of
lines addressed the same way. Every other byte of the file is rejoined exactly
as it was found.

The rules it enforces, and why each is here:

- **Address by line number, verify by code.** `video_time` cannot name a line:
  within a rally the set, attack, block and dig routinely share one second, 8 of
  one test match's 292 attacks share one with another attack, and it is not even
  monotonic down the file. `scout.get_plays` supplies the line number and checks
  it against the section's own line count. `set_coordinates` and `insert_before`
  then re-read the `code` field at that line and raise unless it is the one
  expected, so a stale index cannot be written through.
- **Line numbers are the file as it was opened.** Every edit and insertion in a
  run is addressed against the file as read, and all of them are applied
  together in `save`. Inserting a line shifts every line below it, so applying
  them one at a time would have each insertion invalidate the addresses of the
  rest.
- **Never rewrite an existing code.** The zone and cone letters in it are the
  human scout's reading of the rally. Disagreeing with a scout inside their own
  file is not something this pipeline does; openvolley's own writer documents
  the same limitation. Adding a line the scout left out is a different thing,
  and the caller decides when that is warranted.
- **Preserve the encoding and the line endings.** `cp1252`, opened with
  `newline=''` so a CRLF file stays CRLF, and an inserted line takes the line
  ending of the line it is inserted in front of. Text-mode round-tripping would
  silently rewrite every line in the file and show up as a total diff.
- **One save per run, not one per action.** Edits accumulate in memory. A crash
  mid-run leaves the file exactly as it was rather than half-patched.
- **Back up before the first write, and never overwrite the backup.** So
  `<name>.dvw.bak` is always the file as it was before this pipeline ever
  touched it - which is not only insurance but the thing that tells a human's
  work from the pipeline's own (see `DvwFile._load`).
- **Replace atomically.** Write a temporary file in the same directory, flush it
  to disk, then `os.replace`. An interrupted write cannot truncate a match file.
- **Verify after saving** by re-parsing with pydatavolley and asserting that
  every original line parses to the same values outside the coordinate columns,
  and that every inserted line parses to what was inserted. That catches a
  mangled delimiter, a lost `~`, an encoding slip or a patch on the wrong line
  long before anyone opens the file in DataVolley.
"""

import difflib
import os
import shutil

from .dv_grid import NO_COORDINATE

ENCODING = 'cp1252'
SCOUT_SECTION = '[3SCOUT]'
BACKUP_SUFFIX = '.bak'

# The `;`-separated fields this module is allowed to touch, and the one it reads
# to confirm it is on the right line. A scout line has many more; they are
# rejoined verbatim.
CODE_FIELD = 0
START_FIELD = 4
MID_FIELD = 5
END_FIELD = 6
MINIMUM_FIELDS = END_FIELD + 1

# Every column pydatavolley derives from the three coordinate fields. The
# post-save check asserts that everything *else* came back identical.
COORDINATE_COLUMNS = (
    'start_coordinate', 'mid_coordinate', 'end_coordinate',
    'start_coordinate_x', 'start_coordinate_y',
    'mid_coordinate_x', 'mid_coordinate_y',
    'end_coordinate_x', 'end_coordinate_y',
)

# Columns that differ between two reads of the same untouched bytes, so they say
# nothing about whether a patch was clean. `match_id` is a fresh `uuid4` that
# pydatavolley mints per call - it is not read from the file at all. Checked:
# parsing one file twice in a row gives two different values. `scout_line` is
# the row's position, added by `scout.get_plays`, and an insertion moves it for
# every line below by design.
UNSTABLE_COLUMNS = ('match_id', 'scout_line')

# Columns pydatavolley derives from the *previous* rows rather than from the line
# itself, so an inserted line legitimately changes them on the line after it.
# `attack_phase` is only set on an attack whose previous row is a set - adding
# the set is exactly what gives an attack one. These are exempted on the row
# directly after each insertion and nowhere else.
NEIGHBOUR_COLUMNS = ('attack_phase',)

# What `_origins` records for a line with no counterpart in the backup.
ADDED = 'added'         # a whole new line: this pipeline's insertion
CHANGED = 'changed'     # stands where a different line stood: treat as human


def _scout_positions(lines):
    """The file positions of the scout lines, or None without a scout section.

    Position *i* in the result is the `scout_line` value `scout.get_plays`
    reports - every non-blank line after the section marker, counted the same
    way `scout._scout_line_count` counts them, so the two cannot disagree.
    """
    try:
        section = next(i for i, line in enumerate(lines)
                       if line.strip() == SCOUT_SECTION)
    except StopIteration:
        return None
    return [i for i in range(section + 1, len(lines)) if lines[i].strip()]


def _identity(line):
    """A scout line with its coordinates blanked: what makes it *that* line.

    Coordinates are what this module writes, so a line whose coordinates this
    pipeline (or the reception pipeline) filled in is still the same line.
    """
    fields = line.rstrip('\r\n').split(';')
    if len(fields) >= MINIMUM_FIELDS:
        for field in (START_FIELD, MID_FIELD, END_FIELD):
            fields[field] = ''
    return ';'.join(fields)


def _carries_coordinates(line):
    fields = line.split(';')
    return len(fields) >= MINIMUM_FIELDS and any(
        fields[field].strip() not in ('', NO_COORDINATE)
        for field in (START_FIELD, END_FIELD))


class DvwFile:
    """One scout file held in memory, with pending edits and insertions.

    Open it, call `set_coordinates` / `insert_before` once per change, call
    `save` once at the end. Nothing reaches the disk until `save`.
    """

    def __init__(self, path, overwrite_existing=False):
        self.path = path
        self.backup_path = path + BACKUP_SUFFIX
        self.overwrite_existing = overwrite_existing
        self._load()

    def _load(self):
        with open(self.path, 'r', encoding=ENCODING, newline='') as scout_file:
            self._lines = scout_file.readlines()

        self._positions = _scout_positions(self._lines)
        if self._positions is None:
            raise ValueError(f"{self.path} has no {SCOUT_SECTION} section.")

        self._origins, self._human_coordinates = self._compare_with_backup()

        self._edits = {}        # scout_line -> {field: value}
        self._inserts = {}      # scout_line -> [fields of each line to insert]
        self.skipped = []       # (scout_line, code) left alone, human-scouted

    def _compare_with_backup(self):
        """Which current lines are the scout's, and which already carried coordinates.

        The backup is what makes both knowable: it is written once and never
        overwritten, so it is the pristine file even on the tenth run. Without
        it, a previous run's own output is indistinguishable from a human's
        work, and the choice would be between refusing to re-run an action
        after a fix and quietly wiping a scout's clicks.

        Lines are matched to the backup **by content with the coordinates
        blanked**, not by line number, because a run that inserted sets has
        shifted every line below each one. The match is a sequence alignment
        (`difflib`), so a line keeps its identity wherever it has moved to:

        - matched: the backup's line, and it carried coordinates if that did;
        - `ADDED`: no counterpart at all - a line this pipeline inserted;
        - `CHANGED`: stands where a *different* line stood in the backup, which
          only a human editing the file does. Treated as carrying a human's
          coordinates, since the safe mistake is to leave a line alone.

        Returns `(origins, human)`: for each current scout line its backup
        line number or one of the two markers, and the set of current scout
        lines whose coordinates are a human's.
        """
        current = [self._lines[p] for p in self._positions]

        pristine = None
        if os.path.exists(self.backup_path):
            with open(self.backup_path, 'r', encoding=ENCODING,
                      newline='') as backup:
                backup_lines = backup.readlines()
            positions = _scout_positions(backup_lines)
            if positions is not None:
                pristine = [backup_lines[p] for p in positions]

        if pristine is None:
            origins = list(range(len(current)))
            human = {i for i, line in enumerate(current)
                     if _carries_coordinates(line)}
            return origins, human

        matcher = difflib.SequenceMatcher(
            None, [_identity(line) for line in pristine],
            [_identity(line) for line in current], autojunk=False)
        origins = [CHANGED] * len(current)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                origins[j1:j2] = range(i1, i2)
            elif tag == 'insert':
                origins[j1:j2] = [ADDED] * (j2 - j1)

        human = set()
        for number, origin in enumerate(origins):
            if origin == CHANGED or (origin != ADDED
                                     and _carries_coordinates(pristine[origin])):
                human.add(number)
        return origins, human

    def _fields(self, scout_line):
        if not 0 <= scout_line < len(self._positions):
            raise IndexError(
                f"{self.path} has {len(self._positions)} scout lines, so there"
                f" is no line {scout_line}.")
        return self._lines[self._positions[scout_line]].rstrip('\r\n').split(';')

    def fields(self, scout_line):
        """That line's `;`-separated fields, as the file holds them."""
        return self._fields(scout_line)

    def code(self, scout_line):
        """The `code` field as it stands on that line."""
        return self._fields(scout_line)[CODE_FIELD]

    def codes(self):
        """Every scout line's `code` field, in file order.

        The file as opened, like every line number in a run - nothing queued for
        insertion appears here. That is what `set_codes.sideout_reception` wants:
        it walks back through the rally looking for the serve, the reception and
        any earlier attack, and the only lines this pipeline ever inserts are
        sets, which it steps over anyway.
        """
        return [self._fields(n)[CODE_FIELD]
                for n in range(len(self._positions))]

    def section_lines(self, section):
        """The non-blank lines of a `[3...]` header section, without endings."""
        found = []
        inside = False
        for line in self._lines:
            text = line.rstrip('\r\n')
            if text.startswith('[3'):
                if inside:
                    break
                inside = text.strip() == section
            elif inside and text.strip():
                found.append(text)
        return found

    def added_by_pipeline(self, scout_line):
        """Whether that line was inserted by this pipeline on an earlier run.

        True only for a line with no counterpart at all in the backup - see
        `_compare_with_backup`. False whenever there is no backup yet, since
        then nothing has been inserted.
        """
        self._fields(scout_line)
        return self._origins[scout_line] == ADDED

    def _check_code(self, scout_line, expect_code):
        found = self.code(scout_line)
        if found != expect_code:
            raise ValueError(
                f"{self.path} line {scout_line} carries the code {found!r}, not"
                f" the expected {expect_code!r}. The line numbering no longer"
                f" matches the file, and nothing may be written until that is"
                f" understood.")
        return found

    def set_coordinates(self, scout_line, expect_code, start=None,
                        mid=NO_COORDINATE, end=None):
        """Queue the coordinate fields of one scout line. Returns whether it will be written.

        `expect_code` is the `code` this line is believed to carry, as
        `scout.get_actions` read it. It is checked rather than trusted: a line
        number that has drifted - a re-scouted file, a stale CSV, a bug in the
        selection - is then impossible to write through, at the cost of one
        string comparison.

        `mid` defaults to `-1-1`, the file's own "no value here", because the
        pipeline reconstructs a contact and a landing and nothing in between.

        A field left as `None` is not touched. A line whose coordinates a human
        entered is left alone entirely and recorded in `skipped`, unless the file
        was opened with `overwrite_existing=True`.
        """
        fields = self._fields(scout_line)
        if len(fields) < MINIMUM_FIELDS:
            raise ValueError(
                f"{self.path} line {scout_line} has only {len(fields)} fields,"
                f" so it is not a play line and has no coordinates to set.")

        found = self._check_code(scout_line, expect_code)

        if scout_line in self._human_coordinates and not self.overwrite_existing:
            self.skipped.append((scout_line, found))
            return False

        edit = self._edits.setdefault(scout_line, {})
        for field, value in ((START_FIELD, start), (MID_FIELD, mid),
                             (END_FIELD, end)):
            if value is not None:
                edit[field] = str(value)
        return True

    def insert_before(self, scout_line, expect_code, fields):
        """Queue a new line directly above `scout_line`.

        `expect_code` is checked against the line being inserted in front of,
        exactly as `set_coordinates` checks the line it writes to, and for the
        same reason. `fields` is the new line split on `;`; it must be a play
        line with at least the coordinate fields, and is written as given.

        The line number is the file as opened: the line lands above that line
        however many other insertions this run makes elsewhere.
        """
        self._check_code(scout_line, expect_code)
        fields = [str(field) for field in fields]
        if len(fields) < MINIMUM_FIELDS or not fields[CODE_FIELD]:
            raise ValueError(
                f"refusing to insert {';'.join(fields)!r} into {self.path}: it"
                f" is not a play line.")
        if any('\n' in field or '\r' in field for field in fields):
            raise ValueError("an inserted line may not contain a line break.")
        self._inserts.setdefault(scout_line, []).append(fields)

    @property
    def pending(self):
        """How many lines `save` would change or add."""
        return len(self._edits) + sum(len(v) for v in self._inserts.values())

    def save(self, backup=True, verify=True):
        """Apply every queued edit and insertion, atomically. Returns the lines changed or added.

        Backing up and verifying are separable only so that a test can turn
        them off; a real run wants both.
        """
        if not self.pending:
            return 0

        before = _parse(self.path) if verify else None

        if backup and not os.path.exists(self.backup_path):
            shutil.copy2(self.path, self.backup_path)

        lines = list(self._lines)
        for scout_line, edit in self._edits.items():
            position = self._positions[scout_line]
            line = lines[position]

            # Split the line's terminator off before touching its fields, so a
            # CRLF file stays CRLF and the last line keeps whatever it had.
            body = line.rstrip('\r\n')
            terminator = line[len(body):]

            fields = body.split(';')
            for field, value in edit.items():
                fields[field] = value
            lines[position] = ';'.join(fields) + terminator

        # Insertions last, into a new list, so every address above was resolved
        # against the file as opened. An inserted line ends the way the line
        # below it does - or, for the file's unterminated last line, the way
        # the file's first line does.
        inserts_at = {self._positions[n]: new for n, new in self._inserts.items()}
        scout_positions = set(self._positions)
        first = lines[0]
        default_ending = first[len(first.rstrip('\r\n')):] or '\n'
        expected = []       # (scout-line row in the saved file, code)
        written = []
        row = 0
        for position, line in enumerate(lines):
            ending = line[len(line.rstrip('\r\n')):] or default_ending
            for fields in inserts_at.get(position, ()):
                written.append(';'.join(fields) + ending)
                expected.append((row, fields[CODE_FIELD]))
                row += 1
            written.append(line)
            if position in scout_positions:
                row += 1

        temporary = self.path + '.tmp'
        with open(temporary, 'w', encoding=ENCODING, newline='') as out:
            out.writelines(written)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, self.path)

        changed = self.pending
        if verify:
            _verify(self.path, before, expected)
        # Re-read, so the line numbers and the comparison with the backup
        # describe the file as it now is rather than as it was opened.
        self._load()
        return changed


def _parse(path):
    """The file as pydatavolley reads it. Imported late - it is not cheap."""
    from .scout import get_plays
    return get_plays(path)


def _verify(path, before, inserted):
    """Raise unless the saved file is `before` plus coordinates and `inserted`.

    `inserted` is `(row, code)` for every line added, with `row` its position
    among the saved file's scout lines. The check:

    - the saved file has exactly that many more rows, with the same columns;
    - each inserted row parses back to its code;
    - with the inserted rows taken out, every original row is unchanged in every
      column except the coordinates - and, on the row directly after an
      insertion only, the columns pydatavolley derives from the row above.

    It is not a formality: the failure this guards against - a delimiter count
    changed, a `~` lost, an encoding slip, a patch or an insertion on the wrong
    line - is invisible in the output of a run and shows up days later as a
    file DataVolley will not open.

    The file has already been written when this runs, which is why the backup
    exists: the message says where to find the original.
    """
    after = _parse(path)
    original = path + BACKUP_SUFFIX

    if (list(before.columns) != list(after.columns)
            or len(after) != len(before) + len(inserted)):
        raise ValueError(
            f"{path} no longer parses the same shape after patching"
            f" ({len(before)} rows x {len(before.columns)} columns before,"
            f" {len(after)} x {len(after.columns)} after, with {len(inserted)}"
            f" line(s) inserted). The original is at {original}.")

    for row, code in inserted:
        if after['code'].iloc[row] != code:
            raise ValueError(
                f"{path}: the line inserted as {code!r} reads back as"
                f" {after['code'].iloc[row]!r} at scout line {row}. The"
                f" original is at {original}.")

    inserted_rows = {row for row, _ in inserted}
    kept = after.drop(index=after.index[sorted(inserted_rows)]).reset_index(drop=True)
    before = before.reset_index(drop=True)

    # The kept rows that sit directly below an insertion, in kept numbering.
    below_insertion = set()
    for row in inserted_rows:
        if row + 1 < len(after) and row + 1 not in inserted_rows:
            below_insertion.add(row + 1 - sum(1 for r in inserted_rows if r < row + 1))

    ignored = COORDINATE_COLUMNS + UNSTABLE_COLUMNS
    differing = []
    for column in before.columns:
        if column in ignored:
            continue
        old, new = before[column], kept[column]
        if column in NEIGHBOUR_COLUMNS:
            keep = [i for i in range(len(before)) if i not in below_insertion]
            old, new = old.iloc[keep], new.iloc[keep]
        # Element by element rather than `Series.equals`: an inserted set can
        # give a column its first real value and change its dtype, which says
        # nothing about the rows that were already there.
        same = (old == new).fillna(False).astype(bool) | (old.isna() & new.isna())
        if not same.all():
            differing.append(column)

    if differing:
        raise ValueError(
            f"{path} changed in columns that hold no coordinates:"
            f" {', '.join(differing)}. Only fields {START_FIELD}, {MID_FIELD}"
            f" and {END_FIELD} should have been touched, and lines only added."
            f" The original is at {original}.")
