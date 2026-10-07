"""The set code that goes in front of an attack: which one, and what it says.

Pure string handling on scout-line fields - no I/O, no pandas. `dvw_edit` does
the inserting and `attacks.write_to_scout_file` decides where; this module only
answers "should this attack have a set added before it, and if so, what is the
line".

A set is its own line in `[3SCOUT]`, directly before the attack it fed:

    *13ET+K1F;;;;;;;16.05.22;1;6;2;1;1078;;4;10;6;14;3;13;2;16;9;7;1;4;
    *06AT#X5~42~H2;p;r;;;;;16.05.22;1;6;2;1;1078;;4;10;6;14;3;13;2;16;9;7;1;4;

The code reads: team (`*` home, `a` visiting), the setter's two-digit number,
`E` for set, the tempo, the grade, a two-character setter call (`~~` for
none), and the target attack - `F` front, `C` centre, `B` back, `P` pipe, `S`
setter. DataVolley can carry zones after that; they are not written.

Every field after the code is copied from the attack's line, and none of it is
a guess. **The clock time and `video_time` are the attack's own**, as are the
set number and video file - the attack's `video_time` is hand-synced to the
setter's touch in these files, and 232 of the 235 sets in `&svk-ork_test.dvw`
share their attack's `video_time` exactly, 212 its clock time. The setter
positions and the twelve players on court are the rotation, which cannot change between a set and the attack it
fed. Fields 1-3 are left empty: DataVolley computes them for attacks and leaves
them blank on sets.

The rules, each checked against hand-scouted files before being written down
(`Elitserien 25-26/`, 2,817 same-team sets directly before an attack):

- **The tempo is the attack's.** The fifth character of the attack code (`X5` is
  scouted `AT`, so its set is `ET`). The hand-scouted sets agree 98% of the
  time, and nearly every disagreement is a set scouted `H` before a faster
  attack.
- **The grade is always `+`** for now.
- **No set before `PR`, `PP` or `P2`.** An attack on the opponent's overpass, a
  setter tip and a second-hand attack are, by definition, not set.
- **A quick attack gets its setter call, but only on a sideout:** `X1` ->
  `K1`, `X2` -> `K2`, `X7` -> `K7`, and only if the file's own `[3SETTERCALL]`
  table defines that call. A code DataVolley has no definition for is worse
  than none. The sideout restriction is the scouts' own practice: of the 712
  quick calls in the hand-scouted files, **689 (97%) sit on a sideout attack**
  - `K1` 529 of 540, `K7` 143 of 154, `K2` 17 of 18. A call names the play the
  setter ran off a serve receive; in transition there is no called play to
  name.
- **A set made from out by a sideline gets the shifted call**, `KM` (shifted to
  zone 2) or `KP` (shifted to zone 4), whatever the attack combination is - a
  shifted setter is a property of where the setter was standing, not of who
  they set to. It needs a measured set location, a sideout, and a reception the
  scout graded `#`, `+` or `!`; the caller (`attacks.shifted_setter_call`)
  decides "out by a sideline" from the reconstruction and this module applies
  the rest. All three conditions are the hand-scouted files' own: **all 71
  shifted calls there are on sideout attacks**, 32 `KM` and 39 `KP` without one
  exception, and 70 of the 71 follow a reception graded `#`, `+` or `!`. The
  shifted call **wins over a quick one** when both would apply, because it is
  the one carrying information the line does not otherwise hold - `X1` already
  says the attack was a front quick, while nothing else records that the setter
  was pulled to the sideline.
- **The target attack is the attack combination's**, from the ninth field of
  the file's own `[3ATTACKCOMBINATION]` table (`X5;4;R;T;Shoot in 4;;…;F;;`).
  That is where DataVolley takes it from: across the hand-scouted files, 2,145
  of the 2,153 sets that carry a target agree with their attack's table entry,
  and the 8 that do not are all scouted `F` (six of them before an `X1`). An attack with no combination code, or one the
  table gives no target (`-`), gets none, and the code then ends at the call.
- **The setter is the player in the setter's zone**, read off the attack's own
  line: field 9 is the home setter's zone (1-6), field 10 the visiting one's,
  and fields 14-19 / 20-25 are the players in zones 1-6. Checked on both teams
  of the hand-scouted file: `*13ET+` with the home setter in zone 6 and home
  zone 6 holding 13, `a16ET+` with the visiting setter in zone 2 and visiting
  zone 2 holding 16.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from .dv_grid import NO_COORDINATE

SET_SKILL = 'E'
SET_EVALUATION = '+'

# Where the skill and the grade sit in a play code: `*06AT#X5~42~H2` is team,
# two-digit number, skill `A`, tempo `T`, grade `#`, combination `X5`.
SKILL_INDEX = 3
EVALUATION_INDEX = 5

SERVE_SKILL = 'S'
RECEPTION_SKILL = 'R'
ATTACK_SKILL = 'A'

# Attack combinations that are never set. `PR` is an attack on the opponent's
# freeball, `PP` a setter tip and `P2` a second-hand attack.
UNSET_ATTACKS = ('PR', 'PP', 'P2')

# The setter call a quick attack implies. Only added on a sideout - see the
# module docstring for what the hand-scouted files measure.
QUICK_SETTER_CALLS = {'X1': 'K1', 'X2': 'K2', 'X7': 'K7'}

# The setter call a set made from out by a sideline implies, keyed by the
# front-row zone that sideline belongs to in the setting team's **own**
# numbering: zone 4 is the left-front corner and zone 2 the right-front one
# (`dv_grid._NEAR_HALF_ZONES`, itself a measured layout). The names are the
# file's own - `[3SETTERCALL]` reads `KM;;shifted to 2` and `KP;;Shifted to 4`.
SHIFTED_SETTER_CALLS = {2: 'KM', 4: 'KP'}

# The reception grades a shifted set is added after: a pass good enough that the
# setter played a real ball rather than scrambling one up. 70 of the 71 shifted
# calls in the hand-scouted files follow one of these. A tuple rather than a
# string because `'' in '#+!'` is True, and a truncated code would then pass.
SHIFTED_RECEPTION_GRADES = ('#', '+', '!')

# DataVolley's tempo letters: high, half, quick, tense, super, fast, other.
TEMPOS = 'HMQTUNO'

# Scout-line fields read here.
CODE_FIELD = 0
COMPUTED_FIELDS = (1, 2, 3)         # point/attack phase - blank on a set
COORDINATE_FIELDS = (4, 5, 6)       # start, mid, end
SETTER_ZONE_FIELD = {'*': 9, 'a': 10}
FIRST_PLAYER_FIELD = {'*': 14, 'a': 20}

# The target-attack letters DataVolley uses: front, centre, back, pipe, setter.
TARGET_ATTACKS = 'FCBPS'
TARGET_FIELD = 8                    # in an `[3ATTACKCOMBINATION]` line

SETTER_CALL_SECTION = '[3SETTERCALL]'
ATTACK_COMBINATION_SECTION = '[3ATTACKCOMBINATION]'


def is_play(code):
    """Whether a code is a player's action: team, two-digit number, skill."""
    return (len(code) >= 4 and code[0] in SETTER_ZONE_FIELD
            and code[1:3].isdigit())


def is_set_by(code, team):
    """Whether a code is a set by `team` (`*` or `a`)."""
    return (is_play(code) and code[0] == team
            and code[SKILL_INDEX] == SET_SKILL)


def sideout_reception(codes, line):
    """The reception this attack sided out of, or None when it is not a sideout.

    `codes` is every `code` in the `[3SCOUT]` section in file order - as the
    file was opened, since nothing this pipeline inserts is a serve, a reception
    or an attack - and `line` is the attack's position in it.

    A **sideout attack** is the receiving team's first swing of the rally: the
    rally has a reception, it is by the same team as the attack, and no attack
    by either team came between the two. Everything after that is transition,
    where the rally has been rallied and the setter is no longer running a
    called play off a serve receive.

    That is what the hand-scouted files show setter calls to mean. Walking back
    from each of their 2,817 sets, 689 of the 712 quick calls (97%) and **all 71
    shifted calls** sit on an attack this function calls a sideout, against
    1,210 of the 2,084 uncalled sets that do not.

    The walk back ends at the rally's serve, which every rally has and which is
    above the reception, so the point and rotation lines between rallies are
    never reached. Lines that are not plays at all are stepped over rather than
    stopped at: `$$&` lines - an action by an unknown player, 89 of them in
    `&svk-ork_test.dvw` - sit inside a rally, and treating one as a boundary
    would quietly call a real sideout attack a transition one.
    """
    team = codes[line][0]
    reception = None
    for above in reversed(codes[:line]):
        if not is_play(above):
            continue
        skill = above[SKILL_INDEX]
        if skill == SERVE_SKILL:
            break
        if skill == ATTACK_SKILL:
            return None
        if skill == RECEPTION_SKILL:
            reception = above
    if reception is None or reception[0] != team:
        return None
    return reception


def setter_calls(section_lines):
    """The setter calls a file defines, from its `[3SETTERCALL]` lines."""
    return {line.split(';')[0] for line in section_lines if ';' in line}


def attack_targets(section_lines):
    """Attack combination -> target-attack letter, from `[3ATTACKCOMBINATION]` lines.

    Combinations without a usable letter (`-`, or a short line) are left out.
    """
    targets = {}
    for line in section_lines:
        fields = line.split(';')
        if len(fields) > TARGET_FIELD and fields[TARGET_FIELD] in TARGET_ATTACKS:
            targets[fields[0]] = fields[TARGET_FIELD]
    return targets


def setter_number(fields, team):
    """The setter's number as two digits, or None when the line cannot say.

    Read off the rotation on the line itself, so it is the setter on court at
    that moment - after substitutions and rotations alike.
    """
    zone_field = SETTER_ZONE_FIELD[team]
    first = FIRST_PLAYER_FIELD[team]
    if len(fields) < first + 6:
        return None
    zone = fields[zone_field].strip()
    if not zone.isdigit() or not 1 <= int(zone) <= 6:
        return None
    player = fields[first + int(zone) - 1].strip()
    if not player.isdigit():
        return None
    return f'{int(player):02d}'


@dataclass(frozen=True)
class SetDecision:
    """What to do in front of one attack.

    Exactly one of three things: `code` is the set to add; `existing` says a set
    is already there; otherwise `reason` says why nothing is added. The reasons
    are split into the expected (`skipped_by_rule`: the attack is never set) and
    the ones worth reporting, where a set probably belongs but cannot be written
    from what the line says.
    """

    code: Optional[str] = None
    existing: bool = False
    reason: Optional[str] = None
    skipped_by_rule: bool = False


def _setter_call(combination, reception, shifted_call):
    """The call this attack's set carries, before checking the file defines it.

    None off a sideout, whatever else is true. On a sideout the shifted call
    comes first when there is one and the pass was good enough for the setter to
    have played a real ball (`SHIFTED_RECEPTION_GRADES`); otherwise a quick
    attack gets the call its combination implies.

    The shifted call winning is deliberate. The two can only collide when a
    shifted setter still ran a quick, and of the two codes only `KM`/`KP` says
    something the line does not already hold - `X1` in the attack code is
    already the front quick.
    """
    if reception is None:
        return None
    grade = reception[EVALUATION_INDEX:EVALUATION_INDEX + 1]
    if shifted_call is not None and grade in SHIFTED_RECEPTION_GRADES:
        return shifted_call
    return QUICK_SETTER_CALLS.get(combination)


def decide(attack_fields, previous_code, calls, targets, reception=None,
           shifted_call=None):
    """Whether a set should be added in front of this attack, and which.

    `attack_fields` is the attack's line split on `;`, `previous_code` the code
    of the line directly above it (None at the top of the section), `calls`
    the setter calls the file defines and `targets` its attack combinations'
    target letters (`attack_targets`).

    "Already set" means the line directly above is a set **by the same team**.
    A set by the other team there is not this attack's set - it is the ball
    they put over - and does not count.

    `reception` is the reception this attack sided out of (`sideout_reception`)
    or None, and `shifted_call` the call the measured set location implies
    (`attacks.shifted_setter_call`) or None. **A setter call is only ever added
    on a sideout**, so a caller that passes neither adds none - which is the
    honest answer when the rally context is not known, since a call names a play
    run off a serve receive.
    """
    code = attack_fields[CODE_FIELD]
    team = code[0]
    if previous_code is not None and is_set_by(previous_code, team):
        return SetDecision(existing=True)

    combination = code[6:8]
    if combination in UNSET_ATTACKS:
        return SetDecision(reason=f'{combination} is never set',
                           skipped_by_rule=True)

    tempo = code[4:5]
    if tempo not in TEMPOS:
        return SetDecision(reason=f'no tempo in the attack code ({tempo!r})')

    setter = setter_number(attack_fields, team)
    if setter is None:
        return SetDecision(reason='the line has no setter on court')
    if setter == code[1:3]:
        # The rotation's setter hit it, so someone else set it - and nothing
        # on the line says who.
        return SetDecision(reason=f'the attacker, {setter}, is the setter on court')

    call = _setter_call(combination, reception, shifted_call)
    if call not in calls:
        # A call DataVolley has no definition for in this file is worse than
        # none at all.
        call = None

    # The call slot is two characters wide, so a target with no call behind it
    # needs `~~` holding the place: `*13ET+~~F`, not `*13ET+F`.
    target = targets.get(combination, '')
    call = call or ('~~' if target else '')

    return SetDecision(code=f'{team}{setter}{SET_SKILL}{tempo}{SET_EVALUATION}'
                            f'{call}{target}')


def set_line_fields(attack_fields, set_code, start=None):
    """The set's line, as fields: the attack's line with the set in front.

    `start` is the set location as a formatted grid index, or None. With one,
    the coordinates are written the way a hand-scouted set carries them -
    `start;-1-1;-1-1` - and without, the three fields stay empty like every
    other set in these files.
    """
    fields = list(attack_fields)
    fields[CODE_FIELD] = set_code
    for field in COMPUTED_FIELDS:
        fields[field] = ''
    coordinates = (('', '', '') if start is None
                   else (start, NO_COORDINATE, NO_COORDINATE))
    for field, value in zip(COORDINATE_FIELDS, coordinates):
        fields[field] = value
    return fields
