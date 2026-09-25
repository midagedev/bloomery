#!/usr/bin/env python3
"""The prediction card: what a box run is for, written down before the run.

A run that holds the machine-wide lease (tools/ref/lease.sh `lease_take`, tools/ref/lease-hold.sh)
starts only with a card that names the one term it measures, the band the derivation predicts, what
must happen inside the measured window, and the next action for each outcome. A run whose outcomes
all lead to the same action, or an A/B whose predicted effect sits inside the ruler at its round
count, is refused on paper, before it costs box time (AGENTS.md 「Derive first, measure the gap」,
「Know the ruler」). A job that takes the lease only to have the box to itself (a conversion, a
build that must not share the cores) carries an `exclusive` card instead: why, and for how long.
The card changes when a run may start, never what a runner measures.

  card.py check <file> [--rounds N] [--round-minutes M]
  card.py lease <path> [--rounds N] [--round-minutes M]

check   validates <file>, any path, and prints `card: ok …`, or `card: refused <NAME> (rc <n>): …` on
        stderr with the exit code below.
lease   what lease_take runs. <path> is tree-relative, docs/cards/<slug>.card (slug: letters, digits,
        `.`, `_`, `-`), and names a file of this tree; docs/cards/example-*.card are the format's
        documentation, not a run's card, and are refused. After the same checks it prints the card's
        path, sha256 and body as `[lease]` lines, so the run's log carries its prediction before
        the measurement starts.
--rounds N         the runner's actual round count: an ab card is checked at N instead of its own
                   `rounds`, and a difference is printed. The other kinds take no rounds and ignore it.
--round-minutes M  the runner's box minutes per round: a ruler refusal then prices the rounds that
                   would resolve it.

Format. UTF-8 text, one `key: value` per line. A line whose first non-blank character is `#` is a
comment and blank lines are skipped; a line that starts with a space or a tab continues the value of
the line right above it (a key line or another continuation), joined with one space. Each key once.
An unknown key, a key the card's kind does not take, a missing or empty required key, or a value
that does not parse is CARD_MALFORMED.

  kind          ab | profile | calibration | baseline | exclusive
  question      what the run answers
  term          the one term the run measures, with its conditions (`tok/s @ n=N, depth D, card`)
  unit          the unit of `predict`; `%` for ab
  predict       the band the derivation gives: `lo..hi`, decimal numbers with lo <= hi, optionally
                followed by the unit. For ab it is the effect in percent, signed: (changed - base) /
                base of `term`, so + means the changed arm's value is the larger one
  rounds        ab only, required: rounds per arm; a round runs every arm once
  sd            ab only, optional: the scatter of one run in percent, then where it comes from, e.g.
                `sd: 0.45 % <rig-log entry>, paired ratio SD over 5 rounds`; a number with no source
                is malformed. Default: 0.6 %, AGENTS.md 「Know the ruler」
  condition     what must happen inside the measured window, and how the run ensures it
  decide-in     the next action when the measured value falls inside the band
  decide-below  the next action when it falls below lo
  decide-above  the next action when it falls above hi
  reason        calibration, baseline and exclusive only, required: which constant or reference row
                the run sets, and why now; for exclusive, why the job needs the box to itself, and
                why now
  minutes       exclusive only, required: the expected hold in minutes, `n` or `lo..hi`, optionally
                followed by `min`; a hold past 30 minutes is printed as needing the user's approval

An exclusive card takes kind, reason and minutes and nothing else: it predicts nothing, so it has no
prediction, decision or ruler check. Every other kind takes kind, question, term, unit, predict,
condition and the three decisions, plus the keys named above for it.

Checks, in this order. The text parses (CARD_MALFORMED). At least two of the three decisions differ,
compared with case and whitespace folded (CARD_UNDECIDABLE). An ab card meets the ruler: the 95 %
half-width of a difference of two arm means at N rounds per arm,

    h = t(0.975, 2N - 2) * sd * sqrt(2 / N)

the pooled two-sample interval tools/gpu-ab.py reports (its t table is the one used here), which is
the form that gives AGENTS.md's ±1.0 % at four rounds and ±0.8 % at six with sd 0.6 % — the normal
quantile 1.96 in place of t gives 0.83 % and 0.68 %. The card is refused (CARD_UNDER_RULER) when the
band's edge nearest zero is inside h, |edge| <= h; a band that contains 0 has its nearest edge at 0,
and no round count resolves it. The refusal names the smallest N that resolves the band and, given
--round-minutes, the box minutes those rounds cost.

Exit codes (never 75, which is lease contention):
   0  the card passes
  64  a bad command line
  65  CARD_MALFORMED    the text does not parse against the format above
  66  CARD_ABSENT       (lease) no card named, a path that is not docs/cards/<slug>.card of this
                        tree, an example card, or no such file; (check) no such file
  67  CARD_UNDECIDABLE  the three decisions are one action
  68  CARD_UNDER_RULER  an ab card's predicted effect is inside the ruler at its rounds
  70  the t table in tools/gpu-ab.py does not load
"""
import hashlib
import importlib.util
import math
import os
import re
import sys

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

USAGE, MALFORMED, ABSENT, UNDECIDABLE, UNDER_RULER, SOFTWARE = 64, 65, 66, 67, 68, 70
NAMES = {MALFORMED: 'CARD_MALFORMED', ABSENT: 'CARD_ABSENT', UNDECIDABLE: 'CARD_UNDECIDABLE',
         UNDER_RULER: 'CARD_UNDER_RULER'}

KINDS = ('ab', 'profile', 'calibration', 'baseline', 'exclusive')
PREDICTION = ('question', 'term', 'unit', 'predict', 'condition', 'decide-in', 'decide-below',
              'decide-above')
REQUIRED = {'ab': ('kind',) + PREDICTION + ('rounds',), 'profile': ('kind',) + PREDICTION,
            'calibration': ('kind',) + PREDICTION + ('reason',),
            'baseline': ('kind',) + PREDICTION + ('reason',),
            'exclusive': ('kind', 'reason', 'minutes')}
OPTIONAL = {'ab': ('sd',)}
KEYS = {'kind', 'rounds', 'sd', 'reason', 'minutes'} | set(PREDICTION)
DECISIONS = ('decide-in', 'decide-below', 'decide-above')
SD_DEFAULT = (0.6, 'the default, AGENTS.md 「Know the ruler」: same-binary runs scatter with SD 0.6 % (18 runs)')
ROUNDS_CAP = 100000

NUM = r'[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?'
BAND = re.compile(rf'({NUM})\s*\.\.\s*({NUM})\s*(\S+)?')
MINUTES = re.compile(rf'({NUM})(?:\s*\.\.\s*({NUM}))?\s*(?:min)?')
APPROVAL_MINUTES = 30
SD = re.compile(r'(\d+(?:\.\d+)?)\s*(%?)\s*(.*)')
KEY_LINE = re.compile(r'([a-z][a-z-]*)\s*:\s*(.*)')
CARD_PATH = re.compile(r'docs/cards/[A-Za-z0-9][A-Za-z0-9._-]*\.card')


class Refused(Exception):
    def __init__(self, code, lines):
        super().__init__(code)
        self.code = code
        self.lines = lines


def parse(text):
    """The card's fields, key -> (line number, value), or Refused(CARD_MALFORMED) naming every bad line."""
    fields, errors = {}, []
    above = None  # the key a continuation line extends; '' after a bad line, whose continuations are dropped
    for no, raw in enumerate(text.split('\n'), 1):
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith('#'):
            above = None
            continue
        if line[0] in ' \t':
            if above is None:
                errors.append(f'line {no}: an indented line continues no key line above it')
            elif above:
                at, value = fields[above]
                fields[above] = (at, f'{value} {line.strip()}'.strip())
            continue
        m = KEY_LINE.fullmatch(line)
        if not m:
            errors.append(f'line {no}: not `key: value`: {line.strip()!r}')
            above = ''
            continue
        key, value = m.group(1), m.group(2).strip()
        if key not in KEYS:
            errors.append(f'line {no}: unknown key {key!r}')
            above = ''
        elif key in fields:
            errors.append(f'line {no}: {key!r} again (first on line {fields[key][0]})')
            above = ''
        else:
            fields[key] = (no, value)
            above = key
    if errors:
        raise Refused(MALFORMED, errors)
    return fields


def band(value, unit):
    m = BAND.fullmatch(value)
    if not m:
        return None, f'`predict: {value}` is not a band lo..hi of decimal numbers'
    lo, hi, named = float(m.group(1)), float(m.group(2)), m.group(3)
    if named is not None and named != unit:
        return None, f'`predict: {value}` names the unit {named!r}; the card says `unit: {unit}`'
    if lo > hi:
        return None, f'`predict: {value}`: lo {lo:g} is above hi {hi:g}'
    return (lo, hi), None


def validate(fields):
    """The checked card as a dict, or Refused(CARD_MALFORMED) naming every problem."""
    errors = []
    value = {k: v for k, (_, v) in fields.items()}
    kind = value.get('kind')
    if kind is None:
        raise Refused(MALFORMED, [f'no `kind` ({", ".join(KINDS)})'])
    if kind not in KINDS:
        raise Refused(MALFORMED, [f'line {fields["kind"][0]}: unknown kind {kind!r} ({", ".join(KINDS)})'])
    for k in REQUIRED[kind]:
        if k not in value:
            errors.append(f'no `{k}` (kind {kind} requires it)')
        elif not value[k]:
            errors.append(f'line {fields[k][0]}: `{k}` is empty')
    allowed = set(REQUIRED[kind]) | set(OPTIONAL.get(kind, ()))
    for k, (at, _) in sorted(fields.items(), key=lambda kv: kv[1][0]):
        if k not in allowed:
            errors.append(f'line {at}: kind {kind} takes no `{k}`')
    card = {'kind': kind, 'value': value}
    unit = value.get('unit', '')
    if kind == 'ab' and unit and unit != '%':
        errors.append(f'line {fields["unit"][0]}: an ab card predicts its effect in percent (`unit: %`), got {unit!r}')
    if 'predict' in allowed and value.get('predict'):
        card['band'], err = band(value['predict'], unit)
        if err:
            errors.append(f'line {fields["predict"][0]}: {err}')
    if kind == 'ab' and value.get('rounds'):
        r = value['rounds']
        if re.fullmatch(r'\d+', r) and int(r) >= 1:
            card['rounds'] = int(r)
        else:
            errors.append(f'line {fields["rounds"][0]}: `rounds: {r}` is not a positive integer')
    if kind == 'ab':
        card['sd'] = SD_DEFAULT
        if 'sd' in value:
            m = SD.fullmatch(value['sd'])
            if not m or float(m.group(1)) <= 0:
                errors.append(f'line {fields["sd"][0]}: `sd: {value["sd"]}` is not a positive percentage and its source')
            elif not m.group(3).strip():
                errors.append(f'line {fields["sd"][0]}: `sd: {value["sd"]}` names no source (an override says where it comes from)')
            else:
                card['sd'] = (float(m.group(1)), m.group(3).strip())
    if kind == 'exclusive' and value.get('minutes'):
        m = MINUTES.fullmatch(value['minutes'])
        lo = float(m.group(1)) if m else 0.0
        hi = float(m.group(2)) if m and m.group(2) else lo
        if not m or lo <= 0 or lo > hi:
            errors.append(f'line {fields["minutes"][0]}: `minutes: {value["minutes"]}` is not a positive '
                          'number of minutes or a band lo..hi of them')
        else:
            card['minutes'] = (lo, hi)
    if errors:
        raise Refused(MALFORMED, errors)
    return card


def fold(text):
    return re.sub(r'\s+', ' ', text).strip().rstrip('.').casefold()


def t_table():
    path = os.path.join(TREE, 'tools', 'gpu-ab.py')
    try:
        spec = importlib.util.spec_from_file_location('gpu_ab', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.t975
    except Exception as e:  # a missing or broken gpu-ab.py must stop the check, never skip it
        print(f'card.py: the t table ({path}, t975) does not load: {e}', file=sys.stderr)
        sys.exit(SOFTWARE)


def half_width(t975, sd, n):
    df = 2 * n - 2
    return t975(df) * sd * math.sqrt(2 / n), t975(df), df


def ruler(card, runner_rounds, round_minutes):
    """The ab card's ruler line, or Refused(CARD_UNDER_RULER) with the rounds that resolve it."""
    notes = []
    n = card['rounds']
    source = 'the card'
    if runner_rounds is not None:
        if runner_rounds != n:
            notes.append(f'note: the runner runs {runner_rounds} rounds and the card says {n}: checked at {runner_rounds}')
        n, source = runner_rounds, 'the runner'
    sd, sd_source = card['sd']
    lo, hi = card['band']
    edge = 0.0 if lo <= 0 <= hi else min(abs(lo), abs(hi))
    if n < 2:
        raise Refused(UNDER_RULER, notes + [f'{n} round has no interval: an ab run needs 2 rounds or more'])
    t975 = t_table()
    h, t, df = half_width(t975, sd, n)
    ruler_text = f'h = {h:.3f} % at {n} rounds ({source}; t {t:.3f}, df {df}; sd {sd:g} % — {sd_source})'
    if edge > h:
        return notes, f'{ruler_text}; the band\'s edge nearest zero, {edge:g} %, is outside h'
    lines = notes + [f'{ruler_text} >= the band\'s edge nearest zero, {edge:g} %: this run cannot tell '
                     'the prediction from no effect']
    if edge == 0:
        lines.append(f'the band {lo:g}..{hi:g} contains 0, so its nearest edge is 0: no round count resolves it')
        raise Refused(UNDER_RULER, lines)
    need = next((m for m in range(2, ROUNDS_CAP + 1) if half_width(t975, sd, m)[0] < edge), None)
    if need is None:
        lines.append(f'more than {ROUNDS_CAP} rounds would be needed to resolve {edge:g} %')
        raise Refused(UNDER_RULER, lines)
    lines.append(f'resolves at {need} rounds: h = {half_width(t975, sd, need)[0]:.3f} % < {edge:g} %')
    if round_minutes is None:
        lines.append('box minutes: unknown — the runner does not state its minutes per round (ROUND_MINUTES, --round-minutes)')
    else:
        lines.append(f'box minutes at {need} rounds: {need} x {round_minutes:g} = {need * round_minutes:g} min '
                     '(the runner\'s minutes per round)')
    raise Refused(UNDER_RULER, lines)


def check(text, runner_rounds, round_minutes):
    """(notes, verdict) for a card that passes; Refused otherwise."""
    card = validate(parse(text))
    if card['kind'] == 'exclusive':
        lo, hi = card['minutes']
        notes = []
        if hi > APPROVAL_MINUTES:
            notes.append(f'note: an expected hold over {APPROVAL_MINUTES} minutes needs the user\'s approval before '
                         'it starts (AGENTS.md 「Never」)')
        held = f'{lo:g}' if lo == hi else f'{lo:g}..{hi:g}'
        return notes, f'ok kind=exclusive minutes={held} (no prediction: the lease is held for the box, not a measurement)'
    if len({fold(card['value'][k]) for k in DECISIONS}) < 2:
        raise Refused(UNDECIDABLE, ['decide-in, decide-below and decide-above are one action: '
                                    'no outcome of this run changes what happens next'])
    if card['kind'] != 'ab':
        return [], f'ok kind={card["kind"]} (no ruler: not an ab card)'
    notes, line = ruler(card, runner_rounds, round_minutes)
    return notes, f'ok kind=ab {line}'


def refuse(e, where, pre):
    print(f'{pre}card: refused {NAMES[e.code]} (rc {e.code}): {where}', file=sys.stderr)
    for line in e.lines:
        print(f'{pre}card:   {line}', file=sys.stderr)
    return e.code


def read(path):
    try:
        with open(path, 'rb') as f:
            data = f.read()
    except OSError as e:
        raise Refused(ABSENT, [f'cannot read it: {e.strerror}'])
    try:
        return data, data.decode('utf-8')
    except UnicodeDecodeError as e:
        raise Refused(MALFORMED, [f'not UTF-8 text: {e}'])


def lease_path(arg):
    """The card file lease_take may use: docs/cards/<slug>.card inside this tree, else Refused(CARD_ABSENT)."""
    if not arg:
        lines = ['no card: a run under the lease needs BLOOMERY_LEASE_CARD=docs/cards/<slug>.card, '
                 'passed through BLOOMERY_BOX_ENV (the format: tools/ref/card.py)']
        if os.environ.get('BLOOMERY_CARD'):
            lines.append(f'BLOOMERY_CARD={os.environ["BLOOMERY_CARD"]} is set: that name is tools/box.sh\'s '
                         'card pick (3090, a6000, both); the lease\'s card goes in BLOOMERY_LEASE_CARD')
        raise Refused(ABSENT, lines)
    if not CARD_PATH.fullmatch(arg):
        raise Refused(ABSENT, [f'{arg!r} is not a tree-relative docs/cards/<slug>.card path '
                               '(slug: letters, digits, `.`, `_`, `-`)'])
    if os.path.basename(arg).startswith('example-'):
        raise Refused(ABSENT, [f'{arg} is an example: the format\'s documentation, not a prediction; '
                               'write docs/cards/<slug>.card for this run'])
    path = os.path.realpath(os.path.join(TREE, arg))
    if os.path.dirname(path) != os.path.realpath(os.path.join(TREE, 'docs', 'cards')):
        raise Refused(ABSENT, [f'{arg} resolves to {path}, outside {TREE}/docs/cards'])
    if not os.path.isfile(path):
        raise Refused(ABSENT, [f'no such card in {TREE}'])
    return path


def main(argv):
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding='utf-8')
    usage = 'usage: card.py check <file> | lease <docs/cards/<slug>.card> [--rounds N] [--round-minutes M]'
    if len(argv) < 2 or argv[0] not in ('check', 'lease'):
        print(usage, file=sys.stderr)
        return USAGE
    mode, target, rest = argv[0], argv[1], argv[2:]
    rounds = minutes = None
    while rest:
        opt = rest.pop(0)
        if opt not in ('--rounds', '--round-minutes') or not rest:
            print(f'card.py: {opt!r} is not an option here; {usage}', file=sys.stderr)
            return USAGE
        arg = rest.pop(0)
        if opt == '--rounds':
            if not re.fullmatch(r'\d+', arg) or int(arg) < 1:
                print(f'card.py: --rounds takes a positive integer, got {arg!r}', file=sys.stderr)
                return USAGE
            rounds = int(arg)
        else:
            try:
                minutes = float(arg)
            except ValueError:
                minutes = -1.0
            if not minutes > 0 or math.isinf(minutes):
                print(f'card.py: --round-minutes takes a positive number, got {arg!r}', file=sys.stderr)
                return USAGE
    pre = '[lease] ' if mode == 'lease' else ''
    try:
        path = lease_path(target) if mode == 'lease' else target
        data, text = read(path)
        head = f'{pre}card: {target} sha256={hashlib.sha256(data).hexdigest()}'
        notes, verdict = check(text, rounds, minutes)
    except Refused as e:
        where = target if mode == 'check' else f'BLOOMERY_LEASE_CARD={target}' if target else 'BLOOMERY_LEASE_CARD is empty'
        return refuse(e, where, pre)
    print(head)
    if mode == 'lease':
        for line in text.rstrip('\n').split('\n'):
            print(f'{pre}card | {line.rstrip()}')
    for line in notes:
        print(f'{pre}card: {line}')
    print(f'{pre}card: {verdict}')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
