#!/usr/bin/env python3
"""Interleaved GPU A/B across trees and levers: the lead's timing rounds as one tool.

An arm is a (tree, environment) pair. Every round runs every arm once, through the tree's own
timing recipe — `just time-gpu-generate` — so every arm takes the machine-wide
lease and prints its witness blocks through tools/ref/time-gate.sh, like a hand-run of that recipe.
The order rotates by one arm per round (round r starts at arm r-1): in a fixed order a round's
first arm read slow (AGENTS.md), and rotation spreads that over every arm. Only interleaved,
same-window numbers are compared, and only as each arm's difference from the first arm.

  gpu-ab.py run [--dry-run] --out DIR --rounds N --recipe RECIPE [--recipe ...] --arm NAME=TREE[:K=V,K=V] ...
  gpu-ab.py summary DIR
  gpu-ab.py summary --instrument generate NAME=LOG_GLOB ...

run      one --arm per arm, the first is the reference. TREE is a path or a directory name beside this
         repository's own; its box directory is ~/repo/<tree name>. K=V pairs reach the timed binary
         through tools/box.sh's BLOOMERY_BOX_ENV (a lever read once at load needs its own process,
         which every arm is), after the caller's own BLOOMERY_BOX_ENV and before
         BLOOMERY_AB_ROUNDS=<rounds>, which every arm gets; a tree whose box.sh predates
         BLOOMERY_BOX_ENV is refused for an env arm, since it would drop the lever silently. The run's
         card (BLOOMERY_LEASE_CARD, in the caller's BLOOMERY_BOX_ENV, relative to this tree) is checked
         here at --rounds by tools/ref/card.py before any arm starts, with card.py's exit code on a
         refusal: an arm whose tree predates the card check takes its lease unchecked, and the rotation
         can start with it. Every arm tree that checks cards must hold the same card bytes at that path;
         an arm may not set BLOOMERY_LEASE_CARD or BLOOMERY_AB_ROUNDS. Logs go to
         DIR/<recipe>-<arm>-r<round>.log, one row per arm run to DIR/runs.tsv. A failed arm (a red
         check, a lease that timed out, rc 75) or a card that stops answering ends the run there, no
         retry; the summary covers what finished.
         --dry-run prints the arm order and the exact commands, and runs nothing.
summary  per recipe it knows: each arm's rounds, mean, SD, and difference from the first arm with
         its 95 % interval (two-sample t, pooled SD, df = n1 + n2 - 2 — the ruler of AGENTS.md
         "Know the ruler"). time-gpu-generate: the SMOKE line's p50_ms. The second form reads any
         logs, e.g. the lead's older ones.
"""
import glob
import math
import os
import re
import statistics as st
import subprocess
import sys
import time

# The t table: tools/ref/tdist.py, the one card.py and the depth runners read too.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), 'ref'))
from tdist import t975  # noqa: E402

INSTRUMENTS = {'time-gpu-generate': 'generate'}
KV = re.compile(r'(\S+?)=(\S+)')


def describe(xs):
    m = st.mean(xs)
    return m, (st.stdev(xs) if len(xs) > 1 else float('nan'))


def versus(ref, xs):
    """Difference of means (xs - ref) and its 95 % half-interval; nan when a side has one round."""
    m0, s0 = describe(ref)
    m1, s1 = describe(xs)
    df = len(ref) + len(xs) - 2
    if len(ref) < 2 or len(xs) < 2:
        return m1 - m0, float('nan')
    sp = math.sqrt(((len(ref) - 1) * s0 ** 2 + (len(xs) - 1) * s1 ** 2) / df)
    return m1 - m0, t975(df) * sp * math.sqrt(1 / len(ref) + 1 / len(xs))


def parse_generate(path):
    for line in open(path, errors='replace'):
        if line.startswith('SMOKE '):
            d = dict(KV.findall(line))
            return {'p50_ms': float(d['p50_ms']), 'mean_ms': float(d['mean_ms'])}
    return None


def row(name, xs, ref, unit, digits):
    m, s = describe(xs)
    line = (f'  {name:10s} rounds={len(xs)} mean={m:.{digits}f} {unit} sd={s:.{digits}f} {unit} '
            f'({100 * s / m:.2f} %)')
    if ref is not xs:
        d, h = versus(ref, xs)
        m0 = st.mean(ref)
        ci = '' if math.isnan(h) else f' ± {h:.{digits}f} {unit} (± {100 * h / m0:.2f} %)'
        line += f'  vs first: {d:+.{digits}f} {unit} ({100 * d / m0:+.2f} %){ci}'
    return line


def summarize(instrument, arms):
    """arms: [(name, [log paths])] with the reference first."""
    data = []
    for name, logs in arms:
        runs = [(p, parse_generate(p)) for p in logs]
        bad = [p for p, r in runs if r is None]
        for p in bad:
            print(f'  {name}: no {instrument} result in {p} (left out)')
        data.append((name, [r for _, r in runs if r is not None]))
    data = [(n, rs) for n, rs in data if rs]
    if not data:
        print(f'  no {instrument} results')
        return
    print('p50_ms of the SMOKE line, per arm (graph mode, one process per arm and round)')
    ref = [r['p50_ms'] for r in data[0][1]]
    for name, rs in data:
        xs = [r['p50_ms'] for r in rs]
        print(row(name, xs, ref if name != data[0][0] else xs, 'ms', 4)
              + f'  tok/s={1e3 / st.mean(xs):.2f}  p50s={[round(x, 4) for x in xs]}')


def summary(args):
    if args[:1] == ['--instrument']:
        instrument, specs = args[1], args[2:]
        if instrument not in INSTRUMENTS.values():
            print(f'gpu-ab: no instrument {instrument}; the instruments are {", ".join(INSTRUMENTS.values())}',
                  file=sys.stderr)
            return 2
        arms = []
        for spec in specs:
            name, _, pattern = spec.partition('=')
            arms.append((name, sorted(glob.glob(pattern))))
        print(f'instrument {instrument}')
        summarize(instrument, arms)
        return 0
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    out = args[0]
    rows = [line.rstrip('\n').split('\t') for line in open(os.path.join(out, 'runs.tsv'))][1:]
    for recipe in dict.fromkeys(r[0] for r in rows):
        runs = [r for r in rows if r[0] == recipe and r[6] == '0']
        instrument = INSTRUMENTS.get(recipe)
        print(f'recipe {recipe}' + (f' (instrument {instrument})' if instrument else ' — no summarizer; logs only'))
        if not instrument:
            continue
        arms = {}
        for r in runs:
            arms.setdefault(r[3], []).append(os.path.join(out, r[9]))
        order = list(dict.fromkeys(r[3] for r in rows if r[0] == recipe))
        summarize(instrument, [(a, arms.get(a, [])) for a in order])
    return 0


def tree_path(t):
    if '/' in t:
        return os.path.abspath(t)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(os.path.dirname(here), t)


def lease_card(rounds, trees):
    """0 when the run's card passes card.py's lease check at `rounds` and every card-checking arm tree
    holds it; else card.py's exit code (2 for a BOX_ENV that names more than one card)."""
    cards = [e.split('=', 1)[1] for e in os.environ.get('BLOOMERY_BOX_ENV', '').split()
             if e.startswith('BLOOMERY_LEASE_CARD=')]
    if len(cards) > 1:
        print(f'gpu-ab: BLOOMERY_BOX_ENV names {len(cards)} cards: {" ".join(cards)}', file=sys.stderr)
        return 2
    card = cards[0] if cards else ''
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f'gpu-ab: the card, checked here at {rounds} rounds before any arm:')
    rc = subprocess.run([sys.executable, os.path.join(here, 'tools', 'ref', 'card.py'), 'lease', card,
                         '--rounds', str(rounds)]).returncode
    if rc != 0:
        return rc
    with open(os.path.join(here, card), 'rb') as f:
        body = f.read()
    for tree in trees:
        if not os.path.isfile(os.path.join(tree, 'tools', 'ref', 'card.py')):
            print(f'gpu-ab: {tree} predates the card check: its arms take the lease unchecked, on the check above')
            continue
        try:
            with open(os.path.join(tree, card), 'rb') as f:
                held = f.read()
        except OSError:
            held = None
        if held != body:
            print(f'gpu-ab: {tree} checks cards and {"has no" if held is None else "holds another"} {card}: '
                  'copy this tree\'s card there before the run', file=sys.stderr)
            return 66
    return 0


def healthy(host):
    probe = "nvidia-smi --query-gpu=index,name,pstate --format=csv,noheader 2>&1; echo n=$(nvidia-smi -L 2>/dev/null | grep -c GPU)"
    got = subprocess.run(['ssh', host, probe], capture_output=True, text=True).stdout
    return not re.search(r'ERR!|No devices|Unable to determine', got) and 'n=2' in got, got.strip().replace('\n', ';')


def run(args):
    dry, out, rounds, recipes, arms = False, None, None, [], []
    it = iter(args)
    for a in it:
        if a == '--dry-run':
            dry = True
        elif a == '--out':
            out = os.path.abspath(next(it))
        elif a == '--rounds':
            rounds = int(next(it))
        elif a == '--recipe':
            recipes.append(next(it))
        elif a == '--arm':
            name, _, rest = next(it).partition('=')
            tree, _, env = rest.partition(':')
            if not re.fullmatch(r'[A-Za-z0-9_]+', name) or not tree:
                print(f'gpu-ab: an arm is NAME=TREE[:K=V,K=V] with NAME in [A-Za-z0-9_], got {name}={rest}', file=sys.stderr)
                return 2
            kvs = [kv for kv in env.split(',') if kv]
            for kv in kvs:
                if not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*=[^\s\'"]*', kv):
                    print(f'gpu-ab: arm {name}: {kv!r} is not K=V', file=sys.stderr)
                    return 2
            arms.append((name, tree_path(tree), kvs))
        else:
            print(f'gpu-ab: unknown argument {a}', file=sys.stderr)
            return 2
    if not out or not rounds or rounds < 1 or not recipes or len(arms) < 2 or len({a[0] for a in arms}) != len(arms):
        print('gpu-ab: run needs --out, --rounds >= 1, one --recipe or more and two arms or more, named apart', file=sys.stderr)
        return 2
    for name, tree, kvs in arms:
        box = os.path.join(tree, 'tools', 'box.sh')
        if not os.path.isfile(box):
            print(f'gpu-ab: arm {name}: no {box}', file=sys.stderr)
            return 2
        if kvs and 'BLOOMERY_BOX_ENV' not in open(box).read():
            print(f'gpu-ab: arm {name}: {box} predates BLOOMERY_BOX_ENV and would drop {kvs} — refused', file=sys.stderr)
            return 2
        owned = [kv for kv in kvs if kv.split('=', 1)[0] in ('BLOOMERY_LEASE_CARD', 'BLOOMERY_AB_ROUNDS')]
        if owned:
            print(f'gpu-ab: arm {name}: {" ".join(owned)} belongs to the run, not an arm — refused', file=sys.stderr)
            return 2
    caller = os.environ.get('BLOOMERY_BOX_ENV', '').split()
    clash = [e for e in caller if e.startswith('BLOOMERY_AB_ROUNDS=') and e != f'BLOOMERY_AB_ROUNDS={rounds}']
    if clash:
        print(f'gpu-ab: BLOOMERY_BOX_ENV has {" ".join(clash)} and --rounds is {rounds} — refused', file=sys.stderr)
        return 2
    rc = lease_card(rounds, sorted({tree for _, tree, _ in arms}))
    if rc != 0:
        return rc
    caller = [e for e in caller if not e.startswith('BLOOMERY_AB_ROUNDS=')]
    plan = []
    for recipe in recipes:
        for r in range(1, rounds + 1):
            for slot in range(len(arms)):
                name, tree, kvs = arms[(slot + r - 1) % len(arms)]
                plan.append((recipe, r, slot + 1, name, tree, kvs))
    print(f'gpu-ab plan: {len(arms)} arms x {rounds} rounds x {len(recipes)} recipe(s) = {len(plan)} arm runs; '
          f'reference arm {arms[0][0]}; order rotates one arm per round')
    for recipe in recipes:
        for r in range(1, rounds + 1):
            print(f'  {recipe} round {r}: ' + ' '.join(p[3] for p in plan if p[0] == recipe and p[1] == r))
    cmds = []
    for recipe, r, slot, name, tree, kvs in plan:
        remote = f'~/repo/{os.path.basename(tree)}'
        env = {'BLOOMERY_REMOTE': remote, 'BLOOMERY_BOX_ENV': ' '.join(caller + kvs + [f'BLOOMERY_AB_ROUNDS={rounds}'])}
        log = f'{recipe}-{name}-r{r}.log'
        shown = ' '.join(f"{k}='{v}'" for k, v in env.items())
        print(f'  r{r}.{slot} {name}: (cd {tree} && {shown} just {recipe}) > {out}/{log}')
        cmds.append((recipe, r, slot, name, tree, kvs, env, log))
    if dry:
        return 0
    os.makedirs(out, exist_ok=True)
    tsv = os.path.join(out, 'runs.tsv')
    new = not os.path.exists(tsv)
    host = os.environ.get('BLOOMERY_BOX', 'ws')
    stamp = lambda: time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    stopped = False
    with open(tsv, 'a') as f:
        if new:
            f.write('recipe\tround\tslot\tarm\ttree\tenv\trc\tstart\tend\tlog\tcommit\n')
        for recipe, r, slot, name, tree, kvs, env, log in cmds:
            commit = subprocess.run(['git', '-C', tree, 'describe', '--always', '--dirty', '--abbrev=12'],
                                    capture_output=True, text=True).stdout.strip() or 'no-git'
            start = stamp()
            with open(os.path.join(out, log), 'w') as lf:
                lf.write(f'# gpu-ab recipe={recipe} round={r} slot={slot} arm={name} tree={tree} '
                         f'env={",".join(kvs) or "-"} commit={commit} start={start}\n')
                lf.flush()
                rc = subprocess.run(['just', recipe], cwd=tree, env=dict(os.environ, **env),
                                    stdout=lf, stderr=subprocess.STDOUT).returncode
            f.write(f'{recipe}\t{r}\t{slot}\t{name}\t{tree}\t{",".join(kvs) or "-"}\t{rc}\t{start}\t{stamp()}\t{log}\t{commit}\n')
            f.flush()
            print(f'r{r}.{slot} {recipe} {name} rc={rc}', flush=True)
            ok, state = healthy(host)
            if rc != 0 or not ok:
                why = f'rc {rc}' if rc != 0 else f'a card stopped answering: {state}'
                print(f'gpu-ab: STOP after r{r}.{slot} {name} — {why}; no retry', file=sys.stderr)
                stopped = True
                break
    summary([out])
    return 3 if stopped else 0


def main(argv):
    if argv[:1] == ['run']:
        return run(argv[1:])
    if argv[:1] == ['summary']:
        return summary(argv[1:])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
