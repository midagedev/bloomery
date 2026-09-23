#!/usr/bin/env python3
"""Interleaved GPU A/B across trees and levers: the lead's timing rounds as one tool.

An arm is a (tree, environment) pair. Every round runs every arm once, through the tree's own
timing recipe — `just time-gpu-v41`, `just time-gpu-generate` — so every arm takes the machine-wide
lease and prints its witness blocks through tools/ref/time-gate.sh, like a hand-run of that recipe.
The order rotates by one arm per round (round r starts at arm r-1): in a fixed order a round's
first arm read slow (AGENTS.md), and rotation spreads that over every arm. Only interleaved,
same-window numbers are compared, and only as each arm's difference from the first arm.

  gpu-ab.py run [--dry-run] --out DIR --rounds N --recipe RECIPE [--recipe ...] --arm NAME=TREE[:K=V,K=V] ...
  gpu-ab.py summary DIR
  gpu-ab.py summary --instrument v41|generate NAME=LOG_GLOB ...

run      one --arm per arm, the first is the reference. TREE is a path or a directory name beside this
         repository's own; its box directory is ~/repo/<tree name>. K=V pairs reach the timed binary
         through tools/box.sh's BLOOMERY_BOX_ENV (a lever read once at load needs its own process,
         which every arm is); a tree whose box.sh predates BLOOMERY_BOX_ENV is refused for an env arm,
         since it would drop the lever silently. Logs go to DIR/<recipe>-<arm>-r<round>.log, one row
         per arm run to DIR/runs.tsv. A failed arm (a red check, a lease that timed out, rc 75) or a
         card that stops answering ends the run there, no retry; the summary covers what finished.
         --dry-run prints the arm order and the exact commands, and runs nothing.
summary  per recipe it knows: each arm's rounds, mean, SD, and difference from the first arm with
         its 95 % interval (two-sample t, pooled SD, df = n1 + n2 - 2 — the ruler of AGENTS.md
         "Know the ruler"). time-gpu-v41: the token graphs' us_mean (plans `today`, `wo_a_dense`)
         and the per-site table (graph µs per launch and µs per token). time-gpu-generate: the
         SMOKE line's p50_ms. The second form reads any logs, e.g. the lead's older ones.
"""
import glob
import math
import os
import re
import statistics as st
import subprocess
import sys
import time

INSTRUMENTS = {'time-gpu-v41': 'v41', 'time-gpu-generate': 'generate'}
# Two-sided 95 % t quantiles, df 1..30; past 30, 1.96 + 2.4/df is within 0.003 of the table.
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042]
KV = re.compile(r'(\S+?)=(\S+)')


def t975(df):
    return T975[df - 1] if df <= 30 else 1.96 + 2.4 / df


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


def parse_v41(path):
    shapes, plans = {}, {}
    for line in open(path, errors='replace'):
        if line.startswith('bench shape='):
            d = dict(KV.findall(line))
            shapes[d['shape']] = (float(d['graph_us_mean']), float(d['token_graph_us']))
        elif line.startswith('token plan='):
            d = dict(KV.findall(line))
            plans[d['plan']] = float(d['us_mean'])
    return {'plans': plans, 'shapes': shapes} if plans else None


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
    parse = parse_v41 if instrument == 'v41' else parse_generate
    data = []
    for name, logs in arms:
        runs = [(p, parse(p)) for p in logs]
        bad = [p for p, r in runs if r is None]
        for p in bad:
            print(f'  {name}: no {instrument} result in {p} (left out)')
        data.append((name, [r for _, r in runs if r is not None]))
    data = [(n, rs) for n, rs in data if rs]
    if not data:
        print(f'  no {instrument} results')
        return
    if instrument == 'generate':
        print('p50_ms of the SMOKE line, per arm (graph mode, one process per arm and round)')
        ref = [r['p50_ms'] for r in data[0][1]]
        for name, rs in data:
            xs = [r['p50_ms'] for r in rs]
            print(row(name, xs, ref if name != data[0][0] else xs, 'ms', 4)
                  + f'  tok/s={1e3 / st.mean(xs):.2f}  p50s={[round(x, 4) for x in xs]}')
        return
    for plan in ('today', 'wo_a_dense'):
        have = [(n, [r['plans'][plan] for r in rs if plan in r['plans']]) for n, rs in data]
        have = [(n, xs) for n, xs in have if xs]
        if not have:
            continue
        print(f'token plan={plan}: us_mean per round, per arm')
        ref = have[0][1]
        for name, xs in have:
            print(row(name, xs, ref if name != have[0][0] else xs, 'us', 1) + f'  per-round={[round(x, 1) for x in xs]}')
    sites = [s for s in data[0][1][0]['shapes']]
    head = f'{"site":24s}' + ''.join(f' {n + " us":>10s} {n + " tok":>10s}' for n, _ in data)
    head += ''.join(f' {"d" + n + " tok":>10s}' for n, _ in data[1:])
    print('per site: graph us per launch and us per token, mean over rounds; d = arm - first, per token')
    print(head)
    total = [0.0] * len(data)
    for s in sites:
        cells, toks = '', []
        for name, rs in data:
            got = [r['shapes'][s] for r in rs if s in r['shapes']]
            g = st.mean(x[0] for x in got) if got else float('nan')
            t = st.mean(x[1] for x in got) if got else float('nan')
            toks.append(t)
            cells += f' {g:10.3f} {t:10.1f}'
        cells += ''.join(f' {t - toks[0]:+10.1f}' for t in toks[1:])
        for i, t in enumerate(toks):
            total[i] += t - toks[0]
        print(f'{s:24s}{cells}')
    for i, (name, _) in enumerate(data[1:], 1):
        print(f'sum of per-site token deltas, {name} - {data[0][0]}: {total[i]:+.1f} us')


def summary(args):
    if args[:1] == ['--instrument']:
        instrument, specs = args[1], args[2:]
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
        env = {'BLOOMERY_REMOTE': remote}
        if kvs:
            env['BLOOMERY_BOX_ENV'] = ' '.join(kvs)
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
