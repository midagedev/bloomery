#!/usr/bin/env python3
"""errsrc: judge simulated activation-rounding arms (exact_ref --act) against f64.

usage: errsrc-judge.py F64_DIR ARM=DIR [ARM=DIR ...] [--gate NAME=LOG[:IDX] ...]
F64_DIR/ARM dirs hold exact_ref logs p<id>*.log. Gate logs are gate_e2e output
(the table lists only positions where the arm disagrees with ik; elsewhere it
equals ik). Positions judged = those present in the f64 set and every arm."""
import re, sys, glob, os

FLOOR = 0.5

def exact_logs(d):
    out = {}
    for f in glob.glob(os.path.join(d, 'p*.log')):
        if 'f16kv' in f:
            continue
        pid = int(re.search(r'p(\d+)', os.path.basename(f)).group(1))
        for line in open(f):
            c = line.rstrip('\n').split('\t')
            if len(c) == 7 and c[0].isdigit():
                # theirs, top1, margin, theirs_gap, ref_margin
                out[(pid, int(c[0]))] = (int(c[2]), int(c[3]), float(c[4]), float(c[5]), float(c[6]))
    return out

def gate_table(path, idx):
    runs, cur, on = [], None, False
    for line in open(path):
        if re.match(r'\s*id\s+step\s+ours\s+theirs\s+margin', line):
            cur, on = {}, True
            continue
        if on and line.startswith('forced positions='):
            runs.append(cur); on = False; continue
        if on:
            f = line.split()
            if len(f) == 5:
                cur[(int(f[0]), int(f[1]))] = int(f[2])
    return runs[idx]

args = sys.argv[1:]
f64 = exact_logs(args[0])
arms, gates, i = [], [], 1
while i < len(args):
    if args[i] == '--gate':
        name, rest = args[i + 1].split('=', 1)
        log, _, idx = rest.partition(':')
        gates.append((name, gate_table(log, int(idx or 0))))
        i += 2
    else:
        name, d = args[i].split('=', 1)
        arms.append((name, exact_logs(d)))
        i += 1

keys = set(f64)
for _, a in arms:
    keys &= set(a)
keys = sorted(keys)
print(f'positions judged: {len(keys)} (f64 has {len(f64)})')

def score(top1_of):
    w = [k for k in keys if top1_of(k) != f64[k][1]]
    wc = [k for k in w if f64[k][2] >= FLOOR]
    return w, wc

rows = [('ik CUDA (theirs)', lambda k: f64[k][0])]
for name, t in gates:
    rows.append((f'engine {name}', lambda k, t=t: t.get(k, f64[k][0])))
for name, a in arms:
    rows.append((f'sim {name}', lambda k, a=a: a[k][1]))
print(f'{"arm":<28} {"!=f64":>6} {"!=f64 & m>=0.5":>15}  clear positions')
tops = {}
for name, fn in rows:
    w, wc = score(fn)
    tops[name] = fn
    print(f'{name:<28} {len(w):>6} {len(wc):>15}  {" ".join(f"{a}/{b}" for a, b in wc)}')

# pairwise top1 disagreement between arms (how well a sim predicts an engine)
names = [n for n, _ in rows]
print('\npairwise top1 disagreements (count over judged positions):')
print(' ' * 28 + ''.join(f'{n[:14]:>16}' for n in names))
for a in names:
    print(f'{a:<28}' + ''.join(f'{sum(tops[a](k) != tops[b](k) for k in keys):>16}' for b in names))

# margin shift: mean |sim margin - f64 margin| on positions where both agree with f64 top1
for name, a in arms:
    d = [abs(a[k][2] - f64[k][2]) for k in keys if a[k][1] == f64[k][1]]
    print(f'sim {name}: mean |margin - f64 margin| = {sum(d)/len(d):.4f} over {len(d)}')

# sigma: arm margin (signed toward the true top1) minus the true margin.
# Where the arm's top1 differs from the truth, its margin for the true top1
# is approximated by -(arm margin) (the truth is at best the arm's runner-up).
from math import erf, sqrt
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
def sig_rows():
    out = [('ik CUDA (theirs)', lambda k: (f64[k][0], f64[k][4]))]
    for name, a in arms:
        out.append((f'sim {name}', lambda k, a=a: (a[k][1], a[k][2])))
    return out
print('\nsigma (RMS of signed margin error), mean bias, expected vs actual mismatches at true margin >= 0.5:')
for name, fn in sig_rows():
    d = []
    for k in keys:
        top, m = fn(k)
        sm = m if top == f64[k][1] else -m
        d.append(sm - f64[k][2])
    rms = sqrt(sum(x * x for x in d) / len(d))
    bias = sum(d) / len(d)
    clear = [k for k in keys if f64[k][2] >= FLOOR]
    exp = sum(Phi(-f64[k][2] / rms) for k in clear)
    exp_all = sum(Phi(-f64[k][2] / rms) for k in keys)
    act = sum(fn(k)[0] != f64[k][1] for k in clear)
    act_all = sum(fn(k)[0] != f64[k][1] for k in keys)
    print(f'{name:<28} sigma {rms:.4f} bias {bias:+.4f}  clear: expected {exp:.2f} actual {act}   all: expected {exp_all:.1f} actual {act_all}')
