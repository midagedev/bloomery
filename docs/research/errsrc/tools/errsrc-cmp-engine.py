"""Per-position comparison of gate_e2e --margins dumps (scalar, MMA) with the
f64 truth and the simulation of our own rounding rule (exact_ref --act ours).

usage: errsrc-cmp-engine.py DUMP_DIR SIM_DIR
DUMP_DIR holds scalar.tsv and mma.tsv; SIM_DIR holds f64/ and ours/ exact_ref logs.
Margins are signed toward the truth's top1 the way sigma_forced does it: an
arm that misses the truth gets -(its own margin)."""
import collections
import glob
import os
import re
import sys
from math import erf, sqrt

D, SIM = sys.argv[1], sys.argv[2]


def eng(p):
    out = {}
    for line in open(p):
        if line.startswith('#'):
            continue
        i, s, t, m, et, em = line.rstrip('\n').split('\t')
        out[(int(i), int(s))] = (int(t), float(m), int(et), float(em))
    return out


def sim(d):
    out = {}
    for f in glob.glob(os.path.join(d, 'p*.log')):
        if 'f16kv' in f:
            continue
        pid = int(re.search(r'p(\d+)', os.path.basename(f)).group(1))
        for line in open(f):
            c = line.rstrip('\n').split('\t')
            if len(c) == 7 and c[0].isdigit():
                out[(pid, int(c[0]))] = (int(c[3]), float(c[4]))
    return out


sc, mm = eng(f'{D}/scalar.tsv'), eng(f'{D}/mma.tsv')
f64, ours = sim(f'{SIM}/f64'), sim(f'{SIM}/ours')
keys = sorted(set(sc) & set(mm) & set(ours) & set(f64))
bad = [k for k in keys if sc[k][2] != f64[k][0] or abs(sc[k][3] - f64[k][1]) > 1e-3]
print(f'positions: engine {len(sc)}, common with both sims {len(keys)}, '
      f'truth disagreement between the gate file and the f64 logs {len(bad)}')


def signed(top, m, truth):
    return m if top == truth else -m


S_sc = {k: signed(sc[k][0], sc[k][1], sc[k][2]) for k in keys}
S_mm = {k: signed(mm[k][0], mm[k][1], mm[k][2]) for k in keys}
S_si = {k: signed(ours[k][0], ours[k][1], sc[k][2]) for k in keys}
T = {k: sc[k][3] for k in keys}
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))


def rep(name, a, b):
    d = {k: a[k] - b[k] for k in keys}
    n = len(d)
    r = sqrt(sum(x * x for x in d.values()) / n)
    bias = sum(d.values()) / n
    tails = []
    for z in (2, 3, 4, 5):
        cnt = sum(abs(x) > z * r for x in d.values())
        tails.append(f'>{z}rms {cnt} (gauss {n * 2 * (1 - Phi(z)):.1f})')
    print(f'{name:<20} rms {r:.4f} bias {bias:+.4f}  ' + '  '.join(tails))
    return d, r


print()
d1, _ = rep('scalar - truth', S_sc, T)
d2, _ = rep('mma - truth', S_mm, T)
d3, _ = rep('sim ours - truth', S_si, T)
d4, _ = rep('scalar - sim ours', S_sc, S_si)
d5, _ = rep('mma - sim ours', S_mm, S_si)
d6, r6 = rep('mma - scalar', S_mm, S_sc)

print('\nlargest |mma - scalar|  (id/step: d | scalar top1 margin | mma top1 margin | truth top1 margin | sim top1 margin)')
for k in sorted(keys, key=lambda k: -abs(d6[k]))[:15]:
    print(f'  {k[0]:>2}/{k[1]:<2} d={d6[k]:+.3f} | sc {sc[k][0]:>6} {sc[k][1]:.3f} | mma {mm[k][0]:>6} {mm[k][1]:.3f}'
          f' | truth {sc[k][2]:>6} {T[k]:.3f} | sim {ours[k][0]:>6} {ours[k][1]:.3f}')

print('\nlargest |mma - sim ours|  (with scalar - sim ours at the same position)')
for k in sorted(keys, key=lambda k: -abs(d5[k]))[:10]:
    print(f'  {k[0]:>2}/{k[1]:<2} d={d5[k]:+.3f} (scalar {d4[k]:+.3f}) | mma {mm[k][0]:>6} {mm[k][1]:.3f} | truth {sc[k][2]:>6} {T[k]:.3f}')

for label, idx in (('step', 1), ('prompt', 0)):
    g = collections.defaultdict(list)
    for k in keys:
        g[k[idx]].append(d6[k])
    print(f'\nrms(mma - scalar) by {label}: ' +
          ' '.join(f'{s}:{sqrt(sum(x * x for x in v) / len(v)):.2f}' for s, v in sorted(g.items())))

clear = [k for k in keys if T[k] >= 0.5]
print()
for name, top in (('scalar', sc), ('mma', mm)):
    w = [k for k in clear if top[k][0] != sc[k][2]]
    print(f'{name}: clear mismatches {len(w)}: ' + ' '.join(f'{a}/{b}(true m {T[(a, b)]:.2f})' for a, b in w))
