#!/usr/bin/env python3
"""The A/B ruler's interval, one owner for everyone who prints or predicts it: the depth runners'
ratio tables (ratio_table in depth-ds41.sh, depth-qwen3moe.sh, depth-glm5next.sh) and the card's
check (tools/ref/card.py).

A sitting runs its arms rotated within a round, one row an arm, and reads its effect as the mean of
the per-round ratios q_r = base_r / ref_r — the round pairs the two arms' rows, so what the round's
machine state does to both cancels. The interval of that mean is the paired t interval

    h = t975(N - 1) * sd(q) / sqrt(N)

with sd(q) the scatter of one round's ratio and t975 the quantile table tools/ref/tdist.py owns.
The df is N - 1, one per round, not the 2N - 2 of two independent arm means: at two rounds t is
12.706 against the two-sample 4.303, and a card checked against the two-sample form passes runs
whose printed interval is then ~3x its prediction [derived] — the runner's table is the paired
form, so the ruler is too.

The sd a card assumes is one round's ratio scatter. Both arms' runs scatter at the same-binary SD
s (AGENTS.md 「Know the ruler」: 0.6 %, 18 runs); with none of that shared within a round the
ratio's scatter is sqrt(2) * s, and what the two arms do share within a round the pairing cancels,
so sqrt(2) * s is the conservative end — the card's default [derived]. An override names its
source, a measured per-round ratio SD.

  from ruler import half_width           # card.py's check: (h, t, df) at an assumed sd
  python3 tools/ref/ruler.py factors N   # f(1) ... f(N), f(c) = t975(c - 1) / sqrt(c) for c >= 2
                                         # (`-` at c = 1: one round has no interval), the runners'
                                         # awk: its interval is f[c] * the ratios' sample SD
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from tdist import t975  # noqa: E402


def half_width(sd, rounds):
    """The 95 % half-width of the mean of `rounds` per-round ratios whose scatter is `sd`:
    t975(rounds - 1) * sd / sqrt(rounds), as (half_width, t, df). One round has no interval."""
    if isinstance(rounds, bool) or not isinstance(rounds, int) or rounds < 2:
        raise ValueError(f'the paired interval takes 2 rounds or more, got {rounds!r}')
    t = t975(rounds - 1)
    return t * sd / math.sqrt(rounds), t, rounds - 1


def factor(rounds):
    """half_width's multiplier on the sample SD: t975(rounds - 1) / sqrt(rounds), the number the
    runners' awk takes from here (`factors` below)."""
    return half_width(1.0, rounds)[0]


def main(argv):
    if len(argv) == 2 and argv[0] == 'factors' and argv[1].isdigit() and int(argv[1]) >= 1:
        print(' '.join('-' if c < 2 else repr(factor(c)) for c in range(1, int(argv[1]) + 1)))
        return 0
    print('usage: ruler.py factors N — prints f(1) ... f(N), f(c) = t975(c - 1) / sqrt(c) for c >= 2', file=sys.stderr)
    return 64


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
