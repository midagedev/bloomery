#!/usr/bin/env python3
"""The two-sided 95 % Student t quantile, the one table every interval here is drawn from.

  from tdist import t975          # tools/gpu-ab.py, tools/ref/card.py
  python3 tools/ref/tdist.py N    # t975(1) … t975(N), space-separated (the depth runners' awk)

t975(df) is the table below for df 1..30 and 1.96 + 2.4 / df past it, which stays within 0.003 of
the true quantile; the largest gap is at df 31, 2.0374 against 2.0395 [derived: bisection on the
numerically integrated t density]. A df below 1 has no quantile and is refused by name, never
answered with a table row.
"""
import sys

# Two-sided 95 % t quantiles, df 1..30.
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
        2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
        2.052, 2.048, 2.045, 2.042]


def t975(df):
    if isinstance(df, bool) or not isinstance(df, int) or df < 1:
        raise ValueError(f't975: df must be an integer >= 1, got {df!r}')
    return T975[df - 1] if df <= len(T975) else 1.96 + 2.4 / df


def main(argv):
    if len(argv) != 1 or not argv[0].isdigit() or int(argv[0]) < 1:
        print('usage: tdist.py N — prints t975(1) … t975(N), N a positive integer', file=sys.stderr)
        return 64
    print(' '.join(repr(t975(df)) for df in range(1, int(argv[0]) + 1)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
