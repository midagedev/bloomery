#!/usr/bin/env python3
"""Source-page reader: one ncu source page (per-instruction SASS counters) read as a step's phases.

An instrument, not a gate. tools/ref/ncu-gpu.sh writes the page (`BLOOMERY_NCU_SOURCE=1`, the gemm
form's `…source.csv`, with the details page `….csv` beside it); this reads it back on any machine.

The page is `--page source --csv` of every profiled launch: per launch a `"Kernel Name"` line, a
header, then one row per SASS instruction. ncu writes each launch's block twice (the page holds
2 x launches blocks); a block equal to an earlier one in every counter is dropped as a repeat, and
what remains is one block per distinct launch. Counts are per launch: summed over the distinct
launches, divided by their number.

Warp-cycles. A stall sample stands for (warp-cycles a launch) / (samples a launch), and a launch's
warp-cycles are CPI x issued instructions (ncu's `Warp Cycles Per Issued Instruction` and `Issued
Instructions`, read from --details, the mean over its launches, or given as --cpi and --issued).
Without them the tables print samples instead of warp-cycles.

Readings, in this order:
  launches   the page's blocks, the repeats dropped, samples and warp-cycles a sample
  loop       the loop (`--loop LO:HI`, offsets from the entry's first instruction as this script
             prints them, or `auto`: the backward branch whose body holds the most samples), its
             head's executions a launch (warp-iterations), its BAR count and steps an iteration
             (BAR count / --bars-per-step)
  phases     the loop cut at its BARs: a segment runs up to a BAR, and the BAR's phase runs from
             the BAR to its landing (the first instruction after it, within 16, whose samples are
             at least half barrier stalls; the BAR itself when none is) plus an unconditional
             branch right after the landing (a loop's back edge). The segment after the last
             landing wraps into the first. Phases are folded by their place in a step (--names
             names them, 2 x --bars-per-step of them): per warp-step, the instructions executed,
             the warp-cycles and their five largest stall reasons, then the step's total by reason
  top        the loop's instructions with the most samples (--top N)
  sectors    every instruction with `L2 Theoretical Sectors Global`: sectors a launch, ideal,
             their ratio and share; then --group sums, sums by opcode, and the total. The sectors
             are what ncu computes from the addresses (requests x sectors), not `lts__t_sectors`
             traffic
--listing prints the page's SASS as a condensed listing (`<offset> <instruction>` a line) instead,
the form tools/sass_inflight.py reads (`--step` on the JIT's own code).

usage: srcpage.py PAGE [--details CSV | --cpi X --issued N] [--loop auto|LO:HI]
                  [--bars-per-step B] [--names N,N,...] [--group NAME=ADDR[-ADDR],...]...
                  [--top N] [--listing]
  PAGE         the source page CSV
  --group      a named set of instructions for the sector sums: offsets and inclusive ranges
               (0x1040,0x75a0-0x75b0), repeatable
Exit status: 0 with tables; 1 when the page holds no block, the loop is not found, or a --group
offset names no instruction with sectors; 2 on a usage error.
"""
import csv
import re
import sys

SAMPLES = 'Warp Stall Sampling (All Samples)'
EXECUTED = 'Instructions Executed'
SECTORS = 'L2 Theoretical Sectors Global'
IDEAL = 'L2 Theoretical Sectors Global Ideal'
SHORT = {'barrier': 'bar', 'branch_resolving': 'brr', 'dispatch': 'disp', 'drain': 'drain',
         'imc': 'imc', 'lg': 'lg', 'long_sb': 'lsb', 'math': 'math', 'membar': 'mbar', 'mio': 'mio',
         'misc': 'misc', 'no_inst': 'noi', 'not_selected': 'nsel', 'selected': 'sel',
         'short_sb': 'ssb', 'sleep': 'slp', 'tex': 'tex', 'wait': 'wait'}
TARGET = re.compile(r'\b(?:BRA|CALL(?:\.[A-Z]+)*)\s+(?:!?U?P[T0-9]+\s*,\s*)?(0x[0-9a-f]+)')
GUARD = re.compile(r'^@!?U?P[T0-9]+\s+')
LANDING_WINDOW = 16


def usage(msg):
    print(f'srcpage.py: {msg}', file=sys.stderr)
    print(__doc__.split('usage: ')[1].split('\nExit status')[0], file=sys.stderr)
    sys.exit(2)


def num(s):
    s = s.replace(',', '').strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def read_page(path):
    """Distinct launch blocks: (header, [block], repeats dropped). A block is [(address, source,
    {column: value})] in page order."""
    rows = list(csv.reader(open(path, newline='', errors='replace')))
    starts = [i for i, r in enumerate(rows) if r and r[0] == 'Kernel Name']
    if not starts:
        return None, [], 0
    header = rows[starts[0] + 1]
    blocks, seen, dropped = [], set(), 0
    for a, b in zip(starts, starts[1:] + [len(rows)]):
        body = [r for r in rows[a + 2:b] if r and r[0].startswith('0x')]
        key = tuple(tuple(r) for r in body)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        blocks.append([(int(r[0], 16), r[1].strip(),
                        {h: num(r[i]) for i, h in enumerate(header) if i >= 2 and i < len(r)})
                       for r in body])
    return header, blocks, dropped


def per_launch(blocks):
    """One row per instruction: offset from the entry's first, source, counters averaged over
    the distinct launches. Every block must list the same instructions."""
    first = blocks[0]
    for b in blocks[1:]:
        if [(x[1]) for x in b] != [(x[1]) for x in first]:
            sys.exit('srcpage.py: the launches list different code; profile one binary per page')
    base = first[0][0]
    out = []
    for i, (addr, src, _) in enumerate(first):
        vals = {}
        for k in first[i][2]:
            vals[k] = sum(b[i][2][k] for b in blocks) / len(blocks)
        out.append({'off': addr - base, 'abs': addr, 'src': src, 'v': vals})
    return out, base


def read_details(path):
    """Mean CPI and issued instructions a launch from a details CSV (`--page details --csv`)."""
    cpi, issued = [], []
    for r in csv.DictReader(open(path, newline='', errors='replace')):
        name, value = r.get('Metric Name', ''), r.get('Metric Value', '')
        if name == 'Warp Cycles Per Issued Instruction':
            cpi.append(num(value))
        elif name == 'Issued Instructions':
            issued.append(num(value))
    if not cpi or not issued:
        sys.exit(f'srcpage.py: {path} has no `Warp Cycles Per Issued Instruction` / '
                 '`Issued Instructions` rows')
    return sum(cpi) / len(cpi), sum(issued) / len(issued)


def opcode(src):
    s = GUARD.sub('', src)
    return s.split()[0] if s else ''


def find_loops(ins, base):
    """Backward branches: (head offset, back-edge offset)."""
    index = {d['off'] for d in ins}
    loops = []
    for d in ins:
        m = TARGET.search(d['src'])
        if not m:
            continue
        tgt = int(m.group(1), 16) - base
        if tgt < d['off'] and tgt in index:
            loops.append((tgt, d['off']))
    return loops


def parse_offsets(spec):
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part[2:]:
            lo, hi = part.split('-', 1) if not part.startswith('-') else (part, part)
            out.append((int(lo, 16), int(hi, 16)))
        else:
            v = int(part, 16)
            out.append((v, v))
    return out


def main(argv):
    page, details, cpi, issued, loop_arg = None, None, None, None, 'auto'
    bars_per_step, names, groups, top, listing = 2, None, [], 30, False
    it = iter(argv)
    for a in it:
        if a == '--details':
            details = next(it, None)
        elif a == '--cpi':
            cpi = float(next(it, 'x'))
        elif a == '--issued':
            issued = float(next(it, 'x'))
        elif a == '--loop':
            loop_arg = next(it, None)
        elif a == '--bars-per-step':
            bars_per_step = int(next(it, '0'))
        elif a == '--names':
            names = next(it, '').split(',')
        elif a == '--group':
            g = next(it, '')
            if '=' not in g:
                usage(f'--group wants NAME=ADDR,..., got {g!r}')
            n, spec = g.split('=', 1)
            groups.append((n, parse_offsets(spec)))
        elif a == '--top':
            top = int(next(it, '0'))
        elif a == '--listing':
            listing = True
        elif a.startswith('--'):
            usage(f'unknown option {a}')
        elif page is None:
            page = a
        else:
            usage(f'one page, got {a} too')
    if page is None or loop_arg is None or bars_per_step < 1:
        usage('a page, and --loop / --bars-per-step need values')
    if (cpi is None) != (issued is None):
        usage('--cpi and --issued go together')

    header, blocks, dropped = read_page(page)
    if not blocks:
        print(f'srcpage.py: {page} holds no launch block', file=sys.stderr)
        return 1
    ins, base = per_launch(blocks)
    if listing:
        for d in ins:
            src = TARGET.sub(lambda m: m.group(0).replace(
                m.group(1), f'{int(m.group(1), 16) - base:#x}'), d['src'])
            print(f"{d['off']:04x}  {src}")
        return 0
    if details:
        cpi, issued = read_details(details)
    reasons = [h for h in header if h.startswith('stall_') and '(Not Issued)' not in h]
    samples = sum(d['v'].get(SAMPLES, 0.0) for d in ins)
    wcps = cpi * issued / samples if cpi is not None and samples else None
    # Without CPI the cycle columns are samples a launch, not per warp-step.
    unit = 'warp-cycles' if wcps else 'samples'
    print(f'launches: {len(blocks)} distinct, {dropped} repeated block(s) dropped; '
          f'samples {samples:,.0f} a launch'
          + (f'; CPI {cpi:.2f} x issued {issued:,.0f} -> {wcps:,.1f} warp-cycles a sample'
             if wcps else '; no CPI: tables in samples'))

    loops = find_loops(ins, base)
    if loop_arg == 'auto':
        if not loops:
            print('srcpage.py: no backward branch on the page', file=sys.stderr)
            return 1

        def weight(lp):
            return sum(d['v'].get(SAMPLES, 0.0) for d in ins if lp[0] <= d['off'] <= lp[1])
        lo, hi = max(loops, key=weight)
    else:
        try:
            lo, hi = (int(x, 16) for x in loop_arg.split(':'))
        except ValueError:
            usage(f'--loop wants auto or LO:HI in hex, got {loop_arg!r}')
    body = [d for d in ins if lo <= d['off'] <= hi]
    if not body or body[0]['off'] != lo:
        print(f'srcpage.py: no instruction at the loop head {lo:#x}', file=sys.stderr)
        return 1
    head_exec = body[0]['v'].get(EXECUTED, 0.0)
    bars = [i for i, d in enumerate(body) if opcode(d['src']).startswith('BAR')]
    loop_samples = sum(d['v'].get(SAMPLES, 0.0) for d in body)
    if not bars or len(bars) % bars_per_step:
        print(f'srcpage.py: the loop {lo:#x}-{hi:#x} holds {len(bars)} BAR, not a multiple of '
              f'--bars-per-step {bars_per_step}', file=sys.stderr)
        return 1
    steps = len(bars) // bars_per_step
    ws = head_exec * steps
    scale = wcps / ws if wcps else 1.0
    print(f'loop {lo:#06x}-{hi:#06x}: head executed {head_exec:,.0f} a launch, {len(bars)} BAR, '
          f'{steps} step(s) an iteration -> {ws:,.0f} warp-steps a launch; '
          f'{100 * loop_samples / samples:.1f} % of the samples')

    # Phases: segment before each BAR, then the BAR through its landing.
    def barrier_share(d):
        s = d['v'].get(SAMPLES, 0.0)
        return d['v'].get('stall_barrier', 0.0) / s if s else 0.0
    cuts = []  # (bar index, landing index)
    for b in bars:
        land = b
        for j in range(b + 1, min(b + 1 + LANDING_WINDOW, len(body))):
            if barrier_share(body[j]) >= 0.5:
                land = j
                break
        if land + 1 < len(body) and opcode(body[land + 1]['src']) == 'BRA' \
                and not GUARD.match(body[land + 1]['src']):
            land += 1
        cuts.append((b, land))
    kinds = 2 * bars_per_step
    names = names or [f'{k}{i}' for i in range(bars_per_step) for k in ('seg', 'bar')]
    if len(names) != kinds:
        usage(f'--names wants {kinds} names (segment, barrier per BAR of a step), got {len(names)}')
    phase_of = [None] * len(body)
    for n_bar, (b, land) in enumerate(cuts):
        pos = n_bar % bars_per_step
        for j in range(b, land + 1):
            phase_of[j] = 2 * pos + 1
        prev_land = cuts[n_bar - 1][1] if n_bar else -1
        for j in range(prev_land + 1, b):
            phase_of[j] = 2 * pos
    for j in range(cuts[-1][1] + 1, len(body)):  # wraps into the first segment
        phase_of[j] = 0
    print('bars (BAR -> landing): ' + ', '.join(
        f"{body[b]['off']:#06x} -> {body[land]['off']:#06x}" for b, land in cuts))
    print(f"\nphases, instructions per warp-step, {unit} {'per warp-step' if wcps else 'a launch'}:")
    print(f"{'phase':<12} {'instrs':>8} {unit:>12}  top reasons")
    totals = {r: 0.0 for r in reasons}
    whole = 0.0
    for k in range(kinds):
        part = [d for j, d in enumerate(body) if phase_of[j] == k]
        n_ins = sum(d['v'].get(EXECUTED, 0.0) for d in part) / ws
        cyc = sum(d['v'].get(SAMPLES, 0.0) for d in part) * scale
        by = {r: sum(d['v'].get(r, 0.0) for d in part) * scale for r in reasons}
        for r, v in by.items():
            totals[r] += v
        whole += cyc
        topr = ' '.join(f"{SHORT.get(r[6:], r[6:])} {v:,.0f}"
                        for r, v in sorted(by.items(), key=lambda kv: -kv[1])[:5] if v > 0)
        print(f'{names[k]:<12} {n_ins:>8.1f} {cyc:>12,.0f}  {topr}')
    print(f"{'step':<12} {sum(d['v'].get(EXECUTED, 0.0) for d in body) / ws:>8.1f} {whole:>12,.0f}")
    print('by reason: ' + ' '.join(f"{SHORT.get(r[6:], r[6:])} {v:,.0f}"
                                   for r, v in sorted(totals.items(), key=lambda kv: -kv[1])
                                   if v >= 0.5))

    if top:
        print(f"\ntop {top} of the loop by samples (executions per warp-step, {unit} {'per warp-step' if wcps else 'a launch'}):")
        for d in sorted(body, key=lambda d: -d['v'].get(SAMPLES, 0.0))[:top]:
            s = d['v'].get(SAMPLES, 0.0)
            by = sorted(((d['v'].get(r, 0.0), r) for r in reasons), reverse=True)[:2]
            why = ' '.join(f"{SHORT.get(r[6:], r[6:])} {100 * v / s:.0f}%" for v, r in by if s)
            print(f"  {d['off']:#06x} {d['v'].get(EXECUTED, 0.0) / ws:5.2f} "
                  f"{s * scale:8.1f}  {100 * s / loop_samples:5.2f}%  {why:<18} "
                  f"{d['src'][:64]}")

    mem = [d for d in ins if d['v'].get(SECTORS, 0.0) > 0]
    if mem:
        tot = sum(d['v'][SECTORS] for d in mem)
        tot_ideal = sum(d['v'].get(IDEAL, 0.0) for d in mem)
        print('\nL2 theoretical sectors a launch (from the addresses, not lts traffic):')
        print(f"  {'offset':<8} {'exec':>10} {'sectors':>12} {'ideal':>12} {'ratio':>7} {'share':>6}"
              '  instruction')
        for d in mem:
            sec, ide = d['v'][SECTORS], d['v'].get(IDEAL, 0.0)
            print(f"  {d['off']:#06x} {d['v'].get(EXECUTED, 0.0):>10,.0f} {sec:>12,.0f} "
                  f"{ide:>12,.0f} {sec / ide if ide else 0:>7.3f} {100 * sec / tot:>5.1f}%  "
                  f"{d['src'][:56]}")
        rc = 0
        rows = []
        for name, spans in groups:
            members = [d for d in mem if any(a <= d['off'] <= b for a, b in spans)]
            for a, b in spans:
                if a == b and not any(d['off'] == a for d in mem):
                    print(f'srcpage.py: --group {name}: {a:#x} has no sectors', file=sys.stderr)
                    rc = 1
            rows.append((name, members))
        by_op = {}
        for d in mem:
            by_op.setdefault(opcode(d['src']), []).append(d)
        rows += sorted(by_op.items())
        rows.append(('total', mem))
        print(f"  {'group':<28} {'sectors':>12} {'ideal':>12} {'ratio':>7} {'excess':>12} "
              f"{'share':>6} {'of excess':>9}")
        excess_all = tot - tot_ideal
        for name, members in rows:
            sec = sum(d['v'][SECTORS] for d in members)
            ide = sum(d['v'].get(IDEAL, 0.0) for d in members)
            print(f"  {name:<28} {sec:>12,.0f} {ide:>12,.0f} {sec / ide if ide else 0:>7.3f} "
                  f"{sec - ide:>12,.0f} {100 * sec / tot:>5.1f}% "
                  f"{100 * (sec - ide) / excess_all if excess_all else 0:>8.1f}%")
        return rc
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
