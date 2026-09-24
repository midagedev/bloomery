#!/usr/bin/env python3
"""SASS in-flight scan: how many global loads an entry issues before it first waits on one.

The SASS twin of tools/ptx-scan.sh, and like it an instrument, not a gate. tools/sass-scan.sh makes
the listings (`cuobjdump -sass` of the gate binary's .oxart PTX assembled by ptxas) and calls this.

The measure is the one a latency-bound kernel is judged by: a load marks its destination registers
in flight, and the first later instruction that reads a register still in flight is where the warp
first waits on memory. The loads issued before that point are the ones whose latencies overlap.
An instruction that writes a register in flight takes it out of flight without waiting. This is
register dependency, not the scoreboard bits the listing also encodes.

Two readings per entry:
  loops  every backward branch is a loop; its body is read in address order from the branch
         target to the branch, and reports its loads, those issued before the body's first wait,
         the rest, and the first wait's address.
  path   one walk from the entry's first instruction, as a warp executes it: an unconditional BRA
         is followed, a conditional one takes the next decision (t = taken, n = not taken) — with
         none given, or none left, it falls through. A CALL is a branch that returns: followed
         when unconditional, a decision when predicated, and RET goes back to the instruction
         after it. BRA.DIV, taken only by a diverged warp, is not taken. A predicated EXIT is read
         as not exiting (the walk follows a lane that stays, the one that stores); an unpredicated
         EXIT ends it. A loop the decisions never leave stops the walk at WALK_LIMIT instructions,
         and the walk says so. It reports the instructions executed, the position of the first
         load among them, the loads, those issued before the first wait, BSSY count and branches
         taken.
Decisions are per entry and per run: pick them from the listing (`list`) for the shape in question.

usage: sass_inflight.py [--filter SUBSTR [--exact]] [--decisions t,n,...|list] [--banner TEXT] FILE...
A FILE is a cuobjdump listing ("Function : <name>" sections) or a condensed one ("<addr> <op>" per
line, one entry named after the file). Exit status: 0 with rows, 1 when no entry was read or none
matches the filter, 2 on a usage error. With --exact the filter is a whole entry name instead of a
substring.
"""
import os
import re
import sys

INS = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')
FUNC = re.compile(r'^\s*Function\s*:\s*(\S+)')
CONDENSED = re.compile(r'^([0-9a-f]{4,})\s+(\S.*?)\s*$')
REG = re.compile(r'\bR(\d+)((?:\.[A-Za-z0-9]+)*)')
GUARD = re.compile(r'^@(!?)(U?P[T0-9]+)\s+')
BRANCH = re.compile(r'^(BRA|CALL)(?:\.[A-Z]+)*\s+(?:!?U?P[T0-9]+\s*,\s*)?(0x[0-9a-f]+)$')
STORES = {'STG', 'ST', 'STS', 'STL', 'RED'}
WALK_LIMIT = 100_000


def read(path):
    """Entries in file order: name -> [(address, instruction text)]."""
    entries, name = {}, None
    lines = open(path, errors='replace').read().splitlines()
    if not any(FUNC.match(line) for line in lines):
        name = os.path.basename(path).rsplit('.', 1)[0]
        entries[name] = [(int(m.group(1), 16), m.group(2)) for m in map(CONDENSED.match, lines) if m]
        return entries
    for line in lines:
        f = FUNC.match(line)
        if f:
            name = f.group(1)
            entries[name] = []
            continue
        m = INS.search(line)
        if m and name is not None:
            entries[name].append((int(m.group(1), 16), m.group(2)))
    return entries


def split(text):
    """(guard, opcode, operands) with the guard ('@P0', '@!P0' or ''), .reuse flags dropped."""
    g = GUARD.match(text)
    guard = g.group(0).strip() if g else ''
    body = text[g.end():] if g else text
    body = body.replace('.reuse', '')
    op, _, rest = body.partition(' ')
    return guard, op, [o.strip() for o in rest.split(',')] if rest.strip() else []


def regs(operand, width=1):
    """Register numbers an operand names; a .64 register pair names two, `width` widens the first."""
    out = []
    for n, mods in REG.findall(operand):
        r = int(n)
        w = 2 if '.64' in mods else width
        out.extend(range(r, r + w))
    return out


def load_width(op):
    return 4 if '.128' in op else 2 if '.64' in op else 1


def is_load(op):
    return op.split('.')[0] == 'LDG'


def step(pending, op, ops):
    """Apply one instruction to the in-flight set; True when it waits on a load."""
    if is_load(op):
        waits = bool(set(r for o in ops[1:] for r in regs(o)) & pending)
        pending.update(regs(ops[0], load_width(op)) if ops else [])
        return waits
    base = op.split('.')[0]
    if base in STORES:
        return bool(set(r for o in ops for r in regs(o)) & pending)
    waits = bool(set(r for o in ops[1:] for r in regs(o)) & pending)
    if ops:
        pending.difference_update(regs(ops[0], 2 if '.WIDE' in op else 1))
    return waits


def loops(ins):
    out = []
    for a, t in ins:
        guard, op, ops = split(t)
        m = BRANCH.match(' '.join([op] + ([', '.join(ops)] if ops else [])))
        if m and m.group(1) == 'BRA' and int(m.group(2), 16) < a:
            lo = int(m.group(2), 16)
            body = [(x, s) for x, s in ins if lo <= x <= a]
            pending, first, before, after, ldg = set(), None, 0, 0, 0
            for x, s in body:
                _, bop, bops = split(s)
                waits = step(pending, bop, bops)
                if waits and first is None:
                    first = x
                if is_load(bop):
                    ldg += 1
                    if first is None:
                        before += 1
                    else:
                        after += 1
            out.append((lo, a, ldg, before, after, first))
    return out


def walk(ins, decisions):
    idx = {a: i for i, (a, _) in enumerate(ins)}
    pending, returns = set(), []
    i = n = ldg = before = bssy = taken = used = 0
    first_ldg = first_wait = None
    end = 'fell off the listing'
    while i < len(ins):
        if n >= WALK_LIMIT:
            end = f'stopped after {WALK_LIMIT} instructions: a loop the decisions never leave'
            break
        a, t = ins[i]
        guard, op, ops = split(t)
        n += 1
        if step(pending, op, ops) and first_wait is None:
            first_wait = a
        if is_load(op):
            ldg += 1
            if first_ldg is None:
                first_ldg = n
            if first_wait is None:
                before += 1
        if op.startswith('BSSY'):
            bssy += 1
        if op == 'EXIT' and not guard:
            end = 'EXIT'
            break
        m = BRANCH.match(' '.join([op] + ([', '.join(ops)] if ops else [])))
        if op.startswith('BRA.DIV'):
            i += 1
            continue
        if m:
            target = int(m.group(2), 16)
            go = guard in ('', '@PT') or (guard != '@!PT' and used < len(decisions) and decisions[used])
            if guard not in ('', '@PT', '@!PT'):
                used += 1
            if go:
                if target not in idx:
                    end = f'branch to 0x{target:04x} outside the listing'
                    break
                if target == a:
                    end = 'BRA to itself'
                    break
                if m.group(1) == 'CALL':
                    returns.append(i + 1)
                taken += 1
                i = idx[target]
                continue
        elif op.split('.')[0] == 'RET' and not guard:
            if not returns:
                end = 'RET with no CALL on the path'
                break
            i = returns.pop()
            continue
        elif op.split('.')[0] in ('BRA', 'BRX', 'JMX', 'CALL', 'RET', 'JMP'):
            end = f'stopped at {t} (not followed)'
            break
        i += 1
    return dict(instrs=n, first_ldg=first_ldg, ldg=ldg, before=before, first_wait=first_wait,
                bssy=bssy, taken=taken, conditional=used, end=end)


def hexa(a):
    return '-' if a is None else f'0x{a:04x}'


def main(argv):
    filt, exact, decisions, banner, files = '', False, None, None, []
    it = iter(argv)
    for arg in it:
        if arg == '--filter':
            filt = next(it, '')
        elif arg == '--exact':
            exact = True
        elif arg == '--decisions':
            decisions = next(it, '')
        elif arg == '--banner':
            banner = next(it, '')
        elif arg.startswith('--'):
            print(f'sass_inflight: unknown flag {arg}', file=sys.stderr)
            return 2
        else:
            files.append(arg)
    if not files:
        print(__doc__.split('usage: ')[1].split('\n')[0], file=sys.stderr)
        return 2
    listing = decisions == 'list'
    choice = []
    if decisions and not listing:
        bad = [d for d in decisions.split(',') if d not in ('t', 'n')]
        if bad:
            print(f'sass_inflight: decisions are t or n, comma-separated; got {bad}', file=sys.stderr)
            return 2
        choice = [d == 't' for d in decisions.split(',')]
    entries = {}
    for f in files:
        entries.update(read(f))
    if exact and not filt:
        print('sass_inflight: --exact needs --filter NAME', file=sys.stderr)
        return 2
    rows = [(name, ins) for name, ins in sorted(entries.items())
            if ins and (name == filt if exact else filt in name)]
    if banner:
        print(banner + (f' filter={filt}' if filt else '') + (' exact' if exact else '')
              + f' path={decisions or "fall-through"}')
    if not rows:
        how = (" is " if exact else " contains ") + filt if filt else ""
        print(f'sass_inflight: no entry{how} in {len(entries)} read', file=sys.stderr)
        return 1
    if listing:
        for name, ins in rows:
            print(f'#### {name}')
            for a, t in ins:
                print(f'{a:04x} {t}')
        return 0
    print(f'{"entry":32s} {"instrs":>6s} {"LDG":>4s} {"BSSY":>4s} {"loops":>5s} {"path":>6s} '
          f'{"1st_LDG":>7s} {"p_LDG":>5s} {"LDG<wait":>8s} {"1st_wait":>8s}')
    details = []
    for name, ins in rows:
        real = [(a, t) for a, t in ins if split(t)[1] != 'NOP']
        ldg = sum(1 for _, t in real if is_load(split(t)[1]))
        bssy = sum(1 for _, t in real if split(t)[1].startswith('BSSY'))
        lp = loops(ins)
        w = walk(ins, choice)
        print(f'{name:32s} {len(real):6d} {ldg:4d} {bssy:4d} {len(lp):5d} {w["instrs"]:6d} '
              f'{w["first_ldg"] if w["first_ldg"] is not None else "-":>7} {w["ldg"]:5d} '
              f'{w["before"]:8d} {hexa(w["first_wait"]):>8s}')
        for lo, hi, n, before, after, first in lp:
            if n:
                details.append(f'  loop {name} 0x{lo:04x}-0x{hi:04x} LDG={n} issued_before_first_wait={before} '
                               f'after={after} first_wait={hexa(first)}')
        details.append(f'  path {name} decisions={decisions or "fall-through"} '
                       f'conditional_branches={w["conditional"]} taken={w["taken"]} bssy={w["bssy"]} '
                       f'end={w["end"]}')
    print('\n'.join(details))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
