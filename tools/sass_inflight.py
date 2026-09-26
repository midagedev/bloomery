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

A third reading, `--step HEAD`, replaces the table: the instructions one iteration of the loop at
HEAD executes along the decisions, by class — the count a latency-bound kernel's step is judged by
(its cycles are instructions × CPI ÷ warps a scheduler). The walk starts at HEAD, the loop's first
instruction (a backward BRA's target; `auto` takes the loop with the largest body), follows the
path rules above, and ends at the first BRA back to HEAD, which it counts. Decisions here may also
be keyed by address, `0x1cf0=t,0x2370=n,0x2030=t2`: a keyed branch takes its decision every time it
is met (`tK`: taken the first K times, then not — a nested loop run K + 1 times), a branch not
keyed falls through, and a keyed address the walk never meets fails the reading (`unmet=`: keys
read off another build's listing name other instructions). The iteration is cut after every second
barrier (BAR), so a loop that runs two steps an iteration reports each (`segs`), the instructions
after the last cut joining the last step. Classes: LDG, LDGSTS, LDS by width (@!PT dummies apart), IMMA (tensor), IMAD (every form),
ALU, FP32 (FFMA, FMUL, FADD), BRA/WARPSYNC (with BSSY, BSYNC, CALL, RET), BAR, other (uniform
datapath, dependency barriers and the rest); a line per class breaks it down by opcode. With a
cuobjdump listing it also sums the stall counts of the control words (`stall`).
`--raw` prints the matching entries of a cuobjdump listing verbatim, control words included, which
this script reads back.

usage: sass_inflight.py [--filter SUBSTR [--exact]] [--decisions t,n,...|ADDR=t|n|tK,...|list]
                        [--step HEAD|auto] [--raw] [--banner TEXT] FILE...
A FILE is a cuobjdump listing ("Function : <name>" sections) or a condensed one ("<addr> <op>" per
line, one entry named after the file). Exit status: 0 with rows, 1 when no entry was read or none
matches the filter (or, for --step, a walk did not end at its back edge or left a keyed decision
unmet), 2 on a usage error. With
--exact the filter is a whole entry name instead of a substring.
"""
import os
import re
import sys

INS = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')
HIWORD = re.compile(r'^\s*/\*\s*0x([0-9a-f]{16})\s*\*/\s*$')
FUNC = re.compile(r'^\s*Function\s*:\s*(\S+)')
CONDENSED = re.compile(r'^([0-9a-f]{4,})\s+(\S.*?)\s*$')
REG = re.compile(r'\bR(\d+)((?:\.[A-Za-z0-9]+)*)')
GUARD = re.compile(r'^@(!?)(U?P[T0-9]+)\s+')
BRANCH = re.compile(r'^(BRA|CALL)(?:\.[A-Z]+)*\s+(?:!?U?P[T0-9]+\s*,\s*)?(0x[0-9a-f]+)$')
STORES = {'STG', 'ST', 'STS', 'STL', 'RED'}
WALK_LIMIT = 100_000
CLASSES = ('LDG', 'LDGSTS', 'LDS', 'IMMA', 'IMAD', 'ALU', 'FP32', 'BRA/WARPSYNC', 'BAR', 'other')
ALU = {'LOP3', 'LOP', 'SHF', 'SGXT', 'IADD3', 'LEA', 'ISETP', 'PLOP3', 'SEL', 'I2FP', 'CS2R', 'HADD2',
       'MOV', 'PRMT', 'IMNMX', 'IABS', 'POPC', 'FLO', 'BMSK', 'BREV', 'P2R', 'R2P', 'FSEL', 'FSETP',
       'FMNMX'}
CONTROL = {'BRA', 'WARPSYNC', 'BSSY', 'BSYNC', 'CALL', 'RET', 'BRX', 'JMP', 'JMX', 'EXIT'}


def read_full(path):
    """Entries in file order: name -> (instructions [(address, text)], control words {address:
    the instruction's high 64 bits}, raw listing lines). A condensed listing has no control words
    and no raw lines."""
    entries, name = {}, None
    lines = open(path, errors='replace').read().splitlines()
    if not any(FUNC.match(line) for line in lines):
        name = os.path.basename(path).rsplit('.', 1)[0]
        entries[name] = ([(int(m.group(1), 16), m.group(2)) for m in map(CONDENSED.match, lines) if m],
                         {}, [])
        return entries
    last = None
    for line in lines:
        f = FUNC.match(line)
        if f:
            name = f.group(1)
            entries[name] = ([], {}, [line])
            last = None
            continue
        if name is None:
            continue
        ins, ctrl, raw = entries[name]
        m = INS.search(line)
        if m:
            last = int(m.group(1), 16)
            ins.append((last, m.group(2)))
            raw.append(line)
            continue
        h = HIWORD.match(line)
        if h and last is not None:
            ctrl[last] = int(h.group(1), 16)
            raw.append(line)
            last = None
    return entries


def read(path):
    """Entries in file order: name -> [(address, instruction text)]."""
    return {name: ins for name, (ins, _, _) in read_full(path).items()}


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


def parse_decisions(text):
    """A decision string: a positional list [bool], or keyed {address: times taken (None = always,
    0 = never)}. Returns (positional, keyed) or raises ValueError."""
    items = [d for d in text.split(',') if d] if text else []
    if not any('=' in d for d in items):
        bad = [d for d in items if d not in ('t', 'n')]
        if bad:
            raise ValueError(f'decisions are t or n, comma-separated; got {bad}')
        return [d == 't' for d in items], {}
    keyed = {}
    for d in items:
        a, _, v = d.partition('=')
        m = re.fullmatch(r't(\d*)|n', v)
        if not re.fullmatch(r'0x[0-9a-f]+', a) or not m:
            raise ValueError(f'keyed decisions are ADDR=t|n|tK (ADDR 0x-hex); got {d!r}')
        keyed[int(a, 16)] = 0 if v == 'n' else (int(m.group(1)) if m.group(1) else None)
    return [], keyed


def loop_heads(ins):
    """(head, back-edge address) of every backward BRA, largest body first."""
    out = []
    for a, t in ins:
        _, op, ops = split(t)
        m = BRANCH.match(' '.join([op] + ([', '.join(ops)] if ops else [])))
        if m and m.group(1) == 'BRA' and int(m.group(2), 16) < a:
            out.append((int(m.group(2), 16), a))
    return sorted(out, key=lambda x: x[0] - x[1])


def step_walk(ins, head, positional, keyed):
    """One iteration of the loop at `head` (the path rules of `walk`; keyed decisions count their
    uses): (executed [(address, text)], decided [(address, text, taken)], end)."""
    idx = {a: i for i, (a, _) in enumerate(ins)}
    if head not in idx:
        return [], [], f'no instruction at 0x{head:04x}'
    i, path, decided, returns, used, seen = idx[head], [], [], [], 0, {}
    while True:
        if len(path) >= WALK_LIMIT:
            return path, decided, f'stopped after {WALK_LIMIT} instructions: a loop the decisions never leave'
        a, t = ins[i]
        guard, op, ops = split(t)
        path.append((a, t))
        if op == 'EXIT' and not guard:
            return path, decided, 'EXIT'
        m = BRANCH.match(' '.join([op] + ([', '.join(ops)] if ops else [])))
        if m and not op.startswith('BRA.DIV') and guard != '@!PT':
            target = int(m.group(2), 16)
            if m.group(1) == 'BRA' and target == head:
                return path, decided, 'back edge'
            if guard in ('', '@PT'):
                go = True
            else:
                if keyed:
                    k = keyed.get(a, 0)
                    go = k is None or seen.get(a, 0) < k
                    seen[a] = seen.get(a, 0) + 1
                else:
                    go = used < len(positional) and positional[used]
                    used += 1
                decided.append((a, t, go))
            if go:
                if target not in idx:
                    return path, decided, f'branch to 0x{target:04x} outside the listing'
                if m.group(1) == 'CALL':
                    returns.append(i + 1)
                i = idx[target]
                continue
        elif op.split('.')[0] == 'RET' and not guard:
            if not returns:
                return path, decided, 'RET with no CALL on the path'
            i = returns.pop()
            continue
        i += 1
        if i >= len(ins):
            return path, decided, 'fell off the listing'


def klass(text):
    """(class, detail) of one instruction: the detail names the opcode, or an LDS/LDGSTS width."""
    guard, op, _ = split(text)
    base = op.split('.')[0]
    width = '128' if '.128' in op else '64' if '.64' in op else '16' if '16' in op else \
        '8' if ('.U8' in op or '.S8' in op) else '32'
    if base in ('LDG', 'LDGSTS'):
        return base, width
    if base == 'LDS':
        return 'LDS', '@!PT' if guard == '@!PT' else width
    if base in ('IMMA', 'HMMA', 'DMMA', 'BMMA'):
        return 'IMMA', base
    if base == 'IMAD':
        return 'IMAD', op
    if base in ('FFMA', 'FMUL', 'FADD'):
        return 'FP32', base
    if base in CONTROL:
        return 'BRA/WARPSYNC', base
    if base == 'BAR':
        return 'BAR', base
    if base in ALU:
        return 'ALU', base
    return 'other', base


def stall_of(ctrl, a):
    """The stall count of the instruction at `a` from its control word (bits 41..44 of the high
    word), or None without one."""
    return None if a not in ctrl else (ctrl[a] >> 41) & 0xf


def segments(path):
    """Instruction counts of the iteration cut after every second barrier; the rest joins the last."""
    cuts, bars = [], 0
    for n, (_, t) in enumerate(path, 1):
        if split(t)[1].split('.')[0] == 'BAR':
            bars += 1
            if bars % 2 == 0:
                cuts.append(n)
    edges = [0] + cuts[:-1] + [len(path)]
    return [edges[k + 1] - edges[k] for k in range(len(edges) - 1)]


def step_report(name, ins, ctrl, head_arg, decisions):
    """The --step lines of one entry, and whether its walk ended at the back edge."""
    heads = loop_heads(ins)
    if head_arg == 'auto':
        if not heads:
            return [f'step {name} no loop'], False
        head = heads[0][0]
    else:
        head = int(head_arg, 16)
    positional, keyed = parse_decisions(decisions)
    path, decided, end = step_walk(ins, head, positional, keyed)
    stalls = [stall_of(ctrl, a) for a, _ in path]
    stall = '-' if not path or None in stalls else str(sum(stalls))
    back = next((b for h, b in heads if h == head), None)
    unmet = sorted(set(keyed) - {a for a, _, _ in decided})
    lines = [f'step {name} head=0x{head:04x} back={hexa(back)} instrs={len(path)} '
             f'segs={"/".join(map(str, segments(path)))} stall={stall} end={end}'
             + (' unmet=' + ','.join(f'0x{a:04x}' for a in unmet) if unmet else '')]
    by, detail = {c: 0 for c in CLASSES}, {c: {} for c in CLASSES}
    for _, t in path:
        c, d = klass(t)
        by[c] += 1
        detail[c][d] = detail[c].get(d, 0) + 1
    lines.append('  classes ' + ' '.join(f'{c}={by[c]}' for c in CLASSES))
    for c in CLASSES:
        if detail[c]:
            parts = sorted(detail[c].items(), key=lambda x: (-x[1], x[0]))
            lines.append(f'  {c:12s} ' + ' '.join(f'{d}:{n}' for d, n in parts))
    lines.append(f'  decided {len(decided)}: '
                 + ' '.join(f'0x{a:04x}={"t" if go else "n"}' for a, _, go in decided))
    return lines, end == 'back edge' and not unmet


def hexa(a):
    return '-' if a is None else f'0x{a:04x}'


def main(argv):
    filt, exact, decisions, banner, files, head, raw = '', False, None, None, [], None, False
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
        elif arg == '--step':
            head = next(it, '')
            if head != 'auto' and not re.fullmatch(r'0x[0-9a-f]+', head):
                print(f'sass_inflight: --step takes a 0x-hex address or auto, got {head!r}', file=sys.stderr)
                return 2
        elif arg == '--raw':
            raw = True
        elif arg.startswith('--'):
            print(f'sass_inflight: unknown flag {arg}', file=sys.stderr)
            return 2
        else:
            files.append(arg)
    if not files:
        print(__doc__.split('usage: ')[1].split('\nA FILE')[0], file=sys.stderr)
        return 2
    listing = decisions == 'list'
    choice = []
    if decisions and not listing:
        try:
            choice, keyed = parse_decisions(decisions)
        except ValueError as e:
            print(f'sass_inflight: {e}', file=sys.stderr)
            return 2
        if keyed and head is None:
            print('sass_inflight: keyed decisions are read by --step only', file=sys.stderr)
            return 2
    full = {}
    for f in files:
        full.update(read_full(f))
    entries = {name: ins for name, (ins, _, _) in full.items()}
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
    if raw:
        for name, _ in rows:
            print('\n'.join(full[name][2]))
        return 0
    if head is not None:
        ok = True
        for name, ins in rows:
            lines, done = step_report(name, ins, full[name][1], head, '' if listing else decisions)
            print('\n'.join(lines))
            ok = ok and done
        return 0 if ok else 1
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
