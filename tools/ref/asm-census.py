#!/usr/bin/env python3
"""Host-code census of a release binary, per function, from `objdump -d --no-show-raw-insn -C`.

An input is an ELF binary (objdump runs on it; the box) or a saved listing (any other file).

    asm-census.py legacy INPUT [--fail]   every legacy-encoded (non-VEX) instruction on an xmm register
                                          inside a function that touches a ymm register, and every SSE4a
                                          instruction (EXTRQ, INSERTQ, MOVNTSS, MOVNTSD) anywhere; with
                                          --fail, rc 1 when either list is not empty
    asm-census.py loop INPUT FUNCTION     the function's outermost loop (its longest backward branch):
                                          instructions, memory operands, stack operands, the mnemonics
    asm-census.py diff BASE NEW           the functions whose instruction sequence differs, addresses and
                                          rip displacements normalized, with each side's count; and the
                                          functions one side lacks
    asm-census.py tile-units LISTING RELOCS QDOT_LIB OPS_RS
                                          the host expert tier's tile cost rows, derived from the disassembly
                                          of qdot's tile and wide kernels and compared with `tile_units` in ops.rs
                                          (rc 1 when a type's row is missing, stale or off); tools/ref/tile-units.sh
                                          builds the inputs
    asm-census.py --self-test             canned listings

A listing from llvm-objdump (`jbe 0x5805 <f+0x5805>`) and from GNU objdump (`jbe 5805 <f+0x5805>`) are both read.

Why legacy encodings matter here: on Zen 3 a legacy-SSE instruction run while the ymm upper halves are
dirty takes a dispatch fault (perf: fp_disp_faults.ymm_spill_fault, then ymm_fill_fault). Under
target-cpu=znver3 LLVM emits VEX for everything that has a VEX form, so what is left is SSE4a, which
has none — the reason .cargo/config.toml turns sse4a off (tools/check-rustflags.sh holds it).
"""

import collections
import functools
import re
import subprocess
import sys

FUNC = re.compile(r"^[0-9a-f]+ <(.*)>:$")
JUMP = re.compile(r"(j\w+)\s+(?:0x)?([0-9a-f]+) <")
SSE4A = {"extrq", "insertq", "movntss", "movntsd"}
CALLER_SAVED = {"rax", "rcx", "rdx", "rsi", "rdi", "r8", "r9", "r10", "r11"}  # a call leaves them unknown
STACK_SLOT = re.compile(r"(-?0x[0-9a-f]+)?\(%rsp\)")
SLOT_BYTES = 8  # a flag's stack slot: the bytes from its offset that the widest integer store of the flag writes
# The instructions the tile derivation reads, by what they write; the first pattern a mnemonic matches wins, and
# one no pattern matches is refused by name. "" nothing; "F" the condition flags only; "D" the last operand; "DF"
# the last operand and the flags; "2", "2F" the last two operands (and the flags); the rest name implicit writes.
WRITE_CLASSES = tuple((re.compile(p), kind) for p, kind in (
    (r"j\w+|nop\w*|data16|prefetch\w*|vzero(?:upper|all)|endbr(?:32|64)", ""),
    (r"(?:cmp|test|bt)[bwlq]?|sahf|v?u?comis[sd]|vptest|vtestp[sd]", "F"),
    (r"(?:mov|movabs|lea|not|bswap|movbe|shlx|shrx|sarx|rorx|pdep|pext)[bwlq]?|movz[bw][wlq]|movs[bw][wlq]|movslq"
     r"|cmov[a-z]+|set[a-z]+", "D"),
    (r"(?:add|sub|adc|sbb|and|or|xor|neg|inc|dec|sh[lr]|sa[lr]|ro[lr]|rc[lr]|sh[lr]d|bt[src]|bextr|bzhi|andn"
     r"|popcnt|tzcnt|lzcnt|bs[fr]|blsr|blsi|blsmsk|adcx|adox)[bwlq]?", "DF"),
    (r"imul[wlq]?", "imul"),
    (r"(?:mul|div|idiv)[bwlq]?", "muldiv"),
    (r"xchg[bwlq]?|mulx[lq]?", "2"),
    (r"xadd[bwlq]?", "2F"),
    (r"cmpxchg[bwlq]?", "cmpxchg"),
    (r"lahf", "lahf"),
    (r"cltq|cwtl|cbtw", "cltq"),
    (r"cqto|cltd|cwtd", "cqto"),
    (r"push[wlq]?", "push"),
    (r"pop[wlq]?", "pop"),
    (r"call[lq]?", "call"),
    (r"v(?:mov(?:aps|ups|apd|upd|dq[au]|ss|sd|d|q|[hl]p[sd]|hlps|lhps|[sd]dup|shdup|sldup|nt(?:dqa?|p[sd])|mskp[sd])"
     r"|broadcast(?:s[sd]|[fi]128)|pbroadcast[bwdq]|cvt\w+|(?:insert|extract)(?:[fi]128|ps)|pinsr[bwdq]|pextr[bwdq]"
     r"|perm(?:[dq]|p[sd]|ilp[sd]|2[fi]128)|fn?m(?:addsub|subadd|add|sub)(?:132|213|231)[ps][sd]"
     r"|(?:add|sub|mul|div|min|max|sqrt|rcp|rsqrt|round|and|andn|or|xor|hadd|hsub|addsub|blendv?|shuf|unpck[hl]|dp)"
     r"(?:p[sd]|s[sd])|cmp\w*[ps][sd]"
     r"|p(?:adds?u?[bwdq]|subs?u?[bwdq]|madd\w+|mul\w+|andn?|x?or|(?:min|max)[su][bwdq]|avg[bw]|abs[bwd]|sign[bwd]"
     r"|shuf(?:b|d|[hl]w)|unpck[hl](?:bw|wd|dq|qdq)|ack[su]s(?:wb|dw)|s(?:ll|rl|ra)v?[wdq]|s[lr]ldq|cmp(?:eq|gt)[bwdq]"
     r"|mov[sz]x(?:bw|bd|bq|wd|wq|dq)|movmskb|alignr|blend(?:w|d|vb)|h(?:add|sub)s?[wd]|sadbw))", "D"),
))
# An integer register operand and the part of its 64-bit family it names (bits 64, 32, 16, 8, or "h" for 8..15).
GPR_PARTS = tuple((re.compile(p), part) for p, part in (
    (r"%r([abcd]x|si|di|bp|sp|\d+)", 64), (r"%e([abcd]x|si|di|bp|sp)", 32), (r"%r(\d+)d", 32),
    (r"%([abcd]x|si|di|bp|sp)", 16), (r"%r(\d+)w", 16), (r"%([abcd])l", 8), (r"%(si|di|bp|sp)l", 8),
    (r"%r(\d+)b", 8), (r"%([abcd])h", "h"),
))
OPERAND = re.compile(r"\$-?(?:0x)?[0-9a-f]+|%[xyz]mm\d+"  # an immediate, a vector register or a memory operand
                     r"|(?:%[c-gs]s:)?(?:-?0x[0-9a-f]+)?\((?:%\w+)?(?:,%\w+(?:,[1248])?)?\)")
VECTOR_BYTES = {"x": 16, "y": 32, "z": 64}
SUFFIX_BYTES = {"b": 1, "w": 2, "l": 4, "q": 8}
PART_BYTES = {64: 8, 32: 4, 16: 2, 8: 1, "h": 1}
Writes = collections.namedtuple("Writes", "regs mem flags")


def listing(path):
    with open(path, "rb") as f:
        elf = f.read(4) == b"\x7fELF"
    if elf:
        out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", path], check=True,
                             capture_output=True, text=True, errors="replace").stdout
        return out.splitlines()
    return open(path, errors="replace").read().splitlines()


def functions(lines):
    """{name: [(address, instruction text)]} in listing order."""
    fns, cur = collections.OrderedDict(), None
    for line in lines:
        m = FUNC.match(line)
        if m:
            cur = m.group(1)
            fns.setdefault(cur, [])
            continue
        if cur is None or "\t" not in line:
            continue
        addr, ins = line.split("\t", 1)
        try:
            a = int(addr.strip().rstrip(":"), 16)
        except ValueError:
            continue
        fns[cur].append((a, ins.strip()))
    return fns


def mnemonic(ins):
    return ins.split()[0] if ins.split() else ""


def legacy(fns):
    """[(function, kind, mnemonic, count)] for the two lists `legacy` prints."""
    rows = []
    for name, body in fns.items():
        ymm = any("%ymm" in i for _, i in body)
        sse4a = collections.Counter(mnemonic(i) for _, i in body if mnemonic(i) in SSE4A)
        for m, c in sorted(sse4a.items()):
            rows.append((name, "sse4a" + ("-in-avx" if ymm else ""), m, c))
        if ymm:
            leg = collections.Counter(
                mnemonic(i) for _, i in body if "%xmm" in i and not mnemonic(i).startswith("v") and mnemonic(i) not in SSE4A
            )
            for m, c in sorted(leg.items()):
                rows.append((name, "legacy-in-avx", m, c))
    return rows


def loop(fns, name):
    if name not in fns:
        raise KeyError(f"no function {name!r} in the listing")
    body = fns[name]
    best = None
    for a, ins in body:
        m = JUMP.match(ins)
        if m:
            t = int(m.group(2), 16)
            if t < a and (best is None or a - t > best[1] - best[0]):
                best = (t, a)
    if best is None:
        raise KeyError(f"{name}: no backward branch")
    ins = [i for a, i in body if best[0] <= a <= best[1]]
    hist = collections.Counter(mnemonic(i) for i in ins)
    mem = sum(1 for i in ins if "(" in i and not mnemonic(i).startswith("j"))
    stack = sum(1 for i in ins if "(%rsp)" in i)
    return len(ins), mem, stack, hist


def normalize(ins):
    ins = re.sub(r"#.*$", "", ins)
    ins = re.sub(r"<([^>+]*)\+0x[0-9a-f]+>", r"<\1+OFF>", ins)
    ins = re.sub(r"\b[0-9a-f]{4,}\b", "ADDR", ins)
    ins = re.sub(r"-?0x[0-9a-f]+\(%rip\)", "X(%rip)", ins)
    return ins.strip()


def diff(a, b):
    """([(function, base count, new count)], [(function, side)], identical count)."""
    moved, alone, same = [], [], 0
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            alone.append((name, "new" if name in b else "base"))
            continue
        x = [normalize(i) for _, i in a[name]]
        y = [normalize(i) for _, i in b[name]]
        if x == y:
            same += 1
        else:
            moved.append((name, len(x), len(y)))
    return moved, alone, same


# ---- tile-units: the host expert tier's tile cost rows ------------------------------------------------------
#
# `tile_units` in crates/model/src/ops.rs prices one unit of a tile kernel for the lane cut, in half-instruction
# units per block (a 256-value super-block, or a 32-value block for the types whose k_granularity is 32): the
# fixed unpack of one run of a weight row, and each column's share, fitted as fixed + C * per_col through the
# `qdot::tile::<C>` instantiations C = 2 and 4, read back at C = 3, 5 and 6. qdot's `tile::<C>` is one function holding every kind's kernel, so a
# kind's loop is found in the function's control-flow graph: the jump table of its `match kind` gives each kind's
# entry block, the natural loops (a back edge to a dominating block) under that entry are the kind's loops, and the
# outer loop with the most instructions is the block loop. A loop inside it runs its trip count, read off its
# latch (`add $s, %r; cmp $n, %r; jne header` from a zeroed %r, or — a loop whose count the compiler carried in
# a flag register — a `jmp header` latch with the exit `test $1, %r; je` inside and the flag cleared in the
# latch); a trip count the code does not show is a named
# error, never a guess. The failure mode this blocks: a qdot tile change (a kernel, a column count, the inlining
# around it) that leaves the balancer's table stale, so the lanes are cut by a cost the code no longer has.
#
# A type with a wide kernel (qdot's `has_wide`) is priced at runs of up to qdot::WIDE_COLS columns, so its row is
# fitted through the wide kernel's block loop at WIDE_FIT columns instead. That kernel takes its column count at run
# time: its block loop holds a group loop whose trip dispatches on the group's width, and at a whole number of groups
# every trip runs the full group's arm. So a trip of the group loop is its heaviest path (its inner loops at their
# trips), the group loop runs C / WIDE_GROUP trips, and the block loop's trip is its own heaviest path around it.

FIT_COLS = (2, 4)       # the column counts the rows are fitted through (two points, the form the table's first rows were derived in)
LIN_COLS = (3, 5, 6)    # read back against the fit: a loop that stops being straight-line code shows here
TILE_COLS = 8           # qdot::TILE_COLS: the widest run a tile takes
WIDE_FIT = (16, 32)     # a wide kernel's rows are fitted through these runs (whole groups of qdot::WIDE_GROUP)
WIDE_FNS = ("qdot::dot_{t}_q8k_wide_avx2", "qdot::wide_{t}", "qdot::dot_row_cols_wide")  # the first one the listing holds: inlining moves the kernel up its callers
TOL = 0.05              # a table row stands while its cost, at every run width 1..TILE_COLS, is within this
X4_BLOCKS = 4           # a 32-value type's block loop takes one x4 group (qdot::Q82X4) of four blocks an iteration
ROW_FMT = "        GgmlType::{ty} => Some(({fixed}, {per_col})),"


class LoopUnread(Exception):
    """A wide kernel's loops the reader cannot price: its row is refused by name, the other rows still derive."""


class TileError(Exception):
    """The listing does not have the shape the derivation reads; the message names what is missing."""


def text_sections(lines):
    """{function: its `.text.*` section name} from the `Disassembly of section` headers."""
    out, sec = {}, None
    for line in lines:
        m = re.match(r"Disassembly of section (.*):$", line)
        if m:
            sec = m.group(1)
            continue
        m = FUNC.match(line)
        if m and sec:
            out.setdefault(m.group(1), sec)
    return out


def jump_table(rel_lines, section):
    """The targets, as offsets into the function, of the jump table in the function's `.rodata.*` section: the
    PC-relative relocations of its 32-bit entries, entry `o` holding `target - table`, so target = addend - o."""
    rodata = section.replace(".text.", ".rodata.", 1)
    cur, ent = None, []
    for line in rel_lines:
        m = re.match(r"RELOCATION RECORDS FOR \[(.*)\]:", line)
        if m:
            cur = m.group(1)
            continue
        m = re.match(r"([0-9a-f]+)\s+R_X86_64_PC32\s+\S+?\+0x([0-9a-f]+)", line)
        if m and cur == rodata:
            ent.append((int(m.group(1), 16), int(m.group(2), 16)))
    return [a - o for o, a in sorted(ent)]


def gpr_part(op):
    """(64-bit family, part) of an integer register operand: `%eax` is (`rax`, 32), `%r9b` (`r9`, 8), `%dil`
    (`rdi`, 8), `%ah` (`rax`, "h", bits 8..15); else None."""
    for pat, part in GPR_PARTS:
        m = pat.fullmatch(op.strip())
        if m:
            name = m.group(1)
            return "r" + (name + "x" if name in ("a", "b", "c", "d") else name), part
    return None


def gpr(op):
    """The 64-bit register family an integer register operand names, else None."""
    part = gpr_part(op)
    return part[0] if part else None


def operands(ins):
    """Operands with the commas inside an address kept together."""
    parts = ins.split(None, 1)
    return [o.strip() for o in re.split(r",(?![^()]*\))", parts[1].split("#")[0])] if len(parts) > 1 else []


@functools.lru_cache(maxsize=None)
def writes(ins):
    """Every write of one instruction, from WRITE_CLASSES: `regs` {family: (constant or None, part written)}, `mem`
    [(memory operand or "*" for any memory, bytes or None, constant or None)], `flags` whether it writes the condition
    flags. A constant is a move's immediate or a zeroing xor's 0. An instruction or operand form the classes do not
    list is a TileError naming it: what is not known to write nothing is never read as writing nothing."""
    ins = re.sub(r"^(?:(?:data16|cs|ds|lock|notrack)(?:\s+|$))+", "", ins.strip())
    if not ins:
        return Writes({}, [], False)
    mn, ops = mnemonic(ins), operands(ins)
    kind = next((k for pat, k in WRITE_CLASSES if pat.fullmatch(mn)), None)
    if kind is None:
        raise TileError(f"`{ins}`: {mn} is not an instruction the write classes list")
    if kind == "":
        return Writes({}, [], False)
    if kind == "call":  # the callee may write the caller-saved registers, any memory and the flags
        return Writes(dict.fromkeys(CALLER_SAVED, (None, 64)), [("*", None, None)], True)
    for o in ops:
        if not gpr_part(o) and not OPERAND.fullmatch(o):
            raise TileError(f"`{ins}`: operand {o} is not a form the write classes read")
    regs, mem = {}, []
    # a store's bytes: the widest register operand or integer suffix, never fewer than it writes
    sizes = [PART_BYTES[p[1]] for p in map(gpr_part, ops) if p]
    sizes += [VECTOR_BYTES[o[1]] for o in ops if re.fullmatch(r"%[xyz]mm\d+", o)]
    if not mn.startswith("v") and mn[-1] in SUFFIX_BYTES:
        sizes.append(SUFFIX_BYTES[mn[-1]])

    def put(op, value=None):
        part = gpr_part(op)
        if part:
            regs[part[0]] = (value, part[1])
        elif "(" in op:
            mem.append((op, max(sizes, default=None), value))

    if kind in ("D", "DF") and ops:
        value = (int(ops[0][1:], 16) if re.fullmatch(r"mov[bwlq]?|movabsq", mn) and ops[0].startswith("$") else
                 0 if mn.startswith("xor") and len(ops) == 2 and ops[0] == ops[1] else None)
        put(ops[-1], value)
    elif kind in ("2", "2F") and len(ops) >= 2:
        if ops[-1] != ops[-2]:  # an exchange of an operand with itself (`xchg %ax, %ax` padding) changes nothing
            put(ops[-2])
            put(ops[-1])
    elif kind == "imul" and len(ops) > 1:
        put(ops[-1])
    elif kind in ("imul", "muldiv") and len(ops) == 1:
        regs.update(rax=(None, 64), rdx=(None, 64))
    elif kind == "cmpxchg" and len(ops) == 2:
        put(ops[-1])
        regs.setdefault("rax", (None, 64))
    elif kind in ("lahf", "cltq", "cqto") and not ops:
        regs["rdx" if kind == "cqto" else "rax"] = (None, "h" if kind == "lahf" else 64)
    elif kind in ("push", "pop") and len(ops) == 1:
        if kind == "pop":
            put(ops[0])
        regs["rsp"] = (None, 64)
    elif kind != "F":
        raise TileError(f"`{ins}`: {len(ops)} operands, not a form of {mn} the write classes read")
    return Writes(regs, mem, kind in ("F", "DF", "2F", "imul", "muldiv", "cmpxchg"))


class Cfg:
    """The basic blocks of one function, their dominators and natural loops. The function's one indirect jump is
    its jump table's dispatch (`table`: entry offsets); a function with neither has no dispatch, and a second
    indirect jump is not modelled."""

    def __init__(self, body, table):
        self.body = body
        self.pos = {a: i for i, (a, _) in enumerate(body)}
        ind = [i for i, (_, ins) in enumerate(body) if re.match(r"jmpq?\s+\*", ins)]
        if (len(ind), bool(table)) not in ((1, True), (0, False)):
            raise TileError(f"want one indirect jump and its jump table, or neither; found {len(ind)} jumps and "
                            f"{len(table)} entries")
        self.entries = []
        for t in table:
            if t not in self.pos:
                raise TileError(f"jump table entry +{t:#x} is no instruction")
            self.entries.append(self.pos[t])
        lead = {0} | set(self.entries)
        for i, (_, ins) in enumerate(body):
            mn = mnemonic(ins)
            if mn.startswith("j") or mn in ("ret", "retq", "ud2"):
                if i + 1 < len(body):
                    lead.add(i + 1)
                m = JUMP.match(ins)
                if m and int(m.group(2), 16) in self.pos:
                    lead.add(self.pos[int(m.group(2), 16)])
        starts = sorted(lead)
        self.bid = {s: k for k, s in enumerate(starts)}
        self.entry_blocks = [self.bid[t] for t in self.entries]
        self.blocks = [(s, starts[k + 1] if k + 1 < len(starts) else len(body)) for k, s in enumerate(starts)]
        n = len(self.blocks)
        self.succ = [[] for _ in range(n)]
        for k, (s, e) in enumerate(self.blocks):
            ins = body[e - 1][1]
            mn = mnemonic(ins)
            m = JUMP.match(ins)
            if mn.startswith("j"):
                if mn in ("jmp", "jmpq") and not m:
                    self.succ[k] = list(self.entry_blocks)
                    continue
                if m and int(m.group(2), 16) in self.pos:
                    self.succ[k].append(self.bid[self.pos[int(m.group(2), 16)]])
                if mn not in ("jmp", "jmpq") and e < len(body):
                    self.succ[k].append(self.bid[e])
            elif mn not in ("ret", "retq", "ud2") and e < len(body):
                self.succ[k].append(self.bid[e])
        self.pred = [[] for _ in range(n)]
        for k in range(n):
            for t in set(self.succ[k]):
                self.pred[t].append(k)
        self._dominators()
        self._loops()

    def _dominators(self):
        post, seen, stack = [], {0}, [(0, iter(self.succ[0]))]
        while stack:
            v, it = stack[-1]
            for w in it:
                if w not in seen:
                    seen.add(w)
                    stack.append((w, iter(self.succ[w])))
                    break
            else:
                post.append(v)
                stack.pop()
        rpo = post[::-1]
        num = {v: i for i, v in enumerate(rpo)}
        idom = {0: 0}

        def meet(a, b):
            while a != b:
                while num[a] > num[b]:
                    a = idom[a]
                while num[b] > num[a]:
                    b = idom[b]
            return a

        changed = True
        while changed:
            changed = False
            for v in rpo[1:]:
                ps = [p for p in self.pred[v] if p in idom]
                if not ps:
                    continue
                d = ps[0]
                for p in ps[1:]:
                    d = meet(p, d)
                if idom.get(v) != d:
                    idom[v] = d
                    changed = True
        self.idom = idom

    def dom(self, a, b):
        """Whether block `a` dominates block `b`."""
        while b in self.idom:
            if a == b:
                return True
            if b == 0:
                return False
            b = self.idom[b]
        return False

    def _loops(self):
        loops = {}
        for u in self.idom:
            for v in set(self.succ[u]):
                if v in self.idom and self.dom(v, u):
                    body, stack = loops.setdefault(v, {v}), [u]
                    while stack:
                        x = stack.pop()
                        if x not in body:
                            body.add(x)
                            stack += self.pred[x]
        self.loops = loops

    def count(self, blocks):
        """Instructions in `blocks`, alignment padding left out."""
        n = 0
        for b in blocks:
            s, e = self.blocks[b]
            n += sum(1 for _, ins in self.body[s:e] if not re.match(r"(nop|data16|cs nop|xchg\s+%ax)", ins))
        return n

    def trips(self, header):
        """The trip count of the loop at block `header`, from its latch and its zeroed start."""
        blocks = self.loops[header]
        latches = [u for u in blocks if header in self.succ[u]]
        where = f"the loop at {self.body[self.blocks[header][0]][0]:#x}"
        if len(latches) != 1:
            raise TileError(f"{where}: {len(latches)} back edges, the trip count is not read")
        s, e = self.blocks[latches[0]]
        ins = [i for _, i in self.body[s:e]]
        if ins and mnemonic(ins[-1]) in ("jmp", "jmpq") and not re.match(r"jmpq?\s+\*", ins[-1]):
            return self._flag_trips(header, latches[0], where)
        if len(ins) < 3 or not JUMP.match(ins[-1]) or mnemonic(ins[-1]) != "jne" or not mnemonic(ins[-2]).startswith("cmp"):
            raise TileError(f"{where}: the latch does not end `cmp $n, %r; jne`")
        c_ops = operands(ins[-2])
        reg = gpr(c_ops[-1]) if len(c_ops) == 2 and c_ops[0].startswith("$") else None
        adds = [operands(i) for i in ins[:-2] if mnemonic(i).startswith("add") and len(operands(i)) == 2 and gpr(operands(i)[1]) == reg]
        if reg is None or not adds or not adds[-1][0].startswith("$"):
            raise TileError(f"{where}: the latch has no `add $s, %r` of the compared register")
        n, step = int(c_ops[0][1:], 16), int(adds[-1][0][1:], 16)
        if step <= 0 or n % step:
            raise TileError(f"{where}: compare {n:#x} is not a multiple of step {step:#x}")
        if sum(self._sets(i, reg) is not False for b in blocks for _, i in self.body[slice(*self.blocks[b])]) != 1:
            raise TileError(f"{where}: %{reg} is written in the loop besides the latch's add")
        if not self._starts_at_zero(header, reg):
            raise TileError(f"{where}: %{reg} is not shown to start at zero")
        return n // step

    def _flag_trips(self, header, latch, where):
        """The trip count of a loop whose latch closes with `jmp` and the count left the latch for a flag
        operand: a block of the loop ends `test $1, FLAG; je` to outside — FLAG a register or a stack slot,
        wherever the compiler kept it — the flag written 1 on entry and cleared in the latch, its only write
        inside the loop, so exactly one latch pass runs and the body takes 2 trips. The latch also writes the
        induction register its last value (`movl $k, %r`, the register zeroed on entry and untouched inside
        the loop besides the latch), and that count must agree: k + 1 = 2. Anything else is a named error."""
        blocks = self.loops[header]
        flag = shown = None
        for b in blocks:
            s, e = self.blocks[b]
            body = [i for _, i in self.body[s:e]]
            m = JUMP.match(body[-1]) if body else None
            if not (m and mnemonic(body[-1]) == "je"):
                continue
            t = self.bid.get(self.pos.get(int(m.group(2), 16), -1))
            if t is None or t in blocks:
                continue
            # the test may sit anywhere in the block, nothing that writes the flags between it and the je
            for k in range(len(body) - 2, -1, -1):
                if not mnemonic(body[k]).startswith("test"):
                    if writes(body[k]).flags:
                        break
                    continue
                t_ops = operands(body[k])
                part = gpr_part(t_ops[-1]) if t_ops else None
                # bit 0 of a register's low part or of a stack slot's first byte; %ah..%dh hold bits 8..15
                if len(t_ops) == 2 and t_ops[0] == "$0x1" and (
                        part and part[1] != "h" or STACK_SLOT.fullmatch(t_ops[1])):
                    flag, shown = part[0] if part else t_ops[1], t_ops[1]
                break
            if flag is not None:
                break
        if flag is None:
            raise TileError(f"{where}: the latch closes with `jmp` and no block of the loop ends "
                            "`test $1, %r; je` to outside, the flag-tested shape")
        s, e = self.blocks[latch]
        latch_ins = [i for _, i in self.body[s:e]]
        if not any((v := self._sets(i, flag, low=True)) is not False and v == 0 for i in latch_ins):
            raise TileError(f"{where}: the flag {shown} is not cleared in the latch")
        inside = [i for b in blocks if b != latch for _, i in self.body[self.blocks[b][0]:self.blocks[b][1]]]
        if any(self._sets(i, flag) is not False for i in inside):
            raise TileError(f"{where}: the flag {shown} is written inside the loop besides the latch")
        if self._entry_value(header, flag, low=True) != 1:
            raise TileError(f"{where}: the flag {shown} is not shown to start at one")
        counted = [(v, r) for i in latch_ins for r in writes(i).regs
                   if (v := self._sets(i, r)) is not None and v > 0 and self._entry_value(header, r) == 0]
        if len(counted) != 1:
            raise TileError(f"{where}: the latch writes {len(counted)} registers a positive immediate "
                            "zeroed on entry, want the induction register's last value")
        k, reg = counted[0]
        if any(self._sets(i, reg) is not False for i in inside):
            raise TileError(f"{where}: the induction register %{reg} is written inside the loop "
                            "besides the latch")
        if k != 1:
            raise TileError(f"{where}: the induction register's last value {k} disagrees with the "
                            "flag's two trips")
        return k + 1

    def _entry_value(self, header, loc, low=False):
        """The constant the entry path into the loop at `header` last writes to `loc` (`_sets`), a stack slot also
        through the register its last store copied (one indirection: `movl $1, %ecx; movl %ecx, 0x20(%rsp)`), or None
        when no constant is shown."""
        want = loc
        for ins in self._entry_ins(header):
            value = self._sets(ins, want, low)
            if value is False:
                continue
            ops = operands(ins)
            src = gpr_part(ops[0]) if want == loc and STACK_SLOT.fullmatch(loc) and len(ops) == 2 else None
            if value is None and src and src[1] != "h" and re.fullmatch(r"mov[bwlq]?", mnemonic(ins)) and ops[1] == loc:
                want = src[0]
                continue
            return value
        return None

    def _entry_ins(self, header):
        """The instructions of the loop's single-entry path, nearest first (each block's own instructions
        reversed, then its predecessor's), up to 8 blocks outside the loop; nothing when the entry is not a
        chain of single predecessors."""
        outside = [p for p in self.pred[header] if p not in self.loops[header]]
        for _ in range(8):
            if len(outside) != 1:
                return
            s, e = self.blocks[outside[0]]
            yield from (ins for _, ins in reversed(self.body[s:e]))
            outside = self.pred[outside[0]]

    def _starts_at_zero(self, header, reg):
        return self._entry_value(header, reg) == 0

    def inner_loops(self, header):
        """The headers of the loops directly inside the loop at `header`."""
        blocks = self.loops[header]
        inner = [h for h, b in self.loops.items() if h != header and h in blocks and b < blocks]
        return [h for h in inner if not any(g != h and h in self.loops[g] and self.loops[h] < self.loops[g] for g in inner)]

    def dynamic(self, header):
        """Instructions one trip of the loop at `header` runs: its blocks once, an inner loop by its trips."""
        blocks = self.loops[header]
        top = self.inner_loops(header)
        own = blocks - set().union(*[self.loops[h] for h in top]) if top else blocks
        return self.count(own) + sum(self.trips(h) * self.dynamic(h) for h in top)

    def heaviest(self, header, inner):
        """Instructions on the heaviest path of one trip of the loop at `header`, from the header to a back edge:
        a loop directly inside it is one step of `inner(its header)` instructions, every other block its own count.
        A trip that takes one of several arms runs the heaviest."""
        blocks = self.loops[header]
        top = set(self.inner_loops(header))
        rep = {b: b for b in blocks}
        for h in top:
            for b in self.loops[h]:
                rep[b] = h
        succ = collections.defaultdict(set)
        for u in blocks:
            for v in self.succ[u]:
                if v in blocks and v != header and rep[v] != rep[u]:
                    succ[rep[u]].add(rep[v])
        latch = {rep[u] for u in blocks if header in self.succ[u]}
        where = f"the loop at {self.body[self.blocks[header][0]][0]:#x}"
        memo, walking = {}, set()

        def best(r):
            if r not in memo:
                if r in walking:
                    raise TileError(f"{where}: a cycle of blocks no loop inside it accounts for (a body whose "
                                    "paths re-enter their own cycle, so no latch shows its trips)")
                walking.add(r)
                ways = [best(w) for w in succ[r]] + ([0] if r in latch else [])
                if not ways:
                    raise TileError(f"{where}: the block at {self.body[self.blocks[r][0]][0]:#x} reaches no back edge")
                walking.discard(r)
                memo[r] = (inner(r) if r in top else self.count([r])) + max(ways)
            return memo[r]

        return best(header)

    @staticmethod
    def _sets(ins, loc, low=False):
        """What `ins` writes to `loc`, a register family or a stack slot, by `writes`: False when it leaves `loc`
        alone, the constant when it writes one, else None. A register's constant counts when the write covers all of
        it (its 32- or 64-bit name), or with `low` (a flag's bit 0) its low byte or word too. A slot's constant counts
        from a store to the slot alone; any other store that may reach its SLOT_BYTES (another base or index, an
        unknown width, a call) or a write of %rsp makes it None."""
        w = writes(ins)
        slot = STACK_SLOT.fullmatch(loc)
        if not slot:
            if loc not in w.regs:
                return False
            value, part = w.regs[loc]
            return value if part in (32, 64) or low and part in (8, 16) else None
        if "rsp" in w.regs:
            return None
        at, hits = int(slot[1] or "0", 16), []
        for dst, size, value in w.mem:
            store = STACK_SLOT.fullmatch(dst)
            start = int(store[1] or "0", 16) if store else None
            if start is None or size is None or (start < at + SLOT_BYTES and at < start + size):
                hits.append((dst, value))
        if not hits:
            return False
        return hits[0][1] if len(w.mem) == 1 and hits[0][0] == loc else None


def tile_arms(lines, rel_lines, c, nkinds):
    """For `qdot::tile::<c>`: [(outer loop instruction counts a trip runs, largest first)] by kind index."""
    secs = text_sections(lines)
    fns = functions(lines)
    names = [n for n in fns if re.fullmatch(rf"qdot::tile::?<{c}>", n)]
    if len(names) != 1:
        raise TileError(f"want one qdot::tile::<{c}> in the listing, found {len(names)}")
    table = jump_table(rel_lines, secs[names[0]])
    if len(table) != nkinds:
        raise TileError(f"qdot::tile::<{c}>: the jump table has {len(table)} entries for {nkinds} TileKind variants")
    cfg = Cfg(fns[names[0]], table)
    arms = []
    for e in cfg.entry_blocks:
        under = [h for h in cfg.loops if cfg.dom(e, h)]
        outer = [h for h in under if not any(g != h and h in cfg.loops[g] and cfg.loops[h] < cfg.loops[g] for g in under)]
        arms.append(sorted((cfg.dynamic(h) for h in outer), reverse=True))
    return arms


def qdot_tiles(src):
    """([TileKind variants in declaration order], {GgmlType name: variant}, {GgmlType name: k_granularity}) from
    qdot's source: the enum, `tile_kind`'s arms and `k_granularity`'s."""
    m = re.search(r"enum TileKind \{(.*?)\}", src, re.S)
    kinds = re.findall(r"^\s*(\w+),", m.group(1), re.M) if m else []
    tk = re.search(r"fn tile_kind\(.*?\{(.*?)has_features", src, re.S)
    ty_kind = dict(re.findall(r"GgmlType::(\w+) => TileKind::(\w+)", tk.group(1))) if tk else {}
    kg = re.search(r"pub fn k_granularity\(.*?\{(.*?)\n\}", src, re.S)
    small = set(re.findall(r"GgmlType::(\w+)", kg.group(1).split("=>")[0])) if kg else set()
    gran = {t: 32 if t in small else 256 for t in ty_kind}
    if not kinds or not ty_kind or not kg or set(ty_kind.values()) != set(kinds):
        raise TileError("qdot's TileKind, tile_kind or k_granularity is not in the shape the derivation reads")
    return kinds, ty_kind, gran


def qdot_wide(src):
    """({GgmlType names qdot's `has_wide` takes}, WIDE_COLS, WIDE_GROUP) from qdot's source; a qdot with no
    `has_wide` has no wide kernel (an empty set)."""
    m = re.search(r"pub fn has_wide\(.*?\{(.*?)\n\}", src, re.S)
    if not m:
        return set(), None, None
    tys = set(re.findall(r"GgmlType::(\w+)", m.group(1)))
    cols = re.search(r"pub const WIDE_COLS: usize = (\d+);", src)
    group = re.search(r"const WIDE_GROUP: usize = (\d+);", src)
    if not tys or not cols or not group:
        raise TileError("qdot's has_wide, WIDE_COLS or WIDE_GROUP is not in the shape the derivation reads")
    cols, group = int(cols.group(1)), int(group.group(1))
    if any(c % group or c > cols for c in WIDE_FIT):
        raise TileError(f"the wide runs {WIDE_FIT} are not whole groups of WIDE_GROUP = {group} within WIDE_COLS = {cols}")
    return tys, cols, group


def wide_units(lines, rel_lines, ty, group):
    """(function, {C: instructions the wide kernel's block loop runs a trip at C columns}) for each C of WIDE_FIT. The
    kernel is the first of WIDE_FNS the listing holds; its block loop is its outer loop with the most instructions,
    the group loop the loop directly inside that with the most."""
    fns, secs = functions(lines), text_sections(lines)
    names = [n.format(t=ty.lower().replace("_", "")) for n in WIDE_FNS]
    held = [n for n in names if n in fns]
    if not held:
        raise TileError(f"{ty} has a wide kernel and the listing holds none of {', '.join(names)}")
    name = held[0]
    cfg = Cfg(fns[name], jump_table(rel_lines, secs[name]))
    try:
        return name, wide_loops(cfg, name, group)
    except TileError as e:
        raise LoopUnread(str(e)) from e


def wide_loops(cfg, name, group):
    """{C: instructions} of `wide_units` for the kernel `name` whose control flow is `cfg`."""
    outer = [h for h in cfg.loops if not any(g != h and h in cfg.loops[g] and cfg.loops[h] < cfg.loops[g] for g in cfg.loops)]
    if not outer:
        raise TileError(f"{name}: no loop")
    block = max(outer, key=lambda h: cfg.count(cfg.loops[h]))
    inner = cfg.inner_loops(block)
    if not inner:
        raise TileError(f"{name}: the block loop holds no group loop")
    grp = max(inner, key=lambda h: cfg.count(cfg.loops[h]))
    trip = cfg.heaviest(grp, lambda h: cfg.trips(h) * cfg.dynamic(h))
    return {c: cfg.heaviest(block, lambda h, c=c: c // group * trip if h == grp else cfg.trips(h) * cfg.dynamic(h))
            for c in WIDE_FIT}


def ops_tile_units(src):
    """{GgmlType name: (fixed, per_col)} from `tile_units` in ops.rs."""
    m = re.search(r"fn tile_units\(ty: GgmlType\) -> Option<\(u64, u64\)> \{\s*match ty \{(.*?)\n    \}", src, re.S)
    if not m:
        raise TileError("no `fn tile_units` in the ops source")
    return {t: (int(f), int(p)) for t, f, p in re.findall(r"GgmlType::(\w+) => Some\(\((\d+), (\d+)\)\)", m.group(1))}


def fit(points):
    """Least-squares fixed + C * per_col through [(C, units)]."""
    n = len(points)
    mx, my = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    slope = sum((x - mx) * (y - my) for x, y in points) / sum((x - mx) ** 2 for x, _ in points)
    return my - slope * mx, slope


def tile_units(lines, rel_lines, qdot_src, ops_src):
    """(rows, red): one row per type a tile kernel serves, `{ty, kind, gran, counts, fixed, per_col, lin, table,
    dev, status}`, and the number of rows that are not green."""
    kinds, ty_kind, gran = qdot_tiles(qdot_src)
    wide_tys, wide_cols, group = qdot_wide(qdot_src)
    table = ops_tile_units(ops_src)
    cs = sorted(FIT_COLS + LIN_COLS)
    per_c = {c: tile_arms(lines, rel_lines, c, len(kinds)) for c in cs}
    rows, red = [], 0
    for ty, kind in ty_kind.items():
        k = kinds.index(kind)
        want = 1 if gran[ty] == 256 else 2
        for c in cs:
            if len(per_c[c][k]) != want:
                raise TileError(f"{ty} at C = {c}: {len(per_c[c][k])} outer loops, want {want} "
                                f"({'the block loop' if want == 1 else 'the x4 group loop and its tail'})")
        blocks = 1 if gran[ty] == 256 else X4_BLOCKS
        units = {c: 2 * per_c[c][k][0] / blocks for c in cs}
        f, p = fit([(c, units[c]) for c in FIT_COLS])
        row = {"ty": ty, "kind": kind, "gran": gran[ty], "counts": {c: per_c[c][k][0] for c in cs},
               "fixed": round(f), "per_col": round(p),
               "lin": max(abs(f + c * p - units[c]) / units[c] for c in LIN_COLS), "table": table.get(ty)}
        runs = TILE_COLS
        if ty in wide_tys:
            try:
                row["wide_fn"], row["wide"] = wide_units(lines, rel_lines, ty, group)
            except LoopUnread as e:
                row["fixed"], row["per_col"], row["dev"], row["status"] = None, None, None, f"REFUSED ({e})"
                red += 1
                rows.append(row)
                continue
            f, p = fit([(c, 2 * row["wide"][c]) for c in WIDE_FIT])
            row["fixed"], row["per_col"], runs = round(f), round(p), wide_cols
        if row["table"] is None:
            row["dev"], row["status"] = None, "MISSING"
        else:
            tf, tp = row["table"]
            row["dev"] = max(abs((tf + c * tp) - (row["fixed"] + c * row["per_col"])) / (row["fixed"] + c * row["per_col"])
                             for c in range(1, runs + 1))
            row["status"] = "ok" if row["dev"] <= TOL else "STALE"
        red += row["status"] != "ok"
        rows.append(row)
    for ty in sorted(set(table) - set(ty_kind)):
        rows.append({"ty": ty, "kind": "-", "gran": 0, "counts": {}, "fixed": None, "per_col": None, "lin": 0.0,
                     "table": table[ty], "dev": None, "status": "NO-TILE"})
        red += 1
    return rows, red


def tile_units_report(rows, red):
    out = [f"{'type':8} {'kind':7} {'gran':>4} {'dynamic loop instrs C=' + ','.join(map(str, sorted(FIT_COLS + LIN_COLS))):28} "
           f"{'fit (fixed, per_col)':>20} {'lin':>6} {'table':>10} {'dev':>6}  status"]
    for r in rows:
        counts = ",".join(str(r["counts"][c]) for c in sorted(FIT_COLS + LIN_COLS)) if r["counts"] else "-"
        fit_s = f"({r['fixed']}, {r['per_col']})" if r["fixed"] is not None else "-"
        tab = f"({r['table'][0]}, {r['table'][1]})" if r["table"] else "-"
        dev = f"{100 * r['dev']:.1f}%" if r["dev"] is not None else "-"
        out.append(f"{r['ty']:8} {r['kind']:7} {r['gran']:>4} {counts:28} {fit_s:>20} {100 * r['lin']:5.1f}% {tab:>10} {dev:>6}  {r['status']}")
    for r in rows:
        if "wide" in r:
            out.append(f"{r['ty']}: fitted through its wide kernel {r['wide_fn']}, block loop instrs C="
                       + ",".join(map(str, WIDE_FIT)) + " " + ",".join(str(r["wide"][c]) for c in WIDE_FIT)
                       + "; the row stands at runs up to WIDE_COLS")
    out.append("rows (derived):")
    out += [ROW_FMT.format(ty=r["ty"], fixed=r["fixed"], per_col=r["per_col"]) for r in rows if r["fixed"] is not None]
    out.append(f"tile-units: {len(rows) - red} of {len(rows)} rows within {int(TOL * 100)} % of the derivation" if not red else
               f"tile-units: {red} of {len(rows)} rows missing, stale, refused or without a tile kernel: the table in ops.rs is not what the code derives")
    return "\n".join(out)

def canned_tile(cs=(2, 3, 4, 5, 6), trip_cmp="cmpq\t$0x20, %rax", tail=True, inner_latch="counted",
                inner="vaddps\t%ymm0, %ymm1, %ymm2", entry=()):
    """(listing lines, relocation lines) of `qdot::tile::<C>` functions shaped as qdot's: a jump-table dispatch on the
    kind to two arms. Arm 0, a 256-value type: a block loop of 5C + 6 instructions, a decode loop inside it. Arm 1, a
    32-value type: the x4 group loop (4C + 8) and a shorter tail loop. `inner_latch` picks arm 0's decode-loop latch:
    "counted", `add $n, %r; cmp $n, %r; jne` of eight trips (5C + 100 instructions a block, fixed 200, per_col 10);
    "flag", the count carried in a flag register (`test $1, %r; je` in the body, the flag cleared and the induction
    register written its last value in a `jmp` latch) of two trips (5C + 39, fixed 78, per_col 10); "spill", the same
    with the flag in a stack slot (5C + 40, fixed 80, per_col 10). Arm 1 runs 4C + 8 for four blocks (fixed 4,
    per_col 2) at every latch. `inner` opens the decode loop; `entry` instructions run just before it."""
    lst, rel = [], []
    for c in cs:
        prog = []  # (label or None, text); a jump names its target label

        def fill(n):
            prog.extend((None, "vaddps\t%ymm0, %ymm1, %ymm2") for _ in range(n))

        prog += [(None, "leaq\t(%rip), %rdi"), (None, "movslq\t(%rdi,%rax,4), %rax"), (None, "addq\t%rdi, %rax"),
                 (None, "jmpq\t*%rax")]
        prog.append(("A", "xorl\t%ecx, %ecx"))
        prog.append(("A_outer", "vxorps\t%xmm0, %xmm0, %xmm0"))
        fill(5 * c + 5)
        if inner_latch == "counted":
            prog.append((None, "xorl\t%eax, %eax"))
            prog += [(None, i) for i in entry]
            prog.append(("A_inner", inner))
            fill(7)
            prog += [(None, "addq\t$0x4, %rax"), (None, trip_cmp), (None, "jne\tA_inner")]
            fill(3)
        else:
            flag = "%ebx" if inner_latch == "flag" else "0x20(%rsp)"
            prog.append((None, "xorl\t%eax, %eax"))
            prog += [(None, "movl\t$0x1, %ecx")] if inner_latch == "spill" else [(None, "movl\t$0x1, %ebx")]
            prog += [(None, "movl\t%ecx, 0x20(%rsp)")] if inner_latch == "spill" else []
            prog += [(None, i) for i in entry]
            prog.append(("A_inner", inner))
            fill(7)
            prog += [(None, f"testb\t$0x1, {flag}"), (None, "je\tA_out")]
            prog += [(None, "movl\t$0x0, 0x20(%rsp)")] if inner_latch == "spill" else [(None, "xorl\t%ebx, %ebx")]
            prog += [(None, "movl\t$0x1, %eax"), (None, "jmp\tA_inner")]
            prog.append(("A_out", "vaddps\t%ymm0, %ymm1, %ymm2"))
            fill(2)
        prog += [(None, "cmpq\t%rdx, %rsi"), (None, "jne\tA_outer"), (None, "jmp\tEXIT")]
        prog.append(("B", "xorl\t%ecx, %ecx"))
        prog.append(("B_main", "vxorps\t%xmm0, %xmm0, %xmm0"))
        fill(4 * c + 5)
        prog += [(None, "cmpq\t%rdx, %rsi"), (None, "jne\tB_main")]
        if tail:
            prog.append(("B_tail", "vxorps\t%xmm0, %xmm0, %xmm0"))
            fill(c + 1)
            prog += [(None, "cmpq\t%rdx, %rsi"), (None, "jne\tB_tail")]
        prog.append((None, "jmp\tEXIT"))
        prog.append(("EXIT", "retq"))
        at = {lab: 4 * i for i, (lab, _) in enumerate(prog) if lab}
        name, sec = f"qdot::tile::<{c}>", f".text._RNtile{c}"
        lst += [f"Disassembly of section {sec}:", "", f"0000000000000000 <{name}>:"]
        for i, (_, text) in enumerate(prog):
            m = re.fullmatch(r"(j\w+)\t(\w+)", text)
            if m:
                text = f"{m.group(1)}\t{at[m.group(2)] :#x} <{name}+{at[m.group(2)]:#x}>"
            lst.append(f"{4 * i:8x}:      \t{text}")
        lst.append("")
        rel += [f"RELOCATION RECORDS FOR [.rodata._RNtile{c}]:", "OFFSET           TYPE                     VALUE"]
        rel += [f"{o:016x} R_X86_64_PC32            {sec}+{at[lab] + o:#x}" for o, lab in ((0, "A"), (4, "B"))]
    return lst, rel


def canned_wide(name="qdot::dot_q4k_q8k_wide_avx2"):
    """(listing lines, relocation lines) of a wide kernel shaped as qdot's: a block loop holding a pack loop of eight
    trips and a group loop, whose trip takes a range check to the full group's arm or a jump table to arms 1..7 (the
    full arm an inner loop of eight trips); a short reduction loop after. A block trip runs 105 instructions besides
    its groups and a full group's trip 160 (the heaviest arm), so C = 16 runs 425 and C = 32 745: fixed 210, per_col
    40 in units."""
    prog = []

    def fill(n):
        prog.extend((None, "vaddps\t%ymm0, %ymm1, %ymm2") for _ in range(n))

    prog += [(None, "pushq\t%rbp"), (None, "xorl\t%r8d, %r8d")]
    prog.append(("OUTER", "vxorps\t%xmm0, %xmm0, %xmm0"))
    fill(37)
    prog.append((None, "xorl\t%eax, %eax"))
    prog.append(("PACK", "vaddps\t%ymm0, %ymm1, %ymm2"))
    fill(3)
    prog += [(None, "addq\t$0x4, %rax"), (None, "cmpq\t$0x20, %rax"), (None, "jne\tPACK")]
    fill(5)
    prog += [(None, "testq\t%rdx, %rdx"), (None, "je\tBLATCH")]
    prog.append(("GROUP", "vaddps\t%ymm0, %ymm1, %ymm2"))
    fill(1)
    prog += [(None, "cmpq\t$0x6, %rcx"), (None, "ja\tARM8"), (None, "leaq\t(%rip), %rdi"),
             (None, "movslq\t(%rdi,%rcx,4), %rcx"), (None, "addq\t%rdi, %rcx"), (None, "jmpq\t*%rcx")]
    for g in range(1, 8):
        prog.append((f"ARM{g}", "vaddps\t%ymm0, %ymm1, %ymm2"))
        fill(10 * g - 1)
        prog.append((None, "jmp\tGLATCH"))
    prog.append(("ARM8", "vaddps\t%ymm0, %ymm1, %ymm2"))
    fill(39)
    prog.append((None, "xorl\t%eax, %eax"))
    prog.append(("A8", "vaddps\t%ymm0, %ymm1, %ymm2"))
    fill(10)
    prog += [(None, "addq\t$0x4, %rax"), (None, "cmpq\t$0x20, %rax"), (None, "jne\tA8")]
    prog += [("GLATCH", "addq\t$0x8, %r8"), (None, "cmpq\t%rdx, %r8"), (None, "jne\tGROUP")]
    prog += [("BLATCH", "vaddps\t%ymm0, %ymm1, %ymm2"), (None, "cmpq\t%rdx, %rsi"), (None, "jne\tOUTER")]
    prog.append(("HSUM", "vaddps\t%ymm0, %ymm1, %ymm2"))
    fill(2)
    prog += [(None, "cmpq\t%rdx, %rsi"), (None, "jne\tHSUM"), (None, "retq")]
    at = {lab: 4 * i for i, (lab, _) in enumerate(prog) if lab}
    sec = ".text._RNwide"
    lst = [f"Disassembly of section {sec}:", "", f"0000000000000000 <{name}>:"]
    for i, (_, text) in enumerate(prog):
        m = re.fullmatch(r"(j\w+)\t(\w+)", text)
        if m:
            text = f"{m.group(1)}\t{at[m.group(2)]:#x} <{name}+{at[m.group(2)]:#x}>"
        lst.append(f"{4 * i:8x}:      \t{text}")
    lst.append("")
    rel = ["RELOCATION RECORDS FOR [.rodata._RNwide]:", "OFFSET           TYPE                     VALUE"]
    rel += [f"{4 * (g - 1):016x} R_X86_64_PC32            {sec}+{at[f'ARM{g}'] + 4 * (g - 1):#x}" for g in range(1, 8)]
    return lst, rel


CANNED_QDOT = """enum TileKind {
    Qa,
    Qb,
}
fn tile_kind(w: GgmlType) -> Option<TileKind> {
    let kind = match w {
        GgmlType::Q4_K => TileKind::Qa,
        GgmlType::Q8_0 => TileKind::Qb,
        _ => return None,
    };
    has_features(w).then_some(kind)
}
pub fn k_granularity(w: GgmlType) -> usize {
    match w {
        GgmlType::Q5_0 | GgmlType::Q8_0 => 32,
        _ => 256,
    }
}
"""
CANNED_WIDE = """pub const WIDE_COLS: usize = 32;
#[cfg(target_arch = "x86_64")]
const WIDE_GROUP: usize = 8;
pub fn has_wide(w: GgmlType) -> bool {
    matches!(w, GgmlType::Q4_K) && has_features(w)
}
"""
CANNED_OPS = """fn tile_units(ty: GgmlType) -> Option<(u64, u64)> {
    match ty {
        GgmlType::Q4_K => Some((200, 10)),
        // a comment between the rows
        GgmlType::Q8_0 => Some((4, 2)),
        _ => None,
    }
}
"""


def tile_units_self_test():
    lst, rel = canned_tile()
    st = {r["ty"]: r for r in tile_units(lst, rel, CANNED_QDOT, CANNED_OPS)[0]}
    got = [(t, st[t]["fixed"], st[t]["per_col"], st[t]["counts"][4], st[t]["status"]) for t in ("Q4_K", "Q8_0")]
    want = [("Q4_K", 200, 10, 120, "ok"), ("Q8_0", 4, 2, 24, "ok")]
    if got != want:
        print(f"asm-census self-test: tile-units {got} != {want}")
        return 1
    # the derivation reads GNU objdump's jump spelling (no 0x) the same
    gnu = [re.sub(r"(\tj\w+\t)0x", r"\1", l) for l in lst]
    if tile_units(gnu, rel, CANNED_QDOT, CANNED_OPS)[0] != tile_units(lst, rel, CANNED_QDOT, CANNED_OPS)[0]:
        print("asm-census self-test: tile-units reads GNU objdump's listing differently")
        return 1
    # a latch that carries its count in a flag — `test $1, %r; je` in the body, the flag cleared and the induction
    # register written its last value in a `jmp` latch — reads two trips, in a register and spilled to a slot alike
    for what, latch, want, slot in (
        ("a flag register", "flag", (78, 10), "0x20(%rsp)"),
        ("a spilled flag slot", "spill", (80, 10), "0x20(%rsp)"),
        ("a red-zone flag slot", "spill", (80, 10), "-0x10(%rsp)"),
    ):
        fl, fr = canned_tile(inner_latch=latch)
        fl = [line.replace("0x20(%rsp)", slot) for line in fl]
        st = {r["ty"]: r for r in tile_units(fl, fr, CANNED_QDOT, CANNED_OPS.replace("(200, 10)", f"{want}"))[0]}["Q4_K"]
        got = (st["fixed"], st["per_col"], st["status"])
        if got != (*want, "ok"):
            print(f"asm-census self-test: tile-units flag latch held in {what}: {got} != {(*want, 'ok')}")
            return 1
    # a row within 5 %, a stale row, a missing row, a row for a type with no tile kernel: the failure each names
    for what, ops, status, red in (
        ("within tolerance", CANNED_OPS.replace("(200, 10)", "(208, 10)"), {"Q4_K": "ok"}, 0),
        ("stale", CANNED_OPS.replace("(200, 10)", "(300, 10)"), {"Q4_K": "STALE"}, 1),
        ("missing", CANNED_OPS.replace("        GgmlType::Q8_0 => Some((4, 2)),\n", ""), {"Q8_0": "MISSING"}, 1),
        ("no tile kernel", CANNED_OPS.replace("        _ => None", "        GgmlType::Q6_K => Some((1, 1)),\n        _ => None"),
         {"Q6_K": "NO-TILE"}, 1),
    ):
        rows, n = tile_units(lst, rel, CANNED_QDOT, ops)
        seen = {r["ty"]: r["status"] for r in rows}
        if n != red or any(seen.get(t) != s for t, s in status.items()):
            print(f"asm-census self-test: tile-units {what}: {seen}, {n} red; want {status}, {red}")
            return 1
    # a wide type's row is fitted through its wide kernel's runs and holds up to WIDE_COLS: the canned kernel's
    # fixed 210, per_col 40 stand, the tile's (200, 10) is stale, as is a row that holds only to TILE_COLS
    wl, wr = canned_wide()
    wq = CANNED_QDOT + CANNED_WIDE
    for what, ops, status, red in (
        ("wide row", CANNED_OPS.replace("(200, 10)", "(210, 40)"), "ok", 0),
        ("wide row from the tile", CANNED_OPS, "STALE", 1),
        ("wide row held only to TILE_COLS", CANNED_OPS.replace("(200, 10)", "(222, 37)"), "STALE", 1),
    ):
        rows, n = tile_units(lst + wl, rel + wr, wq, ops)
        q = {r["ty"]: r for r in rows}["Q4_K"]
        got = (q["fixed"], q["per_col"], q.get("wide"), q.get("wide_fn"), q["status"], n)
        want = (210, 40, {16: 425, 32: 745}, "qdot::dot_q4k_q8k_wide_avx2", status, red)
        if got != want:
            print(f"asm-census self-test: tile-units {what}: {got} != {want}")
            return 1
    # the derivation reads GNU objdump's jump spelling of the wide kernel the same
    gnu = [re.sub(r"(\tj\w+\t)0x", r"\1", l) for l in lst + wl]
    if tile_units(gnu, rel + wr, wq, CANNED_OPS)[0] != tile_units(lst + wl, rel + wr, wq, CANNED_OPS)[0]:
        print("asm-census self-test: tile-units reads GNU objdump's wide kernel differently")
        return 1
    # a shape the derivation does not read is a named error, never a guess
    def edited(lines, was, is_):
        return [l.replace(was, is_) for l in lines]

    flag_lst = canned_tile(inner_latch="flag")[0]
    for what, args in (
        ("a wide type with no wide kernel in the listing", (lst, rel, wq, CANNED_OPS)),
        ("a wide kernel named for another type", (lst + canned_wide("qdot::dot_q8_0_q8k_wide_avx2")[0], rel + wr, wq,
                                                  CANNED_OPS)),
        ("wide runs that are not whole groups", (lst + wl, rel + wr, wq.replace("WIDE_GROUP: usize = 8", "WIDE_GROUP: usize = 12"),
                                                 CANNED_OPS)),
        ("a wide kernel whose jump table is gone", (lst + wl, rel, wq, CANNED_OPS)),
        ("a latch that is no counted loop", (canned_tile(trip_cmp="cmpq\t%rbx, %rax")[0], rel, CANNED_QDOT, CANNED_OPS)),
        ("a 32-value type with no tail loop", (canned_tile(tail=False)[0], rel, CANNED_QDOT, CANNED_OPS)),
        ("a flag latch that clears no flag", (edited(flag_lst, "xorl\t%ebx, %ebx", "nop"), rel, CANNED_QDOT, CANNED_OPS)),
        ("a flag latch whose induction register counts past its flag",
         (edited(flag_lst, "movl\t$0x1, %eax", "movl\t$0x2, %eax"), rel, CANNED_QDOT, CANNED_OPS)),
        ("a flag latch with no exit test", (edited(flag_lst, "testb\t$0x1, %ebx", "nop"), rel, CANNED_QDOT, CANNED_OPS)),
        ("a spilled flag that does not start at one",
         (edited(canned_tile(inner_latch="spill")[0], "movl\t$0x1, %ecx", "movl\t$0x2, %ecx"), rel, CANNED_QDOT,
          CANNED_OPS)),
        ("a jump table of another size", (lst, rel, CANNED_QDOT.replace("Qb,\n", "Qb,\n    Qc,\n"), CANNED_OPS)),
        ("a function with no jump table", (lst, [], CANNED_QDOT, CANNED_OPS)),
    ):
        try:
            tile_units(*args)
            print(f"asm-census self-test: tile-units read {what}")
            return 1
        except TileError:
            pass

    # every write the derivation reads goes through `writes`: a write it would miss, and an instruction or operand form
    # it does not list, is refused, and the refusal says why
    def flagged(latch, slot="0x20(%rsp)", reg="%ebx", edit=("", ""), gap=None, **kw):
        """canned_tile's flag latch with the flag in `reg` or spilled to `slot`, one instruction edited, `gap` between
        the flag test and its je."""
        fl, fr = canned_tile(inner_latch=latch, **kw)
        fl = [l.replace("0x20(%rsp)", slot).replace("%ebx", reg).replace(*edit) for l in fl]
        for k, l in enumerate(fl if gap else ()):
            if "\ttestb\t$0x1, " in l:
                fl[k - 1] = fl[k - 1].split("\t", 1)[0] + "\t" + l.split("\t", 1)[1]
                fl[k] = l.split("\t", 1)[0] + "\t" + gap
        return fl, fr, CANNED_QDOT, CANNED_OPS.replace("(200, 10)", "(80, 10)" if latch == "spill" else "(78, 10)")

    for what, args, why in (
        *((f"a counted loop whose head writes its counter with {ins}", (*canned_tile(inner=ins), CANNED_QDOT, CANNED_OPS),
           "is written in the loop besides the latch's add") for ins in (
               "incq\t%rax", "xchgq\t%rax, %rcx", "xaddq\t%rax, %rcx", "mulxq\t%rsi, %rax, %rcx", "cmpxchgq\t%rbx, %rcx",
               "lahf", "cltq")),
        ("a flag exchanged in the head", flagged("flag", inner="xchgb\t%bl, %cl"), "is written inside the loop"),
        *((f"a flag in {reg} that {byte} clears in the head", flagged("flag", reg=reg, inner=f"movb\t$0x0, {byte}"),
           "is written inside the loop") for reg, byte in (
               ("%edi", "%dil"), ("%esi", "%sil"), ("%ebp", "%bpl"), ("%ebx", "%bh"))),
        ("a flag the latch clears only in %bh", flagged("flag", edit=("xorl\t%ebx, %ebx", "xorb\t%bh, %bh")),
         "is not cleared in the latch"),
        ("a flag set only in %bh on entry", flagged("flag", edit=("movl\t$0x1, %ebx", "movb\t$0x1, %bh")),
         "is not shown to start at one"),
        ("a flag tested in %bh", flagged("flag", edit=("testb\t$0x1, %ebx", "testb\t$0x1, %bh")), "no block of the loop ends"),
        ("a spilled flag stored from %ch", flagged("spill", edit=("movl\t%ecx, 0x20(%rsp)", "movb\t%ch, 0x20(%rsp)")),
         "is not shown to start at one"),
        *((f"a spilled flag at {slot} overwritten in the head by {ins}", flagged("spill", slot, inner=ins),
           "is written inside the loop") for slot, ins in (
               ("0x20(%rsp)", "movq\t$0x0, 0x1c(%rsp)"), ("-0x10(%rsp)", "movq\t$0x0, -0x14(%rsp)"),
               ("-0x10(%rsp)", "movq\t$0x0, (%rsp,%rcx,4)"))),
        # the induction register callee-saved, so only the flag's slot is left for the call to overwrite
        *((f"a spilled flag at {slot} across a call in the head",
           flagged("spill", slot, edit=("%eax", "%r12d"), inner="callq\t0x0 <f>"), "is written inside the loop")
          for slot in ("0x20(%rsp)", "-0x8(%rsp)")),
        ("a spilled flag at -0x8(%rsp) across a call on entry",
         flagged("spill", "-0x8(%rsp)", edit=("%eax", "%r12d"), entry=("callq\t0x0 <f>",)), "is not shown to start at one"),
        *((f"a spilled flag at {slot} overwritten on entry by {'; '.join(ins)}", flagged("spill", slot, entry=ins),
           "is not shown to start at one") for slot, ins in (
               ("0x20(%rsp)", ("movq\t$0x0, 0x1c(%rsp)",)), ("0x20(%rsp)", ("xchgl\t%ecx, %edx", "movl\t%ecx, 0x20(%rsp)")))),
        *((f"a flag test whose je reads the flags {ins} writes", flagged("flag", gap=ins), "no block of the loop ends")
          for ins in ("bsfl\t%ecx, %edx", "vptest\t%ymm0, %ymm1")),
        ("an instruction the write classes do not list", (*canned_tile(inner="cpuid"), CANNED_QDOT, CANNED_OPS),
         "cpuid is not an instruction the write classes list"),
        ("an operand form the write classes do not read",
         (*canned_tile(inner="vaddps\t%ymm0, %ymm1, %ymm2 {%k1}"), CANNED_QDOT, CANNED_OPS), "is not a form the write classes read"),
        ("a flag test gap the write classes do not list", flagged("flag", gap="rdtsc"),
         "rdtsc is not an instruction the write classes list"),
    ):
        try:
            tile_units(*args)
            print(f"asm-census self-test: tile-units read {what}")
            return 1
        except TileError as e:
            if why not in str(e):
                print(f"asm-census self-test: tile-units refused {what} for another reason: {e}")
                return 1
    return 0


def self_test():
    base = """
0000000000001000 <k::avx>:
    1000:\tvpxor  %ymm0,%ymm0,%ymm0
    1004:\tvmovdqu (%rdi),%ymm1
    1008:\textrq  $0x10,$0x8,%xmm3
    100e:\tmovaps %xmm1,%xmm2
    1012:\tvpaddd 0x20(%rsp),%ymm1,%ymm1
    1018:\tadd    $0x20,%rdi
    101c:\tjne    1004 <k::avx+0x4>
    101e:\tvzeroupper
    1021:\tret
0000000000001100 <k::scalar>:
    1100:\tinsertq $0x8,$0x0,%xmm1,%xmm0
    1106:\tlea    -0x10(%rip),%rax        # 10f6 <k::avx+0xf6>
    110d:\tret
""".splitlines()
    new = [l.replace("extrq  $0x10,$0x8,%xmm3", "vpsrlq $0x10,%xmm3,%xmm3").replace(
        "-0x10(%rip)", "-0x30(%rip)").replace("10f6", "10d6") for l in base]
    fns = functions(base)
    got = sorted((f, k, m, c) for f, k, m, c in legacy(fns))
    want = [("k::avx", "legacy-in-avx", "movaps", 1), ("k::avx", "sse4a-in-avx", "extrq", 1), ("k::scalar", "sse4a", "insertq", 1)]
    if got != want:
        print(f"asm-census self-test: legacy {got} != {want}")
        return 1
    n, mem, stack, hist = loop(fns, "k::avx")
    if (n, mem, stack, hist["extrq"]) != (6, 2, 1, 1):
        print(f"asm-census self-test: loop {(n, mem, stack, hist['extrq'])} != (6, 2, 1, 1)")
        return 1
    moved, alone, same = diff(fns, functions(new))
    if moved != [("k::avx", 9, 9)] or alone or same != 1:
        print(f"asm-census self-test: diff {moved} {alone} {same}: want k::avx moved, k::scalar identical (rip normalized)")
        return 1
    try:
        loop(fns, "k::scalar")
        print("asm-census self-test: a function with no loop was read as one")
        return 1
    except KeyError:
        pass
    if tile_units_self_test():
        return 1
    print("asm-census self-test: ok (legacy and sse4a lists, the loop census, the normalized diff, a loopless function, "
          "the tile-units derivation and its refusals)")
    return 0


def main(argv):
    args = argv[1:]
    if args == ["--self-test"]:
        return self_test()
    try:
        if args[:1] == ["legacy"] and len(args) in (2, 3) and (len(args) == 2 or args[2] == "--fail"):
            rows = legacy(functions(listing(args[1])))
            for f, k, m, c in rows:
                print(f"{c}\t{k}\t{m}\t{f}")
            n = sum(c for _, _, _, c in rows)
            print(f"asm-census: {n} legacy-encoded or SSE4a instructions in {len({f for f, *_ in rows})} functions")
            return 1 if rows and len(args) == 3 else 0
        if args[:1] == ["loop"] and len(args) == 3:
            n, mem, stack, hist = loop(functions(listing(args[1])), args[2])
            top = " ".join(f"{m}:{c}" for m, c in hist.most_common())
            print(f"{args[2]}: loop {n} instructions, {mem} with a memory operand, {stack} on the stack\n{top}")
            return 0
        if args[:1] == ["diff"] and len(args) == 3:
            moved, alone, same = diff(functions(listing(args[1])), functions(listing(args[2])))
            for f, x, y in moved:
                print(f"differs\t{x} -> {y}\t{f}")
            for f, side in alone:
                print(f"only-{side}\t{f}")
            print(f"asm-census: {same} functions identical, {len(moved)} differ, {len(alone)} on one side only")
            return 0
        if args[:1] == ["tile-units"] and len(args) == 5:
            lines, rel, qdot, ops = (open(f, errors="replace").read() for f in args[1:])
            rows, red = tile_units(lines.splitlines(), rel.splitlines(), qdot, ops)
            print(tile_units_report(rows, red))
            return 1 if red else 0
    except (OSError, KeyError, TileError, subprocess.CalledProcessError) as e:
        print(f"asm-census: {e}")
        return 1
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
