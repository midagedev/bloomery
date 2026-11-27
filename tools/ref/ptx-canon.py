#!/usr/bin/env python3
"""Are two builds of a PTX entry the same program? The canonical diff of an entry's body.

`tools/ptx-scan.sh` prints, per entry, the md5 of its body with every register, label and local
depot number deleted and every generated module symbol replaced by its declaration's signature
(`ptx::normalize` in crates/gpu-gates/src/ptx.rs). That digest is stable across the backend's free
numbering, and blind to which register an operand names: a rewired operand or a retargeted branch
keeps it. This script renumbers instead of deleting: each register class, the block labels and the
local depots are numbered in order of first appearance inside the entry, so two bodies compare equal
only when they are the same instruction stream over the same dataflow, whatever absolute numbers the
backend chose. When a deletion moves a live entry's md5, this says whether the program moved with it.

A generated module symbol — a whole token starting with `__shared_mem_`, `__device_global_` or
`__dynamic_smem_`, whatever the backend put after the stem (a crate hash, a number, a kernel name) —
is written as `<stem><<signature>>#k`, the rule `ptx::normalize` applies: the signature is the
symbol's declaration in its module with the name taken out (linkage, state space, alignment, type,
dimensions, and `= #<FNV-1a 64 of the initializer>`), k its order of first appearance among the
entry's generated symbols. So a renumbering or reordering of the module's declarations, or a new
crate hash, compares identical, and a changed declaration of a symbol the entry names does not. A
generated symbol its module does not declare, one declared twice, or a declaration line this reader
does not parse is refused by name (exit 2). Every other symbol is kept by name.

  ptx-canon.py [-U N] BASE.ptx NEW.ptx ENTRY[:NEW_ENTRY]...
  ptx-canon.py [-U N] --all BASE.ptx NEW.ptx
  ptx-canon.py --scans [--list-equal] BASE.scan NEW.scan
  ptx-canon.py --self-test

For each entry it prints the unified diff of the two canonical bodies (N lines of context, default
3) and one verdict line:

  ptx-canon: <entry> identical
  ptx-canon: <entry> reordered-only (same line multiset); <whether the moved lines are independent>
  ptx-canon: <entry> value-equal (<what the proof matched>)
  ptx-canon: <entry> differs (<n> changed lines)

ENTRY:NEW_ENTRY compares a renamed entry. --all compares every entry the two files share and lists
the ones only one file has (`only in base`, `only in new`: a deleted or an added kernel), then a
count line. Exit 0 when every compared entry is identical, reordered-only with every swapped pair
independent and the value proof holding (below) or value-equal (below), 1 when one differs or its
reorder is not shown safe, 2 on a usage error or a named entry that a file does not hold. --all also
ends 1 with a named line when an entry is on one side only (a deleted or an added kernel is no proof
of an unchanged program) or when the two files share no entry (nothing was compared), and 2 when a
file holds no `.visible .entry` at all (an empty or unreadable module).

--scans compares two saved `tools/ptx-scan.sh` outputs (stdout, or stdout and stderr together): the
table rows by entry, column by column, and the md5 block by entry. Two md5 blocks compare only under
one digest method — the `method=` of their `ptx-scan-md5:` line; a block with no method comes from an
older rule set — and a pair whose methods differ is refused by name (exit 2): rescan the older binary
with this tree's tools. A failed scan is refused too. Exit 0 when both scans hold the same entries,
every row and every md5 equal, 1 otherwise, with the moved rows and digests listed. `--list-equal`
adds one `ptx-canon: md5-equal <entry>` line for each entry whose digest is equal on both sides.

A `--no-jit` scan (banner `jit=skipped(no-jit)`) has no driver JIT reading: its `jit_regs` and
`jit_local` cells read `skipped(no-jit)`. Those two columns are never compared when either scan is
no-jit — a skipped cell is not a value and equals nothing, not even another skipped cell — and one line
says so (`jit columns not compared (no-jit: base|new|both)`); every other column and the md5 block
compare as before. A scan whose banner and cells disagree (a no-jit banner over a number, a full banner
over a skipped cell) or that holds a skipped cell in any other column is refused by name (exit 2).

`reordered-only` is a same-multiset result, not by itself a proof. Its line therefore also checks
every pair of lines whose relative order changed: the pair is independent when the two lines share
no register and neither is a memory, synchronisation or control line (anything but the pure
arithmetic, move, convert, compare and select opcodes listed in PURE: a load, a store, an atomic,
a barrier, a shuffle, a branch, a call, a label, a directive, and every opcode not in PURE). That
screen reads an opcode's root and a register's name only, so it cannot see a carry flag (`addc`,
`.cc`), a read of `%clock` or of a register outside REG_CLASSES (`%SPL`), or a second instruction
on one line. A reorder therefore reads `reordered-only` with exit 0 only when every swapped pair
passes the screen AND the `value-equal` proof below holds over the same two bodies: the proof
classifies every line with the one rule (`classify`), the screen is a label. A reorder that passes
the screen and not the proof, or fails the screen, says `read the diff` and the exit code is 1: a
same-multiset reorder can still move a value. A reorder that changes the order in which one
register class first appears renumbers everything after it and reads `differs` unless the
`value-equal` proof below holds: the verdict errs toward `differs`, never toward `identical`.

`value-equal` is the verdict for what numbering by first appearance cannot absorb. It is not a
renaming: the same value may sit in another register, and pure lines may change places inside a
basic block. It proves that the two bodies perform the same loads, stores, atomics, barriers and
branches, in the same order, on equal operand values and under equal guards, and hand every
register to the next basic block holding the same value. The proof is checked, and anything it
cannot show reads `differs` (or the `reordered-only` line it already had), plus a `note:` line
naming why.
  - Both bodies are cut into basic blocks (a label, a directive or a control transfer, `bra`, `ret`,
    `exit` or `trap`, ends one) and the two cuts must be the same sequence.
  - Inside a block every register starts as a free symbol and the lines are run symbolically. A pure
    line (PURE, no carry flag, no read of a volatile special register) gets a value term over its
    operands, a register copy being the value it copies and a constant `mov` the constant's bits
    (`0f00000000` and `0` are one value). Every other line (a load, store, atomic, shuffle, barrier,
    branch, a carry or volatile line) is an anchor: kept in order, its text, guard and operand values
    must be equal, and it defines fresh values.
  - A pairing of the two sides' registers is searched, then checked from scratch: the new side re-run
    with its registers renamed through the pairing must give, in every block, the same anchors in the
    same order and the same value in every register the block changed.
  - The two bodies must also hold the same lines once register names are erased: no line added,
    dropped or rewritten, an immediate included.
What it does not prove: it knows no arithmetic. A changed opcode, modifier, width or immediate, a
line added or dropped, a swapped commutative operand and a folded constant all read `differs`. An
integer constant is a decimal (the bare `0` or a nonzero-led one), a negative one or a `0x` hex on an
integer or bit move, and a hex-float literal (`0f…`, `0d…`) one on a bit or float move; an octal literal
(`010` is 8 in PTX), an integer literal on a float move (the value or the bits: the ISA text does not
say) and a hex-float literal on a signed or unsigned move are no constant and compare by their text. A
line holding two instructions (`a; b;`) ends the proof. A line
whose opcode is in neither the PURE list nor the anchor lists ends the proof there (`differs`, never
a guess). It compares values, not timing: a pure line may move, so when a value is ready is not
compared. It says nothing about the registers, spills or SASS the backend makes of the bodies
(`tools/canon-check.sh` takes the verdict as a move only beside equal resource columns) and nothing
about the body of a device function the entry calls. A same-multiset reorder that `reordered-only`
cannot show independent (two lines that read one register) is tried here too.

The body is cut as `ptx::body` cuts it: from just past `.visible .entry <name>(` to the line of
the next `.entry` or `.func` directive. The body of a device function the entry calls lies outside
it, as it does for the scan's digest. Three things are therefore compared by no verdict here, and
`identical` says nothing about them: the body of a `.func` an entry calls, the initializer of a
module table that is not a generated symbol (kept by name), and which of two generated symbols with
one signature an entry names (each entry is a renaming of itself, the module need not be). A change
that touches one of them is not provable by this tool or by `tools/canon-check.sh`: run the owning gates.

Getting the module PTX of a gate binary (what ptx-scan reads), once for the base tree's binary and
once for the changed tree's, on the box:

  objcopy -O binary --only-section=.oxart target/release/<bin> <bin>.sec
  target/release/oxart_ptx <bin>.sec <dir>

The second writes <dir>/mod<N>.ptx per module and prints `mod<N> bundle=<name> bytes=<len>`.

The core kernels are the `bloomery-gpu` bundle (mod1 of gate_e2e, mod2 of generate_ds41).
`oxart_ptx` is host-only (`cargo build --release -p bloomery-gpu-gates --bin oxart_ptx`, what
`just ptx-scan` builds). The base binary is the one the base tree's own remote directory built.
"""
import difflib
import re
import sys
from collections import Counter

# The register classes the backend numbers (`ptx::normalize`'s REG_CLASSES).
REG_CLASSES = {"r", "rd", "rs", "f", "fd", "p", "h", "hh", "rq"}
# Entry-local generated symbols whose number means nothing (`ptx::normalize`'s NUMBERED_STEMS).
NUMBERED_STEMS = ("__local_depot",)
# Generated module symbols, replaced by their declaration's signature (`ptx::GENERATED_STEMS`).
GENERATED_STEMS = ("__shared_mem_", "__device_global_", "__dynamic_smem_")
# What a module-scope declaration opens with (`ptx.rs`'s DECL_OPENERS).
DECL_OPENERS = (".visible", ".extern", ".weak", ".common", ".shared", ".global", ".const")
STATE_SPACES = {".shared", ".global", ".const"}
# Opcodes whose only effect is their destination registers: a line of one of these may swap with
# another such line that shares no register with it.
PURE = {
    "abs", "add", "addc", "and", "bfe", "bfi", "bfind", "brev", "clz", "cnot", "copysign", "cos",
    "cvt", "cvta", "div", "dp2a", "dp4a", "ex2", "fma", "fns", "isspacep", "lg2", "lop3", "mad",
    "mad24", "madc", "max", "min", "mov", "mul", "mul24", "neg", "not", "or", "popc", "prmt", "rcp",
    "rem", "rsqrt", "sad", "selp", "set", "setp", "shf", "shl", "shr", "sin", "slct", "sqrt", "sub",
    "subc", "tanh", "testp", "xor",
}
DIRECTIVE = re.compile(r"(^|[ \t])\.(entry|func)([ \t(]|$)")
IDENT = re.compile(r"[A-Za-z0-9_$]+")
LABEL = re.compile(r"\$L__BB\d+(_\d+)*$")
CANON_REG = re.compile(r"%[a-z]+#\d+")
WINDOW_CAP = 2000


class Refused(Exception):
    """A usage error or a missing entry: exit 2 with the message."""


def body(text, name):
    """The entry's body, cut as `ptx::body` cuts it, or None when the text holds no such entry."""
    head = f".visible .entry {name}("
    at = text.find(head)
    if at < 0:
        return None
    lines = text[at + len(head):].split("\n")
    for i, line in enumerate(lines):
        if DIRECTIVE.search(line):
            return lines[:i]
    return lines


def fnv1a64(data):
    """FNV-1a, 64 bits, of a str's UTF-8 bytes: the initializer digest (`ptx.rs`'s fnv1a64)."""
    h = 0xcbf29ce484222325
    for b in data.encode():
        h = ((h ^ b) * 0x100000001b3) & 0xFFFFFFFFFFFFFFFF
    return h


def declaration(raw):
    """(name, signature) of the generated symbol a line declares, None for any other line. Refused
    for a line that opens like a declaration and names a generated symbol but is not
    `<directives> <name>[dims] [= init];` on one line (`ptx.rs`'s `declaration`)."""
    code = raw.split("//", 1)[0].strip()
    if not any(code.startswith(o) and code[len(o):len(o) + 1] in (" ", "\t") for o in DECL_OPENERS):
        return None
    m = next((m for m in IDENT.finditer(code) if m.group(0).startswith(GENERATED_STEMS)), None)
    if m is None:
        return None
    name = m.group(0)
    head, tail = code[:m.start()], code[m.end():]
    words = head.split()
    if not all(w.startswith(".") or w.isdigit() for w in words) or not STATE_SPACES & set(words):
        raise Refused(f"declaration of {name} does not open with directives naming a state space: "
                      f"{code}")
    if not tail.endswith(";"):
        raise Refused(f"declaration of {name} does not end on its line with `;`: {code}")
    dims, eq, init = tail[:-1].partition("=")
    dims = "".join(dims.split())
    init = " ".join(init.split())
    if not re.fullmatch(r"(\[\d*\])*", dims) or (eq and not init):
        raise Refused(f"declaration of {name} has dimensions or an initializer this reader does not "
                      f"parse: {code}")
    sig = " ".join(words) + (f" {dims}" if dims else "") + (f" = #{fnv1a64(init):016x}" if eq else "")
    return name, sig


def declarations(text, path):
    """{name: signature} of every generated symbol the module text declares; refused when one is
    declared twice."""
    out = {}
    for line in text.split("\n"):
        d = declaration(line)
        if d is None:
            continue
        if d[0] in out:
            raise Refused(f"{path} declares {d[0]} twice")
        out[d[0]] = d[1]
    return out


def entries(text):
    """Every `.visible .entry` name in the text, in declaration order."""
    return re.findall(r"\.visible \.entry ([A-Za-z0-9_$]+)\(", text)


def tokenize(code, name, decls):
    """One comment-free, whitespace-collapsed line cut into literal strings and entities: ("reg", cls,
    tok), ("label", tok), ("depot", stem, tok), ("gen", stem, tok, signature). The entities are the names
    the backend numbers freely; everything else is kept as written. `canon` and the value proof both read
    a line through this one cut."""
    out = []
    i = 0
    while i < len(code):
        c = code[i]
        if c == "%":
            m = re.match(r"%([a-z]*)", code[i:])
            cls = m.group(1)
            j = i + m.end()
            digits = re.match(r"\d*", code[j:]).group(0)
            after = code[j + len(digits)] if j + len(digits) < len(code) else ""
            if cls in REG_CLASSES and digits and not (after.isalnum() or after in "_$"):
                out.append(("reg", cls, cls + digits))
                i = j + len(digits)
                continue
            decl = re.match(r"<\d+>", code[j:]) if cls in REG_CLASSES else None
            if decl:
                out.append(f"%{cls}<N>")
                i = j + decl.end()
                continue
            out.append(f"%{cls}")
            i = j
            continue
        m = IDENT.match(code, i)
        if m and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] in "_$")):
            tok = m.group(0)
            i = m.end()
            if LABEL.match(tok):
                out.append(("label", tok))
                continue
            gen = next((s for s in GENERATED_STEMS if tok.startswith(s)), None)
            if gen:
                if tok not in decls:
                    raise Refused(f"entry {name} names {tok}, which its module does not declare")
                out.append(("gen", gen, tok, decls[tok]))
                continue
            stem = next((s for s in NUMBERED_STEMS
                         if tok.startswith(s) and tok[len(s):].isdigit()), None)
            if stem:
                out.append(("depot", stem, tok))
            elif tok == name:
                out.append("ENTRY")
            elif tok.startswith(name + "_param_"):
                out.append("ENTRY" + tok[len(name):])
            else:
                out.append(tok)
            continue
        out.append(c)
        i += 1
    return out


def clean(raw):
    """A body line with its comment cut and its whitespace runs collapsed; empty for a blank line."""
    return re.sub(r"[ \t\r]+", " ", raw.split("//", 1)[0]).strip()


def numbering():
    """A first-appearance numberer: `number(kind, token)` is 1, 2, ... per kind in the order tokens of
    that kind are first asked for."""
    seen = {}
    count = {}

    def number(kind, tok):
        key = (kind, tok)
        if key not in seen:
            count[kind] = count.get(kind, 0) + 1
            seen[key] = count[kind]
        return seen[key]
    return number


def body_tokens(lines, name, decls):
    """The body's non-empty lines, each cut into tokens: a str for text kept as written, a
    ("reg", class, name) for a register as the backend named it. Block labels, local depots and
    generated module symbols are already their canonical text (numbered by first appearance across
    the body, generated symbols as their declaration's signature), so a line differs between two
    bodies only in its register names."""
    number = numbering()
    out = []
    for raw in lines:
        code = clean(raw)
        if not code:
            continue
        res = []
        for t in tokenize(code, name, decls):
            if isinstance(t, str) or t[0] == "reg":
                res.append(t)
            elif t[0] == "label":
                res.append(f"$L#{number('$L', t[1])}")
            elif t[0] == "gen":
                res.append(f"{t[1]}<{t[3]}>#{number('gen', t[2])}")
            else:
                res.append(f"{t[1]}#{number(t[1], t[2])}")
        out.append(res)
    return out


def canon(lines, name, decls):
    """The body's lines with comments cut, whitespace runs collapsed, empty lines dropped, every
    numbered register, block label and local depot renumbered by first appearance, and every
    generated module symbol written as its declaration's signature (`decls`, the module's
    `declarations`) with its order of first appearance."""
    number = numbering()
    return ["".join(t if isinstance(t, str) else f"%{t[1]}#{number(t[1], t[2])}" for t in toks)
            for toks in body_tokens(lines, name, decls)]


def opcode(line):
    """The mnemonic's root of an instruction line (`mov` of `@%p#1 mov.b32 …`), or None for a
    label, a directive or a brace."""
    s = line
    if s.startswith("@"):
        s = s.split(" ", 1)[1] if " " in s else ""
    if not s or s.endswith(":") or s[0] in ".{}":
        return None
    return re.match(r"[a-z0-9_]*", s).group(0)


def independent(x, y):
    """Two instruction lines that may swap: both pure, no register in common."""
    if opcode(x) not in PURE or opcode(y) not in PURE:
        return False
    return not (set(CANON_REG.findall(x)) & set(CANON_REG.findall(y)))


def moved_pairs(a, b):
    """The pairs of lines whose relative order differs between a and b (same multiset). The k-th
    copy of a text in a is matched with its k-th copy in b; two equal lines never need a swap."""
    lo = 0
    while a[lo] == b[lo]:
        lo += 1
    hi = len(a) - 1
    while a[hi] == b[hi]:
        hi -= 1
    wa, wb = a[lo:hi + 1], b[lo:hi + 1]
    if len(wa) > WINDOW_CAP:
        return None, len(wa)
    occ = {}
    ida = []
    for t in wa:
        occ[t] = occ.get(t, 0) + 1
        ida.append((t, occ[t]))
    occ = {}
    posb = {}
    for i, t in enumerate(wb):
        occ[t] = occ.get(t, 0) + 1
        posb[(t, occ[t])] = i
    pairs = []
    for i in range(len(ida)):
        for j in range(i + 1, len(ida)):
            if ida[i][0] != ida[j][0] and posb[ida[i]] > posb[ida[j]]:
                pairs.append((ida[i][0], ida[j][0]))
    return pairs, len(wa)


# ---------------------------------------------------------------------------------------------------
# value-equal: the same values at every memory and control line, shown by a proof that is checked
# ---------------------------------------------------------------------------------------------------


class Unproved(Exception):
    """The proof does not hold for a pair, or reads a line it does not model: the message says which,
    and the verdict stays what the line diff gave."""


# Special registers a thread reads the same value from for its whole life: a `mov` from one is pure.
# Any other `%name` operand (`%clock`, `%globaltimer`, `%warpid`, `%smid`, a bare `%`) makes its line
# an anchor; `%warpid` and `%nwarpid` are volatile (the hardware may move a warp), so a read of one
# keeps its place against the barriers and memory lines.
FIXED_SPECIAL = {"%tid", "%ntid", "%ctaid", "%nctaid", "%laneid", "%lanemask"}
# The instruction roots that end a basic block: what the registers hold at one is what its target sees.
CONTROL = {"bra", "brx", "ret", "exit", "trap"}
# The anchors: lines whose order against each other is kept and whose operand values must be equal.
# These define no register ...
ANCHOR_NODEF = CONTROL | {"bar", "barrier", "membar", "fence", "st", "red", "prefetch", "prefetchu",
                          "cp", "stmatrix"}
# ... and these define the registers of their first operand (every other register they name is read).
ANCHOR_DEF0 = {"ld", "ldu", "atom", "shfl", "vote", "match", "redux", "ldmatrix", "mma", "activemask",
               "elect"}
# Pure by root, but they carry the hidden carry flag, so they are anchors too.
CARRY_ROOTS = {"addc", "subc", "madc"}
# A PTX decimal literal starts with a nonzero digit: `010` is octal 8, not 10. Only the bare `0`, a nonzero-led decimal
# and a `0x` hex are read as constants; an octal or binary literal (and any suffixed one) is no constant here.
IMMEDIATE = re.compile(r"-?(0[xX][0-9a-fA-F]+|0|[1-9][0-9]*)")
MOV_TYPE = re.compile(r"[busf](8|16|32|64)")


def cls_of(tok):
    """The register class of a register name (`rd` of `rd12`)."""
    return re.match(r"[a-z]+", tok).group(0)


class Terms:
    """Hash-consed value terms, shared by both sides so that an equal id is an equal term. A node is
    (tag, scalars, children): "in" (a register's value at block entry), "res" (what an anchor
    defines), "const", "lit" (an operand written out), "opnd" (an operand with registers in it),
    "op" (a pure line's value), "unpack", "psel" (a guarded line's new or old value)."""

    def __init__(self):
        self.ids = {}
        self.nodes = []
        self.shapes = []
        self.free = []

    def mk(self, tag, scal=(), kids=()):
        key = (tag, scal, kids)
        i = self.ids.get(key)
        if i is None:
            i = self.ids[key] = len(self.nodes)
            self.nodes.append(key)
            self.shapes.append(hash(((tag, scal[:1] if tag == "in" else scal),
                                     tuple(self.shapes[k] for k in kids))))
            self.free.append(tag == "in" or any(self.free[k] for k in kids))
        return i


class Binder:
    """A partial bijection between the base's register names and the new side's, with an undo trail."""

    def __init__(self):
        self.f, self.g, self.trail = {}, {}, []

    def bind(self, x, y):
        if self.f.get(x) == y:
            return True
        if x in self.f or y in self.g or cls_of(x) != cls_of(y):
            return False
        self.f[x], self.g[y] = y, x
        self.trail.append((x, y))
        return True

    def undo(self, mark):
        while len(self.trail) > mark:
            x, y = self.trail.pop()
            del self.f[x]
            del self.g[y]


def unify(terms, bd, a, b):
    """Whether terms a (base) and b (new) are one term once the register names at their leaves are
    paired; binds the pairs it needs in `bd`. A failed call leaves its partial bindings: callers undo."""
    nodes, free = terms.nodes, terms.free
    stack, seen = [(a, b)], set()
    while stack:
        x, y = stack.pop()
        if (x, y) in seen or (x == y and not free[x]):
            continue
        seen.add((x, y))
        tx, sx, kx = nodes[x]
        ty, sy, ky = nodes[y]
        if tx != ty or len(kx) != len(ky):
            return False
        if tx == "in":
            if not bd.bind(sx[1], sy[1]):
                return False
        elif sx != sy:
            return False
        stack.extend(zip(kx, ky))
    return True


def trial(terms, bd, pairs):
    """Whether the (term, term) pairs unify together, leaving no binding behind."""
    mark = len(bd.trail)
    ok = all(unify(terms, bd, a, b) for a, b in pairs)
    bd.undo(mark)
    return ok


def line_parts(toks):
    """(guard, mnemonic, operands) of an instruction line: guard is (negated, register) or None, the
    operands the top-level comma-separated token lists with their spaces dropped."""
    i, guard = 0, None
    if toks[0] == "@":
        neg = len(toks) > 1 and toks[1] == "!"
        j = 2 if neg else 1
        if j >= len(toks) or isinstance(toks[j], str):
            raise Unproved(f"a guard this proof does not read: {line_text(toks)}")
        guard, i = (neg, toks[j]), j + 1
        while i < len(toks) and toks[i] == " ":
            i += 1
    start = i
    while i < len(toks) and toks[i] not in (" ", ";"):
        if not isinstance(toks[i], str):
            raise Unproved(f"a register inside a mnemonic: {line_text(toks)}")
        i += 1
    mnem = "".join(toks[start:i])
    rest = toks[i:]
    if rest and rest[-1] == ";":
        rest = rest[:-1]
    if ";" in rest:
        # a second instruction on the line would be read as operands of the first and its effect left out
        raise Unproved(f"two instructions on one line: {line_text(toks)}")
    ops, cur, depth = [], [], 0
    for t in rest:
        if t == " ":
            continue
        if t in ("[", "{", "("):
            depth += 1
        elif t in ("]", "}", ")"):
            depth -= 1
        if t == "," and depth == 0:
            ops.append(cur)
            cur = []
        else:
            cur.append(t)
    if cur or ops:
        ops.append(cur)
    return guard, mnem, ops


def line_text(toks):
    """A line with every register written as its class (`%r`): its shape."""
    return "".join(t if isinstance(t, str) else "%" + t[1] for t in toks)


def classify(toks, mnem):
    """How the proof reads an instruction line: "pure" (its value is a function of its operands),
    "def0" (an anchor defining its first operand's registers), "nodef" (an anchor defining none).
    A line whose root the proof does not model raises Unproved."""
    parts = mnem.split(".")
    root = parts[0]
    if root in PURE:
        special = any(isinstance(t, str) and t.startswith("%") and t not in FIXED_SPECIAL
                      for t in toks)
        return "def0" if root in CARRY_ROOTS or "cc" in parts or special else "pure"
    if root in ("bar", "barrier") and "red" in parts:
        return "def0"
    if root in ANCHOR_NODEF:
        return "nodef"
    if root in ANCHOR_DEF0:
        return "def0"
    raise Unproved(f"the opcode {root} is not modelled: {line_text(toks)}")


def is_boundary(toks):
    """A label, a directive, a brace or the entry header's parenthesis: a line that is not an
    instruction and is compared as written."""
    s = line_text(toks)
    return s.endswith(":") or s[0] in ".{}()"


def layout(tok_lines):
    """(boundaries, segments): the body cut into runs of instructions. A boundary line (compared as
    written) or a control-transfer instruction (`None`, the instruction ending its run) separates
    segments, so a pure line never moves across either."""
    bounds, segs, cur = [], [], []
    for toks in tok_lines:
        if is_boundary(toks):
            if any(not isinstance(t, str) for t in toks):
                raise Unproved(f"a register on a directive line: {line_text(toks)}")
            segs.append(cur)
            cur = []
            bounds.append(line_text(toks))
            continue
        cur.append(toks)
        if line_parts(toks)[1].split(".")[0] in CONTROL:
            segs.append(cur)
            cur = []
            bounds.append(None)
    segs.append(cur)
    return bounds, segs


def run_segment(terms, si, lines, rename):
    """Run one segment symbolically, registers read through `rename` (new-side name -> base name).
    Returns (anchors, exit): the anchors in order as (shape, operand-value ids, guard id or None) and
    {register: value id} for every register the segment leaves holding other than its entry value."""
    state = {}

    def val(r):
        tok = rename.get(r[2], r[2])
        v = state.get(tok)
        return terms.mk("in", (r[1], tok)) if v is None else v

    def operand(o):
        if len(o) == 1 and not isinstance(o[0], str):
            return val(o[0])
        items = tuple(t if isinstance(t, str) else None for t in o)
        kids = tuple(val(t) for t in o if not isinstance(t, str))
        return terms.mk("opnd", items, kids) if kids else terms.mk("lit", ("".join(items),))

    anchors = []
    for toks in lines:
        guard, mnem, ops = line_parts(toks)
        kind = classify(toks, mnem)
        gterm = val(guard[1]) if guard else None
        if kind == "pure":
            dst = ops[0] if ops else []
            regs = [t for t in dst if not isinstance(t, str)]
            root = mnem.split(".")[0]
            if root == "mov" and dst[:1] == ["{"] and len(ops) == 2:
                elems = dst[1:-1:2]
                if (len(ops[1]) != 1 or isinstance(ops[1][0], str) or dst[-1] != "}"
                        or len(dst) != 2 * len(elems) + 1 or dst[2:-1:2] != [","] * (len(elems) - 1)
                        or any(isinstance(e, str) and e != "_" for e in elems)):
                    raise Unproved(f"a mov into a register vector this proof does not read: "
                                   f"{line_text(toks)}")
                src = val(ops[1][0])
                regs = [e for e in elems if not isinstance(e, str)]
                new = [terms.mk("unpack", (mnem, j), (src,)) for j, e in enumerate(elems)
                       if not isinstance(e, str)]
            elif not regs or any(isinstance(t, str) and t != "|" for t in dst):
                raise Unproved(f"a destination this proof does not read: {line_text(toks)}")
            else:
                new = None
                if root == "mov" and len(regs) == 1 and len(ops) == 2:
                    src = ops[1]
                    if len(src) == 1 and not isinstance(src[0], str) and src[0][1] == regs[0][1]:
                        new = [val(src[0])]
                    else:
                        c = const_bits(mnem, regs[0][1], src)
                        if c is not None:
                            new = [terms.mk("const", c)]
                if new is None:
                    args = tuple(operand(o) for o in ops[1:])
                    new = [terms.mk("op", (mnem, j), args) for j in range(len(regs))]
            if guard:
                new = [terms.mk("psel", (guard[0],), (gterm, v, val(r))) for r, v in zip(regs, new)]
            for r, v in zip(regs, new):
                state[rename.get(r[2], r[2])] = v
            continue
        k = len(anchors)
        defs, uses = [], []
        for oi, o in enumerate(ops):
            for t in o:
                if isinstance(t, str):
                    continue
                if oi == 0 and kind == "def0":
                    defs.append(t)
                else:
                    uses.append(val(t))
        anchors.append((line_text(toks), tuple(uses), gterm))
        new = [terms.mk("res", (si, k, j, r[1])) for j, r in enumerate(defs)]
        if guard:
            new = [terms.mk("psel", (guard[0],), (gterm, v, val(r))) for r, v in zip(defs, new)]
        for r, v in zip(defs, new):
            state[rename.get(r[2], r[2])] = v
    exit_state = {tok: v for tok, v in state.items()
                  if v != terms.mk("in", (cls_of(tok), tok))}
    return anchors, exit_state


def const_bits(mnem, cls, o):
    """(class, width, bits) of a `mov` immediate, so `0f00000000` and `0` are one value; None when
    the immediate or the move's width is not one this reads."""
    if not all(isinstance(t, str) for t in o):
        return None
    text = "".join(o)
    m = MOV_TYPE.fullmatch(mnem.rsplit(".", 1)[-1])
    if m is None:
        return None
    width, kind = int(m.group(1)), m.group(0)[0]
    # A hex-float literal is the bits of a float: read on a bit or float move only, not on a signed or unsigned one.
    if re.fullmatch(r"0[fF][0-9a-fA-F]{8}", text) and width == 32 and kind in "bf":
        return (cls, 32, int(text[2:], 16))
    if re.fullmatch(r"0[dD][0-9a-fA-F]{16}", text) and width == 64 and kind in "bf":
        return (cls, 64, int(text[2:], 16))
    if IMMEDIATE.fullmatch(text) and kind != "f":
        # An integer literal on a float move may be the float's value or its bits (ptxas decides, the ISA text does not):
        # it is no constant, and stays a literal operand compared by its text.
        neg = text.startswith("-")
        digits = text.lstrip("-")
        v = int(digits, 16) if digits[:2] in ("0x", "0X") else int(digits, 10)
        return (cls, width, (-v if neg else v) % (1 << width))
    return None


def registers(tok_lines):
    """Every register name of the lines, in order of first appearance."""
    seen = {}
    for toks in tok_lines:
        for t in toks:
            if not isinstance(t, str):
                seen.setdefault(t[2], None)
    return list(seen)


def find_renaming(terms, ra, rb, regs_a, regs_b):
    """A bijection of the base's register names onto the new side's, searched, not trusted: the
    anchors' operand values force part of it, a block's exit registers pair by value, and what stays
    tied (registers of equal value) is paired in order. Returns (binder, ties chosen). Raises Unproved
    when no pairing exists; `certify` then checks what this found."""
    bd = Binder()
    for si, ((aa, _), (ab, _)) in enumerate(zip(ra, rb)):
        if len(aa) != len(ab):
            raise Unproved(f"block {si} holds {len(aa)} anchors in the base and {len(ab)} in the new")
        for k, (x, y) in enumerate(zip(aa, ab)):
            if x[0] != y[0]:
                raise Unproved(f"block {si} anchor {k} differs: `{x[0]}` against `{y[0]}`")
            pairs = list(zip(x[1], y[1])) + ([(x[2], y[2])] if x[2] is not None else [])
            mark = len(bd.trail)
            if not all(unify(terms, bd, u, v) for u, v in pairs):
                bd.undo(mark)
                raise Unproved(f"block {si} anchor {k} (`{x[0]}`) reads a value the new side's "
                               "does not hold")
    todo = [(si, dict(ea), dict(eb)) for si, ((_, ea), (_, eb)) in enumerate(zip(ra, rb))]
    ties = 0
    while True:
        progress = False
        for si, ea, eb in todo:
            for r in list(ea):
                if r not in bd.f:
                    continue
                r2 = bd.f[r]
                if r2 not in eb or not unify(terms, bd, ea[r], eb[r2]):
                    raise Unproved(f"block {si}: {r} and its partner {r2} do not end on one value")
                del ea[r], eb[r2]
                progress = True
            for r2 in eb:
                if r2 in bd.g and bd.g[r2] not in ea:
                    raise Unproved(f"block {si}: {r2} ends changed, its partner {bd.g[r2]} does not")
            groups = {}
            for r2, t in eb.items():
                groups.setdefault((terms.shapes[t], cls_of(r2)), []).append(r2)
            for r in list(ea):
                if r in bd.f:
                    continue  # paired by a commit earlier in this pass; the next round checks it
                cands = [r2 for r2 in groups.get((terms.shapes[ea[r]], cls_of(r)), ())
                         if r2 in eb and r2 not in bd.g
                         and trial(terms, bd, [(terms.mk("in", (cls_of(r), r)),
                                                terms.mk("in", (cls_of(r2), r2))),
                                               (ea[r], eb[r2])])]
                if not cands:
                    raise Unproved(f"block {si}: {r} ends on a value no register of the new side "
                                   "holds")
                if len(cands) == 1:
                    commit(terms, bd, r, cands[0], ea, eb)
                    progress = True
        if progress:
            continue
        stuck = next(((si, ea, eb) for si, ea, eb in todo if ea), None)
        if stuck is None:
            break
        si, ea, eb = stuck
        r = next(iter(ea))
        r2 = next((r2 for r2 in eb if r2 not in bd.g and cls_of(r2) == cls_of(r)
                   and trial(terms, bd, [(terms.mk("in", (cls_of(r), r)),
                                          terms.mk("in", (cls_of(r2), r2))), (ea[r], eb[r2])])),
                  None)
        if r2 is None:
            raise Unproved(f"block {si}: {r} ends on a value no register of the new side holds")
        commit(terms, bd, r, r2, ea, eb)
        ties += 1
    for si, ea, eb in todo:
        if eb:
            raise Unproved(f"block {si}: {next(iter(eb))} ends changed, no base register does")
    for cls in sorted({cls_of(r) for r in regs_a} | {cls_of(r) for r in regs_b}):
        left_a = [r for r in regs_a if cls_of(r) == cls and r not in bd.f]
        left_b = [r for r in regs_b if cls_of(r) == cls and r not in bd.g]
        if len(left_a) != len(left_b):
            raise Unproved(f"register class {cls}: {len(left_a)} registers of the base and "
                           f"{len(left_b)} of the new side are left unpaired")
        for r, r2 in zip(left_a, left_b):
            bd.bind(r, r2)
    return bd, ties


def commit(terms, bd, r, r2, ea, eb):
    """Pair base register r with new-side register r2 and the leaf names their values share."""
    bd.bind(r, r2)
    unify(terms, bd, ea[r], eb[r2])
    del ea[r], eb[r2]


def prove_value_equal(ba, bb, base_name, new_name, base_decls, new_decls):
    """Shows that two bodies are one program up to the backend's free choices: a bijection of
    register names, the order of pure lines inside a basic block, and which register holds a copy of
    a value. Returns a stats dict; raises Unproved when it cannot show it.

    The two bodies are cut into basic blocks (a label, a directive, or a control-transfer
    instruction ends one). Inside a block every register starts as a free symbol and a symbolic run
    gives each pure line (arithmetic, move, convert, compare, select) a value term over those symbols,
    a register copy being the value it copies; every other line is an anchor, kept in order, whose
    operand values are compared and which defines fresh values (`ld`, `atom`, `mma`, ... define what
    their first operand names; a carry-flag line, a read of a volatile special register and every
    opcode not listed in ANCHOR_NODEF/ANCHOR_DEF0 is an anchor or refused). A bijection `sigma` is
    searched; the certificate is then checked from scratch: the new side re-run with its registers
    renamed through sigma must give, in every block, the same anchors (same text, same operand
    values, same guard) and the same value in every register at the block's end. The same anchors
    in the same order with the same values, and the same state at every block boundary, mean the two
    bodies do the same loads, stores, barriers and branches on the same values and hand every
    register to the next block unchanged. The two bodies must also hold the same lines as a
    multiset once register names are erased, so no line was added, dropped or rewritten."""
    ta, tb = body_tokens(ba, base_name, base_decls), body_tokens(bb, new_name, new_decls)
    count_a, count_b = Counter(line_text(t) for t in ta), Counter(line_text(t) for t in tb)
    if count_a != count_b:
        a_only, b_only = next(iter(count_a - count_b), None), next(iter(count_b - count_a), None)
        raise Unproved("the two bodies hold different lines once register names are erased "
                       f"(a line more in the base: {a_only!r}; more in the new: {b_only!r}; "
                       f"counts {sum(count_a.values())} and {sum(count_b.values())})")
    (bounds_a, segs_a), (bounds_b, segs_b) = layout(ta), layout(tb)
    if bounds_a != bounds_b:
        raise Unproved("the block boundaries (labels, directives, control transfers) differ")
    terms = Terms()
    ra = [run_segment(terms, i, s, {}) for i, s in enumerate(segs_a)]
    rb = [run_segment(terms, i, s, {}) for i, s in enumerate(segs_b)]
    regs_a, regs_b = registers(ta), registers(tb)
    bd, ties = find_renaming(terms, ra, rb, regs_a, regs_b)
    rc = [run_segment(terms, i, s, bd.g) for i, s in enumerate(segs_b)]
    for si, ((aa, ea), (ac, ec)) in enumerate(zip(ra, rc)):
        if aa != ac:
            k = next(k for k, (x, y) in enumerate(zip(aa, ac)) if x != y)
            raise Unproved(f"certificate: block {si} anchor {k} (`{aa[k][0]}`) differs under the "
                           "pairing found")
        if ea != ec:
            r = next((r for r in sorted(set(ea) | set(ec)) if ea.get(r) != ec.get(r)))
            raise Unproved(f"certificate: block {si} ends with {r} holding another value under the "
                           "pairing found")
    return {"lines": sum(len(s) for s in segs_a), "blocks": len(segs_a),
            "anchors": sum(len(a) for a, _ in ra), "exits": sum(len(e) for _, e in ra),
            "regs": len(regs_a), "renamed": sum(1 for r, r2 in bd.f.items() if r != r2),
            "ties": ties}


def compare(base, new, base_name, new_name, context):
    """Print one entry's diff and verdict line; return the verdict word and whether it fails the
    run (`differs`, or a reorder not shown safe: dependent pairs, or a screen the value proof does not back).
    `base` and `new` are (path, text, decls)."""
    (base_path, base_text, base_decls), (new_path, new_text, new_decls) = base, new
    ba, bb = body(base_text, base_name), body(new_text, new_name)
    if ba is None:
        raise Refused(f"{base_path} holds no entry {base_name}")
    if bb is None:
        raise Refused(f"{new_path} holds no entry {new_name}")
    a, b = canon(ba, base_name, base_decls), canon(bb, new_name, new_decls)
    label = base_name if base_name == new_name else f"{base_name}:{new_name}"
    if a == b:
        print(f"ptx-canon: {label} identical ({len(a)} lines)")
        return "identical", False
    diff = list(difflib.unified_diff(a, b, f"{base_path}:{base_name}", f"{new_path}:{new_name}",
                                     n=context, lineterm=""))
    for line in diff:
        print(line)
    reorder = None  # a same-multiset reorder's note when it is not shown safe
    screened = None  # the note of a same-multiset reorder whose every swapped pair passes the opcode screen
    if sorted(a) == sorted(b):
        pairs, width = moved_pairs(a, b)
        if pairs is None:
            reorder = f"a window of {width} lines, too wide to check pairs; read the diff"
        else:
            dependent = [p for p in pairs if not independent(*p)]
            if dependent:
                reorder = (f"{len(dependent)} of {len(pairs)} swapped pair(s) share a register or hold a "
                           "memory, synchronisation or control line; read the diff (first: "
                           f"{dependent[0][0]!r} / {dependent[0][1]!r})")
            else:
                screened = f"{len(pairs)} swapped pair(s) in a {width}-line window"
    # The opcode screen reads an opcode's root and a register's name: it cannot see a carry flag, a clock, a register
    # outside REG_CLASSES or a second instruction on a line. So a reorder that passes it is `reordered-only` with exit 0
    # only when the value proof also holds over the same two bodies.
    try:
        st = prove_value_equal(ba, bb, base_name, new_name, base_decls, new_decls)
    except Unproved as e:
        unproved = str(e)
        if screened is not None:
            reorder = f"{screened} pass the opcode screen, but the value proof does not hold; read the diff"
    else:
        if screened is not None:
            print(f"ptx-canon: {label} reordered-only (same line multiset); {screened}, every one independent "
                  "(pure opcodes, no shared register)")
            return "reordered-only", False
        print(f"ptx-canon: {label} value-equal ({st['lines']} lines in {st['blocks']} basic "
              f"blocks; {st['anchors']} anchors in the same order on equal operand values; "
              f"{st['exits']} registers leave their blocks equal under a bijection of {st['regs']} "
              f"registers, {st['ties']} paired among registers of one value)")
        return "value-equal", False
    if reorder is not None:
        print(f"ptx-canon: {label} reordered-only (same line multiset); {reorder}")
        verdict = "reordered-only"
    else:
        changed = sum(1 for d in diff if d[:1] in "+-" and d[:3] not in ("+++", "---"))
        print(f"ptx-canon: {label} differs ({changed} changed lines)")
        verdict = "differs"
    print(f"ptx-canon: {label} note: value-equal not shown: {unproved}")
    return verdict, True


def read(path):
    try:
        with open(path, encoding="utf-8", errors="strict") as f:
            return f.read()
    except OSError as e:
        raise Refused(f"cannot read {path}: {e}") from e


# What a `ptx-scan.sh --no-jit` scan prints in place of the driver JIT's reading: in the banner as
# `jit=skipped(no-jit)`, and in each row's two JIT columns.
SKIPPED = "skipped(no-jit)"
JIT_COLUMNS = ("jit_regs", "jit_local")


def parse_scan(path):
    """A saved ptx-scan output: (method, header, {entry: row fields}, {entry: (md5, lines)}, nojit).
    `method` is the md5 line's `method=` value, or `none` for a block from before the method was
    named. `nojit` is whether the scan skipped the driver JIT: its banner says so and its JIT cells
    read SKIPPED, and a scan where the two disagree, or with a skipped cell in any other column, is
    refused."""
    text = read(path)
    banner = next((l for l in text.split("\n") if l.startswith("ptx-scan bin=")), None)
    if banner is None:
        raise Refused(f"{path} holds no ptx-scan banner line")
    if banner.endswith("scan=failed"):
        raise Refused(f"{path} is a failed scan: {banner}")
    jit = [w[len("jit="):] for w in banner.split() if w.startswith("jit=")]
    if jit not in ([], [SKIPPED]):
        raise Refused(f"{path}: a banner jit={' jit='.join(jit)} this reader does not parse "
                      f"(a scan that ran the JIT names jit-card, jit-cc and jit-cuda, not jit=)")
    nojit = bool(jit)
    rows, md5, method, header, state = {}, {}, None, None, None
    for line in text.split("\n"):
        f = line.split()
        if line.startswith("ptx-scan: "):
            # ptx-scan's notes go to stderr; a log of both streams can hold one inside the table.
            continue
        if line.startswith("entry ") and header is None:
            header, state = f, "rows"
        elif line.startswith("ptx-scan-md5:"):
            m = re.fullmatch(r"ptx-scan-md5:(?: method=(\S+))?", line.strip())
            if m is None:
                raise Refused(f"{path}: an md5 header this reader does not parse: {line}")
            method, state = m.group(1) or "none", "md5"
        elif state == "rows" and len(f) == len(header):
            rows[f[0]] = f[1:]
        elif state == "md5" and len(f) == 3 and re.fullmatch(r"[0-9a-f]{32}", f[1]):
            md5[f[0]] = (f[1], f[2])
        else:
            state = None
    if header is None or method is None:
        raise Refused(f"{path} holds no table header or no md5 block")
    jit_at = [header.index(c) - 1 for c in JIT_COLUMNS if c in header]
    for e, cells in rows.items():
        for i, cell in enumerate(cells):
            if cell != SKIPPED:
                if nojit and i in jit_at:
                    raise Refused(f"{path}: the banner says jit={SKIPPED} but {e}'s {header[i + 1]} "
                                  f"reads {cell}; a skipped scan holds no JIT value")
                continue
            if i not in jit_at:
                raise Refused(f"{path}: {e}'s {header[i + 1]} reads {SKIPPED}; only "
                              f"{' and '.join(JIT_COLUMNS)} can be skipped")
            if not nojit:
                raise Refused(f"{path}: {e}'s {header[i + 1]} reads {SKIPPED} but the banner names "
                              "the JIT (jit-card, jit-cc, jit-cuda)")
    # Every entry has a row and a digest, or no compare of this scan is a proof: an entry the md5 block
    # lacks would be read as unmoved.
    if set(rows) != set(md5):
        odd = sorted(set(rows) ^ set(md5))
        raise Refused(f"{path}: {len(odd)} entries in only one of the table and the md5 block "
                      f"({' '.join(odd[:4])}) — a cut or mixed log")
    return method, header, rows, md5, nojit


def compare_scans(base_path, new_path, list_equal=False):
    """Print the entries, rows and digests two scans do not share (with `list_equal`, also the entries
    whose digest is equal, one line each); return whether any moved."""
    bm, header, br, b5, bj = parse_scan(base_path)
    nm, nheader, nr, n5, nj = parse_scan(new_path)
    if bm != nm:
        raise Refused(f"{base_path} digests are method {bm} and {new_path}'s are method {nm}: md5 "
                      "blocks of two methods do not compare; rescan the older binary with this "
                      "tree's tools/ptx-scan.sh and extractor")
    if header != nheader:
        raise Refused(f"{base_path} and {new_path} have other table columns: {' '.join(header)} "
                      f"against {' '.join(nheader)}")
    # A skipped JIT cell is no value: with either side skipped the two JIT columns are dropped from
    # both sides, whatever the other side holds.
    skipped = {(True, True): "both", (True, False): "base", (False, True): "new"}.get((bj, nj))
    if skipped:
        keep = [i for i, h in enumerate(header[1:]) if h not in JIT_COLUMNS]
        br = {e: [cells[i] for i in keep] for e, cells in br.items()}
        nr = {e: [cells[i] for i in keep] for e, cells in nr.items()}
        header = [header[0]] + [header[1 + i] for i in keep]
    only_b, only_n = sorted(set(br) - set(nr)), sorted(set(nr) - set(br))
    common = sorted(set(br) & set(nr))
    rows_moved = [e for e in common if br[e] != nr[e]]
    md5_common = sorted(set(b5) & set(n5))
    md5_moved = [e for e in md5_common if b5[e] != n5[e]]
    print(f"ptx-canon: scans method={bm} entries base={len(br)} new={len(nr)} "
          f"only-base={len(only_b)} only-new={len(only_n)}")
    if skipped:
        print(f"ptx-canon: jit columns not compared (no-jit: {skipped})")
    for e in only_b:
        print(f"ptx-canon: {e} only in base")
    for e in only_n:
        print(f"ptx-canon: {e} only in new")
    print(f"ptx-canon: table rows identical={len(common) - len(rows_moved)} moved={len(rows_moved)}")
    for e in rows_moved:
        cols = " ".join(f"{h}:{a}->{b}" for h, a, b in zip(header[1:], br[e], nr[e]) if a != b)
        print(f"ptx-canon: row-moved {e} {cols}")
    print(f"ptx-canon: md5 identical={len(md5_common) - len(md5_moved)} moved={len(md5_moved)}")
    if list_equal:
        for e in md5_common:
            if b5[e] == n5[e]:
                print(f"ptx-canon: md5-equal {e}")
    for e in md5_moved:
        print(f"ptx-canon: md5-moved {e} lines {b5[e][1]} -> {n5[e][1]}")
    return bool(only_b or only_n or rows_moved or md5_moved or set(b5) ^ set(n5))


SELF_MODULE = """.version 8.7
.target sm_86
{decls}
.global .align 1 .b8 _$_str[4] = {{97, 98, 99, 0}};

.visible .entry alpha(
\t.param .u64 alpha_param_0
)
{{
\tld.global.nc.u32 \t%r1, [{g}+4];
\tst.shared.b32 \t[{s2}], %r1;
\tst.shared.b32 \t[{s3}+4], %r1;
\tld.shared.b32 \t%r2, [{dy}];
\tmov.u64 \t%rd1, _$_str;
\tret;
}}
.visible .entry beta(
)
{{
\tld.shared.b32 \t%r1, [{s5}];
\tret;
}}
"""
SELF_PLAIN = ("__device_global_4", "__shared_mem_2", "__shared_mem_3", "__shared_mem_5",
              "__dynamic_smem_alpha")


def self_module(names=SELF_PLAIN, order=(0, 1, 2, 3, 4), sizes=(8, 256, 256, 64)):
    """The self-test's module, the twin of ptx.rs's test `module`."""
    g, s2, s3, s5, dy = names
    decls = [f".visible .global .align 4 .b8 {g}[{sizes[0]}] = {{1, 2, 3, 4, 5, 6, 7, 8}};",
             f".visible .shared .align 4 .b8 {s2}[{sizes[1]}];",
             f".visible .shared .align 4 .b8 {s3}[{sizes[2]}];",
             f".visible .shared .align 4 .b8 {s5}[{sizes[3]}];",
             f".extern .shared .align 16 .b8 {dy}[];"]
    return SELF_MODULE.format(decls="\n".join(decls[i] for i in order), g=g, s2=s2, s3=s3, s5=s5,
                              dy=dy)


# The real pair behind the value-equal verdict: the join block of gemm_q6k (basic block `bb50`, 17 PHI
# registers set to zero) as the backend wrote it before and after an unrelated `#[kernel]` was added to
# crates/gpu. Verbatim from the two PTX bodies, comments cut; only the tail is ours (it adds distinct
# constants to four of the PHI registers and stores them, standing in for the block's real consumers,
# so that the pairing of registers is observable). The new side numbers every PHI register one higher,
# writes the integer zero before the float zero (`mov.b32 %r6741, 0` then `%r432, 0f00000000`), and puts
# the float PHI's copy (`%r441`) in the middle of the integer copies instead of last.
REAL_BASE_JOIN = """$L__BB178_39:
bar.sync 0;
mov.b32 %r440, 0f00000000;
mov.b32 %r6740, 0;
mov.b32 %r6741, %r6740;
mov.b32 %r6742, %r6740;
mov.b32 %r6743, %r6740;
mov.b32 %r6744, %r6740;
mov.b32 %r6745, %r6740;
mov.b32 %r6746, %r6740;
mov.b32 %r6747, %r6740;
mov.b32 %r6749, %r6740;
mov.b32 %r6750, %r6740;
mov.b32 %r6751, %r6740;
mov.b32 %r6752, %r6740;
mov.b32 %r6753, %r6740;
mov.b32 %r6754, %r6740;
mov.b32 %r6755, %r6740;
mov.b32 %r6756, %r6740;
mov.b32 %r449, %r440;
@%p637 bra $L__BB178_41;
bra.uni $L__BB178_40;""".split("\n")
REAL_HEAD_JOIN = """$L__BB178_39:
bar.sync 0;
mov.b32 %r6741, 0;
mov.b32 %r432, 0f00000000;
mov.b32 %r6742, %r6741;
mov.b32 %r6743, %r6741;
mov.b32 %r6744, %r6741;
mov.b32 %r6745, %r6741;
mov.b32 %r6746, %r6741;
mov.b32 %r6747, %r6741;
mov.b32 %r6748, %r6741;
mov.b32 %r441, %r432;
mov.b32 %r6750, %r6741;
mov.b32 %r6751, %r6741;
mov.b32 %r6752, %r6741;
mov.b32 %r6753, %r6741;
mov.b32 %r6754, %r6741;
mov.b32 %r6755, %r6741;
mov.b32 %r6756, %r6741;
mov.b32 %r6757, %r6741;
@%p637 bra $L__BB178_41;
bra.uni $L__BB178_40;""".split("\n")
REAL_BASE_TAIL = """$L__BB178_40:
add.s32 %r6741, %r6741, 1;
add.s32 %r6745, %r6745, 2;
add.s32 %r6756, %r6756, 3;
add.s32 %r449, %r449, 4;
st.global.b32 [%rd1], %r6741;
st.global.b32 [%rd1+4], %r6745;
st.global.b32 [%rd1+8], %r6756;
st.global.b32 [%rd1+12], %r449;
ret;""".split("\n")
REAL_HEAD_TAIL = [l.replace("%r6741", "%r6742").replace("%r6745", "%r6746").replace("%r6756", "%r6757")
                  .replace("%r449", "%r441") for l in REAL_BASE_TAIL]


# Three more real pairs from the same two builds (the vecprog round's), as the backend wrote them, comments cut and
# tabs collapsed, cut to the lines that move. REAL_Q6K_* is gemm_q6k from its join block `$L__BB178_39` through the loop
# body's first three definitions: the head numbers the phi registers 6741..6757 where the base has 6740..6756, in
# reverse, and holds the zero in another one of them. REAL_BF16_* is the same place in gemm_q6k_bf16 (`$L__BB179_39`),
# where it is the base that has the float copy in the middle. REAL_BF16_COPY_* is gemm_q6k_bf16's block `$L__BB179_59`:
# the same registers in both, the two zeros swapped and the 64-bit zero's copy moved from the middle of sixteen copies
# of one register to the end.
REAL_Q6K_BASE = """$L__BB178_39:
bar.sync 0;
mov.b32 %r440, 0f00000000;
mov.b32 %r6740, 0;
mov.b32 %r6741, %r6740;
mov.b32 %r6742, %r6740;
mov.b32 %r6743, %r6740;
mov.b32 %r6744, %r6740;
mov.b32 %r6745, %r6740;
mov.b32 %r6746, %r6740;
mov.b32 %r6747, %r6740;
mov.b32 %r6749, %r6740;
mov.b32 %r6750, %r6740;
mov.b32 %r6751, %r6740;
mov.b32 %r6752, %r6740;
mov.b32 %r6753, %r6740;
mov.b32 %r6754, %r6740;
mov.b32 %r6755, %r6740;
mov.b32 %r6756, %r6740;
mov.b32 %r449, %r440;
@%p637 bra $L__BB178_41;
bra.uni $L__BB178_40;
$L__BB178_40:
mul.lo.s64 %rd470, %rd77, 210;
add.s64 %rd471, %rd470, %rd35;
add.s64 %rd472, %rd43, %rd471;
add.s64 %rd473, %rd44, %rd471;
add.s64 %rd474, %rd471, 200;
add.s64 %rd475, %rd471, 208;
and.b64 %rd476, %rd472, -4;
add.s64 %rd477, %rd7, %rd476;
ld.global.nc.b32 %rd478, [%rd477];
add.s64 %rd479, %rd472, 3;
and.b64 %rd480, %rd479, -4;
add.s64 %rd481, %rd7, %rd480;
ld.global.nc.b32 %rd482, [%rd481];
shl.b64 %rd483, %rd482, 32;
or.b64 %rd484, %rd483, %rd478;
cvt.u32.u64 %r2111, %rd471;
shl.b32 %r2112, %r2111, 3;
and.b32 %r2113, %r2112, 16;
shr.u64 %rd485, %rd484, %r2113;
cvt.u32.u64 %r6740, %rd485;
add.s64 %rd486, %rd472, 16;
and.b64 %rd487, %rd486, -4;
add.s64 %rd488, %rd7, %rd487;
ld.global.nc.b32 %rd489, [%rd488];
add.s64 %rd490, %rd472, 19;
and.b64 %rd491, %rd490, -4;
add.s64 %rd492, %rd7, %rd491;
ld.global.nc.b32 %rd493, [%rd492];
shl.b64 %rd494, %rd493, 32;
or.b64 %rd495, %rd494, %rd489;
shr.u64 %rd496, %rd495, %r2113;
cvt.u32.u64 %r6741, %rd496;
add.s64 %rd497, %rd472, 32;
and.b64 %rd498, %rd497, -4;
add.s64 %rd499, %rd7, %rd498;
ld.global.nc.b32 %rd500, [%rd499];
add.s64 %rd501, %rd472, 35;
and.b64 %rd502, %rd501, -4;
add.s64 %rd503, %rd7, %rd502;
ld.global.nc.b32 %rd504, [%rd503];
shl.b64 %rd505, %rd504, 32;
or.b64 %rd506, %rd505, %rd500;
shr.u64 %rd507, %rd506, %r2113;
cvt.u32.u64 %r6742, %rd507;
add.s64 %rd508, %rd472, 48;
and.b64 %rd509, %rd508, -4;
add.s64 %rd510, %rd7, %rd509;""".split("\n")
REAL_Q6K_HEAD = """$L__BB178_39:
bar.sync 0;
mov.b32 %r6741, 0;
mov.b32 %r432, 0f00000000;
mov.b32 %r6742, %r6741;
mov.b32 %r6743, %r6741;
mov.b32 %r6744, %r6741;
mov.b32 %r6745, %r6741;
mov.b32 %r6746, %r6741;
mov.b32 %r6747, %r6741;
mov.b32 %r6748, %r6741;
mov.b32 %r441, %r432;
mov.b32 %r6750, %r6741;
mov.b32 %r6751, %r6741;
mov.b32 %r6752, %r6741;
mov.b32 %r6753, %r6741;
mov.b32 %r6754, %r6741;
mov.b32 %r6755, %r6741;
mov.b32 %r6756, %r6741;
mov.b32 %r6757, %r6741;
@%p637 bra $L__BB178_41;
bra.uni $L__BB178_40;
$L__BB178_40:
mul.lo.s64 %rd470, %rd77, 210;
add.s64 %rd471, %rd470, %rd35;
add.s64 %rd472, %rd43, %rd471;
add.s64 %rd473, %rd44, %rd471;
add.s64 %rd474, %rd471, 200;
add.s64 %rd475, %rd471, 208;
and.b64 %rd476, %rd472, -4;
add.s64 %rd477, %rd7, %rd476;
ld.global.nc.b32 %rd478, [%rd477];
add.s64 %rd479, %rd472, 3;
and.b64 %rd480, %rd479, -4;
add.s64 %rd481, %rd7, %rd480;
ld.global.nc.b32 %rd482, [%rd481];
shl.b64 %rd483, %rd482, 32;
or.b64 %rd484, %rd483, %rd478;
cvt.u32.u64 %r2111, %rd471;
shl.b32 %r2112, %r2111, 3;
and.b32 %r2113, %r2112, 16;
shr.u64 %rd485, %rd484, %r2113;
cvt.u32.u64 %r6757, %rd485;
add.s64 %rd486, %rd472, 16;
and.b64 %rd487, %rd486, -4;
add.s64 %rd488, %rd7, %rd487;
ld.global.nc.b32 %rd489, [%rd488];
add.s64 %rd490, %rd472, 19;
and.b64 %rd491, %rd490, -4;
add.s64 %rd492, %rd7, %rd491;
ld.global.nc.b32 %rd493, [%rd492];
shl.b64 %rd494, %rd493, 32;
or.b64 %rd495, %rd494, %rd489;
shr.u64 %rd496, %rd495, %r2113;
cvt.u32.u64 %r6756, %rd496;
add.s64 %rd497, %rd472, 32;
and.b64 %rd498, %rd497, -4;
add.s64 %rd499, %rd7, %rd498;
ld.global.nc.b32 %rd500, [%rd499];
add.s64 %rd501, %rd472, 35;
and.b64 %rd502, %rd501, -4;
add.s64 %rd503, %rd7, %rd502;
ld.global.nc.b32 %rd504, [%rd503];
shl.b64 %rd505, %rd504, 32;
or.b64 %rd506, %rd505, %rd500;
shr.u64 %rd507, %rd506, %r2113;
cvt.u32.u64 %r6755, %rd507;
add.s64 %rd508, %rd472, 48;
and.b64 %rd509, %rd508, -4;
add.s64 %rd510, %rd7, %rd509;""".split("\n")
REAL_BF16_BASE = """$L__BB179_39:
bar.sync 0;
mov.b32 %r6885, 0;
mov.b32 %r432, 0f00000000;
mov.b32 %r6886, %r6885;
mov.b32 %r6887, %r6885;
mov.b32 %r6888, %r6885;
mov.b32 %r6889, %r6885;
mov.b32 %r6890, %r6885;
mov.b32 %r6891, %r6885;
mov.b32 %r6892, %r6885;
mov.b32 %r441, %r432;
mov.b32 %r6894, %r6885;
mov.b32 %r6895, %r6885;
mov.b32 %r6896, %r6885;
mov.b32 %r6897, %r6885;
mov.b32 %r6898, %r6885;
mov.b32 %r6899, %r6885;
mov.b32 %r6900, %r6885;
mov.b32 %r6901, %r6885;
@%p673 bra $L__BB179_41;
bra.uni $L__BB179_40;
$L__BB179_40:
mul.lo.s64 %rd470, %rd77, 210;
add.s64 %rd471, %rd470, %rd35;
add.s64 %rd472, %rd43, %rd471;
add.s64 %rd473, %rd44, %rd471;
add.s64 %rd474, %rd471, 200;
add.s64 %rd475, %rd471, 208;
and.b64 %rd476, %rd472, -4;
add.s64 %rd477, %rd7, %rd476;
ld.global.nc.b32 %rd478, [%rd477];
add.s64 %rd479, %rd472, 3;
and.b64 %rd480, %rd479, -4;
add.s64 %rd481, %rd7, %rd480;
ld.global.nc.b32 %rd482, [%rd481];
shl.b64 %rd483, %rd482, 32;
or.b64 %rd484, %rd483, %rd478;
cvt.u32.u64 %r2111, %rd471;
shl.b32 %r2112, %r2111, 3;
and.b32 %r2113, %r2112, 16;
shr.u64 %rd485, %rd484, %r2113;
cvt.u32.u64 %r6901, %rd485;
add.s64 %rd486, %rd472, 16;
and.b64 %rd487, %rd486, -4;
add.s64 %rd488, %rd7, %rd487;
ld.global.nc.b32 %rd489, [%rd488];
add.s64 %rd490, %rd472, 19;
and.b64 %rd491, %rd490, -4;
add.s64 %rd492, %rd7, %rd491;
ld.global.nc.b32 %rd493, [%rd492];
shl.b64 %rd494, %rd493, 32;
or.b64 %rd495, %rd494, %rd489;
shr.u64 %rd496, %rd495, %r2113;
cvt.u32.u64 %r6900, %rd496;
add.s64 %rd497, %rd472, 32;
and.b64 %rd498, %rd497, -4;
add.s64 %rd499, %rd7, %rd498;
ld.global.nc.b32 %rd500, [%rd499];
add.s64 %rd501, %rd472, 35;
and.b64 %rd502, %rd501, -4;
add.s64 %rd503, %rd7, %rd502;
ld.global.nc.b32 %rd504, [%rd503];
shl.b64 %rd505, %rd504, 32;
or.b64 %rd506, %rd505, %rd500;
shr.u64 %rd507, %rd506, %r2113;
cvt.u32.u64 %r6899, %rd507;
add.s64 %rd508, %rd472, 48;
and.b64 %rd509, %rd508, -4;
add.s64 %rd510, %rd7, %rd509;""".split("\n")
REAL_BF16_HEAD = """$L__BB179_39:
bar.sync 0;
mov.b32 %r440, 0f00000000;
mov.b32 %r6884, 0;
mov.b32 %r6885, %r6884;
mov.b32 %r6886, %r6884;
mov.b32 %r6887, %r6884;
mov.b32 %r6888, %r6884;
mov.b32 %r6889, %r6884;
mov.b32 %r6890, %r6884;
mov.b32 %r6891, %r6884;
mov.b32 %r6893, %r6884;
mov.b32 %r6894, %r6884;
mov.b32 %r6895, %r6884;
mov.b32 %r6896, %r6884;
mov.b32 %r6897, %r6884;
mov.b32 %r6898, %r6884;
mov.b32 %r6899, %r6884;
mov.b32 %r6900, %r6884;
mov.b32 %r449, %r440;
@%p673 bra $L__BB179_41;
bra.uni $L__BB179_40;
$L__BB179_40:
mul.lo.s64 %rd470, %rd77, 210;
add.s64 %rd471, %rd470, %rd35;
add.s64 %rd472, %rd43, %rd471;
add.s64 %rd473, %rd44, %rd471;
add.s64 %rd474, %rd471, 200;
add.s64 %rd475, %rd471, 208;
and.b64 %rd476, %rd472, -4;
add.s64 %rd477, %rd7, %rd476;
ld.global.nc.b32 %rd478, [%rd477];
add.s64 %rd479, %rd472, 3;
and.b64 %rd480, %rd479, -4;
add.s64 %rd481, %rd7, %rd480;
ld.global.nc.b32 %rd482, [%rd481];
shl.b64 %rd483, %rd482, 32;
or.b64 %rd484, %rd483, %rd478;
cvt.u32.u64 %r2111, %rd471;
shl.b32 %r2112, %r2111, 3;
and.b32 %r2113, %r2112, 16;
shr.u64 %rd485, %rd484, %r2113;
cvt.u32.u64 %r6884, %rd485;
add.s64 %rd486, %rd472, 16;
and.b64 %rd487, %rd486, -4;
add.s64 %rd488, %rd7, %rd487;
ld.global.nc.b32 %rd489, [%rd488];
add.s64 %rd490, %rd472, 19;
and.b64 %rd491, %rd490, -4;
add.s64 %rd492, %rd7, %rd491;
ld.global.nc.b32 %rd493, [%rd492];
shl.b64 %rd494, %rd493, 32;
or.b64 %rd495, %rd494, %rd489;
shr.u64 %rd496, %rd495, %r2113;
cvt.u32.u64 %r6885, %rd496;
add.s64 %rd497, %rd472, 32;
and.b64 %rd498, %rd497, -4;
add.s64 %rd499, %rd7, %rd498;
ld.global.nc.b32 %rd500, [%rd499];
add.s64 %rd501, %rd472, 35;
and.b64 %rd502, %rd501, -4;
add.s64 %rd503, %rd7, %rd502;
ld.global.nc.b32 %rd504, [%rd503];
shl.b64 %rd505, %rd504, 32;
or.b64 %rd506, %rd505, %rd500;
shr.u64 %rd507, %rd506, %r2113;
cvt.u32.u64 %r6886, %rd507;
add.s64 %rd508, %rd472, 48;
and.b64 %rd509, %rd508, -4;
add.s64 %rd510, %rd7, %rd509;""".split("\n")
REAL_BF16_COPY_BASE = """$L__BB179_59:
mov.b32 %r6970, 0;
mov.b64 %rd1903, 0d0000000000000000;
mul.lo.s64 %rd1879, %rd1902, 210;
mov.b32 %r6971, %r6970;
mov.b32 %r6972, %r6970;
mov.b32 %r6973, %r6970;
mov.b32 %r6974, %r6970;
mov.b32 %r6975, %r6970;
mov.b32 %r6976, %r6970;
mov.b32 %r6977, %r6970;
mov.b64 %rd1904, %rd1903;
mov.b32 %r6978, %r6970;
mov.b32 %r6979, %r6970;
mov.b32 %r6980, %r6970;
mov.b32 %r6981, %r6970;
mov.b32 %r6982, %r6970;
mov.b32 %r6983, %r6970;
mov.b32 %r6984, %r6970;
mov.b32 %r6985, %r6970;
@%p673 bra $L__BB179_61;
bra.uni $L__BB179_60;""".split("\n")
REAL_BF16_COPY_HEAD = """$L__BB179_59:
mov.b64 %rd1903, 0d0000000000000000;
mov.b32 %r6970, 0;
mul.lo.s64 %rd1879, %rd1902, 210;
mov.b32 %r6971, %r6970;
mov.b32 %r6972, %r6970;
mov.b32 %r6973, %r6970;
mov.b32 %r6974, %r6970;
mov.b32 %r6975, %r6970;
mov.b32 %r6976, %r6970;
mov.b32 %r6977, %r6970;
mov.b32 %r6978, %r6970;
mov.b32 %r6979, %r6970;
mov.b32 %r6980, %r6970;
mov.b32 %r6981, %r6970;
mov.b32 %r6982, %r6970;
mov.b32 %r6983, %r6970;
mov.b32 %r6984, %r6970;
mov.b32 %r6985, %r6970;
mov.b64 %rd1904, %rd1903;
@%p673 bra $L__BB179_61;
bra.uni $L__BB179_60;""".split("\n")


def entry_text(lines, name="t"):
    """A module holding one entry whose body is `lines`."""
    return f".visible .entry {name}(\n\t.param .u64 {name}_param_0\n)\n{{\n" + "\n".join(lines) + "\n}\n"


def judge(base, new):
    """(verdict, fails, output, stats) of `compare` over two small bodies; stats is the proof's count
    dict, or the Unproved message."""
    import contextlib
    import io
    bt, nt = entry_text(base), entry_text(new)
    bd, nd = declarations(bt, "b"), declarations(nt, "n")
    with contextlib.redirect_stdout(io.StringIO()) as out:
        verdict, fails = compare(("b", bt, bd), ("n", nt, nd), "t", "t", 0)
    try:
        stats = prove_value_equal(body(bt, "t"), body(nt, "t"), "t", "t", bd, nd)
    except Unproved as e:
        stats = str(e)
    return verdict, fails, out.getvalue(), stats


def value_equal_cases(expect):
    """The value-equal verdict: positives, then FAIL-first mutants that must not reach it."""
    def same(base, new, what):
        verdict, fails, out, st = judge(base, new)
        expect(verdict == "value-equal" and not fails, f"{what}: want value-equal, got {verdict} "
               f"fails={fails} ({st if isinstance(st, str) else ''})")
        return st

    def never(base, new, what, why=None):
        verdict, fails, out, st = judge(base, new)
        expect(verdict != "value-equal" and fails, f"{what}: must not be value-equal, got {verdict} "
               f"fails={fails}")
        if why is not None:
            expect(isinstance(st, str) and why in st, f"{what}: the proof should fail with {why!r}, "
                   f"got {st!r}")

    pre = ["ld.param.u64 %rd1, [t_param_0];", "ld.global.u32 %r1, [%rd1];",
           "ld.global.u32 %r2, [%rd1+4];"]
    post = ["st.global.u32 [%rd1+8], %r3;", "st.global.u32 [%rd1+12], %r4;", "ret;"]
    base = pre + ["add.s32 %r3, %r1, 1;", "mul.lo.s32 %r4, %r2, 3;"] + post
    # --- positives
    head = ["ld.param.u64 %rd9, [t_param_0];", "ld.global.u32 %r7, [%rd9];",
            "ld.global.u32 %r5, [%rd9+4];", "mul.lo.s32 %r2, %r5, 3;", "add.s32 %r8, %r7, 1;",
            "st.global.u32 [%rd9+8], %r8;", "st.global.u32 [%rd9+12], %r2;", "ret;"]
    st = same(base, head, "a bijective renaming with two independent pure lines swapped")
    expect(isinstance(st, dict) and st["renamed"] >= 4, f"renamed registers are counted: {st}")
    same(pre + ["add.s32 %r3, %r1, 1;", "add.s32 %r4, %r1, 2;"] + post,
         pre + ["add.s32 %r4, %r1, 2;", "add.s32 %r3, %r1, 1;"] + post,
         "two pure lines that read one register swap (`independent` calls that dependent)")
    same(pre + ["mov.u32 %r5, %tid.x;"] + post, ["ld.param.u64 %rd1, [t_param_0];", "mov.u32 %r5, %tid.x;"]
         + pre[1:] + post, "a read of a fixed special register moves across loads")
    zero_a = ["mov.b32 %r1, 0f00000000;", "mov.b32 %r2, 0;", "mov.b32 %r3, %r2;", "mov.b32 %r4, %r1;",
              "add.s32 %r2, %r2, 1;", "add.s32 %r3, %r3, 2;", "add.s32 %r4, %r4, 3;",
              "st.global.u32 [%rd1], %r2;", "st.global.u32 [%rd1+4], %r3;",
              "st.global.u32 [%rd1+8], %r4;", "ret;"]
    zero_b = ["mov.b32 %r2, 0;", "mov.b32 %r1, 0f00000000;", "mov.b32 %r4, %r1;", "mov.b32 %r3, %r2;",
              "add.s32 %r2, %r2, 1;", "add.s32 %r3, %r3, 2;", "add.s32 %r4, %r4, 3;",
              "st.global.u32 [%rd1], %r2;", "st.global.u32 [%rd1+4], %r3;",
              "st.global.u32 [%rd1+8], %r4;", "ret;"]
    same(zero_a, zero_b, "zero constants spelled 0f00000000 and 0, copies from either holder")
    # --- the real pair
    rb, rh = REAL_BASE_JOIN + REAL_BASE_TAIL, REAL_HEAD_JOIN + REAL_HEAD_TAIL
    ca = canon(body(entry_text(rb), "t"), "t", {})
    cb = canon(body(entry_text(rh), "t"), "t", {})
    expect(ca != cb and sorted(ca) != sorted(cb),
           "the real pair differs as canonical text and is not a same-multiset reorder")
    st = same(rb, rh, "the real gemm_q6k join block")
    expect(isinstance(st, dict) and st["ties"] > 0 and st["renamed"] > 0,
           f"the real pair pairs registers of one value and renames them: {st}")
    same(rb, [rh[0]] + [rh[2]] + [rh[1]] + rh[3:], "the real pair with bar.sync moved below a pure line")
    # --- the real pairs, cut from the two entries
    q6, bf = (REAL_Q6K_BASE, REAL_Q6K_HEAD), (REAL_BF16_BASE, REAL_BF16_HEAD)
    cp = (REAL_BF16_COPY_BASE, REAL_BF16_COPY_HEAD)
    st = same(q6[0], q6[1], "the real gemm_q6k excerpt: the join block and the loop body's first definitions")
    expect(isinstance(st, dict) and st["renamed"] >= 18 and st["ties"] > 0,
           f"the real gemm_q6k excerpt renames the phi registers and pairs equal-valued ones: {st}")
    st = same(bf[0], bf[1], "the real gemm_q6k_bf16 excerpt: the same place, the roles of the two builds exchanged")
    expect(isinstance(st, dict) and st["renamed"] >= 18 and st["ties"] > 0,
           f"the real gemm_q6k_bf16 excerpt renames the phi registers too: {st}")
    verdict, fails, out, st = judge(*cp)
    expect(verdict == "reordered-only" and not fails and "every one independent" in out,
           f"the real bf16 copy block is the older verdict's, a pure reorder: {verdict} {out!r}")
    expect(isinstance(st, dict) and st["renamed"] == 0, f"and the proof reads it as no renaming: {st}")
    # --- FAIL-first mutants of the small programs
    never(pre + ["add.s32 %r3, %r1, 1;", "mul.lo.s32 %r4, %r3, 2;"] + post,
          pre + ["mul.lo.s32 %r4, %r3, 2;", "add.s32 %r3, %r1, 1;"] + post,
          "a def-use dependent swap", "does not hold")
    never(pre + ["ld.global.u32 %r3, [%rd1+8];", "add.s32 %r4, %r3, 1;"] + post,
          pre + ["add.s32 %r4, %r3, 1;", "ld.global.u32 %r3, [%rd1+8];"] + post,
          "a pure line moved above the load it reads")
    never(base, [l.replace("%r4, %r2, 3", "%r4, %r2, 4") for l in base],
          "a changed immediate", "different lines")
    never(pre + ["add.s32 %r3, %r1, 1;", "add.s32 %r4, %r2, 2;"] + post,
          pre + ["add.s32 %r3, %r1, 2;", "add.s32 %r4, %r2, 1;"] + post,
          "two immediates exchanged (same lines, other values)")
    never(pre + post[:2] + ["ret;"], pre + ["ld.global.u32 %r1, [%rd1+4];"] + post[:2] + ["ret;"],
          "a line added")
    never(pre + ["st.global.u32 [%rd1+8], %r1;", "st.global.u32 [%rd1+12], %r2;", "ret;"],
          pre + ["st.global.u32 [%rd1+12], %r2;", "st.global.u32 [%rd1+8], %r1;", "ret;"],
          "two stores swapped", "anchor")
    never(["ld.param.u64 %rd1, [t_param_0];", "ld.global.u32 %r1, [%rd1];",
           "ld.global.u32 %r2, [%rd1+4];", "st.global.u32 [%rd1+8], %r1;",
           "st.global.u32 [%rd1+12], %r2;", "ret;"],
          ["ld.param.u64 %rd1, [t_param_0];", "ld.global.u32 %r1, [%rd1];",
           "ld.global.u32 %r1, [%rd1+4];", "st.global.u32 [%rd1+8], %r1;",
           "st.global.u32 [%rd1+12], %r1;", "ret;"],
          "a non-bijective renaming (two registers merged)")
    never(["ld.param.u64 %rd1, [t_param_0];", "add.s32 %r3, %r1, 1;", "$L__BB0_1:",
           "st.global.u32 [%rd1], %r3;", "ret;"],
          ["ld.param.u64 %rd1, [t_param_0];", "$L__BB0_1:", "add.s32 %r3, %r1, 1;",
           "st.global.u32 [%rd1], %r3;", "ret;"], "a pure line moved across a label")
    never(["ld.param.u64 %rd1, [t_param_0];", "add.s32 %r3, %r1, 1;", "@%p1 bra $L__BB0_1;",
           "st.global.u32 [%rd1], %r3;", "$L__BB0_1:", "ret;"],
          ["ld.param.u64 %rd1, [t_param_0];", "@%p1 bra $L__BB0_1;", "add.s32 %r3, %r1, 1;",
           "st.global.u32 [%rd1], %r3;", "$L__BB0_1:", "ret;"], "a pure line moved across a branch")
    never(["add.cc.u32 %r1, %r2, %r3;", "addc.u32 %r4, %r5, %r6;", "ret;"],
          ["addc.u32 %r4, %r5, %r6;", "add.cc.u32 %r1, %r2, %r3;", "ret;"],
          "a carry chain reversed", "anchor")
    never(pre + ["mov.u32 %r5, %clock;", "st.global.u32 [%rd1+8], %r1;", "ret;"],
          pre + ["st.global.u32 [%rd1+8], %r1;", "mov.u32 %r5, %clock;", "ret;"],
          "a read of the clock moved across a store", "anchor")
    never(pre + ["setp.eq.s32 %p1, %r1, 0;", "@%p1 mov.b32 %r3, 7;"] + post,
          pre + ["setp.eq.s32 %p1, %r2, 0;", "@%p1 mov.b32 %r3, 7;"] + post,
          "a guard on another value")
    never(pre + ["call.uni f, ();", "add.s32 %r3, %r1, 1;", "add.s32 %r4, %r2, 2;"] + post,
          pre + ["call.uni f, ();", "add.s32 %r4, %r2, 2;", "add.s32 %r3, %r1, 1;"] + post,
          "an opcode the proof does not model", "not modelled")
    same(pre + ["mov.b32 %r5, %r1;", "mov.b32 %r6, %r1;", "st.global.u32 [%rd1+8], %r5;",
                "st.global.u32 [%rd1+12], %r6;", "ret;"],
         pre + ["mov.b32 %r5, %r1;", "mov.b32 %r6, %r5;", "st.global.u32 [%rd1+8], %r5;",
                "st.global.u32 [%rd1+12], %r6;", "ret;"],
         "a copy of a register against a copy of that copy")
    gpre = pre + ["setp.eq.s32 %p1, %r1, 0;", "setp.eq.s32 %p2, %r2, 0;"]
    guarded = gpre + ["@%p1 mov.b32 %r3, 7;", "@%p2 st.global.u32 [%rd1+8], %r3;", "ret;"]
    same(guarded, pre + ["setp.eq.s32 %p8, %r2, 0;", "setp.eq.s32 %p9, %r1, 0;", "@%p9 mov.b32 %r3, 7;",
                         "@%p8 st.global.u32 [%rd1+8], %r3;", "ret;"],
         "guards renamed with their predicates, the two compares swapped")
    never(gpre + ["@%p1 mov.b32 %r3, 7;", "st.global.u32 [%rd1+8], %r3;", "ret;"],
          gpre + ["@%p2 mov.b32 %r3, 7;", "st.global.u32 [%rd1+8], %r3;", "ret;"],
          "a pure line guarded by the other predicate")
    never(gpre + ["@%p1 st.global.u32 [%rd1+8], %r1;", "ret;"],
          gpre + ["@%p2 st.global.u32 [%rd1+8], %r1;", "ret;"],
          "a store guarded by the other predicate")
    gold = ["ld.param.u64 %rd1, [t_param_0];", "setp.eq.s32 %p1, %r8, 0;"]
    gtail = ["@%p1 ld.global.u32 %r3, [%rd1+8];", "ld.global.u32 %r6, [%rd1+12];",
             "st.global.u32 [%rd1], %r3;", "st.global.u32 [%rd1+4], %r6;", "ret;"]
    never(gold + ["mov.b32 %r3, 1;", "mov.b32 %r6, 2;"] + gtail,
          gold + ["mov.b32 %r3, 2;", "mov.b32 %r6, 1;"] + gtail,
          "a guarded load over another old value")
    never(pre + ["add.s32 %r3, %r1, %r2;", "sub.s32 %r4, %r1, %r2;"] + post,
          pre + ["sub.s32 %r3, %r1, %r2;", "add.s32 %r4, %r1, %r2;"] + post,
          "the operators of two destinations exchanged")
    # --- FAIL-first mutants of the two excerpts
    def edit(lines, old, new):
        assert old in lines, old
        out = list(lines)
        out[out.index(old)] = new
        return out
    never(q6[0], edit(q6[1], "shr.u64 %rd496, %rd495, %r2113;", "shr.u64 %rd496, %rd495, %r2112;"),
          "real gemm_q6k: a shift amount read from another register", "ends on a value")
    never(q6[0], edit(q6[1], "add.s64 %rd497, %rd472, 32;", "add.s64 %rd497, %rd472, 33;"),
          "real gemm_q6k: an immediate changed", "different lines")
    never(q6[0], edit(q6[1], "and.b64 %rd498, %rd497, -4;", "or.b64 %rd498, %rd497, -4;"),
          "real gemm_q6k: an opcode changed", "different lines")
    never(q6[0], [q6[1][1], q6[1][0]] + q6[1][2:], "real gemm_q6k: bar.sync moved above its label")
    i = q6[1].index("shr.u64 %rd485, %rd484, %r2113;")
    swapped = list(q6[1])
    swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
    never(q6[0], swapped, "real gemm_q6k: a convert above the shift it converts")
    same(bf[0], edit(bf[1], "mov.b32 %r6886, %r6884;", "mov.b32 %r6886, %r6885;"),
         "real gemm_q6k_bf16: one copy reads another copy of the zero instead of the holder (the same value)")
    never(bf[0], edit(bf[1], "mov.b32 %r6886, %r6884;", "mov.b32 %r6886, %r6900;"),
          "real gemm_q6k_bf16: one copy reads a register the block sets only later")
    hoist = list(bf[1])
    hoist.insert(hoist.index("mov.b32 %r449, %r440;"), hoist.pop(hoist.index("@%p673 bra $L__BB179_41;")))
    never(bf[0], hoist, "real gemm_q6k_bf16: the branch moved above the float copy")
    never(bf[0], edit(bf[1], "mov.b32 %r6884, 0;", "mov.b32 %r6884, 1;"),
          "real gemm_q6k_bf16: the integer zero changed to one", "different lines")
    never(bf[0], edit(bf[1], "mov.b32 %r6893, %r6884;", "mov.b64 %r6893, %r6884;"),
          "real gemm_q6k_bf16: a copy's opcode changed", "different lines")
    never(cp[0], edit(cp[1], "mul.lo.s64 %rd1879, %rd1902, 210;", "mul.lo.s64 %rd1879, %rd1902, 211;"),
          "real bf16 copy block: an immediate changed", "different lines")
    never(cp[0], edit(cp[1], "mov.b32 %r6971, %r6970;", "mov.b32 %r6971, %r6985;"),
          "real bf16 copy block: a copy of a register the block defines later")
    never(pre + ["mov.u32 %r5, %warpid;", "st.global.u32 [%rd1+8], %r1;", "ret;"],
          pre + ["st.global.u32 [%rd1+8], %r1;", "mov.u32 %r5, %warpid;", "ret;"],
          "a read of the volatile %warpid moved across a store", "anchor")
    never(base, [l.replace("add.s32", "sub.s32") for l in base], "an opcode changed", "different lines")
    # --- a value term keeps its opcode suffix, its destination slot, its guard's sense and the
    # identity of a register read at a block's entry (one case each, so a term that drops one fails)
    sgn_a = pre + ["add.s32 %r3, %r1, %r2;", "add.u32 %r4, %r1, %r2;"] + post
    sgn_b = pre + ["add.u32 %r3, %r1, %r2;", "add.s32 %r4, %r1, %r2;"] + post
    never(sgn_a, sgn_b, "a signed and an unsigned add of one operand pair exchanged")
    same(sgn_a, pre[:1] + ["ld.global.u32 %r7, [%rd1];", "ld.global.u32 %r8, [%rd1+4];",
                           "add.u32 %r5, %r7, %r8;", "add.s32 %r6, %r7, %r8;",
                           "st.global.u32 [%rd1+8], %r6;", "st.global.u32 [%rd1+12], %r5;", "ret;"],
         "the same two adds renamed, their stores in the other order of definition")
    # a positive of these carries a rename (canonical numbering alone would make it `identical`) and
    # two independent pure lines swapped, so the proof itself has to read the construct
    tail_a = ["add.s32 %r10, %r1, 1;", "mul.lo.s32 %r11, %r2, 3;", "st.global.u32 [%rd1+20], %r10;",
              "st.global.u32 [%rd1+24], %r11;", "ret;"]
    tail_b = ["mul.lo.s32 %r12, %r2, 3;", "add.s32 %r13, %r1, 1;", "st.global.u32 [%rd1+20], %r13;",
              "st.global.u32 [%rd1+24], %r12;", "ret;"]
    sp = pre + ["setp.lt.s32 %p1|%p2, %r1, %r2;", "@%p1 st.global.u32 [%rd1+8], %r1;",
                "@%p2 st.global.u32 [%rd1+12], %r2;", "ret;"]
    never(sp, pre + ["setp.lt.s32 %p1|%p2, %r1, %r2;", "@%p2 st.global.u32 [%rd1+8], %r1;",
                     "@%p1 st.global.u32 [%rd1+12], %r2;", "ret;"],
          "the two predicates of one setp exchanged between their stores")
    same(sp[:-1] + tail_a, pre + ["setp.lt.s32 %p5|%p6, %r1, %r2;",
                                  "@%p5 st.global.u32 [%rd1+8], %r1;",
                                  "@%p6 st.global.u32 [%rd1+12], %r2;"] + tail_b,
         "the two predicates of one setp renamed, two unrelated lines swapped")
    gd = pre + ["setp.lt.s32 %p1, %r1, %r2;", "@%p1 mov.u32 %r3, 5;", "@!%p1 mov.u32 %r3, 6;",
                "st.global.u32 [%rd1+8], %r3;", "ret;"]
    never(gd, pre + ["setp.lt.s32 %p1, %r1, %r2;", "@!%p1 mov.u32 %r3, 5;", "@%p1 mov.u32 %r3, 6;",
                     "st.global.u32 [%rd1+8], %r3;", "ret;"],
          "the sense of a guard exchanged between two guarded moves")
    gs = pre + ["setp.lt.s32 %p1, %r1, %r2;", "@%p1 mov.u32 %r3, %r1;", "@!%p1 mov.u32 %r4, %r2;",
                "st.global.u32 [%rd1+8], %r3;", "st.global.u32 [%rd1+12], %r4;", "ret;"]
    never(gs, [l.replace("@%p1 mov.u32 %r3", "@!%p1 mov.u32 %r3").replace("@!%p1 mov.u32 %r4",
                                                                           "@%p1 mov.u32 %r4")
               for l in gs], "the same two guarded copies with the sense of each guard exchanged",
          "reads a value")
    same(gd[:-1] + tail_a, pre + ["setp.lt.s32 %p4, %r1, %r2;", "@%p4 mov.u32 %r7, 5;",
                                  "@!%p4 mov.u32 %r7, 6;", "st.global.u32 [%rd1+8], %r7;"] + tail_b,
         "two guarded moves renamed, two unrelated lines swapped")
    un = pre + ["ld.global.u64 %rd2, [%rd1+16];", "mov.b64 {%r3, %r4}, %rd2;",
                "st.global.u32 [%rd1+8], %r3;", "st.global.u32 [%rd1+12], %r4;", "ret;"]
    never(un, [l.replace("{%r3, %r4}", "{%r4, %r3}") for l in un],
          "the two halves of a 64-bit unpack exchanged")
    same(un[:-1] + tail_a, [l.replace("{%r3, %r4}", "{%r8, %r9}").replace("%r3;", "%r8;")
                            .replace("%r4;", "%r9;") for l in un[:-1]] + tail_b,
         "a 64-bit unpack renamed, two unrelated lines swapped")
    pk = pre + ["mov.b64 %rd2, {%r1, %r2};", "st.global.u64 [%rd1+16], %rd2;", "ret;"]
    never(pk, [l.replace("{%r1, %r2}", "{%r2, %r1}") for l in pk], "the two halves of a pack exchanged")
    same(pk[:-1] + tail_a, pre[:1] + ["ld.global.u32 %r5, [%rd1];", "ld.global.u32 %r6, [%rd1+4];",
                                      "mov.b64 %rd7, {%r5, %r6};",
                                      "st.global.u64 [%rd1+16], %rd7;"]
         + [l.replace("%r1,", "%r5,").replace("%r2,", "%r6,") for l in tail_b],
         "a pack renamed, two unrelated lines swapped")
    entry_reads = pre + ["$L__BB0_1:", "st.global.u32 [%rd1+8], %r1;", "st.global.u32 [%rd1+12], %r1;", "ret;"]
    never(entry_reads, pre + ["$L__BB0_1:", "st.global.u32 [%rd1+8], %r1;", "st.global.u32 [%rd1+12], %r2;",
                      "ret;"], "one register read twice at a block's entry read as two registers")
    lbl = pre + ["setp.lt.s32 %p1, %r1, %r2;", "@%p1 bra $L__BB0_1;", "st.global.u32 [%rd1+8], %r1;",
                 "$L__BB0_1:", "st.global.u32 [%rd1+12], %r1;", "$L__BB0_2:",
                 "st.global.u32 [%rd1+16], %r1;", "ret;"]
    never(lbl, pre + ["setp.lt.s32 %p1, %r1, %r2;", "@%p1 bra $L__BB0_1;",
                      "st.global.u32 [%rd1+8], %r1;", "$L__BB0_2:", "st.global.u32 [%rd1+12], %r1;",
                      "$L__BB0_1:", "st.global.u32 [%rd1+16], %r1;", "ret;"],
          "a branch target moved across a store, every block's anchors unchanged", "block boundaries")
    # a register paired by the leaves of an earlier register's value in the same pass is not looked up
    # again as an unpaired one: the program renamed whole must prove, not only compare canonical
    for lines_a, lines_b in (
            (["@!%p2 add.u32 %r5, %r4, %r3;", "mov.u32 %r4, %r3;"],
             ["@!%p2 add.u32 %r2, %r4, %r5;", "mov.u32 %r4, %r5;"]),
            (["@!%p2 add.u32 %r5, %r4, %r3;", "@%p1 add.s32 %r5, %r3, %r5;", "mov.u32 %r4, %r3;"],
             ["@!%p2 add.u32 %r2, %r4, %r5;", "mov.u32 %r4, %r5;", "@%p1 add.s32 %r2, %r5, %r2;"])):
        st = judge(lines_a, lines_b)[3]
        expect(isinstance(st, dict), f"a program renamed whole proves: {st}")
    # ... nor is a register an earlier one's value paired in this pass called "changed with no partner"
    st = judge(["mul.lo.s32 %r4, %r2, %r4;", "st.global.u32 [%rd1+32], %r5;", "mul.lo.s32 %r5, %r4, %r1;",
                "@%p1 add.u32 %r6, %r4, %r5;", "@%p1 mul.lo.s32 %r1, %r2, %r6;",
                "st.global.u32 [%rd1+16], %r4;"],
               ["mul.lo.s32 %r4, %r1, %r4;", "st.global.u32 [%rd1+32], %r3;", "mul.lo.s32 %r3, %r4, %r5;",
                "@%p1 add.u32 %r6, %r4, %r3;", "st.global.u32 [%rd1+16], %r4;",
                "@%p1 mul.lo.s32 %r5, %r1, %r6;"])[3]
    expect(isinstance(st, dict), f"two registers paired through each other's values prove: {st}")
    t = Terms()
    add_s, add_u = ("add.s32", 0), ("add.u32", 0)
    leaf1, leaf7 = t.mk("in", ("r", "r1")), t.mk("in", ("r", "r7"))
    expect(not unify(t, Binder(), t.mk("op", add_s, (leaf1,)), t.mk("op", add_u, (leaf1,))),
           "two terms that differ in a scalar do not unify")
    expect(not unify(t, Binder(), t.mk("op", add_s, (leaf1,)), t.mk("op", ("add.s32", 1), (leaf1,))),
           "nor in the destination slot")
    expect(unify(t, Binder(), t.mk("op", add_s, (leaf1,)), t.mk("op", add_s, (leaf7,))),
           "two terms equal up to the name at a leaf do")
    # --- the certificate is checked on its own: a search that returns a wrong pairing is caught
    def certified(base_lines, new_lines, pairing):
        saved = globals()["find_renaming"]
        globals()["find_renaming"] = pairing
        bt, nt = entry_text(base_lines), entry_text(new_lines)
        try:
            prove_value_equal(body(bt, "t"), body(nt, "t"), "t", "t", {}, {})
        except Unproved as e:
            return str(e)
        finally:
            globals()["find_renaming"] = saved
        return None

    def reversed_pairing(terms, ra, rb, regs_a, regs_b):
        bd = Binder()
        for c in {cls_of(r) for r in regs_a}:
            for r, r2 in zip(sorted(r for r in regs_a if cls_of(r) == c),
                             sorted((r for r in regs_b if cls_of(r) == c), reverse=True)):
                bd.bind(r, r2)
        return bd, 0

    def by_name(terms, ra, rb, regs_a, regs_b):
        bd = Binder()
        for r in regs_a:
            bd.bind(r, r)
        return bd, 0
    msg = certified(base, head, reversed_pairing)
    expect(msg is not None and "certificate" in msg, f"a wrong pairing fails the certificate: {msg}")
    swap_a = pre + ["st.global.u32 [%rd1+8], %r1;", "st.global.u32 [%rd1+12], %r2;", "ret;"]
    swap_b = pre + ["st.global.u32 [%rd1+8], %r2;", "st.global.u32 [%rd1+12], %r1;", "ret;"]
    msg = certified(swap_a, swap_b, by_name)
    expect(msg is not None and "certificate: block" in msg and "anchor" in msg,
           f"a wrong pairing fails the anchor check alone: {msg}")
    chain = ["ld.param.u64 %rd1, [t_param_0];", "ld.global.u32 %r1, [%rd1];"]
    msg = certified(chain + ["add.s32 %r3, %r1, 1;", "add.s32 %r4, %r3, 2;", "ret;"],
                    chain + ["add.s32 %r3, %r1, 2;", "add.s32 %r4, %r3, 1;", "ret;"], by_name)
    expect(msg is not None and "certificate: block" in msg and "holding another value" in msg,
           f"a wrong pairing fails the exit-state check: {msg}")
    # --- FAIL-first mutants of the real pair
    sw = list(rh)
    i = sw.index("st.global.b32 [%rd1], %r6742;")
    sw[i], sw[i + 1] = "st.global.b32 [%rd1], %r6746;", "st.global.b32 [%rd1+4], %r6742;"
    never(rb, sw, "real pair: two stored registers exchanged")
    imm = list(rh)
    imm[2] = "mov.b32 %r6741, 1;"
    never(rb, imm, "real pair: the integer zero changed to one", "different lines")
    up = list(rh)
    up.insert(15, up.pop(len(REAL_HEAD_JOIN) - 2))
    never(rb, up, "real pair: the branch moved above five copies")
    across = [l for l in rh if l != "mov.b32 %r6757, %r6741;"]
    across.insert(across.index("$L__BB178_40:") + 1, "mov.b32 %r6757, %r6741;")
    never(rb, across, "real pair: a copy moved across the label")
    dep = list(rh)
    i = dep.index("add.s32 %r6742, %r6742, 1;")
    dep[i], dep[i + 4] = dep[i + 4], dep[i]
    never(rb, dep, "real pair: a store above the add it stores")
    # --- the soundness review's holes, one pair each (each fails on the tool without its fix)
    st_r5 = ["st.global.u32 [%rd1+8], %r5;", "ret;"]
    # an octal literal is not the decimal of its digits: PTX reads `010` as 8
    never(pre + ["mov.u32 %r5, 010;", "mov.u32 %r6, 10;"] + st_r5,
          pre + ["mov.u32 %r5, 10;", "mov.u32 %r6, 010;"] + st_r5,
          "review H1: an octal 010 and a decimal 10 exchanged between two registers", "does not hold")
    same(pre + ["mov.u32 %r5, 0x10;", "mov.u32 %r6, 16;"] + st_r5,
         pre + ["mov.u32 %r5, 16;", "mov.u32 %r6, 0x10;"] + st_r5,
         "review H1: a hex 0x10 and a decimal 16 are one value")
    # a second instruction on a line is read as operands of the first, and its store would leave the anchors
    hidden_a = ["ld.global.u32 %r5, [%rd1+16];", "add.s32 %r6, %r2, 1; st.global.u32 [%rd1+16], %r2;",
                "st.global.u32 [%rd1+20], %r5;", "ret;"]
    hidden_b = [hidden_a[1], hidden_a[0]] + hidden_a[2:]
    never(pre + hidden_a, pre + hidden_b,
          "review H2: a store hidden after a pure instruction on one line moved across the load of its word",
          "two instructions on one line")
    # an integer literal on a float move is not the bits of the float literal of that number
    fmov = ["st.global.f32 [%rd1+8], %f5;", "ret;"]
    never(pre + ["mov.f32 %f5, 1;", "mov.f32 %f6, 0f00000001;"] + fmov,
          pre + ["mov.f32 %f6, 1;", "mov.f32 %f5, 0f00000001;"] + fmov,
          "review H3: an integer literal on a float move read as the bits 0f00000001", "does not hold")
    same(pre + ["mov.f32 %f5, 1;", "mov.f32 %f6, 2;", "st.global.f32 [%rd1+8], %f5;",
                "st.global.f32 [%rd1+12], %f6;", "ret;"],
         pre + ["mov.f32 %f6, 2;", "mov.f32 %f5, 1;", "st.global.f32 [%rd1+8], %f5;",
                "st.global.f32 [%rd1+12], %f6;", "ret;"],
         "review H3: the same integer literals on float moves in the other order still prove")
    # a reorder that passes the opcode screen is `reordered-only` with exit 0 only when the value proof holds too
    def screened_but_unproved(base, new, what, why):
        verdict, fails, out, st = judge(base, new)
        expect(verdict == "reordered-only" and fails and "pass the opcode screen" in out
               and "every one independent" not in out,
               f"{what}: want reordered-only refused with the screen passed and the proof not, got {verdict} "
               f"fails={fails} {out!r}")
        expect(isinstance(st, str) and why in st, f"{what}: the proof should fail with {why!r}, got {st!r}")
    named = ["mov.u32 %r5, 0;", "mov.u32 %r6, 0;", "mov.u32 %r7, 0;"]
    clk = ["sub.s32 %r7, %r6, %r5;", "st.global.u32 [%rd1+16], %r7;", "ret;"]
    screened_but_unproved(
        named + ["mov.u32 %r5, %clock;", "mov.u32 %r6, %clock;"] + clk,
        named + ["mov.u32 %r6, %clock;", "mov.u32 %r5, %clock;"] + clk,
        "review H4: two reads of %clock swapped", "reads a value")
    cry = ["st.global.u32 [%rd1+16], %r6;", "ret;"]
    loaded = pre + ["ld.global.u32 %r3, [%rd1+8];", "ld.global.u32 %r4, [%rd1+12];"] + named
    screened_but_unproved(
        loaded + ["add.cc.u32 %r5, %r1, %r2;", "addc.u32 %r6, %r3, %r3;", "add.cc.u32 %r7, %r4, %r4;"] + cry,
        loaded + ["add.cc.u32 %r5, %r1, %r2;", "add.cc.u32 %r7, %r4, %r4;", "addc.u32 %r6, %r3, %r3;"] + cry,
        "review H4: an addc moved past the add.cc that is not the one it reads", "anchor")
    two = ["mov.b64 %rd2, %rd1;", "mov.b64 %rd3, %rd1;"] + named
    both = ["add.s32 %r5, %r1, 1; st.global.u32 [%rd2+16], %r1;",
            "add.s32 %r6, %r2, 1; st.global.u32 [%rd3+16], %r2;"]
    screened_but_unproved(pre + two + both, pre + two + both[::-1],
                          "review H4: two lines hiding stores to one word through two copies of its address swapped",
                          "two instructions on one line")
    spl = ["mov.b64 %SPL, __local_depot0;", "add.u64 %rd2, %SPL, 0;", "st.local.u32 [%rd2], %r1;",
           "ld.local.u32 %r5, [%rd2];", "st.global.u32 [%rd1+16], %r5;", "ret;"]
    screened_but_unproved(pre + spl, pre + [spl[1], spl[0]] + spl[2:],
                          "review H4: the frame pointer %SPL read before it is set", "anchor")
    # ... and the passing case stays reordered-only with exit 0
    both_named = pre + ["mov.u32 %r3, 0;", "mov.u32 %r4, 0;"]
    verdict, fails, out, st = judge(both_named + ["add.s32 %r3, %r1, 1;", "mul.lo.s32 %r4, %r2, 3;"] + post[:2] + ["ret;"],
                                    both_named + ["mul.lo.s32 %r4, %r2, 3;", "add.s32 %r3, %r1, 1;"] + post[:2] + ["ret;"])
    expect(verdict == "reordered-only" and not fails and "every one independent" in out and isinstance(st, dict),
           f"two independent pure lines swapped: reordered-only, exit 0, proof holding: {verdict} {out!r}")
    expect(const_bits("mov.u32", "r", ["010"]) is None and const_bits("mov.u32", "r", ["00"]) is None
           and const_bits("mov.u32", "r", ["0"]) == ("r", 32, 0)
           and const_bits("mov.u32", "r", ["0x10"]) == const_bits("mov.u32", "r", ["16"]),
           "a decimal is 0 or nonzero-led, a hex is 0x…, an octal is no constant")
    expect(const_bits("mov.f32", "f", ["1"]) is None and const_bits("mov.f16", "h", ["1"]) is None
           and const_bits("mov.f32", "f", ["0f3F800000"]) == ("f", 32, 0x3F800000)
           and const_bits("mov.b32", "r", ["0f3F800000"]) == ("r", 32, 0x3F800000)
           and const_bits("mov.b32", "r", ["1"]) == ("r", 32, 1),
           "an integer literal on a float move is no constant; a hex float and an integer on a bit move are")
    expect(const_bits("mov.u32", "r", ["0f3F800000"]) is None and const_bits("mov.s32", "r", ["0f3F800000"]) is None
           and const_bits("mov.u64", "rd", ["0d3FF0000000000000"]) is None
           and const_bits("mov.b64", "rd", ["0d3FF0000000000000"]) == ("rd", 64, 0x3FF0000000000000),
           "a hex-float literal on a signed or unsigned move is no constant")
    # --- the bijection itself
    from_b = Binder()
    expect(from_b.bind("r1", "r2") and from_b.bind("r1", "r2"), "a pair binds, and again")
    expect(not from_b.bind("r3", "r2"), "two base registers on one new register are refused")
    expect(not from_b.bind("r1", "r5"), "one base register on two new registers is refused")
    expect(not from_b.bind("r4", "rd2"), "a pair of two classes is refused")
    mark = len(from_b.trail)
    expect(from_b.bind("r6", "r7") and not from_b.bind("r6", "r8"), "a second binding of r6")
    from_b.undo(mark)
    expect(from_b.bind("r6", "r8") and "r7" not in from_b.g, "undo frees both names")
    expect(const_bits("mov.b32", "r", ["0f00000000"]) == const_bits("mov.u32", "r", ["0"]),
           "0f00000000 and 0 are one 32-bit constant")
    expect(const_bits("mov.b32", "r", ["0f3F800000"]) != const_bits("mov.b32", "r", ["1"]),
           "1.0f and 1 are not")
    expect(const_bits("mov.b32", "r", ["-", "1"]) == ("r", 32, 0xFFFFFFFF), "-1 wraps to the width")
    expect(const_bits("mov.b64", "rd", ["0f00000000"]) is None, "a 32-bit float literal on a 64-bit move")


def self_test():
    """The declaration rule and the scan comparison against small cases; prints one line."""
    import contextlib
    import io
    import os
    import tempfile
    fails = []

    def expect(ok, what):
        if not ok:
            fails.append(what)

    def canons(text):
        d = declarations(text, "self")
        return [canon(body(text, e), e, d) for e in entries(text)]

    def refused(fn):
        try:
            fn()
        except Refused as e:
            return str(e)
        return None

    plain = canons(self_module())
    hashed = ("__device_global_0123456789abcdef_4", "__shared_mem_0123456789abcdef_2",
              "__shared_mem_0123456789abcdef_3", "__shared_mem_0123456789abcdef_5",
              "__dynamic_smem_0123456789abcdef_alpha")
    expect(canons(self_module(hashed)) == plain, "a hashed name canonicalizes like the plain one")
    expect(any(l == "st.shared.b32 [__shared_mem_<.visible .shared .align 4 .b8 [256]>#2], %r#1;"
               for l in plain[0]), f"the signature form: {plain[0]}")
    expect(any("[8] = #187158eaeba8f101>#1+4]" in l for l in plain[0]),
           "the initializer digest is ptx.rs's FNV-1a 64")
    renumbered = ("__device_global_0", "__shared_mem_9", "__shared_mem_1", "__shared_mem_2",
                  "__dynamic_smem_alpha")
    expect(canons(self_module(renumbered, (4, 3, 0, 2, 1))) == plain,
           "renumbered and reordered declarations canonicalize alike")
    resized = canons(self_module(sizes=(8, 256, 256, 128)))
    expect(resized[0] == plain[0] and resized[1] != plain[1], "a resized declaration moves its user")
    swapped = canons(self_module().replace("__shared_mem_3[256]", "__shared_mem_X[256]")
                     .replace("__shared_mem_5[64]", "__shared_mem_3[64]")
                     .replace("__shared_mem_X[256]", "__shared_mem_5[256]"))
    expect(swapped[0] != plain[0] and swapped[1] != plain[1],
           "declarations swapped under their references move both users")
    merged = canons(self_module().replace("[__shared_mem_3+4]", "[__shared_mem_2+4]"))
    expect(merged[0] != plain[0] and merged[1] == plain[1], "two references merged into one move")
    expect(canons(self_module().replace("_$_str;", "_$_str_$_2;"))[0] != plain[0],
           "a symbol that is not generated is kept by name")
    msg = refused(lambda: canons(self_module().replace(".b8 __shared_mem_5[64]",
                                                       ".b8 __shared_mem_6[64]")))
    expect(msg is not None and "entry beta names __shared_mem_5" in msg, f"undeclared: {msg}")
    msg = refused(lambda: canons(self_module() + ".visible .shared .align 4 .b8 __shared_mem_5[64];\n"))
    expect(msg is not None and "declares __shared_mem_5 twice" in msg, f"twice: {msg}")
    for bad in (".visible .global .align 4 .b8 __device_global_7[8] = {1, 2,",
                ".visible .align 4 .b8 __shared_mem_7[8];",
                ".visible .shared .align 4 .b8 __shared_mem_7[x];"):
        msg = refused(lambda: canons(bad + "\n" + self_module()))
        expect(msg is not None and "_7" in msg, f"unparsed declaration {bad!r}: {msg}")

    row = "{e:<28} 256 no 0 0 1 0 {regs} 0 0 16 {jit}"
    def scan(md5_line, rows, nojit=False, jit=None, regs="12", banner=""):
        """A scan fixture. `nojit` is a `--no-jit` scan (the banner field and the two skipped
        cells); `jit` and `banner` override the two JIT cells and the banner's tail to build a scan
        whose banner and cells disagree."""
        jit = jit or (f"{SKIPPED} {SKIPPED}" if nojit else "12 0")
        banner = banner or (f" jit={SKIPPED}" if nojit else "")
        lines = [f"ptx-scan bin=target/release/x section=.oxart bytes=1 modules=1{banner}",
                 "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill "
                 "blk/SM(static) jit_regs jit_local"]
        lines += [row.format(e=e, regs=regs, jit=jit) for e in rows]
        lines += [md5_line] + [f"{e} {h} 9" for e, h in rows.items()]
        return "\n".join(lines) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        def put(name, text):
            path = os.path.join(tmp, name)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
            return path
        h1, h2 = "0" * 32, "1" * 32
        a = put("a", scan("ptx-scan-md5: method=decl1", {"alpha": h1, "beta": h1}))
        b = put("b", scan("ptx-scan-md5: method=decl1", {"alpha": h1, "beta": h2}))
        old = put("old", scan("ptx-scan-md5:", {"alpha": h1, "beta": h1}))
        other = put("other", scan("ptx-scan-md5: method=decl2", {"alpha": h1, "beta": h1}))
        with contextlib.redirect_stdout(io.StringIO()) as out:
            same = compare_scans(a, a)
            moved = compare_scans(a, b)
        expect(not same and moved and "md5-moved beta" in out.getvalue(),
               f"scans compare by entry: {out.getvalue()!r}")
        expect("md5-equal" not in out.getvalue(), f"the default output names no equal entry: {out.getvalue()!r}")
        with contextlib.redirect_stdout(io.StringIO()) as out:
            moved = compare_scans(a, b, list_equal=True)
            rc_list = main(["--scans", "--list-equal", a, b])
        listed = [l for l in out.getvalue().split("\n") if l.startswith("ptx-canon: md5-equal ")]
        expect(moved and rc_list == 1 and listed == ["ptx-canon: md5-equal alpha"] * 2,
               f"--list-equal names the entries whose digest is equal, and only those: {listed}")
        msg = refused(lambda: main(["--scans", "--list-equal", a]))
        expect(msg is not None and "usage" in msg, f"--list-equal with one scan is a usage error: {msg}")
        for x in (old, other):
            msg = refused(lambda: compare_scans(a, x))
            expect(msg is not None and "do not compare" in msg, f"two methods refused: {msg}")
        # A table and an md5 block that name different entries: a cut or mixed log, refused before any compare.
        cut = put("cut", scan("ptx-scan-md5: method=decl1", {"alpha": h1, "beta": h2}).replace(f"beta {h2} 9\n", ""))
        msg = refused(lambda: compare_scans(a, cut))
        expect(msg is not None and "1 entries in only one of the table and the md5 block (beta)" in msg,
               f"a scan whose md5 block lacks a table entry is refused: {msg}")
        # A note on stderr that lands inside the table (a log of both streams) is skipped, not read as the
        # table's end.
        noted = put("noted", scan("ptx-scan-md5: method=decl1", {"alpha": h1, "beta": h1}).replace(
            "jit_local\n", "jit_local\nptx-scan: entries that call a device function — their rows count "
            "their own body, not the callee's: alpha:1\n", 1))
        with contextlib.redirect_stdout(io.StringIO()):
            msg = refused(lambda: compare_scans(a, noted))
        expect(msg is None, f"a stderr note inside the table is skipped: {msg}")
        failed = put("failed", "ptx-scan bin=target/release/x missing scan=failed\n")
        msg = refused(lambda: compare_scans(a, failed))
        expect(msg is not None and "failed scan" in msg, f"a failed scan refused: {msg}")
        # --no-jit scans: the JIT columns hold a word, not a value, and are never compared
        m = "ptx-scan-md5: method=decl1"
        n1 = put("n1", scan(m, {"alpha": h1, "beta": h1}, nojit=True))
        n2 = put("n2", scan(m, {"alpha": h1, "beta": h1}, nojit=True))
        n_regs = put("n_regs", scan(m, {"alpha": h1, "beta": h1}, nojit=True, regs="13"))
        n_md5 = put("n_md5", scan(m, {"alpha": h1, "beta": h2}, nojit=True))
        n_add = put("n_add", scan(m, {"alpha": h1, "beta": h1, "gamma": h1}, nojit=True))
        f_jit = put("f_jit", scan(m, {"alpha": h1, "beta": h1}, jit="14 8"))
        not_compared = "jit columns not compared (no-jit: "

        def compared(x, y):
            with contextlib.redirect_stdout(io.StringIO()) as o:
                moved = compare_scans(x, y)
            return moved, o.getvalue()

        moved, out = compared(n1, n2)
        expect(not moved and not_compared + "both)" in out and "table rows identical=2 moved=0" in out,
               f"two equal no-jit scans are identical on the other columns and say the JIT columns were "
               f"not compared: moved={moved} {out!r}")
        moved, out = compared(a, n1)
        expect(not moved and not_compared + "new)" in out,
               f"a full base against a no-jit new compares as no-jit: moved={moved} {out!r}")
        moved, out = compared(n1, a)
        expect(not moved and not_compared + "base)" in out,
               f"a no-jit base against a full new compares as no-jit: moved={moved} {out!r}")
        moved, out = compared(f_jit, n1)
        expect(not moved, f"a full scan whose JIT cells differ from a no-jit scan's is not moved: {out!r}")
        moved, out = compared(a, f_jit)
        expect(moved and "row-moved alpha jit_regs:12->14 jit_local:0->8" in out and not_compared not in out,
               f"two full scans still compare the JIT columns: moved={moved} {out!r}")
        moved, out = compared(n1, n_regs)
        expect(moved and "row-moved alpha regs:12->13" in out and "jit_" not in out.split("row-moved", 1)[-1],
               f"a no-jit pair still compares every other column: moved={moved} {out!r}")
        moved, out = compared(n1, n_md5)
        expect(moved and "md5-moved beta" in out, f"a no-jit pair still compares the md5 block: {out!r}")
        moved, out = compared(n1, n_add)
        expect(moved and "gamma only in new" in out, f"a no-jit pair still compares the entry set: {out!r}")
        for what, text, why in (
            ("a skipped cell in the regs column", scan(m, {"alpha": h1}, nojit=True, regs=SKIPPED),
             "only jit_regs and jit_local can be skipped"),
            ("a no-jit banner over numeric JIT cells", scan(m, {"alpha": h1}, banner=f" jit={SKIPPED}"),
             "holds no JIT value"),
            ("a full banner over skipped JIT cells", scan(m, {"alpha": h1}, jit=f"{SKIPPED} {SKIPPED}"),
             "the banner names the JIT"),
            ("a banner jit= the reader does not know", scan(m, {"alpha": h1}, banner=" jit=maybe"),
             "does not parse"),
        ):
            bad = put("bad", text)
            msg = refused(lambda: compare_scans(a, bad))
            expect(msg is not None and why in msg, f"{what} refused by name: {msg}")
            msg = refused(lambda: compare_scans(bad, a))
            expect(msg is not None and why in msg, f"{what} refused by name as the base: {msg}")
        wide = put("wide", scan(m, {"alpha": h1}).replace(" jit_local", " jit_local extra")
                   .replace("16 12 0", "16 12 0 1"))
        msg = refused(lambda: compare_scans(a, wide))
        expect(msg is not None and "other table columns" in msg, f"two tables of other columns refused: {msg}")

        # --all reads nothing it was not given: a file with no entry is refused, and no shared entry or an entry on one
        # side only is a named exit 1 (a deleted or an added kernel is no proof of an unchanged program)
        def entry_module(*names):
            return "".join(f".visible .entry {n}(\n)\n{{\n\tret;\n}}\n" for n in names)

        def run_all(x, y):
            with contextlib.redirect_stdout(io.StringIO()) as o:
                code = main(["--all", x, y])
            return code, o.getvalue()
        ab = put("ab.ptx", entry_module("alpha", "beta"))
        code, out = run_all(ab, ab)
        expect(code == 0 and "2 shared entries: 2 identical" in out and "--all:" not in out,
               f"--all over equal modules is exit 0 and says nothing more: {code} {out!r}")
        for x, y, which in ((ab, put("none.ptx", ""), "none.ptx"), (put("text.ptx", "no entry here\n"), ab, "text.ptx")):
            msg = refused(lambda: run_all(x, y))
            expect(msg is not None and which in msg and "holds no `.visible .entry`" in msg,
                   f"--all over a file with no entry is refused by name ({which}): {msg}")
        code, out = run_all(ab, put("gamma.ptx", entry_module("gamma")))
        expect(code == 1 and "ptx-canon: --all: the two files share no entry: nothing was compared" in out,
               f"--all over modules sharing no entry is exit 1, named: {code} {out!r}")
        for x, y, counts in ((put("abc.ptx", entry_module("alpha", "beta", "gamma")), ab, "1 only in base, 0 only in new"),
                             (ab, put("abc2.ptx", entry_module("alpha", "beta", "gamma")), "0 only in base, 1 only in new")):
            code, out = run_all(x, y)
            expect(code == 1 and f"ptx-canon: --all: entries on one side only ({counts})" in out,
                   f"--all with an entry on one side only is exit 1, named ({counts}): {code} {out!r}")
    value_equal_cases(expect)
    for f in fails:
        print(f"self-test FAIL: {f}", file=sys.stderr)
    print(f"ptx-canon self-test: {'FAIL' if fails else 'ok'} ({len(fails)} failures)")
    return 1 if fails else 0


def main(argv):
    args = list(argv)
    if args == ["--self-test"]:
        return self_test()
    if args[:1] == ["--scans"]:
        rest = args[1:]
        list_equal = rest[:1] == ["--list-equal"]
        if list_equal:
            rest = rest[1:]
        if len(rest) != 2:
            raise Refused("usage: ptx-canon.py --scans [--list-equal] BASE.scan NEW.scan")
        return int(compare_scans(rest[0], rest[1], list_equal))
    context = 3
    if args[:1] == ["-U"]:
        if len(args) < 2 or not args[1].isdigit():
            raise Refused("-U takes a line count")
        context = int(args[1])
        args = args[2:]
    everything = args[:1] == ["--all"]
    if everything:
        args = args[1:]
    if len(args) < 2 or (everything and len(args) != 2) or (not everything and len(args) < 3):
        raise Refused("usage: ptx-canon.py [-U N] BASE.ptx NEW.ptx ENTRY[:NEW_ENTRY]... | "
                      "ptx-canon.py [-U N] --all BASE.ptx NEW.ptx | "
                      "ptx-canon.py --scans [--list-equal] BASE.scan NEW.scan | ptx-canon.py --self-test")
    base_path, new_path = args[0], args[1]
    base_text, new_text = read(base_path), read(new_path)
    base = (base_path, base_text, declarations(base_text, base_path))
    new = (new_path, new_text, declarations(new_text, new_path))
    rc = 0
    if everything:
        in_base, in_new = entries(base_text), entries(new_text)
        for path, found in ((base_path, in_base), (new_path, in_new)):
            if not found:
                raise Refused(f"{path} holds no `.visible .entry`: nothing to compare (an empty or unreadable "
                              "module proves nothing)")
        shared = [n for n in in_base if n in set(in_new)]
        tally = {"identical": 0, "reordered-only": 0, "value-equal": 0, "differs": 0}
        for name in shared:
            verdict, fails = compare(base, new, name, name, context)
            tally[verdict] += 1
            rc |= fails
        only_base = [n for n in in_base if n not in set(in_new)]
        only_new = [n for n in in_new if n not in set(in_base)]
        for n in only_base:
            print(f"ptx-canon: {n} only in base")
        for n in only_new:
            print(f"ptx-canon: {n} only in new")
        print(f"ptx-canon: {len(shared)} shared entries: {tally['identical']} identical, "
              f"{tally['reordered-only']} reordered-only, {tally['value-equal']} value-equal, "
              f"{tally['differs']} differ; "
              f"{len(only_base)} only in base, {len(only_new)} only in new")
        if not shared:
            print("ptx-canon: --all: the two files share no entry: nothing was compared")
            rc = 1
        if only_base or only_new:
            print(f"ptx-canon: --all: entries on one side only ({len(only_base)} only in base, {len(only_new)} only in "
                  "new): a deleted or an added kernel is not shown to be the same program")
            rc = 1
        return rc
    for spec in args[2:]:
        base_name, _, new_name = spec.partition(":")
        _, fails = compare(base, new, base_name, new_name or base_name, context)
        rc |= fails
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refused as e:
        print(f"ptx-canon: {e}", file=sys.stderr)
        sys.exit(2)
