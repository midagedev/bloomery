#!/usr/bin/env python3
"""Are two builds of a PTX entry the same program? The canonical diff of an entry's body.

`tools/ptx-scan.sh` prints, per entry, the md5 of its body with every register, label and generated
symbol number deleted (`ptx::normalize` in crates/gpu-gates/src/ptx.rs). That digest is stable
across the backend's free numbering, and blind to which register an operand names: a rewired
operand or a retargeted branch keeps it. This script renumbers instead of deleting: each register
class, the block labels and each generated-symbol stem are numbered in order of first appearance
inside the entry, so two bodies compare equal only when they are the same instruction stream over
the same dataflow, whatever absolute numbers the backend chose. When a deletion moves a live
entry's md5, this says whether the program moved with it.

  ptx-canon.py [-U N] BASE.ptx NEW.ptx ENTRY[:NEW_ENTRY]...
  ptx-canon.py [-U N] --all BASE.ptx NEW.ptx

For each entry it prints the unified diff of the two canonical bodies (N lines of context, default
3) and one verdict line:

  ptx-canon: <entry> identical
  ptx-canon: <entry> reordered-only (same line multiset); <whether the moved lines are independent>
  ptx-canon: <entry> differs (<n> changed lines)

ENTRY:NEW_ENTRY compares a renamed entry. --all compares every entry the two files share and lists
the ones only one file has (`only in base`, `only in new`: a deleted or an added kernel), then a
count line. Exit 0 when every compared entry is identical or reordered-only with every swapped pair
independent (below), 1 when one differs or its reorder is not shown independent, 2 on a usage error
or a named entry that a file does not hold.

`reordered-only` is a same-multiset result, not by itself a proof. Its line therefore also checks
every pair of lines whose relative order changed: the pair is independent when the two lines share
no register and neither is a memory, synchronisation or control line (anything but the pure
arithmetic, move, convert, compare and select opcodes listed in PURE: a load, a store, an atomic,
a barrier, a shuffle, a branch, a call, a label, a directive, and every opcode not in PURE). All
pairs independent: no swapped line touches a register the other touches, and neither touches
memory or control flow, so the swap cannot change a value. Otherwise the line says `read the diff`
and the exit code is 1: a same-multiset reorder can still move a value. A reorder that
changes the order in which one register class first appears renumbers everything after it and
reads `differs`: the verdict errs toward `differs`, never toward `identical`.

The body is cut as `ptx::body` cuts it: from just past `.visible .entry <name>(` to the line of
the next `.entry` or `.func` directive. The body of a device function the entry calls lies outside
it, as it does for the scan's digest.

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

# The register classes the backend numbers (`ptx::normalize`'s REG_CLASSES).
REG_CLASSES = {"r", "rd", "rs", "f", "fd", "p", "h", "hh", "rq"}
# Generated symbols whose number means nothing (`ptx::normalize`'s NUMBERED_STEMS).
NUMBERED_STEMS = ("__shared_mem_", "__local_depot", "__device_global_")
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


def entries(text):
    """Every `.visible .entry` name in the text, in declaration order."""
    return re.findall(r"\.visible \.entry ([A-Za-z0-9_$]+)\(", text)


def canon(lines, name):
    """The body's lines with comments cut, whitespace runs collapsed, empty lines dropped, and every
    numbered register, block label and generated symbol renumbered by first appearance."""
    seen = {}
    count = {}

    def number(kind, tok):
        key = (kind, tok)
        if key not in seen:
            count[kind] = count.get(kind, 0) + 1
            seen[key] = count[kind]
        return seen[key]

    out = []
    for raw in lines:
        code = raw.split("//", 1)[0]
        code = re.sub(r"[ \t\r]+", " ", code).strip()
        if not code:
            continue
        res = []
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
                    res.append(f"%{cls}#{number(cls, cls + digits)}")
                    i = j + len(digits)
                    continue
                decl = re.match(r"<\d+>", code[j:]) if cls in REG_CLASSES else None
                if decl:
                    res.append(f"%{cls}<N>")
                    i = j + decl.end()
                    continue
                res.append(f"%{cls}")
                i = j
                continue
            m = IDENT.match(code, i)
            if m and (i == 0 or not (code[i - 1].isalnum() or code[i - 1] in "_$")):
                tok = m.group(0)
                i = m.end()
                if LABEL.match(tok):
                    res.append(f"$L#{number('$L', tok)}")
                    continue
                stem = next((s for s in NUMBERED_STEMS
                             if tok.startswith(s) and tok[len(s):].isdigit()), None)
                if stem:
                    res.append(f"{stem}#{number(stem, tok)}")
                elif tok == name:
                    res.append("ENTRY")
                elif tok.startswith(name + "_param_"):
                    res.append("ENTRY" + tok[len(name):])
                else:
                    res.append(tok)
                continue
            res.append(c)
            i += 1
        out.append("".join(res))
    return out


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


def compare(base_text, new_text, base_name, new_name, base_path, new_path, context):
    """Print one entry's diff and verdict line; return the verdict word and whether it fails the
    run (`differs`, or a reorder not shown independent)."""
    ba, bb = body(base_text, base_name), body(new_text, new_name)
    if ba is None:
        raise Refused(f"{base_path} holds no entry {base_name}")
    if bb is None:
        raise Refused(f"{new_path} holds no entry {new_name}")
    a, b = canon(ba, base_name), canon(bb, new_name)
    label = base_name if base_name == new_name else f"{base_name}:{new_name}"
    if a == b:
        print(f"ptx-canon: {label} identical ({len(a)} lines)")
        return "identical", False
    diff = list(difflib.unified_diff(a, b, f"{base_path}:{base_name}", f"{new_path}:{new_name}",
                                     n=context, lineterm=""))
    for line in diff:
        print(line)
    if sorted(a) == sorted(b):
        pairs, width = moved_pairs(a, b)
        if pairs is None:
            bad = True
            note = f"a window of {width} lines, too wide to check pairs; read the diff"
        else:
            dependent = [p for p in pairs if not independent(*p)]
            bad = bool(dependent)
            note = (f"{len(pairs)} swapped pair(s) in a {width}-line window, every one independent "
                    "(pure opcodes, no shared register)" if not bad else
                    f"{len(dependent)} of {len(pairs)} swapped pair(s) share a register or hold a "
                    "memory, synchronisation or control line; read the diff (first: "
                    f"{dependent[0][0]!r} / {dependent[0][1]!r})")
        print(f"ptx-canon: {label} reordered-only (same line multiset); {note}")
        return "reordered-only", bad
    changed = sum(1 for d in diff if d[:1] in "+-" and d[:3] not in ("+++", "---"))
    print(f"ptx-canon: {label} differs ({changed} changed lines)")
    return "differs", True


def read(path):
    try:
        with open(path, encoding="utf-8", errors="strict") as f:
            return f.read()
    except OSError as e:
        raise Refused(f"cannot read {path}: {e}") from e


def main(argv):
    args = list(argv)
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
                      "ptx-canon.py [-U N] --all BASE.ptx NEW.ptx")
    base_path, new_path = args[0], args[1]
    base_text, new_text = read(base_path), read(new_path)
    rc = 0
    if everything:
        in_base, in_new = entries(base_text), entries(new_text)
        shared = [n for n in in_base if n in set(in_new)]
        tally = {"identical": 0, "reordered-only": 0, "differs": 0}
        for name in shared:
            verdict, fails = compare(base_text, new_text, name, name, base_path, new_path, context)
            tally[verdict] += 1
            rc |= fails
        only_base = [n for n in in_base if n not in set(in_new)]
        only_new = [n for n in in_new if n not in set(in_base)]
        for n in only_base:
            print(f"ptx-canon: {n} only in base")
        for n in only_new:
            print(f"ptx-canon: {n} only in new")
        print(f"ptx-canon: {len(shared)} shared entries: {tally['identical']} identical, "
              f"{tally['reordered-only']} reordered-only, {tally['differs']} differ; "
              f"{len(only_base)} only in base, {len(only_new)} only in new")
        return rc
    for spec in args[2:]:
        base_name, _, new_name = spec.partition(":")
        _, fails = compare(base_text, new_text, base_name, new_name or base_name, base_path,
                           new_path, context)
        rc |= fails
    return rc


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Refused as e:
        print(f"ptx-canon: {e}", file=sys.stderr)
        sys.exit(2)
