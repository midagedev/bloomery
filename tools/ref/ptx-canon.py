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
  ptx-canon.py --scans BASE.scan NEW.scan
  ptx-canon.py --self-test

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

--scans compares two saved `tools/ptx-scan.sh` outputs (stdout, or stdout and stderr together): the
table rows by entry, column by column, and the md5 block by entry. Two md5 blocks compare only under
one digest method — the `method=` of their `ptx-scan-md5:` line; a block with no method comes from an
older rule set — and a pair whose methods differ is refused by name (exit 2): rescan the older binary
with this tree's tools. A failed scan is refused too. Exit 0 when both scans hold the same entries,
every row and every md5 equal, 1 otherwise, with the moved rows and digests listed.

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


def canon(lines, name, decls):
    """The body's lines with comments cut, whitespace runs collapsed, empty lines dropped, every
    numbered register, block label and local depot renumbered by first appearance, and every
    generated module symbol written as its declaration's signature (`decls`, the module's
    `declarations`) with its order of first appearance."""
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
                gen = next((s for s in GENERATED_STEMS if tok.startswith(s)), None)
                if gen:
                    if tok not in decls:
                        raise Refused(f"entry {name} names {tok}, which its module does not declare")
                    res.append(f"{gen}<{decls[tok]}>#{number('gen', tok)}")
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


def compare(base, new, base_name, new_name, context):
    """Print one entry's diff and verdict line; return the verdict word and whether it fails the
    run (`differs`, or a reorder not shown independent). `base` and `new` are (path, text, decls)."""
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
    return method, header, rows, md5, nojit


def compare_scans(base_path, new_path):
    """Print the entries, rows and digests two scans do not share; return whether any moved."""
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
        for x in (old, other):
            msg = refused(lambda: compare_scans(a, x))
            expect(msg is not None and "do not compare" in msg, f"two methods refused: {msg}")
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
    for f in fails:
        print(f"self-test FAIL: {f}", file=sys.stderr)
    print(f"ptx-canon self-test: {'FAIL' if fails else 'ok'} ({len(fails)} failures)")
    return 1 if fails else 0


def main(argv):
    args = list(argv)
    if args == ["--self-test"]:
        return self_test()
    if args[:1] == ["--scans"]:
        if len(args) != 3:
            raise Refused("usage: ptx-canon.py --scans BASE.scan NEW.scan")
        return int(compare_scans(args[1], args[2]))
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
                      "ptx-canon.py --scans BASE.scan NEW.scan | ptx-canon.py --self-test")
    base_path, new_path = args[0], args[1]
    base_text, new_text = read(base_path), read(new_path)
    base = (base_path, base_text, declarations(base_text, base_path))
    new = (new_path, new_text, declarations(new_text, new_path))
    rc = 0
    if everything:
        in_base, in_new = entries(base_text), entries(new_text)
        shared = [n for n in in_base if n in set(in_new)]
        tally = {"identical": 0, "reordered-only": 0, "differs": 0}
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
              f"{tally['reordered-only']} reordered-only, {tally['differs']} differ; "
              f"{len(only_base)} only in base, {len(only_new)} only in new")
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
