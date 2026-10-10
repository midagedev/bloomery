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
    asm-census.py --self-test             canned listings

Why legacy encodings matter here: on Zen 3 a legacy-SSE instruction run while the ymm upper halves are
dirty takes a dispatch fault (perf: fp_disp_faults.ymm_spill_fault, then ymm_fill_fault). Under
target-cpu=znver3 LLVM emits VEX for everything that has a VEX form, so what is left is SSE4a, which
has none — the reason .cargo/config.toml turns sse4a off (tools/check-rustflags.sh holds it).
"""

import collections
import re
import subprocess
import sys

FUNC = re.compile(r"^[0-9a-f]+ <(.*)>:$")
SSE4A = {"extrq", "insertq", "movntss", "movntsd"}


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
        m = re.match(r"j\w+\s+([0-9a-f]+) <", ins)
        if m:
            t = int(m.group(1), 16)
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
    print("asm-census self-test: ok (legacy and sse4a lists, the loop census, the normalized diff, a loopless function)")
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
    except (OSError, KeyError, subprocess.CalledProcessError) as e:
        print(f"asm-census: {e}")
        return 1
    print(__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    sys.exit(main(sys.argv))
