#!/usr/bin/env python3
"""Find the truncated uniform address of the CUDA 13.0 PTX compiler in compiled GPU code.

The failure it blocks: on sm_120 and sm_121, ptxas 13.0 (the JIT of the R580 drivers) lowers a parameter pointer plus a
shifted u32 that both sit in uniform registers to a carry-less `ULEA URd, URo, URp, k` and then reads `URd:URd+1` as the
address with `URd+1` left at `URZ` or at a value that is not the pointer's high word. The kernel then reads or writes at
the address's low 32 bits (a fault, or silently another buffer). ptxas 13.1 and later emit `ULEA … UPn` +
`ULEA.HI.X`. The toolchain ledger's row 55 (docs/upstream/nvlabs-ledger.md) has the SASS of both.

A suspect is a carry-less `ULEA URd, URa, URb, k` with `URb` the low word of a 64-bit constant-bank load
(`LDCU.64 URb, c[...]`: a pointer parameter), whose `URd` is next read as a 64-bit operand, and whose `URd+1` was last
written by an instruction that does not read `URb+1`. Instruction order stands for control flow, so a suspect names a
site to read; tools/oxart-pack.sh runs this over every cubin a pack ships and fails the pack on any.

usage:
  ulea-scan.py SASS.txt                 the text of `cuobjdump -sass`; one line a suspect, rc 1 if any
  ulea-scan.py --cubins DIR [--cuobjdump PATH]
                                        every DIR/*.cubin through cuobjdump -sass (PATH, else cuobjdump on PATH)
  ulea-scan.py --section FILE [--cuobjdump PATH]
                                        every cubin inside FILE: a packed binary's `.oxart` as objcopy
                                        --dump-section writes it, or what oxart_ptxas / oxart_jit --cubin wrote
  ulea-scan.py --self-test
exit: 0 clean, 1 suspects, 64 usage, 69 a tool or file missing.
"""
import os
import re
import struct
import subprocess
import sys
import tempfile

ULEA = re.compile(r"\bULEA UR(\d+), UR(\d+), UR(\d+), (0x[0-9a-f]+) ;")
INSN = re.compile(r"^\s*/\*[0-9a-f]{4,}\*/")


def _dest(line, reg):
    """Whether `line`'s instruction writes UR`reg` as its first operand (a predicate may sit before it)."""
    return re.search(r"^\s*/\*\w+\*/\s+(@\S+\s+)?\S+ (UP\d, )?UR%d\b" % reg, line) is not None


def _body(line):
    return line.split(";")[0]


def _clip(line):
    return re.sub(r"\s+", " ", _body(line).strip()) if line else "-"


def scan_function(func, lines):
    """The suspects of one function's instruction lines, each `func | ulea | high-word writer | 64-bit use`."""
    out = []
    for i, line in enumerate(lines):
        m = ULEA.search(line)
        if not m:
            continue
        d, b = int(m.group(1)), int(m.group(3))
        src = next((lines[j] for j in range(i - 1, -1, -1) if _dest(lines[j], b)), None)
        if src is None or not re.search(r"LDCU\.64 UR%d, c\[" % b, src):
            continue
        use = None
        for k in range(i + 1, len(lines)):
            if _dest(lines[k], d):
                break
            body = _body(lines[k])
            if re.search(r"\bUR%d\b" % d, body) and (re.search(r"\.64\b", body) or "UR%d.64" % d in body):
                use = k
                break
        if use is None:
            continue
        hw = next((lines[k] for k in range(use - 1, -1, -1) if _dest(lines[k], d + 1)), None)
        if hw is not None and re.search(r"\bUR%d\b" % (b + 1), _body(hw).split(",", 1)[-1]):
            continue
        out.append("%s | %s | %s | %s" % (func, _clip(line), _clip(hw), _clip(lines[use])))
    return out


def scan_text(text):
    out, func, lines = [], None, []
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if func:
                out += scan_function(func, lines)
            func, lines = m.group(1), []
        elif INSN.match(line):
            lines.append(line)
    if func:
        out += scan_function(func, lines)
    return out


def scan_cubins(directory, cuobjdump):
    cubins = sorted(f for f in os.listdir(directory) if f.endswith(".cubin"))
    if not cubins:
        print("ulea-scan: no .cubin in %s" % directory, file=sys.stderr)
        return None
    out = []
    for name in cubins:
        try:
            sass = subprocess.run([cuobjdump, "-sass", os.path.join(directory, name)], capture_output=True, text=True)
        except FileNotFoundError:
            print("ulea-scan: %s is not on PATH (the CUDA toolkit's cuobjdump)" % cuobjdump, file=sys.stderr)
            return None
        if sass.returncode != 0:
            print("ulea-scan: %s -sass %s failed: %s" % (cuobjdump, name, sass.stderr.strip()), file=sys.stderr)
            return None
        found = scan_text(sass.stdout)
        print("ulea-scan: %s functions=%d suspects=%d" % (name, sass.stdout.count("Function : "), len(found)))
        out += ["%s: %s" % (name, f) for f in found]
    return out


EM_CUDA = 190


def _elf_end(blob, at):
    """The length of the little-endian ELF64 CUDA object at `blob[at:]`: the furthest of its header tables and section
    bodies. None when the bytes there are no such object or it runs past the blob."""
    head = blob[at : at + 64]
    if len(head) < 64 or head[4] != 2 or head[5] != 1 or struct.unpack_from("<H", head, 0x12)[0] != EM_CUDA:
        return None
    phoff, shoff = struct.unpack_from("<QQ", head, 0x20)
    phentsize, phnum, shentsize, shnum = struct.unpack_from("<HHHH", head, 0x36)
    if shnum == 0 or shentsize < 64 or at + shoff + shentsize * shnum > len(blob):
        return None
    end = max(64, shoff + shentsize * shnum, phoff + phentsize * phnum)
    for i in range(shnum):
        sh = at + shoff + i * shentsize
        sh_type = struct.unpack_from("<I", blob, sh + 4)[0]
        offset, size = struct.unpack_from("<QQ", blob, sh + 0x18)
        if sh_type != 8:  # SHT_NOBITS holds no bytes in the file
            end = max(end, offset + size)
    return end if at + end <= len(blob) else None


def carve_cubins(blob):
    """The CUDA ELF objects inside `blob` (a section of artifact bundles), each cut at its furthest byte."""
    out, at = [], blob.find(b"\x7fELF")
    while at >= 0:
        end = _elf_end(blob, at)
        if end is None:
            at = blob.find(b"\x7fELF", at + 4)
            continue
        out.append(blob[at : at + end])
        at = blob.find(b"\x7fELF", at + end)
    return out


def scan_section(path, cuobjdump):
    try:
        blob = open(path, "rb").read()
    except OSError as e:
        print("ulea-scan: %s" % e, file=sys.stderr)
        return None
    cubins = carve_cubins(blob)
    if not cubins:
        print("ulea-scan: no cubin in %s (a PTX-only section is not packed)" % path, file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory(prefix="ulea-scan.") as d:
        for n, c in enumerate(cubins, 1):
            with open(os.path.join(d, "%d.cubin" % n), "wb") as f:
                f.write(c)
        return scan_cubins(d, cuobjdump)


# ptxas 13.0 and 13.3 on conv_res at sm_121a (cuobjdump -sass, encodings cut), and the high word of
# rel_prefill_h128_g3_ring under 13.0: the ledger row's lines.
RED_URZ = """
\t\tFunction : conv_res
        /*03c0*/                   LDCU.64 UR10, c[0x0][0x400] ;
        /*0460*/                   UMOV UR9, URZ ;
        /*04b0*/                   ULEA UR8, UR8, UR10, 0x2 ;
        /*0590*/                   IADD.64 R22, R22, UR8 ;
"""
RED_OTHER = """
\t\tFunction : rel_prefill_h128_g3_ring
        /*4750*/                   ULEA UR19, UR13, UR6, 0x18 ;
        /*4790*/                   LDCU.64 UR24, c[0x0][0x3a0] ;
        /*55c0*/                   ULEA UR18, UR10, UR24, 0x3 ;
        /*5d20*/                   IADD.64 R74, R72.reuse, UR18 ;
"""
GREEN_CARRY = """
\t\tFunction : conv_res
        /*03d0*/                   LDCU.64 UR10, c[0x0][0x400] ;
        /*0480*/                   UMOV UR9, URZ ;
        /*04c0*/                   ULEA UR8, UP0, UR7, UR13, 0x2 ;
        /*0550*/                   ULEA.HI.X UR9, UR7, UR11, UR9, 0x2, UP0 ;
        /*0590*/                   IADD.64 R22, R22, UR8 ;
"""
GREEN_HIGH = """
\t\tFunction : f
        /*0010*/                   LDCU.64 UR10, c[0x0][0x400] ;
        /*0020*/                   ULEA UR8, UR8, UR10, 0x2 ;
        /*0030*/                   UIADD3.X UR9, UPT, UPT, URZ, UR11, URZ, UP0, !UPT ;
        /*0040*/                   IADD.64 R22, R22, UR8 ;
"""
GREEN_32 = """
\t\tFunction : g
        /*0010*/                   LDCU.64 UR10, c[0x0][0x400] ;
        /*0020*/                   ULEA UR8, UR8, UR10, 0x2 ;
        /*0030*/                   IADD3 R2, PT, PT, R2, UR8, RZ ;
"""


def self_test():
    cases = [
        ("13.0 conv_res: high word URZ", RED_URZ, 1),
        ("13.0 rel_prefill: high word from another ULEA", RED_OTHER, 1),
        ("13.3 conv_res: ULEA.HI.X carries the pointer's high word", GREEN_CARRY, 0),
        ("the high word written from the pointer's", GREEN_HIGH, 0),
        ("a 32-bit use only", GREEN_32, 0),
    ]
    # A CUDA ELF of a header, one 16-byte section body past its section header table, and that table's one row.
    elf = bytearray(64 + 64 + 16)
    elf[0:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", elf, 0x12, EM_CUDA)
    struct.pack_into("<Q", elf, 0x28, 64)
    struct.pack_into("<HHHH", elf, 0x36, 56, 0, 64, 1)
    struct.pack_into("<IQQ", elf, 64 + 4, 1, 0, 0)
    struct.pack_into("<QQ", elf, 64 + 0x18, 128, 16)
    elf = bytes(elf)
    host = bytearray(elf)
    struct.pack_into("<H", host, 0x12, 62)
    carved = carve_cubins(b"head" + elf + b"pad\x7fELFx" + bytes(host) + elf + b"tail")
    bad = 0 if carved == [elf, elf] else 1
    print("%s two CUDA ELF objects carved whole from a bundle section, a host ELF skipped: %d" % ("ok  " if not bad else "FAIL", len(carved)))
    for name, text, want in cases:
        got = len(scan_text(text))
        ok = got == want
        bad += not ok
        print("%s %s: %d suspect(s), want %d" % ("ok  " if ok else "FAIL", name, got, want))
    print("ulea-scan self-test: %d case(s), %d failed" % (len(cases) + 1, bad))
    return 1 if bad else 0


def main(argv):
    if argv == ["--self-test"]:
        return self_test()
    if argv[:1] in (["--cubins"], ["--section"]) and len(argv) in (2, 4) and (len(argv) == 2 or argv[2] == "--cuobjdump"):
        tool = argv[3] if len(argv) == 4 else "cuobjdump"
        found = scan_cubins(argv[1], tool) if argv[0] == "--cubins" else scan_section(argv[1], tool)
        if found is None:
            return 69
    elif len(argv) == 1 and not argv[0].startswith("-"):
        try:
            text = open(argv[0]).read()
        except OSError as e:
            print("ulea-scan: %s" % e, file=sys.stderr)
            return 69
        found = scan_text(text)
    else:
        print(__doc__.split("usage:")[1].split("exit:")[0].rstrip(), file=sys.stderr)
        return 64
    for f in found:
        print(f)
    print("ulea-scan: suspects=%d" % len(found))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
