#!/usr/bin/env bash
# PTX scan — read the device code a gate binary carries, as a table. An instrument, not a gate: it
# asserts nothing about a kernel. But a scan that could not read the kernels is a failure, not a
# table — it exits 0 only when every PTX module came out of the container, assembled under ptxas
# and loaded under the driver's JIT (with --no-jit: assembled under ptxas, the JIT not run).
#
# `cargo oxide` puts the bundle's PTX text into the executable's `.oxart` ELF section verbatim.
# So a kernel's compiled shape — whether registers spilled to local (`__local_depot`), how many
# round trips that costs (ld.local/st.local), how wide the block is (`.reqntid`) — reads out of a
# byte scan with no device. Two root causes were found this way: flash's per-lane accumulator
# array spilling to a depot, and rms_norm's launch geometry leaving one resident warp.
#
# The section is an oxide-artifacts container, and only its parser knows where a payload ends: a
# payload is stored with its length and padded to 8 bytes, so no NUL is promised after the PTX: a
# payload of 8k bytes runs straight into the container's next record (for the last payload, the
# entry-symbol table). The extractor
# target/release/oxart_ptx (crates/gpu-gates/src/bin/oxart_ptx.rs, the parser behind
# `bloomery_gpu_gates::ptx`) writes one file per bundle that carries PTX; every column below is
# read from those files and from nothing else in the section.
#
# Asserting is the gates' job. This script only prints a table:
#   gate_p5 (lib `no_local_depot`)  — depot, ld.local and st.local are all 0 for the three flash
#                                     kernels and the two kv_append entries
#   gate_p4 `norm_geometry`         — .reqntid of rms_norm and norm_quant == RMS_THREADS
#   gate_p4 `argmax_geometry`       — .reqntid of argmax_fault and argmax_rows_fault == ARGMAX_THREADS
#   gate_p6 `router_shape`          — the fma floor and no depot for f32_gemv, q8_0_gemv and
#                                     q8_0_gemv_heads, and .reqntid of router_topk and expert_table
#   gate_p6 `q3k_half_decode_shape` — no clz and a hardware f16 convert in both Q3_K entries
#
# Mind the spelling: PTX writes a fused multiply-add as `fma.rn.f32` (and `fma.rm.f32`). There is
# no `fma.f32` anywhere, so counting that silently returns 0 and the kernel reads as clean — here
# and in the gates, the `fma.` prefix is what is counted.
#
# An entry's body is its own text only: it ends at the next `.entry` or `.func` line, so a device
# function's instructions never land on the entry above it. An entry that `call`s one has the
# callee's work outside its row; the banner names such entries (`calls=NAME:N,…`) and stderr says
# why — the same reading as bloomery_gpu_gates::ptx (`body`, `Counts::calls`).
#
# Usage: tools/ptx-scan.sh [--no-jit] <binary name> [entry substring]
#   Reads target/release/<binary> and runs the extractor on its section. The recipe
#   (`just ptx-scan <binary>`; `--features deepseek41` for a V4.1 binary) builds all three first;
#   with `--no-jit` it builds the binary and the extractor only.
#   PTX_SCAN_PTXAS (default $CUDA_TOOLKIT_PATH/bin/ptxas, else /usr/local/cuda/bin/ptxas),
#   PTX_SCAN_ARCH (default BLOOMERY_ARCH, else sm_86; sm_86 is the only arch accepted, see the blk/SM column),
#   PTX_SCAN_EXTRACT (default target/release/oxart_ptx) and
#   PTX_SCAN_JIT (default oxart_jit, a target/release binary run through tools/gpu-gate.sh)
#   override the tools.
#   --no-jit (anywhere among the words; the only flag) skips the driver JIT: no card, no gate lock, no
#   wait on a landing batch's GPU hold. The banner then names `jit=skipped(no-jit)` in place of
#   jit-card/jit-cc/jit-cuda, and every jit_regs and jit_local cell reads `skipped(no-jit)` — never
#   blank, never 0. Everything else is as a full scan: the table's other columns come from ptxas and
#   the byte scan, the md5 block from the extractor, and each of their failures fails the scan. The
#   scan proves a MOVE (no kernel changed: equal md5 block, equal ptxas columns; identical PTX under one
#   driver JITs to identical rows). The add and occupancy/geometry classes and the spill ratchet
#   (tools/ptx-spill-check.sh) read the JIT columns and need the full scan. Every reader of a scan
#   (tools/ref/ptx-canon.py, tools/recipes.py `affected --narrow`, tools/ptx-spill-check.sh) reads the
#   banner's `jit=` and the cell, and never takes the marker for a value.
#   tools/ptx-scan.sh --self-test exercises synthetic PTX with stub tools, no CUDA toolkit or card.
# Output puts entries that have a depot first, then by name ascending — a depot is the defect and
# the rest is context.
# The appended ld.global.scalar, ld.global.v2 and ld.global.v4 columns count disjoint global
# `ld` instructions, including .nc, .volatile and .acquire/.relaxed qualifiers and predication.
# Scalar means no .vN qualifier. Comments, other spaces, `ldu` and device functions are excluded;
# these are static instruction counts, not bytes or a count of memory transactions at runtime.
#
# After the table, a second block: a `ptx-scan-md5: method=<m>` line, then `<entry> <md5> <lines>` per
# row of the table (the same filter), by name ascending. The md5 is of the entry's normalized body and
# `lines` its line count: the instruction stream with comments, whitespace runs, register numbers,
# block-label numbers, `__local_depotN` numbers and the entry's own name taken out, and every generated
# module symbol (`__shared_mem_…`, `__device_global_…`, `__dynamic_smem_…`, hashed or not) replaced by
# its declaration's signature — space, alignment, type, size, an initializer digest — and its order of
# first appearance in the entry, so neither the backend's crate hash nor its numbering reaches the md5
# (the rule table is in crates/gpu-gates/src/ptx.rs, `normalize`; the extractor writes the text). <m>
# names that rule set (`ptx::DIGEST_METHOD`, written by the extractor): two md5 blocks compare only under
# one method, and a block with no method line comes from an older rule set — rescan the old binary with
# this tree's tools instead, from the base tree's directory: `PTX_SCAN_EXTRACT=<this tree>/target/release/
# oxart_ptx bash <this tree>/tools/ptx-scan.sh <binary>` (`tools/ref/ptx-canon.py --scans` compares two
# scans and refuses a pair of two methods by name).
# The table counts instructions; this block sees their order and operands, so a rewrite that keeps
# every count (predicated selects turned into branches, a `stacksave` appearing) changes a digest.
# Compare the two blocks separately: the table is what the columns measure, the digest is whether
# the entry's code is the same code.
#
# Exit status. 0: a banner line, the header, one row per entry, the digest block. 1: the scan failed — the banner line
# on stdout ends in `scan=failed`, names what failed, and no rows follow; the detail is on stderr:
#   missing              no target/release/<binary>
#   section=none         the binary carries no .oxart section (a host-only binary?)
#   extract=failed       the extractor rejected the section or a declaration, met a generated symbol
#                        its module does not declare, or is not there
#   modules=0            the section carries no PTX payload
#   ptxas=none           no executable ptxas, so the ptxas columns cannot be read
#   ptxas-failed=modN    ptxas rejected these modules (comma-separated)
#   jit=failed           the driver did not load every module, or the gate lock was not free (never under
#                        --no-jit, which runs no JIT)
#   ptxas-unread=NAME    ptxas assembled every module but reported nothing for these entries
#   jit-unread=NAME      the driver loaded every module but reported nothing for these entries (not read
#                        under --no-jit)
#   bytescan=failed      the byte scan itself failed
#   rows=0               no entry, or none the entry substring matches
#   digest-unread=NAME   the extractor wrote no normalized body for these entries (`method`: it named
#                        no digest method)
# 2: a usage error (no binary, more than one entry substring, a flag other than --no-jit).
# 64: an arch other than sm_86 (BLOOMERY_ARCH or PTX_SCAN_ARCH): the blk/SM column is sm_86's arithmetic.
#
# The next four columns come from `ptxas -v` on the same PTX, not from the byte scan:
#   regs   registers per thread ("Used N registers")
#   smem   static shared memory per block, bytes (0 when ptxas prints no smem clause)
#   spill  spill stores + spill loads, bytes, summed from the function-properties line
#   blk/SM(static)  resident blocks per SM, derived from those three and .reqntid
# blk/SM is `-` when the entry carries no .reqntid to size a block.
# The occupancy arithmetic is the sm_86 column of the CUDA programming guide's technical
# specifications: 65536 registers and 100 KB of shared memory per SM, 48 resident warps,
# 16 resident blocks (8.0 allows 32, 8.6 does not), registers allocated 256 per warp, shared
# memory 128 bytes per block plus a 1 KB per-block driver reservation. Static only: dynamic
# shared memory is a launch argument and is nowhere in the PTX, so an entry that asks for
# some at launch is reported here as if it asked for none.
# A spill in SASS is not a PTX local round-trip — an entry can show ld.local/st.local here and 0
# spill bytes.
#
# The jit_regs and jit_local columns are what the card runs. The kernels reach it as PTX and the driver JITs
# them at load with its own compiler, which need not be the toolkit's ptxas; oxart_jit loads every
# module the way cuda-core does (cuModuleLoadData, default JIT options) and reads, per entry:
#   jit_regs   CU_FUNC_ATTRIBUTE_NUM_REGS — the registers a launch of the entry occupies
#   jit_local  CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES — per-thread local memory (spill and local arrays)
# The banner names the card, its compute capability and the driver's CUDA version they came from
# (jit-card, jit-cc, jit-cuda). The load needs a card, so it runs under the gate lock: a scan waits
# while another GPU gate holds it. Under --no-jit both columns read `skipped(no-jit)` and the banner's
# three fields are the one field `jit=skipped(no-jit)`.
set -uo pipefail

# Synthetic PTX through the real scan, with extraction, assembly and JIT stubs;
# no device or CUDA toolkit is opened. check-recipes runs this on every host.
self_test() {
  local self
  self=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/$(basename "${BASH_SOURCE[0]}")
  python3 - "$self" << 'PY'
import importlib.util
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile

scanner = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("ptx_canon", scanner.parent / "ref/ptx-canon.py")
canon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon)

with tempfile.TemporaryDirectory(prefix="ptx-scan-test.") as tmp:
    root = Path(tmp)
    fixture, binaries, tree = (root / name for name in ("fixture", "bin", "tree"))
    for path in (fixture, binaries, tree / "target/release", tree / "tools"):
        path.mkdir(parents=True)

    def executable(path, text):
        path.write_text("#!/bin/sh\n" + text, encoding="utf-8")
        path.chmod(0o755)

    executable(tree / "target/release/gate_x", "exit 0\n")
    executable(binaries / "objcopy", 'for arg; do out=$arg; done\nprintf section > "$out"\n')
    executable(binaries / "extract", '''
mkdir -p "$3/norm"
cp "$PTX_FIXTURE"/mod*.ptx "$3/"
printf 'fixture\n' > "$3/norm-method"
for entry in q4k_gemv zero_loads next_module; do
    printf '%s\n' "$entry" > "$3/norm/$entry"
done
printf 'mod1 bundle=fixture bytes=1\nmod2 bundle=fixture bytes=1\n'
''')
    executable(binaries / "ptxas", '''
case $1 in --version) echo 'Cuda compilation tools, release 13.3, V13.3.73'; exit 0 ;; esac
[ "${PTX_FIX_FAIL:-0}" = 0 ] || { echo 'fixture assembly failure' >&2; exit 1; }
for arg; do case $arg in *.ptx) module=${arg##*/} ;; esac; done
case $module in
    mod1.ptx)
        printf "ptxas info : Compiling entry function 'q4k_gemv' for 'sm_86'\nptxas info : 4 bytes spill stores, 4 bytes spill loads\nptxas info : Used 32 registers, 256 bytes smem\n"
        printf "ptxas info : Compiling entry function 'zero_loads' for 'sm_86'\nptxas info : 0 bytes spill stores, 0 bytes spill loads\nptxas info : Used 12 registers\n" ;;
    mod2.ptx)
        printf "ptxas info : Compiling entry function 'next_module' for 'sm_86'\nptxas info : 0 bytes spill stores, 0 bytes spill loads\nptxas info : Used 16 registers, 128 bytes smem\n" ;;
esac >&2
''')
    executable(tree / "tools/gpu-gate.sh", '''
printf '# card=fixture cc=86 cuda_driver=13030\n'
printf 'q4k_gemv\t32\t0\t256\t128\nzero_loads\t12\t0\t0\t1024\nnext_module\t16\t0\t128\t64\n'
''')
    (fixture / "mod1.ptx").write_text('''
.version 8.0
.visible .entry q4k_gemv()
.reqntid 128, 1, 1
{
    .local .b8 __local_depot0[8];
    ld.local.u32 %r1, [%rd1];
    st.local.u32 [%rd1], %r1;
    fma.rn.f32 %f1, %f2, %f3, %f4;
    cvt.f32.f16 %f1, %h1;
    ld.global.u32 %r1, [%rd1];
    ld.global.nc.f32 %f1, [%rd1];
    ld.volatile.global.u64 %rd2, [%rd1];
    ld.acquire.gpu.global.u32 %r1, [%rd1];
    ld.relaxed.sys.global.u32 %r1, [%rd1];
    @%p1 ld.global.L1::evict_last.L2::128B.u32 %r1, [%rd1];
    ld.global.v2.u32 {%r1, %r2}, [%rd1];
    ld.global.nc.v2.u32 {%r1, %r2}, [%rd1];
    ld.volatile.global.v2.u32 {%r1, %r2}, [%rd1];
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    @!%p1 ld.global.L2::128B.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    ld.global.u32 %r1, [%rd1];ld.global.v2.u32 {%r1, %r2}, [%rd1];
    ld.shared.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    ld.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    ldu.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    st.global.v4.u32 [%rd1], {%r1, %r2, %r3, %r4};
    // ld.global.v2.u32 %r1, [%rd1]; ld.global.v4.u32 %r1, [%rd1];
    /* ld.global.u32 %r1, [%rd1];
       ld.global.v4.u32 %r1, [%rd1]; */
    ret; // ld.global.u32 %r1, [%rd1];
}
.func helper()
{
    ld.global.v4.u32 {%r1, %r2, %r3, %r4}, [%rd1];
    fma.rn.f32 %f1, %f2, %f3, %f4;
}
.visible .entry zero_loads()
{
    ret;
}
''', encoding="utf-8")
    (fixture / "mod2.ptx").write_text('''
.visible .entry next_module()
.reqntid 64, 1, 1
{
    ld.global.v2.u32 {%r1, %r2}, [%rd1];
}
''', encoding="utf-8")
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ["PATH"],
               PTX_FIXTURE=str(fixture), PTX_SCAN_PTXAS=str(binaries / "ptxas"),
               PTX_SCAN_EXTRACT=str(binaries / "extract"), PTX_SCAN_ARCH="sm_86")
    failures, cases = [], 0

    def check(name, ok):
        global cases
        cases += 1
        print(("ok " if ok else "FAIL ") + name)
        if not ok:
            failures.append(name)

    def scan(args, extra=None):
        return subprocess.run(["bash", str(scanner), *args], cwd=tree,
                              env=dict(env, **(extra or {})), capture_output=True, text=True)

    def parsed(result):
        path = root / "scan"
        path.write_text(result.stdout, encoding="utf-8")
        return canon.parse_scan(path)

    result = scan(["--no-jit", "gate_x"])
    check("fixture scan succeeds", result.returncode == 0)
    if result.returncode:
        print(result.stdout + result.stderr)
        sys.exit(1)
    method, header, rows, digests, nojit = parsed(result)
    added = ["ld.global.scalar", "ld.global.v2", "ld.global.v4"]
    legacy = "entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local".split()
    check("global-load columns appended", header == legacy + added)
    if header != legacy + added:
        print("got columns:", " ".join(header))
        sys.exit(1)
    check("scalar/vector widths, qualifiers, predicates and comments",
          rows["q4k_gemv"][-3:] == ["7", "4", "3"])
    check("device function excluded and next entry reset", rows["zero_loads"][-3:] == ["0", "0", "0"])
    check("module boundary resets counts", rows["next_module"][-3:] == ["0", "1", "0"])
    check("legacy counts, occupancy and no-jit cells preserved",
          rows["q4k_gemv"][:-3] == "128 YES 1 1 1 1 32 256 8 12 skipped(no-jit) skipped(no-jit)".split()
          and rows["zero_loads"][:-3] == "- no 0 0 0 0 12 0 0 - skipped(no-jit) skipped(no-jit)".split()
          and rows["next_module"][:-3] == "64 no 0 0 0 0 16 128 0 16 skipped(no-jit) skipped(no-jit)".split())
    check("depot-first row order and digest coverage", list(rows) == ["q4k_gemv", "next_module", "zero_loads"]
          and set(digests) == set(rows) and method == "fixture" and nojit)
    selected = scan(["gate_x", "q4k", "--no-jit"])
    _, sh, sr, sd, _ = parsed(selected)
    check("entry filter preserves row and digest", selected.returncode == 0 and sh == header
          and sr == {"q4k_gemv": rows["q4k_gemv"]} and sd == {"q4k_gemv": digests["q4k_gemv"]})
    assembled = scan(["gate_x"])
    _, ah, ar, ad, an = parsed(assembled)
    check("full scan layout and JIT columns", assembled.returncode == 0 and ah == header and not an
          and ar["q4k_gemv"] == rows["q4k_gemv"][:10] + ["32", "0"] + ["7", "4", "3"]
          and ad == digests)
    missing = scan(["--no-jit", "gate_x", "absent"])
    check("empty filter remains a failed scan", missing.returncode == 1
          and "rows=0 scan=failed" in missing.stdout and "entry " not in missing.stdout)
    rejected = scan(["--no-jit", "gate_x"], {"PTX_FIX_FAIL": "1"})
    check("assembler failure prints no rows", rejected.returncode == 1
          and "ptxas-failed=mod1,mod2 scan=failed" in rejected.stdout and "entry " not in rejected.stdout)
    before, after = root / "before.scan", root / "after.scan"
    before.write_text(result.stdout, encoding="utf-8")
    after.write_text(result.stdout, encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        same = canon.compare_scans(before, after)
    check("scan comparison accepts added columns", not same)
    path = fixture / "mod2.ptx"
    path.write_text(path.read_text(encoding="utf-8").replace("ld.global.v2.u32", "ld.global.v4.u32"), encoding="utf-8")
    changed = scan(["--no-jit", "gate_x"])
    after.write_text(changed.stdout, encoding="utf-8")
    with redirect_stdout(io.StringIO()):
        moved = canon.compare_scans(before, after)
    check("scan comparison sees a load-width change", changed.returncode == 0 and moved)
    print(f"ptx-scan: self-test {'FAIL' if failures else 'ok'} ({cases} cases, {len(failures)} failed)")
    sys.exit(bool(failures))
PY
}
if [ "$#" = 1 ] && [ "$1" = --self-test ]; then
  self_test
  exit $?
fi
NOJIT=0
WORDS=()
for W in "$@"; do
  case $W in
    --no-jit) NOJIT=1 ;;
    -*)
      echo "ptx-scan.sh: '$W': the scan's one flag is --no-jit" >&2
      echo "usage: ptx-scan.sh [--no-jit] <gate_bin_name> [entry-substring]" >&2
      exit 2
      ;;
    *) WORDS+=("$W") ;;
  esac
done
NAME=${WORDS[0]:-}
FILTER=${WORDS[1]:-}
# The toolkit the build uses (the box env's CUDA_TOOLKIT_PATH): toolkits differ here by a register
# on some entries, so the scan reads the one the bindings were generated from.
PTXAS=${PTX_SCAN_PTXAS:-${CUDA_TOOLKIT_PATH:-/usr/local/cuda}/bin/ptxas}
ARCH=${PTX_SCAN_ARCH:-${BLOOMERY_ARCH-sm_86}}
EXTRACT=${PTX_SCAN_EXTRACT:-target/release/oxart_ptx}
JIT=${PTX_SCAN_JIT:-oxart_jit}
if [ -z "$NAME" ]; then
  echo "usage: ptx-scan.sh [--no-jit] <gate_bin_name> [entry-substring]" >&2
  exit 2
fi
if [ "${#WORDS[@]}" -gt 2 ]; then
  echo "ptx-scan.sh: '${WORDS[*]:2}' after the entry substring '$FILTER': the scan takes one binary and at most one entry substring" >&2
  echo "usage: ptx-scan.sh [--no-jit] <gate_bin_name> [entry-substring]" >&2
  exit 2
fi
# The blk/SM column is the sm_86 limits (header) and the file holds no other arch's: a scan at another arch would print sm_86's
# number as that arch's occupancy. Refused here and not through tools/arch.sh: tools/narrow-scan.sh ships this file alone to a
# tree that may predate that helper.
if [ "$ARCH" != sm_86 ]; then
  echo "ptx-scan.sh: arch '$ARCH': the blk/SM column is sm_86's occupancy arithmetic and this scan holds no other arch's limits (BLOOMERY_ARCH, PTX_SCAN_ARCH)" >&2
  exit 64
fi
BIN=target/release/$NAME
# A failed scan prints its banner and no rows: two failed scans must never diff as two equal tables.
fail() { # fail <banner fields...>
  echo "ptx-scan bin=$BIN $* scan=failed"
  exit 1
}
if [ ! -x "$BIN" ]; then
  echo "ptx-scan: no $BIN — run the matching just gate-gpu-* recipe first" >&2
  fail missing
fi
PTX=$(mktemp)
TBL=$(mktemp)
MODS=$(mktemp -d)
trap 'rm -rf "$PTX" "$TBL" "$MODS"' EXIT
if ! objcopy -O binary --only-section=.oxart "$BIN" "$PTX" 2>/dev/null || [ ! -s "$PTX" ]; then
  echo "ptx-scan: $BIN carries no .oxart section (host-only binary?)" >&2
  fail section=none
fi
SEC="section=.oxart bytes=$(wc -c <"$PTX" | tr -d ' ')"
# The container is cut by its own parser, never here: one file per bundle that carries PTX.
if [ ! -x "$EXTRACT" ]; then
  echo "ptx-scan: no extractor $EXTRACT — cargo build --release -p bloomery-gpu-gates --bin oxart_ptx" >&2
  fail "$SEC extract=failed"
fi
if ! "$EXTRACT" --norm "$PTX" "$MODS" >"$MODS/list"; then
  fail "$SEC extract=failed"
fi
sed 's/^/ptx-scan: /' "$MODS/list" >&2
NMOD=$(grep -c '^mod[0-9]' "$MODS/list")
# The version names the assembler the regs, smem, spill and blk/SM(static) columns come from: a toolkit move changes them.
if [ -x "$PTXAS" ]; then
  PTXAS_VER=$("$PTXAS" --version 2>/dev/null | sed -n 's/.*, V\([0-9][0-9.]*\)$/\1/p')
  TOOLS="ptxas=$PTXAS ptxas-version=${PTXAS_VER:-unknown} arch=$ARCH"
else
  TOOLS=ptxas=none
fi
if [ "$NMOD" = 0 ]; then
  echo "ptx-scan: the .oxart section carries no PTX payload" >&2
  fail "$SEC $TOOLS modules=0"
fi
if [ ! -x "$PTXAS" ]; then
  echo "ptx-scan: no $PTXAS — regs/smem/spill/blk cannot be read" >&2
  fail "$SEC $TOOLS modules=$NMOD"
fi
MODFILES=()
for ((i = 1; i <= NMOD; i++)); do MODFILES+=("$MODS/mod$i.ptx"); done
FAILED=
for MOD in "${MODFILES[@]}"; do
  # A module ptxas rejects (a .version it does not know, a wrong -arch) has no ptxas columns.
  if ! "$PTXAS" -arch="$ARCH" -v "$MOD" -o /dev/null 2>"$MOD.err"; then
    echo "ptx-scan: ptxas failed on $(basename "$MOD"): $(head -1 "$MOD.err")" >&2
    FAILED=${FAILED:+$FAILED,}$(basename "$MOD" .ptx)
    continue
  fi
  # Key on the name ptxas quotes, never on a substring: `flash_latent` is a prefix of
  # `flash_latent_q8` and of `flash_latent_mma`.
  LC_ALL=C awk -v q="'" '
    index($0, "Compiling entry function") > 0 {
      i = index($0, q); rest = substr($0, i + 1); j = index(rest, q)
      name = substr(rest, 1, j - 1)
      next
    }
    name != "" && index($0, "bytes spill stores") > 0 {
      st = 0; ld = 0
      if (match($0, /[0-9]+ bytes spill stores/)) { s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); st = s + 0 }
      if (match($0, /[0-9]+ bytes spill loads/))  { s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); ld = s + 0 }
      spill[name] = st + ld
      next
    }
    name != "" && match($0, /Used [0-9]+ registers/) {
      s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s); regs[name] = s + 0
      smem[name] = 0
      if (match($0, /[0-9]+ bytes smem/)) { s = substr($0, RSTART, RLENGTH); sub(/ .*/, "", s); smem[name] = s + 0 }
      seen[name] = 1
      next
    }
    END { for (k in seen) printf "%s\t%d\t%d\t%d\n", k, regs[k], smem[k], spill[k] }
  ' "$MOD.err" >>"$TBL"
done
[ -z "$FAILED" ] || fail "$SEC $TOOLS modules=$NMOD ptxas-failed=$FAILED"
# The driver's JIT on the box's card, the same section: one `# card=… cc=… cuda_driver=…` line, then
# `<entry>\t<regs>\t<local>\t<shared>\t<max_threads>` per entry. Either card, the `any` form: the JIT
# columns follow the card's compute capability and the driver, not its name — both cards are GA102
# (sm_86) under one driver, so the table does not depend on which one JITs it — and the scan does not
# queue behind the 3090's gate lock. A scan pair (tools/recipes.py) compares the capability
# (jit-cc) and the driver (jit-cuda), never the card name. Under --no-jit none of this runs: no card,
# no gate lock, and the one banner field `jit=skipped(no-jit)` stands for the three.
JITTBL=$MODS/jit
if [ "$NOJIT" = 1 ]; then
  echo "ptx-scan: --no-jit: the driver JIT is not run; jit_regs and jit_local read skipped(no-jit)" >&2
  JITS="jit=skipped(no-jit)"
else
  # The runner's own `gpu-gate.sh: waited …` lines go back to stderr, so a landing batch books the lock wait.
  jit_rc=0
  BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh "$JIT" "$PTX" >"$JITTBL" 2>"$MODS/jit.err" || jit_rc=$?
  grep '^gpu-gate\.sh: waited ' "$MODS/jit.err" >&2 || true
  if [ "$jit_rc" != 0 ]; then
    echo "ptx-scan: the driver JIT ($JIT) failed: $(tail -1 "$MODS/jit.err")" >&2
    fail "$SEC $TOOLS modules=$NMOD jit=failed"
  fi
  JITS=$(sed -n 's/^# card=\([^ ]*\) cc=\([^ ]*\) cuda_driver=\([^ ]*\)$/jit-card=\1 jit-cc=\2 jit-cuda=\3/p' "$JITTBL")
  if [ -z "$JITS" ]; then
    echo "ptx-scan: $JIT printed no '# card=… cc=… cuda_driver=…' line (an older oxart_jit names no cc)" >&2
    fail "$SEC $TOOLS modules=$NMOD jit=failed"
  fi
fi
ROWS=$MODS/rows
UNREAD=$MODS/unread
JITUNREAD=$MODS/jitunread
CALLS=$MODS/calls
if ! LC_ALL=C awk -v filter="$FILTER" -v tbl="$TBL" -v unread="$UNREAD" -v jit="$JITTBL" \
  -v jitunread="$JITUNREAD" -v callsf="$CALLS" -v nojit="$NOJIT" '
BEGIN {
  # A skipped cell is a word, not a number: the JIT columns widen to hold it.
  fmt = "%d\t%-28s %8s %6s %9d %9d %6d %8d %6s %6s %6s %14s " (nojit ? "%15s %15s" : "%8s %9s") " %16d %12d %12d\n"
  while ((getline line < tbl) > 0) {
    split(line, f, "\t"); R[f[1]] = f[2]; S[f[1]] = f[3]; P[f[1]] = f[4]
  }
  close(tbl)
  while (!nojit && (getline line < jit) > 0) {
    if (substr(line, 1, 1) == "#") continue
    split(line, f, "\t"); JR[f[1]] = f[2]; JL[f[1]] = f[3]
  }
  close(jit)
}
# Resident blocks per SM on sm_86 from the static resources alone; see the header comment
# for the constants and for what "static" leaves out.
function blocks(ntid, regs, smem,   wpb, rpw, lr, lw, ls, b) {
  if (ntid == "" || regs == "") return "-"
  wpb = int((ntid + 31) / 32)
  if (wpb < 1 || wpb > 48) return "-"
  if (regs < 1) regs = 1
  rpw = int((regs * 32 + 255) / 256) * 256
  lr = int(65536 / (rpw * wpb))
  lw = int(48 / wpb)
  ls = int(102400 / (int((smem + 127) / 128) * 128 + 1024))
  b = lr
  if (lw < b) b = lw
  if (ls < b) b = ls
  if (16 < b) b = 16
  return b
}
function flush(   have, hj) {
  if (name == "") return
  have = (name in R)
  hj = (name in JR)
  if (!have) print name > unread
  if (!nojit && !hj) print name > jitunread
  if (filter == "" || index(name, filter) > 0) {
    printf fmt, (depot ? 0 : 1), name,
           (ntid == "" ? "-" : ntid), (depot ? "YES" : "no"), ldl, stl, fma, cvt,
           (have ? R[name] : "-"), (have ? S[name] : "-"), (have ? P[name] : "-"),
           (have ? blocks(ntid, R[name], S[name]) : "-"),
           (nojit ? "skipped(no-jit)" : (hj ? JR[name] : "-")), (nojit ? "skipped(no-jit)" : (hj ? JL[name] : "-")),
           ldg, ldg2, ldg4
    if (calls > 0) printf "%s:%d\n", name, calls > callsf
  }
  name = ""
}
function reset() { ntid = ""; depot = 0; ldl = 0; stl = 0; fma = 0; cvt = 0; calls = 0; ldg = 0; ldg2 = 0; ldg4 = 0 }
function tally(line, needle,   n, s) { n = 0; s = line
  while ((i = index(s, needle)) > 0) { n++; s = substr(s, i + length(needle)) }
  return n }
# `call` as its own mnemonic (call, call.uni, predicated), not `.callprototype` or `// callseq`.
# The character after a match is kept, so it cannot pass for the start of the next one.
function tally_call(line,   n, s) { n = 0; s = line
  while (match(s, /(^|[ \t{;])call[. \t]/)) { n++; s = substr(s, RSTART + RLENGTH - 1) }
  return n }
# Strip comments only for the new columns; the legacy columns keep their own reading.
function global_code(line,   out, p, q) {
  out = ""
  while (line != "") {
    if (comment) {
      q = index(line, "*/")
      if (!q) return out
      line = substr(line, q + 2); comment = 0
      continue
    }
    p = index(line, "/*"); q = index(line, "//")
    if (q && (!p || q < p)) return out substr(line, 1, q - 1)
    if (!p) return out line
    out = out substr(line, 1, p - 1) " "
    line = substr(line, p + 2); comment = 1
  }
  return out
}
# Read the whole mnemonic: state space and vector width need not follow ld immediately.
function tally_global(line,   op) {
  while (match(line, /(^|[ \t{;])ld\.[[:alnum:]_.:]+/)) {
    op = substr(line, RSTART, RLENGTH); sub(/^[ \t{;]/, "", op)
    line = substr(line, RSTART + RLENGTH)
    if (op !~ /(^|\.)global(\.|$)/) continue
    if (op ~ /\.v2(\.|$)/) ldg2++
    else if (op ~ /\.v4(\.|$)/) ldg4++
    else if (op !~ /\.v[0-9]+(\.|$)/) ldg++
  }
}
# An entry body ends at the next entry or at the end of its module, never in the next module.
FNR == 1 { flush(); reset(); comment = 0 }
{ global_line = global_code($0) }
/\.visible \.entry / {
  flush(); reset()
  name = $0
  sub(/.*\.visible \.entry /, "", name)
  sub(/\(.*/, "", name)
  next
}
# ...and at any other declaration: a device function (`.func`, a definition or a prototype) is
# not the entry above it, and its instructions do not belong to that entry.
/(^|[ \t])\.(entry|func)([ \t(]|$)/ { flush(); reset(); next }
name != "" {
  if (ntid == "" && index($0, ".reqntid ") > 0) {
    ntid = $0; sub(/.*\.reqntid /, "", ntid); sub(/[^0-9].*/, "", ntid)
  }
  if (index($0, "__local_depot") > 0) depot = 1
  ldl += tally($0, "ld.local")
  stl += tally($0, "st.local")
  fma += tally($0, "fma.")
  cvt += tally($0, "cvt.f32.f16")
  calls += tally_call($0)
  tally_global(global_line)
}
END { flush() }
' "${MODFILES[@]}" | LC_ALL=C sort -k1,1n -k2,2 | cut -f2- >"$ROWS"; then
  fail "$SEC $TOOLS modules=$NMOD bytescan=failed"
fi
if [ -s "$UNREAD" ]; then
  echo "ptx-scan: ptxas assembled every module but reported nothing for: $(paste -sd, "$UNREAD")" >&2
  fail "$SEC $TOOLS modules=$NMOD ptxas-unread=$(paste -sd, "$UNREAD")"
fi
if [ -s "$JITUNREAD" ]; then
  echo "ptx-scan: the driver loaded every module but reported nothing for: $(paste -sd, "$JITUNREAD")" >&2
  fail "$SEC $TOOLS modules=$NMOD $JITS jit-unread=$(paste -sd, "$JITUNREAD")"
fi
if [ ! -s "$ROWS" ]; then
  echo "ptx-scan: no entry${FILTER:+ name contains $FILTER}" >&2
  fail "$SEC $TOOLS modules=$NMOD${FILTER:+ filter=$FILTER} rows=0"
fi
# One digest line per row, read before the banner: a scan whose digests could not be taken fails.
DIGESTS=$MODS/digests
: >"$DIGESTS"
NODIGEST=
for E in $(awk '{print $1}' "$ROWS" | LC_ALL=C sort); do
  if [ ! -f "$MODS/norm/$E" ]; then
    NODIGEST=${NODIGEST:+$NODIGEST,}$E
    continue
  fi
  printf '%s %s %s\n' "$E" "$(md5sum <"$MODS/norm/$E" | cut -d' ' -f1)" "$(wc -l <"$MODS/norm/$E" | tr -d ' ')" >>"$DIGESTS"
done
METHOD=$(cat "$MODS/norm-method" 2>/dev/null)
if [ -z "$METHOD" ]; then
  echo "ptx-scan: the extractor named no digest method ($MODS/norm-method)" >&2
  NODIGEST=${NODIGEST:+$NODIGEST,}method
fi
if [ -n "$NODIGEST" ]; then
  echo "ptx-scan: the extractor wrote no normalized body for: $NODIGEST" >&2
  fail "$SEC $TOOLS modules=$NMOD $JITS digest-unread=$NODIGEST"
fi
CALLERS=
if [ -s "$CALLS" ]; then
  CALLERS=$(LC_ALL=C sort "$CALLS" | paste -sd, -)
  echo "ptx-scan: entries that call a device function — their rows count their own body, not the callee's: $CALLERS" >&2
fi
echo "ptx-scan bin=$BIN $SEC $TOOLS modules=$NMOD${FILTER:+ filter=$FILTER} $JITS${CALLERS:+ calls=$CALLERS}"
JFMT='%8s %9s'
[ "$NOJIT" = 1 ] && JFMT='%15s %15s'
printf "%-28s %8s %6s %9s %9s %6s %8s %6s %6s %6s %14s $JFMT %16s %12s %12s\n" \
  entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill 'blk/SM(static)' jit_regs jit_local \
  ld.global.scalar ld.global.v2 ld.global.v4
cat "$ROWS"
echo "ptx-scan-md5: method=$METHOD"
cat "$DIGESTS"
exit 0
