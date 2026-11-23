#!/usr/bin/env bash
# shellcheck shell=bash
# The pack step of the gate runners, one owner for tools/gate.sh and tools/gpu-gate.sh. Sourced, never executed; the caller has
# sourced tools/gate-bound.sh (run_bounded, gate_build_bound).
#
# A driver older than CUDA 13.1 compiles PTX for a cc 12.x card with a defect that truncates some kernels' addresses
# (docs/upstream/nvlabs-ledger.md row 55), and bloomery-gpu refuses by name to hand it PTX there: SM12X_JIT_FIXED in
# crates/gpu/src/lib.rs, the one place that version is written and the file this reads it from. Such a driver runs a binary only
# when it carries cubins, which tools/oxart-pack.sh compiles with the toolkit's ptxas on the build host with no card. This is
# the step that makes a gate's executable one of those binaries, before it takes a card.
#
#   pack_for_driver <caller> <executable>   0 when the executable may run (packed now, already packed, or never needed packing);
#                                           else the code the runner ends with, after a `GATE RED (exit N) in the pack phase — no
#                                           test ran` line (`GATE TIMED OUT in the pack phase …` for a bound)
#
# It packs when all of these hold, the cheapest asked first, so a shell script, a host binary, an sm_86 build and a packed binary
# pass with no libcuda, no lib.rs and no ptxas read: the file is an ELF with an `.oxart` section; it has no `.oxart.ptx`
# (oxart-pack.sh's mark of a packed binary); a bundle of it targets sm_12x (`.target sm_12…` in the PTX); and libcuda's
# cuDriverGetVersion (the call the loader makes, needs no card) is below SM12X_JIT_FIXED. The pack is idempotent: oxart-pack.sh's
# 65 (already packed, or nothing to pack) is no error here. oxart_ptxas is built first (plain cargo: it is a host tool) when
# ${CARGO_TARGET_DIR:-<repo>/target}/release has none. The build and the pack run under the build bound (BLOOMERY_GATE_BUILD_BOUND,
# 2700 s; gate-bound.sh), a bound that ran out ends 124 or 137 as the build phase does, and a TERM to the runner stops them.
#
# The codes it returns of its own: 64 a refused BLOOMERY_GATE_BUILD_BOUND; 69 objcopy, python3, ptxas, cuobjdump or oxart_ptxas
# missing (oxart-pack.sh's own 69 as well); 70 the executable's sections, SM12X_JIT_FIXED or the driver's version cannot be read.
# Beyond those the failing tool's code: oxart-pack.sh's 1 (a cubin failed tools/ulea-scan.py's address check, as ptxas older than
# 13.1 does on sm_12x), cargo's, a bound's 124 or 137. The red line is what tells these from a test's own code: none of them ran one.
# PACK_TARGET is the target of the executable's PTX when it is a candidate for the driver check (`sm_120a`), empty otherwise.

# cuda_dotted <1000 * major + 10 * minor>: the version as `13.4`.
cuda_dotted() { echo "$(($1 / 1000)).$(($1 % 1000 / 10))"; }

# driver_cuda_version <caller>: the CUDA version of the driver the card runs under, 1000 * major + 10 * minor, from libcuda's
# cuDriverGetVersion; the named failure on stderr and a non-zero status when there is none to read.
driver_cuda_version() {
  command -v python3 > /dev/null || { echo "$1: python3 is not on PATH, which reads the driver's CUDA version" >&2; return 69; }
  python3 -I - "$1" << 'PY'
import ctypes
import sys

for name in ("libcuda.so.1", "libcuda.so"):
    try:
        lib = ctypes.CDLL(name)
        break
    except OSError:
        pass
else:
    sys.exit(sys.argv[1] + ": no libcuda to read the driver's CUDA version from")
version = ctypes.c_int()
rc = lib.cuDriverGetVersion(ctypes.byref(version))
if rc != 0:
    sys.exit(sys.argv[1] + ": cuDriverGetVersion failed (" + str(rc) + ")")
print(version.value)
PY
}

# pack_red <rc>: the pack phase's named red on stderr, then the code.
pack_red() {
  if [ "$1" -eq 124 ] || [ "$1" -eq 137 ]; then
    echo "GATE TIMED OUT in the pack phase after the ${BUILD_BOUND}s build bound (exit $1, BLOOMERY_GATE_BUILD_BOUND) — no test ran; a gate that hangs is a red gate, not a silent one" >&2
  else
    echo "GATE RED (exit $1) in the pack phase — no test ran" >&2
  fi
  return "$1"
}

# pack_bounded <command…>: a command of the pack phase under the build bound, in this shell (run_bounded's contract); its status.
pack_bounded() {
  local rc=0
  run_bounded timeout --kill-after=10 "$BUILD_BOUND" "$@" || rc=$?
  [ -z "$BIN_SIGNAL" ] || exit 143
  return "$rc"
}

# pack_probe <caller> <executable>: sets PACK_TARGET to the sm_12x target of the executable's unpacked PTX, or to nothing when it
# is not a candidate (no ELF, no `.oxart`, already packed, another target); 0, or the named failure's code.
pack_probe() {
  local exe=$2 objcopy=${OBJCOPY:-objcopy} tmp rc=0
  PACK_TARGET=''
  [ -f "$exe" ] || return 0
  [ "$(head -c 4 "$exe" 2> /dev/null | od -An -tx1 | tr -d ' \n')" = 7f454c46 ] || return 0
  command -v "$objcopy" > /dev/null || { echo "$1: $objcopy is not on PATH, which reads the sections of $exe to tell whether it needs packing (OBJCOPY names another)" >&2; return 69; }
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/pack-probe.XXXXXX") || { echo "$1: cannot make a directory under ${TMPDIR:-/tmp} to read $exe's sections in" >&2; return 70; }
  # objcopy writes an empty file for a section the binary lacks, so the sizes say which sections are there.
  if "$objcopy" -O binary --only-section=.oxart "$exe" "$tmp/ptx" 2> "$tmp/err" && "$objcopy" -O binary --only-section=.oxart.ptx "$exe" "$tmp/had" 2>> "$tmp/err"; then
    if [ -s "$tmp/ptx" ] && [ ! -s "$tmp/had" ]; then
      PACK_TARGET=$(LC_ALL=C grep -a -m 1 -o '\.target sm_12[0-9a-z]*' "$tmp/ptx" | sed 's/^\.target //') || PACK_TARGET=''
    fi
  else
    echo "$1: $objcopy could not read the sections of $exe: $(head -c 300 "$tmp/err")" >&2
    rc=70
  fi
  rm -rf "$tmp"
  return "$rc"
}

pack_for_driver() {
  local caller=$1 exe=$2 root fixed have packer rc=0
  pack_probe "$caller" "$exe" || { rc=$?; pack_red "$rc"; return "$rc"; }
  [ -n "$PACK_TARGET" ] || return 0
  root=$(cd "${BASH_SOURCE[0]%/*}/.." && pwd) || { pack_red 70; return 70; }
  fixed=$(sed -n 's/^const SM12X_JIT_FIXED: i32 = \([0-9][0-9]*\);.*/\1/p' "$root/crates/gpu/src/lib.rs" 2> /dev/null)
  if [ -z "$fixed" ]; then
    echo "$caller: $root/crates/gpu/src/lib.rs has no 'const SM12X_JIT_FIXED: i32 = <version>;' line to read the first driver with a sound PTX compiler from" >&2
    pack_red 70
    return 70
  fi
  have=$(driver_cuda_version "$caller") || { rc=$?; pack_red "$rc"; return "$rc"; }
  if [ "$have" -ge "$fixed" ]; then
    echo "$caller: $exe carries $PACK_TARGET PTX, driver CUDA $(cuda_dotted "$have") is $(cuda_dotted "$fixed") or later: not packed"
    return 0
  fi
  echo "$caller: $exe carries $PACK_TARGET PTX, driver CUDA $(cuda_dotted "$have") is before $(cuda_dotted "$fixed"): packing it"
  gate_build_bound "$caller" || { rc=$?; pack_red "$rc"; return "$rc"; }
  packer=${CARGO_TARGET_DIR:-$root/target}/release/oxart_ptxas
  if [ ! -x "$packer" ]; then
    pack_bounded env -C "$root" cargo build --release -p bloomery-gpu-gates --bin oxart_ptxas || { rc=$?; pack_red "$rc"; return "$rc"; }
  fi
  pack_bounded bash "$root/tools/oxart-pack.sh" "$exe" "$packer" || rc=$?
  case $rc in
    0 | 65) return 0 ;; # 65: nothing to pack, which oxart-pack.sh said (a racing run packed it, or it has no device code)
    *)
      pack_red "$rc"
      return "$rc"
      ;;
  esac
}
