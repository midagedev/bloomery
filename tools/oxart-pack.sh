#!/usr/bin/env bash
# Packs a binary's `.oxart` bundles as cubins the CUDA toolkit's ptxas compiles, on the build host with no card:
# `.oxart` then carries the cubins and `.oxart.ptx` the PTX they came from (bloomery-gpu loads that PTX for a bundle
# whose cubin the driver refuses). tools/release/build.sh packs the GB10 archive with it. A binary built from source
# for a cc 12.x card whose driver predates CUDA 13.1 is packed with it before it runs: that driver's PTX compiler
# truncates some kernels' addresses on cc 12.x (docs/upstream/nvlabs-ledger.md row 55), and bloomery-gpu refuses by
# name to hand it PTX for such a card.
#
#   tools/oxart-pack.sh <binary> [<oxart_ptxas>]   default oxart_ptxas: ${CARGO_TARGET_DIR:-target}/release/oxart_ptxas
#                                                  (plain `cargo build --release -p bloomery-gpu-gates --bin oxart_ptxas`)
#
# The failure it blocks: a shipped cubin carrying that compiler's truncated uniform address. Every cubin is read back
# with `cuobjdump -sass` through tools/ulea-scan.py before the binary is touched; a suspect leaves the binary as it was
# (rc 1), and so does a binary already packed (it has `.oxart.ptx`) or one with no `.oxart` (rc 65).
# Tools: ptxas and cuobjdump from PATH (the toolkit that compiles and reads the cubins; ptxas 13.1 or later for an
# sm_12x target), objcopy as $OBJCOPY (default objcopy; a cross build names the target's).
# exit: 0 packed, 1 the scan found a suspect, 64 usage, 65 already packed or nothing to pack, 69 a tool or file missing,
# else the failing tool's code.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
case $# in
  1 | 2) ;;
  *)
    sed -n '9,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 64
    ;;
esac
bin=$1
packer=${2:-${CARGO_TARGET_DIR:-$ROOT/target}/release/oxart_ptxas}
objcopy=${OBJCOPY:-objcopy}
[ -f "$bin" ] || { echo "oxart-pack: no binary at $bin" >&2; exit 69; }
[ -x "$packer" ] || { echo "oxart-pack: no oxart_ptxas at $packer (cargo build --release -p bloomery-gpu-gates --bin oxart_ptxas)" >&2; exit 69; }
for t in "$objcopy" ptxas cuobjdump; do
  command -v "$t" > /dev/null || { echo "oxart-pack: $t is not on PATH" >&2; exit 69; }
done

tmp=$(mktemp -d "${TMPDIR:-/tmp}/oxart-pack.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
# objcopy only warns (rc 0) on a section that is not there, so the dumped files say which sections are.
"$objcopy" --dump-section .oxart="$tmp/ptx.oxart" --dump-section .oxart.ptx="$tmp/had.oxart.ptx" \
  "$bin" "$tmp/probe" 2> /dev/null
rm -f "$tmp/probe"
if [ -e "$tmp/had.oxart.ptx" ]; then
  echo "oxart-pack: $bin already carries .oxart.ptx: it is packed" >&2
  exit 65
fi
if [ ! -s "$tmp/ptx.oxart" ]; then
  echo "oxart-pack: $bin has no .oxart section to pack" >&2
  exit 65
fi
mkdir "$tmp/cubins"
"$packer" "$tmp/ptx.oxart" "$tmp/cubin.oxart" --keep "$tmp/cubins"
echo "# cuobjdump $(cuobjdump --version | tail -1)"
rc=0
python3 "$ROOT/tools/ulea-scan.py" --cubins "$tmp/cubins" || rc=$?
if [ "$rc" != 0 ]; then
  echo "oxart-pack: $bin left as it was: the cubins of $(command -v ptxas) failed the address check (rc $rc)" >&2
  exit "$rc"
fi
"$objcopy" --rename-section .oxart=.oxart.ptx --add-section .oxart="$tmp/cubin.oxart" "$bin" "$tmp/packed"
chmod --reference="$bin" "$tmp/packed"
mv "$tmp/packed" "$bin"
echo "oxart-pack: packed $bin"
