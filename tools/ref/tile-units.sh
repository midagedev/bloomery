#!/usr/bin/env bash
# The host expert tier's tile cost rows, derived from the code: builds qdot for x86_64-linux with the repo's
# rustflags (no linking, so it runs on the Mac with no box), disassembles the rlib with the pinned toolchain's
# llvm-objdump, and runs `tools/ref/asm-census.py tile-units`: the fixed and per-column units of every type qdot
# tiles, fitted over the `qdot::tile::<C>` loops, against `tile_units` in crates/model/src/ops.rs.
#
#   tools/ref/tile-units.sh [OUT_DIR]     OUT_DIR (default: a new temp dir, printed) keeps the listing and relocations
#
# Exit: asm-census's. 0, every row of the table is within 5 % of the derivation; 1, a row is missing (a type qdot
# tiles that the balancer prices by bytes × columns), stale (a qdot tile change moved the instruction counts) or for a
# type with no tile kernel, or the listing is not a shape the derivation reads (the message names it). 69: the pinned
# toolchain or its llvm-tools is not installed. The `rows (derived)` lines are the table's rows to paste.
# CARGO_TARGET_DIR is the tree's own target/ unless set.
set -euo pipefail
root=$(cd "${BASH_SOURCE[0]%/*}/../.." && pwd)
out=${1:-$(mktemp -d)}
mkdir -p "$out"
channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$root/rust-toolchain.toml")
tc=$(compgen -G "$HOME/.rustup/toolchains/$channel-*" | head -n1 || true)
[ -n "$tc" ] && [ -x "$tc/bin/cargo" ] || { echo "tile-units: toolchain $channel is not installed under ~/.rustup/toolchains" >&2; exit 69; }
objdump=$(compgen -G "$tc/lib/rustlib/*/bin/llvm-objdump" | head -n1 || true)
[ -n "$objdump" ] || { echo "tile-units: the $channel toolchain has no llvm-tools (rustup component add llvm-tools)" >&2; exit 69; }
export PATH="$tc/bin:$PATH" RUSTC="$tc/bin/rustc" CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/target}"
# The repo's rustflags (.cargo/config.toml) apply only while no encoded flags are set.
unset CARGO_ENCODED_RUSTFLAGS RUSTFLAGS
(cd "$root" && cargo build -p bloomery-qdot --lib --release --target x86_64-unknown-linux-gnu)
rlib="$CARGO_TARGET_DIR/x86_64-unknown-linux-gnu/release/libqdot.rlib"
[ -f "$rlib" ] || { echo "tile-units: the build left no $rlib" >&2; exit 1; }
"$objdump" -d --no-show-raw-insn -C "$rlib" > "$out/qdot.lst"
"$objdump" -r "$rlib" > "$out/qdot.rel"
echo "tile-units: listing $out/qdot.lst, relocations $out/qdot.rel"
exec python3 "$root/tools/ref/asm-census.py" tile-units "$out/qdot.lst" "$out/qdot.rel" \
  "$root/crates/qdot/src/lib.rs" "$root/crates/model/src/ops.rs"
