#!/usr/bin/env bash
# Builds the prebuilt Linux release on the box and packs it: `just release-build <version>` runs it through
# tools/box.sh with the Mac's commit. It builds into its own target dir (target/portable), so the measure runners'
# target/release is never touched.
#
# The failures it blocks, each checked on the built binaries before anything is packed:
#   - a binary that needs a glibc newer than GLIBC_FLOOR through a strong (non-WEAK) symbol version, so it would not
#     start on the oldest distribution the release names;
#   - a binary that links a shared library outside the allowed set (libcuda must be loaded at run time, never linked);
#   - host code built for this box's CPU (znver3) instead of x86-64-v3: an instruction past x86-64-v3 outside a
#     function that runs behind a runtime CPU check;
#   - a binary with no embedded sm_86 PTX, or PTX for another target.
# Usage: tools/release/build.sh <version> <commit>
#        tools/release/build.sh --check <dir>   (the binary checks alone, on <dir>/<bin>; builds and packs nothing)
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
CHECK_ONLY=
if [ "${1:-}" = --check ]; then
  CHECK_ONLY=${2:?usage: tools/release/build.sh --check <dir>}
else
  VERSION=${1:?usage: tools/release/build.sh <version> <commit>}
  COMMIT=${2:?usage: tools/release/build.sh <version> <commit>}
fi

GLIBC_FLOOR=2.34
BINS=(bloomery-serve bloomery_serve_clef)
DISPATCHED='^(sha2::sha256::x86::digest_blocks|sha1::compress::x86::digest_blocks)$'
ALLOWED_NEEDED='^(libc\.so\.6|libm\.so\.6|libgcc_s\.so\.1|ld-linux-x86-64\.so\.2)$'
if [ -n "$CHECK_ONLY" ]; then
  BIN_DIR=$CHECK_ONLY
else
  NAME="bloomery-$VERSION-linux-x86_64-cuda-sm86"
  DIST="$ROOT/target/release-dist"
  STAGE="$DIST/$NAME"
  export CARGO_TARGET_DIR="$ROOT/target/portable"
  BIN_DIR="$CARGO_TARGET_DIR/release"
  # cargo-oxide appends RUSTFLAGS after the project's extra-rustflags, and rustc keeps the last -C target-cpu.
  export RUSTFLAGS="-C target-cpu=x86-64-v3"
  bin_args=()
  for b in "${BINS[@]}"; do bin_args+=(--bin "$b"); done
  # The commit goes to cargo alone: the servers refuse a BLOOMERY_* variable they do not read.
  BLOOMERY_BUILD_COMMIT="$COMMIT" cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features glm5next,clef --release "${bin_args[@]}"
fi

fail=0
for b in "${BINS[@]}"; do
  f="$BIN_DIR/$b"
  [ -x "$f" ] || { echo "release: $f was not built" >&2; exit 1; }

  strong=$(readelf -V "$f" | awk '/Name: GLIBC_/ && /Flags: none/ {sub("GLIBC_", "", $3); print $3}' | sort -V | tail -1)
  if [ "$(printf '%s\n%s\n' "$strong" "$GLIBC_FLOOR" | sort -V | tail -1)" != "$GLIBC_FLOOR" ]; then
    echo "release: $b needs GLIBC_$strong through a strong symbol version, past the floor $GLIBC_FLOOR" >&2
    fail=1
  fi

  bad_needed=$(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | grep -Ev "$ALLOWED_NEEDED" || true)
  if [ -n "$bad_needed" ]; then
    echo "release: $b links a library outside the allowed set: $bad_needed" >&2
    fail=1
  fi

  # Instructions past x86-64-v3 may sit only in functions that run behind a runtime CPU check (sha2's and sha1's SHA-NI arms,
  # selected by cpufeatures); anywhere else they mean the host code was built for this box's CPU.
  past_v3=$(objdump -d --no-show-raw-insn -C "$f" | awk -v allow="$DISPATCHED" '
    /^[0-9a-f]+ <.*>:$/ { fn = $0; sub(/^[0-9a-f]+ </, "", fn); sub(/>:$/, "", fn); next }
    /\t(sha1|sha256)[a-z0-9]* |\t(extrq|insertq|clzero|monitorx|mwaitx|rdpid|vaes[a-z]*|vpclmulqdq|vpdpbusd)[ \t]|%zmm/ {
      if (fn !~ allow) bad[fn]++
    }
    END { for (k in bad) print bad[k], k }')
  if [ -n "$past_v3" ]; then
    echo "release: $b carries instructions past x86-64-v3 outside a runtime-dispatched function:" >&2
    echo "$past_v3" >&2
    fail=1
  fi

  targets=$(strings -n 8 "$f" | sed -n 's/^\.target[[:space:]]\+\([a-z0-9_]*\).*/\1/p' | sort -u | tr '\n' ' ')
  if [ "$targets" != "sm_86 " ]; then
    echo "release: $b embeds PTX targets [$targets], not sm_86 alone" >&2
    fail=1
  fi

  echo "release: $b glibc_strong_max=$strong needed=$(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p' | tr '\n' ' ')ptx=[$targets] bytes=$(stat -c %s "$f")"
done
[ "$fail" = 0 ] || exit 1
if [ -n "$CHECK_ONLY" ]; then
  echo "release: checks passed on $BIN_DIR"
  exit 0
fi

for b in "${BINS[@]}"; do
  v=$("$CARGO_TARGET_DIR/release/$b" --version)
  case "$v" in
    *"$COMMIT"*) echo "release: $b --version: $v" ;;
    *) echo "release: $b --version does not name the commit $COMMIT: $v" >&2; exit 1 ;;
  esac
done

rm -rf "$STAGE"
mkdir -p "$STAGE/bin"
for b in "${BINS[@]}"; do cp "$CARGO_TARGET_DIR/release/$b" "$STAGE/bin/"; done
cp LICENSE THIRD_PARTY_NOTICES.md "$STAGE/"
cp tools/release/README.release.md "$STAGE/README.md"
printf '%s\n%s\n' "$VERSION" "$COMMIT" > "$STAGE/VERSION"
(cd "$DIST" && tar -czf "$NAME.tar.gz" "$NAME" && sha256sum "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "release: $DIST/$NAME.tar.gz $(stat -c %s "$DIST/$NAME.tar.gz") bytes"
cat "$DIST/$NAME.tar.gz.sha256"
