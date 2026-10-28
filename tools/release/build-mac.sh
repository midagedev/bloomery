#!/usr/bin/env bash
# Builds the prebuilt Apple Silicon release on a Mac and packs it: the Metal backend's server, its one-prompt CLI and
# the MLX folder converter, from the tagged commit this tree has checked out.
#
# The failures it blocks, each checked before anything is packed:
#   - a build by another toolchain than rust-toolchain.toml's (a Homebrew `cargo` or `rustc` first in PATH builds with
#     its stable compiler and fails on the pinned crates, or worse, succeeds with another compiler);
#   - a tree that is not the release tag's commit, or has a tracked file changed;
#   - a binary that links a library outside /usr/lib and /System/Library, so it would not start on a Mac without it;
#   - a binary whose LC_BUILD_VERSION minos is not MACOS_FLOOR (the backend's floor, crates/metal/src/device.rs), or
#     that is not arm64, or that carries no code signature (an unsigned arm64 binary does not run);
#   - when the qwen35moe MLX split's first shard is given, a generate_metal whose greedy ids differ from lab-metal-e2e's.
# The GGUF smoke ids left with the ik clauses.
# PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
# Usage: tools/release/build-mac.sh <version> [<qwen35moe-mlxsplit-first-shard>]
#        tools/release/build-mac.sh --check <dir>   (the binary checks alone, on <dir>/<bin>)
# Writes target/mac-release/dist/$NAME.tar.gz and its .sha256; upload them with `gh release upload v<version>` as named.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
BINS=(bloomery-serve-metal generate_metal prepare)
MACOS_FLOOR=26.2
FIXTURE_IDS=95971,52281,135771,76423,11251,224948,16570,68738
fail() { echo "build-mac.sh: $*" >&2; exit 1; }

check_bins() {
  local dir=$1 b libs minos
  for b in "${BINS[@]}"; do
    [ -x "$dir/$b" ] || fail "$dir/$b is missing"
    [[ "$(file "$dir/$b")" == *"Mach-O 64-bit executable arm64"* ]] || fail "$b is not an arm64 Mach-O executable"
    libs=$(otool -L "$dir/$b" | tail -n +2 | awk '{print $1}' | grep -vE '^(/usr/lib/|/System/Library/)' || true)
    [ -z "$libs" ] || fail "$b links outside the system: $libs"
    minos=$(otool -l "$dir/$b" | awk '/LC_BUILD_VERSION/{f=1} f && $1=="minos"{print $2; exit}')
    [ "$minos" = "$MACOS_FLOOR" ] || fail "$b has minos $minos, not $MACOS_FLOOR"
    codesign -v "$dir/$b" 2>/dev/null || fail "$b carries no valid code signature"
  done
  echo "build-mac.sh: ${BINS[*]}: arm64, system libraries only, minos $MACOS_FLOOR, signed"
}

if [ "${1:-}" = --check ]; then
  check_bins "${2:?usage: tools/release/build-mac.sh --check <dir>}"
  exit 0
fi
VERSION=${1:?usage: tools/release/build-mac.sh <version> [<fixture.gguf>]}
FIXTURE=${2:-}
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || fail "this builds on an Apple Silicon Mac; this is $(uname -s)-$(uname -m)"

# The toolchain: rustup's proxies first, so rust-toolchain.toml picks the compiler.
export PATH="$HOME/.cargo/bin:$PATH"
unset RUSTC RUSTC_WRAPPER
channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
active=$(rustup show active-toolchain | awk '{print $1}')
case "$active" in "$channel"-*) ;; *) fail "the active toolchain is $active, not rust-toolchain.toml's $channel" ;; esac

# The tree: the tag's commit, nothing tracked changed.
head=$(git rev-parse HEAD)
tag=$(git rev-parse "v$VERSION^{commit}" 2>/dev/null) || fail "no tag v$VERSION"
[ "$head" = "$tag" ] || fail "HEAD $head is not v$VERSION ($tag)"
[ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked files changed: $(git status --porcelain --untracked-files=no | head -3)"

export MACOSX_DEPLOYMENT_TARGET=$MACOS_FLOOR CARGO_TARGET_DIR="$ROOT/target/mac-release"
cargo build --release --locked -p bloomery-metal --bin bloomery-serve-metal --bin generate_metal \
  -p bloomery-model-hf --bin prepare
OUT="$CARGO_TARGET_DIR/release"
check_bins "$OUT"

# Developer ID: sign with the hardened runtime and a secure timestamp, then notarize, so Gatekeeper opens a binary a
# browser downloaded. Unset, the binaries keep the linker's ad-hoc signature and a browser download is quarantined.
if [ -n "${MAC_SIGN_IDENTITY:-}" ]; then
  [ -n "${MAC_NOTARY_PROFILE:-}" ] || fail "MAC_SIGN_IDENTITY is set without MAC_NOTARY_PROFILE"
  for b in "${BINS[@]}"; do
    codesign --force --options runtime --timestamp --sign "$MAC_SIGN_IDENTITY" "$OUT/$b"
    sig=$(codesign -dvv "$OUT/$b" 2>&1)
    grep -q "^Authority=Developer ID Application" <<< "$sig" || fail "$b is not signed by a Developer ID"
    grep -q 'flags=.*runtime' <<< "$sig" || fail "$b has no hardened runtime"
  done
  zip="$CARGO_TARGET_DIR/notarize.zip"
  rm -f "$zip"
  (cd "$OUT" && zip -qj "$zip" "${BINS[@]}")
  result=$(xcrun notarytool submit "$zip" --keychain-profile "$MAC_NOTARY_PROFILE" --wait --output-format json)
  status=$(printf '%s' "$result" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status"))')
  [ "$status" = Accepted ] || fail "notarization: $status: $result"
  echo "build-mac.sh: signed by $(sed -n 's/^Authority=//p' <<< "$sig" | sed -n 1p), notarized"
fi

if [ -n "$FIXTURE" ]; then
  err="$CARGO_TARGET_DIR/fixture-smoke.err"
  ids=$(BLOOMERY_REF_MODEL="$FIXTURE" "$OUT/generate_metal" --tokens 760,6511,314 -n 8 --print-ids 2>"$err") \
    || fail "generate_metal on $FIXTURE ended rc $?: $(tail -3 "$err")"
  [ "$ids" = "$FIXTURE_IDS" ] || fail "generate_metal on $FIXTURE gave $ids; lab-metal-e2e expects $FIXTURE_IDS"
  echo "build-mac.sh: generate_metal on the fixture gives lab-metal-e2e's ids"
fi

NAME="bloomery-$VERSION-macos-arm64-metal"
DIST="$CARGO_TARGET_DIR/dist"
STAGE="$DIST/$NAME"
rm -rf "$STAGE" "$DIST/$NAME.tar.gz" "$DIST/$NAME.tar.gz.sha256"
mkdir -p "$STAGE/bin"
for b in "${BINS[@]}"; do cp "$OUT/$b" "$STAGE/bin/"; done
cp LICENSE THIRD_PARTY_NOTICES.md "$STAGE/"
sed "s/@VERSION@/$VERSION/g; s/@COMMIT@/$head/g" "$(dirname "$0")/README.release-macos.md" > "$STAGE/README.md"
(cd "$DIST" && COPYFILE_DISABLE=1 tar -czf "$NAME.tar.gz" "$NAME" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "build-mac.sh: $DIST/$NAME.tar.gz ($(stat -f %z "$DIST/$NAME.tar.gz") bytes), sha256 $(cut -d' ' -f1 "$DIST/$NAME.tar.gz.sha256")"
