#!/usr/bin/env bash
# Builds the prebuilt Apple Silicon release on a Mac and packs it: the Metal backend's server, its one-prompt CLI and
# the MLX folder converter, from the tagged commit this tree has checked out.
#
# The failures it blocks, each checked before anything is packed:
#   - a build by another toolchain than rust-toolchain.toml's (a Homebrew `cargo` or `rustc` first in PATH builds with
#     its stable compiler and fails on the pinned crates, or worse, succeeds with another compiler);
#   - a tree that is not the release's commit (the tag's, or --commit's when the tag does not exist yet), or has a tracked file
#     changed;
#   - an archive that would not be signed: a release archive is Developer ID signed and notarized, so MAC_SIGN_IDENTITY and
#     MAC_NOTARY_PROFILE must both be set (the lead's Mac configuration sets them), else the build is refused before it starts;
#     --unsigned is the one way to build without (the archive keeps the linker's ad-hoc signature, a browser download of it is
#     quarantined, and tools/release/publish.sh refuses it);
#   - a binary that links a library outside /usr/lib and /System/Library, so it would not start on a Mac without it;
#   - a binary whose LC_BUILD_VERSION minos is not MACOS_FLOOR (the backend's floor, crates/metal/src/device.rs), or
#     that is not arm64, or that carries no code signature (an unsigned arm64 binary does not run);
#   - when the qwen35moe MLX split's first shard is given, a generate_metal whose greedy ids differ from lab-metal-e2e's.
# The GGUF smoke ids left with the ik clauses.
# PIN(2026-10-28): removed, user decision, Metal verifies MLX only; the GGUF path runs unverified
# Usage: tools/release/build-mac.sh <version> [--commit <commit>] [--unsigned] [<qwen35moe-mlxsplit-first-shard>]
#        tools/release/build-mac.sh --check <dir>   (the binary checks alone, on <dir>/<bin>)
#        tools/release/build-mac.sh --self-test     (the tree and signing refusals on a temporary repository; check-recipes runs it)
# Without --commit the release's tag v<version> must exist and be HEAD. With --commit <commit>, HEAD must be that commit and
# the tag need not exist: the archive is built beside the sitting, before the tag, from the commit the Linux archives were built
# at, and a tag that does exist must name the same commit.
# Writes target/mac-release/dist/$NAME.tar.gz and its .sha256; upload them with `gh release upload v<version>` as named.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
BINS=(bloomery-serve-metal generate_metal prepare)
MACOS_FLOOR=26.2
FIXTURE_IDS=95971,52281,135771,76423,11251,224948,16570,68738
fail() { echo "build-mac.sh: $*" >&2; exit 1; }

# check_tree_commit <version> <commit or ''>: HEAD is the release's commit and no tracked file changed. The release's commit is
# <commit> when one is given (and a tag v<version>, if it exists, names it too), else the tag v<version>'s.
check_tree_commit() {
  local version=$1 want=$2 head full tag
  head=$(git rev-parse HEAD) || fail "cannot read HEAD"
  if [ -n "$want" ]; then
    full=$(git rev-parse --verify --quiet "$want^{commit}") || fail "--commit $want is no commit of this tree"
    [ "$head" = "$full" ] || fail "HEAD $head is not the release commit $want ($full): check out the commit the Linux archives were built at"
    if tag=$(git rev-parse --verify --quiet "v$version^{commit}"); then
      [ "$tag" = "$head" ] || fail "the tag v$version names $tag, not the release commit $head"
    fi
  else
    tag=$(git rev-parse "v$version^{commit}" 2> /dev/null) || fail "no tag v$version (before the tag, name the commit: --commit <commit>)"
    [ "$head" = "$tag" ] || fail "HEAD $head is not v$version ($tag)"
  fi
  [ -z "$(git status --porcelain --untracked-files=no)" ] || fail "tracked files changed: $(git status --porcelain --untracked-files=no | head -3)"
}

# require_signing <unsigned 0|1>: a release archive is signed and notarized, so both of the Mac's settings are present before
# anything is built; --unsigned says so and goes on.
require_signing() {
  local missing=''
  if [ "$1" = 1 ]; then
    echo "build-mac.sh: --unsigned: the archive keeps the linker's ad-hoc signature, a browser download of it is quarantined, and tools/release/publish.sh refuses it" >&2
    return 0
  fi
  [ -n "${MAC_SIGN_IDENTITY:-}" ] || missing=MAC_SIGN_IDENTITY
  [ -n "${MAC_NOTARY_PROFILE:-}" ] || missing="${missing:+$missing and }MAC_NOTARY_PROFILE"
  [ -z "$missing" ] || fail "$missing unset: a release archive is signed with a Developer ID and notarized, and the lead's Mac configuration sets MAC_SIGN_IDENTITY (the signing identity) and MAC_NOTARY_PROFILE (the notarytool keychain profile); --unsigned builds one that tools/release/publish.sh refuses"
}

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

# The refusals on a temporary repository, on any OS: the tree and signing functions are the script's own (nothing here needs a
# Mac), and each guard is held by a mutant of the script that its case must turn red.
self_test() {
  local t r fails=0 cases=0 c1 c2 out rc self
  self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/build-mac.XXXXXX") || return 70
  r=$t/repo
  mkdir "$r"
  git -C "$r" init -q
  printf 'one\n' > "$r/f"
  git -C "$r" add f
  git -C "$r" -c user.name=selftest -c user.email=selftest@invalid commit -q -m one
  c1=$(git -C "$r" rev-parse HEAD)
  git -C "$r" tag v9.9.9
  printf 'two\n' > "$r/f"
  git -C "$r" -c user.name=selftest -c user.email=selftest@invalid commit -q -am two
  c2=$(git -C "$r" rev-parse HEAD)
  # tree_case <script> <loud> <case> <want rc> <want text> <version> <commit or ''>: check_tree_commit of that copy of this script
  tree_case() {
    cases=$((cases + 1))
    # shellcheck source=/dev/null
    out=$( (source "$1" --functions && cd "$r" && check_tree_commit "$6" "$7") 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$4" ] || [[ "$out" != *"$5"* ]]; then
      [ "$2" = 0 ] || echo "build-mac self-test: $3: rc $rc (want $4), want '$5'; got: $out" >&2
      return 1
    fi
  }
  sign_case() { # sign_case <script> <loud> <case> <want rc> <want text> <unsigned> <identity> <profile>
    cases=$((cases + 1))
    # shellcheck source=/dev/null
    out=$( (MAC_SIGN_IDENTITY=$7 MAC_NOTARY_PROFILE=$8 && source "$1" --functions && require_signing "$6") 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$4" ] || [[ "$out" != *"$5"* ]]; then
      [ "$2" = 0 ] || echo "build-mac self-test: $3: rc $rc (want $4), want '$5'; got: $out" >&2
      return 1
    fi
  }
  # all_cases <script> <loud>: every case against that copy; the number of red ones
  all_cases() {
    local n=0 s=$1 l=$2
    tree_case "$s" "$l" "commit mode, HEAD is the commit" 0 "" 9.8.8 "${c2:0:12}" || n=$((n + 1))
    tree_case "$s" "$l" "commit mode, HEAD is not the commit" 1 "HEAD $c2 is not the release commit ${c1:0:12} ($c1)" 9.8.8 "${c1:0:12}" || n=$((n + 1))
    tree_case "$s" "$l" "commit mode, no such commit" 1 "--commit no-such-commit is no commit of this tree" 9.8.8 no-such-commit || n=$((n + 1))
    tree_case "$s" "$l" "commit mode, the tag names another commit" 1 "the tag v9.9.9 names $c1, not the release commit $c2" 9.9.9 "$c2" || n=$((n + 1))
    tree_case "$s" "$l" "tag mode, HEAD is not the tag" 1 "HEAD $c2 is not v9.9.9 ($c1)" 9.9.9 "" || n=$((n + 1))
    tree_case "$s" "$l" "tag mode, no tag" 1 "no tag v9.8.8 (before the tag, name the commit: --commit <commit>)" 9.8.8 "" || n=$((n + 1))
    printf 'dirty\n' > "$r/f"
    tree_case "$s" "$l" "a tracked file changed" 1 "tracked files changed" 9.8.8 "${c2:0:12}" || n=$((n + 1))
    git -C "$r" checkout -q -- f
    sign_case "$s" "$l" "signing, both unset" 1 "MAC_SIGN_IDENTITY and MAC_NOTARY_PROFILE unset" 0 "" "" || n=$((n + 1))
    sign_case "$s" "$l" "signing, no notary profile" 1 "MAC_NOTARY_PROFILE unset" 0 "Developer ID Application: X" "" || n=$((n + 1))
    sign_case "$s" "$l" "signing, no identity" 1 "MAC_SIGN_IDENTITY unset" 0 "" "profile" || n=$((n + 1))
    sign_case "$s" "$l" "signing, both set" 0 "" 0 "Developer ID Application: X" "profile" || n=$((n + 1))
    sign_case "$s" "$l" "signing, --unsigned names the publish refusal" 0 "tools/release/publish.sh refuses it" 1 "" "" || n=$((n + 1))
    # the script, run: signing is refused before the host, the toolchain and the tree are looked at
    cases=$((cases + 1))
    out=$(cd "$r" && env -u MAC_SIGN_IDENTITY -u MAC_NOTARY_PROFILE bash "$s" 9.8.8 --commit "${c2:0:12}" 2>&1) && rc=0 || rc=$?
    if [ "$rc" != 1 ] || [[ "$out" != *"MAC_SIGN_IDENTITY and MAC_NOTARY_PROFILE unset"* ]]; then
      [ "$l" = 0 ] || echo "build-mac self-test: the run: rc $rc (want 1), want the signing refusal first; got: $out" >&2
      n=$((n + 1))
    fi
    cases=$((cases + 1))
    out=$(cd "$r" && bash "$s" 9.8.8 --commit 2>&1) && rc=0 || rc=$?
    if [ "$rc" != 64 ] || [[ "$out" != *"--commit needs a commit"* ]]; then
      [ "$l" = 0 ] || echo "build-mac self-test: --commit without a commit: rc $rc (want 64); got: $out" >&2
      n=$((n + 1))
    fi
    # the build calls both functions (a guard nobody calls holds nothing)
    cases=$((cases + 1))
    if ! grep -q '^check_tree_commit "\$VERSION" "\$COMMIT"$' "$s" || ! grep -q '^require_signing "\$UNSIGNED"$' "$s"; then
      [ "$l" = 0 ] || echo "build-mac self-test: the build does not call check_tree_commit and require_signing" >&2
      n=$((n + 1))
    fi
    return "$n"
  }
  # mutant <name> <old text> <new text>: this script with the first <old text> replaced; at least one case must go red
  mutant() {
    cp "$self" "$t/mutant.sh"
    python3 -I - "$t/mutant.sh" "$2" "$3" << 'PY' || { echo "build-mac self-test: mutant $1: the text to replace is not in the code" >&2; fails=$((fails + 1)); return 0; }
import sys

path, old, new = sys.argv[1:4]
s = open(path).read()
i = s.find(old)
if i < 0:
    sys.exit(1)
open(path, "w").write(s[:i] + new + s[i + len(old):])
PY
    if all_cases "$t/mutant.sh" 0; then
      echo "build-mac self-test: mutant $1 left every case green: the cases do not hold it" >&2
      fails=$((fails + 1))
    fi
  }
  all_cases "$self" 1 || fails=$((fails + $?))
  mutant "HEAD need not be the commit" '[ "$head" = "$full" ] ||' 'true ||'
  mutant "the tag may name another commit" '[ "$tag" = "$head" ] || fail "the tag' 'true || fail "the tag'
  mutant "the release is unsigned without a word" '[ -z "$missing" ] || fail' 'true || fail'
  mutant "signing is asked after the host" $'\nrequire_signing "$UNSIGNED"\n' $'\n:\n'
  rm -rf "$t"
  if [ "$fails" = 0 ]; then
    echo "build-mac.sh: self-test ok ($cases cases)"
  else
    echo "build-mac.sh: self-test $fails failure(s)" >&2
    return 1
  fi
}
if [ "${1:-}" = --functions ]; then return 0 2> /dev/null || true; fi
if [ "${1:-}" = --self-test ]; then
  self_test
  exit $?
fi
if [ "${1:-}" = --check ]; then
  check_bins "${2:?usage: tools/release/build-mac.sh --check <dir>}"
  exit 0
fi
USAGE="usage: tools/release/build-mac.sh <version> [--commit <commit>] [--unsigned] [<fixture.gguf>] | --check <dir> | --self-test"
VERSION='' FIXTURE='' COMMIT='' UNSIGNED=0
while [ $# -gt 0 ]; do
  case $1 in
    --commit) [ $# -ge 2 ] || { echo "build-mac.sh: --commit needs a commit; $USAGE" >&2; exit 64; }; COMMIT=$2; shift 2 ;;
    --unsigned) UNSIGNED=1; shift ;;
    -*) echo "build-mac.sh: unknown option $1; $USAGE" >&2; exit 64 ;;
    *)
      if [ -z "$VERSION" ]; then VERSION=$1; elif [ -z "$FIXTURE" ]; then FIXTURE=$1; else echo "build-mac.sh: unexpected argument $1; $USAGE" >&2; exit 64; fi
      shift ;;
  esac
done
[ -n "$VERSION" ] || { echo "$USAGE" >&2; exit 64; }
# Signing first: a build that would end unsigned is refused before it starts, not after its minutes.
require_signing "$UNSIGNED"
[ "$(uname -s)-$(uname -m)" = Darwin-arm64 ] || fail "this builds on an Apple Silicon Mac; this is $(uname -s)-$(uname -m)"

# The toolchain: rustup's proxies first, so rust-toolchain.toml picks the compiler.
export PATH="$HOME/.cargo/bin:$PATH"
unset RUSTC RUSTC_WRAPPER
channel=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' rust-toolchain.toml)
active=$(rustup show active-toolchain | awk '{print $1}')
case "$active" in "$channel"-*) ;; *) fail "the active toolchain is $active, not rust-toolchain.toml's $channel" ;; esac

# The tree: the release's commit, nothing tracked changed.
check_tree_commit "$VERSION" "$COMMIT"
head=$(git rev-parse HEAD)

export MACOSX_DEPLOYMENT_TARGET=$MACOS_FLOOR CARGO_TARGET_DIR="$ROOT/target/mac-release"
cargo build --release --locked -p bloomery-metal --bin bloomery-serve-metal --bin generate_metal \
  -p bloomery-model-hf --bin prepare
OUT="$CARGO_TARGET_DIR/release"
check_bins "$OUT"

# Developer ID: sign with the hardened runtime and a secure timestamp, then notarize, so Gatekeeper opens a binary a
# browser downloaded. --unsigned: the binaries keep the linker's ad-hoc signature and a browser download is quarantined.
if [ "$UNSIGNED" = 0 ]; then
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
cp LICENSE THIRD_PARTY_NOTICES.md docs/serving.md "$STAGE/"
sed "s/@VERSION@/$VERSION/g; s/@COMMIT@/$head/g" "$(dirname "$0")/README.release-macos.md" > "$STAGE/README.md"
(cd "$DIST" && COPYFILE_DISABLE=1 tar -czf "$NAME.tar.gz" "$NAME" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")
echo "build-mac.sh: $DIST/$NAME.tar.gz ($(stat -f %z "$DIST/$NAME.tar.gz") bytes), sha256 $(cut -d' ' -f1 "$DIST/$NAME.tar.gz.sha256")"
