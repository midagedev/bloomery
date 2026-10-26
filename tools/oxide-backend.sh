#!/usr/bin/env bash
# The cuda-oxide codegen backend bloomery builds with: the pin's backend source plus
# tools/oxide-backend/cuda-rust-1648.patch (NVIDIA/cuda-rust pull request #1648).
#   tools/oxide-backend.sh                build the backend for the pin and install it in its directory
#   tools/oxide-backend.sh --path         where the backend of the pin is installed
#   tools/oxide-backend.sh --check [SO]   SO (default: CUDA_OXIDE_BACKEND, else the installed one) is the pin's backend and
#                                         carries this tree's patch
#   tools/oxide-backend.sh --self-test
# Exit 0 when done; 70 names what is missing or wrong and nothing is installed past it; 64 is a usage error.
#
# Why a patch. dialect-cute's semantic verifier runs on every module cargo-oxide's backend lowers, and its provenance
# facts cost time cubic in the block count of a kernel's scope even when the scope holds no CuTe operation: at the pin,
# bloomery-gpu does not finish in half an hour. The patch is NVIDIA/cuda-rust pull request #1648 (fixes #1647) as its
# author published it: facts indexed by carrier, a story skipped for a scope without its family's operations. The
# verifier has no switch (no option, env or attribute), and a backend built by cargo-oxide reads no patch of this
# workspace: cargo-oxide builds the backend in the dependency's own checkout, a Cargo workspace of its own. So the backend is built here, from a copy of that checkout
# with the patch applied, by cargo-oxide's own backend command, and every build of this workspace names it: the generated
# .cargo/cuda-oxide.toml carries `backend = <path>` (tools/oxide-host.sh), and tools/box.sh exports CUDA_OXIDE_BACKEND to
# the same file.
#
# The directory. <root>/<rev>/ holds librustc_codegen_cuda.so, source-rev.txt (the pin rev) and PROVENANCE, whose first line
# is `patch-sha256: <hex> tools/oxide-backend/cuda-rust-1648.patch`. <root> is BLOOMERY_OXIDE_BACKENDS, else
# ${CARGO_HOME:-~/.cargo}/cuda-oxide-bloomery. A file in it is written once, by temp file and mv (a rustc that has the .so
# mapped keeps the old inode). --check refuses by name a backend with no patch line, with the line of another patch, with
# another source rev, or no file: tools/oxide-host.sh runs it before every `cargo oxide` build, so an unpatched backend
# never runs one. A pin move that the patch no longer applies to is refused by name at build time: drop the patch (the new
# pin holds the change) or rebase it.
#
# The build. The pin's checkout is the one cargo made for the cuda-device dependency (`cargo metadata`, as cargo-oxide
# finds it). It is copied to <root>/work-<rev>/src without .git, patched, and built in place with the toolchain its
# cuda-oxide/rust-toolchain.toml names; the build tree stays in <root>/work-<rev>/target.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
PATCH_REL=tools/oxide-backend/cuda-rust-1648.patch

refuse() { # refuse <message>: the one exit for a backend that is not ready
  echo "oxide-backend: $1" >&2
  exit 70
}

pin_rev() { # the first cuda-device rev of Cargo.toml's [workspace.dependencies], the one tools/box.sh reads
  local rev
  rev=$(sed -n 's/^cuda-device = .*rev = "\([0-9a-f]*\)".*/\1/p' "$ROOT/Cargo.toml" | head -n 1)
  [ -n "$rev" ] || refuse "no cuda-oxide rev in $ROOT/Cargo.toml"
  printf '%s\n' "$rev"
}

# A relative root is the tree's: the build changes directory into its copy, and a later --path or --check runs from anywhere.
backends_root() {
  local root=${BLOOMERY_OXIDE_BACKENDS:-${CARGO_HOME:-$HOME/.cargo}/cuda-oxide-bloomery}
  case $root in
    /*) printf '%s\n' "$root" ;;
    *) printf '%s/%s\n' "$ROOT" "$root" ;;
  esac
}

backend_path() { printf '%s/%s/librustc_codegen_cuda.so\n' "$(backends_root)" "$(pin_rev)"; }

sha256() { # sha256 FILE
  if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

patch_sha() {
  [ -f "$ROOT/$PATCH_REL" ] || refuse "no $PATCH_REL in $ROOT"
  sha256 "$ROOT/$PATCH_REL"
}

# check [SO]: SO is the pin's backend and carries this tree's patch; prints its path and the patch sha256 it was built with.
check() {
  local so=${1:-${CUDA_OXIDE_BACKEND:-$(backend_path)}} dir rev want have recorded
  rev=$(pin_rev)
  want=$(patch_sha)
  dir=$(dirname "$so")
  [ -f "$so" ] || refuse "no backend at $so: build it with tools/oxide-backend.sh"
  recorded=$(cat "$dir/source-rev.txt" 2> /dev/null || true)
  [ "$recorded" = "$rev" ] || refuse "the backend at $so records source rev '$recorded', not the pin $rev: build it with tools/oxide-backend.sh"
  have=$(sed -n 's/^patch-sha256: \([0-9a-f]\{64\}\) .*/\1/p' "$dir/PROVENANCE" 2> /dev/null | head -n 1 || true)
  [ -n "$have" ] || refuse "the backend at $so was built without $PATCH_REL (its PROVENANCE names no patch). Unpatched, the cute verifier's time is cubic in a large kernel's block count and bloomery-gpu does not finish: build it with tools/oxide-backend.sh"
  [ "$have" = "$want" ] || refuse "the backend at $so was built with patch sha256 ${have:0:12}, this tree's $PATCH_REL is ${want:0:12}: build it with tools/oxide-backend.sh"
  echo "oxide-backend: $so rev ${rev:0:8} patch ${want:0:12}" >&2
}

# checkout_of_pin: the cuda-rust checkout cargo made for cuda-device (cargo-oxide builds its backend from there).
checkout_of_pin() {
  local meta
  meta=$(cargo metadata --format-version 1 --manifest-path "$ROOT/Cargo.toml" 2> /dev/null) || refuse "\`cargo metadata\` failed in $ROOT: run \`cargo fetch\` there first"
  printf '%s' "$meta" | python3 -I -c '
import json, os, sys
for pkg in json.load(sys.stdin)["packages"]:
    if pkg["name"] == "cuda-device" and "NVIDIA/cuda-rust" in (pkg.get("source") or ""):
        d = os.path.dirname(pkg["manifest_path"])
        while d != os.path.dirname(d):
            if os.path.isfile(os.path.join(d, "cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml")):
                print(d)
                sys.exit(0)
            d = os.path.dirname(d)
sys.exit(1)
' || refuse "cargo metadata names no NVIDIA/cuda-rust checkout with cuda-oxide/crates/rustc-codegen-cuda for cuda-device"
}

build() {
  local rev co root work src target sysroot so out wall start md5 tool
  for tool in cargo python3 patch tar rustc; do
    command -v "$tool" > /dev/null 2>&1 || refuse "\`$tool\` is not on PATH"
  done
  rev=$(pin_rev)
  patch_sha > /dev/null
  co=$(checkout_of_pin)
  if [ -e "$co/.git" ]; then
    [ "$(git -C "$co" rev-parse HEAD 2> /dev/null)" = "$rev" ] || refuse "the checkout $co is not at the pin $rev"
  else
    case $(basename "$co") in "${rev:0:7}"*) ;; *) refuse "the checkout $co is not named for the pin $rev" ;; esac
  fi
  root=$(backends_root)
  work=$root/work-$rev
  src=$work/src
  target=$work/target
  out=$root/$rev
  mkdir -p "$work"
  rm -rf "$src"
  mkdir -p "$src"
  (cd "$co" && tar --exclude=.git -cf - .) | tar -C "$src" -xf -
  patch -p1 -d "$src" --dry-run -s < "$ROOT/$PATCH_REL" > /dev/null 2>&1 \
    || refuse "$PATCH_REL does not apply to cuda-rust $rev: the pin may hold the change already (drop the patch) or have moved the code (rebase it)"
  patch -p1 -d "$src" -s --no-backup-if-mismatch < "$ROOT/$PATCH_REL"
  # cargo-oxide's backend command (crates/cargo-oxide/src/backend.rs backend_build_command_in): the backend crate's own cargo
  # run for the host, with no ambient rustc flags and the toolchain's lib directory on the library path.
  cd "$src/cuda-oxide/crates/rustc-codegen-cuda"
  sysroot=$(rustc --print sysroot)
  start=$(date +%s)
  (
    unset RUSTFLAGS CARGO_BUILD_RUSTFLAGS CARGO_BUILD_TARGET
    export CARGO_ENCODED_RUSTFLAGS= RUSTC_BOOTSTRAP=1 LIBRARY_PATH="$sysroot/lib" LD_LIBRARY_PATH="$sysroot/lib"
    cargo build --lib --target host-tuple --locked --target-dir "$target" --message-format=short
  )
  wall=$(($(date +%s) - start))
  so=$(ls "$target"/*/debug/librustc_codegen_cuda.so 2> /dev/null | head -n 1 || true)
  [ -n "$so" ] && [ -f "$so" ] || refuse "the backend build left no librustc_codegen_cuda.so under $target"
  mkdir -p "$out"
  cp "$so" "$out/.librustc_codegen_cuda.so.tmp"
  mv "$out/.librustc_codegen_cuda.so.tmp" "$out/librustc_codegen_cuda.so"
  md5=$( (md5sum "$out/librustc_codegen_cuda.so" 2> /dev/null || md5 -r "$out/librustc_codegen_cuda.so") | cut -d' ' -f1)
  printf '%s\n' "$rev" > "$out/.source-rev.tmp"
  mv "$out/.source-rev.tmp" "$out/source-rev.txt"
  printf 'patch-sha256: %s %s\nsource: NVIDIA/cuda-rust %s, checkout %s copied without .git and patched\nbuild: cargo build --lib --target host-tuple --locked (cargo-oxide'"'"'s backend command), CARGO_ENCODED_RUSTFLAGS empty, RUSTFLAGS/CARGO_BUILD_RUSTFLAGS/CARGO_BUILD_TARGET unset, LIBRARY_PATH and LD_LIBRARY_PATH %s/lib\ntoolchain: %s\ntarget dir: %s, wall %s s, on %s\nmd5 %s\n' \
    "$(patch_sha)" "$PATCH_REL" "$rev" "$co" "$sysroot" "$(rustc -V)" "$target" "$wall" "$(date -u +%FT%TZ)" "$md5" > "$out/.PROVENANCE.tmp"
  mv "$out/.PROVENANCE.tmp" "$out/PROVENANCE"
  check "$out/librustc_codegen_cuda.so"
}

self_test() {
  local t fails=0 out rc rev=0123456789abcdef0123456789abcdef01234567 d
  t=$(mktemp -d "${TMPDIR:-/tmp}/oxide-backend.XXXXXX")
  trap 'rm -rf "$t"' RETURN
  mkdir -p "$t/tree/tools/oxide-backend" "$t/bin" "$t/ck/${rev:0:7}/cuda-oxide/crates/rustc-codegen-cuda" "$t/ck/${rev:0:7}/cuda-oxide/crates/cuda-device" "$t/backends"
  cp "$0" "$t/tree/tools/oxide-backend.sh"
  printf '[workspace.dependencies]\ncuda-device = { git = "https://github.com/NVIDIA/cuda-rust.git", rev = "%s" }\n' "$rev" > "$t/tree/Cargo.toml"
  printf 'one\ntwo\nthree\n' > "$t/ck/${rev:0:7}/cuda-oxide/stub.txt"
  : > "$t/ck/${rev:0:7}/cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml"
  : > "$t/ck/${rev:0:7}/cuda-oxide/crates/cuda-device/Cargo.toml"
  printf -- '--- a/cuda-oxide/stub.txt\n+++ b/cuda-oxide/stub.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO-PATCHED\n three\n' > "$t/tree/$PATCH_REL"
  # A stub cargo: `metadata` names the stub checkout, `build` leaves a backend holding the patched stub file.
  cat > "$t/bin/cargo" <<EOF
#!/bin/sh
case "\$1" in
  metadata) printf '{"packages":[{"name":"cuda-device","source":"git+https://github.com/NVIDIA/cuda-rust.git?rev=$rev#0123456","manifest_path":"$t/ck/${rev:0:7}/cuda-oxide/crates/cuda-device/Cargo.toml"}]}' ;;
  build)
    while [ "\$#" -gt 0 ]; do [ "\$1" = --target-dir ] && dir=\$2; shift; done
    mkdir -p "\$dir/x86_64-unknown-linux-gnu/debug"
    cat "\$PWD/../../stub.txt" > "\$dir/x86_64-unknown-linux-gnu/debug/librustc_codegen_cuda.so" ;;
esac
EOF
  printf '#!/bin/sh\ncase "$1" in --print) echo "%s/sysroot" ;; *) echo "rustc 0.0.0 (stub)" ;; esac\n' "$t" > "$t/bin/rustc"
  chmod +x "$t/bin/cargo" "$t/bin/rustc"
  run() { # run NAME WANT_RC PATTERN [ENV…] -- ARGS…
    local name=$1 want=$2 pat=$3
    shift 3
    out=$(cd "$t/tree" && env -u CUDA_OXIDE_BACKEND BLOOMERY_OXIDE_BACKENDS="$t/backends" PATH="$t/bin:$PATH" "$@" 2>&1) && rc=0 || rc=$?
    if [ "$rc" = "$want" ] && printf '%s\n' "$out" | grep -Eq -- "$pat"; then
      echo "oxide-backend self-test: ok  $name"
    else
      echo "oxide-backend self-test: BAD $name: rc $rc (want $want), output: $out"
      fails=$((fails + 1))
    fi
  }
  local ob="$t/tree/tools/oxide-backend.sh" psha
  d="$t/backends/$rev"
  run 'the backend of the pin is under the root' 0 "^$t/backends/$rev/librustc_codegen_cuda\.so$" bash "$ob" --path
  run 'no backend is refused by name' 70 'no backend at .*: build it with tools/oxide-backend\.sh' bash "$ob" --check
  run 'the build installs a patched backend' 0 "oxide-backend: $d/librustc_codegen_cuda\.so rev 01234567 patch" bash "$ob"
  if [ "$(cat "$d/librustc_codegen_cuda.so")" = "$(printf 'one\nTWO-PATCHED\nthree')" ]; then
    echo "oxide-backend self-test: ok  the backend was built from the patched copy"
  else
    echo "oxide-backend self-test: BAD the installed backend is not the patched copy: $(cat "$d/librustc_codegen_cuda.so")"
    fails=$((fails + 1))
  fi
  [ "$(cat "$t/ck/${rev:0:7}/cuda-oxide/stub.txt")" = "$(printf 'one\ntwo\nthree')" ] && echo "oxide-backend self-test: ok  the pin's checkout is untouched" || { echo "oxide-backend self-test: BAD the pin's checkout was patched in place"; fails=$((fails + 1)); }
  psha=$(sha256 "$t/tree/$PATCH_REL")
  [ "$(head -n 1 "$d/PROVENANCE" | cut -d' ' -f1-2)" = "patch-sha256: $psha" ] && echo "oxide-backend self-test: ok  PROVENANCE opens with the patch sha256" || { echo "oxide-backend self-test: BAD PROVENANCE: $(head -n 1 "$d/PROVENANCE")"; fails=$((fails + 1)); }
  run 'the built backend passes --check' 0 'oxide-backend: .* rev 01234567 patch' bash "$ob" --check
  run 'CUDA_OXIDE_BACKEND is the file --check reads' 70 'no backend at /nowhere/lib\.so' env CUDA_OXIDE_BACKEND=/nowhere/lib.so bash "$ob" --check
  # the same files with another PROVENANCE: an unpatched backend
  cp "$d/PROVENANCE" "$t/PROVENANCE.good"
  printf 'built from a checkout with the backend command of cargo-oxide\nmd5 0\n' > "$d/PROVENANCE"
  run 'an unpatched backend is refused by name' 70 'built without tools/oxide-backend/cuda-rust-1648\.patch \(its PROVENANCE names no patch\)' bash "$ob" --check
  cp "$t/PROVENANCE.good" "$d/PROVENANCE"
  printf -- '--- a/cuda-oxide/stub.txt\n+++ b/cuda-oxide/stub.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO-OTHER\n three\n' > "$t/tree/$PATCH_REL"
  run 'a backend built with another patch is refused naming both' 70 "built with patch sha256 ${psha:0:12}, this tree's tools/oxide-backend/cuda-rust-1648\.patch is" bash "$ob" --check
  printf -- '--- a/cuda-oxide/stub.txt\n+++ b/cuda-oxide/stub.txt\n@@ -1,3 +1,3 @@\n one\n-two\n+TWO-PATCHED\n three\n' > "$t/tree/$PATCH_REL"
  printf 'ffffffffffffffffffffffffffffffffffffffff\n' > "$d/source-rev.txt"
  run 'a backend of another rev is refused' 70 "records source rev 'f+', not the pin $rev" bash "$ob" --check
  printf '%s\n' "$rev" > "$d/source-rev.txt"
  run 'the restored backend passes' 0 'rev 01234567' bash "$ob" --check
  # a relative root: the build installs under the tree, and a separate --check from elsewhere finds it there
  # the tree as the script sees it ($ROOT is a cd-and-pwd path; a TMPDIR ending in / doubles the slash in $t)
  local tr
  tr=$(cd "$t/tree" && pwd)
  run 'a relative root builds under the tree' 0 "oxide-backend: $tr/rel/$rev/librustc_codegen_cuda\\.so rev" env BLOOMERY_OXIDE_BACKENDS=rel bash "$ob"
  run 'a relative root names the same backend from another directory' 0 'rev 01234567' sh -c "cd / && BLOOMERY_OXIDE_BACKENDS=rel bash '$ob' --check"
  # a patch the pin no longer takes
  printf 'one\nchanged\nthree\n' > "$t/ck/${rev:0:7}/cuda-oxide/stub.txt"
  rm -rf "$t/backends"
  run 'a patch that does not apply is refused by name and installs nothing' 70 'cuda-rust-1648\.patch does not apply to cuda-rust 0123456789abcdef' bash "$ob"
  [ ! -e "$t/backends/$rev/librustc_codegen_cuda.so" ] && echo "oxide-backend self-test: ok  nothing was installed" || { echo "oxide-backend self-test: BAD a backend was installed"; fails=$((fails + 1)); }
  run 'a usage error ends 64' 64 'usage:' bash "$ob" --nonsense
  if [ "$fails" = 0 ]; then
    echo "oxide-backend self-test: all green"
  else
    echo "oxide-backend self-test: $fails red"
    return 1
  fi
}

case "${1-}" in
  '') build ;;
  --path) backend_path ;;
  --check)
    [ "$#" -le 2 ] || { echo "usage: tools/oxide-backend.sh [--path | --check [SO] | --self-test]" >&2; exit 64; }
    check "${2-}"
    ;;
  --self-test) self_test ;;
  *)
    echo "usage: tools/oxide-backend.sh [--path | --check [SO] | --self-test]" >&2
    exit 64
    ;;
esac
