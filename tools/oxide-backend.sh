#!/usr/bin/env bash
# The cuda-oxide codegen backend bloomery builds with: the pin's backend source, built here with the release profile.
#   tools/oxide-backend.sh                build the backend for the pin and install it in its directory
#   tools/oxide-backend.sh --path         where the backend of the pin is installed
#   tools/oxide-backend.sh --check [SO]   SO (default: CUDA_OXIDE_BACKEND, else the installed one) is the pin's backend
#   tools/oxide-backend.sh --self-test
# Exit 0 when done; 70 names what is missing or wrong and nothing is installed past it; 64 is a usage error.
#
# Why built here. cargo-oxide builds the backend in the dev profile, where the device lowering of a crate as large as
# bloomery-gpu runs several times slower; the profile changes the compiler's speed, not the PTX it writes. It also keeps one
# backend cache slot that every track's build rewrites in place. So the backend is built here, by cargo-oxide's own backend
# command plus --release, in a directory of its own per rev, and every build of this workspace names it: the generated
# .cargo/cuda-oxide.toml carries `backend = <path>` (tools/oxide-host.sh), and tools/box.sh exports CUDA_OXIDE_BACKEND to the
# same file.
#
# The directory. <root>/<rev>/ holds librustc_codegen_cuda.so, source-rev.txt (the pin rev) and PROVENANCE (the source, the
# command, the toolchain, the wall, the md5). <root> is BLOOMERY_OXIDE_BACKENDS, else ${CARGO_HOME:-~/.cargo}/cuda-oxide-bloomery.
# A file in it is written once, by temp file and mv (a rustc that has the .so mapped keeps the old inode). --check refuses by
# name a backend with another source rev or no file: tools/oxide-host.sh runs it before every `cargo oxide` build, so a build
# never runs on the backend of another rev.
#
# The build. The pin's checkout is the one cargo made for the cuda-device dependency (`cargo metadata`, as cargo-oxide finds
# it). Its cuda-oxide/crates/rustc-codegen-cuda is built in place, as cargo-oxide builds it, with the toolchain its
# cuda-oxide/rust-toolchain.toml names; the build tree stays in <root>/work-<rev>/target.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)

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

# check [SO]: SO is the pin's backend; prints its path.
check() {
  local so=${1:-${CUDA_OXIDE_BACKEND:-$(backend_path)}} dir rev recorded
  rev=$(pin_rev)
  dir=$(dirname "$so")
  [ -f "$so" ] || refuse "no backend at $so: build it with tools/oxide-backend.sh"
  recorded=$(cat "$dir/source-rev.txt" 2> /dev/null || true)
  [ "$recorded" = "$rev" ] || refuse "the backend at $so records source rev '$recorded', not the pin $rev: build it with tools/oxide-backend.sh"
  echo "oxide-backend: $so rev ${rev:0:8}" >&2
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
  local rev co root work target sysroot so out wall start md5 tool
  for tool in cargo python3 rustc; do
    command -v "$tool" > /dev/null 2>&1 || refuse "\`$tool\` is not on PATH"
  done
  rev=$(pin_rev)
  co=$(checkout_of_pin)
  if [ -e "$co/.git" ]; then
    [ "$(git -C "$co" rev-parse HEAD 2> /dev/null)" = "$rev" ] || refuse "the checkout $co is not at the pin $rev"
  else
    case $(basename "$co") in "${rev:0:7}"*) ;; *) refuse "the checkout $co is not named for the pin $rev" ;; esac
  fi
  root=$(backends_root)
  work=$root/work-$rev
  target=$work/target
  out=$root/$rev
  # cargo-oxide's backend command (crates/cargo-oxide/src/backend.rs backend_build_command_in): the backend crate's own cargo
  # run for the host, with no ambient rustc flags and the toolchain's lib directory on the library path. Plus --release:
  # cargo-oxide builds the backend in the dev profile, where the device lowering of a crate as large as bloomery-gpu runs
  # several times slower; the profile changes the compiler's speed, not the PTX it writes.
  cd "$co/cuda-oxide/crates/rustc-codegen-cuda"
  sysroot=$(rustc --print sysroot)
  start=$(date +%s)
  (
    unset RUSTFLAGS CARGO_BUILD_RUSTFLAGS CARGO_BUILD_TARGET
    export CARGO_ENCODED_RUSTFLAGS= RUSTC_BOOTSTRAP=1 LIBRARY_PATH="$sysroot/lib" LD_LIBRARY_PATH="$sysroot/lib"
    cargo build --lib --release --target host-tuple --locked --target-dir "$target" --message-format=short
  )
  wall=$(($(date +%s) - start))
  so=$(ls "$target"/*/release/librustc_codegen_cuda.so 2> /dev/null | head -n 1 || true)
  [ -n "$so" ] && [ -f "$so" ] || refuse "the backend build left no librustc_codegen_cuda.so under $target"
  mkdir -p "$out"
  cp "$so" "$out/.librustc_codegen_cuda.so.tmp"
  mv "$out/.librustc_codegen_cuda.so.tmp" "$out/librustc_codegen_cuda.so"
  md5=$( (md5sum "$out/librustc_codegen_cuda.so" 2> /dev/null || md5 -r "$out/librustc_codegen_cuda.so") | cut -d' ' -f1)
  printf '%s\n' "$rev" > "$out/.source-rev.tmp"
  mv "$out/.source-rev.tmp" "$out/source-rev.txt"
  printf 'source: NVIDIA/cuda-rust %s, checkout %s\nbuild: cargo build --lib --release --target host-tuple --locked (cargo-oxide'"'"'s backend command plus --release), CARGO_ENCODED_RUSTFLAGS empty, RUSTFLAGS/CARGO_BUILD_RUSTFLAGS/CARGO_BUILD_TARGET unset, LIBRARY_PATH and LD_LIBRARY_PATH %s/lib\ntoolchain: %s\ntarget dir: %s, wall %s s, on %s\nmd5 %s\n' \
    "$rev" "$co" "$sysroot" "$(rustc -V)" "$target" "$wall" "$(date -u +%FT%TZ)" "$md5" > "$out/.PROVENANCE.tmp"
  mv "$out/.PROVENANCE.tmp" "$out/PROVENANCE"
  check "$out/librustc_codegen_cuda.so"
}

self_test() {
  local t fails=0 out rc rev=0123456789abcdef0123456789abcdef01234567 d
  t=$(mktemp -d "${TMPDIR:-/tmp}/oxide-backend.XXXXXX")
  trap 'rm -rf "$t"' RETURN
  mkdir -p "$t/tree/tools" "$t/bin" "$t/ck/${rev:0:7}/cuda-oxide/crates/rustc-codegen-cuda" "$t/ck/${rev:0:7}/cuda-oxide/crates/cuda-device" "$t/backends"
  cp "$0" "$t/tree/tools/oxide-backend.sh"
  printf '[workspace.dependencies]\ncuda-device = { git = "https://github.com/NVIDIA/cuda-rust.git", rev = "%s" }\n' "$rev" > "$t/tree/Cargo.toml"
  printf 'one\ntwo\nthree\n' > "$t/ck/${rev:0:7}/cuda-oxide/stub.txt"
  : > "$t/ck/${rev:0:7}/cuda-oxide/crates/rustc-codegen-cuda/Cargo.toml"
  : > "$t/ck/${rev:0:7}/cuda-oxide/crates/cuda-device/Cargo.toml"
  # A stub cargo: `metadata` names the stub checkout (the directory $CK_NAME, default the pin's), `build` leaves a backend
  # holding the stub file of the directory it runs in.
  cat > "$t/bin/cargo" <<EOF
#!/bin/sh
case "\$1" in
  metadata) printf '{"packages":[{"name":"cuda-device","source":"git+https://github.com/NVIDIA/cuda-rust.git?rev=$rev#0123456","manifest_path":"%s/ck/%s/cuda-oxide/crates/cuda-device/Cargo.toml"}]}' "$t" "\${CK_NAME:-${rev:0:7}}" ;;
  build)
    while [ "\$#" -gt 0 ]; do [ "\$1" = --target-dir ] && dir=\$2; shift; done
    mkdir -p "\$dir/x86_64-unknown-linux-gnu/release"
    cat "\$PWD/../../stub.txt" > "\$dir/x86_64-unknown-linux-gnu/release/librustc_codegen_cuda.so" ;;
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
  local ob="$t/tree/tools/oxide-backend.sh"
  d="$t/backends/$rev"
  run 'the backend of the pin is under the root' 0 "^$t/backends/$rev/librustc_codegen_cuda\.so$" bash "$ob" --path
  run 'no backend is refused by name' 70 'no backend at .*: build it with tools/oxide-backend\.sh' bash "$ob" --check
  run 'the build installs the backend of the pin' 0 "oxide-backend: $d/librustc_codegen_cuda\.so rev 01234567\$" bash "$ob"
  if [ "$(cat "$d/librustc_codegen_cuda.so")" = "$(printf 'one\ntwo\nthree')" ]; then
    echo "oxide-backend self-test: ok  the backend was built from the pin's checkout"
  else
    echo "oxide-backend self-test: BAD the installed backend is not the checkout's build: $(cat "$d/librustc_codegen_cuda.so")"
    fails=$((fails + 1))
  fi
  [ "$(head -n 1 "$d/PROVENANCE")" = "source: NVIDIA/cuda-rust $rev, checkout $t/ck/${rev:0:7}" ] && echo "oxide-backend self-test: ok  PROVENANCE opens with the source rev and checkout" || { echo "oxide-backend self-test: BAD PROVENANCE: $(head -n 1 "$d/PROVENANCE")"; fails=$((fails + 1)); }
  run 'the built backend passes --check' 0 'oxide-backend: .* rev 01234567$' bash "$ob" --check
  run 'CUDA_OXIDE_BACKEND is the file --check reads' 70 'no backend at /nowhere/lib\.so' env CUDA_OXIDE_BACKEND=/nowhere/lib.so bash "$ob" --check
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
  # a checkout that is not the pin's
  cp -R "$t/ck/${rev:0:7}" "$t/ck/fffffff"
  rm -rf "$t/backends"
  run 'a checkout of another rev is refused by name and installs nothing' 70 "the checkout $t/ck/fffffff is not named for the pin $rev" env CK_NAME=fffffff bash "$ob"
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
