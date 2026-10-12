#!/usr/bin/env bash
# Seed a new worktree's target/ from a warm tree's (the Mac, APFS). A tree builds in its own target/ (tools/mac-check.sh
# sets it: a shared directory would serve one tree's rmeta to another), so a new worktree starts cold. This clones a warm
# tree's target/ into it and gives the unchanged source files the seed's mtimes, so cargo finds the units they feed fresh.
#
#   tools/seed-target.sh SRC_TREE DST_TREE   once, right after `git worktree add`, before DST's first cargo command
#   tools/seed-target.sh --self-test         two scratch trees under $TMPDIR; the refusals; no cargo
#
# 1. SRC_TREE/target is cloned into DST_TREE/target with `cp -c` (APFS clonefile: the files share blocks, so the
#    copy costs no disk until cargo rewrites a file; the mtimes come along).
# 2. Every file of DST_TREE (not target/, not .git) whose content equals the file of the same path in SRC_TREE gets that
#    file's mtime. A file that differs, or that SRC_TREE lacks, keeps its own mtime, so cargo rebuilds exactly the
#    units it feeds. A fresh checkout stamps every file "now", which cargo reads as every unit changed.
# 3. It prints the units (fingerprint directories, both cargo build-dir layouts) and the bytes of DST_TREE/target.
#
# SRC_TREE should be a tree whose last build was green, on the same toolchain and Cargo.lock: cargo's own fingerprints
# rebuild whatever a toolchain, feature or lock difference changes, and a source edit of SRC_TREE since its build
# stays an edit (its mtime is carried over, newer than the build). The script reads SRC_TREE and writes only
# DST_TREE/target and the mtimes of DST_TREE's files; it runs no cargo and no git.
#
# Exit: 0 seeded. 64 an argument (count, not a directory, the same tree). 65 DST_TREE already has a target/ (it is never
# merged into or replaced). 66 SRC_TREE has no target/. 67 the two trees are on different volumes (a clone cannot
# cross one). 69 not macOS, or DST_TREE's volume is not APFS (cp -c would copy every byte, which is what this avoids).
# 70 the clone or the mtime pass failed (the target/ it created is removed).
# Runs under the Mac's bash 3.2: no associative arrays, no mapfile.
set -euo pipefail

say() { printf 'seed-target: %s\n' "$*"; }

# dev_of PATH: the device id of the volume PATH is on.
dev_of() {
  if [ "$(uname)" = Darwin ]; then stat -f %d "$1"; else stat -c %d "$1"; fi
}

# fs_of PATH: the filesystem type of the volume PATH is on, from mount(8).
fs_of() {
  local dev
  dev=$(df -P "$1" | awk 'NR == 2 { print $1 }')
  mount | awk -v d="$dev" '$1 == d { sub(/^[^(]*\(/, ""); sub(/,.*/, ""); print; exit }'
}

# mtime_of PATH: the mtime in whole seconds.
mtime_of() {
  if [ "$(uname)" = Darwin ]; then stat -f %m "$1"; else stat -c %Y "$1"; fi
}

# restore_mtimes SRC DST: the mtime pass of the header, then the unit and byte count of DST/target.
restore_mtimes() {
  python3 - "$1" "$2" << 'PY'
import os, sys

src, dst = sys.argv[1], sys.argv[2]
restored = total = 0
for root, dirs, files in os.walk(dst):
    if root == dst:
        dirs[:] = [d for d in dirs if d not in ("target", ".git")]
    for name in files:
        path = os.path.join(root, name)
        st = os.lstat(path)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        total += 1
        other = os.path.join(src, os.path.relpath(path, dst))
        try:
            ot = os.lstat(other)
        except OSError:
            continue
        if not os.path.isfile(other) or os.path.islink(other) or ot.st_size != st.st_size:
            continue
        with open(path, "rb") as a, open(other, "rb") as b:
            if a.read() != b.read():
                continue
        if ot.st_mtime_ns != st.st_mtime_ns:
            os.utime(path, ns=(st.st_atime_ns, ot.st_mtime_ns))
        restored += 1
units = size = 0
for root, dirs, files in os.walk(os.path.join(dst, "target")):
    base = os.path.basename(root)
    if base == ".fingerprint":
        units += len(dirs)
    elif base == "fingerprint" and os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(root)))) == "build":
        units += 1
    for name in files:
        st = os.lstat(os.path.join(root, name))
        if not os.path.islink(os.path.join(root, name)):
            size += st.st_size
print(f"seed-target: mtime of {restored} of {total} source files restored to the seed's ({total - restored} differ or are new and keep their own)")
print(f"seed-target: seeded {units} units, {size} bytes")
PY
}

# seed_main SRC DST: the whole flow; every refusal is a named line on stderr and its rc.
seed_main() {
  local src dst
  [ $# = 2 ] || { say "usage: tools/seed-target.sh SRC_TREE DST_TREE | --self-test" >&2; return 64; }
  [ -d "$1" ] || { say "$1 is not a directory" >&2; return 64; }
  [ -d "$2" ] || { say "$2 is not a directory" >&2; return 64; }
  src=$(cd "$1" && pwd -P)
  dst=$(cd "$2" && pwd -P)
  [ "$src" != "$dst" ] || { say "SRC_TREE and DST_TREE are the same tree: $src" >&2; return 64; }
  if [ -e "$dst/target" ] || [ -L "$dst/target" ]; then
    say "refused: $dst already has a target/ (a seed is never merged into one: remove it, or build there)" >&2
    return 65
  fi
  [ -d "$src/target" ] || { say "refused: $src has no target/ to seed from" >&2; return 66; }
  if [ "$(dev_of "$src/target")" != "$(dev_of "$dst")" ]; then
    say "refused: $src/target and $dst are on different volumes (a clone cannot cross one)" >&2
    return 67
  fi
  if [ "$(uname)" != Darwin ]; then
    say "refused: not macOS (cp -c clones on APFS only)" >&2
    return 69
  fi
  if [ "$(fs_of "$dst")" != apfs ]; then
    say "refused: the volume of $dst is '$(fs_of "$dst")', not apfs (cp -c would copy every byte)" >&2
    return 69
  fi
  if ! cp -cRp "$src/target" "$dst/target"; then
    rm -rf "$dst/target"
    say "the clone of $src/target into $dst/target failed; the partial target/ is removed" >&2
    return 70
  fi
  if ! restore_mtimes "$src" "$dst"; then
    rm -rf "$dst/target"
    say "the mtime pass failed; the target/ is removed" >&2
    return 70
  fi
}

self_test() {
  local fails=0 t src dst out rc a b
  fail() { say "self-test FAIL: $*"; fails=$((fails + 1)); }
  t=$(mktemp -d "${TMPDIR:-/tmp}/seed-target-selftest.XXXXXX")
  trap "rm -rf '$t'" EXIT
  src=$t/src
  dst=$t/dst
  # SRC: an unchanged file, a file that DST edits to the same size, a file only SRC has, and a warm target/ in both
  # build-dir layouts (3 units, 64 + 36 + 5 bytes) with fixed mtimes.
  mkdir -p "$src/crates/a" "$src/.git" "$src/target/debug/build/p/h1/fingerprint" "$src/target/debug/build/q/h2/fingerprint" \
    "$src/target/debug/.fingerprint/old-unit" "$dst/crates/a" "$dst/.git"
  printf 'fn same() {}\n' > "$src/crates/a/same.rs"
  printf 'fn old_ver() {}\n' > "$src/crates/a/edited.rs"
  printf 'fn only_src() {}\n' > "$src/crates/a/only-src.rs"
  printf 'ref\n' > "$src/.git/HEAD"
  printf '%064d' 0 > "$src/target/debug/build/p/h1/fingerprint/lib-p"
  printf '%036d' 0 > "$src/target/debug/build/q/h2/fingerprint/lib-q"
  printf '%05d' 0 > "$src/target/debug/.fingerprint/old-unit/lib-old"
  find "$src" -exec touch -t 202001010000 {} +
  # DST: written now, so every mtime is later than the seed's
  sleep 1
  printf 'fn same() {}\n' > "$dst/crates/a/same.rs"
  printf 'fn new_ver() {}\n' > "$dst/crates/a/edited.rs"
  printf 'fn only_dst() {}\n' > "$dst/crates/a/only-dst.rs"
  printf 'ref\n' > "$dst/.git/HEAD"
  local seed_m edited_m only_dst_m git_m
  seed_m=$(mtime_of "$src/crates/a/same.rs")
  edited_m=$(mtime_of "$dst/crates/a/edited.rs")
  only_dst_m=$(mtime_of "$dst/crates/a/only-dst.rs")
  git_m=$(mtime_of "$dst/.git/HEAD")
  [ "$edited_m" != "$seed_m" ] || fail "the scratch trees share an mtime, the cases below prove nothing"

  # the refusals, each with its rc and its named line; none leaves a target/ behind
  refused() { # WHAT RC WANT_RC WANT_LINE OUT: the rc, and the named line in the output
    [ "$2" = "$3" ] || fail "$1 ended rc $2, not $3: $5"
    case $5 in *"$4"*) ;; *) fail "$1 does not name '$4': $5" ;; esac
  }
  other_volume() { if [ "${1#"$src"/}" != "$1" ]; then echo 1; else echo 2; fi; }
  rc=0; out=$(seed_main "$src" 2>&1) || rc=$?
  refused "one argument" "$rc" 64 "usage:" "$out"
  rc=0; out=$(seed_main "$src" "$t/missing" 2>&1) || rc=$?
  refused "a missing DST" "$rc" 64 "is not a directory" "$out"
  rc=0; out=$(seed_main "$src" "$src" 2>&1) || rc=$?
  refused "the same tree" "$rc" 64 "the same tree" "$out"
  rc=0; out=$(seed_main "$dst" "$src" 2>&1) || rc=$?
  refused "a DST with a target" "$rc" 65 "already has a target/" "$out"
  rc=0; out=$(seed_main "$dst" "$t" 2>&1) || rc=$?
  refused "a SRC without a target" "$rc" 66 "has no target/ to seed from" "$out"
  rc=0; out=$(dev_of() { other_volume "$@"; }; seed_main "$src" "$dst" 2>&1) || rc=$?
  refused "two volumes" "$rc" 67 "different volumes" "$out"
  rc=0; out=$(uname() { echo Linux; }; dev_of() { echo 1; }; seed_main "$src" "$dst" 2>&1) || rc=$?
  refused "a non-macOS host" "$rc" 69 "not macOS" "$out"
  rc=0; out=$(uname() { echo Darwin; }; dev_of() { echo 1; }; fs_of() { echo hfs; }; seed_main "$src" "$dst" 2>&1) || rc=$?
  refused "a non-APFS volume" "$rc" 69 "not apfs" "$out"
  [ ! -e "$dst/target" ] || fail "a refused run left a target/ in DST"

  # the seed: on macOS the whole flow, elsewhere the mtime pass on a plain copy of target/
  if [ "$(uname)" = Darwin ]; then
    rc=0; out=$(seed_main "$src" "$dst" 2>&1) || rc=$?
    [ "$rc" = 0 ] || fail "the seed ended rc $rc: $out"
  else
    cp -Rp "$src/target" "$dst/target"
    rc=0; out=$(restore_mtimes "$src" "$dst" 2>&1) || rc=$?
    [ "$rc" = 0 ] || fail "the mtime pass ended rc $rc: $out"
  fi
  case $out in *"seeded 3 units, 105 bytes"*) ;; *) fail "the units and bytes line is not 'seeded 3 units, 105 bytes': $out" ;; esac
  case $out in *"mtime of 1 of 3 source files restored"*) ;; *) fail "the restore count is not 1 of 3 (target/ and .git are no source): $out" ;; esac
  [ "$(mtime_of "$dst/crates/a/same.rs")" = "$seed_m" ] || fail "an equal file did not get the seed's mtime"
  [ "$(mtime_of "$dst/crates/a/edited.rs")" = "$edited_m" ] || fail "a same-size file with other content was touched"
  [ "$(mtime_of "$dst/crates/a/only-dst.rs")" = "$only_dst_m" ] || fail "a file the seed lacks was touched"
  [ "$(mtime_of "$dst/.git/HEAD")" = "$git_m" ] || fail ".git was touched"
  [ "$(mtime_of "$dst/target/debug/build/p/h1/fingerprint/lib-p")" = "$seed_m" ] || fail "the target/ file lost the seed's mtime"
  cmp -s "$src/target/debug/build/q/h2/fingerprint/lib-q" "$dst/target/debug/build/q/h2/fingerprint/lib-q" || fail "the target/ content differs from the seed's"
  rc=0; out=$(seed_main "$src" "$dst" 2>&1) || rc=$?
  [ "$rc" = 65 ] || fail "a second seed ended rc $rc, not 65: $out"

  if [ "$fails" != 0 ]; then
    say "self-test $fails failed"
    return 1
  fi
  echo "seed-target: self-test ok"
}

case ${1:-} in
  --self-test) [ $# = 1 ] || { say "--self-test takes nothing"; exit 64; }; self_test; exit $? ;;
esac
seed_main "$@"
