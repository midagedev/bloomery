#!/usr/bin/env bash
# build-k3-quant-tree.sh — the Kimi K3.1 quantization tree: a second checkout of mainline llama.cpp at the K3 oracle's commit
# (78feec062), built CPU-only with the tools the pipeline runs: llama-imatrix, llama-quantize, llama-tokenize and llama-gguf-split (the smoke
# splits a fixture into a split set with it).
#
#   BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh 'bash tools/ref/build-k3-quant-tree.sh'     (just build-k3-quant-tree)
#   BLOOMERY_BOX_READONLY=1 tools/box.sh 'cat /root/lcpp-kimik3-quant/rc; tail -n 5 /root/lcpp-kimik3-quant/log'
#
# The oracle tree /home/user/llama.cpp-kimik3 is never configured, built or edited: the oracle sets name the md5 of its libllama, and
# a rebuild moves it. This tree is a tree of its own (/home/user/llama.cpp-kimik3-quant) built by the one script that builds every
# mainline oracle tree (tools/ref/qvis/build-lcpp-qvis.sh: clone, fetch the pin, detached checkout, cmake, bounded build, pid/log/rc under
# the sentinel directory), called here with this tree's pin, tree, directory and targets. What this wrapper adds is the refusals the
# pipeline needs and the proof about the oracle:
#   - the tree, the sentinel directory and the targets must not be the oracle's: a tree equal to, inside or around the oracle's path is
#     refused (rc 64) before anything runs;
#   - a tree that exists must be at the pin with no tracked change (the generic script refuses it, rc 65), never moved;
#   - the oracle's commit, its `git status --porcelain` and the md5 and mtime of its libllama/libggml libraries are printed before and
#     after, and the build ends rc 65 if any of them moved;
#   - the binaries are printed with their sha256, so the imatrix and quantize runs record which build they ran.
# mtmd is among the targets only because the generic script's post-build check lists libmtmd.so; the tools do not link it.
#
#   K3_ORACLE_TREE   the oracle tree (default /home/user/llama.cpp-kimik3), read only
#   K3_QUANT_TREE    this tree (default /home/user/llama.cpp-kimik3-quant)
#   K3_QUANT_DIR     the sentinel directory: pid, log, rc (default /root/lcpp-kimik3-quant)
#
# Exit: 0 built; 64 a path that would touch the oracle or a usage error; 65 the tree is not at the pin or the oracle moved; 70 the
# lease cannot be tested; 75 the timing lease is held (a sitting runs: start again after it); 2 the build failed (the log is
# $K3_QUANT_DIR/log).
set -uo pipefail
HERE=$(cd "${BASH_SOURCE[0]%/*}" && pwd)
PIN=78feec062d50ec9e8940244cac918ac4480b60cb
ORACLE=${K3_ORACLE_TREE:-/home/user/llama.cpp-kimik3}
TREE=${K3_QUANT_TREE:-/home/user/llama.cpp-kimik3-quant}
DIR=${K3_QUANT_DIR:-/root/lcpp-kimik3-quant}
ORACLE_DIR=/root/lcpp-kimik3
TARGETS=llama,mtmd,llama-common,llama-imatrix,llama-quantize,llama-tokenize,llama-gguf-split

die() { echo "build-k3-quant-tree.sh: $1" >&2; exit "${2:-64}"; }

[ $# = 0 ] || die "takes no arguments (the pin, tree and directory are K3_QUANT_TREE and K3_QUANT_DIR)"

# Canonical forms without requiring the paths to exist (the tree does not before its first build).
canon() {
  local p=$1 d
  case $p in /*) ;; *) die "$p is not an absolute path" ;; esac
  d=$(cd "$(dirname "$p")" 2> /dev/null && pwd -P) || d=$(dirname "$p")
  printf '%s/%s\n' "${d%/}" "$(basename "$p")"
}
O=$(canon "$ORACLE")
T=$(canon "$TREE")
D=$(canon "$DIR")
OD=$(canon "$ORACLE_DIR")
case $T/ in "$O"/* ) die "K3_QUANT_TREE=$T is the oracle tree or inside it: the oracle is never built into" 64 ;; esac
case $O/ in "$T"/* ) die "K3_QUANT_TREE=$T contains the oracle tree $O: a build here would touch it" 64 ;; esac
case $D/ in "$OD"/* ) die "K3_QUANT_DIR=$D is the oracle build's sentinel directory $OD: its rc and log are the oracle's" 64 ;; esac
case $D/ in "$O"/* ) die "K3_QUANT_DIR=$D is inside the oracle tree" 64 ;; esac
[ -d "$O" ] || die "no oracle tree at $O (the pin's reference: $PIN)" 66

# --no-optional-locks: `git status` would otherwise refresh the oracle's .git/index, a write into the tree this script must not touch.
ogit() { git --no-optional-locks -c safe.directory='*' -C "$O" "$@"; }
oracle_state() {
  local libs=() l
  ogit rev-parse HEAD 2> /dev/null || echo "no-git"
  printf 'status-lines=%s\n' "$(ogit status --porcelain 2> /dev/null | wc -l | tr -d ' ')"
  for l in libllama.so.0.8.0 libllama-common.so.0.8.0 libggml.so.0.28.0 libggml-base.so.0.28.0 libggml-cpu.so.0.28.0 libmtmd.so.0.8.0; do
    [ -e "$O/build/bin/$l" ] && libs+=("$O/build/bin/$l")
  done
  [ "${#libs[@]}" -gt 0 ] || { echo "no-libs"; return 0; }
  md5sum "${libs[@]}" 2> /dev/null | sed "s#$O/build/bin/##"
  stat -c '%n mtime=%Y size=%s' "${libs[@]}" 2> /dev/null | sed "s#$O/build/bin/##"
}

BEFORE=$(oracle_state)
echo "build-k3-quant-tree.sh: oracle $O before:"
printf '%s\n' "$BEFORE" | sed 's/^/    /'
[ "$(printf '%s\n' "$BEFORE" | head -1)" = "$PIN" ] || die "the oracle tree is at $(printf '%s\n' "$BEFORE" | head -1), not the pin $PIN: this tree would not be the oracle's commit" 65
[ "$(printf '%s\n' "$BEFORE" | sed -n 2p)" = "status-lines=0" ] || die "the oracle tree has tracked changes: it is not the commit's build, so this tree cannot be said to equal it" 65

echo "build-k3-quant-tree.sh: tree $T, sentinel $D, pin $PIN, targets $TARGETS"
rc=0
LCPP_BUILD_PIN=$PIN LCPP_BUILD_TREE=$T LCPP_BUILD_DIR=$D LCPP_BUILD_CPU_ONLY=1 LCPP_BUILD_TARGETS=$TARGETS \
  bash "$HERE/qvis/build-lcpp-qvis.sh" || rc=$?
# The generic script writes to its own log (exec > log) and its rc file last; its exit status is the same number.
echo "build-k3-quant-tree.sh: generic build script exit $rc; $D/log tail:"
tail -n 8 "$D/log" 2> /dev/null | sed 's/^/    /'
file_rc=$(cat "$D/rc" 2> /dev/null || echo missing)
[ "$rc" = 0 ] && [ "$file_rc" = 0 ] || die "the build did not complete (exit $rc, $D/rc $file_rc): log $D/log" "$([ "$rc" = 0 ] && echo 2 || echo "$rc")"

AFTER=$(oracle_state)
echo "build-k3-quant-tree.sh: oracle after:"
printf '%s\n' "$AFTER" | sed 's/^/    /'
[ "$BEFORE" = "$AFTER" ] || { diff <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER") >&2; die "the oracle tree moved during the build (commit, status, md5 or mtime above): stop and tell the lead" 65; }
echo "build-k3-quant-tree.sh: oracle unchanged (commit, status, md5 and mtime of its libraries)"

have=$(git -c safe.directory='*' -C "$T" rev-parse HEAD) || die "cannot read $T's commit" 2
[ "$have" = "$PIN" ] || die "$T is at $have after the build, not the pin" 65
echo "build-k3-quant-tree.sh: tree $T at $have; status-lines=$(git -c safe.directory='*' -C "$T" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
for b in llama-imatrix llama-quantize llama-tokenize llama-gguf-split; do
  [ -x "$T/build/bin/$b" ] || die "no $T/build/bin/$b after a build that reported success" 2
  printf 'build-k3-quant-tree.sh: %s sha256 %s\n' "$T/build/bin/$b" "$(sha256sum < "$T/build/bin/$b" | cut -d' ' -f1)"
done
printf 'build-k3-quant-tree.sh: %s md5 %s\n' "$T/build/bin/libllama.so.0.8.0" "$(md5sum < "$T/build/bin/libllama.so.0.8.0" 2> /dev/null | cut -d' ' -f1)"
echo "build-k3-quant-tree.sh: done"
