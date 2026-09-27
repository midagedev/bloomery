#!/usr/bin/env bash
# bloomery — the static tier on a second host: tools/box.sh's sibling for the recipes that need
# neither a card, nor data, nor a quiet machine (check, lint, fmt-check and the host gates below).
# It syncs this tree to the second host's WSL and runs the recipe's own box command there, so a
# build beside a timing sitting does not wait for the box's guard. A result from here is NOT
# landing evidence and is recorded in no ledger: the ledger's key has no host in it.
#
#   tools/host2.sh RECIPE       RECIPE is one of ALLOW below; its command is the recipe's
#                               `./tools/box.sh '…'` argument, printed by `tools/recipes.py box-command`
#   tools/host2.sh --self-test  the refusals, the track rule and the rc wrapper, on the Mac, no ssh
#   tools/host2.sh --lists      the allowlist and the refused tokens (read by recipes.py's self-test)
#   tools/host2.sh --check-command CMD   0 when CMD names no refused token, else 64 (the same)
#
#   BLOOMERY_HOST2=<host>       the ssh host, default sihyung-gpu (Windows sshd, PowerShell, WSL Ubuntu)
#   BLOOMERY_HOST2_WAIT=<s>     how long a run waits for the host-wide lock, default 1800; 0 does not
#                               wait. Past it: exit 75, contention
#
# The remote directory is ~/bloomery-<track> in the WSL home, <track> the worktree root's name
# minus `bloomery-` (the main tree is `main`). The sync is one-way (`--delete`, box.sh's excludes):
# never edit there. The remote shell is PowerShell, which parses any command line it is given and
# squashes exit codes to 0/1, so the fixed wrapper WRAP is the only text it sees: the script goes
# over stdin and `exit $LASTEXITCODE` hands the WSL exit code back unchanged. The script sources
# ~/bloomery-env.sh, takes the host-wide flock ~/.bloomery-host2.lock (two tracks' 12-job builds
# would oversubscribe the host) and exports only BLOOMERY_GIT_COMMIT. Nothing is copied back: a run
# that rewrote the remote Cargo.lock says so and exits 65 when its command was green.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ALLOW="check lint fmt-check gate-levers gate-sampler gate-serve gate-threads gate-gpu-gates-lib gate-r8"
# A command naming one of these needs the box: device code, a card's gate lock, data, a model
# profile, or the timing lease.
FORBID=("cargo oxide" "gpu-gate.sh" "BLOOMERY_DATA" "ref-paths" "measure" "lease")
WRAP='wsl -d Ubuntu -- bash -s; exit $LASTEXITCODE'
RSYNC_PATH='wsl -d Ubuntu -- rsync'
# One connection per host, reused by the rsync and the command's ssh and kept 600 s for the next run.
SSH_MUX=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-host2-%C" -o ControlPersist=600)

say() { printf '%s\n' "$*" >&2; }

# 0 when RECIPE is on the allowlist; otherwise a named line and 64.
check_recipe() {
  case " $ALLOW " in
    *" $1 "*) return 0 ;;
  esac
  say "host2.sh: '$1' is not a host2 recipe; the static tier is: $ALLOW"
  return 64
}

# 0 when COMMAND names none of FORBID as a word (bounded by anything but a letter or digit, so
# `--release` does not name `lease`); otherwise a named line and 64.
check_command() {
  local t
  for t in "${FORBID[@]}"; do
    if [[ $1 =~ (^|[^A-Za-z0-9])"$t"([^A-Za-z0-9]|$) ]]; then
      say "host2.sh: the command names '$t', which needs the box: $1"
      return 64
    fi
  done
  return 0
}

# The track of a worktree root: its name minus `bloomery-`, `main` for the main tree.
track_of() {
  local b=${1%/}
  b=${b##*/}
  case $b in
    bloomery) echo main ;;
    bloomery-?*)
      b=${b#bloomery-}
      case $b in
        *[!A-Za-z0-9._-]*) say "host2.sh: the worktree name '$b' has a character outside [A-Za-z0-9._-], which PowerShell may read"; return 64 ;;
      esac
      echo "$b"
      ;;
    *) say "host2.sh: the tree root '$1' is not named bloomery or bloomery-<track>"; return 64 ;;
  esac
}

# box.sh's rsync excludes, one per line, read from its rsync line (box.sh is their one owner).
box_excludes() {
  local line
  line=$(grep -E '^[[:space:]]*rsync -rlpgoDcz --delete ' "$HERE/tools/box.sh") || { say "host2.sh: no rsync line in tools/box.sh to take the excludes from"; return 70; }
  [ "$(printf '%s\n' "$line" | wc -l | tr -d ' ')" = 1 ] || { say "host2.sh: tools/box.sh has more than one rsync line; which one holds the excludes is not decidable"; return 70; }
  printf '%s\n' "$line" | grep -oE -- "--exclude ('[^']*'|[^ '\"]+)" | sed -E "s/^--exclude //; s/^'(.*)'\$/\\1/"
}

# The script the remote bash reads from stdin. Its body is one compound command ending
# `} </dev/null`: bash parses it whole before running it, so a command that reads stdin cannot
# eat the rest of the script.
remote_script() {
  local track=$1 recipe=$2 cmd=$3 commit=$4 lockmd5=$5 wait=$6
  cat << EOF
{
if ! . ~/bloomery-env.sh; then echo "host2.sh: ~/bloomery-env.sh failed on \$(hostname)" >&2; exit 70; fi
cd ~/bloomery-$track || { echo "host2.sh: no ~/bloomery-$track on \$(hostname) after the sync" >&2; exit 66; }
exec 9>> ~/.bloomery-host2.lock
if ! flock -n 9; then
  left=$wait
  [ "\$left" -le 0 ] || echo "[host2] waiting for the host lock ~/.bloomery-host2.lock (last taken by: \$(cat ~/.bloomery-host2.who 2>/dev/null || echo unknown)); bound ${wait}s" >&2
  got=0
  while [ "\$left" -gt 0 ]; do
    step=\$(( left < 60 ? left : 60 ))
    if flock -w "\$step" 9; then got=1; break; fi
    left=\$(( left - step ))
    [ "\$left" -le 0 ] || echo "[host2] still waiting for the host lock, \${left}s left (last taken by: \$(cat ~/.bloomery-host2.who 2>/dev/null || echo unknown))" >&2
  done
  if [ "\$got" != 1 ]; then
    echo "host2.sh: the host lock ~/.bloomery-host2.lock was still held after ${wait}s (last taken by: \$(cat ~/.bloomery-host2.who 2>/dev/null || echo unknown)): contention" >&2
    exit 75
  fi
  echo "[host2] took the host lock" >&2
fi
echo "$track $recipe pid \$\$ since \$(date -u +%FT%TZ)" > ~/.bloomery-host2.who
export BLOOMERY_GIT_COMMIT=$commit
rc=0
(
$cmd
) || rc=\$?
m=\$(md5sum Cargo.lock | cut -d' ' -f1)
if [ "\$m" != $lockmd5 ]; then
  echo "host2: the remote run rewrote Cargo.lock; run \\\`just check\\\` (box) to bring it back, or update the lock" >&2
  [ "\$rc" != 0 ] || rc=65
fi
exit \$rc
} </dev/null
EOF
}

self_test() {
  local fails=0 t out
  fail() { say "host2 self-test FAIL: $*"; fails=$((fails + 1)); }
  # the allowlist: a GPU recipe is refused with 64, a static one passes
  out=$(check_recipe gate-gpu-e2e 2>&1) && fail "gate-gpu-e2e accepted" || { [ $? = 64 ] || fail "gate-gpu-e2e refused with another rc"; }
  case $out in *"not a host2 recipe"*) ;; *) fail "gate-gpu-e2e's refusal does not name the list: $out" ;; esac
  check_recipe lint 2> /dev/null || fail "lint refused"
  # each refused token, fed to the check as a command
  for t in "cargo oxide" "gpu-gate.sh" "BLOOMERY_DATA" "ref-paths" "measure" "lease"; do
    out=$(check_command "cd x && $t y" 2>&1) && fail "a command naming '$t' accepted" || { [ $? = 64 ] || fail "'$t' refused with another rc"; }
  done
  for t in "bash x/lease-hold.sh --card y" "BLOOMERY_DATA=/d cargo test" "bash x/decode-measure.sh"; do
    check_command "$t" 2> /dev/null && fail "a command '$t' accepted"
  done
  check_command "bash tools/gate.sh --release -p bloomery-levers --lib -- --nocapture" 2> /dev/null || fail "a --release gate.sh command refused (a token matched inside a word)"
  # the track rule
  [ "$(track_of /x/bloomery)" = main ] || fail "bloomery -> $(track_of /x/bloomery 2>&1), not main"
  [ "$(track_of /x/bloomery-foo)" = foo ] || fail "bloomery-foo -> $(track_of /x/bloomery-foo 2>&1), not foo"
  [ "$(track_of /x/bloomery-foo-bar/)" = foo-bar ] || fail "bloomery-foo-bar -> $(track_of /x/bloomery-foo-bar 2>&1), not foo-bar"
  track_of /x/other > /dev/null 2>&1 && fail "a root not named bloomery* accepted"
  track_of '/x/bloomery-a$b' > /dev/null 2>&1 && fail "a track name with \$ accepted"
  # the rc passthrough: PowerShell hands back the WSL exit code only through `exit $LASTEXITCODE`
  case $WRAP in *'exit $LASTEXITCODE'*) ;; *) fail "the wrapper lacks 'exit \$LASTEXITCODE': $WRAP" ;; esac
  case $WRAP in *'bash -s'*) ;; *) fail "the wrapper does not read the script from stdin: $WRAP" ;; esac
  # box.sh's excludes, read from box.sh
  out=$(box_excludes) || fail "box_excludes failed"
  printf '%s\n' "$out" | grep -qx 'target/' || fail "the excludes lack target/: $out"
  printf '%s\n' "$out" | grep -qx '.git/' || fail "the excludes lack .git/: $out"
  # the remote script parses, keeps stdin off the command, takes the lock and names 75 and 65
  out=$(remote_script foo lint 'cargo clippy --workspace && echo "$X"' abcd0123 0123456789abcdef0123456789abcdef 1800)
  bash -n <<< "$out" || fail "the remote script does not parse"
  printf '%s\n' "$out" | tail -1 | grep -qx '} </dev/null' || fail "the remote script's body does not end '} </dev/null'"
  for t in 'flock -n 9' 'exit 75' 'rc=65' 'cd ~/bloomery-foo ' 'export BLOOMERY_GIT_COMMIT=abcd0123' 'cargo clippy --workspace && echo "$X"'; do
    case $out in *"$t"*) ;; *) fail "the remote script lacks '$t'" ;; esac
  done
  if [ "$fails" != 0 ]; then
    say "host2: self-test $fails failed"
    return 1
  fi
  echo "host2: self-test ok"
}

case ${1:-} in
  --self-test) self_test; exit $? ;;
  --lists)
    for t in $ALLOW; do echo "allow $t"; done
    for t in "${FORBID[@]}"; do echo "forbid $t"; done
    exit 0
    ;;
  --check-command)
    [ $# = 2 ] || { say "host2.sh: --check-command takes one command"; exit 64; }
    check_command "$2"
    exit $?
    ;;
  '' | -*) say "usage: tools/host2.sh RECIPE | --self-test | --lists (RECIPE: $ALLOW)"; exit 64 ;;
esac
[ $# = 1 ] || { say "host2.sh: one recipe, got $#: $*"; exit 64; }
RECIPE=$1

say "host2: the static tier on a second host — not landing evidence, recorded in no ledger"
check_recipe "$RECIPE" || exit $?
HOST=${BLOOMERY_HOST2:-sihyung-gpu}
case $HOST in
  '' | *[!A-Za-z0-9._@-]*) say "host2.sh: BLOOMERY_HOST2 is an ssh host name, got '$HOST'"; exit 64 ;;
esac
WAIT=${BLOOMERY_HOST2_WAIT:-1800}
case $WAIT in
  '' | *[!0-9]*) say "host2.sh: BLOOMERY_HOST2_WAIT is whole seconds (0: do not wait), got '$WAIT'"; exit 64 ;;
esac
TRACK=$(track_of "$HERE") || exit $?
CMD=$(python3 "$HERE/tools/recipes.py" box-command "$RECIPE") || exit $?
check_command "$CMD" || exit $?
EXCL=()
excl=$(box_excludes) || exit $?
while IFS= read -r x; do [ -z "$x" ] || EXCL+=(--exclude "$x"); done <<< "$excl"
[ ${#EXCL[@]} -gt 0 ] || { say "host2.sh: tools/box.sh's rsync line gave no excludes"; exit 70; }
# box.sh's rule: the short commit, `-dirty` when the tree holds changes the commit does not.
COMMIT=$(git -C "$HERE" rev-parse --short=8 HEAD 2> /dev/null || echo unknown)
[ -z "$(git -C "$HERE" status --porcelain 2> /dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
LOCKMD5=$(md5 -q "$HERE/Cargo.lock" 2> /dev/null || md5sum "$HERE/Cargo.lock" | cut -d' ' -f1)

# The receiver creates ~/bloomery-<track> itself (one level under an existing home), so no mkdir.
rc=0
rsync -rlpgoDcz --delete "${EXCL[@]}" -e "ssh ${SSH_MUX[*]}" --rsync-path="$RSYNC_PATH" "$HERE"/ "$HOST:~/bloomery-$TRACK/" || rc=$?
if [ "$rc" != 0 ]; then
  say "host2.sh: the sync to $HOST:~/bloomery-$TRACK failed (rsync rc $rc)"
  exit "$rc"
fi
remote_script "$TRACK" "$RECIPE" "$CMD" "$COMMIT" "$LOCKMD5" "$WAIT" | ssh "${SSH_MUX[@]}" "$HOST" "$WRAP" || rc=$?
[ "$rc" != 255 ] || say "host2.sh: ssh to $HOST ended with 255 (a connection failure, or the command's own 255)"
exit "$rc"
