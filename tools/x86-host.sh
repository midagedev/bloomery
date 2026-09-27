#!/usr/bin/env bash
# bloomery — x86 host tests on the second host (sihyung-gpu: a Windows PC, WSL Ubuntu, 20 CPUs, the
# owner's CI runners beside us). The Mac runs the static tier (tools/mac-check.sh) but cannot run a test
# that uses x86 intrinsics or Linux APIs; this host runs those, and nothing else. A result from here is
# NOT landing evidence and is recorded in no ledger: the ledger's key has no host in it.
#
#   tools/x86-host.sh RECIPE          RECIPE is one of ALLOW below; its command is the recipe's
#                                     `./tools/box.sh '…'` argument (`tools/recipes.py box-command`)
#   tools/x86-host.sh test CRATE…     `bash tools/gate.sh --release -p CRATE…` (cargo test under the gate
#                                     bound); a crate whose closure reaches a device root is refused. With
#                                     no crate, every crate that does not (`recipes.py x86-crates --names`)
#   tools/x86-host.sh --tracks [--remove NAME…]   the host's ~/bloomery-* against the local worktrees;
#                                     removes only named, still stale ones (tools/box-tracks.sh's rules)
#   tools/x86-host.sh --self-test     the refusals, the disk guard's arithmetic, the target cap's
#                                     selection, the remote script, on the Mac, no ssh
#   tools/x86-host.sh --check-allowlist CLASSES   ALLOW against `gate-batch.sh --classes` output and the
#                                     data rule (`recipes.py x86-allowed`; check-recipes runs it)
#
#   BLOOMERY_X86_HOST=<host>   the ssh host, default sihyung-gpu (Windows sshd, PowerShell, WSL Ubuntu)
#   BLOOMERY_X86_WAIT=<s>      how long a run waits for the host lock, default 1800; 0 does not wait.
#                              Past it: exit 75, contention
#
# Every run, before the sync, reads three numbers in one PowerShell line: WSL's `df -B1 /` used, the
# distro's ext4.vhdx size (its BasePath in the registry) and the free bytes of the drive holding it. The
# vhdx grows into that drive and does not shrink when files inside are deleted, so a run is refused (73)
# when `used + MARGIN > vhdx` (the vhdx may have to grow) and `free < MARGIN` (the drive cannot take it);
# numbers that cannot be read are refused too (74). The Windows side is only read.
#
# The remote directory is ~/bloomery-<track> in the WSL home, <track> the worktree root's name minus
# `bloomery-` (the main tree is `main`); its target/ is the track's own (CARGO_TARGET_DIR is unset: a
# shared one served another tree's rmeta). The sync is one-way (box.sh's excludes, `--delete`): never
# edit there. PowerShell parses any command line it is given and squashes exit codes to 0/1, so the fixed
# WRAP is the only text it sees with the run: the script goes over stdin and `exit $LASTEXITCODE` hands
# the WSL exit code back. The script takes the host-wide flock ~/.bloomery-host2.lock (the one lock every
# build on this host takes), runs under nice 10, stamps target/.x86-last-use, and at its end, still
# holding the lock, removes the targets beyond the newest KEEP (never its own, never one a process runs
# in: tools/box-gc.sh --check). Nothing is copied back: a run that rewrote Cargo.lock exits 65 when its
# command was green.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
ALLOW="gate-levers gate-sampler gate-serve gate-threads gate-gpu-gates-lib gate-r8"
# The static tier runs on the Mac.
STATIC="check lint fmt fmt-check mac-check mac-lint mac-fmt-check"
# A command naming one of these needs the box: device code, a card's gate lock, data, the lease.
FORBID=("cargo oxide" "gpu-gate.sh" "BLOOMERY_DATA" "ref-paths" "measure" "lease")
DISTRO=Ubuntu
WRAP="wsl -d $DISTRO -- bash -s; exit \$LASTEXITCODE"
RSYNC_PATH="wsl -d $DISTRO -- rsync"
# One connection per host, reused by the guard, the rsync and the command, kept 600 s for the next run.
SSH_MUX=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-x86-%C" -o ControlPersist=600)
KEEP=6
MARGIN=20000000000
RC_DISK=73
RC_GUARD=74
# The disk guard's one PowerShell line: the distro's vhdx by its registry BasePath, the free bytes of
# that path's drive, and WSL's used bytes on /. Reads only.
PS_DISK='$p = @(Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | ForEach-Object { Get-ItemProperty $_.PSPath } | Where-Object { $_.DistributionName -eq "'"$DISTRO"'" }); if ($p.Count -ne 1) { "x86-disk error: $($p.Count) registered distros named '"$DISTRO"'"; exit 3 }; $v = Join-Path $p[0].BasePath $p[0].VhdFileName; $d = (Split-Path -Qualifier $v).TrimEnd(":"); $u = (wsl -d '"$DISTRO"' -- df -B1 --output=used / | Select-Object -Last 1); "x86-disk used=$(([string]$u).Trim()) vhdx=$((Get-Item -LiteralPath $v).Length) free=$((Get-PSDrive $d).Free) drive=$d"'

say() { printf '%s\n' "$*" >&2; }

# 0 when RECIPE is on the allowlist; otherwise a named line and 64.
check_recipe() {
  case " $STATIC " in
    *" $1 "*)
      say "x86-host.sh: '$1' is the static tier, which runs on the Mac: just mac-check, just mac-lint, just mac-fmt-check (tools/mac-check.sh)"
      return 64
      ;;
  esac
  case " $ALLOW " in
    *" $1 "*) return 0 ;;
  esac
  say "x86-host.sh: '$1' is not an x86 host recipe; the host runs: $ALLOW (and test CRATE…)"
  return 64
}

# 0 when COMMAND names none of FORBID as a word (bounded by anything but a letter or digit, so
# `--release` does not name `lease`); otherwise a named line and 64.
check_command() {
  local t
  for t in "${FORBID[@]}"; do
    if [[ $1 =~ (^|[^A-Za-z0-9])"$t"([^A-Za-z0-9]|$) ]]; then
      say "x86-host.sh: the command names '$t', which needs the box: $1"
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
        *[!A-Za-z0-9._-]*) say "x86-host.sh: the worktree name '$b' has a character outside [A-Za-z0-9._-], which PowerShell may read"; return 64 ;;
      esac
      echo "$b"
      ;;
    *) say "x86-host.sh: the tree root '$1' is not named bloomery or bloomery-<track>"; return 64 ;;
  esac
}

# box.sh's rsync excludes, one per line, read from its rsync line (box.sh is their one owner).
box_excludes() {
  local line
  line=$(grep -E '^[[:space:]]*rsync -rlpgoDcz --delete ' "$HERE/tools/box.sh") || { say "x86-host.sh: no rsync line in tools/box.sh to take the excludes from"; return 70; }
  [ "$(printf '%s\n' "$line" | wc -l | tr -d ' ')" = 1 ] || { say "x86-host.sh: tools/box.sh has more than one rsync line; which one holds the excludes is not decidable"; return 70; }
  printf '%s\n' "$line" | grep -oE -- "--exclude ('[^']*'|[^ '\"]+)" | sed -E "s/^--exclude //; s/^'(.*)'\$/\\1/"
}

# The guard's output: one `x86-disk used=N vhdx=N free=N drive=D` line among whatever else came back.
# Prints "USED VHDX FREE DRIVE", or a named line and RC_GUARD.
parse_disk() {
  local line
  line=$(printf '%s\n' "$1" | tr -d '\r' | grep -E '^x86-disk ' || true)
  if [[ ! $line =~ ^x86-disk\ used=([0-9]+)\ vhdx=([0-9]+)\ free=([0-9]+)\ drive=([A-Za-z])$ ]]; then
    say "x86-host.sh: the disk guard could not read its three numbers, so nothing runs; the host said: $(printf '%s' "$1" | tr -d '\r' | tr '\n' ' ' | cut -c1-400)"
    return $RC_GUARD
  fi
  echo "${BASH_REMATCH[1]} ${BASH_REMATCH[2]} ${BASH_REMATCH[3]} ${BASH_REMATCH[4]}"
}

gb() { awk -v b="$1" 'BEGIN { printf "%.2f GB", b / 1e9 }'; }

# The header line, and 0, or the refusal and RC_DISK: refused when the run could make the vhdx grow
# (used + MARGIN > vhdx) into a drive that cannot take it (free < MARGIN).
disk_verdict() {
  local used=$1 vhdx=$2 free=$3 drive=$4
  echo "[x86] disk: WSL / used $used B ($(gb "$used")), ext4.vhdx $vhdx B ($(gb "$vhdx")), $drive: free $free B ($(gb "$free")); margin $(gb $MARGIN)"
  if [ $((used + MARGIN)) -gt "$vhdx" ] && [ "$free" -lt "$MARGIN" ]; then
    say "x86-host.sh: refused — the run could grow the vhdx (used $used + margin $MARGIN > vhdx $vhdx) and $drive: has $free B free, under the margin: free space on $drive: first"
    return $RC_DISK
  fi
  return 0
}

# The target cap's choice. stdin: "<last-use epoch> <track>" per target; $1 KEEP, $2 the running track,
# $3 the track the host lock's record names. Prints "keep <track> <why>" or "rm <track>" per line,
# newest first: the running track takes one of the KEEP places whatever its age, the newest KEEP - 1
# others take the rest, the lock's holder is always kept, the rest go.
lru_select() {
  local keep=$1 cur=$2 holder=$3 n=0 stamp t
  sort -k1,1nr -k2,2 | {
    while read -r stamp t; do
      [ -n "$t" ] || continue
      if [ "$t" = "$cur" ]; then
        echo "keep $t running"
      elif [ "$n" -lt $((keep - 1)) ]; then
        echo "keep $t newest"
        n=$((n + 1))
      elif [ "$t" = "$holder" ]; then
        echo "keep $t lock-holder"
      else
        echo "rm $t"
      fi
    done
  }
}

# The script the remote bash reads from stdin. Its body is one compound command ending `} </dev/null`:
# bash parses it whole before running it, so a command that reads stdin cannot eat the rest.
remote_script() {
  local track=$1 what=$2 cmd=$3 commit=$4 lockmd5=$5 wait=$6
  cat << EOF
{
$(declare -f lru_select)
if ! . ~/bloomery-env.sh; then echo "x86-host.sh: ~/bloomery-env.sh failed on \$(hostname)" >&2; exit 70; fi
cd ~/bloomery-$track || { echo "x86-host.sh: no ~/bloomery-$track on \$(hostname) after the sync" >&2; exit 66; }
for v in CARGO_TARGET_DIR BLOOMERY_DATA; do
  if [ -n "\${!v:-}" ]; then echo "[x86] \$v=\${!v} from the environment is unset: each track builds in its own target/, and no data reaches this host" >&2; unset "\$v"; fi
done
renice -n 10 -p \$\$ > /dev/null || echo "[x86] renice -n 10 failed; the run keeps its priority" >&2
exec 9>> ~/.bloomery-host2.lock
if ! flock -n 9; then
  left=$wait
  [ "\$left" -le 0 ] || echo "[x86] waiting for the host lock ~/.bloomery-host2.lock (held by: \$(cat ~/.bloomery-host2.who 2>/dev/null || echo unknown)); bound ${wait}s" >&2
  got=0
  while [ "\$left" -gt 0 ]; do
    step=\$(( left < 60 ? left : 60 ))
    if flock -w "\$step" 9; then got=1; break; fi
    left=\$(( left - step ))
    [ "\$left" -le 0 ] || echo "[x86] still waiting for the host lock, \${left}s left (held by: \$(cat ~/.bloomery-host2.who 2>/dev/null || echo unknown))" >&2
  done
  if [ "\$got" != 1 ]; then
    echo "x86-host.sh: the host lock ~/.bloomery-host2.lock was still held after ${wait}s (by: \$(cat ~/.bloomery-host2.who 2>/dev/null || echo unknown)): contention" >&2
    exit 75
  fi
  echo "[x86] took the host lock" >&2
fi
echo "$track $what pid \$\$ since \$(date -u +%FT%TZ)" > ~/.bloomery-host2.who
mkdir -p target && touch target/.x86-last-use
export BLOOMERY_GIT_COMMIT=$commit
rc=0
(
$cmd
) || rc=\$?
m=\$(md5sum Cargo.lock | cut -d' ' -f1)
if [ "\$m" != $lockmd5 ]; then
  echo "x86-host.sh: the remote run rewrote Cargo.lock; refresh the lock in the tree (just check on the box, or just mac-check) and commit it" >&2
  [ "\$rc" != 0 ] || rc=65
fi
holder=\$(cut -d' ' -f1 ~/.bloomery-host2.who 2>/dev/null || true)
for d in ~/bloomery-*/target; do
  [ -d "\$d" ] || continue
  t=\${d%/target}; t=\${t##*/bloomery-}
  if [ -e "\$d/.x86-last-use" ]; then echo "\$(stat -c %Y "\$d/.x86-last-use") \$t"
  else echo "[x86] ~/bloomery-\$t/target has no .x86-last-use stamp; its last use is target/'s mtime" >&2; echo "\$(stat -c %Y "\$d") \$t"; fi
done | lru_select $KEEP $track "\$holder" | while read -r verb t _; do
  [ "\$verb" = rm ] || continue
  dir=\$HOME/bloomery-\$t
  crc=0
  bash tools/box-gc.sh --check "\$dir" > /dev/null || crc=\$?
  if [ "\$crc" = 10 ]; then echo "[x86] kept ~/bloomery-\$t/target beyond the newest $KEEP: a process runs there" >&2; continue; fi
  if [ "\$crc" != 0 ]; then echo "[x86] kept ~/bloomery-\$t/target beyond the newest $KEEP: the process scan failed (rc \$crc)" >&2; continue; fi
  size=\$(du -sb "\$dir/target" | cut -f1)
  last=\$(date -u -d "@\$(stat -c %Y "\$dir/target/.x86-last-use" 2>/dev/null || stat -c %Y "\$dir/target")" +%FT%TZ)
  if rm -rf -- "\$dir/target"; then echo "[x86] removed ~/bloomery-\$t/target (\$size B, last use \$last): beyond the newest $KEEP" >&2
  else echo "[x86] could not remove ~/bloomery-\$t/target (rm rc \$?)" >&2; fi
done
exit \$rc
} </dev/null
EOF
}

# ssh with the shared connection; stdin is the caller's.
xssh() { ssh "${SSH_MUX[@]}" "$HOST" "$@"; }

# The guard's numbers, the header line, and the verdict's rc.
guard() {
  local out rc=0 nums
  out=$(xssh "$PS_DISK" 2>&1) || rc=$?
  if [ "$rc" = 255 ]; then
    say "x86-host.sh: ssh to $HOST failed (255) before the disk guard could read: $out"
    return 255
  fi
  nums=$(parse_disk "$out") || return $?
  # shellcheck disable=SC2086
  disk_verdict $nums
}

self_test() {
  local fails=0 t out rc
  fail() { say "x86-host self-test FAIL: $*"; fails=$((fails + 1)); }
  # the allowlist: a GPU recipe is refused with 64; the static tier is refused with the Mac's recipes
  rc=0; out=$(check_recipe gate-gpu-e2e 2>&1) || rc=$?
  [ "$rc" = 64 ] || fail "gate-gpu-e2e: rc $rc, not 64"
  case $out in *"not an x86 host recipe"*) ;; *) fail "gate-gpu-e2e's refusal does not name the list: $out" ;; esac
  for t in check lint fmt fmt-check; do
    rc=0; out=$(check_recipe "$t" 2>&1) || rc=$?
    [ "$rc" = 64 ] || fail "$t: rc $rc, not 64"
    case $out in *"just mac-check"*"just mac-lint"*) ;; *) fail "$t's refusal does not point to the Mac's recipes: $out" ;; esac
  done
  for t in $ALLOW; do check_recipe "$t" 2> /dev/null || fail "$t refused"; done
  # each refused token, fed to the check as a command
  for t in "cargo oxide" "gpu-gate.sh" "BLOOMERY_DATA" "ref-paths" "measure" "lease"; do
    rc=0; check_command "cd x && $t y" 2> /dev/null || rc=$?
    [ "$rc" = 64 ] || fail "a command naming '$t': rc $rc, not 64"
  done
  check_command "bash tools/gate.sh --release -p bloomery-levers --lib -- --nocapture" 2> /dev/null || fail "a --release command refused (a token matched inside a word)"
  # the track rule
  [ "$(track_of /x/bloomery)" = main ] || fail "bloomery is not the track main"
  [ "$(track_of /x/bloomery-foo-bar/)" = foo-bar ] || fail "bloomery-foo-bar/ is not the track foo-bar"
  track_of /x/other > /dev/null 2>&1 && fail "a root not named bloomery* accepted"
  track_of '/x/bloomery-a$b' > /dev/null 2>&1 && fail "a track name with \$ accepted"
  # the wrapper: the script on stdin, the WSL exit code through `exit $LASTEXITCODE`
  case $WRAP in *'bash -s; exit $LASTEXITCODE') ;; *) fail "the wrapper does not end 'bash -s; exit \$LASTEXITCODE': $WRAP" ;; esac
  out=$(box_excludes) || fail "box_excludes failed"
  printf '%s\n' "$out" | grep -qx 'target/' || fail "the excludes lack target/"
  printf '%s\n' "$out" | grep -qx '.git/' || fail "the excludes lack .git/"
  # the guard's parse: its line among PowerShell's noise; anything else is a refusal (RC_GUARD)
  out=$(parse_disk $'noise\r\nx86-disk used=132080410624 vhdx=189589880832 free=2329677824 drive=C\r\n') || fail "a good guard line refused"
  [ "$out" = "132080410624 189589880832 2329677824 C" ] || fail "the guard line parsed as [$out]"
  for t in 'x86-disk used= vhdx=1 free=1 drive=C' 'x86-disk used=1 vhdx=1 free=-5 drive=C' 'x86-disk error: 0 registered distros' '' 'x86-disk used=1 vhdx=1 free=1 drive=C extra'; do
    rc=0; parse_disk "$t" > /dev/null 2>&1 || rc=$?
    [ "$rc" = "$RC_GUARD" ] || fail "guard line [$t]: rc $rc, not $RC_GUARD"
  done
  # the verdict: refused only when the vhdx may grow AND the drive is under the margin
  local G=1000000000
  for t in "132 189.6 2.3 0" "170 189 2 $RC_DISK" "170 189 25 0" "169 189 0 0" "169.000000001 189 19.999999999 $RC_DISK" "170 189 20 0" "0 0 0 $RC_DISK"; do
    read -r u v f want <<< "$t"
    u=$(awk -v x="$u" -v g=$G 'BEGIN { printf "%.0f", x * g }'); v=$(awk -v x="$v" -v g=$G 'BEGIN { printf "%.0f", x * g }'); f=$(awk -v x="$f" -v g=$G 'BEGIN { printf "%.0f", x * g }')
    rc=0; out=$(disk_verdict "$u" "$v" "$f" C 2>&1) || rc=$?
    [ "$rc" = "$want" ] || fail "verdict used=$u vhdx=$v free=$f: rc $rc, not $want"
    case $out in *"used $u B"*"vhdx $v B"*"C: free $f B"*) ;; *) fail "the header line lacks the three numbers: $out" ;; esac
  done
  # the target cap: the running track and the lock's holder are kept whatever their age, the newest KEEP
  # (the running one counted) are kept, the rest go
  out=$(printf '%s\n' "90 a" "80 b" "70 c" "60 d" "50 e" "40 f" "30 g" "20 h" "10 cur" "5 hold" | lru_select 6 cur hold | tr '\n' ';')
  [ "$out" = "keep a newest;keep b newest;keep c newest;keep d newest;keep e newest;rm f;rm g;rm h;keep cur running;keep hold lock-holder;" ] || fail "lru_select: [$out]"
  out=$(printf '%s\n' "3 a" "2 b" "1 c" | lru_select 6 a "" | grep -c '^rm' || true)
  [ "$out" = 0 ] || fail "lru_select removed $out of 3 targets under the cap"
  out=$(printf '%s\n' "5 b" "5 a" "9 cur" | lru_select 2 cur "" | tr '\n' ';')
  [ "$out" = "keep cur running;keep a newest;rm b;" ] || fail "lru_select on a tie: [$out]"
  # the remote script parses, keeps stdin off the command, takes the lock, stamps, caps, names 75 and 65
  out=$(remote_script foo gate-levers 'bash tools/gate.sh -p x && echo "$X"' abcd0123 0123456789abcdef0123456789abcdef 1800)
  bash -n <<< "$out" || fail "the remote script does not parse"
  printf '%s\n' "$out" | tail -1 | grep -qx '} </dev/null' || fail "the remote script's body does not end '} </dev/null'"
  for t in 'flock -n 9' 'exit 75' 'rc=65' 'cd ~/bloomery-foo ' 'renice -n 10 -p $$' 'touch target/.x86-last-use' \
    'lru_select 6 foo "$holder"' 'bash tools/box-gc.sh --check "$dir"' 'lru_select ()' 'unset "$v"' \
    'export BLOOMERY_GIT_COMMIT=abcd0123' 'bash tools/gate.sh -p x && echo "$X"'; do
    case $out in *"$t"*) ;; *) fail "the remote script lacks '$t'" ;; esac
  done
  # test mode: a device crate is refused, a host crate passes (recipes.py x86-crates, the Mac's cargo metadata)
  rc=0; out=$(python3 "$HERE/tools/recipes.py" x86-crates bloomery-sampler bloomery-gpu 2>&1) || rc=$?
  [ "$rc" = 1 ] || fail "x86-crates on a device crate: rc $rc, not 1"
  printf '%s\n' "$out" | grep -q '^ok bloomery-sampler$' || fail "x86-crates did not pass bloomery-sampler: $out"
  printf '%s\n' "$out" | grep -q '^device bloomery-gpu bloomery-gpu -> ' || fail "x86-crates did not refuse bloomery-gpu: $out"
  # the default list: the device rule alone, so gguf (x86_64 code the Mac rejects) is in and gpu is out
  out=$(python3 "$HERE/tools/recipes.py" x86-crates --names 2>&1) || fail "x86-crates --names: $out"
  printf '%s\n' "$out" | grep -qx 'bloomery-gguf' || fail "x86-crates --names left out bloomery-gguf: $out"
  printf '%s\n' "$out" | grep -qx 'bloomery-gpu' && fail "x86-crates --names listed bloomery-gpu: $out"
  rc=0; crate_ok 'a;b' 2> /dev/null || rc=$?
  [ "$rc" = 64 ] || fail "a crate name with ';' accepted"
  # the allowlist rule on a synthetic justfile: the clean recipe passes, each mutant is refused by name
  local d
  d=$(mktemp -d)
  cat > "$d/justfile" << 'JF'
clean:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib'
oxide:
    ./tools/box.sh 'cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --release && bash tools/host-gate.sh gate_x'
data:
    ./tools/box.sh 'BLOOMERY_REF_DIR=$BLOOMERY_DATA/x bash tools/gate.sh -p bloomery-levers --lib'
model:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib -- /models/small/x.gguf'
profile:
    ./tools/box.sh 'source tools/ref/ref-paths.sh && bash tools/gate.sh -p bloomery-levers --lib'
prefix:
    BLOOMERY_MODEL=qwen3moe ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib'
dep: clean
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib'
card:
    ./tools/box.sh 'bash tools/gate.sh -p bloomery-levers --lib'
JF
  printf '%s\t%s\t%s\t%s\t%s\n' clean F balanced - "no device code" oxide F balanced - "no device code" \
    data F balanced - "no device code" model F balanced - "no device code" profile F balanced - "no device code" \
    prefix F balanced - "no device code" dep F balanced - "no device code" card A fixed 3090 "card: gpu-gate.sh 3090" > "$d/classes"
  python3 "$HERE/tools/recipes.py" x86-allowed --justfile "$d/justfile" --classes "$d/classes" clean > /dev/null 2>&1 || fail "x86-allowed refused the clean recipe"
  for t in "oxide:runs cargo oxide" "data:the data directory" "model:a model file" "profile:reference paths" \
    "prefix:carries a prefix" "dep:depends on clean" "card:class A" "absent:not a recipe"; do
    rc=0; out=$(python3 "$HERE/tools/recipes.py" x86-allowed --justfile "$d/justfile" --classes "$d/classes" "${t%%:*}" 2>&1) || rc=$?
    [ "$rc" = 1 ] || fail "x86-allowed ${t%%:*}: rc $rc, not 1"
    case $out in *"${t#*:}"*) ;; *) fail "x86-allowed ${t%%:*} does not say '${t#*:}': $out" ;; esac
  done
  rm -rf "$d"
  if [ "$fails" != 0 ]; then
    say "x86-host: self-test $fails failed"
    return 1
  fi
  echo "x86-host: self-test ok"
}

# 0 for a crate name cargo takes and PowerShell cannot read; else a named line and 64.
crate_ok() {
  case $1 in
    '' | -* | *[!A-Za-z0-9_-]*) say "x86-host.sh: '$1' is not a crate name ([A-Za-z0-9_-])"; return 64 ;;
  esac
}

# --tracks [--remove NAME…]: tools/box-tracks.sh's classify and select_removals over the host's dirs.
tracks() {
  # shellcheck source=tools/box-tracks.sh
  source "$HERE/tools/box-tracks.sh"
  local live table plan rc=0 crc verb name why
  # The host names the main tree's directory bloomery-main (track_of), the box names it bloomery.
  live=$(git -C "$HERE" worktree list --porcelain | awk '/^worktree /{print $2}' | xargs -n1 basename | sed 's/^bloomery$/bloomery-main/')
  table=$(printf '%s\n' 'cd ~ && du -sh bloomery-*/ 2>/dev/null; true' | xssh "$WRAP" | tr -d '\r' | classify bloomery "$live")
  awk '{ printf "%-6s %6s  %s%s\n", $1, $2, $3, ($4 != "" ? " (live track " $4 ")" : "") }' <<< "$table"
  if [ $# = 0 ]; then
    echo "$(awk '$1 == "stale"' <<< "$table" | wc -l | tr -d ' ') stale on $HOST — delete by name: tools/x86-host.sh --tracks --remove NAME…"
    return 0
  fi
  plan=$(select_removals bloomery "$table" "$@") || rc=$?
  while read -r verb name why; do
    [ -n "$verb" ] || continue
    if [ "$verb" = skip ]; then say "skip   $name — $why"; continue; fi
    case $name in bloomery-main | bloomery-*[!A-Za-z0-9._-]*) say "refusing odd name: $name"; return 1 ;; bloomery-?*) ;; *) say "refusing odd name: $name"; return 1 ;; esac
    crc=0
    { printf 'set -- --check "$HOME/%s"\n' "$name"; cat "$HERE/tools/box-gc.sh"; } | xssh "$WRAP" > /dev/null || crc=$?
    if [ "$crc" = 10 ]; then say "skip   $name — a process still runs there (exe under its target/, or cwd inside)"; rc=1; continue; fi
    if [ "$crc" != 0 ]; then say "skip   $name — the process scan failed (rc $crc)"; rc=1; continue; fi
    printf 'rm -rf -- ~/%s\n' "$name" | xssh "$WRAP" && echo "removed $HOST:~/$name"
  done <<< "$plan"
  return $rc
}

HOST=${BLOOMERY_X86_HOST:-sihyung-gpu}
case $HOST in
  '' | *[!A-Za-z0-9._@-]*) say "x86-host.sh: BLOOMERY_X86_HOST is an ssh host name, got '$HOST'"; exit 64 ;;
esac
case ${1:-} in
  --self-test) self_test; exit $? ;;
  --lists)
    for t in $ALLOW; do echo "allow $t"; done
    for t in "${FORBID[@]}"; do echo "forbid $t"; done
    exit 0
    ;;
  --check-allowlist)
    [ $# = 2 ] || { say "x86-host.sh: --check-allowlist takes gate-batch.sh --classes output"; exit 64; }
    # shellcheck disable=SC2086
    python3 "$HERE/tools/recipes.py" x86-allowed --classes "$2" $ALLOW
    exit $?
    ;;
  --tracks)
    shift
    if [ "${1:-}" = --remove ]; then
      shift
      [ $# -gt 0 ] || { say "x86-host.sh: --tracks --remove takes the names to delete (as the listing prints them); a bare --remove is refused"; exit 64; }
    elif [ $# -gt 0 ]; then
      say "x86-host.sh: --tracks takes nothing or --remove NAME…"; exit 64
    fi
    tracks "$@"
    exit $?
    ;;
  test)
    shift
    MODE=test
    ;;
  '' | -*) say "usage: tools/x86-host.sh RECIPE | test CRATE… | --tracks [--remove NAME…] | --self-test (RECIPE: $ALLOW)"; exit 64 ;;
  *)
    [ $# = 1 ] || { say "x86-host.sh: one recipe, got $#: $*"; exit 64; }
    MODE=recipe
    ;;
esac

say "x86-host: x86 host tests on $HOST — not landing evidence, recorded in no ledger"
if [ "$MODE" = recipe ]; then
  WHAT=$1
  check_recipe "$WHAT" || exit $?
  CMD=$(python3 "$HERE/tools/recipes.py" box-command "$WHAT") || exit $?
else
  if [ $# = 0 ]; then
    names=$(python3 "$HERE/tools/recipes.py" x86-crates --names) || exit $?
    set -- $names
    say "[x86] no crate given: every crate with no device root in its closure, $*"
  fi
  for c in "$@"; do crate_ok "$c" || exit $?; done
  rc=0
  verdict=$(python3 "$HERE/tools/recipes.py" x86-crates "$@") || rc=$?
  if [ "$rc" != 0 ]; then
    printf '%s\n' "$verdict" | grep -v '^ok ' | sed 's/^device \([^ ]*\) /x86-host.sh: refused \1 — its closure reaches a device root, which needs cargo oxide and the box: /' >&2
    exit 64
  fi
  WHAT="test $*"
  CMD="bash tools/gate.sh --release$(printf ' -p %s' "$@")"
fi
check_command "$CMD" || exit $?
WAIT=${BLOOMERY_X86_WAIT:-1800}
case $WAIT in
  '' | *[!0-9]*) say "x86-host.sh: BLOOMERY_X86_WAIT is whole seconds (0: do not wait), got '$WAIT'"; exit 64 ;;
esac
TRACK=$(track_of "$HERE") || exit $?
EXCL=()
excl=$(box_excludes) || exit $?
while IFS= read -r x; do [ -z "$x" ] || EXCL+=(--exclude "$x"); done <<< "$excl"
[ ${#EXCL[@]} -gt 0 ] || { say "x86-host.sh: tools/box.sh's rsync line gave no excludes"; exit 70; }
# box.sh's rule: the short commit, `-dirty` when the tree holds changes the commit does not.
COMMIT=$(git -C "$HERE" rev-parse --short=8 HEAD 2> /dev/null || echo unknown)
[ -z "$(git -C "$HERE" status --porcelain 2> /dev/null | head -1)" ] || COMMIT="$COMMIT-dirty"
LOCKMD5=$(md5 -q "$HERE/Cargo.lock" 2> /dev/null || md5sum "$HERE/Cargo.lock" | cut -d' ' -f1)

rc=0
guard || rc=$?
[ "$rc" = 0 ] || exit "$rc"
# The receiver creates ~/bloomery-<track> itself (one level under an existing home), so no mkdir.
rsync -rlpgoDcz --delete "${EXCL[@]}" -e "ssh ${SSH_MUX[*]}" --rsync-path="$RSYNC_PATH" "$HERE"/ "$HOST:~/bloomery-$TRACK/" || rc=$?
if [ "$rc" != 0 ]; then
  say "x86-host.sh: the sync to $HOST:~/bloomery-$TRACK failed (rsync rc $rc)"
  exit "$rc"
fi
remote_script "$TRACK" "$WHAT" "$CMD" "$COMMIT" "$LOCKMD5" "$WAIT" | xssh "$WRAP" || rc=$?
[ "$rc" != 255 ] || say "x86-host.sh: ssh to $HOST ended with 255 (a connection failure, or the command's own 255)"
exit "$rc"
