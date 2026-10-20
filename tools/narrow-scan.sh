#!/usr/bin/env bash
# `just narrow BASE [HEAD]`: the scans `just affected --narrow` needs, taken for you.
#   tools/narrow-scan.sh BASE [HEAD] [--out DIR] [--dry-run] [--keep-ptx] [--scan-only]
#   tools/narrow-scan.sh BASE --worktree [--out DIR] [--dry-run] [--keep-ptx] [--scan-only]
#   tools/narrow-scan.sh --self-test
#
# What it does, in order:
#   1. Resolves BASE and HEAD (default: this tree's HEAD commit) to commits. The scans read the commits through
#      `git archive`, never the working tree: uncommitted edits are not scanned (a note says so). `--worktree` is
#      the mode for a tree that may not commit (a round): the head side is this tree's current content, uncommitted
#      and untracked files included, synced as box.sh syncs it (its own tools/box.sh from this tree), and the base
#      side stays BASE's commit. BASE must be an ancestor of this tree's HEAD, so that `tools/affected-gates.sh BASE`
#      (the working tree against the merge base) reads the change the two scans compare; a fingerprint of the tree
#      (tracked diff against BASE plus each untracked file's md5) is taken before the scans and again before the
#      affected run, and a tree that moved in between ends the run (rc 1) instead of narrowing on mixed content.
#   2. Derives each side's scan set from that side's own tree (`tools/recipes.py scan-set [REV]`: BASE's commit for the
#      base side; HEAD's commit, or with `--worktree` this tree as it is, for the head side): the fewest bins of
#      bloomery-gpu-gates whose scans cover every kernel carrier (each device lib, and each bin with kernels of its own),
#      each with the cargo features it builds under on that side. A feature one side's tree lacks (a change that adds or
#      moves a cargo feature) is thus never asked of the other: the bin whose features differ is printed by name with
#      both sets, and each side builds its own. The two sides must scan the same bins, since a pair compares one bin's
#      scan on both: a bin in only one side's set ends the run (rc 2) naming it, with both sets, before any box call.
#   3. For each side in turn, BASE then HEAD: extracts the commit into DIR/src-<side>, syncs it to that side's remote
#      dir with its own tools/box.sh, builds that side's set there (`cargo oxide build`, as the ptx-scan recipe does) and runs
#      `ptx-scan.sh --no-jit` on each bin. This tree's tools/ptx-scan.sh is the scanner on both sides (the commit's
#      own may predate --no-jit), shipped over stdin; the extractor (oxart_ptx) is the commit's.
#   4. Splits the box's output into DIR/<side>/<bin>.log and runs
#      `tools/affected-gates.sh BASE..HEAD --narrow --scan <base log> <head log> ...` (`BASE --narrow …` under
#      `--worktree`) with every pair, printing its output: the narrowed list with its why-lines, and the `recipes:`
#      line to paste. DIR/narrowed.txt keeps it.
#
#   `--keep-ptx` (unset: everything above, byte for byte) has each side's runner keep every bin's PTX modules
#   (`PTX_SCAN_KEEP`, tools/ptx-scan.sh) and send them after the bin's scan block, as `=== narrow-scan ptx begin <bin>`,
#   a base64 of a gzipped tar of its mod<N>.ptx files (the raw module text tools/ref/ptx-canon.py reads), and
#   `=== narrow-scan ptx end <bin> rc=<n>`; they arrive locally under DIR/<side>/<bin>.ptx/. The block sits outside
#   the scan's own, so DIR/<side>/<bin>.log is the unset run's. A side that sends no block, a cut or failed one, or a
#   member that is not a flat mod<N>.ptx ends the run (rc 1) by name; the MANIFEST then carries `keep_ptx=1`.
#   `--scan-only` skips step 4's affected run (it prints no narrowed list, writes no narrowed.txt, and the MANIFEST
#   says `affected_rc=skipped`): the caller reads the scans itself (tools/canon-check.sh).
#
# The base side's cache. BASE is a commit, so its side (the scan logs and, with --keep-ptx, the PTX modules) is the same
# whenever its inputs are: the BASE sha, the md5 of this tree's tools/ptx-scan.sh (the scanner both sides run), the scan set
# (the base side's scan-set tsv's md5, so a head-only feature change leaves it), the base runner's md5 (the build lines), the
# arch and whether --keep-ptx is set. A run whose key has an
# entry under ~/.cache/bloomery/narrow-base/<base8>-<key16>/ prints `narrow-scan: base <sha> from cache <dir>`, copies
# <dir>/base/ into DIR/base/ as a fresh run lays it out and makes no box call for the base side (the head side runs as
# always). Any key part different is a miss. An entry is written whole (a temp dir and a rename) only after the side
# finished with every bin's scan rc 0 and, under --keep-ptx, every bin's modules; a failed or partial side is never kept.
# An entry whose MANIFEST is missing or lacks a key part, or that lacks a bin's log (or modules under --keep-ptx), is named
# in a line (`cache entry <dir> is corrupt (<why>)`), removed and rebuilt: a part of an entry is never used. NARROW_NO_CACHE=1
# forces a miss (the cache is not read); the run's result then replaces that key's entry. The head side is never cached, and
# the toolkit on the box is not in the key: a ptxas move shows as moved table rows against the cached base, so after one
# remove ~/.cache/bloomery/narrow-base or run once with NARROW_NO_CACHE=1. The MANIFEST names it: `base_cache=hit|miss`.
#
# The remote dirs are per track and persist across runs: ~/repo/bloomery-<track>-narrow-base and -narrow-head, <track>
# this tree's directory name without "bloomery-" (the main tree's pair: ~/repo/bloomery-narrow-base and -narrow-head), so
# two tracks never wait on each other's dirs. A second run builds incrementally: the BASE side costs a sync and the scans
# (no scan is cached; they run again), the HEAD side the crates the change touched. A pair belongs to this recipe alone,
# and one run uses it at a time: a lock under ~/.cache/bloomery named by the pair, an fcntl flock on a file
# held for the whole scan (python3's fcntl: the Mac has no flock(1)). Each launch closes that fd before it
# opens a FIFO, so a child blocked on the open does not keep the lock after this process exits. While the
# flock is held this run also creates that directory
# and writes its pid, so an older copy's mkdir of the same path fails. A directory whose pid is live, or not
# written yet, is refused and not deleted. A dead directory is removed only while this flock is already held,
# and this run removes its own directory when it drops the flock. A second run waits up to NARROW_WAIT
# seconds, default 1800, naming the holder. The main tree's pair keeps the lock name older copies of this
# script took for every tree. NARROW_BASE_REMOTE and
# NARROW_HEAD_REMOTE name other dirs. The names follow the
# auxiliary-dir rule (bloomery-<track>-<suffix>): once the track's worktree is gone, `just box-tracks --remove
# <track>-narrow-base <track>-narrow-head` clears them (the main tree's: `narrow-base narrow-head`).
#
# A side whose box command ends rc 255 (ssh lost the box) runs once more: its recorded remote runner is ended first (the
# stop's READONLY call below), then the side runs again in the same remote dir. One line says so, the MANIFEST carries
# `retried=<sides>` and the done line `(retried: <sides>)`; a second 255 ends the run rc 255. No other code is retried.
#
# Stopping (INT/TERM): the side in flight ends from both ends, by pids it recorded and never by a name match. Its box
# command writes the pid of the `timeout` it started to .narrow-scan-<side>-<pid>.pid in the remote dir (removed when the
# runner ends, and by the next sync if a run died); the stop reads that file in one READONLY box call (no sync, no guard
# wait) and signals that pid, whose TERM `timeout` passes to the runner's process group, then signals the local box.sh's
# process group (job control makes it a group of its own: box.sh, its ssh and rsync) by the pid written under DIR at its
# start. A stop that finds no file prints so (the runner had not started or had ended).
#
# It runs no gate and takes no timing lease, no hold, no card and no gate lock: `--no-jit` loads nothing on a card, and
# the box commands are builds and ptxas. box.sh's own guard still waits while a sitting's lease or hold is up (rc 75 is
# contention: run again, BLOOMERY_BOX_WAIT=3600 waits longer). BLOOMERY_CARD, BLOOMERY_BOX_ENV and BLOOMERY_TIER are
# unset for the box calls, so nothing card-related reaches the box, and so are BLOOMERY_BOX_QUEUE and the queue's
# settings: the runner goes over stdin and the stop signals pids the direct path records, and a queued job has neither.
# A runner that arrives empty ends the side 65 by name, never an empty scan read as rc 0. The scans are sm_86's: a BLOOMERY_ARCH naming another
# arch is refused (64) before anything runs.
#
# DIR (default ~/.cache/bloomery/narrow/<tree>/<base8>-<head8>, <base8>-worktree under `--worktree`) keeps: scan-set.tsv (the head
# side's), scan-set-base.tsv, runner.sh (the script the head side runs; the base side runs it too when the two sets are equal, else
# runner-base.sh, built from scan-set-base.tsv), <side>.build.log (the box's stderr: the builds), <side>.scan.out, <side>/<bin>.log (the scan logs a landing
# cites), narrowed.txt and MANIFEST; <side>.box.pid while the side runs. Under --keep-ptx also <side>/<bin>.ptx/mod<N>.ptx; a
# cached base side has no base.build.log and no base.scan.out; base-cache.key holds the cache key's parts.
#
# NARROW_BOUND (default 7200 s, a hang guard: no box-job length needs approval) bounds the whole run: each side's box command runs under `timeout` with what
# is left. A cut build keeps its compiled crates, so running again continues it. Exit codes: 0 done; 64 usage; 75
# contention (the box guard, or the lock); 70 the box's toolchain; 1 a build or scan failed; 2 a scan set could not be derived or
# the sides' sets name different bins; 124 the bound; 130 INT and 143 TERM; the `affected` run's own code otherwise.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck source=tools/nightly/units.sh
source "$ROOT/tools/nightly/units.sh" # tree_commit labels the head tree, archive_tree reads a base commit
PKG=bloomery-gpu-gates # the package the ptx-scan recipe builds; tools/recipes.py SCAN_PACKAGE
ARCH=sm_86 # the arch of the scans, the recipes' default (BLOOMERY_ARCH, tools/arch.sh); the pairs are built at no other
USAGE="usage: narrow-scan.sh BASE [HEAD | --worktree] [--out DIR] [--dry-run] [--keep-ptx] [--scan-only] | --self-test"

die() { # die <rc> <message>
  echo "narrow-scan: $2" >&2
  exit "$1"
}

md5_of() { # md5_of <file>: the Mac has md5, the box md5sum
  if command -v md5sum > /dev/null; then md5sum "$1" | cut -d' ' -f1; else md5 -q "$1"; fi
}

# str_key <text>: a short key of the text, for a lock's name.
str_key() {
  printf '%s' "$1" | if command -v md5sum > /dev/null; then md5sum | cut -c1-12; else md5 -q | cut -c1-12; fi
}

# worktree_key <base sha>: a fingerprint of this tree's content against BASE: the tracked diff (binary) and each
# untracked, not ignored file's md5 by path (git lists them in path order). Equal keys are the same content; the ignored files box.sh also syncs
# are not in it (affected-gates reads git's view of the change too).
worktree_key() {
  {
    git -C "$ROOT" diff --binary --no-renames "$1" || echo "diff failed"
    git -C "$ROOT" ls-files --others --exclude-standard -z | while IFS= read -r -d '' f; do
      printf '%s %s\n' "$(md5_of "$ROOT/$f" 2> /dev/null || echo unreadable)" "$f"
    done
  } | if command -v md5sum > /dev/null; then md5sum | cut -d' ' -f1; else md5 -q; fi
}

# make_runner <scan-set.tsv> <ptx-scan.sh> <runner out> [keep]: the script a side runs on the box. It writes ptx-scan.sh to a
# temp file, builds each feature group (the builds' output goes to stderr: stdout carries the scan blocks only), builds
# the extractor, and prints each bin's scan between `=== narrow-scan begin <bin>` and `end <bin> rc=<n>` markers. With
# keep = 1 each scan also keeps its modules (PTX_SCAN_KEEP) and the bin's modules follow the scan's end marker as a
# base64 gzipped tar between `=== narrow-scan ptx begin <bin>` and `ptx end <bin> rc=<n>`; keep 0 or unset adds nothing.
make_runner() {
  local tsv=$1 scanner=$2 out=$3 keep=${4:-0} bin feats covers f b groups='' line
  local bins=() featl=()
  while IFS=$'\t' read -r bin feats covers; do
    [ -n "$bin" ] || continue
    case $bin in *[!A-Za-z0-9_-]* | -*) echo "narrow-scan: scan-set.tsv: bin name '$bin' is not a cargo bin name" >&2; return 2 ;; esac
    case $feats in *[!a-z0-9_,-]*) echo "narrow-scan: scan-set.tsv: features '$feats' of $bin are not a feature list" >&2; return 2 ;; esac
    bins[${#bins[@]}]=$bin
    featl[${#featl[@]}]=$feats
  done < "$tsv"
  [ "${#bins[@]}" -gt 0 ] || { echo "narrow-scan: $tsv names no bin: nothing to scan" >&2; return 2; }
  if grep -qx 'NARROW_PTX_SCAN_EOF' "$scanner"; then
    echo "narrow-scan: $scanner holds the here-document's end marker" >&2
    return 2
  fi
  {
    echo 'set -uo pipefail'
    echo 'P=$(mktemp) || exit 70'
    echo "cat > \"\$P\" <<'NARROW_PTX_SCAN_EOF'"
    cat "$scanner"
    [ "$(tail -c 1 "$scanner" | wc -l)" -eq 1 ] || printf '\n'
    echo 'NARROW_PTX_SCAN_EOF'
    [ "$keep" != 1 ] || echo 'K=$(mktemp -d) || exit 70'
    # one `cargo oxide build` per distinct feature list, its bins in the set's order
    for f in "${featl[@]}"; do
      case " $groups " in *" $f "*) continue ;; esac
      groups="$groups $f"
      line="cargo oxide build --arch $ARCH -- -p $PKG --features $f --release"
      for b in "${!bins[@]}"; do
        [ "${featl[$b]}" = "$f" ] && line="$line --bin ${bins[$b]}"
      done
      echo 't=$SECONDS'
      echo "$line >&2 </dev/null || { echo 'narrow-scan: the build with features $f failed' >&2; exit 1; }"
      echo "echo \"narrow-scan: wall: build, features $f: \$((SECONDS - t)) s\" >&2"
    done
    echo 't=$SECONDS'
    echo "cargo build --release -p $PKG --bin oxart_ptx >&2 </dev/null || { echo 'narrow-scan: the extractor build failed' >&2; exit 1; }"
    echo "echo \"narrow-scan: wall: build, extractor: \$((SECONDS - t)) s\" >&2"
    for b in "${bins[@]}"; do
      echo 't=$SECONDS'
      if [ "$keep" = 1 ]; then
        echo "echo '=== narrow-scan begin $b'; PTX_SCAN_KEEP=\"\$K/$b\" bash \"\$P\" --no-jit $b 2>&1; echo \"=== narrow-scan end $b rc=\$?\""
        echo "echo '=== narrow-scan ptx begin $b'"
        echo "tar -czf - -C \"\$K/$b\" . | base64"
        echo "echo \"=== narrow-scan ptx end $b rc=\$?\""
      else
        echo "echo '=== narrow-scan begin $b'; bash \"\$P\" --no-jit $b 2>&1; echo \"=== narrow-scan end $b rc=\$?\""
      fi
      echo "echo \"narrow-scan: wall: scan $b: \$((SECONDS - t)) s\" >&2"
    done
    echo 'rm -f "$P"'
    [ "$keep" != 1 ] || echo 'rm -rf "$K"'
  } > "$out"
}

# scan_set_diff <base scan-set.tsv> <head scan-set.tsv>: the two sides' sets compared by bin name. A bin in both whose features
# differ prints `narrow-scan: bin B: features differ between the sides: base F, head G` (each side builds its own). A bin in one set
# only prints which side has it, and the function returns 1: a pair needs the bin scanned on both sides.
scan_set_diff() {
  awk -F'\t' '
    NR == FNR { if ($1 != "") { bf[$1] = $2; bo[++nb] = $1 }; next }
    $1 != "" { hf[$1] = $2; ho[++nh] = $1 }
    END {
      bad = 0
      for (i = 1; i <= nh; i++) {
        b = ho[i]
        if (!(b in bf)) { print "narrow-scan: bin " b ": in the head scan set only (features " hf[b] ")"; bad = 1 }
        else if (bf[b] != hf[b]) print "narrow-scan: bin " b ": features differ between the sides: base " bf[b] ", head " hf[b]
      }
      for (i = 1; i <= nb; i++) if (!(bo[i] in hf)) { print "narrow-scan: bin " bo[i] ": in the base scan set only (features " bf[bo[i]] ")"; bad = 1 }
      exit bad
    }' "$1" "$2"
}

# split_logs <scan.out> <dir> <bin>...: cuts the box's stdout into <dir>/<bin>.log. Non-zero when a block is missing,
# unterminated, nested, or its scan ended non-zero; the reasons are printed.
split_logs() {
  local file=$1 dir=$2
  shift 2
  mkdir -p "$dir"
  awk -v dir="$dir" -v want="$*" '
    BEGIN { n = split(want, W, " "); for (i = 1; i <= n; i++) need[W[i]] = 1 }
    /^=== narrow-scan begin / {
      if (cur != "") { print "block " cur " has no end marker"; bad = 1; close(file) }
      cur = $4; file = dir "/" cur ".log"; printf "" > file; seen[cur] = "open"; next
    }
    /^=== narrow-scan end / {
      if ($4 != cur) { print "end marker for " $4 " outside its block (inside " (cur == "" ? "none" : cur) ")"; bad = 1; next }
      rc = $5; sub(/^rc=/, "", rc); seen[cur] = (rc == "0" ? "ok" : "rc=" rc); close(file); cur = ""; next
    }
    cur != "" { print >> file }
    END {
      if (cur != "") { print "block " cur " has no end marker: the box output was cut"; bad = 1 }
      for (b in need) {
        if (!(b in seen)) { print "no scan block for " b; bad = 1 }
        else if (seen[b] != "ok") { print "the scan of " b " ended " seen[b] " (its log: " dir "/" b ".log)"; bad = 1 }
      }
      exit bad
    }' "$file"
}

# The decoder split_ptx runs, kept in a function so that no here-document sits inside a command substitution (the Mac's bash 3.2).
# argv: the base64 file, the out dir, the bin name. A tar's members must be flat mod<N>.ptx regular files; anything else is
# refused before a byte is written.
ptx_decode_py() {
  cat << 'NARROW_PTX_DECODE_EOF'
import base64, binascii, io, os, re, sys, tarfile
src, out, name = sys.argv[1:4]
try:
    raw = base64.b64decode(open(src, "rb").read(), validate=False)
    tf = tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")
    members = tf.getmembers()
except (binascii.Error, tarfile.TarError, OSError, EOFError, ValueError) as e:
    sys.exit(f"{name}: the ptx block does not decode ({e})")
files = []
for m in members:
    base = m.name[2:] if m.name.startswith("./") else m.name
    if m.isdir() and base in ("", "."):
        continue
    if not (m.isreg() and re.fullmatch(r"mod[0-9]+\.ptx", base)):
        sys.exit(f"{name}: the ptx block holds a member that is not mod<N>.ptx: {m.name!r}")
    files.append((base, m))
if not files:
    sys.exit(f"{name}: the ptx block holds no module")
os.makedirs(out)
for base, m in files:
    with open(os.path.join(out, base), "wb") as f:
        f.write(tf.extractfile(m).read())
NARROW_PTX_DECODE_EOF
}

# split_ptx <scan.out> <dir> <bin>...: the keep runner's ptx blocks of the box's stdout, each cut into <dir>/<bin>.ptx/ as the
# mod<N>.ptx files its tar holds. Non-zero when a block is missing, unterminated, ended non-zero, does not decode, holds no
# module, or holds a member that is not a flat mod<N>.ptx (nothing is written for such a tar); the reasons are printed.
split_ptx() {
  local file=$1 dir=$2 b bad=0 msg
  shift 2
  mkdir -p "$dir"
  for b in "$@"; do rm -rf "$dir/$b.ptx" "$dir/$b.ptx.b64"; done
  awk -v dir="$dir" -v want="$*" '
    BEGIN { n = split(want, W, " "); for (i = 1; i <= n; i++) need[W[i]] = 1 }
    /^=== narrow-scan ptx begin / {
      if (cur != "") { print "ptx block " cur " has no end marker"; bad = 1; close(file) }
      cur = $5; file = dir "/" cur ".ptx.b64"; printf "" > file; seen[cur] = "open"; next
    }
    /^=== narrow-scan ptx end / {
      if ($5 != cur) { print "ptx end marker for " $5 " outside its block (inside " (cur == "" ? "none" : cur) ")"; bad = 1; next }
      rc = $6; sub(/^rc=/, "", rc); seen[cur] = (rc == "0" ? "ok" : "rc=" rc); close(file); cur = ""; next
    }
    cur != "" { print >> file }
    END {
      if (cur != "") { print "ptx block " cur " has no end marker: the box output was cut"; bad = 1 }
      for (b in need) {
        if (!(b in seen)) { print "no ptx block for " b; bad = 1 }
        else if (seen[b] != "ok") { print "the ptx block of " b " ended " seen[b]; bad = 1 }
      }
      exit bad
    }' "$file" || bad=1
  [ "$bad" = 0 ] || return 1
  for b in "$@"; do
    msg=$(python3 -I -c "$(ptx_decode_py)" "$dir/$b.ptx.b64" "$dir/$b.ptx" "$b" 9>&- 2>&1) || { echo "$msg"; bad=1; continue; }
    rm -f "$dir/$b.ptx.b64"
  done
  return "$bad"
}

# The base side's cache (header): the logs and modules of a finished base side, kept by what they were made from. cache_parts
# prints the key's parts, one per line; the entry's dir is <root>/<base8>-<first 16 hex of the parts' md5>, and its MANIFEST
# carries the parts again (compared line by line on a hit), the bins and the date.
cache_parts() { # cache_parts <scanner> <scan-set.tsv> <runner.sh>
  echo "base=$BASE_SHA"
  echo "ptx_scan_md5=$(md5_of "$1" 9>&-)"
  echo "scan_set_md5=$(md5_of "$2" 9>&-)"
  echo "runner_md5=$(md5_of "$3" 9>&-)"
  echo "arch=$ARCH"
  echo "keep_ptx=$KEEP"
}

# cache_check <entry dir> <parts file> <bin>...: 0 when the entry is complete for this key; else 1 and the reason on stdout.
cache_check() {
  local dir=$1 parts=$2 line b
  shift 2
  [ -s "$dir/MANIFEST" ] || { echo "its MANIFEST is missing"; return 1; }
  while IFS= read -r line; do
    grep -qxF "$line" "$dir/MANIFEST" || { echo "its MANIFEST does not carry '$line'"; return 1; }
  done < "$parts"
  grep -qxF "bins=$*" "$dir/MANIFEST" || { echo "its MANIFEST does not carry 'bins=$*'"; return 1; }
  for b in "$@"; do
    [ -s "$dir/base/$b.log" ] || { echo "$dir/base/$b.log is missing or empty"; return 1; }
    if [ "$KEEP" = 1 ] && ! ls "$dir/base/$b.ptx"/mod[0-9]*.ptx > /dev/null 2>&1; then
      echo "$dir/base/$b.ptx holds no module"
      return 1
    fi
  done
  return 0
}

# cache_store <entry dir> <parts file> <bin>...: the finished base side in DIR/base/ becomes the entry: written whole in a
# temporary directory of the root (a dot name no lookup reads) and renamed into place, so a reader never sees part of one. A
# failure to store is a named note, not a failed run: the cache is a saving, the scans are in DIR.
cache_store() {
  local dir=$1 parts=$2 root tmp
  shift 2
  root=$(dirname "$dir")
  tmp=$root/.tmp-$$-$(basename "$dir")
  { mkdir -p "$root" && rm -rf "$tmp" && mkdir "$tmp" && cp -R "$OUT/base" "$tmp/base" \
    && { cat "$parts"; echo "bins=$*"; echo "created=$(date +%Y-%m-%dT%H:%M:%S)"; } > "$tmp/MANIFEST"; } 9>&- \
    && { rm -rf "$dir" && python3 -c 'import os, sys; os.rename(sys.argv[1], sys.argv[2])' "$tmp" "$dir"; } 9>&- \
    || { echo "narrow-scan: the base side's cache entry $dir could not be written; the scans are in $OUT" >&2; rm -rf "$tmp" 9>&-; return 0; }
  echo "narrow-scan: base side cached: $dir"
}

# lock_take <lock path> <wait seconds> <who>: one run at a time uses a pair of persistent remote dirs.
# The lock is an fcntl exclusive flock on fd 9 of <lock path>.flock (python3's fcntl: the Mac has no
# flock(1)). It is the open file description's, so it stays held after python3 exits and drops when this
# process closes the fd. While the flock is held this run creates <lock path> and writes its pid, so an
# older copy's mkdir of that path fails. A directory already there with a live pid, or with no pid written
# yet, is refused by name. A dead one is removed only after this flock is held, so two recoverers cannot
# both delete it. Prints what it waits for; rc 75 at the bound.
lock_take() {
  local lockd=$1 wait=$2 who=$3 waited=0 rc pid='' holder=''
  mkdir -p "$(dirname "$lockd")" || return 1
  while :; do
    if [ -d "$lockd" ]; then
      pid=$(cat "$lockd/pid" 2> /dev/null || true)
      holder=$(cat "$lockd/who" 2> /dev/null || true)
      if [ -z "$pid" ] || kill -0 "$pid" 2> /dev/null; then
        if [ "$waited" -ge "$wait" ]; then
          echo "narrow-scan: $lockd is held by pid ${pid:-?} (${holder}); this pair of remote dirs takes one run at a time (contention, not failure: run again)" >&2
          return 75
        fi
        echo "narrow-scan: waiting for pid ${pid:-?} (${holder}): this pair of remote dirs takes one run at a time (${waited}s of ${wait}s)" >&2
        sleep 30
        waited=$((waited + 30))
        continue
      fi
    fi
    exec 9>>"$lockd.flock" || return 1
    rc=0
    python3 -c 'import fcntl, sys
try:
    fcntl.flock(9, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(75)
except OSError:
    sys.exit(1)' || rc=$?
    case $rc in
      0) ;;
      75)
        pid=''
        holder=''
        { IFS= read -r pid; IFS= read -r holder; } < "$lockd.flock" || true
        exec 9>&-
        if [ "$waited" -ge "$wait" ]; then
          echo "narrow-scan: $lockd is held by pid ${pid:-?} (${holder}); this pair of remote dirs takes one run at a time (contention, not failure: run again)" >&2
          return 75
        fi
        echo "narrow-scan: waiting for pid ${pid:-?} (${holder}): this pair of remote dirs takes one run at a time (${waited}s of ${wait}s)" >&2
        sleep 30
        waited=$((waited + 30))
        continue
        ;;
      *) exec 9>&-; return 1 ;;
    esac
    if [ -d "$lockd" ]; then
      pid=$(cat "$lockd/pid" 2> /dev/null || true)
      holder=$(cat "$lockd/who" 2> /dev/null || true)
      if [ -z "$pid" ] || kill -0 "$pid" 2> /dev/null; then
        exec 9>&-
        echo "narrow-scan: $lockd is held by pid ${pid:-?} (${holder}); this pair of remote dirs takes one run at a time (contention, not failure: run again)" >&2
        return 75
      fi
      echo "narrow-scan: the lock's holder (pid ${pid:-?}, ${holder}) is gone: taking it over" >&2
      rm -rf "$lockd"
    fi
    mkdir "$lockd" 2> /dev/null && printf '%s\n' "$$" > "$lockd/pid" && printf '%s\n' "$who" > "$lockd/who" || { exec 9>&-; return 1; }
    printf '%s\n%s\n' "$$" "$who" > "$lockd.flock" || { exec 9>&-; return 1; }
    NARROW_LOCK=$lockd
    return 0
  done
}

lock_release() { # drops this process's flock. The directory is removed only when its pid is this process.
  if [ -n "${NARROW_LOCK:-}" ] && [ "$(cat "$NARROW_LOCK/pid" 2> /dev/null || true)" = "$$" ]; then
    rm -rf "$NARROW_LOCK"
  fi
  NARROW_LOCK=
  exec 9>&-
  return 0
}

# side_command <bound s> <pid file>: the one box.sh argument each side runs. The runner arrives on stdin, so the command writes it to a
# temp file and runs it under `timeout`, started in the background so that its pid (the `timeout`, which passes a TERM to the runner's
# process group) is written to <pid file> in the remote dir; the file goes when the runner ends. The command's rc comes back whatever
# came before it (the guard's 75, the backend's 70). The leading `:` names `cargo oxide`, which makes box.sh check the pinned backend
# before the command, as it does for a recipe.
side_command() {
  printf '%s' ': cargo oxide && T=$(mktemp) && cat > "$T" && { [ -s "$T" ] || { echo "narrow-scan: the runner arrived empty on stdin" >&2; rm -f "$T"; exit 65; }; } && { timeout --kill-after=10 '"$1"' bash "$T" </dev/null & echo $! > '"$2"'; wait $!; }; rc=$?; rm -f "$T" '"$2"'; exit $rc'
}

# stop_command <pid file>: the READONLY box command of a stop: signals the pid the side command recorded, if the file is there.
stop_command() {
  printf '%s' 'p=$(cat '"$1"' 2>/dev/null); if [ -n "$p" ]; then kill -TERM "$p" && echo "narrow-scan: signalled the recorded remote pid $p"; else echo "narrow-scan: no remote pid recorded in '"$1"' (the runner had not started or had ended)"; fi'
}

# remote_stop <remote dir> <pid file>: the stop's one READONLY box call (no sync, no guard wait: a sitting does not delay it),
# bounded at 30 s.
remote_stop() {
  local bp k=0
  env -u BLOOMERY_CARD -u BLOOMERY_BOX_ENV -u BLOOMERY_TIER BLOOMERY_REMOTE="$1" BLOOMERY_BOX_READONLY=1 BLOOMERY_BOX_WAIT=0 \
    "$ROOT/tools/box.sh" "$(stop_command "$2")" < /dev/null >&2 2>&1 9>&- &
  bp=$!
  while kill -0 "$bp" 2> /dev/null && [ "$k" -lt 150 ]; do
    sleep 0.2 9>&-
    k=$((k + 1))
  done
  if kill -0 "$bp" 2> /dev/null; then
    kill -TERM "$bp" 2> /dev/null || true
    echo "narrow-scan: the stop's box call did not return in 30 s: the runner may still be running on the box ($1/$2; 'just box-gc --dry-run' lists orphans)" >&2
  fi
}

# stop_run <rc>: INT or TERM. The side in flight (ACTIVE_*, set by run_side) is ended from both ends, then the EXIT trap releases the lock.
ACTIVE_NAME='' ACTIVE_DIR='' ACTIVE_RPIDF='' ACTIVE_PIDF=''
stop_run() {
  local bp
  trap '' INT TERM # a second signal does not cut the stop short
  if [ -n "$ACTIVE_NAME" ]; then
    echo "narrow-scan: stopped: ending the $ACTIVE_NAME side ($ACTIVE_DIR/$ACTIVE_RPIDF on the box, $ACTIVE_PIDF here)" >&2
    remote_stop "$ACTIVE_DIR" "$ACTIVE_RPIDF"
    if [ -f "$ACTIVE_PIDF" ]; then
      bp=$(cat "$ACTIVE_PIDF")
      kill -TERM -- "-$bp" 2> /dev/null || kill -TERM "$bp" 2> /dev/null || true
    fi
  fi
  exit "$1"
}

# run_side <name> <rev> <remote dir> <runner> <out dir> <bound s> <bin>...: one side's sync, build and scans on the box.
# `rev` is a commit (extracted from `git archive`) or `worktree`: this tree as it is, synced by its own tools/box.sh. The box.sh runs in
# the background and is waited for, so a stop signal is handled at once (a trap waits for a foreground command); its pid goes to
# <out>/<name>.box.pid, and its stderr through a fifo to <out>/<name>.build.log and ours.
run_side() {
  local name=$1 rev=$2 dir=$3 runner=$4 out=$5 bound=$6 src rc t0 cmd fifo tp bp
  shift 6
  if [ "$rev" = worktree ]; then
    src=$ROOT
  else
    src=$out/src-$name
    rm -rf "$src" 9>&-
    mkdir -p "$src" 9>&-
    archive_tree "$ROOT" "$rev" "$src" 9>&- || die 1 "git archive $rev failed (git $ARCHIVE_GIT_RC, tar $ARCHIVE_TAR_RC)"
  fi
  ACTIVE_RPIDF=.narrow-scan-$name-$$.pid
  cmd=$(side_command "$bound" "$ACTIVE_RPIDF")
  echo "narrow-scan: $name: $rev -> $dir (bound ${bound}s)" >&2
  t0=$SECONDS
  fifo=$out/$name.build.fifo
  rm -f "$fifo" 9>&-
  mkfifo "$fifo" 9>&- || die 70 "cannot make $fifo"
  # fd 9 is the flock. Close it before opening the FIFO: a child blocked on that open keeps the lock
  # after this process exits.
  tee "$out/$name.build.log" 9>&- < "$fifo" >&2 &
  tp=$!
  # git must not find this tree's repository above the archive (box.sh stamps its HEAD into the build as the commit). Job control
  # makes the subshell the leader of a process group of its own, so a stop reaches box.sh's ssh and rsync with the one pid.
  # The close is on the subshell, before its redirect opens the FIFO.
  set -m
  (
    cd "$src" && exec env -u BLOOMERY_CARD -u BLOOMERY_BOX_ENV -u BLOOMERY_TIER \
      -u BLOOMERY_BOX_QUEUE -u BLOOMERY_Q_CLASS -u BLOOMERY_Q_CARD -u BLOOMERY_Q_PRED_S GIT_CEILING_DIRECTORIES="$out" BLOOMERY_REMOTE="$dir" \
      ./tools/box.sh "$cmd" < "$runner" 2>&1 > "$out/$name.scan.out"
  ) 9>&- > "$fifo" &
  bp=$!
  set +m
  ACTIVE_PIDF=$out/$name.box.pid
  echo "$bp" > "$ACTIVE_PIDF"
  ACTIVE_NAME=$name ACTIVE_DIR=$dir
  wait "$bp"
  rc=$?
  ACTIVE_NAME=''
  wait "$tp" 2> /dev/null
  rm -f "$fifo" "$ACTIVE_PIDF" 9>&-
  SIDE_WALL=$((SECONDS - t0))
  [ "$rev" = worktree ] || rm -rf "$src" 9>&-
  case $rc in
    0) ;;
    75) echo "narrow-scan: $name: rc 75: the box is held by a sitting or a lease (contention, not failure; nothing was built): run again, BLOOMERY_BOX_WAIT=3600 waits longer" >&2 ;;
    124 | 137) echo "narrow-scan: $name: the bound of ${bound}s ended the box command (rc $rc); the dir keeps its compiled crates: run again to continue" >&2 ;;
    65) echo "narrow-scan: $name: rc 65: the runner reached the box empty (nothing was scanned)" >&2 ;;
    70) echo "narrow-scan: $name: rc 70: the box's cuda-oxide backend is missing for this commit's pin (tools/box.sh header)" >&2 ;;
    *) echo "narrow-scan: $name: the box command ended rc $rc (the commit's build or a scan: $out/$name.build.log)" >&2 ;;
  esac
  return "$rc"
}

self_test() {
  local t n=0 bad=0 want rc got
  unset BLOOMERY_ARCH
  # macOS tar stores extended attributes as `._*` members, which the ptx decoder refuses as no mod<N>.ptx; the box's
  # GNU tar writes none, so the cases tar as the box does.
  export COPYFILE_DISABLE=1
  # The base side's cache is off for every run below but its own cases (they set NARROW_NO_CACHE=0): a hit would skip the side the
  # older cases assert on.
  export NARROW_NO_CACHE=1
  t=$(mktemp -d "${TMPDIR:-/tmp}/narrow-scan-test.XXXXXX") || die 70 "self-test: no temporary directory"
  # ok <name> <condition rc>
  ok() {
    n=$((n + 1))
    if [ "$2" = 0 ]; then echo "ok $1"; else bad=$((bad + 1)); echo "FAIL $1"; fi
  }
  printf '%s\t%s\t%s\n' bloomery-serve-ds41 deepseek41,vision,gemma-embedding2 'lib a; lib b' gate_kquant gpu 'bin gate_kquant' gate_swap gpu 'bin gate_swap' > "$t/set.tsv"
  printf '#!/usr/bin/env bash\necho "scan $*"\n' > "$t/ptx-scan.sh"
  # the runner: one build per distinct feature list, the bins grouped in the set's order, one extractor build, one scan
  # block per bin, the scanner embedded verbatim
  make_runner "$t/set.tsv" "$t/ptx-scan.sh" "$t/runner.sh"; ok "runner written" $?
  bash -n "$t/runner.sh"; ok "runner parses" $?
  [ "$(grep -c '^cargo oxide build' "$t/runner.sh")" = 2 ]; ok "runner: one cargo oxide build per feature list" $?
  grep -qF "cargo oxide build --arch $ARCH -- -p $PKG --features deepseek41,vision,gemma-embedding2 --release --bin bloomery-serve-ds41 >&2" "$t/runner.sh"; ok "runner: the V4.1 group" $?
  grep -qF "cargo oxide build --arch $ARCH -- -p $PKG --features gpu --release --bin gate_kquant --bin gate_swap >&2" "$t/runner.sh"; ok "runner: the gpu group takes both bins" $?
  [ "$(grep -c "^cargo build --release -p $PKG --bin oxart_ptx" "$t/runner.sh")" = 1 ]; ok "runner: one extractor build" $?
  [ "$(grep -c -- '--no-jit' "$t/runner.sh")" = 3 ]; ok "runner: one --no-jit scan per bin" $?
  sed -n "/^cat > \"\$P\" <<'NARROW_PTX_SCAN_EOF'$/,/^NARROW_PTX_SCAN_EOF$/p" "$t/runner.sh" | sed '1d;$d' | cmp -s - "$t/ptx-scan.sh"; ok "runner: ptx-scan.sh embedded verbatim" $?
  [ "$(grep -c '^echo "narrow-scan: wall: ' "$t/runner.sh")" = 6 ]; ok "runner: a wall line per build group, the extractor and each scan" $?
  case $(side_command 900 .p-1) in *"cargo oxide"*) ;; *) false ;; esac; ok "side command: names cargo oxide, so box.sh checks the backend" $?
  rc=0; got=$(cd "$t" && bash -c "$(side_command 900 .p-empty)" < /dev/null 2>&1) || rc=$?
  [ "$rc" = 65 ] && grep -qF "the runner arrived empty on stdin" <<< "$got" && [ ! -e "$t/.p-empty" ]; ok "side command: an empty runner ends 65 by name, never an empty scan's 0" $?
  grep -qE -- '^ +-u BLOOMERY_BOX_QUEUE -u BLOOMERY_Q_CLASS -u BLOOMERY_Q_CARD -u BLOOMERY_Q_PRED_S GIT_CEILING' "$0"; ok "side call: the queue's settings are unset, so the runner's stdin reaches the box" $?
  case $(side_command 900 .p-1) in *"timeout --kill-after=10 900 bash"*) ;; *) false ;; esac; ok "side command: the runner runs under the bound" $?
  case $(side_command 900 .p-1) in *'echo $! > .p-1'*) ;; *) false ;; esac; ok "side command: the runner's pid is recorded in the named file" $?
  case $(side_command 900 .p-1) in *'rm -f "$T" .p-1'*) ;; *) false ;; esac; ok "side command: the pid file goes when the runner ends" $?
  sc=$(stop_command .p-1 2> /dev/null)
  ro=$(sed -n "s/^    ro_build='\(.*\)'$/\1/p" "$ROOT/tools/box.sh")
  [ -n "$sc" ] && [ -n "$ro" ] && ! [[ $sc =~ $ro ]] && [[ $sc != *target/* ]]; ok "drift: box.sh's READONLY mode accepts the stop command (no build word, no target/)" $?
  got=$(cd "$t" && bash -c "$(stop_command .no-such.pid 2> /dev/null)" 2>&1)
  grep -qF "no remote pid recorded in .no-such.pid" <<< "$got"; ok "stop command: with no pid file it says so and signals nothing" $?
  grep -q "the build with features gpu failed" "$t/runner.sh" && grep -q "the extractor build failed" "$t/runner.sh"; ok "runner: a failed build ends it by name" $?
  # --keep-ptx: the runner keeps each bin's modules and sends them after the scan's block; unset it is the plain runner
  make_runner "$t/set.tsv" "$t/ptx-scan.sh" "$t/runner-0.sh" 0 && cmp -s "$t/runner.sh" "$t/runner-0.sh"; ok "keep off: an explicit 0 writes the runner the default writes" $?
  ! grep -qE 'PTX_SCAN_KEEP|ptx begin|ptx end|^K=|^tar ' "$t/runner.sh"; ok "keep off: the runner names no keep dir, no ptx block, no tar" $?
  make_runner "$t/set.tsv" "$t/ptx-scan.sh" "$t/runner-keep.sh" 1; ok "keep on: runner written" $?
  bash -n "$t/runner-keep.sh"; ok "keep on: runner parses" $?
  [ "$(grep -c 'PTX_SCAN_KEEP="$K/' "$t/runner-keep.sh")" = 3 ] && [ "$(grep -c -- '--no-jit' "$t/runner-keep.sh")" = 3 ]; ok "keep on: each bin's --no-jit scan keeps its modules under its own dir" $?
  [ "$(grep -c '^echo .=== narrow-scan ptx begin ' "$t/runner-keep.sh")" = 3 ] && [ "$(grep -c '^echo "=== narrow-scan ptx end ' "$t/runner-keep.sh")" = 3 ] && [ "$(grep -c '^tar -czf - -C "$K/' "$t/runner-keep.sh")" = 3 ]; ok "keep on: a tar block per bin, after the scan's own end marker" $?
  awk '/narrow-scan end /{e=1; next} /narrow-scan ptx begin/{if (!e) bad=1; e=0} END{exit bad}' "$t/runner-keep.sh"; ok "keep on: the ptx block is outside the scan's block" $?
  sed -e '/^K=/d' -e '/narrow-scan ptx begin/d' -e '/^tar -czf/d' -e '/narrow-scan ptx end/d' -e '/^rm -rf "$K"$/d' -e 's/PTX_SCAN_KEEP="$K\/[^"]*" //' "$t/runner-keep.sh" | cmp -s - "$t/runner.sh"; ok "keep on: the keep runner less its keep lines is the plain runner, byte for byte" $?
  # split_logs
  {
    echo 'noise before'; echo '=== narrow-scan begin a'; echo 'row 1'; echo 'row 2'; echo '=== narrow-scan end a rc=0'
    echo '=== narrow-scan begin b'; echo 'row 3'; echo '=== narrow-scan end b rc=0'
  } > "$t/good.out"
  got=$(split_logs "$t/good.out" "$t/good" a b 2>&1); rc=$?
  [ "$rc" = 0 ] && [ "$(cat "$t/good/a.log")" = "$(printf 'row 1\nrow 2')" ] && [ "$(cat "$t/good/b.log")" = row\ 3 ]; ok "split: two blocks, noise outside them dropped" $?
  { echo '=== narrow-scan begin a'; echo 'row'; echo '=== narrow-scan end a rc=1'; } > "$t/rc1.out"
  got=$(split_logs "$t/rc1.out" "$t/rc1" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "the scan of a ended rc=1" <<< "$got"; ok "split: a scan that ended rc=1 is refused by name" $?
  { echo '=== narrow-scan begin a'; echo 'row'; } > "$t/cut.out"
  got=$(split_logs "$t/cut.out" "$t/cut" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "has no end marker: the box output was cut" <<< "$got"; ok "split: a cut output is refused by name" $?
  got=$(split_logs "$t/good.out" "$t/miss" a b c 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "no scan block for c" <<< "$got"; ok "split: a bin with no block is refused by name" $?
  { echo '=== narrow-scan begin a'; echo '=== narrow-scan begin b'; echo '=== narrow-scan end b rc=0'; } > "$t/nest.out"
  got=$(split_logs "$t/nest.out" "$t/nest" a b 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "block a has no end marker" <<< "$got"; ok "split: a begin inside a block is refused by name" $?
  # split_ptx <scan.out> <dir> <bin>...: the modules the keep runner sent come back as <dir>/<bin>.ptx/mod<N>.ptx
  mkdir -p "$t/mods"
  printf 'module one\n' > "$t/mods/mod1.ptx"
  printf 'module two\n' > "$t/mods/mod2.ptx"
  pb=$(tar -czf - -C "$t/mods" . | base64)
  { echo '=== narrow-scan begin a'; echo 'row'; echo '=== narrow-scan end a rc=0'
    echo '=== narrow-scan ptx begin a'; echo "$pb"; echo '=== narrow-scan ptx end a rc=0'; } > "$t/ptx-good.out"
  got=$(split_ptx "$t/ptx-good.out" "$t/ptx-good" a 2>&1); rc=$?
  [ "$rc" = 0 ] && [ "$(cat "$t/ptx-good/a.ptx/mod1.ptx")" = "module one" ] && [ "$(cat "$t/ptx-good/a.ptx/mod2.ptx")" = "module two" ] \
    && [ "$(ls "$t/ptx-good/a.ptx" | wc -l | tr -d ' ')" = 2 ] && [ ! -e "$t/ptx-good/a.ptx.b64" ]; ok "split_ptx: two modules come back under <bin>.ptx/, no leftover" $?
  split_logs "$t/ptx-good.out" "$t/ptx-good-logs" a > /dev/null 2>&1 && [ "$(cat "$t/ptx-good-logs/a.log")" = row ]; ok "split_ptx: the scan log's split ignores the ptx block (the log holds the scan's rows only)" $?
  { echo '=== narrow-scan begin a'; echo 'row'; echo '=== narrow-scan end a rc=0'; } > "$t/ptx-none.out"
  got=$(split_ptx "$t/ptx-none.out" "$t/ptx-none" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "no ptx block for a" <<< "$got"; ok "split_ptx: a bin with no ptx block is refused by name" $?
  { echo '=== narrow-scan ptx begin a'; echo "$pb"; } > "$t/ptx-cut.out"
  got=$(split_ptx "$t/ptx-cut.out" "$t/ptx-cut" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "ptx block a has no end marker: the box output was cut" <<< "$got"; ok "split_ptx: a cut ptx block is refused by name" $?
  { echo '=== narrow-scan ptx begin a'; echo '=== narrow-scan ptx end a rc=2'; } > "$t/ptx-rc.out"
  got=$(split_ptx "$t/ptx-rc.out" "$t/ptx-rc" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "the ptx block of a ended rc=2" <<< "$got"; ok "split_ptx: a tar that ended rc=2 is refused by name" $?
  { echo '=== narrow-scan ptx begin a'; echo 'not base64 !!'; echo '=== narrow-scan ptx end a rc=0'; } > "$t/ptx-b64.out"
  got=$(split_ptx "$t/ptx-b64.out" "$t/ptx-b64" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "a: the ptx block does not decode" <<< "$got"; ok "split_ptx: a block that is not a gzipped tar is refused by name" $?
  mkdir -p "$t/mods-empty"
  { echo '=== narrow-scan ptx begin a'; tar -czf - -C "$t/mods-empty" . | base64; echo '=== narrow-scan ptx end a rc=0'; } > "$t/ptx-empty.out"
  got=$(split_ptx "$t/ptx-empty.out" "$t/ptx-empty" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "a: the ptx block holds no module" <<< "$got"; ok "split_ptx: a block with no module is refused by name" $?
  mkdir -p "$t/mods-bad/sub"
  printf 'x\n' > "$t/mods-bad/sub/mod1.ptx"
  { echo '=== narrow-scan ptx begin a'; tar -czf - -C "$t/mods-bad" . | base64; echo '=== narrow-scan ptx end a rc=0'; } > "$t/ptx-sub.out"
  got=$(split_ptx "$t/ptx-sub.out" "$t/ptx-sub" a 2>&1); rc=$?
  [ "$rc" != 0 ] && grep -qF "a: the ptx block holds a member that is not mod<N>.ptx" <<< "$got" && [ ! -e "$t/ptx-sub/a.ptx/sub" ]; ok "split_ptx: a member that is not a flat mod<N>.ptx is refused by name, nothing written for it" $?
  # the lock: a free lock is taken, a held one is refused at its bound naming the holder, a dead holder is taken over.
  # The holder is another process. A subshell of this one keeps $$ as this shell, and would inherit
  # fd 9 if this shell already held the flock, so the holder is its own bash.
  rm -rf "$t/lock.d" "$t/lock.d.flock"
  export -f lock_take
  bash -c 'lock_take "$1" 0 "test run" 2>/dev/null || exit 1; exec sleep 30' _ "$t/lock.d" &
  hp=$!
  k=0
  while [ "$k" -lt 50 ]; do
    [ -f "$t/lock.d.flock" ] && [ "$(head -n 1 "$t/lock.d.flock" 2> /dev/null || true)" = "$hp" ] && break
    kill -0 "$hp" 2> /dev/null || break
    sleep 0.05
    k=$((k + 1))
  done
  [ "$(head -n 1 "$t/lock.d.flock" 2> /dev/null || true)" = "$hp" ]; ok "lock: a free lock is taken" $?
  [ -d "$t/lock.d" ] && [ "$(cat "$t/lock.d/pid" 2> /dev/null || true)" = "$hp" ] && ! mkdir "$t/lock.d" 2> /dev/null
  ok "lock: the held directory is the holder's, so an older copy's mkdir fails" $?
  got=$( (lock_take "$t/lock.d" 0 "second" 2>&1); echo "rc=$?")
  grep -q "rc=75" <<< "$got" && grep -qF "held by pid $hp (test run)" <<< "$got"; ok "lock: a held lock is refused rc 75 naming its holder" $?
  kill -TERM "$hp" 2> /dev/null || true
  wait "$hp" 2> /dev/null || true
  lock_take "$t/lock.d" 0 "after the holder" 2> /dev/null; ok "lock: the hold ends when its process does, and a later take succeeds" $?
  [ -d "$t/lock.d" ] && [ "$(cat "$t/lock.d/pid" 2> /dev/null || true)" = "$$" ]; ok "lock: a take creates the directory with this pid" $?
  lock_release
  [ ! -d "$t/lock.d" ]; ok "lock: release removes the directory when the pid is this process" $?
  ( lock_take "$t/lock.d" 0 "after release" > /dev/null 2>&1 ); ok "lock: release drops this process's hold" $?
  lock_release
  rm -rf "$t/lock.d" "$t/lock.d.flock"
  mkdir "$t/lock.d" && echo 99999999 > "$t/lock.d/pid" && echo gone > "$t/lock.d/who"
  got=$(lock_take "$t/lock.d" 0 "after the dead one" 2>&1); rc=$?
  [ "$rc" = 0 ] && grep -qF "is gone: taking it over" <<< "$got"; ok "lock: a dead holder is taken over" $?
  lock_release
  rm -rf "$t/lock.d" "$t/lock.d.flock"
  # Two recoverers both see the dead pid before either removes the directory. Exactly one may
  # come back holding the lock; the other is contention (75). The one that took it keeps the fd
  # until the other's rc is recorded, so a fast exit is not a second success.
  mkdir "$t/lock.d"
  echo 99999999 > "$t/lock.d/pid"
  echo gone > "$t/lock.d/who"
  rec=$t/recover-sync
  rm -rf "$rec"
  mkdir "$rec"
  kill() {
    if [ "$1" = "-0" ]; then
      echo $$ >> "$rec/pids"
      local k=0 n=0
      while [ "$k" -lt 40 ]; do
        n=$(wc -l < "$rec/pids" 2>/dev/null || echo 0)
        n=${n//[[:space:]]/}
        [ "${n:-0}" -ge 2 ] && break
        sleep 0.05
        k=$((k + 1))
      done
      return 1
    fi
    builtin kill "$@"
  }
  ( lock_take "$t/lock.d" 0 "recover-a" >/dev/null 2>&1; echo $? > "$rec/a.rc"; [ "$(cat "$rec/a.rc")" = 0 ] && exec sleep 30 ) &
  pa=$!
  ( lock_take "$t/lock.d" 0 "recover-b" >/dev/null 2>&1; echo $? > "$rec/b.rc"; [ "$(cat "$rec/b.rc")" = 0 ] && exec sleep 30 ) &
  pb=$!
  k=0
  while [ "$k" -lt 80 ]; do
    [ -f "$rec/a.rc" ] && [ -f "$rec/b.rc" ] && break
    sleep 0.05
    k=$((k + 1))
  done
  unset -f kill
  kill -TERM "$pa" "$pb" 2> /dev/null || true
  wait "$pa" "$pb" 2> /dev/null || true
  ra=$(cat "$rec/a.rc" 2> /dev/null || true)
  rb=$(cat "$rec/b.rc" 2> /dev/null || true)
  wins=0
  [ "$ra" = 0 ] && wins=$((wins + 1))
  [ "$rb" = 0 ] && wins=$((wins + 1))
  [ "$wins" = 1 ] && { [ "$ra" = 75 ] || [ "$rb" = 75 ]; }
  ok "lock: two recoverers of a dead holder, and only one takes it" $?
  rm -rf "$t/lock.d" "$t/lock.d.flock"
  mkdir "$t/lock.d" && echo $$ > "$t/lock.d/pid" && echo older > "$t/lock.d/who"
  got=$(lock_take "$t/lock.d" 0 "second" 2>&1); rc=$?
  lock_release
  [ "$rc" = 75 ] && [ -d "$t/lock.d" ] && grep -qF "held by pid $$ (older)" <<< "$got"; ok "lock: a live older copy's directory is refused, not deleted" $?
  rm -rf "$t/lock.d" "$t/lock.d.flock"
  mkdir "$t/lock.d"
  : > "$t/lock.d/pid"
  got=$(lock_take "$t/lock.d" 0 "second" 2>&1); rc=$?
  lock_release
  [ "$rc" = 75 ] && [ -d "$t/lock.d" ] && grep -qF "held by pid ?" <<< "$got"
  ok "lock: a directory whose pid is not written yet is not taken" $?
  # A child inherits fd 9, so it keeps the flock after the holder exits. The same child
  # started with the fd closed does not. Every launch below closes it; a line that loses
  # the close is red.
  lock_release
  rm -rf "$t/lock.d" "$t/lock.d.flock" "$t/sleep.pid"
  export -f lock_take
  bash -c 'lock_take "$1" 0 "inherit" >/dev/null 2>&1 || exit 1; sleep 30 & echo $! > "$2"; wait' _ "$t/lock.d" "$t/sleep.pid" &
  hp=$!
  k=0
  while [ "$k" -lt 50 ]; do
    [ -s "$t/sleep.pid" ] && [ "$(head -n 1 "$t/lock.d.flock" 2> /dev/null || true)" = "$hp" ] && break
    kill -0 "$hp" 2> /dev/null || break
    sleep 0.05
    k=$((k + 1))
  done
  sp=$(cat "$t/sleep.pid" 2> /dev/null || true)
  kill -TERM "$hp" 2> /dev/null || true
  wait "$hp" 2> /dev/null || true
  got=$( (lock_take "$t/lock.d" 0 "after the parent" 2>&1); echo "rc=$?")
  grep -q "rc=75" <<< "$got" && [ -n "$sp" ] && kill -0 "$sp" 2> /dev/null
  ok "lock: a child that inherited the fd keeps it after the holder exits" $?
  kill -TERM "$sp" 2> /dev/null || true
  wait "$sp" 2> /dev/null || true
  rm -rf "$t/lock.d" "$t/lock.d.flock" "$t/sleep.pid"
  bash -c 'lock_take "$1" 0 "closed" >/dev/null 2>&1 || exit 1; sleep 30 9>&- & echo $! > "$2"; wait' _ "$t/lock.d" "$t/sleep.pid" &
  hp=$!
  k=0
  while [ "$k" -lt 50 ]; do
    [ -s "$t/sleep.pid" ] && [ "$(head -n 1 "$t/lock.d.flock" 2> /dev/null || true)" = "$hp" ] && break
    kill -0 "$hp" 2> /dev/null || break
    sleep 0.05
    k=$((k + 1))
  done
  sp=$(cat "$t/sleep.pid" 2> /dev/null || true)
  kill -TERM "$hp" 2> /dev/null || true
  wait "$hp" 2> /dev/null || true
  lock_take "$t/lock.d" 0 "after the closed child" > /dev/null 2>&1
  ok "lock: a child started with the fd closed does not keep it" $?
  lock_release
  kill -TERM "$sp" 2> /dev/null || true
  wait "$sp" 2> /dev/null || true
  # The strings sit in this function, so the check reads the script with this function removed.
  # A child blocked opening a FIFO keeps fd 9 when the close comes after that open. These launches
  # are the script's own lines: the holder is killed while the child waits, and the lock is free.
  body=$(awk '/^self_test\(\)/{skip=1} skip && /^}$/{skip=0; next} !skip' "$0")
  fifo_blocked() {
    local label=$1 launch=$2 hp k kid rc
    rm -rf "$t/lock.d" "$t/lock.d.flock"
    rm -f "$t/side.fifo" "$t/side.build.log" "$t/side.kid"
    mkfifo "$t/side.fifo"
    export -f lock_take
    # $! is the launch, not lock_take's python. The pause lets that child close fd 9 before the holder dies.
    bash -c 'lock_take "$1" 0 "fifo" >/dev/null 2>&1 || exit 1
      out=$2 name=side fifo=$3
      eval "$4"
      echo $! > "$5"
      exec sleep 30' _ "$t/lock.d" "$t" "$t/side.fifo" "$launch" "$t/side.kid" &
    hp=$!
    kid='' k=0
    while [ "$k" -lt 50 ]; do
      [ -s "$t/side.kid" ] && break
      kill -0 "$hp" 2> /dev/null || break
      sleep 0.05
      k=$((k + 1))
    done
    kid=$(cat "$t/side.kid" 2> /dev/null || true)
    sleep 0.2
    kill -TERM "$hp" 2> /dev/null || true
    wait "$hp" 2> /dev/null || true
    rc=1
    if [ -n "$kid" ]; then
      lock_take "$t/lock.d" 0 "after the blocked child" >/dev/null 2>&1
      rc=$?
      kill -TERM "$kid" 2> /dev/null || true
      wait "$kid" 2> /dev/null || true
    fi
    lock_release
    rm -f "$t/side.fifo" "$t/side.build.log"
    [ "$rc" = 0 ]
    ok "$label" $?
  }
  tee_line=$(grep -F 'tee "$out/$name.build.log"' <<< "$body" | head -n 1)
  tee_line=${tee_line#"${tee_line%%[![:space:]]*}"}
  fifo_blocked "lock: a logger blocked on the FIFO does not keep the flock after the holder exits" "$tee_line"
  wline=$(grep -F '> "$fifo" &' <<< "$body" | head -n 1)
  fifo_blocked "lock: a writer blocked on the FIFO does not keep the flock after the holder exits" "( : )${wline#*\)}"
  miss=0
  for site in \
    'tee "$out/$name.build.log" 9>&- < "$fifo" >&2 &' \
    './tools/box.sh "$cmd" < "$runner" 2>&1 > "$out/$name.scan.out"' \
    ') 9>&- > "$fifo" &' \
    '"$ROOT/tools/box.sh" "$(stop_command "$2")" < /dev/null >&2 2>&1 9>&- &' \
    'archive_tree "$ROOT" "$rev" "$src" 9>&- ||' \
    'split_logs "$OUT/$side.scan.out" "$OUT/$side" "${BINS[@]}" 9>&-' \
    'split_ptx "$OUT/$side.scan.out" "$OUT/$side" "${BINS[@]}" 9>&-' \
    'WT_KEY1=$(worktree_key "$BASE_SHA" 9>&-)' \
    '"$ROOT/tools/affected-gates.sh" "$AFFECTED_RANGE" --narrow "${ARGS[@]}" 9>&-' \
    'sleep 0.2 9>&-' \
    'rm -rf "$src" 9>&-' \
    'mkdir -p "$src" 9>&-' \
    'rm -f "$fifo" 9>&-' \
    'mkfifo "$fifo" 9>&-' \
    'rm -f "$fifo" "$ACTIVE_PIDF" 9>&-' \
    'cat "$OUT/$side.split.err" >&2 9>&-' \
    "'; } 9>&-)| \$({" \
    'cat "$OUT/narrowed.txt" 9>&-' \
    'cat "$OUT/narrowed.err" >&2 9>&-' \
    'ptx_scan_md5=$(md5_of "$ROOT/tools/ptx-scan.sh" 9>&-)' \
    'cp -R "$CACHE_ENTRY/base/." "$OUT/base/" 9>&-' \
    'rm -rf "$CACHE_ENTRY" 9>&-' \
    'cache_check "$CACHE_ENTRY" "$OUT/base-cache.key" "${BINS[@]}" 9>&-'
  do
    grep -qF "$site" <<< "$body" || miss=1
  done
  [ "$miss" = 0 ]; ok "lock: a launch while the lock is held closes fd 9" $?
  # the runner's build lines are the ptx-scan recipe's: a recipe edit that this script does not follow is red
  recipe=$(awk '/^ptx-scan /{f=1; next} f && /^[^ ]/{exit} f' "$ROOT/justfile")
  grep -qF "cargo oxide build --arch {{arch}} -- -p $PKG --features {{FEATURES}} --release --bin {{BIN}}" <<< "$recipe"; ok "drift: the ptx-scan recipe still builds with this runner's cargo oxide line" $?
  grep -qF "cargo build --release -p $PKG --bin oxart_ptx && bash tools/ptx-scan.sh" <<< "$recipe"; ok "drift: the ptx-scan recipe still builds the extractor as this runner does" $?
  grep -qF "NARROW_PTX_SCAN_EOF" "$ROOT/tools/ptx-scan.sh"; [ $? = 1 ]; ok "drift: ptx-scan.sh does not hold the runner's here-document marker" $?
  # another arch is refused by name before any git or box call, and the scan's own arch is not
  got=$(env BLOOMERY_ARCH=sm_121 bash "$0" HEAD --dry-run 2>&1); rc=$?
  [ "$rc" = 64 ] && grep -qF "narrow-scan: BLOOMERY_ARCH is 'sm_121': the scan pairs are sm_86's" <<< "$got"; ok "arch: another BLOOMERY_ARCH is refused by name (64)" $?
  # --worktree: this script in a throwaway repository beside stub tools (no box, no card), so the whole run goes through
  # the real argument handling, the side loop and the affected call. The box stub logs which content each side synced.
  w=$t/bloomery-wt
  mkdir -p "$w/tools" "$t/home"
  cp "$0" "$w/tools/narrow-scan.sh"
  # the scan set per tree: STUB_FEATS_<rev> sets gate_x's features (rev: the commit, or `tree` for the working tree), STUB_EXTRA_<rev>
  # adds a second bin, as a tree whose change adds a cargo feature or a kernel carrier would print
  printf '%s\n' 'import os, sys' 'open(os.environ["STUB_LOG"], "a").write("recipes " + " ".join(sys.argv[1:]) + "\n")' \
    'rev = sys.argv[2] if len(sys.argv) > 2 else "tree"' \
    'print("gate_x\t" + os.environ.get("STUB_FEATS_" + rev, "gpu") + "\tbin gate_x")' \
    'extra = os.environ.get("STUB_EXTRA_" + rev)' \
    'if extra:' '    print(extra + "\tgpu\tbin " + extra)' > "$w/tools/recipes.py"
  printf '#!/usr/bin/env bash\necho scan\n' > "$w/tools/ptx-scan.sh"
  printf '#!/usr/bin/env bash\necho "affected $*" >> "$STUB_LOG"\necho "affected stub"\n' > "$w/tools/affected-gates.sh"
  cat > "$w/tools/box.sh" <<'NARROW_STUB_EOF'
#!/usr/bin/env bash
here=$(cd "$(dirname "$0")/.." && pwd)
# The stop's read-only call: its command runs in the fake remote dir, as box.sh's READONLY mode runs it on the box.
if [ "${BLOOMERY_BOX_READONLY:-0}" = 1 ]; then
  echo "readonly $BLOOMERY_REMOTE" >> "$STUB_LOG"
  cd "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")" || exit 66
  exec bash -c "$*"
fi
side=head
case $BLOOMERY_REMOTE in rb | *narrow-base) side=base ;; esac
# STUB_FAIL_SIDE=<side> STUB_FAIL_RC=<rc>: that side's first box call ends <rc> (255: ssh lost the box) before it runs
# anything; STUB_FAIL_ALWAYS=1: every call of that side does.
if [ "${STUB_FAIL_SIDE:-}" = "$side" ] && { [ -n "${STUB_FAIL_ALWAYS:-}" ] || [ ! -e "$STUB_LOG.failed-$side" ]; }; then
  : > "$STUB_LOG.failed-$side"
  echo "$side $BLOOMERY_REMOTE failed rc=$STUB_FAIL_RC" >> "$STUB_LOG"
  cat > /dev/null
  exit "$STUB_FAIL_RC"
fi
# STUB_HANG: the box.sh itself is held before it starts anything on the box (the sync, the guard's wait).
if [ -n "${STUB_HANG:-}" ]; then
  mkdir -p "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")"
  echo $$ > "$STUB_REMOTE/hang.pid"
  sleep 60 &
  echo $! > "$STUB_REMOTE/hang-sleep.pid"
  wait $!
  exit 0
fi
# STUB_BLOCK: the box command itself runs in the fake remote dir, with a runner that never ends.
if [ -n "${STUB_BLOCK:-}" ]; then
  mkdir -p "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")" && cd "$STUB_REMOTE/$(basename "$BLOOMERY_REMOTE")" || exit 70
  echo "$side $BLOOMERY_REMOTE blocked" >> "$STUB_LOG"
  printf 'exec sleep 60\n' | PATH="$STUB_BIN:$PATH" bash -c "$*"
  exit $?
fi
rf=$(mktemp)
cat > "$rf"
new=no
[ -e "$here/new.txt" ] && new=yes
echo "$side $BLOOMERY_REMOTE marker=$(cat "$here/marker.txt") new=$new" >> "$STUB_LOG"
echo "runner $side $(grep -o -- '--features [^ ]*' "$rf" | cut -d' ' -f2 | tr '\n' ' ' | sed 's/ $//')" >> "$STUB_LOG"
[ "$side" = head ] && [ -n "${STUB_EDIT:-}" ] && echo more >> "$here/marker.txt"
echo '=== narrow-scan begin gate_x'
echo 'ptx-scan: mod0 bundle=b bytes=1'
echo "=== narrow-scan end gate_x rc=${STUB_SCANRC:-0}"
# a keep runner (it names PTX_SCAN_KEEP) is answered with the ptx block it would print; STUB_NOPTX leaves it out
if grep -q 'PTX_SCAN_KEEP' "$rf" && [ -z "${STUB_NOPTX:-}" ]; then
  pd=$(mktemp -d)
  echo "$side module" > "$pd/mod1.ptx"
  echo '=== narrow-scan ptx begin gate_x'
  tar -czf - -C "$pd" . | base64
  echo '=== narrow-scan ptx end gate_x rc=0'
  rm -rf "$pd"
fi
rm -f "$rf"
NARROW_STUB_EOF
  chmod +x "$w/tools/"*.sh
  mkdir -p "$w/tools/nightly"
  cp "$ROOT/tools/nightly/units.sh" "$w/tools/nightly/units.sh"
  git -C "$w" init -q && git -C "$w" config user.email test@test && git -C "$w" config user.name test && git -C "$w" config commit.gpgsign false
  echo base > "$w/marker.txt"
  git -C "$w" add -A && git -C "$w" commit -q -m base
  wb=$(git -C "$w" rev-parse HEAD)
  echo edited > "$w/marker.txt"
  echo new > "$w/new.txt"
  # wt_run <stub log> <narrow-scan args...>: the run's output in $got, its code in $rc
  wt_run() {
    local log=$1
    shift
    : > "$log"
    got=$(HOME="$t/home" STUB_LOG="$log" NARROW_BASE_REMOTE=rb NARROW_HEAD_REMOTE=rh bash "$w/tools/narrow-scan.sh" "$@" 2>&1)
    rc=$?
  }
  wt_run "$t/wt.log" "$wb" --worktree --out "$t/wt-out"
  [ "$rc" = 0 ]; ok "worktree: the run ends rc 0" $?
  grep -qx "head rh marker=edited new=yes" "$t/wt.log"; ok "worktree: the head side synced the edited and the untracked file" $?
  grep -qx "base rb marker=base new=no" "$t/wt.log"; ok "worktree: the base side is BASE's commit" $?
  grep -qx "recipes scan-set" "$t/wt.log"; ok "worktree: the scan set is read from the tree, no commit named" $?
  grep -qx "affected $wb --narrow --scan $t/wt-out/base/gate_x.log $t/wt-out/head/gate_x.log" "$t/wt.log"; ok "worktree: affected reads the working tree against BASE with the pair" $?
  grep -qx "head=worktree@$wb" "$t/wt-out/MANIFEST" && grep -q '^worktree_key=[0-9a-f]\{32\}$' "$t/wt-out/MANIFEST"; ok "worktree: the MANIFEST names the tree and its key" $?
  wt_run "$t/cm.log" "$wb" "$wb" --out "$t/cm-out"
  [ "$rc" = 0 ] && grep -qx "head rh marker=base new=no" "$t/cm.log" && grep -qx "recipes scan-set $wb" "$t/cm.log" && grep -qx "affected $wb..$wb --narrow --scan $t/cm-out/base/gate_x.log $t/cm-out/head/gate_x.log" "$t/cm.log"
  ok "worktree: a commit head side still reads the commit, not the edit (the contrast)" $?
  grep -qF "narrow-scan: note: this tree has uncommitted changes; the scans read the commits" <<< "$got"; ok "note: a dirty tree is named when the head side is a commit" $?
  # Each side's scan set is its own tree's. A change that adds a cargo feature has a head scan set that names it; BASE cannot build it.
  # Mutant: both sides take the head tree's set, the base side then asks its tree for a feature it lacks.
  grep -qx "runner base gpu" "$t/wt.log" && grep -qx "runner head gpu" "$t/wt.log" && grep -qx "recipes scan-set $wb" "$t/wt.log"; ok "scan set: equal sets build the same features on both sides, and the base set is read from BASE's tree" $?
  STUB_FEATS_tree=gpu,hf wt_run "$t/fd.log" "$wb" --worktree --scan-only --out "$t/fd-out"
  [ "$rc" = 0 ] && grep -qx "runner base gpu" "$t/fd.log" && grep -qx "runner head gpu,hf" "$t/fd.log"; ok "scan set: a feature only the head tree has is built on the head side, never asked of the base side" $?
  grep -qF "narrow-scan: bin gate_x: features differ between the sides: base gpu, head gpu,hf" <<< "$got" && grep -qx 'scan_set_differs=1' "$t/fd-out/MANIFEST" && [ -s "$t/fd-out/runner-base.sh" ]; ok "scan set: the bin whose features differ is named with both sets, in the output and the MANIFEST" $?
  STUB_EXTRA_tree=gate_y wt_run "$t/fe.log" "$wb" --worktree --scan-only --out "$t/fe-out"
  [ "$rc" = 2 ] && grep -qF "narrow-scan: bin gate_y: in the head scan set only" <<< "$got" && ! grep -qE '^(base|head) ' "$t/fe.log"; ok "scan set: a bin only one side scans ends rc 2 by name before any box call" $?
  STUB_EXTRA_tree=gate_y wt_run "$t/fe2.log" "$wb" --worktree --dry-run --out "$t/fe2-out"
  [ "$rc" = 2 ] && grep -qF "the sides' scan sets name different bins" <<< "$got"; ok "scan set: --dry-run refuses the same way" $?
  STUB_FEATS_tree=gpu,hf wt_run "$t/fdd.log" "$wb" --worktree --dry-run --out "$t/fdd-out"
  [ "$rc" = 0 ] && grep -qF "the base side runs $t/fdd-out/runner-base.sh, the head side $t/fdd-out/runner.sh" <<< "$got" && ! grep -qE '^(base|head) ' "$t/fdd.log"; ok "scan set: --dry-run names the runner each side runs when they differ" $?
  # A `git status` that fails (a corrupt index, HEAD readable) prints nothing, which reads as a clean tree. The note names it unknown and
  # does not claim a dirty or a clean tree; the run goes on, since its scans read commits. Mutant: the failed status read as a clean tree,
  # no note at all.
  printf garbage > "$t/narrow-badindex"
  GIT_INDEX_FILE="$t/narrow-badindex" wt_run "$t/cmb.log" "$wb" "$wb" --out "$t/cmb-out"
  [ "$rc" = 0 ] && grep -Eq "narrow-scan: note: git status failed in .* \(git exit 128\): whether this tree has uncommitted changes is unknown" <<< "$got" \
    && ! grep -qF "has uncommitted changes;" <<< "$got"; ok "note: a git status that fails is named unknown, never read as a clean tree" $?
  STUB_EDIT=1 wt_run "$t/edit.log" "$wb" --worktree --out "$t/edit-out"
  [ "$rc" = 1 ] && grep -qF "the working tree changed during the run" <<< "$got"; ok "worktree: a tree edited during the run is refused by name" $?
  wt_run "$t/arg.log" "$wb" "$wb" --worktree --out "$t/arg-out"
  [ "$rc" = 64 ] && grep -qF -- "--worktree takes no HEAD" <<< "$got"; ok "worktree: a HEAD beside --worktree is refused rc 64 by name" $?
  # --keep-ptx and --scan-only. Unset, the run is the one above: no ptx dir, the affected run, the MANIFEST as before.
  [ ! -e "$t/wt-out/base/gate_x.ptx" ] && [ ! -e "$t/wt-out/head/gate_x.ptx" ] && ! grep -q 'keep_ptx' "$t/wt-out/MANIFEST" && grep -qx 'affected_rc=0' "$t/wt-out/MANIFEST" \
    && ! grep -q 'PTX_SCAN_KEEP' "$t/wt-out/runner.sh"; ok "keep off: no ptx dir, no runner keep line, the MANIFEST as before" $?
  # ssh rc 255 on a side: the side runs once more (its recorded remote runner is ended first, by the READONLY stop call); a second
  # 255 or any other code is not retried.
  mkdir -p "$t/remote/rh"
  STUB_REMOTE=$t/remote STUB_FAIL_SIDE=head STUB_FAIL_RC=255 wt_run "$t/r255.log" "$wb" --worktree --out "$t/r255-out"
  [ "$rc" = 0 ] && [ "$(grep -c '^head rh ' "$t/r255.log")" = 2 ] && grep -qx 'readonly rh' "$t/r255.log" \
    && grep -qF "narrow-scan: head: ssh lost the box (rc 255); ending its recorded remote runner and running the side once more" <<< "$got" \
    && grep -qx 'retried=head' "$t/r255-out/MANIFEST" && grep -q '^narrow-scan: done: .*(retried: head)' <<< "$got"
  ok "retry: a side whose ssh ends rc 255 runs once more, names the retry and the run ends rc 0" $?
  rm -f "$t/r255.log.failed-head"
  STUB_REMOTE=$t/remote STUB_FAIL_SIDE=head STUB_FAIL_RC=255 STUB_FAIL_ALWAYS=1 wt_run "$t/r255b.log" "$wb" --worktree --out "$t/r255b-out"
  [ "$rc" = 255 ] && [ "$(grep -c '^head rh ' "$t/r255b.log")" = 2 ] && grep -qF "narrow-scan: head: ssh lost the box again (rc 255): retried once, not again" <<< "$got"
  ok "retry: a second rc 255 ends the run rc 255 after exactly two tries" $?
  STUB_REMOTE=$t/remote STUB_FAIL_SIDE=head STUB_FAIL_RC=1 wt_run "$t/r1.log" "$wb" --worktree --out "$t/r1-out"
  [ "$rc" = 1 ] && [ "$(grep -c '^head rh ' "$t/r1.log")" = 1 ] && ! grep -q 'once more' <<< "$got"
  ok "retry: a box command that ends rc 1 (a build or a scan) is not retried" $?
  wt_run "$t/keep.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/keep-out"
  [ "$rc" = 0 ] && [ "$(cat "$t/keep-out/base/gate_x.ptx/mod1.ptx" 2> /dev/null)" = "base module" ] && [ "$(cat "$t/keep-out/head/gate_x.ptx/mod1.ptx" 2> /dev/null)" = "head module" ]; ok "keep: each side's modules arrive under DIR/<side>/<bin>.ptx/" $?
  grep -q 'PTX_SCAN_KEEP' "$t/keep-out/runner.sh"; ok "keep: the runner each side runs keeps the modules" $?
  ! grep -q '^affected ' "$t/keep.log" && [ ! -e "$t/keep-out/narrowed.txt" ] && grep -qx 'affected_rc=skipped' "$t/keep-out/MANIFEST" && grep -qx 'keep_ptx=1' "$t/keep-out/MANIFEST"; ok "scan-only: the affected run is skipped and the MANIFEST says so" $?
  [ "$(cat "$t/keep-out/head/gate_x.log")" = "$(cat "$t/wt-out/head/gate_x.log")" ]; ok "keep: the scan log is the unset run's, byte for byte (the ptx block stays out of it)" $?
  STUB_NOPTX=1 wt_run "$t/noptx.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/noptx-out"
  [ "$rc" = 1 ] && grep -qF "no ptx block for gate_x" <<< "$got"; ok "keep: a side that sent no ptx block ends rc 1 by name" $?
  wt_run "$t/so.log" "$wb" --worktree --scan-only --out "$t/so-out"
  [ "$rc" = 0 ] && [ ! -e "$t/so-out/base/gate_x.ptx" ] && grep -qx 'affected_rc=skipped' "$t/so-out/MANIFEST" && ! grep -q 'keep_ptx' "$t/so-out/MANIFEST"; ok "scan-only alone: no ptx dir, no keep line" $?
  wt_run "$t/kd.log" "$wb" --worktree --keep-ptx --dry-run --out "$t/kd-out"
  [ "$rc" = 0 ] && grep -q 'PTX_SCAN_KEEP' "$t/kd-out/runner.sh" && ! grep -qE '^(base|head) ' "$t/kd.log"; ok "keep: --dry-run writes the keep runner and touches no box" $?
  # The base side's cache, ~/.cache/bloomery/narrow-base/<key>/ under each case's own HOME: a run with the same BASE, scanner,
  # scan set, arch and keep flag takes the base side's logs and modules from it and calls no box for that side.
  ch=$t/chome
  cr=$ch/.cache/bloomery/narrow-base
  # cache_run <home> <log> <narrow-scan args...>: a run with the cache on (CACHE_FORCE=1: NARROW_NO_CACHE=1); output in $got, code in $rc
  cache_run() {
    local home=$1 log=$2
    shift 2
    : > "$log"
    got=$(HOME="$home" STUB_LOG="$log" NARROW_NO_CACHE="${CACHE_FORCE:-0}" NARROW_BASE_REMOTE=rb NARROW_HEAD_REMOTE=rh bash "$w/tools/narrow-scan.sh" "$@" 2>&1)
    rc=$?
  }
  # nent <dir>: the entries of the cache root, a temporary directory included
  nent() { ls -A "$1" 2> /dev/null | wc -l | tr -d ' '; }
  cache_run "$ch" "$t/c1.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c1-out"
  ent=$cr/$(ls -A "$cr" 2> /dev/null | head -n 1)
  [ "$rc" = 0 ] && grep -q '^base rb ' "$t/c1.log" && [ "$(nent "$cr")" = 1 ] && [ -s "$ent/base/gate_x.log" ] && [ "$(cat "$ent/base/gate_x.ptx/mod1.ptx" 2> /dev/null)" = "base module" ] \
    && grep -qx "base=$wb" "$ent/MANIFEST" && grep -qx 'keep_ptx=1' "$ent/MANIFEST" && grep -qx "arch=$ARCH" "$ent/MANIFEST" && grep -qx 'bins=gate_x' "$ent/MANIFEST" \
    && grep -qx "ptx_scan_md5=$(md5_of "$w/tools/ptx-scan.sh")" "$ent/MANIFEST" && grep -qx "scan_set_md5=$(md5_of "$t/c1-out/scan-set.tsv")" "$ent/MANIFEST"; ok "cache: a miss runs the base side and keeps its logs, modules and key under <key>/, nothing else in the root" $?
  grep -qx 'base_cache=miss' "$t/c1-out/MANIFEST"; ok "cache: the MANIFEST says miss" $?
  cache_run "$ch" "$t/c2.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c2-out"
  [ "$rc" = 0 ] && ! grep -q '^base rb ' "$t/c2.log" && grep -q '^head rh ' "$t/c2.log" && grep -qxF "narrow-scan: base $wb from cache $ent" <<< "$got"; ok "cache: the same key hits: one line names the cache dir, no box call for the base side, the head side still runs" $?
  grep -q '^narrow-scan: done: base side from cache, head side [0-9]*s' <<< "$got"; ok "cache: the done line says the base side came from the cache, not a wall" $?
  cmp -s "$t/c1-out/base/gate_x.log" "$t/c2-out/base/gate_x.log" && cmp -s "$t/c1-out/base/gate_x.ptx/mod1.ptx" "$t/c2-out/base/gate_x.ptx/mod1.ptx" && grep -qx 'base_cache=hit' "$t/c2-out/MANIFEST"; ok "cache: a hit lays DIR/base/ out as the fresh run did, and the MANIFEST says hit" $?
  # the key holds the base side's set: a feature only the head tree has leaves it, so the base side still comes from the cache
  STUB_FEATS_tree=gpu,hf cache_run "$ch" "$t/c2b.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c2b-out"
  [ "$rc" = 0 ] && ! grep -q '^base rb ' "$t/c2b.log" && grep -q '^head rh ' "$t/c2b.log" && grep -qxF "narrow-scan: base $wb from cache $ent" <<< "$got" && [ "$(nent "$cr")" = 1 ]; ok "cache: a feature only the head tree has still hits (the key holds the base side's scan set)" $?
  cp "$w/tools/ptx-scan.sh" "$t/ptx-scan.orig"
  echo '# the scanner changed' >> "$w/tools/ptx-scan.sh"
  cache_run "$ch" "$t/c3.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c3-out"
  [ "$rc" = 0 ] && grep -q '^base rb ' "$t/c3.log" && [ "$(nent "$cr")" = 2 ]; ok "cache: a changed scanner (its md5) misses, and its entry sits beside the old one" $?
  cp "$t/ptx-scan.orig" "$w/tools/ptx-scan.sh"
  cache_run "$ch" "$t/c4.log" "$wb" --worktree --scan-only --out "$t/c4-out"
  [ "$rc" = 0 ] && grep -q '^base rb ' "$t/c4.log" && [ "$(nent "$cr")" = 3 ] && [ ! -e "$t/c4-out/base/gate_x.ptx" ]; ok "cache: --keep-ptx is part of the key: a run without it misses, and keeps no modules" $?
  CACHE_FORCE=1 cache_run "$ch" "$t/c5.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c5-out"
  [ "$rc" = 0 ] && grep -q '^base rb ' "$t/c5.log" && ! grep -q 'from cache' <<< "$got" && [ "$(nent "$cr")" = 3 ] && [ -s "$ent/base/gate_x.log" ]; ok "cache: NARROW_NO_CACHE=1 forces a miss, and the run's result replaces that key's entry (no leftover temp dir)" $?
  rm -f "$ent/base/gate_x.log"
  cache_run "$ch" "$t/c6.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c6-out"
  [ "$rc" = 0 ] && grep -qF "cache entry $ent is corrupt" <<< "$got" && grep -qF "gate_x.log" <<< "$got" && grep -q '^base rb ' "$t/c6.log" && [ -s "$ent/base/gate_x.log" ] && [ "$(nent "$cr")" = 3 ]; ok "cache: an entry missing a bin's log is named, removed and rebuilt, never reused in part" $?
  rm -f "$ent/MANIFEST"
  cache_run "$ch" "$t/c7.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c7-out"
  [ "$rc" = 0 ] && grep -qF "cache entry $ent is corrupt" <<< "$got" && grep -qF "MANIFEST" <<< "$got" && grep -q '^base rb ' "$t/c7.log" && [ -s "$ent/MANIFEST" ]; ok "cache: an entry with no MANIFEST is named, removed and rebuilt" $?
  rm -rf "$ent/base/gate_x.ptx"
  cache_run "$ch" "$t/c8.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c8-out"
  [ "$rc" = 0 ] && grep -qF "cache entry $ent is corrupt" <<< "$got" && grep -qF "gate_x.ptx" <<< "$got" && grep -q '^base rb ' "$t/c8.log" && [ -s "$ent/base/gate_x.ptx/mod1.ptx" ]; ok "cache: a keep entry with no modules for a bin is named, removed and rebuilt" $?
  STUB_NOPTX=1 cache_run "$t/chome2" "$t/c9.log" "$wb" --worktree --keep-ptx --scan-only --out "$t/c9-out"
  [ "$rc" = 1 ] && [ "$(nent "$t/chome2/.cache/bloomery/narrow-base")" = 0 ]; ok "cache: a base side with no ptx block fails and is not cached" $?
  STUB_SCANRC=1 cache_run "$t/chome2" "$t/c10.log" "$wb" --worktree --scan-only --out "$t/c10-out"
  [ "$rc" = 1 ] && [ "$(nent "$t/chome2/.cache/bloomery/narrow-base")" = 0 ]; ok "cache: a base side whose scan ended rc 1 fails and is not cached" $?
  orphan=$(git -C "$w" commit-tree "$(git -C "$w" rev-parse 'HEAD^{tree}')" -m orphan)
  wt_run "$t/anc.log" "$orphan" --worktree --out "$t/anc-out"
  [ "$rc" = 64 ] && grep -qF "is not an ancestor of this tree's HEAD" <<< "$got" && [ ! -s "$t/anc.log" ]; ok "worktree: a BASE that is not an ancestor is refused rc 64 by name, nothing synced" $?
  # Per-track remote dirs (the header): the tree's directory name minus "bloomery-" names its pair, the main tree keeps the old
  # names. Each case runs a copy of the throwaway tree under another directory name; --dry-run prints the pair and touches no box.
  mkdir -p "$t/names"
  for name in bloomery bloomery-aa bloomery-bb plain 'bloomery-a+b'; do cp -R "$w" "$t/names/$name"; done
  # dirs_of <tree name> [VAR=value…]: the dry run's output in $got, its code in $rc
  dirs_of() {
    local name=$1
    shift
    got=$(env HOME="$t/home" STUB_LOG="$t/dirs.log" "$@" bash "$t/names/$name/tools/narrow-scan.sh" "$wb" --dry-run 2>&1)
    rc=$?
  }
  dirs_of bloomery-aa
  grep -qF "base dir ~/repo/bloomery-aa-narrow-base, head dir ~/repo/bloomery-aa-narrow-head," <<< "$got"; ok "dirs: a track's pair is ~/repo/bloomery-<track>-narrow-base and -narrow-head" $?
  aa=$got
  dirs_of bloomery-bb
  [ "$got" != "$aa" ] && grep -qF "~/repo/bloomery-bb-narrow-base" <<< "$got" && ! grep -qF "bloomery-aa" <<< "$got"; ok "dirs: two tracks get distinct dirs" $?
  dirs_of bloomery
  grep -qF "base dir ~/repo/bloomery-narrow-base, head dir ~/repo/bloomery-narrow-head," <<< "$got"; ok "dirs: the main tree keeps ~/repo/bloomery-narrow-base and -narrow-head (the unchanged names)" $?
  dirs_of plain
  grep -qF "base dir ~/repo/bloomery-plain-narrow-base," <<< "$got"; ok "dirs: a tree not named bloomery-<track> is a track of its own name" $?
  dirs_of bloomery-aa NARROW_BASE_REMOTE=/x/base
  grep -qF "base dir /x/base, head dir ~/repo/bloomery-aa-narrow-head," <<< "$got"; ok "dirs: NARROW_BASE_REMOTE overrides the base dir, the head dir stays the track's" $?
  dirs_of bloomery-aa NARROW_HEAD_REMOTE=/x/head
  grep -qF "base dir ~/repo/bloomery-aa-narrow-base, head dir /x/head," <<< "$got"; ok "dirs: NARROW_HEAD_REMOTE overrides the head dir, the base dir stays the track's" $?
  dirs_of 'bloomery-a+b'
  [ "$rc" = 64 ] && grep -qF "is not a box directory name" <<< "$got"; ok "dirs: a tree name outside [A-Za-z0-9._-] is refused by name (64)" $?
  dirs_of 'bloomery-a+b' NARROW_BASE_REMOTE=/x/b NARROW_HEAD_REMOTE=/x/h
  [ "$rc" = 0 ] && grep -qF "base dir /x/b, head dir /x/h," <<< "$got"; ok "dirs: both overrides given, the tree's name is not read" $?
  # The stop and the lock. trk_run <tree> <stub log> <args…>: a run on the tree's own dirs that does not wait for the lock.
  trk_run() {
    local tree=$1 log=$2
    shift 2
    : > "$log"
    got=$(HOME="$t/home" STUB_LOG="$log" NARROW_WAIT=0 bash "$tree/tools/narrow-scan.sh" "$@" 2>&1)
    rc=$?
  }
  # The fake `timeout` execs the command, so the runner is one process (the real one ends its whole process group on a TERM).
  mkdir -p "$t/fakebin" "$t/remote"
  printf '%s\n' '#!/bin/sh' 'shift 2' 'exec "$@"' > "$t/fakebin/timeout"
  chmod +x "$t/fakebin/timeout"
  : > "$t/stop-stub.log"
  HOME="$t/home" STUB_LOG="$t/stop-stub.log" STUB_BLOCK=1 STUB_REMOTE="$t/remote" STUB_BIN="$t/fakebin" \
    bash "$t/names/bloomery-aa/tools/narrow-scan.sh" "$wb" --worktree --out "$t/stop-out" > "$t/stop.log" 2>&1 &
  sp=$!
  pf='' k=0
  while [ -z "$pf" ] && [ "$k" -lt 100 ] && kill -0 "$sp" 2> /dev/null; do
    for f in "$t"/remote/*/.narrow-scan-*.pid; do [ ! -e "$f" ] || pf=$f; done
    [ -n "$pf" ] || sleep 0.2
    k=$((k + 1))
  done
  [ -n "$pf" ]; ok "stop: the box command records the pid of the runner it started" $?
  rp=$(cat "$pf" 2> /dev/null || echo 0)
  case $rp in '' | *[!0-9]*) rp=0 ;; esac
  [ "$rp" -gt 1 ] && kill -0 "$rp" 2> /dev/null; ok "stop: … and that pid is the live runner" $?
  trk_run "$t/names/bloomery-aa" "$t/same.log" "$wb" "$wb" --out "$t/same-out"
  [ "$rc" = 75 ] && grep -qF "is held by pid $sp" <<< "$got"; ok "lock: a second run on the same pair is refused rc 75, naming the holder" $?
  trk_run "$t/names/bloomery-bb" "$t/bb.log" "$wb" "$wb" --out "$t/bb-out"
  [ "$rc" = 0 ] && grep -qx "base ~/repo/bloomery-bb-narrow-base marker=base new=no" "$t/bb.log" && grep -qx "head ~/repo/bloomery-bb-narrow-head marker=base new=no" "$t/bb.log"
  ok "lock: another track's run is not held up by it and runs on its own pair" $?
  kill -TERM "$sp"
  k=0
  while kill -0 "$sp" 2> /dev/null && [ "$k" -lt 100 ]; do sleep 0.2; k=$((k + 1)); done
  gone=0
  if kill -0 "$sp" 2> /dev/null; then kill -KILL "$sp" 2> /dev/null; gone=1; fi
  src=0
  { wait "$sp"; src=$?; } 2> /dev/null
  [ "$gone" = 0 ] && [ "$src" = 143 ]; ok "stop: a TERM ends the run at once, rc 143" $?
  [ "$rp" -gt 1 ] && ! kill -0 "$rp" 2> /dev/null; ok "stop: the recorded remote pid is gone" $?
  grep -qx "readonly ~/repo/bloomery-aa-narrow-base" "$t/stop-stub.log" && grep -qF "signalled the recorded remote pid $rp" "$t/stop.log"
  ok "stop: it signalled exactly that pid, through a read-only call on the side's dir" $?
  # A directory glob misses the flock file, so the check takes that file. The probe's directory
  # carries this shell's pid (a subshell keeps $$); remove it after a take that succeeded.
  free=1
  for f in "$t/home/.cache/bloomery"/narrow-scan*.lock.d.flock; do
    if [ -e "$f" ]; then
      if ( lock_take "${f%.flock}" 0 probe > /dev/null 2>&1 ); then
        rm -rf "${f%.flock}"
      else
        free=0
      fi
    fi
  done
  [ "$free" = 1 ]; ok "stop: the lock is released" $?
  # The main tree's pair keeps the lock older copies of the script took for every tree: a live holder of it holds that pair, and no other.
  mkdir -p "$t/home/.cache/bloomery/narrow-scan.lock.d"
  echo $$ > "$t/home/.cache/bloomery/narrow-scan.lock.d/pid"
  echo "an older copy" > "$t/home/.cache/bloomery/narrow-scan.lock.d/who"
  trk_run "$t/names/bloomery" "$t/legacy.log" "$wb" "$wb" --out "$t/legacy-out"
  [ "$rc" = 75 ] && grep -qF "is held by pid $$ (an older copy)" <<< "$got"; ok "lock: the main tree's pair is held by a run of an older copy (the old lock name), rc 75 naming it" $?
  trk_run "$t/names/bloomery-bb" "$t/legacy-bb.log" "$wb" "$wb" --out "$t/legacy-bb-out"
  [ "$rc" = 0 ]; ok "lock: … and a track's own pair is not" $?
  rm -rf "$t/home/.cache/bloomery/narrow-scan.lock.d"
  # A stop before the box command has started a runner: nothing is recorded there, and the local box.sh is ended by its pid file.
  # hang_case <TERM|INT> <rc>: a run with the stub held before it starts anything on the box, stopped by that signal. The run starts
  # with INT at its default: a background job, or any step of a parallel runner (mac-static), has it ignored, and an ignored signal
  # cannot be trapped.
  hang_case() {
    local sig=$1 want=$2 hp hs hc k gone src
    : > "$t/hang-stub.log"
    rm -f "$t/remote/hang.pid" "$t/remote/hang-sleep.pid"
    HOME="$t/home" STUB_LOG="$t/hang-stub.log" STUB_HANG=1 STUB_REMOTE="$t/remote" \
      python3 -c 'import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execvp("bash", ["bash"] + sys.argv[1:])' \
      "$t/names/bloomery-bb/tools/narrow-scan.sh" "$wb" "$wb" --out "$t/hang-out-$sig" > "$t/hang-$sig.log" 2>&1 &
    hp=$!
    k=0
    while [ ! -s "$t/remote/hang.pid" ] && [ "$k" -lt 100 ] && kill -0 "$hp" 2> /dev/null; do sleep 0.2; k=$((k + 1)); done
    hs=$(cat "$t/remote/hang.pid" 2> /dev/null || echo 0)
    case $hs in '' | *[!0-9]*) hs=0 ;; esac
    [ "$hs" -gt 1 ] && [ "$(cat "$t/hang-out-$sig/base.box.pid" 2> /dev/null)" = "$hs" ]; ok "stop $sig: the pid file under DIR holds the local box.sh's pid" $?
    kill -"$sig" "$hp"
    k=0
    while kill -0 "$hp" 2> /dev/null && [ "$k" -lt 100 ]; do sleep 0.2; k=$((k + 1)); done
    gone=0
    if kill -0 "$hp" 2> /dev/null; then kill -KILL "$hp" 2> /dev/null; gone=1; fi
    src=0
    { wait "$hp"; src=$?; } 2> /dev/null
    [ "$gone" = 0 ] && [ "$src" = "$want" ] && grep -qF "no remote pid recorded in .narrow-scan-base-$hp.pid" "$t/hang-$sig.log"
    ok "stop $sig: before any runner started it says so and ends rc $want" $?
    [ "$hs" -gt 1 ] && ! kill -0 "$hs" 2> /dev/null; ok "stop $sig: … and the local box.sh is gone" $?
    hc=$(cat "$t/remote/hang-sleep.pid" 2> /dev/null || echo 0)
    case $hc in '' | *[!0-9]*) hc=0 ;; esac
    [ "$hc" -gt 1 ] && ! kill -0 "$hc" 2> /dev/null; ok "stop $sig: … and so is the process it had started (the ssh), by the job's process group" $?
    if [ "$hc" -gt 1 ]; then kill -TERM "$hc" 2> /dev/null || true; fi
  }
  hang_case TERM 143
  hang_case INT 130
  # usage refusals
  for args in "" "--bogus a" "a b c" "x --out"; do
    # shellcheck disable=SC2086
    got=$(bash "$0" $args 2>&1); rc=$?
    [ "$rc" = 64 ]; ok "usage: '$args' is refused rc 64" $?
  done
  got=$(bash "$0" no-such-rev-1234 2>&1); rc=$?
  [ "$rc" = 64 ] && grep -qF "'no-such-rev-1234' is not a commit" <<< "$got"; ok "usage: an unknown BASE is refused by name" $?
  rm -rf "$t"
  want=ok
  [ "$bad" = 0 ] || want=FAIL
  echo "narrow-scan: self-test $want ($n cases, $bad failed)"
  [ "$bad" = 0 ]
}

if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi

# The scan pairs are sm_86's (ARCH): the runner builds both sides at ARCH, so a narrowing asked at another arch would read
# scans of the wrong code.
[ "${BLOOMERY_ARCH-$ARCH}" = "$ARCH" ] || die 64 "BLOOMERY_ARCH is '$BLOOMERY_ARCH': the scan pairs are $ARCH's (this script builds both sides at $ARCH only)"

BASE='' HEAD_REV='' OUT='' DRY=0 WORKTREE=0 KEEP=0 SCAN_ONLY=0
while [ $# -gt 0 ]; do
  case $1 in
    --out)
      [ $# -ge 2 ] || die 64 "--out needs a directory; $USAGE"
      OUT=$2
      shift 2
      ;;
    --dry-run) DRY=1; shift ;;
    --worktree) WORKTREE=1; shift ;;
    --keep-ptx) KEEP=1; shift ;;
    --scan-only) SCAN_ONLY=1; shift ;;
    -*) die 64 "'$1' is not a flag of this tool; $USAGE" ;;
    *)
      if [ -z "$BASE" ]; then BASE=$1
      elif [ -z "$HEAD_REV" ]; then HEAD_REV=$1
      else die 64 "'$1' after BASE and HEAD; $USAGE"
      fi
      shift
      ;;
  esac
done
[ -n "$BASE" ] || die 64 "$USAGE"
[ "$WORKTREE" = 0 ] || [ -z "$HEAD_REV" ] || die 64 "--worktree takes no HEAD ('$HEAD_REV'): its head side is this tree's current content; $USAGE"
HEAD_REV=${HEAD_REV:-HEAD}
BASE_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$BASE^{commit}") || die 64 "'$BASE' is not a commit of this repository"
HEAD_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$HEAD_REV^{commit}") || die 64 "'$HEAD_REV' is not a commit of this repository"
HEAD_NAME=${HEAD_SHA:0:8}
if [ "$WORKTREE" = 1 ]; then
  [ "$(git -C "$ROOT" merge-base "$BASE_SHA" "$HEAD_SHA" 2> /dev/null)" = "$BASE_SHA" ] \
    || die 64 "--worktree: '$BASE' is not an ancestor of this tree's HEAD ${HEAD_SHA:0:8}; the affected run reads the working tree against the merge base, so the base side must be that commit"
  HEAD_NAME=worktree
  WT_KEY0=$(worktree_key "$BASE_SHA")
fi
[ -n "$OUT" ] || OUT=$HOME/.cache/bloomery/narrow/$(basename "$ROOT")/${BASE_SHA:0:8}-$HEAD_NAME
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
mkdir -p "$OUT" || die 70 "cannot make $OUT"
# The pair of remote dirs (the header): this tree's track is its directory name without "bloomery-"; the main tree has none.
TRACK=''
if [ -z "${NARROW_BASE_REMOTE:-}" ] || [ -z "${NARROW_HEAD_REMOTE:-}" ]; then
  case $(basename "$ROOT") in
    bloomery) ;;
    bloomery-?*) TRACK=$(basename "$ROOT") TRACK=${TRACK#bloomery-} ;;
    *) TRACK=$(basename "$ROOT") ;;
  esac
  case $TRACK in
    *[!A-Za-z0-9._-]*) die 64 "this tree's name '$TRACK' is not a box directory name ([A-Za-z0-9._-]): name the remote dirs with NARROW_BASE_REMOTE and NARROW_HEAD_REMOTE" ;;
  esac
fi
PAIR='~/repo/bloomery'
[ -z "$TRACK" ] || PAIR=$PAIR-$TRACK
BASE_DIR=${NARROW_BASE_REMOTE:-$PAIR-narrow-base}
HEAD_DIR=${NARROW_HEAD_REMOTE:-$PAIR-narrow-head}
BOUND=${NARROW_BOUND:-7200}
case $BOUND in '' | *[!0-9]*) die 64 "NARROW_BOUND is whole seconds, got '$BOUND'" ;; esac
if [ "$WORKTREE" = 1 ]; then
  HEAD_LABEL="the working tree (HEAD ${HEAD_SHA:0:8}, key $WT_KEY0)"
  SCAN_SET_REV=()
  AFFECTED_RANGE=$BASE_SHA
  HEAD_SRC=worktree
else
  # The scans read the commits, so a work tree whose state is unknown changes no number: the note says it is unknown, and the run goes on.
  lrc=0
  tree_commit "$ROOT" || lrc=$?
  case $lrc in
    0) [ "$TREE_DIRTY" = 0 ] || echo "narrow-scan: note: this tree has uncommitted changes; the scans read the commits ${BASE_SHA:0:8} and ${HEAD_SHA:0:8} only (--worktree reads the tree)" >&2 ;;
    1) echo "narrow-scan: note: HEAD of $ROOT cannot be read, so whether this tree has uncommitted changes is unknown; the scans read the commits ${BASE_SHA:0:8} and ${HEAD_SHA:0:8} only" >&2 ;;
    *) echo "narrow-scan: note: git status failed in $ROOT (git exit $TREE_GIT_RC): whether this tree has uncommitted changes is unknown; the scans read the commits ${BASE_SHA:0:8} and ${HEAD_SHA:0:8} only" >&2 ;;
  esac
  HEAD_LABEL=${HEAD_SHA:0:8}
  SCAN_SET_REV=("$HEAD_SHA")
  AFFECTED_RANGE=$BASE_SHA..$HEAD_SHA
  HEAD_SRC=$HEAD_SHA
fi

echo "narrow-scan: ${BASE_SHA:0:8}..$HEAD_LABEL (no gate, no lease, no hold, no card, no gate lock: --no-jit scans); out $OUT"
python3 "$ROOT/tools/recipes.py" scan-set ${SCAN_SET_REV[@]+"${SCAN_SET_REV[@]}"} > "$OUT/scan-set.tsv" 2> "$OUT/scan-set.err" || {
  cat "$OUT/scan-set.err" >&2
  die 2 "the scan set could not be derived from $HEAD_LABEL"
}
cat "$OUT/scan-set.err" >&2
# The base side's set comes from BASE's own tree: a feature this change adds is not one BASE can build.
python3 "$ROOT/tools/recipes.py" scan-set "$BASE_SHA" > "$OUT/scan-set-base.tsv" 2> "$OUT/scan-set-base.err" || {
  cat "$OUT/scan-set-base.err" >&2
  die 2 "the scan set could not be derived from BASE ${BASE_SHA:0:8}"
}
[ -s "$OUT/scan-set-base.tsv" ] || die 2 "BASE ${BASE_SHA:0:8} has an empty scan set: nothing to compare the head side against"
BINS=()
while IFS=$'\t' read -r b _; do [ -z "$b" ] || BINS[${#BINS[@]}]=$b; done < "$OUT/scan-set.tsv"
make_runner "$OUT/scan-set.tsv" "$ROOT/tools/ptx-scan.sh" "$OUT/runner.sh" "$KEEP" || die 2 "the runner could not be written"
BASE_SET=$OUT/scan-set.tsv BASE_RUNNER=$OUT/runner.sh SET_DIFF=''
if ! cmp -s "$OUT/scan-set-base.tsv" "$OUT/scan-set.tsv"; then
  SET_DIFF=$(scan_set_diff "$OUT/scan-set-base.tsv" "$OUT/scan-set.tsv") || {
    echo "$SET_DIFF" >&2
    echo "narrow-scan: the base side's scan set (BASE ${BASE_SHA:0:8}):" >&2
    sed 's/^/  /' "$OUT/scan-set-base.tsv" >&2
    echo "narrow-scan: the head side's scan set ($HEAD_LABEL):" >&2
    sed 's/^/  /' "$OUT/scan-set.tsv" >&2
    die 2 "the sides' scan sets name different bins: a pair compares one bin's scan on both sides, so every bin must be in both"
  }
  BASE_SET=$OUT/scan-set-base.tsv BASE_RUNNER=$OUT/runner-base.sh
  make_runner "$BASE_SET" "$ROOT/tools/ptx-scan.sh" "$BASE_RUNNER" "$KEEP" || die 2 "the base runner could not be written"
  echo "$SET_DIFF"
fi
echo "narrow-scan: scan set (bin, features, carriers it covers):"
sed 's/^/  /' "$OUT/scan-set.tsv"
if [ "$DRY" = 1 ]; then
  if [ "$BASE_RUNNER" = "$OUT/runner.sh" ]; then
    echo "narrow-scan: --dry-run: base dir $BASE_DIR, head dir $HEAD_DIR, bound ${BOUND}s; the runner each side runs is $OUT/runner.sh"
  else
    echo "narrow-scan: --dry-run: base dir $BASE_DIR, head dir $HEAD_DIR, bound ${BOUND}s; the base side runs $BASE_RUNNER, the head side $OUT/runner.sh"
  fi
  exit 0
fi

LOCKD=$HOME/.cache/bloomery/narrow-scan-$(str_key "$BASE_DIR $HEAD_DIR").lock.d
# The main tree's pair is the one every copy of this script shared, under one lock of this name before the pairs were per track.
# This run creates that directory while it holds the flock, so an older copy's mkdir fails, and a directory there whose pid is
# live or not yet written is that older copy's lock and is refused, not deleted.
[ "$BASE_DIR $HEAD_DIR" != '~/repo/bloomery-narrow-base ~/repo/bloomery-narrow-head' ] || LOCKD=$HOME/.cache/bloomery/narrow-scan.lock.d
lock_take "$LOCKD" "${NARROW_WAIT:-1800}" "$(basename "$ROOT") ${BASE_SHA:0:8}..$HEAD_NAME since $(date +%H:%M:%S)" || exit $?
trap lock_release EXIT
trap 'stop_run 130' INT
trap 'stop_run 143' TERM

# The base side's cache: BASE is always a commit, so its side is reusable; the head side never is. The key's parts are in
# $OUT/base-cache.key, the entry's dir is named by their md5.
CACHE_ROOT=$HOME/.cache/bloomery/narrow-base
cache_parts "$ROOT/tools/ptx-scan.sh" "$BASE_SET" "$BASE_RUNNER" > "$OUT/base-cache.key"
CACHE_ENTRY=$CACHE_ROOT/${BASE_SHA:0:8}-$(md5_of "$OUT/base-cache.key" 9>&- | cut -c1-16)
BASE_CACHE=miss RETRIED=''
if [ "${NARROW_NO_CACHE:-0}" != 1 ] && [ -d "$CACHE_ENTRY" ]; then
  if why=$(cache_check "$CACHE_ENTRY" "$OUT/base-cache.key" "${BINS[@]}" 9>&-); then
    BASE_CACHE=hit
  else
    echo "narrow-scan: cache entry $CACHE_ENTRY is corrupt ($why); removed, the base side runs" >&2
    rm -rf "$CACHE_ENTRY" 9>&-
  fi
fi

T0=$SECONDS
WALL_BASE=- WALL_HEAD=-
for side in base head; do
  if [ "$side" = base ] && [ "$BASE_CACHE" = hit ]; then
    echo "narrow-scan: base $BASE_SHA from cache $CACHE_ENTRY"
    rm -rf "$OUT/base" 9>&-
    mkdir -p "$OUT/base" 9>&-
    cp -R "$CACHE_ENTRY/base/." "$OUT/base/" 9>&- || die 1 "base: the cache entry $CACHE_ENTRY could not be copied into $OUT/base"
    WALL_BASE=cache
    continue
  fi
  if [ "$side" = base ]; then rev=$BASE_SHA dir=$BASE_DIR runner=$BASE_RUNNER; else rev=$HEAD_SRC dir=$HEAD_DIR runner=$OUT/runner.sh; fi
  left=$((BOUND - (SECONDS - T0)))
  [ "$left" -ge 60 ] || die 124 "the bound of ${BOUND}s is spent before the $side side (the base side took ${WALL_BASE}s): run again, the dirs keep their compiled crates"
  run_side "$side" "$rev" "$dir" "$runner" "$OUT" "$left" "${BINS[@]}"
  rc=$?
  if [ "$rc" = 255 ]; then
    # ssh lost the box: the remote runner it started may still run, so it is ended by its recorded pid first; then the side runs
    # once more, in the same remote dir (its compiled crates kept). Once only: a second 255 ends the run.
    echo "narrow-scan: $side: ssh lost the box (rc 255); ending its recorded remote runner and running the side once more" >&2
    remote_stop "$dir" ".narrow-scan-$side-$$.pid"
    RETRIED=${RETRIED:+$RETRIED,}$side
    left=$((BOUND - (SECONDS - T0)))
    [ "$left" -ge 60 ] || die 124 "the bound of ${BOUND}s is spent before the $side side's retry: run again, the dirs keep their compiled crates"
    run_side "$side" "$rev" "$dir" "$runner" "$OUT" "$left" "${BINS[@]}"
    rc=$?
    [ "$rc" != 255 ] || echo "narrow-scan: $side: ssh lost the box again (rc 255): retried once, not again" >&2
  fi
  [ "$rc" = 0 ] || exit "$rc"
  if [ "$side" = base ]; then WALL_BASE=$SIDE_WALL; else WALL_HEAD=$SIDE_WALL; fi
  split_logs "$OUT/$side.scan.out" "$OUT/$side" "${BINS[@]}" 9>&- > "$OUT/$side.split.err" 2>&1 || {
    cat "$OUT/$side.split.err" >&2 9>&-
    die 1 "$side: the box output does not hold every scan ($OUT/$side.scan.out)"
  }
  [ "$KEEP" = 0 ] || split_ptx "$OUT/$side.scan.out" "$OUT/$side" "${BINS[@]}" 9>&- > "$OUT/$side.split-ptx.err" 2>&1 || {
    cat "$OUT/$side.split-ptx.err" >&2 9>&-
    die 1 "$side: the box output does not hold every bin's ptx modules ($OUT/$side.scan.out)"
  }
  # the side finished with every bin's scan rc 0 (and its modules, under --keep-ptx): only now is it kept
  [ "$side" != base ] || cache_store "$CACHE_ENTRY" "$OUT/base-cache.key" "${BINS[@]}"
done

echo "narrow-scan: bundles each scan carried (base | head):"
for b in "${BINS[@]}"; do
  echo "  $b: $({ sed -n 's/^ptx-scan: mod[0-9]* bundle=\([^ ]*\) .*/\1/p' "$OUT/base/$b.log" | tr '\n' ' '; } 9>&-)| $({ sed -n 's/^ptx-scan: mod[0-9]* bundle=\([^ ]*\) .*/\1/p' "$OUT/head/$b.log" | tr '\n' ' '; } 9>&-)"
done
ARGS=()
for b in "${BINS[@]}"; do ARGS[${#ARGS[@]}]=--scan; ARGS[${#ARGS[@]}]=$OUT/base/$b.log; ARGS[${#ARGS[@]}]=$OUT/head/$b.log; done
if [ "$WORKTREE" = 1 ]; then
  WT_KEY1=$(worktree_key "$BASE_SHA" 9>&-)
  [ "$WT_KEY1" = "$WT_KEY0" ] || die 1 "the working tree changed during the run (key $WT_KEY0 -> $WT_KEY1): the head scans and the affected list would read different content; run again with the tree still"
fi
if [ "$SCAN_ONLY" = 1 ]; then
  arc=skipped
else
  "$ROOT/tools/affected-gates.sh" "$AFFECTED_RANGE" --narrow "${ARGS[@]}" 9>&- > "$OUT/narrowed.txt" 2> "$OUT/narrowed.err"
  arc=$?
  cat "$OUT/narrowed.txt" 9>&-
  cat "$OUT/narrowed.err" >&2 9>&-
fi
{
  echo "base=$BASE_SHA"
  if [ "$WORKTREE" = 1 ]; then
    echo "head=worktree@$HEAD_SHA"
    echo "worktree_key=$WT_KEY1"
  else
    echo "head=$HEAD_SHA"
  fi
  echo "base_dir=$BASE_DIR"
  echo "head_dir=$HEAD_DIR"
  echo "wall_base_s=$WALL_BASE"
  echo "wall_head_s=$WALL_HEAD"
  echo "ptx_scan_md5=$(md5_of "$ROOT/tools/ptx-scan.sh" 9>&-)"
  echo "affected_rc=$arc"
  echo "base_cache=$BASE_CACHE"
  [ -z "$RETRIED" ] || echo "retried=$RETRIED"
  [ -z "$SET_DIFF" ] || echo "scan_set_differs=1"
  [ "$KEEP" = 0 ] || echo "keep_ptx=1"
} > "$OUT/MANIFEST"
BASE_WALL_TXT=${WALL_BASE}s
[ "$BASE_CACHE" != hit ] || BASE_WALL_TXT="from cache"
RETRY_TXT=''
[ -z "$RETRIED" ] || RETRY_TXT=" (retried: $RETRIED)"
if [ "$SCAN_ONLY" = 1 ]; then
  echo "narrow-scan: done: base side $BASE_WALL_TXT, head side ${WALL_HEAD}s$RETRY_TXT, affected run skipped (--scan-only); scan logs are under $OUT"
  exit 0
fi
echo "narrow-scan: done: base side $BASE_WALL_TXT, head side ${WALL_HEAD}s$RETRY_TXT, affected rc $arc; scan logs and the list are under $OUT"
exit "$arc"
