#!/usr/bin/env bash
# The release picker: one table at the ask, for the two commits a release can name, and the pick between them.
#
#   W     the follower's watermark: the newest commit whose real-tier list is green on the lead's ledger. It is the file
#         ${BLOOMERY_FOLLOW_DIR:-~/.cache/bloomery/follow}/W, which the follower's runner writes. Two forms are read: `key=value`
#         lines (`commit=<sha>`, then optionally `time=<when>` and `list=<items>` or `items=<n>`), or one line `<sha> [<when> [<items>]]`.
#         No file, no commit in it, a commit this tree does not have, or one that is not an ancestor of HEAD: W is unusable, named.
#   HEAD  the tree's HEAD (--head REV).
#
# The table, each row from the tool that owns the fact:
#   commits W..HEAD      git
#   late-commit          tools/release/late-commit.sh W HEAD: whether the commits after W move a shipped binary
#   delta batch          `just affected W..HEAD` into `tools/gate-batch.sh --tier real --ledger --dry-run`: the gates the
#                        commits after W select that the lead's ledger has no green row for, and the batch's predicted wall
#                        (the medians of the times file). A delta that selects no gate is 0 s.
#   sitting              tools/release/sitting.sh --dry-run on --cells FILE: the sitting's predicted wall (without --cells, not priced)
#   README and versions  tools/release/publish.sh --tree-check at W and at HEAD: the serve and gpu-gates crates say the release's
#                        version and README.md names its status, tarball and link
#   tarballs             the box's target/release-dist archives (BLOOMERY_BOX and BLOOMERY_REMOTE, as publish.sh reads them): each arch's
#                        sha256 and the commit its VERSION file names, held to the picked commit: the same commit, or an ancestor
#                        with only docs paths between (tools/release/publish.sh --docs-delta: README.md, README.ko.md, CHANGELOG*,
#                        docs/**), so archives the follower built at W are ready for a HEAD that adds the docs commit
#   rotation             tools/rotation.py check
#   needs-gpu PRs        gh: merged since the last release tag with needs-gpu and neither gpu-verified nor gpu-evidence; skipped by
#                        name where gh is absent or cannot answer
#
# The pick. W, unless the HEAD delta proves in at most 25 minutes (the delta batch's predicted wall): then HEAD, whose newer commits
# the release carries. W also when the delta is unproven (the batch could not be planned): an unproven delta proves nothing. With no
# usable W the pick is HEAD and the whole real tier applies to it. W equal to HEAD is W. The late-commit verdict is shown with the
# pick and does not decide it. After the pick, `ready:` lists what still stands between it and the sitting (the README and version
# check at the picked commit, archives missing or built at another commit, a red rotation, needs-gpu PRs).
#
# Usage: tools/release/pick.sh [--version V] [--head REV] [--w REV | --w-file FILE] [--cells FILE]
#        tools/release/pick.sh --self-test   (the pick rule and every row on a temporary repository and stub tools; check-recipes runs it)
# --version defaults to the serve crate's version at HEAD. Exit 0 with a pick, 64 for an argument it does not take, 70 when HEAD
# cannot be read.
# Each tool is a hook, a command line split on blanks (no shell syntax) with the arguments appended: PICK_LATE, PICK_AFFECTED,
# PICK_BATCH, PICK_SITTING, PICK_ROTATION, PICK_GH and PICK_TARBALLS (which prints `<arch> <sha256> <commit>` or `<arch> absent` a
# line for sm86 and sm120a, and takes the version). Unset, each is the tool named above.
set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 70
USAGE="usage: tools/release/pick.sh [--version V] [--head REV] [--w REV | --w-file FILE] [--cells FILE] | --self-test"
LIMIT_S=1500 # the HEAD delta proves inside this: 25 minutes

die() { # die <rc> <message>
  echo "pick.sh: $2" >&2
  exit "$1"
}

# run_hook <ENV NAME> <default command line> [args…]: the hook's command (the variable's words, else the default's) and the args.
run_hook() {
  local name=$1 def=$2 line words
  shift 2
  line=${!name-}
  [ -n "$line" ] || line=$def
  read -r -a words <<< "$line"
  "${words[@]}" "$@"
}

# w_read <file>: W_COMMIT, W_TIME and W_SIZE from the follower's file; rc 1 when it holds no commit.
w_read() {
  local line k first=''
  W_COMMIT='' W_TIME='' W_SIZE=''
  while IFS= read -r line || [ -n "$line" ]; do
    case $line in '' | '#'*) continue ;; esac
    case $line in
      commit=* | time=* | list=* | items=*)
        k=${line%%=*}
        case $k in commit) W_COMMIT=${line#*=} ;; time) W_TIME=${line#*=} ;; *) W_SIZE=${line#*=} ;; esac ;;
      *=*) ;;
      *) [ -n "$first" ] || first=$line ;;
    esac
  done < "$1"
  if [ -z "$W_COMMIT" ] && [ -n "$first" ]; then
    read -r W_COMMIT W_TIME W_SIZE <<< "$first"
  fi
  [[ $W_COMMIT =~ ^[0-9a-fA-F]{7,40}$ ]] || { W_COMMIT=''; return 1; }
}

# pick_rule: PICK and REASON from W_OK (1 when W is usable), W_WHY, SAME (1 when W is HEAD), DELTA_S (the delta's predicted wall in
# seconds, empty when unproven), DELTA_WHY and LIMIT_S.
pick_rule() {
  if [ "$W_OK" != 1 ]; then
    PICK=HEAD REASON="no usable watermark ($W_WHY): HEAD, and the whole real tier applies to it"
  elif [ "$SAME" = 1 ]; then
    PICK=W REASON="W is HEAD: the follower's pass covers it, there is no delta"
  elif [ -z "$DELTA_S" ]; then
    PICK=W REASON="the HEAD delta is unproven ($DELTA_WHY): an unproven delta proves nothing, the newer commits ship next time"
  elif [ "$DELTA_S" -le "$LIMIT_S" ]; then
    PICK=HEAD REASON="the HEAD delta proves in $((DELTA_S / 60)) min, within $((LIMIT_S / 60)): HEAD carries the newer commits"
  else
    PICK=W REASON="the HEAD delta needs $((DELTA_S / 60)) min, past $((LIMIT_S / 60)): the newer commits ship next time"
  fi
}

# tree_row <rev>: `ok`, or the refusal of publish.sh --tree-check at <rev>.
tree_row() {
  local out rc=0
  out=$(bash tools/release/publish.sh --tree-check "$VERSION" "$1" 2>&1) || rc=$?
  if [ "$rc" = 0 ]; then echo ok; else out=${out%%$'\n'*}; echo "${out#publish.sh: }"; fi
}

# tarball_rows <version>: the hook's lines, or the box's own archives over ssh.
tarball_rows() {
  local remote
  if [ -z "${PICK_TARBALLS-}" ]; then
    remote=${BLOOMERY_REMOTE:-"~/repo/$(basename "$ROOT")"}
    # shellcheck disable=SC2029 # the version and the directory are expanded here on purpose
    ssh -o BatchMode=yes -o ConnectTimeout=10 "${BLOOMERY_BOX:-ws}" "cd $remote/target/release-dist 2> /dev/null || { echo 'sm86 absent'; echo 'sm120a absent'; exit 0; }
for a in sm86 sm120a; do n=bloomery-$1-linux-x86_64-cuda-\$a; if [ -f \$n.tar.gz ] && [ -f \$n.tar.gz.sha256 ]; then
echo \"\$a \$(cut -d' ' -f1 \$n.tar.gz.sha256) \$(tar -xzOf \$n.tar.gz \$n/VERSION 2> /dev/null | sed -n 2p)\"; else echo \"\$a absent\"; fi; done"
  else
    run_hook PICK_TARBALLS "" "$1"
  fi
}

self_test() {
  local t r fails=0 cases=0 n_cases=0 self w0 w1 w2 head out rc
  self=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  t=$(mktemp -d "${TMPDIR:-/tmp}/pick.XXXXXX") || return 70
  r=$t/repo
  mkdir -p "$r/tools/release" "$r/crates/serve" "$r/crates/gpu-gates" "$t/stub"
  git -C "$r" init -q
  cp "$(dirname "$self")/publish.sh" "$r/tools/release/publish.sh"
  cp "$self" "$r/tools/release/pick.sh"
  commit() { # commit <message>: README at 0.0.1 (stale: names 0.0.0 until set), a new file, one commit
    printf 'x\n' >> "$r/log"
    git -C "$r" add -A
    git -C "$r" -c user.name=selftest -c user.email=selftest@invalid commit -q -m "$1"
  }
  printf '[package]\nname = "bloomery-serve"\nversion = "0.0.1"\n' > "$r/crates/serve/Cargo.toml"
  printf '[package]\nname = "bloomery-gpu-gates"\nversion = "0.0.1"\n' > "$r/crates/gpu-gates/Cargo.toml"
  printf '## Status (0.0.0)\n' > "$r/README.md"
  commit w
  w0=$(git -C "$r" rev-parse HEAD)
  printf '## Status (0.0.1)\nbloomery-0.0.1-linux-x86_64-cuda-sm86.tar.gz\nreleases/tag/v0.0.1)\n' > "$r/README.md"
  commit readme
  git -C "$r" tag v0.0.0 "$(git -C "$r" commit-tree -p "$w0" -m side "$w0^{tree}")" # a release commit on a line of its own, no ancestor of HEAD
  commit two
  w1=$(git -C "$r" rev-parse HEAD)
  commit three
  head=$(git -C "$r" rev-parse HEAD)
  w2=${w0:0:12}
  # two commits on head, on no branch: one adds a docs path, one a crates path; --head names them
  mk_child() { # mk_child <path>
    local idx="$t/idx" blob tree
    rm -f "$idx"
    blob=$(printf 'child\n' | git -C "$r" hash-object -w --stdin)
    GIT_INDEX_FILE=$idx git -C "$r" read-tree "$head"
    GIT_INDEX_FILE=$idx git -C "$r" update-index --add --cacheinfo "100644,$blob,$1"
    tree=$(GIT_INDEX_FILE=$idx git -C "$r" write-tree)
    git -C "$r" -c user.name=selftest -c user.email=selftest@invalid commit-tree -p "$head" -m child "$tree"
  }
  hd_docs=$(mk_child docs/notes.md)
  hd_code=$(mk_child crates/serve/lib.rs)
  # the stubs, each driven by an environment variable the case sets
  cat > "$t/stub/late" << 'EOF'
#!/bin/sh
[ -n "$STUB_LATE" ] && echo "$STUB_LATE"
exit "${STUB_LATE_RC:-0}"
EOF
  cat > "$t/stub/affected" << 'EOF'
#!/bin/sh
for g in $STUB_GATES; do echo "  $g"; done
exit "${STUB_AFF_RC:-0}"
EOF
  cat > "$t/stub/batch" << 'EOF'
#!/bin/sh
echo called >> "$STUB_BATCH_LOG"
[ -n "$STUB_WALL" ] && echo "gate-batch: predicted laneA=0s laneB=0s laneX=0s chain=0s wall=${STUB_WALL}s (derived: stub)"
exit "${STUB_BATCH_RC:-0}"
EOF
  cat > "$t/stub/sitting" << 'EOF'
#!/bin/sh
echo "[plan] predicted wall = 3480 s (~58 min) [derived]"
EOF
  cat > "$t/stub/rotation" << 'EOF'
#!/bin/sh
[ -n "$STUB_ROT" ] && echo "$STUB_ROT"
exit "${STUB_ROT_RC:-0}"
EOF
  cat > "$t/stub/gh" << 'EOF'
#!/bin/sh
[ -n "$STUB_PRS" ] && echo "$STUB_PRS"
exit "${STUB_GH_RC:-0}"
EOF
  cat > "$t/stub/tarballs" << 'EOF'
#!/bin/sh
for a in $STUB_TARS; do echo "$a" | tr '_' ' '; done
EOF
  chmod +x "$t"/stub/*
  : > "$t/follow-none"
  # run_pick <extra env words…> -- <pick.sh args…>: the output and rc, in $out and $rc
  run_pick() {
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    out=$(cd "$r" && env ${envs[@]+"${envs[@]}"} PICK_LATE="$t/stub/late" PICK_AFFECTED="$t/stub/affected" PICK_BATCH="$t/stub/batch" \
      PICK_SITTING="$t/stub/sitting" PICK_ROTATION="$t/stub/rotation" PICK_GH="$t/stub/gh" PICK_TARBALLS="$t/stub/tarballs" \
      STUB_BATCH_LOG="$t/batch.log" bash "$1" "${@:2}" 2>&1) && rc=0 || rc=$?
  }
  # want <case> <want rc> <text it must hold>…: the last run_pick; a text starting with ! must be absent
  want() {
    local name=$1 want_rc=$2 s
    shift 2
    cases=$((cases + 1))
    [ "$rc" = "$want_rc" ] || { [ "$LOUD" = 0 ] || echo "pick self-test: $name: rc $rc (want $want_rc); got: $out" >&2; BAD=$((BAD + 1)); return 0; }
    for s in "$@"; do
      case $s in
        '!'*) [[ "$out" != *"${s#!}"* ]] || { [ "$LOUD" = 0 ] || echo "pick self-test: $name: the output holds '${s#!}'; got: $out" >&2; BAD=$((BAD + 1)); return 0; } ;;
        *) [[ "$out" == *"$s"* ]] || { [ "$LOUD" = 0 ] || echo "pick self-test: $name: the output lacks '$s'; got: $out" >&2; BAD=$((BAD + 1)); return 0; } ;;
      esac
    done
  }
  # The cases come in groups (p, g and the watermark files are the caller's, all_cases): a mutant runs the group that holds its guard.
  cases_rule() {
    run_pick STUB_GATES="$g" STUB_WALL=1800 STUB_LATE=BLOCKS STUB_LATE_RC=1 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "delta 30 min: W" 0 "pick: W $w2" "1800 s (30 min)" "BLOCKS" "ship next time"
    run_pick STUB_GATES="$g" STUB_WALL=1501 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "delta 1501 s: W" 0 "pick: W $w2"
    run_pick STUB_GATES="$g" STUB_WALL=1500 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "delta exactly 25 min: HEAD" 0 "pick: HEAD ${head:0:12}" "1500 s (25 min)"
    run_pick STUB_GATES="$g" STUB_WALL=1200 -- "$p" --w-file "$t/follow-line" --version 0.0.1
    want "delta 20 min, the one-line form: HEAD" 0 "pick: HEAD ${head:0:12}" "commits W..HEAD      3"
    run_pick STUB_GATES="" -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "a delta that selects no gate: HEAD" 0 "pick: HEAD" "select no gate"
    run_pick STUB_GATES="$g" STUB_WALL= STUB_BATCH_RC=1 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "an unplannable delta: W" 0 "pick: W $w2" "unproven: gate-batch --dry-run ended rc 1" "an unproven delta proves nothing"
    run_pick STUB_GATES="$g" STUB_WALL= -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "a batch with no predicted line: W" 0 "pick: W $w2" "no predicted line" "an unproven delta proves nothing"
    : > "$t/batch.log"
    run_pick STUB_GATES="$g" STUB_WALL=10 -- "$p" --w "$head" --version 0.0.1
    want "W is HEAD: W, the batch never planned" 0 "pick: W ${head:0:12}" "there is no delta"
    [ ! -s "$t/batch.log" ] || { [ "$LOUD" = 0 ] || echo "pick self-test: W is HEAD: the batch was planned" >&2; BAD=$((BAD + 1)); }
    cases=$((cases + 1))
    # a W that cannot be used
    run_pick STUB_GATES="$g" STUB_WALL=10 -- "$p" --w-file "$t/follow-absent" --version 0.0.1
    want "no watermark file: HEAD" 0 "pick: HEAD" "no usable watermark (no watermark file"
    printf 'commit=0123456789abcdef0123456789abcdef01234567\n' > "$t/follow-unknown"
    run_pick STUB_GATES="$g" STUB_WALL=10 -- "$p" --w-file "$t/follow-unknown" --version 0.0.1
    want "a W this tree lacks: HEAD" 0 "pick: HEAD" "is not a commit of this tree"
    printf 'time=now\n' > "$t/follow-nocommit"
    run_pick STUB_GATES="$g" STUB_WALL=10 -- "$p" --w-file "$t/follow-nocommit" --version 0.0.1
    want "a file with no commit: HEAD" 0 "pick: HEAD" "holds no commit"
    run_pick STUB_GATES="$g" STUB_WALL=10 -- "$p" --w "$head" --head "$w1" --version 0.0.1
    want "W after HEAD, not its ancestor: HEAD" 0 "pick: HEAD" "is not an ancestor of HEAD"
  }
  cases_rows() {
    run_pick STUB_GATES="$g" STUB_WALL=1800 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "README row: W refuses, HEAD passes; ready: no" 0 "README and versions  W: README.md at $w0 does not name '## Status (0.0.1)'" "HEAD: ok" "ready: no" "README at the pick"
    run_pick STUB_GATES="$g" STUB_WALL=1800 -- "$p" --w-file "$t/follow-kv" --version 0.0.2
    want "README row: a version the crates do not say" 0 "crates/serve says version 0.0.1" "ready: no"
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_TARS="sm86_abc_${head:0:12} sm120a_absent" -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "tarballs: built at the pick, one absent" 0 "sm86 abc built at ${head:0:12}" "sm120a absent" "archive absent: sm120a"
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_TARS="sm86_abc_deadbeef0000 sm120a_def_${head:0:12}" -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "tarballs: built at another commit" 0 "built at deadbeef0000, not ${head:0:12}" "ready: no" "archive built at another commit: sm86"
  }
  cases_docs() {
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_TARS="sm86_abc_${head:0:12} sm120a_def_${head:0:12}" -- "$p" --w-file "$t/follow-kv" --head "$hd_docs" --version 0.0.1
    want "tarballs: built at W, the pick adds a docs commit" 0 "pick: HEAD ${hd_docs:0:12}" "sm86 abc built at ${head:0:12}, docs-only delta to ${hd_docs:0:12}" "sm120a def built at ${head:0:12}, docs-only delta to ${hd_docs:0:12}" "ready: yes" "!archive built at another commit"
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_TARS="sm86_abc_${head:0:12} sm120a_def_${head:0:12}" -- "$p" --w-file "$t/follow-kv" --head "$hd_code" --version 0.0.1
    want "tarballs: built before a crates change" 0 "pick: HEAD ${hd_code:0:12}" "sm86 abc built at ${head:0:12}, not ${hd_code:0:12}" "ready: no" "archive built at another commit: sm86" "!docs-only delta"
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_TARS="sm86_abc_${head:0:12} sm120a_def_${head:0:12}" -- "$p" --w-file "$t/follow-kv" --head "$head" --version 0.0.1
    want "tarballs: built at the pick itself" 0 "sm86 abc built at ${head:0:12}" "ready: yes" "!docs-only delta" "!archive built at another commit"
  }
  cases_tags() {
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_PRS=$'#41 a kernel\n#44 another' -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "needs-gpu PRs listed" 0 "needs-gpu PRs        2 merged since v0.0.0" "#41 a kernel" "needs-gpu PRs: 2"
  }
  cases_rest() {
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_TARS="sm86_abc_${head:0:12} sm120a_def_${head:0:12}" STUB_LATE="late: ok" STUB_ROT=green -- "$p" --w-file "$t/follow-kv" --version 0.0.1 --cells /x/cells.tsv
    want "a clear table: ready, sitting priced, late-commit and rotation shown" 0 "late-commit          rc 0: late: ok" "predicted wall = 3480 s (~58 min)" "rotation check       green" "ready: yes" "predicted to the sitting's verdict"
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_ROT="rotation: musespark stale" STUB_ROT_RC=1 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "a red rotation" 0 "rotation check       red: rotation: musespark stale" "rotation is red"
    run_pick STUB_GATES="$g" STUB_WALL=1200 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "no --cells: the sitting is not priced" 0 "sitting              not priced (no --cells FILE)" "!predicted to the sitting's verdict"
    run_pick STUB_GATES="$g" STUB_WALL=1200 STUB_GH_RC=4 -- "$p" --w-file "$t/follow-kv" --version 0.0.1
    want "gh that cannot answer: skipped by name" 0 "needs-gpu PRs        skipped (gh: rc 4" "!needs-gpu PRs: "
    cases=$((cases + 1))
    out=$(cd "$r" && PICK_GH=/no/such/gh PICK_LATE="$t/stub/late" PICK_AFFECTED="$t/stub/affected" PICK_BATCH="$t/stub/batch" STUB_GATES="$g" STUB_WALL=1200 \
      PICK_TARBALLS="$t/stub/tarballs" PICK_ROTATION="$t/stub/rotation" bash "$p" --w-file "$t/follow-kv" --version 0.0.1 2>&1) && rc=0 || rc=$?
    want "gh absent: skipped by name" 0 "needs-gpu PRs        skipped (gh: " "pick: HEAD"
    # arguments
    run_pick -- "$p" --no-such-option
    want "an unknown option" 64 "unknown argument --no-such-option"
    run_pick -- "$p" --head no-such-rev
    want "a HEAD that is no commit" 64 "--head no-such-rev is no commit"
  }
  # all_cases <pick.sh copy in the temp repo> <loud> [rule|rows|docs|tags]: the cases (all groups without the third argument); BAD counts the red ones
  all_cases() {
    local p=$1 g="gate-a gate-b" group=${3:-all}
    LOUD=$2 BAD=0
    printf 'commit=%s\ntime=2026-11-03T06:30\nlist=212\n' "$w0" > "$t/follow-kv"
    printf '%s 2026-11-03T06:30 212\n' "$w2" > "$t/follow-line"
    case $group in all | rule) cases_rule ;; esac
    case $group in all | rows) cases_rows ;; esac
    case $group in all | docs) cases_docs ;; esac
    case $group in all | tags) cases_tags ;; esac
    [ "$group" != all ] || cases_rest
    return "$BAD"
  }
  # mutant <name> <group> <old text> <new text>: pick.sh with the first <old text> replaced; at least one case of the group must go red
  mutant() {
    cp "$self" "$r/tools/release/pick-mutant.sh"
    python3 -I - "$r/tools/release/pick-mutant.sh" "$3" "$4" << 'PY' || { echo "pick self-test: mutant $1: the text to replace is not in the code" >&2; fails=$((fails + 1)); return 0; }
import sys

path, old, new = sys.argv[1:4]
s = open(path).read()
i = s.find(old)
if i < 0:
    sys.exit(1)
open(path, "w").write(s[:i] + new + s[i + len(old):])
PY
    if all_cases "$r/tools/release/pick-mutant.sh" 0 "$2"; then
      echo "pick self-test: mutant $1 left every case green: the cases do not hold it" >&2
      fails=$((fails + 1))
    fi
  }
  all_cases "$r/tools/release/pick.sh" 1 || fails=$((fails + BAD))
  n_cases=$cases
  mutant "the 25 minutes are exclusive" rule 'elif [ "$DELTA_S" -le "$LIMIT_S" ]; then' 'elif [ "$DELTA_S" -lt "$LIMIT_S" ]; then'
  mutant "an unproven delta is taken as proven" rule 'elif [ -z "$DELTA_S" ]; then' 'elif false; then'
  mutant "HEAD whatever the delta" rule 'PICK=W REASON="the HEAD delta needs' 'PICK=HEAD REASON="the HEAD delta needs'
  mutant "W whatever the delta" rule 'PICK=HEAD REASON="the HEAD delta proves' 'PICK=W REASON="the HEAD delta proves'
  mutant "a watermark that is not an ancestor is used" rule $'\n    if git merge-base --is-ancestor "$W_FULL" "$HEAD_FULL"; then\n' $'\n    if true; then\n'
  mutant "the last tag is the nearest ancestor" tags $'\nTAGS=$(git tag -l \'v[0-9]*\' --sort=-v:refname)\n' $'\nTAGS=$(git describe --tags --abbrev=0 --match \'v[0-9]*\' "$HEAD_FULL" 2> /dev/null)\n'
  mutant "an archive built before a crates change is ready" docs $'\n    elif [ -n "$tar_c" ] && tar_d=$(bash tools/release/publish.sh --docs-delta "$tar_c" "$PICK_FULL" 2> /dev/null < /dev/null); then\n' $'\n    elif [ -n "$tar_c" ] && tar_d=ok; then\n'
  mutant "an archive behind a docs commit is not ready" docs $'\n    elif [ -n "$tar_c" ] && tar_d=$(bash tools/release/publish.sh --docs-delta "$tar_c" "$PICK_FULL" 2> /dev/null < /dev/null); then\n' $'\n    elif false; then\n'
  mutant "the README is not looked at" rows $'\n[ "$W_OK" != 1 ] || W_TREE=$(tree_row "$W_FULL")\n' $'\nW_TREE=ok\n'
  mutant "an archive of another commit is ready" rows $'\n    elif [ "${tar_c:0:12}" = "$pc" ] && [ -n "$tar_c" ]; then\n' $'\n    elif true; then\n'
  rm -rf "$t"
  if [ "$fails" = 0 ]; then
    echo "pick.sh: self-test ok ($n_cases cases, $((cases - n_cases)) mutant runs)"
  else
    echo "pick.sh: self-test $fails failure(s) in $cases cases" >&2
    return 1
  fi
}

VERSION='' HEAD_REV=HEAD W_REV='' W_FILE=${BLOOMERY_FOLLOW_DIR:-$HOME/.cache/bloomery/follow}/W CELLS=${PICK_CELLS:-}
if [ "${1:-}" = --self-test ]; then
  [ $# = 1 ] || die 64 "$USAGE"
  self_test
  exit $?
fi
while [ $# -gt 0 ]; do
  case $1 in
    --version | --head | --w | --w-file | --cells)
      [ $# -ge 2 ] || die 64 "$1 needs a value; $USAGE"
      case $1 in --version) VERSION=$2 ;; --head) HEAD_REV=$2 ;; --w) W_REV=$2 ;; --w-file) W_FILE=$2 ;; --cells) CELLS=$2 ;; esac
      shift 2 ;;
    *) die 64 "unknown argument $1; $USAGE" ;;
  esac
done

HEAD_FULL=$(git rev-parse --verify --quiet "$HEAD_REV^{commit}") || die 64 "--head $HEAD_REV is no commit of this tree"
[ -n "$VERSION" ] || VERSION=$(git show "$HEAD_FULL:crates/serve/Cargo.toml" 2> /dev/null | awk -F'"' '/^version = "/ {print $2; exit}')
[ -n "$VERSION" ] || die 70 "cannot read the serve crate's version at $HEAD_REV (--version names it)"
WORK=$(mktemp -d "${TMPDIR:-/tmp}/pick.XXXXXX") || die 70 "cannot make a temporary directory"
trap 'rm -rf "$WORK"' EXIT

# W
W_OK=0 W_WHY='' W_FULL='' W_COMMIT='' W_TIME='' W_SIZE='' W_SRC=''
if [ -n "$W_REV" ]; then
  W_COMMIT=$W_REV W_SRC=--w
elif [ -f "$W_FILE" ]; then
  W_SRC=$W_FILE
  w_read "$W_FILE" || W_WHY="$W_FILE holds no commit"
else
  W_WHY="no watermark file $W_FILE: the follower has written none"
fi
if [ -n "$W_COMMIT" ]; then
  if W_FULL=$(git rev-parse --verify --quiet "$W_COMMIT^{commit}"); then
    if git merge-base --is-ancestor "$W_FULL" "$HEAD_FULL"; then
      W_OK=1
    else
      W_WHY="W $W_COMMIT is not an ancestor of HEAD: history moved"
    fi
  else
    W_WHY="W $W_COMMIT is not a commit of this tree"
  fi
fi
SAME=0
[ "$W_OK" != 1 ] || [ "$W_FULL" != "$HEAD_FULL" ] || SAME=1

# the delta
DELTA_S='' DELTA_WHY='' DELTA_NOTE='' NGATES=0 LATE_ROW='' NCOMMITS=0 LOG=''
if [ "$W_OK" = 1 ]; then
  NCOMMITS=$(git rev-list --count "$W_FULL..$HEAD_FULL")
  LOG=$(git log --oneline --no-decorate -n 8 "$W_FULL..$HEAD_FULL")
fi
if [ "$W_OK" = 1 ] && [ "$SAME" = 0 ]; then
  rc=0
  out=$(run_hook PICK_LATE "bash tools/release/late-commit.sh" "$W_FULL" "$HEAD_FULL" 2>&1) || rc=$?
  LATE_ROW="rc $rc: ${out%%$'\n'*}"
  rc=0
  run_hook PICK_AFFECTED "just affected" "$W_FULL..$HEAD_FULL" > "$WORK/affected.txt" 2> "$WORK/affected.err" || rc=$?
  if [ "$rc" != 0 ]; then
    DELTA_WHY="just affected ended rc $rc: $(head -n 1 "$WORK/affected.err")"
  else
    NGATES=$(grep -cE '^  (gate|weekly)-' "$WORK/affected.txt" || true)
    if [ "$NGATES" = 0 ]; then
      DELTA_S=0 DELTA_NOTE="the commits after W select no gate"
    else
      rc=0
      run_hook PICK_BATCH "bash tools/gate-batch.sh" --list "$WORK/affected.txt" --tier real --ledger --dry-run > "$WORK/batch.txt" 2>&1 || rc=$?
      if [ "$rc" != 0 ]; then
        DELTA_WHY="gate-batch --dry-run ended rc $rc: $(grep -m1 -E '^gate-batch:' "$WORK/batch.txt" || head -n 1 "$WORK/batch.txt")"
      else
        DELTA_S=$(awk '/^gate-batch: predicted / { if (match($0, / wall=[0-9]+s/)) { print substr($0, RSTART + 6, RLENGTH - 7); exit } }' "$WORK/batch.txt")
        [ -n "$DELTA_S" ] || DELTA_WHY="gate-batch printed no predicted line"
      fi
    fi
  fi
fi
pick_rule
PICK_FULL=$HEAD_FULL
[ "$PICK" = HEAD ] || PICK_FULL=$W_FULL

# the rows
W_TREE='' H_TREE=''
[ "$W_OK" != 1 ] || W_TREE=$(tree_row "$W_FULL")
H_TREE=$(tree_row "$HEAD_FULL")
SIT_ROW='not priced (no --cells FILE)' SIT_S=''
if [ -n "$CELLS" ]; then
  rc=0
  out=$(run_hook PICK_SITTING "bash tools/release/sitting.sh" --cells "$CELLS" --binary tree --dry-run 2>&1) || rc=$?
  line=$(grep -m1 '^\[plan\] predicted wall' <<< "$out" || true)
  if [ "$rc" = 0 ] && [ -n "$line" ]; then
    SIT_ROW=${line#\[plan\] } SIT_S=$(sed -n 's/^\[plan\] predicted wall [>=]* \([0-9]*\) s.*/\1/p' <<< "$line")
  else
    SIT_ROW="not priced (sitting.sh --dry-run ended rc $rc: ${out%%$'\n'*})"
  fi
fi
rc=0
TARS=$(tarball_rows "$VERSION" 2>&1) || rc=$?
TAR_ROWS=() TAR_BAD=()
PC=${PICK_FULL:0:12}
if [ "$rc" != 0 ]; then
  TAR_ROWS+=("not read (${TARS%%$'\n'*})")
  TAR_BAD+=("archives not read")
else
  while read -r tar_a tar_s tar_c; do
    [ -n "$tar_a" ] || continue
    pc=$PC
    if [ "$tar_s" = absent ]; then
      TAR_ROWS+=("$tar_a absent")
      TAR_BAD+=("archive absent: $tar_a")
    elif [ "${tar_c:0:12}" = "$pc" ] && [ -n "$tar_c" ]; then
      TAR_ROWS+=("$tar_a ${tar_s:0:12} built at ${tar_c:0:12}")
    elif [ -n "$tar_c" ] && tar_d=$(bash tools/release/publish.sh --docs-delta "$tar_c" "$PICK_FULL" 2> /dev/null < /dev/null); then
      TAR_ROWS+=("$tar_a ${tar_s:0:12} built at ${tar_c:0:12}, $tar_d")
    else
      TAR_ROWS+=("$tar_a ${tar_s:0:12} built at ${tar_c:-?}, not $pc")
      TAR_BAD+=("archive built at another commit: $tar_a")
    fi
  done <<< "$TARS"
fi
if [ ${#TAR_ROWS[@]} = 0 ]; then
  TAR_ROWS+=("none reported")
  TAR_BAD+=("no archive reported")
fi
rc=0
ROT=$(run_hook PICK_ROTATION "python3 tools/rotation.py check" 2>&1) || rc=$?
case $rc in
  0) ROT_ROW=green ;;
  1) ROT_ROW="red: $(grep -v '^$' <<< "$ROT" | head -n 1)" ;;
  *) ROT_ROW="could not run (rc $rc): $(grep -v '^$' <<< "$ROT" | head -n 1)" ;;
esac
# The last release tag is the highest version below this release's: a release commit sits on its own line, so no tag need be an
# ancestor of HEAD, and git describe would name an older one.
TAGS=$(git tag -l 'v[0-9]*' --sort=-v:refname)
if grep -qx "v$VERSION" <<< "$TAGS"; then
  LAST_TAG=$(awk -v v="v$VERSION" 'seen { print; exit } $0 == v { seen = 1 }' <<< "$TAGS")
else
  LAST_TAG=$(head -n 1 <<< "$TAGS")
fi
PRS=0 PR_ROW='' PR_LIST=''
if [ -z "$LAST_TAG" ]; then
  PR_ROW="skipped (no release tag before HEAD)"
else
  since=$(git log -1 --format=%cs "$LAST_TAG")
  rc=0
  out=$(run_hook PICK_GH gh pr list --state merged --limit 200 --search "merged:>=$since label:needs-gpu -label:gpu-verified -label:gpu-evidence" \
    --json number,title --jq '.[] | "#\(.number) \(.title)"' 2>&1) || rc=$?
  if [ "$rc" != 0 ]; then
    PR_ROW="skipped (gh: rc $rc: ${out%%$'\n'*})"
  else
    PRS=$(grep -c . <<< "$out" || true)
    PR_ROW="$PRS merged since $LAST_TAG ($since) with needs-gpu and no gpu-verified or gpu-evidence"
    PR_LIST=$out
  fi
fi

# the table
echo "pick.sh: release v$VERSION at ${HEAD_FULL:0:12} ($(date '+%Y-%m-%d %H:%M'))"
if [ "$W_OK" = 1 ]; then
  echo "  W                    ${W_FULL:0:12}  ($W_SRC${W_TIME:+, $W_TIME}${W_SIZE:+, $W_SIZE items})"
else
  echo "  W                    unusable: $W_WHY"
fi
echo "  HEAD                 ${HEAD_FULL:0:12}"
if [ "$W_OK" = 1 ]; then
  echo "  commits W..HEAD      $NCOMMITS"
  [ -z "$LOG" ] || sed 's/^/                         /' <<< "$LOG"
  if [ "$SAME" = 1 ]; then
    echo "  late-commit          not needed (W is HEAD)"
    echo "  delta batch          none (W is HEAD)"
  else
    echo "  late-commit          $LATE_ROW"
    if [ -n "$DELTA_S" ]; then
      if [ "$NGATES" = 0 ]; then
        echo "  delta batch          $DELTA_NOTE: predicted wall 0 s"
      else
        echo "  delta batch          $NGATES gate(s) selected, predicted wall $DELTA_S s ($((DELTA_S / 60)) min) [gate-batch --tier real --ledger --dry-run, the times file's medians]"
      fi
    else
      echo "  delta batch          unproven: $DELTA_WHY"
    fi
  fi
fi
echo "  sitting              $SIT_ROW"
[ "$W_OK" != 1 ] || echo "  README and versions  W: $W_TREE"
echo "  README and versions  HEAD: $H_TREE"
TAR_LINE=''
for row in "${TAR_ROWS[@]}"; do TAR_LINE="${TAR_LINE:+$TAR_LINE; }$row"; done
echo "  tarballs on the box  $TAR_LINE"
echo "  rotation check       $ROT_ROW"
echo "  needs-gpu PRs        $PR_ROW"
if [ -n "$PR_LIST" ]; then
  head -n 12 <<< "$PR_LIST" | sed 's/^/                         /'
  [ "$PRS" -le 12 ] || echo "                         (and $((PRS - 12)) more)"
fi
if [ "$PICK" = HEAD ] && [ -n "$DELTA_S" ] && [ -n "$SIT_S" ]; then
  echo "  predicted to the sitting's verdict  $(((DELTA_S + SIT_S) / 60)) min [derived: the delta batch plus the sitting; builds, the tag and the publish are not in it]"
elif [ "$PICK" = W ] && [ -n "$SIT_S" ]; then
  echo "  predicted to the sitting's verdict  $((SIT_S / 60)) min [derived: the sitting alone; builds, the tag and the publish are not in it]"
fi
echo "pick: $PICK ${PICK_FULL:0:12} — $REASON"
if [ -n "$LATE_ROW" ] && [ "$PICK" = W ]; then echo "  (the newer commits' late-commit verdict: $LATE_ROW)"; fi
# ready: what stands between the pick and the sitting
BLOCK=()
if [ "$PICK" = W ]; then P_TREE=$W_TREE; else P_TREE=$H_TREE; fi
[ "$P_TREE" = ok ] || BLOCK+=("README at the pick: $P_TREE")
[ ${#TAR_BAD[@]} = 0 ] || BLOCK+=("${TAR_BAD[@]}")
case $ROT_ROW in green) ;; *) BLOCK+=("the rotation is ${ROT_ROW%%:*}") ;; esac
[ "$PRS" = 0 ] || BLOCK+=("needs-gpu PRs: $PRS without card evidence")
if [ ${#BLOCK[@]} = 0 ]; then
  echo "ready: yes"
else
  printf 'ready: no — %s\n' "$(IFS=';'; echo "${BLOCK[*]}")"
fi
exit 0
