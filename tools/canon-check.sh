#!/usr/bin/env bash
# `just canon-check BASE [HEAD]`: is every PTX entry of every scanned bin the same program on both sides?
#   tools/canon-check.sh BASE [HEAD] [--out DIR] [--dry-run] [--all]
#   tools/canon-check.sh BASE --worktree [--out DIR] [--dry-run] [--all]
#   tools/canon-check.sh --from DIR [--all]
#   tools/canon-check.sh --self-test
#
# AGENTS.md's change-class table proves a move, split or rename by identical `just ptx-scan` tables, and a delete or a fold
# whose remaining entry's md5 moves by tools/ref/ptx-canon.py (`reordered-only`, `value-equal` or `identical`, equal resource
# columns). This is that proof in one command. It runs tools/narrow-scan.sh with `--keep-ptx --scan-only` (both sides' `--no-jit` scans of
# the scan set, and each bin's PTX modules brought back; the affected run is not wanted here), then for each bin:
#   1. `ptx-canon.py --scans BASE.log HEAD.log`: the entry sets, every table column, the md5 block;
#   2. for each entry whose md5 moved: the module that holds its `.entry` on each side, and
#      `ptx-canon.py BASE.ptx HEAD.ptx ENTRY`, whose diff and verdict go to DIR/canon/<bin>.<entry>.txt.
# ptx-canon reads the raw module text (`oxart_ptx <section> <dir>`'s mod<N>.ptx: its `body` cuts from `.visible .entry
# <name>(` and its `declarations` read the module's generated symbols), not the extractor's `--norm` bodies, which are the
# text the md5 is of and have those symbols already replaced. So the modules kept are the raw ones (tools/ptx-scan.sh's
# PTX_SCAN_KEEP).
#
# Output: one line per entry that moved or is on one side only, then
#   canon-check: <bins> bins, <entries> entries: <i> identical, <r> reordered-only, <v> value-equal, <d> differs, <o> only-on-one-side; rows <equal|moved: ...>
# <entries> counts each bin's entries of both sides once; <i> the entries whose md5 is equal and those whose md5 moved and
# whose canonical body is identical; <r> the reordered-only entries whose every swapped pair ptx-canon shows independent and
# whose value proof also holds (a reorder that passes the opcode screen alone is its exit 1, counted as differs);
# <v> the value-equal entries (ptx-canon's checked proof that the loads, stores, barriers and branches run in the same order on
# equal values, whatever registers hold them); <d> the entries that differ and the reorders it cannot show independent (its own
# exit 1: `read the diff`); <o> the entries one side lacks. `rows equal` carries the note when the scans were `--no-jit` (their
# two JIT columns are never compared). A reordered-only or value-equal entry is a proven move only beside equal resource
# columns: a moved column is rc 1 whatever the verdicts are.
#
# `--all` also reads the entries whose md5 did NOT move. The digest is blind to which register an operand names, so an entry it
# calls equal can still differ, or differ only in numbering that first-appearance numbering cannot absorb. One more line, before
# the summary, counts what ptx-canon found there:
#   canon-check: --all: <k> md5-equal entries read: <i> identical, <r> reordered-only, <v> value-equal, <d> differs
# and each entry that is not identical is named (`md5 equal; ptx-canon: ...`). <d> there makes the exit code 1 (the digest hid a
# changed program); reordered-only and value-equal are information only. The default run reads no such entry and its lines and
# code are the same with or without the flag. It costs one ptx-canon process per pair of kept modules of a bin (about 40 s for
# the 591 entries of one big module), so it is not the default. A module that holds no `.visible .entry` is left out of the pairs
# (ptx-canon --all refuses one); an md5-equal entry that no remaining pair answers is still refused by name.
#
# Exit codes: 0 only when no entry differs, none is on one side only, every swapped pair is independent and every compared
# row column is equal; 1 when any of those fails (under `--all` also an md5-equal entry that ptx-canon reads as differing); 2 on
# a usage error, a scan or log ptx-canon refuses (a failed scan, two digest methods, a column it cannot read), a bin whose two
# scans hold no entry at all (an empty extraction compared nothing), a moved entry
# whose module is missing (or in several), a ptx-canon answer this script cannot read (a verdict word it does not know, a
# known one with the wrong exit code, `renamed-only` included), under `--all` a md5-equal entry no pair of modules answers (or
# several do), or a narrow-scan run that did not complete — each named, never a silent pass; 75 when narrow-scan is held by
# contention (run again); 130/143 on INT/TERM.
#
# `--from DIR` judges a DIR a `narrow-scan.sh --keep-ptx` run left (scan-set.tsv, base/ and head/ with <bin>.log and
# <bin>.ptx/), with no box: the way to re-read a result or to judge one fetched by hand (`--all` too). DIR defaults, for a run, to
# ~/.cache/bloomery/canon/<tree>/<base8>-<head8> (<base8>-worktree under `--worktree`). `--dry-run` is narrow-scan's: the scan
# set and the remote dirs, no box. The base side comes from narrow-scan's cache when BASE, the scanner, the scan set and the
# flags are the same as a run before (tools/narrow-scan.sh header; NARROW_NO_CACHE=1 forces a rebuild).
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CANON=$ROOT/tools/ref/ptx-canon.py
USAGE="usage: canon-check.sh BASE [HEAD | --worktree] [--out DIR] [--dry-run] [--all] | --from DIR [--all] | --self-test"

die() { # die <rc> <message>
  echo "canon-check: $2" >&2
  exit "$1"
}

# refuse <message>: a refusal is named and counted; the run ends 2 after the other bins are read.
REFUSED=0
refuse() {
  echo "canon-check: $1" >&2
  REFUSED=$((REFUSED + 1))
}

# modfor <module dir> <entry>: prints the one module of the dir that holds `.visible .entry <entry>(`, the text ptx-canon cuts a
# body from; else 1 and the reason on stdout.
modfor() {
  local dir=$1 e=$2 f hits=0 found=''
  [ -d "$dir" ] || { echo "no kept modules: $dir is not a directory (the scan ran without --keep-ptx?)"; return 1; }
  for f in "$dir"/mod[0-9]*.ptx; do
    [ -e "$f" ] || continue
    if grep -qF -e ".visible .entry $e(" "$f"; then
      hits=$((hits + 1))
      found=$f
    fi
  done
  case $hits in
    0) echo "no module under $dir holds .entry $e"; return 1 ;;
    1) printf '%s\n' "$found" ;;
    *) echo "$hits modules under $dir hold .entry $e"; return 1 ;;
  esac
}

# The run's counters, read by the summary: identical, reordered-only (every swapped pair independent), value-equal, differs,
# only on one side, the entries seen, the rows that moved ("bin:entry(col:a->b,...)") and the no-jit note of the scans. ALL_*
# are `--all`'s own: the md5-equal entries ptx-canon read, by verdict.
CNT_I=0 CNT_R=0 CNT_V=0 CNT_D=0 CNT_O=0 CNT_E=0 ROWS_MOVED='' JIT_NOTE=''
ALL=0 ALL_I=0 ALL_R=0 ALL_V=0 ALL_D=0

# judge_entry <dir> <bin> <entry>: one entry whose md5 moved: ptx-canon on the two sides' modules.
judge_entry() {
  local dir=$1 b=$2 e=$3 bm hm why out err crc verdict word
  case $e in *[!A-Za-z0-9_\$]* | '') refuse "$b: the entry name '$e' is not a PTX entry name"; return ;; esac
  bm=$(modfor "$dir/base/$b.ptx" "$e") || { refuse "$b $e: $bm"; return; }
  hm=$(modfor "$dir/head/$b.ptx" "$e") || { refuse "$b $e: $hm"; return; }
  out=$dir/canon/$b.$e.txt
  err=$dir/canon/$b.$e.err
  python3 "$CANON" "$bm" "$hm" "$e" > "$out" 2> "$err"
  crc=$?
  if [ "$crc" -gt 1 ]; then
    refuse "$b $e: ptx-canon refused: $(head -n 1 "$err")"
    return
  fi
  verdict=$(awk -v p="ptx-canon: $e " 'index($0, p) == 1 { print substr($0, length(p) + 1); exit }' "$out")
  word=${verdict%% *}
  case $word:$crc in
    identical:0)
      CNT_I=$((CNT_I + 1))
      echo "canon-check: $b $e: md5 moved; ptx-canon: $verdict"
      ;;
    reordered-only:0)
      CNT_R=$((CNT_R + 1))
      echo "canon-check: $b $e: md5 moved; ptx-canon: $verdict"
      ;;
    value-equal:0)
      CNT_V=$((CNT_V + 1))
      echo "canon-check: $b $e: md5 moved; ptx-canon: $verdict"
      ;;
    reordered-only:1)
      CNT_D=$((CNT_D + 1))
      echo "canon-check: $b $e: md5 moved; ptx-canon: $verdict (counted as differs; diff: $out)"
      ;;
    differs:1)
      CNT_D=$((CNT_D + 1))
      echo "canon-check: $b $e: md5 moved; ptx-canon: $verdict (diff: $out)"
      ;;
    *) refuse "$b $e: ptx-canon exited $crc with a verdict this script cannot read ('$verdict'): $out" ;;
  esac
}

# has_entry <module>: whether the module text holds a `.visible .entry` (what `ptx-canon --all` reads; it refuses a module with none).
has_entry() { grep -qE '\.visible \.entry [A-Za-z0-9_$]+\(' "$1"; }

# judge_equal <dir> <bin> <names file>: `--all`'s reading of the bin's entries whose md5 did not move (their names, one per line,
# from ptx-canon --scans --list-equal). One `ptx-canon --all` per pair of kept modules answers every shared entry of the pair; the
# answers for the named entries are kept, and each named entry must be answered by exactly one pair. The entries that are not
# identical are named; the counts go to ALL_*.
judge_equal() {
  local dir=$1 b=$2 names=$3 side bm hm run rc answers cov e runf rest word
  answers=$dir/canon/$b.all.answers
  : > "$answers"
  for side in base head; do
    [ -d "$dir/$side/$b.ptx" ] || { refuse "$b: --all: no kept modules: $dir/$side/$b.ptx is not a directory (the scan ran without --keep-ptx?)"; return; }
  done
  for bm in "$dir/base/$b.ptx"/mod[0-9]*.ptx; do
    [ -e "$bm" ] || continue
    has_entry "$bm" || continue # a module with no entry answers none; the coverage check below still names an entry nobody answers
    for hm in "$dir/head/$b.ptx"/mod[0-9]*.ptx; do
      [ -e "$hm" ] || continue
      has_entry "$hm" || continue
      run=$dir/canon/$b.all.$(basename "$bm" .ptx)-$(basename "$hm" .ptx)
      python3 "$CANON" --all "$bm" "$hm" > "$run.txt" 2> "$run.err"
      rc=$?
      if [ "$rc" -gt 1 ]; then
        refuse "$b: --all: ptx-canon --all $bm $hm refused: $(head -n 1 "$run.err")"
        return
      fi
      awk -v f="$run.txt" 'NR == FNR { eq[$1] = 1; next }
        $1 == "ptx-canon:" && ($2 in eq) && $3 != "note:" && $3 != "only" {
          line = $0
          sub(/^ptx-canon: [^ ]+ /, "", line)
          printf "%s\t%s\t%s\n", $2, f, line
        }' "$names" "$run.txt" >> "$answers"
    done
  done
  cov=$(awk -F'\t' 'NR == FNR { eq[$1] = 1; next } { n[$1]++ }
    END { for (e in eq) { if (n[e] == 0) print "none " e; else if (n[e] > 1) print n[e] " " e } }' "$names" "$answers" | sort | head -n 1)
  case $cov in
    '') ;;
    none\ *) refuse "$b: --all: ptx-canon --all gave no answer for the md5-equal entry ${cov#none } (no pair of kept modules holds it on both sides): $dir/canon/$b.all.*"; return ;;
    *) refuse "$b: --all: the md5-equal entry ${cov#* } was answered by ${cov%% *} module pairs (it is in several modules of a side): $dir/canon/$b.all.*"; return ;;
  esac
  while IFS=$'\t' read -r e runf rest; do
    [ -n "$e" ] || continue
    word=${rest%% *}
    case $word in
      identical) ALL_I=$((ALL_I + 1)) ;;
      value-equal)
        ALL_V=$((ALL_V + 1))
        echo "canon-check: $b $e: md5 equal; ptx-canon: $rest"
        ;;
      reordered-only)
        case $rest in
          *"every one independent"*)
            ALL_R=$((ALL_R + 1))
            echo "canon-check: $b $e: md5 equal; ptx-canon: $rest"
            ;;
          *)
            ALL_D=$((ALL_D + 1))
            echo "canon-check: $b $e: md5 equal; ptx-canon: $rest (counted as differs; diff: $runf)"
            ;;
        esac
        ;;
      differs)
        ALL_D=$((ALL_D + 1))
        echo "canon-check: $b $e: md5 equal; ptx-canon: $rest (diff: $runf)"
        ;;
      *) refuse "$b $e: --all: a ptx-canon verdict this script cannot read ('$rest'): $runf" ;;
    esac
  done < "$answers"
}

# judge_bin <dir> <bin>: ptx-canon --scans on the bin's two logs, then each moved entry.
judge_bin() {
  local dir=$1 b=$2 base head out err src line nb nn ob on idn mvn n e side cols note
  local only_n=0 rows_n=0 moved_n=0 moved='' list='' eq_n=0
  base=$dir/base/$b.log
  head=$dir/head/$b.log
  for line in "$base" "$head"; do
    [ -s "$line" ] || { refuse "$b: $line is missing or empty (the scan of $b)"; return; }
  done
  out=$dir/canon/$b.scans.txt
  err=$dir/canon/$b.scans.err
  [ "$ALL" = 0 ] || list=--list-equal
  python3 "$CANON" --scans $list "$base" "$head" > "$out" 2> "$err"
  src=$?
  if [ "$src" -gt 1 ]; then
    refuse "$b: ptx-canon --scans refused: $(head -n 1 "$err")"
    return
  fi
  line=$(sed -En 's/^ptx-canon: scans method=[^ ]+ entries base=([0-9]+) new=([0-9]+) only-base=([0-9]+) only-new=([0-9]+)$/\1 \2 \3 \4/p' "$out")
  [ -n "$line" ] || { refuse "$b: ptx-canon --scans printed no entries line this script can read: $out"; return; }
  read -r nb nn ob on <<< "$line"
  [ $((nb + nn)) -gt 0 ] || { refuse "$b: both scans hold no entry: nothing was compared (an empty extraction is no proof): $out"; return; }
  line=$(sed -En 's/^ptx-canon: md5 identical=([0-9]+) moved=([0-9]+)$/\1 \2/p' "$out")
  [ -n "$line" ] || { refuse "$b: ptx-canon --scans printed no md5 line this script can read: $out"; return; }
  read -r idn mvn <<< "$line"
  CNT_E=$((CNT_E + nb + on))
  CNT_I=$((CNT_I + idn))
  note=$(sed -En 's/^ptx-canon: jit columns not compared \(no-jit: (base|new|both)\)$/\1/p' "$out")
  case " $JIT_NOTE " in *" $note "*) ;; *) [ -z "$note" ] || JIT_NOTE=${JIT_NOTE:+$JIT_NOTE }$note ;; esac
  while read -r e side; do
    [ -n "$e" ] || continue
    only_n=$((only_n + 1))
    CNT_O=$((CNT_O + 1))
    echo "canon-check: $b $e: only in $side"
  done < <(sed -En 's/^ptx-canon: ([A-Za-z0-9_$]+) only in (base|new)$/\1 \2/p' "$out")
  while read -r e cols; do
    [ -n "$e" ] || continue
    rows_n=$((rows_n + 1))
    echo "canon-check: $b $e: row moved $cols"
    ROWS_MOVED=${ROWS_MOVED:+$ROWS_MOVED }$b:$e'('${cols// /,}')'
  done < <(sed -En 's/^ptx-canon: row-moved ([A-Za-z0-9_$]+) (.*)$/\1 \2/p' "$out")
  moved=$(sed -En 's/^ptx-canon: md5-moved ([A-Za-z0-9_$]+) lines .*$/\1/p' "$out")
  [ -z "$moved" ] || moved_n=$(printf '%s\n' "$moved" | wc -l | tr -d ' ')
  if [ "$ALL" = 1 ]; then
    sed -En 's/^ptx-canon: md5-equal ([A-Za-z0-9_$]+)$/\1/p' "$out" | sort > "$dir/canon/$b.equal.txt"
    eq_n=$(wc -l < "$dir/canon/$b.equal.txt" | tr -d ' ')
    if [ "$eq_n" != "$idn" ]; then
      refuse "$b: ptx-canon --scans's md5 identical count ($idn) does not match the md5-equal lines it printed ($eq_n): $out"
      return
    fi
  fi
  if [ "$only_n" != $((ob + on)) ] || [ "$moved_n" != "$mvn" ]; then
    refuse "$b: ptx-canon --scans's counts (only-on-one-side $((ob + on)), md5 moved $mvn) do not match the lines it printed ($only_n, $moved_n): $out"
    return
  fi
  if [ "$src" = 1 ] && [ "$only_n" = 0 ] && [ "$rows_n" = 0 ] && [ "$moved_n" = 0 ]; then
    refuse "$b: ptx-canon --scans exited 1 and named no entry, row or digest that moved: cannot classify (the md5 block and the table name other entries?): $out"
    return
  fi
  for e in $moved; do judge_entry "$dir" "$b" "$e"; done
  [ "$ALL" = 0 ] || judge_equal "$dir" "$b" "$dir/canon/$b.equal.txt"
}

# judge_dir <dir>: every bin of the dir's scan-set.tsv; prints the entry lines and the summary; the exit code of the header.
judge_dir() {
  local dir=$1 b bins='' nbins=0 rows
  [ -d "$dir" ] || die 2 "$dir is not a directory"
  [ -s "$dir/scan-set.tsv" ] || die 2 "$dir/scan-set.tsv is missing or empty: not a narrow-scan directory"
  while IFS=$'\t' read -r b _; do
    [ -n "$b" ] || continue
    case $b in *[!A-Za-z0-9_-]* | -*) die 2 "$dir/scan-set.tsv: '$b' is not a bin name" ;; esac
    bins="$bins $b"
    nbins=$((nbins + 1))
  done < "$dir/scan-set.tsv"
  [ "$nbins" -gt 0 ] || die 2 "$dir/scan-set.tsv names no bin"
  rm -rf "$dir/canon"
  mkdir -p "$dir/canon" || die 2 "cannot make $dir/canon"
  for b in $bins; do judge_bin "$dir" "$b"; done
  if [ "$REFUSED" -gt 0 ]; then
    echo "canon-check: refused ($REFUSED above): no verdict" >&2
    return 2
  fi
  if [ -n "$ROWS_MOVED" ]; then
    rows="moved: $ROWS_MOVED"
  else
    rows=equal
    [ -z "$JIT_NOTE" ] || rows="equal (jit columns not compared: no-jit ${JIT_NOTE// //})"
  fi
  if [ "$ALL" = 1 ]; then
    echo "canon-check: --all: $((ALL_I + ALL_R + ALL_V + ALL_D)) md5-equal entries read: $ALL_I identical, $ALL_R reordered-only, $ALL_V value-equal, $ALL_D differs"
  fi
  echo "canon-check: $nbins bins, $CNT_E entries: $CNT_I identical, $CNT_R reordered-only, $CNT_V value-equal, $CNT_D differs, $CNT_O only-on-one-side; rows $rows"
  [ "$CNT_D" = 0 ] && [ "$CNT_O" = 0 ] && [ -z "$ROWS_MOVED" ] && [ "$ALL_D" = 0 ]
}

self_test() {
  local t n=0 bad=0 rc got d w wb h
  t=$(mktemp -d "${TMPDIR:-/tmp}/canon-check-test.XXXXXX") || die 2 "self-test: no temporary directory"
  # ok <name> <condition rc>
  ok() {
    n=$((n + 1))
    if [ "$2" = 0 ]; then echo "ok $1"; else bad=$((bad + 1)); echo "FAIL $1"; fi
  }
  # hx <n>: a 32-digit md5-shaped string, the entries' digests of the fixtures
  hx() { printf '%032d' "$1"; }
  # fx_scan <file> <nojit 0|1> <regs> <method> <name=hash index>...: a ptx-scan log as narrow-scan keeps it (the scan's stderr line
  # first), rows in the order given and the md5 block after them
  fx_scan() {
    local f=$1 nojit=$2 regs=$3 meth=$4 spec e i jit banner
    shift 4
    if [ "$nojit" = 1 ]; then jit='skipped(no-jit) skipped(no-jit)' banner=' jit=skipped(no-jit)'; else jit='14 8' banner=' jit-card=fx jit-cc=86 jit-cuda=13030'; fi
    {
      echo 'ptx-scan: mod1 bundle=fixture bytes=100'
      echo "ptx-scan bin=target/release/gx section=.oxart bytes=1 ptxas=/p ptxas-version=13.3.73 arch=sm_86 modules=1$banner"
      echo 'entry reqntid depot ld.local st.local fma cvt.f16 regs smem spill blk/SM(static) jit_regs jit_local'
      for spec in "$@"; do
        printf '%-28s 256 no 0 0 1 0 %s 0 0 16 %s\n' "${spec%%=*}" "$regs" "$jit"
      done
      echo "ptx-scan-md5: method=$meth"
      for spec in "$@"; do
        e=${spec%%=*} i=${spec#*=}
        echo "$e $(hx "$i") 9"
      done
    } > "$f"
  }
  # fx_mod <file> <ra> <rb> <order ab|ba> <const> <rc>: a module with the entries alpha and beta. alpha's two adds touch r<ra>
  # (const) and r<rc> (9); the order swaps them; the numbers are what a renumbering moves.
  fx_mod() {
    local f=$1 ra=$2 rb=$3 order=$4 c1=$5 rc=$6 a1 a2
    a1="add.s32 	%r$ra, %r$ra, $c1;" a2="add.s32 	%r$rc, %r$rc, 9;"
    {
      printf '.version 8.0\n.target sm_86\n.visible .entry alpha(\n\t.param .u64 alpha_param_0\n)\n{\n'
      printf '\tld.param.u64 \t%%rd1, [alpha_param_0];\n\tmov.u32 \t%%r%s, 1;\n\tmov.u32 \t%%r%s, 2;\n' "$ra" "$rb"
      if [ "$order" = ab ]; then printf '\t%s\n\t%s\n' "$a1" "$a2"; else printf '\t%s\n\t%s\n' "$a2" "$a1"; fi
      printf '\tst.global.u32 \t[%%rd1], %%r%s;\n\tret;\n}\n.visible .entry beta(\n)\n{\n\tret;\n}\n' "$ra"
    } > "$f"
  }
  # fx <dir> <base entries> <head entries> [head alpha's fx_mod arguments]: a narrow-scan --keep-ptx directory for the one bin gx.
  # NOJIT_B / NOJIT_H (default 1), REGS_B / REGS_H (12) and METH_B / METH_H (decl1) shape the two scans; the base alpha is
  # fx_mod 1 2 ab 7 2.
  fx() {
    local dir=$1 be=$2 he=$3
    shift 3
    rm -rf "$dir"
    mkdir -p "$dir/base/gx.ptx" "$dir/head/gx.ptx"
    printf 'gx\tgpu\tbin gx\n' > "$dir/scan-set.tsv"
    # shellcheck disable=SC2086
    fx_scan "$dir/base/gx.log" "${NOJIT_B:-1}" "${REGS_B:-12}" "${METH_B:-decl1}" $be
    # shellcheck disable=SC2086
    fx_scan "$dir/head/gx.log" "${NOJIT_H:-1}" "${REGS_H:-12}" "${METH_H:-decl1}" $he
    fx_mod "$dir/base/gx.ptx/mod1.ptx" 1 2 ab 7 2
    if [ $# -gt 0 ]; then fx_mod "$dir/head/gx.ptx/mod1.ptx" "$@"; else fx_mod "$dir/head/gx.ptx/mod1.ptx" 1 2 ab 7 2; fi
  }
  # judge <dir>: the --from run's output in $got, its code in $rc
  judge() {
    got=$(bash "$0" --from "$1" 2>&1)
    rc=$?
  }
  # judge_all <dir>: the same with --all
  judge_all() {
    got=$(bash "$0" --from "$1" --all 2>&1)
    rc=$?
  }
  # fx_val <file> <ab|ba>: a module whose alpha holds two pure adds that read one register; `ba` defines them in the other order,
  # which moves the numbering by first appearance and is no reorder ptx-canon can show independent: it reads value-equal
  fx_val() {
    local f=$1 order=$2 a1 a2
    a1="add.s32 	%r3, %r1, 1;" a2="add.s32 	%r4, %r1, 2;"
    {
      printf '.version 8.0\n.target sm_86\n.visible .entry alpha(\n\t.param .u64 alpha_param_0\n)\n{\n'
      printf '\tld.param.u64 \t%%rd1, [alpha_param_0];\n\tld.global.u32 \t%%r1, [%%rd1];\n'
      if [ "$order" = ab ]; then printf '\t%s\n\t%s\n' "$a1" "$a2"; else printf '\t%s\n\t%s\n' "$a2" "$a1"; fi
      printf '\tst.global.u32 \t[%%rd1+4], %%r3;\n\tst.global.u32 \t[%%rd1+8], %%r4;\n\tret;\n}\n.visible .entry beta(\n)\n{\n\tret;\n}\n'
    } > "$f"
  }
  d=$t/a
  # (a) equal md5 on every entry: nothing to compare in a module, rc 0, the summary counts
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  judge "$d"
  [ "$rc" = 0 ] && grep -qxF "canon-check: 1 bins, 2 entries: 2 identical, 0 reordered-only, 0 value-equal, 0 differs, 0 only-on-one-side; rows equal (jit columns not compared: no-jit both)" <<< "$got"; ok "(a) equal md5s: rc 0, the summary line" $?
  # (b) md5 moved, the canonical body identical (the backend renumbered the registers)
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 5 6 ab 7 6
  judge "$d"
  [ "$rc" = 0 ] && grep -qF "canon-check: gx alpha: md5 moved; ptx-canon: identical" <<< "$got" && grep -qF ": 2 identical, 0 reordered-only, 0 value-equal, 0 differs, 0 only-on-one-side; rows equal" <<< "$got"; ok "(b) md5 moved, ptx-canon identical (renumbered registers): rc 0, counted identical" $?
  # (c) md5 moved, reordered-only with independent swaps
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 1 2 ba 7 2
  judge "$d"
  [ "$rc" = 0 ] && grep -qF "canon-check: gx alpha: md5 moved; ptx-canon: reordered-only (same line multiset); 1 swapped pair(s) in a 2-line window, every one independent" <<< "$got" && grep -qF ": 1 identical, 1 reordered-only, 0 value-equal, 0 differs, 0 only-on-one-side; rows equal" <<< "$got"; ok "(c) md5 moved, reordered-only with independent swaps: rc 0" $?
  # (d) md5 moved, the program differs
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 1 2 ab 8 2
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "canon-check: gx alpha: md5 moved; ptx-canon: differs (2 changed lines)" <<< "$got" && grep -qF ": 1 identical, 0 reordered-only, 0 value-equal, 1 differs, 0 only-on-one-side; rows equal" <<< "$got" && [ -s "$d/canon/gx.alpha.txt" ]; ok "(d) md5 moved, differs: rc 1, the diff kept under DIR/canon/" $?
  # (e) an entry only in base, and one only in head
  fx "$d" "alpha=1 beta=2 gamma=3" "alpha=1 beta=2"
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "canon-check: gx gamma: only in base" <<< "$got" && grep -qF ": 2 identical, 0 reordered-only, 0 value-equal, 0 differs, 1 only-on-one-side; rows equal" <<< "$got"; ok "(e) an entry only in base: rc 1" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2 gamma=3"
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "canon-check: gx gamma: only in new" <<< "$got" && grep -qF "3 entries: 2 identical, 0 reordered-only, 0 value-equal, 0 differs, 1 only-on-one-side" <<< "$got"; ok "(e) an entry only in head: rc 1" $?
  # (f) a table column moved (regs) with equal md5s
  REGS_H=13 fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "canon-check: gx alpha: row moved regs:12->13" <<< "$got" && grep -qF "rows moved: gx:alpha(regs:12->13) gx:beta(regs:12->13)" <<< "$got"; ok "(f) a non-jit column moved with equal md5s: rc 1, the rows named" $?
  # (g) the JIT columns of a no-jit scan are not compared: a full scan's values against a no-jit scan's, and two no-jit scans
  NOJIT_B=0 fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  judge "$d"
  [ "$rc" = 0 ] && grep -qF "rows equal (jit columns not compared: no-jit new)" <<< "$got"; ok "(g) jit columns: a full base against a no-jit head are not compared, said once" $?
  NOJIT_B=0 NOJIT_H=0 REGS_H=12 fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  judge "$d"
  [ "$rc" = 0 ] && grep -qxF "canon-check: 1 bins, 2 entries: 2 identical, 0 reordered-only, 0 value-equal, 0 differs, 0 only-on-one-side; rows equal" <<< "$got"; ok "(g) two full scans: the jit columns are compared (equal here), no note" $?
  # (h) a moved entry whose module is missing: refused by name (2), never a pass
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  rm -f "$d/head/gx.ptx/mod1.ptx"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "no module under $d/head/gx.ptx holds .entry alpha" <<< "$got" && ! grep -q "^canon-check: 1 bins" <<< "$got"; ok "(h) a moved entry whose head module is missing: rc 2 by name, no summary" $?
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  rm -rf "$d/base/gx.ptx"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "no kept modules: $d/base/gx.ptx is not a directory" <<< "$got"; ok "(h) no kept modules at all (the scan ran without --keep-ptx): rc 2 by name" $?
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 1 2 ab 7 2
  cp "$d/head/gx.ptx/mod1.ptx" "$d/head/gx.ptx/mod2.ptx"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "2 modules under $d/head/gx.ptx hold .entry alpha" <<< "$got"; ok "(h) an entry in two modules: rc 2 by name" $?
  # (i) a reorder ptx-canon cannot show independent: counted as differs, rc 1
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  fx_mod "$d/base/gx.ptx/mod1.ptx" 1 2 ab 7 1
  fx_mod "$d/head/gx.ptx/mod1.ptx" 1 2 ba 7 1
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "ptx-canon: reordered-only (same line multiset); 1 of 1 swapped pair(s) share a register" <<< "$got" && grep -qF "counted as differs" <<< "$got" && grep -qF ": 1 identical, 0 reordered-only, 0 value-equal, 1 differs" <<< "$got"; ok "(i) a reorder over a shared register: rc 1, counted as differs" $?
  # (j) md5 moved, value-equal: a proven move beside equal columns, rc 0, counted in its own column
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  fx_val "$d/base/gx.ptx/mod1.ptx" ab
  fx_val "$d/head/gx.ptx/mod1.ptx" ba
  judge "$d"
  [ "$rc" = 0 ] && grep -qF "canon-check: gx alpha: md5 moved; ptx-canon: value-equal (" <<< "$got" && grep -qF ": 1 identical, 0 reordered-only, 1 value-equal, 0 differs, 0 only-on-one-side; rows equal" <<< "$got"; ok "(j) md5 moved, ptx-canon value-equal: rc 0, counted as value-equal" $?
  # (k) the same beside a moved resource column: rc 1 and the rows named, as for reordered-only; the proof alone is no move
  REGS_H=13 fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  fx_val "$d/base/gx.ptx/mod1.ptx" ab
  fx_val "$d/head/gx.ptx/mod1.ptx" ba
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "ptx-canon: value-equal (" <<< "$got" && grep -qF "canon-check: gx alpha: row moved regs:12->13" <<< "$got" && grep -qF "1 value-equal, 0 differs, 0 only-on-one-side; rows moved: gx:alpha(regs:12->13) gx:beta(regs:12->13)" <<< "$got"; ok "(k) value-equal beside a moved resource column: rc 1, the row named" $?
  # (l) a verdict this script cannot read: a stub ptx-canon answers --scans with the real one and every entry with the line and code
  # it is given. The retired name renamed-only, a word nobody defined, value-equal with exit 1 and the known words with the other
  # code are refused by name (rc 2, no summary); value-equal with exit 0 is read.
  st=$t/stub-tree
  mkdir -p "$st/tools/ref"
  cp "$0" "$st/tools/canon-check.sh"
  cat > "$st/tools/ref/ptx-canon.py" << 'CANON_STUB_PY_EOF'
#!/usr/bin/env python3
import os, subprocess, sys
if sys.argv[1:2] == ["--scans"]:
    rc = subprocess.call([sys.executable, os.environ["REAL_CANON"]] + sys.argv[1:])
    sys.stdout.flush()
    if os.environ.get("STUB_SCANS_EXTRA"):
        print(os.environ["STUB_SCANS_EXTRA"])
    sys.exit(rc)
print(os.environ["STUB_CANON_OUT"])
sys.exit(int(os.environ.get("STUB_CANON_RC", "0")))
CANON_STUB_PY_EOF
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  # stubbed <verdict word> <rc>: the stub's answer for alpha, judged --from the fixture
  stubbed() {
    got=$(REAL_CANON="$CANON" STUB_CANON_OUT="ptx-canon: alpha $1 (stub)" STUB_CANON_RC=$2 bash "$st/tools/canon-check.sh" --from "$d" 2>&1)
    rc=$?
  }
  for v in "value-equal 1" "renamed-only 0" "bogus-word 0" "differs 0" "identical 1"; do
    stubbed "${v% *}" "${v#* }"
    [ "$rc" = 2 ] && grep -qF "gx alpha: ptx-canon exited ${v#* } with a verdict this script cannot read ('${v% *} (stub)')" <<< "$got" && ! grep -q '^canon-check: 1 bins' <<< "$got"; ok "(l) an unreadable verdict '$v': refused by name, rc 2, no summary" $?
  done
  stubbed value-equal 0
  [ "$rc" = 0 ] && grep -qF "canon-check: gx alpha: md5 moved; ptx-canon: value-equal (stub)" <<< "$got" && grep -qF "1 value-equal, 0 differs" <<< "$got"; ok "(l) value-equal with exit 0 is read" $?
  # the same under --all: the stub answers every pair of modules with a verdict line for alpha and a readable one for beta
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  for v in bogus-word renamed-only; do
    got=$(REAL_CANON="$CANON" STUB_CANON_OUT=$'ptx-canon: alpha '"$v"$' (stub)\nptx-canon: beta identical (stub)' STUB_CANON_RC=0 bash "$st/tools/canon-check.sh" --from "$d" --all 2>&1)
    rc=$?
    [ "$rc" = 2 ] && grep -qF "gx alpha: --all: a ptx-canon verdict this script cannot read ('$v (stub)')" <<< "$got" && ! grep -q '^canon-check: 1 bins' <<< "$got"; ok "(l) --all: an unreadable verdict '$v': refused by name, rc 2, no summary" $?
  done
  # a --scans whose md5-equal lines do not add up to its own md5 count (the stub adds one) is refused under --all
  got=$(REAL_CANON="$CANON" STUB_SCANS_EXTRA="ptx-canon: md5-equal ghost" STUB_CANON_OUT="ptx-canon: alpha identical (stub)" bash "$st/tools/canon-check.sh" --from "$d" --all 2>&1)
  rc=$?
  [ "$rc" = 2 ] && grep -qF "gx: ptx-canon --scans's md5 identical count (2) does not match the md5-equal lines it printed (3)" <<< "$got"; ok "(l) --all: a --scans whose md5-equal lines do not match its count: rc 2 by name" $?
  # scans ptx-canon refuses are named and end 2
  METH_H=decl2 fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "gx: ptx-canon --scans refused" <<< "$got" && grep -qF "do not compare" <<< "$got"; ok "two digest methods: refused by name, rc 2" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  echo 'ptx-scan bin=target/release/gx section=none scan=failed' > "$d/head/gx.log"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "is a failed scan" <<< "$got"; ok "a failed scan log: refused by name, rc 2" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  rm -f "$d/head/gx.log"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "$d/head/gx.log is missing or empty" <<< "$got"; ok "a bin with no log: rc 2 by name" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  rm -f "$d/scan-set.tsv"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "scan-set.tsv" <<< "$got"; ok "a DIR with no scan-set.tsv: rc 2 by name" $?
  judge "$t/no-such-dir"
  [ "$rc" = 2 ] && grep -qF "no-such-dir" <<< "$got"; ok "--from a missing directory: rc 2 by name" $?
  # an md5 block that names other entries than the rows (a cut or mixed log): ptx-canon refuses the scan, so no verdict; alone,
  # and beside a moved entry whose body is identical (which would otherwise count and pass with beta unexamined)
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  sed -i.bak '/^beta 0000/d' "$d/head/gx.log"
  rm -f "$d/head/gx.log.bak"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "entries in only one of the table and the md5 block (beta)" <<< "$got"; ok "an md5 block missing a table entry: rc 2, the coverage refusal named" $?
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 5 6 ab 7 6
  sed -i.bak '/^beta 0000/d' "$d/head/gx.log"
  rm -f "$d/head/gx.log.bak"
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "entries in only one of the table and the md5 block (beta)" <<< "$got" && ! grep -q '^canon-check: 1 bins' <<< "$got"
  ok "a moved identical alpha beside beta's missing digest: rc 2, the coverage refusal, no summary" $?
  # two bins: counted, and one bin's moved entry judged in its own modules
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 5 6 ab 7 6
  mkdir -p "$d/base/gy.ptx" "$d/head/gy.ptx"
  printf 'gx\tgpu\tbin gx\ngy\tgpu\tbin gy\n' > "$d/scan-set.tsv"
  fx_scan "$d/base/gy.log" 1 12 decl1 alpha=1 beta=2
  fx_scan "$d/head/gy.log" 1 12 decl1 alpha=1 beta=2
  judge "$d"
  [ "$rc" = 0 ] && grep -qxF "canon-check: 2 bins, 4 entries: 4 identical, 0 reordered-only, 0 value-equal, 0 differs, 0 only-on-one-side; rows equal (jit columns not compared: no-jit both)" <<< "$got"; ok "two bins: the counts add up, rc 0" $?
  # (m) --all also reads the entries whose md5 did not move and says what ptx-canon found; the default run is not changed by it
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  judge "$d"
  w0=$got
  judge_all "$d"
  [ "$rc" = 0 ] && grep -qxF "canon-check: --all: 2 md5-equal entries read: 2 identical, 0 reordered-only, 0 value-equal, 0 differs" <<< "$got" && [ "$(grep -v '^canon-check: --all' <<< "$got")" = "$w0" ]; ok "(m) --all, md5s equal and modules the same: rc 0, the --all line, every other line as without it" $?
  fx_val "$d/base/gx.ptx/mod1.ptx" ab
  fx_val "$d/head/gx.ptx/mod1.ptx" ba
  judge "$d"
  [ "$rc" = 0 ] && ! grep -qF -- "--all" <<< "$got" && grep -qxF "canon-check: 1 bins, 2 entries: 2 identical, 0 reordered-only, 0 value-equal, 0 differs, 0 only-on-one-side; rows equal (jit columns not compared: no-jit both)" <<< "$got"; ok "(m) default run, md5 equal over value-equal modules: rc 0, both identical, no --all line" $?
  judge_all "$d"
  [ "$rc" = 0 ] && grep -qF "canon-check: gx alpha: md5 equal; ptx-canon: value-equal (" <<< "$got" && grep -qxF "canon-check: --all: 2 md5-equal entries read: 1 identical, 0 reordered-only, 1 value-equal, 0 differs" <<< "$got"; ok "(m) --all, md5 equal over value-equal modules: the entry named, numbering only is information (rc 0)" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2" 1 2 ba 7 2
  judge_all "$d"
  [ "$rc" = 0 ] && grep -qF "canon-check: gx alpha: md5 equal; ptx-canon: reordered-only (same line multiset); 1 swapped pair(s)" <<< "$got" && grep -qxF "canon-check: --all: 2 md5-equal entries read: 1 identical, 1 reordered-only, 0 value-equal, 0 differs" <<< "$got"; ok "(m) --all, md5 equal over an independent reorder: counted reordered-only, rc 0" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2" 1 2 ab 8 2
  judge "$d"
  [ "$rc" = 0 ] && ! grep -qF -- "--all" <<< "$got"; ok "(m) default run, md5 equal over a changed constant: rc 0 (the digest is the proof there)" $?
  judge_all "$d"
  [ "$rc" = 1 ] && grep -qF "canon-check: gx alpha: md5 equal; ptx-canon: differs (2 changed lines) (diff: " <<< "$got" && grep -qxF "canon-check: --all: 2 md5-equal entries read: 1 identical, 0 reordered-only, 0 value-equal, 1 differs" <<< "$got" && grep -qF -- "-add.s32" "$d"/canon/gx.all.mod1-mod1.txt; ok "(m) --all, md5 equal over a changed constant: counted differs, rc 1, the diff kept" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  fx_mod "$d/base/gx.ptx/mod1.ptx" 1 2 ab 7 1
  fx_mod "$d/head/gx.ptx/mod1.ptx" 1 2 ba 7 1
  judge_all "$d"
  [ "$rc" = 1 ] && grep -qF "share a register" <<< "$got" && grep -qxF "canon-check: --all: 2 md5-equal entries read: 1 identical, 0 reordered-only, 0 value-equal, 1 differs" <<< "$got"; ok "(m) --all, md5 equal over a reorder ptx-canon cannot show independent: counted differs, rc 1" $?
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2" 5 6 ab 7 6
  judge_all "$d"
  [ "$rc" = 0 ] && [ "$(grep -c 'gx alpha' <<< "$got")" = 1 ] && grep -qxF "canon-check: --all: 1 md5-equal entries read: 1 identical, 0 reordered-only, 0 value-equal, 0 differs" <<< "$got"; ok "(m) --all: an entry whose md5 moved is judged once, not read again" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  rm -f "$d/head/gx.ptx/mod1.ptx"
  judge "$d"
  [ "$rc" = 0 ] && ! grep -qF -- "--all" <<< "$got"; ok "(m) default run, md5s equal and a module missing: rc 0 (nothing is read)" $?
  judge_all "$d"
  [ "$rc" = 2 ] && grep -qF "gx: --all: ptx-canon --all gave no answer for the md5-equal entry alpha" <<< "$got" && ! grep -q '^canon-check: 1 bins' <<< "$got"; ok "(m) --all, a module missing: rc 2 by name, no summary" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  rm -rf "$d/base/gx.ptx"
  judge_all "$d"
  [ "$rc" = 2 ] && grep -qF "gx: --all: no kept modules: $d/base/gx.ptx is not a directory" <<< "$got"; ok "(m) --all, no kept modules: rc 2 by name" $?
  fx "$d" "alpha=1 beta=2 gamma=3" "alpha=1 beta=2 gamma=3"
  judge "$d"
  [ "$rc" = 0 ] && ! grep -qF -- "--all" <<< "$got"; ok "(m) default run, a md5-equal entry no module holds: rc 0" $?
  judge_all "$d"
  [ "$rc" = 2 ] && grep -qF "no answer for the md5-equal entry gamma" <<< "$got"; ok "(m) --all, a md5-equal entry no module holds: rc 2 by name" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  cp "$d/base/gx.ptx/mod1.ptx" "$d/base/gx.ptx/mod2.ptx"
  cp "$d/head/gx.ptx/mod1.ptx" "$d/head/gx.ptx/mod2.ptx"
  judge_all "$d"
  [ "$rc" = 2 ] && grep -qF "gx: --all: the md5-equal entry alpha was answered by 4 module pairs" <<< "$got"; ok "(m) --all, an entry in two modules of each side: rc 2 by name" $?
  fx "$d" "alpha=1 beta=2 gamma=3" "alpha=1 beta=2 gamma=3"
  for side in base head; do
    printf '.visible .entry gamma(\n)\n{\n\tmov.u32 \t%%r1, __shared_mem_9;\n\tret;\n}\n' >> "$d/$side/gx.ptx/mod1.ptx"
  done
  judge "$d"
  [ "$rc" = 0 ] && ! grep -qF -- "--all" <<< "$got"; ok "(m) default run, md5s equal over a module ptx-canon would refuse: rc 0 (nothing is read)" $?
  judge_all "$d"
  [ "$rc" = 2 ] && grep -qF "gx: --all: ptx-canon --all" <<< "$got" && grep -qF "entry gamma names __shared_mem_9" <<< "$got" && ! grep -q '^canon-check: 1 bins' <<< "$got"; ok "(m) --all, ptx-canon refuses a module: rc 2, its message named" $?
  # (n) a bin whose two scans hold no entry compared nothing: refused by name (2), never "0 entries" and rc 0
  fx "$d" "" ""
  judge "$d"
  [ "$rc" = 2 ] && grep -qF "gx: both scans hold no entry: nothing was compared" <<< "$got" && ! grep -q '^canon-check: 1 bins' <<< "$got"; ok "(n) both scans with no entry: rc 2 by name, no summary" $?
  judge_all "$d"
  [ "$rc" = 2 ] && grep -qF "gx: both scans hold no entry: nothing was compared" <<< "$got"; ok "(n) --all, both scans with no entry: rc 2 by name" $?
  # (o) a reorder whose every swapped pair passes ptx-canon's opcode screen but not its value proof (two reads of %clock, a
  # volatile special register the screen reads as a plain move): rc 1, counted as differs, by default and under --all
  fx_clk() { # fx_clk <file> <ab|ba>
    local f=$1 order=$2 c1 c2
    c1="mov.u32 	%r5, %clock;" c2="mov.u32 	%r6, %clock;"
    {
      printf '.version 8.0\n.target sm_86\n.visible .entry alpha(\n\t.param .u64 alpha_param_0\n)\n{\n'
      printf '\tld.param.u64 \t%%rd1, [alpha_param_0];\n\tmov.u32 \t%%r5, 0;\n\tmov.u32 \t%%r6, 0;\n\tmov.u32 \t%%r7, 0;\n'
      if [ "$order" = ab ]; then printf '\t%s\n\t%s\n' "$c1" "$c2"; else printf '\t%s\n\t%s\n' "$c2" "$c1"; fi
      printf '\tsub.s32 \t%%r7, %%r6, %%r5;\n\tst.global.u32 \t[%%rd1], %%r7;\n\tret;\n}\n.visible .entry beta(\n)\n{\n\tret;\n}\n'
    } > "$f"
  }
  fx "$d" "alpha=1 beta=2" "alpha=3 beta=2"
  fx_clk "$d/base/gx.ptx/mod1.ptx" ab
  fx_clk "$d/head/gx.ptx/mod1.ptx" ba
  judge "$d"
  [ "$rc" = 1 ] && grep -qF "ptx-canon: reordered-only (same line multiset); 1 swapped pair(s) in a 2-line window pass the opcode screen, but the value proof does not hold" <<< "$got" && grep -qF "counted as differs" <<< "$got" && grep -qF ": 1 identical, 0 reordered-only, 0 value-equal, 1 differs" <<< "$got"; ok "(o) md5 moved, a reorder past the opcode screen but not the value proof: rc 1, counted as differs" $?
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  fx_clk "$d/base/gx.ptx/mod1.ptx" ab
  fx_clk "$d/head/gx.ptx/mod1.ptx" ba
  judge_all "$d"
  [ "$rc" = 1 ] && grep -qF "pass the opcode screen, but the value proof does not hold" <<< "$got" && grep -qxF "canon-check: --all: 2 md5-equal entries read: 1 identical, 0 reordered-only, 0 value-equal, 1 differs" <<< "$got"; ok "(o) --all, md5 equal over the same reorder: counted differs, rc 1" $?
  # (p) --all leaves a module with no entry out of the pairs (ptx-canon --all refuses one) and still reads the others
  fx "$d" "alpha=1 beta=2" "alpha=1 beta=2"
  for side in base head; do printf '.version 8.0\n.target sm_86\n' > "$d/$side/gx.ptx/mod2.ptx"; done
  judge_all "$d"
  [ "$rc" = 0 ] && grep -qxF "canon-check: --all: 2 md5-equal entries read: 2 identical, 0 reordered-only, 0 value-equal, 0 differs" <<< "$got"; ok "(p) --all, a module with no entry beside the real one: left out, rc 0" $?
  # the whole command, in a throwaway tree beside a stub narrow-scan.sh (no box)
  w=$t/bloomery-cc
  mkdir -p "$w/tools/ref" "$t/home"
  cp "$0" "$w/tools/canon-check.sh"
  cp "$CANON" "$w/tools/ref/ptx-canon.py"
  cat > "$w/tools/narrow-scan.sh" << 'CANON_STUB_EOF'
#!/usr/bin/env bash
echo "narrow $*" >> "$STUB_LOG"
out='' prev=''
for a in "$@"; do
  [ "$prev" = --out ] && out=$a
  prev=$a
done
case " $* " in *" --dry-run "*) exit 0 ;; esac
if [ -n "${STUB_FIXTURE:-}" ] && [ -n "$out" ]; then mkdir -p "$out" && cp -R "$STUB_FIXTURE/." "$out/"; fi
exit "${STUB_RC:-0}"
CANON_STUB_EOF
  chmod +x "$w/tools/"*.sh
  git -C "$w" init -q && git -C "$w" config user.email test@test && git -C "$w" config user.name test && git -C "$w" config commit.gpgsign false
  echo base > "$w/marker.txt"
  git -C "$w" add -A && git -C "$w" commit -q -m base
  wb=$(git -C "$w" rev-parse HEAD)
  fx "$t/fixture" "alpha=1 beta=2" "alpha=3 beta=2" 1 2 ba 7 2
  # run_cc <stub log> <args...>: the run's output in $got, its code in $rc; the stub copies $t/fixture into the --out dir
  run_cc() {
    local log=$1
    shift
    : > "$log"
    got=$(HOME="$t/home" STUB_LOG="$log" STUB_FIXTURE="$t/fixture" bash "$w/tools/canon-check.sh" "$@" 2>&1)
    rc=$?
  }
  run_cc "$t/r1.log" "$wb" --out "$t/r1-out"
  [ "$rc" = 0 ] && grep -qx "narrow $wb --keep-ptx --scan-only --out $t/r1-out" "$t/r1.log" && grep -qF ": 1 identical, 1 reordered-only, 0 value-equal, 0 differs" <<< "$got"; ok "run: BASE goes to narrow-scan with --keep-ptx --scan-only and the --out dir, then the judgement" $?
  run_cc "$t/r2.log" "$wb" --worktree --out "$t/r2-out"
  [ "$rc" = 0 ] && grep -qx "narrow $wb --worktree --keep-ptx --scan-only --out $t/r2-out" "$t/r2.log"; ok "run: --worktree is passed on, with no HEAD" $?
  run_cc "$t/r3.log" "$wb" "$wb" --out "$t/r3-out"
  [ "$rc" = 0 ] && grep -qx "narrow $wb $wb --keep-ptx --scan-only --out $t/r3-out" "$t/r3.log"; ok "run: BASE HEAD are passed on" $?
  run_cc "$t/r4.log" "$wb"
  grep -qx "narrow $wb --keep-ptx --scan-only --out $t/home/.cache/bloomery/canon/bloomery-cc/${wb:0:8}-${wb:0:8}" "$t/r4.log"; ok "run: the default --out is ~/.cache/bloomery/canon/<tree>/<base8>-<head8>" $?
  run_cc "$t/r5.log" "$wb" --worktree
  grep -qx "narrow $wb --worktree --keep-ptx --scan-only --out $t/home/.cache/bloomery/canon/bloomery-cc/${wb:0:8}-worktree" "$t/r5.log"; ok "run: … and <base8>-worktree under --worktree" $?
  run_cc "$t/r6.log" "$wb" --dry-run --out "$t/r6-out"
  [ "$rc" = 0 ] && grep -qx "narrow $wb --keep-ptx --scan-only --out $t/r6-out --dry-run" "$t/r6.log" && ! grep -q 'canon-check: 1 bins' <<< "$got"; ok "run: --dry-run is passed on and judges nothing" $?
  STUB_RC=75 run_cc "$t/r7.log" "$wb" --out "$t/r7-out"
  [ "$rc" = 75 ] && grep -qF "contention" <<< "$got"; ok "run: narrow-scan's rc 75 (contention) is rc 75, named" $?
  STUB_RC=1 run_cc "$t/r8.log" "$wb" --out "$t/r8-out"
  [ "$rc" = 2 ] && grep -qF "the scans did not complete (narrow-scan rc 1)" <<< "$got" && ! grep -q 'canon-check: 1 bins' <<< "$got"; ok "run: a narrow-scan that failed is rc 2 by name, nothing judged" $?
  run_cc "$t/r9.log" no-such-rev-1234
  [ "$rc" = 2 ] && grep -qF "'no-such-rev-1234' is not a commit" <<< "$got" && [ ! -s "$t/r9.log" ]; ok "usage: an unknown BASE is rc 2 by name, narrow-scan not called" $?
  run_cc "$t/r10.log" "$wb" "$wb" --worktree
  [ "$rc" = 2 ] && grep -qF -- "--worktree takes no HEAD" <<< "$got"; ok "usage: a HEAD beside --worktree is rc 2 by name" $?
  run_cc "$t/r11.log" "$wb" --all --out "$t/r11-out"
  [ "$rc" = 0 ] && grep -qx "narrow $wb --keep-ptx --scan-only --out $t/r11-out" "$t/r11.log" && grep -qxF "canon-check: --all: 1 md5-equal entries read: 1 identical, 0 reordered-only, 0 value-equal, 0 differs" <<< "$got"; ok "run: --all is the judgement's, narrow-scan is called as without it" $?
  for args in "" "--bogus" "a b c" "--out" "--from" "--from $t/x extra" "--all"; do
    # shellcheck disable=SC2086
    got=$(bash "$0" $args 2>&1)
    rc=$?
    [ "$rc" = 2 ]; ok "usage: '$args' is rc 2" $?
  done
  rm -rf "$t"
  want=ok
  [ "$bad" = 0 ] || want=FAIL
  echo "canon-check: self-test $want ($n cases, $bad failed)"
  [ "$bad" = 0 ]
}

if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi

FROM='' BASE='' HEAD_REV='' OUT='' DRY=0 WORKTREE=0
while [ $# -gt 0 ]; do
  case $1 in
    --from)
      [ $# -ge 2 ] || die 2 "--from needs a directory; $USAGE"
      FROM=$2
      shift 2
      ;;
    --out)
      [ $# -ge 2 ] || die 2 "--out needs a directory; $USAGE"
      OUT=$2
      shift 2
      ;;
    --dry-run) DRY=1; shift ;;
    --worktree) WORKTREE=1; shift ;;
    --all) ALL=1; shift ;;
    -*) die 2 "'$1' is not a flag of this tool; $USAGE" ;;
    *)
      if [ -z "$BASE" ]; then BASE=$1
      elif [ -z "$HEAD_REV" ]; then HEAD_REV=$1
      else die 2 "'$1' after BASE and HEAD; $USAGE"
      fi
      shift
      ;;
  esac
done
if [ -n "$FROM" ]; then
  [ -z "$BASE$OUT" ] && [ "$DRY$WORKTREE" = 00 ] || die 2 "--from takes no BASE, --out, --dry-run or --worktree; $USAGE"
  judge_dir "$FROM"
  exit $?
fi
[ -n "$BASE" ] || die 2 "$USAGE"
[ "$WORKTREE" = 0 ] || [ -z "$HEAD_REV" ] || die 2 "--worktree takes no HEAD ('$HEAD_REV'): its head side is this tree's current content; $USAGE"
BASE_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "$BASE^{commit}") || die 2 "'$BASE' is not a commit of this repository"
if [ "$WORKTREE" = 1 ]; then
  HEAD_NAME=worktree
else
  HEAD_SHA=$(git -C "$ROOT" rev-parse --verify --quiet "${HEAD_REV:-HEAD}^{commit}") || die 2 "'${HEAD_REV:-HEAD}' is not a commit of this repository"
  HEAD_NAME=${HEAD_SHA:0:8}
fi
[ -n "$OUT" ] || OUT=$HOME/.cache/bloomery/canon/$(basename "$ROOT")/${BASE_SHA:0:8}-$HEAD_NAME
case $OUT in /*) ;; *) OUT=$PWD/$OUT ;; esac
cmd=("$ROOT/tools/narrow-scan.sh" "$BASE")
[ -z "$HEAD_REV" ] || cmd[${#cmd[@]}]=$HEAD_REV
[ "$WORKTREE" = 0 ] || cmd[${#cmd[@]}]=--worktree
cmd[${#cmd[@]}]=--keep-ptx
cmd[${#cmd[@]}]=--scan-only
cmd[${#cmd[@]}]=--out
cmd[${#cmd[@]}]=$OUT
[ "$DRY" = 0 ] || cmd[${#cmd[@]}]=--dry-run
"${cmd[@]}"
nrc=$?
case $nrc in
  0) ;;
  75) echo "canon-check: narrow-scan rc 75: held by contention (the box guard or the pair's lock), not a failure: run again" >&2; exit 75 ;;
  130 | 143) exit "$nrc" ;;
  *) die 2 "the scans did not complete (narrow-scan rc $nrc): nothing was judged" ;;
esac
[ "$DRY" = 0 ] || exit 0
judge_dir "$OUT"
exit $?
