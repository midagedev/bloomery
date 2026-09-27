# shellcheck shell=bash
# The depth runners' fault witness, engine blocks and failed arms (depth-ds41.sh, depth-qwen3moe.sh): what
# the two share of the cold tag, of BLOOMERY_AB_ORDER=blocks and of their FAIL rows, one copy. Sourced;
# it defines constants and functions and runs nothing. What a runner's own engines decide stays in that
# runner: which block an arm belongs to (arm_block), how many token ids its process draws (arm_draws),
# the timed window's start (REF_MARK, the `fed` line), why an arm failed and the header text that
# explains them.
#
# The cold tag. A row carries `majflt <n>`, the change in /proc/vmstat pgmajfault across the arm's
# process, and `timed <n>`, the change over the row's measured window where the runner can see that
# window start (a line the engine prints at it: majflt_mark). The count (the timed one where there is
# one) is priced at COLD_US microseconds a fault, the serial cost of one 4 KB fault measured on this
# box (rig-log 2026-09-23: 75.4 µs, taken one after another on one thread) and the largest a fault has
# been measured at here; a row whose faults could have cost COLD_PCT percent of its timed window W or
# more (the ruler at four rounds) ends in ` [cold]`, and its column prints that bound. The counter is
# machine-wide; the lease keeps it to the arm's process. A runner calls majflt_require before its lease:
# a machine whose /proc/vmstat has no pgmajfault would read every count as 0.
#
# The blocks. Under BLOOMERY_AB_ORDER=blocks (ab_order) the arms run in engine blocks, the blocks in the
# order their first arms are given, each block's rounds together and its arms rotated by one slot a
# round; before a block's rounds one discard process (`DISCARD r0` row, in no mean) runs the block's arm
# that reads the most of what the others will: for a reference block the arm with the most token draws
# at the block's largest --n-cpu-moe (every arm's flags carrying one integer), for our engine's the
# longest prompt. Under rotate (the default) every arm runs once a round. depth-ds41.sh's header has the
# reasoning (Order, Discard).
#
# The failed arms. An arm that fails prints a FAIL row where its row would be (arm_fail) and the runner goes
# on; a counted arm's label drops out of the tables at that depth or P (failed_tally), and the runner ends
# with the list and exit 1 (failed_end). A warm-up's or a discard's failure is `FAIL r0 …`, in the list and
# dropping nothing. depth-ds41.sh's header has the contract (Failures).
#
# The runner defines, before arm_fail: ARM_FAIL_STEM, and CPU_BUSY_TAG and OTHER_BUSY_TAG (lease.sh,
# timing-card.sh). Before failed_tally: sums and pp_sums, its decode and prefill records `label|key|…`.
# Before blocks_plan: ARMS, A_KIND (ref for a reference arm), A_ENG, A_DEP, N,
# arm_block <i> (the block's name), arm_draws <i> (the ids arm <i>'s process takes), ref_cmd <engine>
# <dep> (REF_ARGS, honouring REF_K). Before blocks_dry: dry_cmd <i>, round_order <round> <i...>
# (ORDER_ARMS, ORDER_LOADS), ROUNDS, AB_WARMUP. Before blocks_run: run_unit <round> <i...> and run_round
# <round> <i...>.

# The cold tag's constants: microseconds one serial fault costs [measured, rig-log 2026-09-23], and the
# percent of a row's timed window at which the faults' price tags it.
COLD_US=75 COLD_PCT=1
# The word an arm's row starts with: ROW, a row of the tables; WARMUP, the discarded warm-up run; DISCARD,
# a block's discard. counted: whether the row goes into the sums and the row counts (only a ROW row does).
ROW_TAG=ROW
counted() { [ "$ROW_TAG" = ROW ]; }

# The failed arms (the runners' Failures): FAILED, one `r<r> <label> <d|p>=<key> rc=<rc>` each, and
# FAILED_KEYS, the `label|key` pairs of the counted ones, which the tables drop. The runner sets
# ARM_FAIL_STEM, the prefix of each failed arm's output file (its own name without `.sh`).
FAILED=() FAILED_KEYS=()
# arm_fail <round> <label> <d=|p=key> <rc> <why> [<output>]: the FAIL row in place of the arm's row; the
# output, when there is one, goes whole to a file (a loader's reason is many lines above its tail).
arm_fail() {
  local r=$1 label=$2 key=$3 rc=$4 why=$5 out=${6:-} f='' last
  if [ -n "$out" ]; then
    f=${TMPDIR:-/tmp}/$ARM_FAIL_STEM-${label//[^A-Za-z0-9_.=-]/_}-${key/=/}-r$r.log
    printf '%s\n' "$out" > "$f"
    last=$(printf '%s\n' "$out" | grep -v '^[[:space:]]*$' | tail -n 1)
    echo "$out" | tail -n 20 >&2
    [ -z "$last" ] || why="$why; last line: $last"
  fi
  echo "FAIL r$r $label $key rc=$rc | $why${f:+ | full output: $f}$CPU_BUSY_TAG$OTHER_BUSY_TAG"
  FAILED+=("r$r $label $key rc=$rc")
  counted || return 0
  FAILED_KEYS+=("$label|${key#*=}")
}
# drop_failed: the records `label|key|…` on stdin, less those whose `label|key` a counted arm failed at.
drop_failed() {
  awk -F'|' -v ex="$(printf '%s\n' "${FAILED_KEYS[@]}")" '
    BEGIN { n = split(ex, e, "\n"); for (i = 1; i <= n; i++) if (e[i] != "") x[e[i]] = 1 }
    !(($1 "|" $2) in x)'
}
# failed_tally: the closing summary's failed-arm count and, when a counted arm failed, the list of what
# drops, with those records removed from the runner's decode and prefill records (sums, pp_sums).
failed_tally() {
  echo "failed arms: ${#FAILED[@]} (FAIL rows, the warm-up's or the discards' included)"
  [ ${#FAILED_KEYS[@]} -gt 0 ] || return 0
  echo "=== dropped from the means and the ratios below, a failed arm each (the FAIL rows above) ==="
  printf '%s\n' "${FAILED_KEYS[@]}" | sort -u | awk -F'|' '{ printf "    dropped: %s at %s\n", $1, $2 }'
  [ ${#sums[@]} -eq 0 ] || mapfile -t sums < <(printf '%s\n' "${sums[@]}" | drop_failed)
  [ ${#pp_sums[@]} -eq 0 ] || mapfile -t pp_sums < <(printf '%s\n' "${pp_sums[@]}" | drop_failed)
}
# failed_end: the runner's last line when an arm failed, `failed arms: <each>; …`, and exit 1.
failed_end() {
  [ ${#FAILED[@]} -gt 0 ] || return 0
  echo "failed arms: $(printf '%s; ' "${FAILED[@]}")"
  exit 1
}

majflt_now() { awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat; }
# majflt_require <runner>: exit 2 before the lease when /proc/vmstat has no pgmajfault.
majflt_require() {
  [ -n "$(majflt_now 2> /dev/null)" ] && return 0
  echo "$1: /proc/vmstat gives no pgmajfault: every row's majflt column and cold tag would read 0" >&2
  exit 2
}
# majflt_mark <file> <ERE>: its stdin to stdout line by line, and /proc/vmstat's pgmajfault into <file>
# at the first line matching <ERE> (at none when <ERE> is empty): the line an engine prints as its
# measured window starts.
majflt_mark() {
  awk -v f="$1" -v re="$2" '!s && re != "" && $0 ~ re {
    while ((getline l < "/proc/vmstat") > 0) if (l ~ /^pgmajfault /) { sub(/^pgmajfault /, "", l); print l > f; close(f) }
    close("/proc/vmstat"); s = 1
  } { print; fflush() }'
}
# cold_check <faults> <window s>: MAJ_BOUND, faults × COLD_US as a percent of the window, and COLD_TAG
# (` [cold]` at COLD_PCT or more).
cold_check() {
  local c
  read -r MAJ_BOUND c < <(awk -v f="$1" -v w="$2" -v us="$COLD_US" -v t="$COLD_PCT" \
    'BEGIN { b = (w > 0) ? f * us / 1e4 / w : 1e9; printf "%.1f %d\n", b, (b >= t) }')
  COLD_TAG=
  [ "$c" = 0 ] || COLD_TAG=' [cold]'
}

# with_ncmoe <flags> <K>: the flags with their --n-cpu-moe (or -ncmoe) value replaced by K.
with_ncmoe() {
  local f
  f=$(echo " $1 " | sed -E "s/ (--n-cpu-moe|-ncmoe) [0-9]+ / /")
  echo "${f# }--n-cpu-moe $2"
}

# ab_order <runner>: BLOOMERY_AB_ORDER into ORDER, rotate (the default) or blocks; anything else exits 64.
ab_order() {
  ORDER=${BLOOMERY_AB_ORDER:-rotate}
  case $ORDER in
    rotate | blocks) ;;
    *) echo "$1: BLOOMERY_AB_ORDER is rotate (the default: every arm once a round, the order rotated) or blocks (the arms by engine, a discard process before each block), got '$ORDER'" >&2; exit 64 ;;
  esac
}

# arm_k <i>: a reference arm's --n-cpu-moe (or -ncmoe) when its flags carry one integer, else nothing.
arm_k() {
  local w prev='' k=''
  ref_cmd "${A_ENG[$1]}" "${A_DEP[$1]}"
  for w in "${REF_ARGS[@]}"; do
    case $prev in --n-cpu-moe | -ncmoe) k=$w ;; esac
    prev=$w
  done
  case $k in '' | *[!0-9]*) ;; *) echo "$k" ;; esac
}
# blocks_plan: the blocks, fixed before the lease. Per block b: BLK_KEY[b] its name, BLK_ARMS[b] its arms'
# indices in ARMS (space-separated, in the order given), BLK_DISC[b] the index of the arm its discard
# runs, BLK_K[b] the --n-cpu-moe that discard runs at when it is not the arm's own (empty otherwise),
# BLK_WHY[b] what the choice rests on.
BLK_KEY=() BLK_ARMS=() BLK_DISC=() BLK_K=() BLK_WHY=()
blocks_plan() {
  local i j b key best bestd kmax kall list d k
  for i in "${!ARMS[@]}"; do
    key=$(arm_block "$i") b=
    for j in "${!BLK_KEY[@]}"; do [ "${BLK_KEY[$j]}" != "$key" ] || b=$j; done
    if [ -z "$b" ]; then
      b=${#BLK_KEY[@]}
      BLK_KEY+=("$key") BLK_ARMS+=("")
    fi
    BLK_ARMS[b]="${BLK_ARMS[$b]:+${BLK_ARMS[$b]} }$i"
  done
  for b in "${!BLK_KEY[@]}"; do
    best='' bestd=-1 kmax='' kall=1 list=''
    for i in ${BLK_ARMS[$b]}; do
      d=$(arm_draws "$i")
      list="${list:+$list, }${ARMS[$i]} $d"
      if [ "$d" -gt "$bestd" ]; then best=$i bestd=$d; fi
      [ "${A_KIND[$i]}" = ref ] || continue
      k=$(arm_k "$i")
      if [ -z "$k" ]; then
        kall=0
      elif [ -z "$kmax" ] || [ "$k" -gt "$kmax" ]; then
        kmax=$k
      fi
    done
    BLK_DISC[b]=$best BLK_K[b]=''
    if [ "${A_KIND[$best]}" = ref ]; then
      BLK_WHY[b]="the most token draws of the block's arms ($list)"
      if [ "$kall" = 1 ] && [ "$kmax" != "$(arm_k "$best")" ]; then
        BLK_K[b]=$kmax
        BLK_WHY[b]="${BLK_WHY[$b]}, at the block's largest --n-cpu-moe"
      fi
    else
      BLK_WHY[b]="the longest prompt of the block's arms, $bestd ids"
    fi
  done
}
# block_line <b>: `<b>/<blocks> <name>: <arms>; discard <arm>…`, the block's plan in one line.
block_line() {
  local i arms=''
  for i in ${BLK_ARMS[$1]}; do arms="${arms:+$arms }${ARMS[$i]}"; done
  echo "$(($1 + 1))/${#BLK_KEY[@]} ${BLK_KEY[$1]}: $arms; discard ${ARMS[${BLK_DISC[$1]}]}${BLK_K[$1]:+ at --n-cpu-moe ${BLK_K[$1]}}, ${BLK_WHY[$1]}"
}
# blocks_config: the [config] lines of the order and, under blocks, of every block.
blocks_config() {
  local b
  echo "[config] order: $ORDER$([ "$ORDER" = rotate ] || echo ", a discard before each block: $([ "$AB_WARMUP" = 1 ] && echo on || echo 'off (BLOOMERY_AB_WARMUP=0)')")"
  for b in "${!BLK_KEY[@]}"; do echo "[config] block $(block_line "$b")"; done
}
# blocks_dry: a dry run's block section: the warm-up line, then per block its plan, its discard's command
# line and each round's order and loads.
blocks_dry() {
  local b r
  if [ "$AB_WARMUP" = 1 ]; then
    echo "[dry] warmup: the blocks' discards below, each row DISCARD r0 and in no mean, ratio or row count; the first block's discard is the lease's first process (BLOOMERY_AB_WARMUP=0 skips them)"
  else
    echo "[dry] warmup: off (BLOOMERY_AB_WARMUP=0): no discard; each block's round 1 first row follows the block before it, the first one the lease's first process"
  fi
  for b in "${!BLK_KEY[@]}"; do
    echo "[dry] block $(block_line "$b")"
    if [ "$AB_WARMUP" = 1 ]; then
      REF_K=${BLK_K[$b]}
      echo "[dry] block $((b + 1)) discard ${ARMS[${BLK_DISC[$b]}]}: $(dry_cmd "${BLK_DISC[$b]}")"
      REF_K=
    fi
    for r in $(seq "$ROUNDS"); do
      # shellcheck disable=SC2086 # the block's indices, one word each
      round_order "$r" ${BLK_ARMS[$b]}
      echo "[dry] block $((b + 1)) round $r order: $ORDER_ARMS"
      echo "[dry] block $((b + 1)) round $r loads: $ORDER_LOADS"
    done
  done
}
# blocks_run: every block: its [block] line, its discard (under BLOOMERY_AB_WARMUP=1), its rounds.
blocks_run() {
  local b r
  for b in "${!BLK_KEY[@]}"; do
    echo "[block] $(block_line "$b")"
    if [ "$AB_WARMUP" = 1 ]; then
      ROW_TAG=DISCARD REF_K=${BLK_K[$b]}
      run_unit 0 "${BLK_DISC[$b]}"
      ROW_TAG=ROW REF_K=
      echo "[discard] ${ARMS[${BLK_DISC[$b]}]}${BLK_K[$b]:+ at --n-cpu-moe ${BLK_K[$b]}} ran once before block $((b + 1))'s rounds and is discarded (the DISCARD or FAIL r0 row above)"
    fi
    for r in $(seq "$ROUNDS"); do
      # shellcheck disable=SC2086 # the block's indices, one word each
      run_round "$r" ${BLK_ARMS[$b]}
    done
  done
}

# `bash tools/ref/cold-blocks.sh --self-test`: the cold bound, the flag rewrite, the order's refusal and
# the block planner on fixed arms (just check-recipes runs it on the Mac; the runners' use of it is
# depth-ds41-stub.sh's and depth-qwen3moe-stub.sh's, on the box).
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ "${1:-}" = --self-test ]; then
  fails=0
  check() {
    if [ "$2" = "$3" ]; then echo "ok $1"; else echo "FAIL $1: got [$2], want [$3]"; fails=$((fails + 1)); fi
  }
  # 27 faults x 75 us = 2.025 ms of a 0.2 s window: 1.0 %, tagged; 26 is 0.975 %, printed 1.0 and not
  # tagged (the tag compares the unrounded bound); a window of 0 tags any count.
  cold_check 27 0.2
  check cold-at "$MAJ_BOUND|$COLD_TAG" "1.0| [cold]"
  cold_check 26 0.2
  check cold-under "$MAJ_BOUND|$COLD_TAG" "1.0|"
  cold_check 0 0.2
  check cold-zero "$MAJ_BOUND|$COLD_TAG" "0.0|"
  cold_check 0 0
  check cold-nowindow "$COLD_TAG" " [cold]"
  check ncmoe-long "$(with_ncmoe '-ngl 99 --n-cpu-moe 33 -fa on' 2)" "-ngl 99 -fa on --n-cpu-moe 2"
  check ncmoe-short "$(with_ncmoe '-ngl 99 -fa on -lzm off -ncmoe 26' 30)" "-ngl 99 -fa on -lzm off --n-cpu-moe 30"
  check ncmoe-none "$(with_ncmoe '-ngl 99' 4)" "-ngl 99 --n-cpu-moe 4"
  out=$(BLOOMERY_AB_ORDER=sideways ab_order runner.sh 2>&1) && r=0 || r=$?
  check order-bad "$r ${out%%(*}" "64 runner.sh: BLOOMERY_AB_ORDER is rotate "
  BLOOMERY_AB_ORDER=blocks ab_order runner.sh
  check order-blocks "$ORDER" blocks
  ab_order runner.sh
  check order-default "$ORDER" rotate
  counted && c1=y || c1=n
  ROW_TAG=DISCARD
  counted && c2=y || c2=n
  ROW_TAG=ROW
  check counted "$c1$c2" yn
  # The FAIL row and the failed list: a round's arm with its output (the file and the last line), a
  # discard's without one (in the list, dropping nothing), and the drop of a counted arm's label at its key.
  TMPDIR=$(mktemp -d) ARM_FAIL_STEM=runner CPU_BUSY_TAG='' OTHER_BUSY_TAG=' [other-busy]'
  row=$(arm_fail 1 lcpppp4096 p=4096 134 "no 'pp4096' row" "$(printf 'load\nggml_cuda_error: x\n\n')" 2> /dev/null)
  check fail-row "$row" "FAIL r1 lcpppp4096 p=4096 rc=134 | no 'pp4096' row; last line: ggml_cuda_error: x | full output: $TMPDIR/runner-lcpppp4096-p4096-r1.log [other-busy]"
  check fail-file "$(cat "$TMPDIR/runner-lcpppp4096-p4096-r1.log" 2> /dev/null | head -n 2 | paste -sd'|' -)" "load|ggml_cuda_error: x"
  arm_fail 1 lcpppp4096 p=4096 134 "no 'pp4096' row" > /dev/null
  ROW_TAG=DISCARD
  row=$(arm_fail 0 'ours@X=1' d=6 3 "exited 3")
  arm_fail 0 'ours@X=1' d=6 3 "exited 3" > /dev/null
  ROW_TAG=ROW
  check fail-discard "$row" "FAIL r0 ours@X=1 d=6 rc=3 | exited 3 [other-busy]"
  check fail-list "$(printf '%s; ' "${FAILED[@]}")|${FAILED_KEYS[*]}" "r1 lcpppp4096 p=4096 rc=134; r0 ours@X=1 d=6 rc=3; |lcpppp4096|4096"
  check fail-drop "$(printf 'lcpppp4096|4096|1|9\nlcpppp4096|512|1|9\nours|4096|1|9\n' | drop_failed | paste -sd';' -)" "lcpppp4096|512|1|9;ours|4096|1|9"
  rm -rf "$TMPDIR"
  FAILED=() FAILED_KEYS=()
  # The planner: ours 6 and 512 (prompts), a reference engine `x` at K 26 whose pp arm draws 2P, the same
  # engine's `xk` arm at K 30 in the same block, and a `y` engine with no K. Draws: x:6 -> 105, x:512 ->
  # 1024 (pp), xk:6 -> 105, y:4 -> 8.
  N=96
  ARMS=(6 x:6 512 xpp:512 y:4 xk:6)
  A_KIND=(ours ref ours ref ref ref) A_ENG=(ours x ours xpp y xk) A_DEP=(6 6 512 512 4 6)
  arm_block() { case ${A_ENG[$1]} in x*) echo x ;; y) echo y ;; *) echo ours ;; esac; }
  arm_draws() {
    case ${A_ENG[$1]} in xpp) echo $((2 * A_DEP[$1])) ;; x | xk) echo $((A_DEP[$1] + N + 3)) ;; y) echo $((2 * A_DEP[$1])) ;; *) echo "${A_DEP[$1]}" ;; esac
  }
  ref_cmd() {
    local f='-ngl 99 -ncmoe 26'
    case $1 in xk) f='-ngl 99 -ncmoe 30' ;; y) f='-ngl 99' ;; esac
    [ -z "${REF_K:-}" ] || f=$(with_ncmoe "$f" "$REF_K")
    read -r -a REF_ARGS <<< "-m m -p 0 $f"
  }
  blocks_plan
  check blk-keys "${BLK_KEY[*]}" "ours x y"
  check blk-arms "${BLK_ARMS[0]}|${BLK_ARMS[1]}|${BLK_ARMS[2]}" "0 2|1 3 5|4"
  check blk-line-ours "$(block_line 0)" "1/3 ours: 6 512; discard 512, the longest prompt of the block's arms, 512 ids"
  check blk-line-x "$(block_line 1)" "2/3 x: x:6 xpp:512 xk:6; discard xpp:512 at --n-cpu-moe 30, the most token draws of the block's arms (x:6 105, xpp:512 1024, xk:6 105), at the block's largest --n-cpu-moe"
  check blk-line-y "$(block_line 2)" "3/3 y: y:4; discard y:4, the most token draws of the block's arms (y:4 8)"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($fails failures)"
  [ "$fails" = 0 ]
  exit
fi
