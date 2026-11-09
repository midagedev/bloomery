#!/usr/bin/env bash
# The card quantizer's rate on one real expert stack, under the machine-wide lease: for each type (IQ1_S, IQ2_XXS) the
# probe (`iq_quant_probe`, no --check) quantizes the whole stack on the timing card in batches of experts and prints its
# terms; this runner brackets the timed run with witness blocks and prints, per type, weights/s of the whole quantize
# call and its terms apart: the card's (widen, search, finish), the PCIe's (upload, download) and the host's (the
# bytes-to-words copy and the record pack). Lead-only. A prediction card is required
# (BLOOMERY_LEASE_CARD=docs/cards/quant-measure.card; `just quant-measure` sets it). On the box the timing card is the
# A6000 (timing-card.sh).
# Usage: quant-measure.sh <gguf file that holds the stack> [--tensor NAME] [--types iq1_s,iq2_xxs] [--batch N]
#        quant-measure.sh --self-test     (the arguments and the summary on a fixture of the probe's lines; no card, no lease)
# --tensor defaults to the file's first routed-expert stack (the probe names it); --types to both; --batch to 32 experts.
# Exit: 0 every type timed; 64 a usage error; 66 a missing file; the probe's own status when it fails.
set -euo pipefail

usage() { echo "usage: quant-measure.sh <gguf file> [--tensor NAME] [--types iq1_s,iq2_xxs] [--batch N] | --self-test" >&2; }

# parse_args <args…>: sets FILE, TENSOR, TYPES (comma list) and BATCH; returns 64 on a usage error, 66 on a missing file
# (unless QM_NO_FILE_CHECK=1, the self-test's).
parse_args() {
  FILE='' TENSOR='' TYPES='iq1_s,iq2_xxs' BATCH=32
  while [ $# -gt 0 ]; do
    case $1 in
      --tensor) [ $# -ge 2 ] || { usage; return 64; }; TENSOR=$2; shift 2 ;;
      --types) [ $# -ge 2 ] || { usage; return 64; }; TYPES=$2; shift 2 ;;
      --batch) [ $# -ge 2 ] || { usage; return 64; }; BATCH=$2; shift 2 ;;
      -*) echo "quant-measure: unknown argument $1" >&2; usage; return 64 ;;
      *) [ -z "$FILE" ] || { echo "quant-measure: two files named" >&2; usage; return 64; }; FILE=$1; shift ;;
    esac
  done
  [ -n "$FILE" ] || { usage; return 64; }
  case $BATCH in '' | *[!0-9]* | 0) echo "quant-measure: --batch $BATCH is not a positive integer" >&2; return 64 ;; esac
  local t
  for t in ${TYPES//,/ }; do
    case $t in iq1_s | iq2_xxs) ;; *) echo "quant-measure: --types $t: iq1_s or iq2_xxs" >&2; return 64 ;; esac
  done
  if [ "${QM_NO_FILE_CHECK:-0}" != 1 ] && [ ! -f "$FILE" ]; then
    echo "quant-measure: no file $FILE" >&2
    return 66
  fi
}

# summarize <type> <probe output file>: one line from the probe's `tensor`, `card quantize` and `terms` lines: weights/s of the
# whole call and the card, PCIe and host terms apart; rc 65 when a line is missing (a probe that changed its format).
summarize() {
  awk -v ty="$1" '
    /^tensor / { name = $2; dims = $4 " " $5 " " $6 " " $7 }
    /^card quantize: / {
      w = $3
      match($0, /= [0-9.e+]+ weights\/s/); rate = substr($0, RSTART + 2, RLENGTH - 2 - length(" weights/s"))
      match($0, /[0-9.]+ s wall/); wall = substr($0, RSTART, RLENGTH - length(" s wall"))
      have_q = 1
    }
    /^  terms: / {
      for (i = 1; i <= NF; i++) {
        if ($i == "words") hw = $(i + 1); if ($i == "upload") up = $(i + 1); if ($i == "widen") wi = $(i + 1)
        if ($i == "search") { se = $(i + 1); sr = $(i + 3) }
        if ($i == "finish") fi = $(i + 1); if ($i == "download") dn = $(i + 1); if ($i == "pack") pk = $(i + 1)
      }
      have_t = 1
    }
    END {
      if (!have_q || !have_t) { print "quant-measure: the probe printed no `card quantize` and `terms` lines for " ty > "/dev/stderr"; exit 65 }
      sub(/^\(/, "", sr)
      card = wi + se + fi; pcie = up + dn; host = hw + pk
      printf "quant-measure: %s %s: %s weights, wall %s s: quantize call %s weights/s; terms (s): card %.2f (widen %s, search %s = %s weights/s, finish %s), pcie %.2f (upload %s, download %s), host %.2f (words %s, pack %s)\n", ty, name, w, wall, rate, card, wi, se, sr, fi, pcie, up, dn, host, hw, pk
    }' "$2"
}

self_test() {
  local fails=0 t out want rc
  t=$(mktemp -d "${TMPDIR:-/tmp}/qm-test.XXXXXX")
  trap 'rm -rf "$t"' RETURN
  cat > "$t/probe.out" << 'FIX'
tensor blk.3.ffn_gate_exps.weight MXFP4 [3584, 3072, 896] (5.24 GB) -> Iq1S; importance: synthetic (seeded, positive)
card quantize: 9.865e9 weights in 14.9 s wall: read (page-in copy) 1.6 s, quantize call (H2D, three launches, D2H, pack) 13.3 s = 7.417e8 weights/s
  terms: host words 1.70 s, upload 0.50 s, widen 0.40 s, search 9.05 s (1.090e9 weights/s), finish 0.20 s, download 0.30 s, pack 0.50 s
FIX
  want='quant-measure: iq1_s blk.3.ffn_gate_exps.weight: 9.865e9 weights, wall 14.9 s: quantize call 7.417e8 weights/s; terms (s): card 9.65 (widen 0.40, search 9.05 = 1.090e9 weights/s, finish 0.20), pcie 0.80 (upload 0.50, download 0.30), host 2.20 (words 1.70, pack 0.50)'
  out=$(summarize iq1_s "$t/probe.out") || true
  if [ "$out" != "$want" ]; then echo "quant-measure self-test: summary differs:" >&2; echo "  got:  $out" >&2; echo "  want: $want" >&2; fails=$((fails + 1)); fi
  printf 'tensor x MXFP4 [1]\n' > "$t/bad.out"
  rc=0; summarize iq1_s "$t/bad.out" 2> /dev/null || rc=$?
  if [ "$rc" != 65 ]; then echo "quant-measure self-test: a probe with no timing lines is rc $rc, want 65" >&2; fails=$((fails + 1)); fi
  for args in '' '--types' '--batch 0 f' '--types q4_k f' '--bogus f' 'a b'; do
    # shellcheck disable=SC2086 # the words of the case are the arguments
    rc=0; (QM_NO_FILE_CHECK=1 parse_args $args) 2> /dev/null || rc=$?
    if [ "$rc" != 64 ]; then echo "quant-measure self-test: arguments '$args' are rc $rc, want 64" >&2; fails=$((fails + 1)); fi
  done
  rc=0; (parse_args "$t/no-such-file") 2> /dev/null || rc=$?
  if [ "$rc" != 66 ]; then echo "quant-measure self-test: a missing file is rc $rc, want 66" >&2; fails=$((fails + 1)); fi
  (parse_args --tensor blk.1.ffn_up_exps.weight --types iq2_xxs --batch 8 "$t/probe.out" && [ "$TENSOR $TYPES $BATCH" = 'blk.1.ffn_up_exps.weight iq2_xxs 8' ]) ||
    { echo "quant-measure self-test: a full command line was not read" >&2; fails=$((fails + 1)); }
  if [ "$fails" = 0 ]; then echo "quant-measure: self-test ok"; else return 1; fi
}

if [ "${1:-}" = --self-test ]; then
  [ $# = 1 ] || { usage; exit 64; }
  self_test
  exit $?
fi

parse_args "$@" || exit $?
TIMING_LOCAL_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
BIN=target/release/iq_quant_probe
assert_fresh_binary "$BIN" || exit $?
WITNESS=(head-open indent card gpu-apps binary busiest)
OUT=$(mktemp -d "${TMPDIR:-/tmp}/quant-measure.XXXXXX")
trap 'rm -rf "$OUT"' EXIT
timing_cards_before_lease
lease_take
timing_cards_start
witness pre
rc=0
for t in ${TYPES//,/ }; do
  args=("$FILE" --type "$t" --batch "$BATCH" --check none)
  [ -z "$TENSOR" ] || args+=(--tensor "$TENSOR")
  run_bounded timeout --kill-after=10 1500 "$BIN" "${args[@]}" > "$OUT/$t.out" 2>&1 || rc=$?
  cat "$OUT/$t.out"
  [ "$rc" = 0 ] || break
done
if ! timing_cards_arm ''; then
  echo "quant-measure: invalid witness: $TWOCARD_WHY" >&2
  rc=1
fi
witness post
if [ "$rc" = 0 ]; then
  for t in ${TYPES//,/ }; do summarize "$t" "$OUT/$t.out" || rc=$?; done
fi
echo "quant-measure $FILE types=$TYPES batch=$BATCH rc=$rc"
exit $rc
