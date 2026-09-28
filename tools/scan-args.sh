#!/usr/bin/env bash
# The scan recipes' argument check (ptx-scan, sass-scan, lds-scan), on the Mac before the box builds
# anything. Blocks one failure: a cargo feature given where the recipe takes the scan's own words. A
# feature after the binary (`just ptx-scan generate_ds41 gpu,deepseek41`) lands in the recipe's ARGS, the
# build runs at the default `--features gpu`, and the track's working binary is overwritten by a
# host-only build (or cargo refuses it for its required-features). Features go through --features.
#
#   tools/scan-args.sh <ptx|sass|lds> <binary> [word...]   # the recipe's words after the binary
#   tools/scan-args.sh --self-test
#
# Refused by name, exit 2, the message naming the word and the recipe's right form:
#   a word that names a cargo feature of crates/gpu-gates, or a comma list of them;
#   a comma anywhere in the entry substring (the first word that is not a flag): entry names have none;
#   a flag the scan script does not take (ptx and lds take none; sass takes --exact, --raw and
#   --step <HEAD>);
#   more words than the scan script takes (ptx and lds: one entry substring; sass: the entry and its
#   decisions, which are a comma list of t/n or ADDR=t|n and are not refused for their commas).
# A feature table it cannot read ends it with 2 too: an unread table would pass every word.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)

# features <Cargo.toml>: the [features] table's keys, one a line; non-zero when the file does not read.
features() {
  awk '/^\[/ { f = ($0 == "[features]"); next } f && $1 != "" && $1 !~ /^#/ { print $1 }' "$1"
}

# check <scan> <binary> <features, one a line> <word...>: 0 when the words are the scan's, else 2 and the
# reason on stderr.
check() {
  local scan=$1 bin=$2 feats=$3 w why pos=0 max flags form item all
  shift 3
  case $scan in
    ptx) max=1 flags='' form="just ptx-scan $bin --features <features> [entry substring]" ;;
    lds) max=1 flags='' form="just lds-scan $bin --features <features> [entry substring]" ;;
    sass) max=2 flags=' --exact --raw --step ' form="just sass-scan $bin --features <features> [-- --exact|--raw|--step HEAD] [entry substring [decisions|list]]" ;;
    *) echo "scan-args.sh: the scan is ptx, sass or lds, got '$scan'" >&2; return 64 ;;
  esac
  while [ $# -gt 0 ]; do
    w=$1 why=''
    shift
    case $w in
      -*)
        if [[ $flags == *" $w "* ]]; then
          [ "$w" != --step ] || shift
          continue
        fi
        if [ -n "$flags" ]; then why="tools/$scan-scan.sh takes no flag but${flags% }"; else why="tools/$scan-scan.sh takes no flag"; fi
        ;;
      *)
        pos=$((pos + 1))
        if grep -qxF -- "$w" <<< "$feats"; then
          why="a cargo feature of crates/gpu-gates (an entry filter by this name needs a longer substring)"
        elif [[ $w == *,* ]]; then
          all=1
          IFS=, read -r -a items <<< "$w"
          for item in "${items[@]}"; do grep -qxF -- "$item" <<< "$feats" || all=0; done
          if [ "$all" = 1 ]; then
            why="a comma list of cargo features"
          elif [ "$pos" = 1 ]; then
            why="a comma list is a features list (an entry name has no comma)"
          fi
        fi
        [ -n "$why" ] || [ "$pos" -le "$max" ] || why="tools/$scan-scan.sh takes $([ "$max" = 1 ] && echo "one entry substring" || echo "an entry substring and its decisions")"
        ;;
    esac
    if [ -n "$why" ]; then
      echo "$scan-scan: '$w' after $bin: $why; features go through --features ($form)" >&2
      return 2
    fi
  done
}

self_test() {
  local feats n=0 bad=0 name want rc
  feats=$(printf '%s\n' default gpu deepseek41 qwen3moe)
  # case_ <name> <want rc> <scan> <word...>
  case_() {
    name=$1 want=$2
    shift 2
    rc=0
    check "$1" gate_x "$feats" "${@:2}" 2> /dev/null || rc=$?
    n=$((n + 1))
    if [ "$rc" = "$want" ]; then echo "ok $name"; else bad=$((bad + 1)); echo "FAIL $name: rc $rc, want $want"; fi
  }
  case_ ptx-none 0 ptx
  case_ ptx-entry 0 ptx gemm_q4k
  case_ ptx-list 2 ptx gpu,deepseek41
  case_ ptx-feature 2 ptx deepseek41
  case_ ptx-flag 2 ptx --features
  case_ ptx-two 2 ptx gemm flash
  case_ ptx-comma 2 ptx gemm,flash
  case_ lds-entry 0 lds hc_gated_up_mix
  case_ lds-list 2 lds gpu,deepseek41
  case_ lds-feature 2 lds qwen3moe
  case_ lds-two 2 lds hc_gated x
  case_ sass-entry 0 sass gemm_q4k
  case_ sass-decisions 0 sass gemm_q4k n,t,t,n
  case_ sass-list-word 0 sass gemm_q4k list
  case_ sass-exact 0 sass --exact ds41_attn_seg
  case_ sass-step 0 sass --exact --step 0x15f0 gemm_q4k 0x1cf0=t,0x2370=t
  case_ sass-raw 0 sass --raw gemm_q4k
  case_ sass-features-list 2 sass gpu,deepseek41
  case_ sass-feature 2 sass deepseek41
  case_ sass-feature-second 2 sass gemm_q4k deepseek41
  case_ sass-features-second 2 sass gemm_q4k gpu,deepseek41
  case_ sass-flag 2 sass --features deepseek41
  case_ sass-three 2 sass gemm_q4k n,t extra
  case_ sass-entry-comma 2 sass gemm,flash
  case_ scan-unknown 64 cubin gemm
  # The real table: every feature crates/gpu-gates declares is refused as a word, and `gpu` is one.
  feats=$(features "$ROOT/crates/gpu-gates/Cargo.toml") || { echo "FAIL real-table: crates/gpu-gates/Cargo.toml does not read"; bad=$((bad + 1)); }
  n=$((n + 1))
  if grep -qxF gpu <<< "$feats" && grep -qxF deepseek41 <<< "$feats"; then echo "ok real-table"; else bad=$((bad + 1)); echo "FAIL real-table: gpu or deepseek41 not among [$(echo "$feats" | tr '\n' ' ')]"; fi
  echo "scan-args: self-test $([ "$bad" = 0 ] && echo ok || echo FAIL) ($n cases, $bad failed)"
  [ "$bad" = 0 ]
}

if [ "${1:-}" = --self-test ]; then
  self_test
  exit
fi
[ $# -ge 2 ] || { echo "usage: scan-args.sh <ptx|sass|lds> <binary> [word...] | --self-test" >&2; exit 64; }
FEATS=$(features "$ROOT/crates/gpu-gates/Cargo.toml") && [ -n "$FEATS" ] || {
  echo "$1-scan: cannot read crates/gpu-gates/Cargo.toml for its feature names" >&2
  exit 2
}
check "$1" "$2" "$FEATS" "${@:3}"
