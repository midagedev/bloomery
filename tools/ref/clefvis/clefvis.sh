#!/usr/bin/env bash
# The Clef-Flash image-input oracle sets (`just dump-ref-clefvis`): llama.cpp mainline's mtmd `qwen3vl_merger` tower and
# the qwen35 text model on Clef-Flash's published Q8_0 file and the bf16 mmproj, written by dump_mtmd
# (tools/ref/clefvis/dump_mtmd.cpp) into $BLOOMERY_DATA/<set>/ in dump_ref's set format — the sets the refset families
# clefvis-* (crates/refset/src/arch/qwen35/clefvis.rs) check and R1..R4 of the Clef image input read (design §7.1).
#
#   ref_clefvis_preproc          A  the 8 test images after mtmd's preprocess: the tower's graph input, f32 channel-planar
#   ref_clefvis_taps             B  the tower's named graph nodes of blocks 0, 1, 13, 26 and the final embeddings
#   ref_clefvis_hidden_c{1,2,3}   C  result_norm of every position of three Clef prompts and the M-RoPE positions fed
#   ref_clefvis_prose_c{1,2,3}   C' the same prompts with each image's rows replaced by prose-id text rows
#   ref_clefvis_bf16rows_c{1,2,3} C'' the same prompts with mainline's own tower rows rounded to bf16
#   <set>.cpu                    the CPU twin of taps, hidden_* and prose_* (no card in view, -ngl 0, 16 threads)
# A and B come out of one tower run on the card (`preproc` and `taps` name the same job); the twin of B is
# `taps.cpu`. The prompts' ids are `$CLEFVIS_REF/<case>.ids`, written by `clefvis.sh --ids` (the release's tokenizer and
# processor through tools/ref/clef_ref.py images --encode-only); the cases and their images are cases.tsv.
#
#   clefvis.sh [--cpu-twin] [SET...]   the sets named by their base names (every set of the kind when none)
#   clefvis.sh --ids                   the ids of every request (CPU only, no card lock)
#   clefvis.sh --ref-d [--reps N]      set D: the release's answers to the requests on the A6000 (BLOOMERY_CARD=a6000)
#
# The binary must be the one built from this tree's dump_mtmd.cpp (its .build record's source sha256) against the
# profile's LCPP at its current commit, the mmproj must have the sha256 the family pins and the prose ids the profile's:
# either refused names what differs. Each set is written to <set>.staging and moved over the old set only when dump_mtmd
# exits 0 (its manifest's `# complete` trailer is then written). The runs go through tools/gpu-gate.sh, as lev-dump.sh's
# does: the card's gate lock, the bound (CLEFVIS_BOUND, default 1500 s) and the card pick; no timing lease (a functional
# oracle). The script links itself as target/release/clefvis_dump and runs under that name inside the runner. `--ids`
# and `--cpu-twin` open no card (CUDA_VISIBLE_DEVICES empty, -ngl 0), so they take no gate lock, as hidden.sh's twin
# takes none; each run is still bounded by `timeout` (CLEFVIS_BOUND).
set -euo pipefail
REAL=$(readlink -f "${BASH_SOURCE[0]}")
HERE=$(cd "$(dirname "$REAL")/../../.." && pwd)
cd "$HERE"
if [ "$(basename "${BASH_SOURCE[0]}")" != clefvis_dump ] && [ "${1:-}" != --ids ] && [ "${1:-}" != --cpu-twin ]; then
  mkdir -p target/release
  ln -sf ../../tools/ref/clefvis/clefvis.sh target/release/clefvis_dump
  # box.sh's BLOOMERY_CARD=a6000|both names the card; under its 3090 pin the runner takes whichever card is idle
  case "${BLOOMERY_BOX_CARD:-}" in
    a6000 | both) BLOOMERY_GATE_BOUND=${CLEFVIS_BOUND:-1500} bash tools/gpu-gate.sh clefvis_dump "$@" ;;
    *) BLOOMERY_GATE_BOUND=${CLEFVIS_BOUND:-1500} BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh clefvis_dump "$@" ;;
  esac
  exit $?
fi
# shellcheck source=tools/ref/ref-paths.sh
source tools/ref/ref-paths.sh
[ "$MODEL_NAME" = qwen35 ] ||
  { echo "clefvis.sh: the clefvis sets are the qwen35 profile's; this command picked $MODEL_NAME" >&2; exit 2; }
DATA=$BLOOMERY_DATA
BIN=$DATA/bin/dump_mtmd
IMG=$HERE/tools/ref/clefvis/images
REF=${CLEFVIS_REF:-$DATA/clefvis/ref}
MMPROJ=/root/models/clef-flash/mmproj-Cloudflare_clef-flash-bf16.gguf
MMPROJ_SHA256=3c45b34aee6f353a0d41d6b96ba712a498a17a82f0a6152bf68a5021f9652c0f
TEXT=$FLASH_Q8
PY=${CLEFVIS_PYTHON:-/home/user/clef-venv/bin/python}
SNAPSHOT=${CLEFVIS_SNAPSHOT:-/models/clef-flash/hf}
REVISION=17f0b0ad64efb65d273590632833508766b2aae6
BOUND=${CLEFVIS_BOUND:-1500}
CTX=4096
UBATCH=512
THREADS=16

ref_d() {
  "$PY" tools/ref/clef_ref.py images --snapshot "$SNAPSHOT" --revision "$REVISION" --requests tools/ref/clefvis/requests.jsonl \
    --images-dir "$IMG" --out "$REF" --preproc-set "$DATA/ref_clefvis_preproc" "$@"
}
case ${1:-} in
  --ids) shift; ref_d --encode-only "$@"; exit $? ;;
  --ref-d) shift; ref_d "$@"; exit $? ;;
esac

twin=0
if [ "${1:-}" = --cpu-twin ]; then twin=1; shift; fi
want_src=$(sha256sum "$HERE/tools/ref/clefvis/dump_mtmd.cpp" | cut -d' ' -f1)
want_commit=$(git -c safe.directory="$LCPP" -C "$LCPP" rev-parse --short=9 HEAD)
if ! grep -qx "source_sha256 $want_src" "$BIN.build" 2> /dev/null || ! grep -qx "lcpp_commit $want_commit" "$BIN.build" 2> /dev/null; then
  echo "clefvis.sh: $BIN is not this tree's dump_mtmd at $LCPP $want_commit (rebuild: just build-clefvis-ref)" >&2
  exit 2
fi
got=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
[ "$got" = "$MMPROJ_SHA256" ] || { echo "clefvis.sh: $MMPROJ has sha256 $got, the family pins $MMPROJ_SHA256" >&2; exit 2; }
got=$(sha256sum "$PROSE_IDS" | cut -d' ' -f1)
[ "$got" = "$PROSE_SHA256" ] || { echo "clefvis.sh: $PROSE_IDS has sha256 $got, the profile pins $PROSE_SHA256" >&2; exit 2; }
[ -f "$TEXT" ] || { echo "clefvis.sh: the text model $TEXT is not there" >&2; exit 2; }

selected() { # <base name>: no SET argument selects every set; else the base name must be one of them
  [ $# -eq 1 ] || return 2
  [ "${#SETS[@]}" -eq 0 ] && return 0
  printf '%s\n' "${SETS[@]}" | grep -qx "$1"
}
SETS=("$@")
if [ "${#SETS[@]}" -gt 0 ]; then
  known=" ref_clefvis_preproc ref_clefvis_taps"
  while IFS=$'\t' read -r c _; do
    case $c in '' | '#'*) continue ;; esac
    known+=" ref_clefvis_hidden_$c ref_clefvis_prose_$c ref_clefvis_bf16rows_$c"
  done < tools/ref/clefvis/cases.tsv
  for s in "${SETS[@]}"; do
    case "$known " in *" $s "*) ;; *) echo "clefvis.sh: no set named $s (known:$known)" >&2; exit 64 ;; esac
  done
fi

export BLOOMERY_REF_BUILD=$want_commit
if [ "$twin" = 1 ]; then
  device=cpu
  where=(--cpu -t "$THREADS")
  run=(env CUDA_VISIBLE_DEVICES=)
  suffix=.cpu
else
  device=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n 1)
  where=(-ngl 99)
  run=(env)
  suffix=
fi
rc=0
publish() { # <staging> <final>
  rm -rf "$2"
  mv "$1" "$2"
}
fail() { echo "clefvis.sh: $1 failed (rc $2); the old set, if any, stays" >&2; rc=$2; }

img_args=() # <name,name,…>: --image <name>=<png> for each
set_images() {
  img_args=()
  local n
  IFS=, read -ra names <<< "$1"
  for n in "${names[@]}"; do
    [ -f "$IMG/$n.png" ] || { echo "clefvis.sh: no image $IMG/$n.png" >&2; exit 2; }
    img_args+=(--image "$n=$IMG/$n.png")
  done
}

# The tower job: A (the card run only) and B, both from one run; B's twin from the CPU run.
tower=0
if [ "$twin" = 1 ]; then
  ! selected ref_clefvis_taps || tower=1
else
  ! { selected ref_clefvis_preproc || selected ref_clefvis_taps; } || tower=1
fi
if [ "$tower" = 1 ]; then
  set_images "$(awk -F'\t' '!/^#/ && NF { print $1 }' tools/ref/clefvis/images.tsv | paste -sd, -)"
  pre=$DATA/ref_clefvis_preproc
  taps=$DATA/ref_clefvis_taps$suffix
  rm -rf "$taps.staging" "$pre.staging"
  pre_args=()
  [ "$twin" = 1 ] || pre_args=(--out-preproc "$pre.staging")
  SECONDS=0
  if "${run[@]}" timeout --kill-after=10 "$BOUND" "$BIN" tower --mmproj "$MMPROJ" -m "$TEXT" --mmproj-sha256 "$MMPROJ_SHA256" \
      --card "$device" "${img_args[@]}" "${pre_args[@]}" --out-taps "$taps.staging" "${where[@]}"; then
    [ "$twin" = 1 ] || publish "$pre.staging" "$pre"
    publish "$taps.staging" "$taps"
    echo "clefvis.sh: tower -> $taps ($SECONDS s)"
  else
    fail tower $?
  fi
fi

# The prompt sets.
kinds=(hidden:mtmd prose:prose)
[ "$twin" = 1 ] || kinds+=(bf16rows:bf16)
while IFS=$'\t' read -r c imgs _; do
  case $c in '' | '#'*) continue ;; esac
  for k in "${kinds[@]}"; do
    kind=${k%%:*}
    rows=${k#*:}
    name=ref_clefvis_${kind}_$c
    selected "$name" || continue
    ids=$REF/$c.ids
    [ -f "$ids" ] || { echo "clefvis.sh: no $ids (write it: just dump-ref-clefvis --ids)" >&2; rc=2; continue; }
    set_images "$imgs"
    dir=$DATA/$name$suffix
    rm -rf "$dir.staging"
    extra=()
    [ "$rows" != prose ] || extra=(--prose-ids "$PROSE_IDS")
    SECONDS=0
    if "${run[@]}" timeout --kill-after=10 "$BOUND" "$BIN" hidden --mmproj "$MMPROJ" -m "$TEXT" --mmproj-sha256 "$MMPROJ_SHA256" \
        --card "$device" --ids "$ids" "${img_args[@]}" --rows "$rows" "${extra[@]}" --out "$dir.staging" \
        -c "$CTX" -ub "$UBATCH" "${where[@]}"; then
      publish "$dir.staging" "$dir"
      echo "clefvis.sh: $name$suffix -> $dir ($SECONDS s)"
    else
      fail "$name$suffix" $?
    fi
  done
done < tools/ref/clefvis/cases.tsv
exit "$rc"
