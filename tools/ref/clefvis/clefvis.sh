#!/usr/bin/env bash
# The image-input oracle sets of a Qwen3-VL family seat (`just dump-ref-clefvis` for Clef-Flash, `just dump-ref-qvis
# <qwen35moe|qwen4exp>` for Qwen3.6 and Qwen3.8): llama.cpp mainline's mtmd `qwen3vl_merger` tower and the seat's text model
# on the profile's file and bf16 mmproj (tools/ref/clefvis/profile.sh), written by dump_mtmd (tools/ref/clefvis/dump_mtmd.cpp)
# into $BLOOMERY_DATA/<set>/ in dump_ref's set format — the sets the refset families (crates/refset/src/arch/qwen35/clefvis.rs,
# arch/{qwen35moe,qwen4exp}/vis.rs) check and the image-input rounds read. SETPFX is the profile's set prefix.
#
#   <pfx>_preproc          A  the test images after mtmd's preprocess: the tower's graph input, f32 channel-planar (Clef and the
#                             GLM seat: PREPROC=1 in the profile)
#   <pfx>_taps             B  the tower's named graph nodes of blocks 0, 1, 13, 26 and the final embeddings; a Qwen seat taps
#                             the output end only (the post-LN output and the final embeddings of every image), the GLM seat
#                             its output nodes (vit_out, after_fc_proj, after_ffn_proj)
#   <pfx>_hidden_c{1,2,3}   C  result_norm of every position of three prompts and the positions fed (M-RoPE `mrope_pos`; the
#                             GLM seat's normal `pos`)
#   <pfx>_prose_c{1,2,3}   C' the same prompts with each image's rows replaced by prose-id text rows (Qwen seats)
#   <pfx>_bf16rows_c{1,2,3} C'' the same prompts with mainline's own tower rows rounded to bf16 (Qwen seats)
#   <pfx>_chatids_<case>   E  the ids llama-server's chat path gives each request, with its rendered prompt (Qwen seats)
#   <pfx>_decode_<case>    F  32 greedy decode steps after a prompt: id, result_norm, n_past (Qwen seats, cases with steps)
#   <set>.cpu                 the CPU twin of taps, hidden_*, prose_* and decode_* (no card in view, -ngl 0, THREADS threads)
# A and B come out of one tower run on the card (`preproc` and `taps` name the same job); the twin of B is `taps.cpu`. F comes
# out of C's run (the same context, after the prompt). The prompts' ids are `$REF/<case>.ids`, written by `--ids`: Clef's by the
# release's tokenizer and processor through tools/ref/clef_ref.py images --encode-only, a Qwen seat's by dump_mtmd chat, the
# llama-server path itself (tools/ref/qvis/requests.jsonl; each request's set is the chatids set); the cases and their images
# are the profile's CASES file.
#
#   clefvis.sh [--cpu-twin] [SET...]   the sets named by their base names (every set of the kind when none)
#   clefvis.sh --ids                   the ids of every request (CPU only, no card lock)
#   clefvis.sh --ref-d [--reps N]      set D: the release's answers to the requests on the A6000 (Clef only; BLOOMERY_CARD=a6000)
#
# The binary must be the one built from this tree's dump_mtmd.cpp (its .build record's source sha256) against the profile's
# ORACLE at its current commit (build-clefvis.sh), the mmproj must have the sha256 the family pins and the prose ids the
# profile's: either refused names what differs. Each set is written to <set>.staging and moved over the old set only when
# dump_mtmd exits 0 (its manifest's `# complete` trailer is then written). The runs go through tools/gpu-gate.sh, as lev-dump.sh's
# does: the card's gate lock, the bound (CLEFVIS_BOUND, default 1500 s, the whole call; CLEFVIS_RUN_BOUND the one dump's) and the
# card pick; no timing lease (a functional oracle). The script links itself as target/release/clefvis_dump and runs under that name
# inside the runner. `--ids` and `--cpu-twin` open no card (CUDA_VISIBLE_DEVICES empty, -ngl 0), so they take no gate lock, as
# hidden.sh's twin takes none; each run is still bounded by `timeout`. A model that does not fit the card (Qwen3.8) keeps the
# experts of NCMOE blocks on the host (profile.sh; QVIS_NCMOE_A6000 and QVIS_NCMOE_3090 move them). CLEFVIS_SETS_ROOT writes the sets
# (and the chat ids' staging) into another directory than $BLOOMERY_DATA, CLEFVIS_OUT reads the binary from another one (a build
# under test: build-clefvis.sh writes there too), QVIS_REF moves the ids: a re-dump compared with an installed set touches none.
# The GLM seat (BLOOMERY_MODEL=glm5next, BLOOMERY_TIER=fixture) dumps on the respelled copy of the fixture the profile names; its sets
# carry the fixture's `# fixture` line (refset-check --fixture-line, built here) and the copy's `# respell` line.
# MiniCPM-V 4.7 (`just dump-ref-mcvis`, BLOOMERY_MODEL=minicpmv47, FAMILY mcvis in profile.sh) is the canvas seat: sets A and B only,
# one row a tile (`<image>/t<k>/<node>`, dump_mtmd.cpp's header), the images of its own list (IMAGES_LIST), no case, no prompt arm and
# no ids (`--ids` is refused, exit 64); its refset family is crates/refset/src/arch/minicpmv4_7.
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
# shellcheck source=tools/ref/clefvis/profile.sh
source tools/ref/clefvis/profile.sh
IMG=$HERE/tools/ref/clefvis/images
OUT=${CLEFVIS_SETS_ROOT:-$DATA}
PY=${CLEFVIS_PYTHON:-/home/user/clef-venv/bin/python}
SNAPSHOT=${CLEFVIS_SNAPSHOT:-/models/clef-flash/hf}
REVISION=17f0b0ad64efb65d273590632833508766b2aae6
BOUND=${CLEFVIS_RUN_BOUND:-${CLEFVIS_BOUND:-1500}}
CTX=4096
UBATCH=512

ref_d() {
  [ "$FAMILY" = clef ] || { echo "clefvis.sh: set D is Clef's (the release's answers); this seat is $MODEL_NAME" >&2; exit 64; }
  "$PY" tools/ref/clef_ref.py images --snapshot "$SNAPSHOT" --revision "$REVISION" --requests tools/ref/clefvis/requests.jsonl \
    --images-dir "$IMG" --out "$REF" --preproc-set "$DATA/ref_clefvis_preproc" "$@"
}

# The binary is this tree's source at the profile's tree, the projector, prose ids and text model are the ones pinned.
check_inputs() {
  local want_src want_commit got
  want_src=$(sha256sum "$HERE/tools/ref/clefvis/dump_mtmd.cpp" | cut -d' ' -f1)
  want_commit=$(git -c safe.directory="$ORACLE" -C "$ORACLE" rev-parse --short=9 HEAD)
  if ! grep -qx "source_sha256 $want_src" "$BIN.build" 2> /dev/null || ! grep -qx "lcpp_commit $want_commit" "$BIN.build" 2> /dev/null; then
    echo "clefvis.sh: $BIN is not this tree's dump_mtmd at $ORACLE $want_commit (rebuild: just build-clefvis-ref / build-qvis-ref)" >&2
    exit 2
  fi
  got=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
  [ "$got" = "$MMPROJ_SHA256" ] || { echo "clefvis.sh: $MMPROJ has sha256 $got, the family pins $MMPROJ_SHA256" >&2; exit 2; }
  if [ -n "${PROSE_IDS:-}" ]; then
    got=$(sha256sum "$PROSE_IDS" | cut -d' ' -f1)
    [ "$got" = "$PROSE_SHA256" ] || { echo "clefvis.sh: $PROSE_IDS has sha256 $got, the profile pins $PROSE_SHA256" >&2; exit 2; }
  fi
  [ -f "$TEXT" ] || { echo "clefvis.sh: the text model $TEXT is not there" >&2; exit 2; }
  BUILD_COMMIT=$want_commit
}

# What the GLM seat's dumps are told beyond a Qwen seat's: the image token's id, and the fixture the text model is a copy of (its
# `# fixture` line, written by refset-check, and the file dump_mtmd holds the copy's bloomery.respell.source to). A Qwen seat's
# command line is as it was.
seat_args=()
tower_seat_args=()
fixture_args() {
  [ -z "$RESPELL_SOURCE" ] || tower_seat_args=(--fixture-line "$(fixture_line clefvis.sh)" --respell-source "$RESPELL_SOURCE")
  [ -z "$SEAT" ] || tower_seat_args+=(--seat "$SEAT")
  seat_args=("${tower_seat_args[@]}")
  [ -z "$IMAGE_PAD_ID" ] || seat_args+=(--image-pad-id "$IMAGE_PAD_ID")
}

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
all_images() { awk -F'\t' '!/^#/ && NF { print $1 }' "$IMAGES_LIST" ${IMAGES_EXTRA:+"$IMAGES_EXTRA"} | paste -sd, -; }

publish() { # <staging> <final>
  rm -rf "$2"
  mv "$1" "$2"
}
rc=0
fail() { echo "clefvis.sh: $1 failed (rc $2); the old set, if any, stays" >&2; rc=$2; }

# --ids. Clef: the release's tokenizer and processor. A Qwen seat: dump_mtmd chat, the llama-server path, which writes the
# chatids set and the ids file of every request; the set is moved into place, the ids to $REF/<case>.ids.
write_ids() {
  if [ "$FAMILY" = clef ]; then ref_d --encode-only "$@"; return $?; fi
  [ "$FAMILY" != mcvis ] || { echo "clefvis.sh: the $SEAT seat dumps the tower only (sets A and B): it has no chat ids" >&2; exit 64; }
  check_inputs
  fixture_args
  export BLOOMERY_REF_BUILD=$BUILD_COMMIT
  local stage=$OUT/qvis/$MODEL_NAME.ids.staging case_ set_ ok=1
  rm -rf "$stage"
  mkdir -p "$OUT/qvis" "$REF"
  set_images "$(all_images)"
  SECONDS=0
  if env CUDA_VISIBLE_DEVICES= timeout --kill-after=10 "$BOUND" "$BIN" chat --mmproj "$MMPROJ" -m "$TEXT" --mmproj-sha256 "$MMPROJ_SHA256" \
      --card cpu --requests "$REQUESTS" "${img_args[@]}" "${seat_args[@]}" --out "$stage" -t "$THREADS"; then
    while read -r case_; do
      set_=${SETPFX}_chatids_$case_
      [ -d "$stage/$case_" ] && [ -f "$stage/$case_.ids" ] || { echo "clefvis.sh: chat wrote no $case_" >&2; ok=0; continue; }
      cp "$stage/$case_.ids" "$REF/$case_.ids"
      publish "$stage/$case_" "$OUT/$set_"
      echo "clefvis.sh: $set_ -> $OUT/$set_ ($(wc -l < "$REF/$case_.ids") ids)"
    done < <(sed -n 's/^{"id":"\([^"]*\)".*/\1/p' "$REQUESTS")
    [ "$ok" = 1 ] || rc=1
    rm -rf "$stage"
    echo "clefvis.sh: chat ($SECONDS s)"
  else
    fail chat $?
  fi
  return "$rc"
}
case ${1:-} in
  --ids) shift; write_ids "$@"; exit $? ;;
  --ref-d) shift; ref_d "$@"; exit $? ;;
esac

twin=0
if [ "${1:-}" = --cpu-twin ]; then twin=1; shift; fi
check_inputs
fixture_args

selected() { # <base name>: no SET argument selects every set; else the base name must be one of them
  [ $# -eq 1 ] || return 2
  [ "${#SETS[@]}" -eq 0 ] && return 0
  printf '%s\n' "${SETS[@]}" | grep -qx "$1"
}
SETS=("$@")
if [ "${#SETS[@]}" -gt 0 ]; then
  if [ "$PREPROC" = 1 ]; then known=" ${SETPFX}_preproc ${SETPFX}_taps"; else known=" ${SETPFX}_taps"; fi
  while IFS=$'\t' read -r c _ steps _; do
    case $c in '' | '#'*) continue ;; esac
    for arm in $ARMS; do known+=" ${SETPFX}_${arm}_$c"; done
    [ "${steps:-0}" -le 0 ] || known+=" ${SETPFX}_decode_$c"
  done < "$CASES"
  for s in "${SETS[@]}"; do
    case "$known " in *" $s "*) ;; *) echo "clefvis.sh: no set named $s (known:$known)" >&2; exit 64 ;; esac
  done
fi

export BLOOMERY_REF_BUILD=$BUILD_COMMIT
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
  [ "$FAMILY" != mcvis ] || where+=(-t "$THREADS")
  if [ "$FAMILY" = qvis ]; then
    where+=(-t "$THREADS")
    ncmoe=$NCMOE_3090
    case $device in *A6000*) ncmoe=$NCMOE_A6000 ;; esac
    [ "$ncmoe" -le 0 ] || where+=(-ncmoe "$ncmoe")
  fi
fi

# The tower job: Clef's A (the card run only) and B, both from one run; a Qwen seat's B, the output end only. B's twin
# from the CPU run.
tower=0
if [ "$twin" = 1 ] || [ "$PREPROC" != 1 ]; then
  ! selected "${SETPFX}_taps" || tower=1
else
  ! { selected "${SETPFX}_preproc" || selected "${SETPFX}_taps"; } || tower=1
fi
if [ "$tower" = 1 ]; then
  set_images "$(all_images)"
  pre=$OUT/${SETPFX}_preproc
  taps=$OUT/${SETPFX}_taps$suffix
  rm -rf "$taps.staging" "$pre.staging"
  pre_args=() tap_args=()
  [ "$twin" = 1 ] || [ "$PREPROC" != 1 ] || pre_args=(--out-preproc "$pre.staging")
  [ "$FAMILY" != qvis ] || tap_args=(--taps final) # mcvis: every tile's full taps
  SECONDS=0
  if "${run[@]}" timeout --kill-after=10 "$BOUND" "$BIN" tower --mmproj "$MMPROJ" -m "$TEXT" --mmproj-sha256 "$MMPROJ_SHA256" \
      --card "$device" "${img_args[@]}" "${pre_args[@]}" "${tap_args[@]}" "${tower_seat_args[@]}" --out-taps "$taps.staging" "${where[@]}"; then
    [ "${#pre_args[@]}" -eq 0 ] || publish "$pre.staging" "$pre"
    publish "$taps.staging" "$taps"
    echo "clefvis.sh: tower -> $taps ($SECONDS s)"
  else
    fail tower $?
  fi
fi

# The prompt sets (a case with decode steps writes its decode set from the same run).
kinds=()
for arm in $ARMS; do
  case $arm in
    hidden) kinds+=(hidden:mtmd) ;;
    prose) kinds+=(prose:prose) ;;
    bf16rows) [ "$twin" = 1 ] || kinds+=(bf16rows:bf16) ;;
    *) echo "clefvis.sh: the profile's ARMS names '$arm' (hidden, prose or bf16rows)" >&2; exit 64 ;;
  esac
done
while IFS=$'\t' read -r c imgs steps _; do
  case $c in '' | '#'*) continue ;; esac
  for k in "${kinds[@]}"; do
    kind=${k%%:*}
    rows=${k#*:}
    name=${SETPFX}_${kind}_$c
    dec=0
    [ "$kind" != hidden ] || [ "$FAMILY" != qvis ] || dec=${steps:-0}
    if [ "$dec" -gt 0 ] && selected "${SETPFX}_decode_$c"; then :; else
      selected "$name" || continue
    fi
    ids=$REF/$c.ids
    [ -f "$ids" ] || { echo "clefvis.sh: no $ids (write it: just dump-ref-clefvis --ids / dump-ref-qvis <model> --ids)" >&2; rc=2; continue; }
    set_images "$imgs"
    dir=$OUT/$name$suffix
    rm -rf "$dir.staging"
    extra=("${seat_args[@]}")
    [ "$rows" != prose ] || extra+=(--prose-ids "$PROSE_IDS")
    ddir=$OUT/${SETPFX}_decode_$c$suffix
    rm -rf "$ddir.staging"
    [ "$dec" -le 0 ] || extra+=(--decode "$dec" --out-decode "$ddir.staging")
    [ "$FAMILY" != qvis ] || extra+=(--rows-from "$ROWS_FROM")
    SECONDS=0
    if "${run[@]}" timeout --kill-after=10 "$BOUND" "$BIN" hidden --mmproj "$MMPROJ" -m "$TEXT" --mmproj-sha256 "$MMPROJ_SHA256" \
        --card "$device" --ids "$ids" "${img_args[@]}" --rows "$rows" "${extra[@]}" --out "$dir.staging" \
        -c "$CTX" -ub "$UBATCH" "${where[@]}"; then
      publish "$dir.staging" "$dir"
      echo "clefvis.sh: $name$suffix -> $dir ($SECONDS s)"
      if [ "$dec" -gt 0 ]; then
        publish "$ddir.staging" "$ddir"
        echo "clefvis.sh: ${SETPFX}_decode_$c$suffix -> $ddir"
      fi
    else
      fail "$name$suffix" $?
    fi
  done
done < "$CASES"
exit "$rc"
