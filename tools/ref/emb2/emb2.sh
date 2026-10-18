#!/usr/bin/env bash
# The EmbeddingGemma oracle sets (`just dump-ref-embedding MODEL [STEP...]`), under the selected model profile
# (tools/ref/models/ and its seat in tools/ref/clefvis/profile.sh), from the mainline tree LCPP at LCPP_PIN
# (`just build-emb2-lcpp`) and the dump_emb2 binary linked against it (`just build-embedding-ref MODEL`). Each step writes
# one set under $BLOOMERY_DATA with the profile's SETPFX. The text cases and their ids are the tokenizer oracle's
# ref-tokenizer-$EMBED_VOCAB set (`just dump-ref-tokenizer $EMBED_VOCAB`, crates/tokenizer/tools/oracle.sh):
# the shared cases and the profile's EMBED_CASES, tokenized by the same tree's llama-tokenize. EMBED_STEPS names the
# available steps and their order; a text-only profile has no tower steps. See README.md:
#
#   text       ${SETPFX}_text: dump_emb2 text over every case on the card
#   text.cpu   ${SETPFX}_text.cpu: the same with no card in view
#   tower      ${SETPFX}_tower: dump_emb2 tower over IMAGES on the card
#   tower.cpu  ${SETPFX}_tower.cpu: the same with no card in view
#   server.cpu ${SETPFX}_server.cpu: the same requests with both models on CPU
#   server     ${SETPFX}_server: llama-server's /v1/embeddings vectors of every text
#              case that is UTF-8 and every image (server_vectors.py), the server on 127.0.0.1:${EMB2_PORT:-8097} for the
#              step's length
#
#   bash tools/ref/emb2/emb2.sh [STEP...]     (no step: the profile's EMBED_STEPS)
#
# Before every step the tree is checked, not trusted: LCPP at LCPP_PIN with no tracked change, its llama-server loading that
# tree's libllama.so and built from that commit, the binary's .build record (tools/ref/clefvis/build-clefvis.sh) naming this
# tree's dump_emb2.cpp and set_writer.h and that commit, the record's CUDA architectures covering the visible card, and the model file and projector of the sha256 the
# profile pins. A step that reads the ids checks their manifest: complete, of this model, of this tree's llama-tokenize and the
# libllama.so it loads, and of the two cases files as they are now. Each set is written to <set>.staging and moved over the
# old set only when its writer exits 0, so a failed step leaves the old set whole: the old set waits as <set>.old while the
# new one takes its name, an exit or a caught signal in between moves it back, and a step that finds a <set>.old left by a
# killed run restores it first (or retains it with a timestamp, when the new set has its name already). The card steps (text, tower, server)
# go through tools/gpu-gate.sh, as clefvis.sh's do: the card's gate lock and the bound EMB2_BOUND (default 1500 s, parsed as
# tools/gate-bound.sh parses BLOOMERY_GATE_BOUND), for the call and for each step; the script links itself as
# target/release/emb2_dump and runs the card steps under that name inside the runner. The twins open no card
# (CUDA_VISIBLE_DEVICES empty), take no gate lock and run under the same bound through `timeout`. A functional oracle: no
# lease, no timing.
set -euo pipefail
REAL=$(readlink -f "${BASH_SOURCE[0]}")
HERE=$(cd "$(dirname "$REAL")/../../.." && pwd)
cd "$HERE"
# shellcheck source=tools/ref/ref-paths.sh
source tools/ref/ref-paths.sh
# shellcheck source=tools/ref/clefvis/profile.sh
source tools/ref/clefvis/profile.sh
[ "$FAMILY" = emb2 ] ||
  { echo "emb2: the sets need an EmbeddingGemma profile; this one is $MODEL_NAME" >&2; exit 2; }
: "${EMBED_VOCAB:?the profile names EMBED_VOCAB}"
: "${EMBED_CASES:?the profile names EMBED_CASES}"
: "${EMBED_CONTEXT:?the profile names EMBED_CONTEXT}"
: "${EMBED_STEPS[*]:?the profile names EMBED_STEPS}"
ALL=("${EMBED_STEPS[@]}")
steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=("${ALL[@]}")
for s in "${steps[@]}"; do
  [[ -n "$s" && "$s" != *[[:space:]]* && " ${ALL[*]} " == *" $s "* ]] || { echo "emb2: unknown step '$s' (${ALL[*]})" >&2; exit 64; }
done
selected() {
  local s
  for s in "${steps[@]}"; do [ "$s" != "$1" ] || return 0; done
  return 1
}
# shellcheck source=tools/gate-bound.sh
source tools/gate-bound.sh
# shellcheck source=tools/ref/emb2/oracle-tree.sh
source tools/ref/emb2/oracle-tree.sh
BLOOMERY_GATE_BOUND=${EMB2_BOUND-1500} gate_bound "emb2 (EMB2_BOUND)" || exit $?

# Outside the runner, a call with a card step: the card steps in one call through the runner, then the twins, each a call of
# its own.
card_steps=()
for s in text tower server video; do
  if selected "$s"; then card_steps+=("$s"); fi
done
if [ "$(basename "${BASH_SOURCE[0]}")" != emb2_dump ] && [ ${#card_steps[@]} -gt 0 ]; then
  mkdir -p target/release
  ln -sf "$REAL" target/release/emb2_dump
  # box.sh's BLOOMERY_CARD=a6000|both names the card; under its 3090 pin the runner takes whichever card is idle
  case "${BLOOMERY_BOX_CARD:-}" in
    a6000 | both) BLOOMERY_GATE_BOUND=$BOUND bash tools/gpu-gate.sh emb2_dump "${card_steps[@]}" ;;
    *) BLOOMERY_GATE_BOUND=$BOUND BLOOMERY_GATE_CARD=${BLOOMERY_GATE_CARD:-any} bash tools/gpu-gate.sh emb2_dump "${card_steps[@]}" ;;
  esac
  for s in text.cpu tower.cpu server.cpu video.cpu; do
    if selected "$s"; then bash "$REAL" "$s"; fi
  done
  exit 0
fi

export BLOOMERY_REF_BUILD=${LCPP_PIN:0:9}
IDS=$DATA/ref-tokenizer-$EMBED_VOCAB
TOKENIZE=$ORACLE/build/bin/llama-tokenize
SERVER_BIN=$ORACLE/build/bin/llama-server

die() { echo "emb2: $1" >&2; exit "${2:-2}"; }

# Every exit, a caught signal's included: the step's llama-server is stopped and reaped, and a set whose old copy waits as
# <set>.old with no new one in its place gets the old one back.
server_pid=
publishing=
cleanup() {
  bounded_bg_stop || true
  if [ -n "$publishing" ] && [ ! -e "$publishing" ] && [ -e "$publishing.old" ]; then
    mv "$publishing.old" "$publishing"
  fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

sha() { sha256sum "$1" | cut -d' ' -f1; }

# The tree, its build and the files, before every step.
verify() {
  local want
  oracle_tree --built-by 'just build-emb2-lcpp' emb2 "$ORACLE" "$LCPP_PIN" llama-server
  for want in "source_sha256 $(sha "$DUMPER")" "set_writer_sha256 $(sha tools/ref/clefvis/set_writer.h)" "lcpp_commit $BLOOMERY_REF_BUILD" "lcpp $(readlink -f "$ORACLE")"; do
    grep -Fqx "$want" "$BIN.build" 2> /dev/null || die "$BIN is not this tree's dump_emb2 at $BLOOMERY_REF_BUILD (no '$want' in $BIN.build; just build-emb2-ref)"
  done
  [ "$(sha "$MODEL")" = "$MODEL_SHA256" ] || die "$MODEL is not the file of sha256 $MODEL_SHA256 the profile pins"
  if [ -n "${MMPROJ:-}" ]; then
    [ "$(sha "$MMPROJ")" = "$MMPROJ_SHA256" ] || die "$MMPROJ is not the file of sha256 $MMPROJ_SHA256 the profile pins"
  fi
}

# The visible card, which the binary's CUDA architectures must cover; its name is the sets' `# device` card.
card() {
  local cc archs
  cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n 1)
  archs=$(sed -n 's/^cuda_architectures //p' "$BIN.build")
  case ";$archs;" in
    *";${cc/./};"* | *";${cc/./}a;"* | *";${cc/./}f;"*) ;;
    *) die "the card in view has compute capability $cc, and $BIN's tree is built for '$archs'" ;;
  esac
  nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}" | head -n 1
}

# The ids set is complete, of this model, of this tree's llama-tokenize and the libllama.so it loads (found as oracle.sh finds
# it), and of the two cases files as they are now.
ids_current() {
  local key field want got libllama
  [ -f "$IDS/MANIFEST.tsv" ] || die "no $IDS (just dump-ref-tokenizer $EMBED_VOCAB)"
  libllama=$(ldd "$TOKENIZE" | awk '$1 == "libllama.so" || $1 == "libllama.so.0" { print $3; exit }') || die "ldd cannot read $TOKENIZE"
  [ -f "$libllama" ] || die "$TOKENIZE loads no libllama.so (ldd names '$libllama')"
  grep -q $'^# complete\t' "$IDS/MANIFEST.tsv" || die "$IDS has no complete trailer (just dump-ref-tokenizer $EMBED_VOCAB)"
  while IFS=: read -r key field want; do
    got=$(awk -F'\t' -v k="# $key" -v f="$field" '$1 == k { print $f }' "$IDS/MANIFEST.tsv")
    [ "$got" = "$want" ] || die "$IDS is not the ids this tree makes: its $key is '$got', want '$want' (just dump-ref-tokenizer $EMBED_VOCAB)"
  done << EOF
vocabulary:2:$MODEL
tokenizer:2:$TOKENIZE
tokenizer:3:$(md5sum < "$TOKENIZE" | cut -d' ' -f1)
libllama:2:$libllama
libllama:3:$(md5sum < "$libllama" | cut -d' ' -f1)
cases:3:$(md5sum < crates/tokenizer/tests/cases.txt | cut -d' ' -f1)
extra cases:3:$(md5sum < "crates/tokenizer/tests/$EMBED_CASES" | cut -d' ' -f1)
EOF
}

# Replaced sets and incomplete staging directories stay beside the published set under a unique retained name.
retain() {
  if [ -e "$1" ]; then
    local kept=$1.retained.$(date -u +%Y%m%dT%H%M%SZ).$$
    mv "$1" "$kept"
    echo "emb2: retained $kept"
  fi
}

recover() { # <final>
  [ -e "$1.old" ] || return 0
  if [ -e "$1" ]; then
    retain "$1.old"
  else
    mv "$1.old" "$1"
    echo "emb2: restored $1 from $1.old"
  fi
}

publish() { # <staging> <final>: the old set stays whole until the new one has its name
  publishing=$2
  if [ -e "$2" ]; then mv "$2" "$2.old"; fi
  mv "$1" "$2"
  publishing=
  retain "$2.old"
  echo "emb2: $(basename "$2") -> $2"
}

case_args=()
while IFS= read -r id; do case_args+=(--case "$id"); done < <(awk -F'\t' '$1 ~ /^cases\// { print $1 }' "$IDS/MANIFEST.tsv" 2> /dev/null)
image_inputs() {
  local name
  image_args=()
  for name in "$@"; do
    [ -f "tools/ref/clefvis/images/$name.png" ] || die "no image tools/ref/clefvis/images/$name.png"
    image_args+=(--image "$name=tools/ref/clefvis/images/$name.png")
  done
}
image_inputs "${IMAGES[@]}"

server_step() {
  local stage=$1 card_name=$2 device=$3 kind=$4 port=${EMB2_PORT:-8097} up='' log flags media_args inputs
  local image_args=("${image_args[@]}")
  log=$DATA/${SETPFX}_$kind
  if [ "$device" = cpu ]; then log+=.cpu; fi
  log+=.log
  if (: < "/dev/tcp/127.0.0.1/$port") 2> /dev/null; then
    die "something already listens on 127.0.0.1:$port; stop it or set EMB2_PORT"
  fi
  flags=(-m "$MODEL" --embeddings --pooling mean -c "$EMBED_CONTEXT" -b "$EMBED_CONTEXT" -ub "$EMBED_CONTEXT" -np 1 -ngl 99
    --host 127.0.0.1 --port "$port" --no-webui)
  media_args=()
  inputs=("${case_args[@]}")
  if [ -n "${MMPROJ:-}" ]; then
    flags+=(--mmproj "$MMPROJ")
    if [ "$kind" = video ]; then
      : "${VIDEO_IMAGES[*]:?the profile names VIDEO_IMAGES}"
      image_inputs "${VIDEO_IMAGES[@]}"
    fi
    media_args=(--mmproj "$MMPROJ" --mmproj-sha256 "$MMPROJ_SHA256" "${image_args[@]}")
    if [ "$kind" = video ]; then
      flags+=(--image-max-tokens 140 --video-fps 1 --video-timestamp-interval 0)
      media_args+=(--frames)
      inputs=()
    else
      media_args+=(--mix-image free-448x448)
    fi
  fi
  local cpu=()
  if [ "$device" = cpu ]; then
    export CUDA_VISIBLE_DEVICES=
    flags+=(-ngl 0 --no-mmproj-offload)
    cpu=(--cpu)
  fi
  BG_OUT=$log bounded_bg "$BOUND" "$SERVER_BIN" "${flags[@]}"
  server_pid=$BG_PID
  for _ in $(seq 120); do
    kill -0 "$server_pid" 2> /dev/null || break
    curl -sf "http://127.0.0.1:$port/health" > /dev/null && { up=1; break; }
    sleep 1
  done
  [ -n "$up" ] || die "llama-server did not come up (log: $log)" 1
  bounded "$BOUND" python3 tools/ref/emb2/server_vectors.py --url "http://127.0.0.1:$port" --pid "$server_pid" \
    "${cpu[@]}" --model "$MODEL" --arch "$MODEL_NAME" --model-sha256 "$MODEL_SHA256" --card "$card_name" \
    --build-record "$BIN.build" --ids-dir "$IDS" "${inputs[@]}" --out "$stage" "${media_args[@]}" \
    --server-flags "${flags[*]}"
  bounded_bg_stop || die "llama-server outlived its cleanup budget (pid $server_pid)" 70
  server_pid=
}

mkdir -p "$DATA"
for s in "${ALL[@]}"; do
  selected "$s" || continue
  echo "emb2: $s"
  set_dir=$DATA/${SETPFX}_${s%.cpu}
  case $s in *.cpu) set_dir+=.cpu ;; esac
  exec {set_lock}>"$set_dir.lock"
  flock -w "$BOUND" "$set_lock" || die "set lock for $set_dir was not free within $BOUND s" 75
  stage=$set_dir.staging
  recover "$set_dir"
  verify
  retain "$stage"
  SECONDS=0
  case $s in
    text | text.cpu | tower | tower.cpu)
      if [ "${s#*.}" = cpu ]; then
        run=(env CUDA_VISIBLE_DEVICES=) where=(--cpu --card cpu)
      else
        card_name=$(card)
        run=(env) where=(--card "$card_name")
      fi
      if [ "${s%.cpu}" = text ]; then
        ids_current
        args=(text -m "$MODEL" --ids-dir "$IDS" "${case_args[@]}")
      else
        args=(tower -m "$MODEL" --mmproj "$MMPROJ" "${image_args[@]}")
      fi
      bounded "$BOUND" "${run[@]}" "$BIN" "${args[@]}" "${where[@]}" -t "$THREADS" --build-record "$BIN.build" \
        --out "$stage"
      ;;
    server | server.cpu | video | video.cpu)
      if [ "${s%.cpu}" = server ]; then ids_current; fi
      if [ "${s#*.}" = cpu ]; then card_name=cpu; device=cpu; else card_name=$(card); device=cuda; fi
      server_step "$stage" "$card_name" "$device" "${s%.cpu}"
      ;;
  esac
  publish "$stage" "$set_dir"
  echo "emb2: $s ($SECONDS s)"
  flock -u "$set_lock"
  exec {set_lock}>&-
done
