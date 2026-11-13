#!/usr/bin/env bash
# The IQ2_XXS fixture of K-EXAONE-2.0, the file gate-gpu-exaone-verify's second run opens: the Q4_K fixture's cut
# (/models/fixtures/exaoneh, rules tools/fixture-cuts/exaoneh.json) quantized as the real second quant is, the routed gate and
# up IQ2_XXS over the engine's own importance matrix, the downs Q4_K, the trunk, head and embedding Q8_0 (the pick rules,
# tools/quant-rules/exaone-moe-pick.rules). Two steps, run from the repo root on the box with both binaries built:
#
#   cargo oxide build --arch "$(bash tools/arch.sh)" -- -p bloomery-gpu-gates --features flint --release --bin generate_exaone
#   cargo oxide build --arch "$(bash tools/arch.sh)" -- -p bloomery-gpu-gates --features gpu,hf --release --bin bloomery-quantize
#   BLOOMERY_MODEL=exaone-moe ./tools/box.sh 'bash tools/ref/exaone-iq2-fixture.sh [OUT_DIR]'
#
# 1. the walk of the Q4_K fixture over AGENTS.md (`generate_exaone --imatrix`: a card, under the gate lock) writes the
#    importance matrix, a file of a few tens of MB;
# 2. `bloomery-quantize --cpu` over the cut folder writes the one-shard GGUF (about 6.5 GB, minutes on the host's cores).
#
# The matrix is the fixture's own, from one short text: the file is a coverage fixture for the IQ2_XXS read paths (the card's routed
# kernels, the host tier's `iq2_xxs k=6144 path=fused`), not a quality reference. OUT_DIR is the directory the recipe reads
# (default /models/fixtures/exaoneh-iq2-gguf); an existing output is refused by the quantizer.
# Exit: 0 written; 2 a binary missing; 66 the cut folder or the Q4_K fixture missing.
set -euo pipefail
cut=/models/fixtures/exaoneh
q4k=/models/fixtures/exaoneh-gguf/exaoneh-gguf-fixture-00001-of-00001.gguf
out=${1:-/models/fixtures/exaoneh-iq2-gguf}
im=$out/exaoneh.imatrix.gguf
for f in "$cut" "$q4k"; do
  [ -e "$f" ] || { echo "exaone-iq2-fixture.sh: $f is missing (exit 66)" >&2; exit 66; }
done
for b in generate_exaone bloomery-quantize; do
  [ -x "target/release/$b" ] || { echo "exaone-iq2-fixture.sh: target/release/$b is not built: the build lines are in this file's header (exit 2)" >&2; exit 2; }
done
mkdir -p "$out"
BLOOMERY_REF_MODEL=$q4k bash tools/gpu-gate.sh generate_exaone --imatrix "$im" --text AGENTS.md --ctx 512 --arm a
CUDA_VISIBLE_DEVICES='' ./target/release/bloomery-quantize --hf "$cut" "$out/exaoneh-iq2-fixture" Q4_K_M \
  --imatrix "$im" --tensor-type-file tools/quant-rules/exaone-moe-pick.rules \
  --output-tensor-type q8_0 --token-embedding-type q8_0 --allow-requantize --keep-split --cpu
