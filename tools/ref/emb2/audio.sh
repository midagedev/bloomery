#!/usr/bin/env bash
# Dump the pinned audio tower from the existing BF16 projector and E9 mel set.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source tools/ref/models/gemma-embedding2.sh
DATA=${BLOOMERY_DATA:-/root/bloomery-data}
head=$(git -c safe.directory="$LCPP" -C "$LCPP" rev-parse HEAD)
[ "$head" = "$LCPP_PIN" ] || { echo "audio: llama.cpp is $head, want $LCPP_PIN" >&2; exit 2; }
status=$(git -c safe.directory="$LCPP" -C "$LCPP" status --porcelain --untracked-files=no)
[ -z "$status" ] || { echo "audio: llama.cpp has tracked changes" >&2; exit 2; }
printf '%s  %s\n' "$MMPROJ_SHA256" "$MMPROJ" | sha256sum -c -
OUT=$DATA/ref_gemma_embedding2_audio
for path in "$OUT" "$OUT.cpu"; do
    [ ! -e "$path" ] && [ ! -L "$path" ] || { echo "audio: $path exists; move it aside before dumping" >&2; exit 2; }
done
cargo run --release -p bloomery-refset --bin refset-check -- mel-gemma4a "$DATA/ref_gemma_embedding2_mel"
cmake -S "$LCPP" -B "$LCPP/build"
cmake --build "$LCPP/build" --clean-first --parallel 16
mkdir -p target "$DATA"
CXX=${CXX:-c++}
"$CXX" -std=c++17 -O2 -pthread -I"$LCPP/include" -I"$LCPP/ggml/include" -I"$LCPP/tools/mtmd" \
    tools/ref/emb2/dump_audio.cpp -L"$LCPP/build/bin" -Wl,-rpath,"$LCPP/build/bin" \
    -lmtmd -lllama -lggml -lggml-base -o target/dump_audio
scratch=$(mktemp -d "$DATA/.emb2-audio.XXXXXX")
echo "audio: staging set in $scratch"
PYTHONPATH=tools python3 - "$DATA/ref_gemma_embedding2_mel" > "$scratch/args" <<'PY'
import os
import sys
from bloomery.manifest import dump_file_name, read
manifest = read(sys.argv[1])
for case in ["minimum/0", "sine/0", "silence/0", "before-30s/0", "after-30s/0", "after-30s/1"]:
    rows = [row for row in manifest.rows("tensor") if row["name"] == case + "/mel"]
    if len(rows) != 1:
        sys.exit(f"audio: {case}: expected one mel row")
    row = rows[0]
    shape = [row.int(f"ne{i}") for i in range(4)]
    path = os.path.join(sys.argv[1], dump_file_name(case + "/mel", 0, "output", "contig", "f32"))
    if (row.int("occurrence") != 0 or row["type"] != "f32" or row["op"] != "NONE"
            or row.int("contig") != 1 or not 1 <= shape[0] <= 2999 or shape[1:] != [128, 1, 1]
            or row.int("bytes") != shape[0] * 128 * 4 or os.path.getsize(path) != row.int("bytes")):
        sys.exit(f"audio: {case}: malformed mel row or file")
    print(case)
    print(path)
PY
mapfile -t args < "$scratch/args"
for device in gpu cpu; do
    LD_LIBRARY_PATH="$LCPP/build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        BLOOMERY_REF_BUILD=${LCPP_PIN:0:9} target/dump_audio "$MMPROJ" "$scratch/$device" "$device" "${args[@]}"
done
cargo run --release -p bloomery-refset --bin refset-check -- tower-gemma4a "$scratch/gpu"
cargo run --release -p bloomery-refset --bin refset-check -- tower-gemma4a-cpu "$scratch/cpu"
for device in gpu cpu; do
    target=$OUT
    if [ "$device" = cpu ]; then target=$OUT.cpu; fi
    mv -Tn -- "$scratch/$device" "$target"
    [ ! -e "$scratch/$device" ] || { echo "audio: $target appeared while dumping; set not published" >&2; exit 2; }
done
echo "audio: published $OUT and CPU twin from $head"
