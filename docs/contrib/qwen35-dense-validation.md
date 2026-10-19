# Dense Qwen forced-logit validation handoff

The native diagnostic extends the existing `generate_qwen3moe` loop over
`GpuModel<Body35>`. The HTTP checks exercise the existing Qwen seat and
`app::Session<Body35>` server path. `tools/ref/dense_logits_ref.cpp` independently reads the
same model with mainline llama.cpp. `dense-logits.py` builds the reference,
captures immutable input/build identities, runs either binary with bounds, and
compares completed raw rows. A `--kind reference` run on the pinned mainline tree
is the one writer of the registered reference family `logits-qwen38d`
(`crates/refset/src/arch/qwen35/dense.rs`, `just dump-logits-qwen38d`); `floor`
prints that oracle's own spread. The numerical acceptance band and the full A6000
serving gates remain separate work; a successful dump or a comparison without a
band is not a model correctness gate.

## Dependency and review boundary

The dense-generation parent [PR #339](https://github.com/midagedev/bloomery/pull/339)
is merged at `71713898d3f94527b87af2e96aff0afe4ed1f43e`. This follow-up is based on
upstream main `cca9beaef0180caf5fed247ac53a74617a868007`; it contains only the
original diagnostics commits and their compatibility/documentation updates.
It does not carry a duplicate frontend or a local replacement for the parent.

```sh
git diff cca9beaef0180caf5fed247ac53a74617a868007...HEAD
git log cca9beaef0180caf5fed247ac53a74617a868007..HEAD
```

The merged parent's typed dense shortage remains its context chooser's owner.
Host and CPU-reference diagnostics do not establish NVIDIA/27B acceptance; the
native eager/graph, reference and serving runs below remain the device lane.

## Host, static and reference smoke handoff

Run these on the exact source head being reviewed and record that identity;
earlier heads' results do not establish a later rebase:

```sh
RUSTFLAGS='-C target-cpu=native' cargo test --locked -p bloomery-gpu-gates --lib
python3 tools/ref/dense-logits.py --self-test
python3 tools/ref/dense-serve.py --self-test
python3 tools/recipes.py check
just contrib-check --tier t0
```

With the pinned CUDA headers and libclang available, frontend metadata checks
open no device. Use `cargo oxide`, not plain cargo, for linked device executables
and the context regression:

```sh
cargo check --locked -p bloomery-gpu-gates \
  --features glm5next,clef,vision,mimo2,gemma-embedding2 \
  --bin generate_qwen3moe --bin bloomery-serve
cargo oxide test --arch sm_86 -- -p bloomery-gpu-gates --features gpu \
  --bin generate_qwen3moe dense_shortage_halves_the_default_context
```

After building the reference below, this writer/input smoke uses existing
`prompt.ids` and `forced.ids`, opens no model or device and creates no raw output:

```sh
target/release/dense_logits_ref --validate-inputs --model /unused/model.gguf \
  --prompt-file /absolute/path/prompt.ids --forced-file /absolute/path/forced.ids \
  --out /unused/logits.f32 --ctx 4352 --prefill step
```

For an optional CPU writer smoke with the exact 0.8B file below, use a newly
built reference and a new output directory; this is not native CUDA evidence:

```sh
python3 tools/ref/dense-logits.py run --kind reference --backend cpu \
  --target qwen35-0.8b-candidate \
  --binary "$PWD/target/release/dense_logits_ref" \
  --model /absolute/path/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt /absolute/path/prompt.ids --forced /absolute/path/forced.ids \
  --ctx 4352 --ubatch 512 --prefill step --out /absolute/path/new-cpu-smoke
```

Retain malformed-input/nonfinite/provenance failure output alongside successful
smokes. Neither these commands nor a reference CPU dump establish dense A6000
logits, eager/graph equality, HTTP/native parity or full-model correctness.

## Model and row contract

The default target `qwen38-27b` is exactly
[`lmstudio-community/Qwen3.8-27B-GGUF`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/blob/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf),
revision `5a7da681f60570ab5b439a587e912d2e5eddb582`, file
`Qwen3.8-27B-Q4_K_M.gguf`, 16,810,714,336 bytes, SHA-256
`e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`.
It is dense `qwen35`, with 64 trunk layers and vocabulary248320; the 15 unused
next-token tensors do not enable MTP. The host coverage result still lists a
missing tool-call parser. The initial lane is plain-text generation with draft
off and F16 KV.

The optional `--target qwen35-0.8b-candidate` is a different, explicitly selected
file: [`notschmee/Qwen3.5-0.8B-GGUF`](https://huggingface.co/notschmee/Qwen3.5-0.8B-GGUF/blob/1870603ac8ab59c7ddb9d94cccf3b297e379538f/Qwen3.5-0.8B-Q4_K_M.gguf),
590,880,672 bytes, SHA-256
`be5b3cb8c2f3ebd791349c9c775ea38950f23a2b9e792ca2f0411597dd172988`.
It is a preparation/smoke target with Q8_0 tied embedding/head, not a replacement
for the 27B contribution target. The runner checks complete size and SHA before
opening either engine; another Q4_K_M file is refused.

Inputs are two nonempty UTF-8 files, one decimal token id per line, no headers or
blank lines. `prompt.ids` is the complete prompt. `forced.ids` has N continuation
ids. Row0 is the logit row after the complete prompt, predicting `forced[0]`;
row i>0 follows feeding `forced[i-1]`. The last forced id is recorded but is not
fed. No greedy replacement, tokenization or chat template is applied by these
tools. An out-of-vocabulary id is refused, not clamped.

`logits.0.f32` is `[N,248320]` IEEE754 f32, little-endian, contiguous by row. The
raw file's `.meta` sidecar is written after the last finite row and successful
flush (native) or close (reference). The runner verifies the sidecar's ids/shape,
then writes the set's `MANIFEST.tsv` (format `bloomery-forced-logits-v2`: the node dumps' `# model`, `# build` and
`# arch` lines and one `tensor` row, so a refset family reads it) with model SHA, prompt/continuation SHA, native and
reference commits, binary SHA, context, prefill schedule, KV and the raw-file digest. Only a successful
run publishes its files to an exclusively created output directory, with the
complete manifest last; existing output is refused. Failed command output and
exit status stay in the incomplete staging directory.
Every sidecar key must be unique: duplicate keys are refused even when its format,
dtype and completion markers are otherwise valid.
`tools/bloomery/manifest.py` is the manifest reader. Timeout cleanup is limited to
the launched process session, including the gate wrapper's separate timeout
process group; it never looks for or stops an existing model server.

## Build on the A6000 lane

Use a clean contribution checkout, pinned Rust/CUDA toolchain, and the exact model
file above. The following `build-native` command actually runs `cargo oxide` and
stamps both native binaries from that successful build. A hand-written stamp or
an old binary beside a new checkout is not accepted as build evidence.

```sh
python3 tools/ref/dense-logits.py --self-test
python3 tools/ref/dense-logits.py build-native --arch sm_86
```

The reference uses a clean official `ggml-org/llama.cpp` checkout and an existing
configured shared-library build. The harness has been compiled against commit
`fb4b2737a808a3fb7c2117a498f43815dc9be53e`. Pin the chosen commit and preserve its
CUDA build flags. If configuring a new build, use that checkout's documented
CMake invocation with `GGML_CUDA=ON` and `BUILD_SHARED_LIBS=ON`. The helper runs
the normal incremental `llama`/`ggml` build before linking, validates the source
directory and build-info commit, and records the SHA of the `libllama` and `libggml*`
libraries the harness loads. (The maintainers' box builds nothing in its oracle tree:
`just build-dense-logits-ref` passes `--pin <full commit>`, which proves the existing build
with `tools/ref/emb2/oracle-tree.sh` instead.)

```sh
python3 tools/ref/dense-logits.py build \
  --llama-tree /absolute/path/llama.cpp \
  --llama-build /absolute/path/llama.cpp/build \
  --out "$PWD/target/release/dense_logits_ref"
```

`--cached-libraries` is an explicit local smoke option. Its build record states
that cached-library hashes do not prove source/build correspondence. Use the
normal incremental build for the maintainer lane. Runtime prepends the recorded
library directory, checks the loader's resolved llama/ggml paths, refuses
`LD_PRELOAD`/`LD_AUDIT`, and rechecks library and binary hashes after the run.

## Fixed short and 4K inputs

Use a recorded corpus of token ids from this exact model's tokenizer. Prepare
64-token and 4096-token prompts, each with the following 32 ids as teacher-forced
continuation. Keep the original corpus path/hash in the handoff record. For a
corpus with at least 4128 ids, this preparation preserves literal ids:

```sh
python3 - /absolute/path/corpus.ids /absolute/path/dense-inputs <<'PY'
from pathlib import Path
import hashlib, itertools, json, sys
source, out = map(Path, sys.argv[1:])
with source.open() as stream:
    lines = list(itertools.islice(stream, 4128))
ids = [int(line.strip()) for line in lines]
assert len(ids) == 4128 and all(0 <= i < 248320 for i in ids)
out.mkdir()  # a new input directory
for p in (64, 4096):
    (out / f'p{p}.ids').write_text(''.join(f'{i}\n' for i in ids[:p]))
    (out / f'f{p}.ids').write_text(''.join(f'{i}\n' for i in ids[p:p+32]))
with source.open('rb') as stream:
    sha = hashlib.file_digest(stream, 'sha256').hexdigest()
(out / 'source.json').write_text(json.dumps({'path':str(source), 'sha256':sha})+'\n')
PY
```

## Capture and compare

Set `BLOOMERY_GATE_CARD=a6000` in the maintainer box environment. CUDA runs use
the existing `tools/gpu-gate.sh` lock and 900-second execution bound; binaries must
be in this checkout's `target/release`. No timing lease or throughput claim is
made by these functional captures. Keep the machine's GPU/driver line beside the
bundles. Commands below show the 64-token case; repeat with p4096/f4096 and new
output directories. Context 4352 holds either prompt plus 32 continuation ids.

```sh
MODEL=/absolute/path/Qwen3.8-27B-Q4_K_M.gguf
INPUT=/absolute/path/dense-inputs
OUT=/absolute/path/dense-validation-results
export BLOOMERY_GATE_CARD=a6000

python3 tools/ref/dense-logits.py run --kind reference \
  --binary "$PWD/target/release/dense_logits_ref" --model "$MODEL" \
  --prompt "$INPUT/p64.ids" --forced "$INPUT/f64.ids" \
  --prefill step --ctx 4352 --ubatch 512 --out "$OUT/ref-step-p64"

python3 tools/ref/dense-logits.py run --kind native \
  --binary "$PWD/target/release/generate_qwen3moe" --model "$MODEL" \
  --prompt "$INPUT/p64.ids" --forced "$INPUT/f64.ids" \
  --mode eager --prefill pass --ctx 4352 --ubatch 512 --out "$OUT/native-eager-p64"

python3 tools/ref/dense-logits.py run --kind native \
  --binary "$PWD/target/release/generate_qwen3moe" --model "$MODEL" \
  --prompt "$INPUT/p64.ids" --forced "$INPUT/f64.ids" \
  --mode graph --prefill pass --ctx 4352 --ubatch 512 --out "$OUT/native-graph-p64"

python3 tools/ref/dense-logits.py compare \
  "$OUT/native-eager-p64" "$OUT/native-graph-p64" --bitwise
python3 tools/ref/dense-logits.py compare \
  "$OUT/native-graph-p64" "$OUT/ref-step-p64"
```

The native `pass` schedule uses the existing Body35 passes, documented as the
one-token rule; this architecture does not accept `--prefill step`. For the
serving prompt cut, repeat reference with `--prefill batch` and native with
`--prefill auto`. Both feed the last prompt id separately, with the same ubatch
limit and subsequent forced ids. Record these as separate schedules.

`compare --bitwise` is a real equality check only for native eager versus graph,
with identical file/ids/build/binary/prefill/ubatch/context/KV/backend. A mismatch
returns 1. Ordinary `compare` checks completeness, digests, finite values and shape,
then reports per-row max absolute error, relative L2 and argmax. It intentionally
has no numeric pass band: derive one using the activation formats and reference
backend spread before making it a correctness gate. Reference CPU versus CUDA
is likewise a diagnostic floor, not equality. A zero norm is reported explicitly.
`python3 tools/ref/dense-logits.py floor <reference CUDA set>...` prints that floor for the
registered sets: per row the relative L2 of mainline's CUDA rows against the CPU twin
beside the set (`<set>.cpu`, `just dump-logits-qwen38d --cpu-twin`), the argmax agreement, the
top-2 margins and each row's derived stable margin, which a gate's band is derived from.

## Same-seat HTTP/CLI diagnostics

The serving runner uses the same stamped `generate_qwen3moe` and `bloomery-serve`
binaries built above, with the native `qwen3` seat. It first records CLI greedy
tokens, then starts its own loopback server on an ephemeral port under the
existing GPU gate lock. It preserves the repository's `/completion`, `/props`
and `/metrics` protocol. No existing server is selected or stopped.

```sh
python3 tools/ref/dense-serve.py --self-test
python3 tools/ref/dense-serve.py --model "$MODEL" \
  --short "$INPUT/p64.ids" --long "$INPUT/p4096.ids" \
  --out "$OUT/serve-single"
```

Defaults are 16 predicted tokens, context 4352 per slot, ubatch 512, native graph
mode, auto prefill with the CLI's last-prompt-token cut, f16 KV and draft off.
The short input accepts 1–256 literal ids; the long input requires exactly 4096.
The runner records inputs, binary/build/model identity, commands, responses,
placement and metrics. It compares fresh and repeated short/long requests,
reset after long prefill, and an edit at token 4064 with cached and fresh prefix
paths. Early EOS is retained as incomplete evidence even when all returned
tokens match the CLI prefix.

Add `--two-slot` with a new output directory to run resident-slot diagnostics
after the single-slot server has stopped. Alone and concurrent requests are
compared, and exported occupancy is recorded. Rounded occupancy is diagnostic
evidence; it does not become a new numerical gate or a throughput result.
Sampling, tool calls, decision seats and a full serving acceptance gate remain
outside this fixed-id greedy diagnostic.

## Remaining acceptance work

- The dedicated exact-file dense reference family `logits-qwen38d` is registered;
  its gate and band are still to come. Keep the existing Clef and routed Qwen
  sets intact. Preserve unsupported-tool-parser coverage until that optional
  feature has its own implementation and gate.
- Run the same-seat HTTP/CLI diagnostics at short and 4K prompts, then sampling,
  reset, recurrent checkpoint/prefix reuse and resident-slot isolation. The CLI
  comparison needs its existing `--last-step` cut.
- Preserve `/props` placement and current RoPE/slot memory accounting; a dense
  model cannot borrow the routed-expert host-offload gate's assumptions.
- Run the contribution's affected CUDA gates, including existing routed and
  decision seats. Host/static tests and the 0.8B CPU writer smoke cannot replace
  A6000 execution. Native AMD whole-forward/serving and measured comparisons
  remain later stages.
