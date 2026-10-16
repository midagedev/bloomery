# Dense Qwen generation in the existing seat

Dense `qwen35` attaches to the existing Qwen generative seat and CUDA `Body35`.
There is no new seat, model loop, HTTP engine or libllama inference dependency.
The serving path remains `SeatEngine` -> `app::Session<Body35>` -> Bloomery's
native GPU model, with the common context/load choice, placement, resident slots,
prompt scheduling and checkpoint-based prefix reuse.

Use an explicit `bloomery-serve --model qwen3` for generation. With no model word,
the existing decision selection stays in force: a decision model's card or
in-file decision type selects its decision seat, and a bare backbone without a
head is refused by name. A generative word beside a decision head remains an
error. The CLI also refuses a dense file declaring `qwen35.decision.type`.

The CLI and serving planner select decode flash from `models::shape::gqa_row`.
The 27B shape (24 query heads, 4 KV heads, head width 256) uses the scalar pair
entry (`mma=false`); the existing group-of-eight and quad shapes keep their
tensor-core default. Plans and load/listening records name the actual file's
architecture. The planner reads that architecture from the file it already
holds, instead of taking a second, independently supplied architecture argument.
All its existing gate callers use this same owner. Dense generation serves no
draft; a requested draft is refused by name. MTP, media and mixed-IQ support are
outside this attach's verification scope.

## Exact model for NVIDIA validation

- HF repository: [`lmstudio-community/Qwen3.8-27B-GGUF`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF).
- Revision: `5a7da681f60570ab5b439a587e912d2e5eddb582`.
- Filename: [`Qwen3.8-27B-Q4_K_M.gguf`](https://huggingface.co/lmstudio-community/Qwen3.8-27B-GGUF/blob/5a7da681f60570ab5b439a587e912d2e5eddb582/Qwen3.8-27B-Q4_K_M.gguf).
- Size: 16,810,714,336 bytes.
- SHA-256: `e00082f779fa385cee8c68a3ec8833a75778cc87272240b942f74e0b8243e520`.
- Actual architecture: `qwen35`; 65 declared layers, 64 trunk layers, width
  5120, FFN width 17408, vocabulary 248320, no routed experts.
- Tensor inventory: F32 360, Q4_K 439, Q6_K 67; 866 total. The reader marks
  15 next-token tensors unused, so their presence does not enable MTP.

Q4_K_M is a quantization recipe, not a guarantee about every tensor. Candidate
headers from ggml-org, bartowski and unsloth contained additional types; none
is substituted for this file. The older local GSQ-RCO IQ3_S file has unsupported
per-site formats and is not the initial generation target. No requantization
or weight conversion is part of this change.

## Host admission and static checks

The host-only example uses the existing native Hparams, ModelSpec, roles and
coverage table, opens no card, and prints the actual tensor-type inventory:

```sh
RUSTFLAGS='-C target-cpu=native' cargo run --locked -p bloomery-gpu-gates \
  --example qwen35_admission -- /path/to/Qwen3.8-27B-Q4_K_M.gguf
```

This file has no unsupported tensor-format/role entries. The complete coverage
check still reports one missing tool-call parser for its chat template and
returns nonzero. That is an optional chat-feature gap, not a tensor-format
failure or a successful full admission. The example deliberately preserves
that result; no coverage guard, parser row, numerical band or lint is relaxed.
Text generation and structured tool calls must be validated separately.

Run contributor checks with `just contrib-check --tier t0`, plus host tests for
`bloomery-gpu-gates --lib` and `bloomery-model --lib`. Static `cargo check` and
`cargo clippy --no-deps` of the two frontends and their shared planner gate
callers need CUDA headers and libclang but open no device. Plain cargo must not
build an executable linking the CUDA device crates. The contributor machine
has an RX 7900 GRE, so these are host/static checks, not CUDA code generation,
model loading, generated-token or serving evidence.

## Maintainer A6000 lane

Use the exact file and SHA above for both Bloomery and the independent reference.
Build the native executables with `cargo oxide` and the repository's pinned
CUDA toolchain, then run the generated affected gate list. Preserve the existing
Qwen3/Qwen3.6 generation and decision gates. The existing `gate_qwen3_serve`
expects routed reference sets; it is not a dense-27B correctness result.

The dense clause still needs a reference dump from this file on the A6000 48GB,
forced-ID logits against that reference on identical weight bits, eager versus
graph checks, and HTTP versus native CLI `--last-step` at short and 4K prompts.
Then check sampling, reset, prefix/checkpoint reuse and resident slots. Keep
MTP, media and tool calls out of the first plain-text clause; the missing tool
parser remains listed until it is implemented and verified. Only after these
checks should the repository's timing runners compare native serving with
llama.cpp or Strata under matched bits, context, placement and concurrency.

This attach does not provide native AMD whole-model serving. The independent
HIP operator lab is not a backend for this seat.
