<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-dark.svg">
    <img src="docs/assets/logo.svg" width="128" alt="bloomery">
  </picture>
</p>

# bloomery

Hybrid GPU + CPU inference for mixture-of-experts models, written in Rust down to the CUDA kernels.

bloomery runs mixture-of-experts models that do not fit on one GPU. Each routed layer keeps as many of its experts on the card as fit, and the rest run on the CPU inside the same decode step. The host code is Rust, the GPU kernels are CUDA written in Rust ([cuda-oxide](https://github.com/NVlabs/cuda-oxide)), and the CPU expert kernels are AVX2 Rust. Today it runs DeepSeek-V4.1-Flash on one GPU with an eight-channel CPU and 256 GB of RAM (see [Target hardware](#target-hardware)).

## Models

The files below are the ones the gates and the numbers use; each is byte-identical (sha256) to the upload linked.

| Model | File | Runs as | Checked by |
|---|---|---|---|
| [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | [`Q3_K_M`, 9 shards](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF) (vcruz305) | one GPU + CPU experts: `generate_ds41`, `bloomery-chat`, `bloomery-serve-ds41` | per-op reference gates; step, skewed-pass, rollback and prompt = steps bit gates; load, draft, chat and server gates |
| [Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | [`Q4_K_M`](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) (unsloth) | the whole model on one GPU: `generate_qwen3moe` | per-kernel gates and an end-to-end gate (greedy tokens, PPL/KLD against the reference) |
| [DeepSeek-V2-Lite-Chat](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat) | [`Q3_K_M`](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF) (mradermacher) | CPU (`bloomery-decode`) and GPU | the engine's first model; every subsystem gate |

The DSpark draft for V4.1 speculative decoding is not a public upload. It is converted from the draft tensors in DeepSeek's V4.1 checkpoint with the converter in llama.cpp's V4.1 pull request, and then its `dflash.target_layers` is rewritten from [38, 39, 40] to [37, 38, 39]: the converter keeps V4's value, which is one layer off for V4.1. It is not the file in [JigSawPT's DSpark upload](https://huggingface.co/JigSawPT/DeepSeek-V4.1-Flash-DSpark-GGUF).

In progress: [DeepSeek-V4-Flash-0731](https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF).

## What it does

V4.1-Flash does not fit on one consumer card, so each decode step is split across the machine:

- **Experts are split between the card and the host.** Each routed layer keeps some of its experts on the GPU. The rest run on an AVX2 host tier inside the same captured CUDA graph step. A router-frequency list (`BLOOMERY_HOT_LIST`) can pick which experts stay on the card.
- **engram rows come from NVMe.** V4.1's n-gram lookup table does not fit in RAM beside the host experts, so it stays in a memory-mapped file. Each token reads 48 rows; a helper thread asks the kernel for all of them first, then copies them.
- **A skewed two-row pass.** Two positions run one layer apart in one step, so the card's dense work hides behind the host leg. It is the verification pass for speculative decoding.
- **Speculative decoding with the DSpark draft** (`BLOOMERY_DRAFT=dspark`). The draft runs on a second card and proposes one token; the skewed pass checks it. Greedy output is the same with and without the draft; a gate checks that token for token.
- **An n-gram lookup draft** (`BLOOMERY_DRAFT=lookup`) proposes the next token from the text so far, with the same guarantee.

A prompt does not run one decode step per token:

- **It runs in batches of up to 512 positions**, and leaves the model bit for bit in the state that one decode step per token would leave. Each layer runs only at the positions a later reader needs.
- **Each expert's work is batched.** A host expert computes all the batch's tokens routed to it in one pass. A card expert reads its weights once for every tile of up to eight of its tokens.
- **Two batches go through the layers together** (`BLOOMERY_PREFILL_GROUP`, default 2), so the card routes the next batch while the CPU runs this one's host experts.

## Status

**It runs on the public `Q3_K_M` GGUF as uploaded**, and the maintainers' gates run on that file by default ([`docs/BUILD.md`](docs/BUILD.md#the-model-file)). You can run `generate_ds41` (token ids in, greedy ids out), `bloomery-chat` (text in, streamed text out, sampled or greedy) and `bloomery-serve-ds41` (llama-server's HTTP API with streaming, prompt-prefix reuse, reasoning and tool calls; one request at a time). The tokenizer is bit-identical to `llama-tokenize` on its gate's corpora.

Three reference sets were made from the engine's first file (the same upload with attention, shared experts and the engram tables in Q8_0), and they are being made again from the public file: ik_llama.cpp's greedy tokens, the PPL/KLD baseline and the DSpark draft's intermediate dumps. The DSpark graph gate stays red until its set is remade.

In progress: a timed run on one RTX 3090; faster V4.1 prompts on the CPU side; DeepSeek-V4-Flash-0731.

As far as we know (2026-09-24), this is the only Rust engine that runs DeepSeek-V4.1-Flash with CPU expert offloading, with the GPU kernels also written in Rust. If you know of another one, please open an issue.

## Measured numbers

Built for Ampere GPUs (sm_86) and AVX2 CPUs. Every number is a single stream on one RTX A6000 (48 GB, 300 W) in a 32-core AVX2 workstation, instrumentation off, one fresh process per arm, measured under the quiet-machine protocol below. Decode rows generate `n = 96` tokens. Each table comes from one window with its arms alternated, and a ratio is only taken between arms of the same window.

### DeepSeek-V4.1-Flash, public `Q3_K_M` — one GPU plus CPU experts

Placement plan (a): every layer and the head on the card, 2,668 routed experts on the card, the other 12,692 on the host.

**Decode.** The router-ranked hot list picks the card experts. The prompt is the first 512 tokens of a prose or a code corpus, and the DSpark draft runs on the workstation's RTX 3090. Three rounds, rig-log [2026-09-25, dspark-loop-tps](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#dspark-loop-tps):

| Prompt | plain tok/s | DSpark tok/s | DSpark / plain |
|---|---:|---:|---:|
| prose 512 | 44.8 | **51.3** | 1.146 ± 0.014 |
| code 512 | 43.8 | **50.6** | 1.156 ± 0.062 |

The same placement decodes at 39.6 tok/s after a 4096-token prose prompt (plain; 2026-09-26, [cardtile-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#cardtile-ab)). With the card budget of a 24 GB card, emulated on the A6000 (1,171 card experts), prose 512 decodes at 35.8 tok/s (plain; 2026-09-24, [public-q3km-prose-code-and-budget](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#public-q3km-prose-code-and-budget)).

Against llama.cpp's V4.1 pull request ([#28696](https://github.com/ggml-org/llama.cpp/pull/28696) at `5210c7c`, `--n-cpu-moe 33`), in one window with three rounds: on synthetic prompt ids at depth 6, with no hot list, bloomery decodes at 29.7 tok/s and llama.cpp at 21.5, 1.38 ± 0.19. llama.cpp's first round read 11 % below its other two, which is most of that interval (rig-log [2026-09-25, launch-thread-lever-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#launch-thread-lever-ab)).

**Prompt processing** (pp tok/s over a prompt of P tokens):

| Prompt | Card experts by | P = 512 | P = 4096 | Measured |
|---|---|---:|---:|---|
| synthetic ids | id order (no hot list) | 144.6 | **247.5** | 2026-09-26; main at `0bcee2c` (512) and `e690f54` (4096) |
| prose corpus | hot list | **216.1** | 292.1 | 2026-09-26; main at `efc202f`, before the paired batches |

Sources: rig-log [b1-pp-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#b1-pp-ab), [prefillgroup-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#prefillgroup-ab) and [cardtile-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#cardtile-ab). The 247.5 is the clean round of two. The other round read 235.2, because its row was the window's first run and its prompt planning read the file cold (0.7 s). Later changes to the tree are not expected to move the 144.6 [derived: at 512 synthetic tokens the card work already runs under the CPU's]. llama.cpp's V4.1 branch (`-ngl 999 --n-cpu-moe 33 -fa on -t 32 -nopo 1`, `llama-bench -p P -n 0`, which also feeds random ids) ran at 104.6 tok/s at P = 512 and 103.8 at P = 4096 on 2026-09-25 ([P = 512](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#v41-prefill-baseline-p512), [P = 4096](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#v41-prefill-baseline-p4096)). That was a different window from ours, so no ratio is given here; a table from one window is next.

### Qwen3-30B-A3B-Instruct-2507, `Q4_K_M` — the whole model on the GPU

**Decode** against mainline llama.cpp (`53ed051ce`, `llama-bench -ngl 99 -fa on`), arms alternated in one window, four rounds (rig-log [2026-09-24, Qwen3-30B-A3B](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#qwen3-30b-a3b-e28)):

| Depth | bloomery tok/s | llama.cpp tok/s | bloomery / llama.cpp |
|---:|---:|---:|---:|
| 6 | **206.0** | 193.7 | 1.063 ± 0.004 |
| 1024 | **194.4** | 188.9 | 1.029 ± 0.002 |
| 4096 | 160.7 | **173.1** | 0.929 ± 0.006 |

In that window bloomery was ahead at short context and behind at 4096 keys, where its attention was slower. In a later window with no reference decode arms, bloomery alone decoded at 207.5 tok/s at depth 512 and 175.9 at depth 4096 (2026-09-26, [mrs-flash](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#mrs-flash)); a table with llama.cpp and mistral.rs in one window is next.

**Prompt processing** on the same card, arms alternated in one window, three rounds (`llama-bench -p P -n 0`; mistral.rs `d5ae0f18f` built with `--features "cuda flash-attn"`, its recommended build, `bench --prompt-len P --gen-len 1`; rig-log [2026-09-26, mrs-flash](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#mrs-flash)):

| Prompt | bloomery pp tok/s | llama.cpp | llama.cpp `-ub 4096` | mistral.rs |
|---:|---:|---:|---:|---:|
| 512 | **6,821** | 4,285 | — | 4,835 |
| 4096 | **8,009** | 4,184 | 6,860 | 7,890 |

The prompt runs in ubatches of up to 4096 tokens (a load-time size, `BLOOMERY_QWEN3_UBATCH`) through a grouped int8 tensor-core GEMM, each expert read once per ubatch, and a prefill attention kernel that stages each 64-key tile once for 64 query rows and runs both products on the tensor cores. The router's logits for a ubatch run as register tiles, 32 tokens by 32 experts a block. In this window, at 4096 tokens, bloomery's default is 1.167× llama.cpp with `-ub 4096 -b 4096` and 1.015 ± 0.009× mistral.rs, about equal (1.41× at 512 tokens). Since then `gemm_q4k` runs its step in fewer instructions (`ccdd3dc`): 7,706 tok/s at 512 and 8,818 at 4096 against the previous binary's 6,803 and 7,955 in one window (1.133 ± 0.005 and 1.108 ± 0.009; no reference rows in that window; rig-log [q3gemmb-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#q3gemmb-ab)). Before that change, at 4096 tokens, nsys put the GEMMs at 59 % of 510 ms of kernels, prefill attention at 17 % and the per-token kernels at the remaining 23 % ([q3router-ab](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#q3router-ab)).

### Read these numbers with their conditions

- **Rows are plain decoding unless they say DSpark.** The DSpark rows use two cards: the model on the A6000 and the draft on the RTX 3090.
- **The mistral.rs prefill column is its recommended build** (`flash-attn` on, as upstream's release builds and install script build it on Ampere). The rows we published before 2026-09-26, 3,460 and 1,317 tok/s, came from a build without `flash-attn`, whose prompt attention took the eager P × P path ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#mrs-noflash)).
- **The card is an A6000.** The 24 GB row emulates a 3090's budget on the A6000; a run on a real 3090 has not been timed.
- **The V4.1 hot list was built from routing traces of the same corpora** the prose and code prompts come from, so those rows are its favorable case. With synthetic prompt ids, whose output collapses into a few repeating tokens, the same placement decoded at 34.7 tok/s at depth 6 and 32.3 at depth 4096 (2026-09-24).
- **Synthetic and prose prompts do not share a table row.** Random ids route to different experts than text does, and the prose rows run with the hot list.

For V4.1 decode, the host tier's memory bandwidth sets the step: 135–137 GB/s at 32 threads ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#v41-host-tier-k-rows)). The cost of each part is in [`docs/HARDWARE.md`](docs/HARDWARE.md).

## Target hardware

| | Minimum | Extended |
|---|---|---|
| GPU | RTX 3090 24 GB ×1 (sm_86) | RTX 3090 ×2: the second card runs the DSpark draft (splitting the model across two cards is not built) |
| CPU | AVX2, 8 DDR4 channels | same |
| RAM | 256 GB | same |
| Storage | NVMe for the engram table | same |

The DSpark draft (about 8 GB) has been timed only on a second card. On one card it would take the place of card experts, and that case has not been measured. The development machine is a Threadripper PRO 5975WX (32 cores, 8 DDR4 channels, 264 GB) with an RTX A6000 and an RTX 3090. Details and costs: [`docs/HARDWARE.md`](docs/HARDWARE.md).

## How it is verified

- **Oracle sets against ik_llama.cpp.** Intermediate tensors are dumped from ik_llama.cpp's CPU backend for V4.1 and compared per tensor and per layer. Integer outputs (engram row ids, router top-6 ids, indexer top-k lists) must match exactly, except where two candidates are within a tie band. Float outputs must stay inside bands derived from the two engines' rounding rules.
- **Bit gates.** A captured graph replay must equal the eager run bit for bit. The skewed two-row pass must equal two single steps bit for bit, and so must a rollback. A batched prompt must leave the model in the state one step per token leaves, bit for bit, for prompts of 1 to 4096 tokens and under each prompt setting. Node and launch counts are pinned at build time.
- **Drafts change no output.** With the DSpark or the lookup draft, greedy tokens must equal the plain run's.
- **PPL and KLD.** Hot list, wikitext-2 at 2048 context, 4 chunks, 3090, on the engine's first file (Q8_0 attention; see Status): PPL 1.9003 (bloomery) against 1.8989 (ik_llama.cpp CPU), KLD 0.00601 ± 0.00028, same top token 97.87 % ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-24.md#hot-list-ppl-kld)).
- **A gate is shown to fail** on the defect it guards before the fix goes in. Bands are not relaxed to make a gate pass.

Timing runs only on a quiet machine, under a machine-wide lease, with a witness block around every timed region. The protocol is rig-log's [`docs/quiet-machine.md`](https://github.com/midagedev/rig-log/blob/main/docs/quiet-machine.md).

## Limits

- **sm_86 only.** Every kernel is built and gated for Ampere (`--arch sm_86`). Other architectures are not tested.
- **A pinned nightly.** The toolchain is `nightly-2026-08-28`, pinned together with a cuda-oxide git revision. The source comes from our fork of that revision, where fixes wait until upstream takes them (`THIRD_PARTY_NOTICES.md` lists them).
- **One machine.** Timed numbers come from one A6000 in one workstation. The tooling (`tools/box.sh`) assumes a Mac editor and that workstation; [`docs/BUILD.md`](docs/BUILD.md) says what to run on your own host.
- **V4.1 prompts are bound by the CPU expert tier.** At 4096 synthetic tokens each layer and batch takes about 74 ms, 69 of them in the host experts; the card's work runs underneath ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#prefillgroup-ab)). A 4096-token prompt takes about 16.5 s. The next steps are on the host side: faster host-expert kernels, and streaming the hottest host experts to the card over PCIe. The server reuses a cached prompt prefix, so a follow-up turn pays only for its new tokens.

## Build

See [`docs/BUILD.md`](docs/BUILD.md). In one line: a Linux x86-64 host with AVX2, CUDA 13.3, LLVM 21 and `cargo-oxide`, then `cargo oxide build --arch sm_86 -- -p bloomery-gpu-gates --features deepseek41 --release --bin generate_ds41` (or `--bin bloomery-chat`, `--bin bloomery-serve-ds41`).

## More

- Measurements with their command lines: [rig-log](https://github.com/midagedev/rig-log), under `log/` (Korean).
- Plan and cost models: [`docs/plan.md`](docs/plan.md) (Korean). GPU design: [`docs/gpu-design.md`](docs/gpu-design.md) (Korean). Placement: [`docs/v41-placement.md`](docs/v41-placement.md) (Korean).
- The working contract for contributors and agents: [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).

The engine started on DeepSeek-V2-Lite-Chat Q3_K_M, and that path still runs and is gated. On the A6000 (300 W), V2-Lite decode ran at 229.5 tok/s at depth 6 and 199.6 at depth 4096 (2026-09-22, [rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22-o-the-depth-table-on-the-new-default-and-the-reversal-is-gone.md)).

## Upstream

What this work needed from its reference and its toolchain went upstream:

- ik_llama.cpp [#2455](https://github.com/ikawrakow/ik_llama.cpp/pull/2455): DeepSeek-V4.1 support, the oracle above. Merged 2026-09-21.
- ik_llama.cpp [#2444](https://github.com/ikawrakow/ik_llama.cpp/pull/2444): `GGML_CUDA_NO_PINNED_WEIGHTS`, which loading V4.1 with CPU experts needs. Merged 2026-09-15.
- ik_llama.cpp [#2522](https://github.com/ikawrakow/ik_llama.cpp/pull/2522): loading DeepSeek-V4.1 DSpark drafts. Merged 2026-09-25.
- ik_llama.cpp [#2507](https://github.com/ikawrakow/ik_llama.cpp/pull/2507): V4.1 index keys from the pre-RoPE latent. Open, draft.
- cuda-oxide [#1314](https://github.com/NVlabs/cuda-oxide/pull/1314) (a constant-folder crash on mixed-width shifts, merged 2026-09-23) and [#1321](https://github.com/NVlabs/cuda-oxide/pull/1321) (unrolling loops whose exit test adds a constant to the counter, merged 2026-09-24); [#1329](https://github.com/NVlabs/cuda-oxide/pull/1329) (installing the cached backend by rename) is open.
- llama.cpp [#29008](https://github.com/ggml-org/llama.cpp/pull/29008): message delimiters in the DeepSeek V3.2/V4 chat parser. Merged 2026-09-17.
- Open: mistral.rs [#2430](https://github.com/EricLBuehler/mistral.rs/pull/2430) (non-F32 activations in the CPU GGUF MoE gather) and cutile-rs [#309](https://github.com/NVlabs/cutile-rs/pull/309) (`CudaContext::mem_info`).

Ten more of our fixes are merged in ik_llama.cpp; the full lists are [our ik_llama.cpp PRs](https://github.com/ikawrakow/ik_llama.cpp/pulls?q=is%3Apr+author%3Amidagedev) and [our cuda-oxide PRs](https://github.com/NVlabs/cuda-oxide/pulls?q=is%3Apr+author%3Amidagedev). Toolchain issues not filed yet are in [`docs/upstream/nvlabs-ledger.md`](docs/upstream/nvlabs-ledger.md).

## References and credits

[ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) and ggml are the numerical reference. The kernels were written by reading ggml's k-quant code and ik_llama.cpp's `iqk_mul_mat` kernels, the tokenizer follows the reference's `llama_tokenize`, and accuracy is defined against their output. Where code or an algorithm was taken, the source comment points at the original file. [exllamav3](https://github.com/turboderp-org/exllamav3) and [mistral.rs](https://github.com/EricLBuehler/mistral.rs) are design references, cited where used (for example `crates/gpu/src/route_core.rs`). All four are MIT. The GPU kernels compile with [cuda-oxide](https://github.com/NVlabs/cuda-oxide) (Apache-2.0).

AI assistants helped write the code and the documentation in this repository. Every number here was measured on the machine by the runners in `tools/ref/` and the GPU gate runner.

## License

MIT, see [`LICENSE`](LICENSE). `crates/oxide-ice-unroll`, a compiler-bug reproducer kept out of the workspace, carries NVIDIA's Apache-2.0 headers. Third-party notices, including the ggml authors' MIT notice for the generated tokenizer table, are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
