# Performance and verification

The numbers' conditions, why the engine is fast, and how its output is checked. The headline tables are in the
[README](../README.md#numbers).

## Conditions of the numbers

Two requests at once (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together. The host
is the development machine's: a 32-core AVX2 CPU with 8 DDR4 channels and 256 GB of RAM. The RTX 3090 runs at a
250 W cap (a stock 3090 draws 350 W). Each number comes from the runners in `tools/ref/` under the quiet-machine
protocol ([conditions](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#num3090)); one-request rows,
the A6000's own rows, the measured history and the other engines' rows live on
[rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md).

- Qwen3.8 and GLM-5.3 ran with the MTP draft off and adaptive residency on.
- Qwen3.6 runs two requests in turn today, so its total is one request's rate.
- V4.1 on one 3090: a fixed placement with adaptive residency off: a server with the 3090 alone turns residency on,
  which that row does not count yet.
- On two cards Qwen3.8 runs two requests in turn today; one pass of both over the second card is in progress.

Each of two requests runs at 0.62–0.68× its speed alone, so one user waits longer and the machine serves more
([A6000](https://github.com/midagedev/rig-log/blob/main/log/2026-10-06.md#rel021-slots)).

A first start compiles the GPU code for the card (tens of seconds); later starts take seconds. A warm V4.1
load takes about 16 s.

**Other engines.** The bench page also holds rows measured side by side with llama.cpp, mistral.rs and
exllamav3 on the same card in the same window, each with its build and flags; on some rows bloomery is ahead
and on some it is behind. They are not a claim about the other engines: each ran at the fastest flags we found,
which is not a proof of its best, and some rows did not hold both engines to the same conditions: in the
V4.1 rows of 2026-09-28, llama.cpp read the engram table cold on new prompt ids while bloomery's prompt repeated
one ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-27.md#v41-xeng)). If you know faster flags
for a row, please open an issue.

## Why it is fast

Each point names the measurement that isolates it: the same engine, or a bench of that one path, with and
without that one choice, on the development machine. A point with no such measurement carries no number. The
language is not one of the reasons: the kernels are Rust (274 of them, through
[cuda-oxide](https://github.com/NVIDIA/cuda-rust), [how](cuda-oxide.md)), but the speed comes from the choices below.

- **Experts move to where they are used.** V4.1, Qwen3.8 and GLM-5.3 count their own routing as they run and
  swap routed experts between the card and the host between steps; a prompt call streams its hottest host
  experts onto the card, so decode starts warm. V4.1 `Q3_K_M`, 96 decode steps after a 512-token prose prompt:
  29.48 tok/s with it off, 35.70 with the swap rule, 43.86 with the prompt call's streaming as well (the
  default). The card served 18 % of the routed calls with it off; with streaming it served 57 % over the first
  16 steps and 71 % over the last 16 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#residency-clip)).

<p align="center">
  <a href="https://github.com/midagedev/rig-log/blob/main/assets/residency-explainer-v5.mp4">
    <img src="https://raw.githubusercontent.com/midagedev/rig-log/main/assets/residency-explainer-v5-poster.png" width="420" alt="Adaptive expert residency explainer (42 s video)">
  </a>
</p>

- **The card and the CPU work on a prompt at once (V4.1).** A prompt runs in batches of up to 512 positions:
  every routed expert a batch calls is read once for the whole batch. Batching took P = 512 from 30.7 to
  91 tok/s against one position at a time ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#ds41batch-pp)). Two batches stay in
  flight, layer first, so the card routes and attends one while the CPU runs the other's host experts: 1.20×
  at P = 4096, the host's wait for routes falling from 17.8 to 0.8 ms per layer-batch
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-26.md#prefillgroup-ab)). A batched prompt leaves the same state as one step per
  token, bit for bit.
- **A prompt's CPU experts move to the card (Qwen3.8).** Each layer of a prompt seats the host experts it calls
  most on the card before it runs them, and under `--place a` streams more through a ring of card slots when the
  host would still finish last; the seated experts stay for the answer. One binary, the path off against on, a
  4,096-token prose prompt on the A6000 (residency `mid-p148-s1`, MTP off): 772.9 → 1,350.5 tok/s; seating alone
  gave 1,342.1 ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-10-07.md#rel026-xstream)).
- **Qwen prompts as tensor-core matrix products.** The Qwens' prompt weights and routed experts run as grouped
  int8 tensor-core GEMMs: P = 4096 went from 357 tok/s (eight positions a pass) to 2,615
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#qwen3prefill-ab)). A prefill-only flash attention kernel, where a block owns 64
  query rows and loads each 64-key tile once, then took it 2.12× further
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#q3pflash-ab)), and ubatches of 4096 tokens instead of 512 another 1.35×
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-25.md#q3ubatch-ab)).
- **Disk rows read ahead of the step (V4.1).** V4.1's engram table (195 GiB) stays on NVMe, and every token
  reads 48 scattered rows. In a bench of that path, read on demand from a cold cache they took 4.69 ms a token;
  with the next token's read-ahead issued early, 0.31 ms ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-22.md#p-the-engram-path-costs-a-third-of-a-millisecond-and-it-is-syscalls)).
- **CPU experts at memory speed.** AVX2 kernels take the dot products on the quantized weights as stored, and
  a step's host experts go out as 82 grouped dispatches on a pinned worker pool instead of 720 per-matrix
  ones. In a bench of that leg on an earlier V4.1 file: 31.1 ms against 35.5 ms for the same bytes, at 130 GB/s of the machine's 148
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-23.md#v41-host-leg)). The kernels are not faster than ik_llama.cpp's on one core;
  the leg is bound by memory.
- **One CUDA graph a decode step, the CPU experts inside it.** Where experts run on the host, the card waits on
  a counter in pinned memory that the CPU workers write, not on a host synchronization. Not isolated by a
  measurement on today's models.
- **Speculative decoding** with greedy output unchanged: DSpark for V4.1 (the draft on a second card), the MTP
  head for Qwen3.8 and GLM-5.3.
- **Loads stream through a pinned ring** with one sync: a warm V4.1 load went from 36.3–36.6 s to 15.9–16.6 s
  ([rig-log](https://github.com/midagedev/rig-log/blob/main/log/2026-09-30.md#v41-load-upload)).

## How it is verified

- **Against ik_llama.cpp.** Intermediate tensors are dumped from its CPU backend and compared per layer.
  Integer outputs (router ids, engram rows, indexer top-k) match exactly outside tie bands; float outputs stay
  inside bands derived from both engines' rounding. On the public V4.1 file: PPL 2.2401 against ik's 2.2378,
  KLD 0.00987, same top token 97.46 %.
- **Against Cloudflare's own Python, for Clef.** The request encoder's token ids equal the release's; the
  head's logits stay within 2e-5 of an f64 copy of the release's head; the backbone's hidden states gate
  against llama.cpp mainline's final-norm rows. End to end, the top option equals the BF16 release's on 29 to
  31 of 31 questions for every bartowski file from `Q8_0` down to `Q3_K_S` — every miss a question the release
  itself answers within 0.02 of a tie ([table](../tools/ref/clef/agreement.md)).
- **Bit gates.** Graph replay equals eager execution; the skewed pass equals two single steps; a rollback and
  a V4.1 batched prompt leave the same state as one step per token.
- **Each gate is shown to fail** on its defect before the fix lands. Bands are not relaxed to pass.

## Recorded sessions

Recorded sessions, not benchmark rows. bloomery 0.2.1: [V4.1 answering a code review on one
A6000](https://tape.midagedev.com/r/un7mimv23csn2pyhfk3d), [GLM-5.3 on two cards
(`--place bp`)](https://tape.midagedev.com/r/wcp952uyjebbpui7sgch), [Qwen3.8 on one
A6000](https://tape.midagedev.com/r/9d6bv7ssftr4e9cu8wdi), [Qwen3-30B serving four streams on one
3090](https://tape.midagedev.com/r/uxkad26d6jjr26nixrkh). Earlier: [V4.1 on two cards answering a coding
review](https://tape.midagedev.com/r/6w4t9r5nqwtt5c9sagn3); [GLM-5.3's server at its
defaults](https://tape.midagedev.com/r/6kf3sxuqpza6m7k7iwi6) against [llama.cpp's GLM pull request on the
same card](https://tape.midagedev.com/r/6nn6grc6hztpssp88hz5); [Clef-Flash answering seven Korean SystemOne
requests](https://tape.midagedev.com/r/zd3asiqegffcmvky9hti).
