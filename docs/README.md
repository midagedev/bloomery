# docs

Two kinds of pages live here. The **reference** pages describe the current tree, in English; their numbers
carry their dates. The **working records** are the project's dated notebook: plans, designs, audits and research reports, mostly in Korean, each true
as of its date and often superseded since. Measurements live in [rig-log](https://github.com/midagedev/rig-log), not
here.

## Reference

| Page | What it covers |
|---|---|
| [`BUILD.md`](BUILD.md) | Building from source: the toolchain, the cuda-oxide fork, one command block per model |
| [`HARDWARE.md`](HARDWARE.md) | What each part of the machine costs in a DeepSeek-V4.1 decode step, and the RAM floor |
| [`cuda-oxide.md`](cuda-oxide.md) | How the GPU kernels are written in Rust with cuda-oxide |
| [`upstream/`](upstream/) | Toolchain defects (`nvlabs-ledger.md`) and the notes behind upstream pull requests |

The top-level [`README.md`](../README.md) covers install, use, the models, the numbers and why it is fast;
[`AGENTS.md`](../AGENTS.md) is the working contract.

## Working records (Korean, dated)

| Page | What it is |
|---|---|
| [`plan.md`](plan.md) | The plan and its cost models (「모델」, 「흐름 모델」) |
| [`plan-triage.md`](plan-triage.md) | Open work: the one tracker of what is left to do |
| [`plan-ledger.md`](plan-ledger.md) | What is closed: earlier plan and triage text, moved verbatim |
| [`facts.md`](facts.md) | Machine, model-file and toolchain facts the plan stands on |
| [`fair-measure.md`](fair-measure.md) | The conditions every public cross-engine row must meet |
| [`rust-quality.md`](rust-quality.md) | The code-review rules (R1–R29) that reviews cite |
| [`cards/`](cards/) | Prediction cards: one per box sitting or lease (format in `tools/ref/card.py`) |
| [`research/`](research/) | Design and research reports from delegated rounds, each with its sources |
| [`roofline.md`](roofline.md), [`oracle.md`](oracle.md), [`gpu-design.md`](gpu-design.md), [`arch-split.md`](arch-split.md), [`v41-placement.md`](v41-placement.md), [`v41-inventory.md`](v41-inventory.md) | Early designs (2026-09-19 to 09-27): the decode byte budget, the reference boundary, the first GPU stage on V2-Lite, the per-architecture split, V4.1's placement and tensor inventory |
| [`gates-plan.md`](gates-plan.md), [`cpu-dispatch-plan.md`](cpu-dispatch-plan.md), [`ctx-size-plan.md`](ctx-size-plan.md), [`rebuild.md`](rebuild.md) | Plans for one area each: gate time, CPU dispatch, context sizing, the post-hoc redesign audit |
| [`review-2026-09-21.md`](review-2026-09-21.md), [`RESULTS-mul30-gpu-scout.md`](RESULTS-mul30-gpu-scout.md), [`RESULTS-mul35-saturation.md`](RESULTS-mul35-saturation.md) | One-off reviews and scouting results |
