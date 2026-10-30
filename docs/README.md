# docs

Two kinds of pages live here. The **reference** pages describe the current tree, in English; their numbers
carry their dates. The **design records** are the few dated designs that code cites, mostly in Korean, each true
as of its date and often superseded since. Measurements live in [rig-log](https://github.com/midagedev/rig-log), not
here.

The maintainers' working documents moved to rig-log on 2026-10-11: the plan and its cost models, the open-work
tracker, the closed-work ledger, the machine facts, the research reports and the upstream pull-request drafts. They are
in [`docs/bloomery/`](https://github.com/midagedev/rig-log/tree/main/docs/bloomery) of that repository.

## Reference

| Page | What it covers |
|---|---|
| [`evaluating.md`](evaluating.md) | A checklist to run bloomery and compare it fairly; hardware reports |
| [`agents.md`](agents.md) | Starting the server from an agent: the commands, the events to wait on, the exit codes |
| [`claude-code.md`](claude-code.md) | Claude Code on a local model, on an NVIDIA card or a Mac, with measured first-turn times |
| [`pi.md`](pi.md) | pi on a local model: the smallest first request of the surveyed agents, on an NVIDIA card or a Mac |
| [`dsh.md`](dsh.md) | DeepSeek Harness (dsh) on DeepSeek-V4.1-Flash: the server, the profile patch, what to expect |
| [`serving.md`](serving.md) | llama-server compatibility, the flags, the seats, concurrent requests, drafts, residency, limits |
| [`models.md`](models.md) | The recommended files and what each model needs in GPU, RAM and disk |
| [`performance.md`](performance.md) | The numbers' conditions, why it is fast, how it is verified, recorded sessions |
| [`mac.md`](mac.md) | Apple Silicon: install, the Metal backend's build, the files that open, the server, its limits |
| [`BUILD.md`](BUILD.md) | Building from source: the toolchain, the pinned cuda-oxide and its one carried patch, one command block per model |
| [`HARDWARE.md`](HARDWARE.md) | What each part of the machine costs in a DeepSeek-V4.1 decode step, and the RAM floor |
| [`cuda-oxide.md`](cuda-oxide.md) | How the GPU kernels are written in Rust with cuda-oxide |
| [`upstream/nvlabs-ledger.md`](upstream/nvlabs-ledger.md) | Toolchain defects found in cuda-oxide, cutile-rs and the driver; the pull-request drafts behind them are in rig-log |
| [`contrib/`](contrib/) | For contributors: a local check, how to add a model, good first issues, the DGX Spark and Mac ports |

The top-level [`README.md`](../README.md) covers the status, install, use and the headline numbers;
[`AGENTS.md`](../AGENTS.md) is the working contract.

## Design records and rules (dated)

| Page | What it is |
|---|---|
| [`fair-measure.md`](fair-measure.md) | The conditions every public cross-engine row must meet |
| [`rust-quality.md`](rust-quality.md) | The code-review rules (R1–R29) that reviews cite |
| [`cards/`](cards/) | Prediction cards: one per box sitting or lease (format in `tools/ref/card.py`) |
| [`research/`](research/) | The design and research reports that code comments cite, each with its sources |
| [`roofline.md`](roofline.md), [`oracle.md`](oracle.md), [`gpu-design.md`](gpu-design.md), [`arch-split.md`](arch-split.md), [`v41-placement.md`](v41-placement.md) | Early designs (2026-09-19 to 09-27): the decode byte budget, the reference boundary, the first GPU stage on V2-Lite, the per-architecture split, V4.1's placement |
