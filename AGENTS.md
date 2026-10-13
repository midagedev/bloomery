# AGENTS.md — bloomery

An LLM inference engine for one workstation: Rust host, CUDA-Rust kernels (cuda-oxide today, cutile-rs later), AVX2
kernels for the CPU expert tier. This file is the working contract for everyone who changes the code, people and their
agents alike. It holds only rules in force; their history is in the commit log and in
[rig-log](https://github.com/midagedev/rig-log) (`docs/bloomery/`: the plan, the cost models, the measured record).

To run or evaluate bloomery rather than develop it, read the README's "Status" section and `docs/evaluating.md`. How a
change is checked and merged is in `CONTRIBUTING.md`. A maintainer's machine may carry its own operating rules in a
`CLAUDE.local.md` that is not in the repo; those add to this file and never loosen it.

## Never

- **Never build a device crate with plain `cargo`.** `crates/gpu` and every crate that links it (`gpu-deepseek41`,
  `gpu-glm5next`, `gpu-vision`, `gpu-gates` with its `gpu` feature) hold `#[cuda_module]`/`#[kernel]` code that needs
  the CUDA codegen backend. Use `cargo oxide` (the recipes do). One exception: the host nightly (`tools/nightly/run.sh`)
  runs `bloomery-app`'s default-feature tests under plain cargo, which open no card (`tools/recipes.py`
  HOST_DEVICE_LINKED holds the reason); its numbers are host-tier evidence, never a kernel's.
- **Never run a gate where it cannot run.** Gates need x86_64 Linux and an NVIDIA card: `qdot` uses
  `std::arch::x86_64`, `threads` Linux affinity calls. On an arm64 Mac run only the static tier: `just mac-check` and
  `just mac-lint` (check and clippy as an x86_64-linux cross check, no linker), `just mac-fmt-check`,
  `just mac-static [BASE]` (fmt-check, lint, the feature combos scoped to BASE's changed files and every check script,
  in parallel; `tools/mac-static.sh`) and the pure crates' native tests (`just mac-test`, the crates
  `tools/recipes.py pure-crates` selects). A static result is development-loop evidence, not a gate's.
- **Never time by hand.** The project's numbers come from the runners in `tools/ref/` (`decode-measure.sh`,
  `ab-decode.sh`, `depth-gpu.sh`, `depth-ds41.sh`, `nsys-gpu.sh`, `ncu-gpu.sh` and their recipes), each taking the
  timing lease through `lease_take`: a machine-wide lock, a GPU-idle wait, and witness blocks around every timed
  region. A number from another machine carries `CONTRIBUTING.md`'s evidence list instead (the card and driver line,
  both commits, the model file, the commands and their output). A number with neither is not admissible.
- **Never relax a gate or a lint to make it pass.** Raise it with a dated comment and a reason, or file an issue. The
  lint levels in the root `Cargo.toml` carry the hit counts they were chosen from.
- **Never write a number you did not measure.** If it is derived, say so. If it turns out wrong, strike it through and
  correct it in place.

## Commands

The recipes' own lines and the tool headers (`tools/recipes.py`, `tools/gate.sh`, `tools/gpu-gate.sh`) are the full
reference; what follows is the contract.

    just check / lint / fmt   # check --all-targets (also refreshes Cargo.lock: commit the lock with the edit, or the
                              # timing runners refuse stale binaries, rc 3); clippy, 0 errors; cargo fmt, pinned nightly
    just contrib-check        # what a machine with no card can run: --tier t0 any OS, t1 adds the host tier (CI runs t0)
    just build-ref            # the C++ reference harnesses that link ggml (rerun when ik moves; gate-qdot reads them)
    just deny                 # cargo deny; fails if the cuda-oxide pin floats; reaches the network
    just affected [BASE|A..B] # the gate-* recipes a change selects, from the crate graph and module trees, the scripts
                              # a recipe names, the cargo globals and each manifest by table; builds nothing
    just gate / smoke         # static checks, lint, fast library gates / the round loop's subset
    just gate-<name>          # one subsystem's tests, bounded, real exit code
    just gate-ptx-spill       # every PTX entry's spill/jit_local bytes against tools/ref/ptx-shapes.tsv
    just mac-check / mac-lint / mac-fmt-check / mac-static / mac-test   # the Mac's static tier (above)
    just records-refresh      # tools/bloomery/schema/ and tools/flow/plans/ after a record kind or prompt-call plan
                              # change, each plan at the plan room (tools/ref/plan-room.sh)
    just lab-engram           # the engram IO lab's tests; lab-, not gate-, so no engine landing selects it
    just gate-refset / refset-check   # the reference-set readers and every family's sets in place

- **Bounds and exit codes.** Every `gate-*` recipe runs under `timeout --kill-after=10 900`; the exit code has one
  owner, `tools/gate.sh`. GPU gate binaries run through `tools/gpu-gate.sh`: it takes a card's gate lock, runs under
  the same bound (`BLOOMERY_GATE_BOUND`, whole seconds ≥ 1, parsed by `tools/gate-bound.sh`; any other value ends the
  runner with 64), and returns the binary's code (124/137 timed out, 75 lock contention, 69 a lock file it cannot open).
  `just check-recipes` fails on a test recipe with `||` or a bare `cargo test`, on a `--bin`/`--test`/`-p`/feature
  that names nothing, a script not in the tree, a `gate-*` recipe with no cargo target or one that runs its binary with
  `cargo run`, a bin built without its `required-features`, a bin two recipes build into one target dir with different
  resolved feature sets (one set a bin, so no build replaces the binary another lane runs), a recipe that takes a lock
  itself, and on a Python tool under `tools/` whose self-test is not listed.
- **Narrowing.** The graph over-selects a host-only change. When `just ptx-scan` equals the base for every bin the
  change reaches, no kernel moved, and the gates to run are those that run the changed host path plus the static
  checks; `just narrow BASE [HEAD]` scans the covering bin set (`tools/recipes.py scan-set`, every kernel carrier) with
  `--no-jit` on both sides and prints that list and why (`just affected BASE --narrow --scan BASE_LOG NEW_LOG …` by
  hand; an uncommitted tree: `tools/narrow-scan.sh BASE --worktree`). A kernel, a launch or a byte formula moved runs
  the whole list. Always `just ptx-scan <bin> --features …`. A `--no-jit` scan (no card, no gate lock) proves only a
  move: a pair with one narrows when it is identical and is refused by name otherwise; the add and occupancy classes
  and `gate-ptx-spill` read the JIT columns and take the full scan.
- **Weekly and opt-in.** `just affected` selects a `weekly-*` recipe only when a changed file matches one of its trigger
  rows in `tools/gate-paths.tsv` or its own text changes. `just stage-gpu-load-v41 [--plan a]` (plan (b) staged on
  both cards, checked byte for byte; solo) is never selected: a change to `crates/model/src/placement.rs`,
  `placement/{workstation,host_lock}.rs`, `crates/gpu/src/{hybrid,weights}.rs` or `gate_load_v41.rs` runs it by name.
- `just gate` excludes the measure targets (they need a quiet machine and a lock) and `just deny`. Build artifacts land
  in the workspace root `target/`; the measure runners read only from there.

## Derive first, measure the gap

A change opens with a prediction: the value and band derived from the cost and error models in rig-log
`docs/bloomery/plan.md` (「모델」), and the proof its change class needs. When the measurement misses, which term was
wrong is the finding.

| Change class | Proof | Runtime gate | Timed A/B |
|---|---|---|---|
| move, split, rename | `just ptx-scan` identical (`--no-jit` suffices: no card, no gate lock); host dispatch code also keeps its structural lines (graph node count, eager = replay, e2e set identical) | none beyond those lines | none |
| delete | ptx-scan of `generate_ds41` and `gate_e2e` = base minus exactly the deleted entries; `gate-ptx-spill` red on exactly those rows before `ptx-shapes.tsv` loses them; per entry, the grep that shows no caller. A remaining entry whose md5 moves needs `tools/ref/ptx-canon.py` = `reordered-only` and equal resource columns | the owning gates of what it touched (bit-identical for a moved md5); a removed gate, case or arm is a coverage change with a dated reason | none |
| add (no engine caller yet) | ptx-scan = base plus exactly the new entries; `gate-ptx-spill` red on exactly them before pinning; the family gate green with FAIL-first per clause | the new family gate, and the owning gates of entries engine bins gained | none |
| integer-path reorder | bit-identical by associativity | the owning gate once | none |
| launch count only | Δt = ΔN × c_node | the owning gate | once, only if occupancy moves too |
| fold a launch into a neighbour | Δt = −ΔN × c_node − the removed kernel's time + the work every host-grid block now repeats or waits on × its blocks (paid in full on a grid near ~700 GB/s) | the owning gate | once |
| instruction count on a kernel near ~700 GB/s | model says 0 — do not open the change | — | — |
| occupancy / geometry | computed from regs, smem, threads, SM count (ptxas `-v`), written in the description | the owning gate | once |
| float sum order or precision | the error model (σ against `exact-forced-32.tsv`, a diagnostic) | `gate-gpu-e2e` count pin | once |

Two habits precede the code. **Decompose on paper:** the terms come from the code (byte, instruction, launch, loop
counts) and numbers already measured; a timed run is for the one term derivation cannot fix, named with its expected
value. `lease_take` enforces it: a timed run without a prediction card (`docs/cards/<slug>.card`; format, kinds and
exit codes in `tools/ref/card.py`) gets no lease, and an A/B whose effect sits inside the ruler at its round count is
refused. **Draw the resource timeline:** per unit of work, how long the host CPU, card SMs, PCIe, host DRAM and NVMe are
each busy, and whether the wall is their sum or one resource's max; ask whether the flow should change (asynchrony,
bulk, SIMD, SIMT) before a term is shortened. A term under another resource's shadow is worth 0 (worked example: rig-log
`docs/bloomery/plan.md` 「흐름 모델」).

The busiest unit is not the bound until the step's cycles are its demand: write every unit's demand next to the step's
cycles in the same unit; when the largest is well under the step, the step is latency the resident warps do not hide,
and a lever on that unit predicts 0 (rig-log 09-26#q3gemma-ab: L1TEX wavefronts −39 %, cycles unmoved).

Measurement comes first only for the named residue: hardware faults, register allocation, cache effects with no
mechanism yet. Build-time facts (node, launch, instruction, register counts) are compile-time ratchets, not runtime
measurements. Every number carries its conditions — `tok/s @ n=N, depth D, card` — or it is not a number.

## Performance first, accuracy opt-in

When a choice trades speed against closeness to the exact result, the default is the faster one, and the engine carries
only that path. A more exact variant lives on the verification side, in the f64 referee
(`exact_ref --act f64|ours|ik|ik16`). Bug hunting compares the engine with the simulation of its own rule
(`--act ours`): off by more than σ is a bug candidate, within σ is rounding. Accuracy measures (σ, forced_exact
buckets) are diagnostics and do not block a performance change; the only accuracy pins are bug catchers (the
forced_exact count pin, the reference gates).

## Layout

    crates/gguf/        GGUF reader, dequant, per-weight-type activation format
    crates/threads/     resident pinned worker pool (spin, then park)
    crates/qdot/        fused quantized dot kernels, AVX2
    crates/model/       the CPU engine (ops, attn, ffn, moe, head, forward, kv, derived, profile); tests/ are the gates
    crates/tokenizer/   byte-level BPE from the GGUF header, bit-identical to llama-tokenize
    crates/sampler/     the sampling chain in the reference's order
    crates/runtime/     the generation loop, host only: Target/Verify/Draft, plain and speculative advances
    crates/app/         Session<B> over a GpuModel: open, plan, capture, the card drafts
    crates/serve/       llama-server-compatible HTTP API over an Engine trait
    crates/refset/      the reference sets' readers and the family table per architecture (src/arch)
    crates/gpu*/        the card engine and its per-family crates; gpu-gates holds the GPU gates and bins
    crates/oxide-ice-unroll/   a compiler-bug reproducer that must NOT compile; excluded from the workspace
    tools/ref/          C++ harnesses linking ggml, and the timing runners
    tools/bloomery/     the Python side's one reader per fact: records.py, manifest.py
    tools/review/       the pull-request review bot's tools (.github/review/ holds its instructions)
    docs/research/      the design docs code cites, and the v41-placement scripts
    rig-log docs/bloomery/   plan.md what is live, facts.md machine and toolchain facts, plan-triage.md open items,
                        plan-ledger.md what is closed, research/ sourced surveys

## Conventions

- **No silent failure.** Undefined input gets a named panic or error, never a defined output: a NaN block does not
  quantize to code 0, a NaN router lane does not yield duplicate ids, an out-of-range index is not clamped. Take the
  named error over "both paths give the same defined output", even at the cost of an API change. A device kernel that
  cannot panic raises the fault word (`crates/gpu/src/fault.rs`: one `u32` per `Gpu`, `(layer << 8) | site` by atomic
  min, read back with the token); `Head::tokens` returns `GpuError::Fault` and the model stays `Poisoned` until
  `reset`. The kernel keeps its memory accesses defined and never writes a plausible value.
- Correctness is defined against ggml: kernels match its output within the band in each crate's RESULTS file.
- **Toolchain defects** (cuda-oxide, cutile-rs's cuda-core/cuda-bindings) go into `docs/upstream/nvlabs-ledger.md` the
  moment they are met, before any workaround. A cuda-oxide defect that shapes our code is fixed in our fork as it is
  met: one commit on the `bloomery` branch with its reproducer and FAIL-first, the pin moved (below), the workaround
  removed by the change that moves the pin.
- **Common code first, so the next model attaches cheaply.** A concept two models share (tier legs, prompt walks,
  seats, placement sizing, drafts, residency glue) has one common owner. A model's own code holds only what an
  architecture fact forces (tensor shapes, the recurrent state, the attention kind, the quant), and its description
  names that fact. A per-model copy of shared logic stays only for a large measured speed gain, named in the code
  beside it. Lift what two models already do the same way, not what a future model might need. A common owner that
  grows a branch, flag or trait method per model is the wrong cut; the model-specific part goes back to the model.
- **Code shape** is judged by `docs/rust-quality.md` (R1–R29); reviews cite rule numbers.
- **Comments state what is true now.** Keep `// SAFETY:` (the invariant), one line of why for a non-obvious choice,
  "this order is the gate" on a load-bearing float reduction, a caller's contract. Remove issue numbers, dates, measured
  ms/GB/s/tok/s and how a bug was found. One exception: a re-pinned gate constant keeps its dated one-line attribution,
  `PIN(YYYY-MM-DD):` (a band, a tolerance, `KNOWN_DIVERGENCE`). `tools/check-comments.sh` enforces it in `crates/*/src`.
- **Tests are the gates, and only the gates.** One test per contract; shared harness code in `tests/common`; no
  print-only tests, no second test pinning what another pins. Removing a gate needs the same dated reason as relaxing
  one. Gate tests are `hw_`-prefixed and `#[ignore]`d (`.config/nextest.toml` fences them); the `just gate-*` recipes
  run them.
- **Record lines have one owner:** a `Kind` in `crates/gpu-gates/src/record.rs`, rendered by `Record`, which panics by
  name on a value out of order, missing or of another type (`<bin> --records-schema` prints the kinds). Readers go
  through `tools/bloomery/records.py` by kind and field. `generate_ds41 --plan` prints the prompt call's records and
  exits before the load; a batched run stops with a named error when the call it ran differs from its printed plan.
  Under `BLOOMERY_STEP_STATS=1`, `tools/flow/ds41_prefill.py --counts <log>` holds the flow model's queue-entry counts
  to the engine's `stat prefill front` and `stat prefill lb` records.
- **Reference sets have one reader:** `crates/refset` (Rust) or `tools/bloomery/manifest.py` (Python), by column name.
  Every set opens through its family's check (`crates/refset/src/arch`), which refuses by name a set dumped from another
  model file (the first shard's full path is the identity), another ik build or architecture, or without its trailer.
  A new kind of set is a new family row with its writer's recipe.
- **A feature-gated bin declares `required-features`** in `crates/gpu-gates/Cargo.toml`, so cargo refuses to build it
  without them instead of linking the stub `main`.
- **Do not split a `#[target_feature]` kernel body into helpers** (10–13 % loss). A helper touching `_mm*` intrinsics
  is `#[inline(always)]` or carries the attribute itself. `.cargo/config.toml` sets `-C target-cpu=znver3` for plain
  cargo builds, and `.cargo/cuda-oxide.toml` carries it for oxide builds (`CARGO_ENCODED_RUSTFLAGS` masks the former),
  both per target triple (aarch64 takes `native`); `just check-rustflags` holds each triple's two lists equal. The
  attributes stay.
- **An unprofiled chunk takes no lock and reports nothing.** Per-chunk collectors are for `BLOOMERY_PROFILE` and
  errors; an unconditional lock per chunk is a futex convoy (symptom: a flat thread-scaling curve; first question
  `SWEEP="8 16 24 32" just measure-decode`).
- **The step does no load-time work.** Anything that does not depend on the tokens is resolved once into `Derived`.
  `just gate-alloc` counts allocator calls per steady step and only ratchets down. Activation blocks come from
  `Tensor2::scratch` when every cell is written; `BLOOMERY_POISON=1` turns a missed cell into a NaN the gates catch.
- **Runtime levers** are the rows of `crates/levers/src/registry.rs` and nothing else documents them. A binary parses the
  `parsed` rows first thing in `main` (`bloomery_levers::at_main`), refuses a value its kind does not take, and hands
  typed values down; `<bin> --levers` prints them. An `in place` row is a line of `tools/levers-direct.txt` with the
  change that converts it (`just check-levers`). A retired name that is set is refused by name. `just gate-levers`
  prints the table.
- **Toolchain pins.** `rust-toolchain.toml` pins the nightly and moves only with the cuda-oxide pin. `cuda-oxide` is
  pinned by `rev` in `[workspace.dependencies]` and taken from our fork's `bloomery` branch through `[patch]` (patches
  listed in `THIRD_PARTY_NOTICES.md`); `just deny` fails if either floats. Before the fork's branch moves, the old pin
  gets the tag `pin/<short rev>`; pin tags are never deleted. A fork patch that changes codegen is a pin move (every
  gate); one that must not proves it with identical `just ptx-scan` tables of `generate_ds41` and `gate_e2e`.

## Measuring

- **Cards.** A functional gate runs on any card unless a fact pins it. A gate bin that runs the gate plan
  (`workstation::plan_gate`, the 3090's bytes the README's 3090 rows are measured at) plans those bytes on the card it
  holds (`crates/gpu-gates/src/bin/shared/gate_card.rs`), so its pins hold on any card with the room.
- **Public numbers name their machine.** Anyone's number can be published, a maintainer's or a contributor's, when it
  carries its conditions and evidence. Each card or card set, and each machine, has a table of its own: a table never
  mixes cards. A row beside a reference engine gives that engine the same cards in the same session (`-ts` for two
  cards). A contributor's number decides their pull request on their card; where two cards measure a change
  differently, both stand, separated by a gate on the card fact, not a veto.
- **No gate guards speed, so a change that touches the dispatch path ends with a same-lease A/B.** Build the base in a
  worktree (`just build-decode`), then `just ab-decode <that tree>`; judge by the interleaved relative numbers (the
  same commit moves ~5 % between windows). The runner rotates arm order every round (a fixed first arm reads 0.3–0.8 %
  slow). `BLOOMERY_AB_ENVS="K=V;K=V"` adds same-binary lever arms; put an A/A arm in for any claim under 1 %.
- **Know the ruler before reading it.** Same-binary runs scatter with SD 0.6 %, so the 95 % interval on a difference of
  two arm means is ±1.0 % at four rounds and ±0.8 % at six; ±0.5 % takes 13 rounds per arm and ±0.3 % takes 32
  (sd·√(2/N) with t(0.975, 2N−2); `tools/ref/card.py` computes it). Report the effect and its interval, not a win
  count. Under 1 %, judge only between same-binary lever arms.
- **Prefill is a headline metric beside decode.** Every model's public numbers carry `pp tok/s @ P = 512 and 4096, card`
  next to decode tok/s, for ours and the reference engines in one lease; a change touching the prompt path is judged
  on it. The reference engines' rows live on rig-log's bench page with their flags; the README links them and states
  no ratio against another engine.
- **A decode headline names its depth.** `tools/ref/depth-decode.sh` runs both engines at each depth in one lease
  (`BLOOMERY_DEPTHS="6 1024 4096"`); a change touching attention or the KV cache is judged on the deep rows. For V4.1,
  `just depth-gpu-ds41 prose:<P>[@K=V,…]` / `code:<P>` feed the first P ids of `corpus-prose.ids` / `corpus-code.ids`;
  each corpus's arms are compared only with arms of the same corpus and P, in their own tables.
  `BLOOMERY_GEN_PLACE=gate` runs our arms at `--place gate` on the 3090. Reference arms are preheated
  (`BLOOMERY_PREHEAT=0` off); rows carry `majflt` and `[cold]`; a failed arm is a `FAIL` row (rc 1 at the end).
  `BLOOMERY_AB_ORDER=blocks` runs a cross-engine window in engine blocks, each opened by a discarded warm-up process
  (the preheat is then off unless `BLOOMERY_PREHEAT=1`). The `card` witness field prints SM clock, the clock-event
  mask and the power/thermal slowdown counters; `cpu-freq` prints `scaling_cur_freq` over the cores.
- **The line to beat is the reference at its fastest flags, measured interleaved** (`decode-measure.sh` runs ik at
  default and `IK_BEST_FLAGS`); under 1 % is claimable only from `BLOOMERY_AB_IK=1 just ab-decode` rounds.
- **Matching the reference's sum order can be faster and exact at once.** Read the reference kernel before assuming
  SIMD opens a gate.
- **Read chunk skew at profile level 1, never level 2** (level 2's per-row timer tax inflates the skew). A microbench
  µs is not a step µs.
- **Profile the binary you think you are profiling.** `perf record -D` skips startup, not teardown: cut with `--time`.
  Use `-e cpu-clock`.
- **Price a serial cost with a doubling probe, not a perf percentage:** a detached worktree whose one-line patch runs
  the suspect work twice, `just build-decode`, `just ab-decode <that-tree>`; the slowdown is the cost per step (a lower
  bound).

## Known state

- `just lint` runs clippy with `--features gpu`; its ratchet is `grep -c '^warning:'` (one per target a warning appears
  in), held by `tools/lint-ratchet.txt` and tracked in `docs/rust-quality.md` §0. It only goes down; the stage-0 crates
  are gone and the tree has no `undocumented_unsafe_blocks`.
- `gate-qdot`'s hw tests read harness dumps under `$BLOOMERY_DATA/ref/` that `just build-ref` writes.
- A V4.1 prompt batch's shadow runs the card's routed experts by (slot tile, row tile) blocks, one path with no lever:
  `ds41_card_buckets` → `grouped_tiles` (≤ 8 slots) → `ds41_card_gather` → `ds41_expert_gate_up_tiles` /
  `q4k_gemv_tiles`, the down scattering back to each slot; it writes the step's bits (`just gate-gpu-ds41-prefill`).
