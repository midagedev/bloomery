# AGENTS.md — bloomery

An LLM inference engine for one workstation: Rust host, CUDA-Rust kernels (cuda-oxide today, cutile-rs later), AVX2
kernels for the CPU expert tier. The plan lives in `docs/plan.md`. This file is the working contract, and it holds only
rules in force; their history is in rig-log, the commit log, and this file at `4e7fa8af`.

To run or evaluate bloomery rather than develop it, read the README's "Status" and "Evaluating bloomery"
sections instead: the rules below are for the maintainers' workstation.

## Never

- **Never build a device crate with plain `cargo`.** `crates/gpu` and every crate that links it (`gpu-deepseek41`,
  `gpu-glm5next`, `gpu-vision`, `gpu-gates` with its `gpu` feature) hold `#[cuda_module]`/`#[kernel]` code that needs
  the CUDA codegen backend. Use `cargo oxide` (the recipes do).
- **Never run a gate on the Mac.** The Mac is arm64 and an editor: `qdot` uses `std::arch::x86_64`, `threads` Linux
  affinity calls. On the Mac run only the static tier (`just mac-check`, `just mac-lint`: check and clippy as an
  x86_64-linux cross check, no linker; `just mac-fmt-check`) and the pure crates' native tests (`just mac-test`, the
  crates `tools/recipes.py pure-crates` selects). A Mac result is development-loop evidence; landing evidence is the
  box's record. Every box command goes through `tools/box.sh`, whose rsync is `--delete`: never edit on the box.
- **Never hand-run a benchmark.** Measurements belong to the lease runners in `tools/ref/` (`decode-measure.sh`,
  `ab-decode.sh`, `depth-gpu.sh`, `depth-ds41.sh`, `nsys-gpu.sh`, `ncu-gpu.sh` and their recipes), each taking the
  timing lease through `lease_take`: a machine-wide lock, GPU-idle wait, and witness blocks around every timed region.
  A number produced outside them is not admissible. While the lock is held, `netdata-lease-gate.service` (rig-log
  `configs/`) freezes netdata; `lease_take` prints its state.
- **Both cards are ours.** A compute process on either card that is not one of our rounds is a surprise to
  investigate. **Timed numbers are taken on the A6000** (the 3090 fell off the bus under load, Xid 79): the timing
  runners pin it through `TIMING_GPU` (override `BLOOMERY_TIMING_GPU`), and every witness block opens with the card
  name and power limit. 3090 numbers from before 2026-09-22 and A6000 numbers never share a table. A model that does
  not fit one card may carry a separate "A6000+3090" table: the reference engine gets the same two cards (`-ts`) in the
  same lease, the 3090 stays at its 250 W cap, the witness blocks check for Xid before and after. The 3090 is the
  gate-and-build card: `tools/box.sh` defaults to it, `gpu-power-limit.service` caps it at 250 W, and a compute process
  on it does not stop a timing run (recorded `[other-busy]`; abort with `BLOOMERY_OTHER_STRICT=1`).
  `BLOOMERY_CARD=a6000|both tools/box.sh …` runs functional work on the A6000 and refuses (rc 75) while that card has
  a compute process; the pick reaches the box as `BLOOMERY_BOX_CARD`.
- **Never start a box job longer than 30 minutes without the user's approval.** Estimate the wall first and batch long
  jobs so one approval covers one sitting. A round spec that needs one names the estimate; the lead asks the user.
  Exception: the lead's landing batch of gates through `tools/gate-batch.sh` needs none; it still prints its predicted
  wall before it starts.
- **Never relax a gate or a lint to make it pass.** Raise it with a dated comment and a reason, or file an issue. The
  lint levels in the root `Cargo.toml` carry the hit counts they were chosen from.
- **Never write a number you did not measure.** If it is derived, say so. If it turns out wrong, strike it through and
  correct it in place.

## Commands

The recipes' own lines and the tool headers (`tools/gate-batch.sh`, `tools/recipes.py`, `tools/gpu-gate.sh`,
`tools/box.sh`) are the full reference; what follows is the contract.

    just check / lint / fmt   # check --all-targets (also refreshes Cargo.lock: commit the lock tools/lock-back.sh copies
                              # back with the edit, or the timing runners refuse stale binaries, rc 3); clippy, 0 errors;
                              # cargo fmt with the pinned nightly
    just build-ref            # the C++ reference harnesses that link ggml (rerun when ik moves; gate-qdot reads them)
    just deny                 # cargo deny; fails if the cuda-oxide pin floats; reaches the network
    just affected [BASE|A..B] # the gate-* recipes a change selects, from the crate graph and module trees, the scripts
                              # a recipe names, the cargo globals and each manifest by table; Mac only, builds nothing
    just gate / smoke         # static checks, lint, fast library gates / the round loop's subset (never a landing batch)
    just gate-<name>          # one subsystem's tests on the box, bounded, real exit code
    just gate-ptx-spill       # every PTX entry's spill/jit_local bytes against tools/ref/ptx-shapes.tsv
    just weekly               # every weekly-* recipe in one sitting on the lead's ledger
    just box-gc / box-tracks  # orphans under this track's remote dir / remote dirs vs local worktrees
    just mac-check / mac-lint / mac-fmt-check / mac-test   # the Mac's static tier (above)
    just records-refresh      # tools/bloomery/schema/ and tools/flow/plans/ after a record kind or prompt-call plan change
    just lab-engram           # the engram IO lab's tests; lab-, not gate-, so no engine landing selects it
    just gate-refset / refset-check   # the reference-set readers and every family's sets in place

- **FAIL-first on the box.** `box.sh` syncs by content checksum and does not carry the Mac's mtimes, so a changed file
  gets the box's "now" and cargo rebuilds it. Before trusting a run whose source changed, the build log must show
  `Compiling <crate>`. A Mac `touch` changes no content; to rebuild unchanged source, touch on the box.
- **Bounds and exit codes.** Every `gate-*` recipe runs under `timeout --kill-after=10 900`; the exit code has one
  owner, `tools/gate.sh`. GPU gate binaries run through `tools/gpu-gate.sh`: it takes a card's gate lock, runs under
  the same bound (`BLOOMERY_GATE_BOUND`, whole seconds ≥ 1, parsed by `tools/gate-bound.sh`; any other value ends the
  runner with 64), and returns the binary's code (124/137 timed out, 75 lock contention, 69 a lock file it cannot open).
  `just check-recipes` fails on a test recipe with `||` or a bare `cargo test`, on a `--bin`/`--test`/`-p`/feature
  that names nothing, a script not in the tree, a `gate-*` recipe with no cargo target or one that runs its binary with
  `cargo run`, a bin built without its `required-features`, a recipe that takes a lock itself, and on a Python tool
  under `tools/` whose self-test is not listed.
- **Card locks.** `/root/bloomery-gate.lock` (3090) and `/root/bloomery-gate-a6000.lock`. Under
  `BLOOMERY_CARD=a6000|both` the runner takes that card's lock, or both for `both` (the 3090's first, then the A6000's;
  every other run holds one, so the order cannot deadlock), and refuses (64) a `BLOOMERY_GATE_CARD` naming another card.
  Under box.sh's default pin, `BLOOMERY_GATE_CARD=3090|a6000|any` picks (default `3090`; `any` takes an idle A6000 when
  no timing lease is held, else the 3090). Pass it through `BLOOMERY_BOX_ENV`.
- **Landing batches.** `tools/gate-batch.sh --list FILE` takes `just affected` output and runs it in lanes (A the 3090,
  B the A6000, X alone after both: `[group('solo')]` or `BLOOMERY_CARD=both`); it refuses to start while the timing
  lease is held and refuses a recipe that runs a timing runner. `--ledger` (the lead's) skips an item whose input key
  (`tools/recipes.py key`) is green in `~/.cache/bloomery/gate-ledger.tsv`; `--round-ledger` (a round's) records only in
  `gate-ledger-rounds.tsv`. The lead reads the rounds' file only with `--trust-rounds`, and only for a change that moves
  no behaviour (`just ptx-scan` equal to the base). A rebase moves the key of every item the landed commits touch.
- **Narrowing.** The graph over-selects a host-only change. When `just ptx-scan` equals the base for every bin the
  change reaches, no kernel moved, and the landing batch is the gates that run the changed host path plus the static
  checks; `just affected BASE --narrow --scan BASE_LOG NEW_LOG …` prints that list and why. A kernel, a launch or a byte
  formula moved runs the whole list. Always `just ptx-scan <bin> --features …`.
- **Weekly and opt-in.** `just affected` selects a `weekly-*` recipe only when a changed file matches one of its trigger
  rows in `tools/gate-paths.tsv` or its own text changes. `just stage-gpu-load-v41 [--plan a]` (plan (b) staged on
  both cards, checked byte for byte; solo) is never selected: a change to `crates/model/src/placement.rs`,
  `placement/{workstation,host_lock}.rs`, `crates/gpu/src/{hybrid,weights}.rs` or `gate_load_v41.rs` runs it by name.
- `just gate` excludes the measure targets (they need a quiet machine and a lock) and `just deny`. Build artifacts land
  in the workspace root `target/`; the measure runners read only from there.

## Derive first, measure the gap

A round opens with a prediction: the value and band derived from the cost and error models in `docs/plan.md`
(「모델」), and the proof its change class needs. When the measurement misses, which term was wrong is the finding.

| Change class | Proof | Runtime gate | Timed A/B |
|---|---|---|---|
| move, split, rename | `just ptx-scan` identical; host dispatch code also keeps its structural lines (graph node count, eager = replay, e2e set identical) | none beyond those lines | none |
| delete | ptx-scan of `generate_ds41` and `gate_e2e` = base minus exactly the deleted entries; `gate-ptx-spill` red on exactly those rows before `ptx-shapes.tsv` loses them; per entry, the grep that shows no caller. A remaining entry whose md5 moves needs `tools/ref/ptx-canon.py` = `reordered-only` and equal resource columns | the owning gates of what it touched (bit-identical for a moved md5); a removed gate, case or arm is a coverage change with a dated reason | none |
| add (no engine caller yet) | ptx-scan = base plus exactly the new entries; `gate-ptx-spill` red on exactly them before pinning; the family gate green with FAIL-first per clause | the new family gate, and the owning gates of entries engine bins gained | none |
| integer-path reorder | bit-identical by associativity | the owning gate once | none |
| launch count only | Δt = ΔN × c_node | the owning gate | once, only if occupancy moves too |
| fold a launch into a neighbour | Δt = −ΔN × c_node − the removed kernel's time + the work every host-grid block now repeats or waits on × its blocks (paid in full on a grid near ~700 GB/s) | the owning gate | once |
| instruction count on a kernel near ~700 GB/s | model says 0 — do not open the round | — | — |
| occupancy / geometry | computed from regs, smem, threads, SM count (ptxas `-v`), written in the spec | the owning gate | once |
| float sum order or precision | the error model (σ against `exact-forced-32.tsv`, a diagnostic) | `gate-gpu-e2e` count pin | once |

Two habits precede every round's code. **Decompose on paper:** the terms come from the code (byte, instruction, launch,
loop counts) and numbers already measured; a box run is for the one term derivation cannot fix, named with its expected
value. `lease_take` enforces it: a run without a prediction card
(`BLOOMERY_BOX_ENV='BLOOMERY_LEASE_CARD=docs/cards/<slug>.card'`; format, kinds and exit codes in `tools/ref/card.py`)
gets no lease, an A/B whose effect sits inside the ruler at its round count is refused, and a lease without a runner
goes through `tools/ref/lease-hold.sh` (a raw `flock` on the lease file is not a lease). A runner's child keeps the
lease (fd 9) until it exits; the waiter names every holder within seconds, and every child runs under a `timeout`.
**Draw the resource timeline:** per unit of work, how long the host CPU, card SMs, PCIe, host DRAM and NVMe are each
busy, and whether the wall is their sum or one resource's max; ask whether the flow should change (asynchrony, bulk,
SIMD, SIMT) before a term is shortened. A term under another resource's shadow is worth 0
(worked example: `docs/plan.md` 「흐름 모델」).

The busiest unit is not the bound until the step's cycles are its demand: write every unit's demand next to the step's
cycles in the same unit; when the largest is well under the step, the step is latency the resident warps do not hide,
and a lever on that unit predicts 0 (rig-log 09-26#q3gemma-ab: L1TEX wavefronts −39 %, cycles unmoved).

Measurement comes first only for the named residue: hardware faults (Xid 79), register allocation, cache effects with
no mechanism yet. Build-time facts (node, launch, instruction, register counts) are compile-time ratchets, not runtime
measurements. Every number carries its conditions — `tok/s @ n=N, depth D, card` — or it is not a number.

## Performance first, accuracy opt-in

When a choice trades speed against closeness to the exact result, the default is the faster one, and the engine carries
only that path. A more exact variant lives on the verification side, in the f64 referee (`exact_ref --act f64|ours|ik|ik16`). Bug hunting compares the engine with the simulation of its own rule (`--act ours`): off by more than
σ is a bug candidate, within σ is rounding. Accuracy measures (σ, forced_exact buckets) are diagnostics and do not block
a performance change; the only accuracy pins are bug catchers (the forced_exact count pin, the reference gates).

## Parallel tracks (subagent rounds)

Independent rounds run as git worktrees, each with its own remote directory
(`BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh '...'`). Two trees never share one remote dir. Lease-taking
measurements stay with the main track.

**Every box command waits for a sitting.** `tools/box.sh` runs `lease_guard` (`tools/ref/lease-probe.sh`) first: while
the timing lease or a hold `/root/bloomery-<owner>-hold` is up, the command waits (naming what is up, polling every
30 s, starting after two quiet polls) and exits 75 after `BLOOMERY_BOX_WAIT` seconds (default 1800; 0 does not wait). A
sitting puts its hold up first and exports `BLOOMERY_HOLD_OWNER=<owner>`; of two holds, the later gives way. A command
that builds nothing (`ps`, `tail`, `nvidia-smi`, `just box-gc`) passes with `BLOOMERY_BOX_READONLY=1`, which refuses
cargo, just, make, cmake, ninja or `target/`, syncs nothing, and is refused (66) in a remote dir that is not there.
The lease's one probe is `lease_free`, a shared lock; never probe with an exclusive `flock -n`.

**Batch only the overlap; land the rest as it comes** (user, 2026-10-01). The cost to cut is a gate run that repeats.
Before landing, run `just affected` on each pending piece and sort:
- **Express.** A piece whose gate set is small (predicted wall ≤ 15 min) or disjoint from every other pending piece's
  lands alone as soon as it is Mac-green: its own `tools/gate-batch.sh --ledger`, ff-merge, push.
- **Train.** Only pieces whose sets overlap wait and land together; one run of the union replaces one run per piece
  (`crates/gpu`, `crates/model`, `crates/levers` select 77–119 gates).

A rebase moves only the keys of gates whose closure holds the landed files, so an express landing leaves a train's
greens standing. The box already schedules small runs (one gate lock per card, the timing lease, `any` on an idle
card), so a gate set needs no window of its own; a hold is for a timing sitting only. Rounds run their owning gates
through `--round-ledger`. Inside a gate too, a clause another gate pins, a load a sibling arm repeats, or a gate the
ptx-scan shows untouched is duplicate work to remove.

**Track checklist.**
1. First: `just box-gc` (an orphaned test binary can hold the cargo build lock forever).
2. Run long box commands through `gate-*` recipes or under an explicit `timeout`. box.sh's guard prints `[guard]` lines
   while it waits, so minutes of silence mean a hang on the box: check `just box-gc --dry-run` (it selects by
   `/proc/<pid>/exe`; a `pgrep -f '<dir>'` matches its own shell).
3. Last: `just box-gc`, remove the worktree, then `just box-tracks --remove <track> [<track>-<suffix>…]`. It removes
   only the names given, each only if still stale, never a directory a process runs in; a bare `--remove` is refused.
   An auxiliary dir is `bloomery-<track>-<suffix>` and goes only after its track's worktree.

A quiet agent is a symptom: the first suspect is a hung gate on the box.

## Sessions, tracking and delegation

- **Open work has one owner, `docs/plan-triage.md`.** The self-hosted tracker's **MUL** project holds only stage
  milestones (MUL-1 to MUL-5), machine changes (MUL-6) and upstream work (MUL-7), never a copy of a triage item:
  `GADAK_HOME=$HOME/.gadak gadak --workspace gdk` (on the Mac `/opt/homebrew/bin/gadak`); comments are
  `gadak --workspace gdk comment <KEY> "<body>"` (not `issue comment`). Do not open `TODO.md`.
- **Records.** A round's detail (sitting tables, [derived] predictions, round names) lives in this repo's docs. A
  [rig-log](https://github.com/midagedev/rig-log) `log/` section holds what was measured on this machine and links here;
  read `~/repo/rig-log/CLAUDE.md` before writing one. rig-log is public: grep for `192.168`, `100.` addresses, `.ts.net`
  and `admin` before committing; never BMC addresses, credentials or the box's nvidia-bug-report archive. Disclose AI
  help in prose; rig-log commits pass `tools/check-log.py`.
- **Delegation.** Spec-able rounds go to GLM-5.3 through the `outsource` skill (`--effort max`, claude-code harness, one
  worktree per track); investigation-only rounds may go to agy; opus subagents (`model:"opus"` stated) are the
  exceptions: vision, multi-turn cause narrowing, re-judging a verdict that disagrees with its instrument. Round
  operation is `docs/plan.md` 「라운드 운영」. Copy the relevant clauses of this file into the spec. A delegate's report
  is not evidence: count the tool calls in its transcript and read its worktree's `git status`, then re-run the gates
  under the lead's ownership. Commits and pushes are lead-only.

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
    tools/ref/          C++ harnesses linking ggml, and the lease runners
    tools/bloomery/     the Python side's one reader per fact: records.py, manifest.py
    docs/plan.md        what is live; docs/facts.md machine and toolchain facts; docs/plan-triage.md open items;
                        docs/plan-ledger.md what is closed; docs/research/ sourced surveys

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
  removed by the round that moves the pin.
- **Common code first, so the next model attaches cheaply.** A concept two models share (tier legs, prompt walks,
  seats, placement sizing, drafts, residency glue) has one common owner. A model's own code holds only what an
  architecture fact forces (tensor shapes, the recurrent state, the attention kind, the quant), and its spec names that
  fact. A per-model copy of shared logic stays only for a large measured speed gain, named in the code beside it.
  Lift what two models already do the same way, not what a future model might need. A common owner that grows a
  branch, flag or trait method per model is the wrong cut; the model-specific part goes back to the model.
- **Code shape** is judged by `docs/rust-quality.md` (R1–R29); review reports cite rule numbers.
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
  cargo builds, and `.cargo/cuda-oxide.toml` carries it for oxide builds (`CARGO_ENCODED_RUSTFLAGS` masks the former);
  `just check-rustflags` holds the two equal. The attributes stay.
- **An unprofiled chunk takes no lock and reports nothing.** Per-chunk collectors are for `BLOOMERY_PROFILE` and
  errors; an unconditional lock per chunk is a futex convoy (symptom: a flat thread-scaling curve; first question
  `SWEEP="8 16 24 32" just measure-decode`).
- **No gate guards speed, so a round that touches the dispatch path ends with a same-lease A/B.** Build the base in a
  worktree (`just build-decode`), then `just ab-decode bloomery-<track>`; judge by the interleaved relative numbers (the
  same commit moves ~5 % between windows). The runner rotates arm order every round (a fixed first arm reads 0.3–0.8 %
  slow). `BLOOMERY_AB_ENVS="K=V;K=V"` adds same-binary lever arms; put an A/A arm in for any claim under 1 %.
- **Know the ruler before reading it.** Same-binary runs scatter with SD 0.6 %, so the 95 % interval on a difference of
  two arm means is ±1.0 % at four rounds and ±0.8 % at six; ±0.5 % takes 13 rounds per arm and ±0.3 % takes 32
  (sd·√(2/N) with t(0.975, 2N−2); `tools/ref/card.py` computes it). Report the effect and its interval, not a win
  count. Under 1 %, judge only between same-binary lever arms.
- **Prefill is a headline metric beside decode.** Every model's public numbers carry `pp tok/s @ P = 512 and 4096, card`
  next to decode tok/s, for ours and the reference engines in one lease; a round touching the prompt path is judged on
  it. The measured history is in rig-log. The reference engines' rows live on rig-log's bench page with their flags;
  the README links them and states no ratio against another engine.
- **A decode headline names its depth.** `tools/ref/depth-decode.sh` runs both engines at each depth in one lease
  (`BLOOMERY_DEPTHS="6 1024 4096"`); a round touching attention or the KV cache is judged on the deep rows. For V4.1,
  `just depth-gpu-ds41 prose:<P>[@K=V,…]` / `code:<P>` feed the first P ids of `corpus-prose.ids` / `corpus-code.ids`;
  each corpus's arms are compared only with arms of the same corpus and P, in their own tables. `BLOOMERY_GEN_PLACE=gate`
  runs our arms at `--place gate` on the 3090. Reference arms are preheated (`BLOOMERY_PREHEAT=0` off); rows carry
  `majflt` and `[cold]`; a failed arm is a `FAIL` row (rc 1 at the end). `BLOOMERY_AB_ORDER=blocks` runs a cross-engine
  window in engine blocks, each opened by a discarded warm-up process (the preheat is then off unless
  `BLOOMERY_PREHEAT=1`). The `card` witness field prints SM clock, the
  clock-event mask and the power/thermal slowdown counters; `cpu-freq` prints `scaling_cur_freq` over the cores.
- **The step does no load-time work.** Anything that does not depend on the tokens is resolved once into `Derived`.
  `just gate-alloc` counts allocator calls per steady step and only ratchets down. Activation blocks come from
  `Tensor2::scratch` when every cell is written; `BLOOMERY_POISON=1` turns a missed cell into a NaN the gates catch.
- **Read chunk skew at profile level 1, never level 2** (level 2's per-row timer tax inflates the skew). A microbench
  µs is not a step µs.
- **Profile the binary you think you are profiling.** box.sh builds nothing: run `just build-decode` first. `perf record -D`
  skips startup, not teardown: cut with `--time`. Use `-e cpu-clock`.
- **Price a serial cost with a doubling probe, not a perf percentage:** a detached worktree whose one-line patch runs the
  suspect work twice, `just build-decode`, `just ab-decode <that-tree>`; the slowdown is the cost per step (a lower
  bound).
- **The line to beat is the reference at its fastest flags, measured interleaved** (`decode-measure.sh` runs ik at
  default and `IK_BEST_FLAGS`); under 1 % is claimable only from `BLOOMERY_AB_IK=1 just ab-decode` rounds.
- **Matching the reference's sum order can be faster and exact at once.** Read the reference kernel before assuming SIMD
  opens a gate.
- **Toolchain pins.** `rust-toolchain.toml` pins the nightly and moves only with the cuda-oxide pin. `cuda-oxide` is
  pinned by `rev` in `[workspace.dependencies]` and taken from our fork's `bloomery` branch through `[patch]` (patches
  listed in `THIRD_PARTY_NOTICES.md`); `just deny` fails if either floats. Before the fork's branch moves, the old pin
  gets the tag `pin/<short rev>`; pin tags are never deleted. The box never rebuilds the backend in place: `box.sh`
  exports `CUDA_OXIDE_BACKEND=~/.cargo/cuda-oxide-bloomery/<rev>/librustc_codegen_cuda.so`, and `cargo oxide` stops
  (rc 70) when that backend is missing or its `source-rev.txt` names another commit. A pin move that leaves the backend
  crate's closure untouched may reuse the previous backend (md5 and reason in that directory's `PROVENANCE`). A fork
  patch that changes codegen is a pin move (every gate); one that must not proves it with identical `just ptx-scan`
  tables of `generate_ds41` and `gate_e2e`.
- **Runtime levers** are the rows of `crates/levers/src/registry.rs` and nothing else documents them. A binary parses the
  `parsed` rows first thing in `main` (`bloomery_levers::at_main`), refuses a value its kind does not take, and hands
  typed values down; `<bin> --levers` prints them. An `in place` row is a line of `tools/levers-direct.txt` with the
  round that converts it (`just check-levers`). A retired name that is set is refused by name. `just gate-levers` prints
  the table.

## Known state

- `just lint` runs clippy with `--features gpu`; its ratchet is `grep -c '^warning:'` (one per target a warning appears
  in), held by `tools/lint-ratchet.txt` and tracked in `docs/rust-quality.md` §0. It only goes down; the stage-0 crates
  are gone and the tree has no `undocumented_unsafe_blocks`.
- `gate-qdot`'s hw tests read harness dumps under `$BLOOMERY_DATA/ref/` that `just build-ref` writes.
- Timed recipes (`time-gpu-*`, `prof-gpu-p8`, `bench-gpu-kernels`) run on the A6000 through `tools/ref/timing-card.sh`.
- A V4.1 prompt batch's shadow runs the card's routed experts by (slot tile, row tile) blocks, one path with no lever:
  `ds41_card_buckets` → `grouped_tiles` (≤ 8 slots) → `ds41_card_gather` → `ds41_expert_gate_up_tiles` /
  `q4k_gemv_tiles`, the down scattering back to each slot; it writes the step's bits (`just gate-gpu-ds41-prefill`).
