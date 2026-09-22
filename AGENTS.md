# AGENTS.md — bloomery

An LLM inference engine for one workstation: Rust host, CUDA-Rust kernels
(cuda-oxide today, cutile-rs later), AVX2 kernels for the CPU expert tier.
Stages and gates live in `docs/plan.md`. This file is the working contract.

## Never

- **Never build a device crate with plain `cargo`.** `q3k-gemv` contains
  `#[cuda_module]`/`#[kernel]` code that needs the CUDA codegen backend.
  `cargo build -p q3k-gemv` produces nothing usable. Use `cargo oxide`.
- **Never run a gate on the Mac.** The Mac is arm64 and an editor; `q3k-cpu`
  uses `std::arch::x86_64` and Linux affinity calls and does not compile there.
  Every command below goes through `tools/box.sh`, which rsyncs the tree to the
  workstation and runs there. That rsync is `--delete`: never edit on the box.
- **Never hand-run a benchmark.** Measurements belong to `tools/ref/measure.sh`
  (GPU) and `tools/ref/cpu-measure.sh` (CPU). They own the quiet-machine
  protocol: a machine-wide lock, GPU-idle wait, and witness blocks around every
  timed region. A number produced outside them is not admissible.
- **Both cards are ours** (user, 2026-09-22: "gpu 2개 다 쓰기로 했으니"). Nothing
  else holds the A6000 any more — the nightly vocoder training ended 2026-09-21
  and `llm.service` is inactive and disabled — so a compute process on either
  card that is not one of our rounds is a surprise to investigate, not a tenant
  to yield to. **Timed numbers are taken on the A6000** (user, 2026-09-22 —
  the 3090 fell off the bus twice that day under load, Xid 79 at 01:51 and
  21:38 UTC; the timing runners `tools/ref/depth-gpu.sh`, `nsys-gpu.sh`,
  `ncu-gpu.sh` pin it through `TIMING_GPU`, override with
  `BLOOMERY_TIMING_GPU`, and every witness block opens with the card name and
  power limit). The ik baselines are re-measured there; **3090 numbers from
  before that date and A6000 numbers never share a table**. The 3090 is the
  gate-and-build card: `tools/box.sh` defaults to it (the box env pin), it is
  capped at 300 W by a systemd oneshot (`gpu-power-limit.service`), and a
  compute process on it does not stop a timing run — the runner records it as
  `[other-busy]` (abort with `BLOOMERY_OTHER_STRICT=1`). `BLOOMERY_CARD=a6000|
  both tools/box.sh …` still exists for functional runs on the A6000 and
  refuses (rc 75) while that card has a compute process — i.e. while a timing
  run holds it.
- **Never start a box job longer than 30 minutes without the user's approval**
  (user, 2026-09-22). Estimate the wall time before launching — full-set CPU
  simulations, truth-file builds, depth sweeps, timing tables — and batch such
  jobs so one approval covers one sitting of box time. A round spec that needs
  one says so and names the estimate; the round waits for the lead, and the
  lead asks the user. Gates and builds that finish inside 30 minutes need no
  approval. (Context: the error-source round queued ~2 h of CPU simulation,
  50 min of it a duplicate arm, without anyone asking.)
- **Never relax a gate or a lint to make it pass.** Raise it with a dated
  comment and a reason, or file an issue. The lint levels in the root
  `Cargo.toml` carry the hit counts they were chosen from.
- **Never write a number you did not measure.** If it is derived, say so. If it
  turns out wrong, strike it through and correct it in place; do not silently edit.

## Commands

    just check        # cargo check --workspace --all-targets — also the step that
                      # refreshes Cargo.lock after a dependency edit — `cargo oxide
                      # build` does not rewrite the lock (lock md5 unchanged
                      # across oxide builds in the gates-v2 round)
    just lint         # cargo clippy --workspace; contract is 0 errors
                      # FAIL-first on the box: `box.sh` syncs by content checksum
                      # and does NOT carry the Mac's mtimes, so a changed file
                      # gets the box's own "now" and cargo rebuilds it. (It used
                      # to carry them: a restore handed back an older mtime, and
                      # even a `touch` lost to the box clock running 4.1 s ahead
                      # of the Mac — the MUTATED binary was served twice.)
                      # Before trusting a run whose source changed, the build
                      # log must still show `Compiling <crate>`. The flip side
                      # (2026-09-21): a Mac `touch` changes no content, so the
                      # box never sees it — to force a rebuild of UNCHANGED
                      # source, touch on the box (`box.sh 'touch <file> && …'`).
    just fmt          # cargo fmt --all
    just build-gpu    # cargo oxide build --arch sm_86 -- -p q3k-gemv
    just build-cpu    # release build with -C target-cpu=znver3
    just build-ref    # the C++ reference harnesses that link ggml
    just deny         # cargo deny check; fails if the cuda-oxide pin ever floats
    just measure-gpu  # quiet-machine GPU measurement, witnesses included
    just measure-cpu  # same for the CPU tier, serialized by a file lock
    just gate         # check-recipes + fmt-check + lint + both builds + gate-1-1
    just gate-<name>  # one subsystem's tests on the box, bounded, real exit code:
                      # ops attn ffn moe head forward kv derived mt profile
                      # threads qdot prompts 1-1
    just box-gc       # kill orphan processes under this track's remote dir
    just box-tracks   # remote track dirs vs local worktrees; --remove deletes stale ones

`just gate` excludes the measure targets on purpose: they need a quiet machine
and take a lock, so running them is a separate, deliberate act. It also excludes
`just deny`, which reaches the network for the advisory database.

Build artifacts land in the workspace root `target/`, not under `crates/*/`.
The measure runners read from there and exit rather than fall back, because a
stale binary at an old path is a wrong number, not a missing one.

Every `gate-*` recipe runs its `cargo test` under
`timeout --kill-after=10 900`: warm runs take seconds, cold builds a few
minutes, and a gate that hangs must fail loudly in fifteen minutes rather than
hang a pipeline forever. (2026-09-20: an infinite loop in a first-draft chunk
walk hung `gate-mt` past an hour; two parallel agents died as "inactive"
watching it, and orphaned test processes piled up on the box until they were
found by hand. The bound, `box-gc`, and the checklist below all come from that
incident. The first form of the bound was `cargo test … || echo "TIMED OUT"`,
which made every gate exit 0 whether it passed, failed or timed out; it was
caught in review the same evening, and a 13-gate rerun found no red hidden in
that window. The exit code now has one owner, `tools/gate.sh`, and
`just check-recipes` fails on a test recipe that carries `||` or a bare
`cargo test`.)

## Derive first, measure the gap (2026-09-22)

A round opens with a prediction, not only an end number: the value and band
derived from the cost and error models in `docs/plan.md` (section 「모델」),
and the proof its change class needs. Measurement checks the prediction; when
it misses, which term of the model was wrong is the round's finding. Looking
back, several closed rounds were arithmetic before they were code — A4d's 8.3 %
occupancy is 4 warps of 48 on the SMs its 18 blocks reach; A3f's f16 lever
could not win on a kernel already at ~700 GB/s.

| Change class | Proof | Runtime gate | Timed A/B |
|---|---|---|---|
| move, split, rename (semantics kept) | `just ptx-scan` table identical — that proves the kernels only; host dispatch code also needs its structural lines unchanged (graph node count, eager = replay, e2e set identical) | none beyond those structural lines | none |
| integer-path reorder | bit-identical by associativity | the owning gate once | none |
| launch count only | Δt = ΔN × c_node, predicted | the owning gate | once, only if occupancy moves too |
| instruction count on a kernel near ~700 GB/s | model says 0 — do not open the round | — | — |
| occupancy / geometry | occupancy computed from regs, smem, threads, SM count (ptxas `-v`) and written in the spec | the owning gate | once, to confirm |
| float sum order or precision | the error model (σ against `exact-forced-32.tsv`, a diagnostic — see the next section) | `gate-gpu-e2e` count pin | once, to confirm |

Measurement comes first only for the named residue: hardware faults (Xid 79),
compiler register allocation, cache effects with no mechanism yet. Anything
fixed at build time (node count, launch count, per-kernel instructions and
registers) is a compile-time ratchet and is not re-measured at runtime.

Every number carries its conditions — `tok/s @ n=N, depth D, card` — or it is
not a number: the n=32 vs n=96 ratios lost on 2026-09-22 were a units error.

## Performance first, accuracy opt-in (user, 2026-09-22)

When a choice trades speed against closeness to the exact result, the default
is the faster one, and the engine carries only that path. A more exact variant
is not built into the engine as a second path; it lives on the verification
side, in the f64 referee (`exact_ref --act f64|ours|ik|ik16` simulates each
rounding rule). Bug hunting compares the engine with the simulation of its own
rule (`--act ours`): a position off by more than σ is a bug candidate, one
within σ is rounding. Accuracy measures (σ, forced_exact buckets) are
diagnostics that tell how far the default sits from the truth; they do not
block a performance change. The only accuracy pins are bug catchers — the
existing forced_exact count pin and the reference gates — not a precision
ranking to ratchet down. Case that set it: the 32-value activation block
(errsrc) would bring σ from 0.378 to ~0.26, ik's level, but costs 4× the
activation scales, a re-pin of every bit gate and a CPU/GPU rule split; an
opt-in GPU path was considered and dropped (a second path needs its own gate
to not rot), so 32-value exists only as `exact_ref --act ik`.

## Parallel tracks (subagent rounds)

Independent rounds run as git worktrees, each with its own remote directory via
`BLOOMERY_REMOTE='~/repo/bloomery-<track>' tools/box.sh '...'` — the protocol
box.sh's header already reserved. Lease-taking measurements (decode-measure,
profile-of-record) stay with the main track: the lease is machine-wide, so
measurement is serialized by design.

Track checklist, first and last:

1. **First**: `just box-gc` — clear anything a previous track left under this
   remote dir. An orphaned test binary can hold the cargo build lock and make
   every later command wait forever.
2. **Always** run long box commands through the `gate-*` recipes or with an
   explicit `timeout` — never bare `cargo test` at a prompt you are not
   watching. If a command produces no output for minutes, assume it is hung on
   the box, not thinking: check `just box-gc --dry-run` (it selects by
   `/proc/<pid>/exe`, never by cmdline — a `pgrep -f '<dir>'` also matches the
   shell that runs it).
3. **Last**: `just box-gc` again, remove the worktree, then `just box-tracks
   --remove` — a removed worktree leaves its remote directory (and a
   `target/` of several hundred MB) behind on the box.

A quiet agent is a symptom, not a state: when a subagent goes inactive, the
first suspect is a hung gate on the box, not the agent.

## Layout

    crates/gguf/           GGUF reader, dequant, per-weight-type activation format
    crates/threads/        resident pinned worker pool (spin, then park)
    crates/qdot/           fused quantized dot kernels, AVX2: Q3_K x q8_K,
                           Q4_K/Q5_0/Q5_1/Q6_K x q8_2_x4, Q8_0 x Q8_0 cells
    crates/model/          the engine: ops (matmul_q), attn (MLA, flash), ffn,
                           moe, head, forward, kv, derived, profile; bin
                           bloomery-decode; tests/ are the gates
    crates/q3k-gemv/       stage 0: Q3_K gemv, CUDA-Rust device code (cargo oxide only)
    crates/q3k-cpu/        stage 0: Q3_K x Q8_K gemv, AVX2 intrinsics, pinned threads
    crates/oxide-ice-unroll/   a compiler-bug reproducer that must NOT compile;
                               excluded from the workspace on purpose
    tools/ref/             C++ harnesses linking ggml: ground truth and baseline
    tools/box.sh           the only way code reaches the workstation
    tools/gate.sh          the gate runner: 900 s bound, cargo's own exit code
    docs/plan.md           stages, gates, and the machine facts they rest on
    docs/research/         sourced surveys behind the conventions here

## Conventions

- Correctness is defined against ggml, not against intuition: kernels must match
  its output within the relative-error band recorded in each crate's RESULTS file.
- **GPU-stage toolchain defects and gaps (cuda-oxide, cutile-rs's cuda-core/cuda-bindings) go
  into `docs/upstream/nvlabs-ledger.md` the moment they are met** — one line, before any
  workaround is written; a workaround erases the evidence. The user wants these as upstream
  issues/PRs; the ledger is where the lead triages them.
- **Code shape is judged by `docs/rust-quality.md`** (numbered rules R1–R22: unsafe scope,
  unit-carrying types, error enums per crate, clippy ratchet, refactor = bit-identical gates +
  same-lease A/B). Review reports cite rule numbers; the warning baseline in Known state only
  goes down.
- Prose in this repository is Korean; code comments and anything headed upstream
  are English. This file is English because other tools and outside readers read it.
- **Comments state what is true now, not how we got here.** Keep: `// SAFETY:`
  (the invariant, one to three lines); one line of *why* for a non-obvious
  choice; "this order is the gate" on a load-bearing float reduction; the
  contract a caller must meet. Remove: issue numbers, dates, measured
  ms/GB/s/tok/s, "until round N this was…", how a bug was found. That history
  lives in rig-log and the commit message; a comment repeating it is dual
  bookkeeping that rots. One exception, and it is the existing contract: a
  re-pinned gate constant (band, tolerance, `KNOWN_DIVERGENCE`) keeps its dated
  one-line attribution, marked `PIN(YYYY-MM-DD):`. `tools/check-comments.sh`
  fails on an issue number or a date in `crates/*/src` outside a `PIN` line.
- **Tests are the gates, and only the gates.** One test per contract; shared
  harness code lives in `tests/common`. No print-only tests, no second test
  pinning what another already pins. Removing a gate is a coverage change and
  needs the same dated reason as relaxing one.
- **Do not split a `#[target_feature]` kernel body into helpers** (measured:
  10-13 % loss). Orchestration code is ordinary Rust: a function that no longer
  fits on two screens gets split.
- **An unprofiled chunk takes no lock and reports nothing.** Per-chunk
  collectors (`Mutex<Vec<_>>`) are for `BLOOMERY_PROFILE` and for errors only:
  balanced chunks finish together, so an unconditional lock at the end of each
  is a futex convoy per dispatch. The symptom is a flat thread-scaling curve
  (8 threads as fast as 32) — `SWEEP="8 16 24 32" just measure-decode` is the
  first question to ask of any dispatch-path slowness.
- **No gate guards speed, so a round that touches the dispatch path ends with
  a same-lease A/B.** Build the base commit in a worktree (`just build-decode`
  there), then `just ab-decode bloomery-<track>` from the changed tree; judge
  by the interleaved relative numbers only — the same commit moves ~5 % between
  windows. Bit-identical is not speed-identical: an expression moved from a
  write-into-zeros loop to `map().collect()` doubled its site. The runner
  rotates arm order every round and prints per-arm means: in a fixed order the
  round's first arm read 0.3–0.8 % slow (same-binary A/A), enough to flip a
  small verdict. `BLOOMERY_AB_ENVS="K=V;K=V"` adds same-binary arms that differ
  only by a lever — put an A/A arm in whenever the claim is under 1 %.
- **Know the ruler before reading it.** Same-binary runs scatter with SD
  0.6 % (18 runs, 2026-09-21), so the 95 % interval on a difference of two arm
  means is ±1.0 % at four rounds and ±0.8 % at six; ±0.5 % takes about 23
  rounds per arm, ±0.3 % about 63. Report the effect and its interval, not a
  win count — 4/4 is p = 0.06 by the sign test. Under 1 %, judge only between
  same-binary lever arms: two builds differ by link layout alone.
- **A decode headline names its depth.** tg96 after a 6-token prompt measures
  the n → 0 end of attention. `tools/ref/depth-decode.sh` runs both engines at
  each depth in one lease (`BLOOMERY_DEPTHS="6 1024 4096"`, ik via
  `llama-bench -gp d,96`); 2026-09-21: +1.7 % at depth 6, −15 % at 1024, −39 %
  at 4096 (3.4 vs 0.87 µs per cached key per step). Any round that touches
  attention or the KV cache is judged on the deep rows too.
- **The step does no load-time work.** Anything that does not depend on the
  tokens — tensor lookups, names, metadata keys, views, decoded gains — is
  resolved once into `Derived`; the calling thread's serial time is the step's
  length because every worker waits on it. `just gate-alloc` counts allocator
  calls per steady step and only ratchets down. Activation blocks come from
  `Tensor2::scratch` (no zero fill) when every cell is written;
  `BLOOMERY_POISON=1` turns a missed cell into a NaN the gates catch.
- **Read chunk skew at profile level 1, never level 2.** Level 2 times every
  row, and the timer tax scales with row count — it inflated "slowest chunk vs
  mean" from 11 % to 20–40 % and a whole round was aimed at the difference.
  `span ms` / `slowest ms` and the per-chunk line print at level 1 for this. A
  microbench µs is not a step µs either: the pool bench went 4.56 → 1.61 µs
  per dispatch and decode did not move.
- **Profile the binary you think you are profiling.** `tools/box.sh` syncs
  source and builds nothing; run `just build-decode` first. `perf record -D`
  skips startup, not teardown — cut the report with `--time`, or the
  `munmap` of the populated mapping reads as step cost. Use `-e cpu-clock`
  (the default IBS event misattributes symbols on this CPU).
- **Price a serial cost with a doubling probe, not a perf percentage.** A
  detached worktree, a one-line patch that runs the suspect work twice, `just
  build-decode` there, then `just ab-decode <that-tree>`: the slowdown is the
  work's cost per step (a lower bound — the second run is cache-warm). perf put
  caller-side quantization at 14–15 % of the main thread twice; the probe
  priced it at 1.3 %. Do this before spending a round on the item.
- **The line to beat is the reference at its fastest flags, measured
  interleaved.** `decode-measure.sh` runs ik twice (default, and
  `IK_BEST_FLAGS`); a difference under 1 % is only claimable from
  `BLOOMERY_AB_IK=1 just ab-decode` rounds, never from one headline.
- **Matching the reference's sum order can be faster and exact at once.** The
  F32 router dot and `rms_norm` were scalar "to keep the bits"; the reference
  sums in lanes (`dot_f32`) and in f64 (`sum_sq_f64`), and porting that order
  took both to max|diff| 0 against the oracle. Read the reference kernel before
  assuming SIMD opens a gate.
- `rust-toolchain.toml` at the root pins the nightly; it moves only when the
  cuda-oxide pin moves. `cuda-oxide` itself is pinned by `rev` in
  `[workspace.dependencies]`; `just deny` fails if that ever floats.
- Issues live in the self-hosted tracker, project MUL. Measurements are written
  up in the rig-log repository first and linked from the issue.

## Known state, 2026-09-20

All 13 subsystem gates pass on main with real exit codes (rerun after the
`tools/gate.sh` fix). Gate tests are `hw_`-prefixed and `#[ignore]`d: a plain
`cargo test` runs almost nothing by design, and `.config/nextest.toml` fences
them out of the fast loop. The `just gate-*` recipes are how they run.

`just lint` runs clippy **with `--features gpu`** (without it the GPU gpu-gates binaries
are dummy mains and their bodies are never linted). The ratchet counter is
`grep -c '^warning:'` on that output, which counts a warning once per target it
appears in (an agent's unique-count will read lower — same direction, different
ruler). 2026-09-22 night, measured: 308 before the quality rounds; 234 after
`gpusafety`, 305 after `gatesdedup` on its own base — the merged value is re-measured
and written here when a round lands — 221 on main `0ff785e` after the four rounds, **211 on main `48ee5c2`** (after fnsplit).
`gate-qdot`'s four hw tests read `$BLOOMERY_DATA/ref/*-ik-dot.txt`; `just build-ref` writes
them (`tools/ref/build-qdot-ref.sh` builds and runs the x4 harnesses) — before 2026-09-22
no recipe did, and the four were a standing red on every tree. Rerun `build-ref` when ik
moves. `docs/rust-quality.md` §0 is the table that tracks the lint count. Timed recipes (`time-gpu-*`, `prof-gpu-p8`, `bench-gpu-kernels`) run on the
A6000 since the runners share `tools/ref/timing-card.sh`; their earlier 3090 numbers do
not belong in the same table.

Older baselines, kept for the slope: 144 warnings on main (2026-09-21 morning;
169 on the `gpu-p0` tree with `crates/gpu` + `crates/gpu-spike` as members; 114 on 2026-09-20; 74 on
09-19). 81 are in the stage-0 crates (q3k-gemv 60, q3k-cpu 21); the engine
crates carry 33 (model 16, qdot 13, gguf 4). 63 are
`undocumented_unsafe_blocks` (q3k-gemv 49, q3k-cpu 10, qdot 4), kept at `warn`;
MUL-10 ratchets it to `deny`. New engine code should not add to that count.

No `RUSTFLAGS`/`.cargo/config` enables AVX2 globally: every kernel's
`#[target_feature]` is load-bearing. A helper that touches `_mm*` intrinsics
is either `#[inline(always)]` (it inherits the caller's features) or carries
the attribute itself; with neither it compiles and runs tens of times slower.

Compile-time levers in `q3k-cpu` are `const` values with dead branches behind
them; MUL-11 converts them to `#[cfg(feature)]` so the on-side also compiles.
Runtime levers: `BLOOMERY_THREADS`, `BLOOMERY_SPIN` (threads),
`BLOOMERY_PROFILE` (model::profile), `BLOOMERY_FLASH_SIMD=0` (attn, scalar
rollback), `BLOOMERY_GATE_BOUND` (tools/gate.sh), `BLOOMERY_FLASH_SEG` (gpu flash: keys per
segment, a multiple of 32; an unusable value panics instead of falling back).
Each is read once, at first use.
