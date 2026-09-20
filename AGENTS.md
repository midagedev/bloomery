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
- **Never touch the A6000.** It serves models and runs nightly training. The
  dev card is the 3090, pinned by `CUDA_VISIBLE_DEVICES` in the box env.
- **Never relax a gate or a lint to make it pass.** Raise it with a dated
  comment and a reason, or file an issue. The lint levels in the root
  `Cargo.toml` carry the hit counts they were chosen from.
- **Never write a number you did not measure.** If it is derived, say so. If it
  turns out wrong, strike it through and correct it in place; do not silently edit.

## Commands

    just check        # cargo check --workspace --all-targets
    just lint         # cargo clippy --workspace; contract is 0 errors
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
   the box, not thinking: check `pgrep -fa '<remote dir>'`.
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
- **No gate guards speed, so a round that touches the dispatch path ends with
  a same-lease A/B.** Build the base commit in a worktree (`just build-decode`
  there), then `just ab-decode bloomery-<track>` from the changed tree; judge
  by the interleaved relative numbers only — the same commit moves ~5 % between
  windows. Bit-identical is not speed-identical: an expression moved from a
  write-into-zeros loop to `map().collect()` doubled its site.
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

`just lint` reports 0 errors and 114 warnings (measured 2026-09-20; 74 on
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
rollback), `BLOOMERY_GATE_BOUND` (tools/gate.sh). Each is read once, at first use.
