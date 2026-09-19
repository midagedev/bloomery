# AGENTS.md — mulle

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
    just gate         # fmt-check + lint + both builds; run before committing

`just gate` excludes the measure targets on purpose: they need a quiet machine
and take a lock, so running them is a separate, deliberate act. It also excludes
`just deny`, which reaches the network for the advisory database.

Build artifacts land in the workspace root `target/`, not under `crates/*/`.
The measure runners read from there and exit rather than fall back, because a
stale binary at an old path is a wrong number, not a missing one.

## Layout

    crates/q3k-gemv/       Q3_K gemv, CUDA-Rust device code (cargo oxide only)
    crates/q3k-cpu/        Q3_K x Q8_K gemv, AVX2 intrinsics, pinned threads
    crates/oxide-ice-unroll/   a compiler-bug reproducer that must NOT compile;
                               excluded from the workspace on purpose
    tools/ref/             C++ harnesses linking ggml: ground truth and baseline
    tools/box.sh           the only way code reaches the workstation
    docs/plan.md           stages, gates, and the machine facts they rest on
    docs/research/         sourced surveys behind the conventions here

## Conventions

- Correctness is defined against ggml, not against intuition: kernels must match
  its output within the relative-error band recorded in each crate's RESULTS file.
- Prose in this repository is Korean; code comments and anything headed upstream
  are English. This file is English because other tools and outside readers read it.
- `rust-toolchain.toml` at the root pins the nightly; it moves only when the
  cuda-oxide pin moves. `cuda-oxide` itself is pinned by `rev` in
  `[workspace.dependencies]`; `just deny` fails if that ever floats.
- Issues live in the self-hosted tracker, project MUL. Measurements are written
  up in the rig-log repository first and linked from the issue.

## Known state, 2026-09-19

`just gate` and `just deny` both pass. `just lint` reports 0 errors and 74
warnings; 54 of those are `undocumented_unsafe_blocks` (53 blocks and one
`unsafe impl`), kept at `warn` because writing 54 safety comments in one pass
would produce 54 rote comments. MUL-10 ratchets it to `deny` as the kernels are
rewritten for stage 1. The remaining 20 are ordinary style lints.

There are no tests yet. The oracle-comparison tests are MUL-12, and
`.config/nextest.toml` already fences hardware tests (`hw_` name prefix) out of
the fast loop so the first test written lands on the right side of that line.
Compile-time levers in `q3k-cpu` are `const` values with dead branches behind
them; MUL-11 converts them to `#[cfg(feature)]` so the on-side also compiles.
