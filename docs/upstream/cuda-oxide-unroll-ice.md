<!-- Upstream issue draft for NVlabs/cuda-oxide — NOT filed. Lead pastes and
     submits after review. Everything below is measured on our box; the
     reproducer was reduced with AI assistance and every compile claim was
     re-run and verified by hand on the machine. -->

# Title

`#[unroll]` on a `while` loop with a `usize` counter: mixed-width shift constant-folding panics with `APInt::shl: bitwidth mismatch` (full and partial unroll both affected)

## Environment

- cargo-oxide 0.2.1, cuda-oxide git deps at `b9847e9515ed3a23096f22567d3eaf0a6e3e440c` (HEAD: "fix(codegen): preserve assertions during loop unrolling (#1293)")
- pliron at `edd41fe`
- rustc 1.100.0-nightly (e457a7b0d 2026-08-27) — nightly-2026-08-28
- CUDA 13.0 (V13.0.88), RTX 3090 (sm_86), driver 615.71.09
- Linux, standalone project scaffolded by `cargo oxide new`

## Minimal reproducer

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(domain = 1, block = (32, 1, 1))]
    pub fn ice(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        if let Some(o) = out.get_mut(tid) {
            let mut i: usize = 0;
            #[unroll]
            while i < 1 {
                *o = 1u32 << i;
                i += 1;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 32)?;
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx)? };
    let prepared = module.prepare_ice(LaunchConfig1D::new(1, 32, 0))?;
    module.ice(&stream, &prepared, &mut out_dev)?;
    let out_host = out_dev.to_host_vec(&stream)?;
    assert!(out_host.iter().all(|&v| v == 1));
    Ok(())
}
```

`cargo oxide build --arch sm_86`

## Expected

The kernel compiles. Rust legalizes mixed-width shifts — `u32 << usize` is
well-formed (the shift amount may be any integer type), and lowering already
handles the width mismatch (`convert_shift` in
`crates/mir-lower/src/convert/ops/arithmetic.rs:477` zext/truncs and masks the
amount).

## Actual

Device codegen dies with an internal compiler error:

```text
error: [rustc_codegen_cuda] Internal compiler error in device codegen: assertion `left == right` failed: APInt::shl: bitwidth mismatch (32 vs 64)
         left: 32
        right: 64. This is a bug in cuda-oxide. Please file at https://github.com/NVlabs/cuda-oxide/issues
error: could not compile `oxide-ice-unroll` (bin "oxide-ice-unroll") due to 1 previous error
```

With `RUST_BACKTRACE=1`, the deciding frames:

```text
   8: <pliron::utils::apint::APInt>::shl
             at pliron .../src/utils/apint.rs:180:9
   9: <dialect_mir::ops::arithmetic::MirShlOp as pliron::opts::constants::ConstFoldInterface>::check_fold
             at crates/dialect-mir/src/const_fold.rs:200:64
  10-13: pliron::opts::constants::sccp::{process_fold_op, process_op, process_block, sccp}
  14: mir_transforms::unroll::unroll_annotated_loops
             at crates/mir-transforms/src/unroll.rs:387:13
```

## What one-variable-at-a-time interventions show

| variant | compiles | note |
|---|---|---|
| baseline above | **no** — `APInt::shl: bitwidth mismatch (32 vs 64)` | |
| counter `usize` → `u32`, nothing else changed | **yes** | the single-change flip |
| `#[unroll]` removed | **yes** | |
| `<< i` → `>> i` (`0x8000_0000u32 >> i`) | **no** — `APInt::lshr: bitwidth mismatch (32 vs 64)` | `MirShrOp` has the same hole |
| `1u32 << i` → `(i + 1) << 2u32` | **no** — `APInt::shl: bitwidth mismatch (64 vs 32)` | both width directions panic |
| amount routed through a cast (`i as u32` / `i as u64`) | **yes** | the cast's result is not a constant attr, so the folder never sees both operands constant |
| `#[unroll(4)]` with a **runtime** trip count (`n: usize` param) | **no** — same panic, same backtrace | not limited to constant trip counts |
| no loop: `let k: usize = 0; *o = 1u32 << k;` | **yes** | the panic needs the unroll pass's SCCP, not constants per se |
| upstream `unroll_smoke` example unchanged | **yes** | control |
| independent reproducer (`usize` counter, 8 trips, accumulator): `i << 1u64` vs `i << 1u32` | `1u64` **yes**, `1u32` **no** | with counter and `#[unroll]` held fixed, only the amount's width flips it; note a bare `1` literal in that position is `i32` |

## Suspected location

`crates/dialect-mir/src/const_fold.rs`:
- `MirShlOp::check_fold` — `lhs.value().shl(&rhs.value())` at line 200
- `MirShrOp::check_fold` — `lhs.value().ashr(...)` / `.lshr(...)` at lines 223/225

`shift_in_range` (line 187) validates the amount's *value* but not its
*bitwidth*; pliron's `APInt::shl/lshr/ashr` assert `lhs.bw() == rhs.bw()`.

Why `#[unroll]` is the trigger: the unroll pass is the only place that makes
both operands of such a shift simultaneously constant. Full unrolling
materializes the counter as literals of the counter's width (`make_const`
call at `crates/mir-transforms/src/unroll.rs:975`), and after any unroll the
pass runs SCCP over the whole function (`crates/mir-transforms/src/unroll.rs:387`).
SCCP's optimistic lattice evaluates loop phis at their initial constants
while iterating, which is why `#[unroll(N)]` on a runtime-trip-count loop
panics even though the counter is never a compile-time constant in the final
IR. Since Rust permits mixed-width shifts, a `mir.shl`/`mir.shr` with
operands of different widths is legal IR (lowering handles it), so the folder
must be total over it — currently it panics instead.

A possible fix, mirroring `convert_shift`'s semantics: in `check_fold`,
resize the amount to the value's width (keeping the existing
`shift_in_range` rejection for amounts that do not fit / are out of range)
before calling the APInt op — or simply decline to fold when the widths
differ.

## Workaround

Use a `u32` counter in `#[unroll]` loops (verified to build and run
correctly on sm_86), or route the shift amount through an `as` cast, which
blocks constant propagation to the shift.

## Disclosure

The reproducer was reduced with AI assistance; every compile claim in this
report was re-run and verified by hand on the machine above.
