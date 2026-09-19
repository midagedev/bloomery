# `#[unroll]` on a `usize`-counter loop with a mixed-width shift ICEs: `APInt::shl: bitwidth mismatch (64 vs 32)`

## Environment

- cuda-oxide at `b9847e95` (`fix(codegen): preserve assertions during loop unrolling (#1293)`),
  `cuda-device`/`cuda-host` at the same commit; `cargo-oxide 0.2.1`
- `rustc 1.100.0-nightly (e457a7b0d 2026-08-27)` (toolchain `nightly-2026-08-28`)
- LLVM 21.1.8, CUDA 13.0 (V13.0.88), driver 615.71.09, RTX 3090, `--arch sm_86`

## Minimal reproducer

`cargo oxide new` project with this kernel (`cargo oxide build --arch sm_86`):

```rust
#[kernel]
#[launch_bounds(256)]
#[launch_contract(domain = 1, block = (256, 1, 1))]
pub fn unroll_usize(mut out: DisjointSlice<u32>) {
    let tid = thread::index_1d();
    if let Some(out_elem) = out.get_mut(tid) {
        let mut acc: u32 = 0;
        let mut i: usize = 0;
        #[unroll]
        while i < 8 {
            acc = acc.wrapping_add((i << 1) as u32);
            i += 1;
        }
        *out_elem = acc;
    }
}
```

The shift is `usize << i32` (the literal `1` defaults to `i32`), i.e. a
shift with mixed 64/32-bit operands inside the unrolled loop.

## Expected vs actual

Expected: the loop unrolls and the kernel compiles (as the `u32`-counter
version does — it folds to a constant store and runs correctly).

Actual (exit 101):

```text
error: [rustc_codegen_cuda] Internal compiler error in device codegen: assertion `left == right` failed: APInt::shl: bitwidth mismatch (64 vs 32)
error: could not compile `oxide-ice-unroll` (bin "oxide-ice-unroll") due to 1 previous error
```

Interventions (each a single change to the kernel above; full table with
outputs available on request):

- `usize` -> `u32` counter: compiles, runs, correct (`PASSED` on GPU).
- `i << 1u64` (same-width amount): compiles; `i << 1u32`: ICEs.
  With counter type and `#[unroll]` held fixed, only the amount's type
  flips the outcome.
- Runtime (non-constant) shift amount, or no `#[unroll]`: compiles.
- `1u32 << i` with `i: usize`: ICEs with `(32 vs 64)`; `(i >> 1)`:
  ICEs with `APInt::lshr: bitwidth mismatch (64 vs 32)` — both shift
  directions and both operand orders are affected.

## Suspected location

`RUST_BACKTRACE=1` shows the panic path:

```text
pliron::utils::apint::APInt::shl (apint.rs:180)
dialect_mir::ops::arithmetic::MirShlOp as ConstFoldInterface>::check_fold
    (crates/dialect-mir/src/const_fold.rs:200)
pliron::opts::constants::sccp::...
mir_transforms::unroll::unroll_annotated_loops
    (crates/mir-transforms/src/unroll.rs:387)
```

What seems to happen: `mir-importer` only inserts a unifying cast for
comparisons (`mir-importer/src/translator/rvalue/expr.rs`, `if
is_comparison`), so `mir.shl` keeps mixed operand widths; after
unrolling substitutes the induction variable,
`unroll_annotated_loops` runs `sccp` (unroll.rs:387), and
`MirShlOp::check_fold` (const_fold.rs:200) calls
`lhs.value().shl(&rhs.value())` on mixed-width pliron `APInt`s, whose
`assert_eq!` on bit widths panics. The non-folded path is fine because
`convert_shift` in `mir-lower/src/convert/ops/arithmetic.rs` casts the
count to the value width before emitting `llvm.shl` — the fold skips
that step. `MirShrOp::check_fold` (~line 216) has the same shape and
presumably the same bug (confirmed for `>>` above).

## Workaround

Use `u32` loop counters (and same-width shift amounts) in `#[unroll]`ed
loops; e.g. the reproducer with `let mut i: u32 = 0` compiles and runs
correctly.

## Note

This reproducer was reduced with AI assistance and verified by hand on
the machine described above; every compiles/does-not-compile claim was
checked by building that exact variant.
