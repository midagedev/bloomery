<!-- Upstream PR draft for NVlabs/cuda-oxide — NOT filed. The patch lives on
     the local branch fix/const-fold-mixed-width-shift (one DCO-signed commit).
     Title: fix(dialect-mir): fold shifts whose amount has a different width -->

## Summary

Constant-folding a shift whose amount has a different integer width from the shifted value panics the compiler:

```text
error: [rustc_codegen_cuda] Internal compiler error in device codegen: assertion `left == right` failed: APInt::shl: bitwidth mismatch (32 vs 64)
```

Rust allows this shape (`u32 << usize`), and `convert_shift` in `mir-lower` already widens or narrows the amount when lowering. The folder in `dialect-mir/src/const_fold.rs` did not: `MirShlOp`/`MirShrOp::check_fold` passed both operands straight to `APInt::shl`/`lshr`/`ashr`, which assert equal bit widths. `shift_in_range` only checked the amount's magnitude.

`#[unroll]` makes it easy to reach: the pass materialises a `usize` loop counter as a 64-bit constant and then runs SCCP, so `1u32 << i` arrives at the folder with a 32-bit value and a 64-bit amount. Smallest kernel that hits it:

```rust
let mut i: usize = 0;
#[unroll]
while i < 1 {
    *o = 1u32 << i;
    i += 1;
}
```

Without `#[unroll]` the same kernel compiles; with a `u32` counter it compiles; casting the amount (`i as u32`) also avoids it, because the cast keeps the folder from seeing two constants.

## Changes

- `crates/dialect-mir/src/const_fold.rs`: replace `shift_in_range` with `shift_amount`, which returns the amount re-expressed at the shifted value's width, or `None` when it is `>= width`. The range check reads the amount at its own width first, so a wide amount such as `1 << 32` is refused rather than truncated to `0` and folded. `MirShlOp` and `MirShrOp` use it; the result type is still the shifted value's type.
- `crates/dialect-mir/tests/const_fold.rs`: `shifts_fold_with_a_shift_amount_of_a_different_width` — wider amount for `shl`, logical and arithmetic `shr`; narrower amount (`u64 << u8`); and the two refusals (`1u32 << (1u64 << 32)`, `1u32 >> 32u64`).

## Testing

- The new test fails before the fix with exactly the message above (`apint.rs:180`, `32 vs 64`) and passes after. `cargo test -p dialect-mir`: all suites green. `cargo fmt --check` and `cargo clippy -p dialect-mir --all-targets -- -D warnings` clean.
- End to end on an RTX 3090 (sm_86, CUDA 13.0, nightly-2026-08-28), using `CUDA_OXIDE_BACKEND` to select a backend built from each tree:
  - backend built from `b0f961d` (main): the kernel above fails to compile with the ICE, exit 101;
  - backend built from this branch: it compiles and runs, all 32 elements `== 1`;
  - a four-trip variant (`acc |= 1u32 << (2 * i)`, `while i < 4`) compiles and runs, all elements `== 85` (`0x55`), so the folded values are right, not merely accepted.
- [ ] `just check` passes — <!-- fill from the local run before filing -->

## Checklist

- [x] All commits signed off (`git commit -s`)
- [x] No new source files

<!-- Disclosure, as prose: the reproducer was reduced and this patch drafted with AI assistance; every compile and run result quoted here was re-run by hand on the machine named above. -->
