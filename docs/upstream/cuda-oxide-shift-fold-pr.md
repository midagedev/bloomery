<!-- Filed as https://github.com/NVlabs/cuda-oxide/pull/1314 on 2026-09-21; body below is what went up. -->

## Summary

Constant-folding a shift whose amount has a different integer width from the shifted value panics the compiler:

```text
Internal compiler error in device codegen: assertion `left == right` failed: APInt::shl: bitwidth mismatch (32 vs 64)
```

Rust allows the shape (`u32 << usize`), and `convert_shift` in `mir-lower` already widens or narrows the amount when lowering. The folder did not: `MirShlOp`/`MirShrOp::check_fold` passed both operands straight to `APInt::shl`/`lshr`/`ashr`, which assert equal widths. `#[unroll]` is the easy way to reach it, since the pass materialises a `usize` loop counter as a 64-bit constant and then runs SCCP:

```rust
let mut i: usize = 0;
#[unroll]
while i < 1 {
    *o = 1u32 << i;
    i += 1;
}
```

A `u32` counter, or `i as u32`, avoids it; the fold is the only place that does not accept the mixed pair.

## Changes

- `dialect-mir/src/const_fold.rs`: `shift_in_range` becomes `shift_amount`, which returns the amount re-expressed at the shifted value's width, or `None` when it is `>= width` (unchanged: a Rust shift overflow is not folded). The range check reads the amount at its own width first, so a wide amount such as `1 << 32` is refused rather than truncated to `0` and folded.
- `dialect-mir/tests/const_fold.rs`: one test — wider amount for `shl`, logical and arithmetic `shr`; narrower amount (`u64 << u8`); the two refusals.

The fix mirrors the lowering rather than touching pliron's `APInt` asserts, so the folder and `convert_shift` agree on what a legal shift is.

## Testing

- The new test fails before the fix with exactly the message above and passes after.
- End to end on an RTX 3090 (sm_86, CUDA 13.0, nightly-2026-08-28), selecting the backend with `CUDA_OXIDE_BACKEND`: built from `b0f961d`, the kernel above fails with the ICE (exit 101); built from this branch it compiles and runs, all 32 elements `== 1`; a four-trip variant (`acc |= 1u32 << (2 * i)`) gives `== 85`, so the folded values are right, not merely accepted.
- `scripts/smoketest.sh --compile-only` over all 230 examples, patched backend: 227 pass, 3 fail (`gemm_sol`, `gemm_sol_final`, `tcgen05_matmul`). The `b0f961d` backend gives the same 227/3 with the same three, all for one reason: the machine's LLVM 21 `llc` does not know `llvm.nvvm.stmatrix.sync.aligned.m8n8.x2.b16.p3`. `unroll_smoke` and `unroll_bounds_check` executed on the 3090: pass.
- `just check` minus `fmt-check`, run recipe by recipe on the same machine (no `just` there): clippy, test (5359 tests, 0 failed), test-cuda, check-guards, doc-check all pass. `cargo oxide fmt --check` reports the same set on unpatched `main` and on this branch (two example manifests need a nightly Cargo feature; not touched here); the two files this PR changes are rustfmt-clean.
- [x] `cargo test -p dialect-mir`, `cargo clippy -p dialect-mir --all-targets -- -D warnings` pass
- [ ] `just check` as one command: not run (see above for the per-recipe results)
- [ ] New example: none — the dialect test carries the regression, and the kernel above is the runtime check

## Checklist

- [x] All commits signed off (`git commit -s`)
- [x] SPDX headers on new source files (none added)

The reproducer was reduced and this patch drafted with AI assistance; every result above was re-run by hand on the machine named.
