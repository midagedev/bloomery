# REPRO: `#[unroll]` + `usize` counter + shift ICEs device codegen

Reduced 2026-09-19 on the box (`ws`) from a Q3_K kernel that died with
`APInt::shl: bitwidth mismatch (64 vs 32)` when `#[unroll]` was put on
`while` loops with `usize` counters.

## Environment

- cuda-oxide checkout `/root/src/cuda-oxide` at `b9847e9515ed3a23096f22567d3eaf0a6e3e440c`
  (`fix(codegen): preserve assertions during loop unrolling (#1293)`)
- `cargo-oxide 0.2.1`, `rustc 1.100.0-nightly (e457a7b0d 2026-08-27)`
  (toolchain `nightly-2026-08-28`)
- LLVM 21.1.8 (`~/opt/LLVM-21.1.8-Linux-X64`), CUDA 13.0 (V13.0.88),
  driver 615.71.09, GPU RTX 3090 (`sm_86`)
- Crate deps pin `cuda-device`/`cuda-host` to the same `rev` (see `Cargo.toml`)

## Minimal reproducer

`crates/oxide-ice-unroll/` is `cargo oxide new` output with this kernel
(`src/main.rs`); build with `cargo oxide build --arch sm_86` from that dir:

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

Exact compiler output (exit 101):

```text
error: [rustc_codegen_cuda] Internal compiler error in device codegen: assertion `left == right` failed: APInt::shl: bitwidth mismatch (64 vs 32)
error: could not compile `oxide-ice-unroll` (bin "oxide-ice-unroll") due to 1 previous error
```

Reduction notes: the first candidate body (`i as u32 & 3`, no shift)
compiled, so a shift of the counter is required, not just a `usize`
counter. Slice/table indexing of the counter is not required (row (c)).
`i << 1` is `usize << i32` — the literal defaults to `i32` — so the MIR
shift has mixed 64/32-bit operands.

## Intervention table

Method: one edit to the kernel above per row, then
`cargo oxide build --arch sm_86` in `crates/oxide-ice-unroll/`.
"Compiles" = exit 0; "ICE" = exit 101 with the error line shown.

| Row | Variant (single change vs baseline) | Compiles | Exact error line / note |
|-----|--------------------------------------|----------|-------------------------|
| (a) | Failing baseline (`usize` + `#[unroll]` + `i << 1`) | no | `APInt::shl: bitwidth mismatch (64 vs 32)` |
| (b) | `usize` -> `u32` counter, nothing else | **yes** | PTX folds the loop to `st.global.b32 [%rd1], 56`; `cargo oxide run` prints `PASSED: all 256 elements correct` |
| (c) | (b) + body indexes `table[i as usize]` (local `[u32; 8]`) instead of the shift | yes | — |
| (d) | Baseline without `#[unroll]` | yes | — |
| (e1) | Baseline with `1u64` shift amount (`i << 1u64`) | yes | same-width (64,64) fold is fine |
| (e2) | Baseline with `1u32` shift amount (`i << 1u32`) | no | `APInt::shl: bitwidth mismatch (64 vs 32)` |
| (f) | `#[unroll] for i in 0..8` (same shift body) | yes | **Not E0658**: compiles with `warning: #[unroll] requested but the loop was not unrolled: no recognized induction variable (loop counter)`; the loop stays rolled. The `#[kernel]` macro consumes the attribute before rustc's E0658 check, and `mir-transforms` still only recognizes counted `while` shapes at this commit (cf. #557, closed, and unmerged PR #559). |
| (g) | Upstream `unroll_smoke` example, `src/main.rs` byte-identical (`cmp` clean), only `Cargo.toml` path deps rewritten to the same git rev + toolchain pin added (the checkout itself was not touched) | yes | two expected warnings from its deliberate negative tests (`loop-invariant bound`, `early break`); exit 0 |
| (h) | Mirrored widths: `1u32 << i` with `i: usize` | no | `APInt::shl: bitwidth mismatch (32 vs 64)` |
| (i) | `>>` instead of `<<`: `(i >> 1)` with `i: usize` | no | `APInt::lshr: bitwidth mismatch (64 vs 32)` |
| (j) | Shift amount is runtime (`tid`-derived `s: u32`, `(i << s)`) | yes | mixed widths are legal at runtime; only the constant fold asserts |

Deciding rows: **(e1) vs (e2)** — with `usize` counter and `#[unroll]`
held fixed, only the shift amount's type changes, and the outcome flips.
So the trigger is not "a `usize` counter" alone, and not `#[unroll]`
alone: it is a shift whose two operands have different bit widths *after
both become constant*. Any single change that removes one of the three
ingredients (mixed widths, both-constant, post-unroll fold) flips
ICE -> compiles: (b)/(e1) remove the width mix, (j) removes
both-constant, (d) removes the unroll fold.

## Where in the compiler

The message is **not** LLVM's: it is pliron's `APInt` assertion at
`pliron/src/utils/apint.rs:180` (`assert_eq!(self.bw(), rhs.bw(),
"APInt::shl: bitwidth mismatch ({} vs {})", ...)`).

`RUST_BACKTRACE=1` on row (a):

```text
8: <pliron::utils::apint::APInt>::shl
       at pliron-9a7b9a8b55345c4e/edd41fe/src/utils/apint.rs:180:9
9: <dialect_mir::ops::arithmetic::MirShlOp as pliron::opts::constants::ConstFoldInterface>::check_fold
       at cuda-oxide-6d394bb007f5e114/b9847e9/crates/dialect-mir/src/const_fold.rs:200:64
10-13: pliron::opts::constants::sccp::process_fold_op / process_op / process_block / sccp
14: mir_transforms::unroll::unroll_annotated_loops
       at cuda-oxide-6d394bb007f5e114/b9847e9/crates/mir-transforms/src/unroll.rs:387:13
```

The chain, with quotes (all paths under `crates/` of cuda-oxide @ b9847e95):

1. `mir-importer/src/translator/rvalue/expr.rs` inserts a unifying cast
   **only for comparisons** (`if is_comparison`, ~line 101); a Rust shift
   like `usize << i32` arrives at the dialect as `mir.shl` with mixed
   operand widths and no cast.
2. After unrolling, `mir-transforms/src/unroll.rs:385-387` runs cleanup
   on the function:
   ```rust
   // Fold and clean only this function. `sccp` folds constant index
   // arithmetic and branch conditions, ...
   sccp(func_op, ctx)?;
   ```
   Unrolling substitutes the induction variable's per-iteration value,
   so the shift's operands are both `mir.constant`.
3. `dialect-mir/src/const_fold.rs:200` folds without a width check:
   ```rust
   let res = IntegerAttr::new(lhs.get_type(), lhs.value().shl(&rhs.value()));
   ```
   (`MirShrOp::check_fold` at ~line 216 has the same shape via
   `ashr`/`lshr` — row (i) confirms both arms assert.)
4. The non-folded path is fine, which is why rows (d) and (j) compile:
   `mir-lower/src/convert/ops/arithmetic.rs` (`convert_shl`/`convert_shr`
   via `convert_shift`) casts the shift count to the value width before
   emitting `llvm.shl`, whose operands must match.

Suspected fix location: `crates/dialect-mir/src/const_fold.rs`,
`MirShlOp::check_fold` (line 200) and `MirShrOp::check_fold`
(~line 216) — extend/truncate the amount to the value width (mirroring
`convert_shift`), or decline the fold when widths differ.

Ruled out: LLVM/NVVM (panic is before IR export, in pliron `APInt`);
the `#[kernel]` macro (it only tags the loop); `llvm-export`
(it sees only width-matched `llvm.shl`).

## Duplicate search (2026-09-19, `gh`, NVlabs/cuda-oxide)

| Query | Command | Result count |
|-------|---------|--------------|
| `unroll` | `gh issue list --repo NVlabs/cuda-oxide --search "unroll" --state all --limit 30` | 7 issues (#811, #1235, #557, #495, #399, #1150, #397) — none about this ICE |
| `APInt` | `... --search "APInt" ...` | 1 issue (#481, SwitchInt on u128 — unrelated) |
| `bitwidth mismatch` | `... --search "bitwidth mismatch" ...` | 0 |
| `usize` | `... --search "usize" ...` | 30 (limit hit) — titles scanned, none about an unroll/shift ICE |
| `ICE` | `... --search "ICE" ...` | 1 issue (#79, tuple-returning functions — unrelated) |
| `unroll` PRs | `gh pr list --repo NVlabs/cuda-oxide --search "unroll" --state all --limit 30` | 24 shown — none about a shift-width ICE |

Two near-misses checked by body, both ruled out: #1235 (OPEN feature
request for u32 shared-memory addressing — no unroll, no panic) and #557
(CLOSED for-loop unroll recognition — explains row (f)'s warning path,
not this panic; its implementation PR #559 is CLOSED unmerged).

Verdict: **no duplicate — file a new issue** (`docs/upstream/cuda-oxide-unroll-ice.md`).
