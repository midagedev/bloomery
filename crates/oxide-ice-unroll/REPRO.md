# Reproducer: `#[unroll]` + `usize` counter → `APInt::shl: bitwidth mismatch` ICE

Device-codegen panic in cuda-oxide when a `#[unroll]`-annotated `while` loop
has a 64-bit (`usize`) counter and the loop body contains a shift whose
counter-side operand and other operand have different widths.

All builds below ran on the workstation through `tools/box.sh`, one command per
row, on 2026-09-19. Nothing else was building on the 3090 (this round is
compile-time work; the GPU was touched only by the single workaround run in
§Run). Machine: RTX 3090 (sm_86), driver 615.71.09, CUDA 13.0 (V13.0.88),
rustc 1.100.0-nightly (e457a7b0d 2026-08-27) = nightly-2026-08-28,
cargo-oxide 0.2.1, cuda-oxide git deps at `b9847e9515ed3a23096f22567d3eaf0a6e3e440c`
(checkout HEAD: "fix(codegen): preserve assertions during loop unrolling (#1293)"),
pliron at `edd41fe`.

Every row is: `cp variants/<v>.rs src/main.rs && cargo oxide build --arch sm_86`
from `crates/oxide-ice-unroll/`, log kept in `/tmp/ice-log/<v>.log` on the box.

## The minimal reproducer (`src/main.rs`)

```rust
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
```

Output (verbatim, `s5-trip1-no-acc.log`):

```text
   Compiling oxide-ice-unroll v0.1.0 (/root/repo/mulle-ice-glm/crates/oxide-ice-unroll)
[rustc_codegen_cuda] note: run with `RUST_BACKTRACE=1` to display a backtrace
error: [rustc_codegen_cuda] Internal compiler error in device codegen: assertion `left == right` failed: APInt::shl: bitwidth mismatch (32 vs 64)
         left: 32
        right: 64. This is a bug in cuda-oxide. Please file at https://github.com/NVlabs/cuda-oxide/issues
error: could not compile `oxide-ice-unroll` (bin "oxide-ice-unroll") due to 1 previous error
Build failed with exit code: Some(101)
```

### Shrink path (each step still ICEs unless noted)

| step | change from previous | ICEs? |
|---|---|---|
| `a-baseline` | failing shape as first hit in the wild: trip count 4, accumulator `acc \|= 1u32 << (2*i)` | yes `(32 vs 64)` |
| `s1-trip1` | trip count 4 → 1 | yes |
| `s2-trip1-bare-counter` | drop the `2 *` multiplier — amount is the bare counter | yes |
| `s3-trip1-overwrite` | `\|=` → overwrite | yes |
| `s5-trip1-no-acc` | drop the accumulator — write the shift result straight out | yes |
| `s4-trip1-no-launch-attrs` | drop `#[launch_bounds]`/`#[launch_contract]` | n/a — not a valid shrink: the host-side launch API (`prepare_ice`) is generated from the launch contract; removing it fails host compilation with E0599 instead |

`s5` is minimal: from it, removing the `u32` from `1u32` (making both sides
`usize`) compiles — the width mismatch *is* the bug — and removing `#[unroll]`
compiles (row d). The `if let Some(o)` guard and the launch attributes are
scaffold requirements, not part of the defect.

## Intervention table

| row | variant | single change vs minimal | compiles | evidence (verbatim from the build log) |
|---|---|---|---|---|
| a | `s5-trip1-no-acc` | — (baseline) | **no** | `error: ... ICE in device codegen: assertion 'left == right' failed: APInt::shl: bitwidth mismatch (32 vs 64) ... Build failed with exit code: Some(101)` |
| b | `b-u32-counter` | `let mut i: usize` → `let mut i: u32` | **yes** | `Finished \`release\` profile [optimized] target(s) in 0.43s` |
| c | `c-u32-index` | `u32` counter, body indexes `w[i as usize]`, no shift | **yes** | `Finished \`release\` profile [optimized] target(s) in 0.43s` |
| d | `d-no-unroll` | `#[unroll]` removed | **yes** | `Finished \`release\` profile [optimized] target(s) in 0.43s` |
| e1 | `e1-u64-amount` | amount `i` → `i as u64` | **yes** | `Finished ... in 0.44s` — no skip warning, so the unroll fired; the `as` cast is what saves it (see below) |
| e2 | `e2-u32-amount` | amount `i` → `i as u32` | **yes** | `Finished ... in 0.43s` — same reason |
| e3 | `e3-rev-64-32` | `1u32 << i` → `(1u64 << (i as u32)) as u32` | **yes** | `Finished ... in 0.43s` — cast on the amount blocks the fold |
| e4 | `e4-usize-value-u32-amount` | `1u32 << i` → `(i + 1) << 2u32` (value side 64-bit, amount a `u32` literal — no cast in the fold path) | **no** | `error: ... APInt::shl: bitwidth mismatch (64 vs 32) ... Build failed with exit code: Some(101)` |
| f | `f-for-loop` | `while` → `for i in 0..1usize` | **yes** | `warning: #[unroll] requested but the loop was not unrolled: no recognized induction variable (loop counter)` + `Finished ... in 0.43s` |
| g | `variants/unroll_smoke` | upstream example unchanged (control) | **yes** | `Finished \`release\` profile [optimized] target(s) in 14.16s` |
| h | `h-partial-runtime` | `#[unroll]`, trip 1 → `#[unroll(4)]` + runtime bound `n: usize` | **no** | `error: ... APInt::shl: bitwidth mismatch (32 vs 64) ... Build failed with exit code: Some(101)` |
| i | `i-shr` | `<<` → `>>` (`0x8000_0000u32 >> i`) | **no** | `error: ... APInt::lshr: bitwidth mismatch (32 vs 64) ... Build failed with exit code: Some(101)` |
| j | `j-const-no-loop` | no loop at all: `let k: usize = 0; *o = 1u32 << k;` | **yes** | `Finished ... in 0.42s` |

### The deciding rows

**b is the single-change flip**: the *only* difference between row a (ICE) and
row b (clean build) is the counter's type, `usize` → `u32`. e4 and i extend it:
the panic appears in **both width directions** (`shl 32←64`, `shl 64←32`,
`lshr 32←64`) — whatever side the counter-derived constant lands on — and
a mixed-width shift whose counter-side operand passes through an `as` cast
(e1/e2/e3) does *not* panic, because the cast's result is not a constant attr,
so the folder never sees both operands constant.

### Cause statement

Rust legalizes mixed-width shifts (`u32 << usize`, `usize << u32`); mir-lower
handles them correctly (`convert_shift` at
`crates/mir-lower/src/convert/ops/arithmetic.rs:477` widens/narrows and
masks the amount). The **constant folder** does not:
`MirShlOp::check_fold` / `MirShrOp::check_fold` in
`crates/dialect-mir/src/const_fold.rs` (`shl` at line 200, `ashr`/`lshr` at
lines 223-225) call pliron `APInt::shl/lshr/ashr`, which assert equal
bitwidths (`pliron .../utils/apint.rs:180`). Their `shift_in_range` guard
(`const_fold.rs:187`) checks the amount's *value*, never its *width*. The
`#[unroll]` pass is the measured trigger because (1) full unrolling replaces
the counter with literals of the counter's width (`make_const` call at
`crates/mir-transforms/src/unroll.rs:975`) and (2) after *any* unroll the
pass runs SCCP over the function (`crates/mir-transforms/src/unroll.rs:387`),
whose optimistic lattice
evaluates loop phis at their initial constants — which is why even `#[unroll(N)]`
on a runtime-trip-count loop panics (row h) though the counter is never a
compile-time constant in the final IR. Without the annotation the function is
never visited by that SCCP (row j compiles with an all-constant mixed-width
shift and no loop; row d compiles with the loop un-annotated).

### Backtrace (row a, `RUST_BACKTRACE=1`; frames 8-15)

```text
   8: <pliron::utils::apint::APInt>::shl
             at /root/.cargo/git/checkouts/pliron-9a7b9a8b55345c4e/edd41fe/src/utils/apint.rs:180:9
   9: <dialect_mir::ops::arithmetic::MirShlOp as pliron::opts::constants::ConstFoldInterface>::check_fold
             at /root/.cargo/git/checkouts/cuda-oxide-6d394bb007f5e114/b9847e9/crates/dialect-mir/src/const_fold.rs:200:64
  10: pliron::opts::constants::sccp::process_fold_op   .../sccp.rs:87:34
  11: pliron::opts::constants::sccp::process_op        .../sccp.rs:155:9
  12: pliron::opts::constants::sccp::process_block     .../sccp.rs:165:9
  13: pliron::opts::constants::sccp::sccp              .../sccp.rs:189:13
  14: mir_transforms::unroll::unroll_annotated_loops
             at .../crates/mir-transforms/src/unroll.rs:387:13
  15: cuda_oxide_codegen::prep::prepare_mir_module     .../prep.rs:148:5
```

(The partial-unroll ICE, row h, produces the identical frame chain.)

## Run: the workaround is correct, not just accepted

`variants/b-u32-counter.rs` (counter `u32`, everything else identical) built
and ran on the 3090:

```text
$ cargo oxide run --arch sm_86
   Compiling oxide-ice-unroll v0.1.0 (/root/repo/mulle-ice-glm/crates/oxide-ice-unroll)
    Finished `release` profile [optimized] target(s) in 0.43s
     Running `target/release/oxide-ice-unroll`
PASSED: all 32 elements == 1
```

## Notes on two spec premises that measurement corrected

- **"for-loop attributes are rejected (E0658)"** (from the 2026-09-19 GLM-arm
  note): not what happens inside `#[kernel]`. `#[unroll]` on a `for` loop is
  consumed and *skipped with a warning* — `warning: #[unroll] requested but the
  loop was not unrolled: no recognized induction variable (loop counter)` (row f).
  The E0658 rejection the earlier round saw must have come from a different
  attribute position (e.g. a plain helper function), not this one.
- **Full unroll is not the only trigger.** The partial-unroll path (`#[unroll(N)]`
  with a *runtime* trip count) panics too (row h), via the pass's cleanup SCCP
  rather than literal materialization. Any issue text that scopes the bug to
  constant trip counts understates it.

## Duplicate search (NVlabs/cuda-oxide, 2026-09-19)

Queries and result counts, verbatim:

```text
gh issue list --repo NVlabs/cuda-oxide --search "unroll" --state all --limit 30            → 7
gh issue list --repo NVlabs/cuda-oxide --search "APInt" --state all --limit 30             → 1  (#481, 128-bit SwitchInt — different)
gh issue list --repo NVlabs/cuda-oxide --search "bitwidth mismatch" --state all --limit 30 → 0
gh issue list --repo NVlabs/cuda-oxide --search "usize" --state all --limit 30             → 30 (limit hit; no shift/fold ICE in titles)
gh issue list --repo NVlabs/cuda-oxide --search "ICE" --state all --limit 30               → 1  (#79, tuple-return lowering — different)
gh issue list --repo NVlabs/cuda-oxide --search "const fold" --state all --limit 30        → 5  (none matching)
gh issue list --repo NVlabs/cuda-oxide --search "shl" --state all --limit 30               → 1  (#422, integer type-ID dedup — different)
gh issue list --repo NVlabs/cuda-oxide --search "shift amount" --state all --limit 30      → 0
gh pr list     --repo NVlabs/cuda-oxide --search "unroll" --state all --limit 30           → 24
```

Adjacent-but-different, both read in full: PR #1293 (the checkout's HEAD;
preserves *user runtime assertions* through unrolling — not the folder's own
assert) and PR #284 (introduced the pass). Issue #557 (range-`for` recognition)
matches row f's warning, not this panic. **No duplicate found.**

## Lead verification (2026-09-19, same box)

Every `variants/*.rs` row above was rebuilt by the lead in one loop
(`cp variant → cargo oxide build --arch sm_86 → restore baseline → build`),
and the baseline re-ICE'd after every row, so no row's verdict came from a
stale build cache. Results matched the table exactly: a/e4/h/i/s1/s2/s3/s5
panic with the widths listed, b/c/d/e1/e2/e3/j finish, f warns
"no recognized induction variable", s4 fails host compilation (not a valid
shrink), the `unroll_smoke` control finishes. The `b-u32-counter` build was
also run on the 3090: `PASSED: all 32 elements == 1`.

The independent muse arm (`REPRO-arm-muse.md`, a 8-trip reproducer with an
accumulator) reached the same frames (`const_fold.rs:200` via
`unroll.rs:387` → sccp) and adds one row worth carrying into the issue: with
the `usize` counter and `#[unroll]` held fixed, `i << 1u64` compiles and
`i << 1u32` ICEs — a bare `1` literal in shift-amount position is `i32`, so
the innocent-looking `i << 1` is already a mixed-width shift.
