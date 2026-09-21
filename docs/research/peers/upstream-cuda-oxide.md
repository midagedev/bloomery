# Pre-Submission Research Report: NVlabs/cuda-oxide Upstream Candidates

- **Repository**: [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)
- **Pinned Revision**: `b9847e9515ed3a23096f22567d3eaf0a6e3e440c`
- **Upstream HEAD Revision**: `b0f961df3af0ff140b3b006fa2b6750b71f43f62` (commit date: 2026-09-20T10:52:57Z)
- **Relationship between Pin and HEAD**: HEAD is exactly 1 commit ahead of the pin (`b0f961d` has parent `b9847e9`). The only diff between the pin and HEAD is in `.vscode/settings.json` and `crates/mir-lower/src/convert/ops/control_flow.rs` ("fix(tests): use the upstream volatility interface").

---

## Candidate 1: `#[unroll]` Compiler Panic (`APInt::shl: bitwidth mismatch`)

### 1. Still Present at HEAD?
- **Read in code**:
  - The constant folding logic for shifts is located in `crates/dialect-mir/src/const_fold.rs` at lines 186–240 in both the pinned rev (`b9847e9`) and HEAD (`b0f961d`). The file is 100% byte-identical between the pin and HEAD.
  - At line 187–189: `shift_in_range(value: &APInt, amount: &APInt)` checks only `amount.to_u128() < value.bw() as u128`. It validates the numerical magnitude of the shift amount against the value's bitwidth, but does not validate that `value.bw() == amount.bw()`.
  - At line 200: `MirShlOp::check_fold` invokes `lhs.value().shl(&rhs.value())`.
  - At lines 223–225: `MirShrOp::check_fold` invokes `lhs.value().ashr(&rhs.value())` or `lhs.value().lshr(&rhs.value())`.
  - In Pliron's `APInt::shl/ashr/lshr`, an assertion `assert_eq!(self.bw(), rhs.bw(), "APInt::shl: bitwidth mismatch")` is evaluated. When `lhs` is 32-bit (`u32`) and `rhs` is 64-bit (`usize`), `APInt::shl` panics in `check_fold`.
  - In `crates/mir-transforms/src/unroll.rs:975`, full loop unrolling materializes the loop counter as an induction variable literal of `s.iv_type` (`usize` = 64-bit on 64-bit targets) using `make_const(ctx, s.iv_type, iv_literal, prev_tail)`.
  - In `crates/mir-transforms/src/unroll.rs:387`, `unroll_annotated_loops` runs `sccp(func_op, ctx)?` on any function where unrolling occurred.
  - In `crates/mir-lower/src/convert/ops/arithmetic.rs:477–505`, `convert_shift` explicitly accommodates mixed-width shifts in lowering by widening (`zext`) or narrowing (`trunc`) the shift amount to match `lhs_width`, proving that mixed-width shifts are valid IR in cuda-oxide.
- **Upstream commit history**:
  - `git log b9847e9..b0f961d` contains only commit `b0f961df3af0ff140b3b006fa2b6750b71f43f62` ("fix(tests): use the upstream volatility interface"), which touches only `.vscode/settings.json` and `crates/mir-lower/src/convert/ops/control_flow.rs`.
  - A search across all 1,172 commits in the repository history for `unroll`, `APInt`, `bitwidth`, and `usize` confirmed that no commit has ever modified `crates/dialect-mir/src/const_fold.rs:186–240` since it was first introduced in commit `9e923e93` ("feat(unroll): loop-unrolling pass with opt-independent constant folding (#284)").
  - **Conclusion**: The bug is **unquestionably still present at HEAD**.

### 2. Duplicates
Searched upstream issues, pull requests, and discussions via GitHub CLI read-only queries (`gh issue list`, `gh pr list`, and `gh api graphql`).
- **Query log and hit counts**:
  - `unroll`: 7 issues (#811, #1235, #557, #495, #399, #397, #1150), 24 PRs (#1299, #1293, #1205, #972, #559, #284, #636, #320, #398, #965, #818, #1050, #1221, #496, #510, #850, #1151, #1203, #964, #608, #387, #314, #650, #684).
  - `APInt`: 1 issue (#481: "SwitchInt on a 128-bit integer scrutinee is rejected"), 3 PRs (#482, #650, #284).
  - `bitwidth mismatch`: 0 issues, 0 PRs.
  - `APInt::shl`: 0 issues, 0 PRs.
  - `bitwidth`: 0 issues, 0 PRs.
  - `shl`: 1 issue (#422: "Deduplicate primitive integer type IDs in generation"), 0 PRs.
  - Discussions: 8 total discussions in repository; none mention unroll, APInt, bitwidth, or shift folding.
- **Duplicate finding**: **0 duplicates found**. The issue has never been reported, tracked, or discussed upstream.

### 3. Contribution Rules
- **Stated by upstream** (`CONTRIBUTING.md:18–98`):
  - Upstream requires the **Developer Certificate of Origin (DCO) 1.1**.
  - Every commit must include a `Signed-off-by: Name <email>` trailer matching `user.name` and `user.email` (`git commit -s`).
  - No CLA is required; DCO is the sole sign-off mechanism.
  - No explicit "AI-generated code" prohibition exists in `CONTRIBUTING.md`, `README.md`, or the issue/PR templates.
  - All contributions are subject to NVIDIA's internal IP review process (`CONTRIBUTING.md:241–246`).
  - All new first-party source files must carry the exact SPDX block header:
    `/*\n * SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.\n * SPDX-License-Identifier: Apache-2.0\n */`
- **Does upstream accept external PRs?**
  - **Yes, actively.** Multiple external (non-NVIDIA) contributors have had PRs merged. Representative merged PRs from non-NVIDIA authors:
    1. PR #1288: "fix(atomics): support core AtomicPtr lowering" by `@YunusMutlu` (`mutluyunus708@gmail.com`).
    2. PR #1282: "fix(debug): support runtime-indexed fixed-array references" by `@uurl` (`raulestradaa@gmail.com`).
    3. PR #1275: "refactor(llvm): use upstream global constant semantics" by `@uurl` (`raulestradaa@gmail.com`).
    4. PR #1270: "fix(cuda-device): pack coalesced ballot masks" by `@YunusMutlu` (`mutluyunus708@gmail.com`).
    5. PR #1264: "fix(mir-lower): lower redux.sync results to signless integers" by `@sachinsharma3191` (`sachinsharma3191@gmail.com`).
- **Classification**: **Bug Report**. Device codegen crashes with an ICE during compilation (`rustc_codegen_cuda` assertion failure). Matches `.github/ISSUE_TEMPLATE/bug_report.md` (labels: `bug`, `TBD`).

### 4. Merged Sibling Precedent
- **Closest merged PR**: PR #1293 ([#1293](https://github.com/NVlabs/cuda-oxide/pull/1293)):
  - **Title**: `fix(codegen): preserve assertions during loop unrolling`
  - **Author**: `nihalpasham` (merged 2026-09-18).
  - **Size**: +1,688 / -192 across 11 files.
  - **Test style**: Added a dedicated example / regression test (`unroll_bounds_check`) verifying that assertions survive full and partial loop unrolling. Coupled with CI checks inspecting generated PTX (`assert!(ptx.contains(...))`) and unit tests in `crates/mir-transforms/tests/assert_preservation.rs`.
- **Foundational precedent**: PR #284 ([#284](https://github.com/NVlabs/cuda-oxide/pull/284)):
  - **Title**: `feat(unroll): loop-unrolling pass with opt-independent constant folding`
  - **Size**: Introduced the unroll pass and `crates/dialect-mir/src/const_fold.rs`. Tested via `unroll_smoke` and dialect unit tests.

### 5. Patch Surface (Read Only)
- **Likely root cause** (*our inference*):
  - Rust permits mixed-width shifts (e.g. `u32 << usize`, `u64 << u32`).
  - `crates/mir-lower/src/convert/ops/arithmetic.rs:477–505` (`convert_shift`) already legalizes mixed-width shifts by casting the amount operand to the shifted value's width.
  - In `crates/dialect-mir/src/const_fold.rs:200` (`MirShlOp::check_fold`) and `lines 223–225` (`MirShrOp::check_fold`), `lhs.value().shl(&rhs.value())` directly passes `rhs` into Pliron's `APInt::shl`. Pliron's `APInt` arithmetic methods assert identical bitwidths.
  - The unroll pass materializes the loop counter as a 64-bit constant (`usize`), and then immediately runs SCCP (`crates/mir-transforms/src/unroll.rs:387`), which exposes both operands as compile-time constants of different bitwidths to `check_fold`.
- **Where a fix would go**:
  - In `crates/dialect-mir/src/const_fold.rs`: Inside `MirShlOp::check_fold` and `MirShrOp::check_fold`.
  - Fix implementation: Since `shift_in_range(&lhs.value(), &rhs.value())` (line 187) already verifies that `rhs.to_u128() < lhs.bw() as u128`, `rhs` can be cast/resized to `lhs.value().bw()` before invoking `APInt::shl`/`ashr`/`lshr`. Alternatively, `check_fold` can construct an `APInt` of width `lhs.value().bw()` with value `rhs.value().to_u64()`.
  - Tests: Add unit tests in `crates/dialect-mir/` verifying `MirShlOp` and `MirShrOp` constant folding with operands of mismatched bitwidths (e.g., `ui32` value shifted by `ui64` amount and vice versa).

### 6. Adversarial Review of Our Draft
A maintainer reading `bloomery/docs/upstream/cuda-oxide-unroll-ice.md` would push back on the following points:
1. **Missing `cargo oxide doctor` output**:
   - Upstream's `.github/ISSUE_TEMPLATE/bug_report.md:24–36` explicitly requests pasting `cargo oxide doctor` inside a `<details>` block to capture resolved `llc`, libNVVM, toolchain components, and driver configuration. The draft provides handwritten bullet points instead.
   - *Edit*: Run `cargo oxide doctor` and embed its verbatim output under an `Environment` details fold.
2. **Reproducer contains unnecessary host runtime and GPU scaffold**:
   - The minimal reproducer includes `CudaContext::new(0)`, `DeviceBuffer::zeroed`, stream synchronization, and GPU result assertions in `fn main()`.
   - The bug is a compiler crash (ICE) during device codegen (`cargo oxide build`). No GPU, CUDA context, or host execution is needed to reproduce the crash. Including runtime execution code gives the false impression that reproduction requires physical hardware.
   - *Edit*: Simplify `fn main()` to `fn main() {}`, removing all `cuda-core` / `DeviceBuffer` runtime calls.
3. **Kernel signature carries unrelated DSL complexity**:
   - The kernel in the reproducer uses `DisjointSlice<u32>`, `out.get_mut(tid)`, `#[launch_bounds(32)]`, and `#[launch_contract(...)]`.
   - None of these attributes or types are involved in the defect. An issue reviewer might question whether `DisjointSlice` or contract lowering is causing the crash.
   - *Edit*: Use a plain slice parameter `pub fn ice(out: &mut [u32])` without `#[launch_contract]` or `DisjointSlice`, focusing 100% on the `#[unroll]` loop and the shift.
4. **Internal workflow comments**:
   - Lines 1–4 of the draft include internal instructions: `<!-- Upstream issue draft for NVlabs/cuda-oxide — NOT filed. Lead pastes and submits after review... -->`.
   - *Edit*: Strip all internal harness / workflow commentary before submission.

---

## Candidate 4: `launch_contract` `requires` Arithmetic Has No Division

### 1. Still Present at HEAD?
- **Read in code**:
  - The `requires` relation grammar at HEAD (`b0f961d`) is defined in `crates/cuda-macros/src/cuda_module/contract.rs`.
  - Lines 406–411 (`requires_grammar_help`):
    ```rust
    format!(
        "each requires relation is one comparison (`>=`, `>`, `<=`, `<`, `==`, `!=`) between \
         expressions built from slice parameters as `<param>.len()`, unsigned integer scalar \
         parameters (u8/u16/u32/u64/usize), integer literals, parentheses, and `+`, `-`, `*`; \
         available operands: {available}"
    )
    ```
  - Lines 458–463 (`requires_arithmetic_op`):
    ```rust
    fn requires_arithmetic_op(op: &syn::BinOp) -> bool {
        matches!(
            op,
            syn::BinOp::Add(_) | syn::BinOp::Sub(_) | syn::BinOp::Mul(_)
        )
    }
    ```
  - Lines 760–765 (`requires_operand_tokens`):
    ```rust
    let checked = match binary.op {
        syn::BinOp::Add(_) => quote! { checked_add },
        syn::BinOp::Sub(_) => quote! { checked_sub },
        syn::BinOp::Mul(_) => quote! { checked_mul },
        _ => unreachable!("requires operators are validated during contract construction"),
    };
    ```
  - Division (`/`, `syn::BinOp::Div`) is explicitly rejected.
  - `crates/cuda-macros/tests/compile_fail/launch_contract_requires_bad_operator.rs` (lines 4–5) specifically tests and asserts this omission:
    `// requires arithmetic is limited to +, -, and *; division is not part of the v1 relation grammar.`
    with `.stderr` confirming rejection of `input.len() / 2 >= n`.
  - **Conclusion**: Division is **completely absent at HEAD**, and is deliberately prohibited by design in the v1 grammar.

### 2. Duplicates
Searched upstream issues, pull requests, and discussions via GitHub CLI read-only queries (`gh issue list`, `gh pr list`, and `gh api graphql`).
- **Query log and hit counts**:
  - `launch_contract`: 6 issues (#1277, #1192, #534, #507, #730, #782), 12 PRs (#1056, #1193, #535, #514, #850, #509, #1028, #505, #319, #318, #320, #477).
  - `requires`: 30 issues, 30 PRs (all examined; none propose contract arithmetic expansion or division).
  - `division`: 0 issues, 3 PRs (#1293, #373, #473).
  - `contract arithmetic`: 7 issues (#1235, #1036, #274, #968, #611, #315, #190), 16 PRs (#1081, #830, #484, #655, #612, #326, #391, #279, #477, #117, #474, #202, #691, #187, #406, #191).
  - `divide`: 1 issue (#205), 9 PRs (#655, #573, #511, #513, #545, #473, #206, #638, #12).
  - `checked_div`: 0 issues, 0 PRs.
  - `LaunchContractError`: 1 issue (#480), 1 PR (#1014).
- **Duplicate finding**: **0 duplicates found**. No issue or PR has requested adding division to `launch_contract(requires = ...)`.

### 3. Contribution Rules
- **Stated by upstream** (`CONTRIBUTING.md:100–110`):
  - Standard process: Open an issue describing the feature, discuss, submit PR with DCO sign-off.
  - For language and macro design changes, `CONTRIBUTING.md:13–15` specifically advises:
    *"If you are unsure whether something is worth a full issue or PR, the Discord `#contributors` channel is a good place to ask first."*
- **Does upstream accept external PRs?**
  - Yes (see Candidate 1 analysis above for merged examples).
- **Classification**: **Feature Request / Proposal**. This is an intentional absence in the v1 grammar, not a compiler crash or regression. Matches `.github/ISSUE_TEMPLATE/feature_request.md` (labels: `enhancement`, `TBD`).

### 4. Merged Sibling Precedent
- **Closest merged PR**: PR #477 ([#477](https://github.com/NVlabs/cuda-oxide/pull/477)):
  - **Title**: `Bounds-check elision: safe views + size contracts, and an opt-in unchecked switch`
  - **Size**: Major feature PR introducing `requires` on `#[launch_contract]`.
  - **Test style**: Added 12 `trybuild` compile-fail tests for syntax/type validation (`crates/cuda-macros/tests/compile_fail/`), unit tests for macro expansion in `crates/cuda-macros/src/tests/cuda_module.rs`, and an end-to-end example (`gemm_views`) with runtime verification.
  - **Design note from PR #477**: The PR description states:
    *"Relations are a tiny grammar, not arbitrary Rust: identifiers are validated against the kernel signature (typos are compile errors) and all arithmetic is checked u64 (a huge m * k cannot wrap and falsely pass)."*
    And under "Not in this PR": deliberately restricted grammar to avoid undefined semantics.

### 5. Patch Surface (Read Only)
Adding `/` to `launch_contract` `requires` clauses would touch three areas:
1. **Parser and AST validation**:
   - `crates/cuda-macros/src/cuda_module/contract.rs:458–463`: Add `syn::BinOp::Div(_)` to `requires_arithmetic_op`.
   - `crates/cuda-macros/src/cuda_module/contract.rs:406–411`: Update `requires_grammar_help` message to list `/`.
2. **Codegen and Semantics of Runtime Check**:
   - `crates/cuda-macros/src/cuda_module/contract.rs:760–775`: Add `syn::BinOp::Div(_) => quote! { checked_div }`.
   - **Zero-divisor semantics**: If `rhs == 0`, `checked_div` returns `None`. Currently, `requires_operand_tokens` maps `None` to `::cuda_core::LaunchContractError::SizeRequirementOverflow`. Calling division-by-zero an "overflow" is inaccurate. However, `cuda_core::LaunchContractError` is defined in an external crate (`cuda-core`, published from `NVlabs/cutile-rs`). Adding a new error variant like `SizeRequirementDivisionByZero` would require an upstream release of `cuda-core` first.
   - **Rounding / Truncation semantics**: Rust integer division truncates toward zero (floor division for positive numbers). In a contract like `qs.len() >= n_rows * k / 4`, if `n_rows * k == 7`, `7 / 4 == 1`. If the buffer actually requires 2 words to store 7 elements, integer floor division permits an undersized buffer of 1 word. For kernel safety contracts, ceiling division (`div_ceil`) or exact divisibility is often what is intended.
3. **Upstream test updates**:
   - `crates/cuda-macros/tests/compile_fail/launch_contract_requires_bad_operator.rs` currently tests that `/` is rejected; it would need to be updated to test a different illegal operator (e.g. `%` or `&`).
   - `crates/cuda-macros/tests/compile_fail/launch_contract_requires_bad_operator.stderr` would need re-generation.
   - New tests in `crates/cuda-macros/tests/pass/` and `crates/cuda-macros/src/tests/cuda_module.rs` covering `/`.

---

## (a) Go / No-Go / Needs-More Recommendations

| Candidate | Status | Single Strongest Reason |
|---|---|---|
| **Candidate 1** (`#[unroll]` ICE) | **GO** | Confirmed 100% reproducible and present at upstream HEAD (`b0f961d`), zero duplicates exist, the root cause in `const_fold.rs:200` is cleanly isolated, and upstream actively accepts bug fixes for compiler ICEs in `unroll` (precedent: PR #1293). |
| **Candidate 4** (`requires` division) | **NEEDS-MORE** | Division was deliberately excluded from v1 (`launch_contract_requires_bad_operator.rs:4–5`), and adding it requires upstream alignment on integer division semantics (floor vs ceiling division) and `cuda-core::LaunchContractError` error variants before any code can be accepted. |

---

## (b) What Could Not Be Determined

1. **Pliron Upstream Shift Plans**: Whether the Pliron project (`pliron::utils::apint::APInt`) intends to relax `APInt::shl` to allow mismatched bitwidths natively, or if Pliron considers mismatched bitwidth shift folding the sole responsibility of the dialect caller (`dialect-mir`).
2. **NVIDIA IP Review Timelines**: While external PRs are regularly merged, the turnaround time of NVIDIA's internal IP review process for non-employee PRs varies from days (e.g. PR #1288) to weeks depending on the subsystem touched.

---

## (c) Contribution Opportunities for Quantized-Matmul Kernels

Reading through `cuda-device` and `cuda-macros` identified the following concrete contribution opportunities relevant to quantized-matmul kernels (e.g. INT4 / FP4 / FP8 GEMM):

1. `crates/cuda-device/src/mma_frag.rs:6`: Add fragment index algebra and unpack abstractions for sub-byte shapes (`m16n8k64_s32_s4`, `m16n8k64_s32_u4`, `m8n8k32_s32_s4/u4`); currently `mma_frag` only derives mappings for `m16n8k16` and `m16n8k32_s32_s8`.
2. `crates/cuda-device/src/lib.rs:45`: Add `bfe` (bit field extract) and `bfi` (bit field insert) PTX intrinsics to `cuda_device`; currently only `prmt` (byte-level permutation) is implemented, forcing 4-bit unpacking through cumbersome arithmetic or shift sequences.
3. `crates/cuda-device/src/lib.rs:45`: Add `lop3.b32` (three-input bitwise logic) intrinsic to `cuda_device`; essential for single-cycle branchless INT4-to-FP16 dequantization without lookup tables.
4. `crates/cuda-macros/src/cuda_module/contract.rs:458`: Propose `div_ceil` or integer division in `launch_contract` `requires` clauses, enabling contracts over packed sub-byte tensor buffers (e.g. `weights.len() >= (rows * cols + 1) / 2`).
5. `crates/cuda-device/src/wmma.rs:75`: Provide sub-byte staging abstractions over `ldmatrix.sync.aligned.m8n8.x4.shared.b16`, reducing boilerplate when loading packed 4-bit weights from shared memory to register fragments.
