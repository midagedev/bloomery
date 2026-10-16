## What

Two bit-identical changes to the i-quant `_sel` decode kernels Bolt's routed and shared experts run on (IQ2_XXS):

- `ab44b6ce` (was ef515b31) IQ2_XXS's decode reads a block-shared grid and sign-selector table (`Iq2XxsTab`, filled once a block) instead of the global grid per lookup; the selectors are `prmt` byte picks. Every output bit as before.
- `2e30c849` (was 4b83c062) the `_sel` bodies (`gate_up_sel_body`, `down_sel_body`, `iq4_nl_gemv_sel32`) split a thread row into slot and expert row in u32, not in the device's emulated 64-bit division; each launcher refuses by name an `n_slots · rows_per_expert` past u32.

Change class: instruction count / table read on a kernel well under ~700 GB/s (routed 324–427 GB/s, shared 345–411 GB/s on the A6000), so the model does not say 0; proof is bit identity (owning gates) plus one timed kernel A/B.

## Prediction (docs/cards/bolt-iq2opt.card)

iq2xxs_5376 (shared up) 13.89 µs → 8.9..11.5 µs on the A6000 [derived from the SASS a sub-block, −36/−39 %].

## Build-time counts (tools/sass-scan.sh gate_iq_sel, ptxas 13.4.92, sm_86)

| kernel | base | here |
|---|---|---|
| iq2_xxs_gemv_sel | 401 | 304 |
| iq2_xs_gemv_sel | 420 | 344 |
| iq3_xxs_gemv_sel | 442 | 365 |
| iq4_nl_gemv_sel32 | 368 | 287 |
| iq2_xxs_rows (no `_sel` body) | 582 | 582 |

## GPU Evidence (ws, RTX A6000, branch head 4b83c062 on d0e8a8e1)

Rebased since onto main 85511826 as ab44b6ce / 2e30c849: `git diff 4b83c062 2e30c849 -- crates/gpu/src/iq.rs crates/gpu/src/iq_sel.rs` is empty (main's 18 commits do not touch either file); the ptx tables of main's other entries may move with main's own commits; the one conflict was `tools/unsafe-ratchet.txt` (main's head-norm entry 1862 → 1863 / 201 → 202 plus this branch's +3 / +1 → `gpu 1866 203 10`, `bash tools/check-unsafe.sh` rc 0). The runs below are at the pre-rebase head.

| gate | job | result |
|---|---|---|
| `just gate-gpu-iq` | s1644 | `gate_iq: PASS` (host rule bit-identical every geometry) |
| `just gate-gpu-iq-sel` | s1651 | `gate_iq_sel: PASS` |
| `BLOOMERY_TIER=fixture just gate-gpu-nemotron-h-moe-e2e` | s1654 | PASSED |
| `just gate-gpu-nemotron-h-moe-e2e` (real Bolt file) | s1656 | PASSED; structure graph_nodes=1084, iq2_xxs_gemv_sel=160/160 |
| `just gate-gpu-glm-sel` | s1657 | `gate_glm_sel: 98 clauses, 0 failed — PASS` |
| `just gate-gpu-routed-legs` | s1663 | `gate_routed_legs: PASS` |
| `just gate-gpu-qwen4exp-iqleg` | s1675 | PASS (`clause 2 bits: 8939556 bytes vs 8939556, 0 differ`) |
| `./tools/ptx-compare.sh d0e8a8e1 gate_e2e --features gpu` | s1689 | pending (queued); added to this body when it lands |

Kernel A/B (`tools/ref/time-gate.sh gate_iq_sel --bench-kernels --bench-arm iq2xxs` under the lease, card `docs/cards/bolt-iq2opt.card`, one sitting a tree back to back; a second interleaved round was cancelled for the deadline):

| op (A6000, graph_us_mean, 64-launch graphs × 7) | main d0e8a8e1 (s1660) | here 4b83c062 (s1671) | here / main |
|---|---|---|---|
| iq2xxs_5376 | 13.824 µs (412.7 GB/s) | 10.549 µs (542.2 GB/s) | 0.763 |
| iq2xxs_4096 | 16.519 µs (345.2 GB/s) | 11.954 µs (477.7 GB/s) | 0.724 |
| iq2xxs_moe_up_240x1536 | 11.371 µs (326.1 GB/s) | 9.130 µs (406.8 GB/s) | 0.803 |
| iq2xxs_moe_down_240x1536 | 12.365 µs (299.8 GB/s) | 9.294 µs (398.8 GB/s) | 0.752 |
| iq2xxs_moe_up_304x2048 | 13.770 µs (365.5 GB/s) | 11.508 µs (429.5 GB/s) | 0.836 |
| iq2xxs_moe_down_304x2048 | 11.709 µs (425.0 GB/s) | 9.534 µs (517.6 GB/s) | 0.814 |
| iq2xxs_moe_up_432x2304 | 15.363 µs (368.5 GB/s) | 12.992 µs (428.1 GB/s) | 0.846 |
| iq2xxs_moe_down_432x2304 | 16.176 µs (349.2 GB/s) | 12.380 µs (447.6 GB/s) | 0.765 |
| iq2xxs_moe_up_496x2560 | 18.159 µs (360.1 GB/s) | 14.400 µs (429.3 GB/s) | 0.793 |
| iq2xxs_moe_down_496x2560 | 16.172 µs (383.7 GB/s) | 12.269 µs (501.4 GB/s) | 0.759 |

The card's term, iq2xxs_5376, lands at **10.549 µs inside the predicted 8.9..11.5** (main 13.824 µs, the s1326 13.89 reproduced). Every routed geometry moves 0.72–0.85×. Model [derived]: the routed/shared IQ2 kernels are 2.57 ms of our 12.18 ms decode step (nsys, A6000, main), so at ~0.78× ~0.55 ms a token ≈ +4–5 % decode; the end-to-end decode A/B is not yet run.

Not merged; for review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
