## What

Bolt's decode step folds 80 rms_norm + quantize_q8_1 pairs (1084 → 1004 graph nodes) into the existing fused `norm_quant` launch (host dispatch only, no kernel touched, every output bit as before by `FusedKernels::enqueue_norm_quant`'s contract):

- `4cd38136` (P3a, implemented by GLM-5.3 from a perf spec, reviewed): `hybrid::mamba_layer` — when `ssm_in` reads the q8_1 form (`SiteTy::reads() == Form::Q8x128`, Bolt's NVFP4), the pre-norm and `ssm_in`'s quantize are one `norm_quant` launch, then the site's kgemv alone. `Site::project`'s q8_1 arm is split into its quantize and kgemv halves (no copied kgemv). The Nano's Q5_0 `ssm_in` keeps `norm` + `project`.
- `77180fa2` (P3b, perf): `nemotron_h_moe::moe_layer` — the MoE pre-norm is the same `norm_quant` launch, writing `xn` (the router and the latent down read it as before) and the hidden-width q8_1 form the shared expert's up reads; the up's own quantize of `xn` is gone (no launch between writes that form; `Acts::q8_1_at` keys forms by (k, m)). The structure pin becomes one `norm_quant` a Mamba-2 and a MoE layer.
- `cea33b80` (review): the shared expert's up runs right after the fused norm, before the latent's quantize. `Acts` keys a q8_1 form by (k, m) alone, so a file whose latent width equals its hidden would have had the latent row quantized into the buffer the up then read (finite wrong values, no fault). Bolt's file is 4096 hidden / 1024 latent, two buffers, so no measured run read a wrong form: the box's fixture keeps the real file's widths (`KeyRule::Copy` of `embedding_length` and `moe_latent_size`), which is why its logits did not move. The equal-width layout exists only as spec.rs's header-reader `FIXTURE`, which builds no GPU model, so the GPU gates hold that case only through the order. The order no longer depends on the widths. The two single-caller `Site` wrappers are inlined (`m.ssm_in.ty.reads() == Form::Q8x128`; `project`'s q8_1 arm quantizes inline).

Change class: fold a launch into a neighbour. The Bolt e2e `structure` clause pins `norm_quant` (a revert fails it).

## Prediction [derived]

nsys of main's decode (A6000, below) puts the graph's gaps near 0 (12175 µs of kernel time a 12.2 ms step), so each fold is worth the removed launch's own time: `q3k_quantize_q8_1` ~1.9–2.3 µs plus the norm's difference, ~0.08 ms a token for 40 folds (~0.6 %) — ~~40~~ 80 folds, see the A/B. Under the timing runner's ruler (±1.0 % at 4 rounds).

## Timed A/B (fold class, same lease)

`tools/ref/depth-qwen3moe.sh bin:<main d0e8a8e1>:512 bin:<#326 8a16e7a4>:512 512` from this tree (77180fa2), RTX A6000, `BLOOMERY_AB_ROUNDS=4`, rotating arm order, card `docs/cards/bolt-decode-ab-a6000.card` (s1724):

| arm | tok/s @ n=96, depth 512, A6000 (mean of 4) | range | nodes |
|---|---|---|---|
| main d0e8a8e1 | 81.19 | 80.79..82.21 | 1084 |
| **this PR 77180fa2** | **82.90** | 82.63..83.30 | 1004 |
| #326 8a16e7a4 (beside it) | 84.84 | 84.45..85.45 | 1084 |

+2.1 % (±1.0 % at 4 rounds) against the derived ~0.6 %. The miss, by term: the prediction counted 40 folds where the step has 80 (40 Mamba-2 + 40 MoE), so ~1.3 % at ~2 µs a fold; the A/B's 12.317 → 12.063 ms is 0.254 ms over 80 folds, ~3.2 µs a fold [derived from the means], ~1.2 µs a fold more than the removed quantize's own kernel time. That residue is unexplained here; no mechanism is claimed. Rows tagged cpu-busy/other-busy are read beside the means (2 of 12 other-busy).

## GPU Evidence (ws, RTX 3090 lane; ptxas/toolchain as main)

| gate | job | result |
|---|---|---|
| `BLOOMERY_TIER=fixture just gate-gpu-nemotron-h-moe-e2e` @ 4cd38136 | s1701 | PASSED; `structure: … graph_nodes=59 … norm_quant=1/1` |
| `just gate-gpu-nemotron-h-moe-e2e` (real Bolt file) @ 4cd38136 | s1703 | PASSED; `structure: layers mamba=40 attn=8 moe=40, graph_nodes=1044` (main 1084, s1656) `norm_quant=40/40` |
| `just gate-gpu-nemotron-e2e` (Nano) @ 4cd38136 | s1704 | PASSED; structure unchanged (`graph_nodes=380`) |
| FAIL-first `tools/mutant-run.sh` (`if false && m.ssm_in.reads_q8_1()`), fixture tier | s1707 | killed: `norm_quant=0/1 … clause structure FAIL`, only that clause; clean-after s1709 green (`norm_quant=1/1`); `mutant-run: 1 mutants, 1 killed, 0 survived … — PASS` |
| `just gate-gpu-nemotron-h-moe-e2e` (real) @ 77180fa2 | s1711 | PASSED; `graph_nodes=1004` `norm_quant=80/80` |
| `BLOOMERY_TIER=fixture just gate-gpu-nemotron-h-moe-e2e` @ 77180fa2 | s1713 | PASSED; `graph_nodes=57` `norm_quant=3/3` |
| FAIL-first P3b (MoE norm + quantize restored), fixture tier | s1716 | killed: `norm_quant=1/3 … clause structure FAIL`, only that clause; clean-after s1718 green; `mutant-run: 1 mutants, 1 killed, 0 survived … — PASS` |

| `BLOOMERY_TIER=fixture just gate-gpu-nemotron-h-moe-e2e` @ cea33b80 | s1741 | PASSED; `graph_nodes=57` |
| `just gate-gpu-nemotron-h-moe-e2e` (real) @ cea33b80 | s1744 | PASSED; `graph_nodes=1004`; the 24 printed logit/route/reset lines byte-identical (sorted, `cmp` rc 0) to 77180fa2's s1711 |

Bit identity: at both commits, the real tier's 24 printed logit/route lines (`eager_equals_replay`, `logits_near_mainlines_real` per prompt with max |delta|, cosine and top-5 logits, `top1_is_mainlines_real` margins) are byte-identical (`cmp` rc 0) to the same gate's run on the bit-identical iq2opt branch (s1656, A6000), and eager = replay holds.

Not merged; for review.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
