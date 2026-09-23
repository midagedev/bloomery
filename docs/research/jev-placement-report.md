The kernel reads the slot map from a device table, so a same-count swap needs no graph re-capture. I have what I need, so I'm writing the report.

**Report: Jev-style classifiers for the draft side and expert re-placement (2026-09-24, read-only research)**

No files edited, no box commands, no git changes.

**Bottom line.** The open-source Jev ecosystem is real and big. None of it works inside inference. For (b) re-placement, the prompt's own prefill router statistics are the exact per-request signal, and we get them for free. A classifier only helps when the list must be chosen before prefill, and our batch-1 single-stream shape does not force that. For (a) the draft, per-token gating is too tight a budget for any Jev-style model, and our lookup draft already has a free confidence signal: match length. Try (a) first as an offline gate study, then a "prefix predicts the rest" column for (b).

## 1. Open-source Jev implementations (fetched)

| Project | Model | Runs on | Published latency per decision |
|---|---|---|---|
| Shalimov04/open-jev | ~140M mmBERT-small, distilled from a Qwen 3.8-27B teacher | CPU serving | **5.3–7.1 ms**; 182–2644 ex/s at batch 64 |
| wfzyx/von | 395M ModernBERT-Large, 1.5 GB | CUDA, ROCm, MPS, CPU | README claims "sub-15 ms"; its own table says **~18 ms** on GPU. Accuracy is 72.0 % vs Jev's 96.6 % on 49 tasks |
| Heman10x-NGU/Verdict-open-jev | 151M ModernBERT-base + GLiClass head, ONNX | CPU, WebGPU | **p50 35.58 ms** single-thread. Scores 48.07 % on TypeSafe's 337-case eval, vs 88.43 % for a 26B model |
| com-kotobalabs/open-jev-deberta-v3-large | 0.4B | H100, M1 Max | **28 ms for 10 questions on H100**; 1.8 s for 4 questions on M1 Max. Accuracy 85.4 % in-domain, 69 % out-of-distribution |
| razorback16/openjev | DiffusionGemma 26B-A4B | RTX PRO 6000 Blackwell, Apple MLX | **27–94 ms** for a single request |
| openjev/openjev (Hugging Face) | 27B BF16, plus FP8 and MLX quants | H100 | **~80 ms** for short text; ~210 ms for 1,460 tokens with 23 options. 84.0 % on 10k questions; CC BY-NC |
| ZefanCai/Open-Jev-9B | LoRA + head on Qwen3.5-9B | not stated | **No latency published.** ECE 0.0077 on test, 0.037 out-of-distribution |
| Meanblock/JEV-CPU | Qwen3-0.6B, fp32 | CPU | **~1 s** |
| featherless-ai/simple-jev | any Hugging Face model, reads label-token logits | CPU, CUDA, ROCm | **No measured latency** |

Speed bands:
- **Under 5 ms:** nothing is published.
- **Under 50 ms:** open-jev (CPU), von (GPU), Verdict (CPU), and the DeBERTa model (H100).
- **Under 500 ms:** the 27B models and razorback's 26B model.

On our box, a GPU-resident model competes for VRAM; the A6000 has about 1.5 GB spare per the dspark report. A CPU model competes with the host expert tier.

**Negative finding.** A summarizer read the README of heyjunpenn/awesome-jev, which catalogs about 832 projects. It found **no project on speculative decoding, drafts, MoE placement or KV cache**. I did not read the list myself.

## 2. Expert re-placement per request

**Our slot map is a device table.** `crates/gpu-deepseek41/src/chain/ffn.rs:509` takes `slots: &DeviceTensor<u32>`. The `SlotMap` docs (`crates/gpu/src/hybrid.rs:147-152`) say a chain uploads the host map to the card. So a swap that keeps each layer's count is a table write plus expert uploads, with no re-capture of kernel arguments. I did not check two things: whether the host tier's own copy can be swapped live, and whether any grid size depends on the per-layer count.

**What each signal gives:**
- **A classifier** gives a label such as code, Korean, prose or threads. That picks a corpus-average list. At best it reaches the in-domain held-out figure of 57.6–73.5 %, against 40.1–48.9 % for the mixed static list (B12 row, rig-log 09-23#v41-router-four-corpora). That is a gap of +10–33 pp, minus any misclassification.
- **Prefill router statistics** show this request's actual expert selections. They are strictly more specific than a domain label and cost nothing.
- **Prior art uses the model's own signals.** DAOP uses prefill activation to swap about half of each layer's experts between CPU and GPU. It swaps only during prefill, with a 1.05 activity-ratio threshold. It ran Mixtral and Phi-3.5 MoE on an A6000 and reports "up to 8.20 %" over caching and prefetching. fMoE's "semantic hint" comes from the MoE's **own embedding layer**, not an external model. It reports −47 % latency and +39 % expert hit rate on 3090s; a search snippet of an older version said +36 %.

**Arithmetic.** Inputs are the measured pinned H2D rate of 26.2 GB/s (26.15 with the compute stream saturated, `docs/research/v41-ports.md:38`), 16,773,120 B per expert, and 38 routed layers (2–39) holding 888 card slots.

| Quantity | Value |
|---|---|
| Upload one expert | 0.640 ms [derived] |
| Swap 4 per layer | 97 ms [derived] |
| Swap 8 per layer | 195 ms [derived] |
| Replace all 888 slots | 568 ms [derived] |
| Today's prefill, 200-token prompt | ~7.8 s [derived: 160 s ÷ 4096 = 39 ms/token] |
| Gain for +10–33 pp card share | 3.0–9.9 ms/step [derived, B12 slope ~0.30 ms/pp] |
| Tokens to repay a fully exposed swap | 10–189 [derived, 97–568 ms ÷ 3.0–9.9 ms] |

- **The swap hides under prefill.** Even a 200-token prompt takes seconds to prefill, so the upload fits on the copy stream.
- **The gain is a ceiling.** The B12 slope was 7.0–9.8 ms for 23.5–32.8 pp. E11 measured −4.5 ms against a predicted 7–10 ms, because out-of-sample card hit came in lower, so treat 3.0–9.9 ms as an upper bound.
- **It repays within the request.** The worst case of 189 tokens sits inside a 100–400-token request.
- **One cost is underived.** H2D reads host DRAM at 26 GB/s while the host tier reads at about 121 GB/s, so a swap overlapping decode would slow the host tier. I have no measurement for that.

**When a pre-prefill decision pays:**
1. **The list must be resident before any routing is seen.** Examples are a cold multi-tenant preload or a per-tenant card partition.
2. **The prompt is too short for prefix statistics.** Five tokens give 30 selections per layer over 384 experts.
3. **Time to first token cannot absorb the swap, and prefill is chunked so wide that every expert is touched.** Then prefill statistics arrive only after prefill ends.

We are batch-1 single stream with per-token prefill, so none of these apply today. Case 2 is also covered by starting from the static list and updating from the tokens as they are generated.

## 3. Classifiers that gate speculative decoding

| Work | Mechanism | Measured effect |
|---|---|---|
| SpecDec++ (arXiv 2405.19715) | Acceptance-prediction head on the draft; stops drafting when P(any reject) passes a threshold | 2.04×/2.26×/2.23× on Alpaca/GSM8K/HumanEval, which is **+7.2/9.4/11.1 %** over baseline speculative decoding. Llama-2-chat 7B draft, 70B target. Head size not in the abstract |
| DISCO (arXiv 2405.04304) | Classifier decides whether to keep drafting or verify | **+10 %** average over the best static lookahead, same text. The "2-layer FFN" detail comes from a search snippet, not the paper |
| Cascade (arXiv 2506.20675) | Utility = token gain ÷ verify cost, test-and-set phases, suspend when utility < 1 | On MoE, plain speculative decoding slows up to **1.5×**. Cascade limits the slowdown to **5 %** and gives **+7–14 %** over static K in vLLM on 5 MoE models |
| fergusfinn blog | Running average of acceptance or drafter confidence, plus a cost model | **Simulation, not a measurement** |

Cascade's 2–3× verification cost is batched verification of several tokens, the plain batch we already rejected in ktok. What transfers is its gating rule.

**What this means for us:**
- **A Jev-style model cannot gate per token.** Even the fastest published one, 5.3 ms, is 13–15 % of a 35–40 ms step.
- **We already have a free confidence feature.** The papers train heads because their drafts are models with logits. Our lookup draft has n-gram match length and occurrence count for nothing, and E5 found the winning share is n=3 real repeats.
- **Per-request drafter choice is the only Jev-shaped slot.** The choice is lookup vs DSpark by domain: E5 measured code 0.595 vs Korean 0.288. A Hangul-ratio rule or a running acceptance average does the same job without a model.

## 4. Recommendation

**(a) first. It is offline and takes minutes.** Extend `tools/ref/draft-accept.py` on the four corpora, and on the E5b greedy output once it exists.
- **Record** acceptance conditioned on match length n = 1, 2, 3, ≥4 and occurrence count.
- **Compare four policies:** always pair, pair only if n ≥ k, a logistic gate on these features, and an oracle.
- **Score** each policy as expected throughput Σ(1 + accepted) ÷ Σ(r if paired, else 1), at r = 1.21 and 1.29.
- **Decide:**
  - If the oracle beats always-pair by less than 2 %, drop gating.
  - If the n ≥ k rule gets within 1 point of the oracle, ship the rule. Measure it on the box with a same-lease A/B, knowing the ruler is ±1.0 % at four rounds.
  - Watch Korean. It is the borderline case, 0.288 against the 0.21–0.29 break-even.

**(b) second. Add a prefix-predicts-rest column to E17.**
- **Build** each layer's ranking from the first N router selections of a document, for N = 64, 256 and 1024 tokens.
- **Measure** byte-weighted card hit on the rest of the document, per corpus.
- **Compare** against the mixed static list (40.1–48.9 %) and the in-domain list, which stands in for a perfect domain classifier (57.6–73.5 %).
- **Decide:**
  - If prefix at N = 256 reaches the in-domain level, open E18 with prefill statistics and no classifier.
  - If prefix sits near static, drop E18.
  - A classifier is justified only if in-domain beats prefix by more than 5 pp at small N. It would then be a simple rule first and an encoder only if the rule loses.
- **Inputs** are existing router traces. No lease; box CPU for minutes.

**Drop:**
- Any external classifier in the decode loop.
- The hosted Jev API.
- GPU-resident 26–27B OpenJev variants, which would take VRAM.

Keep simple-jev's logit-reading only as the cheap post-change evaluator already noted on 09-21.

**Could not open or verify:**
- The DISCO PDF body.
- The SpecDec++ head size.
- fMoE's ablation separating semantic from trajectory search. The paper does not isolate it.
- The dev.to "Open-Source Jev Alternatives" article.
- The Hugging Face ZefanCai collection and dataset.

All latency figures are self-reported by their authors.

**Spots beyond the brief (report only, not touched):**
- `docs/plan.md:636`, the E18 row, says "그래프 재캡처 비용 포함". The kernel reads the map from a device table (`crates/gpu-deepseek41/src/chain/ffn.rs:509`), so a same-count swap may need no re-capture. The open cost is the host tier's copy of the map. That is XS to verify.
- `docs/plan.md:635`, the E17 row, has no prefix-of-same-document column, which is the column that decides E18. S.

## Sources
- https://github.com/Shalimov04/open-jev
- https://github.com/wfzyx/von
- https://github.com/Heman10x-NGU/Verdict-open-jev
- https://github.com/razorback16/openjev
- https://github.com/featherless-ai/simple-jev
- https://github.com/heyjunpenn/awesome-jev (summarizer read only)
- https://huggingface.co/openjev/openjev
- https://huggingface.co/ZefanCai/Open-Jev-9B
- https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large
- https://huggingface.co/Meanblock/JEV-CPU
- https://arxiv.org/abs/2405.19715 (SpecDec++)
- https://arxiv.org/abs/2405.04304 (DISCO, abstract only)
- https://arxiv.org/abs/2506.20675 (Cascade)
- https://fergusfinn.com/blog/adaptive-speculation/
- https://arxiv.org/html/2501.10375v2 (DAOP)
- https://arxiv.org/abs/2502.05370 and https://arxiv.org/html/2502.05370 (fMoE/FineMoE)
- Seen in search results only, not opened: https://dev.to/rupesh_poojary_ce8e5e7994/open-source-jev-alternatives-run-typed-calibrated-llm-decisions-locally-4dfb, https://huggingface.co/collections/ZefanCai/open-jev, https://huggingface.co/AlexWortega/openjev
- Repo inputs: `docs/plan.md` rows B12, E5, E11, E12, E16, E17, E18; `docs/research/v41-ports.md:38`; `crates/gpu/src/hybrid.rs:140-235`; `crates/gpu-deepseek41/src/chain/ffn.rs:509`
