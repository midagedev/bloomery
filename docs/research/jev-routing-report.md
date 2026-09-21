# "jev" and the 2026 routing trend: identification, survey, and what survives on our box

Investigation round, 2026-09-21. Written for the bloomery/rig-log lead.

**Provenance key.** Every claim below carries one tag:

- **[M]** MEASURED by the cited source, with the conditions I could actually read (hardware, model, baseline, N).
- **[C]** CLAIMED by the source with no data I could open behind it (vendor page, blog, tweet, README assertion).
- **[I]** MY inference — not measured by anyone, derived here.
- **[M-box]** measured on our machine, from our own docs.

**Opened vs. not opened.** I only cite URLs I fetched and read. Pages that appeared in search results but that I did not open are listed in §5 and are *not* cited as evidence anywhere above it.

---

## 1. Abstract

"jev" is **Jev**, the first public model from **TypeSafe AI** (founder Diogo Almeida, ex-OpenAI), announced **2026-09-15** and covered by TechCrunch on 2026-09-18 — i.e. **six days old at the time of writing**. It is not a language model and has nothing to do with MoE expert routing. It is a *decision* model: it emits typed, calibrated probabilities over a caller-defined output schema instead of text, trained with "RLCD" on synthetic data, priced at $42 per *billion* input tokens with output tokens free. The single largest use the community found for it in its launch week is exactly the one the user saw: **replacing an LLM call inside a model-router / classifier / judge**, which is category **(b)** in the round spec — request-level routing between models — and *not* categories (a), (c) or (d).

The "raises tok/s dramatically" part of the user's premise does not survive contact with the sources. **No source I opened — English or Korean — claims that Jev raises the decode throughput of any model.** What the sources claim, and in two cases measure, is *latency and cost per routing decision*, plus *pipeline* throughput from sending more traffic to cheaper models. The Korean coverage says the same thing in the same units: "70 ms 의사결정으로 채팅 모델보다 빠른 라우팅" — milliseconds per decision, never tokens per second. The most likely origin of the tok/s framing is TypeSafe's published API **rate limit of "250,000 tokens per second"**, which the vendor's own docs present as a quota, not a decode rate [M, docs.typesafe.ai/models]. And the one reproducible third-party integration I opened measures Jev making a router **slower**: p95 end-to-end 77.93 ms → 490.38 ms versus the same router's local deterministic features, with identical routes and accuracy [M, slo-router README]. TypeSafe itself says its multiples are "a ceiling rather than what you'll measure" [C].

On our machine, the honest transfer is **zero for the engine and possibly positive for the serving layer, but only as an output-changing cascade**. Nothing in this trend touches the batch-1 decode of DeepSeek-V4.1-Flash. Our own docs already bound the categories that would: expert-ID prediction buys ≈0 when weights are in DRAM, identity caching is bounded by a router trained flat (chance overlap k²/N = 0.094 experts), and the only route past the residual chain is output-changing deferral [M-box, `hybrid-lit-report.md`, `hybrid.md`].

---

## 2. Identification of "jev"

### 2.1 Verdict

**Jev = TypeSafe AI's "System One Model", announced 2026-09-15.** Confidence: high. It matches all three parts of the user's description — it is a *model*, it is *being used for routing*, and there is a *boom* (launch week dominated Hacker News; a second-hand ecosystem of clones appeared within days).

### 2.2 Evidence, source by source

| source | opened | what it establishes | tag |
|---|---|---|---|
| [TechCrunch, 2026-09-18](https://techcrunch.com/2026/09/18/a-new-kind-of-ai-model-from-a-chatgpt-inventor-is-thrilling-developers/) | yes | Transformer that "produces probabilities, or what the company calls 'calibrated decisions'"; TypeSafe AI, founder Diogo Almeida (ex-OpenAI, RLHF); cannot hallucinate because "users define the outputs in advance"; output tokens free, input metered per billion | [C] — reporting, not benchmarking |
| same | yes | Vercel case study: replacing an OpenAI model gave "results five to 18 times more quickly and with greater accuracy"; Bryo AI found Gemini slightly *more accurate* on email classification | [C] — secondhand, no methodology. The WebFetch summariser itself flagged the Bryo cost sentence as possibly inverted; I do not carry that number. |
| [latent.space AINews, 2026-09-15](https://www.latent.space/p/ainews-jev-a-system-one-model-that) | yes | Announced 2026-09-15 after two years in stealth; "decides/classifies/routes/scores"; RLCD training; 20–200× faster, 40–400× cheaper; parallel sampling; HN front page all day; engineers read it as a drop-in for "structured classifiers / judges / routing policies" | [C] for the multiples, [M] only for the fact of the launch date and reception |
| [typesafe.ai](https://typesafe.ai/) | yes | Vendor headline: "193.6x Faster, 244.6x Cheaper", 0.114 s vs 8.566 s on a "System One task"; "$42 Per Billion input tokens", "238x Lower input price than Claude Fable 5.1" | [C] — vendor, single unnamed workflow, no N |
| [flaviocopes.com/jev](https://flaviocopes.com/jev/), updated 2026-09-18 | yes | TypeSafe quotes **"70 to 500 milliseconds end to end"**, most queries ~100 ms from US West Coast; and TypeSafe **explicitly says its multiples are "the high end of what you'd see in the real world, so treat them as a ceiling rather than what you'll measure"** | [C] for the latency, but the ceiling caveat is the vendor's own and is the most useful sentence in the whole discourse |
| [docs.typesafe.ai/models](https://docs.typesafe.ai/models) | yes | `jev-1.13.0`, aliases `jev-latest`/`jev-preview`; 64k total budget, 32k for `state` + longest question; **rate limits "250,000 tokens per second / 1,200 requests per minute"**, adjusting dynamically; no latency figure published | [M] — this is the vendor's own spec sheet |

### 2.3 Near-misses and the negative sweeps

Queries I ran, with their results, so the absence is checkable:

- `site:github.com "jev" llama.cpp OR vllm OR sglang OR ktransformers expert routing issue` → returned KTransformers, SGLang's `sgl-router` refactor, and the llama.cpp discussion #24528 on MoE expert VRAM caching. **No "jev" in any MoE-inference tracker.**
- `arxiv "jev" mixture of experts routing 2026` → returned a dozen 2026 MoE-routing papers (equifinality, path-constrained MoE, routing-free MoE, geometric routing). **No arXiv paper titled or about "jev".**
- `huggingface models "jev" router classifier 2026` → returned a **live `jev` model tag** on Hugging Face plus three concrete artefacts (§2.4). This is the sweep that confirmed the ecosystem, not just the vendor.

Other things named "jev" that are not this: **JEV = Japanese encephalitis virus** is the dominant meaning of the trigram in the literature and will pollute any bare search; it is not what the user saw. I found no "jev-router", "JEVA", "jve", "jeb" project in the inference space.

### 2.4 The open clones — this is what makes it a "trend" and not one vendor

Within days of launch a self-hostable layer appeared. This matters to us more than TypeSafe's API does:

- **[featherless-ai/simple-jev](https://github.com/featherless-ai/simple-jev)** — "turn any open model into a classifier/jev endpoint". Mechanically: apply the chat template, tokenise each question, **read the model's next-token logits for the label tokens** and assemble a JSON of choices / rubric scores / entailment judgements — no classifier head, no generation. It also finds the **exact common token prefix** across questions about the same context and reuses one KV cache. Examples name Qwen 3.5 0.8B, Gemma 4 26B-A4B. **It reports no measured speed at all**; its "4,200 → 1,200 tokens" figure is explicitly labelled as avoided input processing, "not a measured latency ratio". [M, README — the honesty is the finding]
- **AlexWortega/openjev** and **com-kotobalabs/open-jev-deberta-v3-large** — two further repos whose names appear under the HF `jev` tag. **I did not open either model card and make no claim about what they contain**; I record only that the names exist in the listing, as evidence of ecosystem breadth.
- Prior-art dispute: [HN 49736660](https://news.ycombinator.com/item?id=49736660) — a developer claims to have open-sourced a non-autoregressive "probability prediction with a JSON schema" architecture in March 2025, with paper, weights and dataset, and that a frontier lab later presented the same idea with none of those. Commenters largely agree the differentiator was productisation. One commenter reports local architectures beating the cloud one "marginally" with 712 ms vs 0.14 ms latency — **an uninterpretable pair of numbers with no setup given; I record it only as the temperature of the thread.** [C, and weak]

### 2.5 The Korean-language surface — where the user most likely saw it

Because the framing "routing that raises tok/s" did not appear anywhere in English, I swept the Korean surface (`jev 라우팅 tok/s 속도 모델`). It is active and it dates the boom precisely:

- [okky.kr/spaces/it-news/1564103](https://okky.kr/spaces/it-news/1564103) — "Jev: 40~400배 저렴하고 20~200배 빠른 자동화 특화 모델". Vendor multiples, relayed. [C]
- [promppy.com/item/1805258](https://www.promppy.com/item/1805258), **2026-09-20** — "Jev 모델: **70ms 의사결정으로 채팅 모델보다 빠른 라우팅** 구현하기". I opened this one. It frames Jev as a routing layer at 70–500 ms, names request routing / tool-call decisions / support-ticket classification as the fit, and warns that probabilistic answers inside a fixed schema still need validation. **It gives no tokens-per-second figure, no throughput comparison, and no baseline.** [M — read directly]
- promppy also ran two other Jev items (1737806 on the launch, 1745487 on ecosystem spread) and wikidocs/머니업그레이드 carried "출력 무료·70~500ms 초고속". Not opened individually.

> **[I] The Korean discourse says exactly what the English one says: *milliseconds per decision*, never tokens per second.** The user's "tok/s가 확 올랐다" is, on this evidence, a compression of "the routing step got ~10× faster and the pipeline got cheaper" — plus, plausibly, TypeSafe's published **250,000 tokens/second** rate limit being read as a speed. I could not find a post making the tok/s claim literally; see §6.

> **[I] The identification is secure, but the thing identified is six days old.** Treat every multiple in §2.2 as launch-week marketing until a third party reproduces it. Two have tried; §3.1 has what they got.

---

## 3. The trend, by the spec's categories

### 3.1 Category (b) — routing requests between models. This is where Jev lives.

**Mechanically**, all of these sit *in front of* inference: a cheap decision function reads the request and picks which model (or which tier, or whether to escalate) serves it. None of them changes how any model decodes a token. **All of them are output-changing at the system level** — a different model answers — even though each individual model's decode is untouched.

**The two measurements I could open, and they disagree with the marketing:**

| source | what was measured | number | conditions / baseline | tag |
|---|---|---|---|---|
| [DevelopersIO / classmethod, Sept 2026](https://dev.classmethod.jp/en/articles/jev-for-llm-model-routing/) | Jev as the classifier in NVIDIA NeMo Switchyard's 4-tier routing task, called directly against the TypeSafe API | **median 0.643–0.674 s, mean 0.652–0.690 s per call**; cost $0.000025–0.000027/call; confidence 1.0 except the "medium" tier at 0.57–0.67 | **N = 40 total (10 per tier)**; author states these are direct API calls, **not** integrated into Switchyard, so integration overhead is excluded; "~3× faster than Gemini 3.5 Flash, ~10–11× faster than DeepSeek V4 Flash" | [M], small N, weak baseline |
| [zeeshan8281/slo-router](https://github.com/zeeshan8281/slo-router) README | swapping Jev 1.13 in as the semantic-feature source of an SLO-aware OpenAI-compatible router, versus the router's own deterministic local features | **p95 end-to-end 77.93 ms → 490.38 ms**, "preserved the same routes and accuracy" | bundled fixture; README explicitly warns "do not present the eight-row demo as a model benchmark"; the live-OpenRouter run is in `results/live-jev-analysis.md`, **which I did not open** | [M] for the fixture number, and it is the sharpest datum in this report |

> **Finding 1 [I].** The two independent parties who measured Jev got **~0.65 s** (classmethod) and **~0.49 s p95** (slo-router) per decision. The vendor's own comparison workflow is **0.114 s** [M, typesafe.ai] and the quoted envelope is 70–500 ms [C, flaviocopes]. So third-party wall-clock is **4–6× slower than the vendor figure**, which is the ordinary API-vs-datasheet gap, not misconduct — but it changes the engineering conclusion. A router whose decision costs 0.5–0.65 s is a **latency tax** unless the model it avoids would have cost more than that, and slo-router's result is exactly that case: against *deterministic local features* Jev is 6× worse for the same routes.

**The baseline problem, stated plainly [I].** "40–200× faster than frontier LLMs" compares a non-generative classifier against an autoregressive model generating text. That is a category comparison, not a speedup. The fair baseline for "route this request" is a fine-tuned small encoder — DeBERTa-v3, a 0.5B classifier, or a regex — and those are also ~100× faster than a frontier LLM and run locally for free. The `open-jev-deberta-v3-large` name appearing under the HF `jev` tag is the market making this observation. **TypeSafe itself agrees:** its multiples are "the high end of what you'd see in the real world … a ceiling rather than what you'll measure" [C, flaviocopes.com/jev, 2026-09-18].

**One line worth carrying to the user [I].** Note what the classmethod baseline actually is: Jev is "~10–11× faster than **DeepSeek V4 Flash**" *at emitting a classification over an API*. In this discourse a DeepSeek model is the thing being **replaced**, not the thing being sped up. That is the whole relationship between the jev trend and our machine.

**The surrounding router landscape (2025–2026).** From [pakodas, *State of LLM Routers in 2026*, 2026-07-28](https://pakodas.substack.com/p/llm-routers) and [lowpassfilter, 2026-09-12](https://lowpassfilter.substack.com/p/your-model-router-must-earn-its-keep), both opened:

- Named production systems: **OpenRouter Fusion** (fan-out to several frontier models, synthesise), **Devin Fusion** (frontier model + cheap "sidekick" throughout a task), **Harvey** (workflow-type routing), **Vercel AI Gateway**, **Splunk**, **vLLM Semantic Router**. Neither article gives latency, overhead, or throughput numbers for any of them [C].
- Research anchors: **RouteLLM** — 85 % cost reduction on MT-Bench at 95 % of GPT-4-Turbo quality, matrix-factorisation router sending 14 % of queries to the strong model [C via these two secondary sources; I did not open the RouteLLM paper]. **FrugalGPT** — savings "reaching 98 % in the strongest setting" [C, same caveat]. **HydraFusion** — 36–67 % less estimated cost than Claude Opus 5 across three benchmarks, quality from 1.5 points below to 4.9 above [C].
- **Neither routing survey mentions Jev.** The July survey predates it; the September 12 one predates the launch by three days. **[I] This is the clearest evidence that "jev routing" is a three-to-six-day-old phenomenon, not an established trend** — the two people who write about routers for a living have not yet written about it.
- The lowpassfilter piece's own thesis is the one worth carrying: it deliberately gives no universal numbers and argues routing must be measured in your own context. It quantifies **nothing** about added latency or break-even, which is exactly the number slo-router supplies.

**What survives on our box: nothing, at the engine layer.** Our decode is one model on one machine at batch 1. There is no second model to route to and no per-request cost to save. [I]

### 3.2 Category (d) — draft/verify schemes marketed as routing

- **RLM-Cascade** ([arXiv 2606.22840](https://arxiv.org/pdf/2606.22840), 2026-06-23) — response-level speculative decoding: a small draft LLM writes a whole candidate response, a large verifier validates/corrects. Positioned against token-level speculative decoding (Medusa, EAGLE) and cascades (LLM-Cascade, FrugalGPT). **I opened the PDF but the summariser could not extract the experimental table**; hardware, batch, models and the actual speedup are **not verified** — see §5. Output preservation is *claimed* by construction but response-level verification is not token-exact.
- The generic claim circulating with this family is "**3–5× throughput on appropriate workloads**" [C, from the search surface only — I did not open a source that measures it, and I do not treat it as evidence].
- **What survives on our box:** we already occupy this lane. The DSpark draft takes us 17.7 → 22.8 tok/s [M-box], and `plan.md:124` (MUL-43) has the τ threshold for whether n-gram drafting is even worth opening. Response-level cascade is the *output-changing* cousin of what we already run exactly. [I]

### 3.3 Category (c) — routing that changes which computation runs

- **Informed Routing** ([arXiv 2510.13831](https://arxiv.org/abs/2510.13831)) — token-level, not request-level: a **Lightweight Feature Forecaster** estimates a unit's output *before* the routing decision, turning execute-or-skip into **execute-or-approximate**. Claims state-of-the-art efficiency/performance trade-offs across sparsity levels and >50 % less training time. **The abstract gives no hardware, model size, batch, or concrete speedup** [C for the trade-off claim]. The abstract's phrase "preserves model fidelity" is about benchmark scores, not bit-exactness — **this is output-changing** and I classify it as such against the paper's own framing. [I]
- **What survives on our box:** this is the same family as KTransformers' **Expert Deferral**, which our lit report already has measured: **+33 % decode on DeepSeek-V3 at −0.5 % average LiveBench accuracy at 6 deferred experts**, collapsing to −6.7 % at 8; the skip variant is **−13.3 % at 6 skipped** [M-paper, `hybrid-lit-report.md:607-608`]. Our own doc already notes our shared-expert share is smaller, so our gain would be smaller too [M-box, `hybrid.md:53`].

### 3.4 Category (a) — MoE expert-routing prediction / pre-gating

Nothing in the jev trend is in this category, and our docs already close it. Recording the bound so the lead does not have to re-derive it:

- **Prediction of expert identities can only buy a prefetch**, and prefetch cannot start the computation, because `e_ℓ(x_ℓ)` needs `x_ℓ` [M-box, `hybrid-lit-report.md:596-601`]. Our weights are already in DDR4, so the prefetch has nothing to fetch: **≈ 0**, and a wrong prefetch *costs* bandwidth.
- **Identity-based caching is bounded by a flat router.** V4.1's selection-only bias is trained for load balance, the shared expert absorbs the common-mode signal, and N/k = 64 makes two tokens' chance overlap **k²/N = 0.094 experts** [M-box, `hybrid.md:39`]. A better policy is worth **+6–8 pp of hit rate at 25 % capacity** (HybriMoE) ≈ **3 % of our step** [M-paper via `hybrid-lit-report.md:523-524`].
- **Transfer vs. host compute is decided by one ratio:** β_h/β_link = 147.7/26.2 = **5.6×** against transfer, independent of expert size and quant, flipping only above ~800 concurrent tokens [M-box/D, `hybrid-lit-report.md:316-322`].

### 3.5 What is genuinely new in the jev trend, stripped of hype

**[I] One idea, and it is in the open clone, not the vendor model.** `simple-jev` does *constrained decoding degenerated to a single step*: instead of generating a decision, read the next-token logits over the label tokens and normalise. That is a real, output-preserving-by-construction trick — the "decision" is a projection of a distribution the model computes anyway — and it costs one prefill and **zero decode steps**. It is not new science (it is how zero-shot classification with LMs has always worked), but it is newly packaged and newly attached to a pricing model where output tokens are free. The second idea, **shared-prefix KV reuse across several questions about one context**, is prompt caching, which we already have as a planned row (`plan.md:79`, C2).

---

## 4. Exact vs. output-changing

| thing | changes model output? | quality cost, if measured |
|---|---|---|
| Jev / any request-level router (b) | **Yes, at system level** — a different model answers. Each model's decode is untouched | RouteLLM: 95 % of GPT-4-Turbo quality at 14 % strong-model traffic [C]; HydraFusion: −1.5 to +4.9 points [C]; Bryo AI found Gemini slightly more accurate than Jev on email classification [C] |
| `simple-jev` logit-reading (b, mechanism) | No, for the *decision* — it reads the base model's own distribution | none by construction; label-token choice is a design decision, not an approximation [I] |
| Response-level cascade / RLM-Cascade (d) | **Yes** — verification is at response level, not token level | **not obtained** (§5) |
| Token-level speculative decoding (our DSpark) (d) | **No** — exact, verified per token | none; 17.7 → 22.8 tok/s [M-box] |
| Informed Routing / execute-or-approximate (c) | **Yes** — units are approximated, not executed | not stated in the abstract [C] |
| Expert Deferral (c) | **Yes** | −0.5 % avg LiveBench at 6 deferred, −6.7 % at 8 [M-paper] |
| Expert Skipping (c) | **Yes** | −13.3 % at 6 skipped, −88.7 % at 8 [M-paper] |
| Expert-ID prefetch (a) | **No** | none — and ≈ 0 gain for us [M-box] |
| Expert row-splitting across CPU/GPU (our §2.6 lever) | **No** (float summation order only) | band gate, not bit gate; ceiling **1.15×** at our VRAM, 1.37× uncapped [M-box/D] |

---

## 5. Verdict for the lead

**Is our engine plan missing anything?** At the engine layer, **no**. Not one item in this trend operates below the request boundary, and the four categories that do — (a), (c), and the two halves of (d) — are already bounded in `hybrid-lit-report.md` with numbers better than anything the jev discourse has produced. The trend's headline multiples are a classifier being compared to a text generator; our bottleneck is DRAM bandwidth on 3.34 GB of expert weights per token.

**Where it touches us is `plan.md:25` and `plan.md:81` — the serving layer, and there is a real gap there.** Stage 4 of the plan says, in five words, "DSpark draft + scheduler, fast/slow model routing", with a pass criterion only about acceptance rate and tok/s. **That single phrase is the entire (b) category and it is undefined**: it does not say what decides, what the decision costs, or what quality is allowed to move. The jev trend's one durable contribution is the number that makes that row decidable — **slo-router's 77.93 ms → 490.38 ms** shows a router's decision cost can exceed everything it saves, and the classmethod 0.65 s shows that a hosted decision model is not a free lunch at our latency scale. If we ever build "fast/slow routing", the decision must be **local and sub-millisecond** (logit-read on the draft model we already load, or a DeBERTa-class encoder), never a network call.

**The single measurement on our box that would tell.** Take the prompts we already have — the 32-prompt fixture behind `gate-prompts` plus real traffic from `llm.service` — and offline, with **no engine change and no GPU**, answer: *what fraction φ of them does the **DSpark draft model answering alone** get acceptably right?*

Then compute effective throughput the only way that is meaningful — **total tokens generated ÷ total wall time over the whole prompt set**, not a blend of rates:

```
  T_eff = Σ tokens / [ φ·(tokens/draft_rate) + (1−φ)·(tokens/17.7) ]
```

and compare it against **22.8 tok/s**, which is what the *exact* draft+verify path already gives us [M-box]. **If T_eff does not clear 22.8, the entire (b) category is closed for this box** — because a cascade would then be paying quality for throughput we already have for free — **and `plan.md:25` should say so in one line.** This is the same shape as the MUL-43 τ test at `plan.md:124` and can share its harness; unlike everything else in this report it costs one offline run.

(I deliberately do not offer "V4.1 with thinking off" as the cheap arm: it is the same weights at the same 17.7 tok/s. Thinking-off reduces the token *count*, not the decode *rate*, so it belongs to a different measurement.)

**Second-order note [I]:** `simple-jev`'s logit-reading is worth 20 minutes of the lead's attention *not* as a router but as a **cheap evaluator**. Our gates judge logits and argmax; a single-step label projection over the model we are already running is a way to ask "did this answer stay acceptable" after an output-changing change (Expert Deferral, informed routing) without standing up a judge model. That is a gate-authoring tool, not a speed lever.

---

## 6. Not obtained / could not verify

- **RLM-Cascade (arXiv 2606.22840)** — PDF fetched but the experimental table did not extract. Hardware, models, batch size, baselines and the actual speedup number are **unverified**. Its claim of output preservation is structural, not measured, as far as I can see.
- **`results/live-jev-analysis.md`** inside slo-router — the *real* benchmark against live OpenRouter. The README points at it; I only read the README's fixture line. This file is the single most valuable unread artefact in this report.
- **RouteLLM, FrugalGPT, HydraFusion primary papers** — every number I quote for these came through two secondary blog posts. Not opened, not independently verified.
- **Informed Routing (2510.13831)** — abstract only; no hardware, model, batch, or numeric speedup obtained.
- **AlexWortega/openjev** and **com-kotobalabs/open-jev-deberta-v3-large** model cards — seen in a HF listing, not opened. No claims made about them.
- **TypeSafe's "193.6× / 244.6×" workflow** — the vendor does not say what the task was, what the LLM was, or N. Uncheckable as published.
- **The user's premise itself** — I could not find any post, thread or README claiming that jev routing raises *decode* tok/s, in English **or in Korean** (§2.5). The closest artefacts are TypeSafe's 250,000 tok/s rate limit and the Korean framing "70 ms 의사결정으로 채팅 모델보다 빠른 라우팅". Not swept: X/Twitter and Discord (authenticated, unreachable from here), Zhihu / WeChat mirrors, Qiita. **If the lead can get the original post the user saw, that link settles whether §1's inference is right** — and it is the only thing in this report I would change my conclusion for.
- **promppy items 1737806 and 1745487, okky 1564103, the wikidocs piece** — Korean coverage seen in listings, only 1805258 opened.
- The "3–5× throughput" figure for speculative cascades appeared only in a search-result summary; I did not open a source for it and it is not evidence.

---

## 7. Improvement opportunities in our docs, outside this round's scope

Report only — I changed nothing.

- `/Users/hckim/repo/bloomery/docs/plan.md:25` — stage-4's "빠른/느린 모델 라우팅" is the whole (b) category compressed to five words with no pass criterion for what decides or what the decision may cost.
- `/Users/hckim/repo/bloomery/docs/plan.md:81` — C4 says "k* ≈ 2.2 실측" while `:124` opens MUL-43 with a τ threshold of 1.3/1.5; two speculative-decoding numbers in one file with no line relating them.
- `/Users/hckim/repo/bloomery/docs/plan.md:121` — the CPU tok/s table stops at 2026-09-20 and delegates to `HANDOFF.md`; plan.md is no longer the single owner of its own scoreboard.
- `/Users/hckim/repo/bloomery/docs/plan.md:246` — a measured negative result (THP never fired, `AnonHugePages` 0 kB, effect unconfirmed) is buried *inside the argument list of a code span*, `madvise(MADV_HUGEPAGE — …)`; it reads as a code sample and is easy to miss as a finding.
- `/Users/hckim/repo/bloomery/docs/research/hybrid.md:49` — every multiplier in that doc is gated on WKS-35 (β_h 147.7 vs 215–233) with no owner or date on the blocker.
- `/Users/hckim/repo/bloomery/docs/research/hybrid-lit-report.md:424` — the ReMoE row carries +26.4 %/+27.2 % numbers in a table while its own read-level column says "abstract/secondary only — not obtained".
- `/Users/hckim/repo/bloomery/docs/research/hybrid-lit-report.md:576` — CoX-MoE (arXiv 2605.17889) is the one paper that might pre-empt our row-splitting lever and its PDF would not extract; still unread.
- `/Users/hckim/repo/bloomery/docs/research/hybrid-engines.md:47` — the `model.py` read pass on V4.1's one-sublayer-late hyper-connection is an open TODO with no owner, and it is the only thing standing between "cross-layer overlap is impossible" and a possible exact win.

DONE-jev
