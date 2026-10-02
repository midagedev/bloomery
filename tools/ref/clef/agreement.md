# Clef-Flash: each published GGUF against the BF16 release

Cloudflare's [Clef-Flash](https://huggingface.co/Cloudflare/clef-flash) answered by bloomery (`bloomery_serve_clef`, `POST /v1/systemone`) from each of [bartowski's GGUFs](https://huggingface.co/bartowski/Cloudflare_clef-flash-GGUF) (rev `d7f376ea`), with the release's own head file, against the release's own Python at BF16 (`tools/ref/clef_ref.py`, rev `17f0b0ad`). The requests are `suite.jsonl` (8, English) and `suite-ko.jsonl` (7, Korean): 31 questions. `e2e.py` runs a file and prints its row.

| file | size (GB) | tops /31 | English /16 | Korean /15 | max abs dp | mean abs dp | prefill tok/s (n 4,067) | head_ms median | misses (question: official, ours) |
|---|---|---|---|---|---|---|---|---|---|
| bartowski Q8_0 | 9.55 | 30/31 | 15/16 | 15/15 | 0.0136 | 0.0033 | 3,289 | 22.1 | risk (1 0.474, 0 0.481) |
| self-made Q8_0 | 9.53 | 30/31 | 15/16 | 15/15 | 0.0269 | 0.0034 | 3,266 | 22.1 | risk (1 0.474, 0 0.473) |
| bartowski Q6_K_L | 8.11 | 31/31 | 16/16 | 15/15 | 0.0418 | 0.0057 | 3,333 | 22.0 | — |
| bartowski Q6_K | 7.79 | 31/31 | 16/16 | 15/15 | 0.0253 | 0.0061 | 3,363 | 22.6 | — |
| bartowski Q6_K_S | 7.51 | 31/31 | 16/16 | 15/15 | 0.0335 | 0.0063 | 3,383 | 22.4 | — |
| bartowski Q5_K_M | 6.88 | 31/31 | 16/16 | 15/15 | 0.0517 | 0.0082 | 3,862 | 22.2 | — |
| bartowski Q5_K_S | 6.50 | 31/31 | 16/16 | 15/15 | 0.0566 | 0.0085 | 3,998 | 22.2 | — |
| bartowski Q4_K_L | 6.20 | 30/31 | 15/16 | 15/15 | 0.0765 | 0.0099 | 4,128 | 21.8 | risk (1 0.474, 0 0.482) |
| bartowski Q4_K_M | 5.84 | 29/31 | 14/16 | 15/15 | 0.0717 | 0.0117 | 4,322 | 22.0 | risk (1 0.474, 0 0.524); bug_report (yes 0.512, no 0.484) |
| self-made Q4_K_M | 5.63 | 30/31 | 15/16 | 15/15 | 0.1209 | 0.0208 | 4,402 | 22.4 | bug_report (yes 0.512, no 0.391) |
| bartowski Q4_K_S | 5.48 | 29/31 | 14/16 | 15/15 | 0.0784 | 0.0159 | 4,526 | 22.2 | risk (1 0.474, 0 0.553); bug_report (yes 0.512, no 0.488) |
| bartowski Q3_K_L | 4.66 | 30/31 | 15/16 | 15/15 | 0.0804 | 0.0141 | 4,167 | 22.3 | risk (1 0.474, 0 0.504) |
| bartowski Q3_K_M | 4.48 | 30/31 | 15/16 | 15/15 | 0.1024 | 0.0219 | 4,087 | 22.6 | risk (1 0.474, 0 0.480) |
| bartowski Q3_K_S | 4.26 | 29/31 | 14/16 | 15/15 | 0.1363 | 0.0309 | 4,033 | 22.6 | risk (1 0.474, 0 0.479); bug_report (yes 0.512, no 0.487) |

Not supported yet (refused by name at load, before any upload):

| file | size (GB) | blocking type |
|---|---|---|
| bf16 | 17.92 | BF16 sites |
| Q4_1 / Q4_0 | 5.94 / 5.48 | Q4_1 / Q4_0 sites |
| IQ4_NL / IQ4_XS | 5.83 / 5.23 | IQ4_NL / IQ4_XS sites and token_embd |
| IQ3_M / IQ3_XS / IQ3_XXS | 4.85 / 4.27 / 4.14 | IQ3_S sites (IQ3_XXS: also IQ3_XXS, IQ4_XS) |
| Q2_K / IQ2_M | 3.64 / 3.54 | Q2_K sites (IQ2_M: IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS); token_embd Q3_K |

Answers are compared with the official BF16 release (tools/ref/clef/e2e.py; |dp| = |p(ours, our top) − p(official, its top)|; tops and |dp| repeat exactly between runs). The official BF16 is itself near a tie on `risk` (top "1" at p 0.474). Prefill and head_ms come from one sitting, each server alone on the A6000, one request (functional, no lease); the same file moves about 3–4 % between sittings. head_ms is the host joint head, the same work for every file. Sizes are the files' bytes / 1e9; each sha256 equals Hugging Face's LFS oid (rev d7f376ea).
