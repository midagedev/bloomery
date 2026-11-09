# Quantization segment: chart data and storyboard

One CSV per chart, written by the lead from the named sitting's log; a chart renders only rows whose `status` is `measured` (or
`derived`, drawn hatched and labelled). Every cell that a sitting has not filled is empty and its row `status=placeholder`; no
value here is invented. Comparisons, predictions and sittings: `quant-promo.md`; cards: `docs/cards/quant-promo-*.card`.

Common columns, on every CSV that carries a timing (so each number names its conditions, AGENTS.md "Measuring"):
`card` (name and power limit, e.g. `RTX A6000 300W`, or `none` for the CPU arms), `driver`, `commit` (bloomery, 40 hex),
`tool` (`bloomery-quantize`, `bloomery-quantize --cpu`, `llama-quantize 78feec062`), `model_file` (full path), `model_bytes_in`,
`imatrix` (full path), `rules` (full path), `threads` (the arm's own, from its first log line), `lease` (the card file's sha256 from
the `[lease]` line, or `none` for a production-log row), `status` (`measured` | `derived` | `placeholder`), `source` (the log or
file the value was read from, or the derivation in one clause).

## Chart 1 - `speed_k31.csv` (per-stack rate, K3.1, bars: card / host rule / llama-quantize)

Columns: `arm` (`card`|`cpu_bq`|`cpu_llq`), `type` (`IQ1_S`|`IQ2_XXS`), `stacks` (IQ1_S stacks inside the window), `weights` (= stacks x
9.865e9), `seconds` (their summed seconds), `weights_per_s`, `mweights_per_s`, `seconds_per_stack`, `window_s`, + the common columns.

| arm | row today | source / status |
|---|---|---|
| card | `seconds_per_stack` 13.4 (13.2 .. 13.9), `mweights_per_s` 736 | `/root/k31quant/gpuset.log`, `iq_quant_set` (predecessor), A6000 beside a GPU imatrix, unleased: `measured`, **not** the leased row |
| cpu_llq | 46.2, 214 | `/root/k31quant/quantize.progress`, 13 stacks / 600 s, nice 10, 64 threads, contended: `measured`, **not** the leased row |
| cpu_bq | empty | `placeholder`: sitting S1, card `quant-promo-k31-cpu-bq` (prediction 150 .. 300 Mweights/s) |
| card (leased) | empty | `placeholder`: S1, `quant-promo-k31-card` (700 .. 1000) |
| cpu_llq (leased) | empty | `placeholder`: S1, `quant-promo-k31-cpu-llq` (190 .. 290) |

Derived on the chart: ratio card / cpu_llq = 736 / 214 = 3.4x [derived from the two measured rows, until S1 replaces them].

## Chart 2 - `speed_k31_total.csv` (whole K3.1 run, two stacked bars: CPU path, card path)

Columns: `path` (`cpu`|`card`), `phase` (`experts`|`set_write`|`trunk_and_copy`|`total`), `seconds`, + `status`, `source`.

| path | phase | seconds | status |
|---|---|---|---|
| cpu | total | 13,970 | `derived` (`k31-quant.sh quantize --pred-s`: 2.7227e12 / (40 x 5.4e6) = 12,600 s + read 975 s + write 387 s; the measured stack rate, 214 against 216 Mweights/s, agrees) |
| card | total | 5,450 | `derived` (`iq-quant-card.md`: 276 x 14.3 s + 440 s set write + llama-quantize over the set 1,060 s; 254 x 13.4 + 22 x 16.4 = 3,765 s card from the production log gives 4,200 + 1,060 = 5,260 s) |
| cpu / card | experts, set_write, trunk_and_copy | empty | `placeholder`: the production v0 run's own logs (chain12: `gpuset.log` end line, `quantize.log` and the runner's wall lines), unleased, contended by the v1 imatrix: `measured` once the run ends |

## Chart 3 - `speed_mid.csv` (MiMo-V2.6-Flash-RL MXFP4, 167 GB, end to end, three bars + stage split)

Columns: `arm` (`A_card`|`B_cpu`|`C_llq`), `stage` (`quantize`|`set_write`|`llama_quantize`|`total`), `seconds`, `weights`
(3.028e11 for the experts), `input_drive` (`nvme0` | `hdd`), `output_drive`, + common columns.

| arm | total, derived | band | status |
|---|---|---|---|
| A_card | 540 (409 + 44 + 85) | 450 .. 750 | `derived`; `placeholder` for the measured row: sitting S2, `quant-promo-mid-a-card` |
| B_cpu | 1,545 (1,415 + 44 + 85) | 1,100 .. 2,100 | `derived`; `placeholder`: `quant-promo-mid-b-cpu` |
| C_llq | 1,445 (1,415 + 28) | 1,200 .. 1,750 | `derived`; `placeholder`: `quant-promo-mid-c-llq` |

Derived on the chart: speed-up C / A = 2.7x (band 1.6 .. 3.9). A row whose `input_drive` is `hdd` is drawn separately and labelled
read-bound (`quant-promo.md` 1b, prerequisite 1).

## Chart 4 - `terms.csv` (where a stack's time goes on the card: stacked bar)

Columns: `type`, `term` (`read`|`host_words`|`upload`|`widen`|`search`|`finish`|`download`|`pack`), `seconds`, + common columns. Read from
`just quant-measure` / `iq_quant_probe`'s `terms:` line (`tools/ref/quant-measure.sh summarize`).

| type | term | seconds | status |
|---|---|---|---|
| IQ1_S | search | 9.05 .. 9.12 | `measured` (`iq-quant-card.md`, unleased, beside a 64-thread imatrix); the rest of the call 14.0 s: read 3.4 .. 4.4, words 1.9, upload 0.33, widen 0.13, finish 0.06, download + pack 0.8 .. 2.8 |
| IQ2_XXS | search | 11.8 | `measured`, same source; call 17.4 s |
| both | all terms, leased | empty | `placeholder`: `just quant-measure` under `docs/cards/quant-measure.card` (predict 690 .. 1090 Mweights/s) |

## Chart 5 - `quality.csv` (our K3.1 quant against the released model, grouped bars per category)

Columns, exactly `tools/ref/k31-eval.py score`'s: `set` (`all`|`en`|`ko`|`code`), `prompts`, `positions`, `top1_agree`, `top1_ci95`,
`dnll_nats`, `dnll_ci95`, `kl_top`, `kl_top_ci95`, `kl_union`, `kl_union_ci95`; plus `run` (`v0`|`v1`|`ud_k3`), `imatrix` (full path),
`calibration_text` (full path), `map` (`tight`|`trunk-q5`|`ud`), `type` (`IQ1_S`), `set_dir` (full path), `student_commit`,
`teacher` (`moonshotai/kimi-k3.1-pro`, no-thinking render), `card`, `status`.

| run | imatrix | status |
|---|---|---|
| v0 | `/root/k31quant/kimi-k31-c8192x4.imatrix.gguf` (text `calib-mix-c8192x4.txt`, md5 `c0ec381caf5f97242b0a61ba0c5b6499`) | `placeholder`: the lead scores the production set |
| v1 | the chat-rendered corpus `calib-chat-c8192.txt` imatrix (chainv1, `--seqs 3`) | `placeholder`: needs its own set + score |
| ud_k3 | none (K3's `UD-IQ1_S`, a different model) | `placeholder`, optional |

## Chart 6 - `exact.csv` (identity frames)

Columns: `frame` (`fixture_whole`|`mid_whole`|`k31_check`), `file_a`, `file_b` (full paths), `scope` (`whole-file`|`rows`), `bytes`,
`sha256_a`, `sha256_b`, `equal` (`1`|`0`), `check_rows` (for `k31_check`: N), `tensors_checked`, `first_diff` (tensor, expert, row, byte, or empty),
`commit`, `card`, `status`.

| frame | what | status |
|---|---|---|
| fixture_whole | K3 fixture: `bloomery-quantize` + `llama-quantize` against `llama-quantize` alone | proved by `gate-gpu-iq-quant`'s end-to-end clause; `placeholder` for the two hashes (the lead's gate record) |
| mid_whole | MiMo finals A, B, C: three sha256 | `placeholder`: S2, predicted equal |
| k31_check | `bloomery-quantize --check N` over K3.1's stacks, and the gate's flipped-byte stop | `placeholder`: N and the tensor count |

## Chart 7 - `size.csv` (file size by map, and what fits where)

Columns: `map`, `experts_gb`, `trunk_gb`, `total_mib`, `total_gb`, `bpw`, `fits_a6000_trunk` (`1` if the trunk <= 48 GB), `fits_card_plus_host_ram`
(`1` if total <= 48 + 251 GB), `status`, `source`. Rows from `quant-promo.md` Size: all `derived` (`llama-quantize --dry-run`):
tight 553,113.74 MiB (580.0 GB, 1.67 bpw), trunk-q5 559,452.10 (586.6, 1.69), ud 566,514.80 (594.0, 1.71), all-IQ2_XXS 702,428.62 (736.5, 2.12),
source 1,488,919.97 (1,561, 4.49). The `fits_*` columns are arithmetic only (trunk 30.36 GB against 48 GB: 1; totals against 299 GB: 0); real
placement is the plan's line from the final speed sitting.

## Storyboard (8 beats)

1. Hook: the 1.56 TB source on a bar, next to the 580 GB tight set (chart 7, two rows).
2. The old way: `llama-quantize` on 64 threads, a stack counter ticking at 46 s a stack (terminal clip from S1, chart 1's CPU bar).
3. Our way: `bloomery-quantize` on the card, the same counter at 13 s a stack (terminal clip, chart 1's card bar, ratio label).
4. Whole run: 3.9 h against 91 min (chart 2, two stacked bars, `derived` hatching until the production run's logs fill them).
5. A real model end to end: MiMo's three bars (chart 3), with the card arm's stage split (chart 4 inset).
6. Exact: two `sha256sum` lines, identical, then the three MiMo hashes (chart 6); one flipped-byte `--check` stop named by tensor and row.
7. Quality: EN / KO / code, v0 and v1 imatrix, top-1 and KL against the released model (chart 5), one line: bytes are ggml's.
8. Fit: the file on one A6000 + host + NVMe (chart 7's fits columns, then the `--plan` placement line), end card with the command.

Beats 2, 3 and 5 take terminal clips from the S1 and S2 sittings (a card's log is its clip's source); beats 1, 4 and 7 are charts only;
beats 6 and 8 mix a clip with a chart. The assembler (the prior promo's film runner and Mac-side assembler) is not on this
machine, so the beats name sources, not its inputs.
