# Router dials: steering a routed MoE toward the experts on the card

An experiment report. Two opt-in levers change which experts a routed MoE decode step runs, to trade a little
closeness to the model's own routing for decode speed on a load that keeps only part of the experts on the card. Both
are off by default and the engine's output is unchanged when they are unset. A third and fourth lever (an expert mask
and a route census) are tools for the same line of work.

| Lever | What it does |
|---|---|
| `BLOOMERY_ROUTE_RESIDENT_BONUS=λ` | adds λ to the router's selection value of every expert that is on the stage card at that moment; the gate weights still come from the unbiased scores |
| `BLOOMERY_ROUTE_TOPK=k'` | the decode step weighs only the router's first k' picks; the tail gets weight 0 and is not computed. The prompt pass keeps the model's k |
| `BLOOMERY_ROUTE_MASK=<file>` | bans the listed experts per layer (`<layer> <id> <id> ...`), so the router never picks them |
| `BLOOMERY_ROUTE_CENSUS=<file>` | writes how often each expert was picked and the weight it carried (`# route-census v1`) |

`crates/levers/src/registry.rs` holds the exact rule of each; `tools/route-mask.py` turns a census into a mask.

## Why it can work

On a card that holds a subset of the experts, a decode step's wall is mostly the picks that miss the card: each one is
served from host memory. The router's top-k is decided by small score gaps, so a small bias toward experts that are
already resident replaces a near-tie pick on the host by a near-tie pick on the card. The bias changes the choice only;
the chosen experts are weighed by the model's own scores.

## Measurements

All on one RTX 5090 (32 GB), DeepSeek-V4.1-Flash bq3km, residency on at its default.

### Decode speed and perplexity

`just depth-gpu-ds41 prose:512 ...`: one lease, three rounds, arm order rotated, the first 512 ids of the prose corpus,
96 generated tokens at depth 512. Perplexity: `ppl_ds41` over the prose corpus, 512 x 8 chunks, 2,040 scored
positions. The ratios are derived from the two columns beside them.

| Arm | decode tok/s, mean [range] | vs unset | pp tok/s (P = 512) | perplexity | vs unset |
|---|---|---|---|---|---|
| unset | 51.07 [50.56..51.65] | | 316.86 | 3.8742 | |
| bonus 0.1 | 69.86 [69.60..70.27] | +36.8 % | 316.70 | 3.8684 | −0.15 % |
| bonus 0.2 | 80.54 [78.33..82.25] | +57.7 % | 316.57 | 3.9306 | +1.5 % |
| bonus 0.3 | 89.31 [88.80..90.04] | +74.9 % | 316.78 | 4.1499 | +7.1 % |
| top-k 5 (of 6) | 60.59 [60.44..60.71] | +18.6 % | 316.43 | 3.9344 | +1.6 % |
| top-k 4 | 73.44 [73.25..73.66] | +43.8 % | 315.90 | 4.0607 | +4.8 % |
| bonus 0.2 and top-k 5 | 83.80 [81.81..85.07] | +64.1 % | 316.67 | 3.9955 | +3.1 % |

A first sitting (same runner, same conditions, unset 50.88 tok/s) read the large bonuses:

| Arm | decode tok/s (mean of 3) | perplexity | residency flips per run |
|---|---|---|---|
| bonus 1 | 109.08 | 8.4390 | 35–37 |
| bonus 2 | 115.13 | 17.5021 | 4–6 |
| bonus 4 | 117.08 | 27.3675 | 0 |

Perplexity only, no timed row: bonus 0.05 3.8589; bonus 0.15 3.9032; bonus 0.5 4.7577; top-k 3 4.5264; bonus 0.1
and top-k 5 3.9580.

Reading:

- A bonus of 0.1 leaves the perplexity where it was and decodes 37 % faster. The prompt pass does not move: the
  bonus and the cut act on the decode step only.
- At an equal perplexity cost (about +1.5 %), the bonus buys +58 % and the cut +19 %. Combining them costs more
  perplexity than the bonus alone at the same speed.
- A bonus of 1 or more overrides the router: the model is no longer usable.

### Task scores

`toktape eval` against `bloomery-serve` built from the same tree, one server per arm, temperature 0, thinking off:
KMMLU (first 300), MMLU-Pro (first 300), HumanEval and HumanEval+ (164). `p` is McNemar's exact two-sided test
against the unset arm on the same items. The decode column is the server's own request lines summed over the whole
eval, a look and not a timed number.

| Arm | KMMLU | MMLU-Pro | HumanEval | HumanEval+ | decode ids/s (look) |
|---|---|---|---|---|---|
| unset | 75.3 % | 73.3 % | 97.0 % | 92.7 % | 32.0 |
| bonus 0.1 | 75.3 % (p 1.0) | 73.7 % (p 1.0) | 97.0 % | 92.7 % (p 1.0) | 36.5 |
| bonus 0.2 | 75.0 % (p 1.0) | 73.3 % (p 1.0) | 95.1 % | 90.9 % (p 0.25) | 42.2 |
| top-k 5 | 75.3 % (p 1.0) | 74.0 % (p 0.69) | 96.3 % | 92.1 % | |
| bonus 0.3 | 74.7 % | 74.3 % | 95.1 % | 90.9 % | |

No arm differs from the unset one at these sample sizes.

ERNIE 5.5 Flash Preview F8M4 on the same card took the same levers, and its eval was stopped early: unset KMMLU
69.0 %, HumanEval 98.8 %, HumanEval+ 93.9 %, 43.4 decode ids/s (look); bonus 0.1 KMMLU 69.3 %, its other suites not
run. ERNIE has no timed row and no perplexity yet.

## What is not known

- The serving path gained less than the runner measured (+14 % against +37 % at bonus 0.1). The eval's requests are
  short and the server runs at its serving defaults; which of the two explains the gap is not measured.
- Perplexity is from one English prose corpus. Code and Korean text are unread.
- The prediction card expected 20–45 % for the whole dial range; bonuses of 1 and more gave over 2x. The term that
  was wrong is not identified (the step is more bound by host picks than the 85 % assumed, or the residency flips
  themselves cost).
- The cut is a one-column decode rule: a batched or speculative pass weighs all k slots.
- The mask and the census have run in no timed or scored experiment yet.
- ERNIE 5.5 has one score at one bonus and no timed row.

## Related: pass count and loop contrast on Inkling 2 Mini

The same night read Inkling 2 Mini's two load-time levers with HumanEval (164 items, `toktape eval`, the shipped
0.3.17 server, untimed).

| Arm | HumanEval | HumanEval+ |
|---|---|---|
| default (3 passes) | 86.6 % | 83.5 % |
| `BLOOMERY_INK2_PASSES=2` | 4.3 % | 4.3 % |
| `BLOOMERY_INK2_PASSES=4` | 2.4 % | 2.4 % |
| `BLOOMERY_INK2_PASSES=5` | 10.4 % | 10.4 % |
| `BLOOMERY_INK2_LOOPCD=0.5` | 87.2 % | 85.4 % (p 0.61) |
| `BLOOMERY_INK2_LOOPCD=1.0` | 85.4 % | 83.5 % (p 1.0) |
| `BLOOMERY_INK2_LOOPCD=1.0`, `_REF=2` | 82.9 % | 79.9 % (p 0.15) |

The Mini is a fixed-loop checkpoint: at another pass count its replies stop closing (at 2 the channel switch leaks into
the content, at 4 the thinking block never ends). The lever's text already says its output at another count is not
guaranteed. Loop contrast does not move HumanEval at these sample sizes.
