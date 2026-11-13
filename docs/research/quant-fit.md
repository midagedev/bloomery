# quant-fit: `bloomery-quantize --fit <bytes>` picks a per-tensor-class mix from measured sensitivity

A design note with a working cut (2026-11-14, track qfit). Every number is measured, cited, or marked `[derived]` with its
terms. Paths are this repo's.

## 0. Result in one paragraph

A mix is a rules file. Today a person writes it (`tools/quant-rules/exaone-moe-{bootstrap,pick,iq1gu}.rules`) and the engine's
own loss tool (`ppl_exaone` + `kld_diff`) only scores it afterwards. `--fit` closes the loop: `bloomery-quantize --fit-ladder`
lists the model's units and what each may become, `tools/ref/qfit.py measure` scores every unit at every lower type through the
engine's own loss path, and `bloomery-quantize --fit BYTES --sensitivity TABLE` picks the mix of least KLD for the budget,
writing the rules file after the plan under it has agreed with the pick to the byte. The selection is pure
(`crates/quantize/src/fit.rs`, `crates/quantize/tests/fit.rs`); the measurement is the one part that costs card and host time.

## 1. The unit

A unit is a tensor class over a layer range. A class is the set of tensors that share a stem and a role; the exaone-moe
classes (`fit.rs` `CLASSES`) are `routed_gate_up` (the `ffn_gate_exps` and `ffn_up_exps` stacks together: one fused entry
in the engine, one rule token covers both), `routed_down`, `attn`, `shared`, `dense_ffn`, `nextn`, `head`, `embed`. A layer
range is `--fit-group N` consecutive routed layers (default 0: the class over all its layers). `UnitKey { class, layers }` is
the key; its rule token is `^blk\.(a|...|b)\.(stem|stem)$=type`, which the reader's ECMAScript subset accepts. Rules are
first-match, so the unit tokens come before the fixed rules in every file the tool writes.

**What may be a unit** is decided by the reader, not by this tool: the candidate types of a unit are `FIT_TYPES` (Q4_K, Q5_K,
Q6_K, Q8_0, IQ2_XXS, IQ1_S: what the quantizer writes) intersected with the reader's list for each stem
(`gguf::arch::allowed_types` for exaone-moe, `gguf::arch::kimi_k3::allowed_types`) and with the types whose block divides the
rows. A class with one legal type is no unit: on exaone-moe the attention, shared, dense, nextn, head and embedding classes
are Q8_0 only, so they are bytes in the file that no choice moves (`fixed`).

The fixture's ladder (`--fit-ladder --fit-group 1`, box, measured) is 8 units: the four routed layers' gate/up
(`Q4_K`, `IQ2_XXS`, `IQ1_S`: 905,969,664 / 415,236,096 / 314,572,800 B a layer) and down (`Q4_K` 452,984,832, `IQ1_S`
157,286,400 B; the reader allows no IQ2_XXS down), base plan 8,496,145,440 B, equal to the Q4_K fixture file. So there are
12 variants and the base.

## 2. How a unit is measured

The sensitivity of a unit at a type is **the mean KLD, against the base mix, of the base with that one unit retyped**, over a
fixed id file, through the engine's own forward pass:

1. base = every unit at its largest legal type (`--fixed-rules` holds what no unit holds);
2. one variant per (unit, smaller type): the base's rules with that unit's token first, quantized by `bloomery-quantize`;
3. `ppl_exaone --ids F --run CxN={kld}` walks the variant (one id per decode step, writes ik's `.kld`: about 2 n_vocab + 8 B a
   position), `kld_diff base.kld variant.kld` prints the mean KLD with its standard error;
4. the table row is `class layers type dkld se positions`; the base's own types are anchor rows at 0.

The additivity of single-unit KLD is the model's first-order assumption: for small perturbations KLD is a quadratic form in the
logit error and the cross term of two roughly uncorrelated errors is small, so the sum of the units' rows predicts the
mix's KLD. `qfit.py verify` measures the picked mix and prints it against the sum; the ratio is the model's error, and it is
the one number a table needs to be trusted with.

**Bins per family.** The measurement needs (a) a reader type list for each stem (`allowed_types`), (b) the quantizer to know
the family (`resolve::ARCHS`), (c) a ppl bin that writes the `.kld` format. `kld_diff` and `kld.rs` are family-free.

| family | quantizer `ARCHS` | reader list | ppl/kld bin | `--fit` today |
|---|---|---|---|---|
| exaone-moe | yes | `gguf::arch::exaone_moe` | `ppl_exaone` | yes (this cut) |
| kimi-k3 | yes | `gguf::arch::kimi_k3` | none (`generate_kimi_k3`, `gate_kimi_k3_imatrix` only) | table by hand only; a ppl bin is the gap, about 60 lines over the shared walk if `ppl_exaone`'s CLI is lifted `[derived]` |
| deepseek41, glm5next, qwen3moe | no | per reader | none | the quantizer does not write them |

## 3. The selection

Each unit's candidates form points (bytes, dkld). Dominated points (more bytes and no less loss than another) are dropped, the
rest are cut to the lower convex hull. The hull's edges are the steps: step cost = dkld increase, step saving = bytes saved,
slope = cost / saving. All units start at their largest candidate; steps are taken in ascending slope (ties by unit, then
edge, so the order is fixed) until `fixed + Σ unit bytes <= budget`. This is the greedy of the multiple-choice knapsack on
convex hulls: optimal among the hull's choices up to the last step, and monotone by construction, since a larger budget stops
earlier in the same list. A budget below the smallest reachable file is a named refusal with both numbers
(`--fit B cannot be met: the smallest file this table can pick is ...`); an unmeasured or mistyped table is refused by line
(`Table::parse`), a table row for a type the reader does not allow, a unit with no row, overlapping units and a type fixed
by `--output-tensor-type` or `--token-embedding-type` likewise.

**Bytes are exact, not estimated.** The file is `overhead + Σ pad32(tensor bytes)` where the overhead (headers, metadata) does
not depend on any tensor's type (a tested identity). So the pick's bytes are the plan's bytes; `--fit` re-plans under the rules
it generated and refuses to write them when the two differ, and the rules file is created with `create_new` (an existing
file is never overwritten). The predicted loss printed is the additive sum, labelled as such.

## 4. The output

    bloomery-quantize --hf DIR OUT Q4_K_M --fit 6.5GB --sensitivity table.tsv --fit-out fit.rules [the run's own flags]

writes `fit.rules` (no comments; unit tokens, then the rules of what no unit holds, as `--tensor-type-file` reads them) and
prints each unit's pick, the file's bytes and the predicted sum. The same command with `--tensor-type-file fit.rules --plan`
prints the plan at those bytes; `just quantize` takes the file as it takes a hand-written one. `--fit` and `--fit-ladder`
open no card and take no output directory room.

## 5. Cost, priced on paper

Terms: quantize wall = host weights / (threads x rate), the quantizer's own derived-wall line (rate 8.0e6 weights/s/thread
under the card's; 2878 s on 32 threads for the 750B IQ1_S mix, from its `--plan` print on the real source `[derived]`);
walk wall = positions x step time (decode step 100 ms for K-EXAONE `[derived from the cards' step lines]`).

| | fixture `exaoneh` (6 layers, 8.5 GB Q4_K) | K-EXAONE-2.0 750B, whole class a unit | same, 8 layers a unit |
|---|---|---|---|
| variants | 13 | 4 | 25 (3 a group of 8, 8 groups, and the base) |
| quantize a variant | ~~45 s `[derived: 11.6e9 weights / (32 x 8e6)]`~~ 70 to 93 s `[measured: the quantizer's own "done in" lines, 70 s and 93 s, at 1.9e8 and 1.2e8 weights/s over their quantizing]` | 2878 s `[derived: the plan's line at 8e6/thread; the fixture ran 1.6 to 2.1 times slower than that formula, so 4600 to 6000 s if the ratio holds]` | the same, whole file |
| walk a variant (2032 scored positions of 256 x 16) | 25 s `[measured, all 15 walks]` | 410 s + load 150 s `[derived]` | same |
| total | pipelined (next quantize under this walk): **1135 s `[measured]`** for the 13; sequential 13 x (85 + 25 + 3) = 24 min `[derived from those terms]` | 4 variants: 4 x (3.4 to 6 ks) = 3.8 to 6.7 h | 25 variants: 24 to 42 h sequential |
| disk at the peak | 2 variants x 8.5 GB + 0.62 GB a kld `[measured: base.kld 624,263,060 B]` | 285 GB a variant: the box's 334 GB free holds one | same |

Three consequences. (1) On the 750B the whole-file variant quantize (48 min derived, more if the fixture's ratio holds) is over the 30-minute box-job cap by itself, so a
large-model table is a named sitting, never a round's run. (2) The box cannot hold the base file and a variant of a
280 GB model together, so the large-model path is the **delta variant**: `--keep-split` writes one output shard a source shard,
a unit of g layers changes only the shards that hold those layers, and the other shards are hardlinks of the base's. A group of 8 of 64
routed layers then quantizes 2878 s x 8/64 = 360 s (to 750 s at the fixture's ratio) and adds about 36 GB, around the walk's 560 s, so
25 variants run in about 4 to 5 h `[derived]`. This path is not built: it needs the quantizer to write into a directory that already holds the base's shards.
(3) The layer axis can be had far cheaper from the fixture: it is a cut of real layers 2, 5, 62 and 65 of the 750B (kept
`[0, 1, 2, 5, 62, 65]` in `tools/fixture-cuts/exaoneh.json`), so its per-layer rows are the early/late spread of the real
model's, with the caveat that the widths are cut (expert_ff 512) and the importance matrix is from one short text.

## 6. Resource timeline, per variant on the fixture

| resource | busy | what |
|---|---|---|
| host CPU | 70 to 90 s at 32 threads `[measured]`, then 1 thread for the walk, ~3 s `kld_diff` `[derived]` | quantize is the host's; the walk's one host thread shares a core with it |
| card SMs | 25 s walk `[measured]`, a few percent busy `[derived]` (6 layers, 1 id a step: latency, not bandwidth) | no lever: the card is not the bound |
| PCIe | model upload 8.5 GB once a variant, about 1 s at 12 GB/s `[derived]` | under the walk's load |
| host DRAM | the quantizer's working set; the page cache holds the variant just written | the walk reads it back from cache |
| NVMe | write 8.5 GB (6 s at 1.5 GB/s `[derived]`), read the cut source (cached after the first) and write a 0.62 GB kld `[measured]` | well under the host's 70 to 90 s |

The wall is **one resource's max**, not the sum: the quantize on the host (a variant's pipelined period was 85 s `[measured: 1135 s
for 13, the base's 142 s included]` against 25 s of walk). With `--ahead 1` the walk of variant v runs under the quantize of v+1, so
the card's 25 s is in the host's shadow (worth 0 to shorten) and the lever is the flow, already taken.
The only term left to shorten is the host's quantize, which is rules over all weights though a variant changes one unit: the
delta variant of section 5 is that lever (bulk: one unit's tensors instead of the file).

## 7. Predictions, written before the box runs

The fixture is a coverage cut (a few real layers, cut widths), so absolute KLD values are not predicted; relations are.

| id | claim | band | if outside, next |
|---|---|---|---|
| P1 | each gate/up unit: dkld(IQ1_S) > dkld(IQ2_XXS) > 0 | all 4 units | a unit that breaks it means noise (se) or an imatrix hole: read its se before anything |
| P2 | each layer: dkld(down IQ1_S) > dkld(gate/up IQ2_XXS) | all 4 layers | picks at the hand budget then differ from the hand mix: read the table for which unit |
| P3 | the table's sum for the hand IQ2 mix (4 gate/up at IQ2_XXS) predicts its measured KLD | measured/predicted in 0.8 to 1.3 | above 1.3: the additive model is the weak term, add a refine pass (swap one unit of the pick and re-measure); below 0.8: the table is conservative, fine |
| P4 | the fit's pick at the hand mix's bytes is no worse than the hand mix | measured pick <= 1.05 x measured hand | above: the table's noise or the additivity, same next step as P3 |
| P5 | measure wall | 8 to 15 min for 13 variants, host-bound | a wall over 17 min means the host's IQ rates are under the plan's 8e6/thread: read the quantizer's `weights/s` line |
| P6 | the fit's pick under the iq1gu hand mix's budget (6,130,558,208 B `[derived]`: 8,496,145,664 B base with the imatrix's 224 B, less 2,365,587,456 B) | predicted sum <= the iq1gu's table sum | if the iq1gu is the pick, the hand mix was right by this measure |

Written after the table (`exaoneh-sensitivity.tsv`, section 8) and before the verify runs, from the table's hull steps by hand
`[derived]`. The steps in slope order: L5 gate/up IQ2 (1.0e-11 a byte), L5 down IQ1 (1.5e-11), L5 gate/up IQ1 (4.0e-11), L4 gate/up
IQ2 (1.1e-10), L4 down IQ1 (1.7e-10), L3 gate/up IQ2 (2.5e-10), L2 gate/up IQ2 (3.4e-10), L3 down IQ1 (5.4e-10).

| id | claim | value |
|---|---|---|
| P7 | the pick at the IQ2 hand mix's bytes (6,533,211,392 B) stops after the sixth step: L5 gate/up IQ1_S and down IQ1_S, L4 gate/up IQ2_XXS and down IQ1_S, L3 gate/up IQ2_XXS, L2 and L3 down Q4_K | file 6,331,884,800 B (201,326,592 B under: one step's granularity), predicted sum 0.242566 against the hand mix's 0.347698 |
| P8 | its measured KLD against the base | 0.7 x the hand mix's measured KLD within 0.15 (measured pick / measured hand in 0.55 to 0.85) |
| P9 | the pick at the iq1gu mix's bytes adds L2 gate/up IQ2_XXS | file 5,841,151,232 B, predicted sum 0.407221 against the iq1gu mix's 0.697604 |

## 8. Measured

The fixture `exaoneh` (`/models/fixtures/exaoneh`: layers 2, 5, 62, 65 of the 750B as blk.2..5, plus two dense layers and the MTP
layer; Q4_K file 8,496,145,440 B) on the box, 2026-11-14. Base = every routed unit Q4_K, trunk/head/embedding Q8_0
(`exaone-moe-bootstrap.rules`), importance matrix `generate_exaone --imatrix` over AGENTS.md (14 chunks of 512, 52 entries, 53 MB,
6 s `[measured]`; `exaoneh.imatrix.gguf` adds 224 B to every plan, so the base is 8,496,145,664 B). Loss: `ppl_exaone` over the
first 256 x 16 ids of `corpus-prose.ids`, 2032 scored positions, A6000 gate lock, no lease. Walk 25 s each, 15 walks; `measure` 1135 s.

The sensitivity table (`--fit-group 1`; dkld = mean KLD of the base with that one unit retyped, against the base walk):

| unit | Q4_K | IQ2_XXS | IQ1_S | bytes Q4_K / IQ2_XXS / IQ1_S |
|---|---|---|---|---|
| gate/up blk.2 (real layer 2) | 0 | 0.164655 ± 0.004783 | 0.287246 ± 0.008533 | 905,969,664 / 415,236,096 / 314,572,800 |
| gate/up blk.3 (real 5) | 0 | 0.124033 ± 0.003241 | 0.285116 ± 0.008784 | same |
| gate/up blk.4 (real 62) | 0 | 0.054074 ± 0.001028 | 0.116266 ± 0.002519 | same |
| gate/up blk.5 (real 65) | 0 | 0.004936 ± 0.000079 | 0.008976 ± 0.000148 | same |
| down blk.2 | 0 | | 0.257522 ± 0.007735 | 452,984,832 / 157,286,400 (IQ1_S) |
| down blk.3 | 0 | | 0.161064 ± 0.004341 | same |
| down blk.4 | 0 | | 0.051181 ± 0.000992 | same |
| down blk.5 | 0 | | 0.004302 ± 0.000083 | same |

Sensitivity falls 33 times (gate/up IQ2_XXS) to 60 times (down IQ1_S) from the first routed layer to the last. The fixture is a coverage cut, not a language
model (the base's own PPL over this text is 4.36e6 in `kld_diff`'s line), so these numbers say what the pipeline does, not how
the 750B behaves; section 5's point (3) is how to use the layer axis as a proxy.

Verification (`qfit.py verify`: the hand mix and the fit's pick each quantized, walked and compared with the base):

| budget | mix | file B | measured KLD | table's sum | measured / sum |
|---|---|---|---|---|---|
| 6,533,211,392 (the IQ2 hand mix, `exaone-moe-pick.rules`) | hand: gate/up IQ2_XXS x4, downs Q4_K | 6,533,211,392 | 0.289668 ± 0.008077 | 0.347698 | 0.833 |
| | **fit's pick**: blk.5 gate/up IQ1_S and down IQ1_S, blk.4 gate/up IQ2_XXS and down IQ1_S, blk.3 gate/up IQ2_XXS, blk.2 and blk.3 down and blk.2 gate/up Q4_K | 6,331,884,800 (3.1 % smaller) | 0.253356 ± 0.006745 | 0.242566 | 1.044 |
| 6,130,558,208 (the IQ1_S gate/up hand mix, `exaone-moe-iq1gu.rules`) | hand: gate/up IQ1_S x4, downs Q4_K | 6,130,558,208 | 0.687838 ± 0.019241 (top-1 39.2 %) | 0.697604 | 0.986 |
| | **fit's pick**: the above plus blk.2 gate/up IQ2_XXS | 5,841,151,232 (4.7 % smaller) | 0.354826 ± 0.010571 | 0.407221 | 0.871 |

At the iq1gu budget the pick has half the hand mix's KLD (0.355 against 0.688, 15 sigma apart on the unpaired SEs) in fewer bytes.
At the IQ2 budget it is 12.5 % lower (0.253 against 0.290), inside the unpaired SEs: no paired `kld_diff` of the two was run, so
that gap is not established. The additive sum predicted the four measured KLDs within 0.83 to 1.04 of their value.

Predictions against the outcome (section 7):

| id | outcome |
|---|---|
| P1 | held: IQ1_S > IQ2_XXS > 0 on all 4 gate/up units |
| P2 | **missed** on 2 of 4 layers: down IQ1_S < gate/up IQ2_XXS on blk.4 (0.0512 < 0.0541) and blk.5 (0.0043 < 0.0049); held on blk.2 and blk.3. The picks took the cheap downs early, as the table said |
| P3 | held: hand IQ2 mix 0.833 and the iq1gu mix 0.986 of the sum (band 0.8 to 1.3) |
| P4 | held: pick 0.875 of the hand mix (band <= 1.05) |
| P5 | **missed**: measure took 1135 s = 18.9 min against 8 to 15; the quantize is 70 to 93 s a variant, not ~45 s (section 5) |
| P6 | held: predicted 0.407 <= the iq1gu's 0.698, measured 0.355 against 0.688 |
| P7 | held exactly: the pick, its 6,331,884,800 B and its sum 0.242566 |
| P8 | **missed narrowly**: pick / hand 0.875 against the band 0.55 to 0.85 (the hand mix's sum overshot its measurement by 17 %) |
| P9 | held exactly: the pick (blk.2 gate/up IQ2_XXS added), 5,841,151,232 B, sum 0.407221 |

Derived numbers that were wrong, struck where they stood: the imatrix walk (1 to 2 min, measured 6 s), the quantize wall (45 s,
measured 70 to 93 s), the kld file (1.26 GB for 4096 positions: ik's chunks score the second half, 2032 positions, 0.62 GB), the
base's size (8,496,145,440 B: the imatrix adds 224 B).

**One trap met.** The ssh control master dropped (`mux_client_request_session: read from master failed`, rc 255) during the second
verify; the remote run went on with its output to a closed pipe and died at its next print after the walk, leaving the walked KLD
file. The hand mix's KLD above comes from that file through `kld_diff`, the pick's from a rerun. A long run should write its
log on the box (`> LOG 2>&1 < /dev/null`) and be polled.

## 9. Limits

- The table is first-order: it does not see interactions between units; `verify` prices that error per run.
- One id file, one context: KLD against the base on 4096 prose positions. A mix chosen on prose is not tested on code.
- The pick stops at the first step that fits, so it can end up to one step under the budget (201 MB and 289 MB under in the two
  fixture picks; one 8-layer gate/up step on the 750B is about 29 GB). Size the group so the largest step is under about 2 % of the
  file. A refill pass after the walk would use the slack but picks differently per budget; `tests/fit.rs` is red on that mutant
  (the monotone gate), so it is a design change, not a tweak.
- The base is the largest legal type. A base that cannot be materialised (the 750B's Q4_K file is about 440 GB, over the box's free
  disk) means the table covers the steps below the best existing mix.
- The resumable driver (`qfit.py measure --resume`) keeps one checkpoint line per finished variant; the delta variant of section
  5 is the missing piece for models above the box's disk.
