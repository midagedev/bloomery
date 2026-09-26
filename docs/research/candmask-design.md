# V4.1 candidate mask — design memo

The design memo of round `candmask` (rebuild wave 2, research only), kept as it reported, under the
lead's dispositions below. It designs the two-level candidate mask that V4.1 applies once layer 20's
visible rows pass 2,048 blocks of 8 — the reason every V4.1 entry refuses positions past 16,384 by
name since `ef80083`.

## Lead's dispositions (2026-09-27)

- **Semantics checked against the reference by the lead** (`model.py:498-506`, `:560-612`;
  `config.json` `index_source_layer_ids`, `candidate_source_layer_id`): layer 20 ranks the blocks by
  its own indexer scores before its own top-k, which the mask does not touch; the consumers are the
  indexers of layers 24, 28, 32 and 36; the block holding the query's newest row is pinned; and the
  engine's refusal bound (2,048 × 8 × layer 20's ratio 1 = 16,384) is where the mask first acts.
- **Built inside the rebuild's waves, not beside them.** The ModelSpec design already carries the mask
  (`Selector.candidates: Option<Candidates>`, `modelspec-design.md`). R1's kernels go to wave 3
  `opslib` as a family with no model name in it, with R1's bit-exact unit gate (G1) as specified here;
  R2's wiring goes to wave 4 `layerprog` (the candidate source as one more CED source, the refusal
  lifted by a delete-class change). Until wave 4 lands, positions past 16,384 stay a named refusal.
- **R3 builds the mask in a derived ik tree.** Every V4.1 oracle set is pinned to `# build db517b69`
  by a constant in each gate that reads it (`IK_BUILD`, `ORACLE_BUILD`, `SET_BUILD`, `BUILD`). The mask
  in that tree would move every family's build id and re-take every set. A derived tree (db517b69 plus
  the mask, as `/home/user/ik-dspark-draft` is db517b69 plus the draft script) gives the candidate sets
  (d1c, d3) their own family and build id — the per-family build column of `crates/refset` — and
  sitting 10's regeneration is not affected.
- **What is the round's, not the lead's:** the Python reference's sm_89 FP8 requirement and checkpoint
  size, skel's three blockers, exllamav3's and mistral.rs's missing V4.1 paths, and every cost and
  oracle-time figure, which the memo marks [derived]. Nothing in the memo was measured.
- **Beyond the brief** is disposed in `docs/plan-triage.md`. The d3 / CPU end-to-end truth run
  (6–48 min [derived]) is the user's call when R4 comes due.

## Round memo


I wrote no files in any repository, changed no git state, and ran no builds, binaries, lease or GPU work.

The spec's read path had one side effect you should know about. I made 8 box reads through `BLOOMERY_BOX_READONLY=1 tools/box.sh` as prescribed. Each call rsyncs the Mac tree to `~/repo/bloomery` on the box (`tools/box.sh:63`). So the main tree's two uncommitted edits (`docs/rebuild.md` and `tools/verdict-diff.py`, not mine) went to the box copy, as any box.sh call does. No user input arrived during the round.

**Verdict: build now.** The semantics are fully fixed by the reference. The first two gate rungs are bit-exact rule checks that need no outside oracle. The cost is at most ±0.1 % of a step at any depth, and it lifts the 16,384 refusal to the serving plan's 32,768. Hold public long-context numbers until an end-to-end check against ik with the mask exists (round R4 below).

### 0. Work done before answering (the two required habits)

#### 0a. The question split into terms

| # | Term | Settled by |
|---|---|---|
| T1 | Selection semantics | Reading only: `model.py:569-610` |
| T2 | The bound position | Integer arithmetic (Q1.7) |
| T3 | New work per step | Byte and launch counts from the code |
| T4 | Change in what the consumers scan | Bytes, using the per-position formula in `docs/v41-placement.md:113-118` (1,664 B per cached position per step) |
| T5 | Bandwidth | 567–701 GB/s (`docs/plan.md:73`) |
| T6 | Cost per graph node, `c_node` | 0.852 µs (`docs/plan.md:71`) |
| T7 | Time of the single-block passes at 16k–32k rows | **The one term arithmetic cannot fix.** Predicted: score pass 6–7 µs at 16,384 rows and 12–14 µs at 32,768; top-k pass 4–8 / 6–12 µs; new block top-k 2–5 µs [derived] |
| T8 | Oracle cost | Measured ik CPU rates (rig-log 09-23:143, :425) and recorded gate wall times (`~/.cache/bloomery/gate-times.tsv`) |

**No run is proposed for T7.** The mask's total effect is inside the ruler at every depth. Resolving 0.1 % would take about 280 rounds per arm [derived: 1.97·0.6·√(2/N) = 0.1]. So no decision depends on T7, and `lease_take` would refuse the card anyway.

#### 0b. Resource timeline

**Decode, per layer (A6000, plan (a)).**
- The card runs the attention sub-layer, including the indexer, while the host tier waits for that layer's router output.
- Then host experts run beside card experts.
- The step is host-bound: 12.9–13.3 ms of card kernels per replay, of which 4.4–4.8 ms overlap the host; 16.5–18.5 ms exposed waiting for the host (rig-log 09-24:195).
- The mask touches only the card-serial attention segment. Its terms therefore add to the step; none sits under another resource's shadow. PCIe, host CPU, host DRAM and NVMe: 0.

**Prompt batch (group of 2 or more).**
- Layer 20's new launches join the card route.
- On lcg prompts the host union binds (68.6 of 73.8 ms per layer-batch, `docs/plan.md:111`), so the new card term is hidden and worth 0.
- On prose the card binds (49.5 ms modelled), so it adds.

**Flow levers checked.**
- **Asynchrony:** running the block top-k on the side branch HC_PRE already uses would save 2–5 µs but cost a fork and a join (about 1.7 µs). Net about 0; not worth it.
- **Bulk:** consumers skipping the key tiles of excluded blocks is the design's own saving.
- **SIMT:** two larger levers are in "Beyond the brief" below; both are under the ruler at 32k.

### 1. Semantics, exactly

1. **Which scores rank the blocks.** Layer 20's own indexer scores.
   - Formula: `Σ_h relu(q_h·k_t)·w_h`, with `w = weights_proj(x)·d^-½·H^-½` (`model.py:550-557`).
   - In a prompt, rows the query cannot reach are set to −inf first (`:563-565`). In decode there is no fill: the width is `end_pos//ratio` (`:554`, `:567`).
   - Taken **before** layer 20's own top-k (`:569-572` precede `:577-580`). Layer 20's own top-k is **not** masked (the `elif`, `:573`).
   - One mask per query row per forward call (`shared_attn.candidates`, `:570`, `:1166-1176`). Nothing persists across steps.

2. **The block key.** The maximum over the block's 8 positions; the last block is padded with −inf (`model.py:597-600`). In a prompt, a partly reachable block takes the max of its visible rows; a fully unreachable block is −inf, meaning "not reachable yet" (`:591-592`).

3. **Ties.** The reference uses `scores.topk(min(2048, num_blocks))` (`:607`), and torch does not document which of equal values wins (listed under Open).
   - The reference's scores are probably bf16. This is **inferred** from the dtype flow: the fp8 GEMM returns the default dtype (`kernel.py:299`), `weights_proj` is bf16 (`model.py:515`), the index-key cache takes the default dtype (`:521-525`), and model.py's own `__main__` sets bf16 (`:1296`). `generate.py`, which would set it for real runs (`README.md:53`), is not in this tree.
   - If so, equal block maxima are common there. Ours are f32, so they are rare.
   - Proposed rule: the lower block id wins. That matches our row top-k (`indexer.rs:31-34`) and exllamav3's DSA top-k (`dsa_topk.cu:284`).

4. **The pin.** `last = (compress_lens−1)//8` is set to +inf (`model.py:604-605`). That is the block holding the query's newest visible row, pinned whether it is partial or full.
   - The kept set is {pinned block} ∪ the 2,047 best other blocks.
   - `top.values > -inf` then drops picks that were unreachable (`:608-609`); this only matters for prompt rows that reach fewer than 2,048 blocks.

5. **Which layers consume the mask, and how.**
   - Consumers are the indexers above layer 20: **24, 28, 32, 36** (`uses_candidates`, `model.py:503`; indexers exist only on index-source layers, `:655-661`, `config.json:112-121`).
   - Each masks its **own** scores (its own q and weights, layer 20's keys) to −inf outside the candidates (`:573-575`), then takes its usual top-k (`:578-580`).
   - Layers 21–23, 25–27, 29–31, 33–35 and 37–39 reuse their source's list (`:725-726`).
   - Masking and restricting the top-k to candidate rows give the same result whenever it is active: at least 2,047·8 + 1 = 16,377 finite candidates ≥ 512 [derived]. So we are free to build it as a candidate-only scan.
   - Layers 2–19 (ratio 2) and the draft layers (ratio 0, `config.json:101-103`) are untouched.

6. **A prompt batch whose queries see different lengths.** Everything is per row: visible count per query (`:564`), −inf per row, pin per row (`:604`).
   - A row that reaches 2,048 blocks or fewer keeps all its reachable blocks; a row that reaches more gets a real selection. A batch crossing 16,384 therefore mixes unmasked and masked rows.
   - Each row equals the decode computation at its position [derived].
   - The reference itself prefills only from position 0 (`:563`, `:466`). A multi-token call at an offset would take its decode branch (`:566-567`). So for our offset batches the defining semantics are the per-position decode ones, which the prefill gate already enforces.

7. **The exact position where the mask stops being a no-op.**
   - At query position p (0-based), layer 20 (ratio 1, `config.json:81`) sees D = p+1 rows (`model.py:567`), in ⌈D/8⌉ blocks (`:598-600`).
   - All blocks are kept while ⌈D/8⌉ ≤ 2,048, i.e. p ≤ 16,383.
   - The mask first acts at **p = 16,384**: 2,049 blocks, block 2,048 holds one row and is pinned, so exactly one full block is dropped (the lowest-max of blocks 0–2,047).
   - Our bound is 2,048 × 8 × 1 = 16,384 (`hparams.rs:40-44`, `:525-545`). `check_defined` refuses `end > 16,384`, with end = pos+1 for a step (`body.rs:838-852`, `:1953`), pos+2 for a pair (`:1150`), the call's end for a prompt (`prefill.rs:999`), and the history length (`body.rs:866`). **It matches exactly.**
   - Whether a consumer's list actually changes at a given p is data-dependent.
   - The GGUF carries no candidate keys. I listed shard 1's header keys on the box (47 `deepseek41.*` keys, none named `candidate*`), so `hparams.rs:40-44` is the only source.

### 2. Where it sits in our step

#### Decode (captured graph)
**Current launch path:**
- `AttnChain::enqueue_layer_of` (`chain/attn.rs:1251`)
- → `Selection::Run` (`:1360`) → `enqueue_indexer` (`:1759`) → `enqueue_select` (`:1817`)
- → `IndexerKernels::enqueue` (`indexer.rs:1049`), which launches `ds41_indexer_score` (`:270`) and `ds41_indexer_topk` (`:597`).
- Per-layer wiring is resolved at load: `ListOf` / `LayerStep` (`body.rs:448`, `:463`), `layer_io` (`:1625`); lists per pair row (`:2168-2175`).

**New at layer 20 (the source).**
- **Placement is a correctness constraint:** the new launches must be enqueued *inside `enqueue_select`, right after the two existing passes*. The reason: `IndexerScratch.scores` is shared by every indexer layer of the stream and by both pair rows (`chain/attn.rs:631`, `:922`), so layer 24 overwrites it.
- (a) Block maxima of the scores: ⌈n_vis/8⌉ f32 values per token.
- (b) Block top-k:
  - block count ⌈n_vis/8⌉ from the device word the indexer already reads;
  - pin = (n_vis−1)>>3;
  - keep {pin} ∪ the best 2,047 others, ties to the lower id;
  - a NaN block raises the fault word instead of writing a plausible mask;
  - output is a keep bitmask.
- Below the bound both new launches exit early, and consumers skip the mask by the same device predicate. Nothing stale is read.

**Consumers (24/28/32/36).**
- The score pass reads the keep bitmask. A 16-key tile is two 8-key tensor-core tiles, i.e. exactly two blocks, so excluded blocks skip their key loads and matrix multiplies, get −inf written, and stay out of the histogram.
- `ds41_indexer_topk` is **reused unchanged**: the k-th key is finite, so −inf keys never reach it.
- The consumer's score buffer then equals the reference's masked scores (`model.py:575`), which gives a free tap for gates.

**Graph.** Nothing becomes data-dependent in a way the graph cannot hold:
- grids depend only on the token count and the SM count;
- counts are device words;
- work past the live count exits early;
- no host readback.

What moves is the node count (below) and the `--structure` node-count pins.

**Kernel entries: two options.**
- **A (recommended).** Leave `ds41_indexer_score` byte-identical and add three new entries: `ds41_cand_blockmax`, `ds41_cand_topk`, and a consumer variant compiled from the same core.
  - `just ptx-scan` then shows only new entries; every existing md5 stays the same. That is a structural proof.
  - Cost: +2 nodes per step = +1.7 µs [derived: 2 × 0.852].
  - Needed: one comparison in the index gate showing the consumer variant equals the plain entry bit for bit below the bound.
- **B.** Fuse the block maximum into the existing score pass (after the cross-lane reduction, lanes 0–7 and 8–15 hold the tile's two blocks) and add a flag for consumers.
  - Cost: +1 node (0.85 µs).
  - The hot entry's md5 changes, so the proof becomes the owning bit gates at every oracle set.
- I'd pick A: the extra node is about 0.003 % of a 29 ms step.

**Pair pass** (`body.rs:1092`, `:1138`): each row gets its own bitmask, as it already has its own lists.

#### Prompt batch
- Per chunk of up to 8 tokens: `chunk_rows` (`chain/attn/batch.rs:1058`) → `enqueue_select` (`:1206`); the same placement rule applies at layer 20.
- The bitmask must be kept per chunk and per batch set, like `BatchSet::lists` (`prefill.rs:394-397`, `:883-887`, served by `LayerCaches::chunk` `:697-716`). Layer 20's chunks all run before any consumer's, and a group holds several batches.

#### Does the CED triangle still hold?
Yes, with no new term.
- A consumer at position p reads the mask layer 20 wrote at the same p.
- Block starts are monotone in the layer (`full = floor(read_from)`, `read_from = part`, `ced.rs:286`, `:297`). So layer 20's block ran at every position where a consumer's block runs.
- This is the same argument `ced.rs:26-28` makes for lists.
- Add the candidate source to `CedLayer.sources` (`ced.rs:157-172`) so `exact()` (`:180`) checks it.

#### Two-card plan (b)
The cut is at layer 20 (`v41-placement.md:170-173`), so the source and all consumers sit on the 3090 side. The mask never crosses the link.

#### Buffers (ctx_max 32,768, `workstation.rs:64`)

| Buffer | Per query | Decode (2 rows) | Prompt batch |
|---|---|---|---|
| Block maxima (transient) | 4,096 × 4 B = 16 KB | 32 KB | 128 KB per chunk, reused |
| Keep bitmask (layer 20 → 36) | 512 B | 1 KB | 65 chunks × 8 × 512 B = 266 KB per set; group of 2 → 532 KB, group of 8 → 2.1 MB |

All [derived]; negligible against the A6000's margin of about 1.09 GB (`v41-placement.md:192`).

#### New versus reused
- **New:** the block-max reduction, the block top-k with pin (reusing `order_key`, `block_scan`, `pick_bin` and `fine_count` from `indexer.rs`), the consumers' bitmask read, the hyperparameter fields, and two constant words (block count and block size) alongside `top_k`.
- **Not new:** no gather (the attention's list gather is unchanged) and no persistent state (the mask is recomputed every step, so rollback and keep points need nothing).

### 3. Cost, derived

#### Decode, per step (A6000; D = p+1 rows in the ratio-1 stream)
Key bytes scanned per step:
- **No mask:** 1,664·D.
- **With the mask and a candidate-only consumer scan:** 640·D + 4 × 256 × 16,384 B.

| D | Blocks | Dropped | Bytes, no mask | Bytes, with mask | Saved | Saved time at 600–700 GB/s |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 2,048 | 0 | 27.26 MB | 27.26 MB | 0 | 0 |
| 24,576 | 3,072 | 1,024 | 40.89 MB | 32.51 MB | 8.39 MB | 12.0–14.0 µs |
| 32,768 | 4,096 | 2,048 | 54.53 MB | 37.75 MB | 16.78 MB | 24.0–28.0 µs |

**New work:**
- block maxima write 8 / 12 / 16 KB;
- the block top-k reads them about 3 times from L2 and writes 256–512 B;
- consumers read 4 × 512 B;
- total ≤ 70 KB, about 0 in bandwidth;
- +2 launches per step at every depth = +1.7 µs;
- when active, the block top-k takes 2–5 µs (T7).

**Net per step [derived]:**
- 16,384: +1.7 µs (+0.006 %);
- 24,576: −5 to −10 µs;
- 32,768: −17 to −24 µs.

Against a 25–31 ms step at depth 4096 (prose 24.89–25.26 ms, rig-log 09-26:291; lcg 30.81–30.90 ms, rig-log 09-24:168), that is at most 0.1 %.

**Reference-literal alternative** (score every row, then mask): no saving; +1.7 µs, plus 2–5 µs once past the bound.

**Does the step get cheaper past the bound?**
- Relative to an unmasked engine at the same depth: yes, by at most 0.1 %.
- Past 16,384 the per-position byte slope drops from 1,664 to 640 B (−62 %), because only layers 2, 8, 14 and 20 keep scanning every row.
- **No timed A/B is warranted.**

#### Prompt batch, P = 512 at positions 16,384–16,895
Blocks per row are 2,049–2,112, so 1–64 blocks are dropped.

- **New work:** 64 chunks × 2 launches = 0.2–0.4 ms of card time. Host issue is 128 × 2.42 µs (`docs/plan.md:109`) = 0.31 ms, hidden under the card route.
- **Consumer saving:** about 260 rows per token × 256 B × 4 layers × 512 tokens ≈ 136 MB ≈ 0.2 ms.
- **Net:** about ±0.3 ms against a batch wall of 2.37 s (prose pp512 216.1) to 3.54 s (lcg pp512 144.6), i.e. about 0.01 % [derived].
- **Shadow:** hidden on lcg; adds on prose.

For comparison, a batch at 32,256–32,767 would save about 8.6 GB ≈ 12–14 ms (0.4–0.6 % of the batch) [derived]. That is the only case where the saving rises above noise.

### 4. The oracle

#### Which implementation can produce truth past 16,384 for our file

| Implementation | Has the mask? | Runs our file? | Usable as truth past 16,384? |
|---|---|---|---|
| Python reference | Yes | **No, at any context.** Its FP8 tensor-core GEMMs (`kernel.py:266`, `:550`; FP4 is cast to FP8 first, `:496-497`) need sm_89 or newer, and our cards are sm_86. It also needs the HF FP8/FP4 checkpoint split per tensor-parallel rank (`README.md:17-31`, MP=8), which is not on the box and is about 500 GB (skel quotes 507.9 GB, ports report §8) against 72 GB of VRAM. `generate.py` and `convert.py` are absent from this tree. | Only its pure-torch pieces run anywhere (`:583-610`, `:575`, `:578-580`), usable as a rule simulator |
| skel | Yes: `build_deepseek4.cpp:1036-1094`; pin `:150-181` and `llama-dsv4.cpp:928-950`; the source layer takes the unfused path | Tensor and arch names match (`llama-arch.cpp:59`, `llama-model.cpp:1297-1338`); missing candidate keys default to 20/8/2048 (`llama-hparams.cpp:2294-2299`). **But three blockers:** (1) engram constants come from a sidecar whose generator is not in the tree; without it engram is disabled with a warning, which is a different model (`llama-engram.h:3-15`, `llama-engram.cpp:202-208`). (2) The engram gather hardcodes Q8_0 rows and checks only the row count (`llama-engram.h:35`, `.cpp:156-167`, `:246-252`, `:426-453`), while our `engram_embd` is Q3_K, so it reads garbage or out of bounds. (3) Its indexer rounds the query to fp4, and that is "not optional"; it changed 7.33 % of top-k slots (`build_deepseek4.cpp:1096-1102`). | Only after a sidecar writer (S) and a Q3_K patch (about 10 lines), and even then its lists differ from ours by more than the mask does. A semantics cross-check, not a numeric oracle |
| ik (our port; `/home/user/ik-idxkey` db517b69 on the box and `ik_llama.cpp` 41b17995 on the Mac) | No: `grep candidate` in `src/` finds nothing in either tree | Yes | No: past 16,384 it silently computes the unmasked model |
| mainline llama.cpp (V4.1 branch) | No; it caps the context at the threshold (`llama.cpp-fork/src/models/deepseek41.cpp:32-34`) | Yes | No |
| exllamav3 0740edc | No. V4 only: `_RATIO_TO_TYPE = {0, 4, 128}` (`deepseek_v4.py:19`, `:56`), so V4.1's ratios 2 and 1 do not load | No | No |
| mistral.rs d5ae0f1 | No V4 path at all (only `deepseek2.rs`, `deepseek3.rs`) | No | No |

**Where the designs differ:**
- exllamav3's top-k: fp16 scores ranked on a 16-bit key in two 256-bucket passes (`dsa_topk.cu:16-18`); for decode rows of 32,768 or more it splits the row into 32 spans and merges (`:279-289`); ties go to the lower index.
- Ours: f32 keys, three passes (10 + 11 + 11 bits), one block per token.
- skel: keeps −inf picks that the reference drops (`build_deepseek4.cpp:1067-1079` vs `model.py:608-609`). This is harmless, because the consumer's causal mask reapplies −inf.

**Proposed oracle:**
- Add the mask to our ik port (skel's graph code is the template).
- Read `deepseek41.attention.candidate_*` with defaults, so `--override-kv` can shrink the block count the same way the d1 oracle set shrinks top_k (`tools/ref/models/deepseek41.sh:218-219`).

#### Truth-run prices [derived]

**d1c** = the d1 set plus 16 candidate blocks (prefill 301, context 512, CPU).
- At position 301 there are 38 blocks; 16 are kept, giving 128 candidate rows for a top-64, so the mask bites at every consumer.
- Estimated time: **2–9 min.** Bases: the 5-token set's pre-launch estimate was 1.5–8 min (rig-log 09-23:96), and the d1 and d2 plain sets' manifests were written 5 min 13 s apart (box file times 09-24 15:44:02 → 15:49:15).

**d3** = the real 2,048-block setting, prefill 16,448 on the CPU.
- Two measured ik CPU rates, both on the older mixed file:
  - 66.5–75.4 tok/s (router trace, rig-log 09-23:143) → 6–13 min including load;
  - 6.8–7.2 tok/s (perplexity run, rig-log 09-23:425) → 40–48 min.
- **The pessimistic end crosses 30 min, so this needs your approval.**

**End-to-end, ik side:**
- On the GPU (A6000, ik pp512 128.5 tok/s, rig-log 09-25:415): **3–8 min**.
- On the CPU: 6–48 min, same as d3.

#### Gate ladder

| Rung | What it checks | Oracle | Bit-exact? | Box time |
|---|---|---|---|---|
| **G1 unit** (in `gate-gpu-ds41-index`, beside its existing 16,384 / 32,768-row synthetic cases `(v)`, `gate_deepseek41_index.rs:44-49`) | Synthetic scores at n_vis = 16,384 / 16,385 / 16,392 / 24,576 / 32,767 / 32,768; an 8-token launch straddling the bound; equal block maxima planted across the 2,048th boundary; a ±0 pair. Keep bitmask vs a host transcription of `select_candidate_blocks` with a stable sort (value descending, id ascending). Consumer list = exact top-512 of the masked scores. Consumer entry = plain entry bit for bit below the bound. NaN → fault word. Captured replay = eager. FAIL-first: turn the pin off, or send ties to the higher id | Host rule | **Yes.** Only max and order-key comparisons (exact) and integer steps decide it | Currently recorded at 3–4 s (gate-times.tsv, 09-26; I did not verify that this includes reading the oracle sets), plus under 1 s of new GPU work |
| **G2 rule at real depth** (new arm of `gate_deepseek41_long`, which already opens the gate placement at 32,768 context, `:144`, `:244`) | Prose ids to 16,448 via the prompt batch, then 16–64 eager steps. Taps: layer 20's scores, the bitmask, each consumer's scores and list. Host checks are exact. Counts the positions where the mask changed a list (FAIL-first witness; must be > 0). Finite probe on every position | Our own taps + host rule | Yes | 2–5 min (prompt rate on the gate placement not measured) |
| **G2b prefill crossing** (`gate-gpu-ds41-prefill`) | One call to 16,300; keep point; then 16,300–16,500 as a batch vs as 200 steps; state and logits bit-identical | Steps | Yes | +1.5–3 min over today's recorded 329–414 s |
| **G3 ids vs oracle** (`gate-gpu-ds41-step --select` on d1c) | Consumers' lists vs ik+mask within the tie band, at both block and row level | ik+mask CPU dump | Tie band | Our side ≈ the step gate's recorded 41–96 s; dump 2–9 min |
| G3b (optional) | Same on d3 | ik+mask CPU dump | Tie band | 6–48 min; approval |
| **G4 end-to-end** (`--deep` arm) | 16,448 prose ids + 64 greedy tokens vs ik+mask greedy; red on a first difference where our margin ≥ 1.5 (the existing rule) | ik+mask | First-difference rule | Ours 2–5 min; ik 3–8 min on GPU or 6–48 min on CPU |

### 5. Build order and size

| Round | Size | Scope | Change class and gates |
|---|---|---|---|
| **R1 `candmask`** | M | Hyperparameters: a `Candidates` struct built from the constants, source/consumer roles per layer, and `with_candidates(blocks)` for gates, refusing `blocks·size < top_k + size` by name. The three kernel entries (option A). G1. | New entries only. `ptx-scan` of `generate_ds41` and `gate_e2e` = base plus the new entries, every existing md5 identical; new rows in `ptx-shapes.tsv`. No engine wiring, no timed A/B. |
| **R2 `candwire`** | M | Wire the source and consumers through the decode step, the pair pass (per-row bitmasks) and the prompt batch (per-chunk, per-set bitmasks). Enqueue inside `enqueue_select`. Add the candidate source to `CedLayer.sources`. Lift the refusal. G2, G2b. | Below the bound, "launch count only": +2 nodes per step (+1.7 µs predicted) and +128 launches per 512-position batch; owning gates bit-identical; `--structure` node pins re-pinned with a PIN line. Past the bound: new semantics. |
| **R3 `candoracle`** | M | The mask in our ik port (also an upstream triage item); the d1c set variant; G3. | Box work is one CPU dump sitting of 2–9 min. |
| **R4 `canddeep`** | S | G4 after R3; d3 and G3b only with your approval. | — |

R1 and R2 could be merged into one L round if you'd rather have fewer rounds. Keeping them apart isolates the kernel's structural proof.

**What the refusal becomes (a delete-class change in R2):**
- Remove `check_defined` (`body.rs:838-852`), its callers (`:866`, `:1150`, `:1953`, `prefill.rs:999`) and the `defined` field (`body.rs:589-595`, `:2215`). The remaining bounds are `positions()` (`body.rs:867`, `:1943`, `prefill.rs:1114`) and the planner (`plan.rs:197-198`).
- Retire `candidate_free_positions` and its test (`hparams.rs:525-545`, `:1559`) as a coverage change with the dated reason "the mask is built".
- Replace them with one load-time invariant: layer 20 owns the ratio-1 compressor and index keys and runs the indexer, and every later indexer reads that stream; otherwise the load refuses by name.
- After R2, a V4.1 call is bounded only by ctx_max (32,768).

### Recommendation
Build R1 and R2 now as correctness rounds, with no timed A/B.
- **Specified:** the semantics are 28 lines of reference code (`model.py:583-610`) plus the consumer mask (`:569-575`).
- **Gated without outside help:** both first rungs are bit-exact rule gates that need no external oracle.
- **Cheap:** the cost is at most ±0.1 % of a step at every depth, inside the ruler.
- **Unblocks the serving plan:** it lifts the refusal to 32,768 and makes the file's 1M-token header context semantically correct.
- **Stronger public claim:** mainline caps at 16,384, and ik silently computes a different model past it.

Put R3 (ik with the mask, d1c via `--override-kv`) into the next CPU-dump sitting. Run d3 and the CPU end-to-end truth only with approval. Publish long-context numbers only after R4.

### Open
1. **torch.topk tie order at the 2,048th block.** Smallest step: a roughly 20-line torch script (CPU and CUDA) planting equal block maxima at the boundary. It only matters for bit-exact comparison with the reference, which bf16 vs f32 already rules out.
2. **The reference's score dtype is inferred as bf16.** Smallest step: read the upstream `generate.py` named in `README.md:53`.
3. **The T7 kernel times at 16k–32k rows.** Smallest step: nsys over the index gate's existing synthetic depth cases (about 2 minutes, needs a lease card). Not recommended, since no decision depends on it.
4. **ik's CPU prefill rate on the public file at 16k** (the two measured anchors differ tenfold, 6.8 vs 75 tok/s). Smallest step: the wall time of the d1c dump, plus one 4,096-token CPU prefill in the next CPU sitting.
5. **The 3090 gate placement's prompt rate at 16k.** Smallest step: the `time prompt` line from G2's first run.
6. **Whether the index gate's recorded 3–4 s includes reading the oracle sets.** Smallest step: one `just gate-gpu-ds41-index` log.
7. **Whether ik's CUDA backend supports the mask's operations** (block max pooling; argsort over up to 4,096 blocks) for a GPU end-to-end run. Skel's comment about running its candidate-source layer at 20,480 rows suggests yes (`build_deepseek4.cpp:1284-1291`), but that is **inferred**.
8. **The mainline cap's mechanism.** I read only its comment.

### Beyond the brief (reported only; nothing touched)
1. `indexer.rs:136-141`: `relu` maps NaN to +0, and the indexer kernels have no fault sink, so a NaN in the indexer's own query or weights silently becomes a plausible score and list. Add a fault site (S).
2. `indexer.rs:597`: the top-k pass runs one block per token, i.e. one of 84 SMs, at 16k–32k rows. exllamav3's split-and-merge (`dsa_topk.cu:279-289`) would save about 30–60 µs per step at 32k [derived]. That is under the ruler, so open it only if ctx_max moves toward 1M (M).
3. `indexer.rs:297` (`t = bid / blocks`): the prompt batch's score pass reads every index key once per token. At depth 32k a 512-position batch streams about 27.9 GB [derived]. Sharing key tiles across a chunk's 8 tokens would cut that about 8×; this is the batch-wide lever for long prompts (M).
4. `docs/v41-placement.md:120`: the 1M-depth figure of about 9 ms per token becomes about 3.5 ms with the mask [derived: (640·1,048,576 + 16.8 MB) / 195 GB/s] (XS, after landing).
5. `hparams.rs:40-44`: the candidate settings are global constants because the GGUF carries none (verified). A converter PR using skel's key names (`skel/src/llama-arch.cpp:244-246`) would let the file carry them (S, upstream).
6. ik past 16,384 is a silent-failure upstream: mainline caps there (`deepseek41.cpp:32-34`), ik does not. Even a cap is more than a one- or two-line fix, so under the CLAUDE.md rules this goes to triage, not an immediate PR.
7. `model.py:578-580` quirk: if candidates < top_k, consumers would attend −inf rows. The `with_candidates` override must refuse that by name (XS).
8. `docs/research/v41-ops-report.md:91-93`: "consumed by indexers of layers > 20" should say precisely layers 24/28/32/36; the others reuse lists (XS).
9. `tools/ref/models/deepseek41.sh:210-236`: add the d1c and d3 variants once ik reads the keys (XS, part of R3).
10. `tools/box.sh:63`: `BLOOMERY_BOX_READONLY=1` still rsyncs the tree, including other people's uncommitted edits. A read-only call could skip the sync (XS).
