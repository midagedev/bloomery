# Qwen3.8 expert routing locality on recent prose and code

CPU eval-callback traces of 8,192 prose tokens and 8,192 code tokens support the locality premise in issue #31 for these inputs. Mean LRU-64 hit rates are 67.7% and 73.3%; mean LRU-128 hit rates are 83.2% and 86.9%. These are routed-selection replay rates, not measured GPU speedups. The per-layer rates vary substantially, and a fixed static list transfers poorly between these two genres.

## Model and collection

Measured on 2026-10-12 with Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS, architecture `qwen4exp`, 48 routed layers, 512 experts per layer, and top-10 routing. The machine is an Intel i5-12600K with 64 GiB RAM running Linux 7.0.0-38-generic. The Radeon RX 7900 GRE is not used for routing collection: `-ngl 0`, KV offload disabled, and operation offload disabled.

The collector adapts `tools/ref/router_trace.cpp` to mainline llama.cpp commit `fb4b2737a808a3fb7c2117a498f43815dc9be53e`. The linked local build has Vulkan-specific changes; the CPU backend files are unchanged. The [API adapter patch](router-locality-mainline.patch) reproduces the compiled collector source. It requests output logits for every input token so the graph retains every token's final routed layers.

The eval callback reads `ffn_moe_topk-<layer>` as an I32 view of ARGSORT, following both `nb` strides before storing little-endian U16 expert ids. Each call requires exactly one selection for every routed layer. Counts must match the recorded top-k rows, expert ids must be valid and distinct within a token, and incomplete manifests are rejected.

Each corpus is tokenized with the model's own vocabulary, adding special tokens and treating special-token-looking source text as plain text. No source repetition is needed: the prose has 10,069 token ids and the code has 9,855. Only the first 8,192 ids are collected. Contexts reset every 2,048 tokens; batch and microbatch are 256, with 10 CPU threads. The callback uses the top-k-only schedule. These are teacher-forced input traces, not generated-answer traffic.

## Replay rules and results

LRU-64 and LRU-128 mean that each layer gets its own cache of 64 or 128 expert ids. Caches start empty, process the router's rank order, refresh hits, insert misses, and persist across the four context resets. Static top-32 learns from tokens [0,4096) and is evaluated on the independent contexts [4096,8192). All percentages count routed selections, and the all-layer row is the mean because all layers have the same selection count.

| corpus | LRU 64 % | LRU 128 % | static top 32 held % | static top 32 in sample % |
|---|---:|---:|---:|---:|
| prose | 67.7 | 83.2 | 35.8 | 39.2 |
| code | 73.3 | 86.9 | 44.1 | 46.8 |

Both mean LRU-64 rates exceed the issue's 60% criterion. Layer 0 is much less local than layer 15, so the mean does not justify giving every layer the same physical byte budget. A model cache must also account for expert byte sizes and the engine's shared pools, copies, and admission rules.

A static top-32 list learned from all prose tokens serves 17.2% of code selections. A list learned from all code tokens serves 14.0% of prose selections. These cross-genre evaluations use different learning and evaluation corpora; they are separate from the within-genre held-out rows above. Resetting LRU at each context changes the aggregate results by less than 0.3 percentage points.

## Schedule sensitivity and repeatability

Both schedules process the same first 512 input tokens, with independent 2048-token contexts and the same CPU collector. Expert sets differ under floating-point schedule changes; agreement and locality are separate quantities.

| corpus | shared ids % | batch LRU-64 % | scalar LRU-64 % | batch LRU-128 % | scalar LRU-128 % | maximum layer drift pp |
|---|---:|---:|---:|---:|---:|---:|
| prose | 95.76 | 67.16 | 67.14 | 82.89 | 82.85 | 0.59 |
| code | 94.23 | 70.44 | 70.52 | 86.56 | 86.55 | 0.80 |

The 512-token comparison bounds only this prefix. The corresponding maximum per-layer change in LRU hit rate is 0.80 percentage points; it is not a proof of bitwise routing equivalence across schedules or devices. A second full 8,192-token collection for each corpus, with its input MD5 recorded at launch, reproduced every top-k file and counts file byte for byte. The earlier collection and scalar checks are preserved separately.

## Text sources and provenance

The prose is recent technical discussion on public Bloomery issues and pull requests. Fenced code, table rows, and details/summary tag lines are removed. The code is source from the contributor's two pull requests created on 2026-10-12. Dated text reduces the chance of training overlap, but cannot establish that the model never saw it.

- prose: https://github.com/midagedev/bloomery/issues/48
- prose: https://github.com/midagedev/bloomery/issues/54
- prose: https://github.com/midagedev/bloomery/issues/13
- prose: https://github.com/midagedev/bloomery/issues/17
- code: https://github.com/Burntgogi/bloomery/blob/1f5e9ca75258ef812994bb2bc1d36dfb00ceb46c/fuzz/src/lib.rs
- code: https://github.com/Burntgogi/bloomery/blob/1f5e9ca75258ef812994bb2bc1d36dfb00ceb46c/fuzz/fuzz_targets/jinja_parse.rs
- code: https://github.com/Burntgogi/bloomery/blob/19285c84720cf434607e5f94f308aa316c0304ff/crates/sampler/examples/sample_bench/threshold.rs
- code: https://github.com/Burntgogi/bloomery/blob/19285c84720cf434607e5f94f308aa316c0304ff/crates/sampler/examples/sample_bench.rs
- prose: https://github.com/midagedev/bloomery/issues/22
- prose: https://github.com/midagedev/bloomery/pull/36
- prose: https://github.com/midagedev/bloomery/pull/49
- prose: https://github.com/midagedev/bloomery/pull/57

- `prose.txt` SHA-256: `9b79107e149cef59768a72fdef24a1dea54d4dcb94238f5e569572e96a8e979a`; `prose.ids` MD5: `95ee78d18f59219f3c99682fa079db00`.
- `code.txt` SHA-256: `8e6ab9b0549a41badb75234ed3eaaa03dc3e0680d9a78a1cf8ad6168eaf0ad51`; `code.ids` MD5: `33a59998ab0f940fa2acc1a74583303f`.

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf` SHA-256: `219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15`.
- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf` SHA-256: `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.

Raw traces, exact text/id snapshots, compiler commands, and all file hashes are preserved in the contributor's local evidence directory. Tables below are the per-layer observations, not synthetic fixtures.

## Reproduction

Build shared mainline llama.cpp libraries at the pinned commit, copy `tools/ref/router_trace.cpp`, and apply the API adapter patch to the copy. Link it with the common, llama, and ggml libraries. Tokenize the archived text through `llama_tokenize` with `add_special=true` and `parse_special=false`, one decimal id per line. Build paths and the compiler are local environment choices; the model, ids, schedule, and flags below define the collection.

```sh
export BLOOMERY_DATA="$EVIDENCE/data" BLOOMERY_ROUTER_WRITE=1
export BLOOMERY_LOCK_DIR="$EVIDENCE/locks" BLOOMERY_REF_BUILD=mainline-fb4b2737-adapter
for corpus in prose code; do
  export BLOOMERY_ROUTER_IDS_MD5="$(md5sum "$CORPUS_DIR/$corpus.ids" | cut -d' ' -f1)"
  bash tools/ref/lease-hold.sh --card docs/cards/contributor-router-locality.card -- timeout 1800     "$COLLECTOR" -m "$MODEL" --expect-arch qwen4exp --ids "$CORPUS_DIR/$corpus.ids"     --out "$BLOOMERY_DATA/router/$corpus" --chunk 2048 --max-tokens 8192 --top-k-only     -c 2048 -b 256 -ub 256 -ngl 0 -t 10 -tb 10
  python3 tools/ref/router-coverage.py locality "$BLOOMERY_DATA/router/$corpus"
done
```

The collection card uses the profile kind and a 0..100% band: this is a locality profile, not a wall-time comparison. For the scalar sensitivity check, keep the same ids and chunk size, use `-b 1 -ub 1 --max-tokens 512`, then compare only the first 512 tokens.

## Per layer results

### Prose

| layer | LRU 64 % | LRU 128 % | static top 32 held % |
|---:|---:|---:|---:|
| 0 | 27.8 | 49.5 | 16.3 |
| 1 | 48.8 | 67.5 | 25.6 |
| 2 | 49.8 | 68.9 | 24.7 |
| 3 | 78.8 | 91.1 | 57.1 |
| 4 | 66.6 | 84.6 | 38.6 |
| 5 | 61.5 | 79.0 | 34.7 |
| 6 | 55.7 | 73.7 | 25.5 |
| 7 | 56.9 | 74.3 | 30.4 |
| 8 | 71.3 | 85.2 | 33.1 |
| 9 | 76.6 | 88.6 | 41.3 |
| 10 | 66.5 | 81.6 | 29.2 |
| 11 | 73.7 | 87.1 | 33.4 |
| 12 | 57.0 | 76.8 | 29.6 |
| 13 | 66.2 | 81.6 | 29.8 |
| 14 | 70.7 | 85.6 | 35.6 |
| 15 | 91.9 | 96.9 | 61.2 |
| 16 | 86.0 | 94.6 | 46.9 |
| 17 | 70.7 | 85.7 | 32.8 |
| 18 | 55.7 | 74.5 | 30.0 |
| 19 | 72.5 | 87.5 | 37.3 |
| 20 | 69.1 | 85.2 | 33.7 |
| 21 | 61.5 | 78.7 | 29.7 |
| 22 | 48.2 | 68.9 | 24.6 |
| 23 | 74.7 | 87.6 | 42.3 |
| 24 | 69.9 | 85.9 | 32.3 |
| 25 | 74.4 | 88.5 | 38.5 |
| 26 | 65.3 | 82.2 | 28.3 |
| 27 | 72.4 | 87.4 | 33.0 |
| 28 | 57.8 | 77.5 | 32.7 |
| 29 | 65.9 | 81.5 | 30.9 |
| 30 | 68.8 | 84.7 | 32.7 |
| 31 | 88.5 | 95.5 | 54.7 |
| 32 | 82.2 | 92.6 | 38.8 |
| 33 | 69.4 | 84.6 | 32.2 |
| 34 | 57.2 | 75.8 | 28.3 |
| 35 | 70.9 | 86.1 | 34.1 |
| 36 | 71.7 | 87.0 | 34.4 |
| 37 | 63.7 | 81.7 | 31.2 |
| 38 | 50.1 | 70.4 | 25.9 |
| 39 | 81.9 | 92.4 | 50.6 |
| 40 | 81.1 | 91.2 | 50.6 |
| 41 | 74.5 | 88.4 | 43.2 |
| 42 | 64.1 | 83.6 | 34.1 |
| 43 | 78.1 | 90.9 | 44.4 |
| 44 | 73.2 | 88.4 | 42.6 |
| 45 | 72.2 | 87.6 | 43.1 |
| 46 | 69.9 | 86.9 | 39.0 |
| 47 | 68.1 | 87.3 | 39.4 |
| all | 67.7 | 83.2 | 35.8 |

### Code

| layer | LRU 64 % | LRU 128 % | static top 32 held % |
|---:|---:|---:|---:|
| 0 | 31.8 | 58.2 | 25.7 |
| 1 | 53.0 | 75.5 | 33.2 |
| 2 | 55.3 | 75.4 | 35.1 |
| 3 | 73.9 | 88.0 | 51.0 |
| 4 | 68.4 | 86.2 | 41.7 |
| 5 | 65.0 | 83.3 | 38.3 |
| 6 | 64.2 | 80.9 | 37.7 |
| 7 | 63.6 | 81.5 | 38.1 |
| 8 | 77.9 | 89.0 | 42.2 |
| 9 | 80.5 | 90.9 | 40.9 |
| 10 | 72.5 | 85.7 | 36.6 |
| 11 | 79.5 | 91.0 | 38.9 |
| 12 | 68.3 | 85.1 | 37.5 |
| 13 | 71.8 | 85.2 | 34.8 |
| 14 | 78.1 | 90.3 | 42.4 |
| 15 | 92.4 | 97.4 | 61.8 |
| 16 | 88.4 | 96.2 | 54.5 |
| 17 | 78.7 | 90.6 | 44.3 |
| 18 | 70.9 | 86.3 | 42.6 |
| 19 | 78.3 | 90.3 | 46.5 |
| 20 | 77.4 | 89.5 | 46.1 |
| 21 | 73.7 | 85.7 | 44.6 |
| 22 | 63.1 | 80.8 | 43.1 |
| 23 | 73.2 | 86.3 | 43.8 |
| 24 | 77.3 | 89.3 | 47.4 |
| 25 | 79.5 | 91.2 | 42.2 |
| 26 | 72.0 | 86.1 | 39.5 |
| 27 | 78.2 | 91.0 | 42.2 |
| 28 | 71.3 | 87.1 | 42.2 |
| 29 | 74.4 | 87.9 | 39.2 |
| 30 | 76.6 | 90.0 | 43.1 |
| 31 | 89.3 | 96.2 | 60.5 |
| 32 | 85.5 | 94.7 | 50.0 |
| 33 | 77.6 | 89.6 | 44.8 |
| 34 | 73.2 | 86.7 | 44.2 |
| 35 | 78.0 | 89.2 | 46.4 |
| 36 | 79.6 | 90.0 | 50.0 |
| 37 | 77.6 | 88.3 | 48.1 |
| 38 | 68.7 | 83.9 | 44.4 |
| 39 | 79.2 | 90.8 | 46.8 |
| 40 | 75.7 | 87.6 | 46.1 |
| 41 | 76.6 | 87.8 | 54.0 |
| 42 | 74.0 | 86.5 | 51.9 |
| 43 | 75.4 | 86.4 | 48.9 |
| 44 | 74.7 | 87.0 | 52.2 |
| 45 | 70.0 | 84.7 | 41.2 |
| 46 | 70.3 | 85.9 | 44.1 |
| 47 | 64.8 | 82.1 | 44.0 |
| all | 73.3 | 86.9 | 44.1 |
