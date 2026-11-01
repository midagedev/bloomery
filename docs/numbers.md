# Numbers

The README's [Status](../README.md#status-0310) shows one table for each of three common machines. This page holds every
other record. Every number here is a release check or a sitting of the runners in `tools/ref/`, with its machine and conditions.
The reference engines' rows are on
[rig-log's bench page](https://github.com/midagedev/rig-log/blob/main/docs/bloomery-bench.md) with their flags,
measured in the same sitting as ours (the Mac's in [its own log](https://github.com/midagedev/rig-log/blob/main/log/2026-10-25.md#mac-m4pro-xeng)). Conditions: [`performance.md`](performance.md#conditions-of-the-numbers).

## RTX A6000 48 GB + RTX 3090 24 GB, 0.3.10

Large mixture-of-experts models on one workstation: an RTX A6000 48 GB and an RTX 3090 24 GB (at its 250 W cap),
32-core AVX2 host, 256 GB of RAM. Serving defaults: `--place` unset (it picks `bp`: the A6000 holds the model's stage,
the 3090 holds experts, the host runs the rest), every lever unset, no `--parallel`, `--ctx-size 4300`. Means of two
rounds, tok/s.

| Model | Prompt, 512 tokens | Prompt, 4,096 tokens | Decode at depth 512 | Decode at depth 4,096 |
|---|---|---|---|---|
| MiniMax-M3.1-Flash `UD-Q2_K_XL` (GGUF) | 160.16 | 165.92 | 28.63 | 29.41 |
| MiniMax-M3.1-Flash, NVFP4 folder | 129.08 | 131.56 | 21.12 | 19.53 |

## One request at the serving defaults, RTX A6000 (+ RTX 3090 for `bp`)

Every lever unset, no `--parallel`, `--ctx-size 16384`, greedy, thinking
off, one load per row, the release candidate's `bloomery-serve`; the release each row
was checked at is in its first column. Prompt sizes are the server's prompt tokens.

| Model, place (release) | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3.8 `UD-Q4_K_XL`, `a` (0.3.0) | 5,673, 1,693, 3.38 s | 75.1 tok/s | 86.3 tok/s |
| Qwen3.8 `UD-Q4_K_XL`, `bp` (0.3.1) | 5,673, 1,707, 3.35 s | 84.3 tok/s | 94.0 tok/s |
| Qwen3.6-35B-A3B `UD-Q4_K_XL`, unset → `a` (0.3.1) | 5,673, 7,083, 0.81 s | 115.0 tok/s | 117.5 tok/s |
| GLM-5.3-Flash `UD-IQ4_XS`, unset → `bp` (0.3.1) | 5,548, 172, 32.3 s | 35.1 tok/s | 35.0 tok/s |
| DeepSeek-V4.1-Flash `Q3_K_M`, unset → `a` (0.3.1) | 5,642, 403, 14.0 s | 41.9 tok/s | 42.7 tok/s |
| MiMo-V2.6-Flash `MXFP4`, unset → `a` (0.3.1) | 5,620, 104, 53.9 s | 34.8 tok/s | 36.1 tok/s |
| Nemotron 3.5 Bolt 74B-A9B `24GB`, unset → `a` (0.3.4) | 5,680, 2,716, 2.10 s | 87.5 tok/s | 88.7 tok/s |

## One RTX 3090, two cards, and two requests at once (0.3.0)

Two requests at once (`--parallel 2`, both busy) after a 512-token prompt, decode tok/s of both together, and one
request's decode at depth 4096. Measured on 0.3.0 (2026-10-12), on a 32-core AVX2 host with 8 DDR4 channels and 256 GB
of RAM; the 3090 at a 250 W cap; the MTP draft off.

**One RTX 3090 24 GB**

| Model | Where it runs | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M` | the whole model on the card | 252.5 | 164.1 | 6,496 / 6,801 |
| Qwen3.6-35B-A3B `Q4_K_M` | the whole model on the card | 257.3 | 181.8 | 5,766 / 6,685 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | card + CPU | 64.3 ¹ | 51.5 ¹ | 502 / 664 ¹ |
| GLM-5.3-Flash `UD-Q4_K_XL` | card + CPU | 25.3 ² | 19.5 | 176 / 205 |
| DeepSeek-V4.1-Flash `Q3_K_M` | card + CPU | 33.5 | 26.9 | 195 / 376 |
| Nemotron 3.5 Bolt 74B-A9B `24GB` (0.3.4) | the whole model on the card | — ⁴ | 86.9 | 1,773 / 2,047 |

**RTX A6000 48 GB + RTX 3090 24 GB (`--place bp`)**

| Model | Two requests, tok/s in total | Decode at depth 4096 | Prompt tok/s (P = 512 / 4096) |
|---|---|---|---|
| DeepSeek-V4.1-Flash `Q3_K_M` | 56.1 | 39.0 | 233 / 398 |
| GLM-5.3-Flash `UD-Q4_K_XL` | 34.0 | 22.7 | 218 / 262 |
| Qwen3.8-Flash-Next `UD-Q4_K_XL` | — ³ | 90.5 | — / 1,777 |

¹ with `BLOOMERY_RESIDENCY=mid-p32-s1` · ² 29.1 with `BLOOMERY_RESIDENCY=mid-p0-s1` · ³ under `--place bp` the Qwen3.8
seat runs two requests in turn, a slot at a time · ⁴ the nemotron seat serves one slot. Bolt's rows are 0.3.4's
sittings (n=4, prompt at e92fa8bde, decode at 44495f0ba); on one RTX A6000 the same sittings read 87.4 tok/s at depth
4096 and 2,143 / 2,772 prompt tok/s.


## One RTX 3060 12 GB (0.3.0, WSL2)

One request at the serving defaults (`--ctx-size 16384`, every lever unset), 0.3.0's release tarball, an i7-12700F
desktop with 32 GB of RAM under WSL2, greedy, thinking off.

| Model | Long prompt: tokens, prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|
| Qwen3-30B-A3B `Q4_K_M`, card + CPU | 5,619, 103 ¹, 54.4 s | 38.8 tok/s | 40.4 tok/s |
| Qwen3.8-Flash-Next `UD-Q4_K_XL`, experts paged from NVMe ² | 5,673, 84.9, 66.8 s | 4.5 tok/s | 5.4 tok/s |

¹ 0.3.0's measurement: the prompt ran one row at a time on this load. 0.3.1 runs it in wide units; not re-measured
here. ² 0.3.1's release candidate with no flags at all: the plan chose one slot of 28,881 positions and pages 75 GB of
experts from NVMe through a 19 GB RAM arena; a paged plan runs with the MTP draft off. 0.3.0 refused this load at the
defaults.


## DGX Spark, Qwen3.8-Flash-Next on the FP4-Local and NVFP4 folders (0.3.10)

Measured by [@juliankang4](https://github.com/juliankang4) on one DGX Spark at 0.3.10: `nvidia/Qwen3.8-Flash-Next-FP4-Local`
and NVIDIA's NVFP4 folder as downloaded, the MTP draft from Unsloth's `mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`, 128
tokens, greedy, `cache_prompt` off, median of three rounds. Decode is 1000 / TPOT; prefill is prompt tokens / TTFT.
Two and four requests at once on the FP4-Local folder, and the slots and context each folder's plan holds, are in
[`contrib/dgx-spark.md`](contrib/dgx-spark.md#23-qwen38-on-one-spark-slots-mtp-and-memory), section 2.3.

| Prompt | Text | Decode, draft off | Decode, MTP on | First token, off / on | Prefill, draft off |
|---|---|---|---|---|---|
| 4K, FP4-Local | code | 30.8 | 54.2 | 1,984 / 2,053 ms | 2,065 |
| 4K, FP4-Local | English prose | 29.8 | 39.9 | 2,078 / 2,078 ms | 1,971 |
| 4K, NVFP4 | code | 22.7 | 47.9 | 2,112 / 2,215 ms | 1,939 |
| 4K, NVFP4 | English prose | 22.7 | 33.4 | 2,068 / 2,192 ms | 1,981 |

## Contributors' cards: RTX 5090, DGX Spark (0.3.0, by @juliankang4)

One request at the serving defaults, measured by [@juliankang4](https://github.com/juliankang4) on their own hardware
at 0.3.0, greedy, thinking off, 256 tokens. Each row is their measurement on that card; the conditions, logs and tapes
are in [#46](https://github.com/midagedev/bloomery/issues/46). Their 0.3.9 DGX Spark work (prompts up to about twice
as fast) is in the [release notes](https://github.com/midagedev/bloomery/releases/tag/v0.3.9#from-our-contributors).

| Card | Model | ~4K prompt: prefill tok/s, first token | Decode after it | Short prompt decode |
|---|---|---|---|---|
| RTX 5090 32 GB | Qwen3.6-35B-A3B `Q4_K_M`, whole on the card | 14,937, 276 ms | 345.2 tok/s | 356.1 tok/s |
| RTX 5090 32 GB | Qwen3.8-Flash-Next `UD-Q4_K_XL` ¹ | 447, 9.12 s | 54.5 tok/s | 47.7 tok/s |
| DGX Spark (GB10) | Qwen3.6-35B-A3B `Q4_K_M`, whole | 4,576, 899 ms | 89.8 tok/s | 93.5 tok/s |
| DGX Spark (GB10) | Qwen3.8-Flash-Next `UD-Q4_K_XL`, whole, MTP draft on | 1,038, 4.11 s | 41.6 tok/s | 41.5 tok/s |

¹ with `BLOOMERY_HOST_ROOM=56G` on 64 GB of RAM: 0.3.0 refuses this load at the defaults.

## RTX 5090, Qwen3.6-35B-A3B with its MTP draft (by @juliankang4)

One RTX 5090 32 GB at a power limit of 400 W (the card's default is 600 W), measured by
[@juliankang4](https://github.com/juliankang4) on their own machine at `3ebdff767`: unsloth's
`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, which carries the model's next-token layer, with the MTP draft on
(`BLOOMERY_DRAFT=mtp`), `--parallel 1`, `--ctx-size 16896`, 4,096-id prompts (8 code and 8 English prose windows), 128
greedy tokens, one request at a time. Each value is the median of three rounds, each round the median of the 16
requests, the rounds' range in brackets. End-to-end is (prompt ids + generated ids) over the request's whole time.

| Metric | `3ebdff767` |
|---|---|
| Prefill, tok/s | 8,747.2 [8,743.6 to 8,754.5] |
| Decode, tok/s | 268.2 [268.1 to 268.3] |
| Time to first token, ms | 470 [470 to 471] |
| End-to-end, tok/s | 4,456.9 [4,442.6 to 4,457.1] |

Decode by corpus: code 282.8 tok/s, English prose 254.3 tok/s.

## Apple Silicon: M4 Pro, M3 Ultra, M5 Ultra (Qwen3.6-35B-A3B)

On an M4 Pro 48 GB (development loop, three rounds, tok/s): decode 101.2 and a 4,096-id prompt at 1,076.5 on the MLX
4-bit weights; 85.6 and 1,044.5 on the Q4_K_M file. mlx-lm and llama.cpp ran on the same files in the same sitting:
[their rows](https://github.com/midagedev/rig-log/blob/main/log/2026-10-25.md#mac-m4pro-xeng). On
[@avlp12](https://github.com/avlp12)'s machines at 0.3.9's Metal code, a 4,096-id prompt runs at 3,300 tok/s on an M3
Ultra and 5,972 on an M5 Ultra, whose neural accelerators run the prompt's GEMMs, and decode after 512 ids at 141–142
and 211–212 tok/s ([#593](https://github.com/midagedev/bloomery/issues/593)). The conditions of each row:
[`docs/mac.md`](mac.md#numbers).


