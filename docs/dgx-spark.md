# Running bloomery on a DGX Spark

This page is for someone with an NVIDIA DGX Spark (or another GB10 machine) who wants to serve a model with bloomery.
It covers the install, the first start on Qwen3.8-Flash-Next, which checkpoint folder to pick, how many requests and
how much context one Spark holds, the MTP draft, and the measured speed. The general serving guide is
[`serving.md`](serving.md). Porting work, the machine's measured facts and the multi-Spark design are in
[`contrib/dgx-spark.md`](contrib/dgx-spark.md).

To start, install the release, download `nvidia/Qwen3.8-Flash-Next-FP4-Local` and run `bloomery-serve -m <folder>`.
The defaults serve one request at a time at the longest context that fits, with the model's MTP draft on.

## 1. Install

The release carries a GB10 archive (`linux-aarch64-cuda-sm121a`) whose GPU code is compiled for the GB10 ahead of
time, so the first start compiles nothing. The machine needs DGX OS 7 (Ubuntu 24.04, glibc 2.39) and an NVIDIA
driver of R580 or newer (CUDA 13).

```sh
# the installer: it checks that every GPU is a GB10 and links bloomery-serve into ~/.local/bin
curl -fsSL https://raw.githubusercontent.com/midagedev/bloomery/main/tools/release/install.sh | sh

# or Docker (needs the NVIDIA Container Toolkit); Docker pulls the linux/arm64 image by itself
docker run --gpus all -p 8080:8080 -v /path/to/models:/models \
  ghcr.io/midagedev/bloomery -m /models/Qwen3.8-Flash-Next-FP4-Local
```

The installer refuses an aarch64 machine whose GPU is not a GB10; such a machine builds from source. Building on a
Spark (the container, the `llc` and the `sm_121a` target) is in
[`contrib/dgx-spark.md`](contrib/dgx-spark.md#21-build), section 2.1.

## 2. Serve Qwen3.8-Flash-Next

A 128 GB Spark holds every routed expert of Qwen3.8-Flash-Next on the GB10. NVIDIA publishes the model in two
safetensors checkpoint folders, and bloomery serves either as downloaded, with no conversion step:

```sh
bloomery-serve -m /path/to/Qwen3.8-Flash-Next-FP4-Local
```

`--hf nvidia/Qwen3.8-Flash-Next-NVFP4` downloads and serves the NVFP4 folder in one step. Before it listens, the
server prints a `setup` line: the draft and where it reads it, the slots, the context each serves and the settings
that change them. The server is ready when `GET /health` answers 200; [`serving.md`](serving.md#2-send-a-request)
shows a first request.

## 3. Which folder

Start with FP4-Local.

| Folder | On disk | Weights |
|---|---|---|
| `nvidia/Qwen3.8-Flash-Next-FP4-Local` | 102.3 GB (revision `b9ce18a6`) | NVFP4 experts with compressed block scales, MXFP8 and FP8 dense weights |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 132.7 GB | NVFP4 experts |

FP4-Local takes less of the Spark's memory at the same settings (section 4) and was faster in every cell of the speed
table (section 6).
On the GB10's `sm_121a` build, both folders' routed expert prompt products run on the FP4 tensor cores (weights and
activations in FP4), and the MXFP8 sites on the FP8 tensor cores.

FP4-Local needs a writable scale store ([`serving.md`](serving.md#92-qwen38-checkpoint-folders)). The KV cache is
f16 for both folders unless `--cache-type-k q8_0` asks ([`serving.md`](serving.md#6-memory-beside-the-weights)).

## 4. Requests at once, context and memory

With no context flag each slot asks the trained 262,144 positions, and `--parallel auto` counts the slots from
memory; the rules are in [`serving.md`](serving.md#3-slots-and-context). On a 128 GB Spark every row of the tables
below kept the context it asked for.

The GB10 and the CPU share one memory pool, so the plan reads its room from the host's `MemAvailable`, not from the
GPU driver's free-memory figure. A Spark with more running beside the server fits less. Read your own numbers with
the same flags and `--plan`, which prints the plan and exits before it loads anything:

```sh
bloomery-serve -m /path/to/Qwen3.8-Flash-Next-FP4-Local --parallel auto --slot-ctx 65536 --plan
```

The tables are `--plan` on one 128 GB Spark at main `e7180ea9` (GB10, driver 595.84), with 115.3 to 116.5 GiB of
`MemAvailable` before each plan. With every routed expert on the GB10, one slot of the full 262,144 positions needs
104.0 GB on FP4-Local and 107.5 GB on NVFP4. Where more slots do not fit beside every expert, the plan keeps the
context and moves routed experts to the host tier instead: the `plan` line's `host_experts` counts them, out of the
model's 24,576. Need is the `need` record's bytes, what the load takes of the pool (the GB10's share, the slots'
checkpoints in pinned host memory and file pages). The prompt cache is the `need` record's `cache_ram`, its budget on
top of that (by default 8 GiB, or half of what the plan leaves where that is less); a row with 0 runs no prompt cache.

FP4-Local:

| Settings beside `-m <folder>` | Slots | Context a slot | Routed experts on the host | Need GB | Prompt cache GiB |
|---|---:|---:|---:|---:|---:|
| none | 1 | 262,144 | 0 | 104.0 | 8.0 |
| `--parallel 2` | 2 | 262,144 | 0 | 115.5 | 3.9 |
| `--parallel 4` | 4 | 262,144 | 5,878 | 123.5 | 0.0 |
| `--parallel auto --slot-ctx 131072` | 8 | 131,072 | 12,576 | 123.6 | 0.0 |
| `--parallel auto --slot-ctx 65536` | 8 | 65,536 | 1,934 | 123.6 | 0.0 |
| `--parallel auto --slot-ctx 32768` | 8 | 32,768 | 0 | 111.6 | 5.8 |
| `BLOOMERY_DRAFT=off` | 1 | 262,144 | 0 | 101.8 | 8.0 |
| `BLOOMERY_DRAFT=off --parallel 4` | 4 | 262,144 | 4,435 | 123.6 | 0.0 |

NVFP4:

| Settings beside `-m <folder>` | Slots | Context a slot | Routed experts on the host | Need GB | Prompt cache GiB |
|---|---:|---:|---:|---:|---:|
| none | 1 | 262,144 | 0 | 107.5 | 7.7 |
| `--parallel 2` | 2 | 262,144 | 0 | 119.0 | 2.4 |
| `--parallel 4` | 4 | 262,144 | 7,140 | 123.7 | 0.0 |
| `--parallel auto --slot-ctx 131072` | 8 | 131,072 | 13,774 | 123.7 | 0.0 |
| `--parallel auto --slot-ctx 65536` | 8 | 65,536 | 3,209 | 123.7 | 0.0 |
| `--parallel auto --slot-ctx 32768` | 8 | 32,768 | 0 | 115.1 | 4.3 |
| `BLOOMERY_DRAFT=off` | 1 | 262,144 | 0 | 104.1 | 8.0 |
| `BLOOMERY_DRAFT=off --parallel 4` | 4 | 262,144 | 5,182 | 123.7 | 0.0 |

The MTP draft stayed on in every row that did not set `BLOOMERY_DRAFT=off`, and no row printed a `draft yield` record.

A context the plan cut to fit prints `rule=fitted` with what it was short of; a context set by `--slot-ctx` or
`--ctx-size` prints `rule=set`. To fit a context or a slot count the default does not, a smaller `-ub` frees GPU
bytes and runs long prompts slower ([`serving.md`](serving.md#6-memory-beside-the-weights)).

## 5. The MTP draft

The draft is on by default. With `BLOOMERY_MTP_DRAFT` unset, it is the checkpoint's own MTP layer, so the folder is
the only download.

It goes off by itself, with a `load draft=off` or `draft yield` record that says why. The conditions are the yield
rule in [`serving.md`](serving.md#4-speculative-decoding). `BLOOMERY_DRAFT=mtp` keeps it on whatever the context, and
`BLOOMERY_DRAFT=off` serves without it.

## 6. Speed

The [README's Spark table](../README.md#status-0311) holds the values and the request conditions for both checkpoint
folders at main `e7180ea9` on a DGX Spark (GB10, driver 595.84, `sm_121a`). On a 4,096-id prompt with the MTP draft on,
FP4-Local prefilled 2,840 to 2,894 tok/s and decoded 39 to 70 tok/s, and NVFP4 was slower in every cell.

Decode depends on how many draft tokens the model keeps, so it differs by prompt more than prefill does. These are the
servers the README table came from, one per folder:

```sh
BLOOMERY_DRAFT=mtp BLOOMERY_HOST_ROOM=96G BLOOMERY_MTP_WIDTH=cost \
  bloomery-serve -m /path/to/Qwen3.8-Flash-Next-FP4-Local --host 127.0.0.1 --port 18731 --parallel 1 --ctx-size 16896
BLOOMERY_DRAFT=mtp BLOOMERY_HOST_ROOM=96G BLOOMERY_MTP_WIDTH=cost \
  bloomery-serve -m /path/to/Qwen3.8-Flash-Next-NVFP4 --host 127.0.0.1 --port 18731 --parallel 1 --ctx-size 16896
```

`BLOOMERY_DRAFT=mtp` keeps the draft on (section 5), and `cost` is what an unset `BLOOMERY_MTP_WIDTH` means. The
context and `BLOOMERY_HOST_ROOM` were the same in every round. Earlier Spark measurements, with their releases,
are in [`numbers.md`](numbers.md).

## 7. Other models on a Spark

Any model that fits the pool whole runs on the GB10 with no host tier. In the order to try them (sizes from
[`models.md`](models.md#what-each-model-needs), the GLM file's from
[`contrib/dgx-spark.md`](contrib/dgx-spark.md#22-first-placement-and-the-models-to-try) section 2.2): Clef-Flash 4.8 to 9.7 GB, Qwen3-30B-A3B `Q4_K_M` 18.6 GB, Qwen3.6-35B-A3B `Q4_K_M`
21.2 GB, then the Qwen3.8 folders above, Qwen3.8 `UD-Q4_K_XL` 111.3 GB and GLM-5.3-Flash `UD-Q2_K_XL` 101.3 GiB. A
64 GB Spark holds only the first three.

A model larger than the pool, such as GLM-5.3-Flash `UD-Q4_K_XL` or DeepSeek-V4.1-Flash `Q3_K_M`, loads at the
defaults and answers, with its overflow paged from the NVMe drive ([`serving.md`](serving.md#54-when-the-host-cannot-hold-the-experts-either)).
How fast that runs depends on the drive; [`contrib/dgx-spark.md`](contrib/dgx-spark.md#8-models-larger-than-the-pool-p4)
section 8 has the read rates measured on two Sparks.

One model across several linked Sparks is not served yet: `--place tp` plans GLM-5.3-Flash over several Sparks and
loads rank 0, then stops by name. Its state is in [`contrib/dgx-spark.md`](contrib/dgx-spark.md#9-several-sparks-p5),
section 9.

## 8. Troubleshooting

- **A start is refused for room.** The refusal names the bytes it asked for and the reading that gave the room. Free
  memory the Spark holds elsewhere (other processes, containers), or lower the slot count or context. The plan reads
  `MemAvailable` at start, so a `drop_caches` before the load is not needed.
- **Set the room by hand.** `BLOOMERY_HOST_ROOM=100G` makes the plan read that room instead of `MemAvailable`, for
  example to plan a start while something else still holds memory. A refusal then names the value as given.
- **`nvidia-smi` shows `[N/A]` for memory.** The GB10 reports no separate GPU memory there; bloomery reads the driver
  API and `MemAvailable` instead.
