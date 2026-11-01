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

FP4-Local takes less of the Spark's memory at the same settings (section 4) and decoded faster when the two were
measured side by side ([`numbers.md`](numbers.md#dgx-spark-qwen38-flash-next-on-the-fp4-local-and-nvfp4-folders-0310)).
On the GB10's `sm_121a` build, both folders' routed expert prompt products run on the FP4 tensor cores (weights and
activations in FP4), and the MXFP8 sites on the FP8 tensor cores.

Both folders keep the KV cache as q8_0, and FP4-Local needs a writable scale store; both are in
[`serving.md`](serving.md#92-qwen38-checkpoint-folders).

## 4. Requests at once, context and memory

With no context flag a slot asks the trained 262,144 positions and takes less where the Spark's memory cannot hold
that plan; `--parallel auto` counts the slots from memory. The rules are
[`serving.md`](serving.md#3-slots-and-context).

The GB10 and the CPU share one memory pool, so the plan reads its room from the host's `MemAvailable`, not from the
GPU driver's free-memory figure. A Spark with more running beside the server fits less. Read your own numbers with
the same flags and `--plan`, which prints the plan and exits before it loads anything:

```sh
bloomery-serve -m /path/to/Qwen3.8-Flash-Next-FP4-Local --parallel auto --slot-ctx 65536 --plan
```

The table is `--plan` on one Spark at `TBD-SITTING` (commit), with `TBD-SITTING` of `MemAvailable` before each plan.
Need is the `need` record's bytes: what the load takes of the pool (the GB10's share, the slots' checkpoints in
pinned host memory and file pages). The prompt cache's budget (`--cache-ram`; by default 8 GiB, or half of what the
plan leaves where that is less) comes on top.

| Settings beside `-m <folder>` | FP4-Local: slots × context | FP4-Local: need | NVFP4: slots × context | NVFP4: need |
|---|---|---:|---|---:|
| none | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `--parallel 2` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `--parallel 4` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `--parallel auto --slot-ctx 131072` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `--parallel auto --slot-ctx 65536` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `--parallel auto --slot-ctx 32768` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `BLOOMERY_DRAFT=off` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `BLOOMERY_DRAFT=off --parallel 4` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |

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

Measured at `TBD-SITTING` (commit) on a DGX Spark (GB10, driver `TBD-SITTING`, `sm_121a`), on the FP4-Local folder
at the serving defaults beside the flags named. The client sends each request to `/v1/completions` with a 4,096-id
prompt from a code corpus (`code-fn`) or an English prose corpus (`prose-en`), `max_tokens` 128, `temperature` 0 and
`cache_prompt` false; no two requests share a prompt. Decode is 1000 / TPOT, prefill the prompt tokens over TTFT, and total the tokens generated after each request's
first token over the time from the first first token to the last token. Each
value is the median of three rounds after one warm-up round; a round sends eight requests of each corpus one at a
time, then the two-request phase. tok/s unless named.

| One request | Decode, off | Decode, MTP | TTFT ms, off | TTFT ms, MTP | Prefill, off | Prefill, MTP |
|---|---:|---:|---:|---:|---:|---:|
| `code-fn` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `prose-en` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |

| Two requests at once | Total, off | Total, MTP | Decode a request, off | Decode a request, MTP | TTFT ms, off | TTFT ms, MTP |
|---|---:|---:|---:|---:|---:|---:|
| `code-fn` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |
| `prose-en` | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING | TBD-SITTING |

The servers:

```sh
TBD-SITTING
```

Earlier Spark measurements, with their releases, are in [`numbers.md`](numbers.md).

## 7. Other models on a Spark

Any model that fits the pool whole runs on the GB10 with no host tier. In the order to try them (sizes from
[`models.md`](models.md)): Clef-Flash 4.8 to 9.7 GB, Qwen3-30B-A3B `Q4_K_M` 18.6 GB, Qwen3.6-35B-A3B `Q4_K_M`
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
