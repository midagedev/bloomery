# The quantization segment of the release promo: what it compares, what each frame needs, how it is measured

`bloomery-quantize` ships its own quantizer for the routed-expert stacks (IQ1_S, IQ2_XXS on the card or `--cpu`); its bytes
are the bytes `llama-quantize` writes. The promo's quantization segment shows four things: it is faster, it is exact, the
model it makes is good, and the file fits. This page fixes each comparison, the numbers it needs, the sitting that
measures them and the prediction with its band. Chart data and the storyboard are in `quant-promo-data.md`; the
prediction cards are `docs/cards/quant-promo-*.card`. Written 2026-11-06 before any promo sitting. Tags: `[measured]` a
number a log or run holds, with its conditions; `[derived]` the arithmetic is shown; `[placeholder]` a named sitting fills it.

## Inputs this page stands on

| fact | value | source |
|---|---|---|
| K3.1 routed-expert stack | 3584 x 3072 x 896 = 9.865e9 weights, MXFP4 in (17/32 B a weight, 5.24 GB), IQ1_S out (50 B a 256 block) | `tools/ref/k31-quant.sh` header |
| K3.1 stacks | 276 (254 IQ1_S + 22 IQ2_XXS on the tight map), 2.7227e12 weights | same |
| card, one stack, IQ1_S, A6000 | 13.2 - 13.9 s = 7.1 - 7.7e8 weights/s, `read overlapped` | `/root/k31quant/gpuset.log`, 41 lines read 11-06 23:19 [measured, unleased] |
| card, one stack, IQ2_XXS | 16.4 s = 6.0e8 weights/s | same, blk.11 [measured, unleased] |
| card, the search kernel alone | 1.09e9 (IQ1_S), 8.3e8 (IQ2_XXS) weights/s | `docs/research/iq-quant-card.md` [measured, unleased] |
| llama-quantize CPU, 64 threads | experts 3 -> 16 in 600 s = 13 stacks = **46.2 s a stack = 214 Mweights/s** (trunk tensors inside the window, so a floor) | `/root/k31quant/quantize.progress`, the run 11-06 22:15-22:27 (`nice -n 10 ionice -c2 -n7`, `CUDA_VISIBLE_DEVICES=` empty, mainline 78feec062) [measured, contended] |
| the CPU model | 5.4 Mweights/s/thread x 40 effective threads = 216 Mweights/s | `k31-quant.sh` QUANT_RATE_MW x QUANT_EFF_THREADS [derived]; the measured 214 agrees |
| NVMe read / write | 1.6 / 1.5 GB/s | `k31-quant.sh` NVME_READ_GBS (measured floor), NVME_WRITE_GBS (derived) |
| K3.1 sizes by map | see the Size section | `docs/cards/k31-quantize.card` and `plan-*.log` dry-runs |

Two caveats the chart must carry. (1) The 13.4 s a stack is the **predecessor binary** (`iq_quant_set`), run **beside a GPU
`llama-imatrix` on the same A6000** (handoff 11-06: chainv1 on the A6000, chain12's card set on the A6000 by UUID), unleased. It is a
floor of the card rate, and `bloomery-quantize` is not the binary measured. (2) The 214 is contended (nice 10) and CPU-only; no
`--cpu` rate of `bloomery-quantize` is measured anywhere.

## 1. Speed

Question: how much faster is the card path than the CPU path, on the same input, imatrix and rules?

Headline [derived from the two measured rows]: per IQ1_S stack, 46.2 s (llama-quantize, 64 threads) against 13.4 s (card)
= **3.4x**; as rates 214 against 736 Mweights/s. Whole K3.1: the CPU path is 13,970 s (3.9 h, `k31-quant.sh quantize --pred-s`)
and the card path 5,450 s (91 min, `iq-quant-card.md`: 276 stacks x 14.3 s + the 0.66 TB set written in series at 1.5 GB/s +
llama-quantize over the set for the trunk), **2.6x** end to end. The stack ratio is larger than the end-to-end one because the
trunk, the set write and the read stay.

### 1a. K3.1, per stack (three arms, one sitting)

| arm | tool | prediction (Mweights/s) | band | card |
|---|---|---|---|---|
| card | `bloomery-quantize` (A6000, batch 32) | 736 [derived: the production log's mean, quiet floor] | 700 .. 1000 | `quant-promo-k31-card.card` |
| CPU reference | `llama-quantize` at 64 threads, normal priority | 214 -> 240 [derived: the contended measured rate; a quiet one is higher] | 190 .. 290 | `quant-promo-k31-cpu-llq.card` |
| host rule | `bloomery-quantize --cpu` | 214 [derived: same rule, same work; the band is a guess] | 150 .. 300 | `quant-promo-k31-cpu-bq.card` |

Common to the three: input `/home/user/stage/k3shape/Kimi-K3.1-Pro-k3shape-00001-of-00031.gguf` (the MXFP4 K3-shaped set, the
production run's input), imatrix `/root/k31quant/kimi-k31-c8192x4.imatrix.gguf` (the v0 production imatrix, md5 of its text
`c0ec381c...`), rules `/root/k31quant/rules-IQ1_S.txt`, the tight map. One lease (`tools/ref/lease-hold.sh --card
docs/cards/<slug>.card -- <command>`), each arm a time-boxed window (card 600 s, CPU arms 900 s), killed by `timeout`, the
partial output removed by name. Rate = sum of the IQ1_S stacks' weights over their seconds (the binary's per-tensor lines for
the card and the host rule; the 120 s progress file's counts for llama-quantize, whose log prints no per-tensor time).
IQ2_XXS stacks inside a window are reported apart (the chart shows IQ1_S). Run order inside the sitting: card, llama-quantize,
`--cpu`; the host arms need the card idle and the card arm needs the host quiet, so none overlaps.

The thread count: `docs/quantize.md` lists no thread flag; `crates/quantize/src/host.rs` prints `host rule, N threads` from
the resident pool. The row records N next to llama-quantize's 64; if N is not 64 the chart says so (the lead decides whether to
pin it).

Sitting wall [derived]: three windows (600 + 900 + 900 s) + three process starts and set reads (~60 s each) + witnesses
= ~48 min. Needs the A6000 free: today it holds the v0 card set (chain12, ~3.3 h from 23:10) and the v1 imatrix (chainv1, plan
10,972 s, ~02:13); the lead places the sitting after them (or after the release path: the handoff's own order).

### 1b. A mid-size model, end to end

Pick: **MiMo-V2.6-Flash-RL MXFP4**, `/data/moved/models/MiMo-V2.6-Flash-RL/MiMo-V2.6-Flash-RL-MXFP4-0000{1,2}-of-00002.gguf`
(167 GB). It is the only model on the box that the oracle covers as a source and that has routed-expert stacks: every other
mid-size GGUF on `/models` or `/data/moved/models` (Qwen3.6-35B-A3B Q8_0, Qwen3-30B-A3B Q8_0/Q6_K, Ornith/MiniCPM/Nanbeige Q4_K_M, the
fixtures) is Q8_0, Q6_K or Q4_K, which `bloomery-quantize` refuses by name (sources are F32, F16, BF16, MXFP4 only). Header read
on the box (read-only, 11-06): arch `mimo2`, 48 blocks, 256 experts of 2048 ff, embedding 4096; shard 2 holds 472 tensors:
**141 MXFP4 expert stacks of 4096 x 2048 x 256 = 2.147e9 weights, 3.028e11 weights in all**, 101 Q8_0 trunk tensors
(5.93e9 weights), 230 F32 (5.0e7). Every row length (4096, 2048) is a multiple of 256. It is the same size order as K3.1's
experts' share, in 1/10 of the bytes, so the three arms end in minutes, not hours.

Output [derived]: experts IQ1_S 3.028e11 / 256 x 50 B = 59.1 GB; the card set adds the trunk copied (Q8_0 5.93e9 x 34/32 =
6.3 GB, F32 0.2 GB) = **65.6 GB**; the final (trunk requantized Q8_0 -> Q4_K_M by llama-quantize) ~**63 GB**.

Three arms, the exactness frame's three files:

| arm | what runs | derived wall | band | card |
|---|---|---|---|---|
| A card | `bloomery-quantize` (card) -> `llama-quantize` over its set | 409 s card + 44 s set write + ~85 s stage 2 = **~540 s** | 450 .. 750 | `quant-promo-mid-a-card.card` |
| B host | `bloomery-quantize --cpu` -> `llama-quantize` over its set | 1415 s + 44 s + ~85 s = **~1545 s** | 1100 .. 2100 | `quant-promo-mid-b-cpu.card` |
| C reference | `llama-quantize` alone, 64 threads | 1415 s + trunk <= 28 s = **~1445 s** | 1200 .. 1750 | `quant-promo-mid-c-llq.card` |

Derivation of the terms: card stage 3.028e11 / 7.4e8 = 409 s (the production rate); CPU stage 3.028e11 / 2.14e8 = 1415 s (the
K3.1 measured rate; MiMo's stacks are the same rule on the same block shapes, so the per-weight cost is the same to first
order; IQ1_S's off-grid neighbour scan is data dependent, which is what the band covers); set write 65.6 GB / 1.5 = 44 s;
stage 2 = set read 41 s + trunk requantize <= 28 s (the IQ rate as an upper bound for K-quants) + final write 42 s, partly
overlapped (60 .. 110 s). Predicted speed-up C / A = **2.7x** (band 1.6 .. 3.9, the extremes of the two bands).

Prerequisites, in order, each a lead decision:
1. **The input's drive.** `/data` is sda, a rotational 17 TB disk (`lsblk` ROTA=1); at an assumed 0.2 GB/s [placeholder: unmeasured] the
   167 GB read is 835 s, over the card stage's 409 s, so the card arm would measure the disk. Staging the file to `/models/<name>`
   (903 GB free, nvme0, 1.6 GB/s floor = 104 s read, under the card's shadow) makes arm A a card measurement. The copy writes
   167 GB to nvme0, which dropped off the bus on 11-06 under conversion writes (memory: box-models-nvme-dropout): the lead
   decides; the alternative is a row named `[hdd-read-bound]`. No symlink (workstation rule): the arms name the final path.
2. **An imatrix for MiMo.** None exists and IQ1_S refuses to run without one (`llama-quantize` plan: "will require an imatrix").
   `quant-promo-mid-imatrix.card` (an exclusive hold, 10 .. 40 min [derived, loose]). Open: whether the quant tree 78feec062 loads arch
   `mimo2` for `llama-imatrix` is unchecked; if not, a speed-only run needs a synthetic imatrix GGUF (positive, seeded: the
   probe already uses a synthetic importance), and no tool writes one today (see "Outside the assignment").
3. **Rules.** `--tensor-type-file`: `blk\..*\.ffn_(gate|up|down)_exps\.weight=iq1_s` (inside the matcher's subset) and the
   remaining trunk by `llama-quantize`'s default for ftype Q4_K_M; `bloomery-quantize --plan --llama-plan <llama-quantize --dry-run
   TSV>` cross-checks them and prints the exact `llama-quantize` command for arm A/B stage 2. The tight-map machinery of
   `k31-quant.sh` is K3-specific (its reference table), so the three arms are bare commands under `lease-hold.sh`, not a runner.
4. **Drives for the outputs**: arm A writes 65.6 GB (set) + 63 GB (final); B the same; C 63 GB: ~320 GB, on the drive that
   is not the input's (nvme1 has 685 GB free today; K3.1's stage dir is in use until the release path ends).

Sitting wall [derived]: imatrix 10 .. 40 min, then A 9 min + B 26 min + C 24 min + sha256 of three 63 GB files (below) +
process starts = ~75 min, plus the copy (167 GB at the HDD's read rate: [placeholder: the copy's own log]).

## 2. Exactness

Question: are the bytes the bytes `llama-quantize` writes? The frame shows identity, not closeness.

- **Fixture, already proved**: `just gate-gpu-iq-quant`'s end-to-end clause on the K3 fixture: `bloomery-quantize` then
  `llama-quantize` equals `llama-quantize` alone, byte for byte. The frame is two `sha256sum` lines (whole file) that match,
  next to the gate's green line. The box run is the gate's: [placeholder: filled from the lead's next `gate-gpu-iq-quant`
  record, with its commit and card line].
- **At scale**: the mid-size run's three finals, `sha256sum` of A, B and C: the same hash three times (63 GB each, cached or
  read at the drive's rate; ~40 s each at ~1.5 GB/s single-thread SHA-NI [derived: assumed rate, unmeasured]). This is the whole-file
  claim on a real 167 GB model and the arm B line is the `--cpu` equality. Predicted: equal; a mismatch is not a result for the
  video, it is a bug (the gate's stop-by-name `--check` names the tensor, expert, row and byte).
- **K3.1, per tensor**: `bloomery-quantize --check N` (N rows of every quantized tensor through the host rule too) over the
  production input prints per-tensor `ok`; the frame is that line plus one flipped-byte run the gate already proves (FAIL-first:
  stops by name with tensor, expert, row, byte). Per-tensor sha256 against `llama-quantize`'s own K3.1 tensors would be the
  stronger frame, and the only llama-quantize K3.1 expert tensors were written by the killed CPU run of 11-06 (22:15-22:27) into
  `/home/user/stage/Kimi-K3.1-Pro-IQ1_S-k3shape.partial`; `k31-quant.sh` removes a stale `.partial` at the next run's start. One
  read-only `ls` on 11-06 late showed no usable listing of it (the output was empty): the lead checks before the next quantize run,
  and knows that llama-quantize writes the GGUF header at the end, so a killed run's partial file has data and no parsable header
  (tensors would be compared by offset from the plan's TSV, unverified). Given that, the MiMo whole-file frame is the real-model one.

## 3. Quality

Question: how good is the model our quantizer made, against the released K3.1?

Harness: `tools/ref/k31-eval.py` (teacher = the released model through OpenRouter, 150 held-out prompts in
`tools/ref/k31-eval-prompts.jsonl`, EN / KO / code; student = `just gen-kimi-k3 --score`; scored: per-position top-1 agreement,
NLL difference, truncated KL over the teacher's top 20, KL over the union). The lead scores; this page defines the frame and the
data columns (`quant-promo-data.md`, chart 5). Columns are `score`'s exact ones: set, prompts, positions, top1 agree,
dNLL (nats), KL top, KL union, each with a 95 % interval as the tool prints it.

The two sets that carry the story, both the tight map (IQ1_S experts, Q4_K trunk) over the same K3.1 input:
- **v0**: imatrix `kimi-k31-c8192x4.imatrix.gguf`, calibration `calib-mix-c8192x4.txt` (raw text: prose and code), the production set.
- **v1**: imatrix from the chat-rendered corpus `calib-chat-c8192.txt` (K3.1 wrote the chats, rendered with its own encoder,
  `--corpus-special`; the GPU imatrix is chainv1, `--seqs 3`, plan 10,972 s, ~02:13 KST 11-07 per the handoff).
- optional third row: the K3 `UD-IQ1_S` file on the same prompts, the comparison the `ud` map exists for. [placeholder: only if the lead
  wants it; K3 is a different model, so it is a size-class reference, not an equal-model comparison.]

Each scored set is one full quantize run: the card path, 91 min [derived] (`k31-quant.sh --gpu-quant`), then a scoring pass on
the box. Predictions are the lead's to set once a first score exists; none is invented here. What the frame shows: the three
category bars (EN, KO, code) for v0 and v1, top-1 agreement and truncated KL, and one line stating that bytes are ggml's so
quality is the imatrix's and the map's, not the quantizer's.

## 4. Size and fit

Bytes per map [derived: `llama-quantize --dry-run` over K3's 2573 tensor shapes, `plan-*.log`; K3.1 has K3's shapes]:

| set | MiB | GB (x 1.048576) | bpw | note |
|---|---|---|---|---|
| source (MXFP4 experts + BF16 trunk) | 1,488,919.97 | 1,561 | 4.49 | `prepare` output, 1.56 TB |
| tight (IQ1_S + 22 IQ2_XXS stacks, Q4_K trunk) | 553,113.74 | 580.0 | 1.67 | experts 545.35 GB, trunk 30.36 GB |
| trunk-q5 | 559,452.10 | 586.6 | 1.69 | trunk 37.04 GB (over the 35 GB `--max-trunk-gb` of tight) |
| ud (K3's UD-IQ1_S table, Q6_K trunk) | 566,514.80 | 594.0 | 1.71 | equals the K3 file's tensor bytes, 594,033,688,806 B |
| all-IQ2_XXS (tight) | 702,428.62 | 736.5 | 2.12 | the second run's size |

What fits where (one A6000 of 48 GB + 251 GB host RAM + NVMe, the box): the tight trunk (30.36 GB) is under the card's 48 GB; the
experts (545.35 GB) exceed card + host RAM (48 + 251 = 299 GB), so the set streams from NVMe through the host tier, which is the
box's serving path; the source (1.56 TB) fits no tier but the drive. Placement numbers beyond these byte sums are the plan's:
`bloomery-serve --plan` / the placement census prints them for the machine and the video shows that line, not a derived one
[placeholder: the lead's final speed sitting record].

## What the lead must run, in order (predicted walls)

1. Nothing for the promo while the v0 card set, the v1 imatrix and the release path hold the A6000 (handoff 11-06). The promo
   sittings follow them.
2. Read-only, now or any time: `ls` the `.partial` and keep it if usable (2 above); decide the MiMo copy (1b prerequisite 1).
3. **Sitting S1 (K3.1 speed rows, A6000, ~48 min)**: cards `quant-promo-k31-card`, `-k31-cpu-llq`, `-k31-cpu-bq`, in that order,
   each under `lease-hold.sh`; per-arm window as above.
4. **Sitting S2 (mid-size, A6000, ~75 min + the copy)**: imatrix (`quant-promo-mid-imatrix`, 10 .. 40 min), then arms A, B, C
   (`quant-promo-mid-{a-card,b-cpu,c-llq}`; 9 + 26 + 24 min), then the three `sha256sum`s. The arm order is A, C, B (card first
   while the page cache is clean, the reference before the host rule) with the output drive emptied between arms.
5. **Quality (Q1)**: v0 is the production set, so only the scoring pass is new; v1 needs its own card set + trunk run (91 min each
   on the card path) and the scoring pass; the lead scores and fills chart 5.
6. `just gate-gpu-iq-quant` (fixture frame) with its record line, and `bloomery-quantize --plan` on K3.1 for the frame's plan
   lines (reads headers only, no write).

Sittings S1 and S2 can share one lease window per card; they need the A6000 and the whole host, so they do not overlap each
other nor the production quantize.

## Outside the assignment (report only)

- `tools/ref/k31-quant.sh` takes only the K3 reference table (`--ref-table`): a generic "quantize any MXFP4/BF16 MoE with a rules
  file" runner would make the mid-size arms one command; ~150 lines, medium.
- A **synthetic-imatrix writer** (a GGUF with positive seeded entries for a model's `*_exps` stacks) would let speed-only runs of
  any model skip `llama-imatrix`; the probe's synthetic importance has no GGUF writer today. Small.
- `docs/quantize.md` has no thread flag for `--cpu`, so "same thread count" is not controllable: a `--threads N` that sizes the
  pool would make the host-vs-reference row exact. Small.
- `llama-quantize`'s log has no per-tensor time; only the runner's progress file prices it. A `--time-per-tensor` wrapper or the
  progress file at 30 s granularity would sharpen the CPU row. Small.
- `gpu-gate.sh` outside `box.sh` leaves the device to CUDA's order (handoff 11-06 22:34): the A6000/3090 identity of a row is
  not in its log; the witness block would carry it if `gpu-gate.sh` printed the chosen UUID. Small.
- The film pipeline named in the brief (`~/.cache/bloomery/film/tools/`) exists on neither the Mac nor the box, and the handoff
  mentions it nowhere: the assembler's interface is unread; the chart CSVs and beats are written to be tool-agnostic.
