#!/usr/bin/env bash
# V4.1 decode by depth and prefill by prompt length, three engines in one lease on the timing card
# (run on the box, lead-only): our engine (generate_ds41 --depth D --time --place a|gate), ik
# (llama-bench -gp D,N, or -p P -n 0 for prefill, at the profile's IK_GPU_FLAGS, under its
# IK_GPU_ENV) and mainline llama.cpp (llama-bench -d D, or -p P -n 0, at the profile's
# LCPP_GPU_FLAGS), alternated arm by arm or run in engine blocks (Order below).
#
#   BLOOMERY_MODEL=deepseek41 tools/box.sh 'bash tools/ref/depth-ds41.sh 6 ik:6 lcpp:6'
#   just depth-gpu-ds41 6 lcpp:6 4096 lcpp:4096
#   just depth-gpu-ds41 512 ikpp:512 lcpppp:512 4096 ikpp:4096 lcpppp:4096
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just depth-gpu-ds41 6 lcpp:6    # the command lines, no lease, no load
#   BLOOMERY_BOX_ENV='BLOOMERY_AB_ORDER=blocks' just depth-gpu-ds41 6 512 lcpp:6 lcpppp:512   # engine blocks
#   BLOOMERY_BOX_ENV='BLOOMERY_AB_ORDER=blocks' just depth-gpu-ds41 6 512 lcpp:6 lcpppp:512 lcppfit:6 lcppppfit:512
#   tools/ref/depth-ds41.sh --parse FILE    # an ours arm's lines, row and residency curve from a saved output
#
# The V4.1 sibling of depth-gpu.sh, and it blocks the same failure: a ratio read at one depth and
# quoted as "decode is faster" — a step's attention term grows with the cached keys, so the depth
# goes into every number (`tok/s @ n=N, depth D, <card>`).
#
# Arms, in lease order under the default order (it rotates by one slot each round — the position
# bias ab-decode.sh names; Order below has the engine blocks):
#   <D>      ours: generate_ds41 --depth D -n N --time. The fed ids are lease.sh's lcg_prompt D,
#            one real decode step each, untimed; generated token 0 comes out of the last of them and
#            the N - 1 steps after it are timed. The context is the binary's default, the serving
#            ctx_max the plan (a) is made for, at every depth: the plan's card expert prefix depends
#            on ctx_max, so a per-depth ctx would move experts between the card and the host.
#            The row also carries `pp_tok/s <v> (n=D, passes=K)` from the binary's `time prompt`
#            row, the wall of those D fed steps (Prefill below): one arm reports both the prefill
#            of D ids and the decode at depth D.
#   ik:<D>   ik: llama-bench -p 0 -n 0 -gp D,N -r 1 (D = 0: plain tg N). llama-bench sizes its
#            context to D + N itself and feeds its own prompt ids, not lcg_prompt's. It runs as
#            `env $IK_GPU_ENV`: without GGML_CUDA_NO_PINNED_WEIGHTS the CPU expert overrides turn
#            the host context into one pinned allocation larger than RAM and the load fails (the
#            profile's IK_GPU_ENV comment).
#   lcpp:<D> mainline: llama-bench -p 0 -n N -d D -r 1 $LCPP_GPU_FLAGS (D = 0: plain tg N).
#            Mainline has no -gp; -d prefills D tokens before its clock starts, and its row label
#            is `tgN @ dD` where ik's is `tgN@ppD`. llama-bench sizes n_ctx to D + N and the
#            context pads it to 256 (src/llama-context.cpp). No environment: mainline keeps the
#            host context on the file mapping (the profile's LCPP_GPU_FLAGS comment).
#   lcpp<K>:<D>  the same with --n-cpu-moe K in place of the profile's LCPP_NCMOE: the sweep arm.
#            The profile's LCPP_NCMOE_SWEEP names the values that load, e.g.
#            `lcpp:6 lcpp34:6 lcpp35:6` interleaved with `6`.
#   ikpp:<P> ik's prefill: llama-bench -p P -n 0 -r 2 -o json with the ik:<D> arm's binary, flags
#            ($IK_GPU_FLAGS) and environment ($IK_GPU_ENV); the row is the second repetition,
#            P / samples_ns[1], as `tok/s(pp) … @ n=0, prompt P`, and names the first as `cold`.
#            The file is larger than RAM, so an arm that follows another engine refaults its own
#            pages during its first repetition; one load, two samples, the warm one counts.
#   lcpppp:<P>  mainline's prefill, the same at $LCPP_GPU_FLAGS, the lcpp:<D> arm's.
#   ikpp<U>:<P>, lcpppp<U>:<P>  the same with -ub U -b max(U, 2048): the ubatch lever, not a default
#            (Prefill below). Refused when the profile's flags already name -ub or -b.
#   lcppfit:<D>, lcppppfit[<U>]:<P>  mainline at its own placement: the lcpp:<D> and lcpppp[<U>]:<P>
#            arms' command lines with the profile's placement options (-ngl, --n-cpu-moe, -ts, -ot)
#            removed and `-fitt 1024 -v` added, so llama-bench's fit chooses the layers and the
#            overrides (Fit below). The row carries `fit <what it chose>`; refused before the lease when
#            the tree's llama-bench has no --fit-target, or when the profile's flags carry a fit option.
#   lcppsrv:<D>, lcppsrvpp[<U>]:<P>  mainline warm, on our ids: the tree's llama-server (lcpp-warm.sh:
#            LCPPSRV, else the llama-server beside LCPPBIN) at LCPP_GPU_FLAGS in the server's spellings, one
#            process a row. After it listens, one POST /completion of the arm's ids is sent and discarded
#            (the warm-up), then the same request is timed. The ids are the ones ours feeds at that D or P:
#            lease.sh's lcg_prompt, which --depth feeds. A decode arm asks n_predict N, greedy, and its row
#            is predicted_per_second (the N - 1 decode steps over their wall); a prompt arm asks n_predict 1
#            and its row is prompt_per_second (lcpp-warm.sh has both against llama-bench's, and every flag
#            the server gets that the bench has no word for). <U> is the ubatch lever as for lcpppp<U>. The
#            row carries `ids=lcg`, the warm-up's rate and fault count, whether the timed continuation is
#            the warm-up's, prompt_n, build, device, majflt (timed: around the timed request, against W =
#            its prompt_ms + predicted_ms) and wall. prompt_n other than the ids sent, cache_n other than 0,
#            predicted_n other than asked, a server that exits or never answers /health inside
#            BLOOMERY_ARM_BOUND: each a FAIL row. Refused before the lease when the tree has no llama-server,
#            its --help lacks a flag the arm passes, or LCPP_GPU_FLAGS holds a word lcpp-warm.sh's table
#            does not translate. No preheat: the warm-up request reads the arm's set in its own process.
#   lcppsrvfit:<D>, lcppsrvppfit[<U>]:<P>  the same at the server's fit: the lcppfit arms' flags, the
#            server's `-fit on -fitt 1024 -v`, and their `fit` column.
#   lcppsrv…:prose:<P>, lcppsrv…:code:<P>  a server arm fed the corpus's first P ids, the ids a prose:<P> or
#            code:<P> arm feeds: row label `<engine>@prose` (`@code`), `ids=prose`, in that corpus's tables.
#   <D>@NAME=VALUE[,NAME=VALUE...]  ours at depth D with those variables set (`env NAME=VALUE ...`):
#            a lever arm of the same binary, row label `ours@NAME=VALUE[,...]`. Beside a plain `<D>`
#            arm it is the same-binary A/B, e.g. `6 6@BLOOMERY_PIN_MAIN=0`.
#            An arm with BLOOMERY_DRAFT=dspark also sees the other card, where the draft runs, and
#            gets the profile's DSPARK_MODEL unless it names one (timing-card.sh dspark_env).
#   prose:<P>[@NAME=VALUE[,NAME=VALUE...]]  ours fed the first P ids of
#            $BLOOMERY_DATA/engram/corpus-prose.ids (one id per line) instead of the LCG prompt:
#            generate_ds41 --tokens <those ids> -n N --time, with the variables set as in <D>@…. Row label
#            `prose`, or `prose@NAME=VALUE[,...]`; the depth column is P. The prompt's routing, and so
#            its card and host work, is prose's, not the LCG walk's: a prose arm is compared only with
#            prose arms of the same P — `prose:512 prose:512@BLOOMERY_PREFILL_GROUP=1` is the
#            same-binary A/B on the prose prompt — in its own decode and prefill tables (prose / each
#            prose@ label), never with ours or the references. A P the file cannot supply (P < 1, or
#            past its line count) is refused before anything runs.
#   code:<P>[@NAME=VALUE[,NAME=VALUE...]]  the same on $BLOOMERY_DATA/engram/corpus-code.ids, row label
#            `code` or `code@NAME=VALUE[,...]`, in tables of its own (code / each code@ label). prose
#            and code are the corpus arms (CORPORA): corpus-<name>.ids, one id per line; a corpus is
#            compared only with arms of its own name and P, never with another corpus, ours or the
#            references.
#   bin:<path>:<D>  a second generate_ds41 (an absolute path on the box, a base tree's build) at depth
#            D, row label `bin:<basename of its tree>` (the tree is the path above `target/`). It is a
#            base by construction, so its freshness is not asked; its tree line (sha256, HEAD, dirty
#            files) is printed with the references'.
# Every label is its own engine in the per-arm means and in the ratio table (ours / each other label,
# the corpus labels excepted; <corpus> / each <corpus>@ label in that corpus's tables).
# Prefill values (ours' `time prompt`, the pp arms) have their own means and ratio table per prompt
# length P, never the decode ones'.
# Our arms run at any depth up to the plan's ctx_max (every indexer layer selects its list at every
# position); generate_ds41 refuses only D + N - 1 > --ctx, and a refused arm is a FAIL row (Failures).
#
# Placement. BLOOMERY_GEN_PLACE (a, the default, gate or bp; anything else is refused) is the placement
# every generate_ds41 arm — ours, the corpus arms, bin: — loads by, passed as --place; the [config]
# line and every such row name it (`place <p>`), and a row whose SMOKE footer names another placement
# is a FAIL row. Plan (a) loads on the card named A6000 and the gate plan on the one named 3090
# (workstation::plan_a, plan_gate: the card is found by name), and the arms see the timing card only,
# so a placement whose card is not the timing card (BLOOMERY_TIMING_GPU, timing-card.sh) is refused
# (64) before the lease and before a dry run's command lines — when an arm runs generate_ds41. With the
# 3090 as the timing card the profile sizes the references' --n-cpu-moe for its 24 GB
# (tools/ref/models/deepseek41.sh, LCPP_NCMOE). bp is plan (b′) (workstation::plan_bp: plan (a) on the
# A6000, the 3090 its expert tier, the DSpark draft beside the tier), which loads both cards: it runs only
# in the two-card mode (Two cards, below), and outside it is refused (64) as a and gate are inside it.
# What follows describes plan (a).
# Ours is plan (a): every layer and the head on the A6000, each routed layer's experts
# [0, n_l) on the card (n_l 63-64 of 384, the budget's), the rest on the host tier — the plan line
# generate_ds41 prints. ik moves experts by tensor, and a layer's 384 experts are one tensor, so it
# cannot keep a prefix of every layer; the profile's --n-cpu-moe keeps the experts of the first
# layers on the CPU and the last ones' whole on the card, sized to the bytes plan (a) gives the
# card's experts. The per-token host work is the same in expectation (6 routed experts per layer,
# each with the host's share of the probability), the shape is not: ours joins the host once per
# layer, ik's card layers never wait on the host and its CPU layers never use the card's share.
# Mainline places by the same rule (--n-cpu-moe), so an lcpp row has ik's shape; the profile sizes
# both counts for MODEL (IK_NCMOE and LCPP_NCMOE, one arithmetic — the two engines place
# the same tensors on the card), so the two references hold the same layers on the card.
# Fit. The lcppfit arms leave the placement to llama-bench's fit (tools/ref/lcpp-fit.sh has what it
# does and where): every layer's dense part on the card, then whole layers' experts front to back, then
# part of one more layer (its up, then its gate projection), the rest of the experts on the host — the
# last layers', where --n-cpu-moe K's are the first K. -fitt 1024 is the margin llama.cpp's `-fit on`
# leaves (common/common.h:481). Predicted on this file and the A6000 [derived, from the profile's
# LCPP_NCMOE bytes and the fit's rule, up and gate taken as Q3_K beside the Q4_K down; the card's free
# memory and the compute buffer not measured]: 47,568 MiB to fill (49,140 less the driver's 548 and the
# 1,024 margin), less 3,429 MiB of card dense and a compute buffer the hand-set load leaves at most
# ~2,000 MiB for, leaves 42,100-43,700 MiB for experts. Layers 0-1 (6,682.5 MiB each) and 2-5 (6,142.5)
# whole are 37,935 MiB; a seventh whole layer (44,077.5) does not fit, its up and gate (about 3,713)
# do: layers 0-5 whole, layer 6's down and layers 7-39 on the host, 41,648 MiB of experts on the card
# against --n-cpu-moe 33's 42,997.5. The host then reads 205,133 MiB of experts against 203,783 (+0.7 %),
# so a lcppfit row predicts at or up to about 1 % below its lcpp twin, inside the ruler at three rounds.
# Not in that figure: the partial layer's pattern is by name (blk.6.ffn_down.* for the gate fraction),
# so it also moves that layer's shared-expert down, a dense tensor every token uses, to the host — a
# host op and a split in layer 6 that --n-cpu-moe never makes (the `fit` column's `(blk 6: <k>)`).
# The row's `fit` column is the measured placement (predicted `overridden CPU:<n> in blk 6-39 (blk 6:
# <k>)`), and every arm records its own: n_ctx (the KV cache) and -ub (the compute buffer) enter the fit.
#
# Paging. The file (about 347 GB) is larger than the page cache can grow (Cached peaks near 262 GB
# in the witness blocks), and the engines' host expert sets differ: ours is experts n_l.. of every
# layer, populated at our load before our timer; a reference's is every expert of its first K layers
# (--n-cpu-moe K), read through the file mapping inside its timer. A reference arm that follows ours
# therefore starts with part of its set evicted and pays NVMe reads inside its timed window.
# Preheat. Before every reference arm (ik:, ikpp…, lcpp…, lcpppp…, a warm-up included) the runner
# reads that arm's host set into the page cache: token_embd (both engines keep the input embedding on
# the host) and every tensor matching blk.<i>.ffn_(up|down|gate|gate_up)_(ch|)exps with i < K, K the
# arm's own --n-cpu-moe (the override list llama-bench builds, common/common.h LLM_FFN_EXPS_REGEX).
# tools/ref/gguf-ranges.py reads the ranges from the header of every shard, once per K before the
# lease, and preheats them (pread in chunks, the data discarded). The read sits outside the arm's
# witness blocks and its row's wall, and prints `preheat <engine> K=<k> bytes=<b> s=<t> gbps=<rate>`:
# the rate tells how much came from NVMe and how much from the page cache. Not preheated: the engram
# tables and what the engine uploads to the card. The engram tables are 84.5 GB of lazy rows, and a
# token reads about 49 4-KB pages of them (48 rows: 2 layers x 3 n-gram orders x 8 heads, plus the
# rows that straddle a page) [derived, round memfit]. They are no small term: llama.cpp's V4.1 build
# faults each row in alone (MADV_RANDOM on the lazy range, GET_ROWS as one task), about 1.9 s per 512
# tokens it has not read before — the whole of a cold reference pp row's deficit (rig-log
# 2026-09-27#v41-xeng) — and which rows a run reads depends on its token ids, which only the block
# order's discard process covers (Order below). What the engine uploads to the card it reads once at
# load, before its timer; preheating that too would evict the host set it is meant to keep: at
# K = 33 the host set and the card set are 262.8 GB together [derived], more than the file cache holds.
# Flags whose host set this rule does not model (-ot, --override-tensor,
# -cmoe, --cpu-moe, -fitt, --fit-target, a list for --n-cpu-moe, -ngl below the block count + 1) are
# refused before the lease — so a fit arm runs under BLOOMERY_AB_ORDER=blocks (its block's discard reads
# its set) or with BLOOMERY_PREHEAT=0. BLOOMERY_PREHEAT=0 turns the preheat off (the same-lease A/B of
# it); any value but 0 or 1 is refused. Its default follows the order: 1 under rotate, 0 under blocks. Under blocks the block's
# discard process has just read the host set through the same mapping, with no other engine between it
# and the block's arms: at -nopo 1 a prompt runs the host layers' experts on the host, and a 4096-token
# prompt leaves a host layer's expert unread with probability (1 - 6/384)^4096 < e^-64 [derived]. The
# preheat would read a resident set — 6.4-28.5 s of lease wall before each reference arm in the two
# windows of rig-log 2026-09-27 — for a window term of 0 [derived]. BLOOMERY_PREHEAT=1 under blocks
# preheats every reference process, the discard included. A dry run prints each reference arm's K,
# bytes and ranges and the seconds they take cold at PH_RATE, and reads nothing but the headers.
# Cold tag. Every row carries `majflt <n>`, the change in /proc/vmstat pgmajfault across the arm's
# process — a reference's load and warm-up included — and `timed <n>`, the change over the row's
# measured window, where the runner can see that window start: an ours, corpus or bin arm counts from
# its `fed` line (generate_ds41 prints it just before its prompt timer starts) to its exit; an lcpp arm
# (lcpp…, lcpppp…) runs llama-bench with --progress and counts from the progress line of its last
# repetition's timed test to its exit — `prompt run 2/2` for a pp arm, `generation run 1/1` for a
# decode arm, each printed after the repetition's clock starts (tools/llama-bench/llama-bench.cpp:2440,
# 2444, 2457 on the V4.1 branch; `depth run` comes before the clock). ik's llama-bench has no
# --progress, so an ik row counts its whole process. Both counts read the machine-wide counter, which
# the lease keeps to this arm's process, as the `fed` count does. The count (the timed one where there
# is one) is set against the row's own timed window W: P / tok/s for a pp row, N / tok/s for a decode
# row, the prompt's ms plus N × mean ms for ours. A fault is priced at COLD_US microseconds, the serial
# cost of one engram fault measured on this box (rig-log 2026-09-23: 75.4 µs a 4 KB fault, taken one
# after another on one thread, the way llama.cpp's V4.1 build reads its engram rows) and the largest
# serial cost a fault has been measured at here. A row whose faults could have cost COLD_PCT percent of
# W or more (the ruler at four rounds) ends in ` [cold]`, and the column prints that bound (`≤ <x> % of
# W`). The counter does not tell the two kinds of fault apart: an engram fault reads 4 KB and waits
# alone; a host-set fault reads around (up to the device's read-ahead) on one of the threads that share
# the expert work, and its serial cost has not been measured. So the bound is an upper bound for engram
# faults and a price by count for the others: an untagged row lost under 1 % of W if its faults were
# engram faults, and a tagged row faulted at least that often inside its window — or, an ik row, in its
# load or warm-up, which a whole-process count cannot tell apart. The witness still prints the page
# cache and the fault count before and after every arm. BLOOMERY_GEN_WARM (generate_ds41 --warm) trims
# our arm's first steps.
# With BLOOMERY_STEP_STATS=1 (BLOOMERY_BOX_ENV on the Mac side) our arm's `stat summary` line is
# echoed with its load lines; the per-step `stat step` lines stay in the arm's output only.
#
# Prefill. pp_tok/s is P over the wall of processing a P-token prompt, in each engine's terms:
#   ours     generate_ds41's `time prompt` row: before the first fed step to after the readback of
#            generated token 0, `passes` the steps it took. V4.1's ours is one decode step per
#            token today, so its pp equals its step rate at the fed depths by construction — at or
#            above it: the plain feed is one body per id and one readback at the end, a timed step
#            one body and one readback. A DSpark arm's feed reads back per id and also reads each
#            position's features into the draft (`kind=dspark`).
#   ik, lcpp llama-bench's pp test: llama_decode over the P ids in batches of -b and ubatches of
#            -ub, then one synchronize (test_prompt), one repetition, llama-bench's own value.
#            Both trees default to -ub 512 -b 2048; the row names the batch sizes that ran.
# The arms do not start alike. The warmup before llama-bench's timed repetition is a 1-token prompt
# in ik (examples/llama-bench/llama-bench.cpp:2230) and the whole prompt in mainline
# (tools/llama-bench/llama-bench.cpp:2378 on the V4.1 branch); ours has none: the prompt is the
# process's first work after the capture. So ik's and our rows carry first-touch costs mainline's
# does not; the witness's page cache and pgmajfault lines show the paging part of them.
# Where the host experts run in prefill: at -ub 512 ik's CUDA backend copies no host MUL_MAT_ID
# weight to the card (it does at ubatch × 6 used experts >= 32 × 384, -ub >= 2048 [derived, from the
# profile's IK_NCMOE comment and ggml/src/ggml-cuda.cu:5226 at the default offload batch size]), and
# mainline runs -nopo 1, so both prefill the host layers' experts on the host threads, as ours does
# at every position. The ubatch lever:
# ikpp<U> at U >= 2048 crosses ik's offload line, whose compute buffer holds a layer's expert
# tensors, more than the profile's placement leaves free on the card [derived, the profile's
# LCPP_GPU_FLAGS comment] — pair it with an IK_GPU_FLAGS override of --n-cpu-moe inside the box
# command; lcpppp<U> keeps -nopo 1.
#
# Failures. An arm that exits non-zero, prints no row (no SMOKE line, no time prompt row, no
# llama-bench value), fails its preheat, ran another placement than it was given, or broke the residency
# seed condition (Residency below, rc=residency) prints
# `FAIL r<r> <label> d=<D>|p=<P> rc=<rc> | <why or its last line> | full output: <file>` where its
# row would be, and the runner goes on with the next arm. That label at that depth or P drops out of
# the means and the ratios (the tables name what they dropped), and the runner ends with `failed arms:
# …` and exits 1 (cold-blocks.sh's arm_fail, failed_tally and failed_end, depth-qwen3moe.sh's too). A
# warm-up that fails is `FAIL r0 …` and counts in that list. An arm that fails in a
# load shared with other arms ends that process: the arms after it in the round re-run in a fresh load
# (Loads below), never on the failed one.
#
# Loads. A round's ours and corpus arms that share a load key — this binary, --place and the arm's
# NAME=VALUE list, every variable of which the binary consumes at its load — run in one process:
# generate_ds41 --arm <D|prose:P|code:P> ... --arm-sync, the engine cleared between two arms
# (app::Session::clear, bit for bit a fresh process's state). The binary waits after each arm's `arm`
# line, so the guards and witness blocks stand before and after every arm as before; the row's wall,
# majflt (from the go) and timed majflt (from its fed line) are the arm's own, and the row carries
# `slot <k>/<n>`, its place in its load. The load's own lines print once, under `[load]`. The grouping,
# the order (units and the arms inside each rotated by round) and the process driver are
# tools/ref/load-groups.sh's, shared with depth-qwen3moe.sh. BLOOMERY_AB_LOAD=arm runs every arm in a
# process of its own, and an arm's own `@BLOOMERY_AB_LOAD=arm` runs that arm alone (its label keeps
# it: `prose:512 512 prose:512@BLOOMERY_AB_LOAD=arm` is the A/A of the clear, the fresh process against
# the in-load arm). A BLOOMERY_DRAFT or BLOOMERY_CHECK_FINITE arm always runs alone (a draft's state has
# no clear), and a bin: arm is its own process with the one-arm command line. The warm-up and the
# blocks' discards are one arm in a process of its own.
#
# Residency. An arm whose environment turns adaptive residency on — BLOOMERY_RESIDENCY set to anything
# but off, by its own @ list, else inherited by the runner (the lever's words are off, mid-p0-s1 and
# mid-p40-s1; unset is off) — must start its timed work from the seed map, the load's placement
# (docs/fair-measure.md 2.5). The runner holds each such arm to the engine's own records, per
# generate_ds41 process: the arm's first `residency pass` is pass=none boundary=0 (the first boundary
# after the load or after a reset: at slot 1 the fresh load's, at a later slot the one after its clear);
# and at slot k > 1 the lines between the previous arm's work and this arm's `arm` line — where
# generate_ds41 prints the clear's report, load-groups.sh's LG_PREV_OUT — hold exactly one `residency
# reset`, with diff=0 and dropped_bytes equal to the profile's RESIDENCY_RESET_DROPPED_BYTES (V4.1: 0,
# models/deepseek41.sh). A `residency reset` among an arm's own lines, before the last record of another
# kind, fails that arm too. The rotate warm-up and the blocks' discards are processes of their own, so
# they warm no timed arm's map; a same-id PRIME (Warm rows), or any arm before this one in its load, is
# what the reset clause covers: with no reset between, the arm's first pass is not none/0 and there is no
# reset record, and it fails. An arm with residency off that prints a `residency pass` or `residency
# reset` record fails as well (the lever leaked into an off arm). A failure is `FAIL r<r> <label> d=<D>
# rc=residency | <each clause that failed> | full output: <file>` in place of the row. A residency arm
# under a profile without RESIDENCY_RESET_DROPPED_BYTES is refused (64) before the lease. Every
# residency row is followed by `residency curve <label> r<r> d=<D> windows=<w> tok/s=<a>,<b>,…
# flips=<n> | <its row's tag> slot <k> | seed <what held>`: the arm's timed passes (`time step`, one
# position each; under a draft `time pass`, its positions) in windows of RES_WINDOW (16) passes, in
# order, the last window short when they do not divide, and the flips its `residency pass` records made
# (the sum of `made`) — a printed line in no mean; the prompt call is not in it (the row's pp column is
# its rate).
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 3),
# BLOOMERY_GEN_WARM, BLOOMERY_GEN_BIN (default target/release/generate_ds41), BLOOMERY_GEN_PLACE and
# BLOOMERY_PREHEAT (above), BLOOMERY_ARM_BOUND (seconds one arm, or one preheat, may run, default
# 900: a hung arm ends at rc 124/137 as a FAIL row instead of holding the lease; a shared load's
# process has that bound per arm and one more for its load, and one that prints nothing for that long
# is killed), BLOOMERY_AB_LOAD (Loads above), BLOOMERY_AB_WARMUP and BLOOMERY_AB_ORDER (below), BLOOMERY_WARM_ROWS (Warm rows), BLOOMERY_DRY=1 (print each arm's command line, the binaries' tree
# lines, the preheat plan, the CPU guard's settings and reading, the warm-up or the blocks' discards,
# and the order of every round, then exit 0 before the lease: nothing is loaded and nothing is timed).
#
# Order. BLOOMERY_AB_ORDER=rotate (the default) runs every arm once a round, the arms of all engines
# in one order rotated by one slot each round. BLOOMERY_AB_ORDER=blocks runs the arms in engine blocks,
# the blocks in the order their first arms are given; a block runs all its rounds together, its own
# arms' order rotated by one slot each round. Any other value is refused. The blocks:
#   ours         every <D> and <D>@… arm: this tree's binary on the LCG prompt
#   prose, code  each corpus's arms, its @ arms included: the same binary on that corpus's ids
#   bin:<tree>   each second binary's arms
#   lcpp         every mainline arm, lcpp, lcpp<K>, lcpppp and lcpppp<U>: one binary and one token
#                stream; a sweep value only moves whole layers between the card and the host, and
#                every K reads the same non-lazy bytes of the file
#   lcppfit      every fit arm, lcppfit, lcppppfit and lcppppfit<U>: the same binary, but the fit's
#                host set is the last layers' experts and --n-cpu-moe K's the first K layers', so in
#                one block with the lcpp arms the discard would read neither and their union is nearly
#                every expert [derived]. The fit arms of a block place alike but for the partial layer
#                (n_ctx moves the KV cache by 40 KiB a position, a layer's experts are 6,142.5 MiB;
#                -ub moves the compute buffer), so the discard's set covers the others' but for that
#                layer, and a row's `timed` count shows what it missed
#   ik           every ik arm, ik, ikpp and ikpp<U>
#   lcppsrv      every server arm at the hand-set placement, lcppsrv and lcppsrvpp[<U>], whatever its ids;
#   lcppsrvfit   and at the server's fit. A server block's discard is its arm with the longest prompt, as
#                an ours block's: each server row has its own warm-up request besides
# Same-binary lever arms stay interleaved inside their block. Rotation across engines is dropped here:
# what it guards is the position bias inside a round, 0.3-0.8 % for a round's first arm (AGENTS.md,
# the ab-decode.sh rotation), while the cross-engine ratios are 1.2-3.2x (our rows in rig-log
# 2026-09-27#v41-xeng against round memfit's warm llama.cpp predictions [derived]); what it costs
# is a swap of two engines' sets through the page cache at every arm (the two host sets' union was
# about 357 GB beside a file
# cache of about 261 GB [derived, round memfit]), which is the larger term. The ratio tables still pair
# round r of one block with round r of another, now measured minutes apart; the witness blocks carry
# the card's clocks and the page cache across that gap.
# Discard. Before a block's rounds the runner runs one discard process, its row printed as `DISCARD r0
# …` between its own witness blocks and in no mean, ratio or row count, as the warm-up's is; a discard
# that fails is `FAIL r0 …` and counts in the failed list. It is the block's arm that reads the most of
# what the others will:
#   a reference block: the arm with the most token draws, at the block's largest --n-cpu-moe (the
#     `[block]` line names it when that is not the arm's own). llama-bench seeds no generator, so every
#     process draws its ids from one std::rand() stream from its start, and an arm's ids are the stream's
#     first draws. Mainline draws 3P for a pp arm (-r 2, its warm-up the whole prompt) and D + N + 3 for a
#     decode arm (-r 1: 2 in its 1-token warm-up, then D, then N + 1); ik draws 2P + 1 and D + N + 4, or
#     N + 3 at D = 0 (its warm-up prompt is 1 token) — the V4.1 branch's tools/llama-bench/llama-bench.cpp
#     :2146-2178 and :2370-2460, ik's examples/llama-bench/llama-bench.cpp:2108-2130 and :2226-2250. So the
#     discard reads every engram row the block's arms will, bar the n-grams that straddle a boundary
#     between two of its prompts (two positions a boundary, tens of 4 KB faults) [derived]. The block's
#     largest K makes the discard's host set the union of the block's.
#   an ours, corpus or bin block: the arm with the longest prompt, with that arm's variables — the LCG
#     walk and a corpus's first P ids are prefixes of each other, so it reads every shorter arm's prompt
#     ids, though not the tokens another arm generates. Our engine reads a step's engram rows ahead of
#     it (WILLNEED, then the copy), so those are not a serial term in our rows. A lever arm whose
#     variable moves the host set (a hot list) reads its own set after the discard; its `timed` count
#     shows what that cost.
# Its cost is one row of the arm a block, 22-230 s for the V4.1 rows (the `wall` column of rig-log
# 2026-09-27's two windows). Under blocks BLOOMERY_AB_WARMUP=1 (the default) is the discards — the first
# block's discard is the lease's first process, so there is no separate warm-up — and 0 skips them.
#
# Warm rows. BLOOMERY_WARM_ROWS=1 takes every counted row warm and names the one that is not
# (cold-blocks.sh has the rule): a same-id PRIME run of each of our arms right before its row, in its load,
# and one retry of a row the cold tag marks, printed first as `COLD r<r> …` — a reference arm as a second
# process, ours as a fresh load of its prime and the arm, a server arm as one more timed request; cold
# again it is `FAIL … rc=cold`. Off by default: nothing prints differently.
#
# Warm-up (under rotate; under blocks, Discard above). The lease's first process reads the model's
# pages cold — the plan's reads of the file and
# the host tier's experts through the mapping, both inside our arm's timed window — so the first row
# of a lease reads low. The runner runs the first arm once before round 1 and discards it: its row
# prints as `WARMUP r0 …`, between its own witness blocks and after the same guards, and is in no
# mean, ratio or row count; a warm-up that fails is a FAIL row as a round's arm is. It costs one
# row's wall, 73-128 s for the V4.1 rows at P = 512 and 4096 (lcg and prose) [derived: those rows'
# `wall` column], so the lease is that much longer. BLOOMERY_AB_WARMUP=0 skips it; any value but 0 or
# 1 is refused. depth-qwen3moe.sh takes the same variable with 0 as its default under rotate: its
# Qwen3 and Qwen3.6 models sit on the card, so no timed window of theirs reads the file.
#
# Contention. Before every arm the runner checks both the other card (guard_other, timing-card.sh:
# `[other-busy]`) and the CPU (guard_cpu, lease.sh: `[cpu-busy]` when builds or reference engines it
# did not start — BLOOMERY_CPU_BUSY_COMMS — sum past BLOOMERY_CPU_BUSY_PCT percent of one cpu), and
# checks the CPU again after the arm; a row that met CPU contention ends in ` [cpu-busy]`, one whose
# arm started beside a compute process on the other card in ` [other-busy]` (guard_other's
# OTHER_BUSY_TAG), and the closing summary counts both. BLOOMERY_OTHER_STRICT=1 aborts (rc 75) on
# either instead.
#
# Two cards. BLOOMERY_TIMING_CARDS=a6000+3090 (timing-card.sh has the mode) runs the arms on both cards,
# the A6000 as device 0 and the 3090 as device 1, for the separate "A6000+3090" table (AGENTS.md, user
# 2026-09-28: a model that does not fit one card; the reference on the same two cards, in the same
# lease; the 3090 at its 250 W cap; the witness counting the kernel's Xid lines). Every row's card field
# reads `A6000+3090`, so no reader puts it in the A6000 table. The profile's two-card line
# (TWO_CARD_PLACEMENT, models/deepseek41.sh, V41_PUBLIC only) gives mainline its --n-cpu-moe and -ts;
# the lcpp arms run (lcpp, lcpp<K>, lcpppp[<U>], the fit arms, whose fit places over both cards, and the
# server arms lcppsrv…, the same flags in the server's spellings), and
# ours, corpus and bin: arms under BLOOMERY_GEN_PLACE=bp; under a or gate they are refused by name before
# anything runs, with the hint to set bp (timing-card.sh's TIMING_CARDS_PLACE). After each of our arms its
# `load` record's `cards` must name the A6000 and the 3090 (records.py), or the arm is a FAIL row. An ik
# arm is refused by name (ik's -ts is a
# byte split over its own layer sizes, src/llama.cpp get_layer_sizes, which nobody has read against this
# placement; the public reference is llama.cpp). Before the lease a card that does not answer, a 3090 off
# its cap or an unpatched lease.sh refuses the run; after every arm an Xid since the last arm, a card lost
# or off its cap, or a llama-bench or llama-server that did not see both cards (its ggml_cuda_init lines)
# makes the arm a FAIL row. A compute process on either card as an arm starts is waited out (10 minutes,
# then rc 75). A dry run prints the pre-lease checks' verdict and goes on.
set -uo pipefail
# An ours arm's output is read by tools/bloomery/records.py, which owns the record kinds
# crates/gpu-gates/src/record.rs declares; the runner names kinds and fields, never a column.
RECORDS="${BASH_SOURCE[0]%/*}/../bloomery/records.py"
# An arm's place in its load, ` | slot <k>/<n>` on its row (ours_post; empty for a one-arm command line).
SLOT_COL=
# ours_parse: an ours or bin arm's output on stdin, into what its row reads: the SMOKE footer's p50,
# mean, warm and placement (P50 empty without a footer) with its depth and generated count, the `time
# step`/`time pass` walls in order (SERIES), the draft summary (D_*), the generated tokens past token
# 0 (TOKENS), and the first `time prompt` row (PP_*, empty without one). Returns 2 with FAIL_WHY set
# when records.py cannot read the output.
ours_parse() {
  local rec
  rec=$(python3 "$RECORDS" sh - P50=smoke.p50_ms MEAN=smoke.mean_ms WARMCOL=smoke.warm \
    GEN=smoke.generated DEPTH=smoke.depth PLACE_RAN=smoke.place 'SERIES=time_step|time_pass.ms*' \
    D_PROP=draft_summary.proposals D_ACC=draft_summary.accepts D_POS=draft_summary.positions \
    D_PASSES=draft_summary.passes 'D_TPS=draft_summary.tok/s(positions)' 'TOKENS=step.token*' \
    PP_N=time_prompt.n PP_MS=time_prompt.ms 'PP_TPS=time_prompt.tok/s' PP_PASSES=time_prompt.passes \
    PP_KIND=time_prompt.kind) || { FAIL_WHY="records.py did not read the output"; return 2; }
  eval "$rec"
}
# pp_col <arm kind>: an ours or bin arm's prefill column from its parsed time prompt row (ours_parse),
# into PP_COL. A bin arm's base build may print no time prompt row (PP_N empty, the column says so);
# an ours arm's binary is this tree's, so a missing row fails the arm (1, FAIL_WHY).
pp_col() {
  if [ -n "$PP_N" ]; then
    PP_COL=" | pp_tok/s $PP_TPS (n=$PP_N, passes=$PP_PASSES)"
    [ "$PP_KIND" = steps ] || PP_COL="${PP_COL%)}, kind=$PP_KIND)"
  elif [ "$1" = bin ]; then
    PP_COL=" | pp_tok/s ? (the binary prints no time prompt row)"
  else
    FAIL_WHY="no time prompt row"
    return 1
  fi
}
# ours_row <arm kind> <label> <depth> <round> <output> <wall s>: an ours or bin arm's lines — the
# records it echoes, then its row — from its output; into TPS_MEAN, TPS_P50 and DRAFT for the sums.
# The majflt column and the cold tag come from MAJ_WHOLE and MAJ_TIMED (ours_arm; empty under
# --parse: no column). An output with no row, or one that ran another placement than PLACE, prints
# nothing and returns non-zero with FAIL_WHY set; 3 is a warm-rows retry that was cold again, its FAIL row
# printed.
ours_row() {
  local kind=$1 label=$2 dep=$3 r=$4 out=$5 wall=$6 h10 t10 uniq_tok win
  ours_parse <<< "$out" || return
  [ -n "$P50" ] || { FAIL_WHY="no SMOKE line"; return 1; }
  if [ "$PLACE" != - ] && [ -n "$PLACE_RAN" ] && [ "$PLACE_RAN" != "$PLACE" ]; then
    FAIL_WHY="its SMOKE footer names place=$PLACE_RAN; the runner passed --place $PLACE"
    return 1
  fi
  pp_col "$kind" || return
  echo "$out" | grep -E '^(plan|load|capture|fed|prefill|stat prefill|stat summary|time prompt|call|arm|residency host) '
  # `time step` rows are one position each; under BLOOMERY_DRAFT the rows are `time pass … positions=1|2`
  # and the `draft summary` line carries the positions-per-second rate the verdict reads.
  DRAFT=
  [ -z "$D_PROP" ] || DRAFT="p=$D_PROP/$D_PASSES q=$D_ACC/$D_PROP positions=$D_POS tok/s(positions)=$D_TPS"
  h10=$(echo "$SERIES" | head -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  t10=$(echo "$SERIES" | tail -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  uniq_tok=$(printf '%s' "$TOKENS" | sort -u | grep -c .)
  TPS_MEAN=$(awk -v m="$MEAN" 'BEGIN{printf "%.2f", 1e3/m}')
  TPS_P50=$(awk -v p="$P50" 'BEGIN{printf "%.2f", 1e3/p}')
  # The timed window: the prompt's wall and the N generated steps at the mean.
  MAJ_COL='' COLD_TAG=''
  if [ -n "${MAJ_WHOLE:-}" ]; then
    win=$(awk -v p="${PP_MS:-0}" -v n="$N" -v m="$MEAN" 'BEGIN { printf "%.4f", (p + n * m) / 1e3 }')
    cold_check "${MAJ_TIMED:-$MAJ_WHOLE}" "$win"
    MAJ_COL=" | majflt $MAJ_WHOLE (timed ${MAJ_TIMED:-? (no fed line)}; ≤ $MAJ_BOUND % of W ${win} s)"
    # 3: a retry cold again, whose FAIL row cold_verdict printed (cold-blocks.sh).
    cold_verdict "$r" "$label" "d=$dep" "${MAJ_TIMED:-$MAJ_WHOLE}" "$win" || return 3
  fi
  echo "$ROW_TAG r$r $label d=$dep n=$N | tok/s(mean) $TPS_MEAN @ n=$N, depth $dep, $CARD_NAME | place ${PLACE_RAN:-$PLACE} | p50 $P50 ms | mean $MEAN ms | tok/s(p50) $TPS_P50 | warm ${WARMCOL:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok${DRAFT:+ | draft $DRAFT}$PP_COL$MAJ_COL$SLOT_COL | wall ${wall}s$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
}
# The residency seed condition and curve (the header's Residency). RES_WINDOW: the passes a curve window
# holds.
RES_WINDOW=16
# res_value <index>: the BLOOMERY_RESIDENCY arm <index> runs with — its own @ list's, else the runner's.
res_value() {
  local e v=${BLOOMERY_RESIDENCY:-}
  local -a kv=()
  [ -z "${A_ENV[$1]}" ] || IFS=, read -r -a kv <<< "${A_ENV[$1]}"
  for e in "${kv[@]}"; do
    case $e in BLOOMERY_RESIDENCY=*) v=${e#*=} ;; esac
  done
  echo "$v"
}
# res_on <index>: whether arm <index> runs with adaptive residency on.
res_on() {
  local v
  v=$(res_value "$1")
  [ -n "$v" ] && [ "$v" != off ]
}
# res_read <output>: the residency records and timed passes of an arm's output, by kind and field, into
# RP_PASS, RP_BOUNDARY, RP_MADE (every `residency pass`, one a line), RR_ALL (every `residency reset`'s
# diff) and RR_TAIL (the `residency reset` lines after the last record of another kind: the report of a
# clear that follows the arm), CV_STEP (the `time step` walls), CV_PASS and CV_POS (the `time pass`
# walls and positions). Returns 2 with RES_WHY set when records.py cannot read the output.
res_read() {
  local rec
  RES_WHY="records.py did not read the output"
  rec=$(python3 "$RECORDS" sh - 'RP_PASS=residency_pass.pass*' 'RP_BOUNDARY=residency_pass.boundary*' \
    'RP_MADE=residency_pass.made*' 'RR_ALL=residency_reset.diff*' 'CV_STEP=time_step.ms*' \
    'CV_PASS=time_pass.ms*' 'CV_POS=time_pass.positions*' <<< "$1") || return 2
  RR_TAIL=$(python3 "$RECORDS" tail - residency_reset <<< "$1") || return 2
  RES_WHY=''
  eval "$rec"
}
# res_seed <index> <slot>: the seed condition of the output res_read read, arm <index>'s at its slot in its
# load (1 for a process of its own), the previous arm's output in LG_PREV_OUT when the slot is past 1.
# Returns 1 with RES_WHY naming every clause that failed; on success RES_SEED says what held.
res_seed() {
  local i=$1 slot=$2 first b0 n_all n_tail rec
  local -a why=()
  RES_WHY='' RES_SEED=''
  n_all=$(grep -c . <<< "$RR_ALL")
  n_tail=$(grep -c . <<< "$RR_TAIL")
  if ! res_on "$i"; then
    [ -z "$RP_PASS" ] || why+=("$(grep -c . <<< "$RP_PASS") residency pass record(s) in an arm with BLOOMERY_RESIDENCY off: the lever leaked into an off arm")
    [ "$n_all" = 0 ] || why+=("$n_all residency reset record(s) in an arm with BLOOMERY_RESIDENCY off: the lever leaked into an off arm")
    if [ ${#why[@]} -gt 0 ]; then
      RES_WHY=$(printf '%s; ' "${why[@]}")
      RES_WHY=${RES_WHY%; }
      return 1
    fi
    return 0
  fi
  if [ -z "$RP_PASS" ]; then
    why+=("no residency pass record: BLOOMERY_RESIDENCY=$(res_value "$i") ran no residency machine")
  else
    first=$(head -n 1 <<< "$RP_PASS") b0=$(head -n 1 <<< "$RP_BOUNDARY")
    if [ "$first" != none ] || [ "$b0" != 0 ]; then
      why+=("its first residency pass is pass=$first boundary=$b0, not the seed's none/0 ($([ "$slot" = 1 ] && echo "slot 1: the load's first boundary" || echo "slot $slot: the first boundary after its clear"))")
    fi
  fi
  [ $((n_all - n_tail)) = 0 ] || why+=("$((n_all - n_tail)) residency reset record(s) among its own lines: a reset inside its timed work")
  RES_SEED="first pass ${first:-?}/${b0:-?}"
  if [ "$slot" != 1 ]; then
    local tail_prev R_DIFF='' R_DROP='' r_n
    RES_WHY="records.py did not read the previous arm's output"
    tail_prev=$(python3 "$RECORDS" tail - residency_reset <<< "$LG_PREV_OUT") || return 1
    rec=$(python3 "$RECORDS" sh - 'R_DIFF=residency_reset.diff*' 'R_DROP=residency_reset.dropped_bytes*' <<< "$tail_prev") || return 1
    RES_WHY=''
    eval "$rec"
    r_n=$(grep -c . <<< "$R_DIFF")
    if [ "$r_n" != 1 ]; then
      why+=("$r_n residency reset record(s) between the previous arm and its arm line, want 1: its map did not start from the seed")
    else
      [ "$R_DIFF" = 0 ] || why+=("the reset before it left diff=$R_DIFF map entries off the seed, want 0")
      [ "$R_DROP" = "$RESIDENCY_RESET_DROPPED_BYTES" ] || why+=("the reset before it released dropped_bytes=$R_DROP, the profile's RESIDENCY_RESET_DROPPED_BYTES is $RESIDENCY_RESET_DROPPED_BYTES")
    fi
    RES_SEED+=", reset diff=${R_DIFF:-?} dropped_bytes=${R_DROP:-?}"
  fi
  if [ ${#why[@]} -gt 0 ]; then
    RES_WHY=$(printf '%s; ' "${why[@]}")
    RES_WHY=${RES_WHY%; }
    return 1
  fi
}
# res_curve <label> <round> <depth> <slot>: the curve line of an output res_read read (the header's
# Residency), when it holds a `residency pass` record.
res_curve() {
  local w
  [ -n "$RP_PASS" ] || return 0
  if [ -n "$CV_PASS" ]; then
    w=$(paste -d' ' <(echo "$CV_PASS") <(echo "$CV_POS"))
  else
    w=$(awk 'NF { print $1, 1 }' <<< "$CV_STEP")
  fi
  w=$(awk -v n="$RES_WINDOW" 'NF {
    ms += $1; pos += $2; c++
    if (c == n) { out = out sep sprintf("%.2f", 1e3 * pos / ms); sep = ","; k++; ms = pos = c = 0 }
  } END {
    if (c) { out = out sep sprintf("%.2f", 1e3 * pos / ms); k++ }
    printf "windows=%d tok/s=%s", k, (k ? out : "-")
  }' <<< "$w")
  echo "residency curve $1 r$2 d=$3 $w flips=$(awk '{ s += $1 } END { print s + 0 }' <<< "$RP_MADE") | $ROW_TAG slot $4 | seed ${RES_SEED:-not checked}"
}
# The cold tag's constants and cold_check, the fault counter and majflt_mark, ROW_TAG and counted, the
# order and the blocks' planner and loops: shared with depth-qwen3moe.sh.
# shellcheck source=tools/ref/cold-blocks.sh
source "${BASH_SOURCE[0]%/*}/cold-blocks.sh" || exit 2
# `--parse FILE`: ours_row over a saved output (`-` for stdin) — its depth and N the SMOKE footer's,
# the runner's context (round, card, contention) `-` — and nothing loaded, timed or leased.
if [ "${1:-}" = --parse ]; then
  [ $# -eq 2 ] || { echo "usage: depth-ds41.sh --parse FILE" >&2; exit 64; }
  out=$(cat -- "$2") || exit 2
  ours_parse <<< "$out" || { echo "$2: $FAIL_WHY" >&2; exit 2; }
  ROW_TAG=ROW N=${GEN:--} CARD_NAME=- CPU_BUSY_TAG='' OTHER_BUSY_TAG='' PLACE=-
  ours_row ours ours "${DEPTH:--}" - "$out" - || { echo "$2: $FAIL_WHY" >&2; exit 1; }
  # No arm environment here: the curve without the seed verdict.
  res_read "$out" || { echo "$2: $RES_WHY" >&2; exit 2; }
  res_curve ours - "${DEPTH:--}" -
  exit 0
fi
# The profile (MODEL, IK, IKBIN, IK_GPU_FLAGS, IK_GPU_ENV, LCPP, LCPPBIN, LCPP_GPU_FLAGS,
# LCPP_NCMOE); tools/box.sh exports its MODEL to our binary as BLOOMERY_REF_MODEL, so the three
# engines open one file.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
# The fit arms' flags, probe and column (lcppfit, lcppppfit[<U>]).
# shellcheck source=tools/ref/lcpp-fit.sh
source "${BASH_SOURCE[0]%/*}/lcpp-fit.sh" || exit 2
# The server arms' start, requests, flags and stop (lcppsrv…).
# shellcheck source=tools/ref/lcpp-warm.sh
source "${BASH_SOURCE[0]%/*}/lcpp-warm.sh" || exit 2
[ "$MODEL_NAME" = deepseek41 ] || {
  echo "depth-ds41.sh: the profile is $MODEL_NAME — pick deepseek41 on the Mac side (BLOOMERY_MODEL=deepseek41)" >&2
  exit 64
}
N=${BLOOMERY_DECODE_N:-96}
WARM=${BLOOMERY_GEN_WARM:-}
ROUNDS=${BLOOMERY_AB_ROUNDS:-3}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_ds41}
BOUND=${BLOOMERY_ARM_BOUND:-900}
DRY=${BLOOMERY_DRY:-}
AB_WARMUP=${BLOOMERY_AB_WARMUP:-1}
case $AB_WARMUP in
  0 | 1) ;;
  *) echo "depth-ds41.sh: BLOOMERY_AB_WARMUP is 1 (the default: one discarded run of the first arm) or 0, got '$AB_WARMUP'" >&2; exit 64 ;;
esac
ab_order depth-ds41.sh
warm_rows_init depth-ds41.sh
PLACE=${BLOOMERY_GEN_PLACE:-a}
case $PLACE in
  a | gate | bp) ;;
  *) echo "depth-ds41.sh: BLOOMERY_GEN_PLACE is a (plan (a), on the A6000; the default), gate (the gate plan, on the 3090) or bp (plan (b′), on both cards), got '$PLACE'" >&2; exit 64 ;;
esac
# The preheat's default follows the order (the header's Preheat): the blocks' discards read the host set.
PREHEAT_DEFAULT=1
[ "$ORDER" = rotate ] || PREHEAT_DEFAULT=0
PREHEAT=${BLOOMERY_PREHEAT:-$PREHEAT_DEFAULT}
case $PREHEAT in
  0 | 1) ;;
  *) echo "depth-ds41.sh: BLOOMERY_PREHEAT is 1 (read each reference arm's host set before it; the default under BLOOMERY_AB_ORDER=rotate) or 0 (the default under blocks), got '$PREHEAT'" >&2; exit 64 ;;
esac
GGUF_RANGES="${BASH_SOURCE[0]%/*}/gguf-ranges.py"
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6)
# ours: an arm runs this tree's generate_ds41; gen: an arm runs a generate_ds41 (ours, corpus or bin); srv:
# a server arm.
ours=0 gen=0 ik=0 lcpp=0 lcppfit=0 srv=0
# Per arm, by its index in ARMS: the kind (ours, corpus, ref, srv or bin), the depth, the row label, the
# reference engine (ref and srv; the corpus name for a corpus arm), the binary (ours, corpus and bin), the
# NAME=VALUE list (comma-separated) and a server arm's ids (lcg, or the corpus's name).
A_KIND=() A_DEP=() A_LABEL=() A_ENG=() A_BIN=() A_ENV=() A_IDS=()
# A corpus arm's prompt, the ids as generate_ds41 --tokens takes them (empty for every other arm).
A_TOK=()
# The corpus arms' names: corpus-<name>.ids under $BLOOMERY_DATA/engram, one id per line; each file's
# id count is read once, into CORPUS_N_<name>, when an arm names it.
CORPORA="prose code"
corpus_file() { echo "${BLOOMERY_DATA:-}/engram/corpus-$1.ids"; }
# A prefill arm's engine (ikpp[<U>], lcpppp[<U>], lcppppfit[<U>]), and its ubatch lever U (empty: the
# default).
pp_eng() { case $1 in ikpp* | lcpppp*) return 0 ;; *) return 1 ;; esac; }
pp_ub() { local u=${1#ikpp}; u=${u#lcpppp}; echo "${u#fit}"; }
arm_usage() {
  echo "depth-ds41.sh: arm '$1' is <D>, <D>@NAME=VALUE[,NAME=VALUE...], prose:<P>[@NAME=VALUE,...], code:<P>[@NAME=VALUE,...], ik:<D>, lcpp:<D>, lcpp<K>:<D>, lcppfit:<D>, ikpp[<U>]:<P>, lcpppp[<U>]:<P>, lcppppfit[<U>]:<P>, lcppsrv[fit]:[prose:|code:]<D>, lcppsrvpp[fit][<U>]:[prose:|code:]<P> or bin:<path>:<D>" >&2
  exit 64
}
# arm_envs_ok <arm> <NAME=VALUE list>: the list is one or more NAME=VALUE, no spaces or commas in a value.
arm_envs_ok() {
  local -a kv
  IFS=, read -r -a kv <<< "$2"
  [ ${#kv[@]} -gt 0 ] || arm_usage "$1"
  for e in "${kv[@]}"; do
    [[ $e =~ ^[A-Za-z_][A-Za-z0-9_]*=[^[:space:],]+$ ]] || arm_usage "$1"
  done
}
# corpus_check <arm> <name> <P>: the file's id count read once; P outside 1..count is refused.
corpus_check() {
  local file var
  file=$(corpus_file "$2") var=CORPUS_N_$2
  if [ -z "${!var:-}" ]; then
    [ -r "$file" ] || { echo "depth-ds41.sh: arm '$1': no $2 prompt file at $file (BLOOMERY_DATA)" >&2; exit 2; }
    printf -v "$var" '%s' "$(($(wc -l < "$file")))"
  fi
  if [ "$3" -lt 1 ] || [ "$3" -gt "${!var}" ]; then
    echo "depth-ds41.sh: arm '$1': a $2 prompt of $3 ids; $file holds ${!var} (1..${!var})" >&2
    exit 64
  fi
}
for a in "${ARMS[@]}"; do
  kind=ref eng=${a%%:*} dep=${a#*:} label='' bin='' envs='' tok='' ids=''
  if srv_eng "$eng"; then
    kind=srv srv=1 ids=lcg label=$eng
    case $dep in
      prose:* | code:*)
        ids=${dep%%:*} dep=${dep#*:}
        case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
        corpus_check "$a" "$ids" "$dep"
        tok=$(head -n "$dep" "$(corpus_file "$ids")" | paste -sd, -)
        label=$eng@$ids
        ;;
    esac
    case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
    [ "$dep" -ge 1 ] || { echo "depth-ds41.sh: arm '$a': a server arm sends at least one id" >&2; exit 64; }
    srv_check_arm "$a" || { echo "depth-ds41.sh: arm '$a': $SRV_WHY" >&2; exit 64; }
    A_KIND+=("$kind") A_DEP+=("$dep") A_LABEL+=("$label") A_ENG+=("$eng") A_BIN+=('') A_ENV+=('') A_TOK+=("$tok") A_IDS+=("$ids")
    continue
  fi
  case $a in
    prose:* | code:*)
      kind=corpus bin=$BIN label=$eng
      case $dep in *@*) envs=${dep#*@} dep=${dep%%@*} label=$eng@$envs && arm_envs_ok "$a" "$envs" ;; esac
      case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
      corpus_check "$a" "$eng" "$dep"
      tok=$(head -n "$dep" "$(corpus_file "$eng")" | paste -sd, -)
      ours=1 gen=1
      ;;
    bin:*)
      kind=bin eng=bin bin=${a#bin:}
      dep=${bin##*:} bin=${bin%:*}
      case $bin in /*) ;; *) arm_usage "$a" ;; esac
      tree=${bin%/target/*}
      [ "$tree" != "$bin" ] || tree=${bin%/*}
      label=bin:${tree##*/}
      gen=1
      ;;
    *@*)
      kind=ours eng=ours dep=${a%%@*} envs=${a#*@} bin=$BIN label=ours@$envs
      arm_envs_ok "$a" "$envs"
      ours=1 gen=1
      ;;
    *:*)
      case $eng in
        ik) ik=1 ;;
        lcpp | lcpp[0-9] | lcpp[0-9][0-9]) lcpp=1 ;;
        ikpp | ikpp[1-9]*) ik=1 ;;
        lcpppp | lcpppp[1-9]*) lcpp=1 ;;
        lcppfit | lcppppfit | lcppppfit[1-9]*)
          lcpp=1 lcppfit=1
          lcpp_fit_flags "$LCPP_GPU_FLAGS" || { echo "depth-ds41.sh: arm '$a': $FIT_WHY" >&2; exit 64; }
          ;;
        *) arm_usage "$a" ;;
      esac
      label=$eng
      ;;
    *) kind=ours eng=ours dep=$a bin=$BIN label=ours ours=1 gen=1 ;;
  esac
  case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
  if pp_eng "$eng"; then
    ub=$(pp_ub "$eng")
    case $ub in *[!0-9]*) arm_usage "$a" ;; esac
    [ "$dep" -ge 1 ] || { echo "depth-ds41.sh: arm '$a': a prompt of 0 ids has no prefill to time" >&2; exit 64; }
    if [ -n "$ub" ]; then
      case $eng in ikpp*) flags=$IK_GPU_FLAGS ;; *) flags=$LCPP_GPU_FLAGS ;; esac
      case " $flags " in
        *" -ub "* | *" --ubatch-size "* | *" -b "* | *" --batch-size "*)
          echo "depth-ds41.sh: arm '$a': the profile's flags already set the batch sizes ($flags); the ubatch lever would add a second value" >&2
          exit 64
          ;;
      esac
    fi
  fi
  A_KIND+=("$kind") A_DEP+=("$dep") A_LABEL+=("$label") A_ENG+=("$eng") A_BIN+=("$bin") A_ENV+=("$envs") A_TOK+=("$tok") A_IDS+=("$ids")
done
# The load keys (the header's Loads): an ours or corpus arm's binary, placement and variables, the solo
# marker left out; `|solo` on an arm that runs alone. A reference or bin: arm has none.
# shellcheck source=tools/ref/load-groups.sh
source "${BASH_SOURCE[0]%/*}/load-groups.sh" || exit 2
for i in "${!ARMS[@]}"; do
  case ${A_KIND[$i]} in
    ours | corpus)
      LG_KEY[i]="$BIN|place=$PLACE|$(lg_env_key "$(lg_strip_solo "${A_ENV[$i]}")")"
      if lg_is_solo "${A_ENV[$i]}" || [[ ,${A_ENV[$i]}, =~ ,BLOOMERY_(DRAFT|CHECK_FINITE)= ]]; then
        LG_KEY[i]+='|solo'
      fi
      ;;
    *) LG_KEY[i]= ;;
  esac
done
# The residency arms (the header's Residency), and the profile's reset bytes their later slots are held to:
# refused before the lease when the profile gives none.
RES_ARMS=()
for i in "${!ARMS[@]}"; do
  case ${A_KIND[$i]} in
    ours | corpus | bin) if res_on "$i"; then RES_ARMS+=("${ARMS[$i]}"); fi ;;
  esac
done
if [ ${#RES_ARMS[@]} -gt 0 ]; then
  case ${RESIDENCY_RESET_DROPPED_BYTES:-} in
    '' | *[!0-9]*)
      echo "depth-ds41.sh: arms ${RES_ARMS[*]} run with BLOOMERY_RESIDENCY on, and the profile $MODEL_NAME gives no RESIDENCY_RESET_DROPPED_BYTES (the host bytes a residency reset releases, which every later slot's reset must carry; got '${RESIDENCY_RESET_DROPPED_BYTES:-}'): set it in tools/ref/models/$MODEL_NAME.sh" >&2
      exit 64
      ;;
  esac
fi
# res_config: the residency check's [config] and [dry] line.
res_config() {
  if [ ${#RES_ARMS[@]} -eq 0 ]; then
    echo "residency: no arm runs with BLOOMERY_RESIDENCY on; an arm that prints a residency pass or reset record is a FAIL rc=residency row (the lever leaked)"
  else
    echo "residency: ${RES_ARMS[*]} run with BLOOMERY_RESIDENCY on: each starts from the seed (its first residency pass none/0, and at slot k > 1 one residency reset before its arm line with diff=0 and dropped_bytes=$RESIDENCY_RESET_DROPPED_BYTES, the profile's) or is a FAIL rc=residency row; a residency curve line after each row, windows of $RES_WINDOW passes"
  fi
}
# The card pin, the card's witness lines, the other-card guard and the binary's freshness; this runner
# has the two-card mode (the header's Two cards).
TIMING_CARDS_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
timing_cards_mode || exit $?
TC_ARMS=()
for i in "${!ARMS[@]}"; do TC_ARMS+=("${ARMS[$i]}" "${A_KIND[$i]}" "${A_ENG[$i]}"); done
# shellcheck disable=SC2034 # read by timing_cards_arms (timing-card.sh)
TIMING_CARDS_PLACE=bp TIMING_CARDS_PLACE_RAN=$PLACE
timing_cards_arms "$BIN" "${TC_ARMS[@]}" || exit $?
# The placement's card must be the timing card, the only one the arms see: plan (a) loads on the card
# named A6000, the gate plan on the one named 3090 (workstation::plan_a, plan_gate).
if [ "$gen" = 1 ]; then
  if [ "$PLACE" = a ] && [ "$TIMING_GPU" = "$GPU_3090" ]; then
    echo "depth-ds41.sh: BLOOMERY_GEN_PLACE=a is plan (a), which loads on the A6000, and the timing card is the 3090 (BLOOMERY_TIMING_GPU=$TIMING_GPU): generate_ds41 would refuse every arm; set BLOOMERY_GEN_PLACE=gate" >&2
    exit 64
  fi
  if [ "$PLACE" = bp ] && [ -z "$TIMING_CARDS" ]; then
    echo "depth-ds41.sh: BLOOMERY_GEN_PLACE=bp is plan (b′), which loads on both cards (the A6000 and its 3090 expert tier); it runs in the two-card mode, BLOOMERY_TIMING_CARDS=a6000+3090" >&2
    exit 64
  fi
  if [ "$PLACE" = gate ] && [ "$TIMING_GPU" != "$GPU_3090" ]; then
    echo "depth-ds41.sh: BLOOMERY_GEN_PLACE=gate is the gate plan, which loads on the 3090, and the timing card is $TIMING_GPU, not the 3090 ($GPU_3090): name the 3090 in BLOOMERY_TIMING_GPU, or leave BLOOMERY_GEN_PLACE at a" >&2
    exit 64
  fi
fi
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# The t quantiles of the ratio intervals, df 1..ROUNDS: tools/ref/tdist.py, the table gpu-ab.py and
# card.py read.
T975=$(python3 "${BASH_SOURCE[0]%/*}/tdist.py" "$ROUNDS") || {
  rc=$?
  echo "depth-ds41.sh: tools/ref/tdist.py gave no t quantiles for ROUNDS=$ROUNDS (rc $rc; ROUNDS is a positive integer)" >&2
  exit "$rc"
}
# Each engine's binary matters only to its own arms: a reference-only run neither builds ours (the
# recipe skips the build) nor reads it, so its freshness is not asked. A dry run asks nothing of
# our binary: it prints the command line it would run.
if [ "$ours" = 1 ] && [ -z "$DRY" ]; then assert_fresh_binary "$BIN" || exit $?; fi
# A bin:<path> arm's binary is checked where its tree line is taken, below.
if [ "$ik" = 1 ]; then [ -x "$IKBIN" ] || { echo "depth-ds41.sh: no llama-bench at $IKBIN" >&2; exit 2; }; fi
if [ "$lcpp" = 1 ]; then [ -x "$LCPPBIN" ] || { echo "depth-ds41.sh: no llama-bench at $LCPPBIN" >&2; exit 2; }; fi
if [ "$lcppfit" = 1 ]; then
  # shellcheck disable=SC2153 # LCPP is the profile's, as below
  lcpp_fit_probe "$LCPPBIN" || { echo "depth-ds41.sh: the lcppfit/lcppppfit arms need llama-bench's fit: $FIT_WHY (tree $LCPP)" >&2; exit 64; }
fi
# A server arm needs the tree's llama-server, every flag it passes in that server's --help (run with no
# card), and curl; the server is a reference engine to the CPU guard.
SRVBIN=
if [ "$srv" = 1 ]; then
  srv_preflight depth-ds41.sh
  CPU_BUSY_COMMS="$CPU_BUSY_COMMS llama-server"
fi
if [ -n "$TIMING_CARDS" ]; then
  CARD_NAME=$TIMING_CARDS_NAME
else
  CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
fi
# A binary's sha256 and its tree's HEAD and dirty count, once. GIT_OPTIONAL_LOCKS=0 keeps `git
# status` from rewriting the index of a tree this root process does not own. A tree box.sh synced
# has no .git (the rsync leaves it out): when that tree is the one this runner stands in, its commit
# is box.sh's BLOOMERY_GIT_COMMIT (`-dirty` when the Mac tree held uncommitted changes), printed as
# `head=<commit>(box.sh) dirty_files=?`; any other tree without git prints `head=? dirty_files=?`.
tree_line() {
  local bin=$1 tree=$2 sha head dirty
  sha=$(sha256sum "$bin" | cut -c1-12)
  if head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short=9 HEAD 2> /dev/null); then
    dirty=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
  else
    head='?' dirty='?'
    if [ -n "${BLOOMERY_GIT_COMMIT:-}" ] && [ "$(cd "$tree" 2> /dev/null && pwd -P)" = "$(pwd -P)" ]; then
      head="$BLOOMERY_GIT_COMMIT(box.sh)"
    fi
  fi
  echo "$bin sha256=$sha head=$head dirty_files=$dirty"
}
IK_LINE='' LCPP_LINE='' BIN_LINES=()
[ "$ik" = 0 ] || IK_LINE=$(tree_line "$IKBIN" "$IK")
# shellcheck disable=SC2153 # LCPP is the profile's, which shellcheck does not follow
[ "$lcpp" = 0 ] || LCPP_LINE=$(tree_line "$LCPPBIN" "$LCPP")
# The server's launcher is a few KB; what it runs is the tree's libllama-server-impl.so beside it.
SRV_LINE=
[ "$srv" = 0 ] || SRV_LINE=$(srv_tree "$(tree_line "$SRVBIN" "$LCPP")")
for i in "${!ARMS[@]}"; do
  [ "${A_KIND[$i]}" = bin ] || continue
  b=${A_BIN[$i]}
  [ -x "$b" ] || { echo "depth-ds41.sh: arm '${ARMS[$i]}': no binary at $b" >&2; exit 2; }
  t=${b%/target/*}
  [ "$t" != "$b" ] || t=${b%/*}
  BIN_LINES+=("${A_LABEL[$i]}: $(tree_line "$b" "$t")")
done
ref_witness() {
  [ -z "$IK_LINE" ] || echo "    ik: $IK_LINE"
  [ -z "$LCPP_LINE" ] || echo "    lcpp: $LCPP_LINE"
  [ -z "$SRV_LINE" ] || echo "    lcppsrv: $SRV_LINE"
  [ ${#BIN_LINES[@]} -eq 0 ] || printf '    %s\n' "${BIN_LINES[@]}"
}

# The witness before and after every row: the timing card's lines (with our binary), the busiest
# processes, the page cache and the major fault count.
WITNESS=(head-open indent card busiest model mem pgmajfault)

# ref_cmd <engine> <depth or prompt length>: the reference arm's binary, arguments and row label,
# into REF_ENV, REF_BIN, REF_ARGS and REF_LABEL, and for a prefill arm the batch sizes its row names
# into REF_BATCH. The flags are word-split on purpose: the profile keeps them as one string. An
# lcpp<K> engine is LCPP_GPU_FLAGS with its --n-cpu-moe value replaced by K. A prefill arm is its
# decode twin's binary, flags and environment with -p P -n 0 in place of the decode test. REF_K, when
# set, is the --n-cpu-moe every engine runs at instead (a block's discard, the header's Discard). An
# lcpp arm also gets --progress, and REF_MARK is the ERE of the progress line its measured window
# starts at (the header's Cold tag); REF_MARK is empty for ik, whose llama-bench has no --progress. A
# fit arm's flags are lcpp_fit_flags' rewrite, taken after REF_K (which would add --n-cpu-moe back), and
# REF_FIT says so for the dry run.
REF_K=
ref_cmd() {
  local eng=$1 dep=$2 flags ub reps=(-r 1)
  REF_ENV=() REF_BATCH='' REF_MARK='' REF_FIT=''
  case $eng in
    ik | ikpp*)
      # shellcheck disable=SC2206
      REF_ENV=($IK_GPU_ENV)
      REF_BIN=$IKBIN flags=$IK_GPU_FLAGS
      ;;
    *)
      REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS
      case $eng in
        lcpp[0-9]*) flags=$(with_ncmoe "$flags" "${eng#lcpp}") ;;
      esac
      ;;
  esac
  [ -z "$REF_K" ] || flags=$(with_ncmoe "$flags" "$REF_K")
  if lcpp_fit_eng "$eng"; then
    lcpp_fit_flags "$flags"
    flags=$FIT_FLAGS
    REF_FIT="placement: llama-bench's fit at -fitt $LCPP_FIT_TARGET MiB (dropped: $FIT_DROPPED), -v for the fit column"
  fi
  case $eng in
    ikpp* | lcpppp*)
      ub=$(pp_ub "$eng")
      REF_ARGS=(-p "$dep" -n 0) REF_LABEL="pp$dep |" REF_BATCH="ub 512 b 2048 (llama-bench defaults)"
      reps=(-r 2 -o json)
      if [ -n "$ub" ]; then
        REF_ARGS+=(-ub "$ub" -b "$((ub > 2048 ? ub : 2048))")
        REF_BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
      fi
      ;;
    ik) if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n 0 -gp "$dep,$N"); REF_LABEL="tg$N@pp$dep |"; fi ;;
    *) if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n "$N" -d "$dep"); REF_LABEL="tg$N @ d$dep |"; fi ;;
  esac
  # The last repetition's timed test: its prompt for a pp arm, its generation for a decode arm.
  case $eng in
    ik | ikpp*) ;;
    lcpppp*) REF_MARK=": prompt run ${reps[1]}/${reps[1]}\$" flags="$flags --progress" ;;
    *) REF_MARK=": generation run ${reps[1]}/${reps[1]}\$" flags="$flags --progress" ;;
  esac
  # shellcheck disable=SC2206
  REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" "${reps[@]}" $flags)
}

# The preheat's plan, before the lease: per --n-cpu-moe K a reference arm names, the ranges file
# PH_DIR/k<K>.tsv and gguf-ranges.py's `host` line (PH_LINE_<K>_<ngl>). PH_RATE is the read rate a dry
# run prices a cold preheat at: GB/s, the NVMe populate rate measured on this box (the V4.1 oracle dump
# read 474.18 GB through the mapping at 1.33 GB/s, rig-log 2026-09-23#v41-oracle-dump); a sequential
# pread of the /models drive (Phison E18, PCIe 4.0 x4) has not been measured, so it is the slow end.
PH_RATE=1.33
PH_DIR=
# ph_k <arm>: the host set's K (--n-cpu-moe, 0 without one) and -ngl from REF_ARGS (ref_cmd first),
# into PHK and PHNGL. Flags whose host set the rule does not model are refused (64).
ph_k() {
  local w prev='' k=0 ngl=''
  for w in "${REF_ARGS[@]}"; do
    case $w in
      -ot | --override-tensor | --override-tensor=* | -cmoe | --cpu-moe | -fitt | --fit-target)
        echo "depth-ds41.sh: arm '$1': its flags carry $w, a host set the preheat does not model; set BLOOMERY_PREHEAT=0 to run it unpreheated" >&2
        exit 64
        ;;
    esac
    case $prev in
      --n-cpu-moe | -ncmoe) k=$w ;;
      -ngl | --n-gpu-layers | --gpu-layers) ngl=$w ;;
    esac
    prev=$w
  done
  case $k in '' | *[!0-9]*) echo "depth-ds41.sh: arm '$1': --n-cpu-moe '$k' is not one layer count (a list runs several placements in one llama-bench)" >&2; exit 64 ;; esac
  case $ngl in *[!0-9]*) echo "depth-ds41.sh: arm '$1': -ngl '$ngl' is not one layer count" >&2; exit 64 ;; esac
  PHK=$k PHNGL=$ngl
}
if [ "$PREHEAT" = 1 ] && [ "$ik$lcpp" != 00 ]; then
  PH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/depth-ds41-preheat.XXXXXX") || exit 2
  trap 'rm -rf "$PH_DIR"' EXIT
  for i in "${!ARMS[@]}"; do
    [ "${A_KIND[$i]}" = ref ] || continue
    ref_cmd "${A_ENG[$i]}" "${A_DEP[$i]}"
    ph_k "${ARMS[$i]}"
    var=PH_LINE_${PHK}_${PHNGL:-none}
    [ -z "${!var:-}" ] || continue
    line=$(python3 "$GGUF_RANGES" host "$MODEL" --n-cpu-moe "$PHK" ${PHNGL:+--ngl "$PHNGL"} --out "$PH_DIR/k$PHK.tsv") || {
      rc=$?
      echo "depth-ds41.sh: arm '${ARMS[$i]}': no preheat ranges for K=$PHK (tools/ref/gguf-ranges.py rc $rc)" >&2
      exit "$rc"
    }
    printf -v "$var" '%s' "$line"
  done
fi
# ph_bytes <K> <ngl>: the host set's bytes from its `host` line.
ph_bytes() {
  local var=PH_LINE_${1}_${2:-none} w
  for w in ${!var}; do case $w in bytes=*) echo "${w#bytes=}" ;; esac; done
}

# The blocks (the header's Order and Discard), fixed before the lease; none under rotate. Per block b:
# BLK_KEY[b] its name, BLK_ARMS[b] its arms' indices in ARMS (space-separated, in the order given),
# BLK_DISC[b] the index of the arm its discard runs, BLK_K[b] the --n-cpu-moe that discard runs at when
# it is not the arm's own (empty otherwise), BLK_WHY[b] what the choice rests on.
# arm_block <i>: the block arm <i> belongs to.
arm_block() {
  case ${A_KIND[$1]} in
    ref) case ${A_ENG[$1]} in ik | ikpp*) echo ik ;; lcppfit | lcppppfit*) echo lcppfit ;; *) echo lcpp ;; esac ;;
    srv) case ${A_ENG[$1]} in *fit*) echo lcppsrvfit ;; *) echo lcppsrv ;; esac ;;
    ours) echo ours ;;
    corpus) echo "${A_ENG[$1]}" ;;
    *) echo "${A_LABEL[$1]}" ;;
  esac
}
# arm_draws <i>: how many token ids arm <i>'s process takes — a reference arm's std::rand() draws, an
# ours, corpus, bin or server arm's prompt length.
arm_draws() {
  local e=${A_ENG[$1]} d=${A_DEP[$1]}
  if [ "${A_KIND[$1]}" != ref ]; then
    echo "$d"
    return
  fi
  case $e in
    lcpppp*) echo $((3 * d)) ;;
    ikpp*) echo $((2 * d + 1)) ;;
    ik) if [ "$d" = 0 ]; then echo $((N + 3)); else echo $((d + N + 4)); fi ;;
    *) echo $((d + N + 3)) ;;
  esac
}
[ "$ORDER" = rotate ] || blocks_plan
# preheat_off: why the preheat is off, for the dry run and the [config] line.
preheat_off() {
  if [ "$ik$lcpp" = 00 ]; then
    echo "off (no reference arm)"
  elif [ -n "${BLOOMERY_PREHEAT:-}" ]; then
    echo "off (BLOOMERY_PREHEAT=0)"
  else
    echo "off (BLOOMERY_AB_ORDER=blocks: each block's discard reads its host set; BLOOMERY_PREHEAT=1 turns it on)"
  fi
}

# majflt_mark's marks (cold-blocks.sh): generate_ds41's `fed` record, printed just before its prompt
# timer starts, or an lcpp arm's progress line of its measured repetition, printed just after that
# repetition's clock starts.

# The arms that failed: arm_fail, FAILED and FAILED_KEYS (cold-blocks.sh); an arm's whole output goes to
# ${TMPDIR:-/tmp}/depth-ds41-<label>-<d|p><key>-r<round>.log.
ARM_FAIL_STEM=depth-ds41
# preheat_arm <engine>: the host set of K = PHK into the page cache (gguf-ranges.py preheat, under the
# arm bound) and its `preheat` line; non-zero with FAIL_WHY on a failure.
preheat_arm() {
  local out rc=0
  out=$(timeout --kill-after=10 "$BOUND" python3 "$GGUF_RANGES" preheat "$PH_DIR/k$PHK.tsv" 2>&1) || rc=$?
  if [ "$rc" -ne 0 ]; then
    FAIL_WHY="its preheat failed (rc $rc): ${out##*$'\n'}"
    return "$rc"
  fi
  echo "preheat $1 K=$PHK $out"
}

# One reference arm: its preheat, its llama-bench on its flags, the row, and the sum; a FAIL row when
# any of them fails. The llama-bench output passes through majflt_mark on its way into `raw` (stderr
# alone for a pp arm, whose stdout is the json), so the fault count at an lcpp arm's measured window's
# start is known.
# ref_arm <engine> <depth> <round>
ref_arm() {
  local eng=$1 dep=$2 r=$3 raw rc=0 val build dev t0 t1 cold errf key f0 f1 win tags markf mark whole fitsrc cnt
  FIT_COL=''
  ref_cmd "$eng" "$dep"
  if pp_eng "$eng"; then key=p=$dep; else key=d=$dep; fi
  if [ -n "$PH_DIR" ]; then
    ph_k "$eng:$dep"
    preheat_arm "$eng" || { rc=$?; arm_fail "$r" "$eng" "$key" "$rc" "$FAIL_WHY"; return 0; }
  fi
  markf=$(mktemp "${TMPDIR:-/tmp}/depth-ds41-mark.XXXXXX") || exit 2
  witness "pre r$r $eng d=$dep"
  ref_witness
  t0=$(date +%s)
  f0=$(majflt_now)
  if pp_eng "$eng"; then
    # The json goes to stdout alone (fd 3 out of the pipe); the loader's log, through majflt_mark,
    # goes to a file, shown on a failure. One pipeline, so both are whole when `raw` is.
    errf=${TMPDIR:-/tmp}/depth-ds41-$eng-d$dep-r$r.err
    raw=$({ timeout --kill-after=10 "$BOUND" env "${REF_ENV[@]}" "$REF_BIN" "${REF_ARGS[@]}" 2>&1 1>&3 3>&- | majflt_mark "$markf" "$REF_MARK" > "$errf"; exit "${PIPESTATUS[0]}"; } 3>&1) || rc=$?
  else
    raw=$(timeout --kill-after=10 "$BOUND" env "${REF_ENV[@]}" "$REF_BIN" "${REF_ARGS[@]}" 2>&1 | majflt_mark "$markf" "$REF_MARK"; exit "${PIPESTATUS[0]}") || rc=$?
  fi
  f1=$(majflt_now)
  t1=$(date +%s)
  mark=$(cat "$markf")
  rm -f "$markf"
  witness "post r$r $eng d=$dep"
  guard_cpu "post r$r $eng d=$dep"
  # Two cards: an Xid, a card lost or off its cap, or an engine that saw one card fails the arm (a pp
  # arm's ggml_cuda_init lines are on its stderr, the errf file).
  if pp_eng "$eng"; then fitsrc=$(cat "$errf" 2> /dev/null); else fitsrc=$raw; fi
  if ! timing_cards_arm "$fitsrc"; then
    arm_fail "$r" "$eng" "$key" "$rc" "two cards: $TWOCARD_WHY" "$fitsrc"
    return 0
  fi
  if pp_eng "$eng"; then
    # Exactly one test and two samples, or no value: a missing field is a failed arm, not a 0.
    val=$(echo "$raw" | jq -r 'if length == 1 and (.[0].samples_ns | length) == 2
      then .[0] | (.n_prompt * 1e9 / .samples_ns[1] * 100 | round / 100) else empty end' 2>/dev/null)
    cold=$(echo "$raw" | jq -r '.[0] | (.n_prompt * 1e9 / .samples_ns[0] * 100 | round / 100)' 2>/dev/null)
    [ -z "$val" ] && raw="$raw
$(tail -n 40 "$errf" 2>/dev/null)"
  else
    val=$(echo "$raw" | grep -F "$REF_LABEL" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g')
  fi
  # A fit arm's loader lines (stderr: the errf file for a pp arm) say what the fit chose; a fit that
  # failed or never ran is a FAIL row, whatever llama-bench measured after it.
  if lcpp_fit_eng "$eng"; then
    if pp_eng "$eng"; then fitsrc=$(cat "$errf" 2> /dev/null); else fitsrc=$raw; fi
    lcpp_fit_col "$fitsrc" || { arm_fail "$r" "$eng" "$key" "$rc" "$FIT_WHY" "$fitsrc"; return 0; }
  fi
  if [ $rc -ne 0 ] || [ -z "$val" ]; then
    arm_fail "$r" "$eng" "$key" "$rc" "no '${REF_LABEL% |}' row" "$raw"
    return 0
  fi
  # The timed window of one repetition: P / tok/s for a pp row, N / tok/s for a decode row.
  if pp_eng "$eng"; then win=$(awk -v p="$dep" -v v="$val" 'BEGIN { printf "%.4f", p / v }'); else win=$(awk -v n="$N" -v v="$val" 'BEGIN { printf "%.4f", n / v }'); fi
  whole=$((f1 - f0)) cnt=$((f1 - f0))
  if [ -z "$REF_MARK" ]; then
    cold_check "$whole" "$win"
    MAJ_COL=" | majflt $whole (whole process; ≤ $MAJ_BOUND % of W ${win} s)"
  elif [ -n "$mark" ]; then
    cnt=$((f1 - mark))
    cold_check "$((f1 - mark))" "$win"
    MAJ_COL=" | majflt $whole (timed $((f1 - mark)); ≤ $MAJ_BOUND % of W ${win} s)"
  else
    cold_check "$whole" "$win"
    MAJ_COL=" | majflt $whole (timed ? (no progress line: the whole process); ≤ $MAJ_BOUND % of W ${win} s)"
  fi
  # Under the warm rows a tagged first run prints as COLD, a tagged retry as its FAIL row (cold-blocks.sh).
  cold_verdict "$r" "$eng" "$key" "$cnt" "$win" || return 0
  tags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
  # The reference's own table, header and row: its columns name every setting it ran with that
  # differs from its defaults, so the log shows which flags took.
  if pp_eng "$eng"; then
    # The json's settings, samples dropped: the same "which flags took" as the md table.
    echo "$raw" | jq -c '.[0] | del(.samples_ns, .samples_ts)' | sed "s/^/    $eng params /"
    [ -z "$FIT_COL" ] || echo "$FIT_LINES" | sed "s/^/    $eng fit /"
    build=$(echo "$raw" | jq -r '.[0] | "\(.build_commit) (\(.build_number))"')
    dev=$(echo "$raw" | jq -r '.[0].gpu_info')
  else
    echo "$raw" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $eng table /"
    [ -z "$FIT_COL" ] || echo "$FIT_LINES" | sed "s/^/    $eng fit /"
    build=$(echo "$raw" | sed -n 's/^build: //p' | head -n 1)
    dev=$(echo "$raw" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
    [ -z "$TWOCARD_DEVS" ] || dev=$TWOCARD_DEVS
  fi
  if pp_eng "$eng"; then
    echo "$ROW_TAG r$r $eng p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME$FIT_COL | cold ${cold:-?} (repetition 1 of 2) | $REF_BATCH | build ${build:-?} | device ${dev:-?}$MAJ_COL | wall $((t1 - t0))s$tags"
    counted || return 0
    pp_sums+=("$eng|$dep|$r|$val|$tags")
  else
    echo "$ROW_TAG r$r $eng d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME$FIT_COL | build ${build:-?} | device ${dev:-?}$MAJ_COL | wall $((t1 - t0))s$tags"
    counted || return 0
    sums+=("$eng|$dep|$r|$val||$tags")
  fi
  count_row
}

# A server arm's row after its device column (lcpp-warm.sh srv_row), and its CPU guard after the server.
srv_tail() { echo "$MAJ_COL | wall ${1}s$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"; }
srv_after() { guard_cpu "post r$1 $2 $3"; }

# The closing summary's row counts: every ROW line, and those that carried each tag.
count_row() {
  n_rows=$((n_rows + 1))
  [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
  [ -z "$OTHER_BUSY_TAG" ] || other_rows=$((other_rows + 1))
  [ -z "$COLD_TAG" ] || cold_rows=$((cold_rows + 1))
}

# The variables arm <i> of ours runs with, into ARM_ENVS: its own, and for a DSpark arm
# (BLOOMERY_DRAFT=dspark) the other card's visibility and the draft file (timing-card.sh dspark_env).
# The dry run prints the same list.
arm_envs() {
  local envs
  ARM_ENVS=()
  envs=$(lg_strip_solo "${A_ENV[$1]}")
  [ -z "$envs" ] || IFS=, read -r -a ARM_ENVS <<< "$envs"
  if [[ ",${A_ENV[$1]}," == *",BLOOMERY_DRAFT=dspark,"* ]]; then
    local -a extra=()
    mapfile -t extra < <(dspark_env)
    ARM_ENVS=("${extra[@]}" "${ARM_ENVS[@]}")
  fi
}
# The prompt arm <i> of ours feeds, into ARM_FEED: --tokens <the corpus ids> for a corpus arm, else
# --depth <D> (the binary's LCG prompt).
arm_feed() {
  if [ "${A_KIND[$1]}" = corpus ]; then ARM_FEED=(--tokens "${A_TOK[$1]}"); else ARM_FEED=(--depth "${A_DEP[$1]}"); fi
}
# One arm of a generate_ds41 at --place PLACE in a process of its own, with the one-arm command line:
# a bin: arm (a second binary, which may know no --arm). The row and the sum under the arm's label, or
# a FAIL row (ours_post). The output passes through majflt_mark on its way into `out`, so the fault
# count at the prompt timer's start is known: MAJ_WHOLE over the process, MAJ_TIMED from the fed line on.
# ours_arm <index> <round>
ours_arm() {
  local i=$1 r=$2 dep bin out rc t0 t1 f0 f1 fedf fed
  local -a envs=() feed=()
  dep=${A_DEP[$i]} bin=${A_BIN[$i]}
  arm_envs "$i"
  envs=("${ARM_ENVS[@]}")
  arm_feed "$i"
  feed=("${ARM_FEED[@]}")
  fedf=$(mktemp "${TMPDIR:-/tmp}/depth-ds41-fed.XXXXXX") || exit 2
  ours_pre "$i" "$r"
  t0=$(date +%s)
  f0=$(majflt_now)
  if [ ${#envs[@]} -eq 0 ]; then
    out=$(timeout --kill-after=10 "$BOUND" "$bin" "${feed[@]}" -n "$N" --place "$PLACE" --time ${WARM:+--warm "$WARM"} 2>&1 | majflt_mark "$fedf" '^fed '; exit "${PIPESTATUS[0]}")
  else
    out=$(timeout --kill-after=10 "$BOUND" env "${envs[@]}" "$bin" "${feed[@]}" -n "$N" --place "$PLACE" --time ${WARM:+--warm "$WARM"} 2>&1 | majflt_mark "$fedf" '^fed '; exit "${PIPESTATUS[0]}")
  fi
  rc=$?
  f1=$(majflt_now)
  t1=$(date +%s)
  fed=$(cat "$fedf")
  rm -f "$fedf"
  MAJ_WHOLE=$((f1 - f0)) MAJ_TIMED=
  [ -z "$fed" ] || MAJ_TIMED=$((f1 - fed))
  ours_post "$i" "$r" "$rc" "$out" "$((t1 - t0))"
}
# ours_pre <index> <round>: the witness block before a generate_ds41 arm.
ours_pre() { witness "pre r$2 ${A_LABEL[$1]} d=${A_DEP[$1]} n=$N"; }
# ours_post <index> <round> <rc> <output> <wall s>: the witness block after a generate_ds41 arm, then its
# row and sums, or its FAIL row; MAJ_WHOLE and MAJ_TIMED are the arm's. An output that opens with an
# `arm` record (an --arm list's) gives the row its slot in the load.
ours_post() {
  local i=$1 r=$2 rc=$3 out=$4 wall=$5 dep label tags a slot
  dep=${A_DEP[$i]} label=${A_LABEL[$i]}
  witness "post r$r $label d=$dep n=$N"
  guard_cpu "post r$r $label d=$dep"
  SLOT_COL=
  a=$(sed -nE '1s/^arm i=([0-9]+) arms=([0-9]+) .*/\1 \2/p' <<< "$out")
  [ -z "$a" ] || SLOT_COL=" | slot $((${a% *} + 1))/${a#* }"
  if [ "$rc" -ne 0 ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "exited $rc" "$out"
    return 0
  fi
  # Two cards: an Xid, a card lost or off its cap, or a load that did not name both cards fails the arm
  # (a grouped arm's load record is its load's header).
  if ! timing_cards_arm "$out"$'\n'"${LG_HEADER:-}" ours; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "two cards: $TWOCARD_WHY" "$out"
    return 0
  fi
  # The residency seed condition (the header's Residency): the slot from the arm record, 1 for a process
  # of its own; LG_PREV_OUT is read only past slot 1, which only the load driver's processes have.
  res_read "$out" || {
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "$RES_WHY" "$out"
    return 0
  }
  slot=1
  [ -z "$a" ] || slot=$((${a% *} + 1))
  res_seed "$i" "$slot" || {
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" residency "$RES_WHY" "$out"
    return 0
  }
  ours_row "${A_KIND[$i]}" "$label" "$dep" "$r" "$out" "$wall" || {
    [ $? = 3 ] || arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "$FAIL_WHY" "$out"
    return 0
  }
  res_curve "$label" "$r" "$dep" "$slot"
  counted || return 0
  count_row
  tags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
  if [ -n "$DRAFT" ]; then TPS_MEAN=${DRAFT##*tok/s(positions)=}; fi
  sums+=("$label|$dep|$r|$TPS_MEAN|$TPS_P50|$tags")
  [ -z "$PP_N" ] || pp_sums+=("$label|$PP_N|$r|$PP_TPS|$tags")
  [ -z "$SLOT_COL" ] || slot_sums+=("$label|${SLOT_COL##*slot }|$PP_N|${PP_TPS:-}|$TPS_MEAN")
}
# The driver's hooks (tools/ref/load-groups.sh): a load's command line and environment, and an arm's
# guards and witness blocks around it. The load's lines echoed once: its plan, load, host set, capture
# and prompt buffer lines.
LG_HEADER_RE='^(plan|load|host|capture|prefill|residency host) '
lg_cmd() {
  local i
  arm_envs "$1"
  LG_ENV=("${ARM_ENVS[@]}")
  LG_CMD=("$BIN")
  for i in "$@"; do
    case ${A_KIND[$i]} in
      corpus) LG_CMD+=(--arm "${A_ENG[$i]}:${A_DEP[$i]}") ;;
      *) LG_CMD+=(--arm "${A_DEP[$i]}") ;;
    esac
  done
  # shellcheck disable=SC2206 # an empty WARM adds nothing
  LG_CMD+=(-n "$N" --place "$PLACE" --time ${WARM:+--warm "$WARM"} --arm-sync)
}
lg_pre() {
  prime_tag "$1"
  CPU_BUSY_TAG=
  guard_other
  guard_cpu "$(guard_label "$1" "$2")"
  ours_pre "$1" "$2"
}
# A COLD row's arm goes on COLD_LIST, its unit's retry (run_unit); a PRIME or COLD row's tag is undone.
lg_post() {
  COLD_QUEUED=0
  ours_post "$@"
  [ "$COLD_QUEUED" = 0 ] || COLD_LIST+=("$1")
  [ "$PRIMING" = 0 ] && [ "$COLD_QUEUED" = 0 ] || ROW_TAG=ROW
}
# guard_label <index> <round>: the CPU guard's label before arm <index>.
guard_label() {
  case $ROW_TAG in
    WARMUP) echo "pre warmup ${ARMS[$1]}" ;;
    DISCARD) echo "pre discard ${ARMS[$1]}" ;;
    *) echo "pre r$2 ${ARMS[$1]}" ;;
  esac
}

# ratio_table <prefix> <keys> <labels> <tag field> [base]: records `label|key|round|value|…` on stdin;
# for every key and every label of <labels>, each round's base / label ratio (arms that ran more than
# once in a round averaged first), their mean with its 95 % interval (Student t at rounds - 1 degrees
# of freedom, T975) and the ratio of the arm means. The base is ours (the default), or a corpus name
# for that corpus's tables.
# A tag field above 0 is the record's field holding the row's tags (5 in the prefill records, 6 in
# the decode ones), and the line ends with each side's count of [cpu-busy], [other-busy] and [cold].
ratio_table() {
  awk -F'|' -v prefix="$1" -v deps="$2" -v refs="$3" -v tagged="$4" -v base="${5:-ours}" -v rounds="$ROUNDS" -v t975="$T975" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
  if (tagged) { if ($tagged ~ /cpu-busy/) bc[a]++; if ($tagged ~ /other-busy/) bo[a]++; if ($tagged ~ /cold/) bk[a]++ }
} END {
  nt = split(t975, t, " ")
  nd = split(deps, d, " ")
  nr = split(refs, rf, " ")
  for (i = 1; i <= nd; i++) for (j = 1; j <= nr; j++) {
    ref = rf[j]
    if (!((base SUBSEP d[i]) in an) || !((ref SUBSEP d[i]) in an)) continue
    c = 0; m = 0; list = ""
    for (r = 1; r <= rounds; r++) {
      ko = base SUBSEP d[i] SUBSEP r; kr = ref SUBSEP d[i] SUBSEP r
      if (!(ko in rn) || !(kr in rn)) continue
      q = (rs[ko] / rn[ko]) / (rs[kr] / rn[kr]); c++; v[c] = q; m += q
      list = list sprintf(" r%d %.4f", r, q)
    }
    if (c == 0) continue
    m /= c; ss = 0
    for (x = 1; x <= c; x++) ss += (v[x] - m) ^ 2
    if (c < 2) ci = "(one round: no interval)"
    else if (c - 1 > nt) ci = sprintf("(no t quantile for df %d)", c - 1)
    else ci = sprintf("± %.4f", t[c - 1] * sqrt(ss / (c - 1)) / sqrt(c))
    ao = base SUBSEP d[i]; ar = ref SUBSEP d[i]
    busy = tagged ? sprintf("  busy: %s [cpu-busy %d/%d] [other-busy %d/%d] [cold %d/%d], %s [cpu-busy %d/%d] [other-busy %d/%d] [cold %d/%d]", base, bc[ao], an[ao], bo[ao], an[ao], bk[ao], an[ao], ref, bc[ar], an[ar], bo[ar], an[ar], bk[ar], an[ar]) : ""
    printf "%s%-5s %s/%-6s  mean %.4f %s (n=%d)  of means %.4f  per round:%s%s\n", prefix, d[i], base, ref, m, ci, c, (as[ao] / an[ao]) / (as[ar] / an[ar]), list, busy
  }
}'
}

# dry_cmd <i>: arm <i>'s command line as the dry run prints it (a reference arm at REF_K when set).
dry_cmd() {
  local i=$1 dep=${A_DEP[$1]} note='' feedline var
  if [ "${A_KIND[$i]}" = srv ]; then
    srv_dry_cmd "$i"
    return
  fi
  if [ "${A_KIND[$i]}" = ref ]; then
    ref_cmd "${A_ENG[$i]}" "$dep"
    echo "timeout --kill-after=10 $BOUND env ${REF_ENV[*]} $REF_BIN ${REF_ARGS[*]}   # row label '${REF_LABEL% |}'${REF_BATCH:+, $REF_BATCH}${REF_MARK:+, measured window from /${REF_MARK}/}${REF_FIT:+, $REF_FIT}"
    return
  fi
  if lg_grouped "$i"; then
    lg_cmd "$i"
    echo "one arm of a load: timeout --kill-after=10 \$((BOUND x arms + BOUND)) ${LG_ENV[*]:+env ${LG_ENV[*]} }${LG_CMD[*]}   # row label '${A_LABEL[$i]}', load key ${LG_KEY[$i]}"
    return
  fi
  feedline="--depth $dep"
  [ "${A_LABEL[$i]}" = ours ] || note="   # row label '${A_LABEL[$i]}'"
  if [ "${A_KIND[$i]}" = corpus ]; then
    var=CORPUS_N_${A_ENG[$i]}
    feedline="--tokens \"\$(head -n $dep $(corpus_file "${A_ENG[$i]}") | paste -sd, -)\""
    note="$note, $dep of the file's ${!var} ids, first ${A_TOK[$i]%%,*}, last ${A_TOK[$i]##*,}"
  fi
  arm_envs "$i"
  echo "timeout --kill-after=10 $BOUND ${ARM_ENVS[*]:+env ${ARM_ENVS[*]} }${A_BIN[$i]} $feedline -n $N --place $PLACE --time${WARM:+ --warm $WARM}$note"
}
# run_unit <round> <index...>: one unit of a round (tools/ref/load-groups.sh): the arms of one load key
# in one process through the driver, or one reference or bin: arm after the contention guards; into
# their rows or FAIL rows.
# Under the warm rows (cold-blocks.sh) a unit of our arms runs each arm after its PRIME in one load, and
# its COLD rows' arms again in a fresh load, prime and arm; a reference arm's COLD row runs its process
# again; a server arm retries inside srv_arm.
run_unit() {
  local r=$1
  shift
  COLD_QUEUED=0
  if lg_grouped "$1"; then
    if [ "$WARM_ROWS" = 0 ] || ! counted; then
      lg_run_unit "$r" "$@"
      return
    fi
    COLD_LIST=()
    prime_list "$@"
    lg_run_unit "$r" "${PRIME_LIST[@]}"
    PRIMING=0
    [ ${#COLD_LIST[@]} -gt 0 ] || return 0
    prime_list "${COLD_LIST[@]}"
    COLD_TRY=1
    lg_run_unit "$r" "${PRIME_LIST[@]}"
    COLD_TRY=0 PRIMING=0
    return
  fi
  unit_guard "$1" "$r"
  case ${A_KIND[$1]} in
    ref)
      ref_arm "${A_ENG[$1]}" "${A_DEP[$1]}" "$r"
      [ "$COLD_QUEUED" = 1 ] || return 0
      ROW_TAG=ROW COLD_TRY=1
      unit_guard "$1" "$r"
      ref_arm "${A_ENG[$1]}" "${A_DEP[$1]}" "$r"
      COLD_TRY=0
      ;;
    srv) srv_arm "$1" "$r" ;;
    *)
      if [ "$WARM_ROWS" = 1 ] && counted; then
        ROW_TAG=PRIME
        ours_arm "$1" "$r"
        ROW_TAG=ROW
        unit_guard "$1" "$r"
      fi
      COLD_QUEUED=0
      ours_arm "$1" "$r"
      [ "$COLD_QUEUED" = 1 ] || return 0
      ROW_TAG=PRIME COLD_TRY=1
      unit_guard "$1" "$r"
      ours_arm "$1" "$r"
      ROW_TAG=ROW
      unit_guard "$1" "$r"
      ours_arm "$1" "$r"
      COLD_TRY=0
      ;;
  esac
}
# unit_guard <index> <round>: the contention guards before an arm's process.
unit_guard() {
  CPU_BUSY_TAG=
  guard_other
  guard_cpu "$(guard_label "$1" "$2")"
}
# run_round <round> <index...>: those arms' units in the round's order.
run_round() {
  local r=$1 line
  local -a units idx
  shift
  lg_units "$@"
  mapfile -t units < <(lg_round "$r")
  for line in "${units[@]}"; do
    read -r -a idx <<< "$line"
    run_unit "$r" "${idx[@]}"
  done
}
# round_order <round> <index...>: the round's arms in order, and its loads: `<arms>` and `[<a b> <c>]`.
round_order() {
  local r=$1 line i o='' u=''
  local -a units idx
  shift
  lg_units "$@"
  mapfile -t units < <(lg_round "$r")
  for line in "${units[@]}"; do
    read -r -a idx <<< "$line"
    local -a names=()
    for i in "${idx[@]}"; do names+=("${ARMS[$i]}"); done
    o+="${o:+ }${names[*]}"
    if lg_grouped "${idx[0]}"; then u+="${u:+ }[${names[*]}]"; else u+="${u:+ }${names[*]}"; fi
  done
  ORDER_ARMS=$o ORDER_LOADS=$u
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES place=$PLACE preheat=$PREHEAT order=$ORDER"
  ref_witness | sed 's/^   /[dry]/'
  echo "[dry] cpu guard: comms=[$CPU_BUSY_COMMS] threshold=${CPU_BUSY_PCT}% strict=${BLOOMERY_OTHER_STRICT:-0} now: $(cpu_busy_reading)"
  if [ -n "$TIMING_CARDS" ]; then
    echo "[dry] two cards: $TIMING_CARDS_NAME, the profile's two-card line: $TWO_CARD_PLACEMENT"
    tc_rc=0
    timing_cards_precheck '[dry] ' || tc_rc=$?
    if [ "$tc_rc" = 0 ]; then
      echo "[dry] two-card precheck: ok"
    else
      echo "[dry] two-card precheck: refused (rc $tc_rc): $TWOCARD_WHY — a real run stops here, before the lease"
    fi
  fi
  ph_round=0
  for i in "${!ARMS[@]}"; do
    a=${ARMS[$i]}
    dep=${A_DEP[$i]}
    echo "[dry] $a: $(dry_cmd "$i")"
    if [ "${A_KIND[$i]}" = ref ] && [ -n "$PH_DIR" ]; then
      ref_cmd "${A_ENG[$i]}" "$dep"
      ph_k "$a"
      b=$(ph_bytes "$PHK" "$PHNGL")
      var=PH_LINE_${PHK}_${PHNGL:-none}
      ph_round=$((ph_round + b))
      echo "[dry] $a: preheat K=$PHK: $b B ($(awk -v b="$b" 'BEGIN { printf "%.1f", b / 1e9 }') GB), $(awk -v b="$b" -v r="$PH_RATE" 'BEGIN { printf "%.0f", b / 1e9 / r }') s if all of it is cold at $PH_RATE GB/s; ${!var}"
    fi
  done
  if [ -n "$PH_DIR" ]; then
    echo "[dry] preheat: $ph_round B a round over the reference arms, $(awk -v b="$ph_round" -v r="$PH_RATE" 'BEGIN { printf "%.0f", b / 1e9 / r }') s a round if every byte is cold at $PH_RATE GB/s (the upper end: an arm after one of its own engine finds its set cached)"
  else
    echo "[dry] preheat: $(preheat_off)"
  fi
  [ "$WARM_ROWS" = 0 ] || echo "[dry] warm rows: each of our arms after a same-id PRIME in its load; a counted row tagged [cold] prints as COLD and runs once more (cold-blocks.sh)"
  [ "$gen" = 0 ] || echo "[dry] $(res_config)"
  if [ "$ORDER" = rotate ]; then
    if [ "$AB_WARMUP" = 1 ]; then
      echo "[dry] warmup: ${ARMS[0]} once before round 1 (its command line above), discarded — its row prints as WARMUP r0 and is in no mean, ratio or row count (BLOOMERY_AB_WARMUP=0 skips it)"
    else
      echo "[dry] warmup: off (BLOOMERY_AB_WARMUP=0): round 1's first row is the lease's first process"
    fi
    for r in $(seq "$ROUNDS"); do
      round_order "$r" "${!ARMS[@]}"
      echo "[dry] round $r order: $ORDER_ARMS"
      echo "[dry] round $r loads: $ORDER_LOADS"
    done
    exit 0
  fi
  blocks_dry
  exit 0
fi

majflt_require depth-ds41.sh
timing_cards_precheck || {
  rc=$?
  echo "depth-ds41.sh: two cards, refused before the lease: $TWOCARD_WHY" >&2
  exit "$rc"
}
lease_take
timing_cards_start
[ -z "$TIMING_CARDS" ] || echo "[config] two cards: $TIMING_CARDS_NAME, the profile's two-card line: $TWO_CARD_PLACEMENT"
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s warmup=$AB_WARMUP"
echo "[config] ours: $BIN (--place $PLACE, default ctx)"
for c in $CORPORA; do
  var=CORPUS_N_$c
  [ -z "${!var:-}" ] || echo "[config] $c: the first P ids of $(corpus_file "$c") (${!var} ids)"
done
if [ -n "$PH_DIR" ]; then
  for var in ${!PH_LINE_*}; do echo "[config] preheat: ${!var}"; done
else
  echo "[config] preheat: $(preheat_off)"
fi
blocks_config
echo "[config] cold tag: majflt in the row's measured window (ours: from its fed line; lcpp: from its --progress line; ik: the whole process) × ${COLD_US} µs ≥ ${COLD_PCT} % of that window"
echo "[config] ik: $IKBIN flags=$IK_GPU_FLAGS env=$IK_GPU_ENV"
echo "[config] lcpp: $LCPPBIN flags=$LCPP_GPU_FLAGS (lcpp<K>: --n-cpu-moe K)"
if [ "$lcppfit" = 1 ]; then
  lcpp_fit_flags "$LCPP_GPU_FLAGS"
  echo "[config] lcppfit: $LCPPBIN flags=$FIT_FLAGS (llama-bench's fit places the model; dropped: $FIT_DROPPED; lcppfit and lcppppfit get --progress as lcpp and lcpppp do)"
fi
echo "[config] prefill: ikpp/lcpppp run llama-bench -p P -n 0 -r 2 -o json at the flags above, the row is repetition 2 (<U>: -ub U -b max(U, 2048)); ours from its time prompt row"
[ "$srv" = 0 ] || srv_config
warm_rows_config
[ "$gen" = 0 ] || echo "[config] $(res_config)"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[config] cpu guard: comms=[$CPU_BUSY_COMMS] threshold=${CPU_BUSY_PCT}% strict=${BLOOMERY_OTHER_STRICT:-0}"
witness pre
ref_witness
guard_other
guard_cpu pre

sums=() pp_sums=() slot_sums=()
n_rows=0 busy_rows=0 other_rows=0 cold_rows=0
if [ "$ORDER" = rotate ]; then
  if [ "$AB_WARMUP" = 1 ]; then
    ROW_TAG=WARMUP
    run_unit 0 0
    ROW_TAG=ROW
    echo "[warmup] ${ARMS[0]} ran once before round 1 and is discarded (the WARMUP or FAIL r0 row above): the lease's first process reads the model's pages cold"
  fi
  for r in $(seq "$ROUNDS"); do run_round "$r" "${!ARMS[@]}"; done
else
  blocks_run
fi
echo
echo "cpu-busy rows: $busy_rows of $n_rows (BLOOMERY_CPU_BUSY_PCT=${CPU_BUSY_PCT}% over [$CPU_BUSY_COMMS])"
echo "other-busy rows: $other_rows of $n_rows (a compute process on the other card as the arm started)"
echo "cold rows: $cold_rows of $n_rows (the measured window's majflt × ${COLD_US} µs ≥ ${COLD_PCT} % of that window)"
warm_rows_summary
# A failed arm drops out at its depth or P (cold-blocks.sh).
failed_tally
echo "=== per-arm means (tok/s @ n=$N, $CARD_NAME). First column: ours from mean_ms, the references"
echo "    llama-bench's own mean over the N steps — the cross-engine ratio reads these. The p50"
echo "    column is ours only. ==="
[ ${#sums[@]} -eq 0 ] || printf '%s\n' "${sums[@]}" | awk -F'|' '{
  k = $1 " d=" $2; s[k] += $4; n[k]++; if ($5 != "") { sp[k] += $5; np[k]++ }
  if ($6 ~ /cold/) c[k]++
  if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
} END { for (k in s) {
  spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
  printf "mean %-14s %8.2f tok/s  [%s..%s, spread %.2f%%]  %s (n=%d)  [cold %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], spread, (np[k] ? sprintf("%.2f tok/s(p50)", sp[k] / np[k]) : ""), n[k], c[k], n[k] } }' | sort
echo
echo "=== ours / reference per depth: each round's ratio of the pair measured in that round, their"
echo "    mean with its 95 % interval (Student t, rounds - 1 degrees of freedom; 2.0 past 21 rounds),"
echo "    the ratio of the arm means, and each side's tagged rows ==="
deps=$(printf '%s\n' "${A_DEP[@]}" | sort -un | tr '\n' ' ')
# The corpus labels have their own tables: their prompt is not the one ours and the references ran.
corpus_re=${CORPORA// /|}
refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -vx ours | grep -vE "^($corpus_re)(@|$)|@($corpus_re)$" | sort -u | tr '\n' ' ')
printf '%s\n' "${sums[@]}" | ratio_table "ratio d=" "$deps" "$refs" 6 ours
for c in $CORPORA; do
  c_refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -E "^$c@|@$c$" | sort -u | tr '\n' ' ')
  [ -n "$c_refs" ] || continue
  echo
  echo "=== the $c prompt: $c / each $c@ arm per P, the same statistics ==="
  printf '%s\n' "${sums[@]}" | ratio_table "ratio $c d=" "$deps" "$c_refs" 6 "$c"
done
if [ ${#pp_sums[@]} -gt 0 ]; then
  echo
  echo "=== prefill per prompt length (tok/s(pp) @ n=0, prompt P, $CARD_NAME). Ours: its time prompt"
  echo "    row (the P fed steps through token 0's readback); the references: llama-bench's pp value"
  echo "    over one repetition. The tags count the rows that met contention or faults. ==="
  printf '%s\n' "${pp_sums[@]}" | awk -F'|' '{
    k = $1 " p=" $2; s[k] += $4; n[k]++
    if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
    if ($5 ~ /cpu-busy/) c[k]++; if ($5 ~ /other-busy/) o[k]++; if ($5 ~ /cold/) f[k]++
  } END { for (k in s) {
    spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
    printf "mean pp %-14s %8.2f tok/s(pp)  [%s..%s, spread %.2f%%]  (n=%d)  [cpu-busy %d/%d] [other-busy %d/%d] [cold %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], spread, n[k], c[k], n[k], o[k], n[k], f[k], n[k] } }' | sort
  echo
  echo "=== ours / reference prefill per prompt length: the decode table's statistics over the pp"
  echo "    values, then how many of each side's rows carried [cpu-busy], [other-busy] and [cold] ==="
  pp_keys=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  pp_refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep -vx ours | grep -vE "^($corpus_re)(@|$)|@($corpus_re)$" | sort -u | tr '\n' ' ')
  printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp p=" "$pp_keys" "$pp_refs" 5 ours
  for c in $CORPORA; do
    c_refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep -E "^$c@|@$c$" | sort -u | tr '\n' ' ')
    [ -n "$c_refs" ] || continue
    echo
    echo "=== the $c prompt's prefill: $c / each $c@ arm per P, the same statistics ==="
    printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp $c p=" "$pp_keys" "$c_refs" 5 "$c"
  done
fi
if [ ${#slot_sums[@]} -gt 0 ]; then
  echo
  echo "=== load slots: each label's rows first after their load (slot 1) and after another arm of it"
  echo "    (slot 2 on), the prefill per P and the decode ==="
  printf '%s\n' "${slot_sums[@]}" | awk -F'|' '{
    split($2, sl, "/"); later = (sl[1] > 1); k = $1 " p=" $3
    if ($4 != "") { pp[k, later] += $4; np[k, later]++ }
    tg[k, later] += $5; nt[k, later]++; keys[k] = 1
  } END { for (k in keys) {
    line = sprintf("slots %-24s", k)
    for (l = 0; l <= 1; l++) line = line sprintf("  %s: pp %s tg %s (n=%d)", l ? "later" : "first", \
      np[k, l] ? sprintf("%.2f", pp[k, l] / np[k, l]) : "-", nt[k, l] ? sprintf("%.2f", tg[k, l] / nt[k, l]) : "-", nt[k, l] + 0)
    if (np[k, 0] && np[k, 1]) line = line sprintf("  later/first pp %.4f", (pp[k, 1] / np[k, 1]) / (pp[k, 0] / np[k, 0]))
    print line } }' | sort
fi
witness post
ref_witness
failed_end
