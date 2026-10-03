#!/usr/bin/env bash
# GLM-5.3-Flash decode by depth and prefill by prompt length on the timing card, three engines in one
# lease (run on the box, lead-only): our engine (generate_glm5next --time), llama.cpp on the two open
# PR branches that build glm5next (#27752, #27754; mainline does not) and exllamav3's own bench on its
# EXL3 quantization of the model.
#
#   just depth-gpu-glm5next 512 lcpp27752:512 lcpp27754:512
#   just depth-gpu-glm5next 512 lcpp27752pp:512 lcpp27754pp:512 exl3pp:512
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just depth-gpu-glm5next 512 lcpp27754:512   # the command lines, no lease, no load
#
# depth-qwen3moe.sh's shape and tables; what differs is below. Every number carries its conditions:
# `tok/s @ n=N, depth D, <card>`, and the prefill `tok/s(pp) @ n=0, prompt P, <card>`.
#
# Arms:
#   <D>       ours: generate_glm5next --tokens <D ids of GLM_PROSE from GLM_PROSE_FROM> -n N --ctx C --place
#             PLACE --time. The prompt runs as the binary's --prefill default, its batches (the row's
#             pp_tok/s `kind=` names the feed): D over the wall from the first fed position to the
#             readback of generated token 0. The N - 1 steps after token 0 are timed. C is one context
#             for every ours arm and every server arm (BLOOMERY_GEN_CTX, default 2048; docs/fair-measure.md
#             1.5): the plan refuses more than the deepest context a reference set checks
#             (place::ORACLE_POSITIONS, 16,384), and D + N <= C is checked before the lease. The binary
#             takes no --arm list, so an ours arm's warm-up is a process of its own (WARMUP r0, below),
#             not a same-id PRIME in its load. The prompt is prose under this model's own
#             vocabulary (models/glm5next.sh GLM_PROSE, its sha256 checked before the lease): the D
#             ids from 0-based index GLM_PROSE_FROM (models/glm5next.sh: past the ids the router set
#             was traced from, docs/fair-measure.md 4.2). Each ours and server row prints
#             `prompt ids <a>..<b>`.
#   oursmtp:<D>  ours with the MTP draft: the same command under `env BLOOMERY_DRAFT=mtp`, its rows
#             labelled `oursmtp`, its prompt row a prefill record under that label. A drafted arm takes
#             one position more than ours, for the verify's last window: D + N + 1 <= C, checked before
#             the lease. Before the lease the binary's `--levers` under BLOOMERY_DRAFT=mtp must pass (a
#             binary that does not act on the lever refuses it by name), and a BLOOMERY_DRAFT already set
#             in the runner's environment is refused: the plain ours arms would draft too. Its greedy ids
#             are the plain run's, so each row is held to ours' on the same ids and length as a server row
#             is (the xcheck below; oursmtp is never the ours side of it). When generate_glm5next's
#             checked-in schema declares the `mtp summary` record, the row carries its fields through
#             tools/bloomery/records.py (` | mtp positions/pass <positions / passes> = positions P / passes
#             Q, kept [..]`), and an oursmtp row that prints none is a FAIL row; a schema that declares
#             none reads none, the row carries nothing for it, and the [config] (or [dry]) `oursmtp:` line
#             says which. Its row's tok/s is the SMOKE mean's, as ours'. After the tables, `ratio mtp d=`:
#             ours / oursmtp per depth, paired by round (below 1: the draft is faster); oursmtp is in no
#             other ratio table but the placement pairs' (Placement below).
#   <D>@NAME=VALUE[,NAME=VALUE...], oursmtp:<D>@NAME=VALUE[,...]   ours (or oursmtp) with those
#             variables set for this arm's process only (`env NAME=VALUE ... <binary>`, after oursmtp's
#             BLOOMERY_DRAFT=mtp; the item place=<a|gate|bp> is the arm's placement, Placement below):
#             a lever arm, row label `ours@NAME=VALUE[,...]` (`oursmtp@…`), so two arms of one depth are
#             two rows and two means. Beside a plain `<D>` arm it is the same-binary
#             A/B, paired by round in the ratio table's `ours/ours@…` line, e.g. `512
#             512@BLOOMERY_RESIDENCY=mid-p40-s1`; an oursmtp lever arm is in no ratio table, as oursmtp is
#             in none but its own and the placement pairs'. Every arm here is a process of its own, so arms whose variables differ
#             never share one. The list is checked as depth-qwen3moe.sh's is (tools/ref/lever-arms.sh), each
#             refusal by name before anything runs: an empty list, item, name or value; white space; an
#             item that is no NAME=VALUE; a bad variable name; `@` or `|` in a value; a name given twice; a
#             name the runner's own environment already sets; a name that is no lever row of
#             crates/levers/src/registry.rs (BLOOMERY_AB_LOAD included: this runner has no load driver).
#             Also refused: BLOOMERY_DRAFT in any list (oursmtp:<D> is the drafted arm, and sets it), and
#             `@` on a reference arm, whose engine takes no lever of ours. Before the lease the binary's
#             `--levers` under each lever arm's variables (oursmtp's with BLOOMERY_DRAFT=mtp) must pass: a
#             binary that does not act on a lever, or a value its kind does not take, refuses it by name.
#             A lever arm is not in the greedy cross-check (below): its lever may move the arithmetic.
#             Residency records: when generate_glm5next's checked-in schema declares the `residency
#             lever` and `residency pass` records, every ours row carries them through records.py
#             (cold-blocks.sh's residency sums: ` | residency <word> (<why>) passes n kept k landed l late
#             t made m bytes b`), with each label's per-pass means after the tables, and an arm that runs
#             BLOOMERY_RESIDENCY and prints no lever record is a FAIL row; a schema that declares none reads
#             none, and an arm that runs BLOOMERY_RESIDENCY under it is a FAIL row naming the schema
#             (depth-qwen3moe.sh's rule).
#   lcpp27752:<D>, lcpp27754:<D>   the PR branch's llama-bench -p 0 -n N -d D -r 1 at the profile's
#             LCPP27752_GPU_FLAGS / LCPP27754_GPU_FLAGS (the second under LCPP27754_ENV). -d prefills
#             D of llama-bench's own std::rand() ids before its clock starts; its row label is `tgN @
#             dD`.
#   lcpp27752pp:<P>, lcpp27754pp:<P>   the branch's prefill: llama-bench -p P -n 0 -r 1 at the same
#             flags, `ppP`. lcpp27752pp<U>:<P> (and 27754's) adds -ub U -b max(U, 2048): the ubatch
#             lever, not a default. P is not bounded by our context C, so a P above C has no ours row
#             beside it.
#   lcpp27752fit:<D>, lcpp27754fit:<D>, lcpp27752ppfit[<U>]:<P>, lcpp27754ppfit[<U>]:<P>   the
#             branch at llama.cpp's own placement: the lcpp2775x:<D> and lcpp2775xpp[<U>]:<P> command
#             lines with the profile's placement options (-ngl, --n-cpu-moe, -ts, -ot) removed and
#             `-fitt 1024 -v` added, so llama-bench's fit chooses the layers and the overrides
#             (tools/ref/lcpp-fit.sh: what the fit does and where, the flags, the probe, the column);
#             #27754's twins keep LCPP27754_ENV and -fa off. The fit keeps the trailing blocks'
#             experts on the host where --n-cpu-moe keeps the leading ones' (models/glm5next.sh has
#             the predicted placement). The row carries `fit <what it chose>` from -v's loader
#             lines, echoed as `<label> fit …`. A fit that failed or never ran is a FAIL row naming
#             it, whatever the exit code: llama-bench ignores the fit's status and loads at -ngl -1
#             with no overrides, which on this file runs out of card memory or times a placement
#             nobody chose. Refused before the lease when that branch's llama-bench --help lists no
#             --fit-target, or when its flags already carry a fit option.
#             The server arms have no fit twin: an MTP context's fit counts the NextN block
#             (common/fit.cpp:141 in #27754, 140 in #27752, under load_mtp) and a server without the
#             draft's does not, so a fitted srv/mtp pair would sit at two placements and their ratio
#             would no longer be the draft's alone; the placement question is the llama-bench twins'.
#   lcpp27754srv:<D>, lcpp27754srvpp[<U>]:<P>, lcpp27754mtp:<D> (and lcpp27752's)   the branch's
#             llama-server, because llama-bench feeds its own ids and drives no speculation: one server
#             process a row, built on tools/ref/lcpp-warm.sh's functions — the branch's
#             LCPP2775x_GPU_FLAGS in the server's spellings (that file's table, -fit off), its
#             LCPP_SRV_FIXED (-np 1 -ctxcp 0 --cache-ram 0: without -ctxcp 0 this model's recurrent
#             layers have the server copy their state to the host inside the prompt clock) and -c C,
#             ours' --ctx; #27754's under LCPP27754_ENV; the MTP arm at GLM_NCMOE_MTP with GLM_MTP_FLAGS
#             (--spec-type draft-mtp --spec-draft-n-max 2) after them. A server arm whose ids + n_predict +
#             1 pass C is refused before the lease, and a server whose log names another n_ctx is a FAIL
#             row. After /health answers, one POST /completion of ours' D prompt ids (from
#             GLM_PROSE_FROM), so the draft sees prose, is sent and discarded (the warm-up,
#             docs/fair-measure.md 2.1), then the same request is timed: greedy, ignore_eos, cache_prompt
#             off. A decode row (srv, mtp) is predicted_per_second at n_predict N (the decode, drafts
#             included) and carries prompt_n and prompt_per_second, which is also a prefill record under
#             its label; a srvpp row is prompt_per_second at n_predict 1, <U> the ubatch lever (-ub U -b
#             max(U, 2048)). The MTP row carries draft_n / draft_n_accepted beside the server's own
#             `draft acceptance = … (A accepted / G generated), mean len = …` log line, verbatim.
#             predicted_n other than asked, prompt_n other than the ids, cache_n other than 0, or an MTP
#             arm that drafted nothing, is a FAIL row. The srv arm is the MTP arm's twin without the
#             draft: the same binary, request and path, so their ratio is the draft's alone. An engine
#             word takes its own flags, `+t<N>`, `+nopo<0|1>`, `+k<K>` (-t, -nopo, --n-cpu-moe in place
#             of the profile's; lcpp-warm.sh's Per-arm flags), its label with them. Each server row's
#             first ids are held to ours' on the same ids and length (lcpp-warm.sh's Cross-check): one
#             whose first id parts from ours is a FAIL xcheck line in the failed arms and drops out of
#             the means and ratios; one that parts later is a [xcheck-tail] line and stays. A server arm runs no WARMUP
#             r0 process: its warm-up is its own discarded request.
#   exl3:<D>  exllamav3's eval/perf.py -spf --max_length D + 256 at EXL3_FLAGS on EXL3_MODEL, the
#             `Context D` row of its Generation table: 100 steps at depth D (its fixed count, the
#             row says n=100) over wikitext-2 ids, the recurrent state a test state of that depth.
#             D is 0 or a multiple of 256 (perf.py's lengths). exl3pp:<P> is perf.py -sg --max_length
#             P, the `Length P` row of its Prefill table: P ids in chunks of its default 4096, P a
#             multiple of 256 up to 4096. Another quantization: its rows are in the means and in a
#             table of their own, never in a ratio against ours.
# Every arm is one process: a load, the prefill, the steps. The references open the model through a
# file mapping; ours reads its host set at load (MADV_POPULATE_READ), outside its timer.
#
# Paging. The GGUF split set (199.7 GB) fits in the page cache (about 245 GB on this box), ours and
# the branches read the same file pages, so their arms rotate freely. The EXL3 directory (154 GB) and
# its CPU-tier copies do not fit beside it: the exllamav3 arms run as a block after every GGUF round,
# opened by a discarded process (`DISCARD r0`), so no GGUF row is timed after the EXL3 set evicted the
# file. Every ours and llama-bench arm runs once, discarded, right before its round-1 row, on the same
# ids (`WARMUP r0`, docs/fair-measure.md 2.1; a server arm's warm-up is its own discarded request, the
# exllamav3 block keeps its one discard; BLOOMERY_AB_WARMUP=0 skips them): the sitting's earlier
# segments leave another model's pages cached.
# The file does not stay whole in the cache on its own: our load drops the file pages of what it
# uploads (BLOOMERY_CARD_DONTNEED, 1 by default: the card trunk, 8.97 GB, and the card experts,
# 39.94 GB), so the next arm whose host set holds those pages reads them from the drive — ours in its
# populate, before its timer, llama-bench through its mapping, at its load and inside its timer.
# Preheat. Before every GGUF arm (the warm-up included; no exllamav3 arm) the runner reads a host set
# into the page cache with tools/ref/gguf-ranges.py (host, once per K before the lease; preheat, pread
# in chunks, the data discarded): token_embd and the routed experts of blocks 0..K-1. A llama.cpp
# arm's K is its own --n-cpu-moe (the MTP arm's GLM_NCMOE_MTP); ours' and a fit twin's is the
# profile's GLM_PREHEAT_K, every routed expert of the engine's blocks, a superset of ours' host set
# under any card rule and of the fit's trailing blocks' routed experts — gguf-ranges.py cannot read a
# plan, and the whole set fits beside the rest of the file. The read sits outside the arm's witness
# blocks, its wall and its fault counts, and prints `preheat <label> K=<k> bytes=<b> s=<t>
# gbps=<rate>`: the rate tells how much the arm before left uncached. For ours it moves the drive
# reads out of the load and makes the load's faults independent of the arm before; it cannot move the
# timed window, which starts after the populate. Flags whose host set the rule does not model (-ot,
# --override-tensor, -cmoe, --cpu-moe, a list for --n-cpu-moe) are refused before the lease.
# BLOOMERY_PREHEAT=0 turns the preheat off (1, the default; anything else is refused). A dry run
# prints each arm's K and the `host` line of every K, and reads nothing but the headers.
# Every row carries `majflt <n>` and ` [cold]` when those faults at COLD_US µs each (75.4: the serial
# 4 KB fault measured on this box, rig-log 2026-09-23, depth-ds41.sh's price) could be 1 % of the
# row's timed window or more. Ours counts from its `fed` record, which generate_glm5next prints just
# before its feed's timer starts, to its exit (majflt_mark, tools/ref/cold-blocks.sh), and prints the
# whole process's count beside it; an ours output with no `fed` line counts the whole process and says
# so. Every other row counts the change in /proc/vmstat pgmajfault across the arm's whole process
# (load and warm-up included, so an upper bound on its timed window's). exllamav3 rows print their
# count untagged: its load reads the 154 GB directory and copies the CPU experts into its own memory,
# faulting in every process, and perf.py's clock starts after that load, reading no file.
# A round's row that reads [cold] prints as `COLD r<r> …`, is in no mean, and its arm runs once more
# right away (its preheat included); a second [cold] prints `FAIL-cold r<r> …`, stays out of the means
# and the ratios, and joins the failed list (docs/fair-measure.md 2.3). Warm-up rows are not re-run.
#
# Placement. BLOOMERY_GEN_PLACE (a, the default; gate; bp) is the --place every ours and oursmtp arm runs
# at, unless the arm's own list names one: `place=<a|gate|bp>` among its NAME=VALUE items (`512@place=bp`,
# `512@place=a,BLOOMERY_RESIDENCY=mid-p33-s1`) runs that arm at that placement. The item is the runner's,
# never a variable the binary sees and no lever row (lever-arms.sh never reads it); the label keeps it
# (`ours@place=bp`), so `512 512@place=bp` is the same-binary placement A/B in the `ratio d=` table (ours /
# ours@place=bp), whose direction follows the arms' order. Any two arms of one engine whose variables are the
# same, in any order, and whose placements differ (`oursmtp:512 oursmtp:512@place=a`;
# `512@place=a,BLOOMERY_RESIDENCY=mid-p40-s1` and `512@BLOOMERY_RESIDENCY=mid-p40-s1,place=bp`) are paired by
# round in the `ratio place d=` table (`ratio place pp p=` for their prefill), ordered by placement alone: the
# later placement over the earlier (bp / a, gate / a), whatever the arms' order or BLOOMERY_GEN_PLACE, so its
# per-round list is what `tools/ref/card.py verdict` reads for a bp-against-a card. place= on a reference arm, a word outside a|gate|bp, an empty one and place given twice are
# refused by name before anything runs (tools/ref/arm-place.sh, shared with depth-ds41.sh). Plan (a) loads
# on the largest visible card (workstation::ALIASES: the binary finds its cards by device — on this box
# the A6000), the gate plan on the one card named 3090 (plan_gate), plan (b′) on the two largest
# (plan_bp: the 3090 the host tier's expert tier); the timing card is TIMING_GPU's, timing-card.sh's pick
# (BLOOMERY_TIMING_GPU overrides), never a name the binary sees. Under one card the arms see the timing
# card alone, so a placement whose card is not the timing card is refused (64) before the lease, and bp is
# refused outside the two-card mode. Every ours row names its placement (`place <p>`), one whose SMOKE
# names another is a FAIL row, and with a place= arm the [config] line has a `placements` line.
#
# Two cards. BLOOMERY_TIMING_CARDS=a6000+3090 (timing-card.sh's mode, depth-ds41.sh's) shows both cards to
# every ours arm, the A6000 as device 0 and the 3090 as device 1, for the separate "A6000+3090" table: every
# row's card field reads `A6000+3090`, so no reader puts it in the A6000 table. An ours arm at bp loads both
# cards; one at a loads plan (a) on the A6000 and leaves the 3090 idle; gate (a 3090-only row) is refused by
# name. After each ours arm that exited 0 its load record's `cards` (records.py, generate_glm5next's
# load_generator record; tools/ref/arm-place.sh place_arm_cards) must name the A6000 and the 3090 under bp and the A6000 alone under a, or the arm
# is a FAIL row; a load record with no cards field is one too. The reference arms stay on the A6000 alone:
# each runs with CUDA_VISIBLE_DEVICES the A6000's UUID, and a llama-bench or llama-server log that shows
# other than that one device is a FAIL row (exllamav3 prints no device lines; its CUDA_VISIBLE_DEVICES is
# the check). No reference arm here splits over both cards (no -ts arm for this model): the [config] line
# says so. The witness is depth-ds41.sh's: before the lease a card that does not answer, a 3090 off its
# 250 W cap or an unreadable kernel journal refuses the run; after every arm an Xid since the last arm, a
# card lost or off its cap fails the arm; a compute process on either card as an arm starts is waited out
# (10 minutes, then rc 75). A dry run prints the pre-lease checks' verdict and goes on.
#
# Contention. Before every arm: the other card (guard_other, ` [other-busy]`), the timing card
# (guard_timing: waits for another round's process on it, rc 75 after 10 minutes) and the CPU
# (guard_cpu, before and after, ` [cpu-busy]`: every engine here runs routed experts on the host
# cores). BLOOMERY_OTHER_STRICT=1 aborts on either tag instead.
#
# Failures. An arm that exits non-zero or prints no value prints `FAIL
# r<r> <label> <d|p>=<X> rc=<rc> | <why> | full output: <file>` where its row would be; the runner
# goes on, that label at that key drops out of the means and ratios, and the runner exits 1 at the end
# with the list.
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (default 2), BLOOMERY_GEN_WARM
# (--warm), BLOOMERY_GEN_CTX (C), BLOOMERY_GEN_BIN (default target/release/generate_glm5next),
# BLOOMERY_GEN_PLACE (a, the default, needs the A6000 as the timing card; gate the 3090; bp the two-card
# mode, BLOOMERY_TIMING_CARDS=a6000+3090: Placement and Two cards above),
# BLOOMERY_GEN_PAIR (1 passes --pair to our arms; unset or empty passes nothing),
# BLOOMERY_AB_WARMUP (1 or 0), BLOOMERY_PREHEAT (1 or 0, above), BLOOMERY_ARM_BOUND (seconds one
# arm, or one preheat, may run, default 900), BLOOMERY_DRY=1 (every arm's command line and preheat,
# the trees, the checks and each round's order, then exit 0 before the lease: nothing is loaded and
# nothing is timed).
set -uo pipefail
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
[ "$MODEL_NAME" = glm5next ] || {
  echo "depth-glm5next.sh: the profile is $MODEL_NAME — BLOOMERY_MODEL=glm5next on the Mac side" >&2
  exit 64
}
RECORDS="${BASH_SOURCE[0]%/*}/../bloomery/records.py"
N=${BLOOMERY_DECODE_N:-96}
ROUNDS=${BLOOMERY_AB_ROUNDS:-2}
WARM=${BLOOMERY_GEN_WARM:-}
CTX=${BLOOMERY_GEN_CTX:-2048}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_glm5next}
PLACE=${BLOOMERY_GEN_PLACE:-a}
PAIR=${BLOOMERY_GEN_PAIR:-}
BOUND=${BLOOMERY_ARM_BOUND:-900}
WARMUP=${BLOOMERY_AB_WARMUP:-1}
PREHEAT=${BLOOMERY_PREHEAT:-1}
# majflt_mark (the timed window's fault mark) is cold-blocks.sh's; sourced before COLD_US, which this
# runner keeps at its own value.
# shellcheck source=tools/ref/cold-blocks.sh
source "${BASH_SOURCE[0]%/*}/cold-blocks.sh" || exit 2
COLD_US=75.4
DRY=${BLOOMERY_DRY:-}
PROSE=$BLOOMERY_DATA/$GLM_PROSE
PROSE_FROM=$GLM_PROSE_FROM
case $PROSE_FROM in '' | *[!0-9]*) echo "depth-glm5next.sh: GLM_PROSE_FROM is an id count, got '$PROSE_FROM'" >&2; exit 64 ;; esac
for v in N:$N ROUNDS:$ROUNDS CTX:$CTX BOUND:$BOUND; do
  case ${v#*:} in '' | *[!0-9]* | 0*) echo "depth-glm5next.sh: ${v%%:*} is a positive integer, got '${v#*:}'" >&2; exit 64 ;; esac
done
case $PLACE in a | gate | bp) ;; *) echo "depth-glm5next.sh: BLOOMERY_GEN_PLACE is a (the default), gate or bp (the two-card mode's), got '$PLACE'" >&2; exit 64 ;; esac
case $PAIR in '' | 1) ;; *) echo "depth-glm5next.sh: BLOOMERY_GEN_PAIR is 1 or unset, got '$PAIR'" >&2; exit 64 ;; esac
case $WARMUP in 0 | 1) ;; *) echo "depth-glm5next.sh: BLOOMERY_AB_WARMUP is 0 or 1, got '$WARMUP'" >&2; exit 64 ;; esac
case $PREHEAT in 0 | 1) ;; *) echo "depth-glm5next.sh: BLOOMERY_PREHEAT is 1 (read each GGUF arm's host set before it, the default) or 0, got '$PREHEAT'" >&2; exit 64 ;; esac
case $WARM in '' | [0-9] | [1-9][0-9]*) ;; *) echo "depth-glm5next.sh: BLOOMERY_GEN_WARM is a count, got '$WARM'" >&2; exit 64 ;; esac

ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(512 lcpp27754:512)
# Per arm: the engine (ours, oursmtp, lcpp27752, lcpp27754, exl3), whether it is a prefill arm, its
# ubatch lever, its depth or prompt length, its row label (a prefill arm's names its ubatch, a lever arm's
# its variables) and a lever arm's NAME=VALUE list as given (comma-separated; empty for every other arm).
A_ENG=() A_PP=() A_UB=() A_DEP=() A_LABEL=() A_ENV=()
# An ours arm's placement (its place= item, else BLOOMERY_GEN_PLACE; empty for a reference arm) and whether
# place= set it (1, or empty). A_ENV is the arm's list less its place= item: the variables its process gets.
A_PLACE=() A_PLACE_SET=()
ours=0 ours_mtp=0 lcpp=0 exl3=0 gguf=0 srv=0 fit27752=0 fit27754=0 levers=0
usage() {
  echo "depth-glm5next.sh: arm '$1' is <D>[@NAME=VALUE,...], oursmtp:<D>[@NAME=VALUE,...], lcpp27752[fit]:<D>, lcpp27754[fit]:<D>, lcpp27752pp[fit][<U>]:<P>, lcpp27754pp[fit][<U>]:<P>, lcpp2775{2,4}{srv,mtp}[+t<N>][+nopo<0|1>][+k<K>]:<D>, lcpp2775{2,4}srvpp[<U>][+…]:<P>, exl3:<D> or exl3pp:<P>${2:+ — $2}" >&2
  exit 64
}
# A lever arm's list checks (the header's <D>@NAME=VALUE): tools/ref/lever-arms.sh, shared with
# depth-qwen3moe.sh; a refusal is this runner's usage line with its reason.
arm_refuse() { usage "$1" "$2"; }
LEVER_ARM_RUNNER=depth-glm5next.sh
LEVER_REGISTRY=${BASH_SOURCE[0]%/*}/../../crates/levers/src/registry.rs
# shellcheck source=tools/ref/lever-arms.sh
source "${BASH_SOURCE[0]%/*}/lever-arms.sh" || exit 2
# An ours arm's place= (the header's Placement): tools/ref/arm-place.sh, shared with depth-ds41.sh and
# depth-qwen3moe.sh; our load record is generate_glm5next's kind, read by its schema.
# shellcheck disable=SC2034 # read by tools/ref/arm-place.sh
PLACE_RUNNER=depth-glm5next.sh PLACE_BIN=generate_glm5next PLACE_WORDS='a gate bp' PLACE_LOAD_KIND=load_generator PLACE_LOAD_BIN=generate_glm5next
# shellcheck source=tools/ref/arm-place.sh
source "${BASH_SOURCE[0]%/*}/arm-place.sh" || exit 2
# The fit arms' flags, probe and column (lcpp-fit.sh); fit_eng names this runner's fit engines.
# shellcheck source=tools/ref/lcpp-fit.sh
source "${BASH_SOURCE[0]%/*}/lcpp-fit.sh" || exit 2
# The server arms' flags, start, requests, context check and stop, and the greedy cross-check.
# shellcheck source=tools/ref/lcpp-warm.sh
source "${BASH_SOURCE[0]%/*}/lcpp-warm.sh" || exit 2
# srv_glm <engine>: true for a server engine (srv, srvpp[<U>], mtp, each with its per-arm flags).
srv_glm() { case ${1%%+*} in lcpp2775[24]srv | lcpp2775[24]srvpp | lcpp2775[24]srvpp[1-9]* | lcpp2775[24]mtp) return 0 ;; *) return 1 ;; esac; }
fit_eng() { case $1 in lcpp2775[24]fit | lcpp2775[24]ppfit | lcpp2775[24]ppfit[1-9]*) return 0 ;; *) return 1 ;; esac; }
# gpu_flags <engine>: the profile's llama-bench flags of that engine's branch.
gpu_flags() { case $1 in lcpp27752*) echo "$LCPP27752_GPU_FLAGS" ;; *) echo "$LCPP27754_GPU_FLAGS" ;; esac; }
for a in "${ARMS[@]}"; do
  # `@` is ours only: split at it first, so a value with a `:` is not read as an engine's arm.
  arm_head=${a%%@*} envs='' at='' aplace=''
  eng=${arm_head%%:*} dep=${arm_head#*:} pp=0 ub=''
  [ "$arm_head" != "$eng" ] || { eng=ours dep=$arm_head; }
  if [ "$arm_head" != "$a" ]; then
    at=${a#*@}
    case $eng in
      ours | oursmtp) ;;
      *)
        ! arm_has_place "$at" || place_ref_refuse "$a" "${eng%%+*}"
        arm_refuse "$a" "'@' sets a lever of ours, and $eng: is a reference engine's arm — a lever of ours is not a reference's"
        ;;
    esac
    # The place= item is the runner's (tools/ref/arm-place.sh); the rest is the lever list, checked as before.
    [ -n "$at" ] || arm_envs_ok "$a" "$at"
    arm_place_split "$a" "$at"
    envs=$ARM_REST aplace=$ARM_PLACE
    [ -z "$envs" ] || arm_envs_ok "$a" "$envs"
    case ,$envs in
      *,BLOOMERY_DRAFT=*)
        if [ "$eng" = oursmtp ]; then
          arm_refuse "$a" "BLOOMERY_DRAFT is the oursmtp arm's own (it sets BLOOMERY_DRAFT=mtp): give the arm its other variables only"
        else
          arm_refuse "$a" "BLOOMERY_DRAFT is the oursmtp arm's own: the drafted arm is oursmtp:${dep}[@…], which sets BLOOMERY_DRAFT=mtp"
        fi
        ;;
    esac
    [ -z "$envs" ] || levers=1
  fi
  case $dep in '' | *[!0-9]*) usage "$a" ;; esac
  if [ "${eng%%+*}" != "$eng" ]; then
    srv_glm "$eng" || usage "$a" "+t<N>, +nopo<0|1> and +k<K> are a server arm's flags"
    srv_mods_split "$eng" || usage "$a" "$SRV_WHY"
  fi
  case ${eng%%+*} in
    ours)
      ours=1
      [ "$dep" -ge 1 ] && [ $((dep + N)) -le "$CTX" ] || usage "$a" "ours needs 1 <= D and D + N <= C ($N + D against --ctx $CTX)"
      ;;
    oursmtp)
      ours=1 ours_mtp=1
      [ "$dep" -ge 1 ] && [ $((dep + N + 1)) -le "$CTX" ] || usage "$a" "oursmtp needs 1 <= D and D + N + 1 <= C ($dep + $N + 1 = $((dep + N + 1)) against --ctx $CTX): the verify's last window takes one position past the N-th token"
      ;;
    lcpp27752 | lcpp27754) lcpp=1 ;;
    lcpp27752fit | lcpp27754fit) lcpp=1 ;;
    lcpp27752srv | lcpp27754srv | lcpp27752mtp | lcpp27754mtp | lcpp2775[24]srvpp | lcpp2775[24]srvpp[1-9]*)
      lcpp=1 srv=1 np=$N
      case ${eng%%+*} in
        *srvpp*)
          pp=1 np=1 ub=${eng%%+*} ub=${ub#lcpp2775?srvpp}
          case $ub in *[!0-9]* | 0*) usage "$a" ;; esac
          ;;
      esac
      [ "$dep" -ge 1 ] || usage "$a" "a server arm feeds D >= 1 prose ids"
      [ $((dep + np + 1)) -le "$CTX" ] || usage "$a" "the server's -c is ours' --ctx $CTX (docs/fair-measure.md 1.5), and $dep ids + n_predict $np + 1 take $((dep + np + 1)) positions; raise BLOOMERY_GEN_CTX"
      srv_mods_split "$eng"
      lcpp_srv_flags "$(srv_mods_apply "$(gpu_flags "$eng")" "$SRV_MODS")" || usage "$a" "$SRV_WHY"
      ;;
    lcpp27752pp* | lcpp27754pp*)
      lcpp=1 pp=1 ub=${eng#lcpp2775?pp}
      ub=${ub#fit}
      case $ub in *[!0-9]* | 0*) usage "$a" ;; esac
      [ "$dep" -ge 1 ] || usage "$a" "a prompt of 0 ids has no prefill to time"
      ;;
    exl3)
      exl3=1
      [ $((dep % 256)) = 0 ] || usage "$a" "perf.py measures depth 0 and multiples of 256"
      ;;
    exl3pp)
      exl3=1 pp=1
      [ "$dep" -ge 256 ] && [ "$dep" -le 4096 ] && [ $((dep % 256)) = 0 ] || usage "$a" "perf.py measures prompts of multiples of 256, up to one 4096 chunk here"
      ;;
    *) usage "$a" ;;
  esac
  if fit_eng "$eng"; then
    lcpp_fit_flags "$(gpu_flags "$eng")" || usage "$a" "$FIT_WHY"
    case $eng in lcpp27752*) fit27752=1 ;; *) fit27754=1 ;; esac
  fi
  case $eng in exl3*) ;; *) gguf=1 ;; esac
  A_ENG+=("$eng") A_PP+=("$pp") A_UB+=("$ub") A_DEP+=("$dep") A_LABEL+=("$eng${at:+@$at}") A_ENV+=("$envs")
  case $eng in ours | oursmtp) A_PLACE+=("${aplace:-$PLACE}") A_PLACE_SET+=("${aplace:+1}") ;; *) A_PLACE+=('') A_PLACE_SET+=('') ;; esac
done
# The draft is the oursmtp arm's alone: set here, every ours arm would draft and their ratio would read 1.
if [ "$ours" = 1 ] && [ -n "${BLOOMERY_DRAFT+set}" ]; then
  echo "depth-glm5next.sh: BLOOMERY_DRAFT is set to '$BLOOMERY_DRAFT' in the runner's environment, so the plain ours arms would run it too; unset it (the oursmtp:<D> arm sets BLOOMERY_DRAFT=mtp for its own command)" >&2
  exit 64
fi

# The card pin, the witness, the other-card guard; this runner has the two-card mode (the header's Two
# cards).
TIMING_CARDS_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
timing_cards_mode || exit $?
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
CPU_BUSY_COMMS=${BLOOMERY_CPU_BUSY_COMMS:-$CPU_BUSY_COMMS generate_glm5next generate_qwen3moe llama-server python3}
WITNESS=(head-open indent card busiest model mem pgmajfault)
T975=$(python3 "${BASH_SOURCE[0]%/*}/tdist.py" "$ROUNDS") || {
  echo "depth-glm5next.sh: tools/ref/tdist.py gave no t quantiles for ROUNDS=$ROUNDS" >&2
  exit 2
}
# Each ours arm's placement against the mode and the timing card (tools/ref/arm-place.sh): plan (a) is made
# for the largest visible card (the binary finds its cards by device; on this box the A6000), the gate plan
# for the one card named 3090, plan (b′) for the two largest in the two-card mode; the timing card is
# TIMING_GPU's, timing-card.sh's pick (BLOOMERY_TIMING_GPU overrides), never a name the binary sees.
# BLOOMERY_GEN_PLACE once for the arms that follow it, an arm's place= for that arm.
PLACE_SEEN=0 PLACES_SET='' PLACE_ARMS=()
for i in "${!ARMS[@]}"; do
  [ -n "${A_PLACE[$i]}" ] || continue
  PLACE_ARMS+=("$i")
  if [ -n "${A_PLACE_SET[$i]}" ]; then
    PLACES_SET=1
    place_check "${ARMS[$i]}" "${A_PLACE[$i]}"
  elif [ "$PLACE_SEEN" = 0 ]; then
    place_check '' "$PLACE"
    PLACE_SEEN=1
  fi
done
# The reference arms stay on the A6000 alone in the two-card mode (no -ts arm for this model): their
# processes get the A6000 alone as CUDA_VISIBLE_DEVICES (REF_PIN, REF_CVD), and their logs are held to one
# device, the A6000.
REF_PIN=() REF_CVD=$CUDA_VISIBLE_DEVICES
if [ -n "$TIMING_CARDS" ]; then
  REF_PIN=("CUDA_VISIBLE_DEVICES=$GPU_A6000") REF_CVD=$GPU_A6000
  CARD_NAME=$TIMING_CARDS_NAME
else
  CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
fi
# tc_config: the two-card mode's [config] and [dry] line.
tc_config() {
  echo "two cards: $TIMING_CARDS_NAME, our arms at their placements; the reference arms on the A6000 alone (CUDA_VISIBLE_DEVICES=$GPU_A6000), no -ts arm for this model"
}

# What every arm needs, checked before the lease (a dry run prints the findings and goes on). Ours:
# a binary no older than its sources that declares the timing records, the prose ids at their sha256.
# The references: their binaries, and exllamav3's venv, bench and model.
CHECKS=()
check() { # check <rc> <message>: fatal before a real run, printed in a dry one
  if [ -n "$DRY" ]; then CHECKS+=("$2"); else echo "depth-glm5next.sh: $2" >&2; exit "$1"; fi
}
if [ "$ours" = 1 ] || [ "$srv" = 1 ]; then
  if [ ! -f "$PROSE" ]; then
    check 66 "no prose ids at $PROSE"
  elif [ "$(sha256sum "$PROSE" | cut -d' ' -f1)" != "$GLM_PROSE_SHA256" ]; then
    check 65 "$PROSE is not the prose the profile pins (sha256 $GLM_PROSE_SHA256)"
  else
    for i in "${!ARMS[@]}"; do
      case ${A_ENG[$i]} in exl3* | lcpp2775[24] | lcpp2775[24]fit | lcpp2775[24]pp*) continue ;; esac
      [ $((PROSE_FROM + A_DEP[i])) -le "$(wc -l < "$PROSE")" ] || check 64 "arm ${ARMS[$i]}: $PROSE holds fewer than GLM_PROSE_FROM + ${A_DEP[$i]} = $((PROSE_FROM + A_DEP[i])) ids"
    done
  fi
fi
if [ "$ours" = 1 ]; then
  if [ -z "$DRY" ]; then assert_fresh_binary "$BIN" || exit $?; fi
  if [ -x "$BIN" ]; then
    kinds=$("$BIN" --records-schema 2> /dev/null | python3 -c 'import sys, json; print(" ".join(json.loads(l).get("kind", "") for l in sys.stdin if l.strip()))')
    for k in time_prompt time_step smoke plan; do
      case " $kinds " in *" $k "*) ;; *) check 2 "$BIN declares no '$k' record (--records-schema): it has no --time, so no ours arm can be timed" ;; esac
    done
  else
    check 2 "no binary at $BIN"
  fi
fi
# The oursmtp arm: its binary takes BLOOMERY_DRAFT=mtp (at_main refuses a lever the binary does not act
# on, --levers included), and the checked-in schema says whether its `mtp summary` record is read (MTP_REC
# 1) or none is (0), probed as cold-blocks.sh's res_kinds probes the residency records.
MTP_REC=0 MTP_NOTE=''
if [ "$ours_mtp" = 1 ]; then
  if [ -x "$BIN" ]; then
    lv=$(env BLOOMERY_DRAFT=mtp "$BIN" --levers 2>&1) || check 2 "$BIN refuses BLOOMERY_DRAFT=mtp (its --levers under it: ${lv##*$'\n'}): no oursmtp arm can run"
  fi
  if lv=$(python3 "$RECORDS" sh --bin generate_glm5next /dev/null 'MK=mtp_summary.kept' 'MP=mtp_summary.positions' 'MQ=mtp_summary.passes' 2>&1); then
    MTP_REC=1 MTP_NOTE="generate_glm5next's checked-in schema declares the mtp summary record: each oursmtp row carries its positions, passes and kept, and one that prints none is a FAIL row"
  else
    case $lv in
      *"prints no kind mtp_summary"*) MTP_NOTE="generate_glm5next's checked-in schema declares no mtp summary record: the oursmtp rows carry no draft fields" ;;
      *) check 2 "records.py could not read generate_glm5next's schema for the mtp summary record: $lv" ;;
    esac
  fi
fi
# arm_envs <i>: arm <i>'s own variables, into ARM_ENVS (NAME=VALUE words; none for an arm without a list).
arm_envs() {
  ARM_ENVS=()
  [ -z "${A_ENV[$1]}" ] || IFS=, read -r -a ARM_ENVS <<< "${A_ENV[$1]}"
}
# A lever arm's variables: its binary takes them (at_main refuses a lever the binary does not act on, and
# a value its kind does not take, --levers included), probed once per engine and list.
if [ "$levers" = 1 ] && [ -x "$BIN" ]; then
  probed='|'
  for i in "${!ARMS[@]}"; do
    [ -n "${A_ENV[$i]}" ] || continue
    case $probed in *"|${A_ENG[$i]}@${A_ENV[$i]}|"*) continue ;; esac
    probed+="${A_ENG[$i]}@${A_ENV[$i]}|"
    arm_envs "$i"
    draft=()
    [ "${A_ENG[$i]}" = ours ] || draft=(BLOOMERY_DRAFT=mtp)
    lv=$(env ${draft[@]+"${draft[@]}"} "${ARM_ENVS[@]}" "$BIN" --levers 2>&1) ||
      check 2 "arm ${ARMS[$i]}: $BIN refuses ${draft[*]:+${draft[*]} }${ARM_ENVS[*]} (its --levers under them: ${lv##*$'\n'}): the arm cannot run as written"
  done
fi
# The residency records (cold-blocks.sh's residency sums): G_RES 0 when generate_glm5next's checked-in
# schema declares them, 1 when it declares none (an arm that runs BLOOMERY_RESIDENCY then fails by name).
G_RES=1
if [ "$ours" = 1 ]; then
  G_RES=0
  res_kinds generate_glm5next || G_RES=$?
  [ "$G_RES" != 2 ] || check 2 "$RS_WHY"
fi
# res_asked <i>: the BLOOMERY_RESIDENCY arm <i> runs with (its own list's, else the runner's), empty when
# neither sets it.
res_asked() {
  local e v=${BLOOMERY_RESIDENCY:-}
  arm_envs "$1"
  for e in ${ARM_ENVS[@]+"${ARM_ENVS[@]}"}; do
    case $e in BLOOMERY_RESIDENCY=*) v=${e#*=} ;; esac
  done
  echo "$v"
}
# RS_NOTE: what the rows carry of the residency records, printed in [config] (and [dry]) when an arm runs
# BLOOMERY_RESIDENCY.
RS_NOTE=''
for i in "${!ARMS[@]}"; do
  case ${A_ENG[$i]} in ours | oursmtp) ;; *) continue ;; esac
  [ -n "$(res_asked "$i")" ] || continue
  if [ "$G_RES" = 0 ]; then
    RS_NOTE="generate_glm5next's checked-in schema declares the residency records: each ours row carries its residency lever and passes"
  else
    RS_NOTE="generate_glm5next's checked-in schema declares no residency lever record: every row of an arm that runs BLOOMERY_RESIDENCY is a FAIL row naming it"
  fi
  break
done
case " ${A_ENG[*]}" in *" lcpp27752"*) [ -x "$LCPP27752BIN" ] || check 2 "no llama-bench at $LCPP27752BIN (PR #27752's tree)" ;; esac
case " ${A_ENG[*]}" in *" lcpp27754"*) [ -x "$LCPP27754BIN" ] || check 2 "no llama-bench at $LCPP27754BIN (PR #27754's tree)" ;; esac
# A fit arm needs its branch's llama-bench to have the fit (lcpp_fit_probe runs its --help with no card).
# shellcheck disable=SC2153 # LCPP27752 and LCPP27754 are the profile's
{ [ "$fit27752" = 0 ] || [ ! -x "$LCPP27752BIN" ] || lcpp_fit_probe "$LCPP27752BIN"; } || check 64 "the lcpp27752fit/lcpp27752ppfit arms need llama-bench's fit: $FIT_WHY (tree $LCPP27752)"
{ [ "$fit27754" = 0 ] || [ ! -x "$LCPP27754BIN" ] || lcpp_fit_probe "$LCPP27754BIN"; } || check 64 "the lcpp27754fit/lcpp27754ppfit arms need llama-bench's fit: $FIT_WHY (tree $LCPP27754)"
if [ "$exl3" = 1 ]; then
  [ -x "$EXL3_PY" ] || check 2 "no exllamav3 python at $EXL3_PY"
  # shellcheck disable=SC2153 # EXL3 is the profile's
  [ -f "$EXL3/eval/perf.py" ] || check 2 "no bench at $EXL3/eval/perf.py"
  [ -f "$EXL3_MODEL/config.json" ] || check 2 "no EXL3 model at $EXL3_MODEL"
  [ -f "$EXL3_WIKITEXT" ] || check 2 "no wikitext-2 test text at $EXL3_WIKITEXT"
fi

# A binary's sha256 and its tree's HEAD and dirty count. GIT_OPTIONAL_LOCKS=0 keeps `git status` from
# rewriting the index of a tree this root process does not own.
tree_line() {
  local bin=$1 tree=$2 sha head dirty
  sha=$(sha256sum "$bin" 2> /dev/null | cut -c1-12)
  head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short=10 HEAD 2> /dev/null || echo '?')
  dirty=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
  echo "$bin sha256=${sha:-?} head=$head dirty_files=$dirty"
}
REF_LINES=()
for e in lcpp27752 lcpp27754; do
  case " ${A_ENG[*]} " in *" $e"*) ;; *) continue ;; esac
  if [ "$e" = lcpp27752 ]; then REF_LINES+=("$e: $(tree_line "$LCPP27752BIN" "$LCPP27752")"); else REF_LINES+=("$e: $(tree_line "$LCPP27754BIN" "$LCPP27754")"); fi
done
if [ "$exl3" = 1 ]; then
  v=$(find "${EXL3_PY%/bin/*}"/lib/python3*/site-packages -maxdepth 1 -name 'exllamav3-*.dist-info' 2> /dev/null | sed 's/.*exllamav3-//; s/\.dist-info$//' | head -n 1)
  REF_LINES+=("exl3: $EXL3_PY exllamav3 ${v:-?} bench $(tree_line "$EXL3/eval/perf.py" "$EXL3") model $EXL3_MODEL ($EXL3_BPW bpw)")
fi
ref_witness() { [ ${#REF_LINES[@]} -eq 0 ] || printf '    %s\n' "${REF_LINES[@]}"; }

# prompt_ids <D>: the D prompt ids, one a line: GLM_PROSE from 0-based index PROSE_FROM (the header's
# Arms). prompt_col <D>: the row's `prompt ids` column.
prompt_ids() { tail -n +"$((PROSE_FROM + 1))" "$PROSE" | head -n "$1"; }
prompt_col() {
  PROMPT_COL=" | prompt ids $PROSE_FROM..$((PROSE_FROM + $1 - 1))"
}
# The command of arm <i>, into CMD (an array, its environment first through env), LABEL_TEST
# (the row the reference's output is read by) and, for a fit arm, FIT_NOTE (what its flags dropped).
arm_cmd() {
  local i=$1 eng=${A_ENG[$1]} dep=${A_DEP[$1]} ub=${A_UB[$1]} flags server
  local -a words=() mtpw=()
  CMD=() LABEL_TEST='' BATCH='' FIT_NOTE='' SRV_ENVS=() SRV_NP=$N
  case ${eng%%+*} in
    ours | oursmtp)
      CMD=("$BIN" --tokens "$(prompt_ids "$dep" | paste -sd, -)" -n "$N" --ctx "$CTX" --place "${A_PLACE[$i]}" --time ${WARM:+--warm "$WARM"} ${PAIR:+--pair})
      # A lever arm's variables, after oursmtp's draft.
      arm_envs "$i"
      [ ${#ARM_ENVS[@]} -eq 0 ] || CMD=("${ARM_ENVS[@]}" "${CMD[@]}")
      if [ "$eng" = ours ]; then
        [ ${#ARM_ENVS[@]} -eq 0 ] || CMD=(env "${CMD[@]}")
      else
        CMD=(env BLOOMERY_DRAFT=mtp "${CMD[@]}")
      fi
      ;;
    lcpp2775[24]srv | lcpp2775[24]srvpp* | lcpp2775[24]mtp)
      # The branch's bench flags (its GLM_NCMOE_MTP for the MTP arm) with the arm's own, in the server's
      # spellings (lcpp-warm.sh), the MTP words after them, at -c C; #27754's environment.
      case $eng in
        lcpp27752*) server=$LCPP27752SRV ;;
        *) server=$LCPP27754SRV && read -r -a SRV_ENVS <<< "$LCPP27754_ENV" ;;
      esac
      SRV_ENVS=(${REF_PIN[@]+"${REF_PIN[@]}"} ${SRV_ENVS[@]+"${SRV_ENVS[@]}"})
      flags=$(gpu_flags "$eng")
      case ${eng%%+*} in *mtp) flags=$(with_ncmoe "$flags" "$GLM_NCMOE_MTP") ;; esac
      srv_mods_split "$eng"
      flags=$(srv_mods_apply "$flags" "$SRV_MODS")
      if [ "${A_PP[$i]}" = 1 ]; then
        SRV_NP=1 BATCH="ub 512 b 2048 (llama-server defaults)"
        if [ -n "$ub" ]; then
          flags="$flags -ub $ub -b $((ub > 2048 ? ub : 2048))" BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
        fi
      fi
      lcpp_srv_flags "$flags"
      read -r -a words <<< "$SRV_FLAGS"
      case ${eng%%+*} in *mtp) read -r -a mtpw <<< "$GLM_MTP_FLAGS" && words+=("${mtpw[@]}") ;; esac
      lcpp_srv_cmd "$server" "$MODEL" "$CTX" "${words[@]}"
      CMD=("${SRV_CMD[@]}")
      LABEL_TEST="POST /completion: GLM_PROSE ids $PROSE_FROM..$((PROSE_FROM + dep - 1)), a discarded warm-up then the same timed, n_predict $SRV_NP, temperature 0, ignore_eos, -c $CTX (ours' --ctx)"
      ;;
    lcpp*)
      # shellcheck disable=SC2206 # the profile's NAME=VALUE words
      case $eng in lcpp27752*) CMD=(env ${REF_PIN[@]+"${REF_PIN[@]}"} "$LCPP27752BIN") flags=$LCPP27752_GPU_FLAGS ;; *) CMD=(env ${REF_PIN[@]+"${REF_PIN[@]}"} $LCPP27754_ENV "$LCPP27754BIN") flags=$LCPP27754_GPU_FLAGS ;; esac
      if fit_eng "$eng"; then
        lcpp_fit_flags "$flags"
        flags=$FIT_FLAGS
        FIT_NOTE="placement: llama-bench's fit at -fitt $LCPP_FIT_TARGET MiB (dropped: $FIT_DROPPED), -v for the fit column"
      fi
      if [ "${A_PP[$i]}" = 1 ]; then
        CMD+=(-m "$MODEL" -p "$dep" -n 0 -r 1) LABEL_TEST="pp$dep |" BATCH="ub 512 b 2048 (llama-bench defaults)"
        if [ -n "$ub" ]; then
          CMD+=(-ub "$ub" -b "$((ub > 2048 ? ub : 2048))") BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
        fi
      else
        CMD+=(-m "$MODEL" -p 0 -n "$N" -d "$dep" -r 1) LABEL_TEST="tg$N @ d$dep |"
      fi
      # shellcheck disable=SC2206 # the profile keeps the flags as one string
      CMD+=($flags)
      ;;
    exl3)
      # shellcheck disable=SC2206
      CMD=(sudo -u user env --chdir=/ "CUDA_VISIBLE_DEVICES=$REF_CVD" "$EXL3_PY" "$EXL3/eval/perf.py" -m "$EXL3_MODEL" -spf --max_length "$((dep + 256))" $EXL3_FLAGS)
      LABEL_TEST="Context $dep:"
      ;;
    exl3pp)
      # shellcheck disable=SC2206
      CMD=(sudo -u user env --chdir=/ "CUDA_VISIBLE_DEVICES=$REF_CVD" "$EXL3_PY" "$EXL3/eval/perf.py" -m "$EXL3_MODEL" -sg --max_length "$dep" $EXL3_FLAGS)
      LABEL_TEST="Length $dep:"
      ;;
  esac
}
# A server arm needs its branch's llama-server, whose --help (run with no card) lists every flag the arm
# passes, and curl.
for i in "${!ARMS[@]}"; do
  srv_glm "${A_ENG[$i]}" || continue
  case ${A_ENG[$i]} in lcpp27752*) sb=$LCPP27752SRV ;; *) sb=$LCPP27754SRV ;; esac
  if [ ! -x "$sb" ]; then
    check 2 "no llama-server at $sb"
  else
    arm_cmd "$i"
    lcpp_srv_probe "$sb" || check 64 "arm ${ARMS[$i]}: $SRV_WHY"
  fi
done
[ "$srv" = 0 ] || command -v curl > /dev/null || check 2 "no curl for the server arms"
# exllamav3 runs as the tree's owner, from / (its CPU-tier worker process opens the working directory,
# which that user cannot enter under /root): perf.py keeps a disk cache beside itself, and its token stream
# reads the wikitext-2 test text from the temp dir, downloading it when absent; the staged copy
# keeps the network out of the lease.
stage_exl3() {
  local d=/tmp/llama_cpp_ppl_wikitext2/wikitext-2-raw
  [ -f "$d/wiki.test.raw" ] && return 0
  sudo -u user mkdir -p "$d" && sudo -u user cp "$EXL3_WIKITEXT" "$d/wiki.test.raw"
}

# The order: GGUF arms rotated by one slot each round, all their rounds, then the EXL3 arms likewise.
order_of() { # order_of <round> <exl3 0|1>: the indices of that group's arms in that round's order
  local r=$1 want=$2 i
  local -a idx=()
  for i in "${!ARMS[@]}"; do
    case ${A_ENG[$i]} in exl3*) [ "$want" = 1 ] && idx+=("$i") ;; *) [ "$want" = 0 ] && idx+=("$i") ;; esac
  done
  local n=${#idx[@]} k
  for ((k = 0; k < n; k++)); do printf '%s ' "${idx[$(((k + r - 1) % n))]}"; done
}
first_of() { local o; o=$(order_of 1 "$1"); echo "${o%% *}"; }

# The preheat's plan, before the lease (the header's Preheat): per arm A_PHK, the K of the host set it
# reads (empty for an exllamav3 arm, with BLOOMERY_PREHEAT=0, or in a dry run whose ranges failed), and
# per K and -ngl gguf-ranges.py's `host` line PH_LINE_<K>_<ngl> and the ranges file PH_DIR/k<K>.tsv.
# PH_RATE prices a cold preheat in a dry run: GB/s, depth-ds41.sh's NVMe populate rate, the slow end.
GGUF_RANGES="${BASH_SOURCE[0]%/*}/gguf-ranges.py"
PH_RATE=1.33
PH_DIR='' A_PHK=()
# ph_k <index>: arm <index>'s host set K and -ngl into PHK and PHNGL (PHK empty: no preheat). Flags whose
# host set the rule does not model exit 64.
ph_k() {
  local i=$1 w prev='' k=0 ngl='' fit=0
  PHK='' PHNGL=''
  case ${A_ENG[$i]} in
    exl3*) return 0 ;;
    ours | oursmtp) PHK=$GLM_PREHEAT_K; return 0 ;;
  esac
  arm_cmd "$i"
  for w in "${CMD[@]}"; do
    case $w in
      -ot | --override-tensor | --override-tensor=* | -cmoe | --cpu-moe)
        echo "depth-glm5next.sh: arm '${ARMS[$i]}': its flags carry $w, a host set the preheat does not model; set BLOOMERY_PREHEAT=0 to run it unpreheated" >&2
        exit 64
        ;;
      -fitt | --fit-target) fit=1 ;;
    esac
    case $prev in
      --n-cpu-moe | -ncmoe) k=$w ;;
      -ngl | --n-gpu-layers | --gpu-layers) ngl=$w ;;
    esac
    prev=$w
  done
  if [ "$fit" = 1 ]; then
    PHK=$GLM_PREHEAT_K
    return 0
  fi
  case $k in '' | *[!0-9]*) echo "depth-glm5next.sh: arm '${ARMS[$i]}': --n-cpu-moe '$k' is not one layer count (a list runs several placements)" >&2; exit 64 ;; esac
  case $ngl in *[!0-9]*) echo "depth-glm5next.sh: arm '${ARMS[$i]}': -ngl '$ngl' is not one layer count" >&2; exit 64 ;; esac
  PHK=$k PHNGL=$ngl
}
if [ "$PREHEAT" = 1 ] && [ "$gguf" = 1 ]; then
  PH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/depth-glm5next-preheat.XXXXXX") || exit 2
  trap 'rm -rf "$PH_DIR"' EXIT
  for i in "${!ARMS[@]}"; do
    ph_k "$i"
    A_PHK[i]=$PHK
    [ -n "$PHK" ] || continue
    var=PH_LINE_${PHK}_${PHNGL:-none}
    [ -z "${!var:-}" ] || continue
    # shellcheck disable=SC2086 # an empty PHNGL adds nothing
    if line=$(python3 "$GGUF_RANGES" host "$MODEL" --n-cpu-moe "$PHK" ${PHNGL:+--ngl "$PHNGL"} --out "$PH_DIR/k$PHK.tsv"); then
      printf -v "$var" '%s' "$line"
    else
      rc=$?
      check "$rc" "arm '${ARMS[$i]}': no preheat ranges for K=$PHK (tools/ref/gguf-ranges.py rc $rc)"
      A_PHK[i]=''
    fi
  done
fi
# ph_bytes <K> <ngl>: the host set's bytes from its `host` line.
ph_bytes() {
  local var=PH_LINE_${1}_${2:-none} w
  for w in ${!var}; do case $w in bytes=*) echo "${w#bytes=}" ;; esac; done
}
# preheat_arm <index>: the host set of arm <index>'s K into the page cache (gguf-ranges.py preheat, under
# the arm bound) and its `preheat` line; non-zero with FAIL_WHY on a failure. Nothing without a plan.
preheat_arm() {
  local k=${A_PHK[$1]:-} out rc=0
  [ -n "$PH_DIR" ] && [ -n "$k" ] || return 0
  out=$(timeout --kill-after=10 "$BOUND" python3 "$GGUF_RANGES" preheat "$PH_DIR/k$k.tsv" 2>&1) || rc=$?
  if [ "$rc" -ne 0 ]; then
    FAIL_WHY="its preheat failed (rc $rc): ${out##*$'\n'}"
    return "$rc"
  fi
  echo "preheat ${A_LABEL[$1]} K=$k $out"
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL n=$N rounds=$ROUNDS ctx=$CTX place=$PLACE warm=${WARM:-0} card=$CARD_NAME timing_gpu=$TIMING_GPU arm_bound=${BOUND}s warmup=$WARMUP cold_us=$COLD_US"
  echo "[dry] ours: $BIN prose=$PROSE"
  [ -z "$PLACES_SET" ] || echo "[dry] placements: $(place_line "${PLACE_ARMS[@]}")"
  if [ -n "$TIMING_CARDS" ]; then
    echo "[dry] $(tc_config)"
    tc_rc=0
    timing_cards_precheck '[dry] ' || tc_rc=$?
    if [ "$tc_rc" = 0 ]; then
      echo "[dry] two-card precheck: ok"
    else
      echo "[dry] two-card precheck: refused (rc $tc_rc): $TWOCARD_WHY — a real run stops here, before the lease"
    fi
  fi
  [ "$ours_mtp" = 0 ] || echo "[dry] oursmtp: ours' command under env BLOOMERY_DRAFT=mtp; ${MTP_NOTE:-the schema of the mtp summary record was not read (the check lines)}"
  [ -z "$RS_NOTE" ] || echo "[dry] residency: $RS_NOTE"
  echo "[dry] prompt: GLM_PROSE ids from index $PROSE_FROM"
  ref_witness | sed 's/^   /[dry]/'
  [ ${#CHECKS[@]} -eq 0 ] || printf '[dry] check: %s\n' "${CHECKS[@]}"
  for i in "${!ARMS[@]}"; do
    arm_cmd "$i"
    row=${LABEL_TEST% |}
    echo "[dry] ${ARMS[$i]}:${row:+ row \"$row\"}${BATCH:+, $BATCH}${FIT_NOTE:+, $FIT_NOTE}"
    pre="timeout --kill-after=10 $BOUND " post=''
    ! srv_glm "${A_ENG[$i]}" || [ ${#SRV_ENVS[@]} -eq 0 ] || pre+="env ${SRV_ENVS[*]} "
    echo "[dry]     $pre$(printf '%q ' "${CMD[@]}" | sed -E 's/--tokens [^ ]+/--tokens <GLM_PROSE ids '"$PROSE_FROM..$((PROSE_FROM + A_DEP[i] - 1))"'>/')$post"
    [ -n "${A_PHK[$i]:-}" ] || continue
    ph_k "$i"
    b=$(ph_bytes "$PHK" "$PHNGL")
    ph_round=$((${ph_round:-0} + b))
    echo "[dry]     preheat K=$PHK: $b B ($(awk -v b="$b" 'BEGIN { printf "%.1f", b / 1e9 }') GB), $(awk -v b="$b" -v r="$PH_RATE" 'BEGIN { printf "%.0f", b / 1e9 / r }') s if all of it is cold at $PH_RATE GB/s"
  done
  if [ -n "$PH_DIR" ]; then
    for var in ${!PH_LINE_*}; do echo "[dry] preheat: ${!var}"; done
    echo "[dry] preheat: ${ph_round:-0} B a round over the GGUF arms, $(awk -v b="${ph_round:-0}" -v r="$PH_RATE" 'BEGIN { printf "%.0f", b / 1e9 / r }') s a round if every byte is cold at $PH_RATE GB/s (the upper end: a set the arm before left cached reads at page-cache speed)"
  else
    echo "[dry] preheat: off ($([ "$PREHEAT" = 0 ] && echo BLOOMERY_PREHEAT=0 || echo 'no GGUF arm'))"
  fi
  [ "$srv" = 0 ] || echo "[dry] server arms: -c $CTX (ours' --ctx), a discarded request then the timed one; each row's first $XC_N ids held to ours' on the same ids (xcheck)"
  [ "$gguf" = 0 ] || [ "$WARMUP" = 0 ] || echo "[dry] WARMUP r0: every ours and llama-bench arm once on its own ids, discarded, right before its round-1 row (a server arm's warm-up is its own discarded request)"
  for r in $(seq "$ROUNDS"); do
    o='' ; for i in $(order_of "$r" 0); do o+="${ARMS[$i]} "; done
    [ "$gguf" = 0 ] || echo "[dry] round $r gguf order: $o"
  done
  [ "$exl3" = 0 ] || [ "$WARMUP" = 0 ] || echo "[dry] DISCARD r0: ${ARMS[$(first_of 1)]} (discarded, opens the exllamav3 block)"
  for r in $(seq "$ROUNDS"); do
    o='' ; for i in $(order_of "$r" 1); do o+="${ARMS[$i]} "; done
    [ "$exl3" = 0 ] || echo "[dry] round $r exl3 order: $o"
  done
  exit 0
fi

[ "$exl3" = 0 ] || stage_exl3 || { echo "depth-glm5next.sh: could not stage $EXL3_WIKITEXT for perf.py" >&2; exit 2; }

# A compute process on the timing card as an arm starts: another round's functional run. Every arm
# loads onto that card, so wait for it, polling every 10 s, and stop the runner after 10 minutes
# (rc 75: contention, not a result).
guard_timing() {
  local apps i
  for ((i = 0; i < 60; i++)); do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$TIMING_GPU")
    if [ -z "$apps" ]; then
      [ "$i" = 0 ] || echo "[timing-busy] $(now) the timing card is free after $((i * 10)) s" >&2
      return 0
    fi
    [ "$i" != 0 ] || echo "[timing-busy] $(now) compute apps on the timing card: [$(echo "$apps" | tr '\n' ';')]; waiting up to 10 min" >&2
    sleep 10
  done
  witness abort-timing >&2
  exit 75
}
majflt() { awk '$1 == "pgmajfault" { print $2 }' /proc/vmstat; }
strip() { sed 's/\x1b\[[0-9;]*m//g' | tr '\r' '\n'; }
# cold_col <faults> <timed window, s> [<what the count spans>]: the majflt column, and COLD_TAG when the
# faults could cost 1 % of the window. The span defaults to the whole process.
cold_col() {
  local f=$1 w=$2 pct
  pct=$(awk -v f="$f" -v us="$COLD_US" -v w="$w" 'BEGIN { printf "%.1f", (w > 0) ? 100 * f * us / 1e6 / w : 0 }')
  COLD_TAG=''
  awk -v p="$pct" 'BEGIN { exit !(p >= 1) }' && COLD_TAG=' [cold]'
  MAJ_COL=" | majflt $f (${3:-whole process}) <= $pct % of the timed window"
}

sums=() pp_sums=() failed=()
n_rows=0 busy_rows=0 other_rows=0 cold_rows=0 retry_rows=0 fail_cold=0
# cold_pass <tag> <round> <index>: PTAG, the word the row prints under the cold rule (the header):
# <tag> for a row that is not [cold] or not a round's; COLD on a round row's first pass, which sets
# COLD_AGAIN so run_row runs the arm once more; FAIL-cold on its second, which joins the failed list.
# Only a ROW row goes into the sums.
COLD_PASS=0 COLD_AGAIN=0
cold_pass() {
  local key=d
  PTAG=$1
  [ "$1" = ROW ] && [ -n "$COLD_TAG" ] || return 0
  if [ "$COLD_PASS" = 0 ]; then
    PTAG=COLD COLD_AGAIN=1
  else
    [ "${A_PP[$3]}" = 0 ] || key=p
    PTAG=FAIL-cold
    failed+=("r$2:${A_LABEL[$3]}@$key=${A_DEP[$3]}(cold)")
    fail_cold=$((fail_cold + 1))
  fi
}
# fail_row <tag> <round> <index> <rc> <why> <output>
fail_row() {
  local f=${TMPDIR:-/tmp}/depth-glm5next-${A_LABEL[$3]//\//_}-${A_DEP[$3]}-r$2.log key=d
  [ "${A_PP[$3]}" = 0 ] || key=p
  echo "$6" > "$f"
  echo "FAIL r$2 ${A_LABEL[$3]} $key=${A_DEP[$3]} rc=$4 | $5 | full output: $f$CPU_BUSY_TAG$OTHER_BUSY_TAG"
  echo "$6" | tail -n 12 | sed 's/^/    /' >&2
  failed+=("r$2:${A_LABEL[$3]}@$key=${A_DEP[$3]}")
}

# srv_start_arm <index> <depth>: a server arm's process (lcpp-warm.sh: SRV_CMD under its own bound,
# #27754's environment, a port the kernel picks), its context against -c C, the discarded warm-up request
# and the timed one, then the server stopped by the pid it was started with, on every path. out is the
# server's log; SRV_RUN_WHY and SRV_FAIL_RC say what failed (empty: every step ran), the timed request's
# SRV_* its values.
srv_start_arm() {
  local ids log
  ids=$(prompt_ids "$2" | paste -sd, -)
  log=$(mktemp "${TMPDIR:-/tmp}/depth-glm5next-server.XXXXXX") || exit 2
  SRV_RUN_WHY='' SRV_FAIL_RC=0
  if ! lcpp_srv_start "$BOUND" "$log" ${SRV_ENVS[@]+"${SRV_ENVS[@]}"}; then
    SRV_FAIL_RC=$SRV_RC SRV_RUN_WHY="-c $CTX (ours' --ctx): $SRV_WHY"
  elif ! lcpp_srv_ctx_check "$log" "$CTX"; then
    SRV_RUN_WHY=$SRV_WHY
  elif ! lcpp_srv_arm "$ids" "$SRV_NP" "$BOUND"; then
    SRV_RUN_WHY=$SRV_WHY
  fi
  lcpp_srv_stop
  out=$(cat "$log")
  rm -f "$log"
}

# run_arm <tag> <round> <index>: tag is ROW (a round's arm), WARMUP or DISCARD (discarded rows).
run_arm() {
  local tag=$1 r=$2 i=$3 eng=${A_ENG[$3]} dep=${A_DEP[$3]} label=${A_LABEL[$3]} out rc t0 t1 m0 m1 val w rowtags why line markf mark
  CPU_BUSY_TAG='' FIT_COL='' FIT_LINES=''
  guard_other
  # The two-card mode's guard_other waits on both cards (guard_cards); one card's waits on the timing card.
  [ -n "$TIMING_CARDS" ] || guard_timing
  preheat_arm "$i" || {
    rc=$?
    [ "$tag" != ROW ] || n_rows=$((n_rows + 1))
    if [ "$tag" = ROW ]; then fail_row "" "$r" "$i" "$rc" "$FAIL_WHY" ""; else fail_row "" 0 "$i" "$rc" "$FAIL_WHY" ""; fi
    return
  }
  guard_cpu "pre r$r $label $dep"
  arm_cmd "$i"
  witness "pre $tag r$r $label ${dep}"
  ref_witness
  m0=$(majflt) t0=$(date +%s)
  case ${eng%%+*} in
    lcpp2775[24]srv | lcpp2775[24]srvpp* | lcpp2775[24]mtp)
      srv_start_arm "$i" "$dep"
      rc=0
      ;;
    ours | oursmtp)
      # Through majflt_mark: the fault count at the `fed` line, where the feed's timer starts.
      markf=$(mktemp "${TMPDIR:-/tmp}/depth-glm5next-fed.XXXXXX") || exit 2
      out=$(lease_bounded "$BOUND" "${CMD[@]}" 2>&1 | majflt_mark "$markf" '^fed '; exit "${PIPESTATUS[0]}")
      rc=$?
      mark=$(cat "$markf")
      rm -f "$markf"
      ;;
    *)
      out=$(lease_bounded "$BOUND" "${CMD[@]}" 2>&1)
      rc=$?
      ;;
  esac
  t1=$(date +%s) m1=$(majflt)
  witness "post $tag r$r $label ${dep}"
  guard_cpu "post r$r $label $dep"
  [ "$tag" != ROW ] || {
    n_rows=$((n_rows + 1))
    [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
    [ -z "$OTHER_BUSY_TAG" ] || other_rows=$((other_rows + 1))
  }
  local r_tag=r$r tc
  [ "$tag" = ROW ] || r_tag=r0
  # Two cards: an Xid, a card lost or off its cap, or an engine off the cards it was given fails the arm.
  # Ours is held to its load record's cards (bp both, a the A6000 alone) once it exited 0; a reference,
  # given the A6000 alone, to its log's one device (exllamav3 prints none: its CUDA_VISIBLE_DEVICES is the
  # check).
  if [ -n "$TIMING_CARDS" ]; then
    case ${eng%%+*} in
      ours | oursmtp)
        if [ "$rc" = 0 ] && ! place_arm_cards "$out" "${A_PLACE[$i]}"; then
          fail_row "" "${r_tag#r}" "$i" "$rc" "two cards: $TWOCARD_WHY" "$out"
          return
        fi
        ;;
      *)
        case $eng in exl3*) tc=none ;; *) tc=stage ;; esac
        timing_cards_arm "$out" "$tc" || { fail_row "" "${r_tag#r}" "$i" "$rc" "two cards: $TWOCARD_WHY" "$out"; return; }
        ;;
    esac
  fi
  # A fit arm's loader lines say what the fit chose; a fit that failed or never ran fails the arm by
  # name, whatever llama-bench measured or how it exited after it.
  if fit_eng "$eng" && ! lcpp_fit_col "$out"; then
    why=$FIT_WHY
    [ "$rc" = 0 ] || why="exited $rc; $why"
    fail_row "" "${r_tag#r}" "$i" "$rc" "$why" "$out"
    return
  fi
  if [ "$rc" -ne 0 ]; then fail_row "" "${r_tag#r}" "$i" "$rc" "exited $rc" "$out"; return; fi
  case ${eng%%+*} in
    ours | oursmtp)
      local rec P50 MEAN WARMCOL PLACE_RAN SERIES TOKENS PP_N PP_MS PP_TPS PP_PASSES PP_KIND CARD_EXP HOST_EXP XTOK
      rec=$(python3 "$RECORDS" sh --bin generate_glm5next - P50=smoke.p50_ms MEAN=smoke.mean_ms WARMCOL=smoke.warm \
        PLACE_RAN=smoke.place 'SERIES=time_step.ms*' 'TOKENS=step.token*' PP_N=time_prompt.n PP_MS=time_prompt.ms \
        'PP_TPS=time_prompt.tok/s' PP_PASSES=time_prompt.passes PP_KIND=time_prompt.kind CARD_EXP=plan.card_experts \
        HOST_EXP=plan.host_experts XTOK=tokens.tokens LCTX=load_generator.ctx <<< "$out") || { fail_row "" "${r_tag#r}" "$i" 0 "records.py did not read the output" "$out"; return; }
      eval "$rec"
      [ -n "$P50" ] && [ -n "$PP_N" ] || { fail_row "" "${r_tag#r}" "$i" 0 "no SMOKE or time prompt record" "$out"; return; }
      [ "$PLACE_RAN" = "${A_PLACE[$i]}" ] || { fail_row "" "${r_tag#r}" "$i" 0 "its SMOKE names place=$PLACE_RAN; the runner passed --place ${A_PLACE[$i]}" "$out"; return; }
      # The context the servers are given is the one this load made (docs/fair-measure.md 1.5).
      [ "$LCTX" = "$CTX" ] || { fail_row "" "${r_tag#r}" "$i" 0 "its load record names ctx=${LCTX:-none}, and the server arms run at -c $CTX (BLOOMERY_GEN_CTX, ours' --ctx)" "$out"; return; }
      # The oursmtp row's draft fields: the mtp summary record, when the schema declares it (MTP_REC).
      local MK='' MP='' MQ='' MTP_COL='' mrec mlines
      if [ "$eng" = oursmtp ] && [ "$MTP_REC" = 1 ]; then
        if ! mrec=$(python3 "$RECORDS" sh --bin generate_glm5next - 'MK=mtp_summary.kept' 'MP=mtp_summary.positions' 'MQ=mtp_summary.passes' <<< "$out" 2>&1) ||
          ! mlines=$(python3 "$RECORDS" lines --bin generate_glm5next - mtp_summary <<< "$out" 2>&1); then
          fail_row "" "${r_tag#r}" "$i" 0 "records.py did not read its mtp summary record: $mrec${mlines:+ $mlines}" "$out"
          return
        fi
        eval "$mrec"
        [ -n "$mlines" ] || { fail_row "" "${r_tag#r}" "$i" 0 "the oursmtp arm printed no mtp summary record, which its schema declares" "$out"; return; }
        [ -n "$MK" ] && [ -n "$MP" ] && [ -n "$MQ" ] || { fail_row "" "${r_tag#r}" "$i" 0 "its mtp summary record has no kept, positions or passes (kept='$MK' positions='$MP' passes='$MQ')" "$out"; return; }
        MTP_COL=" | mtp positions/pass $(awk -v p="$MP" -v q="$MQ" 'BEGIN { printf "%.3f", (q > 0) ? p / q : 0 }') = positions $MP / passes $MQ, kept $MK"
      fi
      # The residency records: the lever and the timed passes' sums (cold-blocks.sh's residency sums).
      RS_WORD='' RS_COL=''
      if [ "$G_RES" = 0 ]; then
        res_sums generate_glm5next "$out" || { fail_row "" "${r_tag#r}" "$i" 0 "$RS_WHY" "$out"; return; }
        if [ -z "$RS_WORD" ] && [ -n "$(res_asked "$i")" ]; then
          fail_row "" "${r_tag#r}" "$i" 0 "the arm runs BLOOMERY_RESIDENCY=$(res_asked "$i") and printed no residency lever record, which its schema declares" "$out"
          return
        fi
      elif [ -n "$(res_asked "$i")" ]; then
        fail_row "" "${r_tag#r}" "$i" 0 "the arm runs BLOOMERY_RESIDENCY=$(res_asked "$i"), and generate_glm5next's checked-in schema (tools/bloomery/schema/generate_glm5next.jsonl) declares no residency record, so its row cannot carry them: refresh the schema from the binary that prints them (just records-refresh)" "$out"
        return
      fi
      echo "$out" | grep -E '^(plan|load|capture|fed|step 0|time prompt) '
      local h10 t10 uniq tps tps50
      h10=$(echo "$SERIES" | head -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
      t10=$(echo "$SERIES" | tail -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
      uniq=$(echo "$TOKENS" | sort -u | grep -c .)
      tps=$(awk -v m="$MEAN" 'BEGIN{printf "%.2f", 1e3/m}')
      tps50=$(awk -v p="$P50" 'BEGIN{printf "%.2f", 1e3/p}')
      w=$(awk -v a="$PP_MS" -v m="$MEAN" -v n="$N" 'BEGIN{print (a + (n - 1) * m) / 1e3}')
      if [ -n "$mark" ]; then
        cold_col "$((m1 - mark))" "$w" "timed, from the fed line; whole process $((m1 - m0))"
      else
        cold_col "$((m1 - m0))" "$w" "whole process: no fed line"
      fi
      cold_pass "$tag" "$r" "$i"
      prompt_col "$dep"
      rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
      echo "$PTAG $r_tag $label d=$dep n=$N ctx=$CTX | tok/s(mean) $tps @ n=$N, depth $dep, $CARD_NAME | place $PLACE_RAN card_experts $CARD_EXP host_experts $HOST_EXP$PROMPT_COL | p50 $P50 ms | mean $MEAN ms | tok/s(p50) $tps50 | warm ${WARMCOL:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq | pp_tok/s $PP_TPS (n=$PP_N, passes=$PP_PASSES, kind=$PP_KIND)$MTP_COL$RS_COL$MAJ_COL | wall $((t1 - t0))s$rowtags"
      [ "$PTAG" = ROW ] || { cold_count "$tag"; return 0; }
      sums+=("$label|$dep|$r|$tps")
      pp_sums+=("$label|$PP_N|$r|$PP_TPS")
      res_sums_add "$label" "$dep" "$r"
      # The drafted run's ids are held to ours' as a server row's are; it is never the ours side. A lever
      # arm is in no cross-check: its lever may move the arithmetic.
      case $label in
        ours) xc_add ours prose "$dep" "$r" ours "$XTOK" ;;
        oursmtp) xc_add srv prose "$dep" "$r" "$label" "$XTOK" ;;
      esac
      ;;
    lcpp2775[24]srv | lcpp2775[24]srvpp* | lcpp2775[24]mtp)
      local acc prps wtps w
      [ -z "$SRV_RUN_WHY" ] || { fail_row "" "${r_tag#r}" "$i" "$SRV_FAIL_RC" "$SRV_RUN_WHY" "$out"; return; }
      acc=$(grep -o 'draft acceptance = .*' <<< "$out" | tail -n 1)
      case ${eng%%+*} in *mtp) [ "${SRV_DRAFT_N:-0}" -gt 0 ] || { fail_row "" "${r_tag#r}" "$i" 0 "the MTP arm drafted nothing (draft_n ${SRV_DRAFT_N:-0})" "$out"; return; } ;; esac
      echo "$out" | grep -E '^build:|model buffer size|speculative|draft-mtp|nextn|llama_context: n_ctx ' | head -n 8 | sed "s/^/    $label load /"
      wtps=$(awk -v v="$WARM_TPS" 'BEGIN{printf "%.2f", v}')
      prps=$(awk -v v="$SRV_PROMPT_TPS" 'BEGIN{printf "%.2f", v}')
      # The timed window: the timed request's prompt and decode; its faults counted around it alone.
      w=$(awk -v p="$SRV_PROMPT_MS" -v g="$SRV_PRED_MS" 'BEGIN { print (p + g) / 1e3 }')
      cold_col "$((SRV_F1 - SRV_F0))" "$w" "timed: the timed request; whole process $((m1 - m0))"
      cold_pass "$tag" "$r" "$i"
      prompt_col "$dep"
      rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
      if [ "${A_PP[$i]}" = 1 ]; then
        echo "$PTAG $r_tag $label p=$dep n=0 | tok/s(pp) $prps @ n=0, prompt $dep, $CARD_NAME | llama-server /completion$PROMPT_COL | warm-up $wtps tok/s(pp) majflt $WARM_FAULTS, continuation $SRV_SAME | -c $CTX | $BATCH$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$PTAG" = ROW ] || { cold_count "$tag"; return 0; }
        pp_sums+=("$label|$dep|$r|$prps")
      else
        val=$(awk -v v="$SRV_PRED_TPS" 'BEGIN{printf "%.2f", v}')
        echo "$PTAG $r_tag $label d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME | llama-server /completion$PROMPT_COL | warm-up $wtps tok/s majflt $WARM_FAULTS, continuation $SRV_SAME | -c $CTX | prompt_n $SRV_PROMPT_N prompt tok/s $prps | draft_n $SRV_DRAFT_N draft_n_accepted $SRV_DRAFT_ACC | ${acc:-no draft acceptance line}$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$PTAG" = ROW ] || { cold_count "$tag"; return 0; }
        sums+=("$label|$dep|$r|$val")
        pp_sums+=("$label|$dep|$r|$prps")
      fi
      xc_add srv prose "$dep" "$r" "$label" "$SRV_TOKENS"
      ;;
    lcpp*)
      val=$(echo "$out" | grep -F "$LABEL_TEST" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//; s/ //g' | head -n 1)
      [ -n "$val" ] || { fail_row "" "${r_tag#r}" "$i" 0 "no '${LABEL_TEST% |}' row" "$out"; return; }
      echo "$out" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $label table /"
      [ -z "$FIT_COL" ] || while IFS= read -r line; do echo "    $label fit $line"; done <<< "$FIT_LINES"
      local build dev
      build=$(echo "$out" | sed -n 's/^build: //p' | head -n 1)
      dev=$(echo "$out" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
      if [ "${A_PP[$i]}" = 1 ]; then
        cold_col "$((m1 - m0))" "$(awk -v p="$dep" -v v="$val" 'BEGIN{print p / v}')"
        cold_pass "$tag" "$r" "$i"
        rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
        echo "$PTAG $r_tag $label p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME$FIT_COL | $BATCH | build ${build:-?} | device ${dev:-?}$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$PTAG" = ROW ] && pp_sums+=("$label|$dep|$r|$val")
      else
        cold_col "$((m1 - m0))" "$(awk -v n="$N" -v v="$val" 'BEGIN{print n / v}')"
        cold_pass "$tag" "$r" "$i"
        rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG"
        echo "$PTAG $r_tag $label d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME$FIT_COL | build ${build:-?} | device ${dev:-?}$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$PTAG" = ROW ] && sums+=("$label|$dep|$r|$val")
      fi
      ;;
    exl3*)
      local clean
      clean=$(echo "$out" | strip)
      val=$(echo "$clean" | grep -E "^${LABEL_TEST%%[0-9]*} +$dep:" | grep -oE '[0-9]+\.[0-9]+ +tokens/s' | head -n 1 | sed 's/ .*//')
      [ -n "$val" ] || { fail_row "" "${r_tag#r}" "$i" 0 "no '$LABEL_TEST' row" "$out"; return; }
      echo "$clean" | grep -E '^ -- (Bitrate|Chunk size)|CPU MoE worker started' | sed "s/^/    $label load /"
      echo "    $label load CPU split: $(echo "$clean" | grep -c 'CPU split experts') layers, first: $(echo "$clean" | grep -m1 'CPU split experts' | sed 's/.*mlp //')"
      echo "$clean" | grep -E '^(Context|Length) +[0-9]+:' | sed "s/^/    $label table /"
      MAJ_COL=" | majflt $((m1 - m0)) (whole process, its load's; untagged)" COLD_TAG='' PTAG=$tag
      rowtags="$CPU_BUSY_TAG$OTHER_BUSY_TAG"
      if [ "${A_PP[$i]}" = 1 ]; then
        echo "$tag $r_tag $label p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME | EXL3 $EXL3_BPW bpw, another quantization | chunk 4096$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$tag" = ROW ] && pp_sums+=("$label|$dep|$r|$val")
      else
        echo "$tag $r_tag $label d=$dep n=100 | tok/s $val @ n=100, depth $dep, $CARD_NAME | EXL3 $EXL3_BPW bpw, another quantization$MAJ_COL | wall $((t1 - t0))s$rowtags"
        [ "$tag" = ROW ] && sums+=("$label|$dep|$r|$val")
      fi
      ;;
  esac
  cold_count "$tag"
  return 0
}
# cold_count <tag>: a round's [cold] row into cold_rows.
cold_count() { [ "$1" != ROW ] || [ -z "$COLD_TAG" ] || cold_rows=$((cold_rows + 1)); }
# run_row <round> <index>: a round's row of arm <index> under the cold rule (the header): a COLD row runs
# the arm once more.
run_row() {
  COLD_PASS=0 COLD_AGAIN=0
  run_arm ROW "$1" "$2"
  [ "$COLD_AGAIN" = 1 ] || return 0
  retry_rows=$((retry_rows + 1))
  echo "[cold] r$1 ${ARMS[$2]} read [cold]: its arm runs once more (docs/fair-measure.md 2.3)"
  COLD_PASS=1 COLD_AGAIN=0
  run_arm ROW "$1" "$2"
  COLD_PASS=0
}

# ratio_table <prefix> <keys> <labels> [base]: records `label|key|round|value` on stdin; for every key and
# every label, each round's base / label ratio (the base ours, the default, or the label given), their
# mean with its 95 % interval (Student t at rounds - 1 degrees of freedom) and the ratio of the arm means
# (depth-qwen3moe.sh's table).
ratio_table() {
  awk -F'|' -v prefix="$1" -v deps="$2" -v refs="$3" -v base="${4:-ours}" -v rounds="$ROUNDS" -v t975="$T975" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
} END {
  nt = split(t975, t, " "); nd = split(deps, d, " "); nr = split(refs, rf, " ")
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
    printf "%s%-5s %s/%-16s mean %.4f %s (n=%d)  of means %.4f  per round:%s\n", prefix, d[i], base, ref, m, ci, c, (as[base SUBSEP d[i]] / an[base SUBSEP d[i]]) / (as[ref SUBSEP d[i]] / an[ref SUBSEP d[i]]), list
  }
}'
}
# place_rank <word>: the word's position in PLACE_WORDS (a 0, gate 1, bp 2).
place_rank() {
  local w r=0
  for w in $PLACE_WORDS; do
    [ "$w" != "$1" ] || { echo "$r"; return; }
    r=$((r + 1))
  done
}
# place_pairs: every two of our arms of one engine (ours or oursmtp) whose variables are the same, in any
# order, and whose placements differ, as `<numerator label>|<denominator label>` into PL_PAIRS, once a pair
# of labels. The pair is ordered by placement, never by the arms' order or which one names place=: the
# denominator is the placement first in PLACE_WORDS (a before bp), so a two-card pair reads bp / a.
place_pairs() {
  local i j n d p ki
  PL_PAIRS=()
  for i in "${!A_LABEL[@]}"; do
    case ${A_ENG[$i]} in ours | oursmtp) ;; *) continue ;; esac
    ki=$(tr , '\n' <<< "${A_ENV[$i]}" | sort | paste -sd, -)
    for j in "${!A_LABEL[@]}"; do
      [ "$j" -gt "$i" ] && [ "${A_ENG[$j]}" = "${A_ENG[$i]}" ] && [ "${A_PLACE[$j]}" != "${A_PLACE[$i]}" ] || continue
      [ "$(tr , '\n' <<< "${A_ENV[$j]}" | sort | paste -sd, -)" = "$ki" ] || continue
      if [ "$(place_rank "${A_PLACE[$i]}")" -gt "$(place_rank "${A_PLACE[$j]}")" ]; then n=${A_LABEL[$i]} d=${A_LABEL[$j]}; else n=${A_LABEL[$j]} d=${A_LABEL[$i]}; fi
      p="$n|$d"
      [[ " ${PL_PAIRS[*]+${PL_PAIRS[*]}} " == *" $p "* ]] || PL_PAIRS+=("$p")
    done
  done
}
means() { # means <unit>: `label|key|round|value` on stdin, one mean line per label and key
  awk -F'|' -v unit="$1" '{
    k = $1 " " $2; s[k] += $4; n[k]++
    if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
  } END { for (k in s) printf "mean %-26s %9.2f %s  [%s..%s, spread %.2f%%]  (n=%d)\n", k, s[k] / n[k], unit, mn[k], mx[k], (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0, n[k] }' | sort
}

majflt_require depth-glm5next.sh
timing_cards_precheck || {
  rc=$?
  echo "depth-glm5next.sh: two cards, refused before the lease: $TWOCARD_WHY" >&2
  exit "$rc"
}
lease_take
timing_cards_start
[ -z "$TIMING_CARDS" ] || echo "[config] $(tc_config)"
echo "[config] model=$MODEL n=$N rounds=$ROUNDS card=$CARD_NAME timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU arm_bound=${BOUND}s cold_us=$COLD_US"
[ "$ours" = 0 ] || echo "[config] ours: $BIN --ctx $CTX --place $PLACE warm=${WARM:-0} prose=$PROSE"
[ -z "$PLACES_SET" ] || echo "[config] placements: $(place_line "${PLACE_ARMS[@]}")"
[ "$ours_mtp" = 0 ] || echo "[config] oursmtp: ours' command under env BLOOMERY_DRAFT=mtp; $MTP_NOTE"
[ -z "$RS_NOTE" ] || echo "[config] residency: $RS_NOTE"
[ "$ours$srv" = 00 ] || echo "[config] prompt: GLM_PROSE ids from index $PROSE_FROM"
[ "$lcpp" = 0 ] || echo "[config] lcpp27752 flags=$LCPP27752_GPU_FLAGS | lcpp27754 env=$LCPP27754_ENV flags=$LCPP27754_GPU_FLAGS"
for e in 27752 27754; do
  case $e in 27752) [ "$fit27752" = 1 ] || continue ;; *) [ "$fit27754" = 1 ] || continue ;; esac
  lcpp_fit_flags "$(gpu_flags "lcpp$e")"
  echo "[config] lcpp${e}fit: flags=$FIT_FLAGS (llama-bench's fit places the model; dropped: $FIT_DROPPED; lcpp${e}ppfit<U>: -ub U -b max(U, 2048))"
done
[ "$srv" = 0 ] || echo "[config] server arms: the branch's bench flags in llama-server's spellings (lcpp-warm.sh) with $LCPP_SRV_FIXED, -c $CTX (ours' --ctx; a load record naming another ctx, or a server log naming another n_ctx, is a FAIL row), a discarded request then the timed one on ours' prose ids; MTP at --n-cpu-moe $GLM_NCMOE_MTP with $GLM_MTP_FLAGS; each row's first $XC_N ids held to ours' on the same ids (xcheck)"
[ "$exl3" = 0 ] || echo "[config] exl3: $EXL3_MODEL ($EXL3_BPW bpw) flags=$EXL3_FLAGS, perf.py's defaults otherwise (cache 32768, chunk 4096)"
echo "[config] arms=${ARMS[*]}"
if [ -n "$PH_DIR" ]; then
  for var in ${!PH_LINE_*}; do echo "[config] preheat: ${!var}"; done
else
  echo "[config] preheat: off ($([ "$PREHEAT" = 0 ] && echo BLOOMERY_PREHEAT=0 || echo 'no GGUF arm'))"
fi
witness pre
ref_witness
for grp in 0 1; do
  case $grp in 0) [ "$gguf" = 1 ] || continue ;; 1) [ "$exl3" = 1 ] || continue ;; esac
  [ "$WARMUP" = 0 ] || [ "$grp" = 0 ] || run_arm DISCARD 0 "$(first_of 1)"
  for r in $(seq "$ROUNDS"); do
    for i in $(order_of "$r" "$grp"); do
      # Each GGUF arm's warm-up: the same ids, discarded, right before its first round's row.
      [ "$WARMUP" = 0 ] || [ "$grp" = 1 ] || [ "$r" != 1 ] || srv_glm "${A_ENG[$i]}" || run_arm WARMUP 0 "$i"
      run_row "$r" "$i"
    done
  done
done

echo
echo "arm runs: $n_rows (FAIL rows included); rows tagged [cpu-busy] $busy_rows, [other-busy] $other_rows, [cold] $cold_rows; cold re-runs $retry_rows, FAIL-cold $fail_cold"
# The greedy cross-check (lcpp-warm.sh), before the tables: a server row whose first id parts from ours
# joins the failed arms and its label drops out of the means and ratios at that depth or P (xc_drop); a
# later id's part is a [xcheck-tail] line and stays.
xc_table
if [ ${#XC_FAILED[@]} -gt 0 ]; then
  failed+=("${XC_FAILED[@]}")
  printf '%s\n' "${XC_DROP[@]}" | sort -u | awk -F'|' '{ printf "    dropped: %s at %s (FAIL xcheck)\n", $1, $2 }'
  kept=()
  while IFS= read -r l; do [ -z "$l" ] || kept+=("$l"); done < <(printf '%s\n' ${sums[@]+"${sums[@]}"} | xc_drop)
  sums=(${kept[@]+"${kept[@]}"})
  kept=()
  while IFS= read -r l; do [ -z "$l" ] || kept+=("$l"); done < <(printf '%s\n' ${pp_sums[@]+"${pp_sums[@]}"} | xc_drop)
  pp_sums=(${kept[@]+"${kept[@]}"})
fi
echo "=== per-arm decode means (tok/s @ n=$N, $CARD_NAME; exl3 rows @ n=100, another quantization) ==="
[ ${#sums[@]} -eq 0 ] || printf '%s\n' "${sums[@]}" | means tok/s
echo "=== per-arm prefill means (tok/s(pp) @ n=0, prompt P, $CARD_NAME; ours its time prompt row, a server decode row its prompt_per_second) ==="
[ ${#pp_sums[@]} -eq 0 ] || printf '%s\n' "${pp_sums[@]}" | means 'tok/s(pp)'
# Ratios against ours: the same file's engines only. exl3 runs another quantization. The engines do not
# share a routing condition: ours and the server arms feed the same prose ids, ours places card experts by
# id; llama-bench feeds std::rand() ids and places whole layers, whose host bytes a token do not depend on
# the ids; exllamav3 feeds wikitext-2 and adapts its placement to that stream.
same_file() { grep -vE '^exl3|^oursmtp(@|$)' | grep -vx ours; }
if [ ${#sums[@]} -gt 0 ]; then
  echo
  echo "=== ours / each engine on the same file, per depth: each round's ratio, their mean ± 95 % (t, rounds - 1 df) ==="
  keys=$(printf '%s\n' "${sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  refs=$(printf '%s\n' "${sums[@]}" | cut -d'|' -f1 | sort -u | same_file | tr '\n' ' ')
  printf '%s\n' "${sums[@]}" | ratio_table "ratio d=" "$keys" "$refs"
fi
if [ ${#pp_sums[@]} -gt 0 ]; then
  echo
  echo "=== ours / each engine on the same file, prefill per prompt length (ours: its time prompt row) ==="
  keys=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | sort -u | same_file | tr '\n' ' ')
  printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp p=" "$keys" "$refs"
fi
# The draft's own effect: the same binary, ids, context and placement, BLOOMERY_DRAFT=mtp the one difference.
if [[ " ${sums[*]+${sums[*]}}" == *" oursmtp|"* ]]; then
  echo
  echo "=== ours / oursmtp per depth (below 1: the MTP draft is faster): each round's ratio, their mean ± 95 % (t, rounds - 1 df) ==="
  keys=$(printf '%s\n' "${sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  printf '%s\n' "${sums[@]}" | ratio_table "ratio mtp d=" "$keys" oursmtp
fi
# One arm at two placements: the same engine and variables, the --place the one difference.
place_pairs
if [ ${#PL_PAIRS[@]} -gt 0 ]; then
  for what in decode prefill; do
    if [ "$what" = decode ]; then recs=(${sums[@]+"${sums[@]}"}) pre="ratio place d="; else recs=(${pp_sums[@]+"${pp_sums[@]}"}) pre="ratio place pp p="; fi
    [ ${#recs[@]} -gt 0 ] || continue
    echo
    echo "=== one arm at two placements, $what per $([ "$what" = decode ] && echo depth || echo 'prompt length'): the later placement / the earlier (bp / a), each round's ratio, their mean ± 95 % (t, rounds - 1 df) ==="
    keys=$(printf '%s\n' "${recs[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
    for p in "${PL_PAIRS[@]}"; do printf '%s\n' "${recs[@]}" | ratio_table "$pre" "$keys" "${p#*|}" "${p%%|*}"; done
  done
fi
res_sums_table
witness post
ref_witness
if [ ${#failed[@]} -gt 0 ]; then
  echo "failed arms: ${failed[*]}"
  exit 1
fi
