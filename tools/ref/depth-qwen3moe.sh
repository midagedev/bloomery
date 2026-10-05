#!/usr/bin/env bash
# Qwen3-30B-A3B decode by depth and prefill by prompt length with the whole model on the timing
# card, four engines in one lease (run on the box, lead-only): our engine (generate_qwen3moe
# --time), ik (llama-bench -gp D,N; -p P -n 0 for prefill), mainline llama.cpp (llama-bench -d D;
# -p P -n 0) and mistral.rs (mistralrs bench --depth D; --prompt-len P for prefill), alternated arm
# by arm or run in engine blocks (Order below).
#
#   BLOOMERY_MODEL=qwen3moe tools/box.sh 'bash tools/ref/depth-qwen3moe.sh 6 ik:6 lcpp:6'
#   just depth-gpu-qwen3moe 6 ik:6 ikdef:6 lcpp:6 1024 ik:1024 ikdef:1024 lcpp:1024 4096 ik:4096 ikdef:4096 lcpp:4096
#   just depth-gpu-qwen3moe 6 lcpp:6 mrs:6 4096 lcpp:4096 mrs:4096
#   just depth-gpu-qwen3moe 512 ikpp:512 lcpppp:512 mrspp:512 4096 ikpp:4096 lcpppp:4096 mrspp:4096
#   BLOOMERY_BOX_ENV=BLOOMERY_DRY=1 just depth-gpu-qwen3moe 6 lcpp:6 mrs:6    # the command lines, no lease, no load
#   BLOOMERY_BOX_ENV='BLOOMERY_AB_ORDER=blocks' just depth-gpu-qwen4exp 6 lcpp:6 512 lcpppp:512   # engine blocks
#   just depth-gpu-qwen4exp prose:512 prose:512@BLOOMERY_QWEN38_EXPERTS=host   # the prose corpus's first 512 ids, prose's own tables
#   just depth-gpu-qwen3moe 6 bin:/root/repo/bloomery-<track>-base/target/release/generate_qwen3moe:6 lcpp:6
#
# depth-ds41.sh's shape, and it blocks the same failure: a ratio read at one depth and quoted as
# "decode is faster" — a step's attention term grows with the cached keys, so the depth goes into
# every number (`tok/s @ n=N, depth D, <card>`) and the ratio table at the end is per depth.
#
# Arms, in lease order (the order rotates by one slot each round — the position bias ab-decode.sh
# names):
#   <D>       ours, D >= 1: generate_qwen3moe --tokens <lcg_prompt D> -n N --ctx C --time. The D
#             fed ids are prefilled untimed (Qwen3moeModel::prefill: D >= 9 in ubatches of up to the
#             ubatch size — UBATCH, or BLOOMERY_QWEN3_UBATCH; ubatch.rs:1-4 — through the grouped
#             GEMM, a tail of at most 8 and a D <= 8 as one pass; the `load` line's `ubatch=` names
#             the size, the `step 0` line's `plan=` the units), from lease.sh's lcg_prompt: 100000,
#             then ids in [1000, 91000), all inside this vocabulary; generated token 0 comes out of
#             the last pass and the N - 1 steps after it are timed. C is D + N rounded up to 256: both
#             llama-benches size n_ctx to D + N for this test and pad it to 256 under flash
#             attention (ik: GGML_PAD(n_ctx, llama_kv_cache::get_padding(flash_attn)) in
#             src/llama.cpp; mainline: GGML_PAD(n_ctx, 256) in src/llama-context.cpp), so the
#             caches have one height (mistral.rs: below).
#             Our flash grid no longer depends on C (flash_gqa::SEGMENTS); C still sets the cache
#             height every engine gets, so C goes into the row.
#             BLOOMERY_GEN_CTX fixes C for every ours arm instead, e.g. at a serving height.
#             The row also carries `pp_tok/s <v> (n=D, passes=K)` from the binary's `time prompt`
#             row, the wall of that prefill (Prefill below): one arm reports both the prefill of
#             D ids and the decode at depth D.
#   <D>@NAME=VALUE[,NAME=VALUE...]  ours at depth D, this binary and the <D> arm's command line, with
#             those variables set for this arm only (`env NAME=VALUE ... <binary>`; depth-ds41.sh's
#             grammar): a lever arm, row label `ours@NAME=VALUE[,...]`, in the same tables as the ours
#             rows of that depth. Beside a plain `<D>` arm it is the same-binary A/B, paired by round in
#             the ratio table's `ours/ours@…` line, e.g. `6 6@BLOOMERY_QWEN38_EXPERTS=host`. A lever arm is
#             compared only with ours arms of the same binary, never with a bin: arm or a reference: under
#             1 % two builds differ by link layout alone (AGENTS.md). The label is what its witness blocks
#             and its row print, so both name the variables. The variables are load-time
#             (tools/ref/load-groups.sh), so the load key holds them: a lever arm is a load of its own, a
#             unit the round's rotation (Order) moves like any other. BLOOMERY_AB_LOAD=arm in the list runs
#             the arm alone and stays in the label; the binary never sees it. Refused by name before
#             anything runs: an empty list, item, name or value; a name that is no lever row of
#             crates/levers/src/registry.rs (a row the crate parses or a file reads in place: a retired
#             name, a runner's or a path's variable and a name no row names are refused, the registry
#             read only when an arm has an `@`); a name given twice; a value holding `,`, `@`, `|` or
#             white space; a name the runner's own environment already sets (BLOOMERY_BOX_ENV reaches
#             every arm, so the plain rows' labels would hide it); and `@` on a reference, server or bin:
#             arm — a lever of ours is not a reference's. The item place=<a|gate|bp> is no lever: it is the
#             arm's placement (Placement below), in its load key and its --place, never a variable its
#             process gets; the label keeps it.
#             `depth-qwen3moe.sh --parse-arms [--registry <registry.rs>] <arms...>` parses the arms as a
#             run does and prints each arm's kind, depth, label, variables and load key — a prose arm's
#             line also its corpus path and the file's first three ids (`-` when the file is not readable
#             where it runs: the box's path, read on the Mac — a run refuses it) — each round's
#             order and round 1's load command lines (`env NAME=VALUE ...` first), then exits 0 before the card, the binaries and the lease (it runs on the Mac);
#             `--self-test` runs it on fixed arms, on the Mac. check-recipes does not run it: a script it
#             names that calls lease_take makes gate-batch.sh refuse the check as a timed recipe.
#   <D>@BLOOMERY_GEN_SLOTS=N[,NAME=VALUE...]  (and prose:<P>@BLOOMERY_GEN_SLOTS=N…) a lever arm read
#             as an aggregate row: generate_qwen3moe decodes N streams in one pass (the lever's row in
#             crates/levers/src/registry.rs). The arm feeds N·D ids — lcg_prompt N·D, whose first D are
#             the <D> arm's prompt and the rest the walk's next ids, or the corpus's first N·P — which the
#             binary cuts into N windows of D, window j prefilled into slot j; its `[parse]` line ends in
#             `slots=N feed=N·D`. Its row reads the counted (`warm` left out) `time pass … kind=slots`
#             records through records.py: `tok/s(aggregate) <Σ positions · 1000 / Σ ms> @ n=N·<rounds>,
#             depth D, <card>`, then `slots N`, the per stream rate and the SMOKE footer's p50 and mean
#             ms a pass; `nodes` is the captured pass's (`capture slots=`). The per-arm means hold it
#             beside the plain arm, `(aggregate of N slots)`; it stays out of the `ratio d=` and prose
#             tables, and `ratio slots d=` is its aggregate over its plain twin's tok/s per round — the
#             twin is its label less the BLOOMERY_GEN_SLOTS item (ours, ours@prose, ours@<the rest>) — so
#             above 1 N streams in one pass outrun one stream. Refused by name before anything runs: an N
#             that is not a whole number, N·D past the 20000 ids one argument holds, N·P past the corpus,
#             and BLOOMERY_GEN_SLOTS in the runner's own environment (an arm names its own). A FAIL row: a
#             slots arm with no counted kind=slots record or one whose positions is not N, and a kind=slots
#             record from an arm that names no N. N = 1 is a plain lever arm. The binary refuses N past a
#             pass's rows, a qwen35moe or qwen4exp file and a placed load by name (a FAIL row here).
#   prose:<P>[@NAME=VALUE[,NAME=VALUE...]]  ours fed the first P ids of
#            $BLOOMERY_DATA/$MODEL_NAME/corpus-prose.ids (the profile's own prose corpus, one id a line —
#            the file its d1k reference set is cut from) through --tokens instead of the LCG prompt, the
#            NAME=VALUE list read as a <D> arm's: row label `ours@prose` (`ours@prose@NAME=VALUE[,...]`),
#            the depth column P. The prompt's routing, and so its host and card work, is the prose's,
#            not the LCG walk's (V4.1's lcg and prose prompts moved its card experts differently, so a
#            lever tuned on one can mean nothing on the other): a prose arm is compared only with prose
#            arms of the same P — `prose:512 prose:512@BLOOMERY_QWEN38_EXPERTS=host` is the same-binary
#            A/B on the prose prompt — in its own decode and prefill tables (ours@prose / each
#            ours@prose@ label), never with an lcg arm or a reference. The ids arrive at the prompt, so
#            the load is an lcg arm's: C is P + N rounded up to 256 as for a <D> arm, and a prose arm
#            shares a load (and its key: binary, C, variables) with an lcg arm of the same C and
#            variables. Refused by name before anything runs: a corpus file that is missing or
#            unreadable (under --parse-arms one names itself on the arm's line instead), a P past its
#            line count, a P < 1, a line among its first P that is not one id, and prose: on a
#            reference arm — a corpus arm feeds our binary, a second build (bin:<path>:prose:<P>) or a
#            server (lcppsrv…:prose:<P>, below).
#   bin:<path>:<D>  a second generate_qwen3moe (an absolute path on the box, a base tree's build) at
#             depth D with the ours arm's command line, row label `bin:<basename of its tree>` (the
#             tree is the path above `target/`). Beside a plain `<D>` arm it is the same-lease A/B of
#             two builds. bin:<path>:prose:<P> feeds it the prose arms' prompt, row label
#             `bin:<tree>@prose`, in the prose table beside `prose:<P>`. It is a base by construction, so its freshness is not asked; its tree line
#             (sha256, HEAD, dirty files) is printed with the references'. Every label is its own
#             engine in the per-arm means and in the ratio table (ours / each other label).
#   ik:<D>    ik: llama-bench -p 0 -n 0 -gp D,N -r 1 $IK_GPU_FLAGS (D = 0: plain tg N). The D-token
#             prefill runs first and untimed: ik restarts its clock after it.
#   ikdef:<D> the same at $IK_GPU_DEFAULT_FLAGS: llama-bench's own defaults with every layer on the
#             card. IK_GPU_FLAGS opts into two merges nobody has timed on this model; this arm,
#             interleaved with ik:<D>, says whether they are ik's faster set.
#   lcpp:<D>  mainline: llama-bench -p 0 -n N -d D -r 1 $LCPP_GPU_FLAGS (D = 0: plain tg N).
#             Mainline has no -gp; -d prefills D tokens before its clock starts. Its label is
#             `tgN @ dD`, ik's `tgN@ppD`.
#   mrs:<D>   mistral.rs, D >= 1: mistralrs bench -f <MODEL> --prompt-len 0 --gen-len N --depth D
#             --iterations 1 --warmup 1 --max-seq-len C --pa-context-len C+32 $MRS_FLAGS. One request
#             of D prompt ids with greedy sampling and EOS stop off; its clock (mistralrs-cli
#             src/commands/bench.rs, recv_measurement) runs from the first streamed token to the
#             last, over N - 1 intervals, so the timed tokens sit at the positions ours times. The
#             warmup is one whole discarded request, as llama-bench's own warmup run is.
#             --pa-context-len sizes the PagedAttention pool (on by default on CUDA) to the
#             other engines' cache height C instead of 90 % of the free memory: the pool keeps
#             one 32-token block back, so C + 32 leaves C usable. --max-seq-len C is the
#             automatic device mapper's length, which only places layers. The weights stay in
#             the file's quantized types; activations are bf16 (`DType selected`) and so is the
#             pool (`KV cache type`) — load lines echoed with the row. mistral.rs
#             refuses --depth 0 with a decode length (bench.rs run_bench), so D = 0 is refused here.
#             Its row reads the `Decode (N tokens @ dD)` row of the bench's table (T/s, one decimal).
#   mrspa0:<D> the same with --paged-attn off in place of --pa-context-len: mistral.rs's own KV
#             cache and attention path, interleaved with mrs:<D>, says which is its faster set.
#             That cache is sized by mistral.rs, not by C: this arm is outside the one-height
#             statement above.
#   ikpp:<P>  ik's prefill: llama-bench -p P -n 0 -r 1 $IK_GPU_FLAGS, the ik:<D> arm's binary and
#             flags; the row reads llama-bench's `ppP` row as `tok/s(pp) … @ n=0, prompt P`.
#   lcpppp:<P> mainline's prefill: llama-bench -p P -n 0 -r 1 $LCPP_GPU_FLAGS, the lcpp:<D> arm's.
#   ikpp<U>:<P>, lcpppp<U>:<P>  the same with -ub U -b max(U, 2048): the ubatch lever, not a default
#             (Prefill below). Refused when the profile's flags already name -ub or -b.
#   lcppfit:<D>, lcppppfit[<U>]:<P>  mainline at its own placement: the lcpp:<D> and lcpppp[<U>]:<P>
#             arms' command lines with the profile's placement options (-ngl, --n-cpu-moe, -ts, -ot)
#             removed and `-fitt 1024 -v` added, so llama-bench's fit chooses the layers and the
#             overrides (tools/ref/lcpp-fit.sh has what the fit does and where). The row carries `fit
#             <what it chose>` from -v's loader lines, which are echoed as `lcppfit fit …`; a fit that
#             failed or never ran is a FAIL row (Failures below). Refused before the lease when
#             the tree's llama-bench has no --fit-target, or when the profile's flags carry a fit option.
#   lcppsrv:<D>, lcppsrvpp[<U>]:<P>, lcppsrvfit:<D>, lcppsrvppfit[<U>]:<P>  mainline warm, on our ids: the
#             tree's llama-server at LCPP_GPU_FLAGS in its spellings (the fit twins at its fit), one process
#             a row, a discarded POST /completion of lcg_prompt D (P), then the same timed: decode rows
#             predicted_per_second at n_predict N (and their prompt_per_second a prefill record under the
#             same label), prompt rows prompt_per_second at n_predict 1, `ids=lcg` (depth-ds41.sh's arms of
#             the same names; lcpp-warm.sh has the server's flags, its rates against llama-bench's and
#             every failure that is a FAIL row). The server's -c is ours' at that D or P: BLOOMERY_GEN_CTX,
#             else D + N rounded up to 256 (docs/fair-measure.md 1.5), and an arm whose ids + n_predict + 1
#             do not fit it is refused before the lease. An engine word takes its own flags, `+t<N>`,
#             `+nopo<0|1>`, `+k<K>` (-t, -nopo, -ncmoe in place of the profile's: lcpp-warm.sh's Per-arm
#             flags), its label with them.
#   lcppsrv…:prose:<P>  a server arm fed the corpus's first P ids, the ids a prose:<P> arm feeds: row
#             label `<engine>@prose`, `ids=prose`, in the prose tables beside ours@prose.
# Every server row's first ids are held to ours' on the same ids (lcpp-warm.sh's Cross-check: the plain
# `ours` and `ours@prose` rows' `tokens` lines); a server row whose first id parts from ours is a FAIL
# xcheck line in the failed arms and drops out of the means and ratios; one that parts later is a
# [xcheck-tail] line and stays.
#   mrspp:<P> mistral.rs's prefill: mistralrs bench -f <MODEL> --prompt-len P --gen-len 1
#             --iterations 1 --warmup 1 --max-seq-len C --pa-context-len C+32 $MRS_FLAGS, C = P + 1
#             rounded up to 256 (or BLOOMERY_GEN_CTX): a gen length below 2 skips the decode case,
#             and the one request of P ids plus its first token must fit --max-seq-len
#             (bench.rs run_bench). Its row reads the `TTFT (P input tokens)` row of the bench's
#             table: T/s = P / TTFT.
# ik and mainline feed std::rand() ids, prefill and steps alike; mistral.rs feeds prompt ids 1000 +
# (start + i) mod 2048 and then its greedy continuation; ours feeds its greedy continuation, which
# is why our row carries distinct_tokens. Every reference arm is one process — a model load, the
# prefill, N steps (mistral.rs: its warmup request, then the timed one) — and the four engines open
# the one file. Our arms of one round that share a load key — this binary and C — run in one process
# (generate_qwen3moe --arm <ids> ... --arm-sync), the engine cleared between them (app::Session::clear),
# each arm between its own witness blocks, its row carrying `slot <k>/<n>`, its place in its load; the
# load's lines print once under `[load]`, and the timing card's guard runs before each load's process
# (after that the process holds the card). The grouping, the order (the units and the arms in each
# rotated by round) and the driver are tools/ref/load-groups.sh's, shared with depth-ds41.sh:
# BLOOMERY_AB_LOAD=arm runs every arm in a process of its own. An arm that fails in a shared load ends
# that process: the arms after it in the round re-run in a fresh load (Failures below).
#
# Flags. The ours, ik and lcpp arms hold every layer, the output head and an f16 K/V cache on the
# card (llama-bench's -ctk/-ctv default is f16; ours keeps f16 planes), so a step reads the same
# weight and cache bytes in each. The input embedding differs: both references keep it on the host and
# copy one row to the card per step (ik: buft_input in src/llama.cpp; mainline: dev_input in
# src/llama-model.cpp), ours gathers it on the card. The references' flag sets are the profile's
# (models/qwen3moe.sh), chosen by reading each tree's CLI, docs and loader; no flag sweep has been
# run, so a row is "at these flags". A sweep overrides them inside the box command, the profile
# keeps a caller's value:
#   BLOOMERY_MODEL=qwen3moe tools/box.sh 'IK_GPU_FLAGS="-ngl 99 -fa 1" bash tools/ref/depth-qwen3moe.sh ik:6'
#   ik    -ngl 99   every layer and the output head on the card (ikdef: this flag alone)
#         -fa 1     flash attention; ik's default, spelled out
#         -fmoe 1   the fused MoE up-gate op; ik's default, spelled out
#         -mqkv 1   attn_q/k/v loaded as one matrix where their types agree (the 24 layers whose
#                   attn_v is Q4_K; the 24 with a Q6_K attn_v merge q and k): fewer launches a layer
#         -muge 1   ffn_gate_exps and ffn_up_exps loaded as one matrix (Q4_K in all 48 layers)
#         With these two, ik's size and params columns count each merged matrix twice
#         (llama_model_size in src/llama.cpp sums tensors_by_name, which lists the merged matrix
#         and its views), so its size reads larger than the file's. The views allocate nothing.
#         left out: -gr (graph reuse, on by default); -rcache (build_qwen3moe never builds the
#         rope cache); -ger (BailingMoE only); -sas (split mode graph only); -mla (MLA models);
#         -cuda (fusion and graphs are on in this build; offload-batch-size* tune host offload,
#         mmq-id-size batched expert matmuls, enable-p2p multi-card copies, fa-offset the flash
#         softmax's arithmetic); -rtr, -thp, -mmp, --defer-experts, -t
#         (host tensors, load time and host threads: at -ngl 99 a step's only host work is the
#         embedding row, the input layer stays on the host); -b, -ub, -amb (the untimed prefill);
#         -ser (drops experts: another computation, not a faster one); -ctk/-ctv (a quantized
#         cache: likewise)
#   lcpp  -ngl 99   every layer and the output head on the card
#         -fa on    flash attention; mainline's default is auto
#         left out: -lm, -lzm (load mode); -nopo, --no-host (host tensors); -t, --poll, -C (host
#         threads, as above); -b, -ub (the untimed prefill); -ctk/-ctv (as above). Mainline has
#         no fused-MoE or merge flag: CUDA graphs and op fusion are build options, on in this
#         build (GGML_CUDA_GRAPHS=ON in the tree's CMakeCache.txt).
#   lcppfit  the lcpp flags less -ngl, then -fitt 1024 (the margin llama.cpp's `-fit on` leaves) and
#         -v. The file is well under the card's free memory, so the fit returns at its first check
#         with every layer on the card (common/fit.cpp:354, n_gpu_layers -1 = all, src/llama-model.cpp:1926):
#         the lcpp placement, and a lcppfit row predicts the lcpp row's value within the ruler [derived];
#         its `fit` column is the check (`overridden none`, every layer offloaded).
#   mrs   --format gguf  the file's format, spelled out (the suffix would pick it)
#         left out: a subcommand (bench's own options take the model; `auto` is the default path);
#         -n/--device-layers and --topology (one visible card, automatic mapping puts every layer
#         on it); --isq, --quant, --dtype (another computation, or the file's own types); --cpu;
#         --pa-block-size, --pa-cache-type (their defaults: 32 tokens, the compute dtype); --seed
#         (greedy sampling); --max-batch-size (one sequence). The binary does not print its cargo
#         features; the witness records its sha256, tree and version, and the load lines where
#         its layers and cache went.
#
# The trees are the profile's IK, LCPP and MRS. The witness prints each binary's sha256 with its
# tree's HEAD and dirty count, and each reference row the binary's own `build:` line (mistral.rs:
# its --version line) — a tree can move after its binary was built — and the device llama-bench
# opened.
#
# Paging. The witness prints the page cache and the major fault count before and after every arm:
# the file is 18.6 GB, and another round's host set can evict it between two arms. Qwen3.8 is not
# card-resident: our engine keeps the routed experts past each layer's card prefix on the host tier
# (all of them under BLOOMERY_QWEN38_EXPERTS=host) and reads the PLE table's rows a token at a time, and mainline keeps the first --n-cpu-moe layers' experts on the host, read
# through the file mapping inside its timer; after another model's host set has evicted its pages, a
# row reads the file cold. The cold tag and the blocks below are what tell such a row apart and keep
# it out of the table.
#
# Cold tag (tools/ref/cold-blocks.sh has the rule, shared with depth-ds41.sh). Every row ends in
# ` | majflt <n> (timed <n>; ≤ <x> % of W <w> s)`: the change in /proc/vmstat pgmajfault across the arm's
# process, and across its measured window where the runner sees that window start — ours from its
# `prompt_ids` line (in an --arm list printed after the go, just before its prefill timer starts), an
# lcpp arm (lcpp, lcpppp…, and the fit arms) from its --progress line of the timed repetition
# (`generation run 1/1` for a decode arm, `prompt run 1/1` for a pp arm, printed after that
# repetition's clock starts: tools/llama-bench/llama-bench.cpp:2437, 2447 at 53ed051ce), mrs from its
# `Iteration 1/1...` log line (mistralrs-cli src/commands/bench.rs:263, printed before the timed
# request is sent). ik's llama-bench has no --progress, so an ik row counts its whole process, and a
# bin: arm's one-arm command line prints its prompt ids before its load, so it counts the whole process
# too; the column says which. W is the row's timed window: P / tok/s for a pp row, N / tok/s for a
# llama-bench decode row, (N - 1) / tok/s for a mrs decode row (its clock spans N - 1 intervals), the
# prompt's ms plus N × mean ms for ours. A row whose count × COLD_US could be COLD_PCT % of W or more
# ends in ` [cold]`. The column is appended after the row's `wall` field and its [other-busy] tag, so
# every earlier field stays where it was; the tag comes last.
#
# Order. BLOOMERY_AB_ORDER=rotate (the default) runs every arm once a round, the order rotated by one
# slot each round, as before. BLOOMERY_AB_ORDER=blocks runs the arms in engine blocks, each block's
# rounds together, after one discarded process (`DISCARD r0`, in no mean, ratio or row count) — the
# rule and its reasoning are depth-ds41.sh's (Order, Discard). The blocks:
#   ours         every <D> arm
#   prose        every prose:<P> arm, its lever arms with them: the corpus's ids are not the LCG
#                walk's, so its discard reads its own prompt's pages, not the lcg block's
#   bin:<tree>   each second binary's arms
#   lcpp         lcpp, lcpppp, lcpppp<U>: one binary at one placement (the profile's --n-cpu-moe)
#   lcppfit      lcppfit, lcppppfit, lcppppfit<U>: llama-bench's fit keeps the last layers' experts on
#                the host, --n-cpu-moe K the first K, so they are a block of their own
#   ik           ik, ikdef, ikpp, ikpp<U>
#   mrs          mrs, mrspa0, mrspp
#   lcppsrv      lcppsrv, lcppsrvpp[<U>]; lcppsrvfit: the fit twins. Their discard is the longest prompt
# A block's discard is its arm with the longest prompt (ours, prose, bin) or the most token ids its process
# takes (a reference block): at -r 1 mainline draws 2P for a pp arm (its warm-up is the whole prompt)
# and D + N + 3 for a decode arm, ik P + 1 and D + N + 4 (N + 3 at D = 0), mistral.rs feeds 2 (P + 1)
# and 2 (D + N) ids (its warm-up request, then the timed one), and the discard runs at the block's
# largest --n-cpu-moe when every arm names one.
# Warm rows. BLOOMERY_WARM_ROWS=1: each of our arms after a same-id PRIME run in its load, and one retry of
# a counted row the cold tag marks (`COLD r<r> …`, then the row or `FAIL … rc=cold`), cold-blocks.sh's rule
# and depth-ds41.sh's use of it. Off by default.
# Warm-up. BLOOMERY_AB_WARMUP=1 runs the first arm once before round 1 under rotate (`WARMUP r0`) and
# is the discards under blocks; 0 skips them. Its default follows the order: 0 under rotate (the
# Qwen3 and Qwen3.6 rows' behaviour: the model sits on the card, so no timed window reads the file),
# 1 under blocks.
#
# Prefill. pp_tok/s is P over the wall of processing a P-token prompt, in each engine's terms:
#   ours      generate_qwen3moe's `time prompt` row: the wall of Qwen3moeModel::prefill through the
#             readback of its token — P >= 9 in ubatches of up to the ubatch size (`load ubatch=`)
#             through the grouped GEMM (`kind=gemm`), a P <= 8 as one pass (`kind=prefill`),
#             `passes` the units; the `stat prompt` line is the host prologue inside it. The prefill
#             buffers are allocated at load, so the wall carries no allocation. The arm's cache
#             height is C (D + N rounded up to 256), not P.
#   ik, lcpp  llama-bench's pp test: llama_decode over the P ids in batches of -b and ubatches of
#             -ub, then one synchronize (test_prompt), one repetition, llama-bench's own value, at
#             n_ctx = P padded to 256. Both trees default to -ub 512 -b 2048; the row names them.
#   mrs       the TTFT case: from sending the request to its first streamed token
#             (run_single_bench, recv_measurement), so it also holds the request's scheduling and
#             the first sample, as ours holds the readback of token 0. mistral.rs batches by its
#             scheduler's defaults, max_num_batched_tokens 4096 per step and
#             max_prefill_chunk_tokens 512 while a decode is resident (mistralrs-cli
#             src/args/mod.rs); bench sets neither, so mrspp has no ubatch lever.
#             A hybrid-cache model under PagedAttention (the qwen35moe profile) has its prompt run in
#             chunks of min(512, budget) even with no decode resident (d5ae0f1 paged_attention/
#             scheduler.rs:217-224, engine/mod.rs:604-606): the row is still its rate for P ids.
# The arms do not start alike. The warmup before the timed run is a 1-token prompt in ik
# (examples/llama-bench/llama-bench.cpp:2230), the whole prompt in mainline
# (tools/llama-bench/llama-bench.cpp:2382) and one whole request in mistral.rs (--warmup 1); ours
# has none. So ik's and our rows carry first-touch costs the other two do not.
# The ubatch lever: every expert sits on the card here, and a ubatch reads each routed expert's
# weights once whatever its row count, so a larger ubatch shares that read among more tokens
# [derived]. ikpp<U> / lcpppp<U> interleaved with the default arm find the references' faster
# setting; the line to beat is a reference at its fastest flags.
#
# Records in the row. Our row reads its engine's records through tools/bloomery/records.py by kind and
# field (--bin generate_qwen3moe): under BLOOMERY_DRAFT=mtp the `mtp summary` (` | mtp E(4) <positions /
# passes> = positions P / passes Q, kept [..]`) and, from the SMOKE line's `passes=` (which a plain run's
# lacks), the wall of one verify pass (` | ms/pass X (mean_ms × steps S / passes Q)`: the counted passes'
# wall over their number; `passes=` with a zero or unreadable pass or step count is a FAIL row), which
# the tables after the decode ratios sum per label (`mean pass`) and compare as passes a second (`ratio
# pass d=`, `ratio pass prose d=`): two builds' MTP rows can generate different text, so their tok/s
# ratio carries E(4) on different text and their pass-time ratio does not; under BLOOMERY_STEP_STATS=1
# the `stat summary`'s host slots a token (` | host slots/token X`), and under adaptive residency the
# `residency lever` word and why and the timed passes' `residency pass` fields (cold-blocks.sh's
# residency sums: ` | residency <word> (<why>) passes n kept k landed l late t made m bytes b`), with each
# label's per-pass means after the tables. A checked-in schema that declares no residency record reads
# none, and an arm that sets BLOOMERY_RESIDENCY under it is a FAIL row naming the schema.
# Co-tenants. A compute process on the other card is recorded ([other-busy], timing-card.sh), and
# the arm's row ends in ` [other-busy]`; the closing summary counts those rows. The CPU is checked before
# and after every arm (guard_cpu, lease.sh, as depth-ds41.sh does: builds and the engines this runner did
# not start, CPU_BUSY_COMMS with generate_qwen3moe, llama-server and mistralrs added unless
# BLOOMERY_CPU_BUSY_COMMS names the list, past
# BLOOMERY_CPU_BUSY_PCT percent of one cpu): a row that met it ends in ` [cpu-busy]`, before its
# [other-busy]; Qwen3.8's host tier and a reference's host layers run on those cores. A compute process on
# the timing card as an arm starts holds the arm until it is gone, and after 10 minutes stops the runner
# with rc 75 and no summary (guard_timing below). BLOOMERY_OTHER_STRICT=1 aborts on either tag instead.
#
# Failures (depth-ds41.sh's contract; the FAIL row, the failed list and the drop are cold-blocks.sh's).
# An arm that exits non-zero, prints no row (no SMOKE line or one without its p50 and mean, no time
# prompt row from ours, no value from a llama-bench or mistralrs bench), or whose fit failed or never ran
# (a fit arm) prints `FAIL r<r> <label> d=<D>|p=<P> rc=<rc> | <why or its last line> | full output:
# <file>` where its row would be, and the runner goes on with the next arm. That label at that depth or
# P drops out of the means and the ratios (the tables name what they dropped), and the runner ends with
# `failed arms: …` and exits 1. A warm-up or a block's discard that fails is `FAIL r0 …` and counts in
# that list; the block's rounds still run. An arm that fails in a load shared with other arms ends that
# process: the arms after it in the round re-run in a fresh load (tools/ref/load-groups.sh), never on
# the failed one; a load that fails before its first arm fails every arm of it. A row's rc is the
# process's (134 an abort, 124/137 the arm bound's timeout), 0 when it exited cleanly with no row.
#
# Environment: BLOOMERY_DECODE_N (N, default 96), BLOOMERY_AB_ROUNDS (rounds, default 4),
# BLOOMERY_GEN_WARM (generate_qwen3moe --warm), BLOOMERY_GEN_CTX (above; mrs arms take the same C),
# BLOOMERY_GEN_BIN (default target/release/generate_qwen3moe), BLOOMERY_ARM_BOUND (seconds one arm
# may run, default 900: a hung arm ends at rc 124/137 as a FAIL row instead of holding the lease; a
# shared load's process has that bound per arm and one more for its load, and one that prints nothing
# for that long is killed),
# BLOOMERY_GEN_PLACE (Qwen3.8 only: a, gate or bp, passed as --place; unset passes none, the binary's a;
# Placement below),
# BLOOMERY_AB_ORDER, BLOOMERY_AB_WARMUP and BLOOMERY_WARM_ROWS (above), BLOOMERY_DRY=1 (print each arm's command line,
# the binaries' tree lines and the rotation, or the blocks and their discards, then exit 0 before the
# lease: nothing is loaded and nothing is timed).
#
# Qwen3.6-35B-A3B runs under the qwen35moe profile (`just depth-gpu-qwen35moe`, BLOOMERY_MODEL=qwen35moe)
# with the same arms, command lines and tables: generate_qwen3moe reads the architecture from the file's
# header. What differs is the ours arm's prefill: Qwen3.6 runs its prompt through `prefill_with` as qwen3moe
# does (`--prefill auto|gemm`: `kind=gemm`, `plan=ubatch:<U>…`; `pass` keeps the 8-position passes), so its
# pp_tok/s is the ubatch path's. The
# profile's reference trees and flags are its own (models/qwen35moe.sh).
#
# Qwen3.8-Flash-Next runs under the qwen4exp profile (`just depth-gpu-qwen4exp`, BLOOMERY_MODEL=qwen4exp)
# the same way: generate_qwen3moe opens Body38, its plan on the A6000 (`--place a`, the default; under
# BLOOMERY_GEN_PLACE=gate every ours arm runs `--place gate`, the 3090's plan, which needs the 3090 as the
# timing card, BLOOMERY_TIMING_GPU, and the profile's -ncmoe is then the 3090's; each row's `load` line
# must name the placement it was given, or the arm is a FAIL row),
# each layer's routed expert prefix on the card as its budget holds and the rest on the host tier
# (BLOOMERY_QWEN38_EXPERTS unset is `card`; an arm with no such variable is a card arm, and `…=host`
# every routed expert on the host tier, the same-binary arm; a row of one does not share a table with
# a row from before the card default). Its prompt runs as the binary's `--prefill` says (`kind=`,
# `plan=` in its `time prompt` row), and `--seed-depth` is refused. Its reference is mainline llama.cpp at
# its profile's hand-set -ncmoe placement and llama-bench's fit (models/qwen4exp.sh).
#
# Placement. A qwen4exp arm's own list may name its placement, `place=<a|gate|bp>` among its NAME=VALUE
# items (`prose:512@place=bp`, `6@place=a,BLOOMERY_QWEN38_EXPERTS=host`): that arm runs at --place <word>
# over BLOOMERY_GEN_PLACE, in a load of its own (the place is in the load key); the label keeps it
# (`ours@prose@place=bp`), so `prose:512 prose:512@place=bp` is the same-binary placement A/B in the prose
# table. place= on another profile, on a reference or server arm, a word outside a|gate|bp, an empty one
# and place given twice are refused by name before anything runs (tools/ref/arm-place.sh, shared with
# depth-ds41.sh and depth-glm5next.sh, its refusals one text in all three). Under one card a placement whose
# card is not the timing card is refused (64), bp too; every row of an arm with a --place names it (`place
# <p>`), and one whose load line names another placement is a FAIL row.
#
# Two cards. BLOOMERY_TIMING_CARDS=a6000+3090 (timing-card.sh has the mode) runs the arms on both cards,
# the A6000 as device 0 and the 3090 as device 1, for the separate "A6000+3090" table (AGENTS.md, user
# 2026-09-28: a model that does not fit one card; the reference on the same two cards, in the same
# lease; the 3090 at its 250 W cap; the witness counting the kernel's Xid lines). Every row's card field
# reads `A6000+3090`, so no reader puts it in the A6000 table. Only a profile with a two-card line
# (TWO_CARD_PLACEMENT, models/qwen4exp.sh: its LCPP_GPU_FLAGS then carry -ts) runs in the mode, and only
# its llama.cpp arms: lcpp, lcpppp[<U>] at the profile's split, the fit arms, whose fit places over
# both cards, and the server arms lcppsrv… on the same flags in the server's spellings, and the ours and
# bin: arms at bp or a, each by its placement (Placement above; no --place runs a): a bp arm loads both cards,
# an a arm plan (a) on the A6000 with the 3090 idle, its row in the A6000+3090 table all the same, and a gate
# arm (a 3090-only row) is refused by name. After each of our arms the `cards=` field of its load line
# (tools/ref/arm-place.sh place_arm_cards) must name the A6000 and the 3090 under bp and the A6000 alone
# under a, or the arm is a FAIL row; a load line with no cards= field (generate_qwen3moe's prints none
# today) is one too. ik and mistral.rs arms are refused by name (no two-card line). Before the lease a card that does not answer, a 3090 off its cap or an unpatched lease.sh
# refuses the run; after every arm an Xid since the last arm, a card lost or off its cap, or a
# llama-bench or llama-server that did not see both cards (its ggml_cuda_init lines) makes the arm a FAIL
# row. A compute
# process on either card as an arm starts is waited out (10 minutes, then rc 75). A dry run prints the
# pre-lease checks' verdict and goes on.
set -uo pipefail
# q3_self_test: `depth-qwen3moe.sh --self-test`, the lever arms' parse and refusals (the header's
# <D>@NAME=VALUE) on fixed arms against this tree's lever registry, and the prose arms' grammar and
# refusals (the header's prose:<P>) against a temp corpus file in the self-test's own temp dir, each a
# --parse-arms run of this file
# in a clean environment under the qwen3moe profile: no card, no binary, no lease (it runs on the Mac,
# under bash 3.2). One line per check, `ok <name>` or `FAIL <name>: …`; the last
# line is the verdict.
q3_self_test() {
  local fails=0 checks=0 out rc me=$0 bin=target/release/generate_qwen3moe
  local nope="BLOOMERY""_NOPE" ex
  # run_parse [NAME=VALUE...] -- <arms...>: this file's --parse-arms under env -i, into out and rc.
  run_parse() {
    local -a pre=()
    while [ "$1" != -- ]; do
      pre+=("$1")
      shift
    done
    shift
    out=$(env -i PATH="$PATH" BLOOMERY_MODEL=qwen3moe BLOOMERY_AB_ROUNDS=2 ${pre[@]+"${pre[@]}"} "$BASH" "$me" --parse-arms "$@" 2>&1)
    rc=$?
  }
  # want <name> <rc> <line or fixed text>...: rc as given; with rc 0 each text is a whole line of the
  # output, otherwise a part of it.
  want() {
    local name=$1 want_rc=$2 t bad=''
    shift 2
    checks=$((checks + 1))
    [ "$rc" = "$want_rc" ] || bad="rc $rc, want $want_rc"
    for t in "$@"; do
      if [ "$want_rc" = 0 ]; then
        grep -qxF -- "$t" <<< "$out" || bad="${bad:+$bad; }no line [$t]"
      else
        grep -qF -- "$t" <<< "$out" || bad="${bad:+$bad; }no [$t]"
      fi
    done
    if [ -z "$bad" ]; then
      echo "ok $name"
    else
      echo "FAIL $name: $bad"
      while IFS= read -r t; do echo "    | $t"; done <<< "$out"
      fails=$((fails + 1))
    fi
  }
  run_parse -- 6 6@BLOOMERY_THREADS=8 ik:6
  want lever-arm 0 \
    "[parse] 6: kind=ours depth=6 label=ours env=- load=$bin|ctx=256" \
    "[parse] 6@BLOOMERY_THREADS=8: kind=ours depth=6 label=ours@BLOOMERY_THREADS=8 env=BLOOMERY_THREADS=8 load=$bin|ctx=256|BLOOMERY_THREADS=8" \
    "[parse] ik:6: kind=ref depth=6 label=ik env=- load=(a process of its own)" \
    "[parse] round 1 order: 6 6@BLOOMERY_THREADS=8 ik:6" \
    "[parse] round 2 order: 6@BLOOMERY_THREADS=8 ik:6 6" \
    "[parse] load: $bin --arm <lcg_prompt 6> -n 96 --ctx 256 --time --arm-sync" \
    "[parse] load: env BLOOMERY_THREADS=8 $bin --arm <lcg_prompt 6> -n 96 --ctx 256 --time --arm-sync"
  # Two lever arms of one list in another order share a load; the solo marker takes the arm out of it
  # and stays in the label only.
  run_parse -- 6@BLOOMERY_THREADS=8,BLOOMERY_SPIN=0 6@BLOOMERY_SPIN=0,BLOOMERY_THREADS=8 5@BLOOMERY_THREADS=8,BLOOMERY_AB_LOAD=arm
  want lever-load 0 \
    "[parse] 6@BLOOMERY_THREADS=8,BLOOMERY_SPIN=0: kind=ours depth=6 label=ours@BLOOMERY_THREADS=8,BLOOMERY_SPIN=0 env=BLOOMERY_THREADS=8,BLOOMERY_SPIN=0 load=$bin|ctx=256|BLOOMERY_SPIN=0,BLOOMERY_THREADS=8" \
    "[parse] 6@BLOOMERY_SPIN=0,BLOOMERY_THREADS=8: kind=ours depth=6 label=ours@BLOOMERY_SPIN=0,BLOOMERY_THREADS=8 env=BLOOMERY_SPIN=0,BLOOMERY_THREADS=8 load=$bin|ctx=256|BLOOMERY_SPIN=0,BLOOMERY_THREADS=8" \
    "[parse] 5@BLOOMERY_THREADS=8,BLOOMERY_AB_LOAD=arm: kind=ours depth=5 label=ours@BLOOMERY_THREADS=8,BLOOMERY_AB_LOAD=arm env=BLOOMERY_THREADS=8 load=$bin|ctx=256|BLOOMERY_THREADS=8|solo" \
    "[parse] round 1 loads: [6@BLOOMERY_THREADS=8,BLOOMERY_SPIN=0 6@BLOOMERY_SPIN=0,BLOOMERY_THREADS=8] [5@BLOOMERY_THREADS=8,BLOOMERY_AB_LOAD=arm]" \
    "[parse] round 2 loads: [5@BLOOMERY_THREADS=8,BLOOMERY_AB_LOAD=arm] [6@BLOOMERY_SPIN=0,BLOOMERY_THREADS=8 6@BLOOMERY_THREADS=8,BLOOMERY_SPIN=0]" \
    "[parse] load: env BLOOMERY_THREADS=8 BLOOMERY_SPIN=0 $bin --arm <lcg_prompt 6> --arm <lcg_prompt 6> -n 96 --ctx 256 --time --arm-sync" \
    "[parse] load: env BLOOMERY_THREADS=8 $bin --arm <lcg_prompt 5> -n 96 --ctx 256 --time --arm-sync"
  # A run with no lever arm never reads the registry (the box stub test's tree has none).
  run_parse -- --registry /nonexistent/registry.rs 6 lcpp:6
  want no-lever-no-registry 0 "[parse] 6: kind=ours depth=6 label=ours env=- load=$bin|ctx=256"
  run_parse -- --registry /nonexistent/registry.rs 6@BLOOMERY_THREADS=8
  want registry-missing 2 "no lever registry at /nonexistent/registry.rs"
  # The refusals, each by name before anything runs.
  run_parse -- 6@
  want empty-list 64 "an empty NAME=VALUE list"
  run_parse -- 6@BLOOMERY_THREADS=8,
  want empty-item 64 "an empty item in"
  run_parse -- 6@=8
  want empty-name 64 "'=8' has an empty name"
  run_parse -- 6@BLOOMERY_THREADS=
  want empty-value 64 "BLOOMERY_THREADS has an empty value"
  run_parse -- 6@BLOOMERY_THREADS=8,9
  want comma-in-value 64 "'9' is no NAME=VALUE"
  run_parse -- 6@BLOOMERY_THREADS=8@9
  want at-in-value 64 "the value of BLOOMERY_THREADS holds '@'"
  run_parse -- '6@BLOOMERY_THREADS=8 9'
  want space-in-value 64 "white space"
  run_parse -- '6@BLOOMERY_THREADS=8|9'
  want bar-in-value 64 "the value of BLOOMERY_THREADS holds '|'"
  run_parse -- 6@BLOOMERY_THREADS=8,BLOOMERY_THREADS=9
  want twice 64 "BLOOMERY_THREADS is given twice"
  run_parse -- 6@FOO=1
  want no-row 64 "FOO is no row of the lever registry"
  run_parse -- "6@$nope=1"
  want no-row-bloomery 64 "$nope is no row of the lever registry"
  run_parse -- 6@BLOOMERY_GEN_CTX=512
  want runner-row 64 "BLOOMERY_GEN_CTX is no lever"
  run_parse -- 6@BLOOMERY_CARD_EXPERTS=tile
  want retired-row 64 "BLOOMERY_CARD_EXPERTS is a retired name"
  run_parse BLOOMERY_THREADS=8 -- 6 6@BLOOMERY_THREADS=4
  want inherited 64 "BLOOMERY_THREADS is set in the runner's own environment"
  for ex in ik:6@BLOOMERY_THREADS=8 lcpp:6@BLOOMERY_THREADS=8 lcppsrv:6@BLOOMERY_THREADS=8; do
    run_parse -- "$ex"
    want "ref-${ex%%:*}" 64 "a lever of ours is not a reference's"
  done
  run_parse -- bin:/root/repo/b/target/release/generate_qwen3moe:6@BLOOMERY_THREADS=8
  want bin-arm 64 "a bin: arm is another build"
  # The reader holds the registry's shape: a row it cannot read stops it by name.
  printf 'pub(crate) static REGISTRY: &[LeverSpec] = &[\n    LeverSpec {\n        class: Class::A,\n    },\n];\n' > "${TMPDIR:-/tmp}/q3arm-registry.$$.rs"
  run_parse -- --registry "${TMPDIR:-/tmp}/q3arm-registry.$$.rs" 6@BLOOMERY_THREADS=8
  rm -f "${TMPDIR:-/tmp}/q3arm-registry.$$.rs"
  want registry-shape 2 "1 LeverSpec rows, 0 read with a name and a site"
  # The aggregate arm (the header's <D>@BLOOMERY_GEN_SLOTS=N): a lever arm whose load feeds N·D ids, in a
  # load of its own; N = 1 a plain lever arm; each refusal by name.
  run_parse -- 6 6@BLOOMERY_GEN_SLOTS=2 6@BLOOMERY_GEN_SLOTS=1
  want slots-arm 0 \
    "[parse] 6@BLOOMERY_GEN_SLOTS=2: kind=ours depth=6 label=ours@BLOOMERY_GEN_SLOTS=2 env=BLOOMERY_GEN_SLOTS=2 load=$bin|ctx=256|BLOOMERY_GEN_SLOTS=2 slots=2 feed=12" \
    "[parse] 6@BLOOMERY_GEN_SLOTS=1: kind=ours depth=6 label=ours@BLOOMERY_GEN_SLOTS=1 env=BLOOMERY_GEN_SLOTS=1 load=$bin|ctx=256|BLOOMERY_GEN_SLOTS=1" \
    "[parse] load: env BLOOMERY_GEN_SLOTS=2 $bin --arm <lcg_prompt 12> -n 96 --ctx 256 --time --arm-sync" \
    "[parse] load: env BLOOMERY_GEN_SLOTS=1 $bin --arm <lcg_prompt 6> -n 96 --ctx 256 --time --arm-sync"
  run_parse -- 6@BLOOMERY_GEN_SLOTS=x
  want slots-count 64 "BLOOMERY_GEN_SLOTS=x is no slot count"
  run_parse -- 10001@BLOOMERY_GEN_SLOTS=2
  want slots-cap 64 "feeds 2 slots × 10001 = 20002 ids, past the 20000"
  run_parse BLOOMERY_GEN_SLOTS=2 -- 6
  want slots-env 64 "BLOOMERY_GEN_SLOTS=2 is set in the runner's own environment"
  # The prose arms (the header's prose:<P>): the grammar and the load it shares with an lcg arm of
  # the same C against a temp corpus in this self-test's own temp dir, then every refusal.
  local pt
  pt=$(mktemp -d "${TMPDIR:-/tmp}/q3arm-prose.XXXXXX")
  mkdir -p "$pt/data/qwen3moe" "$pt/bad/qwen3moe" "$pt/none"
  seq 100 800 > "$pt/data/qwen3moe/corpus-prose.ids"
  { head -n 4 "$pt/data/qwen3moe/corpus-prose.ids"; echo x1; tail -n +5 "$pt/data/qwen3moe/corpus-prose.ids"; } > "$pt/bad/qwen3moe/corpus-prose.ids"
  run_parse BLOOMERY_DATA="$pt/data" -- 512 prose:512 prose:512@BLOOMERY_QWEN38_EXPERTS=host 4 prose:4 prose:4@BLOOMERY_AB_LOAD=arm
  want prose-arm 0 \
    "[parse] prose:512: kind=ours depth=512 label=ours@prose env=- load=$bin|ctx=768 corpus=$pt/data/qwen3moe/corpus-prose.ids ids=100,101,102" \
    "[parse] prose:512@BLOOMERY_QWEN38_EXPERTS=host: kind=ours depth=512 label=ours@prose@BLOOMERY_QWEN38_EXPERTS=host env=BLOOMERY_QWEN38_EXPERTS=host load=$bin|ctx=768|BLOOMERY_QWEN38_EXPERTS=host corpus=$pt/data/qwen3moe/corpus-prose.ids ids=100,101,102" \
    "[parse] prose:4: kind=ours depth=4 label=ours@prose env=- load=$bin|ctx=256 corpus=$pt/data/qwen3moe/corpus-prose.ids ids=100,101,102" \
    "[parse] prose:4@BLOOMERY_AB_LOAD=arm: kind=ours depth=4 label=ours@prose@BLOOMERY_AB_LOAD=arm env=- load=$bin|ctx=256|solo corpus=$pt/data/qwen3moe/corpus-prose.ids ids=100,101,102" \
    "[parse] round 1 loads: [512 prose:512] [prose:512@BLOOMERY_QWEN38_EXPERTS=host] [4 prose:4] [prose:4@BLOOMERY_AB_LOAD=arm]" \
    "[parse] load: $bin --arm <lcg_prompt 512> --arm <prose_prompt 512> -n 96 --ctx 768 --time --arm-sync" \
    "[parse] load: env BLOOMERY_QWEN38_EXPERTS=host $bin --arm <prose_prompt 512> -n 96 --ctx 768 --time --arm-sync" \
    "[parse] load: $bin --arm <lcg_prompt 4> --arm <prose_prompt 4> -n 96 --ctx 256 --time --arm-sync"
  # Under --parse-arms a corpus that is not readable (the box's path, read on the Mac) is named on
  # the arm's line, not refused; a run refuses it before anything else.
  run_parse BLOOMERY_DATA="$pt/none" -- prose:512
  want prose-parse-nofile 0 "[parse] prose:512: kind=ours depth=512 label=ours@prose env=- load=$bin|ctx=768 corpus=$pt/none/qwen3moe/corpus-prose.ids ids=-"
  out=$(env -i PATH="$PATH" BLOOMERY_MODEL=qwen3moe BLOOMERY_DATA="$pt/none" "$BASH" "$me" prose:512 2>&1)
  rc=$?
  want prose-nofile 2 "no prose corpus at $pt/none/qwen3moe/corpus-prose.ids"
  run_parse BLOOMERY_DATA="$pt/data" -- prose:702
  want prose-past 64 "a prose prompt of 702 ids; $pt/data/qwen3moe/corpus-prose.ids holds 701 (1..701)"
  # An aggregate prose arm feeds the corpus's first N·P ids, the plain prose arm's P first.
  run_parse BLOOMERY_DATA="$pt/data" -- prose:4 prose:4@BLOOMERY_GEN_SLOTS=2
  want prose-slots 0 \
    "[parse] prose:4@BLOOMERY_GEN_SLOTS=2: kind=ours depth=4 label=ours@prose@BLOOMERY_GEN_SLOTS=2 env=BLOOMERY_GEN_SLOTS=2 load=$bin|ctx=256|BLOOMERY_GEN_SLOTS=2 corpus=$pt/data/qwen3moe/corpus-prose.ids ids=100,101,102 slots=2 feed=8" \
    "[parse] load: env BLOOMERY_GEN_SLOTS=2 $bin --arm <prose_prompt 8> -n 96 --ctx 256 --time --arm-sync"
  run_parse BLOOMERY_DATA="$pt/data" -- prose:400@BLOOMERY_GEN_SLOTS=2
  want prose-slots-past 64 "a prose prompt of 800 ids (2 slots × 400); $pt/data/qwen3moe/corpus-prose.ids holds 701 (1..701)"
  run_parse BLOOMERY_DATA="$pt/data" -- prose:0
  want prose-zero 64 "a prose prompt of 0 ids; $pt/data/qwen3moe/corpus-prose.ids holds 701 (1..701)"
  run_parse BLOOMERY_DATA="$pt/bad" -- prose:6
  want prose-notid 2 "line 5 of the first 6 of $pt/bad/qwen3moe/corpus-prose.ids ('x1') is no id"
  # A prose arm's NAME=VALUE list is a <D> arm's: the registry still gates it.
  run_parse BLOOMERY_DATA="$pt/data" -- prose:4@FOO=1
  want prose-lever 64 "FOO is no row of the lever registry"
  # prose: feeds ours or a server, never a reference or a bin: arm.
  run_parse -- ik:prose:512
  want prose-ref 64 "ik: is a reference engine, which feeds its own prompt ids"
  # A server arm on the prose ids, and the arms after it keep their own prompts: the lcg arm the LCG
  # walk, the prose arm the corpus (a server arm that took no prompt slot swaps them).
  run_parse BLOOMERY_DATA="$pt/data" -- lcppsrv:prose:512 lcppsrvpp4096+nopo0:prose:512 512 prose:512
  want prose-srv 0 \
    "[parse] lcppsrv:prose:512: kind=srv depth=512 label=lcppsrv@prose env=- load=(a process of its own)" \
    "[parse] lcppsrvpp4096+nopo0:prose:512: kind=srv depth=512 label=lcppsrvpp4096+nopo0@prose env=- load=(a process of its own)" \
    "[parse] load: $bin --arm <lcg_prompt 512> --arm <prose_prompt 512> -n 96 --ctx 768 --time --arm-sync"
  run_parse BLOOMERY_DATA="$pt/data" -- lcppsrv:prose:702
  want prose-srv-past 64 "a prose prompt of 702 ids; $pt/data/qwen3moe/corpus-prose.ids holds 701 (1..701)"
  run_parse -- lcppsrv+x1:512
  want srv-flag-bad 64 "'+x1' in 'lcppsrv+x1' is none of +t<N> (-t), +nopo<0|1> (-nopo), +k<K> (--n-cpu-moe)"
  # The placement (BLOOMERY_GEN_PLACE): a qwen4exp file's only, in the load key and the command line.
  run_parse BLOOMERY_GEN_PLACE=gate -- 6
  want place-other-file 64 "BLOOMERY_GEN_PLACE=gate: only a qwen4exp file takes --place"
  out=$(env -i PATH="$PATH" BLOOMERY_MODEL=qwen4exp BLOOMERY_AB_ROUNDS=2 BLOOMERY_GEN_PLACE=gate "$BASH" "$me" --parse-arms 6 2>&1)
  rc=$?
  want place-gate 0 "[parse] 6: kind=ours depth=6 label=ours env=- load=$bin|ctx=256|place=gate" \
    "[parse] load: $bin --arm <lcg_prompt 6> -n 96 --ctx 256 --place gate --time --arm-sync"
  run_parse BLOOMERY_GEN_PLACE=b -- 6
  want place-bad 64 "BLOOMERY_GEN_PLACE is a (the A6000's plan), gate (the 3090's) or bp"
  # An arm's place= (tools/ref/arm-place.sh): a qwen4exp file's only, the arm's own load key and --place,
  # never a variable of its process; the lever list beside it still gated by the registry.
  run_parse -- 6@place=a
  want place-arm-other-file 64 "depth-qwen3moe.sh: arm '6@place=a': place=a: only a qwen4exp file takes --place"
  out=$(env -i PATH="$PATH" BLOOMERY_MODEL=qwen4exp BLOOMERY_AB_ROUNDS=2 "$BASH" "$me" --parse-arms 6 6@place=gate 6@place=gate,BLOOMERY_QWEN38_EXPERTS=host 2>&1)
  rc=$?
  want place-arm 0 "[parse] 6: kind=ours depth=6 label=ours env=- load=$bin|ctx=256" \
    "[parse] 6@place=gate: kind=ours depth=6 label=ours@place=gate env=- load=$bin|ctx=256|place=gate" \
    "[parse] 6@place=gate,BLOOMERY_QWEN38_EXPERTS=host: kind=ours depth=6 label=ours@place=gate,BLOOMERY_QWEN38_EXPERTS=host env=BLOOMERY_QWEN38_EXPERTS=host load=$bin|ctx=256|place=gate|BLOOMERY_QWEN38_EXPERTS=host" \
    "[parse] load: $bin --arm <lcg_prompt 6> -n 96 --ctx 256 --place gate --time --arm-sync"
  out=$(env -i PATH="$PATH" BLOOMERY_MODEL=qwen4exp BLOOMERY_AB_ROUNDS=2 "$BASH" "$me" --parse-arms 6@place=b2 2>&1)
  rc=$?
  want place-arm-word 64 "depth-qwen3moe.sh: arm '6@place=b2': place=b2: generate_qwen3moe takes --place a, gate or bp in this runner"
  run_parse -- lcpp:6@place=a
  want place-arm-ref 64 "depth-qwen3moe.sh: arm 'lcpp:6@place=a': place= sets generate_qwen3moe's --place, and lcpp is a reference engine's arm, which takes no placement of ours"
  run_parse -- 6@place=a,FOO=1
  want place-arm-lever 64 "FOO is no row of the lever registry"
  run_parse BLOOMERY_DATA="$pt/data" -- prose:4 bin:/root/r/t/target/release/generate_qwen3moe:prose:4
  want prose-bin 0 \
    "[parse] bin:/root/r/t/target/release/generate_qwen3moe:prose:4: kind=bin depth=4 label=bin:t@prose env=- load=(a process of its own) corpus=$pt/data/qwen3moe/corpus-prose.ids ids=100,101,102"
  run_parse BLOOMERY_DATA="$pt/data" -- bin:/root/r/t/release/generate_qwen3moe:prose
  want prose-bin-nop 64 "bin:<path>:prose:<P> takes a prompt length P after prose:"
  run_parse BLOOMERY_DATA="$pt/data" -- bin:/root/r/t/release/generate_qwen3moe:prose:900
  want prose-bin-past 64 "a prose prompt of 900 ids"
  rm -rf "$pt"
  echo "self-test: $([ "$fails" = 0 ] && echo ok || echo FAIL) ($checks checks, $fails failures)"
  [ "$fails" = 0 ]
}
# The modes that run nothing on the box (the header's <D>@NAME=VALUE): --parse-arms prints the parsed
# arms and exits before the card, the binaries and the lease; --registry names another registry file.
PARSE_ONLY=
LEVER_REGISTRY=${BASH_SOURCE[0]%/*}/../../crates/levers/src/registry.rs
case ${1:-} in
  --self-test)
    q3_self_test
    exit
    ;;
  --parse-arms)
    PARSE_ONLY=1
    shift
    if [ "${1:-}" = --registry ]; then
      [ $# -ge 2 ] || { echo "depth-qwen3moe.sh: --registry takes a file" >&2; exit 64; }
      LEVER_REGISTRY=$2
      shift 2
    fi
    ;;
esac
# The profile (MODEL, IK, IKBIN, IK_GPU_FLAGS, IK_GPU_DEFAULT_FLAGS, LCPP, LCPPBIN, LCPP_GPU_FLAGS,
# MRS, MRSBIN, MRS_FLAGS); tools/box.sh exports its MODEL to our binary as BLOOMERY_REF_MODEL, so
# the four engines open one file.
# shellcheck source=tools/ref/ref-paths.sh
source "${BASH_SOURCE[0]%/*}/ref-paths.sh"
case $MODEL_NAME in
  qwen3moe | qwen35moe | qwen4exp) ;;
  *)
    echo "depth-qwen3moe.sh: the profile is $MODEL_NAME — pick qwen3moe, qwen35moe or qwen4exp on the Mac side (BLOOMERY_MODEL=…)" >&2
    exit 64
    ;;
esac
N=${BLOOMERY_DECODE_N:-96}
WARM=${BLOOMERY_GEN_WARM:-}
ROUNDS=${BLOOMERY_AB_ROUNDS:-4}
BIN=${BLOOMERY_GEN_BIN:-target/release/generate_qwen3moe}
BOUND=${BLOOMERY_ARM_BOUND:-900}
GEN_CTX=${BLOOMERY_GEN_CTX:-}
PLACE=${BLOOMERY_GEN_PLACE:-}
DRY=${BLOOMERY_DRY:-}
case $PLACE in
  '') ;;
  a | gate | bp)
    [ "$MODEL_NAME" = qwen4exp ] || { echo "depth-qwen3moe.sh: BLOOMERY_GEN_PLACE=$PLACE: only a qwen4exp file takes --place (generate_qwen3moe refuses it on $MODEL_NAME's)" >&2; exit 64; }
    ;;
  *) echo "depth-qwen3moe.sh: BLOOMERY_GEN_PLACE is a (the A6000's plan), gate (the 3090's) or bp (plan (b′), the two-card mode's), or unset (no --place: the binary's a), got '$PLACE'" >&2; exit 64 ;;
esac
case $GEN_CTX in
  *[!0-9]* | 0) echo "depth-qwen3moe.sh: BLOOMERY_GEN_CTX is a positive integer, got '$GEN_CTX'" >&2; exit 64 ;;
esac
# The aggregate arm's lever is an arm's own (the header's <D>@BLOOMERY_GEN_SLOTS=N): set here it would
# reach every ours arm, whose plain rows would read a per stream rate under the label `ours`.
if [ -n "${BLOOMERY_GEN_SLOTS+x}" ]; then
  echo "depth-qwen3moe.sh: BLOOMERY_GEN_SLOTS=$BLOOMERY_GEN_SLOTS is set in the runner's own environment, which every arm inherits: name it per arm (<D>@BLOOMERY_GEN_SLOTS=N), whose row reads the aggregate" >&2
  exit 64
fi
# The fault witness, the cold tag, ROW_TAG, the order's blocks and the FAIL rows: shared with
# depth-ds41.sh. A failed arm's whole output goes to
# ${TMPDIR:-/tmp}/depth-qwen3moe-<label>-<d|p><key>-r<round>.log.
# shellcheck source=tools/ref/cold-blocks.sh
source "${BASH_SOURCE[0]%/*}/cold-blocks.sh" || exit 2
ARM_FAIL_STEM=depth-qwen3moe
ab_order depth-qwen3moe.sh
warm_rows_init depth-qwen3moe.sh
AB_WARMUP_DEFAULT=0
[ "$ORDER" = rotate ] || AB_WARMUP_DEFAULT=1
AB_WARMUP=${BLOOMERY_AB_WARMUP:-$AB_WARMUP_DEFAULT}
case $AB_WARMUP in
  0 | 1) ;;
  *) echo "depth-qwen3moe.sh: BLOOMERY_AB_WARMUP is 1 (one discarded run of the first arm under rotate; the blocks' discards under blocks, the default there) or 0 (none; the default under rotate), got '$AB_WARMUP'" >&2; exit 64 ;;
esac
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(6 ik:6 lcpp:6)
ours=0 ik=0 lcpp=0 lcppfit=0 mrs=0 srv=0
# Per arm, by its index in ARMS: the kind (ours, ref, srv or bin), the depth, the row label, the engine
# (ours, bin, or the reference's), the binary (ours and bin) and a server arm's ids (lcg).
A_KIND=() A_DEP=() A_LABEL=() A_ENG=() A_BIN=() A_IDS=()
# A lever arm's NAME=VALUE list as given less its place= item (comma-separated; empty for every other
# arm): the variables its process gets.
A_ENV=()
# An ours or bin: arm's placement (its place= word, else BLOOMERY_GEN_PLACE; empty: no --place) and whether
# place= set it (1, or empty); empty for a reference or server arm.
A_PLACE=() A_PLACE_SET=()
# A prose arm's prompt, the corpus's first P ids comma-separated as --tokens takes them (empty for
# every other arm); under --parse-arms the placeholder `<prose_prompt P>` its load lines print.
A_TOK=()
# An aggregate arm's N (the header's <D>@BLOOMERY_GEN_SLOTS=N, N >= 2; empty for every other arm).
A_SLOTS=()
arm_usage() {
  echo "depth-qwen3moe.sh: arm '$1' is <D>, <D>@NAME=VALUE[,NAME=VALUE...], prose:<P>[@NAME=VALUE,...], ik:<D>, ikdef:<D>, lcpp:<D>, lcppfit:<D>, mrs:<D>, mrspa0:<D>, ikpp[<U>]:<P>, lcpppp[<U>]:<P>, lcppppfit[<U>]:<P>, mrspp:<P>, lcppsrv[fit]:<D>, lcppsrvpp[fit][<U>]:<P> or bin:<path>:<D>" >&2
  exit 64
}
arm_refuse() {
  echo "depth-qwen3moe.sh: arm '$1': $2" >&2
  exit 64
}
# The prose corpus (the header's prose:<P>): the profile's own file, one id a line — the file its d1k
# reference set is cut from. PROSE_N, its line count, is read once, when the first prose arm names it.
PROSE_N=
corpus_file() { echo "${BLOOMERY_DATA:-}/$MODEL_NAME/corpus-prose.ids"; }
# corpus_check <arm> <P> [<note>]: the corpus file's checks, each refusal by name before anything runs: the
# file readable, P within 1..its line count, and every one of its first P lines one id. Under
# --parse-arms a file that is not readable (the box's path, read on the Mac) is named on the arm's
# line, not refused: a run reads it, and refuses.
corpus_check() {
  local file bad
  file=$(corpus_file)
  if [ ! -r "$file" ]; then
    [ -z "$PARSE_ONLY" ] || return 0
    echo "depth-qwen3moe.sh: arm '$1': no prose corpus at $file (\$BLOOMERY_DATA/$MODEL_NAME/corpus-prose.ids, one id a line)" >&2
    exit 2
  fi
  [ -n "$PROSE_N" ] || PROSE_N=$(($(wc -l < "$file")))
  if [ "$2" -lt 1 ] || [ "$2" -gt "$PROSE_N" ]; then
    arm_refuse "$1" "a prose prompt of $2 ids${3:+ ($3)}; $file holds $PROSE_N (1..$PROSE_N)"
  fi
  bad=$(head -n "$2" "$file" | grep -nvE '^[0-9]+$' | head -n 1)
  if [ -n "$bad" ]; then
    echo "depth-qwen3moe.sh: arm '$1': line ${bad%%:*} of the first $2 of $file ('${bad#*:}') is no id: the file is one id a line" >&2
    exit 2
  fi
}
# corpus_ids <P>: the corpus's first P ids, comma-separated, as --tokens takes them; under
# --parse-arms the placeholder `<prose_prompt P>`, as the load lines print `<lcg_prompt D>`.
corpus_ids() {
  if [ -n "$PARSE_ONLY" ]; then echo "<prose_prompt $1>"; else head -n "$1" "$(corpus_file)" | paste -sd, -; fi
}
# The lever arms' list checks (the header's <D>@NAME=VALUE): tools/ref/lever-arms.sh, shared with
# depth-glm5next.sh. BLOOMERY_AB_LOAD=arm is the load driver's solo marker (load-groups.sh), never the
# binary's.
LEVER_ARM_RUNNER=depth-qwen3moe.sh LEVER_ARM_PASS=BLOOMERY_AB_LOAD=arm
# shellcheck source=tools/ref/lever-arms.sh
source "${BASH_SOURCE[0]%/*}/lever-arms.sh" || exit 2
# An ours arm's place= (the header's Placement): tools/ref/arm-place.sh, shared with depth-ds41.sh and
# depth-glm5next.sh. generate_qwen3moe's load line is no records.py kind: its cards are read from it here.
# shellcheck disable=SC2034 # read by tools/ref/arm-place.sh
PLACE_RUNNER=depth-qwen3moe.sh PLACE_BIN=generate_qwen3moe PLACE_WORDS='a gate bp'
# shellcheck source=tools/ref/arm-place.sh
source "${BASH_SOURCE[0]%/*}/arm-place.sh" || exit 2
# split_at <arm> <list>: an ours arm's @ list into AT (as given: its label's), ENVS (its lever list, checked
# by lever-arms.sh) and APLACE (its place= word, empty without one), which only a qwen4exp file takes.
split_at() {
  AT=$2
  [ -n "$2" ] || arm_envs_ok "$1" "$2"
  arm_place_split "$1" "$2"
  ENVS=$ARM_REST APLACE=$ARM_PLACE
  [ -z "$ENVS" ] || arm_envs_ok "$1" "$ENVS"
  if [ -n "$APLACE" ] && [ "$MODEL_NAME" != qwen4exp ]; then
    place_refuse "$1" "place=$APLACE: only a qwen4exp file takes --place (generate_qwen3moe refuses it on $MODEL_NAME's)"
  fi
}
# arm_slots <arm> <lever list>: an aggregate arm's N into ASLOTS (the header's <D>@BLOOMERY_GEN_SLOTS=N),
# empty when the list names none or N = 1 (a plain lever arm); an N that is not a whole number is refused
# by name, the feed being N·D ids. Its range is the binary's (at_main refuses it by name).
arm_slots() {
  local e
  local -a kv=()
  ASLOTS=''
  [ -z "$2" ] || IFS=, read -r -a kv <<< "$2"
  for e in ${kv[@]+"${kv[@]}"}; do
    case $e in BLOOMERY_GEN_SLOTS=*) ASLOTS=${e#*=} ;; esac
  done
  case $ASLOTS in
    '') return 0 ;;
    0* | *[!0-9]*) arm_refuse "$1" "BLOOMERY_GEN_SLOTS=$ASLOTS is no slot count (a whole number from 1, no leading zero), and the arm feeds N·D ids" ;;
  esac
  [ "$ASLOTS" != 1 ] || ASLOTS=''
}
# arm_feed <i>: the ids arm <i> feeds: its depth, N·D for an aggregate arm.
arm_feed() { echo $((A_DEP[$1] * ${A_SLOTS[$1]:-1})); }
# A prefill arm's engine (ikpp[<U>], lcpppp[<U>], lcppppfit[<U>], mrspp), and its ubatch lever U (empty:
# the default).
pp_eng() { case $1 in ikpp* | lcpppp* | mrspp) return 0 ;; *) return 1 ;; esac; }
pp_ub() { local u=${1#ikpp}; u=${u#lcpppp}; u=${u#fit}; echo "${u#mrspp}"; }
# The fit arms' flags, probe and column (lcppfit, lcppppfit[<U>]).
# shellcheck source=tools/ref/lcpp-fit.sh
source "${BASH_SOURCE[0]%/*}/lcpp-fit.sh" || exit 2
# The server arms (lcppsrv…): lcpp-warm.sh.
# shellcheck source=tools/ref/lcpp-warm.sh
source "${BASH_SOURCE[0]%/*}/lcpp-warm.sh" || exit 2
for a in "${ARMS[@]}"; do
  kind=ref eng=${a%%:*} dep=${a#*:} label='' bin='' envs='' tok='' AT='' APLACE='' ASLOTS=''
  # `@` is ours only (a <D> arm's or a prose arm's list): split at it first, so a value with a `:` is
  # not read as a reference's arm. place= on a reference arm is refused by name (the header's Placement).
  case ${a%%@*} in
    "$a") ;;
    prose:*) ;;
    bin:*) arm_refuse "$a" "'@' sets a lever of this tree's binary, and a bin: arm is another build, run as it is" ;;
    *:*)
      ! arm_has_place "${a#*@}" || place_ref_refuse "$a" "${eng%%+*}"
      arm_refuse "$a" "'@' sets a lever of ours, and ${a%%:*}: is a reference engine's arm — a lever of ours is not a reference's"
      ;;
  esac
  if srv_eng "$eng"; then
    ids=lcg label=$eng
    case $dep in
      prose:*)
        dep=${dep#prose:} ids=prose label=$eng@prose
        case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
        corpus_check "$a" "$dep"
        tok=$(corpus_ids "$dep")
        ;;
    esac
    case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
    [ "$dep" -ge 1 ] || { echo "depth-qwen3moe.sh: arm '$a': a server arm sends at least one id" >&2; exit 64; }
    srv_check_arm "$a" || { echo "depth-qwen3moe.sh: arm '$a': $SRV_WHY" >&2; exit 64; }
    srv=1
    A_KIND+=(srv) A_DEP+=("$dep") A_LABEL+=("$label") A_ENG+=("$eng") A_BIN+=('') A_IDS+=("$ids") A_ENV+=('') A_TOK+=("$tok") A_SLOTS+=('')
    A_PLACE+=('') A_PLACE_SET+=('')
    continue
  fi
  case $a in
    bin:*)
      kind=bin eng=bin bin=${a#bin:}
      dep=${bin##*:} bin=${bin%:*}
      corpus=''
      if [ "${bin##*:}" = prose ]; then
        # bin:<path>:prose:<P>: the other build on the corpus's first P ids, the prose arms' prompt.
        corpus=1 bin=${bin%:prose}
      elif [ "$dep" = prose ]; then
        arm_refuse "$a" "bin:<path>:prose:<P> takes a prompt length P after prose:"
      fi
      case $bin in /*) ;; *) arm_usage "$a" ;; esac
      tree=${bin%/target/*}
      [ "$tree" != "$bin" ] || tree=${bin%/*}
      label=bin:${tree##*/}
      if [ -n "$corpus" ]; then
        case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
        corpus_check "$a" "$dep"
        tok=$(corpus_ids "$dep")
        label+=@prose
      fi
      ;;
    prose:*)
      # prose:<P>[@NAME=VALUE,...]: ours on the corpus's first P ids (the header's prose:<P>).
      kind=ours eng=prose bin=$BIN
      dep=${a#prose:}
      label=ours@prose
      case $dep in *@*) split_at "$a" "${dep#*@}" && envs=$ENVS dep=${dep%%@*} label=ours@prose@$AT ;; esac
      case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
      # An aggregate arm feeds the corpus's first N·P ids, slot j the window j.
      arm_slots "$a" "$envs"
      corpus_check "$a" "$((dep * ${ASLOTS:-1}))" "${ASLOTS:+$ASLOTS slots × $dep}"
      tok=$(corpus_ids "$((dep * ${ASLOTS:-1}))")
      ours=1
      ;;
    *@*)
      kind=ours eng=ours dep=${a%%@*} bin=$BIN label=ours@${a#*@} ours=1
      split_at "$a" "${a#*@}"
      envs=$ENVS
      arm_slots "$a" "$envs"
      ;;
    *:*)
      case $dep in prose:*) arm_refuse "$a" "prose:<P> is ours on the corpus's first P ids, and ${eng}: is a reference engine, which feeds its own prompt ids" ;; esac
      case $eng in
        ik | ikdef | ikpp | ikpp[1-9]*) ik=1 ;;
        lcpp | lcpppp | lcpppp[1-9]*) lcpp=1 ;;
        lcppfit | lcppppfit | lcppppfit[1-9]*)
          lcpp=1 lcppfit=1
          lcpp_fit_flags "$LCPP_GPU_FLAGS" || { echo "depth-qwen3moe.sh: arm '$a': $FIT_WHY" >&2; exit 64; }
          ;;
        mrs | mrspa0 | mrspp) mrs=1 ;;
        *) arm_usage "$a" ;;
      esac
      label=$eng
      ;;
    *) kind=ours eng=ours dep=$a bin=$BIN label=ours ours=1 ;;
  esac
  case $dep in '' | *[!0-9]*) arm_usage "$a" ;; esac
  case $kind:$eng:$dep in
    ref:mrs:0 | ref:mrspa0:0) echo "depth-qwen3moe.sh: arm '$a': mistral.rs refuses --depth 0 with a decode length" >&2; exit 64 ;;
  esac
  if [ "$kind" = ref ] && pp_eng "$eng"; then
    ub=$(pp_ub "$eng")
    case $ub in *[!0-9]*) arm_usage "$a" ;; esac
    [ "$dep" -ge 1 ] || { echo "depth-qwen3moe.sh: arm '$a': a prompt of 0 ids has no prefill to time" >&2; exit 64; }
    if [ -n "$ub" ]; then
      case $eng in ikpp*) flags=$IK_GPU_FLAGS ;; *) flags=$LCPP_GPU_FLAGS ;; esac
      case " $flags " in
        *" -ub "* | *" --ubatch-size "* | *" -b "* | *" --batch-size "*)
          echo "depth-qwen3moe.sh: arm '$a': the profile's flags already set the batch sizes ($flags); the ubatch lever would add a second value" >&2
          exit 64
          ;;
      esac
    fi
  fi
  # Our prompt is one argument of D ids of up to six characters each; the kernel caps one
  # argument at 128 KiB.
  if [ "$kind" != ref ] && { [ "$dep" -lt 1 ] || [ "$dep" -gt 20000 ]; }; then
    echo "depth-qwen3moe.sh: our arm '$a' needs 1 <= D <= 20000 (D fed ids in one --tokens argument)" >&2
    exit 64
  fi
  if [ -n "$ASLOTS" ] && [ $((dep * ASLOTS)) -gt 20000 ]; then
    echo "depth-qwen3moe.sh: our arm '$a' feeds $ASLOTS slots × $dep = $((dep * ASLOTS)) ids, past the 20000 one --tokens argument holds" >&2
    exit 64
  fi
  A_KIND+=("$kind") A_DEP+=("$dep") A_LABEL+=("$label") A_ENG+=("$eng") A_BIN+=("$bin") A_IDS+=('') A_ENV+=("$envs") A_TOK+=("$tok") A_SLOTS+=("$ASLOTS")
  # An ours or bin: arm's placement: its place=, else BLOOMERY_GEN_PLACE (empty: no --place, the binary's a).
  if [ "$kind" = ref ]; then A_PLACE+=('') A_PLACE_SET+=(''); else A_PLACE+=("${APLACE:-$PLACE}") A_PLACE_SET+=("${APLACE:+1}"); fi
done
# The load keys (tools/ref/load-groups.sh): an ours arm's binary and its --ctx, the cache height and the
# flash grid the load fixes, so ours arms share a load when they share C (BLOOMERY_GEN_CTX, or one D).
# shellcheck source=tools/ref/load-groups.sh
source "${BASH_SOURCE[0]%/*}/load-groups.sh" || exit 2
# An arm of an --arm list prints its prompt ids after the go and before its prefill timer starts: the
# start of its measured window (the header's Cold tag).
LG_FED_RE='^prompt_ids '
arm_ctx() { echo "${GEN_CTX:-$(((A_DEP[$1] + N + 255) / 256 * 256))}"; }
# arm_prompt <i>: the prompt ids arm <i> feeds: a prose arm's corpus ids (A_TOK), else the LCG walk of
# its feed (arm_feed: its depth, N·D for an aggregate arm).
arm_prompt() {
  if [ -n "${A_TOK[$1]}" ]; then printf '%s' "${A_TOK[$1]}"; else lcg_prompt "$(arm_feed "$1")"; fi
}
# arm_env_list <i>: the variables arm <i> runs with, comma-separated, the solo marker left out. A
# list of the marker alone answers the empty string without lg_strip_solo: bash 3.2 (the Mac) reads
# the ${out[*]} of its emptied array as unbound under set -u.
arm_env_list() {
  [ -z "${A_ENV[$1]}" ] && return 0
  [ "${A_ENV[$1]}" = "$LG_SOLO" ] && return 0
  lg_strip_solo "${A_ENV[$1]}"
}
# A lever arm's variables are load-time, so they go into its key, sorted; `|solo` on an arm whose list
# holds BLOOMERY_AB_LOAD=arm.
for i in "${!ARMS[@]}"; do
  LG_KEY[i]=
  [ "${A_KIND[$i]}" = ours ] || continue
  LG_KEY[i]="$BIN|ctx=$(arm_ctx "$i")${A_PLACE[$i]:+|place=${A_PLACE[$i]}}"
  [ -n "${A_ENV[$i]}" ] || continue
  env_key=$(lg_env_key "$(arm_env_list "$i")")
  [ -z "$env_key" ] || LG_KEY[i]+="|$env_key"
  if lg_is_solo "${A_ENV[$i]}"; then LG_KEY[i]+='|solo'; fi
done
# arm_envs <i>: the variables arm <i> of ours runs with, into ARM_ENVS (NAME=VALUE words); a unit's arms
# share them (their load key holds them).
arm_envs() {
  local envs
  ARM_ENVS=()
  envs=$(arm_env_list "$1")
  [ -z "$envs" ] || IFS=, read -r -a ARM_ENVS <<< "$envs"
}
# lg_cmd <indices...>: the driver's hook (tools/ref/load-groups.sh): one load's command line into LG_CMD,
# and into LG_ENV the variables its process runs under, its arms' (one list: their load key holds it).
lg_cmd() {
  local i
  arm_envs "$1"
  LG_ENV=(${ARM_ENVS[@]+"${ARM_ENVS[@]}"})
  LG_CMD=("$BIN")
  for i in "$@"; do LG_CMD+=(--arm "$(arm_prompt "$i")"); done
  # shellcheck disable=SC2206 # an empty WARM adds nothing
  LG_CMD+=(-n "$N" --ctx "$(arm_ctx "$1")" ${A_PLACE[$1]:+--place "${A_PLACE[$1]}"} --time ${WARM:+--warm "$WARM"} --arm-sync)
}
# srv_ctx_of <i> <n_predict>: a server arm's -c, ours' context at its D or P (lcpp-warm.sh's Context).
srv_ctx_of() { echo "${GEN_CTX:-$(((A_DEP[$1] + N + 255) / 256 * 256))}"; }
# shellcheck disable=SC2034 # read by lcpp-warm.sh's srv_cmd_of
SRV_CTX_SRC="ours' --ctx at that D or P: ${GEN_CTX:+BLOOMERY_GEN_CTX }${GEN_CTX:-D + N rounded up to 256}"
# --parse-arms: the arms as parsed and each round's order, then exit before the card, the binaries and
# the lease.
if [ -n "$PARSE_ONLY" ]; then
  # Each load of round 1 as the driver starts it (lg_cmd), its prompts by name: lease.sh is not sourced here.
  lcg_prompt() { echo "<lcg_prompt $1>"; }
  for i in "${!ARMS[@]}"; do
    extra=
    if [ "${A_ENG[$i]}" = prose ] || { [ "${A_KIND[$i]}" = bin ] && [ -n "${A_TOK[$i]}" ]; }; then
      f=$(corpus_file)
      if [ -r "$f" ]; then three=$(head -n 3 "$f" | paste -sd, -); else three=-; fi
      extra=" corpus=$f ids=$three"
    fi
    [ -z "${A_SLOTS[$i]}" ] || extra+=" slots=${A_SLOTS[$i]} feed=$(arm_feed "$i")"
    echo "[parse] ${ARMS[$i]}: kind=${A_KIND[$i]} depth=${A_DEP[$i]} label=${A_LABEL[$i]} env=$(e=$(arm_env_list "$i"); echo "${e:--}") load=${LG_KEY[$i]:-(a process of its own)}$extra"
  done
  echo "[parse] order: $ORDER"
  if [ "$ORDER" = rotate ]; then
    lg_units "${!ARMS[@]}"
    for r in $(seq "$ROUNDS"); do
      o='' u=''
      while read -r -a idx; do
        names=''
        for i in "${idx[@]}"; do names+="${names:+ }${ARMS[$i]}"; done
        o+="${o:+ }$names"
        if lg_grouped "${idx[0]}"; then u+="${u:+ }[$names]"; else u+="${u:+ }$names"; fi
      done < <(lg_round "$r")
      echo "[parse] round $r order: $o"
      echo "[parse] round $r loads: $u"
    done
    while read -r -a idx; do
      lg_grouped "${idx[0]}" || continue
      lg_cmd "${idx[@]}"
      if [ ${#LG_ENV[@]} -eq 0 ]; then e=''; else e="env ${LG_ENV[*]} "; fi
      echo "[parse] load: $e${LG_CMD[*]}"
    done < <(lg_round 1)
  else
    echo "[parse] the ours block holds every ours arm, the lever arms with them, each round rotated as under rotate (a dry run prints the blocks)"
  fi
  exit 0
fi
# The card pin, the card's witness lines, the other-card guard and the binary's freshness; this runner
# has the two-card mode (the header's Two cards).
TIMING_CARDS_RUNNER=1
# shellcheck source=tools/ref/timing-card.sh
source "${BASH_SOURCE[0]%/*}/timing-card.sh"
timing_cards_mode || exit $?
# The reference and server arms' two-card lines (timing-card.sh); our arms are held to their placements
# below.
TC_ARMS=()
for i in "${!ARMS[@]}"; do
  case ${A_KIND[$i]} in ref | srv) TC_ARMS+=("${ARMS[$i]}" "${A_KIND[$i]}" "${A_ENG[$i]}") ;; esac
done
timing_cards_arms "$BIN" ${TC_ARMS[@]+"${TC_ARMS[@]}"} || exit $?
# Each ours and bin: arm's placement against the mode and the timing card (tools/ref/arm-place.sh): Qwen3.8's
# placement finds its cards by device (workstation::ALIASES: a the largest visible card, gate the one card
# named 3090, bp the two largest), and the timing card is TIMING_GPU's, timing-card.sh's pick
# (BLOOMERY_TIMING_GPU overrides), never a name the binary sees;
# BLOOMERY_GEN_PLACE once for the arms that follow it, an arm's place= for that arm. With no --place the
# binary runs a, which the two-card mode checks as a; under one card it is not checked, as before.
PLACE_SEEN=0 PLACES_SET='' PLACE_ARMS=()
for i in "${!ARMS[@]}"; do
  case ${A_KIND[$i]} in ours | bin) ;; *) continue ;; esac
  PLACE_ARMS+=("$i")
  if [ -n "${A_PLACE_SET[$i]}" ]; then
    PLACES_SET=1
    place_check "${ARMS[$i]}" "${A_PLACE[$i]}"
  elif [ "$PLACE_SEEN" = 0 ] && { [ -n "$PLACE" ] || [ -n "$TIMING_CARDS" ]; }; then
    place_check '' "${PLACE:-a}"
    PLACE_SEEN=1
  fi
done
# The lease and the witness fields.
# shellcheck source=tools/ref/lease.sh
source "${BASH_SOURCE[0]%/*}/lease.sh"
# The t quantiles of the ratio intervals, df 1..ROUNDS: tools/ref/tdist.py, the table gpu-ab.py and
# card.py read.
T975=$(python3 "${BASH_SOURCE[0]%/*}/tdist.py" "$ROUNDS") || {
  rc=$?
  echo "depth-qwen3moe.sh: tools/ref/tdist.py gave no t quantiles for ROUNDS=$ROUNDS (rc $rc; ROUNDS is a positive integer)" >&2
  exit "$rc"
}
# Each engine's binary matters only to its own arms: a reference-only run neither builds ours (the
# recipe skips the build) nor reads it, so its freshness is not asked. A dry run asks nothing of
# our binary: it prints the command line it would run.
if [ "$ours" = 1 ] && [ -z "$DRY" ]; then assert_fresh_binary "$BIN" || exit $?; fi
# The residency records (cold-blocks.sh's residency sums): Q_RES 0 when generate_qwen3moe's checked-in
# schema declares them, 1 when it declares none (an arm that sets BLOOMERY_RESIDENCY then fails by name).
Q_RES=0
res_kinds generate_qwen3moe || Q_RES=$?
[ "$Q_RES" != 2 ] || { echo "depth-qwen3moe.sh: $RS_WHY" >&2; exit 2; }
# res_asked <i>: the BLOOMERY_RESIDENCY arm <i> runs with (its own list's, else the runner's), empty when
# neither sets it.
res_asked() {
  local e v=${BLOOMERY_RESIDENCY:-}
  local -a kv=()
  [ -z "$(arm_env_list "$1")" ] || IFS=, read -r -a kv <<< "$(arm_env_list "$1")"
  for e in ${kv[@]+"${kv[@]}"}; do
    case $e in BLOOMERY_RESIDENCY=*) v=${e#*=} ;; esac
  done
  echo "$v"
}
# A bin:<path> arm's binary is checked where its tree line is taken, below.
if [ "$ik" = 1 ]; then [ -x "$IKBIN" ] || { echo "depth-qwen3moe.sh: no llama-bench at $IKBIN" >&2; exit 2; }; fi
if [ "$lcpp" = 1 ]; then [ -x "$LCPPBIN" ] || { echo "depth-qwen3moe.sh: no llama-bench at $LCPPBIN" >&2; exit 2; }; fi
if [ "$lcppfit" = 1 ]; then
  # shellcheck disable=SC2153 # LCPP is the profile's, as below
  lcpp_fit_probe "$LCPPBIN" || { echo "depth-qwen3moe.sh: the lcppfit/lcppppfit arms need llama-bench's fit: $FIT_WHY (tree $LCPP)" >&2; exit 64; }
fi
if [ "$mrs" = 1 ]; then [ -x "$MRSBIN" ] || { echo "depth-qwen3moe.sh: no mistralrs at $MRSBIN" >&2; exit 2; }; fi
SRVBIN=
[ "$srv" = 0 ] || srv_preflight depth-qwen3moe.sh
# The CPU guard's names: builds and the engines this runner did not start (its own arms run under its pid);
# a BLOOMERY_CPU_BUSY_COMMS the caller gives is the whole list.
[ -n "${BLOOMERY_CPU_BUSY_COMMS:-}" ] || CPU_BUSY_COMMS="$CPU_BUSY_COMMS generate_qwen3moe llama-server mistralrs"
if [ -n "$TIMING_CARDS" ]; then
  CARD_NAME=$TIMING_CARDS_NAME
else
  CARD_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$TIMING_GPU" | sed 's/^NVIDIA //; s/^GeForce //; s/^RTX //')
fi
# A binary's sha256 and its tree's HEAD and dirty count, once. GIT_OPTIONAL_LOCKS=0 keeps `git
# status` from rewriting the index of a tree this root process does not own.
tree_line() {
  local bin=$1 tree=$2 sha head dirty
  sha=$(sha256sum "$bin" | cut -c1-12)
  head=$(git -c safe.directory="$tree" -C "$tree" rev-parse --short=9 HEAD 2> /dev/null || echo '?')
  dirty=$(GIT_OPTIONAL_LOCKS=0 git -c safe.directory="$tree" -C "$tree" status --porcelain --untracked-files=no 2> /dev/null | wc -l | tr -d ' ')
  echo "$bin sha256=$sha head=$head dirty_files=$dirty"
}
IK_LINE='' LCPP_LINE='' MRS_LINE='' MRS_VERSION='' BIN_LINES=() SRV_LINE=''
[ "$srv" = 0 ] || SRV_LINE=$(srv_tree "$(tree_line "$SRVBIN" "$LCPP")")
[ "$ik" = 0 ] || IK_LINE=$(tree_line "$IKBIN" "$IK")
# shellcheck disable=SC2153 # LCPP is the profile's, which shellcheck does not follow
[ "$lcpp" = 0 ] || LCPP_LINE=$(tree_line "$LCPPBIN" "$LCPP")
if [ "$mrs" = 1 ]; then
  # shellcheck disable=SC2153 # MRS is the profile's, as LCPP
  MRS_LINE=$(tree_line "$MRSBIN" "$MRS")
  MRS_VERSION=$("$MRSBIN" --version 2>&1 | head -n 1)
  MRS_LINE="$MRS_LINE version=${MRS_VERSION:-?}"
fi
for i in "${!ARMS[@]}"; do
  [ "${A_KIND[$i]}" = bin ] || continue
  b=${A_BIN[$i]}
  [ -x "$b" ] || { echo "depth-qwen3moe.sh: arm '${ARMS[$i]}': no binary at $b" >&2; exit 2; }
  t=${b%/target/*}
  [ "$t" != "$b" ] || t=${b%/*}
  BIN_LINES+=("${A_LABEL[$i]}: $(tree_line "$b" "$t")")
done
ref_witness() {
  [ -z "$IK_LINE" ] || echo "    ik: $IK_LINE"
  [ -z "$LCPP_LINE" ] || echo "    lcpp: $LCPP_LINE"
  [ -z "$MRS_LINE" ] || echo "    mrs: $MRS_LINE"
  [ -z "$SRV_LINE" ] || echo "    lcppsrv: $SRV_LINE"
  [ ${#BIN_LINES[@]} -eq 0 ] || printf '    %s\n' "${BIN_LINES[@]}"
}

# The witness before and after every row: the timing card's lines (with our binary), the busiest
# processes, the page cache and the major fault count.
WITNESS=(head-open indent card busiest model mem pgmajfault)

# A compute process on the timing card as an arm starts is another round's functional run
# (BLOOMERY_CARD=a6000 is refused while the card is busy, not while this lease is held, so one can
# start in the gap between two arms). Every arm here loads the whole model onto that card, so it
# would be timed beside that process or fail its allocation. Such a run lasts minutes: wait for it,
# polling every 10 s, and refuse after 10 minutes, rc 75 — contention, not a result.
guard_timing() {
  local apps i
  for ((i = 0; i < 60; i++)); do
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader -i "$TIMING_GPU")
    if [ -z "$apps" ]; then
      [ "$i" = 0 ] || echo "[timing-busy] $(now) the timing card is free after $((i * 10)) s" >&2
      return 0
    fi
    if [ "$i" = 0 ]; then
      echo "[timing-busy] $(now) compute apps on the timing card ($TIMING_GPU): [$(echo "$apps" | tr '\n' ';')]; waiting up to 10 min" >&2
      witness wait-timing >&2
    fi
    sleep 10
  done
  echo "[timing-busy] $(now) still busy after 10 min: [$(echo "$apps" | tr '\n' ';')]" >&2
  witness abort-timing >&2
  exit 75
}

# ref_cmd <engine> <depth or prompt length>: the reference arm's binary, arguments and row label,
# into REF_BIN, REF_ARGS and REF_LABEL, and for a prefill arm the batch sizes its row names into
# REF_BATCH. The flags are word-split on purpose: the profile keeps them as one string. A prefill
# arm is its decode twin's binary and flags with the prompt test in place of the decode test. REF_K,
# when set, is the --n-cpu-moe a llama-bench arm runs at instead (a block's discard, the header's
# Order). A fit arm's flags are lcpp_fit_flags' rewrite, taken after REF_K (which would add --n-cpu-moe
# back), and REF_FIT says so for the dry run. An lcpp arm also gets --progress, and REF_MARK is the ERE
# of the line its measured window starts at (the header's Cold tag): the progress line of its timed
# repetition, mistral.rs's `Iteration 1/1...`; empty for ik, whose llama-bench has no --progress.
REF_K=
ref_cmd() {
  local eng=$1 dep=$2 flags ctx ub
  REF_BATCH='' REF_FIT='' REF_MARK=''
  case $eng in
    ik | ikdef)
      REF_BIN=$IKBIN flags=$IK_GPU_FLAGS
      [ "$eng" = ik ] || flags=$IK_GPU_DEFAULT_FLAGS
      [ -z "$REF_K" ] || flags=$(with_ncmoe "$flags" "$REF_K")
      if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n 0 -gp "$dep,$N"); REF_LABEL="tg$N@pp$dep |"; fi
      # shellcheck disable=SC2206
      REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
      ;;
    lcpp | lcppfit)
      REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS
      [ -z "$REF_K" ] || flags=$(with_ncmoe "$flags" "$REF_K")
      if [ "$dep" = 0 ]; then REF_ARGS=(-p 0 -n "$N"); REF_LABEL="tg$N |"; else REF_ARGS=(-p 0 -n "$N" -d "$dep"); REF_LABEL="tg$N @ d$dep |"; fi
      if [ "$eng" = lcppfit ]; then
        lcpp_fit_flags "$flags"
        flags=$FIT_FLAGS
        REF_FIT="placement: llama-bench's fit at -fitt $LCPP_FIT_TARGET MiB (dropped: $FIT_DROPPED), -v for the fit column"
      fi
      REF_MARK=': generation run 1/1$' flags="$flags --progress"
      # shellcheck disable=SC2206
      REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
      ;;
    ikpp* | lcpppp*)
      case $eng in
        ikpp*) REF_BIN=$IKBIN flags=$IK_GPU_FLAGS ;;
        *) REF_BIN=$LCPPBIN flags=$LCPP_GPU_FLAGS ;;
      esac
      [ -z "$REF_K" ] || flags=$(with_ncmoe "$flags" "$REF_K")
      if lcpp_fit_eng "$eng"; then
        lcpp_fit_flags "$flags"
        flags=$FIT_FLAGS
        REF_FIT="placement: llama-bench's fit at -fitt $LCPP_FIT_TARGET MiB (dropped: $FIT_DROPPED), -v for the fit column"
      fi
      case $eng in lcpppp*) REF_MARK=': prompt run 1/1$' flags="$flags --progress" ;; esac
      ub=$(pp_ub "$eng")
      REF_ARGS=(-p "$dep" -n 0) REF_LABEL="pp$dep |" REF_BATCH="ub 512 b 2048 (llama-bench defaults)"
      if [ -n "$ub" ]; then
        REF_ARGS+=(-ub "$ub" -b "$((ub > 2048 ? ub : 2048))")
        REF_BATCH="ub $ub b $((ub > 2048 ? ub : 2048)) (the arm's lever)"
      fi
      # shellcheck disable=SC2206
      REF_ARGS=(-m "$MODEL" "${REF_ARGS[@]}" -r 1 $flags)
      ;;
    mrspp)
      REF_BIN=$MRSBIN flags=$MRS_FLAGS REF_MARK='Iteration 1/1[.][.][.]'
      ctx=${GEN_CTX:-$(((dep + 1 + 255) / 256 * 256))}
      # shellcheck disable=SC2206
      REF_ARGS=(bench $flags -f "$MODEL" --prompt-len "$dep" --gen-len 1 --iterations 1 --warmup 1 --max-seq-len "$ctx" --pa-context-len "$((ctx + 32))")
      REF_LABEL="TTFT ($dep input tokens)"
      REF_BATCH="scheduler defaults: max_num_batched_tokens 4096, max_prefill_chunk_tokens 512 (bench sets neither)"
      ;;
    mrs | mrspa0)
      REF_BIN=$MRSBIN flags=$MRS_FLAGS REF_MARK='Iteration 1/1[.][.][.]'
      ctx=${GEN_CTX:-$(((dep + N + 255) / 256 * 256))}
      # shellcheck disable=SC2206
      REF_ARGS=(bench $flags -f "$MODEL" --prompt-len 0 --gen-len "$N" --depth "$dep" --iterations 1 --warmup 1 --max-seq-len "$ctx")
      # The pool holds (blocks - 1) whole 32-token blocks of the context asked for (the scheduler keeps
      # one back: paged_attention/mod.rs, available_context_tokens), so one block more than C.
      if [ "$eng" = mrs ]; then REF_ARGS+=(--pa-context-len "$((ctx + 32))"); else REF_ARGS+=(--paged-attn off); fi
      REF_LABEL="Decode ($N tokens @ d$dep)"
      ;;
  esac
}

# ref_val <engine> <label>: the arm's tok/s from its output on stdin. llama-bench prints a markdown
# table (`| … | t/s ± sd |`); mistralrs bench a box-drawn one (`│ Decode (N tokens @ dD) ┆ T/s ± sd
# ┆ … ms TPOT │`) whose first `<number> ± ` is the T/s cell. Its log lines carry ANSI colour codes.
ref_val() {
  case $1 in
    mrs | mrspa0 | mrspp) sed 's/\x1b\[[0-9;]*m//g' | grep -F "$2" | grep -oE '[0-9][0-9.]* ± ' | head -n 1 | sed 's/ ± //' ;;
    *) grep -F "$2" | awk -F'|' '{print $(NF-1)}' | sed 's/ ±.*//;s/ //g' ;;
  esac
}

# One reference arm: its binary on its flags, the row, and the sum; a FAIL row when it exits non-zero,
# prints no value, or (a fit arm) its fit failed or never ran. The output passes through majflt_mark on
# its way into `raw`, so the fault count at the measured window's start is known.
# ref_arm <engine> <depth> <round>
ref_arm() {
  local eng=$1 dep=$2 r=$3 raw rc val build dev t0 t1 key f0 f1 markf mark whole win cnt
  FIT_COL=''
  ref_cmd "$eng" "$dep"
  if pp_eng "$eng"; then key=p=$dep; else key=d=$dep; fi
  markf=$(mktemp "${TMPDIR:-/tmp}/depth-qwen3moe-mark.XXXXXX") || exit 2
  witness "pre r$r $eng d=$dep"
  ref_witness
  t0=$(date +%s)
  f0=$(majflt_now)
  raw=$(timeout --kill-after=10 "$BOUND" "$REF_BIN" "${REF_ARGS[@]}" 2>&1 | majflt_mark "$markf" "$REF_MARK"; exit "${PIPESTATUS[0]}")
  rc=$?
  f1=$(majflt_now)
  t1=$(date +%s)
  mark=$(cat "$markf")
  rm -f "$markf"
  witness "post r$r $eng d=$dep"
  guard_cpu "post r$r $eng d=$dep"
  # Two cards: an Xid, a card lost or off its cap, or an engine that saw one card fails the arm.
  if ! timing_cards_arm "$raw"; then
    arm_fail "$r" "$eng" "$key" "$rc" "two cards: $TWOCARD_WHY" "$raw"
    return 0
  fi
  val=$(echo "$raw" | ref_val "$eng" "$REF_LABEL")
  # A fit arm's loader lines say what the fit chose; a fit that failed or never ran is a FAIL row,
  # whatever llama-bench measured after it.
  if lcpp_fit_eng "$eng" && ! lcpp_fit_col "$raw"; then
    arm_fail "$r" "$eng" "$key" "$rc" "$FIT_WHY" "$raw"
    return 0
  fi
  if [ $rc -ne 0 ] || [ -z "$val" ]; then
    arm_fail "$r" "$eng" "$key" "$rc" "no '${REF_LABEL% |}' row" "$raw"
    return 0
  fi
  case $eng in
    mrs | mrspa0 | mrspp)
      # The load lines that say what ran (weight dtype, layer placement, the KV cache and its
      # length, the decode graphs, the tree), then the bench's own table.
      echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | grep -E 'DType selected|Layers [0-9]+-[0-9]+: |PagedAttention (KV cache type|with block)|CUDA decode graphs|git revision' \
        | sed 's/^[^ ]* *INFO [^ ]* //' | sed "s/^/    $eng load /"
      echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | grep -E '^│' | sed "s/^/    $eng table /"
      build=$MRS_VERSION
      dev=$(echo "$raw" | sed 's/\x1b\[[0-9;]*m//g' | sed -n 's/.*Layers \([0-9]*-[0-9]*: .*\)$/layers \1/p' | head -n 1)
      ;;
    *)
      # The reference's own table, header and row: its columns name every setting it ran with that
      # differs from its defaults, so the log shows which flags took.
      echo "$raw" | grep -E '^\| ' | grep -vE '^\| *-' | sed "s/^/    $eng table /"
      [ -z "$FIT_COL" ] || echo "$FIT_LINES" | sed "s/^/    $eng fit /"
      build=$(echo "$raw" | sed -n 's/^build: //p' | head -n 1)
      dev=$(echo "$raw" | sed -n 's/^ *Device 0: \([^,]*\),.*/\1/p' | head -n 1)
      [ -z "$TWOCARD_DEVS" ] || dev=$TWOCARD_DEVS
      ;;
  esac
  # The timed window of the row: P / tok/s for a pp row, N / tok/s for a llama-bench decode row,
  # (N - 1) / tok/s for a mistral.rs decode row (its clock spans N - 1 intervals).
  case $eng in
    ikpp* | lcpppp* | mrspp) win=$(awk -v p="$dep" -v v="$val" 'BEGIN { printf "%.4f", p / v }') ;;
    mrs | mrspa0) win=$(awk -v n="$N" -v v="$val" 'BEGIN { printf "%.4f", (n - 1) / v }') ;;
    *) win=$(awk -v n="$N" -v v="$val" 'BEGIN { printf "%.4f", n / v }') ;;
  esac
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
  if pp_eng "$eng"; then
    echo "$ROW_TAG r$r $eng p=$dep n=0 | tok/s(pp) $val @ n=0, prompt $dep, $CARD_NAME$FIT_COL | $REF_BATCH | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$CPU_BUSY_TAG$OTHER_BUSY_TAG$MAJ_COL$COLD_TAG"
    counted || return 0
    pp_sums+=("$eng|$dep|$r|$val|$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG")
  else
    echo "$ROW_TAG r$r $eng d=$dep n=$N | tok/s $val @ n=$N, depth $dep, $CARD_NAME$FIT_COL | build ${build:-?} | device ${dev:-?} | wall $((t1 - t0))s$CPU_BUSY_TAG$OTHER_BUSY_TAG$MAJ_COL$COLD_TAG"
    counted || return 0
    sums+=("$eng|$dep|$r|$val||$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG")
  fi
  count_row
}

# The closing summary's row counts: every ROW line, and those that carried [cpu-busy], [other-busy] and
# [cold].
count_row() {
  n_rows=$((n_rows + 1))
  [ -z "$CPU_BUSY_TAG" ] || busy_rows=$((busy_rows + 1))
  [ -z "$OTHER_BUSY_TAG" ] || other_rows=$((other_rows + 1))
  [ -z "$COLD_TAG" ] || cold_rows=$((cold_rows + 1))
}

# pp_rows: every `time prompt n=<P> ms=<ms> tok/s=<v> passes=<K> kind=<k>` row of a generate_qwen3moe
# run on stdin, as `<P> <v> <K> <k> <ms>`; fields after `kind=` are allowed and dropped (an aggregate
# arm's ` slot=<j>`: it prints one a slot); nothing when the run printed none (a base tree's build).
pp_rows() {
  sed -nE 's/^time prompt n=([0-9]+) ms=([0-9.]+) tok\/s=([0-9.]+|inf) passes=([0-9]+) kind=([a-z]+)( .*)?$/\1 \3 \4 \5 \2/p'
}
# parse_pp: the first of them, the one-stream prompt's (an aggregate arm's slot 0).
parse_pp() { pp_rows | head -n 1; }

# pp_col <arm kind> <output>: an ours or bin arm's prefill column from its run's output, into PP_COL,
# with the parsed P, tok/s and ms in PP_N, PP_TPS and PP_MS for the closing tables and the timed
# window. A bin arm's base build may print no time prompt row (PP_N empty, the column says so); an ours
# arm's binary is this tree's, so a missing row fails the arm (1, FAIL_WHY).
pp_col() {
  local passes kind
  read -r PP_N PP_TPS passes kind PP_MS <<< "$(echo "$2" | parse_pp)"
  if [ -n "$PP_N" ]; then
    PP_COL=" | pp_tok/s $PP_TPS (n=$PP_N, passes=$passes)"
    [ "$kind" = prefill ] || PP_COL="${PP_COL%)}, kind=$kind)"
  elif [ "$1" = bin ]; then
    PP_COL=" | pp_tok/s ? (the binary prints no time prompt row)"
  else
    FAIL_WHY="no time prompt row"
    return 1
  fi
}

# One arm of a generate_qwen3moe in a process of its own with the one-arm command line: a second
# binary (bin:), which may know no --arm. The row and the sum under the arm's label. Its prompt ids
# print before its load, so its fault count is the whole process's (MAJ_TIMED empty).
# ours_arm <index> <round>
ours_arm() {
  local i=$1 r=$2 out rc t0 t1 f0 f1
  ours_pre "$i" "$r"
  t0=$(date +%s)
  f0=$(majflt_now)
  out=$(timeout --kill-after=10 "$BOUND" "${A_BIN[$i]}" --tokens "$(arm_prompt "$i")" -n "$N" --ctx "$(arm_ctx "$i")" ${A_PLACE[$i]:+--place "${A_PLACE[$i]}"} --time ${WARM:+--warm "$WARM"} 2>&1)
  rc=$?
  f1=$(majflt_now)
  t1=$(date +%s)
  MAJ_WHOLE=$((f1 - f0)) MAJ_TIMED=
  ours_post "$i" "$r" "$rc" "$out" "$((t1 - t0))"
}
# ours_pre <index> <round>: the witness block before an ours or bin arm.
ours_pre() { witness "pre r$2 ${A_LABEL[$1]} d=${A_DEP[$1]} n=$N ctx=$(arm_ctx "$1")"; }
# ours_post <index> <round> <rc> <output> <wall s>: the witness block after an ours or bin arm, then its
# row, or its FAIL row when it exited non-zero or printed no row (no SMOKE line, or one without its p50
# and mean, no time prompt row). An output that opens with an `arm` line (an --arm list's) gives the
# row its slot in the load. MAJ_WHOLE and MAJ_TIMED are the arm's (the driver's, or ours_arm's).
ours_post() {
  local i=$1 r=$2 rc=$3 out=$4 wall=$5 dep label ctx smoke p50 mean warmcol nodes series h10 t10 uniq_tok tps_mean tps_p50 a slot='' win timed ran mtp=''
  dep=${A_DEP[$i]} label=${A_LABEL[$i]} ctx=$(arm_ctx "$i")
  witness "post r$r $label d=$dep n=$N ctx=$ctx"
  guard_cpu "post r$r $label d=$dep"
  a=$(sed -nE '1s/^arm i=([0-9]+) arms=([0-9]+) .*/\1 \2/p' <<< "$out")
  [ -z "$a" ] || slot=" | slot $((${a% *} + 1))/${a#* }"
  if [ "$rc" -ne 0 ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "exited $rc" "$out"
    return 0
  fi
  smoke=$(echo "$out" | grep -E '^SMOKE ')
  [ -n "$smoke" ] || { arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "no SMOKE line" "$out"; return 0; }
  p50=$(echo "$smoke" | sed -n 's/.*p50_ms=\([0-9.]*[0-9]\).*/\1/p' | head -n 1)
  mean=$(echo "$smoke" | sed -n 's/.*mean_ms=\([0-9.]*[0-9]\).*/\1/p' | head -n 1)
  if [ -z "$p50" ] || [ -z "$mean" ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "its SMOKE line has no p50_ms or mean_ms" "$out"
    return 0
  fi
  # An MTP run's SMOKE line carries passes=: its steps are the positions the counted passes kept and its
  # mean_ms their wall over those positions, so mean_ms × steps / passes is the wall of one verify pass.
  # A plain run's has no passes= and gets no column.
  local passes='' steps='' passcol='' pps=''
  if [[ " $(head -n 1 <<< "$smoke")" == *" passes="* ]]; then
    passes=$(head -n 1 <<< "$smoke" | sed -n 's/.* passes=\([0-9]*\).*/\1/p')
    steps=$(head -n 1 <<< "$smoke" | sed -n 's/.* steps=\([0-9]*\).*/\1/p')
    if ! [[ $passes =~ ^[0-9]+$ && $steps =~ ^[0-9]+$ ]] || [ "$((10#$passes))" = 0 ] || [ "$((10#$steps))" = 0 ]; then
      arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "its SMOKE line carries passes= and no pass time can be read from it (passes='$passes' steps='$steps'; ms/pass is mean_ms × steps / passes, each a positive count)" "$out"
      return 0
    fi
    read -r passcol pps < <(awk -v m="$mean" -v s="$steps" -v q="$passes" 'BEGIN { printf "%.4f %.6f\n", m * s / q, 1e3 * q / (m * s) }')
    passcol=" | ms/pass $passcol (mean_ms × steps $steps / passes $passes)"
  fi
  pp_col "${A_KIND[$i]}" "$out" || { arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "$FAIL_WHY" "$out"; return 0; }
  # The placement it was given: its own `load` line, or its load's (LG_HEADER).
  if [ -n "${A_PLACE[$i]}" ]; then
    ran=$(printf '%s\n%s\n' "$out" "${LG_HEADER:-}" | sed -n 's/^load .* place=\([a-z]*\).*/\1/p' | head -n 1)
    [ "$ran" = "${A_PLACE[$i]}" ] || { arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "its load line names place=${ran:-none}; the runner passed --place ${A_PLACE[$i]}" "$out"; return 0; }
  fi
  # Two cards: an Xid, a card lost or off its cap, or a load whose cards are not its placement's (bp both
  # cards, a the A6000 alone; no --place is a) fails the arm. The cards are the load line's `cards=` field
  # (tools/ref/arm-place.sh place_arm_cards), none when it has no such field.
  if [ -n "$TIMING_CARDS" ]; then
    local lcards
    lcards=$(printf '%s\n%s\n' "$out" "${LG_HEADER:-}" | sed -n 's/^load .* cards=\(\[[^]]*\]\).*/\1/p' | head -n 1)
    place_arm_cards "$out" "${A_PLACE[$i]:-a}" "$lcards" || { arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "two cards: $TWOCARD_WHY" "$out"; return 0; }
  fi
  # Under BLOOMERY_DRAFT=mtp the `mtp summary` record: E(4), the positions a four-row window kept on
  # average, beside the per-position rate the SMOKE mean already is. Under BLOOMERY_STEP_STATS=1 the
  # `stat summary` record's host slots a token. The `time pass` records, an aggregate arm's rounds. All
  # through records.py, by kind and field.
  local MK='' MP='' MQ='' HS='' SP_MS='' SP_POS='' SP_KIND='' SP_WARM='' rec lines n_mtp
  if ! rec=$(python3 "$RS_RECORDS" sh --bin generate_qwen3moe - 'MK=mtp_summary.kept' 'MP=mtp_summary.positions' \
    'MQ=mtp_summary.passes' 'HS=stat_summary_host.host_slots_mean' 'SP_MS=time_pass.ms*' \
    'SP_POS=time_pass.positions*' 'SP_KIND=time_pass.kind*' 'SP_WARM=time_pass.warm*' <<< "$out" 2>&1) ||
    ! lines=$(python3 "$RS_RECORDS" lines --bin generate_qwen3moe - mtp_summary <<< "$out" 2>&1); then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "records.py did not read its mtp, stat and time pass records: $rec${lines:+ $lines}" "$out"
    return 0
  fi
  n_mtp=$(grep -c . <<< "$lines")
  eval "$rec"
  # An aggregate arm (the header's <D>@BLOOMERY_GEN_SLOTS=N) reads its rounds from the `time pass …
  # kind=slots` records: the counted ones (`warm` left out), each of N positions, and every one for the
  # timed window; an arm that names no N prints none. sl_*: the counted rounds, their positions and ms,
  # those whose positions is not N, then every kind=slots record and its ms.
  local nslots=${A_SLOTS[$i]} sl_n sl_pos sl_ms sl_bad sl_all sl_all_ms
  read -r sl_n sl_pos sl_ms sl_bad sl_all sl_all_ms < <(paste -d' ' <(printf '%s\n' "$SP_MS") <(printf '%s\n' "$SP_POS") \
    <(printf '%s\n' "$SP_KIND") <(printf '%s\n' "$SP_WARM") | awk -v want="${nslots:-0}" '$3 == "slots" {
      all++; all_ms += $1
      if ($4 == "1") next
      n++; p += $2; ms += $1; if ($2 != want) bad++
    } END { printf "%d %d %.4f %d %d %.4f\n", n, p, ms, bad, all, all_ms }')
  if [ -z "$nslots" ] && [ "$sl_all" -gt 0 ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "its output holds $sl_all time pass record(s) of kind=slots and the arm names no BLOOMERY_GEN_SLOTS: its row would read one stream's rate off several" "$out"
    return 0
  fi
  if [ -n "$nslots" ] && [ "$sl_n" = 0 ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "the arm runs BLOOMERY_GEN_SLOTS=$nslots and printed no counted time pass record of kind=slots: no aggregate to read" "$out"
    return 0
  fi
  if [ -n "$nslots" ] && [ "$sl_bad" -gt 0 ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "$sl_bad of its $sl_n counted kind=slots records hold positions other than its $nslots slots" "$out"
    return 0
  fi
  mtp=''
  if [ "$n_mtp" -gt 0 ]; then
    [ -n "$MK" ] && [ -n "$MP" ] && [ -n "$MQ" ] || { arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "its mtp summary record has no kept, positions or passes (kept='$MK' positions='$MP' passes='$MQ')" "$out"; return 0; }
    mtp=" | mtp E(4) $(awk -v p="$MP" -v q="$MQ" 'BEGIN { printf "%.3f", (q > 0) ? p / q : 0 }') = positions $MP / passes $MQ, kept $MK"
  fi
  mtp+=$passcol
  [ -z "$HS" ] || mtp+=" | host slots/token $HS"
  # The residency records: the lever and the timed passes' sums (cold-blocks.sh's residency sums).
  RS_WORD='' RS_COL=''
  if [ "$Q_RES" = 0 ]; then
    res_sums generate_qwen3moe "$out" "${LG_HEADER:-}" || { arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "$RS_WHY" "$out"; return 0; }
    if [ -z "$RS_WORD" ] && [ -n "$(res_asked "$i")" ]; then
      arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "the arm runs BLOOMERY_RESIDENCY=$(res_asked "$i") and printed no residency lever record, which its schema declares" "$out"
      return 0
    fi
  elif [ -n "$(res_asked "$i")" ]; then
    arm_fail "$(fail_round "$r")" "$label" "d=$dep" "$rc" "the arm runs BLOOMERY_RESIDENCY=$(res_asked "$i"), and generate_qwen3moe's checked-in schema (tools/bloomery/schema/generate_qwen3moe.jsonl) declares no residency record, so its row cannot carry them: refresh the schema from the binary that prints them (just records-refresh)" "$out"
    return 0
  fi
  mtp+=$RS_COL
  # The prompt_ids line is the whole prompt; the load, capture, step-0, time prompt and stat prompt
  # lines are the arm's configuration, the prefill's time and its host prologue; under
  # BLOOMERY_STEP_STATS the ubatch walk's split and per-layer-batch records follow, then each generated
  # step's host counters and their summary (the host slots a step, from which the residency hit reads).
  # An MTP arm's window walls and kept rows, a residency arm's boundaries and the greedy ids stay in the
  # log beside the row that sums them.
  echo "$out" | grep -E '^(load|capture|step 0|time prompt|stat prompt|stat step|stat summary|residency pass|mtp summary|time pass|tokens) '
  warmcol=$(echo "$smoke" | sed -n 's/.* warm=\([0-9]*\).*/\1/p')
  nodes=$(echo "$out" | sed -n 's/^capture graph_nodes=\([0-9]*\).*/\1/p')
  [ -n "$nodes" ] || nodes=$(sed -n 's/^capture graph_nodes=\([0-9]*\).*/\1/p' <<< "${LG_HEADER:-}")
  series=$(echo "$out" | awk '/^time step /{sub(/.*ms=/,""); print}')
  # An aggregate arm's series are its rounds, its nodes the captured pass's.
  if [ -n "$nslots" ]; then
    series=$(paste -d' ' <(printf '%s\n' "$SP_MS") <(printf '%s\n' "$SP_KIND") | awk '$2 == "slots" { print $1 }')
    nodes=$(echo "$out" | sed -n 's/^capture slots=[0-9]* rows=1 graph_nodes=\([0-9]*\).*/\1/p')
  fi
  h10=$(echo "$series" | head -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  t10=$(echo "$series" | tail -n 10 | sort -n | awk '{a[NR]=$1} END{if(NR)print a[int((NR+1)/2)]}')
  uniq_tok=$(echo "$out" | awk '/^step / && $2 != 0 {print $4}' | sort -u | wc -l | tr -d ' ')
  tps_mean=$(awk -v m="$mean" 'BEGIN{printf "%.2f", 1e3/m}')
  tps_p50=$(awk -v p="$p50" 'BEGIN{printf "%.2f", 1e3/p}')
  # The timed window: the prompt's wall and the N generated steps at the mean; an aggregate arm's, every
  # slot's prompt and every round.
  win=$(awk -v p="${PP_MS:-0}" -v n="$N" -v m="$mean" 'BEGIN { printf "%.4f", (p + n * m) / 1e3 }')
  [ -z "$nslots" ] || win=$(echo "$out" | pp_rows | awk -v r="$sl_all_ms" '{ p += $5 } END { printf "%.4f", (p + r) / 1e3 }')
  cold_check "${MAJ_TIMED:-$MAJ_WHOLE}" "$win"
  if [ -n "$MAJ_TIMED" ]; then
    timed=$MAJ_TIMED
  elif [ -n "$a" ]; then
    timed='? (no prompt_ids line: the whole arm)'
  else
    timed='? (a one-arm run prints its prompt ids before its load: the whole process)'
  fi
  cold_verdict "$r" "$label" "d=$dep" "${MAJ_TIMED:-$MAJ_WHOLE}" "$win" || return 0
  # The row names its placement: the arm's --place, or a in the two-card mode with none (the binary's).
  local rowplace=${A_PLACE[$i]}
  [ -n "$rowplace" ] || [ -z "$TIMING_CARDS" ] || rowplace=a
  if [ -n "$nslots" ]; then
    # The aggregate: Σ positions · 1000 / Σ ms over the counted rounds; the SMOKE footer's p50 and mean are
    # a round's, so 1000 / mean is one stream's rate.
    local agg
    agg=$(awk -v p="$sl_pos" -v ms="$sl_ms" 'BEGIN { printf "%.2f", p * 1e3 / ms }')
    echo "$ROW_TAG r$r $label d=$dep n=$N ctx=$ctx | tok/s(aggregate) $agg @ n=$nslots·$sl_n, depth $dep, $CARD_NAME${rowplace:+ | place $rowplace} | slots $nslots | tok/s(per stream, mean) $tps_mean | p50 $p50 ms/pass | mean $mean ms/pass | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok | nodes ${nodes:-?}$PP_COL$mtp$slot | wall ${wall}s$CPU_BUSY_TAG$OTHER_BUSY_TAG | majflt $MAJ_WHOLE (timed $timed; ≤ $MAJ_BOUND % of W ${win} s)$COLD_TAG"
    counted || return 0
    count_row
    sums+=("$label|$dep|$r|$agg||$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG|$nslots")
  else
    echo "$ROW_TAG r$r $label d=$dep n=$N ctx=$ctx | tok/s(mean) $tps_mean @ n=$N, depth $dep, $CARD_NAME${rowplace:+ | place $rowplace} | p50 $p50 ms | mean $mean ms | tok/s(p50) $tps_p50 | warm ${warmcol:-0} | first10_p50 $h10 | last10_p50 $t10 | distinct_tokens $uniq_tok | nodes ${nodes:-?}$PP_COL$mtp$slot | wall ${wall}s$CPU_BUSY_TAG$OTHER_BUSY_TAG | majflt $MAJ_WHOLE (timed $timed; ≤ $MAJ_BOUND % of W ${win} s)$COLD_TAG"
    counted || return 0
    count_row
    sums+=("$label|$dep|$r|$tps_mean|$tps_p50|$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG")
  fi
  [ -z "$pps" ] || pass_sums+=("$label|$dep|$r|$pps|$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG")
  [ -z "$PP_N" ] || pp_sums+=("$label|$PP_N|$r|$PP_TPS|$CPU_BUSY_TAG$OTHER_BUSY_TAG$COLD_TAG")
  res_sums_add "$label" "$dep" "$r"
  # The greedy cross-check's ours side: the plain rows (no NAME=VALUE list) of this tree's binary.
  case ${A_KIND[$i]}:$label in
    ours:ours) xc_add ours lcg "$dep" "$r" "$label" "$(sed -n 's/^tokens //p' <<< "$out" | tail -n 1)" ;;
    ours:ours@prose) xc_add ours prose "$dep" "$r" "$label" "$(sed -n 's/^tokens //p' <<< "$out" | tail -n 1)" ;;
  esac
}
# The driver's hooks (tools/ref/load-groups.sh). The timing card must be free before the load's process
# starts: after that the process itself holds it between its arms. The load's lines echoed once; its
# capture line's node count goes into every row of the load (LG_HEADER). lg_cmd is above, with the load
# keys.
LG_HEADER_RE='^(load|capture) '
lg_before_load() {
  guard_other
  guard_timing
}
lg_pre() {
  prime_tag "$1"
  CPU_BUSY_TAG=
  guard_cpu "pre r$2 ${ARMS[$1]}"
  ours_pre "$@"
}
# A COLD row's arm goes on COLD_LIST, its unit's retry (run_unit); a PRIME or COLD row's tag is undone.
lg_post() {
  COLD_QUEUED=0
  ours_post "$@"
  [ "$COLD_QUEUED" = 0 ] || COLD_LIST+=("$1")
  [ "$PRIMING" = 0 ] && [ "$COLD_QUEUED" = 0 ] || ROW_TAG=ROW
}
# A server arm's row after its device column (lcpp-warm.sh srv_row).
srv_tail() { echo " | wall ${1}s$CPU_BUSY_TAG$OTHER_BUSY_TAG$MAJ_COL$COLD_TAG"; }
srv_after() { guard_cpu "post r$1 $2 $3"; }
# unit_guard <index> <round>: the contention guards before an arm's own process: the other card, the
# timing card and the CPU.
unit_guard() {
  CPU_BUSY_TAG=
  guard_other
  guard_timing
  guard_cpu "pre r$2 ${ARMS[$1]}"
}
# run_unit <round> <index...>: one unit of a round: the ours arms of one load key in one process, or one
# reference, server or bin: arm after the card guards. Under the warm rows (cold-blocks.sh) our arms run
# each after its PRIME, and a COLD row's arm runs once more: ours in a fresh load of its prime and the arm,
# a reference as a second process, a server arm inside srv_arm.
run_unit() {
  local r=$1 a
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
  a=${ARMS[$1]}
  unit_guard "$1" "$r"
  case ${A_KIND[$1]} in
    ref)
      ref_arm "${a%%:*}" "${a#*:}" "$r"
      [ "$COLD_QUEUED" = 1 ] || return 0
      ROW_TAG=ROW COLD_TRY=1
      unit_guard "$1" "$r"
      ref_arm "${a%%:*}" "${a#*:}" "$r"
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
# round_order <round> <index...>: those arms' order in the round and its loads, into ORDER_ARMS and
# ORDER_LOADS.
round_order() {
  local r=$1 line i o='' u=''
  local -a units idx names
  shift
  lg_units "$@"
  mapfile -t units < <(lg_round "$r")
  for line in "${units[@]}"; do
    read -r -a idx <<< "$line"
    names=()
    for i in "${idx[@]}"; do names+=("${ARMS[$i]}"); done
    o+="${o:+ }${names[*]}"
    if lg_grouped "${idx[0]}"; then u+="${u:+ }[${names[*]}]"; else u+="${u:+ }${names[*]}"; fi
  done
  ORDER_ARMS=$o ORDER_LOADS=$u
}

# The blocks (the header's Order), fixed before the lease under blocks; cold-blocks.sh's planner reads
# these two. arm_block <i>: the block arm <i> belongs to.
arm_block() {
  case ${A_KIND[$1]} in
    ref)
      case ${A_ENG[$1]} in
        ik | ikdef | ikpp*) echo ik ;;
        lcppfit | lcppppfit*) echo lcppfit ;;
        lcpp | lcpppp*) echo lcpp ;;
        *) echo mrs ;;
      esac
      ;;
    srv) case ${A_ENG[$1]} in *fit*) echo lcppsrvfit ;; *) echo lcppsrv ;; esac ;;
    ours) case ${A_ENG[$1]} in prose) echo prose ;; *) echo ours ;; esac ;;
    *) echo "${A_LABEL[$1]}" ;;
  esac
}
# arm_draws <i>: how many token ids arm <i>'s process takes: a llama-bench arm's std::rand() draws at
# -r 1 (mainline: 2P for a pp arm, whose warm-up is the whole prompt, and D + N + 3 for a decode arm;
# ik: P + 1 and D + N + 4, or N + 3 at D = 0: its warm-up prompt is 1 token), a mistral.rs arm's ids
# over its warm-up request and its timed one, an ours or bin arm's prompt length (N·D for an aggregate
# arm).
arm_draws() {
  local e=${A_ENG[$1]} d=${A_DEP[$1]}
  if [ "${A_KIND[$1]}" != ref ]; then
    arm_feed "$1"
    return
  fi
  case $e in
    lcpppp*) echo $((2 * d)) ;;
    ikpp*) echo $((d + 1)) ;;
    ik | ikdef) if [ "$d" = 0 ]; then echo $((N + 3)); else echo $((d + N + 4)); fi ;;
    mrspp) echo $((2 * (d + 1))) ;;
    mrs | mrspa0) echo $((2 * (d + N))) ;;
    *) echo $((d + N + 3)) ;;
  esac
}
[ "$ORDER" = rotate ] || blocks_plan

# dry_cmd <i>: arm <i>'s command line as the dry run prints it (a reference arm at REF_K when set). A
# prose arm always runs in a load (an ours arm has a load key), and its feed prints as
# `<prose_prompt P>`, as an lcg arm's prints `<lcg_prompt D>`, with the file's count and its first and
# last of the P ids in the note.
dry_cmd() {
  local i=$1 dep=${A_DEP[$1]} ctx note='' feed
  if [ "${A_KIND[$i]}" = srv ]; then
    srv_dry_cmd "$i"
    return
  fi
  if [ "${A_KIND[$i]}" = ref ]; then
    ref_cmd "${A_ENG[$i]}" "$dep"
    echo "timeout --kill-after=10 $BOUND $REF_BIN ${REF_ARGS[*]}   # row label '${REF_LABEL% |}'${REF_BATCH:+, $REF_BATCH}${REF_MARK:+, measured window from /${REF_MARK}/}${REF_FIT:+, $REF_FIT}"
    return
  fi
  ctx=$(arm_ctx "$i")
  [ "${A_LABEL[$i]}" = ours ] || note="   # row label '${A_LABEL[$i]}'"
  feed="<lcg_prompt $(arm_feed "$i")>"
  facts=
  if [ -n "${A_TOK[$i]}" ]; then
    feed="<prose_prompt $(arm_feed "$i")>"
    facts=", $(arm_feed "$i") of the file's ${PROSE_N:-?} ids, first ${A_TOK[$i]%%,*}, last ${A_TOK[$i]##*,}"
  fi
  [ -z "${A_SLOTS[$i]}" ] || facts+=", ${A_SLOTS[$i]} slots of $dep ids in one pass (an aggregate row)"
  if lg_grouped "$i"; then
    arm_envs "$i"
    echo "one arm of a load: timeout --kill-after=10 \$((BOUND x arms + BOUND)) ${ARM_ENVS[*]:+env ${ARM_ENVS[*]} }${A_BIN[$i]} --arm $feed ... -n $N --ctx $ctx${A_PLACE[$i]:+ --place ${A_PLACE[$i]}} --time${WARM:+ --warm $WARM} --arm-sync   # load key ${LG_KEY[$i]}$facts"
  else
    echo "timeout --kill-after=10 $BOUND ${A_BIN[$i]} --tokens $feed -n $N --ctx $ctx${A_PLACE[$i]:+ --place ${A_PLACE[$i]}} --time${WARM:+ --warm $WARM}$note"
  fi
}

# ratio_table <prefix> <keys> <labels> <tagged> <tag field> [base]: records `label|key|round|value|…` on
# stdin; for every key and every label of <labels>, each round's base / label ratio (arms that ran more
# than once in a round averaged first), their mean with its 95 % interval (Student t at rounds - 1
# degrees of freedom, T975) and the ratio of the arm means. The base is ours (the default) or
# ours@prose for the prose table. With tagged = 1 field 5 is the row's tags, and the
# line ends with each side's count of [other-busy]; a <tag field> above 0 is the record's field that
# holds its tags, and the line then ends with each side's count of [cold].
ratio_table() {
  awk -F'|' -v prefix="$1" -v deps="$2" -v refs="$3" -v tagged="$4" -v tf="${5:-0}" -v base="${6:-ours}" -v rounds="$ROUNDS" -v t975="$T975" '{
  k = $1 SUBSEP $2 SUBSEP $3; rs[k] += $4; rn[k]++
  a = $1 SUBSEP $2; as[a] += $4; an[a]++
  if (tagged && $5 ~ /other-busy/) bo[a]++
  if (tagged && $5 ~ /cpu-busy/) bc[a]++
  if (tf && $tf ~ /cold/) bk[a]++
  if (tf && $tf ~ /cpu-busy/) bt[a]++
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
    busy = tagged ? sprintf("  busy: %s [cpu-busy %d/%d] [other-busy %d/%d], %s [cpu-busy %d/%d] [other-busy %d/%d]", base, bc[ao], an[ao], bo[ao], an[ao], ref, bc[ar], an[ar], bo[ar], an[ar]) : ""
    if (!tagged && tf) busy = sprintf("  cpu-busy: %s %d/%d, %s %d/%d", base, bt[ao], an[ao], ref, bt[ar], an[ar])
    cold = tf ? sprintf("  cold: %s %d/%d, %s %d/%d", base, bk[ao], an[ao], ref, bk[ar], an[ar]) : ""
    printf "%s%-5s %s/%-6s  mean %.4f %s (n=%d)  of means %.4f  per round:%s%s%s\n", prefix, d[i], base, ref, m, ci, c, (as[ao] / an[ao]) / (as[ar] / an[ar]), list, busy, cold
  }
}'
}

if [ -n "$DRY" ]; then
  echo "[dry] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s timing_gpu=$TIMING_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "[dry] place: ${PLACE:-unset, the binary default a}"
  [ -z "$PLACES_SET" ] || echo "[dry] placements: $(place_line "${PLACE_ARMS[@]}")"
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
  ref_witness | sed 's/^   /[dry]/'
  for i in "${!ARMS[@]}"; do echo "[dry] ${ARMS[$i]}: $(dry_cmd "$i")"; done
  [ "$WARM_ROWS" = 0 ] || echo "[dry] warm rows: each of our arms after a same-id PRIME in its load; a counted row tagged [cold] prints as COLD and runs once more (cold-blocks.sh)"
  if [ "$ORDER" = rotate ]; then
    [ "$AB_WARMUP" = 0 ] || echo "[dry] warmup: ${ARMS[0]} once before round 1 (its command line above), discarded — its row prints as WARMUP r0 and is in no mean, ratio or row count (BLOOMERY_AB_WARMUP=0 skips it)"
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

majflt_require depth-qwen3moe.sh
timing_cards_precheck || {
  rc=$?
  echo "depth-qwen3moe.sh: two cards, refused before the lease: $TWOCARD_WHY" >&2
  exit "$rc"
}
lease_take
timing_cards_start
[ -z "$TIMING_CARDS" ] || echo "[config] two cards: $TIMING_CARDS_NAME, the profile's two-card line: $TWO_CARD_PLACEMENT"
echo "[config] model=$MODEL n=$N rounds=$ROUNDS warm=${WARM:-0} card=$CARD_NAME arm_bound=${BOUND}s"
echo "[config] ours: $BIN ctx=${GEN_CTX:-D+N rounded up to 256}${PLACE:+ --place $PLACE}"
[ -z "$PLACES_SET" ] || echo "[config] placements: $(place_line "${PLACE_ARMS[@]}")"
echo "[config] cpu guard: comms=[$CPU_BUSY_COMMS] threshold=${CPU_BUSY_PCT}% strict=${BLOOMERY_OTHER_STRICT:-0}"
[ -z "$PROSE_N" ] || echo "[config] prose: the first P ids of $(corpus_file) (${PROSE_N} ids), in prose's own tables"
blocks_config
echo "[config] cold tag: majflt in the row's measured window (ours: from its prompt_ids line; lcpp: from its --progress line; mrs: from its Iteration line; ik and bin: the whole process) × ${COLD_US} µs ≥ ${COLD_PCT} % of that window"
# Each engine's line only when it has arms: a profile names only the engines it runs (qwen35moe.sh).
[ "$ik" = 0 ] || echo "[config] ik: $IKBIN flags=$IK_GPU_FLAGS ikdef flags=$IK_GPU_DEFAULT_FLAGS"
[ "$lcpp" = 0 ] || echo "[config] lcpp: $LCPPBIN flags=$LCPP_GPU_FLAGS (lcpp and lcpppp: --progress)"
if [ "$lcppfit" = 1 ]; then
  lcpp_fit_flags "$LCPP_GPU_FLAGS"
  echo "[config] lcppfit: $LCPPBIN flags=$FIT_FLAGS (llama-bench's fit places the model; dropped: $FIT_DROPPED; lcppppfit<U>: -ub U -b max(U, 2048); --progress as lcpp)"
fi
[ "$mrs" = 0 ] || echo "[config] mrs: $MRSBIN flags=$MRS_FLAGS (mrs: --pa-context-len C, mrspa0: --paged-attn off)"
[ "$srv" = 0 ] || srv_config
warm_rows_config
echo "[config] prefill: ikpp/lcpppp run llama-bench -p P -n 0 -r 1 at the flags above (<U>: -ub U -b max(U, 2048)), mrspp mistralrs bench --prompt-len P --gen-len 1; ours from its time prompt row"
echo "[config] arms=${ARMS[*]} timing_gpu=$TIMING_GPU other_gpu=$OTHER_GPU CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
witness pre
ref_witness
guard_other
guard_timing
guard_cpu pre

sums=() pp_sums=() pass_sums=()
n_rows=0 busy_rows=0 other_rows=0 cold_rows=0
if [ "$ORDER" = rotate ]; then
  if [ "$AB_WARMUP" = 1 ]; then
    ROW_TAG=WARMUP
    run_unit 0 0
    ROW_TAG=ROW
    echo "[warmup] ${ARMS[0]} ran once before round 1 and is discarded (the WARMUP or FAIL r0 row above)"
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
# The greedy cross-check (lcpp-warm.sh), before the tables: a server row whose first id parts from ours
# is a failed arm and drops out at its depth or P with them; a later id's part is a [xcheck-tail] line.
xc_table
[ ${#XC_FAILED[@]} -eq 0 ] || FAILED+=("${XC_FAILED[@]}")
[ ${#XC_DROP[@]} -eq 0 ] || FAILED_KEYS+=("${XC_DROP[@]}")
# A failed arm drops out at its depth or P (cold-blocks.sh).
failed_tally
# failed_tally drops from sums and pp_sums; the pass records drop the same label|key pairs here.
[ ${#pass_sums[@]} -eq 0 ] || mapfile -t pass_sums < <(printf '%s\n' "${pass_sums[@]}" | drop_failed)
echo "=== per-arm means (tok/s @ n=$N, $CARD_NAME). First column: ours from mean_ms, the references"
echo "    their bench's own mean (llama-bench over the N steps, mistralrs bench over N - 1 intervals)"
echo "    — the cross-engine ratio reads these. The p50 column is ours only. ==="
[ ${#sums[@]} -eq 0 ] || printf '%s\n' "${sums[@]}" | awk -F'|' '{
  k = $1 " d=" $2; s[k] += $4; n[k]++; if ($5 != "") { sp[k] += $5; np[k]++ }
  if ($6 ~ /cold/) c[k]++
  if ($7 != "") agg[k] = $7
  if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
} END { for (k in s) {
  spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
  printf "mean %-14s %8.2f tok/s  [%s..%s, spread %.2f%%]  %s (n=%d)  [cold %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], spread, (np[k] ? sprintf("%.2f tok/s(p50)", sp[k] / np[k]) : (k in agg ? sprintf("(aggregate of %d slots)", agg[k]) : "")), n[k], c[k], n[k] } }' | sort
echo
echo "=== ours / reference per depth: each round's ratio of the pair measured in that round (arms"
echo "    that ran more than once in a round are averaged first), their mean with its 95 % interval"
echo "    (Student t, rounds - 1 degrees of freedom; 2.0 past 21 rounds), and the ratio of the arm means ==="
deps=$(printf '%s\n' "${A_DEP[@]}" | sort -un | tr '\n' ' ')
# The prose labels have their own table: their prompt is not the one ours and the references ran. A label
# fed the prose ids is `ours@prose`, `ours@prose@…` or `<server engine>@prose`.
prose_re='^ours@prose(@|$)|^[^@]+@prose$'
# The aggregate arms' labels (the header's <D>@BLOOMERY_GEN_SLOTS=N): their own table below, out of these.
SLOT_LABELS=''
for i in "${!ARMS[@]}"; do [ -z "${A_SLOTS[$i]}" ] || SLOT_LABELS+="${A_LABEL[$i]}"$'\n'; done
# not_slots: the labels on stdin less the aggregate arms'.
not_slots() { awk -v s="$SLOT_LABELS" 'BEGIN { n = split(s, a, "\n"); for (i = 1; i <= n; i++) if (a[i] != "") x[a[i]] = 1 } !($0 in x)'; }
refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -vx ours | grep -vE "$prose_re" | not_slots | sort -u | tr '\n' ' ')
printf '%s\n' "${sums[@]}" | ratio_table "ratio d=" "$deps" "$refs" 0 6
prose_refs=$(printf '%s\n' "${A_LABEL[@]}" | grep -E "$prose_re" | grep -vx ours@prose | not_slots | sort -u | tr '\n' ' ')
if [ -n "$prose_refs" ]; then
  echo
  echo "=== the prose prompt: ours@prose / each arm on the prose ids per P, the same statistics ==="
  printf '%s\n' "${sums[@]}" | ratio_table "ratio prose d=" "$deps" "$prose_refs" 0 6 ours@prose
fi
# The aggregate arms: each one's aggregate over its plain twin's tok/s, the label less its
# BLOOMERY_GEN_SLOTS item (ours, ours@prose, ours@<the rest of its list>), the same statistics.
if [ -n "$SLOT_LABELS" ]; then
  echo
  echo "=== N slots in one pass: each aggregate arm's tok/s (Σ positions / Σ ms of its rounds) over its plain"
  echo "    twin's tok/s per depth, the same statistics; above 1 the N streams in one pass outrun one stream ==="
  while IFS= read -r sl; do
    [ -n "$sl" ] || continue
    twin=$(awk -v l="$sl" 'BEGIN {
      at = index(l, "@prose@") ? index(l, "@prose@") + 6 : index(l, "@"); head = substr(l, 1, at - 1); n = split(substr(l, at + 1), kv, ",")
      for (i = 1; i <= n; i++) if (kv[i] !~ /^BLOOMERY_GEN_SLOTS=/) rest = rest (rest == "" ? "" : ",") kv[i]
      print head (rest == "" ? "" : "@" rest) }')
    if ! printf '%s\n' "${A_LABEL[@]}" | grep -qxF -- "$twin"; then
      echo "ratio slots: $sl has no plain twin $twin among the arms: no ratio"
      continue
    fi
    printf '%s\n' "${sums[@]}" | ratio_table "ratio slots d=" "$deps" "$twin" 0 6 "$sl"
  done < <(printf '%s' "$SLOT_LABELS" | sort -u)
fi
# The pass time of the MTP rows: a row's tok/s is its pass time over E(4), and E(4) moves with the text
# two builds generate, so a code change is read on the pass time. The records hold passes a second
# (1000 / ms a pass), so a ratio above 1 means ours passes faster, as in the decode table.
if [ ${#pass_sums[@]} -gt 0 ]; then
  echo
  echo "=== pass time per arm (ms a verify pass under MTP, mean_ms × steps / passes of the row's SMOKE line:"
  echo "    the counted passes' wall over their number, $CARD_NAME). The ratios below are over passes a"
  echo "    second, 1000 / ms, the decode table's statistics ==="
  printf '%s\n' "${pass_sums[@]}" | awk -F'|' '{
    k = $1 " d=" $2; ms = 1e3 / $4; s[k] += ms; n[k]++
    if ($5 ~ /cold/) c[k]++
    if (mn[k] == "" || ms < mn[k] + 0) mn[k] = ms; if (mx[k] == "" || ms > mx[k] + 0) mx[k] = ms
  } END { for (k in s) printf "mean pass %-14s %9.4f ms/pass  [%.4f..%.4f]  (n=%d)  [cold %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], n[k], c[k], n[k] }' | sort
  printf '%s\n' "${pass_sums[@]}" | ratio_table "ratio pass d=" "$deps" "$refs" 0 5
  if [ -n "$prose_refs" ]; then
    echo
    echo "=== the prose prompt's pass time: ours@prose / each arm on the prose ids per P, the same statistics ==="
    printf '%s\n' "${pass_sums[@]}" | ratio_table "ratio pass prose d=" "$deps" "$prose_refs" 0 5 ours@prose
  fi
fi
if [ ${#pp_sums[@]} -gt 0 ]; then
  echo
  echo "=== prefill per prompt length (tok/s(pp) @ n=0, prompt P, $CARD_NAME). Ours: its time prompt"
  echo "    row (prefill through its token's readback); llama-bench's pp value over one repetition;"
  echo "    mistral.rs P / TTFT over one request. The tags count the rows that met contention or faults. ==="
  printf '%s\n' "${pp_sums[@]}" | awk -F'|' '{
    k = $1 " p=" $2; s[k] += $4; n[k]++
    if (mn[k] == "" || $4 + 0 < mn[k] + 0) mn[k] = $4; if (mx[k] == "" || $4 + 0 > mx[k] + 0) mx[k] = $4
    if ($5 ~ /cpu-busy/) c[k]++; if ($5 ~ /other-busy/) o[k]++; if ($5 ~ /cold/) f[k]++
  } END { for (k in s) {
    spread = (mn[k] > 0) ? 100 * (mx[k] - mn[k]) / mn[k] : 0
    printf "mean pp %-14s %8.2f tok/s(pp)  [%s..%s, spread %.2f%%]  (n=%d)  [cpu-busy %d/%d] [other-busy %d/%d] [cold %d/%d]\n", k, s[k] / n[k], mn[k], mx[k], spread, n[k], c[k], n[k], o[k], n[k], f[k], n[k] } }' | sort
  echo
  echo "=== ours / reference prefill per prompt length: the decode table's statistics over the pp"
  echo "    values, then how many of each side's rows carried [other-busy] and [cold] ==="
  pp_keys=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f2 | sort -un | tr '\n' ' ')
  pp_refs=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep -vx ours | grep -vE "$prose_re" | sort -u | tr '\n' ' ')
  printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp p=" "$pp_keys" "$pp_refs" 1 5
  pp_prose=$(printf '%s\n' "${pp_sums[@]}" | cut -d'|' -f1 | grep -E "$prose_re" | grep -vx ours@prose | sort -u | tr '\n' ' ')
  if [ -n "$pp_prose" ]; then
    echo
    echo "=== the prose prompt's prefill: ours@prose / each arm on the prose ids per P, the same statistics ==="
    printf '%s\n' "${pp_sums[@]}" | ratio_table "ratio pp prose p=" "$pp_keys" "$pp_prose" 1 5 ours@prose
  fi
fi
res_sums_table
witness post
ref_witness
failed_end
