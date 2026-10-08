//! `generate_qwen3moe` — the decode CLI of the qwen3moe family's engines: a
//! qwen3moe (Qwen3-30B-A3B), a qwen35moe (Qwen3.6-35B-A3B) or a qwen4exp
//! (Qwen3.8-Flash-Next) file, the architecture read from the file's header
//! (any other is refused by name), the whole model on one card or placed by
//! a plan over a card and the host tier (`--place`), greedy, one token per
//! step, and the timing runner's ruler.
//!
//!     generate_qwen3moe (--prompt <text> | --tokens a,b,c | --seed-depth D)
//!                       [-n N] [--ctx C] [--mode eager|graph]
//!                       [--prefill auto|pass|gemm|step] [--place a|gate|bp|<cards>]
//!                       [--time [--warm W]] [--logits] [--last-step]
//!     generate_qwen3moe --arm a,b,c[/N] [--arm ...] [--arm-sync] [-n N] [--ctx C] ...
//!     generate_qwen3moe --dump-taps DIR --tokens-file F [--tokens-file F ...]
//!                       --seqs S --prompt-len P [-n N] [--ctx C]
//!
//! Defaults: N 32, C 4096, mode graph, prefill auto, W 0. A flag given twice
//! takes its last value. `--prompt` tokenizes the text with the file's own vocabulary
//! (`tokenizer`, no BOS: the file sets `add_bos_token` false; no
//! chat template) and prints the generated text after the ids.
//!
//! `--last-step` (a qwen3moe or qwen35moe file, refused by name on a
//! qwen4exp one and beside `--time` and `--seed-depth`) feeds the prompt
//! less its last id by `--prefill`, then the last id as one step, whose
//! argmax is generated token 0: the server's cut (`bloomery-serve --model
//! qwen3` prefills a request's prompt less its last id, then steps it), so
//! the two print the same greedy ids past eight prompt ids, where the
//! ubatch and the step compute the last position by different arms.
//!
//! The prompt is prefilled (`Qwen3moeModel::prefill_with` by the `--prefill`
//! path: `auto` takes one pass for a prompt of up to eight ids and GEMM
//! ubatches of up to the load's ubatch size for a longer one, a tail of up
//! to eight after them one pass; `pass` passes of up to eight positions, the
//! same cache rows and answer as one step per token, in graph mode each a
//! replay of the pass of its size; `gemm` ubatches only); the argmax after
//! its last token is generated token 0, and `N − 1` feedback steps follow.
//! The ubatch size is `BLOOMERY_QWEN3_UBATCH` (1..=4096, default 4096),
//! clipped to the cache, read once: the plan counts that ubatch's arena and
//! the load runs it.
//!
//! A qwen4exp file's plan puts each layer's id prefix on the card as its
//! budget holds (`place::Experts::Card`, `BLOOMERY_QWEN38_EXPERTS` unset or
//! `card`), which the step's, the verify's and the pass's card leg and the
//! ubatch walk's card route run, or with `BLOOMERY_QWEN38_EXPERTS=host` every
//! routed expert on the host tier; a set `card` is refused on the other
//! families' files; the `plan` line prints `experts=` and the `load` line `card_layers=` and
//! `card_stacks=` (each card layer's gate·up and down types, with its layers).
//!
//! A qwen35moe file runs `auto` and `gemm` through `Body35`'s prompt call
//! (`Qwen35moeModel::prefill_with`: the same plan, every unit a walk of the
//! layer program over the ubatch arena, eager in either mode — a ubatch of
//! more than eight ids through each op's wide arm, a pass through its gemv
//! arm), its ubatch size `BLOOMERY_QWEN3_UBATCH` too, clipped to the cache;
//! `pass` runs passes of up to eight positions, each a `step_rows` of its
//! size (in graph mode a replay) — bit for bit its one-token steps — and a
//! pass of one id a step. Its `load` line names `ubatch=`; its `stat prompt`
//! line's `ubatch_tokens=` counts the ubatches' ids, and its `image_bytes=`
//! is the whole prompt's image, which the passes after them read too.
//! `--prompt` is tokenized with the file's pre-tokenizer (`qwen35`).
//!
//! A qwen4exp file runs through `Body38`, placed by its own plan
//! (`model::arch::qwen35moe::place`: every layer, the head and the embedding
//! on one card, its routed experts as above) on the card `--place`
//! names — `a` the A6000 (the default), `gate` the 3090, `bp` plan (b′):
//! the A6000 as under `a` with the 3090 as its expert tier
//! (`place::machine_bp`, the tier's experts printed as the `plan` line's
//! `tier=` and `tier_experts=`; the tier serves the decode walks and the
//! ubatch walk, not the pass) — and printed as a `plan` line. A `place unset`
//! record, first of the run's records after a set residency lever's, names
//! the placement and why: the common rule's
//! (`generate::Place::choose` by `q38place::Q38_RULE`), so unset keeps the
//! cards' offer — `bp` on two cards while its plan holds the rule's
//! break-even experts on the tier card or more and more than none, else `a`
//! — and a card list that is none of `a`, `gate` and `bp` is refused by
//! name. Its `--prefill` is `auto` (the default: `gemm` for a prompt
//! of nine positions or more, `pass` below; `gemm` at every length under
//! `bp`), `gemm` (ubatches of up to the
//! load's size through the host tier's batch port at their width, the Q8_0
//! projections on q8 activations), `pass` (eager passes of up to eight
//! positions through the same port) or `step` (one captured step a
//! position); `--seed-depth` is refused by name (no synthetic depth). Its
//! `load` line names `store_bytes=`, `prefill=`, `ubatch=` and `place=`; in
//! graph mode its
//! `capture` line counts the step graph's nodes by kind against the
//! program's count, and a mismatch ends the run by name. Its plan prints as
//! `step:1x<P>`, `pass:<sizes>` or `ubatch:<sizes>`, and `time prompt`'s
//! `kind=` is the path the prompt ran — `step`, `pass` or `gemm`, never
//! `auto`.
//!
//! `--place W` on a qwen3moe or qwen35moe file places it by its plan
//! (`shared/qwen3moe_place.rs`): `W` a placement word of `generate::Place`
//! (`a`, `gate`, or a card list of one card — `bp` and every list with a
//! tier card refused by name: the program hangs no expert tier), the plan
//! on that card under `BLOOMERY_CARD_BUDGET` — the trunk on the card, each
//! layer's routed id prefix as the budget holds, the rest on the host tier
//! — printed as a `plan` line (`record::PLAN38`) before the `load` line. A
//! plan with no host expert loads the whole model on the plan's card; any
//! other runs the decode step (graph or eager) through the host tier's step
//! port and the prompt as eager passes of up to eight ids through its batch
//! port (`auto` and `pass`; `--prefill gemm` refused by name), and in graph
//! mode captures the step only. Unset, the load is the whole-card one on
//! `a`'s card, the largest visible card, while that fits the card's free
//! bytes — the plan's own test of the whole file's card bytes plus its KV,
//! context, scratch and margin against the census's free reading — and when
//! it does not, the placed plan on that card runs instead, its `plan` line
//! naming why (`why=whole_does_not_fit`). A `place unset` record names the
//! placement the flag or the common rule gave (`generate::Place::choose` by
//! `q3place::Q3_RULE`: the body serves no tier card, so unset is `a`) before
//! the `load` line; the load itself takes the flag as given, or unset the
//! pick above. The placement's levers
//! (`BLOOMERY_CARD_BUDGET` and the host set's, `q3place::PLACED_LEVERS`)
//! act on a run that names `--place`, on any file; set without it they are
//! refused by name.
//!
//! `BLOOMERY_ROUTE_TRACE=<dir>` (a qwen4exp file only, refused by name on
//! the others) writes the engine's route trace of the run into `dir`, a new
//! directory made before the load (`crates/gpu/src/host/route_trace.rs`):
//! every position's routed ids per layer and the slot each ran in, as a
//! router set. The prompt's ids run one step each and are recorded as the
//! call's positions — `--prefill step` is required, `--prefill pass` and
//! `gemm` (and the unset `auto`) refused by name, and so is `--time` (the
//! trace rewrites its manifest after every position, so a timed run's
//! numbers are not a measurement). A `--arm` list writes one `call` row an
//! arm under one set, and the set's `chunk` header line names the positions
//! every arm writes when they all write the same count. The run ends with
//! `route trace <dir> positions=<n> complete` once the set is sealed.
//!
//! `BLOOMERY_RESIDENCY` set prints as a `residency lever` record first
//! thing. Unset, the Qwen3.8 rule picks the word
//! (`bloomery_levers::residency38_unset`, `residency38_at_plan`) and a
//! `residency unset` record prints it with why: on a qwen4exp file under
//! `--place a` or `bp` `mid-p<P>-s1`, P half the fewest card experts a layer of the
//! plan the load runs, after the `plan` line; `off` under `--place gate`,
//! beside `BLOOMERY_ROUTE_TRACE`, with `--prefill step`, when the plan holds
//! no card expert or its fewest leave no room, and when the churn pool does
//! not fit the plan's host headroom or what `MemAvailable` leaves past the
//! plan's host need (after the `plan` line too), and on a
//! qwen3moe or qwen35moe file and under `--dump-taps` (before the load) —
//! never a refusal. Running `mid-p<P>-s<S>` on a
//! qwen4exp file (plain or drafted, either `--place`), the load runs the
//! common residency machine over the card's routed stacks
//! (`Body38::open_placed_residency`): a `residency host` record follows the
//! `plan` line, each arm prints its boundaries' `residency pass` records
//! after its other lines — after its error, when it failed — and each arm
//! after the first opens with the `residency reset` record of the clear
//! before it. A set `mid-…` is
//! refused by name on a qwen3moe or qwen35moe file, under `--dump-taps` and
//! beside `BLOOMERY_ROUTE_TRACE`, when the plan's host headroom cannot take
//! its churn pool, and by the body at a step-fed prompt.
//!
//! `BLOOMERY_XSTREAM` (a qwen4exp file only, refused by name on the
//! others) sets what each prompt call moves (`Body38::set_xstream`):
//! `admit` streams its hottest host experts into the residency pool, `split`
//! admits and then streams the host experts the stream rule sends to the
//! card through the expert stream's ring. Unset (the rule the Qwen3.8 serve
//! seat shares, `shared/xstream38.rs`) it is `split` under `--place a`
//! with a residency machine — `admit` under `bp`, whose expert
//! tier the ring does not serve, and where the card has no room for the
//! ring, named on the `xstream=` line — and `off` everywhere else; `admit`
//! or `split` beside `BLOOMERY_RESIDENCY=off` is refused by name, and so is
//! `BLOOMERY_HOSTSTREAM` set (V4.1's lever) on a qwen4exp file. The load
//! prints an `xstream=` line after its other lines; a streaming call prints
//! a `call stream` record a pick, an `xstream` record a streamed layer, and
//! `call stream end` and `xstream end` records after its arm's lines.
//!
//! `BLOOMERY_DRAFT` unset on a qwen4exp file follows the placement
//! (`bloomery_levers::draft38_unset`): under `--place a` or `bp` the MTP draft runs
//! when a regular file is where it would be opened; the plain path runs
//! under `--place gate`, beside `--logits` or `BLOOMERY_ROUTE_TRACE`, with no
//! file there, and when an arm's last window would pass `--ctx` (depth + n +
//! 2 positions), with a `load draft=off (<why>)` record after the `load`
//! line — `no file at <path>` for the missing file — never a refusal.
//! `BLOOMERY_DRAFT=off` is the plain path with the same record.
//!
//! Drafting (`BLOOMERY_DRAFT=mtp`, or unset as above; a qwen4exp file only,
//! every other family and word refused by name) the decode runs through the runtime's
//! speculative loop with the file's MTP draft (`app::arch::qwen4exp`'s
//! `MtpDraft` over the shared draft file beside the target or
//! `BLOOMERY_MTP_DRAFT`'s, its head `BLOOMERY_MTP_HEAD_ROWS`'s: unset the
//! shipped list on a target of its tokenizer, `full` the full head):
//! windows of four rows — the target's
//! verify of the draft's three ids, the kept rows committed, the draft's
//! next chain one readback — and the greedy ids are the plain run's. A
//! `load draft=mtp` line follows the `load` line (the draft's resident
//! bytes, its program's arena and its head), then the `mtp head` record
//! (which head, what picked it and why), the `capture` line the verify
//! passes' widths; each pass prints its `step` lines one kept token a line
//! and, under `--time`, a `time pass` row (its wall, positions and kept
//! rows) and a `time step` row a kept position (its pass's wall over its
//! positions, the row a plain run's step wall compares with), and an
//! `mtp summary` record closes the arm: the windows' kept lengths, the
//! positions and their rate. Under `BLOOMERY_MTP_WINDOWS=1` an `mtp window`
//! record a drafted pass follows it: the pass, the target's position, the
//! proposal's ids, each one's probability among the draft head's rows and
//! how many the target kept (`tools/flow/q38width.py` replays them). A
//! sampling request is a server matter; this CLI is greedy.
//!
//! Lines: `prompt_ids`, `load` (`arch=` the file's architecture; with the
//! decode flash pass: `flash_mma=`; for qwen3moe the ubatches' attention,
//! `ubatch_attn=gqa_prefill_flash`, and their size,
//! `ubatch=`, on the `load` line because the `time prompt` row's shape is
//! parsed to its end; and `rope_table_us=`, the host time that computed the
//! rope table every path reads at load, one `RopeTable::push` for each of
//! the `ctx` positions — a runtime value; for qwen35moe `store_bytes=`, the
//! attention layers' K/V planes and the delta layers' states), in graph
//! mode `capture graph_nodes=` and `capture prefill_graphs=<n> nodes=<m=1>,…
//! ms= vram_bytes=` (every pass size captured before the prompt: its wall
//! and the card's free bytes it took, runtime values; for qwen35moe the
//! `m = 1` entry is the decode step's graph, which a pass of one id
//! replays), `step 0 pos tok`
//! (with the units the prompt took: `prefill_steps=`, and their shapes:
//! `plan=ubatch:512x2 pass:1`, a run of equal sizes as `<size>x<k>`), then,
//! all written after the loop,
//! `time prompt n=<P> ms= tok/s= passes=<K> kind=<gemm|prefill>` (`K` the
//! ubatches and passes; `gemm` when a ubatch ran)
//! (the wall of `prefill` through its token's readback, on every run: a
//! runtime value like the `load` line, a measurement only under the lease;
//! the prefill arena is allocated at load and the passes captured before
//! it, so that wall carries neither),
//! `stat prompt ubatch_tokens=<n> image_bytes= fill_us= copy_us=` (the host
//! prologue inside that wall before the first ubatch launch: the prompt
//! image's fill and the enqueue of its copy to the card, which the launches
//! behind it wait for, not the host; runtime values read from three clock
//! reads the engine takes on every prompt; a body that writes no image, or a
//! prompt no ubatch ran, prints `ubatch_tokens=<n> (no prompt image)`),
//! per feedback step `step i pos tok` (and `time step i ms=` under
//! `--time`); then
//! `tokens [..]`, `text` for a `--prompt` run, and under `--time` the
//! `SMOKE` footer with `generate`'s keys (`p50_ms=`, `mean_ms=`, `warm=`,
//! `tok/s(p50)=`).
//!
//! `--seed-depth D` stands the model at depth D: `seed_depth(D − 1)` fills
//! the caches with a pattern (for qwen35moe the attention layers' K/V
//! planes; the delta layers' states stay as they stand), and one literal
//! token (id 0) is prefilled after them, so the timed steps run at the
//! positions a D-token prompt leaves. The tokens it prints are
//! meaningless; only the timing is. Its `time prompt` row is that one
//! token's prefill, `n=1`.
//!
//! `--arm a,b,c[/N]` runs several arms after one load (`app::Session::arms`),
//! in the order given: each prefills its ids and generates its `N` (`-n`
//! when it names none), and each after the first starts from the session's
//! clear, so it prints what it prints in a fresh process. Each arm opens
//! with `arm i=<i> arms=<k> ids=<P> n=<N>` and its `prompt_ids` line, then
//! the lines a one-prompt run prints from `step 0` on; the load and capture
//! lines print once, before arm 0. `--arm` does not mix with `--prompt`,
//! `--tokens` or `--seed-depth`. `--arm-sync` makes each arm wait, after its
//! `arm` line, for one line on stdin: the timing runner takes its witness
//! blocks there. A failed arm ends the process, naming the arm.
//!
//! `BLOOMERY_GEN_SLOTS=N` (2 to the body's pass rows, 8 on every file;
//! unset or 1 is the one-sequence run above) decodes N streams in one pass
//! (`GpuModel::step_slots`) on a whole-card qwen3moe, qwen35moe or qwen4exp
//! file, whose load is then planned for N sequences
//! (`PlanInputs::plan_with_slots`). Each arm's ids (`--tokens`, or an
//! `--arm`'s) are N windows of equal length P, window j prefilled into slot
//! j from its reset by `--prefill` (on a qwen3moe or qwen35moe file in
//! graph mode a slot past 0 captures its prefill passes first, outside the
//! prefill's wall);
//! an arm after the first starts from the residency's seed too (its
//! `residency reset` record before its `arm` line); the pass of a row a slot is
//! captured before the rounds (`capture slots=N rows=1 graph_nodes=`), then
//! `-n` − 1 rounds of one pass follow, each slot fed its own argmax, so
//! slot j prints the ids `--tokens <window j>` prints alone. Each slot's
//! `step 0`, `step` lines carry `slot=<j>` after the token, its `time
//! prompt`, `stat prompt` and `tokens` lines end in it; under `--time` each
//! round prints `time pass <r> ms= positions=N kind=slots` (the pass and the
//! N ids' readback, a `warm` round marked) and the `SMOKE` footer names
//! `slots=`, the counted `rounds=` and `positions=`, the rounds' `p50_ms=`
//! and `mean_ms=` and `tok/s(aggregate)=`; a qwen4exp arm's `residency pass`
//! records follow its lines. Refused by name on a placed qwen3moe or
//! qwen35moe load (`--place`, or the unplaced default's fallback), on a
//! qwen4exp run that drafts (`BLOOMERY_DRAFT=mtp`,
//! or unset under `--place a` with the draft file there: the plain pass
//! runs under `BLOOMERY_DRAFT=off`), for an id count N does not divide, and
//! beside `--prompt`, `--seed-depth`, `--last-step`, `--logits`,
//! `--dump-taps`, and on a qwen4exp file `BLOOMERY_STEP_STATS=1` and
//! `BLOOMERY_ROUTE_TRACE`; the qwen4exp body refuses a pass of several
//! slots on a load with an expert tier card (`--place bp`) by name.
//!
//! `--logits` prints `logits n= argmax= margin= fnv64=` after the `tokens`
//! line: the head's last logits row, read back once — the argmax's lead over
//! the best other logit, and the row by its f32 bits (FNV-1a 64).
//!
//! `BLOOMERY_STEP_STATS=1` on a qwen4exp file reads, before the first
//! generated step and after each one, the host tier's counters
//! (`HybridStats`) with the go waits of the services since the read before,
//! the process's page faults (`/proc/self/stat`) and the card's free device
//! bytes (`host_stats::Probe`), and prints, after the arm's other lines, one
//! `stat step` record a step (`record::STAT_STEP_HOST`: `served` layers,
//! their summed `leg_us` and `host_slots`, the go waits) and a
//! `stat summary` over the steps past `--warm`. It also arms the ubatch
//! walk's timing: after a prompt's `stat prompt host` line, one
//! `stat prompt lb` record a
//! layer-batch its ubatches served and a `stat prompt split` over them
//! (the serve's wait and union beside the card time of every part). Unset,
//! or on another file, nothing is read and nothing recorded.
//!
//! `--dump-taps DIR` (a qwen3moe file only) writes the layer-tap dump of
//! `shared/qwen3moe_taps.rs` into DIR, an empty or new directory: from each
//! `--tokens-file` (one id a line) `S` prompts of `P` ids, windows spread
//! over the file, each run from a reset one eager step per id with the layer
//! taps on, then `N` greedy steps. The steps are the decode path the pass
//! prefill equals bit for bit, so a sequence's ids are what `--tokens <its
//! prompt> -n N --prefill pass --ctx C` prints; the GEMM ubatches, which `auto`
//! takes for a prompt of more than eight ids, have no taps. It prints the
//! `load` line, then a `taps seq` record per sequence and a `taps dump`
//! record (`record::GENERATE_QWEN3MOE`), and mixes with no flag but `-n` and
//! `--ctx`.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh`), never at a bare prompt. The cache's
//! height is `--ctx`: the flash's segment grid is fixed by it, so a timed
//! run names its ctx.
//!
//! On a qwen4exp file `--ctx` is at most the file's `context_length`
//! (`place::serve_ctx`; YaRN scaling past it is not built), refused by name
//! past it; the `load` line prints the stores' `ctx`, that cap as
//! `ctx_max`, `ctx_train` and `verified`, the deepest context the reference
//! sets hold our numbers to ik's at (`refset::arch::qwen4exp::VERIFIED_POSITIONS`),
//! which bounds nothing.

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!("generate_qwen3moe: built without the `gpu` feature; see `just gen-qwen3moe`.");
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_qwen3moe", cli::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_taps.rs"]
mod taps;

#[cfg(feature = "gpu")]
#[path = "shared/qwen3moe_place.rs"]
mod q3place;

#[cfg(feature = "gpu")]
#[path = "shared/qwen38_place.rs"]
mod q38place;

#[cfg(feature = "gpu")]
#[path = "shared/gen_slots.rs"]
mod gen_slots;

#[cfg(feature = "gpu")]
#[path = "shared/xstream38.rs"]
mod xstream38;

#[cfg(feature = "gpu")]
mod cli {
    use super::gen_slots;
    use super::q3place::{self, PlaceQ3};
    use super::q38place;
    use super::taps;
    use super::xstream38::{Stage38, xstream38};
    use app::Session;
    use app::arch::qwen3moe::Q38Cfg;
    use app::mtp::{MtpBody, MtpDraft, WindowDraft};
    use bloomery_gpu::arch::qwen3moe::router::MAX_TOKENS;
    use bloomery_gpu::arch::qwen3moe::ubatch::{ImageWrite, ubatch_for, ubatch_size};
    use bloomery_gpu::arch::qwen3moe::{
        Body, Body35, Body38, KvQ8, Open35, PrefillPath, PrefillPlan, PrefillStep, Prompt38,
        Qwen35moeModel, Qwen38Model,
    };
    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::route_trace::{RouteTrace, TraceHeader};
    use bloomery_gpu::host::swap::{CallPick, CallReport, PassReport, Residency};
    use bloomery_gpu::host::xstream::{XLayer, XReport};
    use bloomery_gpu::hybrid::HybridStats;
    use bloomery_gpu::model::{ChainBody, MAX_PASS_ROWS, SlotRows, StepMode};
    use bloomery_gpu::{Gpu, GpuModel, Qwen3moeModel};
    use bloomery_gpu_gates::generate::{Place, card_words};
    use bloomery_gpu_gates::host_stats::{Probe, print_stats};
    use bloomery_gpu_gates::nodes::count_kinds;
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::residency38::{CARD38, Lever38, residency38};
    use bloomery_gpu_gates::{Fnv1a64, GateError, gpu_census, ref_model_path};
    use bloomery_levers::{
        Draft38At, Draft38Off, Levers, Residency38At, ResidencyPick, ResidencyWhy, draft38_unset,
        residency38_unset,
    };
    use cuda_core::sys;
    use gguf::Split;
    use model::arch::Arch;
    use model::arch::qwen35moe::head_list::head_rows_of;
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, PlanInputs, machine_bp_on, machine_for_experts, serve_ctx, tier_batch,
    };
    use model::placement::workstation::HostRead;
    use model::placement::{Device, Machine, Plan, PlanLevers};
    use refset::arch::qwen4exp::VERIFIED_POSITIONS;
    use refset::arch::qwen4exp::mtp::draft_file;
    use runtime::width::{Choosing, Chosen as _, Mode as WidthMode};
    use runtime::{Advance, Committed, PassSink, Speculative, Stop, Target};
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
    use tokenizer::Tokenizer;

    // A Qwen3.6 prompt is cut by the qwen3moe pass plan (`PrefillPlan`),
    // whose passes are `MAX_TOKENS` long; each must be a pass `step_rows`
    // takes.
    const _: () = assert!(MAX_TOKENS == MAX_PASS_ROWS);

    // A Qwen3.8 pass plan is printed by the same cut; its passes are
    // `Prompt38::PASS_ROWS` long.
    const _: () = assert!(Prompt38::PASS_ROWS == MAX_TOKENS);

    // `BLOOMERY_GEN_SLOTS` runs a row a slot, so its most is a pass's rows.
    const _: () = assert!(bloomery_levers::GEN_SLOTS_MAX == MAX_PASS_ROWS as u64);

    /// The last value of flag `name`, if given.
    fn flag(name: &str) -> Result<Option<String>, GateError> {
        Ok(flags(name)?.pop())
    }

    /// Every value of flag `name`, in order.
    fn flags(name: &str) -> Result<Vec<String>, GateError> {
        let args: Vec<String> = std::env::args().collect();
        let mut out = Vec::new();
        for (i, a) in args.iter().enumerate() {
            if a == name {
                out.push(
                    args.get(i + 1)
                        .ok_or_else(|| format!("{name} needs a value"))?
                        .clone(),
                );
            }
        }
        Ok(out)
    }

    /// Comma-separated ids.
    fn ids_of(s: &str) -> Result<Vec<u32>, GateError> {
        Ok(s.split(',')
            .map(|v| v.trim().parse::<u32>())
            .collect::<Result<_, _>>()?)
    }

    /// The engines this CLI drives, one a file architecture.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Family {
        Qwen3,
        Qwen35,
        /// `qwen4exp`, which the qwen35moe reader reads and `Arch` does not
        /// name.
        Qwen38,
    }

    /// The model file ([`ref_model_path`]) and its engine, read from its
    /// first shard's header: `qwen4exp` by its name, else `Arch::detect`'s
    /// qwen3moe or qwen35moe; any other file is refused by name.
    fn open_file() -> Result<(Split, Family), GateError> {
        let path = ref_model_path()?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() == Some("qwen4exp") {
            return Ok((file, Family::Qwen38));
        }
        let first = file
            .shard(0)
            .ok_or_else(|| format!("{} opened with no shard", path.display()))?;
        match Arch::detect(first)? {
            Arch::Qwen3moe => Ok((file, Family::Qwen3)),
            Arch::Qwen35moe => Ok((file, Family::Qwen35)),
            other => Err(format!(
                "{} is a {} file; generate_qwen3moe runs qwen3moe, qwen35moe and qwen4exp files",
                path.display(),
                file.architecture().unwrap_or(other.name())
            )
            .into()),
        }
    }

    /// `--prefill` for the qwen3moe and qwen35moe bodies: `auto` (the
    /// default), `pass` or `gemm`.
    fn prefill_path(arg: Option<&str>) -> Result<PrefillPath, GateError> {
        match arg {
            None | Some("auto") => Ok(PrefillPath::Auto),
            Some("pass") => Ok(PrefillPath::Pass),
            Some("gemm") => Ok(PrefillPath::Gemm),
            Some(o) => Err(format!("--prefill is auto, pass or gemm, not {o}").into()),
        }
    }

    /// The units a prompt runs as, as the `step 0` and `time prompt` lines
    /// print them.
    struct Units {
        /// `plan=`.
        text: String,
        /// `prefill_steps=` and `passes=`.
        count: usize,
        /// `kind=`.
        kind: &'static str,
        /// The ubatches' ids.
        ubatch_tokens: usize,
    }

    impl From<&PrefillPlan> for Units {
        fn from(plan: &PrefillPlan) -> Units {
            Units {
                text: plan.to_string(),
                count: plan.steps.len(),
                kind: plan.kind(),
                ubatch_tokens: plan.ubatch_tokens(),
            }
        }
    }

    /// A body's prompt schedule as this CLI drives it, beside the
    /// skeleton's own step.
    trait Prompted: ChainBody {
        /// What `--prefill` selects.
        type Path: Copy;
        /// The path the body runs for `--prefill <arg>` (`None` when not
        /// given); a path it has not is refused by name, before the load.
        fn path(arg: Option<&str>) -> Result<Self::Path, GateError>;
        /// The units a prompt of `n` ids runs as by `path`.
        fn plan(m: &GpuModel<Self>, n: usize, path: Self::Path) -> Result<Units, GateError>;
        /// Run `ids` by that plan from the model's position; the argmax
        /// after the last.
        fn prefill(m: &mut GpuModel<Self>, ids: &[u32], path: Self::Path)
        -> Result<u32, GateError>;
        /// The last prompt image a ubatch of `units` wrote; `None` when none
        /// did.
        fn image(m: &GpuModel<Self>, units: &Units) -> Result<Option<ImageWrite>, GateError>;
        /// Fill the first `rows` cache rows with the body's synthetic
        /// pattern (`--seed-depth`).
        fn seed(m: &mut GpuModel<Self>, rows: usize) -> Result<(), GateError>;
        /// One `BLOOMERY_STEP_STATS` probe of the body's host tier — its
        /// counters since load, the go waits of the services since the probe
        /// before, the process's page faults and the card's free device
        /// bytes (`host_stats::Probe::read`); `None` for a body with no
        /// host tier.
        fn host_probe(
            _: &GpuModel<Self>,
            _prev: Option<&Probe>,
        ) -> Result<Option<Probe>, GateError> {
            Ok(None)
        }

        /// Mark a prompt call of `n` ids from cache position `pos` on the
        /// host tier's route trace (`Hybrid::route_prompt`), before the ids
        /// run; nothing for a body with no host tier.
        fn mark_prompt(_m: &mut GpuModel<Self>, _pos: u32, _n: usize) -> Result<(), GateError> {
            Ok(())
        }

        /// Take the route trace the body's host tier holds, if one, finish
        /// it and return its directory and the positions it wrote; `None`
        /// with no trace.
        fn finish_trace(_m: &mut GpuModel<Self>) -> Result<Option<(PathBuf, u64)>, GateError> {
            Ok(None)
        }

        /// The residency boundaries' reports since the last take, each with
        /// the kind of the pass it ended; none for a body with no residency
        /// machine.
        fn residency_passes(
            _m: &mut GpuModel<Self>,
        ) -> Result<Vec<(PassKind, PassReport)>, GateError> {
            Ok(Vec::new())
        }

        /// The last prompt call's streaming picks, each with its ubatch, and
        /// its end; none for a body that does not stream.
        fn stream_records(_m: &mut GpuModel<Self>) -> Result<StreamRecords, GateError> {
            Ok((Vec::new(), None, Vec::new(), None))
        }

        /// What the body refuses of the load `m` before a run of `slots`
        /// streams in one pass (`BLOOMERY_GEN_SLOTS`), by name; nothing for a
        /// body whose own pass names its refusals.
        fn slots_load(_m: &GpuModel<Self>, _slots: usize) -> Result<(), GateError> {
            Ok(())
        }

        /// Before a parked slot's prompt in graph mode, outside its wall, the
        /// captures that prompt would otherwise take inside it; nothing for
        /// a body whose prompt captures nothing a slot holds.
        fn capture_slot_prompt(_m: &mut GpuModel<Self>) -> Result<(), GateError> {
            Ok(())
        }
    }

    impl Prompted for Body {
        type Path = PrefillPath;

        fn path(arg: Option<&str>) -> Result<PrefillPath, GateError> {
            prefill_path(arg)
        }

        fn plan(m: &Qwen3moeModel, n: usize, path: PrefillPath) -> Result<Units, GateError> {
            Ok(Units::from(&m.prefill_plan(n, path)?))
        }

        fn prefill(
            m: &mut Qwen3moeModel,
            ids: &[u32],
            path: PrefillPath,
        ) -> Result<u32, GateError> {
            Ok(m.prefill_with(ids, path)?)
        }

        fn image(m: &Qwen3moeModel, _: &Units) -> Result<Option<ImageWrite>, GateError> {
            Ok(m.ubatch_prologue()?.last)
        }

        fn seed(m: &mut Qwen3moeModel, rows: usize) -> Result<(), GateError> {
            Ok(m.seed_depth(rows)?)
        }

        /// A placed load (the unplaced default's fallback; `--place` is
        /// refused before the load), refused by name: its host tier's ports
        /// serve one sequence ([`Body`]'s `whole_card`).
        fn slots_load(m: &Qwen3moeModel, slots: usize) -> Result<(), GateError> {
            if m.body("generate_qwen3moe")?.placed().is_some() {
                return Err(format!(
                    "BLOOMERY_GEN_SLOTS={slots}: the load is placed (its plan line names why), \
                     and a pass of several slots runs a whole-card load: a placed chain's host \
                     tier serves one sequence"
                )
                .into());
            }
            Ok(())
        }

        /// A slot's captured prefill passes travel with its sequence, so a
        /// parked slot holds none: captured here, as the load captures slot
        /// 0's.
        fn capture_slot_prompt(m: &mut Qwen3moeModel) -> Result<(), GateError> {
            m.capture_prefill()?;
            Ok(())
        }
    }

    /// A Qwen3.6 prompt's plan by `path`: `pass` the captured passes,
    /// `auto` and `gemm` the prompt call's (the seat's `wide` with them, a
    /// caller that names it).
    fn plan35(m: &Qwen35moeModel, n: usize, path: PrefillPath) -> Result<PrefillPlan, GateError> {
        match path {
            PrefillPath::Pass => Ok(PrefillPlan::new(n, path, NonZeroUsize::MIN)?),
            PrefillPath::Auto | PrefillPath::Gemm | PrefillPath::Wide => {
                Ok(m.prefill_plan(n, path)?)
            }
        }
    }

    impl Prompted for Body35 {
        type Path = PrefillPath;

        /// Every path: `pass` the captured passes (`step_rows`), `auto` and
        /// `gemm` the prompt call.
        fn path(arg: Option<&str>) -> Result<PrefillPath, GateError> {
            prefill_path(arg)
        }

        fn plan(m: &Qwen35moeModel, n: usize, path: PrefillPath) -> Result<Units, GateError> {
            Ok(Units::from(&plan35(m, n, path)?))
        }

        /// `pass`: each pass of the plan as one `step_rows` of its size, a
        /// pass of one id as a step, bit for bit one step per token; `auto`
        /// and `gemm`, and every path of a placed load: the prompt call.
        fn prefill(
            m: &mut Qwen35moeModel,
            ids: &[u32],
            path: PrefillPath,
        ) -> Result<u32, GateError> {
            if path != PrefillPath::Pass || m.body("generate_qwen3moe")?.placed().is_some() {
                return Ok(m.prefill_with(ids, path)?);
            }
            let plan = plan35(m, ids.len(), path)?;
            let (mut at, mut next) = (0usize, None);
            for step in &plan.steps {
                let PrefillStep::Pass(k) = *step else {
                    return Err(format!("a qwen35moe pass plan holds a ubatch ({plan})").into());
                };
                let rows = ids
                    .get(at..at + k)
                    .ok_or_else(|| format!("the plan {plan} runs past {} ids", ids.len()))?;
                at += k;
                next = Some(pass35(m, rows)?);
            }
            next.ok_or_else(|| "the prompt has no ids".into())
        }

        /// The prompt call's image when a ubatch of `units` ran, its
        /// `tokens` the ubatches' ids.
        fn image(m: &Qwen35moeModel, units: &Units) -> Result<Option<ImageWrite>, GateError> {
            let ub = units.ubatch_tokens;
            Ok(m.prompt_image()?
                .filter(|_| ub > 0)
                .map(|w| ImageWrite { tokens: ub, ..w }))
        }

        fn seed(m: &mut Qwen35moeModel, rows: usize) -> Result<(), GateError> {
            Ok(m.seed_depth(rows)?)
        }

        /// A placed load (the unplaced default's fallback; `--place` is
        /// refused before the load), refused by name before the run: its
        /// host tier's batch port and its placed chain run one sequence
        /// each, and neither runs a pass of several slots
        /// ([`SlotRows::plan_slots`] names the same refusal past the slots'
        /// prompts).
        fn slots_load(m: &Qwen35moeModel, slots: usize) -> Result<(), GateError> {
            if m.body("generate_qwen3moe")?.placed().is_some() {
                return Err(format!(
                    "BLOOMERY_GEN_SLOTS={slots}: the load is placed (its plan line names why), \
                     and a pass of several slots runs a whole-card load: a placed load's prompt \
                     runs through the host tier's batch port and its step through the placed \
                     chain, one sequence each"
                )
                .into());
            }
            Ok(())
        }

        /// A parked slot's graph cache starts empty ([`GpuModel::add_slots`])
        /// and its `--prefill pass` prompt replays the row graphs of its
        /// passes' sizes, which [`GpuModel::step_rows`] captures inside its
        /// call — inside the prompt's wall: captured here for the selected
        /// slot, the step and every pass size, the set the load captures
        /// for slot 0's cache. The `auto` and `gemm` paths' prompt call is
        /// eager in either mode and replays nothing.
        fn capture_slot_prompt(m: &mut Qwen35moeModel) -> Result<(), GateError> {
            m.capture_step()?;
            capture35_rows(m)?;
            Ok(())
        }
    }

    /// The qwen4exp prompt call's own lines (the `Prompted` impl's and the
    /// drafted path's): the host tier's batch services over the call, and
    /// under `BLOOMERY_STEP_STATS` the ubatch walk's records.
    fn after38_prompt(m: &mut Qwen38Model, before: HybridStats) -> Result<(), GateError> {
        let after = m.body("prefill")?.hybrid().stats();
        let served = after.batch_served - before.batch_served;
        let ns = after.batch_ns - before.batch_ns;
        // No service, no time a service: `-`, never a plausible 0.
        let per_service = if served == 0 {
            "-".to_string()
        } else {
            format!("{:.4}", ns as f64 / 1e6 / served as f64)
        };
        println!(
            "stat prompt host services={served} cols={} host_slots={} union_ms={:.3} \
         per_service_ms={per_service}",
            after.batch_cols - before.batch_cols,
            after.batch_host_slots - before.batch_host_slots,
            ns as f64 / 1e6,
        );
        if let Some(s) = m.take_prompt38_stats()? {
            for r in record::prompt_stats(&s) {
                r.print();
            }
        }
        Ok(())
    }

    impl Prompted for Body38 {
        type Path = Prompt38;

        /// `auto` (the default), `gemm`, `pass` or `step`
        /// (`Prompt38::parse`).
        fn path(arg: Option<&str>) -> Result<Prompt38, GateError> {
            Ok(arg.map_or(Ok(Prompt38::Auto), Prompt38::parse)?)
        }

        /// `step:1x<n>` — one captured step a position — the pass cut
        /// `pass:<sizes>`, or the ubatch cut `ubatch:<sizes>`; the kind is
        /// the path `auto` resolves to on this load (`Body38::resolve_prompt`).
        fn plan(m: &Qwen38Model, n: usize, path: Prompt38) -> Result<Units, GateError> {
            let path = m.body("plan")?.resolve_prompt(path, n);
            Ok(match path {
                Prompt38::Gemm => {
                    let plan = PrefillPlan {
                        steps: m
                            .body("plan")?
                            .ubatch_cut(n)
                            .map(PrefillStep::Ubatch)
                            .collect(),
                    };
                    Units {
                        kind: path.name(),
                        ..Units::from(&plan)
                    }
                }
                Prompt38::Step => Units {
                    text: if n == 1 {
                        "step:1".to_string()
                    } else {
                        format!("step:1x{n}")
                    },
                    count: n,
                    kind: path.name(),
                    ubatch_tokens: 0,
                },
                Prompt38::Pass => {
                    let plan = PrefillPlan::new(n, PrefillPath::Pass, NonZeroUsize::MIN)?;
                    Units {
                        kind: path.name(),
                        ..Units::from(&plan)
                    }
                }
                Prompt38::Auto => {
                    return Err(format!(
                        "prompt path auto at {n} positions: `Prompt38::resolve` returned `auto`"
                    )
                    .into());
                }
            })
        }

        /// The prompt by `path`, then a `stat prompt host` line: the host
        /// tier's batch services the prompt took, their columns and host
        /// slots, and the union calls' wall — the host term of a pass's or a
        /// ubatch's layer. With the ubatch walk's timing armed
        /// (`BLOOMERY_STEP_STATS`), the prompt's `stat prompt` records
        /// follow: one `stat prompt lb` a layer-batch the ubatches served,
        /// then the `stat prompt split` over them.
        fn prefill(m: &mut Qwen38Model, ids: &[u32], path: Prompt38) -> Result<u32, GateError> {
            let before = m.body("prefill")?.hybrid().stats();
            let next = m.prompt38(ids, path)?;
            after38_prompt(m, before)?;
            Ok(next)
        }

        /// The Qwen3.8 walks write no prompt image.
        fn image(_: &Qwen38Model, _: &Units) -> Result<Option<ImageWrite>, GateError> {
            Ok(None)
        }

        fn seed(_: &mut Qwen38Model, rows: usize) -> Result<(), GateError> {
            Err(no_seed38(rows + 1))
        }

        fn host_probe(m: &Qwen38Model, prev: Option<&Probe>) -> Result<Option<Probe>, GateError> {
            Ok(Some(probe38(m, prev)?))
        }

        /// The trace's `call` row: the arm's prompt ids one step each from
        /// `pos` ([`Hybrid::route_prompt`]).
        fn mark_prompt(m: &mut Qwen38Model, pos: u32, n: usize) -> Result<(), GateError> {
            Ok(m.body_parts("generate_qwen3moe")?
                .2
                .hybrid_mut()
                .route_prompt(pos, n)?)
        }

        /// [`Hybrid::take_route_trace`], the set sealed by
        /// [`RouteTrace::finish`].
        fn finish_trace(m: &mut Qwen38Model) -> Result<Option<(PathBuf, u64)>, GateError> {
            let Some(t) = m
                .body_parts("generate_qwen3moe")?
                .2
                .hybrid_mut()
                .take_route_trace()
            else {
                return Ok(None);
            };
            let dir = t.dir().to_path_buf();
            Ok(Some((dir, t.finish()?)))
        }

        /// [`Body38::take_residency_passes`]: empty unless the load runs the
        /// machine and [`log38`] asked for the log.
        fn residency_passes(m: &mut Qwen38Model) -> Result<Vec<(PassKind, PassReport)>, GateError> {
            Ok(m.body_parts("generate_qwen3moe")?.2.take_residency_passes())
        }

        /// [`Body38::take_stream_records`] and
        /// [`Body38::take_xstream_records`].
        fn stream_records(m: &mut Qwen38Model) -> Result<StreamRecords, GateError> {
            let body = m.body_parts("generate_qwen3moe")?.2;
            let (picks, end) = body.take_stream_records();
            let (layers, xend) = body.take_xstream_records();
            Ok((picks, end, layers, xend))
        }
    }

    /// The refusal of `--seed-depth D` on a qwen4exp file.
    fn no_seed38(depth: usize) -> GateError {
        format!(
            "--seed-depth {depth}: qwen4exp has no synthetic depth (its delta states and selection \
             pools are not a pattern a seed can stand for)"
        )
        .into()
    }

    /// `--place` on a qwen4exp file, by the common unset rule (`Place::choose`
    /// by `q38place::Q38_RULE`) on one census reading, its `place unset`
    /// record printed: a set word as given, `a`, `gate` or `bp` (plan (b′): the
    /// next-largest card as the largest's expert tier; a card list spelled as
    /// one of them is it, any other list has no Qwen3.8 plan and is refused by
    /// name), else the cards' offer — kept when its plan holds the rule's
    /// break-even experts on the tier card or more and more than none. The
    /// plan the rule asks for is the load this run would open at the offer
    /// (the draft's MTP plan when the run drafts — the offer's stage is plan
    /// (a)'s, the rule never offering the gate card, so the draft's rule
    /// reads the offer's own decision — else the plain plan of the run's
    /// slot count) at the run's context; a plan that refuses runs `a`, the
    /// refusal named on stderr, and the load plans the chosen placement
    /// again (planning is milliseconds).
    fn place38(
        arg: Option<&str>,
        (file, levers, experts): (&Split, &Levers, Experts),
        (ctx, logits, slots): (usize, bool, usize),
        arms: &[Arm],
    ) -> Result<Place, GateError> {
        let flag = match arg {
            None => None,
            Some(word) => {
                let place = Place::parse(word)?;
                if ![Place::A, Place::Gate, Place::Bp].contains(&place) {
                    return Err(format!(
                        "--place {word}: a qwen4exp plan is placed a, gate or bp (plan (b′): the \
                         next-largest card as the largest's expert tier); a card list that is \
                         none of them has no Qwen3.8 plan"
                    )
                    .into());
                }
                Some(place)
            }
        };
        let chosen = q38place::choose(flag, &gpu_census::census()?, |offer| {
            let inputs = PlanInputs::describe(file)?;
            let (draft, _) = draft38(levers, offer, logits, arms, ctx)?;
            let mtp = match draft {
                Draft38::Off => None,
                Draft38::Mtp => {
                    let head = head_rows_of(levers.mtp_head_rows(), file, inputs.spec.vocab)?;
                    let (draft_path, from) = draft_file(levers.mtp_draft(), &ref_model_path()?);
                    let draft_split = Split::open(&draft_path).map_err(|e| {
                        format!(
                            "open the MTP draft {} ({}): {e}",
                            draft_path.display(),
                            from.describe()
                        )
                    })?;
                    Some(MtpInputs::read(&draft_split, file, &inputs, head.rows)?)
                }
            };
            let specs = offer.card_specs()?;
            q38place::tier_experts(
                &inputs,
                (specs[0], specs[1]),
                (u64::try_from(ctx)?, u64::try_from(ubatch_for(ctx)?)?),
                experts,
                &PlanLevers::from_levers(levers)?,
                mtp.as_ref(),
                if mtp.is_some() { 1 } else { slots },
            )
        })?;
        chosen.record().print();
        Ok(chosen.place)
    }

    /// The placement holds an expert tier card: plan (b′).
    fn tiered(place: Place) -> bool {
        !place.tier_cards().is_empty()
    }

    /// The stage card is the A6000: the placement the Qwen3.8 defaults
    /// (residency, the MTP draft, host streaming) treat as plan (a); the gate
    /// plan's stage is the 3090.
    fn stage_a(place: Place) -> bool {
        place != Place::Gate
    }

    /// The stage as the unset `BLOOMERY_XSTREAM` rule reads it.
    fn stage38(place: Place) -> Stage38 {
        if tiered(place) {
            Stage38::Tiered
        } else if place == Place::Gate {
            Stage38::Other
        } else {
            Stage38::A
        }
    }

    /// The machine a plan of `inputs` at ubatches of `ub` positions under
    /// `experts` runs on, over `place`'s cards; `draft` the MTP draft's card
    /// bytes when the draft runs beside the target, which plan (b′) reserves on
    /// its stage card (`place::machine_bp_on`) and the one-card plans count in
    /// `plan_mtp_with` instead.
    fn machine38(
        place: Place,
        inputs: &PlanInputs,
        ub: u64,
        experts: Experts,
        draft: Option<u64>,
    ) -> Result<Machine, GateError> {
        let layers = inputs.spec.layers.len();
        Ok(match place.card_specs()?.as_slice() {
            &[stage, tier] => {
                machine_bp_on((stage, tier), layers, ub, draft, tier_batch(&inputs.hp, ub))
            }
            &[stage] => machine_for_experts(stage, layers, ub, experts),
            cards => {
                return Err(format!(
                    "--place {}: a qwen4exp plan takes a stage card and at most one tier card, \
                     not {} cards",
                    place.name(),
                    cards.len()
                )
                .into());
            }
        })
    }

    /// The `plan` record of `plan` at `place` under `experts`: the stage
    /// card's experts, the tier card's when the plan has one, the stage
    /// card's free bytes when the census read them, the PLE table's tier and
    /// the reading of the host's room that chose it (`read`), and each plan
    /// card's device (`record::plan_devices`). A plan whose PLE tier is not
    /// one the reader takes is refused by name (`Plan::row_tier`).
    fn plan38_line(
        place: Place,
        experts: Experts,
        plan: &Plan<'_>,
        read: &HostRead,
    ) -> Result<String, GateError> {
        let r = Record::new(&record::PLAN38)
            .w("place", place.name())
            .w("card", place.card_specs()?[0].name)
            .w("experts", experts_name(experts))
            .u("ctx_max", plan.ctx_max)
            .u("host_experts", plan.host.experts)
            .u("card_experts", plan.cards[0].experts);
        let r = match (plan.machine.tiers.first(), plan.tier_n_l.first()) {
            (Some(t), Some(n)) => r
                .w("tier", &t.name)
                .u("tier_experts", n.iter().sum::<u64>()),
            _ => r,
        };
        let r = match plan.machine.cards.first().and_then(|c| c.free_bytes) {
            Some(free) => r.u("card_free", free),
            None => r,
        };
        let r = match plan.row_tier()? {
            Some(Device::Host) => r.w("rows", "host"),
            Some(Device::Nvme) => r.w("rows", "nvme"),
            Some(other) => return Err(format!("the PLE table's tier {other:?}").into()),
            None => r,
        };
        Ok(r.w("read", read.word())
            .csv("devices", record::plan_devices(plan.machine))
            .w("cuda_order", record::cuda_order())
            .line())
    }

    /// One pass of `rows` ids on the Qwen3.6 model: the argmax after the last.
    fn pass35(m: &mut Qwen35moeModel, rows: &[u32]) -> Result<u32, GateError> {
        fn last<const M: usize>(m: &mut Qwen35moeModel, rows: &[u32]) -> Result<u32, GateError> {
            let rows: [u32; M] = rows.try_into()?;
            Ok(m.step_rows::<M>(rows)?[M - 1])
        }
        match rows.len() {
            1 => Ok(m.step(rows)?),
            2 => last::<2>(m, rows),
            3 => last::<3>(m, rows),
            4 => last::<4>(m, rows),
            5 => last::<5>(m, rows),
            6 => last::<6>(m, rows),
            7 => last::<7>(m, rows),
            8 => last::<8>(m, rows),
            k => Err(format!("a pass of {k} ids; a Qwen3.6 pass takes 1..={MAX_PASS_ROWS}").into()),
        }
    }

    /// What every arm of the run shares.
    struct Run {
        timed: bool,
        warm: usize,
        mode: StepMode,
        seed_depth: Option<usize>,
        logits: bool,
        tok: Option<Tokenizer>,
        /// `--ctx`, as the `SMOKE` line prints it.
        ctx: usize,
        /// `BLOOMERY_STEP_STATS`, as the levers hold it.
        stats: bool,
        /// `BLOOMERY_MTP_WINDOWS`, as the levers hold it.
        windows: bool,
        /// `BLOOMERY_MTP_WIDTH`, as the levers hold it: the width chooser's
        /// mode (`cost`) or the fixed window (`fixed`).
        width: WidthMode,
        /// `--last-step`: the prompt's last id is a step of its own.
        last_step: bool,
        /// `BLOOMERY_GEN_SLOTS`: the streams an arm decodes in one pass.
        slots: usize,
    }

    /// A prompt call's streaming picks, each with its ubatch, and its end;
    /// its streamed layers, each with its ubatch, and the stream's end.
    type StreamRecords = (
        Vec<(usize, CallPick)>,
        Option<CallReport>,
        Vec<(usize, XLayer)>,
        Option<XReport>,
    );

    /// One arm: its ids and its generated count.
    struct Arm {
        ids: Vec<u32>,
        n_gen: usize,
    }

    pub fn run() -> Result<(), GateError> {
        // A lever set to a value it does not take, or a retired name that is
        // set, is refused by name before anything loads.
        // The placement's levers act on a run that names `--place`; set on
        // one that does not, they are refused by name.
        let mut acts_on = vec![
            bloomery_levers::STEP_STATS,
            bloomery_levers::QWEN38_EXPERTS,
            bloomery_levers::QWEN3_KV,
            bloomery_levers::ROUTE_TRACE,
            bloomery_levers::DRAFT,
            bloomery_levers::MTP_HEAD_ROWS,
            bloomery_levers::MTP_DRAFT,
            bloomery_levers::MTP_WINDOWS,
            bloomery_levers::MTP_WIDTH,
            bloomery_levers::RESIDENCY,
            bloomery_levers::HOSTSTREAM,
            bloomery_levers::XSTREAM,
            bloomery_levers::GEN_SLOTS,
        ];
        if std::env::args().any(|a| a == "--place") {
            acts_on.extend(q3place::PLACED_LEVERS);
        }
        let levers = bloomery_levers::at_main(&acts_on)?;
        record::at_main("generate_qwen3moe", record::GENERATE_QWEN3MOE);
        // Set, the word runs as given; unset, the Qwen3.8 rule picks it once
        // the file, the flags and plan (a) are known (`residency unset`).
        if let Some(word) = levers.residency() {
            record::residency_lever(ResidencyPick {
                word,
                why: ResidencyWhy::Set,
            })
            .print();
        }
        let word_set = levers.residency().unwrap_or("off");
        let residency = Residency::parse(word_set)?;
        let experts = experts38(&levers)?;
        // The K/V planes' format, read once here: the loads below carry it
        // and the plan's KV term counts it (the lever's kind holds the word
        // to the two spellings).
        let kv = KvQ8::parse(levers.qwen3_kv_set().unwrap_or("f16"))
            .ok_or("BLOOMERY_QWEN3_KV takes f16 or q8_0")?;
        // Unset, the lever is a qwen4exp plan's card experts and nothing on
        // another family's file; only a set `card` is refused there.
        let card_set = levers.qwen38_experts_set() == Some("card");
        if let Some(dir) = flag("--dump-taps")? {
            if card_set {
                return Err(
                    "BLOOMERY_QWEN38_EXPERTS=card places a qwen4exp plan's routed experts; \
                     --dump-taps runs a qwen3moe file"
                        .into(),
                );
            }
            if levers.route_trace().is_some() {
                return Err(
                    "BLOOMERY_ROUTE_TRACE records a qwen4exp host tier's steps; --dump-taps runs \
                     a qwen3moe file"
                        .into(),
                );
            }
            if levers.mtp_draft().is_some() {
                return Err(
                    "BLOOMERY_MTP_DRAFT names a qwen4exp file's MTP draft; --dump-taps runs a \
                     qwen3moe file"
                        .into(),
                );
            }
            if residency != Residency::Off {
                return Err(format!(
                    "BLOOMERY_RESIDENCY={word_set} moves a qwen4exp plan's card experts; \
                     --dump-taps runs a qwen3moe file"
                )
                .into());
            }
            if levers.gen_slots() > 1 {
                return Err(format!(
                    "BLOOMERY_GEN_SLOTS={} decodes several streams in one pass; --dump-taps runs \
                     each prompt one eager step per id",
                    levers.gen_slots()
                )
                .into());
            }
            residency_unset_early(
                &levers,
                Residency38At {
                    qwen38_file: false,
                    dump_taps: true,
                    place_a: false,
                    route_trace: false,
                    prefill_step: false,
                },
            );
            return dump_taps(&dir, &levers);
        }
        let timed = std::env::args().any(|a| a == "--time");
        let sync = std::env::args().any(|a| a == "--arm-sync");
        let logits = std::env::args().any(|a| a == "--logits");
        let last_step = std::env::args().any(|a| a == "--last-step");
        let n_gen: usize = flag("-n")?.map_or(Ok(32), |s| s.parse())?;
        let ctx: usize = flag("--ctx")?.map_or(Ok(4096), |s| s.parse())?;
        let warm: usize = flag("--warm")?.map_or(Ok(0), |s| s.parse())?;
        let mode = match flag("--mode")?.as_deref() {
            None | Some("graph") => StepMode::Graph,
            Some("eager") => StepMode::Eager,
            Some(o) => return Err(format!("--mode is eager or graph, not {o}").into()),
        };
        let prefill = flag("--prefill")?;
        let place = flag("--place")?;
        let seed_depth: Option<usize> = flag("--seed-depth")?.map(|s| s.parse()).transpose()?;
        let text = flag("--prompt")?;
        let tokens = flag("--tokens")?;
        let arm_specs = flags("--arm")?;
        let sources = usize::from(text.is_some())
            + usize::from(tokens.is_some())
            + usize::from(seed_depth.is_some())
            + usize::from(!arm_specs.is_empty());
        if sources != 1 {
            return Err("give exactly one of --prompt, --tokens, --seed-depth, --arm".into());
        }
        if sync && arm_specs.is_empty() {
            return Err("--arm-sync paces the arms of an --arm list, and none is given".into());
        }
        let tok = match &text {
            Some(_) => Some(Tokenizer::from_gguf(ref_model_path()?)?),
            None => None,
        };
        let t = Instant::now();
        let (file, family) = open_file()?;
        let prefill = prefill.as_deref();
        let arms: Vec<Arm> = if arm_specs.is_empty() {
            let ids: Vec<u32> = match (&text, &tokens, seed_depth) {
                (Some(t), _, _) => tok.as_ref().ok_or("no tokenizer")?.encode(t, true, false),
                (_, Some(s), _) => ids_of(s)?,
                (_, _, Some(_)) => vec![0],
                _ => unreachable!("one source by the check above"),
            };
            vec![Arm { ids, n_gen }]
        } else {
            arm_specs
                .iter()
                .map(|spec| {
                    let (ids, n) = match spec.split_once('/') {
                        Some((ids, n)) => (ids, n.parse::<usize>()?),
                        None => (spec.as_str(), n_gen),
                    };
                    Ok(Arm {
                        ids: ids_of(ids)?,
                        n_gen: n,
                    })
                })
                .collect::<Result<_, GateError>>()?
        };
        let slots = levers.gen_slots();
        if slots > 1 {
            slots_refused(
                slots,
                family,
                &[
                    ("--prompt", text.is_some()),
                    ("--seed-depth", seed_depth.is_some()),
                    ("--last-step", last_step),
                    ("--logits", logits),
                ],
                &[
                    ("--place", place.is_some()),
                    ("BLOOMERY_STEP_STATS=1", levers.step_stats()),
                    ("BLOOMERY_ROUTE_TRACE", levers.route_trace().is_some()),
                ],
                &arms,
            )?;
        }
        for arm in &arms {
            if arm.ids.is_empty() {
                return Err("the prompt has no ids".into());
            }
            if arm.n_gen == 0 || (timed && arm.n_gen <= warm + 1) {
                return Err(
                    format!("-n {} leaves no counted step (warm {warm})", arm.n_gen).into(),
                );
            }
            // Under several slots each slot holds one window of the ids.
            let depth = seed_depth.unwrap_or(arm.ids.len() / slots);
            if depth + arm.n_gen > ctx {
                return Err(
                    format!("depth {depth} + {} tokens pass --ctx {ctx}", arm.n_gen).into(),
                );
            }
        }
        let (chosen, draft_off) = match family {
            Family::Qwen3 => {
                draft_refused_on_other(&levers, family)?;
                let place = place_q3(place.as_deref(), prefill)?;
                (Chosen::Qwen3(Body::path(prefill)?, place), None)
            }
            Family::Qwen35 => {
                draft_refused_on_other(&levers, family)?;
                let place = place_q3(place.as_deref(), prefill)?;
                (Chosen::Qwen35(Body35::path(prefill)?, place), None)
            }
            Family::Qwen38 => {
                if let Some(d) = seed_depth {
                    return Err(no_seed38(d));
                }
                let place = place38(
                    place.as_deref(),
                    (&file, &levers, experts),
                    (ctx, logits, slots),
                    &arms,
                )?;
                let (draft, off) = draft38(&levers, place, logits, &arms, ctx)?;
                if slots > 1 && draft == Draft38::Mtp {
                    return Err(format!(
                        "BLOOMERY_GEN_SLOTS={slots} decodes several streams in one plain pass, \
                         and this run drafts (BLOOMERY_DRAFT=mtp, or unset under --place a with \
                         the draft file there): the MTP draft verifies one sequence's window; \
                         set BLOOMERY_DRAFT=off"
                    )
                    .into());
                }
                (Chosen::Qwen38(Body38::path(prefill)?, place, draft), off)
            }
        };
        let draft = match chosen {
            Chosen::Qwen38(_, _, d) => d,
            Chosen::Qwen3(..) | Chosen::Qwen35(..) => Draft38::Off,
        };
        if last_step && (family == Family::Qwen38 || timed || seed_depth.is_some()) {
            return Err(
                "--last-step feeds a qwen3moe or qwen35moe prompt's last id as a step, the \
                 server's cut; it is refused on a qwen4exp file and beside --time and \
                 --seed-depth"
                    .into(),
            );
        }
        if family != Family::Qwen38 && card_set {
            return Err(
                "BLOOMERY_QWEN38_EXPERTS=card places a qwen4exp plan's routed experts; a \
                 qwen3moe or qwen35moe file has no host tier"
                    .into(),
            );
        }
        if family == Family::Qwen38 && kv == KvQ8::Q8 {
            return Err(
                "BLOOMERY_QWEN3_KV=q8_0 quantizes the qwen3 family's K/V planes; a qwen4exp \
                 file's selecting stores carry no q8_0 form"
                    .into(),
            );
        }
        if family != Family::Qwen38 && levers.route_trace().is_some() {
            return Err(
                "BLOOMERY_ROUTE_TRACE records a qwen4exp host tier's routing; a qwen3moe or \
                 qwen35moe plan holds every one on the card"
                    .into(),
            );
        }
        if family != Family::Qwen38 && levers.hoststream().is_some() {
            return Err(
                "BLOOMERY_HOSTSTREAM streams a V4.1 prompt's host experts into its residency \
                 pool; a qwen3moe or qwen35moe plan holds every one on the card"
                    .into(),
            );
        }
        if family != Family::Qwen38 && levers.xstream().is_some() {
            return Err(
                "BLOOMERY_XSTREAM moves a qwen4exp prompt's host experts to the card; a qwen3moe \
                 or qwen35moe plan holds every one on the card"
                    .into(),
            );
        }
        if family != Family::Qwen38 && residency != Residency::Off {
            return Err(format!(
                "BLOOMERY_RESIDENCY={word_set} moves a qwen4exp plan's card experts; a qwen3moe \
                 or qwen35moe plan holds every one on the card"
            )
            .into());
        }
        // The trace is the input the residency model replays under a fixed
        // seed; under the machine its slot files would record the machine's
        // own moves.
        if residency != Residency::Off && levers.route_trace().is_some() {
            return Err(format!(
                "BLOOMERY_ROUTE_TRACE records a fixed placement's routing; \
                 BLOOMERY_RESIDENCY={word_set} moves the slot map under it"
            )
            .into());
        }
        // Why the run drafts nothing, as a refusal of a draft lever names it.
        let no_draft = |need: &str| match &draft_off {
            Some(why) => format!("the run drafts nothing ({why})"),
            None => format!("it needs {need}"),
        };
        if family == Family::Qwen38 && draft == Draft38::Off && levers.mtp_head_rows().is_some() {
            return Err(format!(
                "BLOOMERY_MTP_HEAD_ROWS picks the MTP draft's head; {}",
                no_draft("BLOOMERY_DRAFT=mtp on a qwen4exp file")
            )
            .into());
        }
        if draft == Draft38::Off && levers.mtp_draft().is_some() {
            return Err(format!(
                "BLOOMERY_MTP_DRAFT names the MTP draft file; {}",
                no_draft("BLOOMERY_DRAFT=mtp on a qwen4exp file")
            )
            .into());
        }
        if draft == Draft38::Off && levers.mtp_windows() {
            return Err(format!(
                "BLOOMERY_MTP_WINDOWS prints the MTP draft's windows; {}",
                no_draft("BLOOMERY_DRAFT=mtp on a qwen4exp file")
            )
            .into());
        }
        if draft == Draft38::Off && levers.mtp_width().is_some() {
            return Err(format!(
                "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; {}",
                no_draft("BLOOMERY_DRAFT=mtp on a qwen4exp file")
            )
            .into());
        }
        let (place_a, prefill_step) = match chosen {
            Chosen::Qwen38(path, place, _) => (stage_a(place), path == Prompt38::Step),
            Chosen::Qwen3(..) | Chosen::Qwen35(..) => (false, false),
        };
        let at = Residency38At {
            qwen38_file: family == Family::Qwen38,
            dump_taps: false,
            place_a,
            route_trace: levers.route_trace().is_some(),
            prefill_step,
        };
        if family != Family::Qwen38 {
            residency_unset_early(&levers, at);
        }
        let lever38 = match levers.residency() {
            Some(_) => Lever38::Set(residency, word_set),
            None => Lever38::Unset(residency38_unset(at)),
        };
        if matches!(chosen, Chosen::Qwen35(PrefillPath::Pass, None))
            && logits
            && arms.iter().any(|a| a.n_gen == 1)
        {
            return Err(
                "--logits with -n 1 and --prefill pass on a qwen35moe file: the prompt's last \
                 pass leaves its logits in its last row's head, and --logits reads row 0's"
                    .into(),
            );
        }
        if matches!(chosen, Chosen::Qwen38(.., Draft38::Mtp)) && logits {
            return Err(
                "--logits with BLOOMERY_DRAFT=mtp: the drafted run's last call is a verify of \
                 several rows into one head, and --logits reads the step head's row"
                    .into(),
            );
        }
        let listed = !arm_specs.is_empty();
        if !listed {
            println!("prompt_ids {:?}", arms[0].ids);
        }
        let run = Run {
            timed,
            warm,
            mode,
            seed_depth,
            logits,
            tok,
            ctx,
            stats: levers.step_stats(),
            windows: levers.mtp_windows(),
            width: WidthMode::of(levers.mtp_width())?,
            last_step,
            slots,
        };
        match chosen {
            Chosen::Qwen3(path, place) => {
                let m = open_qwen3(
                    file,
                    (ctx, mode),
                    place.map(|p| (p, &levers)),
                    &levers,
                    kv,
                    t,
                )?;
                if slots > 1 {
                    drive_slots(m, &run, path, &arms, listed, sync)
                } else {
                    drive(m, &run, path, &arms, listed, sync)
                }
            }
            Chosen::Qwen35(path, place) => {
                let m = open_qwen35(
                    file,
                    (ctx, mode),
                    place.map(|p| (p, &levers)),
                    &levers,
                    kv,
                    t,
                )?;
                if slots > 1 {
                    drive_slots(m, &run, path, &arms, listed, sync)
                } else {
                    drive(m, &run, path, &arms, listed, sync)
                }
            }
            Chosen::Qwen38(path, place, draft) => match draft {
                Draft38::Off => {
                    let trace = trace38(
                        &levers,
                        &file,
                        (path, place, experts),
                        arms_chunk(&arms),
                        timed,
                    )?;
                    let (mut m, residency) = open_qwen38(
                        file,
                        &levers,
                        (ctx, mode, slots),
                        (path, place, experts),
                        (lever38, draft_off.as_ref()),
                        t,
                    )?;
                    log38(&mut m, residency, &arms, run.slots)?;
                    stream38(&mut m, &levers, place, residency)?;
                    if let Some(t) = trace {
                        m.body_parts("generate_qwen3moe")?
                            .2
                            .hybrid_mut()
                            .attach_route_trace(t)?;
                    }
                    m.set_prompt38_stats(run.stats)?;
                    if slots > 1 {
                        drive_slots(m, &run, path, &arms, listed, sync)
                    } else {
                        drive(m, &run, path, &arms, listed, sync)
                    }
                }
                Draft38::Mtp => {
                    if levers.route_trace().is_some() {
                        return Err(
                            "BLOOMERY_ROUTE_TRACE records the plain run's routing, one step a \
                                    position; it is refused beside BLOOMERY_DRAFT=mtp"
                                .into(),
                        );
                    }
                    let (mut m, cfg, residency) = open_qwen38_mtp(
                        file,
                        &levers,
                        (ctx, mode),
                        (path, place, experts),
                        lever38,
                        t,
                    )?;
                    log38(&mut m, residency, &arms, run.slots)?;
                    stream38(&mut m, &levers, place, residency)?;
                    m.set_prompt38_stats(run.stats)?;
                    drive38_mtp(m, cfg, &run, &arms, listed, sync)
                }
            },
        }
    }

    /// The flags `--dump-taps` refuses by name: every other mode's.
    const NOT_WITH_DUMP: [&str; 11] = [
        "--prompt",
        "--tokens",
        "--seed-depth",
        "--arm",
        "--arm-sync",
        "--time",
        "--warm",
        "--logits",
        "--prefill",
        "--mode",
        "--place",
    ];

    /// Whether `BLOOMERY_DRAFT` drafts this run: `mtp` on a qwen4exp file;
    /// every other family and word is refused by name.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Draft38 {
        Off,
        Mtp,
    }

    /// `BLOOMERY_DRAFT` on a qwen4exp file run at `place` over `arms` with
    /// `--ctx` `ctx`: `mtp` the MTP draft, `off` the plain path; unset, the
    /// rule's (`bloomery_levers::draft38_unset`); and, drafting nothing, why
    /// (the `load draft=off` record's). The V4.1 words and any other are
    /// refused by name.
    fn draft38(
        levers: &Levers,
        place: Place,
        logits: bool,
        arms: &[Arm],
        ctx: usize,
    ) -> Result<(Draft38, Option<Draft38Off>), GateError> {
        match levers.draft() {
            Some("mtp") => Ok((Draft38::Mtp, None)),
            Some("off") => Ok((Draft38::Off, Some(Draft38Off::Set))),
            Some(other) => Err(format!(
                "BLOOMERY_DRAFT={other}: on a qwen4exp file mtp drafts the window and off runs \
                 the plain path; lookup and dspark are the V4.1 binaries'"
            )
            .into()),
            None => {
                let (file, _) = draft_file(levers.mtp_draft(), &ref_model_path()?);
                // The generation loop runs a window of ROWS rows only while
                // they fit (`runtime::Stop::check`): at `e` tokens emitted the
                // target stands at depth + e − 1, and the last check before
                // -n is at e = n − 1.
                let rows = <Body38 as MtpBody>::VERIFY_ROWS;
                let need = arms
                    .iter()
                    .map(|a| a.ids.len() + a.n_gen + rows - 2)
                    .max()
                    .unwrap_or(0);
                let at = Draft38At {
                    place_a: stage_a(place),
                    logits,
                    route_trace: levers.route_trace().is_some(),
                    file: &file,
                    file_is_there: file.is_file(),
                    need,
                    ctx,
                };
                Ok(match draft38_unset(&at) {
                    None => (Draft38::Mtp, None),
                    Some(off) => (Draft38::Off, Some(off)),
                })
            }
        }
    }

    /// Unset, the `residency unset` record of a run the rule decides with no
    /// plan (`--dump-taps`, another family's file); nothing when set.
    fn residency_unset_early(levers: &Levers, at: Residency38At) {
        if levers.residency().is_some() {
            return;
        }
        if let Some(pick) = residency38_unset(at) {
            record::residency_unset(&pick).print();
        }
    }

    /// `BLOOMERY_DRAFT` on the other families: no word of it drafts them.
    fn draft_refused_on_other(levers: &Levers, family: Family) -> Result<(), GateError> {
        match levers.draft() {
            None => Ok(()),
            Some(word) => Err(format!(
                "BLOOMERY_DRAFT={word}: only a qwen4exp file runs a draft (mtp); this is a {} \
                 file",
                match family {
                    Family::Qwen3 => "qwen3moe",
                    Family::Qwen35 => "qwen35moe",
                    Family::Qwen38 => "qwen4exp",
                }
            )
            .into()),
        }
    }

    /// `--dump-taps DIR`: the tap dump of every `--tokens-file`'s windows on
    /// a qwen3moe file, eager, then the manifest read back.
    fn dump_taps(dir: &str, levers: &Levers) -> Result<(), GateError> {
        let args: Vec<String> = std::env::args().collect();
        if let Some(f) = NOT_WITH_DUMP.iter().find(|f| args.iter().any(|a| a == *f)) {
            return Err(format!(
                "--dump-taps runs each prompt one eager step per id; {f} does not apply to it"
            )
            .into());
        }
        let files = flags("--tokens-file")?;
        if files.is_empty() {
            return Err("--dump-taps needs at least one --tokens-file".into());
        }
        let seqs: usize = flag("--seqs")?
            .ok_or("--dump-taps needs --seqs (prompts per --tokens-file)")?
            .parse()?;
        let len: usize = flag("--prompt-len")?
            .ok_or("--dump-taps needs --prompt-len")?
            .parse()?;
        let n_gen: usize = flag("-n")?.map_or(Ok(32), |s| s.parse())?;
        let ctx: usize = flag("--ctx")?.map_or(Ok(4096), |s| s.parse())?;
        if len + n_gen > ctx {
            return Err(format!("--prompt-len {len} + -n {n_gen} pass --ctx {ctx}").into());
        }
        let mut all = Vec::new();
        for f in &files {
            all.extend(taps::windows(Path::new(f), seqs, len)?);
        }
        let t = Instant::now();
        let (file, family) = open_file()?;
        if family != Family::Qwen3 {
            return Err(format!(
                "--dump-taps runs a qwen3moe file (the taps are its chain's), not a {}",
                file.architecture().unwrap_or("?")
            )
            .into());
        }
        let mut m = open_qwen3(file, (ctx, StepMode::Eager), None, levers, KvQ8::F16, t)?;
        let hidden = m.body("generate_qwen3moe")?.hparams().n_embd;
        let out = Path::new(dir);
        let mut dump = taps::Dump::create(out, &ref_model_path()?, hidden)?;
        let t = Instant::now();
        for s in &all {
            let t_seq = Instant::now();
            let row = dump.seq(&mut m, s, n_gen)?;
            let (ids_b, taps_b) = taps::sizes(row, hidden);
            println!(
                "{}",
                Record::new(&record::TAPS_SEQ)
                    .u("k", row.seq)
                    .w("source", s.source.display())
                    .u("offset", s.offset)
                    .u("n_prompt", row.n_prompt)
                    .u("n_total", row.n_total)
                    .u("bytes", ids_b + taps_b)
                    .f("wall_s", t_seq.elapsed().as_secs_f64())
                    .line()
            );
        }
        let rows = taps::read_manifest(out, hidden)?;
        if rows.len() != all.len() {
            return Err(format!(
                "{}: the manifest reads back {} rows of {} sequences",
                out.display(),
                rows.len(),
                all.len()
            )
            .into());
        }
        let bytes: u64 = rows
            .iter()
            .map(|r| {
                let (a, b) = taps::sizes(r, hidden);
                a + b
            })
            .sum();
        println!(
            "{}",
            Record::new(&record::TAPS_DUMP)
                .w("dir", out.display())
                .u("seqs", rows.len())
                .u("positions", rows.iter().map(|r| r.n_total).sum::<usize>())
                .u("bytes", bytes)
                .csv("layers", taps::TAPS)
                .w("prefill", "step")
                .f("wall_s", t.elapsed().as_secs_f64())
                .line()
        );
        Ok(())
    }

    /// The engine and its `--prefill` path (and, for qwen4exp, its card and
    /// draft), chosen before the load; a qwen4exp plan's expert rule is
    /// `BLOOMERY_QWEN38_EXPERTS`'s.
    #[derive(Clone, Copy)]
    enum Chosen {
        Qwen3(PrefillPath, Option<Place>),
        Qwen35(PrefillPath, Option<Place>),
        Qwen38(Prompt38, Place, Draft38),
    }

    /// `--place` on a qwen3moe or qwen35moe file: the placement word
    /// (`generate::Place`), refused by name beside `--prefill gemm` — a
    /// placed prompt runs as passes through the host tier's batch port. The
    /// common unset rule (`Place::choose` by `q3place::Q3_RULE`) prints its
    /// `place unset` record, set or unset; the flag goes on as given, so an
    /// unset one loads the census pick (`q3place::open_unplaced_qwen3`).
    fn place_q3(arg: Option<&str>, prefill: Option<&str>) -> Result<Option<Place>, GateError> {
        let flag = match arg {
            None => None,
            Some(word) => {
                if prefill == Some("gemm") {
                    return Err(format!(
                        "--prefill gemm beside --place {word}: a placed qwen3moe or qwen35moe \
                         prompt runs as passes through the host tier (auto or pass)"
                    )
                    .into());
                }
                Some(Place::parse(word)?)
            }
        };
        q3place::choose(flag, &gpu_census::census()?)?
            .record()
            .print();
        Ok(flag)
    }

    /// The Qwen3-30B-A3B model of `file` — under `place` by its plan
    /// (`q3place`), the `plan` line first, and with `place` unset the
    /// default ([`q3place::open_unplaced_qwen3`]): the whole-card load on
    /// `a`'s card while that fits the card's free bytes, else the placed
    /// plan on that card with its why — its `load` line, and in graph mode
    /// the step captured and, unplaced, every prefill pass, with their
    /// `capture` lines.
    fn open_qwen3(
        file: Split,
        (ctx, mode): (usize, StepMode),
        place: Option<(Place, &Levers)>,
        levers: &Levers,
        kv: KvQ8,
        t: Instant,
    ) -> Result<Qwen3moeModel, GateError> {
        let opts = Qwen3moeModel::lever_opts(ctx, kv)?;
        let mut m = match place {
            None => q3place::open_unplaced_qwen3(file, ctx, opts, levers, Record::print)?,
            Some((p, levers)) => {
                let q = PlaceQ3::qwen3(&file, p, ctx, kv)?;
                let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
                q.record(&plan, None).print();
                q3place::open_qwen3(file, &plan, opts, levers.host())?
            }
        };
        let placed = m.body("generate_qwen3moe")?.placed().is_some();
        m.set_mode(mode);
        println!(
            "load arch=qwen3moe resident_bytes={} ctx={ctx} cache={} layers={} mode={} \
             flash_mma={} ubatch_attn=gqa_prefill_flash ubatch={} rope_table_us={:.1} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            kv.name(),
            m.layers().len(),
            mode_name(mode),
            m.body("generate_qwen3moe")?.flash_mma(),
            m.ubatch()?,
            m.ubatch_prologue()?.table_build.as_secs_f64() * 1e6,
            t.elapsed().as_secs_f64()
        );
        if mode == StepMode::Graph {
            println!("capture graph_nodes={}", m.capture_step()?);
            if placed {
                return Ok(m);
            }
            let (free0, _) = m.gpu().mem_info()?;
            let t = Instant::now();
            let nodes = m.capture_prefill()?;
            capture_line(&nodes, t, free0, m.gpu())?;
        }
        Ok(m)
    }

    /// The row graphs of the pass sizes 2 to [`MAX_PASS_ROWS`], captured
    /// for the selected slot, each's node count — the graphs a `--prefill
    /// pass` pass of its size replays, which [`GpuModel::step_rows`]
    /// captures inside its call when the slot's cache holds one not. The
    /// load captures them for slot 0's `capture` line; a parked slot's
    /// prompt captures its own outside its wall.
    fn capture35_rows(m: &mut Qwen35moeModel) -> Result<Vec<usize>, GateError> {
        Ok(vec![
            m.capture_rows::<2>()?,
            m.capture_rows::<3>()?,
            m.capture_rows::<4>()?,
            m.capture_rows::<5>()?,
            m.capture_rows::<6>()?,
            m.capture_rows::<7>()?,
            m.capture_rows::<8>()?,
        ])
    }

    /// The Qwen3.6-35B-A3B model of `file` with the tensor-core decode
    /// flash (the engine's) — under `place` by its plan (`q3place`), the
    /// `plan` line first, and with `place` unset the default
    /// ([`q3place::open_unplaced_qwen35`]): the whole-card load on `a`'s
    /// card while that fits the card's free bytes, else the placed plan on
    /// that card with its why — its `load` line, and in graph
    /// mode the step captured and, unplaced, the passes of 2 to
    /// [`MAX_PASS_ROWS`] rows, with their `capture` lines.
    fn open_qwen35(
        file: Split,
        (ctx, mode): (usize, StepMode),
        place: Option<(Place, &Levers)>,
        levers: &Levers,
        kv: KvQ8,
        t: Instant,
    ) -> Result<Qwen35moeModel, GateError> {
        let o = Open35 {
            ctx,
            mma: true,
            ubatch: ubatch_size()?,
            kv,
        };
        let mut m = match place {
            None => q3place::open_unplaced_qwen35(file, o, levers, Record::print)?,
            Some((p, levers)) => {
                let q = PlaceQ3::qwen35(&file, p, o)?;
                let plan = q.plan(ctx, &PlanLevers::from_levers(levers)?)?;
                q.record(&plan, None).print();
                q3place::open_qwen35(file, &plan, o, levers.host())?
            }
        };
        let placed = m.body("generate_qwen3moe")?.placed().is_some();
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen35moe resident_bytes={} ctx={ctx} cache={} layers={} mode={} \
             flash_mma={} store_bytes={} ubatch_attn=gqa_prefill_flash_256 ubatch={} in {:.1} s \
             (runtime value)",
            m.resident_bytes(),
            kv.name(),
            m.layers().len(),
            mode_name(mode),
            body.flash_mma(),
            body.store_bytes(),
            m.ubatch()?,
            t.elapsed().as_secs_f64()
        );
        if mode == StepMode::Graph {
            let step = m.capture_step()?;
            println!("capture graph_nodes={step}");
            if placed {
                return Ok(m);
            }
            let (free0, _) = m.gpu().mem_info()?;
            let t = Instant::now();
            let mut nodes = vec![step];
            nodes.extend(capture35_rows(&mut m)?);
            capture_line(&nodes, t, free0, m.gpu())?;
        }
        Ok(m)
    }

    /// `BLOOMERY_QWEN38_EXPERTS` as a qwen4exp plan's expert rule: as set,
    /// else the card experts.
    fn experts38(levers: &Levers) -> Result<Experts, GateError> {
        match levers.qwen38_experts() {
            "host" => Ok(Experts::Host),
            "card" => Ok(Experts::Card),
            other => Err(format!("BLOOMERY_QWEN38_EXPERTS={other}: host or card").into()),
        }
    }

    /// The name the `plan` line prints for `experts`.
    fn experts_name(experts: Experts) -> &'static str {
        match experts {
            Experts::Host => "host",
            Experts::Card => "card",
        }
    }

    /// The positions every arm of the run writes — its prompt's ids one
    /// step each, then its generated tokens less the first, which the
    /// prompt's last step already answered — when they all write the same
    /// count, for the trace's `chunk` header line; `None` when they differ
    /// (the line names one context shape, so a run of unequal arms writes
    /// none and the replay tool refuses the set's unknown contexts by name).
    fn arms_chunk(arms: &[Arm]) -> Option<usize> {
        let of = |a: &Arm| a.ids.len() + a.n_gen - 1;
        let first = of(arms.first()?);
        arms.iter().all(|a| of(a) == first).then_some(first)
    }

    /// The route trace `BLOOMERY_ROUTE_TRACE` asks for on a qwen4exp file,
    /// its directory made here, before the load: every position's routed
    /// ids per layer and the slot each ran in
    /// (`crates/gpu/src/host/route_trace.rs`), the prompt's ids one step
    /// each recorded as the call's positions. Refused by name under a
    /// prompt path but `step` — a pass runs a multi-row service and a
    /// ubatch a prompt batch, and the trace records one-row steps — and
    /// beside `--time`: the trace rewrites its manifest after every
    /// position, so a timed run's numbers would not be a measurement.
    fn trace38(
        levers: &Levers,
        file: &Split,
        (path, place, experts): (Prompt38, Place, Experts),
        chunk: Option<usize>,
        timed: bool,
    ) -> Result<Option<RouteTrace>, GateError> {
        let Some(dir) = levers.route_trace() else {
            return Ok(None);
        };
        if path != Prompt38::Step {
            return Err(
                "BLOOMERY_ROUTE_TRACE records one-row steps: pass --prefill step (a pass runs a \
                 multi-row service and a ubatch a prompt batch, which the trace does not record)"
                    .into(),
            );
        }
        if timed {
            return Err(
                "BLOOMERY_ROUTE_TRACE rewrites its manifest after every position, so --time \
                 beside it is not a measurement: run the trace without --time"
                    .into(),
            );
        }
        let inputs = PlanInputs::describe(file)?;
        let hp = &inputs.hp;
        let mut extra = vec![
            ("place".to_owned(), place.name().to_owned()),
            ("experts".to_owned(), experts_name(experts).to_owned()),
            ("prefill".to_owned(), path.name().to_owned()),
        ];
        if let Some(n) = chunk {
            extra.push(("chunk".to_owned(), n.to_string()));
        }
        let header = TraceHeader {
            model: ref_model_path()?,
            arch: "qwen4exp".to_owned(),
            build: "generate_qwen3moe".to_owned(),
            n_expert: hp.n_expert,
            n_used: hp.n_used,
            first_layer: 0,
            n_layer: inputs.spec.layers.len(),
            extra,
        };
        Ok(Some(RouteTrace::create(dir, header)?))
    }

    /// The cards `m` loaded at `place`, its `load` line's `cards=` field:
    /// `generate::card_words` of the placement's cards (the stage card, then
    /// the expert tier card; a device that is not the placement's refused by
    /// name), as a record's csv field writes them.
    fn cards38(m: &Qwen38Model, place: Place) -> Result<String, GateError> {
        let specs = place.card_specs()?;
        let planned: Vec<&str> = specs.iter().map(|s| s.name).collect();
        let tiers = m.body("generate_qwen3moe")?.hybrid().tiers();
        let words = card_words(place.name(), &planned, Some(&specs), m.gpu(), tiers)?;
        Ok(format!("[{}]", words.join(",")))
    }

    /// The Qwen3.8-Flash-Next model of `file`, placed by its plan on the
    /// card `place` names, its routed experts where `experts` says, its plan
    /// counting `slots` resident sequences (`BLOOMERY_GEN_SLOTS`; one is the
    /// one-sequence plan), under
    /// the residency `lever` resolves over the plan ([`residency38`]): the
    /// `plan` line, the `residency unset` line when the lever is unset, under
    /// `mid` the `residency host` line, the `load` line, `load draft=off`
    /// with why when `draft_off` names one, and in graph mode the step
    /// captured and its `capture` line, its node kinds held to the program's
    /// count. Returns the residency the load runs.
    fn open_qwen38(
        file: Split,
        levers: &Levers,
        (ctx, mode, slots): (usize, StepMode, usize),
        (path, place, experts): (Prompt38, Place, Experts),
        (lever, draft_off): (Lever38<'static>, Option<&Draft38Off>),
        t: Instant,
    ) -> Result<(Qwen38Model, Residency), GateError> {
        let inputs = PlanInputs::describe(&file)?;
        let cap = serve_ctx(u64::try_from(ctx)?, &inputs.hp)?;
        let ub = ubatch_for(ctx)?;
        let machine = machine38(place, &inputs, u64::try_from(ub)?, experts, None)?;
        // One slot is `plan_with` itself.
        let plan = inputs.plan_with_slots(
            &machine,
            u64::try_from(ctx)?,
            &PlanLevers::from_levers(levers)?,
            experts,
            slots,
        )?;
        println!("{}", plan38_line(place, experts, &plan, &inputs.room.1)?);
        let residency = residency38(&plan, lever, Record::print)?;
        let mut m = Body38::open_placed_residency(
            file,
            &plan,
            &inputs,
            CARD38,
            levers.host(),
            ub,
            residency,
            slots,
        )?;
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen4exp resident_bytes={} ctx={ctx} ctx_max={cap} ctx_train={} \
             verified={VERIFIED_POSITIONS} layers={} mode={} store_bytes={} prefill={} ubatch={} \
             place={} cards={} card_layers={} card_stacks={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            inputs.hp.n_ctx_train,
            m.layers().len(),
            mode_name(mode),
            body.store_bytes(),
            path.name(),
            body.ubatch_rows(),
            place.name(),
            cards38(&m, place)?,
            body.card_layers(),
            body.card_stacks(),
            t.elapsed().as_secs_f64()
        );
        if let Some(why) = draft_off {
            Record::new(&record::LOAD_DRAFT_OFF38).w("why", why).print();
        }
        capture38_check(&mut m)?;
        Ok((m, residency))
    }

    /// The Qwen3.8-Flash-Next model of `file` with its MTP draft loaded
    /// beside it (`Body38::open_placed_mtp_residency`, the draft file `draft_file`
    /// picks, its head `head_list::head_rows_of`'s), the residency
    /// `lever` resolves over the MTP plan: the `plan`, `residency unset`,
    /// `residency host`, `load`, `load draft=mtp` and `capture` lines of the
    /// plain open, the draft's own
    /// resident bytes, its program's arena and its head named on the draft's
    /// line. In graph mode the verify passes of 2 to 4 rows are captured by
    /// the session's `with_draft`, whose log prints each width's nodes.
    fn open_qwen38_mtp(
        file: Split,
        levers: &Levers,
        (ctx, mode): (usize, StepMode),
        (path, place, experts): (Prompt38, Place, Experts),
        lever: Lever38<'static>,
        t: Instant,
    ) -> Result<(Qwen38Model, Q38Cfg, Residency), GateError> {
        let inputs = PlanInputs::describe(&file)?;
        let cap = serve_ctx(u64::try_from(ctx)?, &inputs.hp)?;
        let ub = ubatch_for(ctx)?;
        let head = head_rows_of(levers.mtp_head_rows(), &file, inputs.spec.vocab)?;
        let rows = head.rows.clone();
        let (draft_path, from) = draft_file(levers.mtp_draft(), &ref_model_path()?);
        let draft_split = Split::open(&draft_path).map_err(|e| {
            format!(
                "open the MTP draft {} ({}): {e}",
                draft_path.display(),
                from.describe()
            )
        })?;
        let mtp = MtpInputs::read(&draft_split, &file, &inputs, rows)?;
        let ctx_max = u64::try_from(ctx)?;
        let reserve = if tiered(place) {
            Some(mtp.card_bytes(ctx_max)?)
        } else {
            None
        };
        let machine = machine38(place, &inputs, u64::try_from(ub)?, experts, reserve)?;
        let plan = inputs.plan_mtp_with(
            &machine,
            ctx_max,
            &PlanLevers::from_levers(levers)?,
            &mtp,
            experts,
        )?;
        println!(
            "{}",
            plan38_line(place, experts, &plan.plan, &inputs.room.1)?
        );
        let residency = residency38(&plan.plan, lever, Record::print)?;
        let mut m = Body38::open_placed_mtp_residency(
            file,
            &plan,
            &inputs,
            CARD38,
            levers.host(),
            ub,
            &draft_split,
            &mtp,
            residency,
            1,
        )?;
        m.set_mode(mode);
        let body = m.body("generate_qwen3moe")?;
        println!(
            "load arch=qwen4exp resident_bytes={} ctx={ctx} ctx_max={cap} ctx_train={} \
             verified={VERIFIED_POSITIONS} layers={} mode={} store_bytes={} prefill={} ubatch={} \
             place={} cards={} card_layers={} card_stacks={} in {:.1} s (runtime value)",
            m.resident_bytes(),
            inputs.hp.n_ctx_train,
            m.layers().len(),
            mode_name(mode),
            body.store_bytes(),
            path.name(),
            body.ubatch_rows(),
            place.name(),
            cards38(&m, place)?,
            body.card_layers(),
            body.card_stacks(),
            t.elapsed().as_secs_f64()
        );
        let draft = body.mtp().ok_or("the load opened no MTP draft")?;
        let head_line = match draft.head_map() {
            Some((_, n)) => format!("rows={n}"),
            None => "full".to_string(),
        };
        println!(
            "load draft=mtp resident={} arena={} head={head_line} card_bytes={} in {:.1} s \
             (runtime value)",
            draft.resident_bytes(),
            draft.arena_bytes(),
            plan.draft_card_bytes() + plan.arena_bytes,
            t.elapsed().as_secs_f64()
        );
        Record::new(&record::MTP_HEAD38)
            .w("head", head.head_word())
            .u("rows", head.rows_of(inputs.spec.vocab))
            .w("from", head.why.from_word())
            .w("why", &head.why)
            .print();
        capture38_check(&mut m)?;
        Ok((
            m,
            Q38Cfg {
                prompt: path,
                draft: mode,
            },
            residency,
        ))
    }

    /// Under `mid`, keep the residency boundaries' reports for the `residency
    /// pass` records each arm prints after its lines; nothing under `off`.
    fn log38(
        m: &mut Qwen38Model,
        residency: Residency,
        arms: &[Arm],
        slots: usize,
    ) -> Result<(), GateError> {
        if residency == Residency::Off {
            return Ok(());
        }
        // An arm's boundaries: its slots' prompt calls, then at most one a
        // generated token (a step, a pass of its slots, or a verify keeping
        // at least one).
        let passes = arms.iter().map(|a| a.n_gen).max().unwrap_or(0) + slots;
        m.body_parts("generate_qwen3moe")?.2.log_residency(passes);
        Ok(())
    }

    /// `BLOOMERY_XSTREAM` on the load and its `xstream=` line, by the rule
    /// the Qwen3.8 serve seat shares ([`xstream38`]); `BLOOMERY_HOSTSTREAM`
    /// set here is refused by name.
    fn stream38(
        m: &mut Qwen38Model,
        levers: &Levers,
        place: Place,
        residency: Residency,
    ) -> Result<(), GateError> {
        if levers.hoststream().is_some() {
            return Err(
                "BLOOMERY_HOSTSTREAM is V4.1's lever; a qwen4exp load reads BLOOMERY_XSTREAM \
                 (off, admit or split)"
                    .into(),
            );
        }
        let (gpu, _, body) = m.body_parts("generate_qwen3moe")?;
        let line = xstream38(gpu, body, levers.xstream(), stage38(place), residency)?;
        println!("{line}");
        Ok(())
    }

    /// In graph mode: capture the decode step and hold its node kinds to the
    /// program's count, a mismatch ending the run by name.
    fn capture38_check(m: &mut Qwen38Model) -> Result<(), GateError> {
        if m.mode() != StepMode::Graph {
            return Ok(());
        }
        let (launches, memops) = m.body("generate_qwen3moe")?.step_launches();
        let nodes = m.capture_step()?;
        let kernel = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_KERNEL;
        let memop = sys::CUgraphNodeType_enum_CU_GRAPH_NODE_TYPE_BATCH_MEM_OP;
        let ([k, b], other) = count_kinds(&m.step_graph_nodes()?, [kernel, memop]);
        println!(
            "capture graph_nodes={nodes} kernel={k} batch_mem_op={b} other={other} (the \
             program counts {launches}, {memops} of them batch_mem_op)"
        );
        if nodes != launches || b != memops || k + b != nodes || other != 0 {
            return Err(format!(
                "the captured step is not the program's: {nodes} nodes ({k} kernel, {b} \
                 batch_mem_op, {other} other) against {launches} launches, {memops} of them \
                 batch_mem_op"
            )
            .into());
        }
        Ok(())
    }

    /// The `capture prefill_graphs=` line: each pass size's node count, the
    /// captures' wall since `t` and the card bytes they took since `free0`.
    fn capture_line(nodes: &[usize], t: Instant, free0: usize, gpu: &Gpu) -> Result<(), GateError> {
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (free1, _) = gpu.mem_info()?;
        let list: Vec<String> = nodes.iter().map(usize::to_string).collect();
        println!(
            "capture prefill_graphs={} nodes={} ms={ms:.1} vram_bytes={} (runtime values)",
            nodes.len(),
            list.join(","),
            free0.saturating_sub(free1)
        );
        Ok(())
    }

    /// Every arm of the run on the loaded model `m`, in a session over it;
    /// `listed` for an `--arm` list, whose arms open with their `arm` lines.
    /// A run whose model holds a route trace ends by sealing the set
    /// ([`Prompted::finish_trace`]) once every arm ran; a failed arm leaves
    /// the set without its `complete` line, as a killed run's is.
    fn drive<B: Prompted>(
        m: GpuModel<B>,
        run: &Run,
        path: B::Path,
        arms: &[Arm],
        listed: bool,
        sync: bool,
    ) -> Result<(), GateError> {
        let mut s = Session::from_model(m, u32::try_from(run.ctx)?);
        let count = arms.len();
        let ran = s.arms(arms, |s, i, arm| {
            if let Some(c) = s.take_cleared() {
                record::residency_reset(&c).print();
            }
            if listed {
                arm_head(i, count, arm, sync)?;
            }
            let ran = run_arm(s.model_mut(), run, path, arm);
            after_passes(s.model_mut(), ran)
        });
        ran.map_err(|f| Box::new(f) as GateError)?;
        if let Some((dir, rows)) = B::finish_trace(s.model_mut())? {
            println!("route trace {} positions={rows} complete", dir.display());
        }
        Ok(())
    }

    /// The verify pass's capture log: one line a width.
    struct VerifyCaptures;

    impl app::RowsLog for VerifyCaptures {
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), app::SessionError> {
            println!("capture verify rows={rows} nodes={nodes}");
            Ok(())
        }
    }

    /// Every arm on the loaded model through the MTP draft: the session over
    /// it, the draft opened beside it (the verify passes of 2 to 4 rows
    /// captured in graph mode, each width's nodes printed) behind the width
    /// chooser `BLOOMERY_MTP_WIDTH` names, each arm its prompt — the draft
    /// walked over its units — and its windows; each after the first from
    /// the session's clear, the draft started over.
    fn drive38_mtp(
        m: Qwen38Model,
        cfg: Q38Cfg,
        run: &Run,
        arms: &[Arm],
        listed: bool,
        sync: bool,
    ) -> Result<(), GateError> {
        let mut s = Session::from_model(m, u32::try_from(run.ctx)?);
        let path = cfg.prompt;
        let mut draft = MtpDraft::open(s.model(), path, cfg.draft)?;
        if run.windows {
            draft.keep_windows();
        }
        let mut spec = s.with_draft::<Choosing<MtpDraft<Body38>>, 4>(
            draft.choosing(run.width)?,
            &mut VerifyCaptures,
        )?;
        let count = arms.len();
        for (i, arm) in arms.iter().enumerate() {
            if i > 0 {
                s.clear()?;
                spec.draft_mut().draft_mut().restart();
                if let Some(c) = s.take_cleared() {
                    record::residency_reset(&c).print();
                }
            }
            if listed {
                arm_head(i, count, arm, sync)?;
            }
            let ran = run_arm38_mtp(&mut s, &mut spec, path, run, arm);
            after_passes(s.model_mut(), ran)?;
        }
        Ok(())
    }

    /// An arm's result `ran`, with the `residency pass` records of its
    /// boundaries printed after it ([`print_passes`]): a failed print is the
    /// arm's error, and when the arm failed too the error names both.
    fn after_passes<B: Prompted, T>(
        m: &mut GpuModel<B>,
        ran: Result<T, GateError>,
    ) -> Result<T, GateError> {
        match (ran, print_passes(m)) {
            (ran, Ok(())) => ran,
            (Ok(_), Err(p)) => Err(p),
            (Err(r), Err(p)) => {
                Err(format!("{r}; then printing the residency passes after it: {p}").into())
            }
        }
    }

    /// The `residency pass` records of the boundaries since the last print,
    /// after the arm's lines: nothing prints between two timed steps.
    fn print_passes<B: Prompted>(m: &mut GpuModel<B>) -> Result<(), GateError> {
        let (picks, end, layers, xend) = B::stream_records(m)?;
        for (ubatch, p) in &picks {
            record::call_stream(*ubatch, p).print();
        }
        for (ubatch, x) in &layers {
            record::xstream_layer(*ubatch, x).print();
        }
        if let Some(r) = end {
            record::call_report(&r).print();
        }
        if let Some(r) = xend {
            record::xstream_end(&r).print();
        }
        for (kind, r) in B::residency_passes(m)? {
            record::residency_pass_of(kind, &r).print();
        }
        Ok(())
    }

    /// What the drafted generation's sink keeps: every pass's outcome —
    /// whether it verified a proposal, the rows it kept, the rows it ran —
    /// and its wall, every kept token at its position, and the stats probes.
    struct Windows {
        stats: bool,
        emitted: Vec<(u32, u32)>,
        passes: Vec<(bool, usize, usize, f64, Duration)>,
        probes: Vec<Probe>,
        n_gen: usize,
    }

    impl PassSink<Session<Body38>> for Windows {
        type Error = GateError;

        fn begin(&mut self, t: &Session<Body38>) -> Result<(), GateError> {
            if self.stats {
                self.probes.reserve_exact(self.n_gen);
                self.probes.push(probe38(t.model(), None)?);
            }
            Ok(())
        }

        fn pass(
            &mut self,
            t: &Session<Body38>,
            c: &Committed,
            tokens: &[u32],
            wall: Duration,
        ) -> Result<(), GateError> {
            for (r, &tok) in (0u32..).zip(tokens) {
                self.emitted.push((c.pos + r, tok));
            }
            self.passes.push((
                c.proposed,
                c.kept,
                c.rows,
                wall.as_secs_f64() * 1e3 / c.kept as f64,
                wall,
            ));
            if self.stats {
                self.probes.push(probe38(t.model(), self.probes.last())?);
            }
            Ok(())
        }
    }

    /// One arm through the draft: the prompt — the draft walked over its
    /// units, its own lines as the plain run's — then the windows until `-n`
    /// tokens are out, every kept token the target's own argmax.
    fn run_arm38_mtp(
        s: &mut Session<Body38>,
        spec: &mut Speculative<Choosing<MtpDraft<Body38>>, 4>,
        path: Prompt38,
        run: &Run,
        arm: &Arm,
    ) -> Result<(), GateError> {
        let (ids, n_gen, warm) = (&arm.ids, arm.n_gen, run.warm);
        let plan = <Body38 as Prompted>::plan(s.model(), ids.len(), path)?;
        let t = Instant::now();
        let before = s.model().body("prefill")?.hybrid().stats();
        let next = spec.prompt(s, ids)?;
        after38_prompt(s.model_mut(), before)?;
        let prefill_wall = t.elapsed();
        println!(
            "step 0 {} {next} (the {} prompt ids in prefill_steps={} units, plan={}, {:.2} s, \
             runtime value)",
            s.pos() - 1,
            ids.len(),
            plan.count,
            plan.text,
            prefill_wall.as_secs_f64()
        );
        let mut sink = Windows {
            stats: run.stats,
            emitted: Vec::with_capacity(n_gen),
            passes: Vec::with_capacity(n_gen),
            probes: Vec::new(),
            n_gen,
        };
        let stop = Stop::new(n_gen, s.ctx())?;
        let out = runtime::generate(s, spec, ids, next, &stop, &mut sink)?;
        if out.tokens.len() < n_gen {
            return Err(format!(
                "generate_qwen3moe: the generation stopped at {} after {} tokens, before -n",
                out.stop.name(),
                out.tokens.len()
            )
            .into());
        }
        print_prompt(ids.len(), prefill_wall, &plan, None, "");
        let kept = &sink.emitted[..n_gen - 1];
        for (k, &(pos, tok)) in kept.iter().enumerate() {
            println!("step {} {pos} {tok}", k + 1);
        }
        if run.timed {
            // A pass's wall over its positions is the row a plain run's step
            // wall compares with: one a kept position.
            let mut at = 0usize;
            for (i, &(proposed, rows, _, per, wall)) in sink.passes.iter().enumerate() {
                Record::new(&record::TIME_PASS)
                    .u("i", i + 1)
                    .flag("warm", i < warm)
                    .f("ms", wall.as_secs_f64() * 1e3)
                    .u("positions", rows)
                    .w("kind", if proposed { "mtp" } else { "plain" })
                    .print();
                for _ in 0..rows {
                    if at >= n_gen - 1 {
                        break;
                    }
                    at += 1;
                    let tag = if at <= warm { " warm" } else { "" };
                    println!("time step {at}{tag} ms={per:.4}");
                }
            }
        }
        let tokens: Vec<u32> = std::iter::once(next)
            .chain(kept.iter().map(|&(_, t)| t))
            .collect();
        println!("tokens {tokens:?}");
        if let Some(t) = &run.tok {
            println!("text {:?}", t.decode(&tokens));
        }
        mtp_summary(&sink.passes, warm);
        if run.windows {
            mtp_windows(&sink.passes, &spec.draft_mut().draft_mut().take_windows())?;
        }
        if run.timed {
            let counted = &sink.passes[warm..];
            let positions: usize = counted.iter().map(|&(_, k, ..)| k).sum();
            let ms: f64 = counted
                .iter()
                .map(|&(_, _, _, _, w)| w.as_secs_f64() * 1e3)
                .sum();
            let mut per: Vec<f64> = counted.iter().map(|&(_, _, _, p, _)| p).collect();
            per.sort_by(f64::total_cmp);
            let p50 = per[per.len() / 2];
            let mean = ms / positions as f64;
            println!(
                "SMOKE mode={} prompt_tokens={} depth={} generated={n_gen} warm={warm} \
                 steps={positions} passes={} p50_ms={p50:.4} mean_ms={mean:.4} \
                 tok/s(p50)={:.2} tok/s(mean)={:.2} ctx={}",
                mode_name(run.mode),
                ids.len(),
                ids.len(),
                counted.len(),
                1e3 / p50,
                1e3 / mean,
                run.ctx
            );
        }
        print_stats(&sink.probes, warm);
        Ok(())
    }

    /// The `mtp summary` record: the windows' proposals, the kept lengths'
    /// and the verified widths' histograms, the positions and their rate
    /// over the counted passes.
    fn mtp_summary(passes: &[(bool, usize, usize, f64, Duration)], warm: usize) {
        let mut kept = [0u64; 4];
        let mut widths = [0u64; 4];
        let mut positions = 0usize;
        let mut proposals = 0usize;
        for &(p, k, rows, ..) in passes {
            if p {
                proposals += 1;
            }
            kept[k - 1] += 1;
            widths[rows - 1] += 1;
            positions += k;
        }
        let counted = &passes[warm..];
        let counted_positions: usize = counted.iter().map(|&(_, k, ..)| k).sum();
        let ms: f64 = counted
            .iter()
            .map(|&(_, _, _, _, w)| w.as_secs_f64() * 1e3)
            .sum();
        Record::new(&record::MTP_SUMMARY)
            .u("proposals", proposals)
            .list("kept", &kept)
            .list("widths", &widths)
            .u("positions", positions)
            .u("passes", passes.len())
            .f("tok/s(positions)", counted_positions as f64 * 1e3 / ms)
            .print();
    }

    /// The `mtp window` records: each drafted window beside its pass, which
    /// must keep one row past the ids the draft says the target kept and run
    /// one row past the ids it says the pass verified. A window the passes
    /// do not hold, or a pass the draft kept no window of, is refused by
    /// name.
    fn mtp_windows(
        passes: &[(bool, usize, usize, f64, Duration)],
        windows: &[WindowDraft],
    ) -> Result<(), GateError> {
        let drafted: Vec<(usize, usize, usize)> = passes
            .iter()
            .enumerate()
            .filter(|(_, p)| p.0)
            .map(|(i, p)| (i + 1, p.1, p.2))
            .collect();
        if drafted.len() != windows.len() {
            return Err(format!(
                "generate_qwen3moe: the draft kept {} windows of {} drafted passes",
                windows.len(),
                drafted.len()
            )
            .into());
        }
        for (&(pass, kept, rows), w) in drafted.iter().zip(windows) {
            if w.accepted + 1 != kept
                || w.ids.len() != w.p.len()
                || w.width + 1 != rows
                || w.width > w.ids.len()
            {
                return Err(format!(
                    "generate_qwen3moe: pass {pass} ran {rows} rows and kept {kept}, and its \
                     window verified {} of its {} ids ({} probabilities), the target keeping {}",
                    w.width,
                    w.ids.len(),
                    w.p.len(),
                    w.accepted
                )
                .into());
            }
        }
        for (&(pass, ..), w) in drafted.iter().zip(windows) {
            Record::new(&record::MTP_WINDOW)
                .u("window", pass)
                .u("pos", w.pos)
                .csv("ids", &w.ids)
                .csv("p", w.p.iter().map(|p| format!("{p:.5}")))
                .u("width", w.width)
                .u("accepted", w.accepted)
                .print();
        }
        Ok(())
    }

    fn mode_name(mode: StepMode) -> &'static str {
        if mode == StepMode::Graph {
            "graph"
        } else {
            "eager"
        }
    }

    /// One arm on the loaded model: its prefill, its steps, and every line
    /// from `step 0` on.
    fn run_arm<B: Prompted>(
        m: &mut GpuModel<B>,
        run: &Run,
        path: B::Path,
        arm: &Arm,
    ) -> Result<(), GateError> {
        let (ids, n_gen, warm) = (&arm.ids, arm.n_gen, run.warm);
        if let Some(d) = run.seed_depth.filter(|&d| d > 1) {
            B::seed(m, d - 1)?;
            println!("seed rows={} pos={}", d - 1, m.pos());
        }
        let depth = run.seed_depth.unwrap_or(ids.len());
        // Under `--last-step` the prompt call takes all but the last id.
        let (fed, last) = match ids.split_last() {
            Some((&l, head)) if run.last_step => (head, Some(l)),
            _ => (ids.as_slice(), None),
        };
        B::mark_prompt(m, m.pos(), ids.len())?;
        let plan = B::plan(m, fed.len().max(1), path)?;
        let t = Instant::now();
        let mut next = match (fed.is_empty(), last) {
            (false, None) => B::prefill(m, fed, path)?,
            (false, Some(l)) => {
                B::prefill(m, fed, path)?;
                m.step(&[l])?
            }
            (true, Some(l)) => m.step(&[l])?,
            (true, None) => return Err("the prompt has no ids".into()),
        };
        let prefill_wall = t.elapsed();
        let image = B::image(m, &plan)?;
        println!(
            "step 0 {} {next} (the {} prompt ids in prefill_steps={} units, plan={}, {:.2} s, \
             runtime value)",
            m.pos() - 1,
            ids.len(),
            plan.count,
            plan.text,
            prefill_wall.as_secs_f64()
        );
        let mut tokens_out = Vec::with_capacity(n_gen);
        tokens_out.push(next);
        // Every line of the loop is held and written after it: a write is a
        // syscall, and the steps it would separate are the measurement.
        let mut rows: Vec<(u32, u32, f64)> = Vec::with_capacity(n_gen - 1);
        let mut probes: Vec<Probe> = Vec::with_capacity(if run.stats { n_gen } else { 0 });
        if run.stats {
            probes.extend(B::host_probe(m, probes.last())?);
        }
        for _ in 1..n_gen {
            let t0 = Instant::now();
            next = m.step(&[next])?;
            rows.push((m.pos() - 1, next, t0.elapsed().as_secs_f64() * 1e3));
            if run.stats {
                probes.extend(B::host_probe(m, probes.last())?);
            }
        }
        print_prompt(ids.len(), prefill_wall, &plan, image, "");
        for (k, &(pos, tok, ms)) in rows.iter().enumerate() {
            let i = k + 1;
            tokens_out.push(tok);
            println!("step {i} {pos} {tok}");
            if run.timed {
                let tag = if i <= warm { " warm" } else { "" };
                println!("time step {i}{tag} ms={ms:.4}");
            }
        }
        println!("tokens {tokens_out:?}");
        if run.logits {
            let row = m.logits()?;
            let argmax = row
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.total_cmp(y.1).then(y.0.cmp(&x.0)))
                .map_or(0, |(i, _)| i);
            let fnv = Fnv1a64::default().f32s(&row).value();
            // The argmax's lead over the best other logit: how near a tie the
            // last token was.
            let top = row[argmax];
            let second = row
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != argmax)
                .map(|(_, &v)| v)
                .fold(f32::NEG_INFINITY, f32::max);
            println!(
                "logits n={} argmax={argmax} margin={:.4} fnv64={fnv:016x}",
                row.len(),
                top - second
            );
        }
        if let Some(t) = &run.tok {
            println!("text {:?}", t.decode(&tokens_out));
        }
        if run.timed {
            let counted: Vec<f64> = rows[warm..].iter().map(|r| r.2).collect();
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted.iter().sum::<f64>() / counted.len() as f64;
            println!(
                "SMOKE mode={} prompt_tokens={} depth={depth} seeded={} generated={n_gen} \
                 warm={warm} steps={} p50_ms={p50:.4} mean_ms={mean:.4} tok/s(p50)={:.2} ctx={}",
                mode_name(run.mode),
                ids.len(),
                run.seed_depth.is_some(),
                counted.len(),
                1e3 / p50,
                run.ctx
            );
        }
        print_stats(&probes, warm);
        Ok(())
    }

    /// A prompt's `time prompt` and `stat prompt` lines: `n` ids in `wall` by
    /// the units `plan`, and the image its last ubatch wrote; `tag` ends both
    /// (a slot's ` slot=<j>`, else nothing).
    fn print_prompt(n: usize, wall: Duration, plan: &Units, image: Option<ImageWrite>, tag: &str) {
        let ms = wall.as_secs_f64() * 1e3;
        println!(
            "time prompt n={n} ms={ms:.4} tok/s={:.2} passes={} kind={}{tag}",
            n as f64 * 1e3 / ms,
            plan.count,
            plan.kind
        );
        match image {
            Some(w) => println!(
                "stat prompt ubatch_tokens={} image_bytes={} fill_us={:.1} copy_us={:.1} \
                 (runtime values){tag}",
                w.tokens,
                w.bytes,
                w.fill.as_secs_f64() * 1e6,
                w.copy.as_secs_f64() * 1e6
            ),
            None => println!(
                "stat prompt ubatch_tokens={} (no prompt image){tag}",
                plan.ubatch_tokens
            ),
        }
    }

    /// `BLOOMERY_GEN_SLOTS=slots` beside what it does not run with, each
    /// refused by name: what every body refuses ([`gen_slots::refused`]:
    /// each flag of `given` that is set, and of the three after it the
    /// qwen3moe and qwen35moe files' `--place` — a placed load's host tier
    /// serves one sequence — or the qwen4exp file's step stats and route
    /// trace, which record one sequence's steps), and an arm whose ids
    /// `slots` does not cut into windows of one length
    /// ([`gen_slots::windows`]).
    fn slots_refused(
        slots: usize,
        family: Family,
        given: &[(&str, bool)],
        &[place, stats, trace]: &[(&str, bool); 3],
        arms: &[Arm],
    ) -> Result<(), GateError> {
        let mut beside = given.to_vec();
        match family {
            Family::Qwen3 => {
                beside.push(place);
                gen_slots::refused::<Body>(slots, "qwen3moe", &beside)?;
            }
            Family::Qwen35 => {
                beside.push(place);
                gen_slots::refused::<Body35>(slots, "qwen35moe", &beside)?;
            }
            Family::Qwen38 => {
                beside.extend([stats, trace]);
                gen_slots::refused::<Body38>(slots, "qwen4exp", &beside)?;
            }
        }
        for (i, arm) in arms.iter().enumerate() {
            gen_slots::windows(&arm.ids, slots, &format!("arm {i}'s ids"))?;
        }
        Ok(())
    }

    /// Every arm of the run as `run.slots` streams in one pass
    /// (`BLOOMERY_GEN_SLOTS`) on the model `m`, in a session over it that
    /// serves that many slots; `listed` for an `--arm` list, whose arms open
    /// with their `arm` lines. What the body refuses of the load first
    /// ([`Prompted::slots_load`]); each arm from every slot's reset and, past
    /// the first, the residency's seed ([`gen_slots::fresh`]), its record
    /// before the arm's; each arm's `residency pass` records after its lines.
    fn drive_slots<B: Prompted + SlotRows>(
        m: GpuModel<B>,
        run: &Run,
        path: B::Path,
        arms: &[Arm],
        listed: bool,
        sync: bool,
    ) -> Result<(), GateError>
    where
        B::Seq: 'static,
    {
        B::slots_load(&m, run.slots)?;
        let mut s = Session::from_model(m, u32::try_from(run.ctx)?);
        s.add_slots(run.slots)?;
        let count = arms.len();
        let ran = s.arms(arms, |s, i, arm| {
            if let Some(c) = gen_slots::fresh(s, run.slots, i > 0)? {
                record::residency_reset(&c).print();
            }
            if listed {
                arm_head(i, count, arm, sync)?;
            }
            let ran = run_arm_slots(s, run, path, arm);
            after_passes(s.model_mut(), ran)
        });
        ran.map_err(|f| Box::new(f) as GateError)?;
        Ok(())
    }

    /// One arm as `run.slots` streams in one pass, every slot from its reset
    /// and slot 0 selected ([`gen_slots::fresh`]): slot j prefilled with
    /// window j of the arm's ids, in graph mode the pass of a row a slot
    /// captured, then `-n` − 1 rounds, each slot fed its own argmax. Every
    /// slot's lines from its `step 0` on; the rounds' lines are held and
    /// written after them.
    fn run_arm_slots<B: Prompted + SlotRows>(
        s: &mut Session<B>,
        run: &Run,
        path: B::Path,
        arm: &Arm,
    ) -> Result<(), GateError>
    where
        B::Seq: 'static,
    {
        let (n, n_gen, warm) = (run.slots, arm.n_gen, run.warm);
        let windows = gen_slots::windows(&arm.ids, n, "the arm's ids")?;
        let mut pos0 = Vec::with_capacity(n);
        let mut next = Vec::with_capacity(n);
        let mut prompts = Vec::with_capacity(n);
        for (j, ids) in windows.iter().enumerate() {
            s.select_slot(j)?;
            let m = s.model_mut();
            if j > 0 && run.mode == StepMode::Graph {
                B::capture_slot_prompt(m)?;
            }
            let plan = B::plan(m, ids.len(), path)?;
            let t = Instant::now();
            let tok = B::prefill(m, ids, path)?;
            let wall = t.elapsed();
            let image = B::image(m, &plan)?;
            println!(
                "step 0 {} {tok} slot={j} (the {} prompt ids in prefill_steps={} units, plan={}, \
                 {:.2} s, runtime value)",
                m.pos() - 1,
                ids.len(),
                plan.count,
                plan.text,
                wall.as_secs_f64()
            );
            pos0.push(usize::try_from(m.pos())?);
            next.push(tok);
            prompts.push((ids.len(), wall, plan, image));
        }
        s.select_slot(0)?;
        if run.mode == StepMode::Graph && n_gen > 1 {
            let nodes = gen_slots::capture(s, n)?;
            println!("capture slots={n} rows=1 graph_nodes={nodes}");
        }
        // Every round's line is held and written after the loop: a write is
        // a syscall, and the rounds it would separate are the measurement.
        let gen_slots::Rounds {
            ids: out_ids,
            walls,
        } = gen_slots::rounds(s, next, n_gen)?;
        for (j, (len, wall, plan, image)) in prompts.into_iter().enumerate() {
            print_prompt(len, wall, &plan, image, &format!(" slot={j}"));
        }
        for (k, &ms) in walls.iter().enumerate() {
            let i = k + 1;
            for (j, (p, ids)) in pos0.iter().zip(&out_ids).enumerate() {
                println!("step {i} {} {} slot={j}", p - 1 + i, ids[i]);
            }
            if run.timed {
                Record::new(&record::TIME_PASS)
                    .u("i", i)
                    .flag("warm", i <= warm)
                    .f("ms", ms)
                    .u("positions", n)
                    .w("kind", "slots")
                    .print();
            }
        }
        for (j, ids) in out_ids.iter().enumerate() {
            println!("tokens {ids:?} slot={j}");
        }
        if run.timed {
            let c = gen_slots::counted(&walls, warm, n)?;
            println!(
                "SMOKE mode={} prompt_tokens={} depth={} slots={n} generated={n_gen} warm={warm} \
                 rounds={} positions={} p50_ms={:.4} mean_ms={:.4} tok/s(aggregate)={:.2} ctx={}",
                mode_name(run.mode),
                windows[0].len(),
                windows[0].len(),
                c.rounds,
                c.positions,
                c.p50,
                c.mean,
                c.aggregate,
                run.ctx
            );
        }
        Ok(())
    }

    /// An `--arm` list's arm opening: its `arm` line, under `--arm-sync` the
    /// wait for one line on stdin (the runner's witness block), then its
    /// `prompt_ids` line.
    fn arm_head(i: usize, count: usize, arm: &Arm, sync: bool) -> Result<(), GateError> {
        println!(
            "arm i={i} arms={count} ids={} n={}",
            arm.ids.len(),
            arm.n_gen
        );
        if sync {
            let mut line = String::new();
            if std::io::stdin().read_line(&mut line)? == 0 {
                return Err(format!("--arm-sync: stdin closed before arm {i} of {count}").into());
            }
        }
        println!("prompt_ids {:?}", arm.ids);
        Ok(())
    }

    /// [`Probe::read`] of the qwen4exp body, which always has a host tier.
    fn probe38(m: &Qwen38Model, prev: Option<&Probe>) -> Result<Probe, GateError> {
        Probe::read(m.body("generate_qwen3moe")?.hybrid(), m.gpu(), prev)
    }
}
