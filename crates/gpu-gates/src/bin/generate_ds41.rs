//! `generate_ds41` — the thin end-to-end decode CLI of the V4.1 engine, and
//! its timing ruler: `generate`'s shape over `bloomery_gpu_deepseek41::body`.
//!
//!     generate_ds41 [--prompt-id P | --tokens a,b,c] [--depth D] [-n N]
//!                   [--ctx C] [--place PLACE] [--mode eager|graph]
//!                   [--time [--warm W]] [--plan] [--logits]
//!                   [--top2 K [--rows FILE]] [--repeat R] [--table]
//!                   [--ignore-eos] [--serve-feed] [--dump-table FILE]
//!                   [--card-table FILE]
//!     generate_ds41 --arm SPEC [--arm SPEC ...] [--arm-sync] [-n N] [--ctx C]
//!                   [--place PLACE] [--mode eager|graph] [--time [--warm W]]
//!                   [--logits]
//!     generate_ds41 --records-schema
//!
//! Defaults: prompt 0 (none when `--depth` is given), N 32, C the serving
//! context (`workstation::CTX_MAX`), place `a` (the common unset rule,
//! `generate::Place::choose` by `place::TIER_RULE`: no break-even yet, so
//! `a` on one card or two), mode graph, W 0. A flag given twice takes its
//! last value, so a recipe's default can be overridden by the arguments
//! after it. A `place unset` record names the placement and why, first of the
//! run's records, `--plan` runs included.
//!
//! Greedy only, one token per generated step. The fed ids go in batches of
//! up to `body::T_MAX` positions (`body::prefill`, bit for bit the steps'
//! state); `BLOOMERY_PREFILL=steps` feeds them one real step per id instead,
//! the same binary's timing arm, and the `load` line prints which one ran
//! (`prefill=`, the mode the engine holds). The levers it acts on (`ACTS_ON`)
//! are parsed once, at `main` (`bloomery_levers::at_main`), which refuses by
//! name a lever set outside them and a `BLOOMERY_*` name no registry row
//! names; `--levers` prints them with this process's values and exits.
//! Every line the binary writes is a record of a kind
//! `bloomery_gpu_gates::record` declares; `--records-schema` prints those
//! kinds (fields, types, units) and exits. Prompt ids are
//! the V4.1 file's own: row P of `tools/ref/prompts.tsv` as
//! `tools/ref/ik-greedy.sh` tokenized it into
//! `$BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv` — the ids the long gate's
//! `--free`, the dsloop gate and ik's continuation read.
//!
//! `--place` is the placement the engine loads by: `a` is the serving plan
//! (`workstation::plan_a`, every layer and the head on the A6000, each routed
//! layer's expert prefix the budget allows on the card, the rest on the host
//! tier) and the one the timing runners use; `gate` is the step gate's
//! (`workstation::plan_gate`, the same on the 3090); `bp` is plan (b′)
//! (`workstation::plan_bp`): plan (a) on the A6000 byte for byte, and the
//! 3090 an expert tier under the host tier holding each layer's next ids
//! (`bloomery_gpu::host::tier::TierOpen::of_machine`), with the DSpark draft's reserve
//! when the draft runs (its header's bytes, read before the load; the draft
//! then sits on the 3090, and a `BLOOMERY_DSPARK_CARD` naming another card is
//! refused), and the tier's prompt-batch bytes (its staging and tile scratch
//! for blocks of the host union's columns, from the file's header): a prompt
//! call under `bp` is fed as `BLOOMERY_PREFILL` says, as under `a`, the tier
//! serving its experts' slots of each batch. The `load` line's `cards=`
//! names every card the placement loaded and, under `bp`, the tier's experts
//! and resident bytes (`tier_experts=`, `tier_bytes=`). `PLACE` may also be
//! a card list `<stage>[+<tier>…]` of `workstation::CARDS` names
//! (`generate::Place`): `a6000+3090` is `bp`, `a6000` is `a`, `3090` is
//! `gate`; a list whose stage card is not the A6000 (no gate runs another
//! stage) and a list of more tier cards than the V4.1 body serves are refused
//! by name before any plan. A lost tier card (it
//! stops signalling within the go deadline) is the step's or the call's named
//! error and ends the run with a nonzero exit. The card is found by name, so
//! the box's card pin decides which placements can load. The ring
//! shadows are page-locked host memory: the `plan` line prints the plan's
//! figure (`host_shadow=`), the `load` line the allocation
//! (`shadow=host <bytes>`) and the card's unified addressing the load
//! checked; `resident_bytes` counts device bytes only.
//!
//! `--depth D` stands the run at depth D before its first generated token:
//! the prompt, then the depth tables' sequence (`lcg_prompt` in
//! `tools/ref/lease.sh`) from index `P` on, every id one real step. With no
//! prompt the fed ids are exactly `lcg_prompt D`, the ids the V2-Lite depth
//! runner feeds. There is no synthetic depth: this body has no
//! `seed_depth`.
//!
//! Any depth up to `--ctx` runs: every indexer layer selects its stream's
//! list at every position (the identity while the visible rows fit in
//! `top_k`, the indexer's top-k after; past 16,384 positions layers 24, 28,
//! 32 and 36 take it inside the candidate mask layer 20 ranks). The run
//! selects with the file's
//! `top_k`, the model's own; the load line prints it, and a body that
//! loaded with another is refused before the first step.
//!
//! `--time` is a MEASUREMENT and belongs under the machine-wide lease
//! (`tools/ref/time-gate.sh generate_ds41 … --time`,
//! `tools/ref/depth-ds41.sh`). It times each generated step host-side around
//! `step(&[tok])`; the argmax readback synchronizes inside, and the host
//! tier's share runs inside it, so the wall time is the whole step. The fed
//! ids (the prompt, then the depth ids) are timed as one wall, not step by
//! step: the `time prompt n=<P> ms= tok/s= passes=<K> kind=` row runs from
//! before the first fed step to after the readback of generated token 0.
//! `passes` is the passes the feed took: the batches under `kind=batch`
//! (with the DSpark draft, each batch's feature rows are read and appended
//! to the draft inside the wall), one step per id under `kind=steps` — the
//! step rate by construction, at or above it, since that feed runs one body
//! per id and reads back once at the end — or `dspark` (the step feed with
//! the draft: one readback per id, its feature reads and draft appends) or
//! `checked` (the finite probe's eager steps run inside the wall; the probe
//! always feeds step by step). The row prints on every run,
//! after the loop, as a runtime value like the `load` line; it is a
//! measurement only under the lease, as `time step` is. Nothing is printed
//! between two timed steps. `--warm W` drops the first W generated steps from the
//! statistics and still prints them, marked. The `SMOKE` footer carries the
//! keys `generate`'s does (`p50_ms=`, `mean_ms=`, `warm=`), so the runners
//! that read one read the other.
//!
//! A batched feed prints the batch's device bytes (`prefill batch_bytes=`,
//! of them the batch-wide attention projections' `proj_bytes=` and what the
//! batches past a group's first hold for themselves, `group_bytes=`, for
//! groups of `group=` batches) before the prompt and a `stat prefill` line after its `step 0` line: the
//! host tier's batch services since the arm's start — the load, in a run of one arm
//! (`union_layers`, `union_cols`,
//! `union_host_slots`) and the union calls' wall (`union_ms`), the part of
//! the feed the card waits on the host; a runtime value, as `time prompt` is.
//! A `stat prefill split` line follows (`body::PrefillStats`): the group
//! lever (`group=`), the prologue, the enqueue with its union calls, waits
//! on the route's copies and activation copies, and the enqueue time left,
//! summed and per layer-batch — the waits also per layer-batch of a group's
//! first batch (`wait_first_lb=`: in a group of two or more, the route the
//! previous layer's last batch enqueued ahead); the queue entries — launches, event records, stream waits —
//! the route and the shadow put in a layer-batch (`entries_route=`,
//! `entries_shadow=`, the launch-queue model's N_r and N_s) and the host
//! tier's batch-excluded slots a layer-batch (`excluded_lb=`); with
//! `BLOOMERY_STEP_STATS=1` also each layer's card time by event pairs
//! (`card_out`: its first launch to its route's copies; `card_in`: its
//! shadow; `card_proj`: the batch-wide attention projections, inside
//! `card_out`), whose reads add a wait per batch to the feed.
//! The `load` line's `group=` is `BLOOMERY_PREFILL_GROUP` (default 2; 1 runs
//! each batch alone), the batches whose layers a batched feed runs in turn.
//! A second `stat prefill ced=` line names the triangle's state (the `load`
//! line's `ced=`: `on`, or `off (reason)`) and the last call's needs: its
//! positions, the first whose features were kept, the blocks and latent
//! parts it ran over every layer, and each layer's block and latent starts.
//! With `BLOOMERY_STEP_STATS=1`, after the loop, a `stat prefill front` line
//! per batch and a `stat prefill lb` line per layer-batch: the queue entries
//! each enqueued (`body::Body::prefill_counts`), whose sums the `split`
//! line's `entries_route=`/`entries_shadow=` are, and its card and serve
//! times (`card_out_ms=`, `card_in_ms=` where it has a block, `union_ms=`,
//! `wait_ms=`), whose sums the `split` line's are: the split's record is
//! refused by name when they are not (`body::PromptCounts::check_split`).
//!
//! A batched feed prints the plan of its call before the load (`call plan`,
//! then a `call batch` line per batch): `body::BodyLevers::call_plan`, the
//! functions the call's enqueue runs by, and the feed refuses a call whose
//! needs are not the printed ones. `--plan` prints the whole plan — every
//! group lever's groups (`call groups`), each batch, each layer's facts and
//! its card's experts (`call layer`), the needs (`call need`) and every
//! layer-batch's starts and sub-blocks (`call lb`) — and exits before the
//! load; it is refused with `--time`, beside `BLOOMERY_DRAFT=dspark` (its
//! feed widens the tapped layers' blocks) and beside a feed of steps.
//!
//! The binary owns its main thread, so it pins it to the dispatcher's cpu
//! slot (`threads::pool().pin_caller()`), as `bloomery-decode` and
//! `bench_v41_host` do; `BLOOMERY_PIN_MAIN=0` leaves it floating, for the
//! A/B. The `load` line prints both the ask and the outcome.
//!
//! `BLOOMERY_STEP_STATS=1` reads, after every generated step, the host
//! tier's counters (`HybridStats`), the process's page faults (`getrusage`)
//! and the card's free device bytes (`cuMemGetInfo`), and prints one
//! `stat step` line per step and a `stat summary` over the steps `--warm`
//! keeps, after the loop, as the `time` lines are. The summary's
//! `vram_free_load` is the read before the first generated step (after the
//! load, the capture and the fed ids); a replayed step allocates nothing,
//! so `vram_free` staying there is the expected line. Unset, the stat path
//! does not run: no read, no call.
//!
//! The run drives the V4.1 session (`app::Session`, `runtime::generate`):
//! the fed ids through its prompt, then one pass at a time until `-n`
//! tokens are out.
//!
//! `--arm SPEC` runs several arms after one load (`app::Session::arms`), in
//! the order given: `D` feeds `lcg_prompt D` as `--depth D` does, `prose:P`
//! and `code:P` the first P ids of `$BLOOMERY_DATA/engram/corpus-<name>.ids`
//! as `--tokens` does, each optionally followed by `/N`, its own `-n`. Each
//! arm opens with an `arm` record (its index, the list's length, the feed and
//! its counts), then its call's plan under a batched feed, then every line a
//! one-arm run prints after its capture; each arm after the first starts
//! from the session's clear, so it prints the tokens, logits and counters it
//! prints in a fresh process. Before each arm after the first, the state-back
//! check (`shared/state_back.rs`) asserts the model came back to its load —
//! position 0, not poisoned, each card's free device bytes within a fixed
//! slack of the load's own reading — before the arm's `arm` record, so
//! `--arm-sync`'s wait stays where it was. The since-load counters (`stat prefill`'s
//! `union_*`) count from the arm's start. The load-time lines (`plan`,
//! `load`, `capture`, `prefill`) print once, before arm 0. `--arm` does not
//! mix with `--prompt-id`, `--tokens`, `--depth` or `--plan`, and a list of
//! more than one arm is refused beside `BLOOMERY_DRAFT` (a draft's state has
//! no clear) and `BLOOMERY_CHECK_FINITE`. `--arm-sync` makes each arm wait,
//! after its `arm` record, for one line on stdin: the timing runner takes its
//! witness blocks between two arms of one load there. A failed arm ends the
//! process, naming the arm; the model is never cleared past a fault.
//!
//! `--logits` prints a `logits` record after the loop: the head's last
//! logits row read back once, its length, argmax and FNV-1a 64 of its f32
//! bits — the bit-identity check's handle on the logits, outside every timed
//! window.
//!
//! `--top2 K [--rows FILE]`, `--repeat R`, `--table`, `--ignore-eos`,
//! `--serve-feed`, `--dump-table FILE` and `--card-table FILE` are the
//! generate binaries' shared diagnosis flags (`generate::Diag`, whose doc is
//! theirs). They act on the plain runs of the prompt flags, and each is
//! refused by name beside a draft, the finite probe and `BLOOMERY_GEN_SLOTS`.
//! `--repeat` does not mix with `--arm`: an arm list's runs start from the
//! session's clear, which sends the residency back to its seed, and
//! `--repeat`'s from `Target::reset`, which keeps it. `--card-table` needs
//! `BLOOMERY_RESIDENCY=off` (a machine would move the experts it places) and
//! loads through `app::Loaded::open_edited`. The stage card's copy of the
//! map the table flags read is `Body::slots`, the host tier's map and machine
//! `Body::hybrid` (`generate::Residence`).
//!
//! `BLOOMERY_DRAFT=lookup` serves an n-gram lookup draft (`runtime::Lookup`,
//! fed the fed ids and every kept token) through the skewed two-row pass. A
//! pass with a proposal `d` runs `step_rows([next, d])`: row A's argmax
//! equal to `d` accepts both rows' tokens (two positions), otherwise the
//! second position is taken back and row A's token alone is emitted (one
//! position). A pass with no proposal is one
//! `step`. Greedy either way, so the `tokens` line equals the plain run's;
//! the last pass may overshoot `-n` by one, and the lines print the first
//! `-n` tokens. The run needs `--ctx` to hold that one extra position. Its
//! lines differ from the plain path's only where the lever is: `time pass`
//! rows (`positions=`, `kind=plain|pair-accept|pair-reject`) instead of
//! `time step`, a `draft summary`, and the `SMOKE` line's trailing
//! `positions=` and `tok/s(positions)=`, the positions the kept passes
//! advanced over their summed wall time. The pair pass's heads and capture
//! are made before the prompt (`Session::with_draft`), as the step's capture
//! is. `BLOOMERY_STEP_STATS` reads once per pass.
//!
//! `BLOOMERY_DRAFT=dspark` serves the DSpark draft at width 1
//! (`app::arch::deepseek41::CardDraft`): the draft file is
//! `$BLOOMERY_DSPARK_MODEL`, its card `BLOOMERY_DSPARK_CARD` (the 3090 when
//! unset; `shared/ds41_dspark.rs`), and the target carries the feature tap
//! of the draft's `target_layers`, built before the capture. The fed ids go
//! by the prompt schedule, the draft taking the features the call hands it
//! (every position's under steps, the draft's window under a batch); then
//! each pass the width chooser drafts (`runtime::width`; every pass under
//! `BLOOMERY_MTP_WIDTH=fixed`) proposes, runs the pair over `[next,
//! proposal]` and appends the features of the positions it keeps. `kind=` on
//! the `draft summary` line names the draft and `width=` the chooser's mode,
//! under `cost` with its state at the run's end. The rows and lines are the
//! lookup's; a `load draft=dspark` line follows the `load` line.
//!
//! `BLOOMERY_GEN_SLOTS=N` (unset or 1 is the one-sequence run above, line
//! for line) decodes N streams in one pass (`GpuModel::step_slots`, the
//! body's `SlotRows` pass of a row a slot; N past its `MAX_ROWS` is refused
//! by name). The load plans N resident sequences
//! (`PlanInputs::plan_with_slots`: each one's caches on the card, its ring
//! shadows on the host), the `plan` record that plan's, and the session then
//! serves N slots. Each arm's fed ids are N windows of equal length P, an id
//! count N does not divide refused by name: slot j, from its reset, runs
//! window j as the plain run's prompt call (its `fed`, `step 0` and, under a
//! batched feed, `stat prefill` lines; the call's printed plan is a window's,
//! P ids from position 0, and each slot's call is held to it). In graph mode
//! the pass of a row a slot is captured next, outside every timed window;
//! then `-n` − 1 rounds of one pass, each slot fed its own argmax, nothing
//! printed between two rounds. After the rounds: each slot's `time prompt`,
//! then per round each slot's `step` record and, under `--time`, the round's
//! `time pass <i> ms= positions=N kind=slots` (the pass and its N ids'
//! readback; `warm` on a round `--warm` drops), then each slot's `tokens`;
//! every per-slot record in slot order. The `SMOKE` footer's p50 and mean
//! are a round's, `prompt_tokens` and `depth` a window's, `steps` the counted
//! rounds, and its `positions` and `tok/s(positions)` the counted rounds'
//! positions and Σ positions · 1000 / Σ ms, the aggregate rate. Every arm of
//! an `--arm` list after the first starts from every slot's reset and, on a
//! load that runs the residency machine, the residency's seed (its `residency
//! reset` record before its `arm` record), so it runs as in a fresh process.
//! The residency boundaries after the seed's are the slots' prompt calls
//! (`pass=prompt kept=0`, one or more a slot), then one a round
//! (`pass=slots kept=N`, printed after the arm's lines). Refused by name before the load beside
//! `BLOOMERY_DRAFT` (a draft verifies one sequence; the DSpark draft's feature
//! tap is one sequence's), `BLOOMERY_CHECK_FINITE=1`, `BLOOMERY_STEP_STATS=1`
//! and `--logits`, which this arm does not run.
//!
//! `BLOOMERY_CHECK_FINITE=1` runs every position the run steps — the fed ids
//! and each generated token — first through the finite probe
//! (`shared/ds41_finite.rs`): the position's step eagerly, outside the graph,
//! each sub-layer's streams read where it wrote them; then the position is
//! taken back and the token stepped through the engine as without the lever,
//! so the `tokens` line is the plain run's. After the loop, a `stat finite
//! step` line per generated step (`ok`, or the non-finite seams, the first
//! one's `(layer, site)` and, at a MoE seam, its routing and buffers), a line
//! per fed position that is not `ok`, and a `stat finite summary`. A position
//! whose eager argmax differs from the engine's token says so. Refused with
//! `--time`, `BLOOMERY_DRAFT` and `BLOOMERY_STEP_STATS` (its host-tier
//! counters would count the probe's step too). Unset, the probe does not
//! run.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("generate_ds41: built without the `deepseek41` feature; see `just gen-ds41`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("generate_ds41", drive::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_finite.rs"]
mod finite;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_draft.rs"]
mod draft;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_split.rs"]
mod split;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_place.rs"]
mod place;

#[cfg(feature = "deepseek41")]
#[path = "shared/gen_slots.rs"]
mod gen_slots;

#[cfg(feature = "deepseek41")]
#[path = "shared/state_back.rs"]
mod state_back;

#[cfg(feature = "deepseek41")]
mod drive {
    use std::num::NonZeroUsize;
    use std::ops::Range;
    use std::time::{Duration, Instant};

    use app::arch::deepseek41::Ds41Cfg;
    use app::{Loaded, OpenArgs, OpenLog, RowsLog, Session, SessionError};
    use bloomery_gpu::GpuError;
    use bloomery_gpu::head::Head;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::hybrid::HybridStats;
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu_deepseek41::body::{
        self, Body, BodyMeta, Deepseek41Model, PAIR_ROWS, TierOpen,
    };
    use bloomery_gpu_deepseek41::chain::attn::SUB_TOKENS;
    use bloomery_gpu_deepseek41::swap;
    use bloomery_gpu_gates::generate::{
        Diag, NoEog, Place, PlaceWhy, Residence, ServeFeed, TopRows, before_path, card_table,
        dump_table, mode_name, place_table, read_table, repeat_runs, seed_path, slot_table,
        write_table,
    };
    use bloomery_gpu_gates::record::{self, Record};
    use bloomery_gpu_gates::{Fnv1a64, GateError, data_dir, ref_model_path, residency41};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, CED, CHECK_FINITE, DRAFT, ENGRAM_HELPER, GEN_SLOTS, HOST_LOCK,
        HOST_POPULATE, HOSTSTREAM, Levers, MTP_WIDTH, PIN_MAIN, PREFILL, PREFILL_GROUP, R8,
        RESIDENCY, ResidencyAt, ResidencyPick, STEP_STATS,
    };
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::PlanInputs;
    use model::placement::{Machine, Plan, PlanLevers, workstation};
    use runtime::width::{Choosing, Chosen, Mode as WidthMode};
    use runtime::{
        Advance, Committed, GenOutcome, Lookup, PassSink, Speculative, Stop, StopReason, Target,
        Want,
    };

    use crate::draft::{Draft, open_dspark};
    use crate::state_back::StateBack;
    use crate::{dspark, finite, gen_slots, place, split};

    /// The usage line: the prompt flags with the shared diagnosis flags
    /// (`Diag::USAGE`), or an `--arm` list.
    fn usage() -> String {
        format!(
            "usage: generate_ds41 [--prompt-id P | --tokens a,b,c] [--depth D] [-n N] [--ctx C] \
             [--place a|gate|bp|<stage>[+<tier>…]] [--mode eager|graph] [--time [--warm W]] \
             [--plan] [--logits] {}, or --arm SPEC [--arm SPEC ...] [--arm-sync] in place of \
             the prompt flags",
            Diag::USAGE
        )
    }

    /// Where the prompt comes from.
    #[derive(Clone)]
    enum Prompt {
        /// Row 0, or nothing under `--depth`.
        Default,
        Row(u32),
        Ids(Vec<u32>),
    }

    /// The corpora an `--arm` names: `corpus-<name>.ids` under
    /// `$BLOOMERY_DATA/engram`, one id per line.
    const CORPORA: &[&str] = &["prose", "code"];

    /// What an `--arm` feeds.
    #[derive(Clone, Copy)]
    enum ArmFeed {
        /// `lcg_prompt D`, as `--depth D`.
        Lcg(usize),
        /// The first P ids of a corpus, as `--tokens`.
        Corpus(&'static str, usize),
    }

    /// One `--arm SPEC`: its feed and its own `-n`, if it names one.
    #[derive(Clone, Copy)]
    struct ArmSpec {
        feed: ArmFeed,
        n_gen: Option<usize>,
    }

    impl ArmSpec {
        /// `D`, `prose:P` or `code:P`, each optionally followed by `/N`.
        fn parse(spec: &str) -> Result<ArmSpec, GateError> {
            let bad = || -> GateError {
                format!("--arm {spec:?} is D, prose:P or code:P, optionally followed by /N").into()
            };
            let (feed, n_gen) = match spec.split_once('/') {
                Some((f, n)) => (f, Some(n.parse::<usize>().map_err(|_| bad())?)),
                None => (spec, None),
            };
            let feed = match feed.split_once(':') {
                None => ArmFeed::Lcg(feed.parse().map_err(|_| bad())?),
                Some((name, p)) => {
                    let name = CORPORA
                        .iter()
                        .copied()
                        .find(|&c| c == name)
                        .ok_or_else(bad)?;
                    ArmFeed::Corpus(name, p.parse().map_err(|_| bad())?)
                }
            };
            Ok(ArmSpec { feed, n_gen })
        }

        /// The feed's name, as the `arm` record prints it.
        fn feed_name(self) -> &'static str {
            match self.feed {
                ArmFeed::Lcg(_) => "lcg",
                ArmFeed::Corpus(name, _) => name,
            }
        }
    }

    #[derive(Clone)]
    struct Args {
        prompt: Prompt,
        depth: Option<usize>,
        n_gen: usize,
        ctx: usize,
        /// The placement the run loads by, resolved against this process's
        /// devices by the common unset rule ([`place::choose`]).
        place: Place,
        mode: StepMode,
        timed: bool,
        warm: Option<usize>,
        /// `BLOOMERY_STEP_STATS`, as the levers hold it.
        stats: bool,
        /// `--plan`: print the call's whole plan and load nothing.
        plan: bool,
        /// `--arm`: the arms, in order; empty for a run of the prompt flags.
        arms: Vec<ArmSpec>,
        /// `--arm-sync`: each arm waits for a line on stdin after its record.
        sync: bool,
        /// `--logits`: the `logits` record after the loop.
        logits: bool,
        /// The shared diagnosis flags ([`Diag`]).
        diag: Diag,
        /// Whether this run is the one `--dump-table` reads.
        dump_now: bool,
    }

    fn parse_args(levers: &Levers) -> Result<(Args, PlaceWhy), GateError> {
        let mut a = Args {
            prompt: Prompt::Default,
            depth: None,
            n_gen: 32,
            ctx: usize::try_from(workstation::CTX_MAX)?,
            place: Place::A,
            mode: StepMode::Graph,
            timed: false,
            warm: None,
            stats: levers.step_stats(),
            plan: false,
            arms: Vec::new(),
            sync: false,
            logits: false,
            diag: Diag::default(),
            dump_now: false,
        };
        let (mut row, mut ids) = (None, None);
        let mut place_flag = None;
        let mut it = std::env::args().skip(1);
        while let Some(flag) = it.next() {
            if flag == "--time" {
                a.timed = true;
                continue;
            }
            if flag == "--plan" {
                a.plan = true;
                continue;
            }
            if flag == "--arm-sync" {
                a.sync = true;
                continue;
            }
            if flag == "--logits" {
                a.logits = true;
                continue;
            }
            if a.diag.take(&flag, &mut it)? {
                continue;
            }
            let v = it
                .next()
                .ok_or_else(|| format!("{flag} needs a value, or is unknown: {}", usage()))?;
            match flag.as_str() {
                "--prompt-id" => row = Some(v.parse()?),
                "--tokens" => {
                    ids = Some(
                        v.split(',')
                            .map(|t| t.trim().parse::<u32>())
                            .collect::<Result<Vec<_>, _>>()?,
                    );
                }
                "--depth" => a.depth = Some(v.parse()?),
                "--arm" => a.arms.push(ArmSpec::parse(&v)?),
                "-n" => a.n_gen = v.parse()?,
                "--ctx" => a.ctx = v.parse()?,
                "--warm" => a.warm = Some(v.parse()?),
                "--place" => place_flag = Some(place::parse(&v)?),
                "--mode" => {
                    a.mode = match v.as_str() {
                        "graph" => StepMode::Graph,
                        "eager" => StepMode::Eager,
                        other => {
                            return Err(format!("--mode is eager or graph, not {other}").into());
                        }
                    };
                }
                other => return Err(format!("unknown argument {other:?}: {}", usage()).into()),
            }
        }
        a.prompt = match (row, ids) {
            (Some(_), Some(_)) => {
                return Err("--prompt-id and --tokens both name the prompt. Pass one.".into());
            }
            (Some(p), None) => Prompt::Row(p),
            (None, Some(v)) => Prompt::Ids(v),
            (None, None) => Prompt::Default,
        };
        if !a.arms.is_empty() {
            let beside = [
                (
                    !matches!(a.prompt, Prompt::Default),
                    "--prompt-id or --tokens",
                ),
                (a.depth.is_some(), "--depth"),
                (a.plan, "--plan"),
            ];
            if let Some((_, what)) = beside.iter().find(|(set, _)| *set) {
                return Err(format!("--arm names the prompt: {what} does not mix with it").into());
            }
        } else if a.sync {
            return Err("--arm-sync paces the arms of an --arm list, and none is given".into());
        }
        if a.diag.repeat > 1 && !a.arms.is_empty() {
            return Err(
                "--repeat runs the prompt flags' run again; --arm names its own runs".into(),
            );
        }
        a.diag.finish(a.n_gen, &ref_model_path()?)?;
        check_counts(&a)?;
        for arm in &a.arms {
            check_counts(&a.for_arm(arm))?;
        }
        let placed = place::choose(place_flag)?;
        a.place = placed.place;
        Ok((a, placed))
    }

    impl Args {
        /// The run's arguments for `arm`: its own `-n` when it names one.
        fn for_arm(&self, arm: &ArmSpec) -> Args {
            Args {
                n_gen: arm.n_gen.unwrap_or(self.n_gen),
                ..self.clone()
            }
        }
    }

    /// Refused rather than ignored, as `generate` refuses them: a `--warm`
    /// with nothing to trim, and one that trims every timed step.
    fn check_counts(a: &Args) -> Result<(), GateError> {
        if a.n_gen == 0 {
            return Err("-n wants at least one generated token".into());
        }
        if a.timed && a.n_gen < 2 {
            return Err(
                "--time with -n 1 has no generated step to time: token 0 comes out of the \
                 prompt's own step"
                    .into(),
            );
        }
        match a.warm {
            Some(_) if !a.timed => Err("--warm drops the first W steps from --time's \
                                        statistics, which this run has not asked for. Pass \
                                        both, or neither."
                .into()),
            Some(w) if w >= a.n_gen - 1 => Err(format!(
                "--warm {w} leaves no timed step of the {} that -n {} generates",
                a.n_gen - 1,
                a.n_gen
            )
            .into()),
            _ => Ok(()),
        }
    }

    /// The prompt's ids from `$BLOOMERY_DATA/greedy-ds41/prompt<P>.tsv`
    /// (`tools/ref/ik-greedy.sh`): the first row's third column.
    fn prompt_row(p: u32) -> Result<Vec<u32>, GateError> {
        let path = data_dir()
            .join("greedy-ds41")
            .join(format!("prompt{p}.tsv"));
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("{}: {e} — run just ik-greedy-ds41 {p}", path.display()))?;
        let row = text
            .lines()
            .find(|l| !l.starts_with('#') && !l.is_empty())
            .ok_or_else(|| format!("{}: no prompt row", path.display()))?;
        let ids = row
            .split('\t')
            .nth(2)
            .ok_or_else(|| format!("{}: the row has no ids", path.display()))?;
        Ok(ids
            .split(',')
            .map(|t| t.trim().parse::<u32>())
            .collect::<Result<Vec<_>, _>>()?)
    }

    /// The depth tables' sequence, `lcg_prompt` of `tools/ref/lease.sh`: 100000,
    /// then an LCG walk over [1000, 91000). awk computes it in doubles, so this
    /// does too — past the second id the exact 64-bit LCG is another sequence.
    fn depth_ids(n: usize) -> Vec<u32> {
        let mut out = Vec::with_capacity(n);
        let mut s: f64 = 12345.0;
        for i in 0..n {
            if i == 0 {
                out.push(100_000);
                continue;
            }
            s = (s * 1_103_515_245.0 + 12345.0) % 2_147_483_648.0;
            // An integer-valued double in [0, 90000): the cast is exact.
            out.push(1000 + (s % 90_000.0) as u32);
        }
        out
    }

    /// The ids the run feeds before its first generated token, and how many
    /// of them are the prompt's.
    fn fed_ids(a: &Args) -> Result<(Vec<u32>, usize), GateError> {
        let mut ids = match &a.prompt {
            Prompt::Ids(v) => v.clone(),
            Prompt::Row(p) => prompt_row(*p)?,
            Prompt::Default if a.depth.is_some() => Vec::new(),
            Prompt::Default => prompt_row(0)?,
        };
        let prompt_len = ids.len();
        if let Some(d) = a.depth {
            if d < prompt_len {
                return Err(
                    format!("--depth {d} is shorter than the prompt's {prompt_len} ids").into(),
                );
            }
            ids.extend_from_slice(&depth_ids(d)[prompt_len..]);
        }
        if ids.is_empty() {
            return Err("an empty prompt and no --depth: nothing to feed".into());
        }
        Ok((ids, prompt_len))
    }

    /// An arm's fed ids and how many of them are the prompt's: `--depth D`'s
    /// for `D`, `--tokens`' for a corpus arm.
    fn arm_ids(arm: &ArmSpec) -> Result<(Vec<u32>, usize), GateError> {
        match arm.feed {
            ArmFeed::Lcg(0) => Err("--arm 0: nothing to feed".into()),
            ArmFeed::Lcg(d) => Ok((depth_ids(d), 0)),
            ArmFeed::Corpus(name, p) => {
                let path = data_dir().join("engram").join(format!("corpus-{name}.ids"));
                let text = std::fs::read_to_string(&path)
                    .map_err(|e| format!("--arm {name}:{p}: {}: {e}", path.display()))?;
                let all = text
                    .lines()
                    .map(|t| t.trim().parse::<u32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| format!("--arm {name}:{p}: {}: {e}", path.display()))?;
                if p == 0 || p > all.len() {
                    return Err(format!(
                        "--arm {name}:{p}: {} holds {} ids (1..{})",
                        path.display(),
                        all.len(),
                        all.len()
                    )
                    .into());
                }
                Ok((all[..p].to_vec(), p))
            }
        }
    }

    /// The Parsed levers the run acts on besides the pool's two.
    const ACTS_ON: &[&str] = &[
        CED,
        PREFILL,
        PREFILL_GROUP,
        ENGRAM_HELPER,
        STEP_STATS,
        CARD_BUDGET,
        PIN_MAIN,
        DRAFT,
        MTP_WIDTH,
        CHECK_FINITE,
        HOST_POPULATE,
        HOST_LOCK,
        CARD_DONTNEED,
        R8,
        RESIDENCY,
        HOSTSTREAM,
        GEN_SLOTS,
    ];

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(ACTS_ON)?;
        record::at_main("generate_ds41", record::GENERATE_DS41);
        let (a, placed) = parse_args(&levers)?;
        placed.record().print();
        let draft = Draft::from_levers(&levers)?;
        let width = WidthMode::of(levers.mtp_width())?;
        if draft == Draft::Off && levers.mtp_width().is_some() {
            return Err(
                "BLOOMERY_MTP_WIDTH picks the width a drafted window verifies; \
                 BLOOMERY_DRAFT=off runs the plain path"
                    .into(),
            );
        }
        let check_finite = finite_lever(&a, draft, &levers)?;
        let at = ResidencyAt {
            serving_place: a.place != Place::Gate,
            check_finite,
            route_trace: false,
            prefill_steps: body::PrefillMode::from_name(levers.prefill())
                == Some(body::PrefillMode::Steps),
        };
        let residency = levers.residency_at(at);
        let cfg = body::OpenCfg::from_levers_at(&levers, at)?;
        if cfg.body.residency != Residency::Off && (a.place == Place::Gate || check_finite) {
            return Err(format!(
                "BLOOMERY_RESIDENCY={} {}: the residency machine runs under --place a and bp, \
                 on the engine's own passes",
                residency.word,
                if check_finite {
                    "beside BLOOMERY_CHECK_FINITE=1"
                } else {
                    "under --place gate"
                }
            )
            .into());
        }
        let batched = !check_finite && cfg.body.prefill == body::PrefillMode::Batch;
        if a.plan {
            refuse_plan(&a, draft, batched)?;
        }
        let slots = levers.gen_slots();
        if slots > 1 {
            let mut beside = vec![
                ("BLOOMERY_DRAFT", draft != Draft::Off),
                ("BLOOMERY_CHECK_FINITE=1", check_finite),
                ("BLOOMERY_STEP_STATS=1", a.stats),
                ("--logits", a.logits),
            ];
            beside.extend(a.diag.set().into_iter().map(|flag| (flag, true)));
            gen_slots::refused::<Body>(slots, "deepseek41", &beside)?;
        }
        let pin_main = levers.pin_main();
        let pinned = pin_main && threads::pool().pin_caller();
        let runs = arm_runs(&a)?;
        if slots > 1 {
            for (i, r) in runs.iter().enumerate() {
                gen_slots::windows(&r.ids, slots, &format!("arm {i}'s fed ids"))?;
            }
        }
        if runs.len() > 1 && (draft != Draft::Off || check_finite) {
            return Err(format!(
                "{} arms after one load are refused with {}: its state has no clear",
                runs.len(),
                if check_finite {
                    "BLOOMERY_CHECK_FINITE=1"
                } else {
                    "BLOOMERY_DRAFT"
                }
            )
            .into());
        }
        if draft != Draft::Off && runs.iter().any(|r| r.a.n_gen < 2) {
            return Err(format!(
                "BLOOMERY_DRAFT={} with -n 1 has no pass to draft: token 0 comes out of the \
                 prompt's own step",
                draft.name()
            )
            .into());
        }
        // Positions 0 .. depth − 1 are the fed ids; the N − 1 feedback steps
        // take depth .. depth + N − 2. The plan is checked against the arm
        // that steps the most positions. Under several slots each slot holds
        // one window of the fed ids at its own positions.
        let widest = runs
            .iter()
            .max_by_key(|r| r.ids.len() / slots + r.a.n_gen)
            .ok_or("generate_ds41: no arm to run")?;
        let (depth, fed_n) = (widest.ids.len() / slots, widest.a.n_gen);
        let fed = depth + fed_n - 1;
        let draft_file = match draft {
            Draft::Dspark => Some(dspark::draft_hparams()?),
            _ => None,
        };
        let path = ref_model_path()?;
        let reserve = match &draft_file {
            Some((d, _)) => dspark::draft_reserve(a.place, d, &path)?,
            None => None,
        };

        let t = Instant::now();
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let tier_batch = place::tier_batch(
            a.place,
            &Hparams::read(&file).map_err(|e| format!("{}: {e}", path.display()))?,
        );
        let headers = t.elapsed();
        let machine = a.place.machine(reserve, tier_batch)?;
        // The unset word against the plan the load will use, and the lever
        // record naming what it resolved to, before the plan record the open
        // prints. `--card-table` loads a plan it edits itself, beside a word
        // that is `off` (refused below otherwise): no pre-plan there.
        let (plan_n_l, residency) = match &a.diag.card_table {
            Some(_) => (None, residency),
            None => pre_plan(&file, &a, &cfg.place, machine, slots, residency)?,
        };
        record::residency_lever(residency).print();
        if let Some(d) = residency.why.detail() {
            eprintln!("{d}");
        }
        let cfg = body::OpenCfg::from_levers_with(&levers, Residency::parse(residency.word)?)?;
        // The finite probe feeds step by step, outside the prompt call.
        let feed_mode = if check_finite {
            body::PrefillMode::Steps
        } else {
            cfg.body.prefill
        };
        let args = OpenArgs {
            place: a.place.name(),
            machine,
            ctx: a.ctx,
            mode: a.mode,
            cfg: Ds41Cfg {
                open: cfg.clone(),
                feed: feed_mode,
                card_timing: a.stats,
            },
        };
        let mut log = Log {
            a: &a,
            cfg: &cfg,
            draft,
            batched,
            check_finite,
            depth,
            fed,
            fed_n,
            t,
            headers,
            plan: None,
            pin_main,
            pinned,
            residency,
            plan_n_l,
            hp: None,
            ctx_max: 0,
            call: None,
        };
        for r in &runs {
            a.diag.check_feed(r.ids.len())?;
        }
        let table = match &a.diag.card_table {
            Some(p) if cfg.body.residency == Residency::Off => Some(read_table(p)?),
            Some(_) => {
                return Err(
                    "--card-table places the experts itself: set BLOOMERY_RESIDENCY=off \
                     (the machine would move them)"
                        .into(),
                );
            }
            None => None,
        };
        let opened = match (NonZeroUsize::new(slots).filter(|n| n.get() > 1), &table) {
            (Some(_), Some(_)) => {
                return Err(
                    "--card-table loads one sequence's placement; BLOOMERY_GEN_SLOTS opens \
                     several slots: give one of them"
                        .into(),
                );
            }
            (Some(n), None) => open_slots(file, args, n, &mut log)?,
            (None, Some(t)) => Loaded::<Body>::open_edited(file, args, &mut log, |_, plan| {
                place_table(plan, t).map_err(|e| SessionError::Refused(e.to_string()))
            })?,
            (None, None) => Loaded::<Body>::open(file, args, &mut log)?,
        };
        let Some(mut loaded) = opened else {
            return Ok(());
        };
        let hp = log
            .hp
            .clone()
            .ok_or("generate_ds41: the open planned nothing")?;
        // The draft builds the target's feature tap, so it loads before the
        // capture: the captured step then carries the tap.
        let spark = match &draft_file {
            Some(file) => {
                let (d, load) =
                    open_dspark(&mut loaded, file, &path, "generate_ds41", a.place, reserve)?;
                load.print();
                Some(d)
            }
            None => None,
        };
        // Captured before the prompt, so the first timed step is a replay;
        // the batch's buffers made before it, so the timed feed allocates
        // nothing.
        let mut s = loaded.ready(&mut log)?;
        if let (Some(t), Some(p)) = (&table, &a.diag.card_table) {
            card_table(&residence(&s)?.table()?, t, p)?.print();
        }
        if cfg.body.residency != Residency::Off {
            // An arm's boundaries: one a slot's prompt call, the first
            // pass's, then at most one a generated token.
            let passes = runs.iter().map(|r| r.a.n_gen).max().unwrap_or(0) + slots + 1;
            s.model_mut()
                .body_parts("generate_ds41")?
                .2
                .log_residency(passes);
        }
        if slots > 1 {
            s.add_slots(slots)?;
        }
        let mut check = if check_finite {
            let (gpu, w, _) = s.model_mut().body_parts("generate_ds41")?;
            let head = Head::new(gpu, w, hp.rms_eps)?;
            Record::new(&record::CHECK_FINITE).print();
            Some(FiniteCheck {
                head,
                rows: Vec::with_capacity(fed + 1),
            })
        } else {
            None
        };
        let call = log.call.take();
        let pre = Prelude {
            cfg: &cfg,
            hp: &hp,
            ctx_max: log.ctx_max,
            call: batched && draft != Draft::Dspark,
            sync: a.sync,
            arms: runs.len(),
            slots,
        };
        if slots > 1 {
            return s
                .arms(&runs, |s, i, r| {
                    let ran = arm_slots(s, &pre, i, r, feed_mode, call.as_ref());
                    after_passes(s, ran)
                })
                .map_err(|f| Box::new(f) as GateError);
        }
        if draft == Draft::Off && !check_finite {
            a.diag.begin()?;
            let at_load = match a.diag.table {
                true => Some(s.model().body("generate_ds41")?.slot_map().stage_view()),
                false => None,
            };
            if let Some(p) = &a.diag.dump_table {
                write_table(&seed_path(p), &residence(&s)?.table()?)?;
            }
            let each = |s: &mut Session<Body>, i: usize, r: &ArmRun| -> Result<(), GateError> {
                if let Some(c) = s.take_cleared() {
                    record::residency_reset(&c).print();
                }
                let view = pre.arm(i, r)?;
                let fed = r.fed(feed_mode, view.as_ref().or(call.as_ref()), s)?;
                let ran = if r.a.diag.ignore_eos.is_empty() {
                    decode(s, &r.a, &fed, &mut runtime::Plain, "steps", |_| {})
                } else {
                    let mut adv = NoEog::new(r.a.diag.ignore_eos.clone());
                    decode(s, &r.a, &fed, &mut adv, "steps", |_| {})
                };
                after_passes(s, ran)
            };
            if runs.len() > 1 && runs[0].spec.is_none() {
                let (ids, n_gen) = (runs[0].ids.len(), runs[0].a.n_gen);
                repeat_runs(&mut s, runs.len(), ids, n_gen, "generate_ds41", |s, i| {
                    let r = &runs[i];
                    if let (true, Some(p)) = (r.a.dump_now, &r.a.diag.dump_table) {
                        write_table(&before_path(p), &residence(s)?.table()?)?;
                    }
                    each(s, i, r)
                })?;
            } else {
                // The state the load left, asserted back before each arm
                // after the first, before anything else the arm prints (the
                // `--repeat` path above keeps the residency by design and
                // does not check).
                let base = StateBack::read(s.model())?;
                s.arms(&runs, |s, i, r| {
                    if i > 0 {
                        let now = StateBack::read(s.model())?;
                        base.check(&now, &format!("arm {i}"))?;
                    }
                    each(s, i, r)
                })
                .map_err(|f| Box::new(f) as GateError)?;
            }
            if let Some(v) = at_load {
                let r = residence(&s)?;
                let t = r.table()?;
                slot_table(&t, &r.check(&t)?, t.differ(&v)?).print();
            }
            return Ok(());
        }
        // One arm: the pair pass's capture comes before its prelude, so no
        // arm's window holds it.
        let [r] = runs.as_slice() else {
            return Err("generate_ds41: a draft or the finite probe runs one arm".into());
        };
        let a = &r.a;
        if let Some(flag) = a.diag.set().first() {
            return Err(format!(
                "{flag} acts on the plain runs: not under a draft or the finite probe"
            )
            .into());
        }
        match (draft, spark, check.as_mut()) {
            (Draft::Off, _, None) => {
                Err("generate_ds41: the plain run goes through the arm list".into())
            }
            (Draft::Off, _, Some(c)) => {
                let view = pre.arm(0, r)?;
                let fed = r.fed(feed_mode, view.as_ref().or(call.as_ref()), &s)?;
                let mut checked = Checked { c };
                decode(&mut s, a, &fed, &mut checked, "checked", |k| {
                    print_finite(k.c, fed.ids.len());
                })
            }
            (Draft::Lookup, ..) => {
                let mut spec =
                    s.with_draft::<_, PAIR_ROWS>(behind(Lookup::new(), width)?, &mut log)?;
                let view = pre.arm(0, r)?;
                let fed = r.fed(feed_mode, view.as_ref().or(call.as_ref()), &s)?;
                let ran = decode_draft(&mut s, a, &fed, &mut spec, "steps", "lookup", |_| Ok(()));
                after_passes(&mut s, ran)
            }
            (Draft::Dspark, Some(d), _) => {
                let mut spec = s.with_draft::<_, PAIR_ROWS>(behind(d, width)?, &mut log)?;
                let view = pre.arm(0, r)?;
                let fed = r.fed(feed_mode, view.as_ref().or(call.as_ref()), &s)?;
                let ran = decode_draft(&mut s, a, &fed, &mut spec, "dspark", "dspark", |d| {
                    Ok(d.draft_mut().draft_mut().check_fault()?)
                });
                after_passes(&mut s, ran)
            }
            (Draft::Dspark, None, _) => Err("generate_ds41: the DSpark draft did not load".into()),
        }
    }

    /// `d` behind the width chooser of `mode` (`runtime::width`): this
    /// binary's drafts are the chooser's, whatever their own width.
    fn behind<D: runtime::Draft<Session<Body>>>(
        d: D,
        mode: WidthMode,
    ) -> Result<Choosing<D>, SessionError> {
        Ok(d.choosing(mode)?)
    }

    /// The residency views of `s`'s body ([`Residence`]): its stage card's
    /// copy of the slot map (`Body::slots`) and its host tier.
    fn residence(s: &Session<Body>) -> Result<Residence<'_>, GateError> {
        let m = s.model();
        let b = m.body("generate_ds41")?;
        Ok(Residence::of(m.gpu(), b.slots(), b.hybrid()))
    }

    /// A run's result `ran`, with the `residency pass` records of its
    /// boundaries printed after it ([`print_passes`]): a failed print is the
    /// run's error, and when the run failed too the error names both.
    fn after_passes<T>(s: &mut Session<Body>, ran: Result<T, GateError>) -> Result<T, GateError> {
        match (ran, print_passes(s)) {
            (ran, Ok(())) => ran,
            (Ok(_), Err(p)) => Err(p),
            (Err(r), Err(p)) => {
                Err(format!("{r}; then printing the residency passes after it: {p}").into())
            }
        }
    }

    /// The `residency pass` records of the boundaries since the last print,
    /// after the run's timed window: nothing prints between two timed steps.
    fn print_passes(s: &mut Session<Body>) -> Result<(), GateError> {
        let (_, _, b) = s.model_mut().body_parts("generate_ds41")?;
        for (kind, r) in b.take_residency_passes() {
            record::residency_pass_of(kind, &r).print();
        }
        Ok(())
    }

    /// The plan the load will use, made once before the open: the same
    /// inputs — the file's headers ([`PlanInputs::read`]), the placement's
    /// `machine`, the run's `--ctx`, the placement's levers and the resident
    /// `slots` — the open plans with (`Loaded::open` one sequence,
    /// [`open_slots`] several), for the unset residency word to resolve
    /// against ([`residency41::at_plan`]). Deterministic and milliseconds.
    /// Returns the plan's per-layer card experts, for the open's own plan to
    /// be checked against ([`Log::plan`]), and the word the plan left.
    fn pre_plan(
        file: &Split,
        a: &Args,
        place_levers: &PlanLevers,
        machine: impl Fn(usize) -> Machine,
        slots: usize,
        pick: ResidencyPick,
    ) -> Result<(Option<Vec<u64>>, ResidencyPick), GateError> {
        const WHAT: &str = "generate_ds41 residency pre-plan";
        let inputs = PlanInputs::read(file).map_err(|e| GpuError::plan(WHAT, e))?;
        let machine = machine(inputs.model.layers);
        let ctx = u64::try_from(a.ctx)
            .map_err(|_| format!("{WHAT}: a context of {} positions passes u64", a.ctx))?;
        let plan = match NonZeroUsize::new(slots).filter(|n| n.get() > 1) {
            Some(n) => inputs.plan_with_slots(&machine, ctx, place_levers, n),
            None => inputs.plan(&machine, ctx, place_levers),
        }
        .map_err(|e| GpuError::plan(WHAT, e))?;
        let pick = residency41::at_plan(&plan, pick)?;
        Ok((Some(plan.n_l.clone()), pick))
    }

    /// The load of a plan that counts `slots` resident sequences
    /// (`PlanInputs::plan_with_slots`), told to `log` step for step as
    /// [`Loaded::open`] tells it (`false` from its plan stops here:
    /// `Ok(None)`), loaded by that same plan with the placement's expert tier
    /// cards hung under the host tier ([`Body::open_placed_slots`]). The
    /// session over it serves one slot until [`Session::add_slots`].
    fn open_slots<M: Fn(usize) -> Machine>(
        file: Split,
        args: OpenArgs<Ds41Cfg, M>,
        slots: NonZeroUsize,
        log: &mut Log<'_>,
    ) -> Result<Option<Loaded<Body>>, GateError> {
        const WHAT: &str = "generate_ds41 open_slots";
        let inputs = PlanInputs::read(&file).map_err(|e| GpuError::plan(WHAT, e))?;
        let machine = (args.machine)(inputs.model.layers);
        let plan = inputs
            .plan_with_slots(
                &machine,
                u64::try_from(args.ctx)?,
                &args.cfg.open.place,
                slots,
            )
            .map_err(|e| GpuError::plan(WHAT, e))?;
        if !log.plan(args.place, &inputs, &machine, &plan)? {
            return Ok(None);
        }
        let ctx = u32::try_from(plan.ctx_max)
            .map_err(|_| format!("the plan's ctx_max {} passes u32", plan.ctx_max))?;
        let meta = BodyMeta {
            hp: inputs.hp.clone(),
            levers: args.cfg.open.body,
        };
        let tiers = TierOpen::of_machine(plan.machine);
        let mut m = Body::open_placed_slots(file, &plan, 0, tiers, &meta, slots)?;
        m.set_mode(args.mode);
        log.load(&m)?;
        Ok(Some(Loaded::from_model(m, args.cfg, ctx)))
    }

    /// Arm `i` as `pre.slots` streams in one pass (`BLOOMERY_GEN_SLOTS`, the
    /// module doc): every slot past 0 back to its reset (slot 0 is fresh or
    /// cleared by the arm list) and, past the first arm, the residency to
    /// its seed, its record before the arm's; the arm's records; slot j's
    /// window through the plain run's feed (`mode`, `call` the printed
    /// plan's needs); in graph mode the pass of a row a slot captured; then
    /// the rounds, and every line after them.
    fn arm_slots(
        s: &mut Session<Body>,
        pre: &Prelude<'_>,
        i: usize,
        r: &ArmRun,
        mode: body::PrefillMode,
        call: Option<&CallView>,
    ) -> Result<(), GateError> {
        let (n, a) = (pre.slots, &r.a);
        if let Some(c) = gen_slots::fresh(s, n, i > 0)? {
            record::residency_reset(&c).print();
        }
        let view = pre.arm(i, r)?;
        let call = view.as_ref().or(call);
        let windows = gen_slots::windows(&r.ids, n, "the arm's fed ids")?;
        let w = windows[0].len();
        let mut first = Vec::with_capacity(n);
        let mut pos0 = Vec::with_capacity(n);
        let mut feeds = Vec::with_capacity(n);
        for (j, ids) in windows.into_iter().enumerate() {
            s.select_slot(j)?;
            let f = Fed {
                ids,
                prompt_len: r.prompt_len.saturating_sub(j * w).min(w),
                mode,
                need: call.map(|c| &c.need),
                base: s.model().body("generate_ds41")?.hybrid().stats(),
                serve: false,
            };
            let (tok, time) = feed(s, &mut runtime::Plain, &f, "steps")?;
            first.push(tok);
            pos0.push(s.pos());
            feeds.push(time);
        }
        s.select_slot(0)?;
        if a.mode == StepMode::Graph && a.n_gen > 1 {
            gen_slots::capture(s, n)?;
        }
        let rounds = gen_slots::rounds(s, first, a.n_gen)?;
        let warm = a.warm.unwrap_or(0);
        for t in &feeds {
            t.print();
        }
        for (k, &ms) in rounds.walls.iter().enumerate() {
            let i = k + 1;
            for (&p, ids) in pos0.iter().zip(&rounds.ids) {
                print_step(i, p - 1 + u32::try_from(i)?, ids[i]);
            }
            if a.timed {
                Record::new(&record::TIME_PASS)
                    .u("i", i)
                    .flag("warm", i <= warm)
                    .f("ms", ms)
                    .u("positions", n)
                    .w("kind", "slots")
                    .print();
            }
        }
        for ids in &rounds.ids {
            Record::new(&record::TOKENS).list("tokens", ids).print();
        }
        if a.timed {
            let c = gen_slots::counted(&rounds.walls, warm, n)?;
            smoke(a, r.prompt_len.min(w), w, warm, c.rounds, c.p50, c.mean)
                .u("positions", c.positions)
                .f("tok/s(positions)", c.aggregate)
                .print();
        }
        Ok(())
    }

    /// One arm of the run: its arguments (its own `-n`), its fed ids, how
    /// many of them are the prompt's, and the `--arm` it came from (none for
    /// a run of the prompt flags).
    #[derive(Clone)]
    struct ArmRun {
        a: Args,
        ids: Vec<u32>,
        prompt_len: usize,
        spec: Option<ArmSpec>,
    }

    /// The run's arms: the `--arm` list, or the one arm the prompt flags
    /// name, `--repeat` times.
    fn arm_runs(a: &Args) -> Result<Vec<ArmRun>, GateError> {
        if a.arms.is_empty() {
            let (ids, prompt_len) = fed_ids(a)?;
            let run = ArmRun {
                a: a.clone(),
                ids,
                prompt_len,
                spec: None,
            };
            let mut runs = vec![run; a.diag.repeat];
            if a.diag.dump_table.is_some()
                && let Some(r) = runs.get_mut(1)
            {
                r.a.dump_now = true;
            }
            return Ok(runs);
        }
        a.arms
            .iter()
            .map(|arm| {
                let (ids, prompt_len) = arm_ids(arm)?;
                Ok(ArmRun {
                    a: a.for_arm(arm),
                    ids,
                    prompt_len,
                    spec: Some(*arm),
                })
            })
            .collect()
    }

    impl ArmRun {
        /// The arm's feed under `mode`, with the needs of its call's printed
        /// plan, and the host tier's counters at its start: the zero its
        /// since-load counters are read from.
        fn fed<'a>(
            &'a self,
            mode: body::PrefillMode,
            call: Option<&'a CallView>,
            s: &Session<Body>,
        ) -> Result<Fed<'a>, GateError> {
            Ok(Fed {
                ids: &self.ids,
                prompt_len: self.prompt_len,
                mode,
                need: call.map(|c| &c.need),
                base: s.model().body("generate_ds41")?.hybrid().stats(),
                serve: self.a.diag.serve_feed,
            })
        }
    }

    /// What each `--arm` prints before it runs: the `arm` record, and under
    /// a batched feed (`call`) its call's plan; under `sync`, a line read
    /// from stdin after the record.
    struct Prelude<'a> {
        cfg: &'a body::OpenCfg,
        hp: &'a Hparams,
        /// The plan's `ctx_max`: the call's plan is the one the caches hold.
        ctx_max: usize,
        call: bool,
        sync: bool,
        arms: usize,
        /// `BLOOMERY_GEN_SLOTS`: each slot's call runs one window of an arm's
        /// fed ids.
        slots: usize,
    }

    impl Prelude<'_> {
        /// Arm `i`'s records and its pacing; its call's plan, which a run of
        /// the prompt flags printed before the load instead — under several
        /// slots a window's call, which each slot runs.
        fn arm(&self, i: usize, r: &ArmRun) -> Result<Option<CallView>, GateError> {
            let Some(spec) = r.spec else {
                return Ok(None);
            };
            Record::new(&record::ARM)
                .u("i", i)
                .u("arms", self.arms)
                .w("feed", spec.feed_name())
                .u("ids", r.ids.len())
                .u("n", r.a.n_gen)
                .print();
            if self.sync {
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 {
                    return Err(format!(
                        "--arm-sync: stdin closed before arm {i} of {}",
                        self.arms
                    )
                    .into());
                }
            }
            let view = self
                .call
                .then(|| CallView::of(self.cfg, self.hp, self.ctx_max, r.ids.len() / self.slots));
            if let Some(c) = &view {
                c.print(false, self.hp, &[]);
            }
            Ok(view)
        }
    }

    /// What a run feeds before its first generated token: the ids, how many
    /// of them are the prompt's, the feed's mode, on a planned batched feed
    /// the needs its printed plan gives the call, and the host tier's
    /// counters the arm starts from.
    struct Fed<'a> {
        ids: &'a [u32],
        prompt_len: usize,
        mode: body::PrefillMode,
        need: Option<&'a body::Need>,
        /// The host tier's counters at the arm's start.
        base: HybridStats,
        /// `--serve-feed`: the call runs every id but the last.
        serve: bool,
    }

    /// `--plan` prints a batched call's plan and runs nothing: refused with
    /// `--time`, beside the DSpark draft, whose feed widens the tapped
    /// layers' blocks, and beside a feed of steps, which runs no batch.
    fn refuse_plan(a: &Args, draft: Draft, batched: bool) -> Result<(), GateError> {
        let beside = [
            (a.timed, "--time: it runs nothing to time"),
            (
                draft == Draft::Dspark,
                "BLOOMERY_DRAFT=dspark: its feed widens the tapped layers' blocks",
            ),
            (
                !batched,
                "a feed of one step per id (BLOOMERY_PREFILL=steps or BLOOMERY_CHECK_FINITE=1): \
                 it runs no batch",
            ),
        ];
        match beside.iter().find(|(set, _)| *set) {
            Some((_, why)) => Err(format!("--plan is refused with {why}").into()),
            None => Ok(()),
        }
    }

    /// A prompt call's plan as the `call` records print it: its batches,
    /// their chunks, the triangle's needs, the groups under every group
    /// lever, and each layer-batch's starts and sub-blocks — read off
    /// `body::BodyLevers::call_plan`, the functions the call's enqueue runs
    /// by.
    struct CallView {
        group: usize,
        ring: usize,
        ced: body::CedState,
        batches: Vec<Range<usize>>,
        cuts: Vec<Vec<Range<usize>>>,
        need: body::Need,
        /// Per group lever, from 1: the groups, as ranges of batches.
        groups: Vec<(usize, Vec<Range<usize>>)>,
        /// Per batch, per layer index.
        lbs: Vec<Vec<LbView>>,
    }

    /// One layer-batch of a [`CallView`]: the chunks its latent part and its
    /// block start at, and the sub-blocks the projections run over each.
    struct LbView {
        run: usize,
        full: usize,
        part: Vec<Range<usize>>,
        block: Vec<Range<usize>>,
    }

    impl CallView {
        /// The plan of a call of `n` ids from position 0 on the body `cfg`
        /// opens from `hp` with caches of `ctx_max` positions.
        fn of(cfg: &body::OpenCfg, hp: &Hparams, ctx_max: usize, n: usize) -> CallView {
            let p = cfg.body.call_plan(hp, ctx_max, 0, n);
            let lbs = (0..p.batches.len())
                .map(|b| {
                    (0..p.need.layers.len())
                        .map(|i| {
                            let (run, full) = p.starts(b, i);
                            LbView {
                                run,
                                full,
                                part: p.sub_blocks(b, run, full),
                                block: p.sub_blocks(b, full, usize::MAX),
                            }
                        })
                        .collect()
                })
                .collect();
            let groups = p.every_group();
            CallView {
                group: p.group,
                ring: p.ring,
                ced: p.ced,
                groups,
                lbs,
                batches: p.batches,
                cuts: p.cuts,
                need: p.need,
            }
        }

        /// The groups under the plan's own lever.
        fn own_groups(&self) -> &[Range<usize>] {
            self.groups
                .iter()
                .find(|(g, _)| *g == self.group)
                .map_or(&[], |(_, v)| v.as_slice())
        }

        /// The `call plan` record and a `call batch` record per batch; under
        /// `whole` (`--plan`) also every lever's groups, every layer's facts
        /// with its card's experts `n_l`, the needs and every layer-batch.
        fn print(&self, whole: bool, hp: &Hparams, n_l: &[u64]) {
            let layers = self.need.layers.len();
            Record::new(&record::CALL_PLAN)
                .u("first", self.need.first)
                .u("end", self.need.end)
                .u("batches", self.batches.len())
                .u("group", self.group)
                .u("groups", self.own_groups().len())
                .u("t_max", body::T_MAX)
                .u("chunk", body::CHUNK)
                .u("sub_chunks", SUB_TOKENS / body::CHUNK)
                .u("ring", self.ring)
                .u("layers", layers)
                .w("ced", self.ced)
                .print();
            if whole {
                for (g, groups) in &self.groups {
                    Record::new(&record::CALL_GROUPS)
                        .u("g", g)
                        .csv("sizes", groups.iter().map(ExactSizeIterator::len))
                        .print();
                }
            }
            for (gi, g) in self.own_groups().iter().enumerate() {
                for b in g.clone() {
                    let (r, cuts) = (&self.batches[b], &self.cuts[b]);
                    Record::new(&record::CALL_BATCH)
                        .u("b", b)
                        .u("first", r.start)
                        .u("end", r.end)
                        .u("group", gi)
                        .u("set", b - g.start)
                        .u("chunks", cuts.len())
                        .csv(
                            "cuts",
                            cuts.iter()
                                .map(|c| c.start)
                                .chain(cuts.last().map(|c| c.end)),
                        )
                        .print();
                }
            }
            if !whole {
                return;
            }
            for (l, k) in hp.layers.iter().enumerate() {
                Record::new(&record::CALL_LAYER)
                    .u("l", l)
                    .u("ratio", k.ratio())
                    .w("compressor", k.compressor.is_some())
                    .w("gated", k.compressor.is_some_and(|c| c.gated))
                    .w("index_keys", k.index_keys)
                    .w("indexer", k.indexer)
                    .w("engram", k.engram.is_some())
                    .u("card_experts", n_l[l])
                    .print();
            }
            let (block, part) = self.need.counts();
            Record::new(&record::CALL_NEED)
                .u("features_from", self.need.features)
                .u("block_positions", block)
                .u("part_positions", part)
                .csv("full_from", self.need.layers.iter().map(|n| n.full))
                .csv("part_from", self.need.layers.iter().map(|n| n.part))
                .print();
            let sizes =
                |sbs: &[Range<usize>]| sbs.iter().map(ExactSizeIterator::len).collect::<Vec<_>>();
            for (b, row) in self.lbs.iter().enumerate() {
                for (i, lb) in row.iter().enumerate() {
                    Record::new(&record::CALL_LB)
                        .u("b", b)
                        .u("layer", i)
                        .u("run", lb.run)
                        .u("full", lb.full)
                        .csv("part_sb", sizes(&lb.part))
                        .csv("full_sb", sizes(&lb.block))
                        .print();
                }
            }
        }
    }

    /// `BLOOMERY_CHECK_FINITE` as the levers hold it, refused beside `--time`,
    /// the draft and the step stats.
    fn finite_lever(a: &Args, draft: Draft, levers: &Levers) -> Result<bool, GateError> {
        let on = levers.check_finite();
        let beside = [
            (
                a.timed,
                "--time: the probe's eager step would sit between two timed steps",
            ),
            (
                draft != Draft::Off,
                "BLOOMERY_DRAFT: the probe reads one-row steps",
            ),
            (
                a.stats,
                "BLOOMERY_STEP_STATS=1: the host tier's counters would count the probe's step too",
            ),
        ];
        match beside.iter().find(|(set, _)| on && *set) {
            Some((_, why)) => Err(format!("BLOOMERY_CHECK_FINITE=1 is refused with {why}").into()),
            None => Ok(on),
        }
    }

    /// The finite probe's own head and what each checked position read.
    struct FiniteCheck {
        head: Head,
        rows: Vec<FiniteRow>,
    }

    /// One checked position: the probe's reading of it and the engine's
    /// token after it.
    struct FiniteRow {
        pos: u32,
        observed: finite::Observed,
        stepped: u32,
    }

    impl FiniteRow {
        /// The probe's reading, and the eager argmax where it is not the
        /// engine's token.
        fn describe(&self) -> String {
            let o = &self.observed;
            let mut line = o.describe();
            if o.token() != self.stepped {
                line.push_str(&format!(
                    " eager_differs: observed {} engine {}",
                    o.token(),
                    self.stepped
                ));
            }
            line
        }
    }

    /// One position through the probe, then the engine: `tok`'s step at the
    /// model's position observed eagerly, the position taken back, and `tok`
    /// stepped through the session, whose token comes back.
    fn checked_step(
        s: &mut Session<Body>,
        c: &mut FiniteCheck,
        tok: u32,
    ) -> Result<u32, SessionError> {
        let pos = s.pos();
        let observed =
            finite::observed_step(s.model_mut(), &mut c.head, tok, pos, &mut |_, _, _| Ok(()))
                .map_err(SessionError::Caller)?;
        s.model_mut().rollback(pos)?;
        let stepped = s.step(tok, Want::Argmax)?.argmax();
        c.rows.push(FiniteRow {
            pos,
            observed,
            stepped,
        });
        Ok(stepped)
    }

    /// The plain advance under the finite probe: every fed id and every
    /// generated token through [`checked_step`].
    struct Checked<'c> {
        c: &'c mut FiniteCheck,
    }

    impl Advance<Session<Body>> for Checked<'_> {
        fn prompt(&mut self, t: &mut Session<Body>, ids: &[u32]) -> Result<u32, SessionError> {
            let mut next = 0;
            for &id in ids {
                next = checked_step(t, self.c, id)?;
            }
            Ok(next)
        }

        fn begin(&mut self, _t: &Session<Body>, _p: &[u32], _f: u32) -> Result<(), SessionError> {
            Ok(())
        }

        fn pass(
            &mut self,
            t: &mut Session<Body>,
            last: u32,
            out: &mut Vec<u32>,
        ) -> Result<Committed, SessionError> {
            let pos = t.pos();
            out.push(checked_step(t, self.c, last)?);
            Ok(Committed {
                pos,
                kept: 1,
                rows: 1,
                proposed: false,
            })
        }
    }

    /// The `stat finite` lines: one per generated step (the rows past the
    /// `fed` fed positions), one per fed position that is not `ok`, and the
    /// summary.
    fn print_finite(c: &FiniteCheck, fed: usize) {
        for (k, r) in c.rows.iter().enumerate() {
            if k < fed {
                if r.observed.first_nonfinite().is_some() || r.observed.token() != r.stepped {
                    Record::new(&record::STAT_FINITE_FED)
                        .u("pos", r.pos)
                        .w("observed", r.describe())
                        .print();
                }
            } else {
                Record::new(&record::STAT_FINITE_STEP)
                    .u("step", k + 1 - fed)
                    .u("pos", r.pos)
                    .w("observed", r.describe())
                    .print();
            }
        }
        let bad: Vec<&FiniteRow> = c
            .rows
            .iter()
            .filter(|r| r.observed.first_nonfinite().is_some())
            .collect();
        let differs = c
            .rows
            .iter()
            .filter(|r| r.observed.token() != r.stepped)
            .count();
        let first = bad.first().map_or_else(
            || "none".to_string(),
            |r| {
                format!(
                    "pos {} {}",
                    r.pos,
                    r.observed
                        .first_nonfinite()
                        .map_or_else(String::new, finite::site_name)
                )
            },
        );
        Record::new(&record::STAT_FINITE_SUMMARY)
            .u("positions", c.rows.len())
            .u("nonfinite_positions", bad.len())
            .w("first", first)
            .u("eager_differs", differs)
            .print();
    }

    /// What the open prints and checks at each of its steps: the plan (and,
    /// on a planned batched feed, the call's plan, which `--plan` stops
    /// after), the load, the captures and the prompt call's buffers.
    struct Log<'a> {
        a: &'a Args,
        cfg: &'a body::OpenCfg,
        draft: Draft,
        batched: bool,
        check_finite: bool,
        /// The fed ids of the arm that steps the most positions; under
        /// several slots, one window of them, a slot's.
        depth: usize,
        /// The positions that arm steps (a slot of it steps): its fed ids
        /// and its N − 1 feedback steps.
        fed: usize,
        /// That arm's `-n`.
        fed_n: usize,
        /// Before the file was opened: the `load` line's `load_s`.
        t: Instant,
        /// The file's headers (`Split::open` and the hparams read): the
        /// phases line's `open_s`.
        headers: Duration,
        /// The inputs read and the placement plan, once `plan` ran: the
        /// phases line's `plan_s`.
        plan: Option<Duration>,
        pin_main: bool,
        pinned: bool,
        /// `BLOOMERY_RESIDENCY` as resolved against the pre-plan, for the
        /// `residency host` record.
        residency: ResidencyPick,
        /// The pre-plan's per-layer card experts, once made: the open's own
        /// plan must equal them ([`Log::plan`]). `None` under `--card-table`,
        /// whose load edits the plan itself.
        plan_n_l: Option<Vec<u64>>,
        /// The hyperparameters the plan was made from.
        hp: Option<Hparams>,
        /// The plan's `ctx_max`, once planned.
        ctx_max: usize,
        call: Option<CallView>,
    }

    impl OpenLog<Body> for Log<'_> {
        /// The `plan` record, the run's positions against the plan's
        /// `ctx_max`, then the call's plan; `--plan` ends the run here.
        fn plan(
            &mut self,
            place: &'static str,
            inputs: &PlanInputs,
            machine: &Machine,
            plan: &Plan<'_>,
        ) -> Result<bool, SessionError> {
            // The inputs read and the placement plan end here; the load
            // follows.
            self.plan = Some(self.t.elapsed() - self.headers);
            let a = self.a;
            record::plan(place, machine, plan).print();
            // The open's plan is the pre-plan the unset word resolved
            // against; a load whose plan moved under it names itself.
            if let Some(pre) = &self.plan_n_l
                && plan.n_l != *pre
            {
                return Err(SessionError::Refused(format!(
                    "the plan the residency resolved on holds per-layer card experts {pre:?}; \
                     the open's holds {n_l:?}",
                    n_l = plan.n_l
                )));
            }
            let residency = self.cfg.body.residency;
            if let Some(pool) = swap::churn(plan, 0, residency)? {
                record::residency_host(self.residency.word, &pool, plan).print();
            }
            // The caches hold the plan's ctx_max positions, the value the
            // model is loaded with; --ctx only asks for it.
            let ctx_max = usize::try_from(plan.ctx_max).map_err(|_| {
                SessionError::Refused(format!("the plan's ctx_max {} passes usize", plan.ctx_max))
            })?;
            let (depth, fed) = (self.depth, self.fed);
            if fed > ctx_max {
                return Err(SessionError::Refused(format!(
                    "depth {depth} + {} fed tokens exceed the plan's ctx_max {ctx_max} \
                     (--ctx {})",
                    self.fed_n - 1,
                    a.ctx
                )));
            }
            if self.draft != Draft::Off && fed + 1 > ctx_max {
                return Err(SessionError::Refused(format!(
                    "BLOOMERY_DRAFT={}: depth {depth} + {} fed tokens and the last pair's \
                     overshoot exceed the plan's ctx_max {ctx_max} (--ctx {})",
                    self.draft.name(),
                    self.fed_n - 1,
                    a.ctx
                )));
            }
            // The DSpark feed's taps widen the tapped layers' blocks: its call
            // is not the plain call's plan. An `--arm` list prints each arm's
            // call before it runs.
            // Under `--serve-feed` the call runs every id but the last.
            let called = depth - usize::from(a.diag.serve_feed);
            let call = (self.batched && self.draft != Draft::Dspark && a.arms.is_empty())
                .then(|| CallView::of(self.cfg, &inputs.hp, ctx_max, called));
            if let Some(c) = &call {
                c.print(a.plan, &inputs.hp, &plan.n_l);
            }
            self.hp = Some(inputs.hp.clone());
            self.ctx_max = ctx_max;
            self.call = call;
            Ok(!a.plan)
        }

        /// The body's selection checked against the file's, then the `load`
        /// record, the load's phases and the host set's.
        fn load(&mut self, m: &Deepseek41Model) -> Result<(), SessionError> {
            let a = self.a;
            let hp = self
                .hp
                .as_ref()
                .ok_or_else(|| SessionError::Refused("a load with no plan".into()))?;
            let b = m.body("generate_ds41")?;
            let top_k = b.indexer_top_k();
            let shadow = b.shadow_host();
            if top_k != hp.indexer.top_k {
                return Err(SessionError::Refused(format!(
                    "the body selects {top_k} rows per stream, the file's top_k is {}: a step \
                     past that many visible rows would not be the model's",
                    hp.indexer.top_k
                )));
            }
            let load = Record::new(&record::LOAD)
                .u("resident_bytes", m.resident_bytes())
                .w("shadow", "host")
                .u("shadow_bytes", shadow.bytes)
                .u("unified_addressing", shadow.unified_addressing);
            let load = place::with_cards(m, a.place, "generate_ds41", load)
                .map_err(|e| SessionError::Refused(e.to_string()))?;
            let total = self.t.elapsed();
            load.u("ctx", a.ctx)
                .u("layers", hp.n_layer)
                .u("top_k", top_k)
                .w("mode", mode_name(a.mode))
                .w("place", a.place.name())
                .w("pin_main", if self.pin_main { "on" } else { "off" })
                .w("pinned", self.pinned)
                .w(
                    "prefill",
                    if self.check_finite {
                        "steps"
                    } else {
                        b.prefill_mode().name()
                    },
                )
                .w("ced", b.ced())
                .u("group", b.prefill_group_lever())
                .f("load_s", total.as_secs_f64())
                .print();
            if let Some(times) = m.load_times() {
                record::load_phases(
                    total.as_secs_f64(),
                    self.headers.as_secs_f64(),
                    self.plan.map(|d| d.as_secs_f64()),
                    times.context.as_secs_f64(),
                    times.upload.as_secs_f64(),
                    times.derive.as_secs_f64(),
                    times.host_set.as_secs_f64(),
                    times.body.as_secs_f64(),
                    times.head.as_secs_f64(),
                )
                .print();
            }
            if let Some(h) = b.hybrid().residency() {
                for r in record::host_residency(h) {
                    r.print();
                }
            }
            for h in threads::helper::helpers() {
                record::helper(&h.name, h.asked, h.pinned, h.cpus).print();
            }
            Ok(())
        }

        fn capture(&mut self, nodes: usize) -> Result<(), SessionError> {
            Record::new(&record::CAPTURE)
                .u("graph_nodes", nodes)
                .print();
            Ok(())
        }

        fn prompt_buffers(&mut self, m: &Deepseek41Model) -> Result<(), SessionError> {
            let body = m.body("generate_ds41")?;
            let (group, group_bytes) = body.prefill_group().unwrap_or_default();
            Record::new(&record::PREFILL_BYTES)
                .u("batch_bytes", body.batch_bytes())
                .u("proj_bytes", body.batch_proj_bytes())
                .u("group", group)
                .u("group_bytes", group_bytes)
                .print();
            Ok(())
        }
    }

    impl RowsLog for Log<'_> {
        /// The pair pass's capture: the one verify this body runs.
        fn capture_rows(&mut self, rows: usize, nodes: usize) -> Result<(), SessionError> {
            if rows != PAIR_ROWS {
                return Err(SessionError::Refused(format!(
                    "a verify capture of {rows} rows; the pair pass runs {PAIR_ROWS}"
                )));
            }
            Record::new(&record::CAPTURE_PAIR)
                .u("pair_graph_nodes", nodes)
                .print();
            Ok(())
        }
    }

    /// The host tier's batch services since `base`, the arm's start, one
    /// line: the layers served, the columns and host slots they carried, and the union calls'
    /// wall — the part of a batched feed the card waits on the host; then
    /// the last call's needs and its split, then its host-streaming picks
    /// and end when it streamed.
    fn print_union(m: &mut Deepseek41Model, base: &HybridStats) -> Result<(), GateError> {
        let stats = m.body_parts("generate_ds41")?.2.take_prefill_stats();
        let b = m.body("generate_ds41")?;
        let s = b.hybrid().stats();
        Record::new(&record::STAT_PREFILL)
            .u("union_layers", s.batch_served - base.batch_served)
            .u("union_cols", s.batch_cols - base.batch_cols)
            .u(
                "union_host_slots",
                s.batch_host_slots - base.batch_host_slots,
            )
            .f("union_ms", (s.batch_ns - base.batch_ns) as f64 / 1e6)
            .print();
        if let Some(need) = b.prefill_need() {
            let (block, part) = need.counts();
            Record::new(&record::STAT_PREFILL_CED)
                .w("ced", b.ced())
                .u("first", need.first)
                .u("end", need.end)
                .u("features_from", need.features)
                .u("block_positions", block)
                .u("part_positions", part)
                .csv("full_from", need.layers.iter().map(|n| n.full))
                .csv("part_from", need.layers.iter().map(|n| n.part))
                .print();
        }
        split::split(&stats, b.prefill_counts())?.print();
        let (picks, end) = m.body_parts("generate_ds41")?.2.take_stream_records();
        for (g, p) in &picks {
            record::call_stream(*g, p).print();
        }
        if let Some(r) = end {
            record::call_report(&r).print();
        }
        Ok(())
    }

    /// `BLOOMERY_STEP_STATS`: the last prompt call's queue entries, a record
    /// per batch's first steps and per layer-batch
    /// (`body::Body::prefill_counts`), after the loop.
    fn print_counts(m: &Deepseek41Model) -> Result<(), GateError> {
        if let Some(c) = m.body("generate_ds41")?.prefill_counts() {
            for f in &c.front {
                Record::new(&record::STAT_PREFILL_FRONT)
                    .u("b", f.batch)
                    .u("entries", f.entries)
                    .print();
            }
            let ms = |ns: u64| ns as f64 / 1e6;
            for lb in &c.lbs {
                let mut r = Record::new(&record::STAT_PREFILL_LB)
                    .u("b", lb.batch)
                    .u("layer", lb.layer)
                    .w("block", lb.block)
                    .u("entries_route", lb.route)
                    .u("entries_shadow", lb.shadow);
                // The times only under card timing, as the split line's.
                if let Some(out) = lb.card_out_ms {
                    r = r.f("card_out_ms", out);
                    if let Some(shadow) = lb.card_in_ms {
                        r = r.f("card_in_ms", shadow);
                    }
                    r = r
                        .f("union_ms", ms(lb.union_ns))
                        .f("wait_ms", ms(lb.wait_ns));
                }
                r.print();
            }
        }
        Ok(())
    }

    /// The prompt feed's wall and shape, the `time prompt` row.
    struct FeedTime {
        /// The fed ids.
        n: usize,
        /// The steps the feed took.
        passes: usize,
        /// Before the first fed step to after the readback of generated
        /// token 0.
        wall: Duration,
        /// `steps`, `dspark`, or `checked` under the finite probe.
        kind: &'static str,
    }

    impl FeedTime {
        /// `time prompt n= ms= tok/s= passes= kind=`, written after the loop.
        fn print(&self) {
            let ms = self.wall.as_secs_f64() * 1e3;
            Record::new(&record::TIME_PROMPT)
                .u("n", self.n)
                .f("ms", ms)
                .f("tok/s", self.n as f64 * 1e3 / ms)
                .u("passes", self.passes)
                .w("kind", self.kind)
                .print();
        }
    }

    /// The `fed` record: how many ids, the first and last four, and where
    /// the depth sequence starts.
    fn print_fed(ids: &[u32], prompt_len: usize) {
        let depth = ids.len();
        let head: Vec<u32> = ids.iter().copied().take(4).collect();
        let tail: Vec<u32> = ids.iter().copied().skip(depth.saturating_sub(4)).collect();
        Record::new(&record::FED)
            .u("ids", depth)
            .list("first", &head)
            .list("last", &tail)
            .u("depth_sequence_from", prompt_len)
            .print();
    }

    /// The `step 0` record: generated token 0, the position it follows, and
    /// the feed's wall.
    fn print_step0(pos: u32, token: u32, fed: usize, wall: Duration) {
        Record::new(&record::STEP0)
            .u("pos", pos)
            .u("token", token)
            .u("fed", fed)
            .f("feed_s", wall.as_secs_f64())
            .print();
    }

    /// A planned call's needs against its printed plan's: the call the
    /// `call` records describe is the one that ran, or the run stops.
    fn check_plan(m: &Deepseek41Model, need: Option<&body::Need>) -> Result<(), GateError> {
        match need {
            Some(want) if m.body("generate_ds41")?.prefill_need() != Some(want) => Err(
                "generate_ds41: the prompt call ran other needs than its printed plan's \
                 (`call` records)"
                    .into(),
            ),
            _ => Ok(()),
        }
    }

    /// Feed `f.ids` through `adv`'s prompt — the session's prompt schedule
    /// (in batches under `f.mode` `Batch`, else one real step per id), the
    /// draft's tapped call, or the finite probe's steps — print the `fed` and
    /// `step 0` lines, and return the first generated token and the feed's
    /// wall. `steps` is the `time prompt` row's kind when the feed is not a
    /// batch.
    fn feed<A: Advance<Session<Body>>>(
        s: &mut Session<Body>,
        adv: &mut A,
        f: &Fed<'_>,
        steps: &'static str,
    ) -> Result<(u32, FeedTime), GateError> {
        let ids = f.ids;
        let depth = ids.len();
        print_fed(ids, f.prompt_len);
        let batch = f.mode == body::PrefillMode::Batch;
        let t = Instant::now();
        let next = adv.prompt(s, ids)?;
        let wall = t.elapsed();
        print_step0(s.pos() - 1, next, depth, wall);
        if batch {
            check_plan(s.model(), f.need)?;
            print_union(s.model_mut(), &f.base)?;
        }
        let called = if f.serve { depth - 1 } else { depth };
        let time = FeedTime {
            n: depth,
            passes: if batch {
                body::batch_count(called) + usize::from(f.serve)
            } else {
                depth
            },
            wall,
            kind: if batch { "batch" } else { steps },
        };
        Ok((next, time))
    }

    /// The stop rule of `-n` on `s`: `-n` tokens, generated token 0 counted.
    fn stop_at_n(s: &Session<Body>, a: &Args) -> Result<Stop, GateError> {
        Ok(Stop::new(a.n_gen, s.ctx())?)
    }

    /// A generation that ended by its count: the run checked the context
    /// before the load and names no end-of-generation id.
    fn ran_to_n(out: &GenOutcome) -> Result<(), GateError> {
        match out.stop {
            StopReason::Length => Ok(()),
            other => Err(format!(
                "generate_ds41: the generation stopped at {} after {} tokens, before -n",
                other.name(),
                out.tokens.len()
            )
            .into()),
        }
    }

    /// Each plain step's position, token and wall, the stats probes, and
    /// the rows `--top2` reads ([`TopRows`]).
    struct Steps {
        stats: bool,
        rows: Vec<(u32, u32, f64)>,
        probes: Vec<Probe>,
        n_gen: usize,
        tops: TopRows,
    }

    impl PassSink<Session<Body>> for Steps {
        type Error = GateError;

        fn begin(&mut self, t: &Session<Body>) -> Result<(), GateError> {
            if self.stats {
                self.probes.reserve_exact(self.n_gen);
                self.probes
                    .push(Probe::read(t.model(), self.probes.last())?);
            }
            Ok(())
        }

        fn pass(
            &mut self,
            t: &Session<Body>,
            c: &Committed,
            tokens: &[u32],
            wall: Duration,
        ) -> Result<(), GateError> {
            self.rows.push((c.pos, tokens[0], wall.as_secs_f64() * 1e3));
            if self.tops.wants() {
                self.tops.read(&t.model().logits()?, tokens[0])?;
            }
            if self.stats {
                self.probes
                    .push(Probe::read(t.model(), self.probes.last())?);
            }
            Ok(())
        }
    }

    /// Feed `f`, then the N − 1 feedback steps through `adv` (one step a
    /// pass), timed when asked; every line after the loop, `after` at the
    /// finite probe's place among them.
    fn decode<A: Advance<Session<Body>>>(
        s: &mut Session<Body>,
        a: &Args,
        f: &Fed<'_>,
        adv: &mut A,
        steps: &'static str,
        after: impl FnOnce(&A),
    ) -> Result<(), GateError> {
        let depth = f.ids.len();
        let (first, feed_time) = if a.diag.serve_feed {
            feed(s, &mut ServeFeed { inner: adv }, f, steps)?
        } else {
            feed(s, adv, f, steps)?
        };
        if let (true, Some(p)) = (a.dump_now, &a.diag.dump_table) {
            dump_table(&residence(s)?, p)?.print();
        }
        let mut sink = Steps {
            stats: a.stats,
            rows: Vec::with_capacity(a.n_gen - 1),
            probes: Vec::new(),
            n_gen: a.n_gen,
            tops: TopRows::new(&a.diag),
        };
        if sink.tops.wants() {
            sink.tops.read(&s.model().logits()?, first)?;
        }
        let stop = stop_at_n(s, a)?;
        let out = runtime::generate(s, adv, f.ids, first, &stop, &mut sink)?;
        ran_to_n(&out)?;
        let rows = sink.rows;
        let warm = a.warm.unwrap_or(0);
        feed_time.print();
        for (k, &(pos, tok, ms)) in rows.iter().enumerate() {
            let i = k + 1;
            print_step(i, pos, tok);
            if a.timed {
                Record::new(&record::TIME_STEP)
                    .u("i", i)
                    .flag("warm", i <= warm)
                    .f("ms", ms)
                    .print();
            }
        }
        Record::new(&record::TOKENS)
            .list("tokens", &out.tokens)
            .print();
        if a.logits {
            print_logits(s.model())?;
        }
        sink.tops.finish(a.diag.rows.as_deref())?;
        if a.stats {
            print_counts(s.model())?;
            print_stats(&sink.probes, warm);
        }
        after(adv);
        if a.timed {
            let counted: Vec<f64> = rows[warm..].iter().map(|r| r.2).collect();
            let mut sorted = counted.clone();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = counted.iter().sum::<f64>() / counted.len() as f64;
            smoke(a, f.prompt_len, depth, warm, counted.len(), p50, mean).print();
        }
        Ok(())
    }

    /// The `logits` record: the head's last logits row, its length, argmax
    /// and FNV-1a 64 of its f32 bits in order.
    fn print_logits(m: &Deepseek41Model) -> Result<(), GateError> {
        let row = m.logits()?;
        let argmax = row
            .iter()
            .enumerate()
            .max_by(|x, y| x.1.total_cmp(y.1).then(y.0.cmp(&x.0)))
            .map_or(0, |(i, _)| i);
        let fnv = Fnv1a64::default().f32s(&row).value();
        Record::new(&record::LOGITS)
            .u("n", row.len())
            .u("argmax", argmax)
            .w("fnv64", format!("{fnv:016x}"))
            .print();
        Ok(())
    }

    /// The `step` record: generated token `i`, and the position it is the
    /// argmax after.
    fn print_step(i: usize, pos: u32, token: u32) {
        Record::new(&record::STEP)
            .u("i", i)
            .u("pos", pos)
            .u("token", token)
            .print();
    }

    /// The `SMOKE` record of `--time`: the kept steps' (or passes') p50 and
    /// mean, and the rate at the p50; a draft adds its positions.
    fn smoke(
        a: &Args,
        prompt_len: usize,
        depth: usize,
        warm: usize,
        steps: usize,
        p50: f64,
        mean: f64,
    ) -> Record {
        Record::new(&record::SMOKE)
            .w("mode", mode_name(a.mode))
            .w("place", a.place.name())
            .u("prompt_tokens", prompt_len)
            .u("depth", depth)
            .u("generated", a.n_gen)
            .u("warm", warm)
            .u("steps", steps)
            .f("p50_ms", p50)
            .f("mean_ms", mean)
            .f("tok/s(p50)", 1e3 / p50)
    }

    /// What one draft pass ran, and so how many positions it advanced.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum PassKind {
        /// No proposal: one `step`.
        Plain,
        /// Row A's argmax was the draft: both rows' tokens kept.
        Accept,
        /// Row A's argmax was not the draft: the second position taken back.
        Reject,
    }

    impl PassKind {
        /// What a pass of the pair's draft ran: no proposal is one step, a
        /// proposal whose rows were all kept an accept.
        fn of(c: &Committed) -> PassKind {
            match (c.proposed, c.kept == c.rows) {
                (false, _) => PassKind::Plain,
                (true, true) => PassKind::Accept,
                (true, false) => PassKind::Reject,
            }
        }

        fn name(self) -> &'static str {
            match self {
                PassKind::Plain => "plain",
                PassKind::Accept => "pair-accept",
                PassKind::Reject => "pair-reject",
            }
        }

        fn positions(self) -> usize {
            if self == PassKind::Accept { 2 } else { 1 }
        }
    }

    /// Each draft pass's kept tokens at their positions, its kind and wall,
    /// and the stats probes.
    struct Passes {
        stats: bool,
        /// (position the token is the argmax after, token), past token 0.
        emitted: Vec<(u32, u32)>,
        passes: Vec<(PassKind, f64)>,
        probes: Vec<Probe>,
        n_gen: usize,
    }

    impl PassSink<Session<Body>> for Passes {
        type Error = GateError;

        fn begin(&mut self, t: &Session<Body>) -> Result<(), GateError> {
            if self.stats {
                self.probes.reserve_exact(self.n_gen);
                self.probes
                    .push(Probe::read(t.model(), self.probes.last())?);
            }
            Ok(())
        }

        fn pass(
            &mut self,
            t: &Session<Body>,
            c: &Committed,
            tokens: &[u32],
            wall: Duration,
        ) -> Result<(), GateError> {
            for (r, &tok) in (0u32..).zip(tokens) {
                self.emitted.push((c.pos + r, tok));
            }
            self.passes
                .push((PassKind::of(c), wall.as_secs_f64() * 1e3));
            if self.stats {
                self.probes
                    .push(Probe::read(t.model(), self.probes.last())?);
            }
            Ok(())
        }
    }

    /// `decode` under `BLOOMERY_DRAFT`: passes of `spec` until `-n` tokens are
    /// out, each timed around the proposal, the pair pass and the draft's
    /// update — the lookup's push, or the DSpark draft's features. The draft
    /// takes the fed ids by its own prompt (the DSpark draft every fed
    /// position's features, from the batches' feature rows or one step per
    /// id); `after` runs once the passes are done. `steps` is the `time
    /// prompt` row's kind of a feed that is not a batch, `kind` the `draft
    /// summary`'s.
    fn decode_draft<D: runtime::Draft<Session<Body>>>(
        s: &mut Session<Body>,
        a: &Args,
        f: &Fed<'_>,
        spec: &mut Speculative<Choosing<D>, PAIR_ROWS>,
        steps: &'static str,
        kind: &'static str,
        after: impl FnOnce(&mut Speculative<Choosing<D>, PAIR_ROWS>) -> Result<(), GateError>,
    ) -> Result<(), GateError> {
        let depth = f.ids.len();
        let (first, feed_time) = feed(s, spec, f, steps)?;
        let mut sink = Passes {
            stats: a.stats,
            emitted: Vec::with_capacity(a.n_gen),
            passes: Vec::with_capacity(a.n_gen - 1),
            probes: Vec::new(),
            n_gen: a.n_gen,
        };
        let stop = stop_at_n(s, a)?;
        let out = runtime::generate(s, spec, f.ids, first, &stop, &mut sink)?;
        ran_to_n(&out)?;
        after(spec)?;
        let passes = sink.passes;
        let warm = a.warm.unwrap_or(0);
        if warm >= passes.len() {
            return Err(format!(
                "--warm {warm} leaves no timed pass of the {} this run took",
                passes.len()
            )
            .into());
        }
        feed_time.print();
        print_draft_rows(a, first, &sink.emitted, &passes);
        if a.logits {
            print_logits(s.model())?;
        }
        if a.stats {
            print_counts(s.model())?;
            print_stats(&sink.probes, warm);
        }
        print_draft_summary(a, &passes, f.prompt_len, depth, kind, spec.draft_mut());
        Ok(())
    }

    /// The `step` lines of the first `-n` tokens, the `time pass` rows when
    /// timed, and the `tokens` line, capped at `-n`.
    fn print_draft_rows(a: &Args, first: u32, emitted: &[(u32, u32)], passes: &[(PassKind, f64)]) {
        let kept = &emitted[..a.n_gen - 1];
        for (k, &(pos, tok)) in kept.iter().enumerate() {
            print_step(k + 1, pos, tok);
        }
        if a.timed {
            let warm = a.warm.unwrap_or(0);
            for (k, &(kind, ms)) in passes.iter().enumerate() {
                let i = k + 1;
                Record::new(&record::TIME_PASS)
                    .u("i", i)
                    .flag("warm", i <= warm)
                    .f("ms", ms)
                    .u("positions", kind.positions())
                    .w("kind", kind.name())
                    .print();
            }
        }
        let tokens: Vec<u32> = std::iter::once(first)
            .chain(kept.iter().map(|&(_, t)| t))
            .collect();
        Record::new(&record::TOKENS).list("tokens", &tokens).print();
    }

    /// The `draft summary` over every pass (the rate over the passes past
    /// `--warm`) with the width chooser's mode and, under `cost`, its state
    /// at the run's end and the passes it ran closed ([`record::width_gate`]);
    /// then the `SMOKE` footer when timed: `generate`'s keys over the kept
    /// passes, then `positions=` and `tok/s(positions)=`.
    fn print_draft_summary<D>(
        a: &Args,
        passes: &[(PassKind, f64)],
        prompt_len: usize,
        depth: usize,
        kind: &str,
        chooser: &mut Choosing<D>,
    ) {
        let warm = a.warm.unwrap_or(0);
        let proposals = passes.iter().filter(|p| p.0 != PassKind::Plain).count();
        let accepts = passes.iter().filter(|p| p.0 == PassKind::Accept).count();
        let positions: usize = passes.iter().map(|p| p.0.positions()).sum();
        let kept = &passes[warm..];
        let kept_positions: usize = kept.iter().map(|p| p.0.positions()).sum();
        let kept_ms: f64 = kept.iter().map(|p| p.1).sum();
        let rate = kept_positions as f64 * 1e3 / kept_ms;
        let summary = Record::new(&record::DRAFT_SUMMARY)
            .u("proposals", proposals)
            .u("accepts", accepts)
            .u("positions", positions)
            .u("passes", passes.len())
            .f("tok/s(positions)", rate)
            .w("kind", kind)
            .w("width", chooser.mode().word());
        let summary = match chooser.mode() {
            WidthMode::Fixed => summary,
            WidthMode::Cost => {
                let closed = chooser.take_tally().closed;
                record::width_gate(summary, &chooser.gate_state(), closed)
            }
        };
        summary.print();
        if a.timed {
            let mut sorted: Vec<f64> = kept.iter().map(|p| p.1).collect();
            sorted.sort_by(f64::total_cmp);
            let p50 = sorted[sorted.len() / 2];
            let mean = kept_ms / kept.len() as f64;
            smoke(a, prompt_len, depth, warm, kept.len(), p50, mean)
                .u("positions", kept_positions)
                .f("tok/s(positions)", rate)
                .print();
        }
    }

    /// The counters one `BLOOMERY_STEP_STATS` read takes: the host tier's
    /// since load, the engram rows' since load, the process's page faults
    /// since start, and the card's free device bytes now; and where the
    /// engram rows' helper runs (`None` off, `Some(None)` floating).
    #[derive(Clone, Copy)]
    struct Probe {
        hybrid: HybridStats,
        /// The go waits of the services since the probe before.
        gap: bloomery_gpu::host::step::GapSummary,
        params_ns: u64,
        eng: bloomery_gpu_deepseek41::chain::glue::EngramStats,
        eng_helper: Option<Option<usize>>,
        majflt: u64,
        minflt: u64,
        vram_free: u64,
    }

    impl Probe {
        fn read(m: &Deepseek41Model, prev: Option<&Probe>) -> Result<Probe, GateError> {
            let body = m.body("generate_ds41")?;
            let hybrid = body.hybrid().stats();
            let gap = match prev {
                Some(p) => body.hybrid().gap_summary(p.hybrid.gaps, hybrid.gaps)?,
                None => bloomery_gpu::host::step::GapSummary::default(),
            };
            let params_ns = body.params_ns();
            let eng = body.step_rows().engram_stats();
            let eng_helper = body.step_rows().helper_cpu();
            let vram_free = u64::try_from(m.gpu().mem_info()?.0)?;
            // SAFETY: `rusage` is integers only, so all-zero is a valid value,
            // and `getrusage` writes only through the pointer it is given.
            let (rc, ru) = unsafe {
                let mut ru: libc::rusage = std::mem::zeroed();
                let rc = libc::getrusage(libc::RUSAGE_SELF, &raw mut ru);
                (rc, ru)
            };
            if rc != 0 {
                return Err(format!("getrusage: {}", std::io::Error::last_os_error()).into());
            }
            Ok(Probe {
                hybrid,
                gap,
                params_ns,
                eng,
                eng_helper,
                majflt: u64::try_from(ru.ru_majflt)?,
                minflt: u64::try_from(ru.ru_minflt)?,
                vram_free,
            })
        }
    }

    /// One line per generated step from the deltas of `probes` (one read
    /// before the first generated step, one after each), then the summary
    /// over the steps past `warm`. `straggle_max_us` is the worst single
    /// service since load (a maximum has no delta); `host_w2` is the step's
    /// mean host share of routed weight squared per service. `vram_free` is
    /// the read after the step, not a delta. `eng_warm`/`eng_cold` are the
    /// step's engram rows found in the page cache or not before the read,
    /// `eng_direct` the ones the step thread read itself (the helper off),
    /// `eng_wait_us` the step thread's time in the engram read (the wait on
    /// the helper, or the direct copy), `eng_helper_us` the helper's own and
    /// `eng_classify_us` the time the warm/cold count itself took (inside
    /// `eng_wait_us` with the helper on, outside it off). `gap_min_us`,
    /// `gap_p50_us` and `gap_max_us` are the step's go waits, one per
    /// service: the card's time from the host's signal to the next go (the
    /// lower median); `params_us` the host time in the step's synchronous
    /// parameter copy. `overlap` is the
    /// step's host slot ids of a two-row pass's row 1 that row 0 also sent
    /// to the host at the same layer, `union` the distinct host slots of the
    /// step's rows per layer, summed (`host_slots − overlap`: a one-row step
    /// prints `overlap=0 union=<host_slots>`). The summary's `phi_mean` is
    /// the pooled row overlap over the kept steps, `Σ overlap / Σ` row 1's
    /// host slots, 0 when no kept step ran two rows; the summary's `_mean`s
    /// are over the kept steps.
    fn print_stats(probes: &[Probe], warm: usize) {
        let mut legs: Vec<f64> = Vec::with_capacity(probes.len());
        let mut waits: Vec<f64> = Vec::with_capacity(probes.len());
        let (mut eng_warm, mut eng_cold, mut eng_direct) = (0_u64, 0_u64, 0_u64);
        let (mut eng_helper_ns, mut eng_classify_ns) = (0_u64, 0_u64);
        let mut vram_free_min = u64::MAX;
        let (mut straggle_max, mut slots, mut majflt, mut minflt) = (0.0_f64, 0_u64, 0_u64, 0_u64);
        let (mut overlap_sum, mut row1_sum) = (0_u64, 0_u64);
        for (k, w) in probes.windows(2).enumerate() {
            let i = k + 1;
            let (p, q) = (&w[0].hybrid, &w[1].hybrid);
            let served = q.served - p.served;
            let leg_us = (q.leg_ns - p.leg_ns) as f64 / 1e3;
            let straggle_us = (q.straggle_ns - p.straggle_ns) as f64 / 1e3;
            let host_slots = q.host_slots - p.host_slots;
            let overlap = q.overlap_slots - p.overlap_slots;
            let row1 = q.pair_row1_slots - p.pair_row1_slots;
            let union = host_slots - overlap;
            let host_w2 = if served == 0 {
                0.0
            } else {
                (q.host_w2 - p.host_w2) / served as f64
            };
            let dmaj = w[1].majflt - w[0].majflt;
            let dmin = w[1].minflt - w[0].minflt;
            let (e, f) = (&w[0].eng, &w[1].eng);
            let (ew, ec, ed) = (f.warm - e.warm, f.cold - e.cold, f.direct - e.direct);
            let wait_us = (f.wait_ns - e.wait_ns) as f64 / 1e3;
            let helper_ns = f.helper_ns - e.helper_ns;
            let classify_ns = f.classify_ns - e.classify_ns;
            Record::new(&record::STAT_STEP)
                .u("i", i)
                .flag("warm", i <= warm)
                .u("served", served)
                .f("leg_us", leg_us)
                .f("straggle_us", straggle_us)
                .f("straggle_max_us", q.straggle_max_ns as f64 / 1e3)
                .u("host_slots", host_slots)
                .f("host_w2", host_w2)
                .u("overlap", overlap)
                .u("union", union)
                .u("go_early", q.go_early - p.go_early)
                .u("parks", q.parks_in_service - p.parks_in_service)
                .u("majflt", dmaj)
                .u("minflt", dmin)
                .u("vram_free", w[1].vram_free)
                .u("eng_warm", ew)
                .u("eng_cold", ec)
                .u("eng_direct", ed)
                .f("eng_wait_us", wait_us)
                .f("eng_helper_us", helper_ns as f64 / 1e3)
                .f("eng_classify_us", classify_ns as f64 / 1e3)
                .f("gap_min_us", w[1].gap.min_ns as f64 / 1e3)
                .f("gap_p50_us", w[1].gap.p50_ns as f64 / 1e3)
                .f("gap_max_us", w[1].gap.max_ns as f64 / 1e3)
                .f("params_us", (w[1].params_ns - w[0].params_ns) as f64 / 1e3)
                .print();
            if i > warm {
                waits.push(wait_us);
                eng_warm += ew;
                eng_cold += ec;
                eng_direct += ed;
                eng_helper_ns += helper_ns;
                eng_classify_ns += classify_ns;
                vram_free_min = vram_free_min.min(w[1].vram_free);
                legs.push(leg_us);
                straggle_max = straggle_max.max(straggle_us);
                slots += host_slots;
                overlap_sum += overlap;
                row1_sum += row1;
                majflt += dmaj;
                minflt += dmin;
            }
        }
        let n = legs.len();
        if n == 0 {
            return;
        }
        let mean = legs.iter().sum::<f64>() / n as f64;
        legs.sort_by(f64::total_cmp);
        let wait_mean = waits.iter().sum::<f64>() / n as f64;
        waits.sort_by(f64::total_cmp);
        let phi_mean = if row1_sum == 0 {
            0.0
        } else {
            overlap_sum as f64 / row1_sum as f64
        };
        let helper = match probes[0].eng_helper {
            None => "off".to_string(),
            Some(None) => "floating".to_string(),
            Some(Some(cpu)) => format!("cpu{cpu}"),
        };
        Record::new(&record::STAT_SUMMARY)
            .u("steps", n)
            .f("leg_us_mean", mean)
            .f(
                "leg_us_p50",
                bloomery_gpu_gates::host_stats::lower_median(&legs),
            )
            .f("straggle_us_max", straggle_max)
            .f("host_slots_mean", slots as f64 / n as f64)
            .f("phi_mean", phi_mean)
            .u("majflt", majflt)
            .u("minflt", minflt)
            .u("vram_free_load", probes[0].vram_free)
            .u("vram_free_min", vram_free_min)
            .w("eng_helper", helper)
            .u("eng_warm", eng_warm)
            .u("eng_cold", eng_cold)
            .u("eng_direct", eng_direct)
            .f("eng_wait_us_mean", wait_mean)
            .f("eng_wait_us_p50", waits[n / 2])
            .f("eng_wait_us_max", waits[n - 1])
            .f("eng_helper_us_mean", eng_helper_ns as f64 / 1e3 / n as f64)
            .f(
                "eng_classify_us_mean",
                eng_classify_ns as f64 / 1e3 / n as f64,
            )
            .print();
    }
}
