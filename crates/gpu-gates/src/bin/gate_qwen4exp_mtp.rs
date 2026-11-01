//! The Qwen3.8-Flash-Next MTP draft program's gate: the target opened by
//! its placement on the gate card with the shared draft file beside it
//! (`Body38::open_placed_mtp`, the head a list of every sixth id), the
//! draft's load held to its plan, and the draft's walks
//! (`GpuModel::mtp_draft`) held against ik's MTP draft set (refset family
//! `mtp-qwen4exp`) and against themselves.
//!
//! What is asserted:
//! - (d) right after the load, before any walk: the draft is open: its
//!   weights, its store and the row map hold the bytes its plan
//!   (`PlanInputs::plan_mtp`) gives; the two matrices it borrows
//!   (`token_embd`, `output`) are the target's own buffers, at the target's
//!   addresses — nothing copied, the list's rows among them; the program's
//!   arena is the plan's; the map holds one word a vocabulary id, its first
//!   the list's ids and the rest 0.
//!   Mixed BF16/F32 router parts in either order are planned, uploaded and
//!   joined as F32, with the joined values and resident bytes checked.
//! - (u) before the load: the head `BLOOMERY_MTP_HEAD_ROWS` unset picks on
//!   this target (`head_list::head_rows_of`) is the shipped list — its
//!   76,000 rows, its first line naming the target tokenizer's digest, the
//!   pick `shipped` — and `full` is the full head.
//! - (q) `Mtp38::open` refuses by name, beside the loaded model: a draft
//!   read as carrying its own matrices, and a plan whose row map is not the
//!   one the load makes (after the draft's uploads, which it frees).
//! - (e) teacher-forced, every graph of the set replayed in ik's order —
//!   the warmup, then each block's graphs — each row's token, position and
//!   target hidden row (ik's `inp_tokens`, `inp_pos`, `inp_mtp_states`) fed
//!   in runs of up to eight rows: the streams `eh_proj` writes within
//!   [`eh_band`] of ik's `mtp_eh_proj-48`, a row at a time.
//! - (l) each row's routed set is ik's (`ffn_moe_topk-48`), but where it is
//!   not: such a flip is allowed only when each exchanged pair's gap in
//!   ik's router logits lies within our two logits' error, that error
//!   within [`flip::flip_cap`] and the gap within [`flip::margin_cap`] of the router
//!   input's band ([`router_band`]). The layer's output's distance from
//!   ik's `l_out-48` on the other rows is printed, not held: our f32
//!   activations against ik's q8 ones move each projection by its own
//!   rounding term, which the node-local pins (n) hold instead.
//! - (h) each row's draft token the argmax of our full head's logits, first
//!   index on a tie; on each row ik computes a head for (its `inp_out_ids`)
//!   that argmax is ik's wherever ik's top-2 margin clears [`flip::margin_cap`] at
//!   [`logits_band`], and ik's runner-up otherwise. The logits' distance
//!   from ik's `result_output` is printed, not held.
//! - (n) the node-local pins, on every row: each node the walk taps against
//!   the host's f64 simulation of our own rule on that node's inputs as the
//!   walk read them back, within the bound of the node's own rounding —
//!   `eh_proj` from the fed token and hidden row (the input pack, then the
//!   projection), the attention site's mix, `AttnOut`, the attention's
//!   combine, the feed-forward site's mix, the routed slots' downs and
//!   their weighted sum, the shared expert's down, the gated sum (bit for
//!   bit), `l_out` (the head site's combine), the head site's mix and, on
//!   ik's head rows, the full head's logits. A dot's bound is `γ(k/32 +
//!   5)·Σ|w·x|` ([`dot_depth`]), a mix's the hyper-connection gate's
//!   `MIXED_BAND`, a combine's its weight's `LO_BAND` and one rounding.
//! - (r) the row-list head (`q8_0_gemv_ids` and `_mcol`, the list's rows of
//!   `output` read in place) against the full head over one walk's rows and
//!   over a walk of its last row alone: each list row's logits the full
//!   head's row of its id, bit for bit; each
//!   row's token the id of the list's argmax; each head's probability the
//!   host's softmax maximum over its rows within the sum's rounding bound.
//! - (t) the pairing: after the set's prompt through the target by passes,
//!   the target's streams at position `q` lie within [`FREE_BAND`] of ik's
//!   warmup hidden row at position `q + 1` — ik's MTP row `p` reads the
//!   target's hidden of `p − 1` — and that shift reads closer than 0 or 2;
//!   a draft walk from the position after the prompt's last unit's first,
//!   its rows reading the unit's arena rows in place, equals one fed them
//!   from the host, bit for bit (a walk error here is a FAIL line, not the
//!   gate's end).
//! - (g) a captured walk equals the eager walk over the same rows, bit for
//!   bit — tokens, probabilities, streams, logits — at 1 to 4 rows with the
//!   list head and at 1 row with the full head, each capture of
//!   [`GRAPH_NODES`] nodes; two own rows (`DraftFeed::Own`) in a row, captured
//!   then replayed, equal them eager, and the first equals a host walk of
//!   the previous walk's last token and streams; a window's chain
//!   (`GpuModel::mtp_chain`: a refresh and two own walks, one readback)
//!   reads back each walk's last id and probability, bit for bit.
//! - (f) the walk's refusals by name — its own row before any walk and
//!   after a reset, no rows, nine rows eager, five captured, rows past the
//!   store, a walk past the positions the store holds for the sequence, a
//!   token past the vocabulary, a hidden slice of the wrong length, a walk and
//!   a chain whose rows carry media rows and a store walk whose media rows are
//!   not the embedding's width, a capture with the taps armed, a media load
//!   (`enable_media`) after a captured draft walk, before and after a reset; a store walk (`WalkMode::Store`) past its
//!   `MTP_STORE_ROWS` rows, read back or chained, and the own row and the
//!   streams after one — and a NaN in a
//!   hidden row raised on the fault word as the draft layer's `hc_mix`, the
//!   walk's error; and a walk that reads the target's arenas by position
//!   (`Hidden::Target`): after the prompt by passes, one that reads the
//!   position before the Pass arena's first, one past its last row and one
//!   that starts its rows at the wrong arena row are each refused by name
//!   with the positions the arena holds, and the walk the arena holds runs.
//! - (s) the store walks: over a prompt's rows fed the target's hidden rows
//!   from the host (position 0 beside a zero row, each later position beside
//!   the row of the one before it), store walks in runs of three leave each
//!   position's keys within `key_band` and its values within `value_band`
//!   of what eager walks over the same rows store — the store walk's
//!   projections read q8 activations of 32 values, the eager walk's the f32
//!   rows.
//! - (w) the windows end to end: first the drafted session's prompt call
//!   walks the draft as (s)'s store walks: its warmup, which walks the store
//!   alone in walks of its own width, leaves every prompt position's keys
//!   and values those store walks stored, bit for bit, over a store other
//!   ids overwrote first — a row's stored bits are its own inputs' alone —
//!   and the anchor walk after them is bit for bit the same; then
//!   against the plain run at the set's prompt
//!   and D3K's (depth 3,000 after it): under `app::arch::qwen4exp`'s `Draft`
//!   impl the drafted greedy ids are the plain run's for 64 tokens, with a
//!   rejected row among the windows — else the rollback path never ran and
//!   the clause is red — and the live stores after the run the plain run's
//!   at the same position; at the set's prompt the draft keeps no window
//!   unset, and the same run keeping its windows (`BLOOMERY_MTP_WINDOWS`'s
//!   `MtpDraft::keep_windows`) reads back the same ids through the same
//!   passes, one window a drafted pass, each window's kept count its pass's
//!   kept rows less one and its kept ids the run's tokens after its position;
//!   `commit(k)` is `k` steps for every k, a
//!   scripted draft keeping each exactly through the same verify and commit,
//!   at the deep prompt its rejected row completing a pool leaving the
//!   pooled planes the plain run's (PIN(2026-10-30): the planted-lane case is
//!   removed, as `gate_qwen4exp_e2e`'s (v) says). Last, prompt calls that continue the held
//!   sequence: the draft joins at the call's start and drafts with the plain
//!   run's ids, and restarted beside the held sequence it skips the next
//!   call by name, proposing nothing, the ids still the plain run's.
//! - (fr) a prompt call feeds the draft final rows, the last layer's combine applied to every
//!   row the head's mix does not cover: the set's prompt called once with the draft, one ubatch
//!   walked as one unit (the split walk off), is the reference; the same call under
//!   `SplitWalk::Every` (two units, the head's mix over the second alone) and, on a load of the
//!   same plan whose ubatch holds a third of the prompt, as several ubatches (the head on the
//!   last alone), one unit each and two under `Every`, leaves the draft's store and the next id
//!   the reference's bit for bit; the call by passes (the head on the last pass alone) leaves
//!   the call by steps' (each row its own step's head).
//! - (c) the width chooser (`BLOOMERY_MTP_WIDTH=cost`): the draft behind
//!   `runtime::width`'s `Choosing` over the same model — every window's
//!   verify width at most its proposal, its kept rows the target's, the
//!   greedy ids the plain run's whatever widths ran, the widths that ran
//!   the chooser's tally's, cuts below the draft's own exercised by the
//!   rotation and the probes; and closed (a rule no drafted width wins)
//!   the pass steps plainly with the draft's proposals shadowed, never a
//!   window of one, the ids still the plain run's.
//! - (p) a partial accept's records: the commit of a verify that kept fewer
//!   rows than it ran cuts the draft's records to the kept rows. A window
//!   starting at p0 and keeping k rows leaves the draft store at p0 + 1
//!   and the target at p0 + k. A snapshot preserves both counts. A store
//!   walked past the position (an anchor row) makes the snapshot refused by name, and after
//!   a scripted window that keeps 2 of its 4 rows the pass arena's rejected
//!   row is refused by name for a walk that reads it (the uncut record
//!   would let the walk run).
//! - (x) a resume of a state whose draft side the load does not take (a
//!   state of a load without the draft) is refused by name with the model
//!   untouched: its stores, position and draft store unchanged.
//! - (kp-draft) a cut and a prefix restore to a point keep the draft: a drafted run's state
//!   with a point at its prompt's end below its end, put back whole and cut to the point by
//!   `rollback`, put back as the point's prefix (`seq_resume_prefix` asked one short of the
//!   end), and saved there and put back whole, each stands at the point with the draft's store
//!   holding the positions below it (its rows the run's) and the step arena's row 0 the
//!   target's hidden row before the point bit for bit, as the point kept it; the draft's walk
//!   at the point that reads that row runs, its logits the same bits all three ways. The store
//!   behind the point, its own case: a point past the draft's store (ids fed without the
//!   draft) keeps the draft off, its store forgotten. On a load of twice
//!   `CHECKPOINT_EVERY` positions, the inner mark of a long prompt call: a cut to it and the
//!   run's state put back as its prefix each rejoin with the row a call of its ids alone
//!   leaves. On the two-slot load: two slots cut to points at one position each rejoin with
//!   their own row; a continuing call's start mark after a step keeps the step's row, and one
//!   after a select, whose position before only the load's ubatch arena held for the slot,
//!   keeps none (a cut there leaves the draft off, never with the other slot's row); a state
//!   put back as its prefix on the other slot rejoins with its own rows; and a slot whose cut
//!   left the draft off rides a pass of both slots with plain rows beside the other slot's
//!   drafted window, each slot's ids its own run's.
//! - (sl) a drafted run's state of `n` positions saved on a save lane (`seq_save_async`), whole,
//!   after a rollback the draft rejoins, and with the store behind the state (a prompt call
//!   without the draft): the lane's copy carries the draft's
//!   store rows of those positions beside the target's rows and recurrent stores, so its bytes
//!   are the target's and the store's `n` of its `ctx` positions (the plan's store bytes times
//!   `n / ctx`): they follow `n`, not `ctx`. With the store's rows overwritten by other ids, the
//!   state put back whole holds the draft's count of the save and the store's rows below `n`
//!   bit for bit. A store walked past the state's position (an anchor row) makes the save on a
//!   lane refused by name, as the engine stream's is under (p), nothing enqueued. The lane's
//!   state put back as its point's prefix rejoins the draft there as the engine stream's does,
//!   the store's rows below the point and the step arena's row 0 the same bits.
//! - (i) a pass the width chooser holds back right after the server's cut
//!   step (the prompt call, then its last id as the seat's first step): the
//!   chooser walks the draft's waiting rows first (`Draft::before_plain`,
//!   the call its propose makes on the plain pass it returns), then the
//!   target steps and overwrites the arena the anchor read — after the
//!   walk — so the pass takes the position in instead of refusing, the
//!   draft's store stands at the target's position, a snapshot there is
//!   taken, and the drafted ids stay the plain run's.
//! - (v) a window no verify takes (the width chooser's shadow, or the window
//!   its closing gate cuts to no id), after `begin` and after a drafted pass
//!   from the prompt's anchor: the draft's chain runs, its proposal is
//!   dropped, the target steps and the draft hears of it through
//!   `Draft::held`. The step walks nothing the prompt call left (`begin`
//!   took its anchor), the draft's store stands at the target's position
//!   (the chain took its own rows back), a snapshot there is taken, and the
//!   drafted ids after it are the plain run's.
//! - (y) two drafted sequences resident over a plan of two
//!   (`PlanInputs::plan_mtp_with_slots`, `Body38::open_placed_mtp_slots`),
//!   in the server's order (`worker.rs`'s `start` → `genloop`'s `prompt`):
//!   each request's prompt call and its first plain step run together
//!   before any other slot is selected, then the drafted passes interleave
//!   with a select between every pass, the app-side draft state parked per
//!   slot around each select exactly as a seat will (`MtpDraft::park`
//!   before a select away, `unpark` after a select back): each slot's ids,
//!   draft proposals and acceptances and draft store equal its window's
//!   single-sequence reference, run on the load's own sequence before any
//!   slot exists — references the slots' exchange cannot have shaped. The
//!   windows diverge from position 0 on (the deep prompt read past the
//!   set's own whole length), so no per-sequence state the two share by
//!   content can pass for one the slots exchanged.
//! - (iv) MTP refuses target hidden rows outside the last layer's completed range
//!   (a tapped call that failed past its launch leaves none), and an own row after a
//!   state adoption, reset, store walk, slot exchange or cut until a Rows refresh, or
//!   at a position other than the refreshed walk's next position. A sink's completed rows
//!   and the own row after a refresh run. The checkpoint-only input test exercises
//!   these clauses and the existing plain and drafted greedy runners on one load.
//! - (z) the same two windows' drafted passes run as one pass of both slots'
//!   verify rows (`app::mtp::pass_slots`, `Session::verify_slots` and
//!   `commit_slots`), each window's draft its own and capped at depth 1 (a
//!   pass of 4 rows) and at depth 3 (8 rows): each slot's ids, each pass's
//!   kept rows, its draft's windows (the proposals, their probabilities and
//!   acceptances) and its draft store equal its window run alone on the
//!   load's own sequence before any slot exists through the session's
//!   verify of one sequence, its proposals capped alike (`Capped`); in the
//!   at least one pass at each depth keeps fewer rows than it ran, and in the
//!   real tier one keeps different counts on the two slots (else the clause
//!   is red: the per-slot keep never ran apart; a fixture's generated draft
//!   has every proposal refused, so the fixture tier prints that count and
//!   does not pin it, the step-after arm's fixed counts keeping the two slots
//!   apart there); and while a pass of slot 1's rows alone waits
//!   for its commit a select of slot 1 is refused by name by the model, the
//!   selection and the positions kept. Then (z) step after: both windows'
//!   requests started alone on their own slots, five passes of both slots'
//!   verify rows committed at fixed counts (every pass keeps part of one
//!   slot's rows at least, the two counts differ, and the last keeps 1 of
//!   parked slot 1's 2), then each slot's plain steps — slot 0's right
//!   after the commit, slot 1's after the select that brings it live — the
//!   call that reads a slot's sequence as the pass of several slots left it
//!   and no later pass reads: each slot's ids, each step's argmax and
//!   logits row bit for bit, its committed lane and position its window's
//!   run alone on the load's own sequence before any slot exists, through
//!   the session's verify of one sequence at the same counts.
//!   A plain round immediately after slot 0's GEMM prompt and `begin`,
//!   beside slot 1's started request, matches immediate draft walks.
//!
//! (l) and (h) hold at least one row each off a flip: a run that excuses
//! every row as a flip fails.
//!
//! Tiers (`BLOOMERY_TIER`, `crates/gpu-gates/src/tier.rs`): the fixture tier runs the
//! self-consistency clauses, replays the rows of the real set's mirror on the fixture as the feed
//! (`tier::input_set`; ik's hidden rows are finite inputs), reads the deep prompt from D3K's
//! mirror, and leaves every comparison with the real set's output to the real tier
//! (`deferred(real)`). Beside them it runs the two fixture-oracle
//! clauses against ik's MTP set dumped on the fixture (`refset::arch::qwen4exp::fixture`, each
//! opened by `tier::fixture_set`), declared in [`FX_ORACLE`] and held to the gate's end:
//! - (elhfx) the fixture set's graphs replayed as (e), (l) and (h) replay the real one, at the
//!   bands [`fx_bands`] counts over the draft file's site types, the layer's output held where
//!   the real clause prints it, the argmax a tie band ([`flip::head_tie`]) in place of ik's
//!   runner-up;
//! - (tfx) the pairing of (t) against the fixture set's warmup hidden rows: the shift 1 reads
//!   closer than 0 and 2, its distance printed (the prompt's last rows lie on a flip's path).

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_qwen4exp_mtp", gate::run())
}

#[cfg(feature = "gpu")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
#[path = "shared/q38_arch.rs"]
mod q38_arch;

#[cfg(feature = "gpu")]
#[path = "shared/quiet.rs"]
mod quiet;

#[cfg(feature = "gpu")]
mod gate {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use bloomery_gpu::GpuError;
    use bloomery_gpu::arch::qwen3moe::{
        Body38, CHECKPOINT_EVERY, KvQ8, MTP_GRAPH_ROWS, MTP_ROWS, MTP_STORE_ROWS, Mtp38, MtpDraft,
        MtpNode, MtpTaps, Prompt38, Qwen38Model, Seq38, SplitWalk, Store38Host, TargetRows,
        checkpoint_bytes, seq_positional_bytes, seq38_bytes,
    };
    use bloomery_gpu::draft::{DraftFeed, DraftHead, Feed, Hidden, WalkMode};
    use bloomery_gpu::fault::FaultSite;
    use bloomery_gpu::media::CallRows;
    use bloomery_gpu::weights::DevWeight;
    use bloomery_gpu_gates::act_rule::{Arm, site_rel};
    use bloomery_gpu_gates::flip::{self, Flip};
    use bloomery_gpu_gates::q38_fixture::{Shape, key_ring_live};
    use bloomery_gpu_gates::rounding::{U, gamma, q8_32_rel};
    use bloomery_gpu_gates::slots_gate;
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{GateError, RefManifest, checks_failed, ref_model_path, verdict};
    use bloomery_levers::{CARD_BUDGET, CardBudget, HostCfg, Levers};
    use gguf::digest::hex;
    use gguf::quant::{GgmlType, half_to_f32};
    use gguf::{Gguf, Split};
    use model::arch::models::headpick::{HeadSet, vocab_sha256};
    use model::arch::models::{Borrows, MtpSource};
    use model::arch::qwen35moe::head_list::{HeadWhy, SHIPPED, head_rows_of};
    use model::arch::qwen35moe::place::{
        Experts, MtpInputs, MtpPlan, PlaceError, PlanInputs, machine, machine_for_experts,
    };
    use model::placement::workstation::HostRead;
    use model::placement::{PlacementError, PlanLevers};
    use refset::arch::qwen4exp::IK;
    use refset::arch::qwen4exp::MODEL;
    use refset::arch::qwen4exp::fixture;
    use refset::arch::qwen4exp::mtp::{DRAFT, DraftFrom, MTP, MTP_SET, draft_file};
    use refset::ik::Layout;
    use refset::mtpref::{Graph, MtpSet};
    use runtime::hc_gated::{Geometry, LO_BAND, MIXED_BAND, MixWeights, mix_ref};
    use runtime::prompt::CallEnd;
    use runtime::width::Chosen as _;
    use runtime::{Advance, Committed, Draft, Out, PassSink, TapNeed, Target, Verify, Want};

    use crate::quiet::Quiet;

    /// The gate's one reading of what the run is made from: the target file
    /// (`ref_model_path`, a whole fixture under the fixture tier), the draft
    /// file beside it (the engine's own discovery), the host load config, the
    /// plan's budget (the caller's lever in the real tier, the header's in the
    /// fixture tier, [`plan_levers`]) and the layer counts the header names.
    /// Set once, first thing in `run`, before any load.
    struct Cfg {
        path: std::path::PathBuf,
        draft: std::path::PathBuf,
        host: HostCfg,
        header_budget: Option<tier::HeaderBudget>,
        lever_budget: CardBudget,
        shape: Shape,
        n_expert: usize,
    }

    static CFG: std::sync::OnceLock<Cfg> = std::sync::OnceLock::new();

    fn cfg() -> &'static Cfg {
        CFG.get().expect("init runs before any clause")
    }

    /// The draft layer's index in the draft file: the target's layer count,
    /// as the engine names the layer past the target's (the real file's 48).
    fn draft_layer() -> usize {
        cfg().shape.n_layer
    }

    /// The target file, opened as a split of architecture `qwen4exp`.
    fn open_target() -> Result<Split, GateError> {
        let path = &cfg().path;
        let file = Split::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        if file.architecture() != Some("qwen4exp") {
            return Err(format!(
                "{} is {:?}, not qwen4exp",
                path.display(),
                file.architecture()
            )
            .into());
        }
        Ok(file)
    }

    /// The MTP draft file a run of `target` opens: the engine's own discovery (`BLOOMERY_MTP_DRAFT`,
    /// else the shared draft's name beside the target, else the family's path), and in the fixture tier
    /// a whole fixture beside the target — never the real draft the discovery falls back to.
    ///
    /// # Errors
    /// The fixture tier's discovery landing on the family's real draft, or on a file that is not a
    /// whole fixture.
    fn draft_path(levers: &Levers, target: &Path) -> Result<PathBuf, GateError> {
        let (path, from) = draft_file(levers.mtp_draft(), target);
        let tier = Tier::from_env()?;
        if tier == Tier::Fixture {
            if matches!(from, DraftFrom::Family(_)) {
                return Err(format!(
                    "the fixture tier found no shared draft beside {} and would open the real one at \
                     {}",
                    target.display(),
                    path.display()
                )
                .into());
            }
            let g = Gguf::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
            tier.check_file(&path, g.iter_kv().map(|(k, _)| k))?;
        }
        Ok(path)
    }

    /// The draft file beside the target.
    fn open_draft() -> Result<Split, GateError> {
        let path = &cfg().draft;
        Split::open(path).map_err(|e| format!("open {}: {e}", path.display()).into())
    }

    /// The plan's levers for a load that reserves `extra` card bytes for the
    /// draft out of the same budget (0: the target alone): the real tier's
    /// lever, or the header's budget on the fixture tier's card plus `extra`
    /// (`tier::budget_levers`).
    fn plan_levers(extra: u64) -> Result<PlanLevers, GateError> {
        tier::budget_levers(
            Tier::from_env()?,
            cfg().header_budget,
            cfg().lever_budget.clone(),
            extra,
        )
    }

    /// The card a plan is made on: the real tier's the gate runner's (the
    /// 3090's bytes on the card in view), the fixture tier's `a`
    /// (`tier::card`).
    fn card() -> Result<model::placement::workstation::CardSpec, GateError> {
        tier::card(crate::gate_card::card)
    }

    /// Read the headers once and print each derived value beside the real
    /// file's literal it replaced ([`tier::witness`]).
    fn init() -> Result<bool, GateError> {
        let levers = bloomery_levers::at_main(&[
            CARD_BUDGET,
            bloomery_gpu::arch::qwen3moe::ubatch::UBATCH_ENV,
        ])?;
        let path = ref_model_path()?;
        let draft = draft_path(&levers, &path)?;
        let file = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let inputs = PlanInputs::describe(&file)?;
        let shape = crate::q38_arch::shape_of(&inputs.spec, &file)?;
        let header_budget = tier::header_budget(&file)?;
        let n_expert = inputs.spec.routed().map_or(0, |m| m.experts as usize);
        let cfg = Cfg {
            path,
            draft,
            host: levers.host(),
            header_budget,
            lever_budget: levers.card_budget(),
            shape,
            n_expert,
        };
        let mut ok = tier::witness(
            "model file",
            cfg.path.to_string_lossy().into_owned(),
            MODEL.to_string(),
        );
        ok &= tier::witness(
            "draft file",
            cfg.draft.to_string_lossy().into_owned(),
            DRAFT.to_string(),
        );
        ok &= tier::witness("draft layer", cfg.shape.n_layer, REAL_DRAFT_LAYER);
        ok &= tier::witness(
            "card budget",
            cfg.lever_budget.clone(),
            PlanLevers::default().card_budget_bytes,
        );
        ok &= tier::witness_card(&card()?);
        if CFG.set(cfg).is_err() {
            return Err("the gate's configuration was read twice".into());
        }
        Ok(ok)
    }

    /// The draft layer's index in the real file's draft (the target's 48
    /// layers): the literal [`draft_layer`] replaced.
    const REAL_DRAFT_LAYER: usize = 48;

    /// Cache rows: the e2e gate's.
    const CTX: u64 = 3072;
    /// The head's rows: every sixth id, 0 to 245,754.
    const LIST_ROWS: u32 = 40_960;
    /// The shipped list's rows ([`SHIPPED`]'s first line).
    const SHIPPED_ROWS: usize = 76_000;
    /// Streams of a hidden row, and the values of one.
    const STREAMS: usize = 4;
    const HIDDEN: usize = 2560;
    const WIDE: usize = STREAMS * HIDDEN;
    /// The hyper-connection mix's rank.
    const RANK: usize = 320;
    /// ggml's IMROPE position streams per row.
    const POS_SECTIONS: usize = 4;
    /// The router's experts and routed picks, and an expert's width.
    const N_EXPERT: usize = 512;
    const N_USED: usize = 10;
    const FF: usize = 640;

    /// PIN(2026-09-30): the streams `eh_proj` writes against ik's on the same
    /// inputs. Both sides compute the two norms in f32 (a few roundings
    /// each); the projection is the one term: ik quantizes its 5,120-value
    /// input to 32-value q8 blocks, ours reads it in f32, and a q8
    /// activation moves a projection's output by at most
    /// `rounding::q8_32_rel` = 1.2858e-2 of its RMS (the largest crest).
    fn eh_band() -> f64 {
        q8_32_rel()
    }

    /// PIN(2026-09-30): the band of the head's logits against ik's that
    /// (h)'s tie cap reads: the error model of the e2e gate's `gemm_band`
    /// over the draft's projections in series, where ik's side reads q8
    /// activations and ours f32 — `eh_proj`, the attention's input and
    /// output projections, the routed experts' gate·up and down, the shared
    /// expert's gate·up and down, and the head's projection: eight terms of
    /// [`q8_32_rel`], √8 · 1.2858e-2 = 3.64e-2.
    fn logits_band() -> f64 {
        8f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-09-30): the router's input against ik's: `eh_proj` and the
    /// attention's input and output projections, √3 · 1.2858e-2 = 2.23e-2.
    fn router_band() -> f64 {
        3f64.sqrt() * q8_32_rel()
    }

    /// PIN(2026-10-01): a store walk's values against an eager walk's over
    /// the same rows, a position's row at a time: the store walk's
    /// projections read 32-value q8 activations where the eager walk reads
    /// the f32 rows — `eh_proj`, then the value's own projection, two
    /// roundings of at most `rounding::q8_32_rel` each in series, which add
    /// (the triangle inequality; a quadrature sum assumes the two independent,
    /// and the clean run's worst position already sits past √2 of it); the
    /// attention site's mix reaches that input through σ, whose slope is at
    /// most ¼, and adds under 4 % of it (the e2e gate's `gemm_band` rule);
    /// each side's f16 store one rounding of 2^−11. 2.772e-2.
    fn value_band() -> f64 {
        2.0 * q8_32_rel() * 1.04 + 2.0 * 2f64.powi(-11)
    }

    /// PIN(2026-10-01): a store walk's keys against an eager walk's: the
    /// value's two projections in series, then each head's RMS norm, which
    /// moves a unit vector by at most twice its input's relative distance,
    /// and the rope, a rotation; each side's f16 store one rounding. 5.446e-2.
    fn key_band() -> f64 {
        2.0 * 2.0 * q8_32_rel() * 1.04 + 2.0 * 2f64.powi(-11)
    }

    /// Rows a host store walk of (s) and (w) takes: a width no caller walks,
    /// so the call's warmup equals it only when a row's stored bits are a
    /// function of that row's inputs alone, not of the walk it lands in.
    const HOST_STORE_RUN: usize = 3;

    /// PIN(2026-09-30): a captured walk's nodes at 1 to 4 rows, derived from
    /// the walk's launch list (`mtp38`'s module doc): the embedding and the
    /// pack (2); `eh_proj` over the `4·m` packed columns in runs of eight
    /// (1, 1, 2, 2); the attention site — its mix (3), q, k, v, the norm and
    /// append, the flash's segment pass and merge, the gate, the output
    /// projection (11); the feed-forward site — its mix (3), the router, the
    /// places, the routed gate, up, SwiGLU and down, the slots' sum, the
    /// shared expert's gate, up, SwiGLU and down, the gated sum (15); the
    /// head — its mix (3), the projection, the argmax (5); the two copies of
    /// the last row (2). 35 + ⌈4m/8⌉.
    const GRAPH_NODES: [usize; 4] = [36, 36, 37, 37];

    /// PIN(2026-09-30): [provisional — backlog] the target's last-layer
    /// streams after the prompt by passes against ik's hidden rows, the e2e
    /// gate's `FREE_BAND` borrowed: its derivation is GLM's (√45 ·
    /// 1.415e-2) carried to this model's 48 layers, not a Qwen3.8 forced
    /// arm, which neither gate has; a forced arm's worst layer replaces it.
    const FREE_BAND: f64 = 0.10;

    /// `‖a − b‖ / ‖b‖` in f64; infinite on a NaN or a length mismatch, so
    /// neither passes a band.
    fn rel(a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - f64::from(y)).powi(2);
            den += f64::from(y).powi(2);
        }
        let r = (num / den.max(f64::MIN_POSITIVE)).sqrt();
        if r.is_nan() { f64::INFINITY } else { r }
    }

    /// The first index of the largest value, and the largest value of the
    /// rest: the head's tie rule.
    fn top2(v: &[f32]) -> (usize, usize) {
        let mut best = 0;
        for (i, &x) in v.iter().enumerate() {
            if x > v[best] {
                best = i;
            }
        }
        let mut second = usize::from(best == 0);
        for (i, &x) in v.iter().enumerate() {
            if i != best && x > v[second] {
                second = i;
            }
        }
        (best, second)
    }

    /// `1 / Σ exp(v_i − max)` in f64: the largest softmax probability.
    fn p_max(v: &[f32]) -> f64 {
        let top = v.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        1.0 / v
            .iter()
            .map(|&x| (f64::from(x) - f64::from(top)).exp())
            .sum::<f64>()
    }

    /// The rounding bound on our probability over `n` rows: each thread's
    /// share of `⌈n/1024⌉` terms in order, the butterfly's five, the 32 warp
    /// slots in order, each term's `exp` within two ulps, and the division.
    fn p_bound(n: usize) -> f64 {
        gamma(n.div_ceil(1024) + 5 + 32 + 1) + 4.0 * U
    }

    /// Column `c` of a `[row][m]` logits readback.
    fn column(v: &[f32], m: usize, c: usize) -> Vec<f32> {
        v.iter().skip(c).step_by(m).copied().collect()
    }

    /// One graph of the set: its block, label and rows, ik's side.
    struct IkGraph {
        block: i32,
        graph: Graph,
        /// The draft layer's index in ik's node names.
        layer: usize,
        tokens: Vec<u32>,
        pos: Vec<u32>,
        states: Vec<f32>,
        out_ids: Vec<usize>,
        eh: Vec<f32>,
        l_out: Vec<f32>,
        logits: Vec<f32>,
        router: Vec<f32>,
        topk: Vec<i32>,
    }

    impl IkGraph {
        fn read(
            set: &MtpSet,
            block: i32,
            graph: Graph,
            layer: usize,
        ) -> Result<IkGraph, GateError> {
            let find = |name: &str| set.find(block, graph, name, 0);
            let ints = |name: &str| -> Result<Vec<i32>, GateError> {
                Ok(set.i32s(find(name)?, Layout::Flat)?)
            };
            let f32s =
                |name: &str| -> Result<Vec<f32>, GateError> { Ok(set.logical_f32s(find(name)?)?) };
            let unsigned = |name: &str| -> Result<Vec<u32>, GateError> {
                ints(name)?
                    .into_iter()
                    .map(|v| u32::try_from(v).map_err(|_| format!("{name} holds {v}").into()))
                    .collect()
            };
            let tokens = unsigned("inp_tokens")?;
            let n = tokens.len();
            // ggml's multi-section positions (IMROPE): four streams of n, the
            // first the text position; a text row's second and third equal it.
            let sections = unsigned("inp_pos")?;
            if sections.len() != POS_SECTIONS * n
                || sections[n..3 * n].chunks(n).any(|s| s != &sections[..n])
            {
                return Err(format!(
                    "block {block} {}: inp_pos holds {} values, not {POS_SECTIONS} sections of \
                     {n} with the second and third the first's",
                    graph.as_str(),
                    sections.len()
                )
                .into());
            }
            let pos = sections[..n].to_vec();
            let out_ids = unsigned("inp_out_ids")?
                .into_iter()
                .map(|v| v as usize)
                .collect::<Vec<_>>();
            let topk_row = find(&format!("ffn_moe_topk-{layer}"))?;
            let topk = set.i32s(topk_row, Layout::Logical)?;
            let g = IkGraph {
                block,
                graph,
                layer,
                states: f32s("inp_mtp_states")?,
                eh: f32s(&format!("mtp_eh_proj-{layer}"))?,
                l_out: f32s(&format!("l_out-{layer}"))?,
                logits: f32s("result_output")?,
                router: f32s(&format!("ffn_moe_logits-{layer}"))?,
                topk,
                tokens,
                pos,
                out_ids,
            };
            let shapes = [
                ("inp_pos", g.pos.len(), n),
                ("inp_mtp_states", g.states.len(), n * WIDE),
                ("mtp_eh_proj", g.eh.len(), n * WIDE),
                ("l_out", g.l_out.len(), n * WIDE),
                ("ffn_moe_logits", g.router.len(), n * N_EXPERT),
                ("ffn_moe_topk", g.topk.len(), n * N_USED),
            ];
            if let Some((what, got, want)) = shapes.iter().find(|(_, got, want)| got != want) {
                return Err(format!(
                    "block {block} {}: {what} holds {got} values, {n} rows take {want}",
                    graph.as_str()
                )
                .into());
            }
            if g.pos.windows(2).any(|w| w[1] != w[0] + 1)
                || g.out_ids.iter().any(|&o| o >= n)
                || g.topk.iter().any(|&e| !(0..N_EXPERT as i32).contains(&e))
            {
                return Err(format!(
                    "block {block} {}: positions not consecutive, an output row past {n}, or a \
                     routed id past {N_EXPERT}",
                    graph.as_str()
                )
                .into());
            }
            Ok(g)
        }

        fn label(&self) -> String {
            format!("block {} {}", self.block, self.graph.as_str())
        }
    }

    /// The draft layer's index in a set's node names: the header's ([`draft_layer`]), which the
    /// set's `mtp_eh_proj-N` rows must name (`MtpSet::check_draft_layer`; the real file's 48 on its
    /// set, the fixture target's layer count on the fixture's).
    fn ik_layer(set: &MtpSet) -> Result<usize, GateError> {
        let layer = draft_layer();
        set.check_draft_layer(layer)?;
        Ok(layer)
    }

    /// The set's graphs in the order ik computed them: the warmup, then each
    /// verified block's graphs in file order.
    fn graphs_of(set: &MtpSet) -> Vec<(i32, Graph)> {
        let mut order: Vec<(i32, Graph)> = Vec::new();
        for r in &set.rows {
            if !order.contains(&(r.block, r.graph)) {
                order.push((r.block, r.graph));
            }
        }
        order
    }

    /// A walk's readback: the draft, the streams and the logits.
    struct Walk {
        draft: MtpDraft,
        l_out: Vec<f32>,
        logits: Vec<f32>,
        rows: usize,
    }

    fn walk(
        m: &mut Qwen38Model,
        feed: DraftFeed<'_, TargetRows>,
        head: DraftHead,
        mode: WalkMode,
    ) -> Result<Walk, GateError> {
        let draft = m.mtp_draft(feed, head, mode)?;
        let l_out = m.mtp_l_out()?;
        let (logits, rows) = m.mtp_logits()?;
        Ok(Walk {
            draft,
            l_out,
            logits,
            rows,
        })
    }

    /// Two walks' readbacks, bit for bit.
    fn same_bits(a: &Walk, b: &Walk) -> bool {
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        a.draft.tokens == b.draft.tokens
            && bits(&a.draft.p) == bits(&b.draft.p)
            && bits(&a.l_out) == bits(&b.l_out)
            && bits(&a.logits) == bits(&b.logits)
    }

    /// A Q8_0 matrix's rows on the host, `k` values a row.
    struct HostQ8<'a> {
        bytes: &'a [u8],
        k: usize,
        rows: usize,
    }

    impl<'a> HostQ8<'a> {
        fn open(split: &'a Split, name: &str) -> Result<HostQ8<'a>, GateError> {
            let (sh, t) = split
                .find(name)
                .ok_or_else(|| format!("no tensor {name}"))?;
            if t.ty != gguf::quant::GgmlType::Q8_0 || !(2..=3).contains(&t.dims.len()) {
                return Err(format!(
                    "{name} is {:?} {:?}, not a Q8_0 matrix or stack",
                    t.ty, t.dims
                )
                .into());
            }
            let g = split
                .shard(sh)
                .ok_or_else(|| format!("{name}: shard {sh}"))?;
            let bytes = g.data(t)?;
            let k = usize::try_from(t.dims[0])?;
            let rows = t.dims[1..]
                .iter()
                .try_fold(1usize, |a, &d| usize::try_from(d).map(|d| a * d))?;
            Ok(HostQ8 { bytes, k, rows })
        }

        /// Row `r`'s 34-byte blocks.
        fn blocks(&self, r: usize) -> &[[u8; 34]] {
            let nb = self.k / 32;
            self.bytes[r * nb * 34..(r + 1) * nb * 34]
                .as_chunks::<34>()
                .0
        }

        /// Row `r` dequantized: each value `q·d`, exact in f32.
        fn row(&self, r: usize) -> Vec<f32> {
            let mut v = Vec::with_capacity(self.k);
            for blk in self.blocks(r) {
                let d = gguf::quant::half_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                v.extend(blk[2..].iter().map(|&q| f32::from(q as i8) * d));
            }
            v
        }

        /// Every row dequantized, row after row.
        fn dequant(&self) -> Vec<f32> {
            (0..self.rows).flat_map(|r| self.row(r)).collect()
        }

        /// Row `r` against `x` in f64 over the exact weight values: `Σ w·x`
        /// and `Σ |w·x|`.
        fn dot_abs(&self, r: usize, x: &[f64]) -> (f64, f64) {
            let (mut acc, mut abs) = (0.0f64, 0.0f64);
            for (b, blk) in self.blocks(r).iter().enumerate() {
                let d = f64::from(gguf::quant::half_to_f32(u16::from_le_bytes([
                    blk[0], blk[1],
                ])));
                for (i, &q) in blk[2..].iter().enumerate() {
                    let t = d * f64::from(q as i8) * x[b * 32 + i];
                    acc += t;
                    abs += t.abs();
                }
            }
            (acc, abs)
        }

        /// Rows `rows` against `x` ([`HostQ8::dot_abs`]), split over the
        /// host's cores.
        fn apply_abs(&self, x: &[f64], rows: std::ops::Range<usize>) -> (Vec<f64>, Vec<f64>) {
            let n = rows.len();
            let lanes = std::thread::available_parallelism().map_or(1, |p| p.get());
            let chunk = n.div_ceil(lanes).max(1);
            let mut out = vec![(0.0f64, 0.0f64); n];
            std::thread::scope(|s| {
                for (c, part) in out.chunks_mut(chunk).enumerate() {
                    let first = rows.start + c * chunk;
                    s.spawn(move || {
                        for (i, o) in part.iter_mut().enumerate() {
                            *o = self.dot_abs(first + i, x);
                        }
                    });
                }
            });
            out.into_iter().unzip()
        }
    }

    /// PIN(2026-09-30): the rounding path of one output of the q8f32 Q8_0
    /// dot over `k` values (`q8_0_lane_partial_1col`, which the gemv, the
    /// token-major mcol gemv and the selecting gemv all run): a lane's `k/32`
    /// fused multiply-adds — its `k/128` words, four values a word, `q·d`
    /// exact in f32 — then the butterfly's five adds, so `|ŷ − Σ w·x| ≤
    /// γ(k/32 + 5)·Σ|w·x|` (Higham's bound for a sum of that depth).
    /// Measured on the set's 127 rows: the worst distance at 0.009 of its
    /// row's bound (the shared expert's down, 1.405e-7).
    fn dot_depth(k: usize) -> usize {
        k / 32 + 5
    }

    /// PIN(2026-09-30): the input pack's values against the host's, relative
    /// (`mtp_input`): the hidden row's sum of squares over its 10,240 values
    /// is 40 fused multiply-adds a thread, the butterfly's 5 and the 7 warp
    /// sums, every term positive: γ(52) of the sum (the embedding's 2,560
    /// values take γ(22), inside it); the root halves that share, the
    /// division by the count, `+ eps`, the root and the reciprocal add 4u,
    /// and `(x·r)·γ` two more: ½γ(52) + 6u. Measured on the set's 127
    /// rows: `eh_proj` over the pack at worst 2.240e-7, 0.001 of its bound.
    fn pack_rel() -> f64 {
        0.5 * gamma(52) + 6.0 * U
    }

    /// One hyper-connection site's weights on the host, dequantized, in
    /// `runtime::hc_gated::MixWeights`' layouts.
    struct HostSite {
        gamma: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Option<Vec<f32>>,
    }

    impl HostSite {
        /// The site `sub` of the draft layer (`attn`, `ffn`), or the head's
        /// (`nextn.hc_head`, no inject).
        fn open(draft: &Split, stem: &str, inject: bool) -> Result<HostSite, GateError> {
            let b = |s: &str| format!("blk.{}.{stem}_{s}.weight", draft_layer());
            let q = |s: &str| HostQ8::open(draft, &b(s)).map(|m| m.dequant());
            Ok(HostSite {
                gamma: bloomery_gpu_gates::split_f32(draft, &b("norm"), WIDE)?,
                down: q("down")?,
                up: q("up")?,
                inject: if inject { Some(q("inject")?) } else { None },
            })
        }

        fn weights(&self) -> MixWeights<'_> {
            MixWeights {
                gamma: &self.gamma,
                down: &self.down,
                up: &self.up,
                inject: self.inject.as_deref(),
            }
        }
    }

    /// The draft layer's weights the node-local pins read, the target's
    /// embedding and head, and the model's norm epsilon.
    struct Host<'a> {
        embd: HostQ8<'a>,
        enorm: Vec<f32>,
        hnorm: Vec<f32>,
        eh: HostQ8<'a>,
        out: HostQ8<'a>,
        down_e: HostQ8<'a>,
        down_s: HostQ8<'a>,
        head: HostQ8<'a>,
        attn: HostSite,
        ffn: HostSite,
        head_site: HostSite,
        geo: Geometry,
        eps: f32,
    }

    impl<'a> Host<'a> {
        fn open(file: &'a Split, draft: &'a Split, eps: f32) -> Result<Host<'a>, GateError> {
            let o =
                |stem: &str| HostQ8::open(draft, &format!("blk.{}.{stem}.weight", draft_layer()));
            let f = |stem: &str, n: usize| {
                bloomery_gpu_gates::split_f32(
                    draft,
                    &format!("blk.{}.{stem}.weight", draft_layer()),
                    n,
                )
            };
            Ok(Host {
                embd: HostQ8::open(file, "token_embd.weight")?,
                enorm: f("nextn.enorm", HIDDEN)?,
                hnorm: f("nextn.hnorm", WIDE)?,
                eh: o("nextn.eh_proj")?,
                out: o("attn_output")?,
                down_e: o("ffn_down_exps")?,
                down_s: o("ffn_down_shexp")?,
                head: HostQ8::open(file, "output.weight")?,
                attn: HostSite::open(draft, "hc_attn", true)?,
                ffn: HostSite::open(draft, "hc_ffn", true)?,
                head_site: HostSite::open(draft, "nextn.hc_head", false)?,
                geo: Geometry::new(STREAMS as u32, RANK as u32, HIDDEN as u32)
                    .map_err(|e| format!("the draft's hyper-connection shape: {e}"))?,
                eps,
            })
        }
    }

    /// PIN(2026-09-30): a hyper-connection mix against `mix_ref` on its
    /// input as read back, `max|ours − ref| / max|ref|`: the hyper-connection
    /// gate's `MIXED_BAND`, whose derivation takes the up's gain on random
    /// inputs. Measured on this set's 127 rows: AttnIn at worst 0.023 of it,
    /// FfnIn 0.021, HeadIn 0.007.
    fn mix_band() -> f64 {
        f64::from(MIXED_BAND)
    }

    /// A node's index in the taps ([`MtpNode::ALL`]'s order).
    fn node_at(n: MtpNode) -> usize {
        MtpNode::ALL
            .iter()
            .position(|&m| m == n)
            .expect("ALL lists every node")
    }

    /// One node-local pin's record over the replay: each row's distance
    /// from the host's simulation of our rule on the node's inputs as the
    /// walk read them back, against its bound.
    struct Pin {
        what: &'static str,
        rule: &'static str,
        rows: usize,
        past: usize,
        worst: f64,
        /// The largest distance over its row's bound.
        ratio: f64,
        first_past: Option<String>,
    }

    impl Pin {
        fn new(what: &'static str, rule: &'static str) -> Pin {
            Pin {
                what,
                rule,
                rows: 0,
                past: 0,
                worst: 0.0,
                ratio: 0.0,
                first_past: None,
            }
        }

        /// A row at distance `d` against its bound `band` (0: bit for bit).
        fn hold(&mut self, d: f64, band: f64, label: &dyn Fn() -> String) {
            self.rows += 1;
            self.worst = self.worst.max(d);
            let r = if band > 0.0 {
                d / band
            } else if d == 0.0 {
                0.0
            } else {
                f64::INFINITY
            };
            self.ratio = self.ratio.max(if r.is_nan() { f64::INFINITY } else { r });
            // A NaN distance is past every bound.
            if d.is_nan() || d > band {
                self.past += 1;
                if self.first_past.is_none() {
                    self.first_past = Some(format!("{} ({d:.3e}, bound {band:.3e})", label()));
                }
            }
        }

        fn ok(&self) -> bool {
            self.past == 0 && self.rows > 0
        }

        fn line(&self) -> String {
            format!(
                "(n) {} = {}: {} rows, worst {:.3e}, worst over its bound {:.3}, {} past{} {}",
                self.what,
                self.rule,
                self.rows,
                self.worst,
                self.ratio,
                self.past,
                self.first_past
                    .as_ref()
                    .map_or(String::new(), |f| format!(", first {f}")),
                verdict(self.ok())
            )
        }
    }

    /// The node-local pins, in the walk's order.
    struct Pins {
        eh: Pin,
        attn_in: Pin,
        attn_out: Pin,
        attn_combined: Pin,
        ffn_in: Pin,
        routed: Pin,
        shared: Pin,
        ffn_out: Pin,
        l_out: Pin,
        head_in: Pin,
        logits: Pin,
    }

    impl Pins {
        fn new() -> Pins {
            Pins {
                eh: Pin::new(
                    "eh_proj",
                    "the input pack of the token's embedding row and the fed hidden row, \
                     then the projection",
                ),
                attn_in: Pin::new("AttnIn", "the attention site's mix of eh_proj's streams"),
                attn_out: Pin::new("AttnOut", "the output projection of AttnGated"),
                attn_combined: Pin::new(
                    "AttnCombined",
                    "eh_proj's streams plus the attention site's weights times AttnOut",
                ),
                ffn_in: Pin::new("FfnIn", "the feed-forward site's mix of AttnCombined"),
                routed: Pin::new(
                    "Routed",
                    "the routed slots' downs of their SwiGLU rows, weighted, in slot order",
                ),
                shared: Pin::new("Shared", "the shared expert's down of its SwiGLU row"),
                ffn_out: Pin::new("FfnOut", "Routed plus Shared times its gate, bit for bit"),
                l_out: Pin::new(
                    "l_out",
                    "AttnCombined plus the feed-forward site's weights times FfnOut",
                ),
                head_in: Pin::new("HeadIn", "the head site's mix of l_out"),
                logits: Pin::new("logits", "the full head's projection of HeadIn"),
            }
        }

        fn all(&self) -> [&Pin; 11] {
            [
                &self.eh,
                &self.attn_in,
                &self.attn_out,
                &self.attn_combined,
                &self.ffn_in,
                &self.routed,
                &self.shared,
                &self.ffn_out,
                &self.l_out,
                &self.head_in,
                &self.logits,
            ]
        }
    }

    /// `‖ours − y‖ / ‖y‖` and `‖Σ|w·x|‖ / ‖y‖`, the host's `y` and
    /// absolute sums in f64: a dot's bound is `γ(depth)` times the second.
    fn dot_pin(ours: &[f32], y: &[f64], abs: &[f64]) -> (f64, f64) {
        let den = y
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let num = ours
            .iter()
            .zip(y)
            .map(|(&o, &v)| (f64::from(o) - v).powi(2))
            .sum::<f64>()
            .sqrt();
        let a = abs.iter().map(|v| v * v).sum::<f64>().sqrt();
        (nan_inf(num / den), a / den)
    }

    /// A NaN distance as infinite, so no band passes it.
    fn nan_inf(d: f64) -> f64 {
        if d.is_nan() { f64::INFINITY } else { d }
    }

    /// `max|ours − ref| / max|ref|`: the mix bands' measure
    /// (`runtime::hc_gated::MIXED_BAND`).
    fn max_rel(ours: &[f32], r: &[f64]) -> f64 {
        let den = r
            .iter()
            .fold(0.0f64, |a, v| a.max(v.abs()))
            .max(f64::MIN_POSITIVE);
        let num = ours
            .iter()
            .zip(r)
            .fold(0.0f64, |a, (&o, &v)| a.max((f64::from(o) - v).abs()));
        nan_inf(num / den)
            + if ours.len() == r.len() {
                0.0
            } else {
                f64::INFINITY
            }
    }

    /// A combine against the host's (`res + wgt·y` per stream, `wgt` the
    /// site's in f64): `‖ours − host‖ / ‖host‖` and its bound over it.
    ///
    /// PIN(2026-09-30): the card's weight is within `LO_BAND` of the largest
    /// of the site's (`runtime::hc_gated::LO_BAND`'s derivation), so a value
    /// moves by at most that times `|y|`, and the fused multiply-add rounds
    /// once: per value `LO_BAND·max|wgt|·|y| + u·|host|`. Measured on the
    /// set's 127 rows: AttnCombined at worst 0.011 of its bound, l_out 0.018.
    fn combine_pin(ours: &[f32], res: &[f32], y: &[f32], wgt: &[f64]) -> (f64, f64) {
        let top = wgt.iter().fold(0.0f64, |a, w| a.max(w.abs()));
        let (mut num, mut den, mut bound) = (0.0f64, 0.0f64, 0.0f64);
        for (s, &w) in wgt.iter().enumerate() {
            let span = s * HIDDEN..(s + 1) * HIDDEN;
            for ((&o, &r), &yi) in ours[span.clone()].iter().zip(&res[span]).zip(y) {
                let h = f64::from(r) + w * f64::from(yi);
                num += (f64::from(o) - h).powi(2);
                den += h * h;
                bound += (f64::from(LO_BAND) * top * f64::from(yi).abs() + U * h.abs()).powi(2);
            }
        }
        let den = den.sqrt().max(f64::MIN_POSITIVE);
        (nan_inf(num.sqrt() / den), bound.sqrt() / den)
    }

    /// The node-local pins of row `t` of a walk of graph `g` at its row
    /// `at` ([`Pins`]): each node against the host's simulation of our rule
    /// on the node's inputs as the walk read them back — the eh_proj pin's
    /// inputs are the fed token and hidden row themselves.
    #[allow(
        clippy::too_many_arguments,
        reason = "one row of one walk: its taps, streams, logits and the host's weights"
    )]
    fn pin_row(
        h: &Host<'_>,
        g: &IkGraph,
        at: usize,
        taps: &MtpTaps,
        t: usize,
        l_out: &[f32],
        p: &mut Pins,
    ) {
        let label = || format!("{} row {at}", g.label());
        let node = |n: MtpNode| {
            let wd = n.width();
            &taps.nodes[node_at(n)].1[t * wd..(t + 1) * wd]
        };
        let f64s = |v: &[f32]| v.iter().map(|&x| f64::from(x)).collect::<Vec<f64>>();

        // eh_proj: the pack `[e·r_e·enorm | h_s·r_h·hnorm_s]` a stream, then
        // the projection of its 5,120 values.
        let e = h.embd.row(g.tokens[at] as usize);
        let hid = &g.states[at * WIDE..(at + 1) * WIDE];
        let inv_rms = |v: &[f32]| {
            1.0 / (v.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / v.len() as f64
                + f64::from(h.eps))
            .sqrt()
        };
        let (re, rh) = (inv_rms(&e), inv_rms(hid));
        let (mut y, mut abs) = (Vec::with_capacity(WIDE), Vec::with_capacity(WIDE));
        for s in 0..STREAMS {
            let mut pack = Vec::with_capacity(2 * HIDDEN);
            pack.extend((0..HIDDEN).map(|i| f64::from(e[i]) * re * f64::from(h.enorm[i])));
            pack.extend((0..HIDDEN).map(|i| {
                let j = s * HIDDEN + i;
                f64::from(hid[j]) * rh * f64::from(h.hnorm[j])
            }));
            let (ys, a) = h.eh.apply_abs(&pack, 0..HIDDEN);
            y.extend(ys);
            abs.extend(a);
        }
        let eh = &taps.eh[t * WIDE..(t + 1) * WIDE];
        let (d, a) = dot_pin(eh, &y, &abs);
        // The pack's own error reaches the output through |W|: at most
        // pack_rel of Σ|w·x|, beside the projection's γ.
        let g = gamma(dot_depth(2 * HIDDEN));
        p.eh.hold(d, (g + pack_rel() * (1.0 + g)) * a, &label);

        // The attention site: its mix of eh_proj's streams, the output
        // projection, the combine.
        let attn = mix_ref(h.geo, h.attn.weights(), eh, h.eps);
        p.attn_in.hold(
            max_rel(node(MtpNode::AttnIn), &attn.mixed),
            mix_band(),
            &label,
        );
        let gated = f64s(node(MtpNode::AttnGated));
        let (y, abs) = h.out.apply_abs(&gated, 0..HIDDEN);
        let (d, a) = dot_pin(node(MtpNode::AttnOut), &y, &abs);
        p.attn_out
            .hold(d, gamma(dot_depth(gated.len())) * a, &label);
        let wgt = attn.wgt.as_deref().unwrap_or(&[]);
        let (d, b) = combine_pin(node(MtpNode::AttnCombined), eh, node(MtpNode::AttnOut), wgt);
        p.attn_combined.hold(d, b, &label);

        // The feed-forward site: its mix, the routed slots, the shared
        // expert, their gated sum.
        let ffn = mix_ref(h.geo, h.ffn.weights(), node(MtpNode::AttnCombined), h.eps);
        p.ffn_in.hold(
            max_rel(node(MtpNode::FfnIn), &ffn.mixed),
            mix_band(),
            &label,
        );
        let slots = taps.slots_row;
        // PIN(2026-09-30): the routed sum's bound per value — each slot's
        // down (γ(640/32 + 5) of Σ|w·x|, `dot_depth`) through its weight,
        // and `ds41_ffn_card_acc_10`'s ten fused multiply-adds in slot order, γ(10)
        // of Σ|w·down|. Measured on the set's 127 rows: at worst 0.006 of it.
        let (mut acc, mut bound) = (vec![0.0f64; HIDDEN], vec![0.0f64; HIDDEN]);
        for j in 0..N_USED {
            let e = taps.ids[t * slots + j] as usize;
            let w = f64::from(taps.weights[t * slots + j]);
            let x = f64s(&taps.routed_h[(t * N_USED + j) * FF..(t * N_USED + j + 1) * FF]);
            let (dj, aj) = h.down_e.apply_abs(&x, e * HIDDEN..(e + 1) * HIDDEN);
            for i in 0..HIDDEN {
                acc[i] += w * dj[i];
                // The down's own bound through the weight, and the ten
                // slots' fused multiply-adds (depth 10) over |w·down|.
                bound[i] += w.abs() * (gamma(dot_depth(FF)) * aj[i] + gamma(N_USED) * dj[i].abs());
            }
        }
        let routed = node(MtpNode::Routed);
        let den = acc
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);
        let num = routed
            .iter()
            .zip(&acc)
            .map(|(&o, &v)| (f64::from(o) - v).powi(2))
            .sum::<f64>()
            .sqrt();
        let b = bound.iter().map(|v| v * v).sum::<f64>().sqrt();
        p.routed.hold(nan_inf(num / den), b / den, &label);
        let x = f64s(&taps.shared_h[t * FF..(t + 1) * FF]);
        let (y, abs) = h.down_s.apply_abs(&x, 0..HIDDEN);
        let (d, a) = dot_pin(node(MtpNode::Shared), &y, &abs);
        p.shared.hold(d, gamma(dot_depth(FF)) * a, &label);
        // `q38_shared_add`: the product rounded, then the sum rounded.
        let gate = taps.weights[t * slots + N_USED];
        let differ = routed
            .iter()
            .zip(node(MtpNode::Shared))
            .zip(node(MtpNode::FfnOut))
            .filter(|&((r, s), o)| (*r + *s * gate).to_bits() != o.to_bits())
            .count();
        p.ffn_out.hold(differ as f64, 0.0, &label);

        // The layer's output: the head site's combine.
        let wgt = ffn.wgt.as_deref().unwrap_or(&[]);
        let (d, b) = combine_pin(
            l_out,
            node(MtpNode::AttnCombined),
            node(MtpNode::FfnOut),
            wgt,
        );
        p.l_out.hold(d, b, &label);
        let head = mix_ref(h.geo, h.head_site.weights(), l_out, h.eps);
        p.head_in.hold(
            max_rel(node(MtpNode::HeadIn), &head.mixed),
            mix_band(),
            &label,
        );
    }

    /// The logits pin of one head row: our logits `ours` against the full
    /// head's projection of our `HeadIn` row `x`.
    fn pin_logits(h: &Host<'_>, x: &[f32], ours: &[f32], p: &mut Pin, label: &dyn Fn() -> String) {
        let x: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let (y, abs) = h.head.apply_abs(&x, 0..h.head.rows);
        let (d, a) = dot_pin(ours, &y, &abs);
        p.hold(d, gamma(dot_depth(HIDDEN)) * a, label);
    }

    /// The worst of each clause over the replay, and the counts.
    struct Replay {
        eh_worst: f64,
        eh_fail: usize,
        /// The layer's output's and the logits' distance from ik's: printed
        /// inside (l) and (h), not held.
        l_out_worst: f64,
        logits_worst: f64,
        rows: usize,
        heads: usize,
        flips: usize,
        flips_bad: usize,
        argmax_same: usize,
        l_out_held: usize,
        /// Held rows whose layer output is past the fixture's band ([`Vs::Fixture`]).
        l_out_bad: usize,
        heads_held: usize,
        argmax_tie: usize,
        argmax_bad: usize,
        kernel_bad: usize,
        /// The held head rows' smallest top-2 margin of ik's logits, over
        /// its tie cap, and the row: how near (h) came to a tie.
        margin_min: Option<(f64, f64, String)>,
        pins: Pins,
    }

    impl Replay {
        fn new() -> Replay {
            Replay {
                eh_worst: 0.0,
                eh_fail: 0,
                l_out_worst: 0.0,
                logits_worst: 0.0,
                rows: 0,
                heads: 0,
                flips: 0,
                flips_bad: 0,
                argmax_same: 0,
                l_out_held: 0,
                l_out_bad: 0,
                heads_held: 0,
                argmax_tie: 0,
                argmax_bad: 0,
                kernel_bad: 0,
                margin_min: None,
                pins: Pins::new(),
            }
        }
    }

    /// What a replay compares with ik's output.
    #[derive(Clone, Copy)]
    enum Vs {
        /// Nothing: the node-local pins (n) and (h)'s kernel against our own full head only.
        Nothing,
        /// The real set, the oracle clauses: (e), (l) and (h) at [`eh_band`], [`router_band`] and
        /// [`logits_band`], the argmax ik's or, under the cap, ik's runner-up, the layer's
        /// output printed.
        Real,
        /// The fixture set, (elhfx): the bands counted over the draft file's sites
        /// ([`fx_bands`]), the argmax a tie band ([`flip::head_tie`]), the layer's output held.
        Fixture(FxBands),
    }

    /// The fixture's bands against ik on the draft layer's rows: `eh_proj`, the router's input,
    /// the layer's output and the logits, as the real clause's PINs count them over the draft
    /// file's own site types ([`fx_bands`]).
    #[derive(Clone, Copy)]
    struct FxBands {
        eh: f64,
        router: f64,
        l_out: f64,
        logits: f64,
    }

    /// Replay graph `g` in runs of up to [`MTP_ROWS`] rows, the full head,
    /// the taps armed, into `r`. ik's rows are the feed in both tiers (the
    /// fixture copies the vocabulary and the hidden rows are finite inputs);
    /// `vs` is what the oracle clauses compare: (e), (l) and (h)'s comparison
    /// with ik's output are made only when it names a set, and the node-local
    /// pins (n) and (h)'s kernel against our own full head are made always.
    fn replay(
        m: &mut Qwen38Model,
        g: &IkGraph,
        host: &Host<'_>,
        r: &mut Replay,
        vs: Vs,
    ) -> Result<(), GateError> {
        let vs_ik = !matches!(vs, Vs::Nothing);
        let (eh_b, router_b, logits_b) = match vs {
            Vs::Fixture(b) => (b.eh, b.router, b.logits),
            Vs::Nothing | Vs::Real => (eh_band(), router_band(), logits_band()),
        };
        let n = g.tokens.len();
        let vocab = m.body("replay")?.vocab();
        if g.logits.len() != g.out_ids.len() * vocab {
            return Err(format!(
                "{}: result_output holds {} values, {} output rows of {vocab}",
                g.label(),
                g.logits.len(),
                g.out_ids.len()
            )
            .into());
        }
        let mut c0 = 0;
        while c0 < n {
            let rows = MTP_ROWS.min(n - c0);
            let w = walk(
                m,
                DraftFeed::Rows(Feed {
                    tokens: &g.tokens[c0..c0 + rows],
                    pos0: g.pos[c0],
                    hidden: Hidden::Host(&g.states[c0 * WIDE..(c0 + rows) * WIDE]),
                    media: None,
                }),
                DraftHead::Full,
                WalkMode::Eager,
            )?;
            let taps = m.mtp_taps()?;
            let mut flipped = vec![false; rows];
            for (t, flipped_t) in flipped.iter_mut().enumerate().take(rows) {
                let at = c0 + t;
                let span = at * WIDE..(at + 1) * WIDE;
                let e = if vs_ik {
                    rel(&taps.eh[t * WIDE..(t + 1) * WIDE], &g.eh[span.clone()])
                } else {
                    0.0
                };
                r.eh_worst = r.eh_worst.max(e);
                if e > eh_b {
                    r.eh_fail += 1;
                    println!("(e) {} row {at}: eh_proj {e:.3e} FAIL", g.label());
                }
                let ours_ids: Vec<u32> =
                    taps.ids[t * taps.slots_row..t * taps.slots_row + N_USED].to_vec();
                let ours_v = &taps.logits[t * taps.logits_row..t * taps.logits_row + N_EXPERT];
                let ik_v = &g.router[at * N_EXPERT..(at + 1) * N_EXPERT];
                let ik_ids = &g.topk[at * N_USED..(at + 1) * N_USED];
                if let Some(f) = vs_ik
                    .then(|| {
                        Flip::between(
                            (g.layer, at),
                            (&ours_ids, ours_v),
                            (ik_ids, ik_v),
                            flip::margin(ik_v, ik_ids),
                        )
                    })
                    .flatten()
                {
                    *flipped_t = true;
                    r.flips += 1;
                    let cap = flip::flip_cap(router_b, ik_v);
                    let gap_cap = flip::margin_cap(router_b, ik_v);
                    let ok = f.allowed(cap) && f.pairs.iter().all(|&(_, _, gap, _)| gap <= gap_cap);
                    if !ok {
                        r.flips_bad += 1;
                    }
                    println!(
                        "(l) {} {} (gap cap {gap_cap:.3e}) {}",
                        g.label(),
                        f.line("mtp", cap),
                        verdict(ok)
                    );
                }
                let l_out = &w.l_out[t * WIDE..(t + 1) * WIDE];
                pin_row(host, g, at, &taps, t, l_out, &mut r.pins);
                if vs_ik && !*flipped_t {
                    r.l_out_held += 1;
                    let d = rel(l_out, &g.l_out[span]);
                    r.l_out_worst = r.l_out_worst.max(d);
                    let held_band = match vs {
                        Vs::Fixture(b) => Some(b.l_out),
                        Vs::Nothing | Vs::Real => None,
                    };
                    if let Some(band) = held_band.filter(|&band| d > band) {
                        r.l_out_bad += 1;
                        println!(
                            "(l) {} row {at}: l_out {d:.3e} past its band {band:.3e} FAIL",
                            g.label()
                        );
                    }
                }
            }
            r.rows += rows;
            for c in 0..rows {
                let ours = column(&w.logits, rows, c);
                let (top, _) = top2(&ours);
                if w.rows != vocab || w.draft.tokens[c] as usize != top {
                    r.kernel_bad += 1;
                    println!(
                        "(h) {} row {}: the draft names {}, our logits' argmax {top} over {} rows \
                         FAIL",
                        g.label(),
                        c0 + c,
                        w.draft.tokens[c],
                        w.rows
                    );
                }
            }
            for (k, &o) in g.out_ids.iter().enumerate() {
                if !(c0..c0 + rows).contains(&o) {
                    continue;
                }
                let c = o - c0;
                r.heads += 1;
                let ours = column(&w.logits, rows, c);
                let head_in = &taps.nodes[node_at(MtpNode::HeadIn)].1[c * HIDDEN..(c + 1) * HIDDEN];
                pin_logits(host, head_in, &ours, &mut r.pins.logits, &|| {
                    format!("{} row {o}", g.label())
                });
                if !vs_ik {
                    continue;
                }
                let ik = &g.logits[k * vocab..(k + 1) * vocab];
                let (top, _) = top2(&ours);
                let (ik_top, ik_2) = top2(ik);
                let margin = f64::from(ik[ik_top]) - f64::from(ik[ik_2]);
                let clears = margin > flip::margin_cap(logits_b, ik);
                if flipped[c] {
                    println!(
                        "(h) {} row {o}: argmax {top} ik {ik_top} on a flip, not held",
                        g.label()
                    );
                    continue;
                }
                r.heads_held += 1;
                r.logits_worst = r.logits_worst.max(rel(&ours, ik));
                let cap = flip::margin_cap(logits_b, ik);
                if r.margin_min
                    .as_ref()
                    .is_none_or(|(_, q, _)| margin / cap < *q)
                {
                    r.margin_min = Some((margin, margin / cap, format!("{} row {o}", g.label())));
                }
                let tie = match vs {
                    Vs::Fixture(_) => flip::head_tie(ik, top as u32, logits_b),
                    Vs::Nothing | Vs::Real => !clears && top == ik_2,
                };
                if top == ik_top {
                    r.argmax_same += 1;
                } else if tie {
                    r.argmax_tie += 1;
                    let (shown, why) = match vs {
                        Vs::Fixture(_) => {
                            (flip::head_cap(logits_b, ik), "within the head's tie cap")
                        }
                        Vs::Nothing | Vs::Real => (cap, "ik's runner-up"),
                    };
                    println!(
                        "(h) {} row {o}: argmax {top}, ik's {ik_top} by {margin:.3e} under the cap \
                         {shown:.3e}: {why}, a tie",
                        g.label()
                    );
                } else {
                    r.argmax_bad += 1;
                    println!(
                        "(h) {} row {o}: argmax {top}, ik's {ik_top} (runner-up {ik_2}) by \
                         {margin:.3e}, cap {:.3e} FAIL",
                        g.label(),
                        flip::margin_cap(logits_b, ik)
                    );
                }
            }
            c0 += rows;
        }
        Ok(())
    }

    /// (r): the list head against the full head over the rows of `g`'s last
    /// run, as one walk of all of them (the m-column entry) and as one walk
    /// of its last row (the one-column entry).
    fn list_head(m: &mut Qwen38Model, g: &IkGraph, ids: &[u32]) -> Result<bool, GateError> {
        let n = g.tokens.len();
        let mut all = true;
        for rows in [MTP_ROWS.min(n), 1] {
            let c0 = n - rows;
            let feed = DraftFeed::Rows(Feed {
                tokens: &g.tokens[c0..],
                pos0: g.pos[c0],
                hidden: Hidden::Host(&g.states[c0 * WIDE..]),
                media: None,
            });
            let full = walk(m, feed, DraftHead::Full, WalkMode::Eager)?;
            let list = walk(m, feed, DraftHead::Rows, WalkMode::Eager)?;
            let mut ok = list.rows == ids.len();
            let mut differ = 0usize;
            let (mut p_worst, mut p_bad) = (0.0f64, 0usize);
            for c in 0..rows {
                let lc = column(&list.logits, rows, c);
                let fc = column(&full.logits, rows, c);
                differ += lc
                    .iter()
                    .zip(ids)
                    .filter(|&(v, &id)| v.to_bits() != fc[id as usize].to_bits())
                    .count();
                let (top, _) = top2(&lc);
                let token_ok = list.draft.tokens[c] == ids[top];
                ok &= token_ok;
                if !token_ok {
                    println!(
                        "(r) row {c}: the list head names {}, its argmax row {top} is id {} FAIL",
                        list.draft.tokens[c], ids[top]
                    );
                }
                for (w, v) in [(&list, &lc), (&full, &fc)] {
                    let want = p_max(v);
                    let e = (f64::from(w.draft.p[c]) - want).abs() / want;
                    p_worst = p_worst.max(e / p_bound(v.len()));
                    if e > p_bound(v.len()) {
                        p_bad += 1;
                        println!(
                            "(r) row {c}: p {} against the host's {want:.9} over {} rows, {e:.3e} \
                             past {:.3e} FAIL",
                            w.draft.p[c],
                            v.len(),
                            p_bound(v.len())
                        );
                    }
                }
            }
            ok &= differ == 0 && p_bad == 0;
            println!(
                "(r) a walk of {rows} rows of {} ({}): list logits = the full head's rows of their \
                 ids, {differ} differ; tokens mapped; p within its bound (worst {p_worst:.2} of it) \
                 {}",
                g.label(),
                list.rows,
                verdict(ok)
            );
            all &= ok;
        }
        Ok(all)
    }

    /// The word a fixture clause prints for its outcome where it holds the rows off every flip:
    /// `unexercised` when none was (which rows a flip leaves off its path moves with the engine's
    /// rounding, so none is neither a pass nor a fail), else [`verdict`].
    fn outcome(ok: bool, held: usize) -> &'static str {
        if ok && held == 0 {
            "unexercised"
        } else {
            verdict(ok)
        }
    }

    /// The fixture-oracle clauses a run declares, held by `tier::expect_fixture_oracle` at the
    /// gate's end in both tiers: (tfx)'s shift pairing ([`pairing_fx`]) and (elhfx) ([`elh_fx`]).
    const FX_ORACLE: usize = 2;

    /// The fixture's bands against ik on the draft layer's rows ([`FxBands`]), counted as the
    /// real clause's PINs count them ([`eh_band`], [`router_band`], [`logits_band`]: one term a
    /// site, an expert stack's gate and up as one) over the sites the files' types name. A site's
    /// term is `site_rel(ty, Arm::Gemv).joint()`: the draft walk reads each projection's input as
    /// f32 rows (the gemvs and `q8_0_gemv_sel_f32`) and ik quantizes it to the weight type's
    /// activation form, at the worst crest ([`site_rel`]); a stack's gate and up and the attention's
    /// q, k and v are one term each, the worst of the two or three; the terms add in quadrature.
    /// The router's input reads `eh_proj`, the attention's input and its output projection; the
    /// layer's output beside them the routed and the shared expert's gate·up and down; the
    /// logits beside those the head (`output`, the target's). Printed beside the real clause's.
    fn fx_bands(target: &Split, draft: &Split) -> Result<FxBands, GateError> {
        let site = |file: &Split, tensor: &str| -> Result<f64, GateError> {
            let ty = file
                .find(tensor)
                .map(|(_, t)| t.ty)
                .ok_or_else(|| format!("fx bands: {tensor} is not in the file"))?;
            let rel = site_rel(ty, Arm::Gemv).ok_or_else(|| {
                format!(
                    "fx bands: {tensor} is {ty:?}, which no launch reads as the gemv \
                     (act_rule::site_rel)"
                )
            })?;
            Ok(rel.joint())
        };
        let named = |stem: &str| format!("blk.{}.{stem}.weight", draft_layer());
        let worst = |stems: &[&str]| -> Result<f64, GateError> {
            let mut w = 0.0f64;
            for stem in stems {
                w = w.max(site(draft, &named(stem))?);
            }
            Ok(w)
        };
        let eh = site(draft, &named("nextn.eh_proj"))?;
        let attn_in = worst(&["attn_q", "attn_k", "attn_v"])?;
        let attn_out = site(draft, &named("attn_output"))?;
        let routed_gate_up = worst(&["ffn_gate_exps", "ffn_up_exps"])?;
        let routed_down = site(draft, &named("ffn_down_exps"))?;
        let shared_gate_up = worst(&["ffn_gate_shexp", "ffn_up_shexp"])?;
        let shared_down = site(draft, &named("ffn_down_shexp"))?;
        let head = site(target, "output.weight")?;
        let quadrature = |terms: &[f64]| terms.iter().map(|q| q * q).sum::<f64>().sqrt();
        let router = quadrature(&[eh, attn_in, attn_out]);
        let l_out = quadrature(&[
            eh,
            attn_in,
            attn_out,
            routed_gate_up,
            routed_down,
            shared_gate_up,
            shared_down,
        ]);
        let logits = quadrature(&[l_out, head]);
        println!(
            "fx bands: eh {eh:.4e} router {router:.4e} l_out {l_out:.4e} logits {logits:.4e} from \
             the terms eh_proj {eh:.3e}, attention in {attn_in:.3e} and out {attn_out:.3e}, routed \
             gate·up {routed_gate_up:.3e} and down {routed_down:.3e}, shared gate·up \
             {shared_gate_up:.3e} and down {shared_down:.3e}, head {head:.3e}; the real clause's \
             PINs eh {:.4e} router {:.4e} logits {:.4e}",
            eh_band(),
            router_band(),
            logits_band()
        );
        Ok(FxBands {
            eh,
            router,
            l_out,
            logits,
        })
    }

    /// (elhfx): every graph of ik's MTP set dumped on the fixture, replayed teacher-forced on
    /// ik's own inputs (the fixture set's tokens, positions and hidden rows) as the real set is:
    /// (e) `eh_proj` within [`FxBands::eh`] of ik's, a row at a time; (l) each row's routed set
    /// ik's, but where it is not, allowed only under the pair cap and margin bound at
    /// [`FxBands::router`]; the layer's output of each row off a flip within [`FxBands::l_out`];
    /// (h) each row's argmax ik's or within the head's tie cap of ik's top at
    /// [`FxBands::logits`] ([`flip::head_tie`]); the node-local pins (n) and the draft's token
    /// against our full head's argmax on the same rows. Where every row of (l) or (h) lies on a
    /// flip no row is held: the clause holds what it can and prints `unexercised` in place of its
    /// verdict ([`outcome`]).
    fn elh_fx(
        m: &mut Qwen38Model,
        host: &Host<'_>,
        (target, draft): (&Split, &Split),
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "(elhfx) every graph of ik's fixture set replayed: eh_proj, routes, \
                              l_out and the argmax against ik's";
        if !tier::run_clause(CLAUSE, Tag::FixtureOracle)? {
            return Ok(true);
        }
        let dir = tier::fixture_set(CLAUSE, &fixture::MTP, fixture::MTP_SET)?;
        let set = MtpSet::open(&dir, &fixture::MTP)?;
        let bands = fx_bands(target, draft)?;
        let layer = ik_layer(&set)?;
        let mut r = Replay::new();
        for (b, g) in graphs_of(&set) {
            let ig = IkGraph::read(&set, b, g, layer)?;
            replay(m, &ig, host, &mut r, Vs::Fixture(bands))?;
        }
        let e_ok = r.eh_fail == 0 && r.rows > 0;
        println!(
            "(elhfx e) {} rows: eh_proj worst {:.3e} (band {:.3e}, {:.3} of it), {} past {}",
            r.rows,
            r.eh_worst,
            bands.eh,
            r.eh_worst / bands.eh,
            r.eh_fail,
            verdict(e_ok)
        );
        let l_ok = r.flips_bad == 0 && r.l_out_bad == 0;
        if r.l_out_held == 0 {
            println!(
                "(elhfx l) unexercised past the flips (l_out held 0 of {} rows)",
                r.rows
            );
        }
        println!(
            "(elhfx l) {} rows, {} on flips, {} flips not allowed; l_out's distance from ik's on \
             the {} rows held, worst {:.3e} (band {:.3e}, {:.3} of it), {} past it {}",
            r.rows,
            r.flips,
            r.flips_bad,
            r.l_out_held,
            r.l_out_worst,
            bands.l_out,
            r.l_out_worst / bands.l_out,
            r.l_out_bad,
            outcome(l_ok, r.l_out_held)
        );
        let h_ok = r.argmax_bad == 0 && r.kernel_bad == 0;
        if r.heads_held == 0 {
            println!(
                "(elhfx h) unexercised past the flips (heads held 0 of {} head rows)",
                r.heads
            );
        }
        println!(
            "(elhfx h) {} head rows, {} held: argmax = ik's {}, ties {}, wrong {}; the draft's \
             token = our argmax but {}; the logits' distance from ik's {:.3e} (printed, band \
             {:.3e}) {}",
            r.heads,
            r.heads_held,
            r.argmax_same,
            r.argmax_tie,
            r.argmax_bad,
            r.kernel_bad,
            r.logits_worst,
            bands.logits,
            outcome(h_ok, r.heads_held)
        );
        let mut n_ok = true;
        for p in r.pins.all() {
            println!("(elhfx n) {}", p.line());
            n_ok &= p.ok();
        }
        Ok(e_ok && l_ok && h_ok && n_ok)
    }

    /// (tfx): the target's streams after the prompt against the hidden rows of ik's fixture MTP
    /// set's warmup graph: ours at `q` lies closest to ik's at `q + 1` — ik's row `p` reads the
    /// target's hidden of `p − 1`, and the fixture's rows at other positions are near orthogonal —
    /// closer than at `q` and at `q + 2`. The distance itself is printed: a free walk of the
    /// prompt's last rows sits on a flip's path, which no band covers. `ours` are the last
    /// `rows` positions' streams from `first` on.
    fn pairing_fx(
        ours: &[f32],
        (rows, first): (usize, usize),
        prompt: &[u32],
    ) -> Result<bool, GateError> {
        const CLAUSE: &str = "(tfx) the target's streams after the prompt against ik's fixture MTP \
                              hidden rows, shifted by one";
        if !tier::run_clause(CLAUSE, Tag::FixtureOracle)? {
            return Ok(true);
        }
        let dir = tier::fixture_set(CLAUSE, &fixture::MTP, fixture::MTP_SET)?;
        let set = MtpSet::open(&dir, &fixture::MTP)?;
        if set.tokens.as_deref() != Some(prompt) {
            return Err(format!(
                "{}: its tokens are not the prompt the target ran ({} ids)",
                dir.display(),
                prompt.len()
            )
            .into());
        }
        let (block, graph) = graphs_of(&set)
            .into_iter()
            .find(|&(_, g)| g == Graph::Warmup)
            .ok_or_else(|| format!("{}: the set has no warmup graph", dir.display()))?;
        let warm = IkGraph::read(&set, block, graph, ik_layer(&set)?)?;
        let at = |pos: usize| warm.pos.iter().position(|&p| p as usize == pos);
        let mut by_shift = [f64::INFINITY; 3];
        for (s, slot) in by_shift.iter_mut().enumerate() {
            let mut worst = 0.0f64;
            let mut pairs = 0;
            for t in 0..rows {
                if let Some(r) = at(first + t + s) {
                    worst = worst.max(rel(
                        &ours[t * WIDE..(t + 1) * WIDE],
                        &warm.states[r * WIDE..(r + 1) * WIDE],
                    ));
                    pairs += 1;
                }
            }
            println!(
                "(tfx) the target's streams at q against ik's fixture MTP hidden at q + {s}: \
                 {pairs} rows, worst {worst:.3e}"
            );
            if pairs > 0 {
                *slot = worst;
            }
        }
        let ok = by_shift[1] < by_shift[0] && by_shift[1] < by_shift[2];
        println!(
            "(tfx) ik's fixture row p reads the target's hidden of p − 1: the shift 1 reads closer \
             than 0 and 2 (worst {:.3e} at 1, printed); a flip's path lies under the last rows {}",
            by_shift[1],
            verdict(ok)
        );
        Ok(ok)
    }

    /// (t): the target's streams after the prompt against ik's warmup hidden
    /// rows, and a draft walk fed them in place against one fed them from the
    /// host.
    fn pairing(m: &mut Qwen38Model, prompt: &[u32], warm: &IkGraph) -> Result<bool, GateError> {
        let next = m.prompt38(prompt, Prompt38::Pass)?;
        let n = prompt.len();
        let rows = (n - 1) % MTP_ROWS + 1;
        let first = n - rows;
        let ours = m.target_streams(TargetRows::Pass, rows)?;
        let at = |pos: usize| warm.pos.iter().position(|&p| p as usize == pos);
        let mut by_shift = Vec::new();
        let vs_ik = tier::run_clause(
            "(t) the target's streams after the prompt against ik's MTP hidden rows, shifted by one",
            Tag::Oracle,
        )?;
        for s in 0..3usize {
            if !vs_ik {
                break;
            }
            let mut worst = 0.0f64;
            let mut pairs = 0;
            for t in 0..rows {
                if let Some(r) = at(first + t + s) {
                    worst = worst.max(rel(
                        &ours[t * WIDE..(t + 1) * WIDE],
                        &warm.states[r * WIDE..(r + 1) * WIDE],
                    ));
                    pairs += 1;
                }
            }
            println!(
                "(t) the target's streams at q against ik's MTP hidden at q + {s}: {pairs} rows, \
                 worst {worst:.3e}"
            );
            by_shift.push(if pairs > 0 { worst } else { f64::INFINITY });
        }
        let shift_ok = !vs_ik
            || (by_shift[1] <= FREE_BAND && by_shift[1] < by_shift[0] && by_shift[1] < by_shift[2]);
        if vs_ik {
            println!(
                "(t) ik's row p reads the target's hidden of p − 1, within {FREE_BAND:.2}: {}",
                verdict(shift_ok)
            );
        }
        // The walk starts one past the arena's first position — each of its
        // rows at q reads the hidden row at q − 1, the arena's rows 0..rows
        // (the prompt's last unit's rows, `first` on) — and runs one row past
        // the prompt's end, the prompt's own next id its last token.
        let mut tokens = prompt[first + 1..].to_vec();
        tokens.push(next);
        let pos0 = u32::try_from(first + 1)?;
        let host = walk(
            m,
            DraftFeed::Rows(Feed {
                tokens: &tokens,
                pos0,
                hidden: Hidden::Host(&ours),
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Eager,
        );
        let place = walk(
            m,
            DraftFeed::Rows(Feed {
                tokens: &tokens,
                pos0,
                hidden: Hidden::Target {
                    walk: TargetRows::Pass,
                    first: 0,
                },
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Eager,
        );
        let same = match (&host, &place) {
            (Ok(h), Ok(p)) => same_bits(h, p),
            _ => false,
        };
        println!(
            "(t) {rows} rows from {pos0} fed the target's streams in place = from the host, bit \
             for bit {}{}{}",
            verdict(same),
            host.err().map_or(String::new(), |e| format!(" ({e})")),
            place.err().map_or(String::new(), |e| format!(" ({e})")),
        );
        let fx_ok = pairing_fx(&ours, (rows, first), prompt)?;
        Ok(shift_ok && same && fx_ok)
    }

    /// (g): captured walks against eager ones.
    fn graphs(m: &mut Qwen38Model, g: &IkGraph) -> Result<bool, GateError> {
        m.set_mtp_taps(false)?;
        let mut ok = true;
        let rows_of = |k: usize| {
            DraftFeed::Rows(Feed {
                tokens: &g.tokens[..k],
                pos0: g.pos[0],
                hidden: Hidden::Host(&g.states[..k * WIDE]),
                media: None,
            })
        };
        let mut shapes: Vec<(usize, DraftHead)> =
            (1..=MTP_GRAPH_ROWS).map(|k| (k, DraftHead::Rows)).collect();
        shapes.push((1, DraftHead::Full));
        for (k, head) in shapes {
            let e = walk(m, rows_of(k), head, WalkMode::Eager)?;
            let c = walk(m, rows_of(k), head, WalkMode::Graph)?;
            let again = walk(m, rows_of(k), head, WalkMode::Graph)?;
            let same = same_bits(&e, &c) && same_bits(&e, &again);
            ok &= same;
            println!(
                "(g) {k} rows, {head:?} head: captured = eager, twice {}",
                verdict(same)
            );
        }
        let pos1 = g.pos[0] + 1;
        let own = |p: u32| DraftFeed::Own { pos0: p };
        let x = walk(m, rows_of(1), DraftHead::Rows, WalkMode::Eager)?;
        let own_e = [
            walk(m, own(pos1), DraftHead::Rows, WalkMode::Eager)?,
            walk(m, own(pos1 + 1), DraftHead::Rows, WalkMode::Eager)?,
        ];
        walk(m, rows_of(1), DraftHead::Rows, WalkMode::Eager)?;
        let own_g = [
            walk(m, own(pos1), DraftHead::Rows, WalkMode::Graph)?,
            walk(m, own(pos1 + 1), DraftHead::Rows, WalkMode::Graph)?,
        ];
        walk(m, rows_of(1), DraftHead::Rows, WalkMode::Eager)?;
        let host = walk(
            m,
            DraftFeed::Rows(Feed {
                tokens: &x.draft.tokens,
                pos0: pos1,
                hidden: Hidden::Host(&x.l_out),
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Eager,
        )?;
        let chain_ok = same_bits(&own_e[0], &own_g[0]) && same_bits(&own_e[1], &own_g[1]);
        let host_ok = same_bits(&own_e[0], &host);
        ok &= chain_ok && host_ok;
        println!(
            "(g) two own rows, captured then replayed = eager {}; the first = a host walk of the \
             last token and streams {}",
            verdict(chain_ok),
            verdict(host_ok)
        );
        // A window's chain — a refresh of two rows, then two own walks, one
        // readback — against the same walks one by one: each walk's last id
        // and probability, bit for bit.
        let chain = m.mtp_chain(rows_of(2), 2, DraftHead::Rows, WalkMode::Graph)?;
        let one = [
            walk(m, rows_of(2), DraftHead::Rows, WalkMode::Eager)?,
            walk(m, own(g.pos[0] + 2), DraftHead::Rows, WalkMode::Eager)?,
            walk(m, own(g.pos[0] + 3), DraftHead::Rows, WalkMode::Eager)?,
        ];
        let want: Vec<(u32, u32)> = one
            .iter()
            .map(|w| {
                let last = w.draft.tokens.len() - 1;
                (w.draft.tokens[last], w.draft.p[last].to_bits())
            })
            .collect();
        let got: Vec<(u32, u32)> = chain
            .tokens
            .iter()
            .zip(&chain.p)
            .map(|(&t, p)| (t, p.to_bits()))
            .collect();
        let chain_walks = got == want;
        ok &= chain_walks;
        println!(
            "(g) a chain of two rows and two own walks = its walks one by one: ids {:?} {}",
            chain.tokens,
            verdict(chain_walks)
        );
        let nodes = m
            .body("graphs")?
            .mtp()
            .map(|d| d.graph_nodes())
            .unwrap_or_default();
        for (rows, own, head, n) in &nodes {
            let want = GRAPH_NODES.get(rows - 1).copied().unwrap_or(0);
            let fits = *n == want;
            ok &= fits;
            println!(
                "(g) capture rows={rows} own={own} head={head:?}: {n} nodes (want {want}) {}",
                verdict(fits)
            );
        }
        let all = nodes.len() == MTP_GRAPH_ROWS + 2;
        ok &= all;
        println!(
            "(g) {} captures (want {}) {}",
            nodes.len(),
            MTP_GRAPH_ROWS + 2,
            verdict(all)
        );
        Ok(ok)
    }

    /// Whether `r` is a refusal whose message holds `want`.
    fn refused<T>(what: &str, r: Result<T, GpuError>, want: &str) -> bool {
        let got = r.err().map_or("ran".to_string(), |e| e.to_string());
        let ok = got.contains(want);
        println!("(f) {what}: {got} {}", verdict(ok));
        ok
    }

    /// (f): the refusals, then the NaN.
    fn refusals(m: &mut Qwen38Model, g: &IkGraph) -> Result<bool, GateError> {
        let (vocab, ctx) = {
            let b = m.body("refusals")?;
            (b.vocab(), b.mtp().map_or(0, |d| d.ctx()))
        };
        let over = MTP_STORE_ROWS + 1;
        let tokens: Vec<u32> = g.tokens.iter().copied().cycle().take(over).collect();
        let states: Vec<f32> = g.states.iter().copied().cycle().take(over * WIDE).collect();
        let feed = |k: usize, pos0: u32| {
            DraftFeed::Rows(Feed {
                tokens: &tokens[..k],
                pos0,
                hidden: Hidden::Host(&states[..k * WIDE]),
                media: None,
            })
        };
        let (h, e) = (DraftHead::Rows, WalkMode::Eager);
        let mut ok = true;
        ok &= refused("no rows", m.mtp_draft(feed(0, 0), h, e), "a walk of 0 rows");
        ok &= refused(
            "nine rows",
            m.mtp_draft(feed(9, 0), h, e),
            "a walk of 9 rows",
        );
        ok &= refused(
            "five captured rows",
            m.mtp_draft(feed(5, 0), h, WalkMode::Graph),
            "a walk of 5 rows",
        );
        ok &= refused(
            "a store walk past its rows",
            m.mtp_walk(feed(over, 0), h, WalkMode::Store),
            &format!("a walk of {over} rows"),
        );
        let last = u32::try_from(ctx - 1)?;
        ok &= refused(
            "rows past the store",
            m.mtp_draft(feed(2, last), h, e),
            "in a store of",
        );
        let past = [u32::try_from(vocab)?];
        ok &= refused(
            "a token past the vocabulary",
            m.mtp_draft(
                DraftFeed::Rows(Feed {
                    tokens: &past,
                    pos0: 0,
                    hidden: Hidden::Host(&states[..WIDE]),
                    media: None,
                }),
                h,
                e,
            ),
            "past the vocabulary",
        );
        ok &= refused(
            "a short hidden slice",
            m.mtp_draft(
                DraftFeed::Rows(Feed {
                    tokens: &tokens[..2],
                    pos0: 0,
                    hidden: Hidden::Host(&states[..WIDE]),
                    media: None,
                }),
                h,
                e,
            ),
            "hidden values for 2 rows",
        );
        // Media rows are a store walk's alone, at the embedding's width: an
        // eager walk and a chain whose rows carry them are refused, and so is
        // a store walk whose rows are of another width, each through the one
        // accessor.
        let at = 0..1;
        let (row, wide) = (vec![0u16; HIDDEN], vec![0u16; WIDE]);
        let media = CallRows::new(0, HIDDEN, [(&at, &row[..])]).map_err(|e| e.to_string())?;
        let too_wide = CallRows::new(0, WIDE, [(&at, &wide[..])]).map_err(|e| e.to_string())?;
        let with = |media| {
            DraftFeed::Rows(Feed {
                tokens: &tokens[..1],
                pos0: 0,
                hidden: Hidden::Host(&states[..WIDE]),
                media: Some(media),
            })
        };
        ok &= refused(
            "a walk with media rows",
            m.mtp_draft(with(&media), h, e),
            "outside a store walk",
        );
        ok &= refused(
            "a chain with media rows",
            m.mtp_chain(with(&media), 1, h, e),
            "outside a store walk",
        );
        ok &= refused(
            "a store walk with media rows of the hidden width",
            m.mtp_walk(with(&too_wide), h, WalkMode::Store),
            "media rows of 10240 values a row",
        );
        m.set_mtp_taps(true)?;
        ok &= refused(
            "a capture with the taps armed",
            m.mtp_draft(feed(1, 0), h, WalkMode::Graph),
            "the taps off for a captured walk",
        );
        m.set_mtp_taps(false)?;
        ok &= refused(
            "a walk past the positions the store holds",
            m.mtp_draft(feed(1, last), h, e),
            "the draft's store holds this sequence's positions below",
        );
        let mut nan = states[..WIDE].to_vec();
        nan[7] = f32::NAN;
        let r = m.mtp_draft(
            DraftFeed::Rows(Feed {
                tokens: &tokens[..1],
                pos0: 0,
                hidden: Hidden::Host(&nan),
                media: None,
            }),
            h,
            e,
        );
        let raised = match &r {
            Err(GpuError::Fault { fault, .. }) => {
                fault.layer as usize == draft_layer()
                    && fault.sites & (1 << FaultSite::HcMix as u32) != 0
            }
            _ => false,
        };
        println!(
            "(f) a NaN in a hidden row: {} {}",
            r.err().map_or("ran".to_string(), |e| e.to_string()),
            verdict(raised)
        );
        m.reset()?;
        // A captured walk bakes the rotation table's address: a load whose draft has captured
        // one is refused a media load, and still after a reset, which keeps the captures.
        m.mtp_draft(feed(1, 0), h, WalkMode::Graph)?;
        ok &= refused(
            "a media load after a captured draft walk",
            m.enable_media(),
            "a load whose MTP draft has no captured walks",
        );
        m.reset()?;
        ok &= refused(
            "a media load after a captured draft walk and a reset",
            m.enable_media(),
            "a load whose MTP draft has no captured walks",
        );
        ok &= refused(
            "the draft's own row after a reset",
            m.mtp_draft(DraftFeed::Own { pos0: 0 }, h, e),
            "a walk before the draft's own row",
        );
        // A store walk writes the store alone: nothing of it is read back,
        // and it leaves no own row for the next walk.
        let store = WalkMode::Store;
        ok &= refused(
            "a store walk read back",
            m.mtp_draft(feed(1, 0), h, store),
            "a walk with a head to read back",
        );
        ok &= refused(
            "a chain of store walks",
            m.mtp_chain(feed(1, 0), 1, h, store),
            "a chain's walks with a head",
        );
        m.mtp_walk(feed(1, 0), h, store)?;
        ok &= refused(
            "the draft's own row after a store walk",
            m.mtp_draft(DraftFeed::Own { pos0: 1 }, h, e),
            "a walk before the draft's own row",
        );
        ok &= refused(
            "the streams of a store walk",
            m.mtp_l_out(),
            "a walk to read",
        );
        Ok(ok && raised)
    }

    /// (f): a walk that reads the target's arenas by position: after the
    /// set's prompt by passes (the Pass arena holding its last unit's rows),
    /// with the draft's store warmed over the prompt and its next position,
    /// a walk that reads the position before the arena's first, one that
    /// reads past its last row, and one that starts its rows at the wrong
    /// arena row are each refused by name with the positions the arena
    /// holds, and the walk the arena does hold — its rows from the position
    /// after the arena's first, the prompt's own next id its last token —
    /// runs.
    fn arena_holds(m: &mut Qwen38Model, prompt: &[u32]) -> Result<bool, GateError> {
        m.reset()?;
        let next = m.prompt38(prompt, Prompt38::Pass)?;
        let n = u32::try_from(prompt.len())?;
        let rows = (prompt.len() - 1) % MTP_ROWS + 1;
        let first = n - rows as u32;
        // The draft's store must hold every walk's start: warm it over the
        // prompt's rows and the position after them (store walks, the prompt
        // call's warmup shape).
        let zeros = vec![0.0f32; MTP_STORE_ROWS * WIDE];
        m.mtp_walk(
            DraftFeed::Rows(Feed {
                tokens: &prompt[..MTP_STORE_ROWS.min(prompt.len())],
                pos0: 0,
                hidden: Hidden::Host(&zeros[..MTP_STORE_ROWS.min(prompt.len()) * WIDE]),
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Store,
        )?;
        m.mtp_walk(
            DraftFeed::Rows(Feed {
                tokens: &prompt[..1],
                pos0: n,
                hidden: Hidden::Host(&zeros[..WIDE]),
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Store,
        )?;
        fn feed(tokens: &[u32], pos0: u32, at: usize) -> DraftFeed<'_, TargetRows> {
            DraftFeed::Rows(Feed {
                tokens,
                pos0,
                hidden: Hidden::Target {
                    walk: TargetRows::Pass,
                    first: at,
                },
                media: None,
            })
        }
        let (h, e) = (DraftHead::Rows, WalkMode::Eager);
        let mut ok = refused(
            "the Pass arena's rows read as the position before their first",
            m.mtp_draft(feed(&prompt[..1], first, 0), h, e),
            &format!(
                "as positions {}..{first} for a walk from {first}: the arena holds positions \
                 {first}..{n}",
                first - 1
            ),
        );
        ok &= refused(
            "the Pass arena's rows read past their last",
            m.mtp_draft(feed(&prompt[..1], n + 1, 0), h, e),
            &format!(
                "as positions {n}..{} for a walk from {}: the arena holds positions {first}..{n}",
                n + 1,
                n + 1
            ),
        );
        ok &= refused(
            "the Pass arena's rows read as other positions",
            m.mtp_draft(feed(&prompt[..1], first + 1, 1), h, e),
            &format!(
                "rows 1..2 of the Pass arena as positions {first}..{} for a walk from {}",
                first + 1,
                first + 1
            ),
        );
        let mut tokens = prompt[first as usize + 1..].to_vec();
        tokens.push(next);
        let runs = m.mtp_draft(
            DraftFeed::Rows(Feed {
                tokens: &tokens,
                pos0: first + 1,
                hidden: Hidden::Target {
                    walk: TargetRows::Pass,
                    first: 0,
                },
                media: None,
            }),
            h,
            e,
        );
        let runs_ok = runs.is_ok();
        println!(
            "(f) the walk the Pass arena holds, its {rows} rows from {}: {} {}",
            first + 1,
            match &runs {
                Ok(w) => format!("ran, {} tokens", w.tokens.len()),
                Err(err) => err.to_string(),
            },
            verdict(runs_ok)
        );
        Ok(ok && runs_ok)
    }

    /// (iv) Only completed target rows may feed MTP: a headless pass or ubatch leaves none,
    /// a headed ubatch leaves its last row, and a sink completes every requested row.
    fn completed_rows(m: &mut Qwen38Model, prompt: &[u32]) -> Result<bool, GateError> {
        let ids = &prompt[..4];
        let zeros = vec![0.0; 5 * WIDE];
        let (h, e) = (DraftHead::Rows, WalkMode::Eager);
        let feed = |walk, first, pos0| {
            DraftFeed::Rows(Feed {
                tokens: &ids[..1],
                pos0,
                hidden: Hidden::Target { walk, first },
                media: None,
            })
        };
        let warm = |m: &mut Qwen38Model| {
            m.mtp_walk(
                DraftFeed::Rows(Feed {
                    tokens: &[ids[0]; 5],
                    pos0: 0,
                    hidden: Hidden::Host(&zeros),
                    media: None,
                }),
                h,
                WalkMode::Store,
            )
        };
        let mut ok = true;
        for (path, walk) in [
            (Prompt38::Gemm, TargetRows::Ubatch),
            (Prompt38::Pass, TargetRows::Pass),
        ] {
            m.reset()?;
            m.body_parts("completed rows")?.2.set_checkpoints(false);
            // Seed finite final streams before overwriting their arena without a reader.
            let mut sink = |_: &mut Qwen38Model, _, _, _| Ok(());
            m.prompt38_call(ids, path, CallEnd::Unread, Some(&mut sink))?;
            m.reset()?;
            m.prompt38_call(ids, path, CallEnd::Unread, None)?;
            warm(m)?;
            ok &= refused(
                &format!("{walk:?} unfinished first hidden row"),
                m.mtp_draft(feed(walk, 0, 1), h, e),
                "final hidden rows",
            );
            ok &= refused(
                &format!("{walk:?} unfinished last hidden row"),
                m.mtp_draft(feed(walk, 3, 4), h, e),
                "final hidden rows",
            );
            m.reset()?;
            m.prompt38_call(ids, path, CallEnd::Unread, Some(&mut sink))?;
            warm(m)?;
            let ran = m.mtp_draft(feed(walk, 0, 1), h, e).is_ok();
            println!(
                "(iv) {walk:?} sink completed first hidden row {}",
                verdict(ran)
            );
            ok &= ran;
            // The finite rows above stay in the arena; a tapped call that fails
            // past its launch must leave no completed record of its own rows.
            m.reset()?;
            m.body_parts("completed rows")?.2.plant_after_launch();
            ok &= refused(
                &format!("{walk:?} tapped call failed past its launch"),
                m.prompt38_call(ids, path, CallEnd::Unread, Some(&mut sink)),
                "planted failure after the launch",
            );
            warm(m)?;
            ok &= refused(
                &format!("{walk:?} first hidden row of the failed call"),
                m.mtp_draft(feed(walk, 0, 1), h, e),
                "arena holds no position",
            );
        }
        m.reset()?;
        m.prompt38(ids, Prompt38::Gemm)?;
        warm(m)?;
        ok &= refused(
            "Ubatch headed but unfinished first hidden row",
            m.mtp_draft(feed(TargetRows::Ubatch, 0, 1), h, e),
            "final hidden rows",
        );
        let ran = m.mtp_draft(feed(TargetRows::Ubatch, 3, 4), h, e).is_ok();
        println!("(iv) Ubatch completed last hidden row {}", verdict(ran));
        m.reset()?;
        Ok(ok && ran)
    }

    /// The shared own row is valid only at the next position of the live slot's walk.
    fn own_row(m: &mut Qwen38Model, prompt: &[u32]) -> Result<bool, GateError> {
        const REFRESH: &str = "Rows refresh required";
        m.add_slots(2)?;
        let zeros = vec![0.0; WIDE];
        let (h, e) = (DraftHead::Rows, WalkMode::Eager);
        let refresh = |m: &mut Qwen38Model, token| {
            m.mtp_walk(
                DraftFeed::Rows(Feed {
                    tokens: &[token],
                    pos0: 0,
                    hidden: Hidden::Host(&zeros),
                    media: None,
                }),
                h,
                e,
            )
        };
        m.reset()?;
        refresh(m, prompt[0])?;
        let mut ok = refused(
            "own row at the wrong position",
            m.mtp_draft(DraftFeed::Own { pos0: 0 }, h, e),
            "requested position 0",
        );
        // Each invalidation below follows a refresh whose next position is the
        // one requested, so only the invalidation itself refuses the walk.
        m.reset()?;
        m.prompt38(&prompt[..2], Prompt38::Pass)?;
        let state = m.seq_save()?;
        m.reset()?;
        refresh(m, prompt[0])?;
        // The model is empty, so the resume adopts the state without a reset.
        m.seq_resume(&state)?;
        ok &= refused(
            "own row after a state adoption",
            m.mtp_draft(DraftFeed::Own { pos0: 1 }, h, e),
            REFRESH,
        );
        refresh(m, prompt[0])?;
        m.reset()?;
        ok &= refused(
            "own row after a reset",
            m.mtp_draft(DraftFeed::Own { pos0: 1 }, h, e),
            REFRESH,
        );
        refresh(m, prompt[0])?;
        m.mtp_store_slots(
            &[(
                0,
                Feed {
                    tokens: &[prompt[0]],
                    pos0: 0,
                    hidden: Hidden::Host(&zeros),
                    media: None,
                },
            )],
            h,
        )?;
        ok &= refused(
            "own row after a store walk",
            m.mtp_draft(DraftFeed::Own { pos0: 1 }, h, e),
            REFRESH,
        );
        m.select_slot(1)?;
        m.reset()?;
        refresh(m, prompt[1])?;
        m.select_slot(0)?;
        ok &= refused(
            "own row after A to B to A",
            m.mtp_draft(DraftFeed::Own { pos0: 1 }, h, e),
            REFRESH,
        );
        refresh(m, prompt[0])?;
        let ran = m
            .mtp_draft(DraftFeed::Own { pos0: 1 }, h, WalkMode::Graph)
            .is_ok();
        println!("(iv) own row after Rows refresh {}", verdict(ran));
        refresh(m, prompt[0])?;
        // A verify's rejected tail cannot leave its branch's own input live.
        m.verify_slots(&[(0, &prompt[..2])])?;
        m.commit_slots(&[1])?;
        ok &= refused(
            "own row after a cut",
            m.mtp_draft(DraftFeed::Own { pos0: 1 }, h, e),
            REFRESH,
        );
        m.reset()?;
        Ok(ok && ran)
    }

    /// A continued image starts at call offset zero and crosses the draft's store block.
    /// Its rows equal token 7's embedding, while its ids are 8: replaying the target's
    /// hidden rows with token 7 gives an independent store reference without media splicing.
    #[cfg(all(test, feature = "checkpoint"))]
    fn media_prompt(mut m: Qwen38Model) -> Result<(Qwen38Model, bool), GateError> {
        use bloomery_gpu::media::{ImageShape, ImageSpan, MediaSpan};
        use bloomery_gpu::{Window, to_host};
        const PREFIX: [u32; 4] = [1, 2, 3, 4];
        const IMAGE: usize = MTP_STORE_ROWS + 8;
        let ctx = m.body("media draft")?.ctx() as u32;
        let embedding = {
            let (gpu, w, _) = m.body_parts("media draft")?;
            let table = w.bf16("media draft", "token_embd.weight")?;
            let row = Window::<u16>::of(table.buf(), 7 * HIDDEN * 2, HIDDEN)?;
            to_host(gpu.stream(), &row)?
        };
        let rows: Vec<u16> = embedding
            .iter()
            .copied()
            .cycle()
            .take(IMAGE * HIDDEN)
            .collect();
        let spans = [ImageSpan {
            span: MediaSpan {
                at: 0..IMAGE,
                rows: &rows,
            },
            shape: ImageShape::Grid(8, IMAGE / 8),
        }];
        let mut ids = vec![8; IMAGE];
        ids.extend_from_slice(&[5, 6]);
        let mut hidden = vec![0.0; WIDE];
        let mut tap = |m: &mut Qwen38Model, arena, first, count| {
            assert_eq!(hidden.len() / WIDE - 1, first as usize);
            hidden.extend(m.target_streams(arena, count)?);
            Ok(())
        };
        m.prompt38_with(&PREFIX, Prompt38::Gemm, Some(&mut tap))?;
        let first = m.media38_frame(&ids, &spans, |m| {
            m.prompt38_with(&ids, Prompt38::Gemm, Some(&mut tap))
        })?;
        let rope = m.rope_saved()?;
        let mut plain = vec![first];
        slots_gate::greedy(&mut m, &mut plain, 7)?;
        let mut reference_ids = PREFIX.to_vec();
        reference_ids.extend(std::iter::repeat_n(7, IMAGE));
        reference_ids.extend_from_slice(&[5, 6]);
        let held = reference_ids.len() - 1;
        // First read the text rotations as a contrast, then the prepared image rotations.
        let mut stores = Vec::new();
        for image in [false, true] {
            m.reset()?;
            if image {
                m.rope_restore(rope.clone())?;
            }
            for (block, tokens) in reference_ids[..held].chunks(MTP_STORE_ROWS).enumerate() {
                let from = block * MTP_STORE_ROWS;
                m.mtp_walk(
                    DraftFeed::Rows(Feed {
                        tokens,
                        pos0: from as u32,
                        hidden: Hidden::Host(&hidden[from * WIDE..(from + tokens.len()) * WIDE]),
                        media: None,
                    }),
                    DraftHead::Rows,
                    WalkMode::Store,
                )?;
            }
            stores.push(store_of(&m, held)?);
        }
        let rotated = stores[0].0 != stores[1].0;
        let (mut session, mut spec) = drafted_session(m, step_cfg(Prompt38::Gemm), ctx)?;
        spec.prompt(&mut session, &PREFIX)?;
        let first = spec.draft_mut().prompt_media(&mut session, &ids, &spans)?;
        let got = store_of(session.model(), held)?;
        let same = same_rows(&got, &stores[1], held);
        let store_ok = same.iter().all(|&same| same);
        println!(
            "(media-draft) image at call offset 0 after {} text ids, {} image rows across store \
             blocks: {} of {held} store rows = embedding-token replay; image keys differ from \
             text rotations {rotated} {}",
            PREFIX.len(),
            IMAGE,
            same.iter().filter(|&&same| same).count(),
            verdict(store_ok && rotated)
        );
        let mut kept = Kept::default();
        let drafted = runtime::generate(
            &mut session,
            &mut spec,
            &ids,
            first,
            &runtime::Stop::new(8, ctx)?,
            &mut kept,
        )?;
        let ids_ok = drafted.tokens.get(..plain.len()) == Some(plain.as_slice());
        let proposed = kept.proposed.iter().any(|&p| p);
        println!(
            "(media-draft) drafted ids = plain ids {ids_ok}, proposed {proposed} {}",
            verdict(ids_ok && proposed)
        );
        Ok((
            session.into_model(),
            store_ok && rotated && ids_ok && proposed,
        ))
    }

    /// The positions and the ubatch of [`hw_mtp_input_validity`]'s load, room for
    /// [`media_prompt`]'s image, and of [`hw_close_fold`]'s, whose small ubatch
    /// puts a closing call's last id past a ubatch edge.
    #[cfg(all(test, feature = "checkpoint"))]
    const CTX_INPUT: u64 = 192;
    #[cfg(all(test, feature = "checkpoint"))]
    const UB_INPUT: usize = 128;
    #[cfg(all(test, feature = "checkpoint"))]
    const CTX_CLOSE: u64 = 64;
    #[cfg(all(test, feature = "checkpoint"))]
    const UB_CLOSE: usize = 8;
    /// The positions of [`hw_quantum_cuts`]'s load: a prompt past two inner
    /// marks ([`CHECKPOINT_EVERY`]) and its steps.
    #[cfg(all(test, feature = "checkpoint"))]
    const CTX_QUANTUM: u64 = 2 * CHECKPOINT_EVERY as u64 + 64;

    /// The checkpoint folder at `$BLOOMERY_QWEN38_FOLDER` loaded with its MTP
    /// layer, two slots of `ctx` positions and ubatches of `ub` rows, its
    /// rope sections read z first.
    #[cfg(all(test, feature = "checkpoint"))]
    fn folder_load(ctx: u64, ub: usize) -> Result<Qwen38Model, GateError> {
        use bloomery_gpu::arch::qwen3moe::Folder38;
        let levers = bloomery_levers::at_main(&[CARD_BUDGET])?;
        let path = PathBuf::from(
            std::env::var_os("BLOOMERY_QWEN38_FOLDER")
                .ok_or("$BLOOMERY_QWEN38_FOLDER names the checkpoint folder")?,
        );
        let mut f = bloomery_gpu_gates::model_file::open_folder(&path)?;
        // Section zero reads z in zyxt: the first image key must use z=0.
        let mut kvs: Vec<_> = f
            .header
            .iter_kv()
            .map(|(k, v)| (k.to_owned(), v.clone()))
            .collect();
        let key = format!(
            "{}.rope.section_order",
            f.header.architecture().ok_or("no architecture")?
        );
        kvs.retain(|(k, _)| k != &key);
        kvs.push((key, gguf::Value::String("zyxt".to_owned())));
        let tensors = f
            .header
            .iter_tensors()
            .map(|(_, t)| gguf::write::TensorDecl {
                name: t.name.clone(),
                dims: t.dims.clone(),
                type_id: t.ty.as_u32(),
                nbytes: t.nbytes,
            })
            .collect();
        let layout = gguf::write::Layout::new(&kvs, tensors)?;
        f.header = Split::header_only(
            f.header.shard_path(0).ok_or("no first shard")?.to_owned(),
            &layout,
        )?;
        let inputs = PlanInputs::describe(&f.header)?;
        let draft = f.draft?.ok_or("the checkpoint has no MTP layer")?;
        let mtp = MtpInputs::read(
            &draft,
            &f.header,
            &inputs,
            HeadSet::List {
                ids: (0..64).collect::<Vec<_>>().into(),
                digest: vocab_sha256(&f.header)?,
            },
        )?;
        let census = bloomery_gpu::census()?;
        let device = census.first().ok_or("no visible card")?;
        let machine = machine_for_experts(
            model::placement::devices::spec_of_device(device),
            inputs.stack(),
            ub as u64,
            Experts::Card,
        );
        let plan_levers = PlanLevers {
            card_budget_bytes: levers.card_budget(),
        };
        let plan = inputs.plan_mtp_with_slots(
            &machine,
            (ctx, ub as u64),
            &plan_levers,
            &mtp,
            Experts::Card,
            2,
        )?;
        let ple = f
            .rows
            .remove(&model::arch::qwen35moe::names::per_layer_token_embd());
        let model = Folder38::with(f.header, f.source, ple);
        Ok(Body38::open_placed_mtp_slots(
            model,
            &plan,
            &inputs,
            0,
            levers.host(),
            ub,
            &draft,
            &mtp,
            2,
        )?)
    }

    /// The input boundary on an existing checkpoint, without an oracle set or a server.
    #[cfg(all(test, feature = "checkpoint"))]
    #[test]
    #[ignore = "needs one NVIDIA card and a Qwen3.8 checkpoint folder with MTP at $BLOOMERY_QWEN38_FOLDER"]
    fn hw_mtp_input_validity() -> Result<(), GateError> {
        let mut m = folder_load(CTX_INPUT, UB_INPUT)?;
        m.enable_media()?;
        let (mut m, media_ok) = media_prompt(m)?;
        m.reset()?;
        m.set_mode(bloomery_gpu::model::StepMode::Graph);
        let prompt = [1, 2, 3, 4];
        let rows_ok = completed_rows(&mut m, &prompt)?;
        // Use the gate's existing greedy runner and drafted session on the same load.
        let (m, plain, _) = plain_to(m, &prompt, Prompt38::Gemm, 8)?;
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Gemm), CTX_INPUT as u32)?;
        let first = spec.prompt(&mut s, &prompt)?;
        let drafted = runtime::generate(
            &mut s,
            &mut spec,
            &prompt,
            first,
            &runtime::Stop::new(8, CTX_INPUT as u32)?,
            &mut Kept::default(),
        )?;
        // The runtime keeps the last pass whole; compare the requested prefix.
        let same = drafted.tokens.get(..plain.len()) == Some(plain.as_slice());
        if !same {
            eprintln!("(iv) greedy plain={plain:?} drafted={:?}", drafted.tokens);
        }
        println!(
            "(iv) first {} drafted greedy ids = plain ids {}; draft kept {} ids",
            plain.len(),
            verdict(same),
            drafted.tokens.len()
        );
        let own_ok = own_row(s.model_mut(), &prompt)?;
        if media_ok && rows_ok && own_ok && same {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// A shared draft prefix stays bit-identical while its source rewrites rows
    /// through each walk entry. A reset releases shared mappings before a put-back.
    /// Mutants: omit either walk's own_from, or retain shared pages in Room::settle.
    #[cfg(test)]
    #[test]
    #[ignore = "needs one NVIDIA card and the Qwen3.8 target and draft files"]
    fn hw_mtp_shared_store() -> Result<(), GateError> {
        assert!(init()?, "the model header witness");
        let (file, draft) = (open_target()?, open_draft()?);
        let inputs = PlanInputs::describe(&file)?;
        let mtp = MtpInputs::read(
            &draft,
            &file,
            &inputs,
            HeadSet::List {
                ids: (0..64).collect::<Vec<_>>().into(),
                digest: vocab_sha256(&file)?,
            },
        )?;
        let (cut, at, ctx) = (
            CHECKPOINT_EVERY as usize,
            2 * CHECKPOINT_EVERY as usize,
            3 * CHECKPOINT_EVERY as u64,
        );
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(ctx as usize)?;
        let machine = machine(card()?, inputs.stack(), ub as u64);
        let plan = inputs.plan_mtp_with_slots(
            &machine,
            (ctx, ub as u64),
            &plan_levers(mtp.card_bytes_of(ctx, 2)?)?,
            &mtp,
            Experts::Host,
            2,
        )?;
        let mut m = Body38::open_placed_mtp_slots(
            file,
            &plan,
            &inputs,
            0,
            cfg().host,
            ub,
            &draft,
            &mtp,
            2,
        )?;
        m.add_slots(2)?;
        m.body_parts("draft share")?.2.set_checkpoints(true);
        // The MTP store is [2][ctx][256] f16 per plane. Whole pages
        // below both points must actually be shared on the card under test.
        let page = m.gpu().allocation_granularity()?;
        let row_bytes = 256 * size_of::<u16>();
        assert!(
            [cut, at, ctx as usize]
                .iter()
                .all(|n| (n * row_bytes).is_multiple_of(page)),
            "draft share points and head runs must end on whole pages ({page} B)"
        );
        let ids: Vec<u32> = (0..=at).map(|i| 1 + (i % 31) as u32).collect();
        m.prompt38(&ids[..at], Prompt38::Auto)?;
        // A saved state's points are below its end. Keep the share point
        // inside the state so a resume carries it into the source slot.
        m.prompt38(&ids[at..], Prompt38::Auto)?;
        overwrite_store(&mut m, &ids, ids.len(), 0)?;
        let original_rows = store_of(&m, at)?;
        let original = m.seq_save()?;
        overwrite_store(&mut m, &ids, ids.len(), 1)?;
        let replacement_rows = store_of(&m, ids.len())?;
        let replacement = m.seq_save()?;
        assert!(
            original_rows.0 != replacement_rows.0[..at * 2 * 256],
            "different keys for the put-back"
        );
        assert!(
            original_rows.1 != replacement_rows.1[..at * 2 * 256],
            "different values for the put-back"
        );
        let zeros = vec![0.0f32; WIDE];
        for entry in ["eager", "graph", "store", "slots", "resume", "prefix"] {
            m.select_slot(1)?;
            m.reset()?;
            m.select_slot(0)?;
            m.seq_resume(&original)?;
            m.select_slot(1)?;
            let (put, moved) = m.seq_share_prefix(0, at as u32)?;
            assert_eq!(put as usize, at, "{entry}: shared point");
            assert!(
                moved.mapped >= 4 * at * row_bytes,
                "{entry}: draft pages mapped"
            );
            assert!(
                store_of(&m, at)? == original_rows,
                "{entry}: shared draft rows"
            );
            let shared = m.body("draft share")?.store_shared();
            m.select_slot(0)?;
            if entry == "resume" || entry == "prefix" {
                m.reset()?;
                assert_eq!(
                    m.body("draft share reset")?.store_shared(),
                    0,
                    "{entry}: reset must release shared pages before the put-back"
                );
                let n = if entry == "prefix" { cut } else { ids.len() };
                let put = if entry == "prefix" {
                    m.seq_resume_prefix(&replacement, n as u32)?
                } else {
                    m.seq_resume(&replacement)?;
                    m.pos()
                };
                assert_eq!(put as usize, n, "{entry}: restored point");
                let restored = store_of(&m, n)?;
                assert!(
                    restored.0 == replacement_rows.0[..n * 2 * 256]
                        && restored.1 == replacement_rows.1[..n * 2 * 256],
                    "{entry}: restored draft rows"
                );
                assert_eq!(
                    m.body("draft share put-back")?.store_shared(),
                    0,
                    "{entry}: restored pages are owned"
                );
            } else {
                let tokens = [ids[cut] + 1];
                let feed = Feed {
                    tokens: &tokens,
                    pos0: cut as u32,
                    hidden: Hidden::Host(&zeros),
                    media: None,
                };
                if entry == "slots" {
                    m.mtp_store_slots(&[(0, feed)], DraftHead::Rows)?;
                } else {
                    let mode = match entry {
                        "eager" => WalkMode::Eager,
                        "graph" => WalkMode::Graph,
                        _ => WalkMode::Store,
                    };
                    m.mtp_walk(DraftFeed::Rows(feed), DraftHead::Rows, mode)?;
                }
                let rewritten = store_of(&m, at)?;
                assert!(
                    rewritten.0 != original_rows.0,
                    "{entry}: keys actually changed"
                );
                assert!(
                    rewritten.1 != original_rows.1,
                    "{entry}: values actually changed"
                );
                assert_eq!(
                    shared - m.body("draft share COW")?.store_shared(),
                    4 * (at - cut) * row_bytes,
                    "{entry}: only the draft tail detached"
                );
            }
            m.select_slot(1)?;
            assert!(
                store_of(&m, at)? == original_rows,
                "{entry}: the other slot's shared K/V must stay bit-identical"
            );
            println!(
                "draft share {entry}: {at} positions mapped, source rewritten, other slot K/V bit-identical PASS"
            );
        }
        Ok(())
    }

    /// (cl) A prompt's last id run as its last call's last row
    /// ([`CallEnd::Close`]), held to the prompt fed in one `Auto` call
    /// ([`close_case`]): at 21 ids from a reset the last id leaves a delta
    /// chunk open, at 32 it completes one, which the call keeps out of the
    /// state; both fold into the last ubatch. At 5 the call is passes. Then a
    /// drafted
    /// run whose prompt call closes gives the plain run's greedy ids, its
    /// draft able to rejoin at the end mark.
    #[cfg(all(test, feature = "checkpoint"))]
    #[test]
    #[ignore = "needs one NVIDIA card and a Qwen3.8 checkpoint folder with MTP at $BLOOMERY_QWEN38_FOLDER"]
    fn hw_close_fold() -> Result<(), GateError> {
        let mut m = folder_load(CTX_CLOSE, UB_CLOSE)?;
        m.set_mode(bloomery_gpu::model::StepMode::Graph);
        let ids: Vec<u32> = (1..=32).collect();
        let mut ok = true;
        let mut rides = Vec::new();
        for (n, folds) in [(21, true), (32, true), (5, false)] {
            ok &= close_case(&mut m, &ids[..n], folds)?;
            rides.push(m.closes_by_ubatch(n, Prompt38::Auto)?);
        }
        let rides_ok = rides == [true, true, false];
        println!(
            "(cl) the last id rides the ubatch walk at 21, 32 and 5 ids from a reset (the seat \
             steps it otherwise): {rides:?} {}",
            verdict(rides_ok)
        );
        ok &= rides_ok;
        ok &= close_steps(&mut m, &ids[..5])?;
        let prompt = &ids[..21];
        let end = 20;
        let (m, plain, _) = plain_to(m, prompt, Prompt38::Auto, 8)?;
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), CTX_CLOSE as u32)?;
        s.model_mut().body_parts("(cl)")?.2.set_checkpoints(true);
        let first = spec
            .draft_mut()
            .prompt_call(&mut s, prompt, CallEnd::Close)?
            .ok_or("the closing call read no token")?;
        let rejoins = s.model().body("(cl)")?.draft_rejoins(end);
        let drafted = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(8, CTX_CLOSE as u32)?,
            &mut Kept::default(),
        )?;
        s.model_mut().body_parts("(cl)")?.2.set_checkpoints(false);
        let same = drafted.tokens.get(..plain.len()) == Some(plain.as_slice());
        println!(
            "(cl) a drafted run whose prompt call closes: the draft rejoins at {end} {rejoins}; \
             greedy ids {:?} = the plain run's {plain:?} {}",
            drafted.tokens,
            verdict(same && rejoins)
        );
        s.model_mut().add_slots(2)?;
        s.reset()?;
        spec.draft_mut().restart();
        let first = spec
            .draft_mut()
            .prompt_call(&mut s, prompt, CallEnd::Close)?
            .ok_or("the closing call read no token")?;
        let width = s.model().body("(cl)")?.mtp_tap_width();
        let source = s.model().body("(cl)")?.target_streams(
            s.model().gpu(),
            TargetRows::Ubatch,
            prompt.len() % UB_CLOSE,
        )?;
        spec.draft_mut().retain_prompt(&mut s)?;
        let refused_owned_arenas = {
            let pos = s.pos();
            let (gpu, _, body) = s.model_mut().body_parts("(cl)")?;
            [TargetRows::Step, TargetRows::Pass]
                .into_iter()
                .all(|walk| match body.retain_prompt_row(gpu, walk, 0, pos) {
                    Err(e) => e.to_string().contains("a shared prompt arena"),
                    Ok(()) => false,
                })
        };
        let anchor = spec.draft().park();
        s.model_mut().select_slot(1)?;
        s.reset()?;
        let other: Vec<u32> = prompt.iter().copied().rev().collect();
        s.model_mut().prompt38(&other, Prompt38::Gemm)?;
        s.model_mut().select_slot(0)?;
        spec.draft_mut().unpark(&anchor);
        let retained =
            s.model()
                .body("(cl)")?
                .target_streams(s.model().gpu(), TargetRows::Step, 1)?;
        let row_equal = retained
            .iter()
            .map(|v| v.to_bits())
            .eq(source[source.len() - width..].iter().map(|v| v.to_bits()));
        let active = !spec.draft().skipping();
        let after_switch = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(8, CTX_CLOSE as u32)?,
            &mut Kept::default(),
        )?;
        let preserved =
            row_equal && active && refused_owned_arenas && after_switch.tokens == drafted.tokens;
        println!(
            "(cl) close then another slot's prompt: retained_row_bits={row_equal} draft_active={active} \
             refused_owned_arenas={refused_owned_arenas} greedy_equal={} {}",
            after_switch.tokens == drafted.tokens,
            verdict(preserved),
        );
        if ok && same && rejoins && preserved {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    /// One (cl) case on `ids` from a reset, checkpoints on: the closing call
    /// whole gives the one `Auto` call's argmax, logits' bits and three steps
    /// after it ([`closed_run`]), but where its last id completes a delta
    /// chunk: the call keeps that chunk out of the state, and the next step
    /// folds it by the step's rule (`gate_linear`'s clause 16), not the
    /// walk's, so there the steps' ids against the one call's and the former
    /// cut's are printed, not held. The call cut after its first half gives
    /// the whole call's. Its end mark
    /// stands before the last id and not past it, and that id stepped from
    /// the mark (a rollback there) gives the server's former cut: the call
    /// less its last id, then its step. The call's last ubatch holds more
    /// than one row (it folded) exactly when `folds`.
    #[cfg(all(test, feature = "checkpoint"))]
    fn close_case(m: &mut Qwen38Model, ids: &[u32], folds: bool) -> Result<bool, GateError> {
        const WHAT: &str = "(cl)";
        let (n, k) = (ids.len(), ids.len() / 2);
        let end = u32::try_from(n - 1)?;
        m.body_parts(WHAT)?.2.set_checkpoints(true);
        m.reset()?;
        let want = closed_run(m, ids, CallEnd::Argmax)?;
        m.reset()?;
        m.prompt38(&ids[..n - 1], Prompt38::Auto)?;
        let cut = closed_run(m, &ids[n - 1..], CallEnd::Argmax)?;
        m.reset()?;
        let whole = closed_run(m, ids, CallEnd::Close)?;
        let body = m.body(WHAT)?;
        let rows = body.ubatch_split().map_or(0, |(a, b)| a + b);
        let points = body.checkpoints().positions();
        m.rollback(end)?;
        let stepped = closed_run(m, &ids[n - 1..], CallEnd::Argmax)?;
        m.reset()?;
        m.prompt38(&ids[..k], Prompt38::Auto)?;
        let split = closed_run(m, &ids[k..], CallEnd::Close)?;
        m.body_parts(WHAT)?.2.set_checkpoints(false);
        m.reset()?;
        let marked = points.contains(&end) && !points.contains(&(end + 1));
        let ids = |c: &Closed| c.2.iter().map(|s| s.0).collect::<Vec<_>>();
        let kept = n.is_multiple_of(bloomery_gpu::linear::CHUNK_ROWS);
        let first = (whole.0, &whole.1) == (want.0, &want.1);
        let like = first && (kept || whole == want);
        let ok = like && split == whole && stepped == cut && marked && (rows > 1) == folds;
        println!(
            "(cl) {n} ids: the closing call whole = the one call's argmax and logits {first}, \
             its steps' bits {} (held {}), step ids {:?} (the one call's {:?}, the former cut's \
             {:?}); cut at {k} = whole {}; its last id stepped from the end mark = the former \
             cut {}; points {points:?}; the last ubatch {rows} rows (folds {folds}) {}",
            whole.2 == want.2,
            !kept,
            ids(&whole),
            ids(&want),
            ids(&cut),
            split == whole,
            stepped == cut,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The argmax after `ids` fed from where a model stands, the logits'
    /// bits, and three greedy steps' ids and logits' bits.
    #[cfg(all(test, feature = "checkpoint"))]
    type Closed = (u32, Vec<u32>, Vec<(u32, Vec<u32>)>);

    /// [`Closed`] of one `Auto` call ending as `end` asks.
    #[cfg(all(test, feature = "checkpoint"))]
    fn closed_run(m: &mut Qwen38Model, ids: &[u32], end: CallEnd) -> Result<Closed, GateError> {
        let bits = |m: &Qwen38Model| -> Result<Vec<u32>, GateError> {
            Ok(m.logits()?.iter().map(|v| v.to_bits()).collect())
        };
        let first = m
            .prompt38_call(ids, Prompt38::Auto, end, None)?
            .ok_or("the call read no token")?;
        let row = bits(m)?;
        let (mut steps, mut feed) = (Vec::new(), first);
        for _ in 0..3 {
            feed = m.step(&[feed])?;
            steps.push((feed, bits(m)?));
        }
        Ok((first, row, steps))
    }

    /// (cl) A closing call fed by steps, checkpoints on: it runs every id, its
    /// end mark before the last, and stands where the steps of the same ids
    /// stand with their argmax and logits' bits.
    #[cfg(all(test, feature = "checkpoint"))]
    fn close_steps(m: &mut Qwen38Model, ids: &[u32]) -> Result<bool, GateError> {
        let run = |m: &mut Qwen38Model, end: CallEnd| -> Result<_, GateError> {
            m.reset()?;
            let next = m
                .prompt38_call(ids, Prompt38::Step, end, None)?
                .ok_or("the call read no token")?;
            let bits: Vec<u32> = m.logits()?.iter().map(|v| v.to_bits()).collect();
            Ok((next, m.pos(), bits))
        };
        m.body_parts("(cl)")?.2.set_checkpoints(true);
        let plain = run(m, CallEnd::Argmax)?;
        let closed = run(m, CallEnd::Close)?;
        let points = m.body("(cl)")?.checkpoints().positions();
        m.body_parts("(cl)")?.2.set_checkpoints(false);
        m.reset()?;
        let end = u32::try_from(ids.len())?;
        let ok = closed == plain && closed.1 == end && points.contains(&(end - 1));
        println!(
            "(cl) {} ids by steps: the closing call stands at {} (want {end}), argmax {} = the \
             steps' {} and their logits {}; points {points:?} {}",
            ids.len(),
            closed.1,
            closed.0,
            plain.0,
            closed.2 == plain.2,
            verdict(ok)
        );
        Ok(ok)
    }

    /// (qc) The serve seat's prompt quantum (`Q38::prompt_quantum`, the inner
    /// checkpoint grid [`CHECKPOINT_EVERY`]): a prompt call cut at the grid's
    /// positions against the one call ([`quantum_case`]), from a reset to 21
    /// ids past the second mark, and after a prefix three ids short of the
    /// first mark to five ids past the second, whose first and last calls
    /// hold fewer ids than a ubatch walk's fewest ([`Prompt38::GEMM_FROM`]).
    /// A call that short runs as passes off the grid, from a reset too, and
    /// as ubatches where it starts or ends on it (`Body38::resolve_prompt`).
    #[cfg(all(test, feature = "checkpoint"))]
    #[test]
    #[ignore = "needs one NVIDIA card and a Qwen3.8 checkpoint folder with MTP at $BLOOMERY_QWEN38_FOLDER"]
    fn hw_quantum_cuts() -> Result<(), GateError> {
        let e = CHECKPOINT_EVERY as usize;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(CTX_QUANTUM as usize)?;
        let mut m = folder_load(CTX_QUANTUM, ub)?;
        m.set_mode(bloomery_gpu::model::StepMode::Graph);
        m.body_parts("(qc)")?.2.set_checkpoints(true);
        let ids: Vec<u32> = (0..2 * e + 21)
            .map(|i| 1 + (i * 7919 % 30_011) as u32)
            .collect();
        let b = m.body("(qc)")?;
        let paths: Vec<(usize, bool)> = [
            (0, Prompt38::Pass),
            (1, Prompt38::Pass),
            (e - 3, Prompt38::Gemm),
            (e, Prompt38::Gemm),
        ]
        .into_iter()
        .map(|(from, want)| {
            (
                from,
                b.resolve_prompt(Prompt38::Auto, from as u32, 3) == want,
            )
        })
        .collect();
        let mut ok = paths.iter().all(|&(_, p)| p);
        println!(
            "(qc) 3 ids from 0 and 1 run as passes, from {} and {e} as ubatches: {paths:?} {}",
            e - 3,
            verdict(ok)
        );
        for (k, n) in [(0, 2 * e + 21), (e - 3, 2 * e + 5)] {
            ok &= quantum_case(&mut m, &ids[..n], k)?;
        }
        m.body_parts("(qc)")?.2.set_checkpoints(false);
        if ok { Ok(()) } else { Err(checks_failed()) }
    }

    /// One (qc) case, checkpoints on: after a reset and `ids[..k]` fed as a
    /// call, the rest as the serve seat feeds a prompt's calls (each but the
    /// last unread (`GpuModel::prompt38_call`), the last the closing call
    /// [`closed_run`], which the seat runs where its last id rides the ubatch
    /// walk, `GpuModel::closes_by_ubatch`), once as one call and once cut at
    /// every multiple of [`CHECKPOINT_EVERY`] inside it. The last call rides
    /// the walk in both, and the cut calls give the one call's argmax, its
    /// logits' bits, three steps' ids and bits and its checkpoints.
    #[cfg(all(test, feature = "checkpoint"))]
    fn quantum_case(m: &mut Qwen38Model, ids: &[u32], k: usize) -> Result<bool, GateError> {
        let e = CHECKPOINT_EVERY as usize;
        let n = ids.len();
        let mut run = |cuts: &[usize]| -> Result<(Closed, bool, Vec<u32>), GateError> {
            m.reset()?;
            let mut at = 0;
            for &c in [k].iter().chain(cuts).filter(|&&c| c > 0) {
                m.prompt38_call(&ids[at..c], Prompt38::Auto, CallEnd::Unread, None)?;
                at = c;
            }
            let folds = m.closes_by_ubatch(n - at, Prompt38::Auto)?;
            let closed = closed_run(m, &ids[at..], CallEnd::Close)?;
            Ok((closed, folds, m.body("(qc)")?.checkpoints().positions()))
        };
        let cuts: Vec<usize> = (k / e + 1..)
            .map(|j| j * e)
            .take_while(|&c| c < n - 1)
            .collect();
        let (one, one_folds, one_points) = run(&[])?;
        let (cut, cut_folds, cut_points) = run(&cuts)?;
        let calls: Vec<usize> = [k]
            .iter()
            .chain(&cuts)
            .zip(cuts.iter().chain([&n]))
            .map(|(a, b)| b - a)
            .collect();
        let ok = cut == one && one_folds && cut_folds && cut_points == one_points;
        println!(
            "(qc) {n} ids after {k}: the calls {calls:?} cut at {cuts:?} against the one call: \
             argmax {} = {}, logits {}, steps' ids {:?} = {:?} and bits {}; the last call folds \
             {cut_folds} (one call {one_folds}); points {cut_points:?} = {one_points:?} {}",
            cut.0,
            one.0,
            cut.1 == one.1,
            cut.2.iter().map(|s| s.0).collect::<Vec<_>>(),
            one.2.iter().map(|s| s.0).collect::<Vec<_>>(),
            cut.2 == one.2,
            verdict(ok)
        );
        Ok(ok)
    }

    /// The target position and the draft's committed store count travel together in a state.
    #[derive(Clone, Copy, PartialEq)]
    struct DraftEnd {
        held: usize,
        pos: u32,
    }

    impl DraftEnd {
        fn read(m: &Qwen38Model) -> Result<Self, GateError> {
            Ok(Self {
                held: held_of(m)?,
                pos: m.pos(),
            })
        }

        // PIN(2026-10-22): the refresh commits p0 + 1, not the target's p0 + kept rows.
        fn after(pos: u32, kept: usize) -> Option<Self> {
            Some(Self {
                held: (pos as usize).checked_sub(kept.checked_sub(1)?)?,
                pos,
            })
        }

        // PIN(2026-10-22): a whole restore preserves both counts, including a trailing store.
        fn restored(self, m: &Qwen38Model) -> Result<bool, GateError> {
            Ok(Self::read(m)? == self)
        }
    }

    /// A real MTP chain, with the first proposal kept and the second rejected by a forced pick:
    /// the draft's store trails the target. Its state is put back whole on the engine stream
    /// and from a lane, then the store walked past the target refuses both saves
    /// ([`past_refused`]).
    fn partial_state(
        m: Qwen38Model,
        prompt: &[u32],
        kv_bytes: u64,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let ctx = m.body("partial state")?.ctx() as u32;
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
        let first = spec.prompt(&mut s, prompt)?;
        spec.begin(&s, prompt, first)?;
        let p0 = s.pos();
        let mut rows = [first; 4];
        let n = Draft::propose(spec.draft_mut(), &mut s, first, &mut rows[1..])?;
        if n != 3 {
            return Err("partial state needs three proposals".into());
        }
        let mut picked = Verify::verify(&mut s, rows)?;
        let vocab = u32::try_from(s.model().body("partial state")?.vocab())?;
        picked[0] = rows[1];
        picked[1] = (rows[2] + 1) % vocab;
        let kept = runtime::accepted_rows(&rows, &picked);
        Draft::accept(spec.draft_mut(), &mut s, &rows, &picked, kept)?;
        Verify::commit(&mut s, kept)?;
        let mut m = s.into_model();
        let end = DraftEnd::read(&m)?;
        let shape = kept == 2 && end.pos == p0 + 2 && end.held == p0 as usize + 1;
        let cut = DraftEnd::after(end.pos, kept) == Some(end);
        let before = store_of(&m, end.pos as usize)?;
        let state = m.seq_save()?;
        m.reset()?;
        m.seq_resume(&state)?;
        let whole = end.restored(&m)? && store_of(&m, end.pos as usize)? == before;
        let (_, _, lane) = lane_round(&mut m, prompt, kv_bytes, ("partial accept", 4))?;
        let past = past_refused(&mut m, prompt, end.pos)?;
        println!(
            "(p-partial) kept {kept}, target {} (p0 {p0}), store {}; shape {}, kept_state {}, prefix_draft {}, lane_save {}, past_refused {}",
            end.pos,
            end.held,
            verdict(shape),
            verdict(cut),
            verdict(whole),
            verdict(lane),
            verdict(past)
        );
        Ok((m, shape && cut && whole && lane && past))
    }

    /// (p): a partial accept's records and snapshot. The commit of a verify
    /// that kept fewer rows than it ran cuts the draft's records to the kept
    /// rows. The store's count follows [`DraftEnd::after`], and a snapshot
    /// preserves both counts. Once the store walks past the position (an anchor row) the snapshot is
    /// refused by name; after a scripted window that keeps 2 of its 4 rows
    /// the pass arena's rejected row is a position the arena no longer
    /// holds — a walk that reads it is refused by name, where the uncut
    /// record would let it run.
    fn kept_state(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 6;
        const KEEP: usize = 2;
        let ctx = m.body("kept state")?.ctx() as u32;
        // A drafted run: the model stands after its last commit.
        let (mut m, kept) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, prompt)?;
            let mut k = Kept::default();
            runtime::generate(
                &mut s,
                &mut spec,
                prompt,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            (s.into_model(), k)
        };
        let at = m.pos() as usize;
        let held = m.body("kept state")?.mtp().map_or(0, |d| d.held());
        let cut = kept.rows.last().and_then(|&k| DraftEnd::after(m.pos(), k))
            == Some(DraftEnd::read(&m)?);
        println!(
            "(p) after the drafted run's {} windows at {at}: the draft store holds {held} \
             positions {}",
            kept.rows.len(),
            verdict(cut)
        );
        let snap = m.seq_save();
        let snap_ok = matches!(&snap, Ok(s) if s.positions() as usize == at);
        println!(
            "(p) a snapshot at {at} holds exactly the accepted rows: {} {}",
            match &snap {
                Ok(s) => format!("{} positions, {} bytes", s.positions(), s.bytes()),
                Err(e) => e.to_string(),
            },
            verdict(snap_ok)
        );
        let mut ok = cut && snap_ok;
        ok &= past_refused(&mut m, prompt, u32::try_from(at)?)?;
        // A scripted window that keeps `KEEP` of its 4 rows: the arena's
        // rows `KEEP` on are the rejected tail the commit takes back. The
        // scripted drafter walks no draft, so the store is warmed over the
        // positions first (store walks of other ids).
        let (m, plain) = plain_run(m, prompt, Prompt38::Auto, KEEP + 2)?;
        let (mut m, w_ok) = scripted_window(m, &plain, prompt, KEEP, ctx, "the partial accept")?;
        let at = m.pos() as usize;
        let pos0 = at - KEEP;
        let zeros = vec![0.0f32; MTP_STORE_ROWS * WIDE];
        let vocab = u32::try_from(m.body("kept state")?.vocab())?;
        let ids: Vec<u32> = (0..at + 2)
            .map(|i| (prompt[i % prompt.len()] + 1) % vocab)
            .collect();
        overwrite_store(&mut m, prompt, at, 1)?;
        m.mtp_walk(
            DraftFeed::Rows(Feed {
                tokens: &ids[at..at + 1],
                pos0: u32::try_from(at)?,
                hidden: Hidden::Host(&zeros[..WIDE]),
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Store,
        )?;
        ok &= w_ok;
        ok &= refused(
            "the pass arena's rejected row",
            m.mtp_draft(
                DraftFeed::Rows(Feed {
                    tokens: &ids[..1],
                    pos0: u32::try_from(at + 1)?,
                    hidden: Hidden::Target {
                        walk: TargetRows::Pass,
                        first: KEEP,
                    },
                    media: None,
                }),
                DraftHead::Rows,
                WalkMode::Eager,
            ),
            &format!(
                "rows {KEEP}..{} of the Pass arena as positions {at}..{} for a walk from {}: the \
                 arena holds positions {pos0}..{at}",
                KEEP + 1,
                at + 1,
                at + 1
            ),
        );
        Ok((m, ok))
    }

    /// The target's final hidden row at the last of a prompt call of `rows` ids from position
    /// 0: what a point taken at `rows` keeps for the draft, read right after the call from the
    /// ubatch arena, which holds the call's last ubatch ([`Body38::ubatch_cut`]).
    fn ubatch_last(m: &Qwen38Model, rows: usize) -> Result<Vec<f32>, GateError> {
        let last = (rows - 1) % m.body("(kp-draft)")?.ubatch_rows();
        let all = m.target_streams(TargetRows::Ubatch, last + 1)?;
        Ok(all[last * WIDE..].to_vec())
    }

    /// The draft's store count of a model.
    fn held_of(m: &Qwen38Model) -> Result<usize, GateError> {
        Ok(m.body("(kp-draft)")?.mtp().map_or(0, |d| d.held()))
    }

    /// What a model cut to a point holds for the draft's first walk there: its position, the
    /// draft store's count and rows below the point, and the step arena's row 0.
    struct AtPoint {
        pos: u32,
        held: usize,
        store: (Vec<u16>, Vec<u16>),
        row: Vec<f32>,
    }

    impl AtPoint {
        fn read(m: &Qwen38Model, at: usize) -> Result<AtPoint, GateError> {
            Ok(AtPoint {
                pos: m.pos(),
                held: held_of(m)?,
                store: store_of(m, at)?,
                row: m.target_streams(TargetRows::Step, 1)?,
            })
        }

        /// Whether it stands at `at` with the draft rejoined: the store holds the positions
        /// below `at`, its rows `rows`, and the step arena's row 0 is `point_row` bit for bit.
        fn rejoined(&self, at: usize, rows: &(Vec<u16>, Vec<u16>), point_row: &[f32]) -> bool {
            self.pos as usize == at
                && self.held == at
                && same_rows(&self.store, rows, at).iter().all(|&s| s)
                && self
                    .row
                    .iter()
                    .map(|x| x.to_bits())
                    .eq(point_row.iter().map(|x| x.to_bits()))
        }
    }

    /// The draft's walk of `id` at `at` that reads the step arena's row 0 as position `at - 1`:
    /// its logits' bits, or the words of its refusal.
    fn walk_at(
        m: &mut Qwen38Model,
        id: u32,
        at: usize,
    ) -> Result<Result<Vec<u32>, String>, GateError> {
        let r = m.mtp_draft(
            DraftFeed::Rows(Feed {
                tokens: &[id],
                pos0: u32::try_from(at)?,
                hidden: Hidden::Target {
                    walk: TargetRows::Step,
                    first: 0,
                },
                media: None,
            }),
            DraftHead::Rows,
            WalkMode::Eager,
        );
        Ok(match r {
            Ok(_) => Ok(m.mtp_logits()?.0.iter().map(|x| x.to_bits()).collect()),
            Err(e) => Err(e.to_string()),
        })
    }

    /// A drafted run of `n` ids from a reset with checkpoints on: the model, the prompt call's
    /// argmax (the id at the prompt's end) and the target's hidden row at the prompt's last
    /// position, read right after the prompt call.
    fn drafted_run(
        m: Qwen38Model,
        prompt: &[u32],
        n: usize,
    ) -> Result<(Qwen38Model, u32, Vec<f32>), GateError> {
        let ctx = m.body("(kp-draft)")?.ctx() as u32;
        let mut m = m;
        m.body_parts("(kp-draft)")?.2.set_checkpoints(true);
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
        let first = spec.prompt(&mut s, prompt)?;
        let point_row = ubatch_last(s.model(), prompt.len())?;
        runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(n, ctx)?,
            &mut Kept::default(),
        )?;
        Ok((s.into_model(), first, point_row))
    }

    /// (kp-draft): a cut to a point and a prefix restore to it keep the draft. A drafted run
    /// with a point at its prompt's end (`p`) below its end is saved; put back whole, the
    /// draft's store holds its saved count, which can trail the target (the contrast).
    /// Then the state put back whole and cut to `p` by `rollback`, the state put back as its first `p` positions
    /// (`seq_resume_prefix` asked one short of the end), and a state saved there put back whole
    /// each stand at `p` with the draft rejoined ([`AtPoint::rejoined`]): the draft's store
    /// holds the `p` positions below the point, its rows the drafted run's, and the step
    /// arena's row 0 is the target's hidden row at `p - 1` bit for bit as the prompt call left
    /// it. The draft's walk at `p` that reads that row runs, its logits the same bits all three
    /// ways. The prefix restore writes no row past `p`: run over another prompt first, the
    /// store keeps that run's rows at `p..n`, which differ from the state's. Mutants: the cut
    /// that forgets the draft's store (`held` 0, the walk refused); a row other than the
    /// point's (the row's bits); the prefix restore that forgets the draft's store before the
    /// rejoin (`held` 0 there, the walk refused); the prefix restore that adopts the whole
    /// saved store (the state's rows past `p`). After a reset, a run of other ids with its own
    /// point at `p` and cut to it leaves that run's row in the step arena, not the first run's:
    /// a reset that keeps the points' rows (the first run's row is found first).
    fn prefix_draft(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 6;
        let (mut m, first, point_row) = drafted_run(m, prompt, N)?;
        let end = DraftEnd::read(&m)?;
        let state = m.seq_save()?;
        let n = state.positions();
        let ask = n.saturating_sub(1);
        let at = state.keep_point(ask);
        if at == 0 || at >= n || at as usize != prompt.len() {
            m.body_parts("(kp-draft)")?.2.set_checkpoints(false);
            m.reset()?;
            println!(
                "(kp-draft) a drafted run's state of {n} positions carries no point at its \
                 prompt's end {} below its end (asked {ask}, kept {at}) FAIL",
                prompt.len()
            );
            return Ok((m, false));
        }
        let p = at as usize;
        m.reset()?;
        m.seq_resume(&state)?;
        let (whole_held, whole_pos) = (held_of(&m)?, m.pos());
        let contrast = end.restored(&m)?;
        let rows = store_of(&m, p)?;
        let saved_all = store_of(&m, n as usize)?;
        m.reset()?;
        m.seq_resume(&state)?;
        m.rollback(at)?;
        let roll = AtPoint::read(&m, p)?;
        let roll_walk = walk_at(&mut m, first, p)?;
        m.reset()?;
        let vocab = u32::try_from(m.body("(kp-draft)")?.vocab())?;
        let other: Vec<u32> = prompt.iter().map(|&t| (t + 1) % vocab).collect();
        let (mut m, _, other_row) = drafted_run(m, &other, N)?;
        let other_all = store_of(&m, n as usize)?;
        m.rollback(at)?;
        let fresh = AtPoint::read(&m, p)?;
        m.reset()?;
        let put = match m.seq_resume_prefix(&state, ask) {
            Ok(put) => put,
            Err(e) => {
                m.body_parts("(kp-draft)")?.2.set_checkpoints(false);
                m.reset()?;
                println!("(kp-draft) the prefix of {ask} positions refused: {e} FAIL");
                return Ok((m, false));
            }
        };
        let pre = AtPoint::read(&m, p)?;
        let past = store_of(&m, n as usize)?;
        let again = m.seq_save();
        let pre_walk = walk_at(&mut m, first, p)?;
        m.reset()?;
        let saved = match again.map(|s| m.seq_resume(&s).map(|()| s.positions())) {
            Ok(Ok(k)) => Ok((k, AtPoint::read(&m, p)?, walk_at(&mut m, first, p)?)),
            Ok(Err(e)) | Err(e) => Err(e.to_string()),
        };
        m.body_parts("(kp-draft)")?.2.set_checkpoints(false);
        m.reset()?;
        let roll_ok = roll.rejoined(p, &rows, &point_row) && roll_walk.is_ok();
        let tail = |a, b| same_rows(a, b, n as usize)[p..].iter().all(|&s| s);
        let (kept_tail, differs) = (tail(&past, &other_all), !tail(&other_all, &saved_all));
        let pre_ok = put == at
            && pre.rejoined(p, &rows, &point_row)
            && pre_walk.is_ok()
            && kept_tail
            && differs;
        let bits = |r: &[f32]| r.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let fresh_ok = bits(&fresh.row) == bits(&other_row) && bits(&other_row) != bits(&point_row);
        let (saved_ok, saved_said) = match &saved {
            Ok((k, a, w)) => (
                *k == at && a.rejoined(p, &rows, &point_row) && w.is_ok() && *w == roll_walk,
                format!(
                    "{k} positions, the draft's store {} at {}, the row {}, the walk {}",
                    a.held,
                    a.pos,
                    a.row == point_row,
                    w.as_ref().map_or_else(Clone::clone, |_| "ran".to_owned())
                ),
            ),
            Err(e) => (false, format!("refused: {e}")),
        };
        let same = roll_walk.is_ok() && pre_walk == roll_walk;
        let said = |a: &AtPoint, w: &Result<Vec<u32>, String>| {
            format!(
                "the draft's store {} at {}, its rows the run's {}, the step row the point's {}, \
                 the walk {}",
                a.held,
                a.pos,
                same_rows(&a.store, &rows, p).iter().all(|&s| s),
                a.row == point_row,
                w.as_ref().map_or_else(Clone::clone, |_| "ran".to_owned())
            )
        };
        println!(
            "(kp-draft) a drafted run's state of {n} positions, a point at {at} (asked {ask}): \
             whole resume holds the draft's {whole_held} at {whole_pos} {}",
            verdict(contrast)
        );
        println!(
            "(kp-draft) resume then rollback to {at}: {} {}",
            said(&roll, &roll_walk),
            verdict(roll_ok)
        );
        println!(
            "(kp-draft) the prefix put back returned {put}: {}; its walk's logits the rollback's \
             {same}; the rows past {at} another run's {kept_tail} (the state's differ there \
             {differs}) {}",
            said(&pre, &pre_walk),
            verdict(pre_ok && same)
        );
        println!(
            "(kp-draft) a reset, then a run of other ids with its own point at {at}, cut to it: \
             the step row the other run's {} (differing from the first run's {}) {}",
            bits(&fresh.row) == bits(&other_row),
            bits(&other_row) != bits(&point_row),
            verdict(fresh_ok)
        );
        println!(
            "(kp-draft) a state saved at the prefix, put back whole: {saved_said} {}",
            verdict(saved_ok)
        );
        Ok((
            m,
            contrast && roll_ok && pre_ok && same && saved_ok && fresh_ok,
        ))
    }

    /// (kp-draft) the store behind the point: a cut to a point the draft's store does not reach
    /// keeps the draft off there. A drafted run, then `K` ids by a prompt call without the
    /// draft (a point at its end, the target's row before it in the ubatch arena, the draft's
    /// store still at the drafted run's end), `K` more the same way without checkpoints, then
    /// `rollback` to the first call's end: the draft's store is forgotten (`held` 0) and the
    /// model stands at the point. The second call takes no point because a full ledger keeps
    /// the drafted run's first point and would evict the one the cut needs. Mutant: the rejoin
    /// without its store condition (`held` the drafted run's).
    fn store_behind(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 6;
        const K: usize = 16;
        let (mut m, _, _) = drafted_run(m, prompt, N)?;
        let (end, before) = (m.pos(), held_of(&m)?);
        m.prompt38(&prompt[..K], Prompt38::Auto)?;
        let at = m.pos();
        m.body_parts("(kp-draft)")?.2.set_checkpoints(false);
        m.prompt38(&prompt[K..2 * K], Prompt38::Auto)?;
        let past = m.pos();
        let cut = m.rollback(at);
        let (held, pos) = (held_of(&m)?, m.pos());
        m.reset()?;
        let behind = before < at as usize;
        let ok = behind && cut.is_ok() && pos == at && held == 0;
        println!(
            "(kp-draft) the store behind the point: a drafted run to {end}, its draft's store \
             {before}, {K} ids without the draft to a point at {at} and {K} more to {past}, cut \
             back to {at} ({}): the draft's store holds {held} at {pos} {}",
            cut.map_or_else(|e| format!("refused: {e}"), |()| "ran".to_owned()),
            verdict(ok)
        );
        Ok((m, ok))
    }

    /// (sl): a drafted run's state saved on a save lane (`seq_save_async`) three times, its copy
    /// landed each time: whole, the draft's store holding its saved count at target `n`,
    /// then after a rollback to a point `at` below `n`, which the draft rejoins (its store counts `at`), then
    /// a state of `BEHIND` ids from a reset by a prompt call without the draft (the store holds 0
    /// of `BEHIND`). Each
    /// copy's bytes (`SaveCost::bytes`) are the target's rows of the state's positions
    /// ([`seq_positional_bytes`]), its recurrent stores at the end and at each point the state
    /// carries ([`checkpoint_bytes`]), and the draft's store rows of the same positions,
    /// `kv_bytes * n / ctx` of the store's `kv_bytes` over its `ctx` positions (the plan's),
    /// whatever the store holds: they follow `n`, not `ctx`. Then store walks of other ids,
    /// each round its own, overwrite the store's rows below the state's positions (seen to
    /// change), and the state put back whole leaves the store holding the positions it held at
    /// the save, and its rows below the state's positions the saved ones bit for bit
    /// ([`store_of`]). Mutants: the draft's store read whole on the engine stream instead of
    /// in the copy (the copies' bytes lack the draft's term); the K plane's spans in place of
    /// the V plane's (the V rows stay overwritten); the planes' spans strided by `n`
    /// positions, not `ctx` (the second KV head's rows stay overwritten); a put-back that
    /// counts the state's positions (the third state's store holds `BEHIND`, not 0).
    fn lane_save(
        m: Qwen38Model,
        prompt: &[u32],
        kv_bytes: u64,
    ) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 6;
        const BEHIND: usize = 16;
        let ctx = m.body("(sl)")?.ctx() as u32;
        let mut m = m;
        m.body_parts("(sl)")?.2.set_checkpoints(true);
        let mut m = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, prompt)?;
            runtime::generate(
                &mut s,
                &mut spec,
                prompt,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut Kept::default(),
            )?;
            s.into_model()
        };
        let n = m.pos();
        let (state, _, whole_ok) = lane_round(&mut m, prompt, kv_bytes, ("whole", 1))?;
        let at = state.keep_point(n.saturating_sub(1));
        let prefix_ok = whole_ok && lane_prefix(&mut m, &state, n, at)?;
        let refused_ok = whole_ok && past_refused(&mut m, prompt, n)?;
        let cut_ok = if at == 0 || at >= n {
            println!(
                "(sl) the state of {n} positions carries no point below its end (asked {}, \
                 kept {at}): nothing to roll back to FAIL",
                n.saturating_sub(1)
            );
            false
        } else {
            m.rollback(at)?;
            let (_, held, ok) = lane_round(&mut m, prompt, kv_bytes, ("rolled back", 2))?;
            ok && held == at as usize
        };
        let behind_ok = cut_ok && {
            m.reset()?;
            m.prompt38(&prompt[..BEHIND], Prompt38::Auto)?;
            let (_, held, ok) = lane_round(&mut m, prompt, kv_bytes, ("store behind", 3))?;
            ok && held == 0
        };
        m.body_parts("(sl)")?.2.set_checkpoints(false);
        m.reset()?;
        Ok((
            m,
            whole_ok && prefix_ok && refused_ok && cut_ok && behind_ok,
        ))
    }

    /// The lane's state of a drafted run (`lane_state`, `n` positions, its point at `at`) put
    /// back as the point's prefix (asked one short of the end) beside the engine stream's state
    /// of the same sequence: each stands at `at` with the draft rejoined, and the lane's store
    /// rows below `at` and step arena row 0 are the engine stream's bit for bit. The lane's
    /// state is put back whole after, as the model stood. Mutant: a lane state without the
    /// points' hidden rows (its prefix put-back leaves the draft off, `held` 0).
    fn lane_prefix(
        m: &mut Qwen38Model,
        lane_state: &Seq38,
        n: u32,
        at: u32,
    ) -> Result<bool, GateError> {
        if at == 0 || at >= n {
            return Ok(false);
        }
        let (here, ask) = (at as usize, n - 1);
        let engine = m.seq_save()?;
        m.reset()?;
        let kept_e = m.seq_resume_prefix(&engine, ask)?;
        let want = AtPoint::read(m, here)?;
        m.reset()?;
        let kept_l = m.seq_resume_prefix(lane_state, ask)?;
        let got = AtPoint::read(m, here)?;
        m.reset()?;
        m.seq_resume(lane_state)?;
        let bits = |r: &[f32]| r.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let same = same_rows(&got.store, &want.store, here).iter().all(|&s| s)
            && bits(&got.row) == bits(&want.row);
        let ok = kept_e == at
            && kept_l == at
            && want.held == here
            && got.held == here
            && got.pos == at
            && same;
        println!(
            "(sl) the lane's state of {n} positions put back as the point at {at}'s prefix \
             (kept {kept_l}, the engine stream's {kept_e}): the draft's store holds {} at {} \
             (the engine stream's {}), its rows below {at} and the step row the engine \
             stream's {same} {}",
            got.held,
            got.pos,
            want.held,
            verdict(ok)
        );
        Ok(ok)
    }

    /// One [`lane_save`] round on the model as it stands, `round` its name and the ids'
    /// shift: the state saved on a lane and its copy landed, the store overwritten
    /// ([`overwrite_store`]), the state put back whole. The state, the positions the draft's
    /// store held at the save, and the round's verdict.
    fn lane_round(
        m: &mut Qwen38Model,
        prompt: &[u32],
        kv_bytes: u64,
        round: (&str, u32),
    ) -> Result<(Seq38, usize, bool), GateError> {
        let (what, shift) = round;
        let n = m.pos() as usize;
        let end = DraftEnd::read(m)?;
        let held = end.held;
        let store_ctx = m
            .body("(sl)")?
            .mtp()
            .ok_or("the load opened no draft")?
            .ctx();
        let before = store_of(m, n)?;
        let (gdn, qsa) = (cfg().shape.n_gdn(), cfg().shape.n_qsa());
        let lane = m.seq_lane(2 * seq38_bytes(gdn, qsa, n, true, KvQ8::F16))?;
        let (state, pending) = m.seq_save_async(&lane)?.map_err(GateError::from)?;
        let cost = pending.land()?;
        let copies = 1 + pending.points().len() as u64;
        let target = seq_positional_bytes(qsa, n, KvQ8::F16) + copies * checkpoint_bytes(gdn, qsa);
        let store = kv_bytes * n as u64 / store_ctx as u64;
        let derived = target + store;
        m.reset()?;
        overwrite_store(m, prompt, n, shift)?;
        let seen = store_of(m, n)? != before;
        m.reset()?;
        m.seq_resume(&state)?;
        let put = held_of(m)?;
        let same = store_of(m, n)? == before;
        let bytes_ok = cost.bytes == derived;
        let back_ok = end.restored(m)? && seen && same;
        println!(
            "(sl) {what}, a state of {n} positions whose draft store holds {held} saved on a \
             lane: the copy {} bytes (derived {derived}: the target's {target}, recurrent copies \
             {copies}, the draft's store {store} = {kv_bytes} * {n} / {store_ctx}) {}; \
             the store overwritten {seen}, put back whole it holds {put} positions, its rows \
             below {n} the saved ones {same} {}",
            cost.bytes,
            verdict(bytes_ok),
            verdict(back_ok)
        );
        Ok((state, held, bytes_ok && back_ok))
    }

    /// The saves of a model at position `n` whose draft store was walked past it: store walks
    /// from position 0 through `n` (the anchor row), wherever the store stood, since a store
    /// left by a partial accept trails the target and a walk from `n` would be refused. The
    /// save on the engine stream and the one on a lane are each refused by name, nothing
    /// enqueued.
    fn past_refused(m: &mut Qwen38Model, prompt: &[u32], n: u32) -> Result<bool, GateError> {
        overwrite_store(m, prompt, n as usize + 1, 1)?;
        let (gdn, qsa) = (cfg().shape.n_gdn(), cfg().shape.n_qsa());
        let lane = m.seq_lane(seq38_bytes(gdn, qsa, n as usize, true, KvQ8::F16))?;
        let want = format!(
            "a state at {n} of a draft store that holds {} positions",
            n + 1
        );
        let sync = refused(
            "a snapshot of a store past the position",
            m.seq_save(),
            &want,
        );
        let on_lane = refused(
            "a lane save of a store past the position",
            m.seq_save_async(&lane),
            &want,
        );
        Ok(sync && on_lane)
    }

    /// The draft's store rows at positions `0..n` written by store walks of other ids (the
    /// prompt's, each `shift` up) beside zero hidden rows, from position 0.
    fn overwrite_store(
        m: &mut Qwen38Model,
        prompt: &[u32],
        n: usize,
        shift: u32,
    ) -> Result<(), GateError> {
        let vocab = u32::try_from(m.body("store walks")?.vocab())?;
        let ids: Vec<u32> = (0..n)
            .map(|i| (prompt[i % prompt.len()] + shift) % vocab)
            .collect();
        let zeros = vec![0.0f32; MTP_STORE_ROWS * WIDE];
        for (i, run) in ids.chunks(MTP_STORE_ROWS).enumerate() {
            m.mtp_walk(
                DraftFeed::Rows(Feed {
                    tokens: run,
                    pos0: u32::try_from(i * MTP_STORE_ROWS)?,
                    hidden: Hidden::Host(&zeros[..run.len() * WIDE]),
                    media: None,
                }),
                DraftHead::Rows,
                WalkMode::Store,
            )?;
        }
        Ok(())
    }

    /// The long point clause's context: a prompt call past the first inner mark
    /// ([`CHECKPOINT_EVERY`]), its drafted run and a few positions of slack.
    const CTX_LONG: u64 = 2 * CHECKPOINT_EVERY as u64;

    /// (kp-draft) a point inside a long prompt call, on a load of [`CTX_LONG`] positions: a
    /// drafted run whose prompt call of `CHECKPOINT_EVERY + prompt.len()` ids takes an inner
    /// mark at [`CHECKPOINT_EVERY`] (an anchor point, which every later save carries) and one
    /// at its end. Cut back to the inner point by `rollback`, and the run's state put back as
    /// the inner point's prefix (asked a few positions past it), each stands at the point with
    /// the draft rejoined ([`AtPoint::rejoined`]): the store holding the positions below it,
    /// its rows the run's, and the step arena's row 0 the target's hidden row at
    /// `CHECKPOINT_EVERY - 1` bit for bit as a call of the first `CHECKPOINT_EVERY` ids alone
    /// leaves it. The draft's walk at the point runs, its logits the same bits both ways.
    /// Mutant: the row of the mark's own position, not the one before it (no row, the draft
    /// off).
    fn long_point(inputs: &PlanInputs, mtp: &MtpInputs, prompt: &[u32]) -> Result<bool, GateError> {
        const N: usize = 6;
        let p = CHECKPOINT_EVERY as usize;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(usize::try_from(CTX_LONG)?)?;
        let (file, draft) = (open_target()?, open_draft()?);
        let machine = machine(card()?, inputs.stack(), u64::try_from(ub)?);
        let extra = mtp.card_bytes(CTX_LONG)?;
        let plan = inputs.plan_mtp(
            &machine,
            (CTX_LONG, u64::try_from(ub)?),
            &plan_levers(extra)?,
            mtp,
        )?;
        let mut m = Body38::open_placed_mtp(file, &plan, inputs, 0, cfg().host, ub, &draft, mtp)?;
        let token = *prompt.first().ok_or("the reset clause has no token")?;
        m.reset()?;
        m.reset()?;
        let empty_room = m.body("draft reset room")?.store_room().0 == 0;
        m.step(&[token])?;
        m.reset()?;
        let fresh_short = m.body("draft reset room")?.store_room().0;
        let vocab = u32::try_from(m.body("(kp-draft)")?.vocab())?;
        let ids: Vec<u32> = (0..p + prompt.len())
            .map(|i| Ok((prompt[i % prompt.len()] + u32::try_from(i / prompt.len())?) % vocab))
            .collect::<Result<_, std::num::TryFromIntError>>()?;
        m.reset()?;
        m.prompt38(&ids[..p], Prompt38::Auto)?;
        let point_row = ubatch_last(&m, p)?;
        let (mut m, _, _) = drafted_run(m, &ids, N)?;
        let state = m.seq_save()?;
        let n = state.positions();
        m.reset()?;
        m.seq_resume(&state)?;
        let rows = store_of(&m, p)?;
        let at = u32::try_from(p)?;
        let cut = m.rollback(at);
        let roll = AtPoint::read(&m, p)?;
        let roll_walk = walk_at(&mut m, ids[p], p)?;
        m.reset()?;
        let ask = at + 3;
        let put = m.seq_resume_prefix(&state, ask);
        let pre = AtPoint::read(&m, p)?;
        let pre_walk = walk_at(&mut m, ids[p], p)?;
        m.body_parts("(kp-draft)")?.2.set_checkpoints(false);
        m.reset()?;
        m.step(&[token])?;
        m.reset()?;
        let after_short = m.body("draft reset room")?.store_room().0;
        let room_ok = empty_room && fresh_short > 0 && after_short == fresh_short;
        println!(
            "draft reset room: empty {empty_room}, fresh short {fresh_short} B, after a long \
             drafted request and a short request {after_short} B {}",
            verdict(room_ok)
        );
        drop(m);
        let said = |a: &AtPoint, w: &Result<Vec<u32>, String>| {
            format!(
                "the draft's store {} at {}, its rows the run's {}, the step row the call's {}, \
                 the walk {}",
                a.held,
                a.pos,
                same_rows(&a.store, &rows, p).iter().all(|&s| s),
                a.row == point_row,
                w.as_ref().map_or_else(Clone::clone, |_| "ran".to_owned())
            )
        };
        let roll_ok = cut.is_ok() && roll.rejoined(p, &rows, &point_row) && roll_walk.is_ok();
        let pre_ok = put.as_ref().is_ok_and(|&k| k == at)
            && pre.rejoined(p, &rows, &point_row)
            && pre_walk.is_ok()
            && pre_walk == roll_walk;
        println!(
            "(kp-draft) a run of {n} positions from a prompt call of {} ids, cut back to its \
             inner point at {p} ({}): {} {}",
            ids.len(),
            cut.map_or_else(|e| format!("refused: {e}"), |()| "ran".to_owned()),
            said(&roll, &roll_walk),
            verdict(roll_ok)
        );
        println!(
            "(kp-draft) its state put back as a prefix asked {ask} (returned {}): {}; its walk's \
             logits the cut's {} {}",
            put.map_or_else(|e| format!("refused: {e}"), |k| k.to_string()),
            said(&pre, &pre_walk),
            pre_walk == roll_walk,
            verdict(pre_ok)
        );
        Ok(roll_ok && pre_ok && room_ok)
    }

    /// (x): a resume refused before any copy. A state of a load without the
    /// draft (its own short prompt run) put back on this load is refused by
    /// name — the store layouts differ — with the model untouched: its
    /// stores still the empty model's bit for bit, its position 0, the
    /// draft's store empty.
    fn resume_refused(
        m: Qwen38Model,
        inputs: &PlanInputs,
        ub: usize,
        prompt: &[u32],
    ) -> Result<(Qwen38Model, bool), GateError> {
        let machine = machine(card()?, inputs.stack(), u64::try_from(ub)?);
        let plain_plan = inputs.plan(&machine, (CTX, u64::try_from(ub)?), &plan_levers(0)?)?;
        let mut b = Body38::open_placed(open_target()?, &plain_plan, inputs, 0, cfg().host, ub)?;
        // Other ids over the same length: the state's rows differ from
        // whatever this model's stores hold, so a copy that should not
        // happen shows.
        let vocab = u32::try_from(b.body("resume refused")?.vocab())?;
        let ids: Vec<u32> = prompt.iter().map(|&t| (t + 1) % vocab).collect();
        b.prompt38(&ids, Prompt38::Pass)?;
        let without = b.seq_save()?;
        drop(b);
        let mut m = m;
        m.reset()?;
        let read = |m: &mut Qwen38Model| -> Result<(Vec<Store38Host>, Vec<f32>), GateError> {
            let (gpu, _, body) = m.body_parts("resume refused")?;
            Ok(body.stores_host(gpu)?)
        };
        let before = read(&mut m)?;
        let r = m.seq_resume(&without);
        let after = read(&mut m)?;
        let pos = m.pos();
        let held = m.body("resume refused")?.mtp().map_or(0, |d| d.held());
        let stores_same = before.0.len() == after.0.len()
            && before.0.iter().zip(&after.0).all(|(a, b)| a.same_bits(b));
        let ring_same = before.1.len() == after.1.len()
            && before
                .1
                .iter()
                .zip(&after.1)
                .all(|(a, b)| a.to_bits() == b.to_bits());
        let named = matches!(&r, Err(e) if e.to_string().contains(
            "a state without the draft's side put back on a load with the draft"
        ));
        let untouched = stores_same && ring_same && pos == 0 && held == 0;
        let ok = named && untouched;
        println!(
            "(x) a state without the draft's side put back on a load with the draft: {} — the \
             model untouched: stores {}, position {pos}, draft store {held} {}",
            match &r {
                Ok(()) => "resumed".to_string(),
                Err(e) => e.to_string(),
            },
            if stores_same && ring_same {
                "same".to_string()
            } else {
                "changed".to_string()
            },
            verdict(ok)
        );
        Ok((m, ok))
    }

    /// D3K's prefill, the e2e family's deep prompt (the set's `# tokens`), from this tier's set
    /// ([`tier::input_set`]).
    fn deep_prompt() -> Result<Vec<u32>, GateError> {
        let d3k = refset::arch::qwen4exp::D3K;
        let set = tier::input_set("the deep prompt", (d3k, &IK), fixture::mirror(d3k))?;
        let man = RefManifest::open(&set.path, set.family)?;
        let (_, _, prefill) = man.step()?;
        Ok(prefill.to_vec())
    }

    /// One plain run: `ids` prefilled by `path`, `n` tokens generated one
    /// step each, every step's stores captured. Its tokens and its stores at
    /// each position are the drafted runs' reference.
    struct Plain {
        tokens: Vec<u32>,
        stores: Vec<(Vec<Store38Host>, Vec<f32>)>,
    }

    fn plain_run(
        m: Qwen38Model,
        ids: &[u32],
        path: Prompt38,
        n: usize,
    ) -> Result<(Qwen38Model, Plain), GateError> {
        let mut m = m;
        m.reset()?;
        let mut tokens = vec![m.prompt38(ids, path)?];
        let mut stores = Vec::with_capacity(n);
        for _ in 1..n {
            let t = m.step(&[tokens[tokens.len() - 1]])?;
            tokens.push(t);
            let (gpu, _, b) = m.body_parts("plain")?;
            stores.push(b.stores_host(gpu)?);
        }
        Ok((m, Plain { tokens, stores }))
    }

    /// A run's live stores as the body reads them back, and its PLE ring.
    type Stores = (Vec<Store38Host>, Vec<f32>);

    /// One plain run of `n` tokens from a reset, as [`plain_run`]'s, its
    /// stores read once at its end.
    fn plain_to(
        m: Qwen38Model,
        ids: &[u32],
        path: Prompt38,
        n: usize,
    ) -> Result<(Qwen38Model, Vec<u32>, Stores), GateError> {
        let mut m = m;
        m.reset()?;
        let mut tokens = vec![m.prompt38(ids, path)?];
        slots_gate::greedy(&mut m, &mut tokens, n.saturating_sub(1))?;
        let stores = {
            let (gpu, _, b) = m.body_parts("plain")?;
            b.stores_host(gpu)?
        };
        Ok((m, tokens, stores))
    }

    /// The store comparison's shape: the e2e gate's `(v)` rule in one bool —
    /// the committed delta lane and its carry, the conv ring's eight slots before the
    /// count, the K/V planes' rows below it, the raw key ring's slots of the
    /// count's pool not yet complete, the pools
    /// complete at the count (a pool a rejected row completed is read only
    /// at a count that completes it again), the PLE ring's fourteen slots
    /// before it.
    fn live_same(
        a: &(Vec<Store38Host>, Vec<f32>),
        b: &(Vec<Store38Host>, Vec<f32>),
        pos: usize,
    ) -> bool {
        const N_KV: usize = 2;
        const HEAD: usize = 256;
        const IDX_DIM: usize = 128;
        const POOL: usize = 4;
        const CONV_RING: usize = 11;
        const PLE_RING: usize = 17;
        let conv_ch = 2 * 16 * 128 + 48 * 128;
        let slots =
            |ring: usize, back: usize| (pos.saturating_sub(back)..pos).map(move |p| p % ring);
        let ring_same = |x: &[f32], y: &[f32], ring: usize, width: usize, back: usize| {
            x.len() == ring * width
                && y.len() == ring * width
                && slots(ring, back).all(|s| {
                    x[s * width..(s + 1) * width]
                        .iter()
                        .zip(&y[s * width..(s + 1) * width])
                        .all(|(u, v)| u.to_bits() == v.to_bits())
                })
        };
        let rows_same = |x: &[u16], y: &[u16], heads: usize, width: usize| {
            let ctx = x.len() / (heads * width);
            x.len() == y.len()
                && x.len() == heads * ctx * width
                && pos <= ctx
                && (0..heads).all(|h| {
                    let at = h * ctx * width;
                    x[at..at + pos * width] == y[at..at + pos * width]
                })
        };
        let layers = a.0.len() == b.0.len()
            && a.0.iter().zip(&b.0).all(|(x, y)| match (x, y) {
                (
                    Store38Host::Rec {
                        state: s,
                        ring: r,
                        carry: c,
                    },
                    Store38Host::Rec {
                        state: s2,
                        ring: r2,
                        carry: c2,
                    },
                ) => {
                    let bits = bloomery_gpu_gates::bits_equal;
                    bits(s, s2) && ring_same(r, r2, CONV_RING, conv_ch, 8) && bits(c, c2)
                }
                (
                    Store38Host::Qsa { k, v, ring, pooled },
                    Store38Host::Qsa {
                        k: k2,
                        v: v2,
                        ring: ring2,
                        pooled: pooled2,
                    },
                ) => {
                    let live = pos / POOL * IDX_DIM;
                    rows_same(k, k2, N_KV, HEAD)
                        && rows_same(v, v2, N_KV, HEAD)
                        && key_ring_live(ring, pos)
                            .is_some_and(|x| Some(x) == key_ring_live(ring2, pos))
                        && pooled.len() == pooled2.len()
                        && live <= pooled.len()
                        && pooled[..live] == pooled2[..live]
                }
                _ => false,
            });
        layers && ring_same(&a.1, &b.1, PLE_RING, STREAMS * HIDDEN, 14)
    }

    /// One drafted pass's kept rows, for the histogram and the rejection.
    #[derive(Default)]
    struct Kept {
        rows: Vec<usize>,
        proposed: Vec<bool>,
        /// The generation's end position: the target's after the last pass.
        pos: u32,
    }

    impl PassSink<app::Session<Body38>> for Kept {
        type Error = GateError;

        fn begin(&mut self, _: &app::Session<Body38>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            t: &app::Session<Body38>,
            c: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            self.rows.push(c.kept);
            self.proposed.push(c.proposed);
            self.pos = t.pos();
            Ok(())
        }
    }

    /// [`PassSink`] that hears nothing.
    struct QuietSink;

    impl PassSink<app::Session<Body38>> for QuietSink {
        type Error = GateError;

        fn begin(&mut self, _: &app::Session<Body38>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &app::Session<Body38>,
            _: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            Ok(())
        }
    }

    /// The drafted session over the model from a reset — the plain run it
    /// is held against starts at position 0 — the draft opened, the verify
    /// widths captured.
    fn drafted_session(
        m: Qwen38Model,
        cfg: app::arch::qwen3moe::Q38Cfg,
        ctx: u32,
    ) -> Result<(app::Session<Body38>, app::arch::qwen3moe::Drafted38), GateError> {
        let mut m = m;
        m.reset()?;
        let mut s = app::Session::from_model(m, ctx);
        let draft = app::mtp::MtpDraft::open(s.model(), cfg.prompt, cfg.draft)?;
        let spec = s.with_draft::<app::mtp::MtpDraft<Body38>, 4>(draft, &mut Quiet)?;
        Ok((s, spec))
    }

    /// The windows the draft keeps ([`app::mtp::MtpDraft::keep_windows`],
    /// `BLOOMERY_MTP_WINDOWS`): `unkept` windows came back from the drafted
    /// run `off` (its ids and passes), which kept none, and must be 0; the
    /// same run keeping its windows reads back the same ids through the same
    /// passes, one window a drafted pass, each kept one row past the ids its
    /// window says the target kept, those ids the run's own tokens at the
    /// positions after the window's, each probability in (0, 1].
    fn kept_windows(
        m: Qwen38Model,
        prompt: &[u32],
        ctx: u32,
        off: (&[u32], &Kept),
        unkept: usize,
    ) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 64;
        let (off_tokens, off_kept) = off;
        let (m, out, kept, windows) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            spec.draft_mut().keep_windows();
            let first = spec.prompt(&mut s, prompt)?;
            let mut k = Kept::default();
            let out = runtime::generate(
                &mut s,
                &mut spec,
                prompt,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            let w = spec.draft_mut().take_windows();
            (s.into_model(), out, k, w)
        };
        let same = out.tokens == off_tokens
            && kept.rows == off_kept.rows
            && kept.proposed == off_kept.proposed;
        let drafted: Vec<usize> = kept
            .rows
            .iter()
            .zip(&kept.proposed)
            .filter(|(_, p)| **p)
            .map(|(k, _)| *k)
            .collect();
        let p0 = prompt.len();
        let mut bad = Vec::new();
        if drafted.len() != windows.len() {
            bad.push(format!(
                "{} windows of {} drafted passes",
                windows.len(),
                drafted.len()
            ));
        }
        for (i, (&k, w)) in drafted.iter().zip(&windows).enumerate() {
            let at = w.pos as usize + 1 - p0;
            let emitted = out.tokens.get(at..at + w.accepted);
            let p_ok = w.p.iter().all(|&p| p > 0.0 && p <= 1.0);
            if w.accepted + 1 != k
                || w.ids.len() != w.p.len()
                || w.ids.is_empty()
                || !p_ok
                || emitted.is_some_and(|e| e != &w.ids[..w.accepted])
            {
                bad.push(format!(
                    "window {i}: pass kept {k}, window {:?} p {:?} accepted {}, emitted {emitted:?}",
                    w.ids, w.p, w.accepted
                ));
            }
        }
        let ok = unkept == 0 && same && bad.is_empty();
        println!(
            "(w) depth {}: unset the draft keeps {unkept} windows; kept, the same ids and passes \
             {} and {} windows of {} drafted passes, each its pass's kept rows and ids{} {}",
            prompt.len(),
            verdict(same),
            windows.len(),
            drafted.len(),
            if bad.is_empty() {
                String::new()
            } else {
                format!(" — {}", bad.join("; "))
            },
            verdict(ok)
        );
        Ok((m, ok))
    }

    /// A draft that proposes the plain run's own next tokens, its `keep`th
    /// id one the target rejects: every window keeps exactly `keep` rows
    /// through the same verify and commit the MTP draft drives.
    struct Scripted<'a> {
        plain: &'a [u32],
        keep: usize,
        vocab: u32,
        at: usize,
    }

    impl Draft<app::Session<Body38>> for Scripted<'_> {
        const WIDTH: usize = 3;
        const TAPS: TapNeed = TapNeed::Final;

        fn begin(
            &mut self,
            _: &app::Session<Body38>,
            _: &[u32],
            _: u32,
        ) -> Result<(), app::SessionError> {
            Ok(())
        }

        fn propose(
            &mut self,
            _: &mut app::Session<Body38>,
            _: u32,
            out: &mut [u32],
        ) -> Result<usize, app::SessionError> {
            // The plain run's next tokens up to the keep − 1st, then ids it
            // never emits: the verify keeps exactly `keep` rows.
            let wrong = |want: u32| (want + 1_000) % self.vocab;
            for (i, o) in out.iter_mut().enumerate() {
                let want = self.plain.get(self.at + 1 + i).copied().unwrap_or(0);
                *o = if i < self.keep - 1 { want } else { wrong(want) };
            }
            self.at += self.keep;
            Ok(Self::WIDTH)
        }

        fn accept(
            &mut self,
            _: &mut app::Session<Body38>,
            _: &[u32],
            _: &[u32],
            _: usize,
        ) -> Result<(), app::SessionError> {
            Ok(())
        }

        fn stepped(
            &mut self,
            _: &mut app::Session<Body38>,
            _: u32,
            _: u32,
        ) -> Result<(), app::SessionError> {
            Ok(())
        }
    }

    /// One scripted window keeping exactly `keep` rows: the kept tokens are
    /// the plain run's, and the live stores after the commit equal its at
    /// the same position — `commit(k)` is `k` steps. The model moves in and
    /// out, from a reset as the plain run's. The label names the prompt in
    /// the clause's line.
    fn scripted_window(
        m: Qwen38Model,
        plain: &Plain,
        ids: &[u32],
        keep: usize,
        ctx: u32,
        label: &str,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let mut m = m;
        m.reset()?;
        let mut s = app::Session::from_model(m, ctx);
        let vocab = u32::try_from(s.model().body("scripted")?.vocab())?;
        let mut spec = s.with_draft::<Scripted, 4>(
            Scripted {
                plain: &plain.tokens,
                keep,
                vocab,
                at: 0,
            },
            &mut Quiet,
        )?;
        let first = spec.prompt(&mut s, ids)?;
        let out = runtime::generate(
            &mut s,
            &mut spec,
            ids,
            first,
            &runtime::Stop::new(keep + 1, ctx)?,
            &mut QuietSink,
        )?;
        let mut m = s.into_model();
        let at = m.pos() as usize;
        let want = &plain.tokens[..=keep];
        let tokens_ok = out.tokens == want;
        let (gpu, _, b) = m.body_parts("scripted")?;
        let stores_ok = live_same(&b.stores_host(gpu)?, &plain.stores[keep - 1], at);
        let ok = tokens_ok && stores_ok;
        println!(
            "(w) {label}: a window that keeps {keep} -> {} tokens = the plain run's {}, the \
             live stores after the commit its at position {at} {}",
            out.tokens.len(),
            verdict(tokens_ok),
            verdict(stores_ok)
        );
        Ok((m, ok))
    }

    /// (s) and (w)'s first clause: the draft's walks over a prompt call are
    /// the walks the module doc of `app::arch::qwen3moe` names. From a reset,
    /// the prompt by passes with every unit's hidden rows read to the host,
    /// then the draft walked from the host as the prompt call walks it —
    /// position 0 with a zero hidden row, each unit's rows from its second
    /// position on (the last unit's last excepted) each beside the hidden row
    /// of the position before it — eagerly in runs of eight (every launch,
    /// the head's included), then over the same rows as store walks
    /// (`WalkMode::Store`) in runs of [`HOST_STORE_RUN`]: (s) each position's
    /// keys and values the store walks leave within [`key_band`] and
    /// [`value_band`] of the eager walks'. Then one anchor walk at the
    /// prompt's end; against the drafted session's prompt call by the same
    /// passes, whose warmup walks the store alone: the store's keys and values
    /// at every prompt position bit for bit the host's store walks', then the
    /// same anchor walk's token, streams and logits bit for bit — the anchor's
    /// attention reads every key the walks before it wrote. Between the two,
    /// eager walks of other ids over the same positions overwrite every
    /// stored row, so a position the prompt call does not write reads as
    /// another sequence's, not as the host run's.
    fn prompt_walks(
        m: Qwen38Model,
        prompt: &[u32],
        path: Prompt38,
        ctx: u32,
    ) -> Result<(Qwen38Model, bool), GateError> {
        let n = prompt.len();
        let mut m = m;
        m.reset()?;
        let mut streams = vec![0.0f32; n * WIDE];
        let mut units: Vec<(usize, usize)> = Vec::new();
        {
            let mut sink = |mm: &mut Qwen38Model,
                            walk: TargetRows,
                            first: u32,
                            rows: usize|
             -> Result<(), GpuError> {
                let f = first as usize;
                let v = mm.target_streams(walk, rows)?;
                streams[f * WIDE..(f + rows) * WIDE].copy_from_slice(&v);
                units.push((f, rows));
                Ok(())
            };
            m.prompt38_with(prompt, path, Some(&mut sink))?;
        }
        let (h, e) = (DraftHead::Rows, WalkMode::Eager);
        let zeros = vec![0.0f32; WIDE];
        // Position 0 beside a zero row, then each unit's rows from its second
        // position on, each beside the hidden row of the position before it,
        // in walks of `mode` of `run` rows.
        let walk_host =
            |m: &mut Qwen38Model, mode: WalkMode, run: usize| -> Result<(), GateError> {
                m.mtp_walk(
                    DraftFeed::Rows(Feed {
                        tokens: &prompt[..1],
                        pos0: 0,
                        hidden: Hidden::Host(&zeros),
                        media: None,
                    }),
                    h,
                    mode,
                )?;
                for &(f, rows) in &units {
                    let end = (f + rows + 1).min(n);
                    for (i, r) in prompt[f + 1..end].chunks(run).enumerate() {
                        let q = f + 1 + i * run;
                        m.mtp_walk(
                            DraftFeed::Rows(Feed {
                                tokens: r,
                                pos0: u32::try_from(q)?,
                                hidden: Hidden::Host(
                                    &streams[(q - 1) * WIDE..(q - 1 + r.len()) * WIDE],
                                ),
                                media: None,
                            }),
                            h,
                            mode,
                        )?;
                    }
                }
                Ok(())
            };
        // Other ids over the same positions, from position 0, eagerly: every
        // row an earlier walk stored is overwritten.
        let vocab = u32::try_from(m.body("prompt_walks")?.vocab())?;
        let other: Vec<u32> = prompt.iter().map(|&t| (t + 1) % vocab).collect();
        let overwrite = |m: &mut Qwen38Model| -> Result<(), GateError> {
            m.mtp_walk(
                DraftFeed::Rows(Feed {
                    tokens: &other[..1],
                    pos0: 0,
                    hidden: Hidden::Host(&zeros),
                    media: None,
                }),
                h,
                e,
            )?;
            for (i, run) in other[1..].chunks(MTP_ROWS).enumerate() {
                let q = 1 + i * MTP_ROWS;
                m.mtp_walk(
                    DraftFeed::Rows(Feed {
                        tokens: run,
                        pos0: u32::try_from(q)?,
                        hidden: Hidden::Host(&streams[(q - 1) * WIDE..(q - 1 + run.len()) * WIDE]),
                        media: None,
                    }),
                    h,
                    e,
                )?;
            }
            Ok(())
        };
        walk_host(&mut m, e, MTP_ROWS)?;
        let eager_store = store_of(&m, n)?;
        overwrite(&mut m)?;
        walk_host(&mut m, WalkMode::Store, HOST_STORE_RUN)?;
        let host_store = store_of(&m, n)?;
        let (dk, dv) = store_rel(&host_store, &eager_store, n);
        let (bk, bv) = (key_band(), value_band());
        let band_ok = dk <= bk && dv <= bv;
        println!(
            "(s) the store walks ({path:?}, runs of {HOST_STORE_RUN}) over a store other ids \
             overwrote, against the eager walks over the same {n} positions: keys within \
             {dk:.3e} (band {bk:.3e}), values within {dv:.3e} (band {bv:.3e}) {}",
            verdict(band_ok)
        );
        let anchor = |m: &mut Qwen38Model| {
            walk(
                m,
                DraftFeed::Rows(Feed {
                    tokens: &prompt[..1],
                    pos0: u32::try_from(n).expect("a prompt fits u32"),
                    hidden: Hidden::Host(&streams[(n - 1) * WIDE..]),
                    media: None,
                }),
                h,
                e,
            )
        };
        let host = anchor(&mut m)?;
        overwrite(&mut m)?;
        let overwritten = same_rows(&host_store, &store_of(&m, n)?, n)
            .iter()
            .filter(|&&same| !same)
            .count();
        let cfg = app::arch::qwen3moe::Q38Cfg {
            prompt: path,
            draft: bloomery_gpu::model::StepMode::Eager,
        };
        let (mut s, mut spec) = drafted_session(m, cfg, ctx)?;
        spec.prompt(&mut s, prompt)?;
        let mut m = s.into_model();
        let same_at = same_rows(&host_store, &store_of(&m, n)?, n);
        let stored = same_at.iter().filter(|&&same| same).count();
        let store_ok = overwritten == n && stored == n;
        let first_off = same_at.iter().position(|&same| !same);
        let call = anchor(&mut m)?;
        let same = same_bits(&host, &call);
        println!(
            "(w) the prompt call ({path:?}) walks the draft over {n} positions in {} units as the \
             store walks fed from the host: other ids overwrote {overwritten} of {n} stored rows, \
             then the call's warmup stored {} of {n} as the host's (first other at {first_off:?}) \
             {}; the anchor's token {} (host {}), streams and logits bit for bit {}",
            units.len(),
            stored,
            verdict(store_ok),
            call.draft.tokens[0],
            host.draft.tokens[0],
            verdict(same)
        );
        Ok((m, band_ok && store_ok && same))
    }

    /// The draft's stored keys and values at positions `0..n`, position-major.
    fn store_of(m: &Qwen38Model, n: usize) -> Result<(Vec<u16>, Vec<u16>), GateError> {
        let d = m.body("store")?.mtp().ok_or("the load opened no draft")?;
        Ok(d.store_host(m.gpu().stream(), n)?)
    }

    /// The largest relative distance over the positions below `n` of the
    /// keys, then of the values, in `a` from those in `b` (two [`store_of`]
    /// reads of `n` positions), a position's row at a time.
    fn store_rel(a: &(Vec<u16>, Vec<u16>), b: &(Vec<u16>, Vec<u16>), n: usize) -> (f64, f64) {
        let row = a.0.len() / n.max(1);
        let f = |v: &[u16]| -> Vec<f32> { v.iter().map(|&x| half_to_f32(x)).collect() };
        let worst = |x: &[u16], y: &[u16]| {
            (0..n)
                .map(|p| {
                    let r = p * row..(p + 1) * row;
                    rel(&f(&x[r.clone()]), &f(&y[r]))
                })
                .fold(0.0f64, f64::max)
        };
        (worst(&a.0, &b.0), worst(&a.1, &b.1))
    }

    /// Whether each position below `n` holds bit for bit the same keys and
    /// values in `a` and `b` (two [`store_of`] reads of `n` positions).
    fn same_rows(a: &(Vec<u16>, Vec<u16>), b: &(Vec<u16>, Vec<u16>), n: usize) -> Vec<bool> {
        let row = a.0.len() / n.max(1);
        (0..n)
            .map(|p| {
                let r = p * row..(p + 1) * row;
                a.0[r.clone()] == b.0[r.clone()] && a.1[r.clone()] == b.1[r]
            })
            .collect()
    }

    /// The drafted session's cfg, its prompt path `path`.
    fn step_cfg(path: Prompt38) -> app::arch::qwen3moe::Q38Cfg {
        app::arch::qwen3moe::Q38Cfg {
            prompt: path,
            draft: bloomery_gpu::model::StepMode::Graph,
        }
    }

    /// What one drafted prompt call leaves ((fr)): the draft's store at the
    /// call's positions, the target's next id, and on a call by ubatches the
    /// ubatches it ran and the units its last walk cut (0 and 0 by steps or
    /// passes).
    struct Stored {
        store: (Vec<u16>, Vec<u16>),
        next: u32,
        ubatches: usize,
        units: usize,
    }

    /// The drafted session's prompt call of `prompt` by `path` from a reset,
    /// its draft eager, the ubatch walks cut by `split` for the call and by
    /// the load's own cut again after it, failed or not: what it leaves.
    fn drafted_store(
        m: Qwen38Model,
        prompt: &[u32],
        path: Prompt38,
        split: SplitWalk,
    ) -> Result<(Qwen38Model, Stored), GateError> {
        let mut m = m;
        let ctx = m.body("(fr)")?.ctx() as u32;
        let own = m.body("(fr)")?.split_walk();
        m.body_parts("(fr)")?.2.set_split_walk(split)?;
        let cfg = app::arch::qwen3moe::Q38Cfg {
            prompt: path,
            draft: bloomery_gpu::model::StepMode::Eager,
        };
        let (mut s, mut spec) = drafted_session(m, cfg, ctx)?;
        let next = spec.prompt(&mut s, prompt);
        let mut m = s.into_model();
        m.body_parts("(fr)")?.2.set_split_walk(own)?;
        let next = next?;
        let b = m.body("(fr)")?;
        let (ubatches, units) = match b.resolve_prompt(path, 0, prompt.len()) {
            Prompt38::Gemm => (b.ubatch_cut(prompt.len()).count(), b.ubatch_units()),
            _ => (0, 0),
        };
        let store = store_of(&m, prompt.len())?;
        Ok((
            m,
            Stored {
                store,
                next,
                ubatches,
                units,
            },
        ))
    }

    /// `got`'s draft store against `want`'s at their `n` positions, in words
    /// for (fr)'s line: the same bits, or how many positions differ from the
    /// first; and whether they are the same.
    fn store_said(got: &Stored, want: &Stored, n: usize) -> (String, bool) {
        let same = same_rows(&got.store, &want.store, n);
        match same.iter().position(|&s| !s) {
            None => ("bit for bit".to_string(), true),
            Some(first) => (
                format!(
                    "{} of {n} positions other from {first}",
                    same.iter().filter(|&&s| !s).count()
                ),
                false,
            ),
        }
    }

    /// (fr) on the gate's load: the set's prompt called once with the draft,
    /// one ubatch walked as one unit (the split walk off), is the reference;
    /// the same call under the split walk (`SplitWalk::Every`: two units, the
    /// head's mix over the second alone) leaves the draft's store and the
    /// next id the reference's bit for bit, and so does the call by passes
    /// (the head on the last pass alone) the call by steps' (each row its own
    /// step's head; a pass's rows are its steps' bit for bit). The reference
    /// goes on to [`final_rows_ubatches`].
    fn final_rows(
        m: Qwen38Model,
        prompt: &[u32],
    ) -> Result<(Qwen38Model, Stored, bool), GateError> {
        let mut m = m;
        let completed = completed_rows(&mut m, prompt)?;
        let n = prompt.len();
        let (m, one) = drafted_store(m, prompt, Prompt38::Gemm, SplitWalk::Off)?;
        let (m, two) = drafted_store(m, prompt, Prompt38::Gemm, SplitWalk::Every)?;
        let (m, steps) = drafted_store(m, prompt, Prompt38::Step, SplitWalk::Off)?;
        let (m, passes) = drafted_store(m, prompt, Prompt38::Pass, SplitWalk::Off)?;
        let shape_ok = (one.ubatches, one.units, two.ubatches, two.units) == (1, 1, 1, 2)
            && n > Prompt38::PASS_ROWS;
        let (split_said, split_same) = store_said(&two, &one, n);
        let (pass_said, pass_same) = store_said(&passes, &steps, n);
        let split_ok = split_same && two.next == one.next;
        let pass_ok = pass_same && passes.next == steps.next;
        println!(
            "(fr) a prompt call of {n} ids with the draft: as one ubatch of {} unit and under \
             the split walk as {} units (want 1 and 2) {}; the split walk's draft store against \
             the one unit's {split_said}, its next id {} (the one unit's {}) {}; by {} passes \
             against by steps {pass_said}, the next id {} (by steps {}) {}",
            one.units,
            two.units,
            verdict(shape_ok),
            two.next,
            one.next,
            verdict(split_ok),
            n.div_ceil(Prompt38::PASS_ROWS),
            passes.next,
            steps.next,
            verdict(pass_ok)
        );
        Ok((m, one, completed && shape_ok && split_ok && pass_ok))
    }

    /// (fr) on a load of the gate's plan whose ubatch holds a third of the
    /// set's prompt: the call runs as several ubatches, the head on the last
    /// alone, each walked as one unit and then under the split walk as two,
    /// and leaves the draft's store and the next id of `one` (the gate's
    /// load's one unit, [`final_rows`]) bit for bit either way.
    fn final_rows_ubatches(
        inputs: &PlanInputs,
        mtp: &MtpInputs,
        prompt: &[u32],
        one: &Stored,
    ) -> Result<bool, GateError> {
        let n = prompt.len();
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(usize::try_from(CTX)?)?;
        let (file, draft) = (open_target()?, open_draft()?);
        let machine = machine(card()?, inputs.stack(), u64::try_from(ub)?);
        let extra = mtp.card_bytes(CTX)?;
        let plan = inputs.plan_mtp(
            &machine,
            (CTX, u64::try_from(ub)?),
            &plan_levers(extra)?,
            mtp,
        )?;
        let small = n.div_ceil(3);
        let m = Body38::open_placed_mtp(file, &plan, inputs, 0, cfg().host, small, &draft, mtp)?;
        let (m, cut) = drafted_store(m, prompt, Prompt38::Gemm, SplitWalk::Off)?;
        let (m, cut_two) = drafted_store(m, prompt, Prompt38::Gemm, SplitWalk::Every)?;
        drop(m);
        let ubatches = n.div_ceil(small);
        let shape_ok = ubatches > 1
            && (cut.ubatches, cut.units, cut_two.ubatches, cut_two.units)
                == (ubatches, 1, ubatches, 2);
        let (said, same) = store_said(&cut, one, n);
        let (two_said, two_same) = store_said(&cut_two, one, n);
        let ok_one = same && cut.next == one.next;
        let ok_two = two_same && cut_two.next == one.next;
        println!(
            "(fr) the call on a load of the same plan with ubatches of {small} rows: {} ubatches \
             of {} unit, and of {} units under the split walk (want {ubatches}, 1 and 2) {}; \
             the draft store against the gate's load's one unit {said}, the next id {} (want \
             {}) {}; under the split walk {two_said}, the next id {} {}",
            cut.ubatches,
            cut.units,
            cut_two.units,
            verdict(shape_ok),
            cut.next,
            one.next,
            verdict(ok_one),
            cut_two.next,
            verdict(ok_two)
        );
        Ok(shape_ok && ok_one && ok_two)
    }

    /// (w): the windows end to end, against the plain run — at the set's
    /// prompt and D3K's (depth 3,000 after it): the drafted greedy ids are
    /// the plain run's for 64 tokens, with a rejected row among the windows
    /// (else the rollback path never ran, and the clause is red) and the
    /// live stores after the run the plain run's at the same position;
    /// `commit(k)` is `k` steps for every k, a scripted draft keeping each
    /// exactly through the same verify and commit, at the deep prompt its
    /// second row completing a pool.
    fn windows(
        m: Qwen38Model,
        prompt: &[u32],
        deep: &[u32],
    ) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 64;
        let ctx = m.body("windows")?.ctx() as u32;
        let (m, auto_ok) = prompt_walks(m, prompt, Prompt38::Auto, ctx)?;
        let (m, pass_ok) = prompt_walks(m, prompt, Prompt38::Pass, ctx)?;
        let mut ok = pass_ok && auto_ok;

        // The plain reference: every step's live stores, four tokens past
        // the drafted loop's count (its last pass may keep rows past it).
        let (m, shallow) = plain_run(m, prompt, Prompt38::Auto, N + 4)?;
        let (mut m, out, kept, unkept) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, prompt)?;
            let mut k = Kept::default();
            let out = runtime::generate(
                &mut s,
                &mut spec,
                prompt,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            let unkept = spec.draft_mut().take_windows();
            (s.into_model(), out, k, unkept)
        };
        let ids_ok = out.tokens[..N] == shallow.tokens[..N];
        ok &= ids_ok;
        println!(
            "(w) depth {}: the drafted run's {N} ids = the plain run's {}",
            prompt.len(),
            verdict(ids_ok)
        );
        let rejections: usize = kept
            .rows
            .iter()
            .zip(&kept.proposed)
            .filter(|(k, p)| **p && **k < 4)
            .count();
        ok &= rejections > 0;
        println!(
            "(w) depth {}: {} windows kept {:?}, {rejections} rejected a row {}",
            prompt.len(),
            kept.rows.len(),
            kept.rows,
            verdict(rejections > 0)
        );
        let at = (kept.pos as usize - prompt.len()).min(N + 3);
        let (gpu, _, b) = m.body_parts("windows")?;
        let stores_ok = live_same(
            &b.stores_host(gpu)?,
            &shallow.stores[at - 1],
            kept.pos as usize,
        );
        ok &= stores_ok;
        println!(
            "(w) depth {}: the live stores after the drafted run = the plain run's at position \
             {} {}",
            prompt.len(),
            kept.pos,
            verdict(stores_ok)
        );

        let (model, kept_ok) = kept_windows(m, prompt, ctx, (&out.tokens, &kept), unkept.len())?;
        m = model;
        ok &= kept_ok;

        // commit(k) is k steps, every k, through the rule's own driving.
        for keep in 1..=4usize {
            let (model, k_ok) = scripted_window(m, &shallow, prompt, keep, ctx, "shallow")?;
            m = model;
            ok &= k_ok;
        }

        // The deep prompt: the drafted ids and stores, and a rejection whose
        // row completes a pool (the scripted window's second row, at position
        // 3,003, completes pool 750).
        let (m, deep_plain) = plain_run(m, deep, Prompt38::Auto, 4)?;
        let (mut m, out, kept) = {
            let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
            let first = spec.prompt(&mut s, deep)?;
            let mut k = Kept::default();
            let out = runtime::generate(
                &mut s,
                &mut spec,
                deep,
                first,
                &runtime::Stop::new(N, ctx)?,
                &mut k,
            )?;
            (s.into_model(), out, k)
        };
        let drafted = {
            let (gpu, _, b) = m.body_parts("windows")?;
            b.stores_host(gpu)?
        };
        // The plain run to the drafted run's end, its stores read there.
        let n_plain = kept.pos as usize - deep.len() + 1;
        let (m2, plain_tokens, plain_stores) = plain_to(m, deep, Prompt38::Auto, n_plain)?;
        let mut m = m2;
        let ids_ok = out.tokens[..N] == plain_tokens[..N];
        ok &= ids_ok;
        println!(
            "(w) depth {}: the drafted run's {N} ids = the plain run's {}",
            deep.len(),
            verdict(ids_ok)
        );
        let rejections: usize = kept
            .rows
            .iter()
            .zip(&kept.proposed)
            .filter(|(k, p)| **p && **k < 4)
            .count();
        ok &= rejections > 0;
        println!(
            "(w) depth {}: {} windows kept {:?}, {rejections} rejected a row {}",
            deep.len(),
            kept.rows.len(),
            kept.rows,
            verdict(rejections > 0)
        );
        let stores_ok = live_same(&drafted, &plain_stores, kept.pos as usize);
        ok &= stores_ok;
        println!(
            "(w) depth {}: the live stores after the drafted run = the plain run's at position \
             {} {}",
            deep.len(),
            kept.pos,
            verdict(stores_ok)
        );
        let (model, p_ok) = scripted_window(
            m,
            &deep_plain,
            deep,
            2,
            ctx,
            "deep, its rejected row completing a pool",
        )?;
        m = model;
        ok &= p_ok;
        m.reset()?;
        Ok((m, ok))
    }

    /// A clock the chooser reads that this clause moves: each pass's wall
    /// a fixed cost of the rows it ran, so the chooser's picks are the same
    /// every run.
    #[derive(Clone)]
    struct Ticks(std::rc::Rc<std::cell::Cell<std::time::Duration>>);

    impl runtime::width::Clock for Ticks {
        fn now(&mut self) -> std::time::Duration {
            self.0.get()
        }
    }

    /// The [`Ticks`] clock's plain step: a synthetic unit, the order of a
    /// real step so the clock's readings stay ordinary times.
    const STEP_UNIT: std::time::Duration = std::time::Duration::from_millis(40);

    /// Every pass's rows, kept rows and whether it verified a proposal, the
    /// clock moved by the pass's cost: `cost[r - 1]` of [`STEP_UNIT`] for `r`
    /// rows.
    struct Cut {
        clock: Ticks,
        cost: [f64; 4],
        rows: Vec<usize>,
        kept: Vec<usize>,
        proposed: Vec<bool>,
    }

    impl PassSink<app::Session<Body38>> for Cut {
        type Error = GateError;

        fn begin(&mut self, _: &app::Session<Body38>) -> Result<(), GateError> {
            Ok(())
        }

        fn pass(
            &mut self,
            _: &app::Session<Body38>,
            c: &Committed,
            _: &[u32],
            _: std::time::Duration,
        ) -> Result<(), GateError> {
            let wall = STEP_UNIT.mul_f64(self.cost[c.rows - 1]);
            self.clock.0.set(self.clock.0.get() + wall);
            self.rows.push(c.rows);
            self.kept.push(c.kept);
            self.proposed.push(c.proposed);
            Ok(())
        }
    }

    /// What [`choosing_run`] leaves: the model, the tokens, the passes, the
    /// draft's windows, the chooser's tally and its incumbent at the end.
    type Chose = (
        Qwen38Model,
        Vec<u32>,
        Cut,
        Vec<app::mtp::WindowDraft>,
        runtime::Tally,
        usize,
    );

    /// One greedy run of `n` tokens from a reset, the draft behind the
    /// chooser on a [`Ticks`] clock of `cost`, its windows kept.
    fn choosing_run(
        m: Qwen38Model,
        prompt: &[u32],
        ctx: u32,
        n: usize,
        cost: [f64; 4],
    ) -> Result<Chose, GateError> {
        let mut m = m;
        m.reset()?;
        let mut s = app::Session::from_model(m, ctx);
        let mut draft = app::mtp::MtpDraft::open(
            s.model(),
            Prompt38::Auto,
            bloomery_gpu::model::StepMode::Graph,
        )?;
        draft.keep_windows();
        let clock = Ticks(std::rc::Rc::default());
        let choosing = draft.choosing_by(
            runtime::width::Mode::Cost,
            runtime::width::Rule::DEFAULT,
            clock.clone(),
        )?;
        let mut spec = s.with_draft::<runtime::Choosing<app::mtp::MtpDraft<Body38>, Ticks>, 4>(
            choosing, &mut Quiet,
        )?;
        let first = spec.prompt(&mut s, prompt)?;
        let mut cut = Cut {
            clock,
            cost,
            rows: Vec::new(),
            kept: Vec::new(),
            proposed: Vec::new(),
        };
        let out = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(n, ctx)?,
            &mut cut,
        )?;
        let tally = spec.draft_mut().take_tally();
        let width = spec.draft().width();
        let windows = spec.draft_mut().draft_mut().take_windows();
        let mut m = s.into_model();
        m.reset()?;
        Ok((m, out.tokens, cut, windows, tally, width))
    }

    /// (c) the width chooser (`BLOOMERY_MTP_WIDTH=cost`, `runtime::width`):
    /// the draft behind it over the same model, its windows kept, on a
    /// [`Ticks`] clock. First the walls of the A6000's passes (2, 3, 4 rows
    /// at 1.383, 1.767, 2.15 steps, W(3)'s line [derived]): the tokens the
    /// plain run's whatever ran; every window's record the width its pass
    /// ran (rows − 1), the proposal at least that width and at most the
    /// draft's three ids, its kept ids the plain run's next tokens and its
    /// first rejected id not; the chooser's tally the passes' widths and kept
    /// rows; the warm-up's cuts below the draft's width ran, and some
    /// proposal stopped short of three ids (a chain draft walks the width its
    /// pass plans, not its own). Then walls on which no drafted width
    /// pays: the chooser closes, the draft's proposals run as shadows (never
    /// a window) and the plain passes are the draft's held steps — the tokens
    /// still the plain run's, the windows the verifies' alone.
    fn width_chooser(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 64;
        let ctx = m.body("width chooser")?.ctx() as u32;
        // A last window may keep up to three ids past the N-th.
        let (m, plain) = plain_run(m, prompt, Prompt38::Auto, N + 3)?;
        let base = prompt.len();
        let mut ok = true;
        let mut m = m;
        for (what, cost) in [
            ("paying", [1.0, 1.383, 1.767, 2.15]),
            ("losing", [1.0, 10.0, 10.0, 10.0]),
        ] {
            let (back, tokens, cut, windows, tally, width) = choosing_run(m, prompt, ctx, N, cost)?;
            m = back;
            let ids_ok = tokens[..N] == plain.tokens[..N];
            // Each verified pass's window record, in order.
            let verified: Vec<usize> = (0..cut.rows.len()).filter(|&i| cut.proposed[i]).collect();
            let mut windows_ok = verified.len() == windows.len();
            for (&i, w) in verified.iter().zip(&windows) {
                let at = w.pos as usize - base;
                let ids_kept = w.ids[..w.accepted]
                    .iter()
                    .zip(&plain.tokens[at + 1..])
                    .all(|(d, t)| d == t);
                let rejected = w.accepted == w.width
                    || plain.tokens.get(at + 1 + w.accepted) != Some(&w.ids[w.accepted]);
                let one = w.width + 1 == cut.rows[i]
                    && w.accepted + 1 == cut.kept[i]
                    // PIN(2026-10-23): chains now stop at the chooser's planned width, so a
                    // proposal holds the verified width up to three ids.
                    && (w.width..=3).contains(&w.ids.len())
                    && ids_kept
                    && rejected;
                if !one {
                    println!(
                        "(c) {what}: pass {i} ran {} rows and kept {}; its window at {} verified \
                         {} of {:?}, the target keeping {}",
                        cut.rows[i], cut.kept[i], w.pos, w.width, w.ids, w.accepted
                    );
                }
                windows_ok &= one;
            }
            let mut widths = vec![0u64; 4];
            let mut kept = vec![0u64; 4];
            for (i, &r) in cut.rows.iter().enumerate() {
                widths[r - 1] += 1;
                if cut.proposed[i] {
                    kept[cut.kept[i] - 1] += 1;
                }
            }
            let tally_ok = tally.widths == widths && tally.kept == kept;
            let shape_ok = match what {
                "paying" => {
                    widths[1] > 0
                        && widths[2] > 0
                        && widths[0] > 0
                        && windows.iter().any(|w| w.ids.len() < 3)
                }
                _ => width == 0 && widths[0] * 2 > widths.iter().sum::<u64>(),
            };
            println!(
                "(c) {what}: the {N} ids = the plain run's {}; {} windows, each its pass's width \
                 with the target's kept ids {}; the chooser's widths {:?} = the passes' {:?} {}; \
                 incumbent {width}, the cuts and the gate {}",
                verdict(ids_ok),
                windows.len(),
                verdict(windows_ok),
                tally.widths,
                widths,
                verdict(tally_ok),
                verdict(shape_ok)
            );
            ok &= ids_ok && windows_ok && tally_ok && shape_ok;
        }
        Ok((m, ok))
    }

    /// (w) prompt calls that continue the held sequence: after the drafted
    /// windows of `prompt`, three ids the model did not generate fed by the
    /// rule's prompt call, then windows. The draft joins at the call's start
    /// (the rows the last window left waiting walked, the call's first id at
    /// their last position) and drafts, and the ids are the plain run's —
    /// the same prompt, the drafted run's fed ids stepped, the same three ids
    /// by the same prompt path. Then a join the draft cannot make: restarted
    /// beside a target that holds the sequence, it skips the next call by
    /// name and proposes nothing, the ids still the plain run's.
    fn continued(m: Qwen38Model, prompt: &[u32]) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 8;
        let ctx = m.body("continued")?.ctx() as u32;
        let ext: Vec<u32> = prompt.iter().copied().skip(1).take(3).collect();
        let p0 = prompt.len();
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
        let first = spec.prompt(&mut s, prompt)?;
        let stop = runtime::Stop::new(N, ctx)?;
        let out1 = runtime::generate(&mut s, &mut spec, prompt, first, &stop, &mut QuietSink)?;
        let p1 = s.pos() as usize;
        let fed1 = out1.tokens[..p1 - p0].to_vec();
        let own = out1.tokens[p1 - p0];
        let first2 = spec.prompt(&mut s, &ext)?;
        let join = spec.draft_mut().take_joined();
        let mut k2 = Kept::default();
        let out2 = runtime::generate(&mut s, &mut spec, &ext, first2, &stop, &mut k2)?;
        let p2 = s.pos() as usize;
        let fed2 = out2.tokens[..p2 - p1 - ext.len()].to_vec();
        spec.draft_mut().restart();
        let first3 = spec.prompt(&mut s, &ext)?;
        let skip = spec.draft_mut().take_joined();
        let mut k3 = Kept::default();
        let out3 = runtime::generate(&mut s, &mut spec, &ext, first3, &stop, &mut k3)?;
        let mut m = s.into_model();

        // The plain run: the same prompt calls, the drafted run's fed ids
        // stepped between them.
        m.reset()?;
        let plain = |m: &mut Qwen38Model, ids: &[u32]| -> Result<Vec<u32>, GateError> {
            let mut t = vec![m.prompt38(ids, Prompt38::Auto)?];
            slots_gate::greedy(m, &mut t, N - 1)?;
            Ok(t)
        };
        let _ = m.prompt38(prompt, Prompt38::Auto)?;
        for &x in &fed1 {
            m.step(&[x])?;
        }
        let plain2 = plain(&mut m, &ext)?;
        m.reset()?;
        let _ = m.prompt38(prompt, Prompt38::Auto)?;
        for &x in fed1.iter().chain(&ext).chain(&fed2) {
            m.step(&[x])?;
        }
        let plain3 = plain(&mut m, &ext)?;
        m.reset()?;

        let joined_ok =
            join.is_some_and(|j| j.start as usize == p1 && j.caught_up >= 1 && j.skipped.is_none());
        let drafted_ok = k2.proposed.iter().any(|&p| p);
        let ids2_ok = out2.tokens[..N] == plain2[..];
        let skip_ok =
            skip.is_some_and(|j| j.start as usize == p2 && j.caught_up == 0 && j.skipped.is_some());
        let none_ok = !k3.proposed.iter().any(|&p| p);
        let ids3_ok = out3.tokens[..N] == plain3[..];
        println!(
            "(w) a prompt call of {ext:?} continuing at {p1} (the model's own next id {own}): \
             joined {join:?} {}, windows kept {:?} drafted {}, its {N} ids = the plain run's {}",
            verdict(joined_ok),
            k2.rows,
            verdict(drafted_ok),
            verdict(ids2_ok)
        );
        println!(
            "(w) the draft restarted, a prompt call continuing at {p2}: joined {skip:?} {}, \
             {} passes proposed nothing {}, its {N} ids = the plain run's {}",
            verdict(skip_ok),
            k3.rows.len(),
            verdict(none_ok),
            verdict(ids3_ok)
        );
        let ok = joined_ok && drafted_ok && ids2_ok && skip_ok && none_ok && ids3_ok;
        Ok((m, ok))
    }

    /// A Q8_0 weight's two planes' device addresses.
    /// The addresses a borrowed matrix's form holds: a Q8_0 weight's two
    /// planes, a Q6_K one's word plane and a zero.
    fn planes(dw: Option<&DevWeight>) -> Option<[u64; 2]> {
        match dw {
            Some(DevWeight::Q8_0 { qs, d, .. }) => {
                Some([qs.buf().cu_deviceptr(), d.buf().cu_deviceptr()])
            }
            Some(DevWeight::KQuant {
                ty: GgmlType::Q6_K,
                w,
                ..
            }) => Some([w.buf().cu_deviceptr(), 0]),
            _ => None,
        }
    }

    /// (d) the head's map holds one word a vocabulary id, its first `ids`
    /// and the rest 0: the head reads the list's rows of the target's
    /// `output` through it, in place.
    fn head_rows(m: &Qwen38Model, mtp: &Mtp38, ids: &[u32]) -> Result<bool, GateError> {
        let stream = m.gpu().stream();
        let Some((map, n)) = mtp.head_map() else {
            println!("(d) the draft has no head map FAIL");
            return Ok(false);
        };
        let map = map.to_host_vec(stream)?;
        let vocab = m.weights().get("output.weight").map_or(0, |dw| match dw {
            DevWeight::Q8_0 { d, .. } => d.rows(),
            _ => 0,
        });
        let ok = n == ids.len()
            && map.len() == vocab
            && map.get(..n) == Some(ids)
            && map[n..].iter().all(|&w| w == 0);
        println!(
            "(d) head map: {} words (output's {vocab} rows), {n} rows, the list's ids then 0 {} \
             (want {LIST_ROWS})",
            map.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// Both mixed router pairs through the MTP classifier and the placed
    /// uploader, with distinct values in the router and the shared gate.
    fn mixed_router_load(
        m: &Qwen38Model,
        draft: &Split,
        mtp: &MtpInputs,
    ) -> Result<bool, GateError> {
        use bloomery_gpu::weights::Weights;
        use model::arch::qwen35moe::{names, place::FileTensor};
        use model::placement;

        let index = mtp.draft.index as usize;
        let names = [names::ffn_gate_inp(index), names::ffn_gate_inp_shexp(index)];
        let path =
            std::env::temp_dir().join(format!("qwen38-mtp-router-{}.gguf", std::process::id()));
        let result = (|| -> Result<bool, GateError> {
            let mut ok = true;
            for types in [
                [GgmlType::BF16, GgmlType::F32],
                [GgmlType::F32, GgmlType::BF16],
            ] {
                let mut file: Vec<FileTensor> = draft
                    .iter_tensors()
                    .map(|(shard, t)| FileTensor {
                        name: t.name.clone(),
                        shard,
                        ty: t.ty,
                        dims: t.dims.clone(),
                        nbytes: t.nbytes,
                    })
                    .collect();
                let mut decls = Vec::new();
                for ((name, ty), value) in names.iter().zip(types).zip([0.125f32, -0.25]) {
                    let t = file
                        .iter_mut()
                        .find(|t| t.name == *name)
                        .ok_or("no router part")?;
                    let bytes = match ty {
                        GgmlType::BF16 => ((value.to_bits() >> 16) as u16).to_le_bytes().to_vec(),
                        _ => value.to_le_bytes().to_vec(),
                    };
                    let data = bytes.repeat(usize::try_from(t.dims.iter().product::<u64>())?);
                    t.ty = ty;
                    t.shard = 0;
                    t.nbytes = data.len() as u64;
                    decls.push((name.clone(), t.dims.clone(), ty, data));
                }
                bloomery_gpu_gates::write_gguf(&path, &[], &decls)?;
                let split = Split::open(&path)?;
                let inputs = MtpInputs::from_parts(mtp.draft.clone(), &file)?;
                let rows = inputs
                    .model
                    .tensors
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| names.contains(&t.name))
                    .map(|(i, t)| placement::whole_on_card(i, t, 0, &inputs.model))
                    .collect::<Result<Vec<_>, _>>()?;
                let counted: u64 = rows
                    .iter()
                    .flat_map(|r| &r.segments)
                    .map(|s| s.resident_bytes)
                    .sum();
                let mut weights =
                    Weights::load_rows(m.gpu().stream(), &split, &inputs.model, &rows, 0)?;
                weights.join_rows(
                    m.gpu().stream(),
                    &[&names[0], &names[1]],
                    "derived.mtp.router".into(),
                )?;
                let joined = weights.f32_tensor("mixed MTP router", "derived.mtp.router")?;
                let hidden = mtp.draft.hidden as usize;
                let values = joined.buf().to_host_vec(m.gpu().stream())?;
                let router_values = usize::try_from(decls[0].1.iter().product::<u64>())?;
                let want_values = router_values + hidden;
                let pass = rows.len() == 2
                    && counted == (want_values * 4) as u64
                    && joined.rows() == want_values / hidden
                    && joined.cols() == hidden
                    && values.len() == want_values
                    && values[..router_values].iter().all(|&v| v == 0.125)
                    && values[router_values..].iter().all(|&v| v == -0.25);
                println!(
                    "(d) mixed MTP router={:?} shared={:?} joined F32 bytes={counted} {}",
                    types[0],
                    types[1],
                    verdict(pass)
                );
                ok &= pass;
            }
            Ok(ok)
        })();
        let cleanup = std::fs::remove_file(&path);
        let ok = result?;
        cleanup?;
        Ok(ok)
    }

    /// (d) and (q), on the model as the load left it: no walk has run.
    fn draft_load(
        m: &Qwen38Model,
        plan: MtpPlan<'_>,
        inputs: &PlanInputs,
        draft: &Split,
        mtp: &MtpInputs,
        ids: &[u32],
    ) -> Result<bool, GateError> {
        let target = open_target()?;
        let d = &plan.draft.cards[0];
        let body = m.body("mtp load")?;
        let Some(mtp38) = body.mtp() else {
            return Err("the load opened no MTP draft".into());
        };
        let want = plan.draft_resident_bytes() + d.kv_bytes + plan.map_bytes;
        let got = mtp38.resident_bytes() as u64;
        let mut ok = got == want;
        println!(
            "(d) draft resident {got} (want {want}: weights {} + store {} + map {}) {}",
            plan.draft_resident_bytes(),
            d.kv_bytes,
            plan.map_bytes,
            verdict(got == want)
        );
        for b in mtp38.borrowed_planes() {
            let target_planes = planes(m.weights().get(&b.name));
            let same = target_planes == Some(b.planes);
            ok &= same;
            println!(
                "(d) {} read at {:x?}, the target's at {target_planes:x?} {}",
                b.name,
                b.planes,
                verdict(same)
            );
        }
        let arena = mtp38.arena_bytes() as u64;
        let arena_ok = arena == plan.arena_bytes;
        ok &= arena_ok;
        println!(
            "(d) the draft program's arena {arena} (the plan's {}) {}",
            plan.arena_bytes,
            verdict(arena_ok)
        );
        let borrowed = mtp38.borrowed(m.weights()).is_ok();
        ok &= borrowed;
        println!("(d) the walk's borrow resolves: {}", verdict(borrowed));
        let rows_ok = head_rows(m, mtp38, ids)?;
        ok &= rows_ok;
        let mut own = MtpInputs::read(draft, &target, inputs, HeadSet::Full)?;
        if let MtpSource::File { borrows, .. } = &mut own.draft.source {
            *borrows = Borrows {
                embedding: false,
                head: false,
            };
        }
        let r1 = Mtp38::open(m.gpu(), m.weights(), (draft, None), &own, &plan, false, 1)
            .err()
            .map_or("opened".to_string(), |e| e.to_string());
        let r1_ok = r1.contains("the load borrows the target's token_embd and output");
        ok &= r1_ok;
        println!("(q) a draft with its own matrices: {r1} {}", verdict(r1_ok));
        let mut other = plan;
        other.map_bytes += 4;
        let r2 = Mtp38::open(m.gpu(), m.weights(), (draft, None), mtp, &other, false, 1)
            .err()
            .map_or("opened".to_string(), |e| e.to_string());
        let r2_ok = r2.contains("resident weights, store and row map hold");
        ok &= r2_ok;
        println!("(q) a plan of another row map: {r2} {}", verdict(r2_ok));
        ok &= mixed_router_load(m, draft, mtp)?;
        Ok(ok)
    }

    /// (u) the head a drafted load of this target picks with
    /// `BLOOMERY_MTP_HEAD_ROWS` unset is the shipped list: its rows, its
    /// first line naming the target tokenizer's digest, the pick's reason
    /// `shipped`; `full` set is the full head.
    fn shipped_head(file: &Split, vocab: u32) -> Result<bool, GateError> {
        let want = vocab_sha256(file)?;
        let first = SHIPPED.text.lines().next().unwrap_or_default();
        let line_ok = first.contains(&format!(" rows={SHIPPED_ROWS} "))
            && first.ends_with(&format!(" vocab_sha256={}", hex(&want)));
        let pick = head_rows_of(None, file, vocab)?;
        let rows_ok = matches!(&pick.rows, HeadSet::List { ids, digest }
            if ids.len() == SHIPPED_ROWS && *digest == want);
        let unset_ok = line_ok && rows_ok && pick.why == HeadWhy::Shipped;
        println!(
            "(u) unset: head {} of {} rows from {} ({}); the list's first line {first:?}, the \
             target's tokenizer {} {}",
            pick.head_word(),
            pick.rows_of(vocab),
            pick.why.from_word(),
            pick.why,
            hex(&want),
            verdict(unset_ok)
        );
        let full = head_rows_of(Some(bloomery_levers::MtpHead::Full), file, vocab)?;
        let full_ok = full.rows == HeadSet::Full && full.why == HeadWhy::SetFull;
        println!(
            "(u) full: head {} of {} rows from {} ({}) {}",
            full.head_word(),
            full.rows_of(vocab),
            full.why.from_word(),
            full.why,
            verdict(full_ok)
        );
        Ok(unset_ok && full_ok)
    }

    // ------------------------------------------------------ (y) two slots

    /// The drafted session's speculative over this body.
    type Spec38 = runtime::Speculative<app::mtp::MtpDraft<Body38>, 4>;

    /// The server's first plain step after a request's prompt call
    /// (`serve_seats::drafted`'s step): the rows the last call left waiting
    /// walked, the step read, then the step told to the draft.
    fn plain_step(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        last: u32,
    ) -> Result<u32, GateError> {
        spec.draft_mut().before_step(s, last)?;
        let next = Target::step(s, last, Want::Argmax)?.argmax();
        Draft::stepped(spec.draft_mut(), s, last, next)?;
        Ok(next)
    }

    /// Make `to` the slot every later call acts on, exactly as a seat
    /// switches: the live slot's draft state parked before the select away,
    /// the target's own put back after the select back, the parked states
    /// kept one a slot.
    fn switch(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        parked: &mut [Option<app::mtp::Parked<TargetRows>>; 2],
        to: usize,
    ) -> Result<(), GateError> {
        let from = s.model().selected();
        let live = spec.draft().park();
        s.model_mut().select_slot(to)?;
        if let Some(back) = parked[to].take() {
            spec.draft_mut().unpark(&back);
        }
        parked[from] = Some(live);
        Ok(())
    }

    /// One drafted request's start as the server runs it (`worker.rs`'s
    /// `start` → `genloop`'s `prompt`): the prompt call, then its first
    /// plain step, together — no select between them. Returns the ids so
    /// far (the prompt's argmax, then the first step's).
    fn begin_request(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        ids: &[u32],
    ) -> Result<Vec<u32>, GateError> {
        let first = Advance::prompt(spec, s, ids)?;
        Ok(vec![first, plain_step(s, spec, first)?])
    }

    /// One drafted request as the server runs it when nothing switches
    /// beneath it (the alone run): the start of [`begin_request`], then
    /// `rounds` drafted passes — every id the passes kept, and every pass's
    /// proposal and kept rows.
    fn drive(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        ids: &[u32],
        rounds: usize,
    ) -> Result<(Vec<u32>, Vec<Committed>), GateError> {
        let mut out = begin_request(s, spec, ids)?;
        let mut passes = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            passes.push(one_pass(s, spec, &mut out)?);
        }
        Ok((out, passes))
    }

    /// One drafted pass from `out`'s last id, its kept ids appended.
    fn one_pass<D: Draft<app::Session<Body38>>>(
        s: &mut app::Session<Body38>,
        spec: &mut runtime::Speculative<D, 4>,
        out: &mut Vec<u32>,
    ) -> Result<Committed, GateError> {
        let last = *out.last().ok_or("no token")?;
        let mut kept = Vec::new();
        let c = Advance::pass(spec, s, last, &mut kept)?;
        out.append(&mut kept);
        Ok(c)
    }

    /// The selected slot's draft store, read back at the count its own walks
    /// hold: the rows the sequence's walks wrote.
    fn slot_store(s: &app::Session<Body38>) -> Result<(Vec<u16>, Vec<u16>), GateError> {
        let m = s.model();
        let n = m
            .body("slots")?
            .mtp()
            .ok_or("the load opened no draft")?
            .held();
        store_of(m, n)
    }

    /// (y) (module doc): two drafted sequences over one plan of two, in the
    /// server's order, against each window's single-sequence reference.
    fn slots_windows(
        inputs: &PlanInputs,
        mtp: &MtpInputs,
        ub: usize,
        a: &[u32],
        b: &[u32],
    ) -> Result<bool, GateError> {
        const ROUNDS: usize = 8;
        let ctx = u32::try_from(CTX)?;
        let (file, draft_file) = (open_target()?, open_draft()?);
        let machine = machine(card()?, inputs.stack(), u64::try_from(ub)?);
        let plan = inputs.plan_mtp_with_slots(
            &machine,
            (CTX, u64::try_from(ub)?),
            &plan_levers(mtp.card_bytes_of(CTX, 2)?)?,
            mtp,
            Experts::Host,
            2,
        )?;
        let mut m = Body38::open_placed_mtp_slots(
            file,
            &plan,
            inputs,
            0,
            cfg().host,
            ub,
            &draft_file,
            mtp,
            2,
        )?;
        m.set_mode(bloomery_gpu::model::StepMode::Graph);
        let mut s = app::Session::from_model(m, ctx);
        let mut spec = s.with_draft::<app::mtp::MtpDraft<Body38>, 4>(
            app::mtp::MtpDraft::open(
                s.model(),
                Prompt38::Auto,
                bloomery_gpu::model::StepMode::Graph,
            )?,
            &mut Quiet,
        )?;
        // The references, each on the load's own single sequence before any
        // slot exists: what the server's one-request runs produce, so the
        // two-slot run is judged against runs the slots' exchange cannot
        // have shaped.
        let (a_ids, a_passes, a_store) = {
            s.reset()?;
            spec.draft_mut().restart();
            let run = drive(&mut s, &mut spec, a, ROUNDS)?;
            (run.0, run.1, slot_store(&s)?)
        };
        let (b_ids, b_passes, b_store) = {
            s.reset()?;
            spec.draft_mut().restart();
            let run = drive(&mut s, &mut spec, b, ROUNDS)?;
            (run.0, run.1, slot_store(&s)?)
        };
        // (z)'s references, as alone as (y)'s.
        let mut capped = Vec::with_capacity(DEPTHS.len());
        for depth in DEPTHS {
            capped.push([
                capped_run(&mut s, a, &[Some(depth); ROUNDS])?,
                capped_run(&mut s, b, &[Some(depth); ROUNDS])?,
            ]);
        }
        let after = [after_alone(&mut s, 0, a)?, after_alone(&mut s, 1, b)?];
        let plain = [0, 1].map(|half| PLAIN_ROUNDS.map(|pass| half_of(pass, half)));
        let plain = [
            capped_run(&mut s, a, &plain[0])?,
            capped_run(&mut s, b, &plain[1])?,
        ];
        let own_ok = own_row(s.model_mut(), a)?;
        s.model_mut().add_slots(2)?;
        // Together, in the server's order: slot 0's prompt call and first
        // plain step run together, then slot 1's, then the drafted passes
        // interleave — a select between every pass. Each slot's request
        // starts from its own reset.
        let mut parked = [None, None];
        s.reset()?;
        spec.draft_mut().restart();
        let mut t_a = begin_request(&mut s, &mut spec, a)?;
        switch(&mut s, &mut spec, &mut parked, 1)?;
        s.reset()?;
        spec.draft_mut().restart();
        let mut t_b = begin_request(&mut s, &mut spec, b)?;
        let (mut p_a, mut p_b) = (Vec::new(), Vec::new());
        for _ in 0..ROUNDS {
            switch(&mut s, &mut spec, &mut parked, 0)?;
            p_a.push(one_pass(&mut s, &mut spec, &mut t_a)?);
            switch(&mut s, &mut spec, &mut parked, 1)?;
            p_b.push(one_pass(&mut s, &mut spec, &mut t_b)?);
        }
        let ids_ok = t_a == a_ids && t_b == b_ids;
        let passes_ok = p_a == a_passes && p_b == b_passes;
        // Each slot's draft store, read back at its own held count: the rows
        // its walks wrote, bit for bit its single-sequence reference's — the
        // ids and the passes' kept counts alone cannot see a store another
        // slot's walks filled (a proposal the target refuses every row of
        // leaves them as it found them).
        let stores_ok = slot_store(&s)? == b_store && {
            switch(&mut s, &mut spec, &mut parked, 0)?;
            slot_store(&s)? == a_store
        };
        println!(
            "(y) two slots in the server's order: {} drafted passes a slot, a select between \
             every pass; each slot's ids {} its single-sequence reference's, its proposals and \
             acceptances {}, its draft store {} {}",
            ROUNDS,
            verdict(ids_ok),
            verdict(passes_ok),
            verdict(stores_ok),
            verdict(ids_ok && passes_ok && stores_ok)
        );
        let mut z_ok = true;
        // The equality of each slot's run with its window alone, and a pass that keeps fewer
        // rows than it ran, are self-consistency; that the two slots keep different counts in
        // one pass is a property of the draft's weights against the target's, which the
        // shipped draft has and a fixture's generated one does not (every proposal is
        // refused there).
        let uneven = tier::run_clause(
            "(z) at each depth a pass keeps different counts on the two slots",
            Tag::FileBound,
        )?;
        for (depth, refs) in DEPTHS.into_iter().zip(&capped) {
            z_ok &= z_red(
                &format!("depth {depth}"),
                one_pass_windows(&mut s, [a, b], [depth; 2], ROUNDS, refs.each_ref(), uneven),
            );
        }
        // Unequal depths, in both orders: one slot's chain ends while the
        // other's own walks go on, so a walk past the first takes fewer
        // blocks than the round staged.
        for (x, y) in [(0, 1), (1, 0)] {
            let depths = [DEPTHS[x], DEPTHS[y]];
            let refs = [&capped[x][0], &capped[y][1]];
            z_ok &= z_red(
                &format!("depths {depths:?}"),
                one_pass_windows(&mut s, [a, b], depths, ROUNDS, refs, uneven),
            );
        }
        z_ok &= z_red("waiting", waiting_select(&mut s));
        z_ok &= z_red("step after", step_after(&mut s, [a, b], &after));
        z_ok &= z_red("plain rounds", plain_rounds(&mut s, [a, b], &plain, uneven));
        z_ok &= z_red("prompt anchor", prompt_anchor(&mut s, [a, b]));
        tier::sc(
            "(kp-draft) two slots cut to points at one position: each rejoins with its own row",
        )?;
        let mut kp_ok = slots_rejoin(&mut s, &mut spec, [a, b])?;
        tier::sc("(kp-draft) the point at a continuing call's start, after a step and a select")?;
        kp_ok &= slots_call_start(&mut s, [a, b])?;
        tier::sc("(kp-draft) a state put back as its prefix on the other slot")?;
        kp_ok &= slots_put_back(&mut s, [a, b])?;
        tier::sc("(kp-draft) an off slot beside a drafted window in one pass")?;
        kp_ok &= slots_off_beside(&mut s, [a, b])?;
        Ok(own_ok && ids_ok && passes_ok && stores_ok && z_ok && kp_ok)
    }

    /// (kp-draft) two slots cut to points at one position: each slot's drafted request with
    /// checkpoints on (its prompt call of the same length, so both points stand at its end,
    /// then its first plain step and one drafted pass), a select between them, then each slot
    /// selected and cut back to its point by `rollback`: each stands there with its own draft's
    /// store holding the positions below the point and the step arena's row 0 its own prompt's
    /// hidden row before the point bit for bit, not the other slot's. Mutant: the select that
    /// leaves the points' rows with the load (a slot rejoins with the other slot's row).
    fn slots_rejoin(
        s: &mut app::Session<Body38>,
        spec: &mut Spec38,
        windows: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let l = windows[0].len().min(windows[1].len());
        let mut parked = [None, None];
        s.model_mut()
            .body_parts("(kp-draft)")?
            .2
            .set_checkpoints(true);
        let mut rows = Vec::with_capacity(2);
        for (slot, ids) in windows.into_iter().enumerate() {
            switch(s, spec, &mut parked, slot)?;
            s.reset()?;
            spec.draft_mut().restart();
            let first = Advance::prompt(spec, s, &ids[..l])?;
            rows.push(ubatch_last(s.model(), l)?);
            let mut out = vec![first, plain_step(s, spec, first)?];
            one_pass(s, spec, &mut out)?;
        }
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let mut ok = bits(&rows[0]) != bits(&rows[1]);
        for slot in 0..2 {
            switch(s, spec, &mut parked, slot)?;
            let cut = s.model_mut().rollback(u32::try_from(l)?);
            let m = s.model();
            let (held, pos) = (held_of(m)?, m.pos());
            let row = bits(&m.target_streams(TargetRows::Step, 1)?);
            let (own, other) = (row == bits(&rows[slot]), row == bits(&rows[1 - slot]));
            let slot_ok = cut.is_ok() && held == l && pos as usize == l && own && !other;
            println!(
                "(kp-draft) slot {slot} cut back to its point at {l} ({}): the draft's store \
                 holds {held} at {pos}, the step row its own prompt's {own}, the other slot's \
                 {other} {}",
                cut.map_or_else(|e| format!("refused: {e}"), |()| "ran".to_owned()),
                verdict(slot_ok)
            );
            ok &= slot_ok;
        }
        s.model_mut()
            .body_parts("(kp-draft)")?
            .2
            .set_checkpoints(false);
        Ok(ok)
    }

    /// The ids a (kp-draft) two-slot case feeds past a slot's first call, in each call that
    /// continues it.
    const KP_MORE: usize = 8;

    /// The selected slot's checkpoints turned on or off.
    fn set_marks(s: &mut app::Session<Body38>, on: bool) -> Result<(), GateError> {
        s.model_mut()
            .body_parts("(kp-draft)")?
            .2
            .set_checkpoints(on);
        Ok(())
    }

    /// (kp-draft) the point at a continuing call's start, each slot's request from its reset
    /// and its own draft, `l` half the shorter window. After a step: slot 0's prompt call of `l`
    /// ids and its first plain step, then a call of [`KP_MORE`] ids with checkpoints on, whose
    /// start mark at `l + 1` reads the row the step left in the step arena; cut back to that
    /// mark, the draft rejoins with that row bit for bit. After a select: slot 0's prompt call
    /// of `l` ids with checkpoints off and no step, slot 1's prompt call of `l` ids (the load's
    /// ubatch arena rewritten), slot 0 selected again and a call of [`KP_MORE`] ids with
    /// checkpoints on: no arena of slot 0's holds `l - 1` at the start mark, so the point at
    /// `l` keeps no row, and a cut back to it leaves the draft off, its store forgotten, never
    /// rejoined with slot 1's row. Mutant: the select that keeps the ubatch arena's record
    /// (slot 0 rejoins with slot 1's row).
    fn slots_call_start(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let l = windows[0].len().min(windows[1].len()) / 2;
        let more = &windows[0][l..l + KP_MORE];
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        s.select_slot(0)?;
        let mut d = fresh_draft(s)?;
        start(s, &mut d, &windows[0][..l])?;
        let row = s.model().target_streams(TargetRows::Step, 1)?;
        set_marks(s, true)?;
        s.model_mut().prompt38(more, Prompt38::Auto)?;
        let p = l + 1;
        let rows = store_of(s.model(), p)?;
        let cut = s.model_mut().rollback(u32::try_from(p)?);
        let after_step = AtPoint::read(s.model(), p)?;
        let step_ok = cut.is_ok() && after_step.rejoined(p, &rows, &row);
        println!(
            "(kp-draft) a call's start mark after a step, cut back to it at {p} ({}): the draft's \
             store holds {} at {}, the step row the step's {} {}",
            cut.map_or_else(|e| format!("refused: {e}"), |()| "ran".to_owned()),
            after_step.held,
            after_step.pos,
            bits(&after_step.row) == bits(&row),
            verdict(step_ok)
        );
        set_marks(s, false)?;
        let mut own = Vec::with_capacity(2);
        for (slot, ids) in windows.into_iter().enumerate() {
            s.select_slot(slot)?;
            s.reset()?;
            let mut d = fresh_draft(s)?;
            Draft::prompt(&mut d, s, &ids[..l])?;
            own.push(ubatch_last(s.model(), l)?);
        }
        s.select_slot(0)?;
        set_marks(s, true)?;
        s.model_mut().prompt38(more, Prompt38::Auto)?;
        let cut = s.model_mut().rollback(u32::try_from(l)?);
        let m = s.model();
        let (held, pos) = (held_of(m)?, m.pos());
        let row = bits(&m.target_streams(TargetRows::Step, 1)?);
        let (mine, other) = (row == bits(&own[0]), row == bits(&own[1]));
        let select_ok = bits(&own[0]) != bits(&own[1])
            && cut.is_ok()
            && pos as usize == l
            && held == 0
            && !other;
        println!(
            "(kp-draft) a call's start mark after a select, its position before held only by \
             the ubatch arena's record of before the select, cut back to it at {l} ({}): the \
             draft's store holds {held} at {pos}, the step row its own prompt's {mine}, the other \
             slot's {other} {}",
            cut.map_or_else(|e| format!("refused: {e}"), |()| "ran".to_owned()),
            verdict(select_ok)
        );
        set_marks(s, false)?;
        Ok(step_ok && select_ok)
    }

    /// (kp-draft) a state put back as its prefix on the other slot: slot 0's drafted request
    /// with checkpoints on (its prompt call of `l` ids, a point at its end, then its first
    /// plain step and one drafted pass) saved, then slot 1 reset and the state put back as the
    /// point's prefix: slot 1 stands at `l` with the draft rejoined ([`AtPoint::rejoined`]),
    /// the store's rows slot 0's and the step arena's row 0 slot 0's prompt's hidden row at
    /// `l - 1` bit for bit.
    fn slots_put_back(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let l = windows[0].len().min(windows[1].len()) / 2;
        s.select_slot(0)?;
        s.reset()?;
        set_marks(s, true)?;
        let mut d = fresh_draft(s)?;
        let first = Draft::prompt(&mut d, s, &windows[0][..l])?;
        let row = ubatch_last(s.model(), l)?;
        d.before_step(s, first)?;
        let next = Target::step(s, first, Want::Argmax)?.argmax();
        Draft::stepped(&mut d, s, first, next)?;
        let mut spec = Spec38::new(d);
        one_pass(s, &mut spec, &mut vec![first, next])?;
        let rows = store_of(s.model(), l)?;
        let state = s.model_mut().seq_save()?;
        let n = state.positions();
        set_marks(s, false)?;
        s.select_slot(1)?;
        s.reset()?;
        let put = s.model_mut().seq_resume_prefix(&state, n - 1);
        let at = AtPoint::read(s.model(), l)?;
        let ok = put.as_ref().is_ok_and(|&k| k as usize == l) && at.rejoined(l, &rows, &row);
        println!(
            "(kp-draft) slot 0's state of {n} positions put back on slot 1 as a prefix (returned \
             {}): the draft's store holds {} at {}, its rows slot 0's {}, the step row slot 0's \
             prompt's {} {}",
            put.map_or_else(|e| format!("refused: {e}"), |k| k.to_string()),
            at.held,
            at.pos,
            same_rows(&at.store, &rows, l).iter().all(|&x| x),
            at.row == row,
            verdict(ok)
        );
        s.reset()?;
        Ok(ok)
    }

    /// The passes of the off slot clause ([`slots_off_beside`]).
    const KP_PASSES: usize = 6;

    /// Slot 0 of the off slot clause from its reset, through `d`: the prompt call of `l` ids
    /// of `ids` with checkpoints on and its first plain step, two calls of [`KP_MORE`] ids
    /// without the draft, the second without checkpoints so a full ledger keeps the point the
    /// cut needs, a cut back to the first's end (the draft's store behind the point there, so
    /// the cut forgets it), then a continuing prompt call of [`KP_MORE`] ids, which
    /// the draft skips, and its first plain step. Returns the ids from that call on and the
    /// draft's store count after the cut.
    fn off_slot(
        s: &mut app::Session<Body38>,
        d: &mut app::mtp::MtpDraft<Body38>,
        ids: &[u32],
        l: usize,
    ) -> Result<(Vec<u32>, usize), GateError> {
        s.reset()?;
        set_marks(s, true)?;
        start(s, d, &ids[..l])?;
        let at = s.model().pos();
        s.model_mut()
            .prompt38(&ids[l..l + KP_MORE], Prompt38::Auto)?;
        set_marks(s, false)?;
        s.model_mut().prompt38(&ids[..KP_MORE], Prompt38::Auto)?;
        set_marks(s, true)?;
        s.model_mut().rollback(at + u32::try_from(KP_MORE)?)?;
        let held = held_of(s.model())?;
        let first = Draft::prompt(d, s, &ids[l + KP_MORE..l + 2 * KP_MORE])?;
        d.before_step(s, first)?;
        let next = Target::step(s, first, Want::Argmax)?.argmax();
        Draft::stepped(d, s, first, next)?;
        set_marks(s, false)?;
        Ok((vec![first, next], held))
    }

    /// (kp-draft) an off slot beside a drafted window in one pass, the server's off slot at the
    /// body: slot 0's draft off ([`off_slot`]: its store forgotten by a cut to a point it does
    /// not reach, its next call skipped), slot 1's drafted request from its reset, then
    /// [`KP_PASSES`] passes of both slots' rows ([`app::mtp::pass_slots`], depth
    /// [`app::mtp::SLOT_DEPTH`]): slot 0 rides every pass with one row, a plain step, and slot
    /// 1 proposes in at least one. Slot 0's ids are the same calls' with plain steps in place
    /// of the passes, and slot 1's ids, kept rows and draft store its window run alone
    /// ([`capped_run`]).
    fn slots_off_beside(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let l = windows[0].len().min(windows[1].len()) / 2;
        let depth = app::mtp::SLOT_DEPTH;
        let mut d0 = fresh_draft(s)?;
        let mut d1 = fresh_draft(s)?;
        s.select_slot(0)?;
        let (mut o0, held) = off_slot(s, &mut d0, windows[0], l)?;
        let skips = d0.skipping();
        s.select_slot(1)?;
        let mut o1 = start(s, &mut d1, windows[1])?;
        let (mut plain_rows, mut drafted_rows) = (true, 0usize);
        let mut passes = Vec::with_capacity(KP_PASSES);
        for _ in 0..KP_PASSES {
            let (l0, l1) = (
                *o0.last().ok_or("slot 0 gave no id")?,
                *o1.last().ok_or("slot 1 gave no id")?,
            );
            let mut w = [
                app::mtp::SlotWindow {
                    slot: 0,
                    draft: &mut d0,
                    last: l0,
                    depth,
                    out: &mut o0,
                    draw: None,
                },
                app::mtp::SlotWindow {
                    slot: 1,
                    draft: &mut d1,
                    last: l1,
                    depth,
                    out: &mut o1,
                    draw: None,
                },
            ];
            let (done, _) = app::mtp::pass_slots(s, &mut w)?;
            let [c0, c1] = <[Committed; 2]>::try_from(done)
                .map_err(|d| format!("a pass of 2 slots committed {} slots", d.len()))?;
            plain_rows &= c0.rows == 1 && c0.kept == 1;
            drafted_rows += usize::from(c1.rows > 1);
            passes.push(c1);
        }
        s.select_slot(1)?;
        let store1 = slot_store(s)?;
        s.select_slot(0)?;
        let mut d = fresh_draft(s)?;
        let (mut plain, _) = off_slot(s, &mut d, windows[0], l)?;
        for _ in 0..KP_PASSES {
            let last = *plain.last().ok_or("slot 0 gave no id")?;
            plain.push(Target::step(s, last, Want::Argmax)?.argmax());
        }
        s.select_slot(1)?;
        let alone = capped_run(s, windows[1], &[Some(depth); KP_PASSES])?;
        let off_same = o0 == plain;
        let drafted_same = o1 == alone.ids && passes == alone.passes && store1 == alone.store;
        let ok = held == 0 && skips && plain_rows && drafted_rows > 0 && off_same && drafted_same;
        println!(
            "(kp-draft) an off slot beside a drafted window, {KP_PASSES} passes of both slots: \
             slot 0's store {held} after its cut, its draft skipping {skips}, one plain row a \
             pass {plain_rows}, its ids its plain steps' {off_same}; slot 1 proposed in \
             {drafted_rows} passes, its ids, kept rows and draft store its window alone's \
             {drafted_same} {}",
            verdict(ok)
        );
        s.reset()?;
        Ok(ok)
    }

    // ------------------------------------------- (z) two slots in one pass

    /// A (z) arm's verdict: a call that failed inside it is its red, the
    /// error printed on its line.
    fn z_red(arm: &str, r: Result<bool, GateError>) -> bool {
        r.unwrap_or_else(|e| {
            println!("(z) {arm}: a call failed: {e} {}", verdict(false));
            false
        })
    }

    /// (z)'s proposal caps: the default shape's 2 rows a slot
    /// (`app::mtp::SLOT_DEPTH`), and 4, the widest pass of two slots.
    const DEPTHS: [usize; 2] = [app::mtp::SLOT_DEPTH, 3];

    /// The MTP draft with each proposal capped to `depth` ids, as a pass of
    /// several slots' caller caps each slot's window by the room it hands
    /// the draft: (z)'s reference, verified through the session's verify of
    /// one sequence.
    struct Capped {
        d: app::mtp::MtpDraft<Body38>,
        depth: usize,
    }

    impl Draft<app::Session<Body38>> for Capped {
        const WIDTH: usize = <app::mtp::MtpDraft<Body38> as Draft<app::Session<Body38>>>::WIDTH;
        const TAPS: TapNeed = TapNeed::Final;

        fn prompt(
            &mut self,
            t: &mut app::Session<Body38>,
            ids: &[u32],
        ) -> Result<u32, app::SessionError> {
            Draft::prompt(&mut self.d, t, ids)
        }

        fn begin(
            &mut self,
            t: &app::Session<Body38>,
            prompt: &[u32],
            first: u32,
        ) -> Result<(), app::SessionError> {
            Draft::begin(&mut self.d, t, prompt, first)
        }

        fn propose(
            &mut self,
            t: &mut app::Session<Body38>,
            last: u32,
            out: &mut [u32],
        ) -> Result<usize, app::SessionError> {
            let room = self.depth.min(out.len());
            Draft::propose(&mut self.d, t, last, &mut out[..room])
        }

        fn accept(
            &mut self,
            t: &mut app::Session<Body38>,
            rows: &[u32],
            out: &[u32],
            accepted: usize,
        ) -> Result<(), app::SessionError> {
            Draft::accept(&mut self.d, t, rows, out, accepted)
        }

        fn stepped(
            &mut self,
            t: &mut app::Session<Body38>,
            last: u32,
            next: u32,
        ) -> Result<(), app::SessionError> {
            Draft::stepped(&mut self.d, t, last, next)
        }

        fn held(
            &mut self,
            t: &mut app::Session<Body38>,
            last: u32,
            next: u32,
        ) -> Result<(), app::SessionError> {
            Draft::held(&mut self.d, t, last, next)
        }
    }

    /// One drafted window's run: its ids (the prompt's argmax, the first
    /// plain step's, then every pass's kept ids), every pass's kept rows,
    /// the windows its draft kept and its draft store.
    #[derive(PartialEq)]
    struct Drafted {
        ids: Vec<u32>,
        passes: Vec<Committed>,
        windows: Vec<app::mtp::WindowDraft>,
        store: (Vec<u16>, Vec<u16>),
    }

    /// A fresh draft of the drafted session's, keeping its windows.
    fn fresh_draft(s: &app::Session<Body38>) -> Result<app::mtp::MtpDraft<Body38>, GateError> {
        let mut d = app::mtp::MtpDraft::open(
            s.model(),
            Prompt38::Auto,
            bloomery_gpu::model::StepMode::Graph,
        )?;
        d.keep_windows();
        Ok(d)
    }

    /// A request's start on the selected slot from its reset, as
    /// [`begin_request`] runs it, through `d`: the prompt call, then the
    /// first plain step.
    fn start(
        s: &mut app::Session<Body38>,
        d: &mut app::mtp::MtpDraft<Body38>,
        ids: &[u32],
    ) -> Result<Vec<u32>, GateError> {
        s.reset()?;
        let first = Draft::prompt(d, s, ids)?;
        d.before_step(s, first)?;
        let next = Target::step(s, first, Want::Argmax)?.argmax();
        Draft::stepped(d, s, first, next)?;
        Ok(vec![first, next])
    }

    /// A plain pass of the width chooser's: a pass its `propose` returns no
    /// verify for, as `runtime::width`'s `Choosing` runs it.
    #[derive(Clone, Copy)]
    enum PlainPass {
        /// (i) a held pass right after the server's cut step (a request's
        /// prompt call, then its last id as the seat's first step,
        /// [`start`]): the draft's waiting rows walked first
        /// ([`Draft::before_plain`]), then the target's step, which
        /// overwrites the arena the anchor read only after the walk.
        Held,
        /// (v) a window no verify takes (the chooser's shadow, or the window
        /// its closing gate cuts to no id), after `begin` and after a
        /// drafted pass from the prompt's anchor: the draft's chain, its
        /// proposal dropped ([`Draft::unproposed`]), then the target's step,
        /// which walks no row of the prompt call's arena (`begin` took its
        /// anchor) and finds the chain's own rows taken back.
        Dropped,
    }

    /// The chooser's plain pass `plain`, then the draft told through
    /// [`Draft::held`]: told without a refusal, the draft's store at the
    /// target's position, a snapshot there, and the drafted ids after it
    /// the plain run's.
    fn chooser_plain(
        m: Qwen38Model,
        prompt: &[u32],
        plain: PlainPass,
    ) -> Result<(Qwen38Model, bool), GateError> {
        const N: usize = 12;
        const WIDTH: usize = <app::mtp::MtpDraft<Body38> as Draft<app::Session<Body38>>>::WIDTH;
        let ctx = m.body("the chooser's plain pass")?.ctx() as u32;
        let (m, plain_ids, _) = plain_to(m, prompt, Prompt38::Auto, N)?;
        let mut s = app::Session::from_model(m, ctx);
        let (label, cases): (&str, &[(&str, bool)]) = match plain {
            PlainPass::Held => ("(i) the held pass", &[("the cut step", false)]),
            PlainPass::Dropped => (
                "(v) a window no verify takes",
                &[("begin", false), ("a drafted pass", true)],
            ),
        };
        let mut ok = true;
        for &(after, drafted) in cases {
            let mut spec = Spec38::new(fresh_draft(&s)?);
            let mut out = match plain {
                PlainPass::Held => start(&mut s, spec.draft_mut(), prompt)?,
                PlainPass::Dropped => {
                    s.reset()?;
                    let first = Advance::prompt(&mut spec, &mut s, prompt)?;
                    Advance::begin(&mut spec, &s, prompt, first)?;
                    vec![first]
                }
            };
            if drafted {
                one_pass(&mut s, &mut spec, &mut out)?;
            }
            let last = *out.last().ok_or("no token before the plain pass")?;
            let d = spec.draft_mut();
            let told = (|| -> Result<u32, app::SessionError> {
                match plain {
                    PlainPass::Held => Draft::before_plain(d, &mut s, last)?,
                    PlainPass::Dropped => {
                        Draft::propose(d, &mut s, last, &mut [0u32; WIDTH])?;
                        Draft::<app::Session<Body38>>::unproposed(d);
                    }
                }
                let next = Target::step(&mut s, last, Want::Argmax)?.argmax();
                Draft::held(d, &mut s, last, next)?;
                Ok(next)
            })();
            let pos = s.pos() as usize;
            let held = held_of(s.model())?;
            let saved = s.model_mut().seq_save().map(|st| st.positions() as usize);
            let told_ok = told.is_ok();
            if let Ok(next) = &told {
                out.push(*next);
                while out.len() < N {
                    one_pass(&mut s, &mut spec, &mut out)?;
                }
            }
            let ids_ok = told_ok && out[..N] == plain_ids[..N];
            let saved_ok = matches!(&saved, Ok(n) if *n == pos);
            println!(
                "{label} after {after}: the step told {} {}; the draft store at {held} of the \
                 target's {pos} {}; a snapshot there {} {}; the ids the plain run's {}",
                match &told {
                    Ok(_) => "ok".to_string(),
                    Err(e) => e.to_string(),
                },
                verdict(told_ok),
                verdict(held == pos),
                match &saved {
                    Ok(n) => format!("{n} positions"),
                    Err(e) => e.to_string(),
                },
                verdict(saved_ok),
                verdict(ids_ok)
            );
            ok &= told_ok && held == pos && saved_ok && ids_ok;
        }
        Ok((s.into_model(), ok))
    }

    /// Window `ids` alone on the selected slot, a pass of each of `depths`:
    /// none for `None`; at 0 one step with the draft told of it as a seat's
    /// step tells it (its waiting rows walked first,
    /// [`app::mtp::MtpDraft::before_step`]); else a drafted pass of
    /// proposals capped to the depth ([`Capped`]), verified through the
    /// session's verify of one sequence.
    fn capped_run(
        s: &mut app::Session<Body38>,
        ids: &[u32],
        depths: &[Option<usize>],
    ) -> Result<Drafted, GateError> {
        let mut d = fresh_draft(s)?;
        let mut out = start(s, &mut d, ids)?;
        let mut spec = runtime::Speculative::<Capped, 4>::new(Capped { d, depth: 0 });
        let mut passes = Vec::with_capacity(depths.len());
        for &depth in depths {
            match depth {
                None => {}
                Some(0) => {
                    let (pos, last) = (s.pos(), *out.last().ok_or("no token")?);
                    let d = &mut spec.draft_mut().d;
                    d.before_step(s, last)?;
                    let next = Target::step(s, last, Want::Argmax)?.argmax();
                    Draft::stepped(d, s, last, next)?;
                    out.push(next);
                    passes.push(Committed {
                        pos,
                        kept: 1,
                        rows: 1,
                        proposed: false,
                    });
                }
                Some(depth) => {
                    spec.draft_mut().depth = depth;
                    passes.push(one_pass(s, &mut spec, &mut out)?);
                }
            }
        }
        Ok(Drafted {
            ids: out,
            passes,
            windows: spec.draft_mut().d.take_windows(),
            store: slot_store(s)?,
        })
    }

    /// Each round's committed rows per slot, and the clauses that differ from their references.
    type SlotRounds = (Vec<Vec<(usize, Committed)>>, Vec<String>);

    /// Each window's request started alone on its own slot (slot 0 window A,
    /// slot 1 window B, in the server's order), then a pass of each of
    /// `rounds`' slots (a depth a slot, `None` for a slot the pass leaves
    /// out, [`app::mtp::pass_slots`]), each slot's draft its own: each
    /// pass's commits by slot, and the parts of each slot's run (its ids,
    /// kept rows, draft windows and draft store) that differ from its
    /// reference in `refs`.
    fn slot_rounds(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
        rounds: &[(Option<usize>, Option<usize>)],
        refs: [&Drafted; 2],
    ) -> Result<SlotRounds, GateError> {
        let mut drafts = [fresh_draft(s)?, fresh_draft(s)?];
        let mut outs = [Vec::new(), Vec::new()];
        for (slot, ((d, out), ids)) in drafts.iter_mut().zip(&mut outs).zip(windows).enumerate() {
            s.select_slot(slot)?;
            *out = start(s, d, ids)?;
        }
        let mut passes = [Vec::new(), Vec::new()];
        let mut by_pass = Vec::with_capacity(rounds.len());
        for &(a, b) in rounds {
            let [d0, d1] = &mut drafts;
            let [o0, o1] = &mut outs;
            let mut w = Vec::with_capacity(2);
            for (slot, depth, draft, out) in [(0, a, d0, o0), (1, b, d1, o1)] {
                if let Some(depth) = depth {
                    let last = *out
                        .last()
                        .ok_or_else(|| format!("slot {slot} gave no id"))?;
                    w.push(app::mtp::SlotWindow {
                        slot,
                        draft,
                        last,
                        depth,
                        out,
                        draw: None,
                    });
                }
            }
            let slots: Vec<usize> = w.iter().map(|w| w.slot).collect();
            let (done, _) = app::mtp::pass_slots(s, &mut w)?;
            if done.len() != slots.len() {
                return Err(format!(
                    "a pass of {} slots committed {} slots",
                    slots.len(),
                    done.len()
                )
                .into());
            }
            for (&slot, &c) in slots.iter().zip(&done) {
                passes[slot].push(c);
            }
            by_pass.push(slots.into_iter().zip(done).collect());
        }
        let mut off = Vec::new();
        for (slot, (((d, out), passes), r)) in drafts
            .iter_mut()
            .zip(outs)
            .zip(passes)
            .zip(refs)
            .enumerate()
        {
            s.select_slot(slot)?;
            let got = Drafted {
                ids: out,
                passes,
                windows: d.take_windows(),
                store: slot_store(s)?,
            };
            let parts = [
                ("ids", got.ids == r.ids),
                ("kept rows", got.passes == r.passes),
                ("draft windows", got.windows == r.windows),
                ("draft store", got.store == r.store),
            ];
            off.extend(
                parts
                    .into_iter()
                    .filter(|&(_, same)| !same)
                    .map(|(part, _)| format!("slot {slot} {part}")),
            );
        }
        Ok((by_pass, off))
    }

    /// A (z) clause's words for [`slot_rounds`]' differing parts `off`.
    fn alone_runs(off: &[String]) -> String {
        if off.is_empty() {
            "each slot's ids, kept rows, draft windows and draft store its window's alone run's"
                .to_string()
        } else {
            format!("differs in {}", off.join(", "))
        }
    }

    /// (z) at `depths`, a slot's each: each window's request started alone
    /// on its own slot (slot 0 window A, slot 1 window B, in the server's
    /// order), then `rounds` passes of both slots' drafted rows in one pass
    /// ([`app::mtp::pass_slots`]), each slot's draft its own; against
    /// `refs`, each window's [`capped_run`] at its slot's depth. Unequal
    /// depths run chains of unequal own walks in one round. Mutant: a
    /// continuing chain's rows at its staged block (`b.block` for `next` in
    /// `chain_rows`). `uneven` holds the passes to
    /// the module doc's coverage of a pass keeping different counts on the
    /// two slots: the real tier's, where the shipped draft's proposals are
    /// accepted unevenly.
    fn one_pass_windows(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
        depths: [usize; 2],
        rounds: usize,
        refs: [&Drafted; 2],
        uneven: bool,
    ) -> Result<bool, GateError> {
        let [d0, d1] = depths;
        let (done, off) = slot_rounds(s, windows, &vec![(Some(d0), Some(d1)); rounds], refs)?;
        let (mut rejected, mut apart) = (0usize, 0usize);
        for pass in &done {
            rejected += pass.iter().filter(|(_, c)| c.kept < c.rows).count();
            if let [(_, c0), (_, c1)] = pass.as_slice() {
                apart += usize::from(c0.kept != c1.kept);
            }
        }
        let same = off.is_empty();
        let ok = same && rejected > 0 && (!uneven || apart > 0);
        println!(
            "(z) two slots in one pass at depths {d0} and {d1} (a pass of {} rows): {rounds} \
             passes; {}; {rejected} slot passes kept fewer rows than they ran, {apart} passes \
             kept different counts on the two slots{} {}",
            d0 + d1 + 2,
            alone_runs(&off),
            if uneven {
                ""
            } else {
                " (printed, not pinned in this tier)"
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (z) plain rounds' passes: each pass's depth a slot (slot 0, slot 1),
    /// `None` for a slot the pass leaves out, 0 a plain row whose draft's
    /// waiting rows the pass's one store walk of every depth zero slot's
    /// rows takes ([`app::mtp::SlotWalks`]), 3 a drafted window of up to 4
    /// rows. A plain pass after a drafted one walks the rows that pass kept,
    /// both slots' together where both are plain, one beside the other's
    /// drafted window where one is. The passes of slot 1 alone run while
    /// slot 0's draft waits on rows of the pass arena, which slot 0's next
    /// pass reads.
    const PLAIN_ROUNDS: [(Option<usize>, Option<usize>); 12] = [
        (Some(0), Some(0)),
        (Some(3), Some(3)),
        (Some(0), Some(0)),
        (Some(3), Some(3)),
        (Some(0), Some(3)),
        (Some(3), Some(0)),
        (None, Some(3)),
        (None, Some(0)),
        (Some(0), Some(0)),
        (Some(3), Some(3)),
        (None, Some(3)),
        (Some(3), Some(3)),
    ];

    /// (z) plain rounds: each window's request started alone on its own
    /// slot, then [`PLAIN_ROUNDS`]' passes of the slots each names
    /// ([`app::mtp::pass_slots`]): each slot's ids, kept rows, draft windows
    /// and draft store equal its window run alone ([`capped_run`]). At
    /// least one pass's store walk takes both slots' rows, and at least one
    /// pass of slot 1 alone runs while slot 0's draft waits on rows of the
    /// pass arena; the captured pass of slot 1's four rows alone holds the
    /// body's own count of its launches (`Body38::slots_launches`: it walks
    /// into slot 1's rows and copies none); under `wide` (the real tier,
    /// where the shipped draft's proposals are accepted) at least one store
    /// walk takes more than one row of a slot, the rows a drafted pass kept
    /// (the fixture's draft has every proposal refused: printed, not pinned
    /// there).
    fn plain_rounds(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
        refs: &[Drafted; 2],
        wide: bool,
    ) -> Result<bool, GateError> {
        let (done, off) = slot_rounds(s, windows, &PLAIN_ROUNDS, refs.each_ref())?;
        let key = [(1, 4)];
        let nodes = s.model_mut().capture_slots(&key)?;
        let own = s.model().body("(z) plain rounds")?.slots_launches(&key);
        // The rows each slot's draft has waiting: its start's step leaves one.
        let mut waiting = [1, 1];
        let (mut both, mut alone, mut deep) = (0usize, 0usize, 0usize);
        for (pass, depths) in done.iter().zip(PLAIN_ROUNDS) {
            both += usize::from(depths == (Some(0), Some(0)));
            alone += usize::from(depths.0.is_none());
            for &(slot, c) in pass {
                deep += usize::from(half_of(depths, slot) == Some(0) && waiting[slot] > 1);
                waiting[slot] = c.kept;
            }
        }
        let same = off.is_empty();
        let ok = same && both > 0 && alone > 0 && nodes == own && (!wide || deep > 0);
        println!(
            "(z) plain rounds: {} passes, a plain slot's waiting rows walked in one store walk \
             of the pass's plain slots; {}; {both} walks took both slots' rows, {alone} passes \
             of slot 1 alone ran beside slot 0's waiting rows, the pass of slot 1's 4 rows alone \
             holds {nodes} nodes against the body's own {own}, {deep} walks took more than one \
             row of a slot{} {}",
            PLAIN_ROUNDS.len(),
            alone_runs(&off),
            if wide {
                ""
            } else {
                " (printed, not pinned in this tier)"
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (z) a shared prompt anchor consumed before selecting the other slot,
    /// against the same round with each draft walked immediately.
    fn prompt_anchor(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
    ) -> Result<bool, GateError> {
        let mut runs = Vec::new();
        for immediate in [true, false] {
            let mut drafts = [
                app::mtp::MtpDraft::open(
                    s.model(),
                    Prompt38::Gemm,
                    bloomery_gpu::model::StepMode::Graph,
                )?,
                fresh_draft(s)?,
            ];
            s.select_slot(1)?;
            let mut out1 = start(s, &mut drafts[1], windows[1])?;
            s.select_slot(0)?;
            s.reset()?;
            let first = Draft::prompt(&mut drafts[0], s, windows[0])?;
            Draft::begin(&mut drafts[0], s, windows[0], first)?;
            let mut out0 = vec![first];
            let last1 = *out1.last().ok_or("slot 1 gave no id")?;
            if immediate {
                drafts[0].before_step(s, first)?;
                s.select_slot(1)?;
                drafts[1].before_step(s, last1)?;
            }
            let [d0, d1] = &mut drafts;
            let (done, passes) = app::mtp::pass_slots(
                s,
                &mut [
                    app::mtp::SlotWindow {
                        slot: 0,
                        draft: d0,
                        last: first,
                        depth: 0,
                        out: &mut out0,
                        draw: None,
                    },
                    app::mtp::SlotWindow {
                        slot: 1,
                        draft: d1,
                        last: last1,
                        depth: 0,
                        out: &mut out1,
                        draw: None,
                    },
                ],
            )?;
            s.select_slot(0)?;
            let store0 = slot_store(s)?;
            s.select_slot(1)?;
            runs.push((out0, out1, done, passes, store0, slot_store(s)?));
        }
        let ok = runs[0] == runs[1] && runs[1].2.len() == 2 && runs[1].3 == 1;
        println!(
            "(z) prompt anchor: a plain round after slot 0's GEMM prompt and begin beside \
             slot 1's first step matches immediate walks in ids, commits and draft stores {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// (z)'s refusal: a pass of slot 1's verify rows alone waiting for its
    /// commit — slot 0 selected and holding no verify of its own, so only
    /// the model's own check stands between a select and the waiting slot —
    /// a select of slot 1 is refused by name by the model, slot 0 still
    /// selected and both positions kept; the commit then keeps row 0.
    fn waiting_select(s: &mut app::Session<Body38>) -> Result<bool, GateError> {
        let pos = |s: &mut app::Session<Body38>| -> Result<[u32; 2], GateError> {
            let mut p = [0; 2];
            for (slot, p) in p.iter_mut().enumerate() {
                s.select_slot(slot)?;
                *p = s.model().pos();
            }
            Ok(p)
        };
        let before = pos(s)?;
        s.verify_slots(&[(1, &[0, 0])])?;
        let waiting = (s.model().selected(), pos_live(s));
        let named = match s.model_mut().select_slot(1) {
            Err(bloomery_gpu::GpuError::Shape { what, detail }) => {
                what == "GpuModel::select_slot" && detail.contains("waits for its commit")
            }
            _ => false,
        };
        let kept = (s.model().selected(), pos_live(s)) == waiting && waiting == (0, before[0]);
        // A select that went through moved the waiting slot live: no commit
        // of the pass can stand after it, so the clause ends red here.
        let moved = named && kept && {
            s.commit_slots(&[1])?;
            pos(s)? == [before[0], before[1] + 1]
        };
        let ok = named && kept && moved;
        println!(
            "(z) waiting: a select of slot 1 while a pass of its verify rows waits for its \
             commit named by the model {named}, slot 0 still selected at its position {kept}; \
             the commit of one row moved slot 1 one position {moved} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// The selected slot's position.
    fn pos_live(s: &app::Session<Body38>) -> u32 {
        s.model().pos()
    }

    /// (z) step after's passes: each pass's rows a slot (slot 0, slot 1) —
    /// the token before a proposal, then its window's next ids — each
    /// slot's up to a verify's rows and both slots' up to the widest pass
    /// of two (8 rows).
    const AFTER_WINDOWS: [(usize, usize); 5] = [(4, 4), (4, 3), (3, 4), (4, 2), (3, 2)];
    /// Each [`AFTER_WINDOWS`] pass's kept rows a slot, fixed: every pass
    /// keeps part of one slot's rows at least, the two counts differ, and
    /// the last keeps 1 of slot 1's 2 — a partial keep of the parked slot
    /// right before its plain steps.
    const AFTER_KEEPS: [(usize, usize); 5] = [(2, 3), (4, 1), (1, 2), (3, 2), (2, 1)];
    /// The plain steps after the passes: the second reads what the first
    /// wrote.
    const AFTER_STEPS: usize = 2;

    /// One side of (z) step after's history: every verify row's id (the
    /// prompt's argmax first), each plain step's argmax and logits row (its
    /// bits), and the committed lane and position after the steps.
    #[derive(PartialEq)]
    struct After {
        ids: Vec<u32>,
        steps: Vec<(u32, Vec<u32>)>,
        lane: u32,
        pos: u32,
    }

    /// Slot `half`'s count of a `(slot 0, slot 1)` pair.
    fn half_of<T>((a, b): (T, T), half: usize) -> T {
        if half == 0 { a } else { b }
    }

    /// A pass's `m` rows of a slot: `last`, then `window`'s ids from `at`.
    fn after_rows(window: &[u32], last: u32, at: usize, m: usize) -> Result<Vec<u32>, GateError> {
        let fill = window
            .get(at..at + m - 1)
            .ok_or("the window holds too few ids for (z) step after's rows")?;
        let mut rows = vec![last; m];
        rows[1..].copy_from_slice(fill);
        Ok(rows)
    }

    /// `rows` as one verify of the selected sequence through the session's
    /// verify of one sequence, each row's argmax; waits for its commit.
    fn verify_rows(s: &mut app::Session<Body38>, rows: &[u32]) -> Result<Vec<u32>, GateError> {
        fn run<const M: usize>(
            s: &mut app::Session<Body38>,
            rows: &[u32],
        ) -> Result<Vec<u32>, GateError> {
            let rows: [u32; M] = rows
                .try_into()
                .map_err(|_| format!("{} rows for a verify of {M}", rows.len()))?;
            Ok(Verify::verify(s, rows)?.to_vec())
        }
        match rows.len() {
            2 => run::<2>(s, rows),
            3 => run::<3>(s, rows),
            4 => run::<4>(s, rows),
            n => Err(
                format!("a verify of {n} rows; (z) step after's passes run 2 to 4 a slot").into(),
            ),
        }
    }

    /// The selected slot's [`AFTER_STEPS`] plain steps from `last`, each
    /// with its logits row, then its committed lane and position: the end
    /// of an [`After`] whose verify rows' ids are `ids`.
    fn after_steps(
        s: &mut app::Session<Body38>,
        ids: Vec<u32>,
        mut last: u32,
    ) -> Result<After, GateError> {
        let mut steps = Vec::with_capacity(AFTER_STEPS);
        for _ in 0..AFTER_STEPS {
            let Out::Logits { argmax, row } = Target::step(s, last, Want::Logits)? else {
                return Err("a step asked for its logits gave the argmax alone".into());
            };
            steps.push((argmax, row.iter().map(|v| v.to_bits()).collect()));
            last = argmax;
        }
        Ok(After {
            ids,
            steps,
            lane: s.model().body("slots")?.lane(),
            pos: s.model().pos(),
        })
    }

    /// (z) step after's reference for slot `half`'s side, on the selected
    /// sequence from its reset: `window`'s prompt call, each
    /// [`AFTER_WINDOWS`] pass of that slot's rows through the session's
    /// verify of one sequence committed at its [`AFTER_KEEPS`] count, then
    /// the plain steps.
    fn after_alone(
        s: &mut app::Session<Body38>,
        half: usize,
        window: &[u32],
    ) -> Result<After, GateError> {
        s.reset()?;
        let mut last = Target::prompt(s, window, Want::Argmax)?.argmax();
        let mut ids = vec![last];
        let mut at = 0;
        for (&w, &k) in AFTER_WINDOWS.iter().zip(&AFTER_KEEPS) {
            let (m, k) = (half_of(w, half), half_of(k, half));
            let rows = after_rows(window, last, at, m)?;
            at += m - 1;
            let out = verify_rows(s, &rows)?;
            Verify::commit(s, k)?;
            ids.extend_from_slice(&out);
            last = out[k - 1];
        }
        after_steps(s, ids, last)
    }

    /// (z) step after (module doc): each window's request started on its
    /// own slot from its reset (slot 0 window A, slot 1 window B), the
    /// [`AFTER_WINDOWS`] each as one pass of both slots' verify rows
    /// committed at its [`AFTER_KEEPS`] counts, then each slot selected and
    /// its plain steps — slot 0's right after the commit (selected by the
    /// pass), slot 1's after the select that exchanges it live. Against
    /// `refs`, each window's [`after_alone`].
    fn step_after(
        s: &mut app::Session<Body38>,
        windows: [&[u32]; 2],
        refs: &[After; 2],
    ) -> Result<bool, GateError> {
        let mut last = [0; 2];
        let mut ids = [Vec::new(), Vec::new()];
        for slot in 0..2 {
            s.select_slot(slot)?;
            s.reset()?;
            last[slot] = Target::prompt(s, windows[slot], Want::Argmax)?.argmax();
            ids[slot].push(last[slot]);
        }
        let mut at = [0; 2];
        for (&(m0, m1), &(k0, k1)) in AFTER_WINDOWS.iter().zip(&AFTER_KEEPS) {
            let rows = [
                after_rows(windows[0], last[0], at[0], m0)?,
                after_rows(windows[1], last[1], at[1], m1)?,
            ];
            at = [at[0] + m0 - 1, at[1] + m1 - 1];
            let out = s.verify_slots(&[(0, &rows[0]), (1, &rows[1])])?.ids;
            if out.len() != m0 + m1 {
                return Err(
                    format!("a pass of {m0} and {m1} rows read back {} ids", out.len()).into(),
                );
            }
            ids[0].extend_from_slice(&out[..m0]);
            ids[1].extend_from_slice(&out[m0..]);
            last = [out[k0 - 1], out[m0 + k1 - 1]];
            s.commit_slots(&[k0, k1])?;
        }
        let mut off = Vec::new();
        for (slot, (ids, r)) in ids.into_iter().zip(refs).enumerate() {
            s.select_slot(slot)?;
            let got = after_steps(s, ids, last[slot])?;
            if got.ids != r.ids {
                let at = got.ids.iter().zip(&r.ids).position(|(x, y)| x != y);
                off.push(format!("slot {slot}'s verify ids apart at {at:?}"));
            }
            for (i, (g, w)) in got.steps.iter().zip(&r.steps).enumerate() {
                if g.0 != w.0 {
                    off.push(format!("slot {slot}'s step {i} id {} (alone {})", g.0, w.0));
                } else if g.1 != w.1 {
                    off.push(format!("slot {slot}'s step {i} logits row"));
                }
            }
            if (got.lane, got.pos) != (r.lane, r.pos) {
                off.push(format!(
                    "slot {slot}'s lane and position {:?} (alone {:?})",
                    (got.lane, got.pos),
                    (r.lane, r.pos)
                ));
            }
            if got.steps.len() != r.steps.len() {
                off.push(format!("slot {slot}'s step count"));
            }
        }
        let ok = off.is_empty();
        println!(
            "(z) step after: {} passes of both slots' verify rows at fixed keeps, the last \
             keeping 1 of parked slot 1's 2 rows, then {AFTER_STEPS} plain steps a slot (slot 0 \
             right after the commit, slot 1 after its select); {} {}",
            AFTER_WINDOWS.len(),
            if ok {
                "each slot's ids, each step's id and logits row bit for bit, its lane and \
                 position its window's alone run's"
                    .to_string()
            } else {
                format!("differs in {}", off.join(", "))
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (d) the drafting load's budget. The draft reserves its card bytes
    /// ([`MtpInputs::card_bytes`]) out of the same card budget, so the
    /// header's budget plus those bytes ([`plan_levers`]) leaves the target's
    /// routed experts on the card as many as the target alone holds under the
    /// header's budget: the card plan's counts a layer, drafted and alone, are
    /// equal, and in the fixture tier each layer's is half the file's experts —
    /// the header's contract, from which a budget one granule off leaves a
    /// layer one expert short. The real tier's budget is the caller's lever
    /// (unset: none), whose plans hold the draft's bytes out of the card's
    /// own: the counts are printed there, not pinned.
    fn draft_budget(inputs: &PlanInputs, mtp: &MtpInputs, ub: usize) -> Result<bool, GateError> {
        let machine =
            machine_for_experts(card()?, inputs.stack(), u64::try_from(ub)?, Experts::Card);
        let alone = inputs.plan_with(
            &machine,
            (CTX, u64::try_from(ub)?),
            &plan_levers(0)?,
            Experts::Card,
        )?;
        let extra = mtp.card_bytes(CTX)?;
        let drafted = inputs.plan_mtp_with(
            &machine,
            (CTX, u64::try_from(ub)?),
            &plan_levers(extra)?,
            mtp,
            Experts::Card,
        )?;
        let fixture = Tier::from_env()? == Tier::Fixture;
        let want = (cfg().n_expert / 2) as u64;
        let (a, d) = (&alone.n_l, &drafted.plan.n_l);
        let held = |v: &[u64]| (v.iter().min().copied(), v.iter().max().copied());
        let ok = !fixture || (a == d && d.iter().all(|&n| n == want));
        println!(
            "(d) the drafting load's budget: card experts a layer alone {:?} and drafted {:?} \
             under {} + the draft's {extra} B{} {}",
            held(a),
            held(d),
            plan_levers(extra)?.card_budget_bytes,
            if fixture {
                format!(" (want {want} on every layer both ways: half the experts)")
            } else {
                " (the real tier's budget is the lever's: counts printed, not pinned)".to_string()
            },
            verdict(ok)
        );
        Ok(ok)
    }

    /// (pg) a set draft on a paged plan: the paged tier's draft rule
    /// ([`bloomery_levers::paged_draft`]) runs a set `BLOOMERY_DRAFT=mtp`
    /// where the plan pages host experts through the NVMe tier's RAM arena
    /// (the rule turned it off before PIN(2026-10-12)), so this clause loads
    /// a second, paged model — the same target, draft and head as the load
    /// above, its host room halved until the plan pages (the dial carves the
    /// arena, never under its own floor; the room that crossed the floor
    /// ends the search) — and holds three things: the drafted greedy ids are
    /// the plain run's on that load (the draft moves no bit); the arena
    /// filled over the drafted run (`fills` past 0) and the verify's step
    /// port took its union call (`step_fills` past 0: the ids the four rows'
    /// union claimed, counted apart from the plain steps'); and the drafted
    /// plan's room (`MtpPlan::headroom_bytes`) is the plain stage card's less
    /// the draft's reserve, give or take what the expert rule moved off the
    /// card, within one expert's slack of either sign (the draft's card bytes
    /// come out of the context's — the seat's yield trade). Acceptance is not
    /// pinned here:
    /// a fixture's generated draft has every proposal rejected, tau = 1, as
    /// (z) says. Where the dial gives the file's host leg no arena at any room
    /// the clause has no input: the fixture tier leaves it to the real tier by
    /// name, the real tier is red on it.
    fn drafted_paged_run(
        inputs: &PlanInputs,
        mtp: &MtpInputs,
        ub: usize,
        prompt: &[u32],
    ) -> Result<bool, GateError> {
        const N: usize = 32;
        let machine =
            machine_for_experts(card()?, inputs.stack(), u64::try_from(ub)?, Experts::Card);
        let extra = mtp.card_bytes(CTX)?;
        // The room halves until the plan pages through an arena; a room the
        // tier's floor refuses ends the search, and the clause is red there.
        let (file, draft_file) = (open_target()?, open_draft()?);
        let mut paged_inputs = PlanInputs::describe(&file)?;
        let mut room = paged_inputs.room.0;
        let mut refused = None;
        // The plan leaves the search by the labelled break, not through a variable the next
        // round's write to `room` could see borrowed (the borrow checker of the stable
        // toolchain reads such a variable as live across the loop).
        let paged = 'search: {
            for _ in 0..24 {
                paged_inputs.room = (room, HostRead::Given);
                match paged_inputs.plan_mtp_with(
                    &machine,
                    (CTX, u64::try_from(ub)?),
                    &plan_levers(extra)?,
                    mtp,
                    Experts::Card,
                ) {
                    Ok(p)
                        if p.plan.host.nvme_expert_bytes > 0
                            && p.plan.host.nvme_arena_bytes > 0 =>
                    {
                        break 'search Some(p);
                    }
                    Ok(_) => room /= 2,
                    Err(e) => {
                        refused = Some(e);
                        break 'search None;
                    }
                }
            }
            None
        };
        let Some(plan) = paged else {
            // The split dial gives an arena only where half the host leg lies above the floor
            // and the arena's least (`ArenaDial::peak`); the fixture's four layers, half of
            // each one's experts on the host, lie under it at any room. The search then ends
            // at the floor with a refusal that counts no arena's least, and the fixture tier
            // leaves the clause to the real tier by name. Any other end of the search, and the
            // real tier, are red.
            let no_deep_side = paged_inputs.arena.is_none()
                && matches!(
                    refused,
                    Some(PlaceError::Placement(PlacementError::HostRoomFloor {
                        arena: None,
                        ..
                    }))
                );
            println!(
                "(pg) the search ended at room {room} B: {}",
                refused.map_or_else(|| "24 halvings".to_string(), |e| e.to_string())
            );
            if no_deep_side
                && !tier::run_clause(
                    "(pg) a room above the tier's floor pages this file's host experts through \
                     an arena",
                    Tag::FileBound,
                )?
            {
                return Ok(true);
            }
            println!(
                "(pg) no room above the tier's floor pages this file's host experts {}",
                verdict(false)
            );
            return Ok(false);
        };
        let plain = paged_inputs.plan_with(
            &machine,
            (CTX, u64::try_from(ub)?),
            &plan_levers(extra)?,
            Experts::Card,
        )?;
        // The fit: the drafted plan's room is `MtpPlan::headroom_bytes`, the field that
        // subtracts the draft's reserve R — its card bytes with its program's arena, the lever's
        // `extra` (the stage card's own headroom does not see R);
        // its partner is the plain plan's stage-card headroom at the same lever. The
        // expected relation, term by term: both stage cards hold the same fixed terms; the
        // planner's identity `MtpPlan::headroom = T.headroom - R` holds; the expert
        // rule took `moved` bytes (experts and their rounding) off the card to make room for R,
        // and the remainder R - moved is the slack of a rule that moves whole experts: under one
        // expert's card bytes plus a granule for each of a layer's three routed stacks. The
        // slack's sign follows the packing at the room, not the draft, so which plan holds
        // less room is printed, not held.
        let (p, t) = (&plain.cards[0], &plan.plan.cards[0]);
        let r = extra;
        let fixed = |c: &model::placement::CardTotals| {
            (
                c.dense_bytes,
                c.kv_bytes,
                c.stage_rows_bytes,
                c.scratch_bytes,
                c.context_bytes,
                c.reserve_bytes,
            )
        };
        let same_terms = fixed(p) == fixed(t);
        let identity = plan.headroom_bytes == t.headroom_bytes - i128::from(r);
        let moved = i128::from(p.expert_bytes + p.rounding_bytes)
            - i128::from(t.expert_bytes + t.rounding_bytes);
        let remainder = i128::from(r) - moved;
        // One expert's card bytes in a layer: the layer's routed stacks' card segments, each
        // one's bytes over its experts; the largest layer's, with a granule a stack.
        let mut per_layer = vec![0u64; plain.model.layers];
        for row in &plain.rows {
            let Some(l) = plain.model.tensors[row.tensor].layer else {
                continue;
            };
            for s in &row.segments {
                if let (model::placement::Device::Card(0), Some(list)) = (&s.device, &s.experts)
                    && !list.is_empty()
                {
                    per_layer[l] += s.resident_bytes.div_ceil(list.len());
                }
            }
        }
        let one = i128::from(
            per_layer.iter().copied().max().unwrap_or(0)
                + 3 * model::placement::workstation::GRANULE.get(),
        );
        let less = plan.headroom_bytes < p.headroom_bytes;
        // PIN(2026-10-15): the fit is the relation above, not "the drafted plan holds less room
        // than the plain plan": the expert rule moves whole experts, so the sign follows the
        // packing [A6000 real, gate-gpu-qwen4exp-mtp: remainder +2,091,180 B at room
        // 30,796,775,936 (fde6499f) and -2,103,124 B at room 31,139,464,192 (cee6e7e4), bound
        // 10,285,056 B; identity true and fixed terms equal at both].
        let fit = same_terms && identity && remainder.abs() < one;
        println!(
            "(pg) room {room} B: the plan pages {} B of host experts through a {} B arena; \
             the drafted plan's headroom {} B (MtpPlan) against the plain plan's {} B, {} B less \
             (the drafted plan holds less: {less}, printed); \
             R {r} B, moved {moved} B, remainder {remainder} B (bound {one}); fixed terms equal \
             {same_terms}, identity {identity} {}",
            plan.plan.host.nvme_expert_bytes,
            plan.plan.host.nvme_arena_bytes,
            plan.headroom_bytes,
            p.headroom_bytes,
            p.headroom_bytes - plan.headroom_bytes,
            verdict(fit)
        );
        let m = Body38::open_placed_mtp(
            file,
            &plan,
            &paged_inputs,
            0,
            cfg().host,
            ub,
            &draft_file,
            mtp,
        )?;
        let tier = m
            .body("(pg)")?
            .nvme_tier()
            .cloned()
            .ok_or("(pg) the paged load built no NVMe tier")?;
        let (m, plain_ids, _) = plain_to(m, prompt, Prompt38::Auto, N)?;
        let mid = tier.stats();
        let ctx = m.body("(pg)")?.ctx() as u32;
        let (mut s, mut spec) = drafted_session(m, step_cfg(Prompt38::Auto), ctx)?;
        let first = spec.prompt(&mut s, prompt)?;
        let out = runtime::generate(
            &mut s,
            &mut spec,
            prompt,
            first,
            &runtime::Stop::new(N, ctx)?,
            &mut Kept::default(),
        )?;
        let after = tier.stats();
        let ids_ok = out.tokens[..N] == plain_ids[..N];
        let fills = after.fills - mid.fills;
        let step_fills = after.step_fills - mid.step_fills;
        let counters_ok = fills > 0 && step_fills > 0;
        println!(
            "(pg) the drafted run's {N} ids = the plain run's {} (tau is not pinned: a              fixture's draft has every proposal rejected); the drafted run filled {fills} slots,              its verify's step port {step_fills} {}",
            verdict(ids_ok),
            verdict(counters_ok)
        );
        Ok(ids_ok && counters_ok && fit)
    }

    pub(super) fn run() -> Result<(), GateError> {
        let witnessed = init()?;
        let input = tier::input_set(
            "the MTP draft set's feed",
            (MTP_SET, &MTP),
            fixture::mirror(MTP_SET),
        )?;
        let dir = input.path;
        let set = MtpSet::open(&dir, input.family)?;
        let prompt = set
            .tokens
            .clone()
            .ok_or_else(|| format!("{}: no # tokens line", dir.display()))?;
        let order = graphs_of(&set);
        println!(
            "(o) {}: {} graphs over {} blocks, prompt {} ids, family {} PASS",
            dir.display(),
            order.len(),
            set.blocks().len(),
            prompt.len(),
            input.family.name
        );
        println!(
            "bands: eh {:.4e} router {:.4e} the argmax cap's {:.4e} free {FREE_BAND:.2}; the \
             node-local pins: γ(k/32 + 5) a dot, the pack ½γ(52) + 6u = {:.3e}, the mixes \
             MIXED_BAND {:.1e}, the combines' weights LO_BAND {:.1e}",
            eh_band(),
            router_band(),
            logits_band(),
            pack_rel(),
            MIXED_BAND,
            LO_BAND
        );
        let (file, draft) = (open_target()?, open_draft()?);
        let t = Instant::now();
        let inputs = PlanInputs::describe(&file)?;
        tier::sc(
            "(u) the unset head is the shipped list of the target's tokenizer (copied into a fixture)",
        )?;
        let shipped_ok = shipped_head(&file, inputs.spec.vocab)?;
        let ids: Vec<u32> = (0..LIST_ROWS).map(|i| i * 6).collect();
        let rows = HeadSet::List {
            ids: ids.clone().into(),
            digest: vocab_sha256(&file)?,
        };
        let mtp = MtpInputs::read(&draft, &file, &inputs, rows)?;
        let ub = bloomery_gpu::arch::qwen3moe::ubatch::ubatch_for(usize::try_from(CTX)?)?;
        let machine = machine(card()?, inputs.stack(), u64::try_from(ub)?);
        let extra = mtp.card_bytes(CTX)?;
        let plan = inputs.plan_mtp(
            &machine,
            (CTX, u64::try_from(ub)?),
            &plan_levers(extra)?,
            &mtp,
        )?;
        let d = &plan.draft.cards[0];
        let draft_kv = d.kv_bytes;
        println!(
            "plan card={} ctx_max={CTX} card_budget={} (the draft's card bytes {extra} in it) \
             draft dense={} experts={} rounding={} kv={} map={} arena={} headroom={}",
            card()?.name,
            plan_levers(extra)?.card_budget_bytes,
            d.dense_bytes,
            d.expert_bytes,
            d.rounding_bytes,
            d.kv_bytes,
            plan.map_bytes,
            plan.arena_bytes,
            plan.headroom_bytes
        );
        let mut m = Body38::open_placed_mtp(file, &plan, &inputs, 0, cfg().host, ub, &draft, &mtp)?;
        let arena = m.body("mtp")?.mtp().map_or(0, |d| d.arena_bytes());
        println!(
            "load card={} ctx_max={CTX} resident_bytes={} in {:.1} s; the draft program's arena \
             {arena} bytes (runtime value, outside the plan)",
            card()?.name,
            m.resident_bytes(),
            t.elapsed().as_secs_f64()
        );
        let mut ok = shipped_ok && witnessed;
        tier::sc("(d)(q) the draft's load holds its plan's bytes; Mtp38::open's refusals")?;
        ok &= draft_load(&m, plan, &inputs, &draft, &mtp, &ids)?;
        tier::sc("(d) the drafting load's budget keeps the target's card experts a layer")?;
        ok &= draft_budget(&inputs, &mtp, ub)?;
        ok &= refused(
            "the draft's own row before any walk",
            m.mtp_draft(DraftFeed::Own { pos0: 0 }, DraftHead::Rows, WalkMode::Eager),
            "a walk before the draft's own row",
        );

        let head_file = open_target()?;
        let host = Host::open(&head_file, &draft, inputs.spec.rms_eps)?;
        m.set_mtp_taps(true)?;
        let mut r = Replay::new();
        let mut last = None;
        let mut warm = None;
        tier::sc("(n) the node-local pins and (h)'s kernel on every replayed row")?;
        // The fixture set's oracle replay comes first: this tier's set's replay, which every later
        // clause starts from, then rewrites every row of the draft's store.
        ok &= elh_fx(&mut m, &host, (&head_file, &draft))?;
        let vs_ik = tier::run_clause(
            "(e)(l)(h) every graph of ik's set replayed: eh_proj, routes, l_out and the argmax against ik's",
            Tag::Oracle,
        )?;
        let vs = if vs_ik { Vs::Real } else { Vs::Nothing };
        let layer = ik_layer(&set)?;
        for &(b, g) in &order {
            let ig = IkGraph::read(&set, b, g, layer)?;
            replay(&mut m, &ig, &host, &mut r, vs)?;
            if g == Graph::Warmup {
                warm = Some(ig);
            } else {
                last = Some(ig);
            }
        }
        let warm = warm.ok_or("the set has no warmup graph")?;
        let last = last.ok_or("the set has no block graph")?;
        let (e_ok, l_ok, h_ok) = if vs_ik {
            let e_ok = r.eh_fail == 0 && r.rows > 0;
            println!(
                "(e) {} rows: eh_proj worst {:.3e} (band {:.3e}), {} past {}",
                r.rows,
                r.eh_worst,
                eh_band(),
                r.eh_fail,
                verdict(e_ok)
            );
            let l_ok = r.flips_bad == 0 && r.l_out_held > 0;
            println!(
                "(l) {} rows, {} on flips, {} flips not allowed; l_out's distance from ik's on \
                 the {} rows held {:.3e} (printed, not held: the rounding of ik's q8 \
                 activations) {}",
                r.rows,
                r.flips,
                r.flips_bad,
                r.l_out_held,
                r.l_out_worst,
                verdict(l_ok)
            );
            let h_ok = r.argmax_bad == 0 && r.kernel_bad == 0 && r.heads_held > 0;
            println!(
                "(h) {} head rows, {} held: argmax = ik's {}, ties {}, wrong {}; ik's smallest \
                 top-2 margin {}; the draft's token = our argmax but {}; the logits' distance \
                 from ik's {:.3e} (printed, not held) {}",
                r.heads,
                r.heads_held,
                r.argmax_same,
                r.argmax_tie,
                r.argmax_bad,
                r.margin_min
                    .as_ref()
                    .map_or("-".to_string(), |(m, q, at)| format!(
                        "{m:.3e}, {q:.2} of its tie cap, at {at}"
                    )),
                r.kernel_bad,
                r.logits_worst,
                verdict(h_ok)
            );
            (e_ok, l_ok, h_ok)
        } else {
            let h_ok = r.kernel_bad == 0 && r.heads > 0 && r.rows > 0;
            println!(
                "(h) {} rows, {} head rows: the draft's token = our full head's argmax but {} \
                 {}",
                r.rows,
                r.heads,
                r.kernel_bad,
                verdict(h_ok)
            );
            (true, true, h_ok)
        };
        let mut n_ok = true;
        for p in r.pins.all() {
            println!("{}", p.line());
            n_ok &= p.ok();
        }
        ok &= e_ok && l_ok && h_ok && n_ok;
        let map = m
            .body("list")?
            .mtp()
            .and_then(|d| d.head_map())
            .map(|(map, n)| -> Result<Vec<u32>, GateError> {
                let mut v = map.to_host_vec(m.gpu().stream())?;
                v.truncate(n);
                Ok(v)
            })
            .transpose()?
            .ok_or("the load opened no row list")?;
        tier::sc("(r) the row-list head against the full head")?;
        ok &= map == ids;
        ok &= list_head(&mut m, &last, &map)?;
        m.set_mtp_taps(false)?;
        tier::sc(
            "(t) the pairing: the draft's walk fed the target's rows in place = from the host",
        )?;
        ok &= pairing(&mut m, &prompt, &warm)?;
        tier::sc("(g) a captured walk = the eager walk")?;
        ok &= graphs(&mut m, &warm)?;
        tier::sc("(f) the walk's refusals and the NaN fault")?;
        ok &= refusals(&mut m, &warm)?;
        tier::sc("(w) the windows end to end: drafted ids = plain ids")?;
        let (m, w_ok) = windows(m, &prompt, &deep_prompt()?)?;
        ok &= w_ok;
        tier::sc("(c) the width chooser")?;
        let (m, width_ok) = width_chooser(m, &prompt)?;
        ok &= width_ok;
        let (mut m, c_ok) = continued(m, &prompt)?;
        ok &= c_ok;
        ok &= arena_holds(&mut m, &prompt)?;
        tier::sc("(fr) a prompt call's rows the head does not cover feed the draft final")?;
        let (m, one_unit, fr_ok) = final_rows(m, &prompt)?;
        ok &= fr_ok;
        tier::sc("(p) a partial accept's records")?;
        let (m, partial_ok) = partial_state(m, &prompt, draft_kv)?;
        ok &= partial_ok;
        let (model, p_ok) = kept_state(m, &prompt)?;
        ok &= p_ok;
        tier::sc("(i) the width chooser's held pass after the server's cut step")?;
        let (model, i_ok) = chooser_plain(model, &prompt, PlainPass::Held)?;
        ok &= i_ok;
        tier::sc("(v) a window no verify takes: no stale walk, the draft store at the step")?;
        let (model, v_ok) = chooser_plain(model, &prompt, PlainPass::Dropped)?;
        ok &= v_ok;
        tier::sc("(x) a resume of a state without the draft's side is refused")?;
        let (model, x_ok) = resume_refused(model, &inputs, ub, &prompt)?;
        ok &= x_ok;
        tier::sc("(kp-draft) a cut and a prefix restore to a point keep the draft")?;
        let (model, kp_ok) = prefix_draft(model, &prompt)?;
        ok &= kp_ok;
        tier::sc("(kp-draft) the store behind the point keeps the draft off")?;
        let (model, behind_ok) = store_behind(model, &prompt)?;
        ok &= behind_ok;
        tier::sc("(sl) a save on a lane copies the draft's store rows of the state's positions")?;
        let (model, sl_ok) = lane_save(model, &prompt, draft_kv)?;
        ok &= sl_ok;
        drop(model);
        tier::sc("(fr) a prompt call of several ubatches feeds the draft final rows")?;
        ok &= final_rows_ubatches(&inputs, &mtp, &prompt, &one_unit)?;
        tier::sc("(kp-draft) a point inside a long prompt call, cut to and put back as a prefix")?;
        ok &= long_point(&inputs, &mtp, &prompt)?;
        // Window B diverges from window A from position 0 on: the corpus's
        // deep window read past A's whole length, so no per-sequence state
        // the two share by content can pass for one the slots exchanged.
        let deep = deep_prompt()?;
        tier::sc("(y)(z) two drafted sequences resident: interleaved and as one pass")?;
        ok &= slots_windows(&inputs, &mtp, ub, &prompt, &deep[1000..])?;
        tier::sc("(pg) a set draft on a paged plan: the ids the plain load's, the arena's fills")?;
        ok &= drafted_paged_run(&inputs, &mtp, ub, &prompt)?;
        println!("{}", tier::tally_line());
        tier::expect_fixture_oracle("gate_qwen4exp_mtp", FX_ORACLE)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
