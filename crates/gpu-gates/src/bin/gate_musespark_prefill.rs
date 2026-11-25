//! The Muse Spark prefill gate: the prompt batch (`GpuModel::<Body>::prefill`, the serve seat's `app::Prompt for Body`:
//! passes of up to the load's pass rows through every layer, the attention over each row's own window of the K/V planes,
//! the latent experts by slot, a placed load's hosted experts through the host tier's batch port) is the one-token step run
//! per id to within the error model's band, in the logits row and in every layer's output, and the stores it leaves are the
//! ones the next step reads.
//!
//! The batch keeps its own arithmetic at the dense sites (`project_wide`: each Q8_0 site's input rounded to q8_1 blocks of
//! 32 where the step's gemv reads it in f32, and every GEMM site's block dots added in another f32 order than the gemv's);
//! the attention, the router, the experts, the norms and the residual mixes are the step's own kernels over the same rows. So
//! the batch is the steps' within a band, not bit for bit, and the band is the one two implementations of one quantized
//! model sit apart by: the batch and the steps differ by a subset of the roundings ours and mainline's differ by
//! (`gate_musespark_e2e` holds those to its `EXPERT_REL_BAND` at `l_out`, a router flip included), and the logits by the
//! family's σ (`Case::sigma_logit`, V4.1's until the family's own is measured).
//!
//! The prompt lengths straddle the sliding window `W` (the file's `attention.sliding_window`, 2,048 in both tiers' files):
//! `1` (a pass of one row, the step's shape), `7` (rows short of any alignment), `W - 1` and `W + 1` (the last length every
//! sliding layer reads whole, and the first whose oldest key leaves the window; both cut into several passes, so the K/V
//! planes cross the cut between two passes), and `W + 552` (the 46 sliding layers read a window of real cells and the 16
//! full layers the whole context); the real tier leaves `W - 1` to the fixture. The ids are the fixed walk of `e2e::walk`: the
//! clauses compare the engine's two paths to each other and read no reference.
//!
//! The clauses (all self-consistency, so both tiers run them) are the body's own and the common ones
//! ([`prefill_clauses`]):
//! - `prefill_pass_cut_is_bits`: at every length of the tier's own list, the prompt in passes of `gate::CUT` rows (odd,
//!   mid-run) gives the load's passes' logits row, every layer's last-row output and every store bit for bit: a row's
//!   launches are the same arithmetic in whatever pass it lands, so this holds every layer where the free chain's band
//!   below holds layer 0 only.
//! - `prefill_layers_forced_near_steps`: each layer alone over the whole prompt (`GpuModel::prefill_layer`) on the rows the
//!   steps' own layer before it left, so no layer carries the distance of the one before it on: the layer's output at the
//!   last row of every pass within `gate::TAP_CEIL` of the steps' at that position, for every layer of both arms. A row
//!   where the batch keeps another expert set than the steps is not compared by distance (an exchanged expert moves a row by
//!   its share, not by a rounding): the two paths round the layer's `ffn_inp` apart, so the router's scores differ, and the
//!   exchange must be the batch's own: its kept set is the top of its own selection values (its scores, which the pass
//!   leaves in the route's `probs`, plus the layer's bias; another is a selection defect and ends the clause by name), and
//!   every exchanged pair's measured drift (how far the batch's scores moved the gained expert above the dropped one
//!   against the steps') sits inside the band `gate::ROUTER_EPS` gives, the rounding model's allowance for that movement,
//!   which is all `ROUTER_EPS` decides; a layer where more than half of the judged rows exchange fails by name. The real tier
//!   runs it, and the cut clause, at `7` and `gate::REAL_OWN` (the steps' outputs of every position are held in host
//!   memory, and the cut clause's passes of 7 rows each cross the host tier).
//! - `prefill_span_is_eager_only`: in graph mode `GpuModel::set_layer_span` and `GpuModel::set_input_override` are refused
//!   by their names (a captured step replays the walk it recorded, so neither would move a replay), and in eager mode both
//!   are taken and cleared; the load is left with its step captured.
//! - `prefill_near_steps`: from a reset, `prompt(P ids)` and `P` steps give last logits rows within the steps' top `gate::TOP`
//!   ids' band and layer 0's output at the last position within `gate::TAP_CEIL` (the layers after it can exchange a routed
//!   expert at a near tie and carry it on, so no error model bands the free chain there; the forced clause holds them), the
//!   same store list (each layer's K and V planes over the `P` written positions; the digests are printed, every one moves
//!   with the first rounding), and the position is `P`.
//! - `prefill_then_step_near_steps`: after `prompt(P ids)` one more step (the captured graph, as the seat serves) gives the
//!   logits row and layer 0's output of `P + 1` steps within the same bands, so the stores a batch leaves are the ones the
//!   next step reads.
//! - `prefill_group_is_group_one`: the body has no prompt group (one pass walk at a time; `gate::Muse::set_group` refuses any
//!   other than 1), so the clause runs at group 1: two prompts of the longest length from a reset give the same bits, the
//!   passes' sums carrying no order a run can change.
//! - `fault_word_clean`: the fault word is clean after every run.
//!
//! The arms: the fixture tier runs the whole-card load (`GpuModel::open`) and a placed load whose card keeps half the
//! routed experts (`mid`: every hosted layer's batch crosses the host tier's port and the card's slots of the same rows
//! add theirs, the budget `gate_musespark_e2e`'s placed arms set), so the batch is held on both bodies; the real tier runs
//! the gate plan's placed load (the e2e gate's real load).
//!
//! The stream arm (`stream`, fixture tier): the `mid` budget of a plan that counts the stream's ring (its floor in the
//! prompt reserve, `PlanInputs::streaming`) under the common residency machine (`Residency::Mid`, half
//! the smallest layer's card experts pinned and one spare, the host set read in) with the prompt calls' expert stream on
//! (`XMode::Split`): the host experts of a hosted layer the stream's rule sends to the card land in its ring and run as
//! the common route GEMM over the ring's half beside the stage slots' `_sel` launches, their rows joined into the
//! card's sum before the host's. It runs the
//! forced clause, whose calls (each layer's passes) stream, and the span clause, and then:
//! - `prefill_stream_streamed`: over every call the forced clause made, the stream's records and the body's ring passes
//!   agree: each streamed layer has one ring pass of its walk and layer that joined as many experts, and no ring pass
//!   joined experts no record names; the call's end counts the layers' sum; the host left out exactly the streamed
//!   columns; and some call streamed an expert, or the clause proves nothing and fails by name.
//!
//! The cut clause does not run on the stream arm (passes of [`CUT`] rows open no stream under the body's
//! `STREAM_FLOOR`, so the two cuts compute other slots on the card), nor do the common clauses (their bands hold the
//! `mid` arm's placement, and a call's stream moves the placement the next call starts from).
//!
//! The two-card arm (`--two-card`, `just gate-gpu-musespark-twocard` under `BLOOMERY_CARD=both`, the only arm it runs):
//! plan (b′) (`--place bp`: the largest card in view the stage card, the next an expert tier holding each layer's next ids after the
//! stage's prefix, the host the rest), planned under one card budget no environment moves ([`TWO_BUDGET`] in the real
//! tier; in the fixture tier the `mid` budget, which leaves the tier the other half of the experts). Before the common
//! clauses it holds:
//! - `two_card_load`: the plan puts experts on the tier, and the load's tier opened on the second card beside the stage card;
//! - `two_card_eager_is_replay`: [`TWO_WALK`] steps of the walk, eager and as the captured graph's replays, read the same
//!   ids and logits bits, every logit finite, and the walk sent every tier layer a routed slot (or the run proves nothing
//!   there and fails by name);
//! - `two_card_first_token_near_union`: the first token's layers against the reference — the plan with every tier
//!   segment on the stage card ([`model::placement::union_plan`]), the stage card alone, opened first and released — each layer
//!   teacher-forced: the step runs that layer alone (`GpuModel::set_layer_span`) on the reference's `l_out` of the layer
//!   before it (`GpuModel::set_input_override`; layer 0 reads the embedding), keeps the reference's picks and keeps
//!   `ffn_moe_out` and `l_out` within [`TWO_STEP_BAND`]; the token sent the tier a slot on one of the tier's layers. The
//!   two loads compute the same rows with the same kernels on the same input; the tier's slots join as a second sum
//!   (`(card + tier) + host`), so the band, not bits. A free chain carries that regrouping through every later layer's
//!   q8_1 activations, whole code steps that no band of this clause's error model bounds.
//! - `two_card_prompt_near_union`: a prompt of one pass and [`TWO_PROMPT_PAST_PASS`] rows (the walk's ids), each layer
//!   run alone as the batch (`GpuModel::prefill_layer`) on both loads from a reset on the same rows, the reference's
//!   steps' outputs of the layer before it: the tier's batch join (its rows through the port, normed and summed beside
//!   the card's slots) against the union's card slots, every layer's `l_out` at the prompt's last row within
//!   [`TWO_REL_BAND`], and the batch sent every tier layer a slot. It holds the tier's batch path to the one-card batch,
//!   apart from the batch-against-steps band the common clauses read.
//!
//! The common clauses then run on the two-card load, and the tier must have served their batches (`two_card_batch_sent`).
//!
//! Cards: the arm asks the view for what it needs before any load, two cards in view and the stage card's usable bytes for
//! the union the reference holds ([`tier::union_lack_of`], the plan's own values). A view without them leaves every clause
//! of the arm, the common ones on the two-card load included, by name (`tier::defer_for_room`, `no-card-room`), in either
//! tier, and loads nothing.
//!
//! The node arm (`--node`, `just gate-gpu-musespark-node`, the fixture tier, the only arm it runs): plan (b′) with its tier card
//! on a node (`BLOOMERY_TIER_NODES`): a `tier_server` beside the gate serves the tier from this machine's own card, so the arm
//! needs one card in view, not two. The whole model on the card is read first and released: the no-node load, which holds every
//! expert the node plan splits between the stage card and the node. The gate then starts the node, hands its address over as
//! the tier node (`bloomery_levers::override_with`, once, after that load: a placed load with no tier refuses a node list), and
//! plans under the fixture's `mid` budget (the stage card half the experts, the node the rest). It holds:
//! - `node_tier_load`: the plan puts experts on the tier, the load's tier opened on the node the gate started, its experts by
//!   layer are the plan's, and the stage card keeps the experts the `mid` arm's card keeps;
//! - `node_tier_eager_is_replay`: [`TWO_WALK`] steps of the walk, eager and as the captured graph's replays, read the same ids and
//!   logits bits, every logit finite, and the walk sent every tier layer a routed slot;
//! - `node_tier_first_token_near_whole`: the first token's layers against the no-node load, each layer teacher-forced as
//!   `two_card_first_token_near_union` runs them, within [`TWO_STEP_BAND`]: the node computes the rows the stage card would, and
//!   only the experts' sum is grouped otherwise (`(card + tier) + host` against the card's);
//! - `node_tier_prompt_is_the_host_loads`: the prompt on the layers' inputs the no-node load's steps give, each layer alone as
//!   the batch, on this load and on the `mid` arm's (the card's half of the experts and the host the rest, no node): the batch's
//!   host serves the node's experts as it serves the host tier's and the stage card the same experts in both, so every layer's
//!   last row is the same bits. Against the no-node load the rows differ by the host's expert input rule (the card quantizes
//!   q8_1, the host Q8_K), so a node load's prompt is not asked to equal it;
//! - `node_tier_prompt_is_the_hosts`: that prompt and the common clauses' sent the node no batch service: a node serves the
//!   step, and the batch's host serves the node's experts (`HostTier::batch_slots`).
//!
//! The common clauses then run on the node load: its batches against its own steps, which cross the node.
//!
//! `--diag` runs no clause: on the whole-card load it prints, per layer, the batch's last-row output against the steps'
//! with both paths' routed ids (lengths 1, 7, 2,047), and the load's passes against passes of one row (7, 600, 2,600):
//! the narrowing that tells an exchanged expert from a pass defect.
//!
//! FAIL-first (box): the one-line mutants under the round's mutant folder name the clause each reddens.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_musespark_prefill", gate::run())
}

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the shared runner serves several gates; each takes the helpers it reads"
)]
#[path = "shared/e2e.rs"]
mod e2e;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the card the gate plans on; the placed arms read its spec"
)]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the engine serves several gates; each takes the calls it reads"
)]
#[path = "shared/lcpp_e2e.rs"]
mod lcpp_e2e;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the prefill clauses serve several gates; each takes the helpers they read"
)]
#[path = "shared/prefill_clauses.rs"]
mod prefill_clauses;

#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the gate starts and reaps its node; stdin, signals and the kill serve other gates"
)]
#[path = "shared/node_proc.rs"]
mod node_proc;
#[cfg(feature = "gpu")]
#[allow(
    dead_code,
    reason = "the two-card gates share the module; each takes the helpers it reads"
)]
#[path = "shared/tier_twocard.rs"]
mod tier_twocard;
#[cfg(feature = "gpu")]
mod gate {
    use crate::e2e::{cross_band, feed, finite, rel, same, selection_order, walk};
    use crate::gate_card;
    use crate::lcpp_e2e::{self, Dims, Seat, fnv1a};
    use crate::node_proc::{Proc, tier_server, tier_server_command};
    use crate::prefill_clauses::{self, Case};
    use crate::tier_twocard::{Sent, sent_between, sent_every, tier_counts, tier_on_card};
    use app::Prompt;
    use bloomery_gpu::arch::musespark::{Body, LayerNodes, LayerRoute};
    use bloomery_gpu::host::node::node_cards;
    use bloomery_gpu::host::served::TierBody;
    use bloomery_gpu::host::swap::Residency;
    use bloomery_gpu::host::xstream::{XLayer, XMode, XReport, XStreamBody};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::{Gpu, GpuError, GpuModel};
    use bloomery_gpu_gates::prompts::{SIGMA_LOGIT_V41, k_of};
    use bloomery_gpu_gates::tier::{self, Tag, Tier};
    use bloomery_gpu_gates::{
        GateError, bits_equal, checks_failed, open_split, ref_model_path, split_f32,
    };
    use model::arch::Arch;
    use model::arch::musespark::hparams::Hparams;
    use model::arch::musespark::place::{self, PlanInputs};
    use model::placement::union_plan;
    use model::placement::{Machine, Plan, PlanLevers, workstation};

    type Model = GpuModel<Body>;

    const NAME: &str = "gate_musespark_prefill";

    /// The recipe the model file is opened for.
    const RECIPE: &str = "gate-gpu-musespark-prefill";

    /// Positions of the cache beyond the longest prompt and the step after it.
    const CTX_SPARE: usize = 8;

    /// The ids of the steps' row the logits band reads.
    const TOP: usize = 40;

    /// The longest prompt's positions past the sliding window.
    const PAST_WINDOW: usize = 552;

    /// The ceiling of a layer output's relative distance between the batch and the steps, both tiers: layer 0 on the free
    /// chain (`prefill_near_steps`, `prefill_then_step_near_steps`) and every layer teacher-forced
    /// (`prefill_layers_forced_near_steps`).
    ///
    /// [derived]: one layer on the steps' own input differs from the steps' layer only at the dense sites (the Q8_0 sites'
    /// q8_1 input rounding the step's f32 gemv does not make, and the GEMM's sum order); ours differs from mainline's at
    /// those same sites (mainline rounds every dense input to q8_1 as the batch does) and in the experts' input rule
    /// besides (Q8_K against q8_1). The batch against the steps is therefore a subset of ours against mainline's in one
    /// teacher-forced layer, which `gate_musespark_e2e` holds at `l_out` to its `EXPERT_REL_BAND`, 0.055 (a router flip's
    /// rows reach 5.08 %, p999 4.85 %): a flip of one of the eight picks at a near tie moves a row by about its weight's
    /// share, and the same flip can land between the batch and the steps. Layer 0 of the free chain is such a layer (it
    /// reads the embedding on both paths). From layer 1 on a free layer reads the layers before it, and a flip upstream
    /// is carried on through every layer after: the free chain's distance there has no band an error model gives
    /// (`gate_minimaxm3_prefill` holds its layers past the forced prefix to none for the same reason), so those layers are
    /// held teacher-forced. A defect in a pass (a gain tiled to one row, an attention run reading another run's key
    /// counts, a hosted layer's host sums dropped) moves a layer's output by its own order, tens of percent.
    // PIN(2026-10-23): was 0.055 at every layer of the free chain; the fixture's `--diag` (whole-card, P = 1, 7, 2047)
    // read layer 0 at 2.2e-2..2.6e-2, layer 1 at 4.6e-2..1.0e-1, layer 2 at 6.8e-2..5.9e-1 and layer 3 at 0.36..1.0, the
    // routed ids exchanged from layer 2 or 3, and passes of 512 rows against passes of one row bit-identical at every layer
    // (no pass defect). The same 0.055 now holds each layer teacher-forced.
    const TAP_CEIL: f64 = 0.055;

    /// The host tier's configuration every placed load opens with: no lever is read, so one left in the runner's
    /// environment cannot move a clause (the e2e gate's).
    const HOST: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: false,
        lock: false,
        card_dontneed: false,
        r8: false,
        route_trace: false,
    };

    /// The stream arm's host configuration: [`HOST`] with the host set read in, which the residency machine's source
    /// stages its parts from.
    const HOST_STREAM: bloomery_levers::HostCfg = bloomery_levers::HostCfg {
        populate: true,
        ..HOST
    };

    /// The two-card arm's routed expert bytes on the stage card in the real tier: the stage card's budget is what it
    /// holds without experts plus these, and the tier's experts are capped by the same budget, so the reference's union
    /// is at most twice the budget [derived from the placement's rule; the arm prints it beside the stage card's usable
    /// bytes and leaves its clauses by name where the card has less].
    const TWO_STAGE_EXPERTS: u64 = 2 << 30;

    /// The two-card arm's own clauses, each declared where it runs and left by name, with the body's and the common ones
    /// the arm runs on the two-card load, where the cards in view lack what the arm needs.
    const TWO_CARD_LOAD: &str = "two_card_load";
    const TWO_CARD_EAGER: &str = "two_card_eager_is_replay";
    const TWO_CARD_FIRST: &str = "two_card_first_token_near_union";
    const TWO_CARD_PROMPT: &str = "two_card_prompt_near_union";
    const TWO_CARD_BATCH: &str = "two_card_batch_sent";
    const TWO_CARD_OWN: [&str; 5] = [
        TWO_CARD_LOAD,
        TWO_CARD_EAGER,
        TWO_CARD_FIRST,
        TWO_CARD_PROMPT,
        TWO_CARD_BATCH,
    ];

    /// The node arm's own clauses, each declared where it runs and left by name, with the body's and the common ones the arm
    /// runs on the node load, where the card in view cannot hold the no-node reference.
    const NODE_LOAD: &str = "node_tier_load";
    const NODE_EAGER: &str = "node_tier_eager_is_replay";
    const NODE_FIRST: &str = "node_tier_first_token_near_whole";
    const NODE_PROMPT: &str = "node_tier_prompt_is_the_host_loads";
    const NODE_BATCH: &str = "node_tier_prompt_is_the_hosts";
    const NODE_COMMON: &str = "node_tier_common_clauses_cross_the_node";
    const NODE_OWN: [&str; 6] = [
        NODE_LOAD,
        NODE_EAGER,
        NODE_FIRST,
        NODE_PROMPT,
        NODE_BATCH,
        NODE_COMMON,
    ];

    /// How long the node may take to exit once the load that used it closed.
    const NODE_CLOSE_WITHIN: std::time::Duration = std::time::Duration::from_secs(60);

    /// Steps of the walk the two-card and node arms run eager and as replays.
    const TWO_WALK: usize = 16;

    /// Rows of the two-card arm's prompt past one pass: a second, short pass.
    const TWO_PROMPT_PAST_PASS: usize = 7;

    /// The ceiling of the first token's `ffn_moe_out` and `l_out` relative distance between the two-card load and the
    /// reference, each layer run alone on the reference's own input.
    ///
    /// [derived]: on the same input bits both loads route the same picks and compute every routed slot's row with the
    /// same kernels on the same bytes (the stage card's `iq2_xxs_gate_up_sel` and `iq2_xxs_gemv_sel` over the q8_1 of the
    /// latent and of `h`, both quantizers `q8_1_quant_block`; the tier card's common leg the same entries, the same sm_86
    /// code) and norm each row by the same launch; only the weighted sum is grouped otherwise, `(card + tier) + host`
    /// against `card + host`, at most nine f32 terms: a relative 1e-6 at `ffn_moe_out`, which the step's latent up (an f32
    /// gemv on the Q8_0 rows), the scale-then-norm and the α/β mix carry linearly into `l_out`. A tier row off by a
    /// relative δ moves the sum by about a third of δ for one slot of eight (its weight's share of eight unit rows): a slot
    /// dropped, unnormed or another layer's by an eighth or more, a row from another q8 form of `h` by ~1e-2 (the q8 step
    /// `amax / 127` over √12, against `h`'s RMS), a scale or an epsilon off by a fraction of a percent by that fraction.
    /// The band, a decade over the regrouping, holds a tier row to ~3e-5.
    // PIN(2026-10-23): new; the clause was the free chain at 1e-3, which the real file's first run passed to layer 21
    // only (1.260e-3 at layer 22, growing to 1.593e-2 at layer 61, every layer's picks kept; the fixture's four layers
    // 2.478e-7): each layer's q8_1 activations turn the regrouping's ulps into whole code steps that every later layer
    // carries, which no band of this error model bounds over 62 layers. The forced run prints every layer's distance.
    const TWO_STEP_BAND: f64 = 1e-5;

    /// The ceiling of the prompt's last-row `l_out` relative distance between the two-card load and the reference, each
    /// layer run alone as the batch on the same rows.
    ///
    /// [derived]: as [`TWO_STEP_BAND`]'s, the slots' rows are the same bits (the pass runs the step's `_sel` launches over
    /// its slots, the tier's batch leg the same entries) and only the experts' sum is regrouped, 1e-6; but the pass's
    /// latent up reads that sum as q8 blocks of 32, so a regrouping ulp at a block's rounding boundary moves one of its
    /// 4,096 codes a step, `amax / 127` against the row's norm `64 · rms` with `amax` about `2.5 · rms`: 3e-4 of the
    /// latent up's row, carried into `l_out` at most whole. A value flips with odds about 5e-6 (an ulp over the step's
    /// half, at `|x| / amax` about 0.3), 0.02 flips a row, about one over the 62 layers. The band sits three times over
    /// one flip; a tier batch defect (its rows dropped, unnormed, another layer's) moves `l_out` by a slot's share.
    // PIN(2026-10-23): was 1e-3 on the free chain, shared with the first token's clause; the prompt clause never ran on
    // the real file (its taps re-armed under the capture were refused); the value kept, now on each layer forced, by
    // the latent up's flip it must pass.
    const TWO_REL_BAND: f64 = 1e-3;

    /// The free chain's ceiling: layer 0 only (the layers after it are held by the forced clause).
    fn tap_ceil(_tier: Tier, layer: usize) -> f64 {
        if layer == 0 { TAP_CEIL } else { f64::INFINITY }
    }

    fn tap_ceils(_tier: Tier) -> String {
        format!("ceiling {TAP_CEIL} at layer 0, none after it (the forced clause holds them)")
    }

    /// The load a clause run holds the batch on.
    #[derive(Clone, Copy)]
    enum Arm {
        /// The whole model on the card (`GpuModel::open`).
        Whole,
        /// The placed load at a card budget that keeps half the routed experts' bytes on the card.
        Mid,
        /// The gate plan's placed load.
        Plan,
        /// Plan (b′) on the two largest cards in view (module doc), run alone by `--two-card`.
        Two,
        /// Plan (b′) with the tier on a node of this machine's card (module doc), run alone by `--node`.
        Node,
        /// The `mid` budget under the residency machine with the prompt calls' expert stream on (module doc).
        Stream,
    }

    impl Arm {
        fn word(self) -> &'static str {
            match self {
                Arm::Whole => "whole",
                Arm::Mid => "mid",
                Arm::Plan => "plan",
                Arm::Two => "two",
                Arm::Node => "node",
                Arm::Stream => "stream",
            }
        }
    }

    /// The arms `tier` runs (module doc).
    fn arms(tier: Tier) -> &'static [Arm] {
        match tier {
            Tier::Fixture => &[Arm::Whole, Arm::Mid, Arm::Stream],
            Tier::Real => &[Arm::Plan],
        }
    }

    /// The prompt lengths around a sliding window of `window` positions in `tier` (module doc), ascending: the real tier
    /// leaves `W - 1` to the fixture, for every length costs its steps twice at the real file's decode rate.
    fn lengths(window: usize, tier: Tier) -> Vec<usize> {
        let mut v = vec![1, 7, window + 1, window + PAST_WINDOW];
        if tier == Tier::Fixture {
            v.push(window.saturating_sub(1));
        }
        v.retain(|&p| p > 0);
        v.sort_unstable();
        v.dedup();
        v
    }

    /// The model of `arm` at `ctx` positions, the taps armed under eager steps and then the step captured, so a replay
    /// and a prompt pass both write them and the step after a prompt is the seat's.
    fn open(arm: Arm, hp: &Hparams, ctx: usize) -> Result<Model, GateError> {
        let file = open_split(Arch::MuseSpark, RECIPE)?;
        let mut m = match arm {
            Arm::Whole => Model::open(Gpu::new()?, file, ctx)?,
            Arm::Two => return Err("the two-card arm opens through two_card".into()),
            Arm::Node => return Err("the node arm opens through node_card".into()),
            Arm::Mid | Arm::Plan | Arm::Stream => {
                let inputs = PlanInputs::read(&file)?;
                // The stream arm's plan counts the ring's term, the only card bytes its ring takes.
                let inputs = match arm {
                    Arm::Stream => {
                        let ring = place::stream_ring(&inputs.hp, &inputs.model)?;
                        inputs.streaming(ring)
                    }
                    Arm::Whole | Arm::Mid | Arm::Plan | Arm::Two | Arm::Node => inputs,
                };
                let machine =
                    workstation::placed_on(gate_card::card()?, hp.n_layer, workstation::SCRATCH);
                let ctx = u64::try_from(ctx)?;
                let levers = match arm {
                    Arm::Mid | Arm::Stream => stage_budget(&inputs, &machine, ctx, |all| all / 2)?,
                    Arm::Whole | Arm::Plan | Arm::Two | Arm::Node => PlanLevers::default(),
                };
                let plan = inputs.plan(&machine, ctx, &levers)?;
                println!(
                    "arm {} plan: card experts {} host experts {} (plan values)",
                    arm.word(),
                    plan.cards[0].experts,
                    plan.host.experts
                );
                match arm {
                    Arm::Stream => {
                        let residency = residency_of(&plan)?;
                        let mut m =
                            Model::open_placed_with(file, &plan, 0, HOST_STREAM, residency)?;
                        let (gpu, _, body) = m.body_parts(NAME)?;
                        body.set_xstream(gpu, XMode::Split)?;
                        println!("arm stream xstream={}", body.xstream_word());
                        m
                    }
                    Arm::Whole | Arm::Mid | Arm::Plan | Arm::Two | Arm::Node => {
                        Model::open_placed(file, &plan, 0, HOST)?
                    }
                }
            }
        };
        m.set_taps(true)?;
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        Ok(m)
    }

    /// The stream arm's residency over `plan`: half the smallest layer's card experts pinned (the seed is the plan's id
    /// prefix, not a frequency list), one spare a layer. Refused by name: a plan whose smallest layer of card experts
    /// holds no pinned expert beside the spare and the slot the machine swaps into.
    fn residency_of(plan: &Plan<'_>) -> Result<Residency, GateError> {
        let slots = plan
            .n_l
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .min()
            .ok_or("the stream arm's plan puts no expert on the card")?;
        let slots = usize::try_from(slots)?;
        let pinned = slots / 2;
        if pinned == 0 || slots < pinned + 2 {
            return Err(format!(
                "the stream arm's plan keeps {slots} card experts on its smallest layer: no residency machine of a \
                 pinned expert, a spare and its swap slot"
            )
            .into());
        }
        println!(
            "arm stream residency mid pinned={pinned} spares=1 over {slots} card experts (plan values)"
        );
        Ok(Residency::Mid { pinned, spares: 1 })
    }

    /// A card budget that keeps `share(e)` of the routed expert bytes on `machine`'s stage card, `e` the bytes it keeps
    /// unbudgeted: what the card holds without experts (its dense, rounding, cache and stage rows, set-aside and
    /// margin) plus that share.
    fn stage_budget(
        inputs: &PlanInputs,
        machine: &Machine,
        ctx: u64,
        share: impl Fn(u64) -> u64,
    ) -> Result<PlanLevers, GateError> {
        // The plan's floor holds the prompt reserve beside the dense, KV and set-aside bytes.
        let all = inputs.plan(machine, ctx, &PlanLevers::default())?;
        let (c, card) = (&all.cards[0], &machine.cards[0]);
        let no_experts = c.dense_bytes
            + c.rounding_bytes
            + c.kv_bytes
            + c.stage_rows_bytes
            + card
                .set_aside_bytes()
                .ok_or("the card's set-aside is unbounded")?
            + card.margin_bytes
            + inputs.prompt_reserve;
        Ok(PlanLevers {
            card_budget_bytes: Some(no_experts + share(c.expert_bytes)).into(),
            ..PlanLevers::default()
        })
    }

    /// What the tier was sent since `m` loaded; `None` on a load with no tier.
    fn counts(m: &Model) -> Result<Option<Sent>, GateError> {
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("the two-card load is not a placed load")?;
        Ok(tier_counts(placed.hybrid()))
    }

    /// Every layer's nodes after `tok` from a reset, an eager step with the taps armed.
    fn first_nodes(m: &mut Model, tok: u32) -> Result<Vec<LayerNodes>, GateError> {
        m.set_mode(StepMode::Eager);
        m.set_taps(true)?;
        m.reset()?;
        m.step(&[tok])?;
        Ok(m.nodes()?)
    }

    /// Every layer of `m` run alone on `tok` from a reset, an eager step of that layer's span with the taps armed, its
    /// input the reference's `l_out` of the layer before it in `want` (layer 0 reads the embedding): each layer's nodes.
    /// The span and the input override come off on the way out.
    fn forced_first_nodes(
        m: &mut Model,
        tok: u32,
        want: &[LayerNodes],
    ) -> Result<Vec<LayerNodes>, GateError> {
        m.set_mode(StepMode::Eager);
        let mut got = Vec::with_capacity(want.len());
        for l in 0..want.len() {
            m.set_layer_span(Some(l..l + 1))?;
            let input = l.checked_sub(1).map(|b| want[b].l_out.as_slice());
            m.set_input_override(l, input)?;
            m.reset()?;
            m.step(&[tok])?;
            let nodes = m.nodes()?.into_iter().nth(l);
            got.push(nodes.ok_or_else(|| format!("the forced step read no nodes of layer {l}"))?);
        }
        m.set_layer_span(None)?;
        m.set_input_override(0, None)?;
        Ok(got)
    }

    /// The prompt clause's rows (module doc): a prompt of one of the reference's passes and [`TWO_PROMPT_PAST_PASS`]
    /// rows of the walk's ids, the pass rows both loads run it in, and every layer's output at every position of the
    /// reference's steps over it, `(t · layers + l) · embd` a row: the rows each layer after the first reads.
    struct ForcedRows {
        ids: Vec<u32>,
        rows: usize,
        steps: Vec<f32>,
    }

    impl ForcedRows {
        /// The reference `r`'s steps from a reset, the taps armed.
        fn read(r: &mut Model, hp: &Hparams) -> Result<ForcedRows, GateError> {
            let rows = r.pass_rows()?;
            let ids = walk(rows + TWO_PROMPT_PAST_PASS, hp.n_vocab);
            r.reset()?;
            let mut steps = Vec::with_capacity(ids.len() * hp.n_layer * hp.n_embd);
            for &id in &ids {
                r.step(&[id])?;
                steps.extend(r.layer_outs()?);
            }
            Ok(ForcedRows { ids, rows, steps })
        }

        /// Every layer of `m` run alone as the batch over the ids in passes of the rows (`GpuModel::prefill_layer`) from
        /// a reset, on the steps' rows of the layer before it (layer 0 on the embedding), the taps armed: each layer's
        /// `l_out` at the prompt's last row.
        fn outs(
            &self,
            m: &mut Model,
            (layers, embd): (usize, usize),
        ) -> Result<Vec<Vec<f32>>, GateError> {
            let p = self.ids.len();
            let mut outs = Vec::with_capacity(layers);
            for l in 0..layers {
                let x: Option<Vec<f32>> = (l > 0).then(|| {
                    (0..p)
                        .flat_map(|t| {
                            self.steps[(t * layers + l - 1) * embd..][..embd]
                                .iter()
                                .copied()
                        })
                        .collect()
                });
                m.reset()?;
                m.prefill_layer(&self.ids, self.rows, (l, x.as_deref()))?;
                outs.push(m.layer_outs()?[l * embd..][..embd].to_vec());
            }
            Ok(outs)
        }
    }

    /// What the two-card arm leaves: the load, whether its own clauses held and the tier's layers.
    type TwoLoad = (Model, bool, Vec<usize>);

    /// The two-card arm's load (module doc): the reference's first token and prompt read and the reference released, then plan
    /// (b′) opened and its own clauses run; the load is left as [`open`] leaves one, the taps armed and the step
    /// captured. `Ok(Ok((m, pass, tier layers)))`; `Ok(Err(why))` where the cards in view lack two cards or the stage card's
    /// room for the union, before anything loads.
    fn two_card(
        hp: &Hparams,
        ctx: usize,
        tier_now: Tier,
    ) -> Result<Result<TwoLoad, String>, GateError> {
        if let Some(why) = tier::two_cards_lack()? {
            return Ok(Err(why));
        }
        let inputs = PlanInputs::read(&open_split(Arch::MuseSpark, RECIPE)?)?;
        let mut two = lcpp_e2e::Arm::Bp.machine(hp.n_layer, place::tier_batch(hp)?)?;
        place::stage_placed(&mut two)?;
        let ctx = u64::try_from(ctx)?;
        // The stage card alone, the reference's machine: the budget reads its unbudgeted plan, which a plan of the two
        // cards refuses where the stage holds every expert (the fixture's), the tier then holding none.
        let flat = Machine {
            tiers: Vec::new(),
            ..two.clone()
        };
        let levers = match tier_now {
            Tier::Fixture => stage_budget(&inputs, &flat, ctx, |all| all / 2)?,
            Tier::Real => stage_budget(&inputs, &flat, ctx, |all| all.min(TWO_STAGE_EXPERTS))?,
        };
        let plan = inputs.plan(&two, ctx, &levers)?;
        let on_tier: Vec<u64> = plan.tier_n_l.first().cloned().unwrap_or_default();
        let tier_layers: Vec<usize> = (0..hp.n_layer)
            .filter(|&l| on_tier.get(l).copied().unwrap_or(0) > 0)
            .collect();
        let tier_bytes = plan.cards.get(1).map_or(0, |c| c.expert_bytes);
        let stage = &plan.cards[0];
        let card = &plan.machine.cards[0];
        println!(
            "arm two plan (b′) under a card budget of {} at ctx {ctx}: stage {} experts ({} B), tier {} experts \
             ({tier_bytes} B) on {} layers, host {} experts ({} B); the reference's union {} B of the stage card {}'s {} \
             usable (plan values)",
            model::placement::card_budget::in_bytes(&levers.card_budget_bytes),
            stage.experts,
            stage.expert_bytes,
            plan.cards.get(1).map_or(0, |c| c.experts),
            tier_layers.len(),
            plan.host.experts,
            plan.host.expert_bytes,
            tier::union_bytes(&plan),
            card.name,
            card.usable_bytes,
        );
        if let Some(why) = tier::union_lack_of(&plan) {
            return Ok(Err(why));
        }
        let walk = walk(TWO_WALK, hp.n_vocab);
        let tok = *walk.first().ok_or("the walk is empty")?;

        // The reference: the union on the stage card alone, its first token, its steps over the prompt and each layer of the
        // prompt alone on them read, then released.
        let uplan = union_plan(&plan, &flat)?;
        let (want, rows_in, want_prompt) = {
            let mut r = Model::open_placed(open_split(Arch::MuseSpark, RECIPE)?, &uplan, 0, HOST)?;
            tier::opened_on_rank("reference", r.gpu(), 0)?;
            let want = first_nodes(&mut r, tok)?;
            let rows_in = ForcedRows::read(&mut r, hp)?;
            let want_prompt = rows_in.outs(&mut r, (hp.n_layer, hp.n_embd))?;
            (want, rows_in, want_prompt)
        };

        let t0 = std::time::Instant::now();
        let mut m = Model::open_placed(open_split(Arch::MuseSpark, RECIPE)?, &plan, 0, HOST)?;
        let mut pass = true;

        tier::sc(TWO_CARD_LOAD)?;
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("the two-card load is not a placed load")?;
        tier_on_card(
            "arm two load",
            t0,
            (&m.gpu().device_name()?, m.resident_bytes()),
            placed.hybrid(),
        )?;
        let loaded: Vec<u64> = (0..hp.n_layer)
            .map(|l| u64::try_from(placed.tier_k(l)))
            .collect::<Result<_, _>>()?;
        let planned: Vec<u64> = (0..hp.n_layer)
            .map(|l| on_tier.get(l).copied().unwrap_or(0))
            .collect();
        let load_ok = !tier_layers.is_empty() && loaded == planned;
        println!(
            "two_card_load: the tier holds experts on {} layers; the load's tier experts by layer {} the plan's {}",
            tier_layers.len(),
            if loaded == planned { "are" } else { "are not" },
            verdict(load_ok)
        );
        if loaded != planned {
            println!("  loaded {loaded:?}\n  planned {planned:?}");
        }
        pass &= load_ok;

        tier::sc(TWO_CARD_EAGER)?;
        m.set_taps(true)?;
        m.set_mode(StepMode::Eager);
        m.reset()?;
        let before = counts(&m)?;
        let eager = feed(&mut m, &walk)?;
        let sent_walk = sent_between(before, counts(&m)?);
        let before = counts(&m)?;
        let got = forced_first_nodes(&mut m, tok, &want)?;
        let sent_first = sent_between(before, counts(&m)?);
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        m.reset()?;
        let replay = feed(&mut m, &walk)?;
        let bits = finite(&eager) && same(&eager, &replay);
        println!(
            "two_card_eager_is_replay: {TWO_WALK} steps, eager and replay the same ids and logits bits, every logit \
             finite: {bits} {}",
            verdict(bits)
        );
        pass &= bits;
        pass &= sent_every(
            "two_card_eager_is_replay walk",
            sent_walk.as_ref(),
            &tier_layers,
            false,
        );

        tier::sc(TWO_CARD_FIRST)?;
        pass &= near_union(
            TWO_CARD_FIRST,
            &want,
            &got,
            sent_first.as_ref(),
            &tier_layers,
        );

        tier::sc(TWO_CARD_PROMPT)?;
        let before = counts(&m)?;
        // The taps the capture holds are the ones a pass writes: no re-arming under the capture.
        let got_prompt = rows_in.outs(&mut m, (hp.n_layer, hp.n_embd))?;
        let sent_prompt = sent_between(before, counts(&m)?);
        pass &= prompt_near_union(
            &want_prompt,
            &got_prompt,
            sent_prompt.as_ref(),
            &tier_layers,
        );
        Ok(Ok((m, pass, tier_layers)))
    }

    /// Plan (b′)'s machine with the stage on `stage` and the tier on the card of a node at `node`'s spec, laid out as the
    /// placement and [`place::stage_placed`] lay one out ([`workstation::placed_on`], the tier card, the host's prompt-batch
    /// rows). The node runs on this machine's own card here, one device twice, which the placement refuses; the plan holds
    /// the two budgets to the device instead (`stage_budget`'s card budget splits it).
    fn node_machine(
        hp: &Hparams,
        stage: model::placement::workstation::CardSpec,
        node: model::placement::workstation::CardSpec,
    ) -> Result<Machine, GateError> {
        let batch = place::tier_batch(hp)?;
        let mut machine = workstation::placed_on(stage, hp.n_layer, workstation::SCRATCH);
        machine.tiers.push(workstation::tier_card(node, batch));
        machine
            .host_mut()
            .reserves
            .push((workstation::TIER_BATCH_HOST_RESERVE.to_string(), batch.host));
        Ok(machine)
    }

    /// What the node arm leaves: the load, whether its own clauses held, the tier's layers and the node process the load's
    /// tier runs on, which outlives the load's use and is reaped after it closes.
    type NodeLoad = (Model, bool, Vec<usize>, Proc);

    /// The node arm's load (module doc): the no-node reference's first token and prompt read and the reference released,
    /// then the node started and named as the tier's node, plan (b′) opened over it and the arm's own clauses run; the load
    /// is left as [`open`] leaves one, the taps armed and the step captured. `Ok(Ok(load))`; `Ok(Err(why))` where the card in
    /// view cannot hold every expert for the reference, before anything loads.
    fn node_card(hp: &Hparams, ctx: usize) -> Result<Result<NodeLoad, String>, GateError> {
        if !bloomery_levers::tier_nodes()?.is_empty() {
            return Err(format!(
                "{} is set: the node arm sets it for its load alone",
                bloomery_levers::TIER_NODES
            )
            .into());
        }
        let inputs = PlanInputs::read(&open_split(Arch::MuseSpark, RECIPE)?)?;
        let ctx = u64::try_from(ctx)?;
        // The reference holds every expert on the card alone: the plan of that card must put none on the host.
        let card = gate_card::card()?;
        let one = workstation::placed_on(card, hp.n_layer, workstation::SCRATCH);
        let whole = inputs.plan(&one, ctx, &PlanLevers::default())?;
        if whole.host.experts > 0 {
            return Ok(Err(format!(
                "the no-node reference holds every expert on the card alone, whose plan puts {} experts on the host \
                 (card {}, {} usable bytes; plan values)",
                whole.host.experts,
                card.name,
                card.usable_bytes()
            )));
        }
        let ctx = usize::try_from(ctx)?;
        let walk = walk(TWO_WALK, hp.n_vocab);
        let tok = *walk.first().ok_or("the walk is empty")?;

        // The reference: the whole model on the card, its first token, its steps over the prompt and each layer of the
        // prompt alone on them read, then released.
        let (want, rows_in) = {
            let mut r = Model::open(Gpu::new()?, open_split(Arch::MuseSpark, RECIPE)?, ctx)?;
            let want = first_nodes(&mut r, tok)?;
            (want, ForcedRows::read(&mut r, hp)?)
        };
        // The prompt's reference: the load of this card's half of the experts and the host the rest, no node, which the
        // node's plan below divides further; its batches run the host's experts as the node load's do.
        let want_prompt = {
            let mut r = open(Arm::Mid, hp, ctx)?;
            if r.pass_rows()? != rows_in.rows {
                return Err(format!(
                    "the host-tier load's passes are {} rows, the reference's {}",
                    r.pass_rows()?,
                    rows_in.rows
                )
                .into());
            }
            rows_in.outs(&mut r, (hp.n_layer, hp.n_embd))?
        };

        let model = ref_model_path()?;
        let (node, addr) = tier_server(tier_server_command(Some(model.as_path()))?, "node")?;
        println!("node: tier_server pid {} serves on {addr}", node.id());
        for note in bloomery_levers::override_with(&[
            (bloomery_levers::TIER_NODES, addr.to_string()),
            (bloomery_levers::TIER_LINK, "tcp".to_string()),
        ])? {
            println!("{note}");
        }
        let ctx64 = u64::try_from(ctx)?;
        let (_, node_card) = node_cards(&[addr])?
            .into_iter()
            .next()
            .ok_or("the node's census holds no card")?;
        let machine = node_machine(hp, card, node_card)?;
        // The stage card alone, the budget's machine: the stage keeps half the experts and the node's card the rest.
        let flat = Machine {
            tiers: Vec::new(),
            ..machine.clone()
        };
        let levers = stage_budget(&inputs, &flat, ctx64, |all| all / 2)?;
        let plan = inputs.plan(&machine, ctx64, &levers)?;
        // The stage card keeps the experts the host-tier load's card keeps: that load's own plan, the same budget rule over
        // the same card with no tier or host reserve.
        let stage_same = {
            let mid = workstation::placed_on(card, hp.n_layer, workstation::SCRATCH);
            let mid_levers = stage_budget(&inputs, &mid, ctx64, |all| all / 2)?;
            plan.n_l == inputs.plan(&mid, ctx64, &mid_levers)?.n_l
        };
        let on_tier: Vec<u64> = plan.tier_n_l.first().cloned().unwrap_or_default();
        let tier_layers: Vec<usize> = (0..hp.n_layer)
            .filter(|&l| on_tier.get(l).copied().unwrap_or(0) > 0)
            .collect();
        println!(
            "arm node plan (b′) under a card budget of {} at ctx {ctx64}: stage {} experts ({} B), the node's tier {} experts \
             ({} B) on {} layers, host {} experts (plan values)",
            model::placement::card_budget::in_bytes(&levers.card_budget_bytes),
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.cards.get(1).map_or(0, |c| c.experts),
            plan.cards.get(1).map_or(0, |c| c.expert_bytes),
            tier_layers.len(),
            plan.host.experts,
        );

        let t0 = std::time::Instant::now();
        let mut m = Model::open_placed(open_split(Arch::MuseSpark, RECIPE)?, &plan, 0, HOST)?;
        let mut pass = true;

        tier::sc(NODE_LOAD)?;
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("the node load is not a placed load")?;
        let tier = placed
            .hybrid()
            .tiers()
            .first()
            .ok_or("the node load holds no tier card")?;
        println!(
            "arm node load in {:.1} s: stage {} ({} B), tier {} experts on the node at {:?} ({} B)",
            t0.elapsed().as_secs_f64(),
            m.gpu().device_name()?,
            m.resident_bytes(),
            tier.set().experts(),
            tier.node(),
            tier.resident_bytes(),
        );
        let on_node = tier.node() == Some(addr);
        let loaded: Vec<u64> = (0..hp.n_layer)
            .map(|l| u64::try_from(placed.tier_k(l)))
            .collect::<Result<_, _>>()?;
        let planned: Vec<u64> = (0..hp.n_layer)
            .map(|l| on_tier.get(l).copied().unwrap_or(0))
            .collect();
        let rows_ok = m.pass_rows()? == rows_in.rows;
        let load_ok =
            !tier_layers.is_empty() && loaded == planned && on_node && rows_ok && stage_same;
        println!(
            "{NODE_LOAD}: the tier holds experts on {} layers; it runs on the node {addr}: {on_node}; the load's tier experts by \
             layer {} the plan's; its pass rows {} equal the reference's {}: {rows_ok}; its stage experts by layer are the \
             host-tier load's: {stage_same} {}",
            tier_layers.len(),
            if loaded == planned { "are" } else { "are not" },
            m.pass_rows()?,
            rows_in.rows,
            verdict(load_ok)
        );
        if loaded != planned {
            println!("  loaded {loaded:?}\n  planned {planned:?}");
        }
        pass &= load_ok;

        tier::sc(NODE_EAGER)?;
        m.set_taps(true)?;
        m.set_mode(StepMode::Eager);
        m.reset()?;
        let before = counts(&m)?;
        let eager = feed(&mut m, &walk)?;
        let sent_walk = sent_between(before, counts(&m)?);
        let before = counts(&m)?;
        let got = forced_first_nodes(&mut m, tok, &want)?;
        let sent_first = sent_between(before, counts(&m)?);
        m.set_mode(StepMode::Graph);
        m.capture_step()?;
        m.reset()?;
        let replay = feed(&mut m, &walk)?;
        let bits = finite(&eager) && same(&eager, &replay);
        println!(
            "{NODE_EAGER}: {TWO_WALK} steps, eager and replay the same ids and logits bits, every logit finite: {bits} {}",
            verdict(bits)
        );
        pass &= bits;
        pass &= sent_every(
            "node_tier_eager_is_replay walk",
            sent_walk.as_ref(),
            &tier_layers,
            false,
        );

        tier::sc(NODE_FIRST)?;
        pass &= near_union(NODE_FIRST, &want, &got, sent_first.as_ref(), &tier_layers);

        tier::sc(NODE_PROMPT)?;
        let before = counts(&m)?;
        // The taps the capture holds are the ones a pass writes: no re-arming under the capture.
        let got_prompt = rows_in.outs(&mut m, (hp.n_layer, hp.n_embd))?;
        let sent_prompt = sent_between(before, counts(&m)?);
        pass &= prompt_is_the_host_loads(NODE_PROMPT, &want_prompt, &got_prompt);

        tier::sc(NODE_BATCH)?;
        pass &= node_no_batch(NODE_BATCH, sent_prompt.as_ref(), true);
        Ok(Ok((m, pass, tier_layers, node)))
    }

    /// Whether `sent`, what a span sent the node tier, holds no batch service of the node and, when `no_hits`, no routed slot
    /// either: a node serves the step, and the batch's host serves its experts (module doc). Prints the counters.
    fn node_no_batch(what: &str, sent: Option<&Sent>, no_hits: bool) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what}: the load holds no tier card");
            return false;
        };
        let hits: u64 = sent.hits.iter().sum();
        let st = sent.stats;
        let ok = st.served == 0 && st.settles == 0 && (!no_hits || hits == 0);
        println!(
            "{what}: the span sent the node {hits} routed slots and {} batch services ({} settles): the batch's host serves \
             the node's experts {}",
            st.served,
            st.settles,
            verdict(ok)
        );
        ok
    }

    /// The first token's layers of the two-card load `got`, each run alone on the reference's input, against the
    /// reference's `want` (module doc): every layer keeps the reference's picks (the same input bits route the same) and
    /// its distance within [`TWO_STEP_BAND`]; the token must have sent the tier a slot on one of the tier's layers.
    fn near_union(
        clause: &str,
        want: &[LayerNodes],
        got: &[LayerNodes],
        sent: Option<&Sent>,
        tier_layers: &[usize],
    ) -> bool {
        if want.len() != got.len() {
            println!(
                "{clause}: {} layers read, the reference {} FAIL",
                got.len(),
                want.len()
            );
            return false;
        }
        let hit = |l: usize| {
            sent.and_then(|s| l.checked_sub(s.first).and_then(|i| s.hits.get(i)).copied())
                .unwrap_or(0)
        };
        let sorted = |ids: &[u32]| {
            let mut v = ids.to_vec();
            v.sort_unstable();
            v
        };
        let (mut ok, mut each, mut reached) = (true, Vec::with_capacity(want.len()), Vec::new());
        for (l, (w, g)) in want.iter().zip(got).enumerate() {
            if sorted(&w.ffn_moe_topk) != sorted(&g.ffn_moe_topk) {
                println!(
                    "{clause}: layer {l} picks {:?} on the reference's input, the reference {:?} FAIL",
                    g.ffn_moe_topk, w.ffn_moe_topk
                );
                ok = false;
            }
            let r = rel(&g.ffn_moe_out, &w.ffn_moe_out).max(rel(&g.l_out, &w.l_out));
            if r > TWO_STEP_BAND {
                println!("{clause}: layer {l} relative distance {r:.3e} past {TWO_STEP_BAND} FAIL");
                ok = false;
            }
            each.push(r);
            if tier_layers.contains(&l) && hit(l) > 0 {
                reached.push(l);
            }
        }
        let sent_tier = !reached.is_empty();
        println!(
            "{clause}: {}; the token sent the tier a slot on layers {reached:?} {}",
            distances(&each, TWO_STEP_BAND),
            verdict(ok && sent_tier)
        );
        if !sent_tier {
            println!(
                "FAIL precondition {clause}: the token sent the tier no slot on any layer; the clause proves nothing"
            );
        }
        ok && sent_tier
    }

    /// The prompt's last-row `l_out` of the two-card load `got` against the reference's `want`, each layer run alone as
    /// the batch on the same rows (module doc): its `ffn_inp` is the attention's on those rows, no tier's, and its picks
    /// are not tapped. The batch must have sent every tier layer a slot and the tier served it.
    fn prompt_near_union(
        want: &[Vec<f32>],
        got: &[Vec<f32>],
        sent: Option<&Sent>,
        tier_layers: &[usize],
    ) -> bool {
        const CLAUSE: &str = TWO_CARD_PROMPT;
        if want.len() != got.len() {
            println!(
                "{CLAUSE}: {} layers read, the reference {} FAIL",
                got.len(),
                want.len()
            );
            return false;
        }
        let each: Vec<f64> = want.iter().zip(got).map(|(w, g)| rel(g, w)).collect();
        let mut ok = true;
        for (l, &r) in each.iter().enumerate() {
            if r > TWO_REL_BAND {
                println!("{CLAUSE}: layer {l} relative distance {r:.3e} past {TWO_REL_BAND} FAIL");
                ok = false;
            }
        }
        println!(
            "{CLAUSE}: {} {}",
            distances(&each, TWO_REL_BAND),
            verdict(ok)
        );
        ok & sent_every(CLAUSE, sent, tier_layers, true)
    }

    /// The node load's prompt rows `got` against the host-tier load's `want`, each layer alone as the batch on the same
    /// rows (module doc): the batch's host serves the node's experts as it serves the host tier's, and the stage card
    /// the same experts in both, so every layer's last-row `l_out` has the same bits. Prints each layer's relative
    /// distance beside the verdict.
    fn prompt_is_the_host_loads(clause: &str, want: &[Vec<f32>], got: &[Vec<f32>]) -> bool {
        if want.len() != got.len() {
            println!(
                "{clause}: {} layers read, the host-tier load's {} FAIL",
                got.len(),
                want.len()
            );
            return false;
        }
        let each: Vec<f64> = want.iter().zip(got).map(|(w, g)| rel(g, w)).collect();
        let ok = want.iter().zip(got).all(|(w, g)| bits_equal(w, g));
        println!(
            "{clause}: {} layers, last-row l_out relative distances [{}] against the host-tier load's: bits equal {ok} {}",
            each.len(),
            each.iter()
                .map(|r| format!("{r:.3e}"))
                .collect::<Vec<_>>()
                .join(" "),
            verdict(ok)
        );
        ok
    }

    /// Each layer's relative distance in layer order, then the worst and its layer against `band`.
    fn distances(each: &[f64], band: f64) -> String {
        let (worst, at) =
            each.iter().enumerate().fold(
                (0.0_f64, 0),
                |(w, at), (l, &r)| if r > w { (r, l) } else { (w, at) },
            );
        format!(
            "{} layers each on the reference's input, distances [{}], the worst {worst:.3e} at layer {at} (band {band})",
            each.len(),
            each.iter()
                .map(|r| format!("{r:.3e}"))
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    fn verdict(pass: bool) -> &'static str {
        if pass { "ok" } else { "FAIL" }
    }

    /// The arm the command line runs alone: `--two-card` or `--node`; no argument, the tier's arms. Any other argument is
    /// refused by name.
    fn alone_arg() -> Result<Option<Arm>, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        match args.as_slice() {
            [] => Ok(None),
            [a] if a == "--two-card" => Ok(Some(Arm::Two)),
            [a] if a == "--node" => Ok(Some(Arm::Node)),
            other => Err(format!(
                "{NAME}: arguments {other:?}: the gate takes none, --two-card or --node"
            )
            .into()),
        }
    }

    /// The model behind the clauses' calls.
    struct Muse {
        m: Model,
        dims: Dims,
        /// The stream arm's tally of its calls' stream records; `None` on the other arms.
        stream: Option<StreamTally>,
    }

    impl Muse {
        /// The last call's logits row.
        fn row(&self) -> Result<Vec<f32>, GateError> {
            let mut row = vec![0.0; self.dims.vocab];
            self.m.logits_into(&mut row)?;
            Ok(row)
        }

        /// Layer `l` alone over `ids` in passes of `rows` (`GpuModel::prefill_layer`); on the stream arm the call's
        /// stream records, its ring passes and the host's left-out slots are taken into the tally.
        fn prefill_layer(
            &mut self,
            ids: &[u32],
            rows: usize,
            at: (usize, Option<&[f32]>),
        ) -> Result<(), GateError> {
            let before = match self.stream {
                Some(_) => excluded(&self.m)?,
                None => 0,
            };
            self.m.prefill_layer(ids, rows, at)?;
            if self.stream.is_some() {
                let left_out = excluded(&self.m)? - before;
                let body = self.m.body_parts(NAME)?.2;
                let (xlayers, xend) = body.take_xstream_records();
                let joined = body.take_stream_joined();
                if let Some(t) = self.stream.as_mut() {
                    t.call(&xlayers, xend.as_ref(), &joined, left_out);
                }
            }
            Ok(())
        }
    }

    /// The host slots the batch services left out of their union since `m` loaded (their expert in the service's
    /// exclusion set: streamed to the card).
    fn excluded(m: &Model) -> Result<u64, GateError> {
        let placed = m
            .body(NAME)?
            .placed()
            .ok_or("the stream arm's load is not a placed load")?;
        Ok(placed.hybrid().stats().batch_excluded_slots)
    }

    /// The stream arm's tally over the forced clause's calls (`prefill_stream_streamed`).
    #[derive(Default)]
    struct StreamTally {
        /// Calls taken, and of them those that streamed an expert.
        calls: usize,
        streaming: usize,
        /// The streamed experts the records name, over every call.
        streamed: usize,
        /// Every disagreement, named by its call.
        bad: Vec<String>,
    }

    impl StreamTally {
        /// One call's records `xlayers` and `xend`, its ring passes `joined` (walk, layer, experts joined) and the host
        /// slots its services left out: each streamed layer's ring pass joined its streamed experts and no other did,
        /// the end counts the layers' sum, and the host left out the streamed columns.
        fn call(
            &mut self,
            xlayers: &[(usize, XLayer)],
            xend: Option<&XReport>,
            joined: &[(usize, usize, usize)],
            left_out: u64,
        ) {
            let call = self.calls;
            self.calls += 1;
            let streamed: usize = xlayers.iter().map(|(_, x)| x.streamed).sum();
            let columns: u64 = xlayers.iter().map(|(_, x)| x.streamed_columns).sum();
            let mut want: Vec<(usize, usize, usize)> = xlayers
                .iter()
                .filter(|(_, x)| x.streamed > 0)
                .map(|(w, x)| (*w, x.layer, x.streamed))
                .collect();
            let mut got: Vec<(usize, usize, usize)> =
                joined.iter().copied().filter(|j| j.2 > 0).collect();
            want.sort_unstable();
            got.sort_unstable();
            if got != want {
                self.bad.push(format!(
                    "call {call}: the records stream {want:?}, the ring passes joined {got:?}"
                ));
            }
            let end = xend.map_or(0, |e| e.streamed);
            if end != streamed {
                self.bad.push(format!(
                    "call {call}: the end counts {end} streamed, the layers {streamed}"
                ));
            }
            if left_out != columns {
                self.bad.push(format!(
                    "call {call}: the host left out {left_out} slots, the stream moved {columns} columns"
                ));
            }
            if streamed > 0 {
                self.streaming += 1;
            }
            self.streamed += streamed;
        }
    }

    impl Seat for Muse {
        fn dims(&self) -> Dims {
            self.dims
        }

        fn reset(&mut self) -> Result<(), GateError> {
            Ok(self.m.reset()?)
        }

        fn pos(&self) -> u32 {
            self.m.pos()
        }

        fn step(&mut self, id: u32) -> Result<Vec<f32>, GateError> {
            self.m.step(&[id])?;
            self.row()
        }

        /// The serve seat's call (`app::Prompt for Body`): passes of the load's rows.
        fn prompt(&mut self, ids: &[u32]) -> Result<Vec<f32>, GateError> {
            <Body as Prompt>::prompt(&mut self.m, ids)?;
            self.row()
        }

        /// Group 1 only: the body walks one pass at a time.
        fn set_group(&mut self, group: usize) -> Result<(), GateError> {
            if group == 1 {
                return Ok(());
            }
            Err(format!(
                "the musespark body has no prompt group (one pass walk at a time); asked {group}"
            )
            .into())
        }

        /// Every layer's `l_out` tap: the last row a pass or a step left.
        fn taps(&mut self) -> Result<Vec<f32>, GateError> {
            Ok(self.m.layer_outs()?)
        }

        fn stores(&mut self) -> Result<Vec<(String, u64)>, GateError> {
            Ok(self
                .m
                .store_planes()?
                .into_iter()
                .map(|(name, bytes)| (name, fnv1a(&bytes)))
                .collect())
        }

        fn fault(&mut self) -> Result<(), GateError> {
            if let Some(fault) = self.m.poisoned() {
                return Err(format!("the model is poisoned by {fault:?}").into());
            }
            let (gpu, _, _) = self.m.body_parts(NAME)?;
            match gpu.fault()? {
                None => Ok(()),
                Some(fault) => Err(format!("the card's fault word holds {fault:?}").into()),
            }
        }
    }

    /// The rows of the pass cut the cut clause runs against the load's: odd, under one 32-row attention run and one
    /// 64-row pass alignment, so every length's passes end mid-run.
    const CUT: usize = 7;

    /// A clause of the body's own: the seat and the lengths it runs.
    type Own = fn(&mut Muse, &[usize]) -> Result<bool, GateError>;

    /// The body's own clauses beside the common ones.
    const CUT_CLAUSE: &str = "prefill_pass_cut_is_bits";
    const FORCED_CLAUSE: &str = "prefill_layers_forced_near_steps";
    const STREAM_CLAUSE: &str = "prefill_stream_streamed";
    const SPAN_CLAUSE: &str = "prefill_span_is_eager_only";

    /// The body's own clauses in the order they run; the stream arm runs the last two, the forced clause and the span
    /// clause, alone.
    const OWN: [(&str, Own); 3] = [
        (CUT_CLAUSE, pass_cut_is_bits),
        (FORCED_CLAUSE, forced_near_steps),
        (SPAN_CLAUSE, span_is_eager_only),
    ];

    /// Whether `r` is the refusal `what` names as a missing state.
    fn refused_by(r: Result<(), GpuError>, what: &str) -> bool {
        matches!(r, Err(GpuError::State { what: w, .. }) if w == what)
    }

    /// `prefill_span_is_eager_only`: in graph mode a layer span and an input override are refused by their names (a
    /// captured step replays the walk it recorded, so neither would move a replay); in eager mode both are taken and
    /// cleared. The load is left as [`open`] leaves one, the step captured. FAIL-first: a body that takes either in
    /// graph mode returns `Ok` and the clause is red.
    fn span_is_eager_only(seat: &mut Muse, _lengths: &[usize]) -> Result<bool, GateError> {
        let m = &mut seat.m;
        m.set_mode(StepMode::Graph);
        let span = refused_by(m.set_layer_span(Some(0..1)), "musespark::set_layer_span");
        let input = refused_by(
            m.set_input_override(0, None),
            "musespark::set_input_override",
        );
        m.set_mode(StepMode::Eager);
        let eager = m.set_layer_span(Some(0..1)).is_ok()
            && m.set_layer_span(None).is_ok()
            && m.set_input_override(0, None).is_ok();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        m.capture_step()?;
        println!(
            "{SPAN_CLAUSE}: graph mode refuses the span {span} and the input override {input}; eager mode takes both {eager}"
        );
        Ok(span && input && eager)
    }

    /// `prefill_pass_cut_is_bits`: at every length, the prompt in passes of [`CUT`] rows gives the load's passes'
    /// logits row, every layer's last-row output and every store bit for bit — each row's launches are the same
    /// arithmetic whatever pass it lands in (the GEMM's sum per output is its column's alone, the attention runs and
    /// the appends one row each, the host's union a column's single-column bits), so a row or a key that a pass reads
    /// from the wrong place moves a bit at the layer it happens in. It holds every layer, where the free chain's band
    /// clauses hold layer 0 only.
    fn pass_cut_is_bits(seat: &mut Muse, lengths: &[usize]) -> Result<bool, GateError> {
        let rows = seat.m.pass_rows()?;
        let mut ok = true;
        for &p in lengths {
            let ids = crate::e2e::walk(p, seat.dims.vocab);
            let mut left = Vec::with_capacity(2);
            for cut in [rows, CUT] {
                seat.m.reset()?;
                seat.m.prefill_with(&ids, cut)?;
                left.push((seat.row()?, seat.taps()?, seat.stores()?));
            }
            let (a, b) = (&left[0], &left[1]);
            let first =
                a.1.chunks(seat.dims.embd)
                    .zip(b.1.chunks(seat.dims.embd))
                    .position(|(x, y)| !bits_equal(x, y));
            let pass = bits_equal(&a.0, &b.0) && first.is_none() && a.2 == b.2;
            println!(
                "{CUT_CLAUSE}: P={p} passes of {rows} against {CUT}: logits {} layer outputs {}{} stores {} {}",
                bloomery_gpu_gates::verdict(bits_equal(&a.0, &b.0)),
                bloomery_gpu_gates::verdict(first.is_none()),
                first.map_or(String::new(), |l| format!(" (first layer {l})")),
                bloomery_gpu_gates::verdict(a.2 == b.2),
                bloomery_gpu_gates::verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    /// The prompt length the real tier runs the body's own clauses at beside `7`: two passes of the load's rows, under the
    /// window (the fixture holds the window's lengths on the same code), for the forced clause holds the steps' outputs
    /// of every position in host memory (`P · layers · embd` f32) and both clauses run
    /// passes the real file's host tier serves.
    const REAL_OWN: usize = 600;

    /// The lengths the body's own clauses run in `tier`: all of them on the fixture, `7` and [`REAL_OWN`] on the real tier.
    fn own_lengths(lengths: &[usize], tier: Tier) -> Vec<usize> {
        match tier {
            Tier::Fixture => lengths.to_vec(),
            Tier::Real => vec![7, REAL_OWN],
        }
    }

    /// The relative error of the batch's router logits against the steps': `max |Δlogit| / peak |logit|` over a row's
    /// experts, which [`swap_of`] turns into the band inside which a pair's measured score drift may stand.
    ///
    /// [derived]: the batch's router is the step's gemv lane for lane (`enqueue_router_rows` runs the f32 tile GEMM bit
    /// for bit as the per-token gemv on one input), so its logits differ from the steps' by the input alone: the layer's
    /// `ffn_norm` row of `ffn_inp`, which the two paths round apart at the attention half's dense sites (the Q8_0 sites'
    /// q8_1 input rounding the step's f32 gemv does not make, and the GEMM's sum order), a subset of the roundings ours
    /// and mainline's logits differ by. `gate_musespark_e2e` holds that error against mainline's (`nodes_near_mainlines`:
    /// eps, the largest over the layer-positions that kept mainline's set) and grows it for its longer walk by
    /// `sqrt(ln N / ln N_a)` (`window_past_mainlines`). A score is `σ(z)`, so a logit error `Δz` moves it by
    /// `σ'(z) · Δz` with `σ'(z) = s (1 − s)`; the two scores of an exchanged pair move apart by at most
    /// `eps · peak · (σ'_dropped + σ'_gained)` (`e2e::cross_band`), which bounds the pair's measured drift.
    // PIN(2026-10-23): 0.051, the e2e fixture's measured eps 0.0398 (118 kept sets) grown by sqrt(ln 2560 / ln 118) for
    // its window walk as that clause grows it; the first run prints every exchange's gap, drift and band.
    const ROUTER_EPS: f32 = 0.051;

    /// The exchange of experts between the steps' kept set at a row and the batch's, read off both paths' own scores.
    struct Swap {
        /// The steps' last kept selection value less the best of the others: the margin the exchange crossed.
        margin: f32,
        /// The exchanged pair (the id the steps keep and the batch dropped, the id the batch kept instead) that stands
        /// furthest outside its band, with the steps' gap between the two selection values, the measured drift that
        /// crossed it (the steps' gap plus the batch's own, which is not negative where the batch kept the pair's
        /// gained id) and the band the rounding model allows the drift.
        pair: (u32, u32),
        gap: f32,
        drift: f32,
        band: f32,
    }

    impl Swap {
        /// Whether every exchanged pair's drift sits inside its band (a NaN drift or band is outside).
        fn inside(&self) -> bool {
            self.drift <= self.band
        }
    }

    /// The exchange `batch`'s kept ids make against the steps' `steps` at one row, `bias` the layer's selection bias:
    /// both paths' selection values are their scores (`steps`' from its logits, `batch`'s from the scores the pass left)
    /// plus the bias. The ids the steps keep and the batch drops pair with the ones it keeps instead in the steps'
    /// selection order (the lowest dropped with the highest gained, the pairing that asks the least of each crossing), a
    /// pair's drift is how far the batch's scores moved the gained id above the dropped one against the steps', and the
    /// pair standing furthest outside its band is the one [`Swap::inside`] judges. A row that cannot be read as a routing
    /// (a bias or scores of another length than the logits, a value not finite, an id past the experts, sets of unequal
    /// size or with an id twice, no id exchanged) and a batch whose kept set is not the top of its own selection values are
    /// named errors, not pairs.
    fn swap_of(steps: &LayerRoute, batch: &LayerRoute, bias: &[f32]) -> Result<Swap, GateError> {
        let n = steps.logits.len();
        let named = |why: &str| -> GateError {
            format!(
                "the routing of a row: {why} ({n} logits, {} scores, {} bias values, steps kept {:?}, batch kept {:?})",
                batch.probs.len(),
                bias.len(),
                steps.ids,
                batch.ids
            )
            .into()
        };
        let sel = selection_order(bias, &steps.logits);
        let theirs: Vec<f32> = batch.probs.iter().zip(bias).map(|(&s, &b)| s + b).collect();
        if bias.len() != n
            || batch.probs.len() != n
            || sel.values.iter().chain(&theirs).any(|v| !v.is_finite())
        {
            return Err(named(
                "a bias or scores of another length or a selection value not finite",
            ));
        }
        if batch.ids.len() != steps.ids.len()
            || steps.ids.iter().chain(&batch.ids).any(|&e| e as usize >= n)
            || sorted(&steps.ids).windows(2).any(|w| w[0] == w[1])
            || sorted(&batch.ids).windows(2).any(|w| w[0] == w[1])
        {
            return Err(named(
                "kept sets of unequal size, with an id twice or an id past the experts",
            ));
        }
        // The batch's own selection: what it kept is no less than what it left.
        let kept_low = batch
            .ids
            .iter()
            .map(|&e| theirs[e as usize])
            .fold(f32::INFINITY, f32::min);
        let left_high = (0..n)
            .filter(|&e| !batch.ids.contains(&(e as u32)))
            .map(|e| theirs[e])
            .fold(f32::NEG_INFINITY, f32::max);
        if kept_low < left_high {
            return Err(named(
                "the batch's kept set is not the top of its own selection values: a selection defect",
            ));
        }
        let mut dropped: Vec<u32> = steps
            .ids
            .iter()
            .copied()
            .filter(|e| !batch.ids.contains(e))
            .collect();
        let mut gained: Vec<u32> = batch
            .ids
            .iter()
            .copied()
            .filter(|e| !steps.ids.contains(e))
            .collect();
        let by_value =
            |a: &u32, b: &u32| sel.values[*a as usize].total_cmp(&sel.values[*b as usize]);
        dropped.sort_by(by_value);
        gained.sort_by(|a, b| by_value(b, a));
        let move_of = |e: u32| theirs[e as usize] - sel.values[e as usize];
        let (pair, gap, drift, band) = dropped
            .iter()
            .zip(&gained)
            .map(|(&kept, &other)| {
                (
                    (kept, other),
                    sel.values[kept as usize] - sel.values[other as usize],
                    move_of(other) - move_of(kept),
                    cross_band(
                        ROUTER_EPS,
                        sel.z_max,
                        sel.scores[kept as usize],
                        sel.scores[other as usize],
                    ),
                )
            })
            .max_by(|a, b| (a.2 - a.3).total_cmp(&(b.2 - b.3)))
            .ok_or_else(|| named("no id exchanged"))?;
        let last = steps
            .ids
            .iter()
            .map(|&e| sel.values[e as usize])
            .fold(f32::INFINITY, f32::min);
        let next = (0..u32::try_from(n)?)
            .filter(|e| !steps.ids.contains(e))
            .map(|e| sel.values[e as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        Ok(Swap {
            margin: last - next,
            pair,
            gap,
            drift,
            band,
        })
    }

    /// `ids` in id order: the kept set whatever the rank order the router wrote it in.
    fn sorted(ids: &[u32]) -> Vec<u32> {
        let mut v = ids.to_vec();
        v.sort_unstable();
        v
    }

    /// Every layer's selection bias (`exp_probs_b`) of `n_expert` experts, read from the model file the gate runs on.
    fn router_bias(layers: usize, n_expert: usize) -> Result<Vec<Vec<f32>>, GateError> {
        let file = open_split(Arch::MuseSpark, RECIPE)?;
        (0..layers)
            .map(|l| split_f32(&file, &format!("blk.{l}.exp_probs_b.bias"), n_expert))
            .collect()
    }

    /// `prefill_layers_forced_near_steps`: from a reset, the prompt runs as steps and every position's output of every
    /// layer is kept, with each layer's routing (kept ids and router logits) at the last row of every pass; then each
    /// layer alone runs the whole prompt in passes (`GpuModel::prefill_layer`) on the rows the steps' layer before it
    /// left (layer 0 on the embedding), from empty stores. Each layer reads the steps' own input, so the layer's rounding
    /// at its dense sites is the only thing between the two paths, with none of the layers before it carried on: the
    /// band `gate_musespark_e2e` holds one teacher-forced layer to. The layer's kept experts at the last row of every
    /// pass are read on both paths:
    /// - the same set: the output is held within [`TAP_CEIL`] of the steps' at that position;
    /// - another set: the output is not compared (an exchanged expert moves a row by its share, which no rounding band
    ///   holds), and the exchange must be the batch's own scores' ([`swap_of`]: its kept set the top of its own selection
    ///   values, every exchanged pair's measured drift inside the band; the steps' margin and the pair's gap, drift and
    ///   band are printed). A layer where more than half of the judged rows exchange fails by name: the excuse of a defect
    ///   that moves every row's router input is not a near tie's.
    fn forced_near_steps(seat: &mut Muse, lengths: &[usize]) -> Result<bool, GateError> {
        let Dims {
            layers,
            embd,
            vocab,
        } = seat.dims;
        let rows = seat.m.pass_rows()?;
        let mut ok = true;
        // The selection biases, read from the file at the first row whose kept sets differ.
        let mut bias: Vec<Vec<f32>> = Vec::new();
        // Per layer over every length: the rows judged and the rows whose kept set is not the steps'.
        let (mut judged, mut exchanged) = (vec![0usize; layers], vec![0usize; layers]);
        for &p in lengths {
            let ids = crate::e2e::walk(p, vocab);
            seat.m.reset()?;
            let mut steps = Vec::with_capacity(p * layers * embd);
            let mut routes: Vec<Vec<LayerRoute>> = Vec::with_capacity(p.div_ceil(rows));
            for (t, &id) in ids.iter().enumerate() {
                seat.m.step(&[id])?;
                steps.extend(seat.m.layer_outs()?);
                if (t + 1).is_multiple_of(rows) || t + 1 == p {
                    routes.push(seat.m.layer_routes(0..layers)?);
                }
            }
            let step_row = |t: usize, l: usize| &steps[(t * layers + l) * embd..][..embd];
            let mut worst = Vec::with_capacity(layers);
            let (mut inside, mut flips, mut widest) = (true, 0usize, None::<f32>);
            for l in 0..layers {
                let x: Option<Vec<f32>> = (l > 0).then(|| {
                    (0..p)
                        .flat_map(|t| step_row(t, l - 1).iter().copied())
                        .collect()
                });
                seat.m.reset()?;
                let mut w = 0.0f64;
                for (i, pass) in ids.chunks(rows).enumerate() {
                    let (a, b) = (i * rows, i * rows + pass.len());
                    let rows_in = x.as_deref().map(|x| &x[a * embd..b * embd]);
                    seat.prefill_layer(pass, rows, (l, rows_in))?;
                    let batch = seat
                        .m
                        .layer_routes(l..l + 1)?
                        .pop()
                        .ok_or_else(|| format!("the model holds no layer {l}"))?;
                    judged[l] += 1;
                    let route = &routes[i][l];
                    if sorted(&route.ids) == sorted(&batch.ids) {
                        w = w.max(crate::e2e::rel(&seat.m.layer_out(l)?, step_row(b - 1, l)));
                        continue;
                    }
                    exchanged[l] += 1;
                    if bias.is_empty() {
                        bias = router_bias(layers, route.logits.len())?;
                    }
                    let swap = swap_of(route, &batch, &bias[l])?;
                    println!(
                        "{FORCED_CLAUSE}: P={p} layer {l} row {}: steps kept {:?} batch kept {:?}, margin {:.4}, {} dropped for {} gap {:.4} drift {:.4} band {:.4} {}",
                        b - 1,
                        sorted(&route.ids),
                        sorted(&batch.ids),
                        swap.margin,
                        swap.pair.0,
                        swap.pair.1,
                        swap.gap,
                        swap.drift,
                        swap.band,
                        if swap.inside() { "inside" } else { "FAIL" }
                    );
                    inside &= swap.inside();
                    flips += 1;
                    widest = Some(widest.map_or(swap.margin, |m| m.max(swap.margin)));
                }
                worst.push(w);
            }
            let pass = inside && worst.iter().all(|&w| w <= TAP_CEIL);
            println!(
                "{FORCED_CLAUSE}: P={p} passes of {rows}: worst layer distances (rows that keep the steps' set) {} ceiling {TAP_CEIL}{}, {flips} of {} rows exchange{} {}",
                worst
                    .iter()
                    .map(|w| format!("{w:.3e}"))
                    .collect::<Vec<_>>()
                    .join(" "),
                worst
                    .iter()
                    .position(|&w| w > TAP_CEIL)
                    .map_or(String::new(), |l| format!(" (first layer over {l})")),
                layers * routes.len(),
                widest.map_or(String::new(), |m| format!(" (widest margin {m:.4})")),
                bloomery_gpu_gates::verdict(pass)
            );
            ok &= pass;
        }
        let spread = exchanged.iter().zip(&judged).all(|(&e, &j)| e * 2 <= j);
        println!(
            "{FORCED_CLAUSE}: rows that exchange per layer {exchanged:?} of {judged:?} (at most half) {}",
            bloomery_gpu_gates::verdict(spread)
        );
        Ok(ok && spread)
    }

    /// `prefill_stream_streamed` (module doc) over the stream arm's tally `t`.
    fn stream_streamed(t: &StreamTally) -> bool {
        for b in &t.bad {
            println!("{STREAM_CLAUSE}: {b} FAIL");
        }
        let pass = t.bad.is_empty() && t.streamed > 0;
        println!(
            "{STREAM_CLAUSE}: {} calls, {} of them streamed, {} experts streamed in all; the records, the ring passes and \
             the host's left-out slots {} {}",
            t.calls,
            t.streaming,
            t.streamed,
            if t.bad.is_empty() {
                "agree"
            } else {
                "disagree"
            },
            verdict(pass)
        );
        if t.streamed == 0 {
            println!(
                "FAIL precondition {STREAM_CLAUSE}: no call of the forced clause streamed an expert; the clause proves \
                 nothing"
            );
        }
        pass
    }

    /// Every layer's last-row output and its routed ids (sorted), as the last call left them.
    /// Each layer's output rows and routed ids.
    type Outs = (Vec<Vec<f32>>, Vec<Vec<u32>>);

    fn outs(m: &mut Model) -> Result<Outs, GateError> {
        let nodes = m.nodes()?;
        let ids = nodes
            .iter()
            .map(|n| {
                let mut v = n.ffn_moe_topk.clone();
                v.sort_unstable();
                v
            })
            .collect();
        Ok((nodes.into_iter().map(|n| n.l_out).collect(), ids))
    }

    /// The narrowing run (`--diag`, whole-card load): the batch against the steps per layer with each layer's routed
    /// ids beside, and the load's passes against passes of one row.
    fn diag(m: &mut Model, hp: &Hparams) -> Result<(), GateError> {
        let rows = m.pass_rows()?;
        for p in [1usize, 7, 2047] {
            let ids = crate::e2e::walk(p, hp.n_vocab);
            m.reset()?;
            for &id in &ids {
                m.step(&[id])?;
            }
            let (s_out, s_ids) = outs(m)?;
            m.reset()?;
            m.prefill(&ids)?;
            let (b_out, b_ids) = outs(m)?;
            for l in 0..hp.n_layer {
                println!(
                    "diag steps-vs-batch P={p} L{l} rel {:.3e} ids {} steps {:?} batch {:?}",
                    crate::e2e::rel(&b_out[l], &s_out[l]),
                    if s_ids[l] == b_ids[l] {
                        "same"
                    } else {
                        "DIFFER"
                    },
                    s_ids[l],
                    b_ids[l]
                );
            }
        }
        for p in [7usize, 600, 2600] {
            let ids = crate::e2e::walk(p, hp.n_vocab);
            m.reset()?;
            m.prefill_with(&ids, rows)?;
            let (a_out, a_ids) = outs(m)?;
            m.reset()?;
            m.prefill_with(&ids, 1)?;
            let (o_out, o_ids) = outs(m)?;
            for l in 0..hp.n_layer {
                println!(
                    "diag rows{rows}-vs-rows1 P={p} L{l} rel {:.3e} bits {} ids {}",
                    crate::e2e::rel(&a_out[l], &o_out[l]),
                    bits_equal(&a_out[l], &o_out[l]),
                    if a_ids[l] == o_ids[l] {
                        "same"
                    } else {
                        "DIFFER"
                    }
                );
            }
        }
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let tier_now = Tier::from_env()?;
        let file = open_split(Arch::MuseSpark, RECIPE)?;
        let hp = Hparams::read(&file).map_err(|e| format!("muse-spark header: {e}"))?;
        drop(file);
        let lengths = lengths(hp.window, tier_now);
        let ctx = lengths.last().copied().unwrap_or(1) + 1 + CTX_SPARE;
        if std::env::args().any(|a| a == "--diag") {
            let mut m = open(Arm::Whole, &hp, ctx)?;
            return diag(&mut m, &hp);
        }
        let own_at = own_lengths(&lengths, tier_now);
        let mut failed = None;
        let alone = alone_arg()?;
        let arms: &[Arm] = match alone {
            Some(Arm::Two) => &[Arm::Two],
            Some(Arm::Node) => &[Arm::Node],
            _ => arms(tier_now),
        };
        for &arm in arms {
            let t = std::time::Instant::now();
            let (m, two_pass, node) = match arm {
                Arm::Two => match two_card(&hp, ctx, tier_now)? {
                    Ok((m, pass, layers)) => (m, Some((pass, layers)), None),
                    Err(why) => {
                        // The body's and the common clauses run on the two-card load too, so they are left with it.
                        let names: Vec<&str> = TWO_CARD_OWN
                            .iter()
                            .copied()
                            .chain(OWN.iter().map(|&(name, _)| name))
                            .chain(prefill_clauses::CLAUSES.iter().map(|&(name, _, _)| name))
                            .collect();
                        tier::defer_for_room(&names, &why)?;
                        continue;
                    }
                },
                Arm::Node => match node_card(&hp, ctx)? {
                    Ok((m, pass, layers, node)) => (m, Some((pass, layers)), Some(node)),
                    Err(why) => {
                        let names: Vec<&str> = NODE_OWN
                            .iter()
                            .copied()
                            .chain(OWN.iter().map(|&(name, _)| name))
                            .chain(prefill_clauses::CLAUSES.iter().map(|&(name, _, _)| name))
                            .collect();
                        tier::defer_for_room(&names, &why)?;
                        continue;
                    }
                },
                Arm::Whole | Arm::Mid | Arm::Plan | Arm::Stream => {
                    (open(arm, &hp, ctx)?, None, None)
                }
            };
            let rows = m.pass_rows()?;
            println!(
                "load arm={} layers={} embd={} vocab={} ctx={ctx} window={} pass_rows={rows} in {:.1} s (runtime value)",
                arm.word(),
                hp.n_layer,
                hp.n_embd,
                hp.n_vocab,
                hp.window,
                t.elapsed().as_secs_f64()
            );
            let mut seat = Muse {
                m,
                dims: Dims {
                    layers: hp.n_layer,
                    embd: hp.n_embd,
                    vocab: hp.n_vocab,
                },
                stream: matches!(arm, Arm::Stream).then(StreamTally::default),
            };
            let case = Case {
                lengths: lengths.clone(),
                tap_ceil,
                tap_ceils,
                group: 1,
                logit_bound: Some(k_of(TOP)),
                sigma_logit: SIGMA_LOGIT_V41,
                vocab: hp.n_vocab,
                embd: hp.n_embd,
                tier: tier_now,
            };
            let before = match two_pass {
                Some(_) => counts(&seat.m)?,
                None => None,
            };
            // The body's own clauses first, so the common run's tally line counts them.
            let mut own = true;
            let clauses = match arm {
                Arm::Stream => &OWN[1..],
                Arm::Whole | Arm::Mid | Arm::Plan | Arm::Two | Arm::Node => &OWN[..],
            };
            for &(clause, f) in clauses {
                if tier::run_clause(clause, Tag::SelfConsistency)? {
                    let pass = f(&mut seat, &own_at)?;
                    if !pass {
                        println!("clause {clause} FAIL");
                    }
                    own &= pass;
                }
            }
            if let Some(t) = seat.stream.take() {
                if tier::run_clause(STREAM_CLAUSE, Tag::SelfConsistency)? {
                    let pass = stream_streamed(&t);
                    if !pass {
                        println!("clause {STREAM_CLAUSE} FAIL");
                    }
                    own &= pass;
                }
                if !own {
                    println!("arm {} FAIL", arm.word());
                    failed.get_or_insert_with(checks_failed);
                }
                continue;
            }
            let common =
                prefill_clauses::run(&format!("{NAME} arm {}", arm.word()), &case, &mut seat);
            if common.is_err() || !own {
                println!("arm {} FAIL", arm.word());
                failed.get_or_insert(common.err().unwrap_or_else(checks_failed));
            }
            if let Some((pass, layers)) = two_pass {
                let sent = sent_between(before, counts(&seat.m)?);
                let held = if matches!(arm, Arm::Node) {
                    tier::sc(NODE_COMMON)?;
                    sent_every(NODE_COMMON, sent.as_ref(), &layers, false)
                        & node_no_batch(NODE_COMMON, sent.as_ref(), false)
                } else {
                    tier::sc(TWO_CARD_BATCH)?;
                    sent_every(TWO_CARD_BATCH, sent.as_ref(), &layers, true)
                };
                if !(pass && held) {
                    println!("arm {} FAIL", arm.word());
                    failed.get_or_insert_with(|| {
                        format!("the {} arm's clauses failed", arm.word()).into()
                    });
                }
            }
            if let Some(mut node) = node {
                // The load's close ends the node's serving: it exits 0 once the stage has let go.
                drop(seat);
                match node.wait_within(NODE_CLOSE_WITHIN)? {
                    Some(status) if status.success() => {
                        println!("ok node: it exited 0 once the load closed");
                    }
                    Some(status) => {
                        println!("FAIL node: {status}");
                        failed.get_or_insert_with(|| "the node exited nonzero".into());
                    }
                    None => {
                        println!(
                            "FAIL node: still running {NODE_CLOSE_WITHIN:?} after the load closed; killed"
                        );
                        failed.get_or_insert_with(|| "the node did not exit".into());
                    }
                }
            }
        }
        failed.map_or(Ok(()), Err)
    }
}
