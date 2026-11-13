//! GPU gate for the V4.1 expert tier (`bloomery_gpu::host::tier`), on one
//! card: the loopback. The stage and the tier both run on the gate card
//! (`crate::gate_card::plan_gate`: the 3090's bytes in the real tier, the largest visible card
//! under the header's card budget in the fixture tier), two `Gpu`s on it. Each routed layer's card
//! list is its id prefix `[0, n_l)`. The tiered plan moves the last `k3` ([`tier_depth`]) of
//! them, ids `n_l - k3 .. n_l`, from card 0 to device 1 (the tier) in each layer the pair rule
//! admits on an expert tier (`accepts::ds41_pair` at `tier`: the fused layer's q4_K down; a q5_K
//! down stays whole on the stage card, as the placement's own tier plan keeps it) through
//! the placement's own split (`placement::routed_row`): the (b′) shape, the
//! stage the first ids and the tier the next ones, on the gate machine
//! with (b′)'s tier card and its prompt-batch reserves beside the stage card
//! (`tier_machine`). The reference is the gate plan itself, whose card holds
//! the union — the same experts on one card. Loads run one after the other,
//! the reference first.
//!
//! - `--union` (T1): the prose prompt [`PROMPT`] fed one decode step per
//!   position, then [`STEPS`] greedy steps, then [`PAIRS`] pair passes, all
//!   in graph mode, then the prompt and four steps eagerly: every position's
//!   argmax and logits bit for bit the reference's. The tiered stage graph
//!   has the reference's node count. Precondition: every layer the tier
//!   holds experts of was sent at least one routed slot over the run (the
//!   host's per-layer tier hits), or the run proves nothing and fails by
//!   name.
//! - `--fault` (T2): a fault raised on the tier card's word at the first
//!   tier layer before a step is that step's error, by its layer and site,
//!   and poisons the model; a reset lifts it and the next step is the
//!   reference's.
//! - `--lost` (T3): the tier's stream held behind a host flag before a
//!   step: within the go deadline and its grace the step fails naming the
//!   lost card, the host tier is poisoned as a lost card, the next step is
//!   refused by that poison (`GpuError::HostPoisoned`, a lost card) and a
//!   reset by name; the flag is then raised and both streams drain.
//! - `--two` (T7): a tiered plan with one expert on both devices is refused
//!   by name before anything is uploaded: as the load's check before the
//!   uploads (`body::TIER_MAP_BEFORE_UPLOAD`), not the one after them.
//! - `--batch` (B7): the prose prompt of [`BATCH_P`] positions fed as one
//!   prompt batch (`body::prefill`), then [`BATCH_STEPS`] greedy steps, walked
//!   against the same prompt fed one decode step per position on the tiered
//!   load (`logits_band::walk`): the call's last position and every greedy
//!   step, in order. The call's stage experts run the grouped GEMM, its
//!   tier's the tile path and the steps' the gemv, the same exact integer
//!   block sums folded in more than one f32 order, so the logits sit within
//!   the logits band of the steps', not on their bits. No reference load.
//! - `--batch2` (B7, a group): [`BATCH_P2`] positions, two batches run as one
//!   group of [`BATCH_GROUP`], then the greedy steps, walked against the
//!   reference's prompt call of the same ids. The reference's experts all
//!   sit on its one card, so all run the grouped GEMM; the tiered call's
//!   tier experts run the tile path, so the logits sit within the logits
//!   band of the reference's, not on its bits.
//!
//!   Both: the walk's tokens agree until a position the reference was not
//!   clear about (clear top-1 equality is the band's hard clause), after
//!   which the contexts differ and the rest is not compared. Precondition,
//!   the prompt call itself sent every tier layer at least one routed slot
//!   (the tier's per-layer hits across the call), and `--batch2`'s call ran
//!   as a group of two, or the clause proves nothing and fails by name.
//! - `--bfault` (B2): a fault raised on the tier card's word at the first
//!   tier layer before a prompt call is the call's error, by its layer and
//!   site, and poisons the model; a reset lifts it and the call is then,
//!   walked as B7's is, within the logits band of the same prompt fed one
//!   decode step per position. No reference load.
//! - `--blost` (B3): the tier's stream held behind a host flag before a
//!   prompt call: within the go deadline and its grace the call fails naming
//!   the lost card, the host tier is poisoned as a lost card, the next call
//!   is refused by that poison at its group's start (`HostTier::begin_group`,
//!   before it enqueues anything) and a reset by name; the flag is then
//!   raised and both streams drain. A lost tier stays lost, so with `--lost`
//!   too the tiered plan is loaded again for T3.
//! - `--bfirst` (B3f): B3 as the first prompt call of a fresh tiered load,
//!   before anything else runs on it: the call must still fail by name
//!   within the same bound, so nothing the call would make for itself (a
//!   module load, an allocation) waits on the held tier stream. The tiered
//!   plan is loaded again for any clause after it.
//! - `--bfeat` (B8): on a load of its own, the DSpark draft's feature tap
//!   (`body::attach_features`, the layers of `$BLOOMERY_DSPARK_MODEL`'s
//!   `target_layers`, header only: no draft is loaded) attached after the
//!   tiered load made the prompt batch's buffers, then `body::prepare_prefill`
//!   again — the order `--place bp` opens a draft in (`app::Loaded::open`,
//!   `open_draft`, `Loaded::ready`). Then the [`FEAT_P`] prose ids fed
//!   one decode step each, each position's features read after its step
//!   (`Body::read_features`), and a prompt call of the same ids
//!   (`body::prefill_with`) handing over every position's rows. A second
//!   call handing over its last `window` positions' rows, for the draft's
//!   `attention.sliding_window`, holds the positions and every row bit for
//!   bit the every-position call's: the same load, path and group of
//!   batches, so the window moves no value. The every-position call's rows
//!   against the steps': the largest |Δ| and the largest |Δ| relative to
//!   the step row's largest |x| over every row are printed, the latter held
//!   to [`FEATURE_REL_PIN`] — the call's experts and the steps' fold the
//!   same exact integer block sums in other f32 orders, so a row sits near
//!   the step's, not on its bits. Preconditions: the calls ran two batches
//!   as one group of [`BATCH_GROUP`], the every-position call handed rows
//!   over from both batches, and it sent every tier layer a routed slot.
//! - `--bnode` (B9): the tier on a node. The gate serves the tiered plan's tier from a
//!   `tier_server` beside it on a free loopback port (`shared/node_proc.rs`) and runs the stage as
//!   its own child (`--bnode-stage`), whose environment alone names that port
//!   (`BLOOMERY_TIER_NODES`) and the TCP link (`BLOOMERY_TIER_LINK=tcp`), so no other clause's
//!   load takes either. The stage feeds B7's prompt of [`BATCH_P`] positions one decode step per
//!   position, the node serving its tier experts, then as one prompt batch, whose every tiered
//!   layer-batch sends the node its block and lands the node's rows (`host::node::block`); each
//!   is followed by [`BATCH_STEPS`] greedy steps, and the call is walked against the steps
//!   (`logits_band::walk`), within the logits band as B7's is. Across the call the node's batch
//!   leg is served and settled once a block, every tier layer is sent a routed slot, no decode
//!   layer is asked of the node, and the slots the host left to the node's leg
//!   (`stats().batch_tier_slots`) are the node's routed slots (`stats().layer_hits`): the host
//!   serves none of them. Then the stage opens the same plan with its tier a card of this process
//!   and runs the call and its steps again: tokens and logits bit for bit the node's — the node
//!   runs the same block on the same q8_1 bytes, so its rows are the tier card's — and the host's
//!   batch slots, those it served and those it left to the tier's leg, the same counts. The residency
//!   lever resolves for a batch feed (its `why` is not `prefill_steps`). The node exits 0 once
//!   the stage closed its load. Precondition: the steps sent every tier layer a routed slot
//!   through the node, and the stage's tier is on the node.
//! - `--bnodeskip` (B11): the layers a step's picks leave the node tier. The gate writes a
//!   rank file from the fixture's own routing — the prompt's picks per layer over the tiered
//!   plan, the ids it never picks coldest — and runs this binary as a child ([`NODESKIP_STAGE`])
//!   twice under the file in its environment alone: over a loopback `tier_server` the gate
//!   serves (`BLOOMERY_TIER_NODES`, the TCP link) and with the tier a card of the child's own
//!   process. Over the node, a layer whose picks name no slot of the node's sends it no frame
//!   and the host signals the layer's counter itself, so the node's stream stays in step; each
//!   step must count its skipped layers (`stats().node_skipped`, the `stat step` record's
//!   `node_skipped` field) and the two children's tokens and logits must agree bit for bit —
//!   the same experts on the same homes, so the skip moves no value. Precondition: the trace
//!   leaves a routed layer's whole tier depth unrouted, so every step skips a layer.
//! - `--rank` (B10): the plan with a router rank (`BLOOMERY_EXPERT_RANK`). The gate writes a
//!   rank file from the model's own routed layers, each layer's ids in a stride order that is not
//!   the id order, and runs this binary as a child (`--rank-stage`) whose environment alone names
//!   it, so no other clause's plan takes it. The child's gate plan names the file, keeps the
//!   counts of the plan without it (`n_l`, the tier depth, the card's experts), and holds in each
//!   routed stack the layer's hottest `n_l` ranked ids on the card, not its id prefix. Under it
//!   the child runs T1: the tiered carve of that plan answers the reference load of the same plan
//!   bit for bit, its stage graph has the reference's node count, every tier layer was sent a
//!   routed slot. That is the slot map, the tier and the host set over lists that are no id
//!   prefix. The child then leaves the [`PROMPT`] positions of its tiered run — the fed id as
//!   each position's token, so every row is same-context evidence — and the gate feeds
//!   the same ids over the tiered carve of the plan without the file and walks the child's
//!   positions against its own within the logits band (`logits_band::walk_at`, every position
//!   at the placement margin `prompts::PLACE_MARGIN_V41`). The two plans keep
//!   other experts on the card and the host, and a card expert and a host expert are not one
//!   result bit for bit, so across plans the rows sit within the band, not on bits.
//!
//! Every clause is self-consistency (the tiered load against the reference load or against the
//! same file fed another way, B10 the same plan under another expert order), so the fixture tier
//! runs all of them; each names its tag once
//! through `crate::ds41_tier::sc` before it runs, and the closing line counts them.

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_tier", gate::run())
}

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_dspark.rs"]
#[allow(
    dead_code,
    reason = "B8 reads the draft's header only; the loop half serves generate_ds41"
)]
mod dspark;

#[cfg(feature = "deepseek41")]
#[path = "shared/gate_card.rs"]
mod gate_card;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_pair.rs"]
mod ds41_pair;

#[cfg(feature = "deepseek41")]
#[path = "shared/node_proc.rs"]
#[allow(
    dead_code,
    reason = "B9 starts and reaps its two children; stdin, signals, kills and stderr serve gate_qwen38_nodes"
)]
mod node_proc;

#[cfg(feature = "deepseek41")]
#[path = "shared/ds41_tier.rs"]
#[allow(
    dead_code,
    reason = "the gate reads the path and the clause tags; the triangle's facts serve the prefill gate"
)]
mod ds41_tier;

#[cfg(feature = "deepseek41")]
mod gate {
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::process::Command;
    use std::time::{Duration, Instant};

    use bloomery_gpu::host::PassKind;
    use bloomery_gpu::host::batch::TierBatchStats;
    use bloomery_gpu::hybrid::{BEGIN_GROUP, Chain, PoisonKind};
    use bloomery_gpu::model::StepMode;
    use bloomery_gpu::q4k_sel::QuantSel;
    use bloomery_gpu::{Fault, FaultSite, GpuError, HostFlags, Q8Act};
    use bloomery_gpu_deepseek41::body::{
        self, Body, BodyMeta, DECODE_INPUT, Deepseek41Model, FeatureRows, OpenCfg, PrefillMode,
        TIER_MAP_BEFORE_UPLOAD, TierOpen,
    };
    use bloomery_gpu_gates::logits_band::{self, Positions};
    use bloomery_gpu_gates::prompts::{MARGIN_FLOOR, PLACE_MARGIN_V41};
    use bloomery_gpu_gates::tier;
    use bloomery_gpu_gates::{GateError, checks_failed, prose_ids, record};
    use bloomery_levers::{
        CARD_BUDGET, CARD_DONTNEED, ENGRAM_HELPER, EXPERT_RANK, HOST_LOCK, HOST_POPULATE, Levers,
        R8, ResidencyAt, ResidencyWhy, TIER_LINK, TIER_NODES, TierLink,
    };
    use cuda_core::DeviceBuffer;
    use gguf::Split;
    use model::arch::deepseek41::hparams::Hparams;
    use model::arch::deepseek41::place::PlanInputs;
    use model::arch::deepseek41::place::tier_batch;
    use model::placement::workstation::DeviceId;
    use model::placement::{self, Device, ExpertList, Machine, Plan, Role, Row, workstation};
    use runtime::swaprule::KeptRows;

    use crate::dspark;

    const NAME: &str = "gate_deepseek41_tier";
    /// The real file's tier depth, as the literal this gate carried before [`tier_depth`] derived
    /// it: the (b′) plan's 831 tier experts over its 38 hybrid layers are 21.9 a layer, which
    /// rounds to 22. The real tier prints the derived depth beside it and requires them equal.
    // PIN(2026-11-12): 831 tier experts where the plan gave 881 (23). What took the 50 is the
    // prompt batch's parts: the plan now counts a prompt front on every card — the tier's too —
    // and takes it off the card's expert budget before the spread (PlanInputs::plan_slots_nodes'
    // `fronts`, charged by loaded_routed's `reserve` and grown into the card's prompt term), and
    // a one-slot plan's pass heads add none (slots < 2) while the plan's levers carry no K/V
    // floor, so no other card term of that batch moved the count. The front's bytes at the tier
    // the plan settles at run time; no log line prints the (b′) tier card's term list.
    // FAIL-first: the logged run derived 22 against the old literal 23; a change that hands the
    // tier back those 50 experts' bytes (dropping the front from its budget) derives 23 and is
    // red, as is any term that takes or gives ≳17 experts (one depth step at 38 layers).
    const REAL_K3: usize = 22;
    /// Prompt positions, fed one decode step each: the batch arm's length, whose
    /// prose sends a slot to every tier layer of the real file.
    const PROMPT: usize = 64;
    /// Greedy steps after the prompt.
    const STEPS: usize = 32;
    /// Pair passes after the prompt.
    const PAIRS: usize = 4;
    /// Steps the eager arm runs after the prompt.
    const EAGER_STEPS: usize = 4;
    /// Positions of the prompt call `--batch` feeds against the steps: one
    /// batch.
    /// PIN(2026-10-10): 64 where it was 512. The batch call against the steps
    /// at depth (the `CASES` of P, 512 among them, and the three `SPLITS`;
    /// every layer's ring, state, shadow, rows, keys and logits) is
    /// `gate-gpu-ds41-prefill`'s Cases clause; this clause keeps what only the
    /// tier loopback load shows, at this size: the batch call is within the
    /// band of the steps' on the tiered load with every tier layer sent a slot,
    /// and `--batch2` keeps the two-batch call against the reference.
    const BATCH_P: usize = 64;
    /// Positions of the prompt call `--batch2` feeds against the reference's:
    /// two batches, one group of two under the default group lever.
    const BATCH_P2: usize = 1024;
    /// Positions of the prompt call `--bfeat` hands the feature tap's rows over
    /// in, against the steps': one past the batch width, which is two batches
    /// of one group.
    /// PIN(2026-10-10): 513 where it was `BATCH_P2` (1024). The tap's rows from
    /// a prompt call (the kept features' positions and rows at each of `CASES`
    /// and of `SPLITS`, and the wide-taps case of 1100 keeping 300)
    /// are `gate-gpu-ds41-prefill`'s Cases and Wide taps clauses; this clause
    /// keeps what only the tier gate shows, at this size: the tap attached
    /// after the tiered load made the batch buffers, every tier layer sent a
    /// slot, and a group of two batches both handing rows over.
    const FEAT_P: usize = 513;
    /// The most the every-position call's features may sit from the steps': the RMS |Δ| over
    /// the steps' RMS |x|, every value of every position. A tap that reads another position or
    /// layer sits near or past 1; the two sum orders sit far under it.
    // PIN(2026-11-04): with layers 0·1's routed experts on the cards, the prompt call runs them by the
    // tile GEMM and the steps by the `_sel` gemv, so the first routed seam's sum-order distance now
    // opens at layer 0 and every layer compounds it: real 0.1156, fixture 0.1656 (against the step one
    // position earlier: 0.944, 1.246), x1.5. With the seam at layer 2 it read 0.0789 real, 0.0644
    // fixture, under 0.16. Not derived.
    const FEATURE_REL_PIN: f64 = 0.25;
    /// The batches a group holds that `--batch2`'s call must run as.
    const BATCH_GROUP: usize = 2;
    /// Greedy steps after a prompt call.
    const BATCH_STEPS: usize = 4;
    /// Positions of the prompt calls B2 and B3 feed: one batch.
    const BATCH_FAULT_P: usize = 64;
    /// The device index the tier's segments sit on in the tiered plan.
    const TIER_DEVICE: usize = 1;
    /// The go deadline and the grace a lost card is named within, plus room.
    const LOST_BOUND_S: f64 = 25.0;
    /// The argument B9's stage child runs under ([`node_case`]).
    const NODE_STAGE: &str = "--bnode-stage";
    /// How long B9's stage child may take: its tiered load, the steps, the prompt call and the
    /// greedy steps after each.
    const NODE_STAGE_WITHIN: Duration = Duration::from_secs(600);
    /// How long the node may take to exit once the stage has ended.
    const NODE_CLOSE_WITHIN: Duration = Duration::from_secs(60);
    /// The argument B10's child runs under ([`rank_case`]).
    const RANK_STAGE: &str = "--rank-stage";
    /// How long B10's child may take: its plans, one tiered load and the prompt's steps.
    const RANK_STAGE_WITHIN: Duration = Duration::from_secs(600);
    /// The file B10's child writes its run to, beside the rank file it ran under.
    const RANK_ROWS: &str = "rows.bin";
    /// The argument B11's stage child runs under ([`bnodeskip_case`]).
    const NODESKIP_STAGE: &str = "--bnodeskip-stage";
    /// How long B11's stage child may take: its tiered load and the prompt's steps.
    const NODESKIP_STAGE_WITHIN: Duration = Duration::from_secs(600);
    /// The file B11's child writes its run to, beside the rank file it ran under.
    const NODESKIP_ROWS: &str = "rows.bin";

    struct Args {
        union: bool,
        fault: bool,
        lost: bool,
        two: bool,
        batch: bool,
        batch2: bool,
        bfault: bool,
        blost: bool,
        bfirst: bool,
        bfeat: bool,
        bnode: bool,
        rank: bool,
        bnodeskip: bool,
        /// B9's stage, the child [`node_case`] runs: alone, never beside a clause.
        bnode_stage: bool,
        /// B11's stage, the child [`bnodeskip_case`] runs: alone, never beside a clause.
        bnodeskip_stage: bool,
        /// B10's stage, the child [`rank_case`] runs: alone, never beside a clause.
        rank_stage: bool,
    }

    impl Args {
        /// Whether a clause other than T7 and B3f runs on the tiered load.
        fn on_load(&self) -> bool {
            self.union
                || self.fault
                || self.lost
                || self.batch
                || self.batch2
                || self.bfault
                || self.blost
        }
    }

    fn parse_args() -> Result<Args, GateError> {
        const USAGE: &str = "usage: gate_deepseek41_tier [--union] [--fault] [--lost] [--two] \
                             [--batch] [--batch2] [--bfault] [--blost] [--bfirst] [--bfeat] \
                             [--bnode] [--bnodeskip] [--rank]";
        let mut a = Args {
            union: false,
            fault: false,
            lost: false,
            two: false,
            batch: false,
            batch2: false,
            bfault: false,
            blost: false,
            bfirst: false,
            bfeat: false,
            bnode: false,
            rank: false,
            bnodeskip: false,
            bnode_stage: false,
            bnodeskip_stage: false,
            rank_stage: false,
        };
        for arg in std::env::args().skip(1) {
            match arg.as_str() {
                "--union" => a.union = true,
                "--fault" => a.fault = true,
                "--lost" => a.lost = true,
                "--two" => a.two = true,
                "--batch" => a.batch = true,
                "--batch2" => a.batch2 = true,
                "--bfault" => a.bfault = true,
                "--blost" => a.blost = true,
                "--bfirst" => a.bfirst = true,
                "--bfeat" => a.bfeat = true,
                "--bnode" => a.bnode = true,
                "--bnodeskip" => a.bnodeskip = true,
                "--rank" => a.rank = true,
                NODE_STAGE => a.bnode_stage = true,
                NODESKIP_STAGE => a.bnodeskip_stage = true,
                RANK_STAGE => a.rank_stage = true,
                other => return Err(format!("unknown argument {other:?}: {USAGE}").into()),
            }
        }
        let clause =
            a.on_load() || a.two || a.bfirst || a.bfeat || a.bnode || a.bnodeskip || a.rank;
        let stages = [a.bnode_stage, a.bnodeskip_stage, a.rank_stage];
        for (on, stage, owner) in [
            (a.bnode_stage, NODE_STAGE, "B9's"),
            (a.bnodeskip_stage, NODESKIP_STAGE, "B11's"),
            (a.rank_stage, RANK_STAGE, "B10's"),
        ] {
            if on && (clause || stages.iter().filter(|s| **s).count() > 1) {
                return Err(format!(
                    "{stage} is {owner} stage, which the gate runs as a child of its own: never \
                     beside a clause or the other stage"
                )
                .into());
            }
        }
        if !(clause || a.bnode_stage || a.bnodeskip_stage || a.rank_stage) {
            return Err(USAGE.into());
        }
        Ok(a)
    }

    /// The gate plan's machine `machine` with the (b′) tier card beside its
    /// one stage card, at index [`TIER_DEVICE`]: the tier card and the host
    /// carry the tier's prompt-batch reserves for the file of `hp`
    /// (`workstation::plan_bp`'s), which a tiered load checks its
    /// allocations against. (b′)'s tier is the 3090, whose bytes the gate
    /// card plans; it goes on the gate card by name, so the stage and the
    /// tier share it.
    fn tier_machine(machine: &Machine, hp: &Hparams) -> Result<Machine, GateError> {
        if machine.cards.len() != TIER_DEVICE || !machine.tiers.is_empty() {
            return Err(format!(
                "the gate machine has {} stage cards and {} tiers; the tier goes at index \
                 {TIER_DEVICE}, after one stage card",
                machine.cards.len(),
                machine.tiers.len()
            )
            .into());
        }
        let bp = workstation::plan_bp(hp.n_layer, None, tier_batch(hp));
        if bp
            .tiers
            .iter()
            .map(|t| t.name.as_str())
            .ne([workstation::RTX_3090.name])
        {
            return Err(format!(
                "plan (b′)'s tier cards {:?} are not the 3090, whose bytes the gate card {} plans",
                bp.tiers.iter().map(|t| &t.name).collect::<Vec<_>>(),
                machine.cards[0].name
            )
            .into());
        }
        let mut out = machine.clone();
        *out.host_mut() = bp.host().clone();
        *out.unified_mut() = bp.unified();
        out.tiers = bp.tiers;
        for t in &mut out.tiers {
            t.name.clone_from(&machine.cards[0].name);
            t.device = machine.cards[0].device;
        }
        Ok(out)
    }

    /// Experts per routed layer the tier takes from the cold end of the gate plan's card list: the
    /// (b′) plan's tier depth — the mean of its tier experts over the layers that have any, rounded
    /// down after a half added, read from `workstation::plan_bp` planned over the same file with
    /// the DSpark draft's reserve (`reserve`, from the draft's header) under `place`, the levers
    /// of that plan. A carve from the gate plan's own card list costs no card bytes, so the 3090's
    /// free bytes after the load bound nothing here; a depth the gate plan's lists cannot give is
    /// [`tiered`]'s named error. The real tier prints the derived depth beside [`REAL_K3`].
    fn tier_depth(
        inputs: &PlanInputs,
        place: &model::placement::PlanLevers,
        reserve: u64,
    ) -> Result<usize, GateError> {
        let bp = workstation::plan_bp(inputs.hp.n_layer, Some(reserve), tier_batch(&inputs.hp));
        let plan = inputs.plan(&bp, workstation::CTX_MAX, place)?;
        let per: Vec<u64> = plan
            .tier_n_l
            .first()
            .ok_or("plan (b′) has no tier card")?
            .iter()
            .copied()
            .filter(|&n| n > 0)
            .collect();
        let layers = per.len() as u64;
        if layers == 0 {
            return Err("plan (b′) puts no expert on its tier card".into());
        }
        let total: u64 = per.iter().sum();
        let depth = usize::try_from((total + layers / 2) / layers)?;
        println!(
            "tier depth: plan (b′) with the draft's reserve of {reserve} B puts {total} experts on \
             its tier over {layers} layers: {depth} a layer"
        );
        if !bloomery_gpu_gates::tier::witness("tier depth K3", depth, REAL_K3) {
            return Err(
                format!("the derived tier depth {depth} is not the literal {REAL_K3}").into(),
            );
        }
        Ok(depth)
    }

    /// Whether `r` is the host tier's refusal ([`GpuError::HostPoisoned`])
    /// for the poison of a lost card, as `what`: the entry that refuses it
    /// before it enqueues anything.
    fn lost_refusal<T>(r: &Result<T, GpuError>, what: &str) -> bool {
        matches!(
            r,
            Err(GpuError::HostPoisoned { what: w, poison })
                if *w == what
                    && poison.mark().kind == PoisonKind::CardLost
        )
    }

    /// `plan` on `machine` (the gate plan's with the tier card,
    /// [`tier_machine`]) with the last `k3` experts of each routed stack's
    /// card-0 list, in id order, moved to device [`TIER_DEVICE`], in each
    /// layer whose gate and down types (`hp`'s layer table) the pair rule
    /// admits on an expert tier (`accepts::ds41_pair` at `tier`); a layer it
    /// keeps off a tier keeps its card-0 lists whole, where the engine's tier
    /// refuses it (`Ds41Tier::new`). A stack of a tier layer whose card list
    /// holds `k3` or fewer is refused by name. With `twice`, the first stack
    /// the tier takes also keeps its first tier expert on card 0 (an expert on
    /// two devices). The host segments stay as they are, so the host set is
    /// the plan's.
    fn tiered<'a>(
        plan: &Plan<'a>,
        machine: &'a Machine,
        hp: &Hparams,
        k3: usize,
        twice: bool,
    ) -> Result<Plan<'a>, GateError> {
        let model = plan.model;
        let mut out = plan.clone();
        out.machine = machine;
        let mut doubled = false;
        for row in &mut out.rows {
            let t = &model.tensors[row.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let Some(card) = row.segments.iter().find(|s| s.device == Device::Card(0)) else {
                continue;
            };
            let layer = t
                .layer
                .ok_or_else(|| format!("{}: a routed stack without a layer", t.name))?;
            if !crate::ds41_pair::tier_admits(hp, layer)? {
                continue;
            }
            let ids = card
                .experts
                .as_ref()
                .ok_or_else(|| format!("{}: a card segment without experts", t.name))?
                .ids()
                .to_vec();
            if ids.len() <= k3 {
                return Err(format!(
                    "{}: the card keeps {} experts, the gate moves {k3} to the tier",
                    t.name,
                    ids.len()
                )
                .into());
            }
            let cut = ids.len() - k3;
            let mut stage_ids = ids[..cut].to_vec();
            if twice && !doubled {
                stage_ids.push(ids[cut]);
            }
            let stage = placement::routed_row(
                row.tensor,
                t,
                0,
                ExpertList::new(stage_ids, model.experts)?,
                model,
            )?;
            let tier = placement::routed_row(
                row.tensor,
                t,
                TIER_DEVICE,
                ExpertList::new(ids[cut..].to_vec(), model.experts)?,
                model,
            )?;
            let host = row
                .segments
                .iter()
                .filter(|s| s.device != Device::Card(0))
                .cloned();
            let segments = stage
                .segments
                .into_iter()
                .filter(|s| s.device == Device::Card(0))
                .chain(
                    tier.segments
                        .into_iter()
                        .filter(|s| s.device == Device::Card(TIER_DEVICE)),
                )
                .chain(host)
                .collect();
            *row = Row {
                segments,
                ..row.clone()
            };
            if let Some(n) = out.n_l.get_mut(layer) {
                *n = cut as u64;
            }
            doubled = true;
        }
        Ok(out)
    }

    /// Every argmax and logits vector a run left, in order.
    #[derive(Default)]
    struct Run {
        tokens: Vec<u32>,
        logits: Vec<Vec<f32>>,
        stage_nodes: Option<usize>,
    }

    /// A staged call's run and what it sent the node tier, when the stage ran one.
    type Staged = (Run, Option<Sent>);

    impl Run {
        fn push(&mut self, token: u32, logits: Vec<f32>) {
            self.tokens.push(token);
            self.logits.push(logits);
        }

        /// The run's tokens, each with the logits row it was taken from.
        fn positions(&self) -> Positions<'_> {
            Positions {
                tokens: &self.tokens,
                rows: &self.logits,
            }
        }
    }

    /// The T1 run on `m`: the prompt one step a position (its argmaxes as
    /// its tokens — T2 and B2 read the last one), the greedy steps, the pair
    /// passes, all in graph mode from a reset; then the prompt and
    /// [`EAGER_STEPS`] steps eagerly from a reset.
    fn union_run(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<Run, GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut tok = 0;
        for &id in prompt {
            tok = m.step(&[id])?;
            run.push(tok, m.logits()?);
        }
        run.stage_nodes = Some(m.step_graph_nodes()?.len());
        for _ in 0..STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        for k in 0..PAIRS {
            let second = prompt[k % prompt.len()];
            let [a, b] = m.step_rows([tok, second])?;
            let [la, lb] = m.rows_logits::<2>()?;
            m.keep_rows(KeptRows::prefix(2), PassKind::Pair)?;
            run.push(a, la);
            run.push(b, lb);
            tok = b;
        }
        m.set_mode(StepMode::Eager);
        m.reset()?;
        for &id in prompt {
            tok = m.step(&[id])?;
            run.push(tok, m.logits()?);
        }
        for _ in 0..EAGER_STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        m.set_mode(StepMode::Graph);
        Ok(run)
    }

    /// The reference load's runs each clause compares with.
    #[derive(Default)]
    struct Want {
        union: Option<Run>,
        batch2: Option<Run>,
    }

    /// What one prompt call sent the tier: its hits per layer from the
    /// tier's first layer, and the batch counters it moved.
    struct Sent {
        first: usize,
        hits: Vec<u64>,
        /// Tier layers the host asked of the tier (replayed or enqueued).
        issued: u64,
        stats: TierBatchStats,
        /// The host's batch slots: those it served, and those it left to a
        /// tier's batch leg (`HybridStats::batch_host_slots`,
        /// `batch_tier_slots`).
        host_slots: u64,
        tier_slots: u64,
    }

    /// The tier's per-layer hits, layers asked and batch counters since load;
    /// `None` without a tier.
    fn tier_counts(m: &Deepseek41Model) -> Result<Option<Sent>, GateError> {
        let hybrid = m.body(NAME)?.hybrid();
        let st = hybrid.stats();
        Ok(hybrid.tiers().first().map(|t| Sent {
            first: t.set().layers().start,
            hits: t.stats().layer_hits,
            issued: t.stats().issued,
            stats: hybrid
                .tier_batch_stats()
                .first()
                .copied()
                .unwrap_or_default(),
            host_slots: st.batch_host_slots,
            tier_slots: st.batch_tier_slots,
        }))
    }

    /// The counters `after` less `before`: what the span between sent the
    /// tier; `None` without a tier.
    fn sent_between(before: Option<Sent>, after: Option<Sent>) -> Option<Sent> {
        let (b, a) = (before?, after?);
        Some(Sent {
            first: a.first,
            hits: a.hits.iter().zip(&b.hits).map(|(a, b)| a - b).collect(),
            issued: a.issued - b.issued,
            stats: TierBatchStats {
                served: a.stats.served - b.stats.served,
                settles: a.stats.settles - b.stats.settles,
                settle_early: a.stats.settle_early - b.stats.settle_early,
                settle_ns: a.stats.settle_ns - b.stats.settle_ns,
                skipped: a.stats.skipped - b.stats.skipped,
            },
            host_slots: a.host_slots - b.host_slots,
            tier_slots: a.tier_slots - b.tier_slots,
        })
    }

    /// From a reset in graph mode: `ids` fed as one prompt call, then
    /// [`BATCH_STEPS`] greedy steps; the call's last position and each step,
    /// and on a tiered load what the call alone sent the tier.
    fn batch_run(m: &mut Deepseek41Model, ids: &[u32]) -> Result<(Run, Option<Sent>), GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let before = tier_counts(m)?;
        let mut tok = body::prefill(m, ids)?;
        let sent = sent_between(before, tier_counts(m)?);
        run.push(tok, m.logits()?);
        for _ in 0..BATCH_STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        Ok((run, sent))
    }

    /// The precondition of a prompt-call clause `what`: the call sent every
    /// layer of `tier_layers` at least one routed slot; prints its counters.
    fn sent_every(what: &str, sent: Option<&Sent>, tier_layers: &[usize]) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what} precondition: the tiered load holds no tier card");
            return false;
        };
        let hit = |l: usize| {
            l.checked_sub(sent.first)
                .and_then(|i| sent.hits.get(i))
                .copied()
                .unwrap_or(0)
        };
        let missing: Vec<usize> = tier_layers
            .iter()
            .copied()
            .filter(|&l| hit(l) == 0)
            .collect();
        let slots: u64 = tier_layers.iter().map(|&l| hit(l)).sum();
        let st = sent.stats;
        println!(
            "{what} tier: {} services, {} settles ({} early, {:.2} ms waited), {slots} tier slots \
             over {} layers",
            st.served,
            st.settles,
            st.settle_early,
            st.settle_ns as f64 / 1e6,
            tier_layers.len()
        );
        if missing.is_empty() && st.served > 0 {
            println!("ok {what} precondition: the prompt call sent every tier layer a routed slot");
            true
        } else {
            println!(
                "FAIL {what} precondition: tier layers {missing:?} were sent no routed slot by the \
                 prompt call ({} services); the clause proves nothing there",
                st.served
            );
            false
        }
    }

    /// [`batch_run`] with `ids` fed one decode step per position.
    fn steps_run(m: &mut Deepseek41Model, ids: &[u32]) -> Result<Run, GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut tok = 0;
        for &id in ids {
            tok = m.step(&[id])?;
        }
        run.push(tok, m.logits()?);
        for _ in 0..BATCH_STEPS {
            tok = m.step(&[tok])?;
            run.push(tok, m.logits()?);
        }
        Ok(run)
    }

    /// Whether `got` is `want` bit for bit; prints the first difference.
    fn same(what: &str, want: &Run, got: &Run) -> bool {
        if want.tokens.len() != got.tokens.len() {
            println!(
                "FAIL {what}: {} positions, the reference ran {}",
                got.tokens.len(),
                want.tokens.len()
            );
            return false;
        }
        for (i, ((wt, wl), (gt, gl))) in want
            .tokens
            .iter()
            .zip(&want.logits)
            .zip(got.tokens.iter().zip(&got.logits))
            .enumerate()
        {
            let bits =
                wl.len() == gl.len() && wl.iter().zip(gl).all(|(a, b)| a.to_bits() == b.to_bits());
            if wt != gt || !bits {
                let first = wl
                    .iter()
                    .zip(gl)
                    .position(|(a, b)| a.to_bits() != b.to_bits());
                println!(
                    "FAIL {what}: position {i}: token {gt} (reference {wt}); first logit apart {first:?}"
                );
                return false;
            }
        }
        println!(
            "ok {what}: {} positions, tokens and logits bit for bit the reference's",
            got.tokens.len()
        );
        true
    }

    /// Whether `got`, a prompt call and the greedy steps after it, is within the logits band of
    /// `want`'s: the two runs' positions walked in order ([`logits_band::walk`]), which prints its
    /// line. A prompt call's stage experts run the grouped GEMM and a tier's the tile path, while
    /// the steps' run the gemv and a reference load's call runs every expert on the GEMM, so the
    /// same exact integer block sums fold in more than one f32 order and the logits are near the
    /// other order's, not on its bits.
    fn in_band(what: &str, want: &Run, got: &Run) -> bool {
        logits_band::walk(what, want.positions(), got.positions()).ok
    }

    /// [`in_band`] for a walk that starts at the prompt's position 0 (the rank clause's): the
    /// runs record the fed id as each prompt position's token, so every position's row is
    /// same-context evidence and the walk judges them all at the placement margin — the real
    /// tier's [`PLACE_MARGIN_V41`], the fixture tier's the e2e floor: the two plans run other
    /// experts on the card and the host.
    // PIN(2026-11-12): the rank walk recorded the argmax as each position's token, so it parted
    // at the first argmax flip — 1 of the reference's 64 positions compared — although both arms
    // were fed the same prompt id and stood at the same context; the fed id is the position's
    // context token, and every row is judged.
    fn in_band_from_zero(what: &str, want: &Run, got: &Run) -> Result<bool, GateError> {
        let margin = match tier::Tier::from_env()? {
            tier::Tier::Real => PLACE_MARGIN_V41,
            tier::Tier::Fixture => MARGIN_FLOOR,
        };
        Ok(logits_band::walk_at(what, want.positions(), got.positions(), margin).ok)
    }

    fn open(
        file: Split,
        plan: &Plan<'_>,
        tier: Option<TierOpen>,
        meta: &BodyMeta,
    ) -> Result<Deepseek41Model, GpuError> {
        Body::open_placed_tiered(file, plan, 0, tier, meta)
    }

    pub fn run() -> Result<(), GateError> {
        let levers = bloomery_levers::at_main(&[
            ENGRAM_HELPER,
            CARD_BUDGET,
            EXPERT_RANK,
            HOST_POPULATE,
            HOST_LOCK,
            CARD_DONTNEED,
            R8,
        ])?;
        crate::gate_card::init()?;
        let args = parse_args()?;
        let mut cfg = OpenCfg::from_levers(&levers)?;
        // The batch clauses run the prompt call, so the tiered load makes its
        // buffers (`Body::open_placed_tiered`).
        cfg.body.prefill = PrefillMode::Batch;
        let path = crate::ds41_tier::model_path()?;
        let split = || Split::open(&path).map_err(|e| format!("open {path}: {e}"));
        // The draft is read for its header only (B8), so the plan reserves nothing for it.
        cfg.place = bloomery_gpu_gates::tier::plan_levers(&split()?, &levers, 0)?;
        let inputs = PlanInputs::read(&split()?)?;
        let machine = crate::gate_card::plan_gate(inputs.model.layers);
        let plan = inputs.plan(&machine, workstation::CTX_MAX, &cfg.place)?;
        for (on, name) in [
            (
                args.union,
                "T1: the tiered load is the reference load, step, pairs, eager",
            ),
            (
                args.fault,
                "T2: a fault on the tier card's word is the step's error and poison",
            ),
            (
                args.lost,
                "T3: the held tier stream is named as a lost card",
            ),
            (
                args.two,
                "T7: an expert on two devices is refused before any upload",
            ),
            (
                args.batch,
                "B7: one batch is within the logits band of the steps",
            ),
            (
                args.batch2,
                "B7 group: two batches as one group are within the logits band of the \
                 reference's call",
            ),
            (
                args.bfault,
                "B2: a tier fault before a prompt call is the call's error",
            ),
            (
                args.blost,
                "B3: the held tier stream fails a prompt call by name",
            ),
            (
                args.bfirst,
                "B3f: the first prompt call of a fresh load fails by name",
            ),
            (
                args.bfeat,
                "B8: the feature tap through a tiered prompt call hands over the call's own rows, \
                 near the steps'",
            ),
            (
                args.bnode,
                "B9: one batch over a tier on a node: the node computes its block, within the \
                 logits band of the steps and bit for bit the same plan's with its tier a card",
            ),
            (
                args.rank,
                "B10: a router rank moves the experts' homes, not the tokens",
            ),
            (
                args.bnodeskip,
                "B11: a layer no pick of the step's sends the node tier nothing",
            ),
        ] {
            if on {
                crate::ds41_tier::sc(name)?;
            }
        }
        let meta = BodyMeta {
            hp: inputs.hp.clone(),
            levers: cfg.body,
        };
        let card = machine.cards[0].name.clone();
        let device = machine.cards[0].device;
        let tier_open = || TierOpen {
            card: TIER_DEVICE,
            name: card.clone(),
            device,
            node: None,
        };
        let tmachine = tier_machine(&machine, &inputs.hp)?;
        let draft = dspark::draft_hparams()?;
        let reserve = dspark::draft_bytes(&draft.0, std::path::Path::new(&path))?;
        let k3 = tier_depth(
            &inputs,
            &bloomery_gpu_gates::tier::plan_levers(&split()?, &levers, reserve)?,
            reserve,
        )?;
        let tplan = tiered(&plan, &tmachine, &inputs.hp, k3, false)?;
        let tier_layers: Vec<usize> = (0..inputs.model.layers)
            .filter(|&l| tplan.n_l.get(l) != plan.n_l.get(l))
            .collect();
        let whole: Vec<usize> = (0..inputs.model.layers)
            .filter(|&l| plan.n_l.get(l).is_some_and(|&n| n > 0) && !tier_layers.contains(&l))
            .collect();
        println!(
            "plan: gate card {card}, {} experts on it by the id prefix; tier: the last {k3} of each \
             tier layer's list on {} layers ({} experts), stage keeps the rest and, whole, layers \
             {whole:?} the pair rule keeps off a tier",
            plan.cards[0].experts,
            tier_layers.len(),
            k3 * tier_layers.len()
        );
        if args.rank_stage {
            let base = inputs.plan(
                &machine,
                workstation::CTX_MAX,
                &model::placement::PlanLevers {
                    expert_rank: None,
                    ..cfg.place.clone()
                },
            )?;
            let mut pass = rank_plan(&plan, &base)?;
            let Some(rank) = plan.expert_rank.as_deref() else {
                return rank_verdict(false);
            };
            let prompt = prose_ids("engram", PROMPT)?;
            let mut reference = open(split()?, &plan, None, &meta)?;
            let want = union_run(&mut reference, &prompt)?;
            drop(reference);
            let mut m = open(split()?, &tplan, Some(tier_open()), &meta)?;
            let got = union_run(&mut m, &prompt)?;
            pass &= tiered_is_reference(
                (
                    "B10",
                    "B10 (the tiered load under the rank is the reference load under it)",
                ),
                &m,
                (&want, &got),
                &tier_layers,
            )?;
            // The rows the gate walks: the prompt's positions with the fed id
            // as each one's token, on the same load the union ran.
            let rows = rank_run(&mut m, &prompt)?;
            drop(m);
            write_rows(
                &Path::new(rank.path()).with_file_name(RANK_ROWS),
                &rows,
                prompt.len(),
            )?;
            return rank_verdict(pass);
        }
        if args.bnodeskip_stage {
            let prompt = prose_ids("engram", PROMPT)?;
            let Some(rank) = plan.expert_rank.as_ref() else {
                println!("FAIL B11 stage: the plan holds no expert rank");
                return stage_verdict(false);
            };
            // Over the node the gate serves, or with the tier a card of this
            // process: the same ranked plan, the skip the node's alone.
            let nodes = bloomery_levers::tier_nodes()?;
            let tier = match nodes.as_slice() {
                ns if ns.len() == 1 => match node_open(&card, device) {
                    Ok(t) => t,
                    Err(e) => {
                        println!("FAIL B11 stage: {e}");
                        return stage_verdict(false);
                    }
                },
                [] => tier_open(),
                ns => {
                    println!(
                        "FAIL B11 stage: {TIER_NODES} names {} nodes; the skip stage takes the \
                         one the gate serves or none",
                        ns.len()
                    );
                    return stage_verdict(false);
                }
            };
            let mut m = open(split()?, &tplan, Some(tier), &meta)?;
            let mut stage_picks = Vec::new();
            let (pass, run) =
                match bnodeskip_stage(&mut m, &prompt, nodes.len() == 1, &mut stage_picks) {
                    Ok(r) => r,
                    Err(e) => {
                        println!("FAIL B11 stage: {e}");
                        return stage_verdict(false);
                    }
                };
            drop(m);
            write_rows(
                &Path::new(rank.path()).with_file_name(NODESKIP_ROWS),
                &run,
                prompt.len(),
            )?;
            let mut text = String::new();
            for (l, ours) in stage_picks.iter().enumerate() {
                let mut ids = ours.clone();
                ids.sort_unstable();
                ids.dedup();
                let row: Vec<String> = ids.iter().map(u32::to_string).collect();
                text.push_str(&format!("{l}\t{}\n", row.join(",")));
            }
            std::fs::write(Path::new(rank.path()).with_file_name("trace.tsv"), text)?;
            return stage_verdict(pass);
        }
        if args.bnode_stage {
            let ids = prose_ids("engram", BATCH_P2)?;
            let node = node_open(&card, device)?;
            let m = open(split()?, &tplan, Some(node), &meta)?;
            let (mut pass, got) = node_stage(m, &levers, &ids[..BATCH_P], &tier_layers)?;
            let local = open(split()?, &tplan, Some(tier_open()), &meta)?;
            pass &= node_exact(local, &ids[..BATCH_P], got)?;
            return stage_verdict(pass);
        }
        let mut pass = true;

        if args.two {
            let bad = tiered(&plan, &tmachine, &inputs.hp, k3, true)?;
            let t0 = Instant::now();
            match open(split()?, &bad, Some(tier_open()), &meta) {
                Err(e @ GpuError::Plan { what, .. })
                    if what == TIER_MAP_BEFORE_UPLOAD && e.to_string().contains("placed twice") =>
                {
                    println!(
                        "ok T7: an expert on two devices refused before any upload in {:.2} s: {e}",
                        t0.elapsed().as_secs_f64()
                    );
                }
                Err(e) => {
                    println!(
                        "FAIL T7: refused, but not as {TIER_MAP_BEFORE_UPLOAD} naming an expert \
                         placed twice: {e}"
                    );
                    pass = false;
                }
                Ok(_) => {
                    println!("FAIL T7: a plan with an expert on two devices loaded");
                    pass = false;
                }
            }
        }

        if args.bnode {
            pass &= node_case(&path)?;
        }
        if args.bnodeskip {
            pass &= bnodeskip_case(
                &path,
                (&inputs, &plan, &tplan, k3, &meta, &card, device),
                &levers,
            )?;
        }
        if args.rank {
            pass &= rank_case(
                &path,
                (&inputs, &plan, &tplan),
                (&meta, tier_open()),
                &levers,
            )?;
        }
        if (args.bnode || args.rank) && !(args.on_load() || args.bfirst || args.bfeat) {
            return verdict(pass);
        }

        let prompt = prose_ids("engram", PROMPT)?;
        let long = if args.batch
            || args.batch2
            || args.bfault
            || args.blost
            || args.bfirst
            || args.bfeat
        {
            prose_ids("engram", BATCH_P2)?
        } else {
            Vec::new()
        };
        let mut want = Want::default();
        if args.union || args.fault || args.batch2 {
            let t0 = Instant::now();
            let mut m = open(split()?, &plan, None, &meta)?;
            println!(
                "reference load (gate plan, one card) in {:.1} s",
                t0.elapsed().as_secs_f64()
            );
            if args.union || args.fault {
                want.union = Some(union_run(&mut m, &prompt)?);
            }
            if args.batch2 {
                body::prepare_prefill(&mut m)?;
                want.batch2 = Some(batch_run(&mut m, &long)?.0);
            }
            drop(m);
        }

        let load = || -> Result<Deepseek41Model, GateError> {
            let t0 = Instant::now();
            let m = open(split()?, &tplan, Some(tier_open()), &meta)?;
            let (free, total) = m.gpu().mem_info()?;
            println!(
                "tiered load (stage + tier on {card}) in {:.1} s; card free {free} of {total} B \
                 after it",
                t0.elapsed().as_secs_f64()
            );
            Ok(m)
        };
        if args.bfeat {
            let (_, dhp) = dspark::draft_hparams()?;
            let mut f = load()?;
            pass &= batch_feature_case(
                &mut f,
                &long[..FEAT_P],
                &dhp.target_layers,
                &[dhp.window],
                &tier_layers,
            )?;
            drop(f);
            if !(args.on_load() || args.bfirst) {
                return verdict(pass);
            }
        }
        let mut m = load()?;
        if args.bfirst {
            pass &= batch_lost_case(&mut m, &long[..BATCH_FAULT_P], "B3f")?;
            if args.on_load() {
                drop(m);
                m = load()?;
            }
        }

        if let Some(want) = &want.union {
            if args.union {
                let got = union_run(&mut m, &prompt)?;
                pass &= tiered_is_reference(
                    ("T1", "T1 (step, 32 steps, pairs, eager)"),
                    &m,
                    (want, &got),
                    &tier_layers,
                )?;
            }
            if args.fault {
                let first = *tier_layers
                    .first()
                    .ok_or("the tiered plan has no tier layer")?;
                pass &= fault_case(&mut m, &prompt, want, first)?;
            }
        }
        if args.batch {
            let want = steps_run(&mut m, &long[..BATCH_P])?;
            let (got, sent) = batch_run(&mut m, &long[..BATCH_P])?;
            pass &= in_band(
                "B7 (one batch against the steps, then greedy steps)",
                &want,
                &got,
            );
            pass &= sent_every("B7", sent.as_ref(), &tier_layers);
        }
        if let Some(want) = &want.batch2 {
            let (got, sent) = batch_run(&mut m, &long)?;
            pass &= in_band(
                "B7 group (two batches against the reference's call, then greedy steps)",
                want,
                &got,
            );
            pass &= sent_every("B7 group", sent.as_ref(), &tier_layers);
            let group = m.body(NAME)?.prefill_group().map(|(g, _)| g);
            if group == Some(BATCH_GROUP) {
                println!("ok B7 group structure: the call ran as a group of {BATCH_GROUP}");
            } else {
                println!(
                    "FAIL B7 group structure: the prompt group is {group:?}, want {BATCH_GROUP}: \
                     the held routing across a group is not exercised"
                );
                pass = false;
            }
        }
        if args.bfault {
            let first = *tier_layers
                .first()
                .ok_or("the tiered plan has no tier layer")?;
            pass &= batch_fault_case(&mut m, &long[..BATCH_FAULT_P], first)?;
        }
        if args.blost {
            pass &= batch_lost_case(&mut m, &long[..BATCH_FAULT_P], "B3")?;
            if args.lost {
                drop(m);
                m = load()?;
            }
        }
        if args.lost {
            pass &= lost_case(&mut m, &prompt)?;
        }
        drop(m);
        verdict(pass)
    }

    fn verdict(pass: bool) -> Result<(), GateError> {
        println!("{NAME}: {}", bloomery_gpu_gates::tier::tally_line());
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME}");
        Ok(())
    }

    /// B9 (module header), the gate's side: a `tier_server` serving `model` on a free loopback
    /// port, then this binary as the stage under [`NODE_STAGE`], with the node's address and the
    /// TCP link in its environment alone. Both must end 0 within their bounds, the node once the
    /// stage has closed its load; a child past its bound is killed and the clause fails by name.
    fn node_case(model: &str) -> Result<bool, GateError> {
        if !bloomery_levers::tier_nodes()?.is_empty() {
            return Err(format!("{TIER_NODES} is set: B9 sets it for its stage alone").into());
        }
        let (mut node, addr) = crate::node_proc::tier_server(
            crate::node_proc::tier_server_command(Some(Path::new(model)))?,
            "B9 node",
        )?;
        println!("B9: tier_server pid {} serves on {addr}", node.id());
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg(NODE_STAGE)
            .env(TIER_NODES, addr.to_string())
            .env(TIER_LINK, "tcp");
        let mut stage = crate::node_proc::Proc::spawn(cmd, "B9 stage")?;
        println!("B9: stage pid {}", stage.id());
        let mut pass = match stage.wait_within(NODE_STAGE_WITHIN)? {
            Some(s) if s.success() => {
                println!("ok B9 stage: it exited 0");
                true
            }
            Some(s) => {
                println!("FAIL B9 stage: {s}");
                false
            }
            None => {
                println!("FAIL B9 stage: still running after {NODE_STAGE_WITHIN:?}; killed");
                false
            }
        };
        drop(stage);
        match node.wait_within(NODE_CLOSE_WITHIN)? {
            Some(s) if s.success() => {
                println!("ok B9 node: it exited 0 once the stage closed its load");
            }
            Some(s) => {
                println!("FAIL B9 node: {s}");
                pass = false;
            }
            None => {
                println!(
                    "FAIL B9 node: still running {NODE_CLOSE_WITHIN:?} after the stage ended; \
                     killed"
                );
                pass = false;
            }
        }
        Ok(pass)
    }

    /// B9's tier on the node `BLOOMERY_TIER_NODES` names, over the link `BLOOMERY_TIER_LINK`
    /// names, which must be TCP: the stage's environment [`node_case`] set. Another count of
    /// nodes or another link is refused by name.
    fn node_open(card: &str, device: Option<DeviceId>) -> Result<TierOpen, GateError> {
        let nodes = bloomery_levers::tier_nodes()?;
        let [addr] = nodes.as_slice() else {
            return Err(format!(
                "{NODE_STAGE}: {TIER_NODES} names {} nodes; B9's stage takes the one the gate \
                 serves",
                nodes.len()
            )
            .into());
        };
        let link = bloomery_levers::tier_link()?;
        if link != TierLink::Tcp {
            return Err(format!("{NODE_STAGE}: {TIER_LINK} is {link:?}; B9 runs over tcp").into());
        }
        Ok(TierOpen {
            card: TIER_DEVICE,
            name: card.to_string(),
            device,
            node: Some(*addr),
        })
    }

    /// B9 (module header), the stage's side, on `m`, the tiered load with its tier on the node:
    /// the residency lever as a batch feed resolves it, `ids` fed one decode step each and then as
    /// one prompt call, each followed by greedy steps, the call walked against the steps, and what
    /// the call and the steps sent the node tier. Returns the verdict and the call's run and
    /// counters, which [`node_exact`] reads. A refused call is the clause's failure, named.
    fn node_stage(
        mut m: Deepseek41Model,
        levers: &Levers,
        ids: &[u32],
        tier_layers: &[usize],
    ) -> Result<(bool, Option<Staged>), GateError> {
        let pick = levers.residency_at(ResidencyAt {
            serving_place: false,
            check_finite: false,
            route_trace: false,
            prefill_steps: false,
        });
        record::residency_lever(pick).print();
        let mut pass = pick.why != ResidencyWhy::PrefillSteps;
        println!(
            "{} B9 residency: the lever resolves {} ({:?}) for a batch feed",
            if pass { "ok" } else { "FAIL" },
            pick.word,
            pick.why
        );
        {
            let tier = m
                .body(NAME)?
                .hybrid()
                .tiers()
                .first()
                .ok_or("the tiered load holds no tier card")?;
            match tier.node() {
                Some(a) => println!(
                    "ok B9 structure: the tier {} runs on the node at {a}",
                    tier.name()
                ),
                None => {
                    println!(
                        "FAIL B9 structure: the tier {} is a card of this process",
                        tier.name()
                    );
                    pass = false;
                }
            }
        }
        let before = tier_counts(&m)?;
        let want = steps_run(&mut m, ids)?;
        let steps = sent_between(before, tier_counts(&m)?);
        pass &= node_served("B9 steps", steps.as_ref(), tier_layers);
        let (got, sent) = match batch_run(&mut m, ids) {
            Ok(r) => r,
            Err(e) => {
                println!("FAIL B9: the prompt call over the node tier: {e}");
                return Ok((false, None));
            }
        };
        pass &= in_band(
            "B9 (one batch over the node tier against the steps, then greedy steps)",
            &want,
            &got,
        );
        pass &= node_batch_served("B9", sent.as_ref(), tier_layers);
        drop(m);
        Ok((pass, Some((got, sent))))
    }

    /// B9's exactness, on `m`, the same plan with its tier a card of this process: `ids` as one
    /// prompt call and its greedy steps ([`batch_run`]), tokens and logits bit for bit `got`'s run,
    /// the node's, and the host's batch slots — served, and left to the tier's leg — `got`'s
    /// counts. The node runs the tier card's block on the q8_1 bytes the tier card quantizes for
    /// itself, and its rows land where the tier card's copy puts them, so the call is the same
    /// program on the same inputs, and the host serves what it serves beside a tier card.
    fn node_exact(
        mut m: Deepseek41Model,
        ids: &[u32],
        got: Option<Staged>,
    ) -> Result<bool, GateError> {
        let Some((got, node)) = got else {
            println!("FAIL B9 exact: no prompt call over the node to compare");
            return Ok(false);
        };
        let (want, card) = batch_run(&mut m, ids)?;
        let mut pass = same(
            "B9 exact (the call over the node against the same plan's with its tier a card)",
            &want,
            &got,
        );
        let (Some(node), Some(card)) = (node, card) else {
            println!("FAIL B9 exact slots: a load holds no tier card");
            return Ok(false);
        };
        let ok = (node.host_slots, node.tier_slots) == (card.host_slots, card.tier_slots);
        println!(
            "{} B9 exact slots: over the call over the node the host served {} slots and left {} \
             to the node's leg; with the tier a card it served {} and left {} to the card's leg",
            if ok { "ok" } else { "FAIL" },
            node.host_slots,
            node.tier_slots,
            card.host_slots,
            card.tier_slots
        );
        pass &= ok;
        Ok(pass)
    }

    /// T1's checks on `m`, the tiered load, after `union_run` left `got`: the run is the
    /// reference's `want` bit for bit, the stage graph has the reference's node count, and every
    /// layer of `tier_layers` was sent at least one routed slot over the run, or the clause proves
    /// nothing there. `tag` names the clause on the lines, `what` the run.
    fn tiered_is_reference(
        (tag, what): (&str, &str),
        m: &Deepseek41Model,
        (want, got): (&Run, &Run),
        tier_layers: &[usize],
    ) -> Result<bool, GateError> {
        let mut pass = same(what, want, got);
        if got.stage_nodes != want.stage_nodes {
            println!(
                "FAIL {tag} structure: stage graph of {:?} nodes, the reference's {:?}",
                got.stage_nodes, want.stage_nodes
            );
            pass = false;
        } else {
            println!(
                "ok {tag} structure: stage graph of {:?} nodes, the reference's",
                got.stage_nodes
            );
        }
        let body = m.body(NAME)?;
        let tier = body
            .hybrid()
            .tiers()
            .first()
            .ok_or("the tiered load holds no tier card")?;
        let st = tier.stats();
        let first = tier.set().layers().start;
        let missing: Vec<usize> = tier_layers
            .iter()
            .copied()
            .filter(|&l| st.layer_hits.get(l - first).copied().unwrap_or(0) == 0)
            .collect();
        let hits: u64 = st.layer_hits.iter().sum();
        println!(
            "tier: {} layers asked, {} settles ({} early), hits {hits} over {} layers; graph \
             nodes step {:?} pair {:?}",
            st.issued,
            st.settles,
            st.settle_early,
            tier_layers.len(),
            tier.graph_nodes(Chain::Step),
            tier.graph_nodes(Chain::Pair)
        );
        if missing.is_empty() {
            println!("ok {tag} precondition: every tier layer was sent a routed slot");
        } else {
            println!(
                "FAIL {tag} precondition: tier layers {missing:?} were sent no routed slot; {tag} \
                 proves nothing there"
            );
            pass = false;
        }
        Ok(pass)
    }

    /// B11 (module header), the gate's side: a rank file built from the
    /// fixture's own routing — the prompt's picks per layer, collected as
    /// [`bnodeskip_stage`] runs them, the ids it never picked coldest — then
    /// this binary as a child ([`NODESKIP_STAGE`]) twice under the file in its
    /// environment alone: once over the loopback `tier_server` the gate serves
    /// (`BLOOMERY_TIER_NODES`, the TCP link) and once with the tier a card of
    /// its own process. The node child must leave a `node_skipped` count on
    /// every step, and the two children's tokens and logits must agree bit for
    /// bit: the skip changes which layers the node is sent, never a value —
    /// the same experts on the same homes. A child past its bound is killed
    /// and the clause fails by name.
    fn bnodeskip_case(
        model: &str,
        (inputs, plan, tplan, k3, meta, card, device): (
            &PlanInputs,
            &Plan<'_>,
            &Plan<'_>,
            usize,
            &BodyMeta,
            &str,
            Option<DeviceId>,
        ),
        levers: &Levers,
    ) -> Result<bool, GateError> {
        if levers.expert_rank().is_some() || plan.expert_rank.is_some() {
            return Err(format!("{EXPERT_RANK} is set: B11 sets it for its stage alone").into());
        }
        if !bloomery_levers::tier_nodes()?.is_empty() {
            return Err(format!("{TIER_NODES} is set: B11 sets it for its stage alone").into());
        }
        let dir = std::env::temp_dir().join(format!("bloomery-b11-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let _clean = RemoveDir(dir.clone());
        // The fixture's own routing: the prompt's picks per layer, over the
        // tiered plan with its tier a card of this process. The routing reads
        // no placement, so the ranked plan picks the same ids.
        let prompt = prose_ids("engram", PROMPT)?;
        let mut picks: Vec<Vec<u32>> = vec![Vec::new(); inputs.model.layers];
        {
            let tier = TierOpen {
                card: TIER_DEVICE,
                name: card.to_string(),
                device,
                node: None,
            };
            let mut m = open(Split::open(model)?, tplan, Some(tier), meta)?;
            m.set_mode(StepMode::Graph);
            m.reset()?;
            for &id in &prompt {
                m.step(&[id])?;
                for (l, ours) in m.body(NAME)?.hybrid().step_ids().iter().enumerate() {
                    picks[l].extend_from_slice(ours);
                }
            }
        }
        let traced: usize = picks.iter().map(Vec::len).sum::<usize>();
        let busiest = picks
            .iter()
            .enumerate()
            .max_by_key(|(l, p)| p.len() * 1000 + l)
            .map_or((0, 0), |(l, p)| (l, p.len()));
        println!(
            "B11 trace: {traced} picks over {} positions (layer {busiest:?} the busiest)",
            prompt.len()
        );
        if traced == 0 {
            return Err("B11: the trace load's steps recorded no picks".into());
        }
        let file = write_skip_rank(&dir, inputs, plan, k3, prompt.len(), &picks)?;
        println!("B11: rank file {}", file.display());
        // The clause's precondition, from the same trace: a layer whose whole
        // tier set (its coldest k3 under the rank) the prompt never routes.
        let mut free = Vec::new();
        for (layer, seen) in picks.iter().enumerate() {
            let routed = plan
                .model
                .tensors
                .iter()
                .any(|t| t.role == Role::RoutedExperts && t.layer == Some(layer));
            if !routed {
                continue;
            }
            let n = plan.model.experts;
            let never = (0..n as u32).filter(|id| !seen.contains(id)).count();
            if never >= k3 {
                free.push(layer);
            }
        }
        if free.is_empty() {
            println!(
                "FAIL B11 precondition: no layer's coldest {k3} ids are all unrouted; the clause \
                 proves nothing"
            );
            return Ok(false);
        }
        println!("ok B11 precondition: the rank holds layer {free:?}'s tier experts unrouted");
        let (mut node, addr) = crate::node_proc::tier_server(
            crate::node_proc::tier_server_command(Some(Path::new(model)))?,
            "B11 node",
        )?;
        println!("B11: tier_server pid {} serves on {addr}", node.id());
        // Each child writes its run beside the rank file; they run one after
        // the other, so the second overwrites the first and each is read as
        // its child ends.
        let rows_of = |_tag: &str| dir.join(NODESKIP_ROWS);
        let mut run = None;
        for (tag, over) in [("node", true), ("card", false)] {
            let mut cmd = Command::new(std::env::current_exe()?);
            cmd.arg(NODESKIP_STAGE).env(EXPERT_RANK, &file);
            if over {
                cmd.env(TIER_NODES, addr.to_string()).env(TIER_LINK, "tcp");
            }
            let mut stage = crate::node_proc::Proc::spawn(cmd, "B11 stage")?;
            println!("B11: {tag} stage pid {}", stage.id());
            let mut pass = match stage.wait_within(NODESKIP_STAGE_WITHIN)? {
                Some(s) if s.success() => {
                    println!("ok B11 {tag} stage: it exited 0");
                    true
                }
                Some(s) => {
                    println!("FAIL B11 {tag} stage: {s}");
                    false
                }
                None => {
                    println!(
                        "FAIL B11 {tag} stage: still running after {NODESKIP_STAGE_WITHIN:?}; killed"
                    );
                    false
                }
            };
            drop(stage);
            let theirs: Vec<Vec<u32>> = std::fs::read_to_string(dir.join("trace.tsv"))?
                .lines()
                .map(|line| {
                    line.split_once('\t').map_or(Vec::new(), |(_, ids)| {
                        ids.split(',')
                            .filter(|s| !s.is_empty())
                            .filter_map(|s| s.parse().ok())
                            .collect()
                    })
                })
                .collect();
            let differs: Vec<usize> = theirs
                .iter()
                .zip(&picks)
                .enumerate()
                .filter(|(_, (a, b))| id_set(a) != id_set(b))
                .map(|(l, _)| l)
                .collect();
            println!(
                "B11 {tag} routing: {} of {} routed layers pick other ids than the trace load",
                differs.len(),
                theirs.len(),
            );
            let rows = read_rows(&rows_of(tag))?;
            match run.as_ref() {
                Some(keep) => {
                    pass &= same(
                        "B11 bits (the skip over the node against the same plan's with its tier a \
                         card)",
                        keep,
                        &rows,
                    );
                }
                None => run = Some(rows),
            }
            if !pass {
                drop(node);
                return Ok(false);
            }
        }
        match node.wait_within(NODE_CLOSE_WITHIN)? {
            Some(s) if s.success() => {
                println!("ok B11 node: it exited 0 once the stage closed its load");
            }
            other => {
                println!("FAIL B11 node: {other:?}");
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// A pick list's distinct ids, ascending.
    fn id_set(ids: &[u32]) -> Vec<u32> {
        let mut set = ids.to_vec();
        set.sort_unstable();
        set.dedup();
        set
    }

    /// B11's stage, on `m`, the ranked tiered plan with its tier on the node
    /// the gate serves or a card of this process: the prompt one decode step a
    /// position, each step's `node_skipped` held to at least one layer (the
    /// rank keeps a layer's tier experts unrouted) and the totals printed.
    /// Returns the run for the gate's bit comparison.
    fn bnodeskip_stage(
        m: &mut Deepseek41Model,
        ids: &[u32],
        on_node: bool,
        picks: &mut Vec<Vec<u32>>,
    ) -> Result<(bool, Run), GateError> {
        picks.clear();
        picks.resize(m.body(NAME)?.hybrid().slots().layers().len(), Vec::new());
        let mut run = Run::default();
        let mut pass = true;
        let (mut before, mut worst) = (0_u64, u64::MAX);
        m.set_mode(StepMode::Graph);
        m.reset()?;
        for &id in ids {
            let tok = m.step(&[id])?;
            run.push(tok, m.logits()?);
            let hybrid = m.body(NAME)?.hybrid();
            for (l, ours) in hybrid.step_ids().iter().enumerate() {
                picks[l].extend_from_slice(ours);
            }
            let st = hybrid.stats();
            let skipped = st.node_skipped - before;
            before = st.node_skipped;
            worst = worst.min(skipped);
        }
        let hybrid = m.body(NAME)?.hybrid();
        let tiers = hybrid.tiers().first();
        if let Some(tier) = tiers {
            let first = tier.set().layers().start;
            let ids: Vec<u32> = tier.set().ids(first).map_or(Vec::new(), <[u32]>::to_vec);
            let in_set = id_set(&picks[first])
                .iter()
                .filter(|id| ids.contains(id))
                .count();
            println!(
                "B11 tier set of layer {first}: {:?} .. {:?} ({} ids); the run's distinct picks \
                 in it {in_set}",
                &ids[..ids.len().min(4)],
                &ids[ids.len().saturating_sub(4)..],
                ids.len(),
            );
        }
        println!(
            "B11 steps: {before} layers skipped over {} positions (least a step {worst}); tier \
             stats: issued {}, hits {}",
            ids.len(),
            tiers.map_or(0, |t| t.stats().issued),
            tiers.map_or(0, |t| t.stats().layer_hits.iter().sum::<u64>()),
        );
        // The rank's trace load routes as an unranked load: the rank moves
        // the experts, and the fixture's near-tied router flips with the sum
        // order, so the skip count is what the run itself routes, held only
        // to happen.
        if on_node && before == 0 {
            println!("FAIL B11: the run over the node skipped no layer; the clause proves nothing");
            pass = false;
        }
        Ok((pass, run))
    }

    /// The rank file for B11 in `dir`: the format of B10's `write_rank`, each
    /// routed layer's ids once, written for the carve the gate's `tiered`
    /// makes of them. That carve takes the tier from the card list's last
    /// `k3` ids in id order, so the row's first `n_l` ids (the rank's stage
    /// card) are the picks the trace saw hottest and the never-picked ids the
    /// trace's routing leaves, ordered so every id the tier takes is one the
    /// prompt never routes. `picks` holds the trace's picked ids per layer;
    /// `n_l` is the plan's card experts a routed layer.
    fn write_skip_rank(
        dir: &Path,
        inputs: &PlanInputs,
        plan: &Plan<'_>,
        k3: usize,
        positions: usize,
        picks: &[Vec<u32>],
    ) -> Result<PathBuf, GateError> {
        let n = plan.model.experts;
        let ids_of = u32::try_from(n).map_err(|_| format!("B11: {n} experts do not fit u32"))?;
        let mut layers = BTreeSet::new();
        for t in &plan.model.tensors {
            if t.role == Role::RoutedExperts {
                layers.insert(
                    t.layer
                        .ok_or_else(|| format!("{}: a routed stack without a layer", t.name))?,
                );
            }
        }
        if layers.is_empty() {
            return Err("B11: the model has no routed layer".into());
        }
        let mut text = format!(
            "# router-rank\tn_expert={n}\tn_used={}\tsets=gate-b11\tpositions={positions}\n",
            u64::try_from(inputs.hp.experts.n_used)?,
        );
        for layer in layers {
            let seen = picks
                .get(layer)
                .map_or(&[][..], <Vec<u32>>::as_slice)
                .iter()
                .copied()
                .collect::<BTreeSet<_>>();
            let mut picked: Vec<u32> = seen.iter().copied().collect();
            let mut never: Vec<u32> = (0..ids_of).filter(|id| !seen.contains(id)).collect();
            // The card keeps the hottest picked and the never-picked the tier
            // takes, all above the picked it keeps, so the carve's id-ordered
            // cut lands inside the never-picked alone.
            let keep = n_l_of(plan, layer).unwrap_or(0).saturating_sub(k3);
            picked.sort_unstable();
            let hot: Vec<u32> = picked[..picked.len().min(keep)].to_vec();
            let floor = hot.last().copied().unwrap_or(u32::MAX);
            while never.len() > k3 && never[never.len() - k3] < floor {
                never.pop();
            }
            let mut card: Vec<u32> = hot;
            card.extend_from_slice(&never[never.len().saturating_sub(k3)..]);
            let held: std::collections::BTreeSet<u32> = card.iter().copied().collect();
            let mut row = card;
            row.extend((0..ids_of).filter(|id| !held.contains(id)));
            let ids: Vec<String> = row.iter().map(u32::to_string).collect();
            text.push_str(&format!("{layer}\t{}\n", ids.join(",")));
        }
        let file = dir.join("rank.tsv");
        std::fs::write(&file, text)?;
        Ok(file)
    }

    /// The plan's card experts of routed `layer`, when it holds them all.
    fn n_l_of(plan: &Plan<'_>, layer: usize) -> Option<usize> {
        plan.n_l
            .get(layer)
            .copied()
            .and_then(|n| usize::try_from(n).ok())
    }

    /// B10 (module header), the gate's side: a rank file for `plan`'s model ([`write_rank`]), then
    /// this binary as the child under [`RANK_STAGE`] with the file in its environment alone, and
    /// the prompt fed one step a position over the tiered carve of `tplan`, the plan without the
    /// file. The child must end 0 within its bound and leave its run beside the rank file; its
    /// positions are walked against the gate's within the logits band. A child past its bound is
    /// killed and the clause fails by name.
    fn rank_case(
        model: &str,
        (inputs, plan, tplan): (&PlanInputs, &Plan<'_>, &Plan<'_>),
        (meta, tier): (&BodyMeta, TierOpen),
        levers: &Levers,
    ) -> Result<bool, GateError> {
        if levers.expert_rank().is_some() || plan.expert_rank.is_some() {
            return Err(format!("{EXPERT_RANK} is set: B10 sets it for its stage alone").into());
        }
        let dir = std::env::temp_dir().join(format!("bloomery-b10-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        let _clean = RemoveDir(dir.clone());
        let file = write_rank(&dir, inputs, plan)?;
        println!("B10: rank file {}", file.display());
        let mut cmd = Command::new(std::env::current_exe()?);
        cmd.arg(RANK_STAGE).env(EXPERT_RANK, &file);
        let mut stage = crate::node_proc::Proc::spawn(cmd, "B10 stage")?;
        println!("B10: stage pid {}", stage.id());
        let ended = stage.wait_within(RANK_STAGE_WITHIN)?;
        drop(stage);
        match ended {
            Some(s) if s.success() => println!("ok B10 stage: it exited 0"),
            Some(s) => {
                println!("FAIL B10 stage: {s}");
                return Ok(false);
            }
            None => {
                println!("FAIL B10 stage: still running after {RANK_STAGE_WITHIN:?}; killed");
                return Ok(false);
            }
        }
        let got = read_rows(&dir.join(RANK_ROWS))?;
        let mut m = open(Split::open(model)?, tplan, Some(tier), meta)?;
        let want = rank_run(&mut m, &prose_ids("engram", PROMPT)?)?;
        drop(m);
        in_band_from_zero(
            "B10 (the prompt's steps under the rank against the plan without it)",
            &want,
            &got,
        )
    }

    /// The prompt `ids` fed one decode step per position on `m`, graph mode from a reset: each
    /// position's logits row with the id that was fed as its token. Both arms of the rank walk
    /// feed the same ids, so the walk judges every position's row instead of parting at the first
    /// argmax flip; every position reads the placement margin ([`in_band_from_zero`]).
    fn rank_run(m: &mut Deepseek41Model, ids: &[u32]) -> Result<Run, GateError> {
        let mut run = Run::default();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        for &id in ids {
            m.step(&[id])?;
            run.push(id, m.logits()?);
        }
        Ok(run)
    }

    /// `run`'s first `n` positions' tokens and logits rows in `path`: the position count and the
    /// row width as two little-endian `u32`s, the tokens, then the rows' `f32`s, all
    /// little-endian.
    fn write_rows(path: &Path, run: &Run, n: usize) -> Result<(), GateError> {
        use std::io::Write;
        let (Some(tokens), Some(rows)) = (run.tokens.get(..n), run.logits.get(..n)) else {
            return Err(format!("B10: the run holds fewer than {n} positions").into());
        };
        let width = rows.first().map_or(0, Vec::len);
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&u32::try_from(n)?.to_le_bytes())?;
        f.write_all(&u32::try_from(width)?.to_le_bytes())?;
        for t in tokens {
            f.write_all(&t.to_le_bytes())?;
        }
        for row in rows {
            if row.len() != width {
                return Err(format!(
                    "B10: a logits row of {} values, the first has {width}",
                    row.len()
                )
                .into());
            }
            for x in row {
                f.write_all(&x.to_le_bytes())?;
            }
        }
        f.flush()?;
        Ok(())
    }

    /// The run [`write_rows`] wrote; a file of another size than its two counts give is refused
    /// by name.
    fn read_rows(path: &Path) -> Result<Run, GateError> {
        let bytes = std::fs::read(path).map_err(|e| format!("B10: {}: {e}", path.display()))?;
        let word = |at: usize| -> Result<[u8; 4], GateError> {
            bytes
                .get(at..at + 4)
                .and_then(|b| b.try_into().ok())
                .ok_or_else(|| "B10: the stage's rows file is cut short".into())
        };
        let n = usize::try_from(u32::from_le_bytes(word(0)?))?;
        let width = usize::try_from(u32::from_le_bytes(word(4)?))?;
        let want = 8 + 4 * n + 4 * n * width;
        if bytes.len() != want {
            return Err(format!(
                "B10: the stage's rows file holds {} bytes, {n} positions of {width} values take \
                 {want}",
                bytes.len()
            )
            .into());
        }
        let tokens = (0..n)
            .map(|i| word(8 + 4 * i).map(u32::from_le_bytes))
            .collect::<Result<Vec<_>, _>>()?;
        let rows_at = 8 + 4 * n;
        let logits = (0..n)
            .map(|i| {
                (0..width)
                    .map(|j| word(rows_at + 4 * (i * width + j)).map(f32::from_le_bytes))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Run {
            tokens,
            logits,
            stage_nodes: None,
        })
    }

    /// A directory removed when dropped.
    struct RemoveDir(PathBuf);

    impl Drop for RemoveDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The rank file for `plan`'s model in `dir`: the format of `tools/ref/router-rank.py`, every
    /// routed layer's `n_expert` ids once in a stride order. The stride is the smallest of 3 or
    /// more with no factor in common with `n_expert`, offset by the layer, so a layer's first ids
    /// are neither its id prefix nor the same ids of another layer. No count of routed experts
    /// that has no such stride, and no model whose routed stacks differ in expert count, has a
    /// rank: refused by name.
    fn write_rank(dir: &Path, inputs: &PlanInputs, plan: &Plan<'_>) -> Result<PathBuf, GateError> {
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        let n = plan.model.experts;
        let stride = (3..n)
            .find(|&s| gcd(s, n) == 1)
            .ok_or_else(|| format!("B10: {n} experts have no stride of 3 or more coprime to it"))?;
        let mut layers = BTreeSet::new();
        for t in &plan.model.tensors {
            if t.role == Role::RoutedExperts {
                layers.insert(
                    t.layer
                        .ok_or_else(|| format!("{}: a routed stack without a layer", t.name))?,
                );
            }
        }
        if layers.is_empty() {
            return Err("B10: the model has no routed layer".into());
        }
        let mut text = format!(
            "# router-rank\tn_expert={n}\tn_used={}\tsets=gate-b10\tpositions=1\n",
            u64::try_from(inputs.hp.experts.n_used)?
        );
        for layer in layers {
            let ids: Vec<String> = (0..n)
                .map(|i| ((i * stride + layer as u64) % n).to_string())
                .collect();
            text.push_str(&format!("{layer}\t{}\n", ids.join(",")));
        }
        let file = dir.join("rank.tsv");
        std::fs::write(&file, text)?;
        Ok(file)
    }

    /// B10's plan clauses on `plan`, the gate plan under the rank, against `base`, the same plan
    /// without it: it names a rank; its counts (`n_l`, the tier's, the card's experts) are
    /// `base`'s; each routed stack's card list is the layer's hottest `n_l` ranked ids; and the
    /// rank moved a card list off the id prefix, or the clause proves nothing.
    fn rank_plan(plan: &Plan<'_>, base: &Plan<'_>) -> Result<bool, GateError> {
        let Some(rank) = plan.expert_rank.as_deref() else {
            println!(
                "FAIL B10 plan: the plan names no rank: {EXPERT_RANK} did not reach the planner"
            );
            return Ok(false);
        };
        println!(
            "B10: the plan names the rank {} ({} layers, sets {}, {} positions)",
            rank.path(),
            rank.layers().count(),
            rank.sets(),
            rank.positions()
        );
        let mut pass = true;
        let same_counts = plan.n_l == base.n_l
            && plan.tier_n_l == base.tier_n_l
            && plan.cards[0].experts == base.cards[0].experts;
        println!(
            "{} B10 plan counts: n_l, the tier's and the card's {} experts are the plan without \
             the rank's",
            if same_counts { "ok" } else { "FAIL" },
            plan.cards[0].experts
        );
        pass &= same_counts;
        let (mut stacks, mut moved) = (0usize, 0usize);
        let mut wrong = Vec::new();
        for row in &plan.rows {
            let t = &plan.model.tensors[row.tensor];
            if t.role != Role::RoutedExperts {
                continue;
            }
            let Some(card) = row.segments.iter().find(|s| s.device == Device::Card(0)) else {
                continue;
            };
            let layer = t
                .layer
                .ok_or_else(|| format!("{}: a routed stack without a layer", t.name))?;
            let have = card
                .experts
                .as_ref()
                .ok_or_else(|| format!("{}: a card segment without experts", t.name))?
                .ids();
            let hottest = rank
                .ranked(layer)
                .and_then(|ids| ids.get(..have.len()))
                .map(|ids| {
                    let mut ids = ids.to_vec();
                    ids.sort_unstable();
                    ids
                });
            stacks += 1;
            if hottest.as_deref() != Some(have) {
                wrong.push(t.name.clone());
            }
            if have.iter().copied().ne(0..u32::try_from(have.len())?) {
                moved += 1;
            }
        }
        if wrong.is_empty() {
            println!(
                "ok B10 plan lists: the card list of each of {stacks} routed stacks is its \
                 layer's hottest ranked ids"
            );
        } else {
            println!(
                "FAIL B10 plan lists: stacks {wrong:?} are not their layer's hottest ranked ids"
            );
            pass = false;
        }
        if moved == 0 {
            println!(
                "FAIL B10 precondition: the rank left every card list its id prefix; the clause \
                 proves nothing"
            );
            pass = false;
        } else {
            println!(
                "ok B10 precondition: the rank moved the card list of {moved} of {stacks} stacks \
                 off the id prefix"
            );
        }
        Ok(pass)
    }

    fn rank_verdict(pass: bool) -> Result<(), GateError> {
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME} {RANK_STAGE}");
        Ok(())
    }

    /// The precondition of B9 `what`: the steps sent every layer of `tier_layers` at least one
    /// routed slot through the node.
    fn node_served(what: &str, sent: Option<&Sent>, tier_layers: &[usize]) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what} precondition: the tiered load holds no tier card");
            return false;
        };
        let hit = |l: usize| {
            l.checked_sub(sent.first)
                .and_then(|i| sent.hits.get(i))
                .copied()
                .unwrap_or(0)
        };
        let missing: Vec<usize> = tier_layers
            .iter()
            .copied()
            .filter(|&l| hit(l) == 0)
            .collect();
        let slots: u64 = tier_layers.iter().map(|&l| hit(l)).sum();
        println!(
            "{what} node: stats().issued {}, stats().layer_hits {slots} over {} tier layers",
            sent.issued,
            tier_layers.len()
        );
        if missing.is_empty() {
            println!(
                "ok {what} precondition: the steps sent every tier layer a slot through the node"
            );
            true
        } else {
            println!(
                "FAIL {what} precondition: tier layers {missing:?} were sent no routed slot by the \
                 steps; the clause proves nothing there"
            );
            false
        }
    }

    /// B9 `what`'s pin on the prompt call: the node computed its experts' block of every tiered
    /// layer-batch — its batch leg served and settled once a block, a routed slot on every layer
    /// of `tier_layers`, no decode layer asked of it — and the host left it exactly the node's
    /// slots: those the host's batch services left to a tier's leg are the slots the legs sent
    /// the node, so the host served none of them.
    // PIN(2026-11-05): inverted from "the call sends the node nothing" — the node now serves its
    // block of a prompt batch (`host::node::block`), so the host no longer serves its experts.
    fn node_batch_served(what: &str, sent: Option<&Sent>, tier_layers: &[usize]) -> bool {
        let Some(sent) = sent else {
            println!("FAIL {what} node: the tiered load holds no tier card");
            return false;
        };
        let hit = |l: usize| {
            l.checked_sub(sent.first)
                .and_then(|i| sent.hits.get(i))
                .copied()
                .unwrap_or(0)
        };
        let missing: Vec<usize> = tier_layers
            .iter()
            .copied()
            .filter(|&l| hit(l) == 0)
            .collect();
        let slots: u64 = tier_layers.iter().map(|&l| hit(l)).sum();
        let st = sent.stats;
        let ok = st.served > 0
            && st.settles == st.served
            && missing.is_empty()
            && sent.issued == 0
            && sent.tier_slots == slots;
        println!(
            "{} {what} node: over the prompt call tier_batch_stats().served {} (settles {}, {:.2} ms \
             waited), stats().issued moved {}, stats().layer_hits on the {} tier layers {slots} \
             (none on {missing:?}); the host's batch slots: {} left to the node's leg, {} served",
            if ok { "ok" } else { "FAIL" },
            st.served,
            st.settles,
            st.settle_ns as f64 / 1e6,
            sent.issued,
            tier_layers.len(),
            sent.tier_slots,
            sent.host_slots
        );
        ok
    }

    /// B9's stage child's end: its checks' verdict, which [`node_case`] reads as its exit status.
    fn stage_verdict(pass: bool) -> Result<(), GateError> {
        if !pass {
            return Err(checks_failed());
        }
        println!("PASSED: {NAME} {NODE_STAGE}");
        Ok(())
    }

    /// One prompt call's hand-over: each handed row with its position, and what the call sent the
    /// tier.
    struct Handed {
        rows: Vec<(usize, Vec<f32>)>,
        sent: Option<Sent>,
    }

    /// `ids` fed as one prompt call from a reset on `m`, whose tap is attached, handing over the
    /// features of its last `window` positions: clause `what`'s call. `None`, after printing why,
    /// when the call fails or hands over rows of other positions than those.
    fn hand_over(
        m: &mut Deepseek41Model,
        ids: &[u32],
        window: usize,
        what: &str,
    ) -> Result<Option<Handed>, GateError> {
        let width = m.body(NAME)?.feature_width();
        m.reset()?;
        let before = tier_counts(m)?;
        let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
        let mut sink = |first: u32, chunk: &[f32]| -> Result<(), GpuError> {
            for (r, row) in chunk.chunks_exact(width).enumerate() {
                rows.push((first as usize + r, row.to_vec()));
            }
            Ok(())
        };
        let call = body::prefill_with(
            m,
            ids,
            Some(FeatureRows {
                window,
                sink: &mut sink,
            }),
        );
        if let Err(e) = call {
            println!("FAIL {what}: the prompt call that hands features over failed: {e}");
            return Ok(None);
        }
        let sent = sent_between(before, tier_counts(m)?);
        let from = ids.len() - window.min(ids.len());
        let positions: Vec<usize> = rows.iter().map(|(p, _)| *p).collect();
        if positions.iter().copied().ne(from..ids.len()) {
            println!(
                "FAIL {what}: rows of positions {:?}..{:?} ({} rows), want {from}..{}",
                positions.first(),
                positions.last(),
                positions.len(),
                ids.len()
            );
            return Ok(None);
        }
        Ok(Some(Handed { rows, sent }))
    }

    /// The every-position call's feature `rows` against the steps' `want`, clause `what`: the
    /// RMS |Δ| over the steps' RMS |x|, every value of every row, held to [`FEATURE_REL_PIN`];
    /// printed beside the largest |Δ| and the row whose |Δ| sits furthest over its own largest
    /// |x|. A NaN on either side is infinitely far.
    fn features_near(what: &str, rows: &[(usize, Vec<f32>)], want: &[Vec<f32>]) -> bool {
        let (mut abs, mut worst, mut at) = (0.0_f64, 0.0_f64, 0);
        let (mut dd, mut xx, mut nd) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (p, row) in rows {
            // The control: the same row against the step one position earlier, which a tap
            // reading the wrong position would match instead.
            if let Some(prev) = p.checked_sub(1).and_then(|q| want.get(q)) {
                nd += row
                    .iter()
                    .zip(prev)
                    .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
                    .sum::<f64>();
            }
            let step = &want[*p];
            if row.len() != step.len() {
                println!(
                    "FAIL {what}: position {p}'s row holds {} values, the step's {}",
                    row.len(),
                    step.len()
                );
                return false;
            }
            let (mut d, mut x) = (0.0_f64, 0.0_f64);
            for (&a, &b) in row.iter().zip(step) {
                let diff = (f64::from(a) - f64::from(b)).abs();
                let diff = if diff.is_nan() { f64::INFINITY } else { diff };
                d = d.max(diff);
                x = x.max(f64::from(b).abs());
                dd += diff * diff;
                xx += f64::from(b) * f64::from(b);
            }
            let r = if x > 0.0 {
                d / x
            } else if d > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            abs = abs.max(d);
            if r > worst {
                (worst, at) = (r, *p);
            }
        }
        let rel = if xx > 0.0 {
            (dd / xx).sqrt()
        } else if dd > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
        let ok = rel <= FEATURE_REL_PIN;
        let control = if xx > 0.0 { (nd / xx).sqrt() } else { 0.0 };
        println!(
            "{} {what} against the steps: {} positions' features, RMS |Δ| over the steps' RMS |x| \
             {rel:.3e} (pin {FEATURE_REL_PIN}; against the step one position earlier \
             {control:.3e}); max |Δ| {abs:.3e}, the furthest row position {at} at {worst:.3e} of \
             its max |x|",
            if ok { "ok" } else { "FAIL" },
            rows.len()
        );
        ok
    }

    /// B8 (module header): the tap attached to the fresh tiered load `m`,
    /// whose prompt batch's buffers the load made, then those buffers
    /// prepared again; `ids` fed one step each against a prompt call of them
    /// handing over every position's rows, and against a call for each of
    /// `windows`, each narrower than `ids`.
    fn batch_feature_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        layers: &[usize],
        windows: &[usize],
        tier_layers: &[usize],
    ) -> Result<bool, GateError> {
        body::attach_features(m, layers)?;
        body::prepare_prefill(m)?;
        let width = m.body(NAME)?.feature_width();
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let mut want: Vec<Vec<f32>> = Vec::with_capacity(ids.len());
        for (p, &id) in ids.iter().enumerate() {
            m.step(&[id])?;
            let (gpu, _, b) = m.body_parts(NAME)?;
            let f = b.read_features(gpu, 1)?;
            if f.pos as usize != p || f.values.len() != width {
                return Err(format!(
                    "B8: the step at position {p} read features of position {} ({} values, the \
                     tap's {width})",
                    f.pos,
                    f.values.len()
                )
                .into());
            }
            want.push(f.values.to_vec());
        }
        let mut ok = true;
        let batches = body::batch_count(ids.len());
        let every = "B8 (every position)";
        let full = hand_over(m, ids, ids.len(), every)?;
        if let Some(full) = &full {
            ok &= features_near(every, &full.rows, &want);
            ok &= sent_every(every, full.sent.as_ref(), tier_layers);
            let first_batch = body::batches(0, ids.len()).first().map_or(0, |r| r.end);
            let from = full.rows.first().map_or(ids.len(), |(p, _)| *p);
            if from < first_batch {
                println!("ok {every} precondition: the first batch handed its rows over too");
            } else {
                println!(
                    "FAIL {every} precondition: no row from the first batch (ends at \
                     {first_batch}); the group's first set's rows are not exercised"
                );
                ok = false;
            }
        } else {
            ok = false;
        }
        for &window in windows {
            let what = format!("B8 (window {window})");
            let Some(handed) = hand_over(m, ids, window, &what)? else {
                ok = false;
                continue;
            };
            let Some(full) = &full else {
                continue;
            };
            let from = ids.len() - window.min(ids.len());
            let apart = handed.rows.iter().find_map(|(p, row)| {
                row.iter()
                    .zip(&full.rows[*p].1)
                    .position(|(a, b)| a.to_bits() != b.to_bits())
                    .map(|v| (*p, v))
            });
            if let Some((p, v)) = apart {
                println!(
                    "FAIL {what}: position {p}'s features apart from the every-position call's at \
                     value {v}"
                );
                ok = false;
            } else {
                println!(
                    "ok {what}: {} positions' features, {from}..{}, bit for bit the every-position \
                     call's",
                    handed.rows.len(),
                    ids.len()
                );
            }
        }
        let group = m.body(NAME)?.prefill_group().map(|(g, _)| g);
        if group == Some(BATCH_GROUP) && batches == BATCH_GROUP {
            println!("ok B8 structure: {batches} batches, one group of {BATCH_GROUP}");
        } else {
            println!(
                "FAIL B8 structure: {batches} batches under a group of {group:?}, want \
                 {BATCH_GROUP} of each"
            );
            ok = false;
        }
        Ok(ok)
    }

    /// T2: a fault on the tier card's word at `layer` before a step.
    fn fault_case(
        m: &mut Deepseek41Model,
        prompt: &[u32],
        want: &Run,
        layer: usize,
    ) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        for &id in &prompt[..prompt.len() - 1] {
            m.step(&[id])?;
        }
        let want_fault = raise_tier_fault(m, layer)?;
        let r = m.step(&[prompt[prompt.len() - 1]]);
        let mut ok = match &r {
            Err(GpuError::Fault { fault, .. }) if *fault == want_fault => {
                println!("ok T2: the step fails with the tier's fault: {fault}");
                true
            }
            other => {
                println!("FAIL T2: the step gave {other:?}, want the tier's fault {want_fault}");
                false
            }
        };
        if m.poisoned() != Some(want_fault) {
            println!("FAIL T2: the model's poison is {:?}", m.poisoned());
            ok = false;
        }
        m.reset()?;
        let mut tok = 0;
        for &id in prompt {
            tok = m.step(&[id])?;
        }
        let (i, logits) = (prompt.len() - 1, m.logits()?);
        let bits = logits
            .iter()
            .zip(&want.logits[i])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        if tok == want.tokens[i] && bits {
            println!("ok T2: after the reset the prompt's last position is the reference's");
        } else {
            println!(
                "FAIL T2: after the reset token {tok} (reference {})",
                want.tokens[i]
            );
            ok = false;
        }
        Ok(ok)
    }

    /// Raise a fault on the tier card's word at `layer`, as a quantizer that
    /// met a NaN there; returns the fault a read of the word gives.
    fn raise_tier_fault(m: &mut Deepseek41Model, layer: usize) -> Result<Fault, GateError> {
        let body = m.body(NAME)?;
        let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
        let g = tier.gpu()?;
        let s = g.stream();
        let x = DeviceBuffer::from_host(s, &[f32::NAN; 256])?;
        let sel = DeviceBuffer::from_host(s, &[0u32])?;
        let mut act = Q8Act::with_k(s, 1, 256)?;
        let q = QuantSel {
            x: &x,
            cols: 0..1,
            sel: &sel,
            n_card: 1,
        };
        g.q4k_sel()
            .enqueue_quantize_sel(s, &q, g.layer_sink(layer)?, &mut act)?;
        s.synchronize()?;
        m.gpu().context().bind_to_thread()?;
        Ok(Fault::at(u32::try_from(layer)?, FaultSite::QuantColumn))
    }

    /// B2: a fault on the tier card's word at `layer` before a prompt call.
    fn batch_fault_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        layer: usize,
    ) -> Result<bool, GateError> {
        let want = steps_run(m, ids)?;
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let want_fault = raise_tier_fault(m, layer)?;
        let r = body::prefill(m, ids);
        let mut ok = match &r {
            Err(GpuError::Fault { fault, .. }) if *fault == want_fault => {
                println!("ok B2: the prompt call fails with the tier's fault: {fault}");
                true
            }
            other => {
                println!(
                    "FAIL B2: the prompt call gave {other:?}, want the tier's fault {want_fault}"
                );
                false
            }
        };
        if m.poisoned() != Some(want_fault) {
            println!("FAIL B2: the model's poison is {:?}", m.poisoned());
            ok = false;
        }
        let (got, _) = batch_run(m, ids)?;
        ok &= in_band(
            "B2 (after the reset, the prompt call against the steps)",
            &want,
            &got,
        );
        Ok(ok)
    }

    /// B3 (B3f on a fresh load, clause `what`): the tier's stream held behind
    /// a host flag before a prompt call.
    fn batch_lost_case(
        m: &mut Deepseek41Model,
        ids: &[u32],
        what: &str,
    ) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        let flags = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu()?.context(), 1)?;
            flags.enqueue_wait(tier.gpu()?.stream(), 0)?;
            flags
        };
        m.gpu().context().bind_to_thread()?;
        let t0 = Instant::now();
        let r = body::prefill(m, ids);
        let secs = t0.elapsed().as_secs_f64();
        let mut ok = match &r {
            Err(e) if e.to_string().contains("the card is lost") && secs < LOST_BOUND_S => {
                println!(
                    "ok {what}: the prompt call fails in {secs:.1} s naming the lost card: {e}"
                );
                true
            }
            other => {
                println!(
                    "FAIL {what}: after {secs:.1} s the prompt call gave {other:?}, want the lost card"
                );
                false
            }
        };
        let kind = m.body(NAME)?.hybrid().last_poison().map(|p| p.mark().kind);
        if kind != Some(PoisonKind::CardLost) {
            println!("FAIL {what}: the host tier's poison is {kind:?}, want CardLost");
            ok = false;
        }
        let next = body::prefill(m, ids);
        if lost_refusal(&next, BEGIN_GROUP) {
            println!(
                "ok {what}: the next prompt call is refused at its group's start by the lost card's \
                 poison: {}",
                next.as_ref()
                    .err()
                    .map_or_else(String::new, ToString::to_string)
            );
        } else {
            println!(
                "FAIL {what}: the next prompt call gave {next:?}, want the host tier's refusal as \
                 {BEGIN_GROUP} naming the lost card"
            );
            ok = false;
        }
        match m.reset() {
            Err(e) if e.to_string().contains("lost") => {
                println!("ok {what}: a reset is refused: {e}")
            }
            other => {
                println!(
                    "FAIL {what}: a reset gave {other:?}, want a refusal naming the lost card"
                );
                ok = false;
            }
        }
        flags.raise(0)?;
        let body = m.body(NAME)?;
        body.hybrid()
            .tiers()
            .first()
            .ok_or("no tier card")?
            .gpu()?
            .stream()
            .synchronize()?;
        m.gpu().stream().synchronize()?;
        println!("{what}: flag raised, both streams drained");
        Ok(ok)
    }

    /// T3: the tier's stream held behind a host flag before a step.
    fn lost_case(m: &mut Deepseek41Model, prompt: &[u32]) -> Result<bool, GateError> {
        m.set_mode(StepMode::Graph);
        m.reset()?;
        m.step(&prompt[..4])?;
        let flags = {
            let body = m.body(NAME)?;
            let tier = body.hybrid().tiers().first().ok_or("no tier card")?;
            let flags = HostFlags::new(tier.gpu()?.context(), 1)?;
            flags.enqueue_wait(tier.gpu()?.stream(), 0)?;
            flags
        };
        let t0 = Instant::now();
        let r = m.step(&[prompt[4]]);
        let secs = t0.elapsed().as_secs_f64();
        let mut ok = match &r {
            Err(e) if e.to_string().contains("the card is lost") && secs < LOST_BOUND_S => {
                println!("ok T3: the step fails in {secs:.1} s naming the lost card: {e}");
                true
            }
            other => {
                println!("FAIL T3: after {secs:.1} s the step gave {other:?}, want the lost card");
                false
            }
        };
        let kind = m.body(NAME)?.hybrid().last_poison().map(|p| p.mark().kind);
        if kind != Some(PoisonKind::CardLost) {
            println!("FAIL T3: the host tier's poison is {kind:?}, want CardLost");
            ok = false;
        }
        let next = m.step(&[prompt[5]]);
        if lost_refusal(&next, DECODE_INPUT) {
            println!(
                "ok T3: the next step is refused by the lost card's poison: {}",
                next.as_ref()
                    .err()
                    .map_or_else(String::new, ToString::to_string)
            );
        } else {
            println!(
                "FAIL T3: the next step gave {next:?}, want the host tier's refusal as \
                 {DECODE_INPUT} naming the lost card"
            );
            ok = false;
        }
        match m.reset() {
            Err(e) if e.to_string().contains("lost") => println!("ok T3: a reset is refused: {e}"),
            other => {
                println!("FAIL T3: a reset gave {other:?}, want a refusal naming the lost card");
                ok = false;
            }
        }
        flags.raise(0)?;
        let body = m.body(NAME)?;
        body.hybrid()
            .tiers()
            .first()
            .ok_or("no tier card")?
            .gpu()?
            .stream()
            .synchronize()?;
        m.gpu().stream().synchronize()?;
        println!("T3: flag raised, both streams drained");
        Ok(ok)
    }
}
