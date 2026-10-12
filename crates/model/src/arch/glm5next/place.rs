//! A glm5next file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants, as `deepseek41::place` is for
//! V4.1. The hyperparameters, every tensor's role, the typed description and
//! each layer's KV bytes ([`KvLayout`]) are read once
//! ([`PlanInputs::read`]); a file with a feature the engine does not run is
//! refused there by the coverage check; then the placement
//! ([`PlanInputs::plan`]) runs the expert rule on the routed layers whose
//! stacks the card experts read ([`card_routed`]), the rest of every layer's
//! experts on the host, and refuses a context past the deepest one a
//! reference set checks the token-pool selector at ([`ORACLE_POSITIONS`]).

use bloomery_placement::slots::{SeqTerms, SlotsOf, Stores};
use gguf::{GgmlType, Split};
use models::ModelSpec;
use runtime::prompt::group_sets;
use runtime::stores;

use super::hparams::{Hparams, Kind};
use super::{roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::placement::accepts::{Program, routed_rule};
use crate::placement::workstation::{self, GRANULE, HostRead, TierBatchBytes};
use crate::placement::{
    self, Card, CardFormat, CardTotals, Device, ExpertList, Format, Host, KvBytes, Machine,
    ModelTensor, ModelTensors, PlacementError, Plan, PlanLevers, Role, Unimplemented, Violation,
    checked_beside, joined,
};

const F16_BYTES: u64 = 2;
const F32_BYTES: u64 = 4;

/// The widest call a later call may roll back into, in positions: the conv
/// ring keeps `conv − 1 + PASS_ROWS` inputs. The card body binds it to its
/// kernels' own constant.
pub const PASS_ROWS: usize = 8;

/// Lanes of every KDA layer's recurrent state, fixed at load, each lane
/// stamped with its position: one on a load that runs one row a pass, two on
/// one that verifies two rows (the NextN load, the two-row verify probe),
/// the state after each of a verify's rows in a lane of its own. A load pays
/// for the second lane only when it verifies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdaLanes {
    One,
    Two,
}

impl KdaLanes {
    /// The most lanes a load holds. The card body binds it to its own
    /// constant.
    pub const MAX: usize = 2;

    /// The lanes' count.
    #[must_use]
    pub const fn count(self) -> usize {
        match self {
            KdaLanes::One => 1,
            KdaLanes::Two => 2,
        }
    }
}

/// Whether a file tensor of type `ty` is one of the four K-quants a dense block's launches read.
fn is_kquant(ty: GgmlType) -> bool {
    matches!(
        ty,
        GgmlType::Q3_K | GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K
    )
}

/// The routed stacks the card experts read, each as the file stores it
/// ([`CardFormat::KQuant`]): the types the common card leg runs
/// ([`placement::kernels::card_runs`]). The rule reads a stack's type alone; a
/// layer with a stack of any other type — the Q6_K downs — keeps its experts
/// on the host.
#[must_use]
pub fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    placement::kernels::card_runs(ty).then_some(CardFormat::KQuant)
}

/// Whether a card layer of gate·up type `gate` and down type `down` runs a
/// prompt batch on the grouped tile kernels (Q4_K and Q5_K, either role); a
/// card layer of any other [`card_routed`] type runs it on the card leg in
/// chunks of [`TIER_CHUNK`] tokens ([`card_leg_bytes`]).
#[must_use]
pub const fn card_tiles_run(gate: GgmlType, down: GgmlType) -> bool {
    matches!(gate, GgmlType::Q4_K | GgmlType::Q5_K)
        && matches!(down, GgmlType::Q4_K | GgmlType::Q5_K)
}

/// Card bytes of the card leg's buffers a prompt batch's card layers run on
/// when their stacks are not [`card_tiles_run`]'s: one run of
/// [`TIER_CHUNK`] tokens of `n` values through experts of `ff` rows, `used`
/// slots a token ([`workstation::tier_chunk_scratch_bytes`], the leg's
/// buffers). The one formula of those bytes: the card body's scratch checks
/// its made bytes against it.
#[must_use]
pub fn card_leg_bytes(n: usize, ff: usize, used: usize) -> usize {
    let b = workstation::tier_chunk_scratch_bytes(n as u64, ff as u64, used as u64, TIER_CHUNK);
    usize::try_from(b).unwrap_or(usize::MAX)
}

/// Whether the file holds a routed stack the card experts read
/// ([`card_routed`]) that the tile kernels do not ([`card_tiles_run`]), so a
/// card layer of it runs a prompt batch on the card leg and the plan reserves
/// [`card_leg_bytes`] beside the tiles' scratch. Read from the headers alone;
/// a load whose card layers are all tile layers makes no leg buffers.
#[must_use]
pub fn card_leg_needed(model: &ModelTensors) -> bool {
    model.tensors.iter().any(|t| {
        t.role == Role::RoutedExperts
            && card_routed(t.ty).is_some()
            && !matches!(t.ty, GgmlType::Q4_K | GgmlType::Q5_K)
    })
}

/// Tokens a chunk of the tier's prompt-batch service runs at most: the
/// stage card's chunk of the card experts' launches, which the body binds
/// to its own constant.
pub const TIER_CHUNK: u64 = 8;

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for blocks of
/// the host union's columns: the tier card's staging, the chunked card
/// path's scratch ([`workstation::tier_chunk_scratch_bytes`]) and the host's
/// rows and places, which the plan reserves on the tier card and the host
/// and the load checks its allocations against.
#[must_use]
pub fn tier_batch(hp: &Hparams) -> TierBatchBytes {
    let (n, ff, used) = (hp.n_embd as u64, hp.expert_ff as u64, hp.n_used as u64);
    let cols = crate::moe::UNION_MAX_COLS as u64;
    TierBatchBytes {
        staging: workstation::tier_batch_staging_bytes(n, used, cols),
        scratch: workstation::tier_chunk_scratch_bytes(n, ff, used, TIER_CHUNK),
        host: workstation::tier_batch_host_bytes(n, used, cols),
    }
}

/// Token columns a tile of the GEMM front's dense table spans, and values
/// one step of its int8 activations quantizes: the card GEMM's constants
/// (`GEMM_BN`, `GEMM32_STEP`), which the card body binds to these.
pub const FRONT_TILE_COLS: usize = 64;
pub const FRONT_STEP: usize = 64;

/// Values one super-block of the K-quants' q8_1 activations spans.
pub const FRONT_SUPER: usize = 256;

/// The quantized forms a dense block's sites read, over the file's dense layers: the GEMM front quantizes the
/// normed input and the SwiGLU rows into exactly these (q8 blocks of 32 values for a Q8_0 site, q8_1 blocks of
/// 128 for a K-quant one).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseForms {
    /// A gate and up are K-quants: the normed input is also held as q8_1 blocks of 128 values.
    pub in128: bool,
    /// A down is Q8_0: the SwiGLU rows are held as q8 blocks of 32 values.
    pub rows32: bool,
    /// A down is a K-quant: the SwiGLU rows are held as q8_1 blocks of 128 values.
    pub rows128: bool,
}

impl DenseForms {
    /// No dense block.
    pub const NONE: DenseForms = DenseForms {
        in128: false,
        rows32: false,
        rows128: false,
    };

    /// Dense blocks of three Q8_0 matrices.
    pub const Q8_0: DenseForms = DenseForms {
        in128: false,
        rows32: true,
        rows128: false,
    };

    /// `self` with one more dense block, whose gate and up are K-quants when `gate_up_kquant` and whose down
    /// is one when `down_kquant`.
    #[must_use]
    pub const fn with_block(self, gate_up_kquant: bool, down_kquant: bool) -> DenseForms {
        DenseForms {
            in128: self.in128 || gate_up_kquant,
            rows32: self.rows32 || !down_kquant,
            rows128: self.rows128 || down_kquant,
        }
    }
}

/// The widths, in values a column, a prompt batch's GEMM front is sized for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrontWidths {
    /// The normed input (`embedding_length`).
    pub embd: usize,
    /// A KDA mixer's low-rank gate rows (one head, `kda.head_dim`).
    pub low: usize,
    /// A KDA mixer's gated rows (every value head).
    pub gated: usize,
    /// A latent mixer's normed query low rank (`attention.q_lora_rank`).
    pub q_low: usize,
    /// A latent mixer's query heads before absorption (every head's
    /// `attention.key_length_mla` values).
    pub q: usize,
    /// A latent mixer's heads' outputs (every head's
    /// `attention.value_length_mla` values).
    pub heads_v: usize,
    /// A dense block's width (`feed_forward_length`); 0 on a load with no
    /// dense block, which makes no feed-forward scratch.
    pub ff: usize,
    /// The forms the dense block's sites read: what is held besides the normed input's q8 blocks of 32 values.
    pub forms: DenseForms,
}

/// Card bytes of a GEMM front of `cols` token columns of widths `w`, four
/// bytes a word: the dense table — the slot list and its all-zero ids, a
/// column a word each, two words a tile (a tile every [`FRONT_TILE_COLS`]
/// columns, and one more) and the two counts — each quantized input's three
/// activation planes, twenty words a column's [`FRONT_STEP`]-value step
/// (sixteen of codes, two scales, two code sums), over the normed input, the
/// KDA gate's low-rank and gated rows, the latent query's low rank, the
/// latent heads' outputs and, for a Q8_0 down, the dense block's SwiGLU rows;
/// the dense block's K-quant forms ([`DenseForms`]) in q8_1 blocks of 128
/// values, a column's [`FRONT_SUPER`]-value super-block holding 256 words of
/// codes at the q3 permutation's pairs and 256 at the q4 permutation's quads
/// and 10 of sums and scales, over the normed input and a K-quant down's
/// SwiGLU rows; and the f32 rows the front holds whole, a word a value: the
/// latent query heads, the heads' outputs, and the dense block's gate and up
/// rows. A width of 0 makes nothing (the dense block's forms none at an `ff` of 0). The one formula of those
/// bytes: the card body's front checks its made bytes against it.
#[must_use]
pub fn front_bytes(cols: usize, w: FrontWidths) -> usize {
    let table = 2 * cols + 2 * (cols / FRONT_TILE_COLS + 1) + 2;
    let act = |k: usize| 20 * cols * k.div_ceil(FRONT_STEP);
    let act128 = |k: usize| {
        let sb = k.div_ceil(FRONT_SUPER);
        cols * (256 * sb.div_ceil(2) + 256 * sb.div_ceil(4) + 10 * sb)
    };
    let forms = if w.ff == 0 { DenseForms::NONE } else { w.forms };
    let rows32 = if forms.rows32 { w.ff } else { 0 };
    let acts: usize = [w.embd, w.low, w.gated, w.q_low, w.heads_v, rows32]
        .into_iter()
        .map(act)
        .sum();
    let in128 = if forms.in128 { w.embd } else { 0 };
    let rows128 = if forms.rows128 { w.ff } else { 0 };
    let acts128: usize = [in128, rows128].into_iter().map(act128).sum();
    let rows = cols * (w.q + w.heads_v + 2 * w.ff);
    4 * (table + acts + acts128 + rows)
}

/// The bytes of the GEMM front a prompt batch of a load `inputs` describes at
/// `ctx_max` positions makes on the stage card ([`front_bytes`]): batches of
/// up to the host union's columns or the context, whichever is fewer, at the
/// load's widths ([`PlanInputs::front_widths`]). The plan reserves them on the
/// stage card ([`prompt_reserve_bytes`], [`PlanInputs::plan_lanes`]).
#[must_use]
pub fn prompt_front_bytes(inputs: &PlanInputs, ctx_max: u64) -> u64 {
    front_bytes(prompt_cols(ctx_max), inputs.front_widths()) as u64
}

/// The columns a prompt batch of a load at `ctx_max` positions is made for:
/// the host union's, or the context when it holds fewer.
fn prompt_cols(ctx_max: u64) -> usize {
    usize::try_from(ctx_max).map_or(crate::moe::UNION_MAX_COLS, |c| {
        c.min(crate::moe::UNION_MAX_COLS)
    })
}

/// The group the plan reserves a prompt batch's units for: the group
/// lever's default (`BLOOMERY_PREFILL_GROUP` unset). A load run at a larger
/// group takes the units past these out of the card's margin.
pub const PROMPT_GROUP: usize = bloomery_levers::PREFILL_GROUP_DEFAULT as usize;

/// Card bytes of one prompt group unit for up to `cap` tokens of `n` values,
/// `streams` hyper-connection streams and `used` routed experts a token, the
/// tier places when `tiered`: per token the two stream buffers, the
/// feed-forward sub-layer's mix (`(2 + streams) · streams` values), the
/// position, the live and the two visible counts, the route's weights and
/// the tier places, four bytes a value. The one formula of those bytes: the
/// card body's units check their made bytes against it.
#[must_use]
pub const fn unit_bytes(n: usize, cap: usize, streams: usize, used: usize, tiered: bool) -> usize {
    let places = if tiered { used } else { 0 };
    4 * cap * (2 * streams * n + (2 + streams) * streams + 4 + used + places)
}

/// Card bytes of the card experts' grouped scratch for prompt batches of up
/// to `cap` tokens of `n` values through routed experts of `ff` rows, `used`
/// slots a token ([`workstation::walk_a_tile_scratch_bytes`]). The one
/// formula of those bytes: the card body's scratch checks its made bytes
/// against it.
#[must_use]
pub fn card_tile_bytes(n: usize, ff: usize, used: usize, cap: usize) -> usize {
    let b = workstation::walk_a_tile_scratch_bytes(n as u64, ff as u64, used as u64, cap as u64);
    usize::try_from(b).unwrap_or(usize::MAX)
}

/// The card bytes the plan reserves on the stage card for a load of `inputs`'s
/// prompt batch at `ctx_max` positions: its GEMM front
/// ([`prompt_front_bytes`]), the card experts' grouped scratch
/// ([`card_tile_bytes`] at the front's columns, plus [`card_leg_bytes`] for a
/// file [`card_leg_needed`] names) and the units a group of
/// [`PROMPT_GROUP`] holds past the first ([`runtime::prompt::group_sets`], [`unit_bytes`] at
/// the front's columns, the tier places when `tiered`). The first unit and
/// the other buffers the units share come out of the card's margin.
#[must_use]
pub fn prompt_reserve_bytes(inputs: &PlanInputs, ctx_max: u64, tiered: bool) -> u64 {
    let hp = &inputs.hp;
    let cols = prompt_cols(ctx_max);
    let unit = unit_bytes(hp.n_embd, cols, hp.hc.streams, hp.n_used, tiered);
    let mut tiles = card_tile_bytes(hp.n_embd, hp.expert_ff, hp.n_used, cols);
    if card_leg_needed(&inputs.model) {
        tiles += card_leg_bytes(hp.n_embd, hp.expert_ff, hp.n_used);
    }
    prompt_front_bytes(inputs, ctx_max) + (tiles + (group_sets(PROMPT_GROUP) - 1) * unit) as u64
}

/// What a plan of a glm5next file is made from, read from its headers.
#[derive(Debug)]
pub struct PlanInputs {
    /// The file's hyperparameters.
    pub hp: Hparams,
    /// Every tensor with its role, and the model's layer and expert counts.
    pub model: ModelTensors,
    /// The typed model description the coverage check reads.
    pub spec: ModelSpec,
    /// Each trunk layer's recurrent and cache bytes, at one KDA lane
    /// ([`KvLayout::with_lanes`] for more).
    pub kv: KvLayout,
    /// The host's available bytes when the inputs were read, with the
    /// reading that decided ([`workstation::host_available_read`]): the
    /// routed experts the room cannot hold follow it to the NVMe tier
    /// ([`placement::expert_nvme_tier`]); a NextN plan splits the trunk at
    /// the room less the draft layer's routed experts. A gate that plans an
    /// arm by a room of its own sets it, read [`HostRead::Given`].
    ///
    /// The plan is made at [`workstation::plan_room`]: a read room less
    /// [`workstation::HOST_DRIFT`], a given room as given.
    pub room: (u64, HostRead),
    /// The arena's bytes `BLOOMERY_NVTIER_BYTES` gave the split dial
    /// ([`workstation::nvtier_arena_given`]), `None` unset; the split takes it
    /// as its `given`. A gate that plans an arm by an arena of its own sets
    /// it.
    pub arena: Option<u64>,
}

/// Why a plan was refused.
#[derive(Debug, thiserror::Error)]
pub enum PlaceError {
    /// The placement could not be built.
    #[error(transparent)]
    Placement(#[from] PlacementError),
    /// The plan was built and breaks these invariants, every one of them.
    #[error("the plan breaks its invariants: {}", joined(.0))]
    Broken(Vec<Violation>),
    /// The context asks for more positions than a reference set checks the
    /// token-pool selector at: past them no oracle defines what it keeps.
    #[error(
        "ctx_max {ctx_max}: the deepest context a reference set checks the token-pool selector \
         at is {served} positions; past it nothing defines what the latent layers keep"
    )]
    PastOracle { ctx_max: u64, served: u64 },
    /// A NextN load on a machine of other than one card: the draft layer
    /// runs beside the target's chain on its card.
    #[error(
        "the NextN load runs the next-token layer on the target's one card; the machine has \
         {cards} cards"
    )]
    NextnCards { cards: usize },
    /// A stage card whose plan leaves less beside it than the prompt
    /// batch's GEMM front and group units ([`prompt_reserve_bytes`]) take:
    /// the dense granules, the cache, context, scratch, reserves and margin
    /// already pass what the card holds less them, so no expert placement
    /// makes room.
    #[error(
        "card {card}: the prompt batch's GEMM front and group units take {front} B, and the plan \
         leaves {left} B beside its dense granules, cache, context, scratch, reserves and margin \
         ({short} B short)"
    )]
    FrontOver {
        card: String,
        front: u64,
        left: u64,
        short: u64,
    },
    /// A plan of no resident sequence: a load serves at least one.
    #[error("a plan of {slots} resident sequences: a load serves at least one")]
    Slots { slots: usize },
}

/// The prompt batch's reserve `front` ([`prompt_reserve_bytes`]) beside
/// `total` bytes of a plan's terms on `card`, `limit` its usable bytes less
/// its margin: refused by name when the terms fit alone and the reserve does
/// not fit beside them (terms past the limit are the plan's own violation).
fn refuse_front(card: &Card, total: u64, limit: u64, front: u64) -> Result<(), PlaceError> {
    match limit.checked_sub(total) {
        Some(left) if front > left => Err(PlaceError::FrontOver {
            card: card.name.clone(),
            front,
            left,
            short: front - left,
        }),
        _ => Ok(()),
    }
}

/// A plan's refusal `e` of a context set for a load with the NextN layer, as
/// text: the plan's own words and, for a refusal of bytes
/// ([`PlaceError::Broken`], [`PlaceError::FrontOver`]), the `nextn_bytes` the
/// plan counted for the layer beside the trunk ([`NextnInputs::card_bytes`]),
/// in the same line, so a context that fits without the draft is not refused
/// with it unexplained. Any other refusal, and a plan without the draft
/// (`nextn_bytes` is `None`), is `e`'s text as it is.
#[must_use]
pub fn refusal_text(e: &PlaceError, nextn_bytes: Option<u64>, ctx: usize, slots: usize) -> String {
    match (e, nextn_bytes) {
        (PlaceError::Broken(_) | PlaceError::FrontOver { .. }, Some(n)) => format!(
            "{e}; the plan counts {n} B for the NextN layer the draft holds on the card beside \
             the trunk (its tensors, its store of {slots} slots at {ctx} positions a slot, and \
             its {NEXTN_ARENA_BYTES} B arena)"
        ),
        _ => e.to_string(),
    }
}

/// A plan's terms on a card: its granules, cache, scratch, context and
/// reserves.
fn card_terms(t: &CardTotals) -> u64 {
    t.dense_bytes
        + t.expert_bytes
        + t.rounding_bytes
        + t.kv_bytes
        + t.scratch_bytes
        + t.context_bytes
        + t.reserve_bytes
}

/// The deepest context the plan serves: ik's `--dsa` step set at position
/// 16,382 of a 16,384-position context (`tools/ref/models/glm5next.sh`,
/// variant `d16kdsa`), the deepest reference the selector is checked at.
pub const ORACLE_POSITIONS: u64 = 16_384;

/// The most visible positions at which the token-pool indexer keeps every
/// one: every whole pool while there are at most `top_k / kpool` of them,
/// plus the tail of fewer than `kpool` positions no pool holds yet. Up to
/// this count a row's list is every position.
#[must_use]
pub fn dense_positions(hp: &Hparams) -> u64 {
    let (top_k, kpool) = (hp.indexer.top_k as u64, hp.indexer.kpool as u64);
    (top_k / kpool) * kpool + kpool - 1
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then the file is
    /// refused if it has a feature the engine does not run
    /// ([`PlacementError::Unimplemented`], every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        let inputs = PlanInputs::describe(split)?;
        let missing = inputs.unimplemented();
        if missing.is_empty() {
            Ok(inputs)
        } else {
            Err(PlacementError::Unimplemented(missing))
        }
    }

    /// [`PlanInputs::read`] without the refusal of unimplemented features,
    /// and with the host's room ([`workstation::host_available_read`]) and the
    /// arena's given bytes, a room that cannot be read refused by name.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::ThinkSpan))?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let kv = KvLayout::of(&hp);
        let room = workstation::host_available_read().map_err(PlacementError::HostRoom)?;
        let arena = workstation::nvtier_arena_given().map_err(PlacementError::HostRoom)?;
        Ok(PlanInputs {
            hp,
            model,
            spec,
            kv,
            room,
            arena,
        })
    }

    /// The forms the file's dense blocks read ([`DenseForms`]), from the types of their gate and down in the
    /// header: [`DenseForms::NONE`] for a file with no dense block.
    #[must_use]
    pub fn dense_forms(&self) -> DenseForms {
        self.model
            .tensors
            .iter()
            .filter(|t| t.role == Role::DenseFfn)
            .fold(DenseForms::NONE, |forms, t| {
                let kquant = is_kquant(t.ty);
                if t.name.ends_with(roles::DENSE_GATE) {
                    DenseForms {
                        in128: forms.in128 || kquant,
                        ..forms
                    }
                } else if t.name.ends_with(roles::DENSE_DOWN) {
                    DenseForms {
                        rows32: forms.rows32 || !kquant,
                        rows128: forms.rows128 || kquant,
                        ..forms
                    }
                } else {
                    forms
                }
            })
    }

    /// The widths of a load's prompt batch's GEMM front ([`FrontWidths`]): one
    /// head count for the KDA and latent heads (`attention.head_count`), the
    /// dense block's forms from the header's types ([`PlanInputs::dense_forms`]).
    #[must_use]
    pub fn front_widths(&self) -> FrontWidths {
        let hp = &self.hp;
        FrontWidths {
            embd: hp.n_embd,
            low: hp.kda_head_dim,
            gated: hp.n_head * hp.kda_head_dim,
            q_low: hp.q_lora,
            q: hp.n_head * hp.head_k,
            heads_v: hp.n_head * hp.head_v,
            ff: hp.dense_ff,
            forms: self.dense_forms(),
        }
    }

    /// Every feature of the file the engine does not run, as
    /// [`coverage::check`] lists them. Empty for a file the program runs.
    pub fn unimplemented(&self) -> Vec<Unimplemented> {
        coverage::check(&self.spec, &self.model)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, each KDA layer's state one lane
    /// ([`PlanInputs::plan_lanes`] at [`KdaLanes::One`]): the plan of a load
    /// that runs one row a pass.
    pub fn plan<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_lanes(machine, ctx_max, levers, KdaLanes::One)
    }

    /// The placement of the file on `machine` at `ctx_max` positions under
    /// the placement's `levers`, each KDA layer's state `lanes` lanes: the
    /// expert rule on the layers whose routed stacks [`card_routed`] runs,
    /// each layer's id prefix, within the stage card's budget less the
    /// prompt batch's GEMM front and group units ([`prompt_reserve_bytes`]);
    /// refused past [`ORACLE_POSITIONS`], when it cannot be built, when it
    /// breaks an invariant, and by name when they do not fit beside the rest
    /// ([`PlaceError::FrontOver`]).
    pub fn plan_lanes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        lanes: KdaLanes,
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_slots(machine, ctx_max, levers, lanes, 1)
    }

    /// [`PlanInputs::plan_lanes`] for a load that serves `slots` resident
    /// sequences (`bloomery_gpu::model::Slots`): every term of a sequence
    /// ([`PlanInputs::seq_terms`]) counts them all — each layer's stores
    /// `slots` times ([`SeqTerms::slots_of`]) and the bytes beside the
    /// stores of every sequence past the live one ([`SeqTerms::plan_beside`]),
    /// those reserved out of the expert budget before it fills, so a card
    /// that cannot hold them is refused as [`PlanInputs::plan_lanes`] refuses
    /// it. The stage card's KV term is [`PlanInputs::kv_term`]. One slot is
    /// [`PlanInputs::plan_lanes`] itself; zero is refused by name. The
    /// routed experts the host's room ([`PlanInputs::room`]) cannot hold are
    /// paged through the NVMe tier ([`placement::expert_nvme_tier`]).
    pub fn plan_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        lanes: KdaLanes,
        slots: usize,
    ) -> Result<Plan<'a>, PlaceError> {
        if ctx_max > ORACLE_POSITIONS {
            return Err(PlaceError::PastOracle {
                ctx_max,
                served: ORACLE_POSITIONS,
            });
        }
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        let kv = self.kv.with_lanes(lanes);
        let terms = self.seq_terms(&kv, None);
        let n = slots as u64;
        let beside = terms.plan_beside(n);
        let front = prompt_reserve_bytes(self, ctx_max, !machine.tiers.is_empty());
        let mut plan = placement::plan_routed_reserving(
            &self.model,
            machine,
            ctx_max,
            &terms.slots_of(n),
            levers,
            routed_rule(Program::Glm5nextBody),
            front.saturating_add(beside),
        )?;
        // The bytes beside the stores ride the stage card's KV term, out of
        // the budget reserved above, so the card's bound holds them.
        if let Some(t) = plan.cards.first_mut() {
            t.grow_kv(beside);
        }
        let room = plan.plan_at(&self.room);
        placement::expert_nvme_tier(&mut plan, room, self.arena)?;
        // The front stays beside the plan's terms (`refuse_front` holds it to
        // the card), so a unified pool counts it beside them.
        let plan = checked_beside(plan, front).map_err(PlaceError::Broken)?;
        if let (Some(card), Some(t)) = (machine.cards.first(), plan.cards.first()) {
            let limit = plan.usable_bytes(card).saturating_sub(card.margin_bytes);
            refuse_front(card, card_terms(t), limit, front)?;
        }
        Ok(plan)
    }
}

/// Card bytes the NextN walk's arena may hold beside the plan's tensors
/// [derived, an upper bound: the one-row step buffers the full row runs on
/// (`RowScratch` at the shared expert's width: the streams 32,768 f32, the
/// latent mixer's rows 101,376 f32 and its attention's partials over a
/// list's 2,051 positions, 33 segments of 64 rows of 512 f32 and their
/// maxima, about 1.09 M f32, the KDA rows 90,112 f32 the full row never
/// writes, the selector's scores 4,096 f32 at 16,384 positions); the walk's
/// rows for eight positions (the embedding, hidden and packed rows, the
/// projection's output, its norm and the joined projection's rows, 77,824
/// f32); the head's input, logits (154,880 f32) and argmax words; the host
/// sums' row and the verify's row-0 streams (4,096 and 16,384 f32) — about
/// 1.45 M values, 5.8 MB, rounded up to 16 MiB]. A load whose arena passes it
/// is refused by name.
pub const NEXTN_ARENA_BYTES: u64 = 16 << 20;

/// What a NextN load is planned from ([`PlanInputs::plan_nextn`]): the
/// file's next-token layer as a model of one layer — its tensors in their
/// [`roles::nextn_role`], layer 0 of one — on a card of its own and a host of
/// its own, both of no bound (the bounds are the target machine's, checked on
/// the sums), and its store: a latent layer's, one latent row and one index
/// row a position and a pool key a pool.
#[derive(Debug)]
pub struct NextnInputs {
    /// The layer's index in the file, the first past the trunk: its tensors
    /// are `blk.{index}.*`.
    pub index: usize,
    /// The layer's tensors as layer 0 of one.
    pub model: ModelTensors,
    /// The layer's own machine: one card and a host, no bound on either.
    pub machine: Machine,
    kv: NextnKv,
}

/// The stems a next-token layer's load needs: its own (the head's norm
/// among them — the load runs the shared head through it), every latent
/// mixer's, a routed block's and a shared expert's the file must carry.
fn nextn_required() -> impl Iterator<Item = &'static str> {
    roles::NEXTN
        .iter()
        .map(|s| s.name)
        .chain(roles::required(roles::LATENT))
        .chain(roles::required(roles::MOE))
        .chain(roles::required(roles::SHARED))
}

impl NextnInputs {
    /// The next-token layer of the file `inputs` describes. Refused by name:
    /// a file of other than one next-token layer, a layer that is not a
    /// latent one, a stem of it no next-token layer carries, and a stem the
    /// load needs absent.
    pub fn read(inputs: &PlanInputs) -> Result<NextnInputs, PlacementError> {
        let hp = &inputs.hp;
        let refuse = |detail: String| PlacementError::Metadata {
            key: "nextn_predict_layers".to_string(),
            detail,
        };
        if hp.n_layer != hp.n_trunk + 1 {
            return Err(refuse(format!(
                "{} next-token layers; the NextN load runs one",
                hp.n_layer - hp.n_trunk
            )));
        }
        let index = hp.n_trunk;
        if hp.kinds.get(index) != Some(&Kind::Latent) {
            return Err(refuse(format!(
                "layer {index} is {:?}; the NextN load runs a latent layer",
                hp.kinds.get(index)
            )));
        }
        let prefix = format!("blk.{index}.");
        let mut tensors = Vec::new();
        for t in &inputs.model.tensors {
            let Some(stem) = t.name.strip_prefix(&prefix) else {
                continue;
            };
            let role = roles::nextn_role(stem).ok_or_else(|| PlacementError::Tensor {
                name: t.name.clone(),
                detail: "a stem no next-token layer carries".to_string(),
            })?;
            tensors.push(ModelTensor {
                layer: Some(0),
                role,
                ..t.clone()
            });
        }
        if let Some(missing) =
            nextn_required().find(|stem| !tensors.iter().any(|t| t.name == prefix.clone() + stem))
        {
            return Err(PlacementError::Tensor {
                name: format!("{prefix}{missing}"),
                detail: "is not in the file; the NextN load needs it".to_string(),
            });
        }
        Ok(NextnInputs {
            index,
            model: ModelTensors {
                tensors,
                layers: 1,
                experts: inputs.model.experts,
                experts_used: inputs.model.experts_used,
            },
            machine: nextn_machine(),
            kv: NextnKv {
                row: row_bytes(hp.kv_lora, hp.indexer.head_dim),
                pool_row: hp.indexer.head_dim as u64 * F16_BYTES,
                kpool: hp.indexer.kpool as u64,
            },
        })
    }
}

/// The NextN layer's own machine: one card of no bound with the
/// workstation's granule, running its one layer, and a host of no bound.
fn nextn_machine() -> Machine {
    Machine {
        cards: vec![Card {
            name: "nextn".to_string(),
            device: None,
            usable_bytes: u64::MAX,
            context_bytes: 0,
            scratch_bytes: 0,
            margin_bytes: 0,
            granule_bytes: GRANULE,
            free_bytes: None,
            held_by: None,
            layers: 0..1,
            head: true,
            token_embedding: false,
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        host: Host {
            usable_bytes: u64::MAX,
            reserves: Vec::new(),
        },
        unified: None,
    }
}

/// The NextN layer's store: a latent layer's ([`KvLayout`]'s rule).
#[derive(Clone, Copy, Debug)]
struct NextnKv {
    row: u64,
    pool_row: u64,
    kpool: u64,
}

impl KvBytes for NextnKv {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        if layer != 0 {
            return 0;
        }
        ctx_max * self.row + ctx_max.div_ceil(self.kpool) * self.pool_row
    }
}

/// A glm5next plan with its NextN layer ([`PlanInputs::plan_nextn`]).
#[derive(Debug)]
pub struct NextnPlan<'a> {
    /// The target's plan: [`PlanInputs::plan`]'s expert rule within the
    /// card's budget less the NextN layer's card bytes and arena, so the sum
    /// fits; its totals are the target's alone.
    pub plan: Plan<'a>,
    /// The NextN layer's plan on [`NextnInputs::machine`]: its tensors but
    /// the routed stacks on its card, the routed stacks every expert on the
    /// host, and its store for every resident sequence (`cards[0].kv_bytes`).
    pub nextn: Plan<'a>,
    /// [`NEXTN_ARENA_BYTES`], counted beside the NextN layer's card bytes.
    pub arena_bytes: u64,
    /// The card's usable bytes (capped by the card budget) less the target's
    /// and the NextN layer's card terms and the prompt batch's GEMM front and
    /// group units; the margin is inside it.
    pub headroom_bytes: i128,
    /// The host's headroom less the NextN layer's routed experts.
    pub host_headroom_bytes: i128,
}

impl NextnPlan<'_> {
    /// Bytes the NextN layer adds to the card: its granules and its store.
    #[must_use]
    pub fn nextn_card_bytes(&self) -> u64 {
        nextn_card_bytes(&self.nextn.cards[0])
    }

    /// Device bytes the NextN layer's tensors hold, the allocator's
    /// rounding excluded: what its load's uploads sum to.
    #[must_use]
    pub fn nextn_resident_bytes(&self) -> u64 {
        self.nextn.cards[0].dense_bytes + self.nextn.cards[0].expert_bytes
    }

    /// The NextN layer's host segments — its routed stacks, every expert —
    /// as runs of the target plan's model: each stack's index among the
    /// target's tensors (where it is [`placement::Role::Unused`]) and its
    /// experts, the host set the load reads in beside the target's own.
    /// With them, their file bytes (the segments' resident bytes).
    /// Refused by name: a host segment of a tensor the target does not
    /// carry, or of a whole tensor rather than an expert stack.
    pub fn host_runs(&self) -> Result<(Vec<(usize, ExpertList)>, u64), PlacementError> {
        let mut out = Vec::new();
        let mut bytes = 0u64;
        for row in &self.nextn.rows {
            let t = self.nextn.model.tensors.get(row.tensor).ok_or_else(|| {
                PlacementError::Host(format!(
                    "a NextN plan row names tensor {}, past its model",
                    row.tensor
                ))
            })?;
            for seg in row
                .segments
                .iter()
                .filter(|s| s.device == Device::Host && s.format == Format::HostFile)
            {
                let refuse = |detail: &str| PlacementError::Tensor {
                    name: t.name.clone(),
                    detail: detail.to_string(),
                };
                let experts = seg
                    .experts
                    .clone()
                    .ok_or_else(|| refuse("a NextN host segment of a whole tensor"))?;
                let at = self
                    .plan
                    .model
                    .tensors
                    .iter()
                    .position(|u| u.name == t.name)
                    .ok_or_else(|| refuse("is not among the target plan's tensors"))?;
                out.push((at, experts));
                bytes += seg.resident_bytes;
            }
        }
        Ok((out, bytes))
    }
}

/// A NextN plan's card terms: its granules and its store.
fn nextn_card_bytes(d: &CardTotals) -> u64 {
    d.dense_bytes + d.expert_bytes + d.rounding_bytes + d.kv_bytes
}

impl PlanInputs {
    /// [`PlanInputs::plan`] with the next-token layer `nextn` loaded on the
    /// machine's one card: the NextN layer's own plan (its routed stacks on
    /// the host), the target's expert rule within the card's budget less the
    /// NextN layer's card bytes and arena and the prompt batch's GEMM front
    /// and group units ([`prompt_reserve_bytes`]), each KDA layer's state
    /// two lanes ([`KdaLanes::Two`]: the load verifies the draft's rows), the
    /// card's bound checked on the sum (the reserve named,
    /// [`PlaceError::FrontOver`]) and the host's on the target's host
    /// set and the NextN layer's routed experts. Refused as
    /// [`PlanInputs::plan`] refuses and on a
    /// machine of other than one card; every violation of either plan is
    /// listed, the card's and the host's bound once.
    pub fn plan_nextn<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        nextn: &'a NextnInputs,
    ) -> Result<NextnPlan<'a>, PlaceError> {
        self.plan_nextn_slots(machine, ctx_max, levers, nextn, 1)
    }

    /// [`PlanInputs::plan_nextn`] for a load that serves `slots` resident
    /// sequences, every term of a sequence ([`PlanInputs::seq_terms`])
    /// counting them all: the NextN layer's store `slots` times in its own
    /// plan ([`NextnInputs::slots_of`]), and on the stage card each trunk
    /// layer's stores `slots` times and the bytes beside the stores of every
    /// sequence past the live one, those reserved out of the expert budget
    /// before it fills ([`PlanInputs::kv_term`]). The two plans' kv classes
    /// sum to [`SeqTerms::plan_kv`]. One slot is [`PlanInputs::plan_nextn`]
    /// itself; zero is refused by name. The trunk's NVMe tier split runs at the
    /// room less the NextN layer's routed experts, which are the host's beside
    /// it; a room under them is refused by name.
    pub fn plan_nextn_slots<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        nextn: &'a NextnInputs,
        slots: usize,
    ) -> Result<NextnPlan<'a>, PlaceError> {
        if ctx_max > ORACLE_POSITIONS {
            return Err(PlaceError::PastOracle {
                ctx_max,
                served: ORACLE_POSITIONS,
            });
        }
        let [card] = machine.cards.as_slice() else {
            return Err(PlaceError::NextnCards {
                cards: machine.cards.len(),
            });
        };
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        let kv = self.kv.with_lanes(KdaLanes::Two);
        let terms = self.seq_terms(&kv, Some(nextn));
        let n = slots as u64;
        let beside = terms.plan_beside(n);
        let draft = placement::plan_host_routed(
            &nextn.model,
            &nextn.machine,
            ctx_max,
            &nextn.slots_of(n),
            &PlanLevers::default(),
        )?;
        let arena = NEXTN_ARENA_BYTES;
        let front = prompt_reserve_bytes(self, ctx_max, !machine.tiers.is_empty());
        let nextn_card = nextn_card_bytes(&draft.cards[0]);
        let reserve = (nextn_card + arena + front).saturating_add(beside);
        let mut plan = placement::plan_routed_reserving(
            &self.model,
            machine,
            ctx_max,
            &terms.slots_of(n),
            levers,
            routed_rule(Program::Glm5nextBody),
            reserve,
        )?;
        // The bytes beside the stores ride the stage card's KV term, as
        // [`PlanInputs::plan_slots`] counts them.
        plan.cards[0].grow_kv(beside);
        // The draft layer's routed experts are the host's beside the trunk's,
        // a plan of their own: the trunk splits at the room they leave.
        let draft_host = draft.host.expert_bytes;
        let planned_at = plan.plan_at(&self.room);
        let room = planned_at.checked_sub(draft_host).ok_or_else(|| {
            PlacementError::HostRoom(format!(
                "the NextN layer's routed experts, {draft_host} B, pass the host room {planned_at} B"
            ))
        })?;
        placement::expert_nvme_tier(&mut plan, room, self.arena)?;
        let total = card_terms(&plan.cards[0]) + nextn_card + arena;
        let usable = plan.usable_bytes(card);
        let limit = usable.saturating_sub(card.margin_bytes);
        // A split plan's headroom is what the room it ran at left, and that
        // room lost the draft's experts already; an unsplit plan's is the
        // machine's figure or a stated room's, which have not.
        let host_headroom = if plan.host.nvme_expert_bytes > 0 {
            plan.host.headroom_bytes
        } else {
            plan.host.headroom_bytes - i128::from(draft_host)
        };
        // On a unified machine the pool holds the draft and the front beside
        // the target's own terms.
        let mut broken: Vec<Violation> = plan
            .violations_beside(nextn_card + arena + front + draft.host.expert_bytes)
            .into_iter()
            .filter(|v| !matches!(v, Violation::CardOver { .. } | Violation::HostOver { .. }))
            .chain(
                draft
                    .violations()
                    .into_iter()
                    .filter(|v| !matches!(v, Violation::HostOver { .. })),
            )
            .collect();
        if total > limit {
            broken.push(Violation::CardOver {
                card: card.name.clone(),
                total,
                limit,
            });
        }
        // The host's own bound, as [`Plan::violations`] holds it: a unified
        // pool's share of the cards is the pool's bound, above.
        let host_own = host_headroom + i128::from(plan.host.pool_card_bytes);
        if host_own < 0 {
            let usable = machine.host.usable_bytes;
            broken.push(Violation::HostOver {
                total: u64::try_from(i128::from(usable) - host_own).unwrap_or(u64::MAX),
                usable,
            });
        }
        if !broken.is_empty() {
            return Err(PlaceError::Broken(broken));
        }
        refuse_front(card, total, limit, front)?;
        Ok(NextnPlan {
            headroom_bytes: i128::from(usable) - i128::from(total) - i128::from(front),
            host_headroom_bytes: host_headroom,
            plan,
            nextn: draft,
            arena_bytes: arena,
        })
    }
}

/// One file's per-layer bytes beside the weights: a KDA layer's recurrent
/// state (`lanes` lanes of `n_head` heads of `d × d` f32, a u32 stamp a
/// lane) and conv ring (`conv − 1 + PASS_ROWS` rows of the q, k and v
/// channels in f32), fixed by the file and the load's lanes; a latent layer's
/// cache, a latent row and an index row (`[key; gate]`, twice the indexer's
/// key width) in f16 a position, and a pool key (the indexer's key width in
/// f16) every `kpool` positions, the last pool whole.
#[derive(Clone, Debug)]
pub struct KvLayout {
    /// Per trunk layer: its kind.
    kinds: Vec<Kind>,
    /// A KDA layer's heads, their width and the conv's taps.
    kda: (usize, usize, usize),
    /// The lanes of a KDA layer's state.
    lanes: KdaLanes,
    /// A latent layer's cache bytes a position.
    row: u64,
    /// A latent layer's pool key bytes, and the positions a pool holds.
    pool_row: u64,
    kpool: u64,
}

impl KvLayout {
    /// The layout `hp` describes, each KDA layer's state one lane.
    #[must_use]
    pub fn of(hp: &Hparams) -> KvLayout {
        KvLayout {
            kinds: hp.kinds[..hp.n_trunk].to_vec(),
            kda: (hp.n_head, hp.kda_head_dim, hp.conv),
            lanes: KdaLanes::One,
            row: row_bytes(hp.kv_lora, hp.indexer.head_dim),
            pool_row: hp.indexer.head_dim as u64 * F16_BYTES,
            kpool: hp.indexer.kpool as u64,
        }
    }

    /// The same layout with each KDA layer's state `lanes` lanes.
    #[must_use]
    pub fn with_lanes(&self, lanes: KdaLanes) -> KvLayout {
        KvLayout {
            lanes,
            ..self.clone()
        }
    }

    /// The lanes of a KDA layer's state.
    #[must_use]
    pub fn lanes(&self) -> KdaLanes {
        self.lanes
    }

    /// The bytes of layers `layers` at `ctx_max` positions: what a card that
    /// runs them holds beside its weights ([`KvBytes::layer_bytes`] summed).
    #[must_use]
    pub fn bytes(&self, layers: std::ops::Range<usize>, ctx_max: u64) -> u64 {
        layers.map(|l| self.layer_bytes(l, ctx_max)).sum()
    }
}

/// A KDA layer's state of `lanes` lanes with their stamps and its conv ring
/// over `heads` heads `d` wide with a `conv`-tap conv, in bytes: the lanes
/// past the first and the stamps counted by `runtime::stores`, the rule the
/// delta stores are allocated by.
fn recurrent_bytes(heads: usize, d: usize, conv: usize, lanes: KdaLanes) -> u64 {
    let more = stores::delta_lane_bytes(heads, d, lanes.count());
    let (heads, d) = (heads as u64, d as u64);
    let ring_rows = (conv as u64).saturating_sub(1) + PASS_ROWS as u64;
    heads * d * d * F32_BYTES + more + ring_rows * 3 * heads * d * F32_BYTES
}

/// A latent layer's cache bytes a position: the `latent`-wide row and the
/// index row of an `index_d`-wide key and its gate.
fn row_bytes(latent: usize, index_d: usize) -> u64 {
    (latent as u64 + 2 * index_d as u64) * F16_BYTES
}

impl KvBytes for KvLayout {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        match self.kinds.get(layer) {
            Some(Kind::Kda) => {
                let (heads, d, conv) = self.kda;
                recurrent_bytes(heads, d, conv, self.lanes)
            }
            Some(Kind::Latent) => ctx_max * self.row + ctx_max.div_ceil(self.kpool) * self.pool_row,
            None => 0,
        }
    }
}

impl PlanInputs {
    /// What one resident sequence of a load of the file holds on its card
    /// ([`SeqTerms`]): its stores over the trunk's layers in `kv` (the
    /// file's layout at the load's KDA lanes, [`KvLayout::with_lanes`]), the
    /// next-token layer's store on a load that carries `nextn`, and the bytes
    /// it holds beside them ([`beside_bytes`]). The one fill the plans of
    /// resident sequences, a load's check of its plan and a gate's byte
    /// clause read.
    #[must_use]
    pub fn seq_terms<'a>(&self, kv: &'a KvLayout, nextn: Option<&'a NextnInputs>) -> SeqTerms<'a> {
        seq_terms(kv, nextn.map(|n| &n.kv), self.hp.hc.streams, self.hp.n_embd)
    }

    /// The KV term a plan of `slots` resident sequences of `ctx_max`
    /// positions at `lanes` KDA lanes counts on a card that runs `layers`
    /// (`nextn` when the load carries the next-token layer): the view the
    /// plan counts the stores by over those layers ([`SeqTerms::slots_of`])
    /// and the bytes beside the stores it adds ([`SeqTerms::plan_beside`]) —
    /// what [`PlanInputs::plan_slots`] and [`PlanInputs::plan_nextn_slots`]
    /// put in the stage card's `kv_bytes`, the next-token layer's store apart
    /// in its own plan, and what a load of that many sequences holds its plan
    /// to. Zero slots is refused by name.
    pub fn kv_term(
        &self,
        layers: std::ops::Range<usize>,
        ctx_max: u64,
        lanes: KdaLanes,
        nextn: Option<&NextnInputs>,
        slots: usize,
    ) -> Result<u64, PlaceError> {
        if slots == 0 {
            return Err(PlaceError::Slots { slots });
        }
        let kv = self.kv.with_lanes(lanes);
        let terms = self.seq_terms(&kv, nextn);
        Ok(stage_term(&terms, layers, ctx_max, slots as u64))
    }
}

impl NextnInputs {
    /// The view the NextN layer's own plan counts `slots` sequences' stores
    /// by: its store ([`NextnKv`], layer 0 of its one-layer model) `slots`
    /// times. The bytes a sequence holds beside its stores are the target's
    /// ([`PlanInputs::seq_terms`]).
    fn slots_of(&self, slots: u64) -> SlotsOf<'_> {
        SeqTerms {
            layers: Stores {
                kv: &self.kv,
                count: 1,
            },
            draft: None,
            beside: 0,
        }
        .slots_of(slots)
    }

    /// The card bytes the NextN layer holds for `slots` sequences at
    /// `ctx_max` positions, its arena counted: the term
    /// [`PlanInputs::plan_nextn_slots`] reserves out of the stage card's
    /// budget ([`NextnPlan::nextn_card_bytes`] plus
    /// [`NextnPlan::arena_bytes`]), read from the layer's own plan, which the
    /// stage card does not bound — so a plan the card refuses still names it.
    pub fn card_bytes(&self, ctx_max: u64, slots: usize) -> Result<u64, PlaceError> {
        let n = slots as u64;
        let draft = placement::plan_host_routed(
            &self.model,
            &self.machine,
            ctx_max,
            &self.slots_of(n),
            &PlanLevers::default(),
        )?;
        Ok(nextn_card_bytes(&draft.cards[0]) + NEXTN_ARENA_BYTES)
    }
}

/// [`PlanInputs::seq_terms`] from its parts: the trunk's layers in `kv`
/// (every layer its kinds name), the next-token layer's store `draft` as one
/// layer, and [`beside_bytes`] of `streams` streams `n_embd` wide at `kv`'s
/// lanes, with the verify's row-0 copy on a load that drafts.
fn seq_terms<'a>(
    kv: &'a KvLayout,
    draft: Option<&'a NextnKv>,
    streams: usize,
    n_embd: usize,
) -> SeqTerms<'a> {
    SeqTerms {
        layers: Stores {
            kv,
            count: kv.kinds.len(),
        },
        draft: draft.map(|d| Stores { kv: d, count: 1 }),
        beside: beside_bytes(streams, n_embd, kv.lanes, draft.is_some()),
    }
}

/// [`PlanInputs::kv_term`] from its fill: `terms`' view of `slots`
/// sequences over `layers` and the bytes beside the stores the plan adds;
/// `u64::MAX` past u64 bytes.
fn stage_term(
    terms: &SeqTerms<'_>,
    layers: std::ops::Range<usize>,
    ctx_max: u64,
    slots: u64,
) -> u64 {
    let view = terms.slots_of(slots);
    layers
        .map(|l| view.layer_bytes(l, ctx_max))
        .fold(terms.plan_beside(slots), u64::saturating_add)
}

/// Card bytes one resident sequence holds beside its per-layer stores: the
/// lane word every KDA launch reads, and the final streams of each row a
/// pass runs (a row a KDA lane) and, on a NextN load, the verify's row-0
/// copy — the rows the sequence's own draft walk reads, held a sequence
/// while the arenas address whichever is live [derived: one u32 word, and
/// `streams · n_embd` f32 a row].
fn beside_bytes(streams: usize, n_embd: usize, lanes: KdaLanes, nextn: bool) -> u64 {
    let row = (streams * n_embd) as u64 * F32_BYTES;
    4 + (lanes.count() as u64 + u64::from(nextn)) * row
}

#[cfg(test)]
mod tests {
    use super::{
        DenseForms, FRONT_STEP, FrontWidths, KdaLanes, Kind, KvBytes, KvLayout, NEXTN_ARENA_BYTES,
        NextnKv, PROMPT_GROUP, PlaceError, card_tile_bytes, front_bytes, group_sets,
        recurrent_bytes, refusal_text, row_bytes, seq_terms, stage_term, unit_bytes,
    };
    use crate::placement::Violation;

    fn over() -> PlaceError {
        PlaceError::Broken(vec![Violation::CardOver {
            card: "A6000".to_owned(),
            total: 52_000_000_000,
            limit: 51_000_000_000,
        }])
    }

    /// A set context the plan refuses on bytes, with the draft, says how many
    /// bytes of the plan are the NextN layer the draft holds, in the one line
    /// that carries the plan's arithmetic; without the draft the refusal is
    /// the plan's, word for word.
    #[test]
    fn a_context_refused_with_the_draft_names_the_nextn_bytes() {
        let plain = over().to_string();
        assert_eq!(refusal_text(&over(), None, 8192, 2), plain);
        let named = refusal_text(&over(), Some(300 << 20), 8192, 2);
        assert_eq!(
            named,
            format!(
                "{plain}; the plan counts 314572800 B for the NextN layer the draft holds on the \
                 card beside the trunk (its tensors, its store of 2 slots at 8192 positions a \
                 slot, and its {NEXTN_ARENA_BYTES} B arena)"
            )
        );
        assert!(!named.contains('\n'), "one line: {named}");
    }

    /// The prompt batch's reserve is a refusal of bytes the NextN layer's
    /// share explains too; a refusal that is no matter of bytes (the context
    /// past what the selector is checked at) is the plan's as it came.
    #[test]
    fn only_a_refusal_of_bytes_names_the_nextn_bytes() {
        let front = PlaceError::FrontOver {
            card: "A6000".to_owned(),
            front: 3,
            left: 1,
            short: 2,
        };
        let named = refusal_text(&front, Some(1 << 20), 4096, 1);
        assert!(
            named.starts_with(&front.to_string()) && named.contains("1048576 B"),
            "{named}"
        );
        let past = PlaceError::PastOracle {
            ctx_max: 20_000,
            served: 16_384,
        };
        assert_eq!(
            refusal_text(&past, Some(1 << 20), 20_000, 1),
            past.to_string()
        );
    }

    /// A prompt group unit at GLM-5.3-Flash's sizes (4,096 values, four
    /// streams, eight routed experts a token, 512 columns) is the bytes the
    /// card body's units make (the `prefill units` line's `unit_bytes`), the
    /// tier places a word a slot more; the plan reserves the two a group of
    /// `PROMPT_GROUP` holds past the first.
    #[test]
    fn prompt_unit_bytes() {
        assert_eq!(unit_bytes(4096, 512, 4, 8, false), 67_182_592);
        assert_eq!(unit_bytes(4096, 512, 4, 8, true), 67_182_592 + 4 * 512 * 8);
        assert_eq!(group_sets(PROMPT_GROUP) - 1, 2);
    }

    /// The GEMM front's K-quant forms in bytes: a Q8_0 dense block makes the SwiGLU rows' q8 blocks of 32
    /// values and nothing of 128; a K-quant down swaps them for q8_1 blocks of 128 over the same rows, and a
    /// K-quant gate and up add the normed input's. A column of `k` values takes `256·⌈sb/2⌉ + 256·⌈sb/4⌉ +
    /// 10·sb` words (`sb` = `k / 256` super-blocks); the 32-value rows twenty words a 64-value step.
    #[test]
    fn front_bytes_count_the_dense_blocks_forms() {
        let cols = 64;
        let w = |forms| FrontWidths {
            embd: 2048,
            low: 128,
            gated: 8192,
            q_low: 1536,
            q: 16_384,
            heads_v: 16_384,
            ff: 6144,
            forms,
        };
        let q8 = front_bytes(cols, w(DenseForms::Q8_0));
        // 2048: sb 8 -> 256·4 + 256·2 + 80 = 1,616 words a column; 6144: sb 24 -> 256·12 + 256·6 + 240 = 4,848.
        let (in128, rows128) = (4 * cols * 1_616, 4 * cols * 4_848);
        let rows32 = 4 * 20 * cols * 6144usize.div_ceil(FRONT_STEP);
        let down_k = DenseForms {
            in128: false,
            rows32: false,
            rows128: true,
        };
        assert_eq!(front_bytes(cols, w(down_k)), q8 - rows32 + rows128);
        let all_k = DenseForms {
            in128: true,
            ..down_k
        };
        assert_eq!(front_bytes(cols, w(all_k)), q8 - rows32 + rows128 + in128);
        let mixed = DenseForms {
            rows32: true,
            ..all_k
        };
        assert_eq!(front_bytes(cols, w(mixed)), q8 + rows128 + in128);
        // A load with no dense block makes no feed-forward rows or forms whatever the forms say.
        let none = |forms| front_bytes(cols, FrontWidths { ff: 0, ..w(forms) });
        assert_eq!(none(mixed), none(DenseForms::NONE));
        assert_eq!(none(DenseForms::Q8_0), none(DenseForms::NONE));
    }

    /// The card experts' grouped scratch at GLM-5.3-Flash's sizes (4,096
    /// values, routed experts of 2,048 rows, eight slots a token, 512
    /// columns: 4,096 slots): the rows' q8_1 6,619,136 B, the table's order
    /// 16,384, starts 4,100 and 1,409 tile words 5,636, the entries' Walk A
    /// planes 4,096 · 4,736 = 19,398,656, the SwiGLU rows 33,554,432 and their
    /// q8_1 26,476,544, the downs 67,108,864.
    #[test]
    fn card_tile_scratch_bytes() {
        assert_eq!(
            card_tile_bytes(4096, 2048, 8, 512),
            6_619_136 + 16_384 + 4_100 + 5_636 + 19_398_656 + 33_554_432 + 26_476_544 + 67_108_864
        );
    }

    /// GLM-5.3-Flash's sizes: a KDA layer holds one lane of 64 heads of 128
    /// × 128 f32 and its u32 stamp, or two lanes and two stamps on a load
    /// that verifies, and eleven conv rows of 24,576 f32 channels; a latent
    /// layer 512 + 256 f16 a position and 128 f16 a pool of four, the last
    /// one whole; a layer past the trunk nothing.
    #[test]
    fn glm_layer_bytes() {
        assert_eq!(
            recurrent_bytes(64, 128, 4, KdaLanes::One),
            4_194_304 + 4 + 11 * 24_576 * 4
        );
        assert_eq!(
            recurrent_bytes(64, 128, 4, KdaLanes::Two),
            2 * 4_194_304 + 2 * 4 + 11 * 24_576 * 4
        );
        assert_eq!(row_bytes(512, 128), 1536);
        let kv = KvLayout {
            kinds: vec![Kind::Kda, Kind::Kda, Kind::Kda, Kind::Latent],
            kda: (64, 128, 4),
            lanes: KdaLanes::One,
            row: row_bytes(512, 128),
            pool_row: 256,
            kpool: 4,
        };
        assert_eq!(kv.layer_bytes(0, 2051), 5_275_652);
        assert_eq!(kv.with_lanes(KdaLanes::Two).layer_bytes(0, 2051), 9_469_960);
        assert_eq!(kv.layer_bytes(3, 2051), 2051 * 1536 + 513 * 256);
        assert_eq!(kv.layer_bytes(3, 16_384), 16_384 * (1536 + 64));
        assert_eq!(kv.layer_bytes(4, 2051), 0);
        assert_eq!(
            kv.bytes(0..5, 2051),
            3 * 5_275_652 + 2051 * 1536 + 513 * 256
        );
    }

    /// One resident sequence of GLM-5.3-Flash (34 KDA and 11 latent layers,
    /// four streams of 4,096) at 16,384 positions, as its fill counts it
    /// (`SeqTerms::bytes`): on a plain load the KDA layers at one lane,
    /// 5,275,652 B each, the latent layers 16,384 · 1,600 B each, and beside
    /// them the lane word and the step's row of 65,536 B; on a NextN load the
    /// KDA layers at two lanes, 9,469,960 B each, the NextN layer's store
    /// 16,384 · 1,600 B, and three rows beside the word (the step's, the
    /// verify's second and its row-0 copy). The stage card's KV term of a
    /// NextN plan of two sequences at 1,024 positions: the trunk's stores
    /// twice and the bytes beside one sequence, the layer's store apart.
    #[test]
    fn glm_sequence_bytes() {
        let kinds: Vec<Kind> = (0..45)
            .map(|l| if l % 4 == 3 { Kind::Latent } else { Kind::Kda })
            .collect();
        let kv = KvLayout {
            kinds,
            kda: (64, 128, 4),
            lanes: KdaLanes::One,
            row: row_bytes(512, 128),
            pool_row: 256,
            kpool: 4,
        };
        let draft = NextnKv {
            row: row_bytes(512, 128),
            pool_row: 256,
            kpool: 4,
        };
        let plain = seq_terms(&kv, None, 4, 4096);
        assert_eq!(
            plain.bytes(16_384, 1),
            34 * 5_275_652 + 11 * 16_384 * 1600 + 4 + 65_536
        );
        assert_eq!(plain.bytes(16_384, 1), 467_796_108);
        let two = kv.with_lanes(KdaLanes::Two);
        let drafted = seq_terms(&two, Some(&draft), 4, 4096);
        assert_eq!(
            drafted.bytes(16_384, 1),
            34 * 9_469_960 + 12 * 16_384 * 1600 + 4 + 3 * 65_536
        );
        assert_eq!(drafted.bytes(16_384, 1), 636_748_052);
        let stage = stage_term(&drafted, 0..45, 1024, 2);
        assert_eq!(
            stage,
            2 * (34 * 9_469_960 + 11 * 1024 * 1600) + 4 + 3 * 65_536
        );
        assert_eq!(stage, 680_198_692);
    }

    /// The multi-slot plans ([`super::PlanInputs::plan_slots`],
    /// [`super::PlanInputs::plan_nextn_slots`]) on a synthetic file of four
    /// trunk layers (three KDA, one latent) at GLM-5.3-Flash's widths, a norm
    /// a layer, beside a next-token layer of one latent store: the stage
    /// card's headroom past the bytes the plans add to its kv class after the
    /// totals.
    mod slots {
        use std::num::NonZeroU64;

        use gguf::GgmlType;
        use models::{Arch, ChatSpec, ModelSpec};

        use super::super::super::hparams::{Hc, Hparams, Indexer};
        use super::super::{
            DenseForms, KdaLanes, Kind, KvLayout, NextnInputs, NextnKv, PlaceError, PlanInputs,
            card_leg_bytes, card_leg_needed, card_routed, prompt_reserve_bytes, row_bytes,
        };
        use crate::placement::workstation::HostRead;
        use crate::placement::{
            Card, Host, Machine, ModelTensor, ModelTensors, Plan, PlanLevers, Role, UnifiedPool,
            Violation,
        };

        const LAYERS: usize = 4;
        const CTX: u64 = 4096;

        /// GLM-5.3-Flash's widths on four trunk layers and one next-token
        /// layer; the rest no plan reads.
        fn hparams() -> Hparams {
            Hparams {
                n_layer: LAYERS + 1,
                n_trunk: LAYERS,
                n_embd: 4096,
                n_head: 64,
                n_vocab: 151_552,
                n_ctx_train: 131_072,
                rms_eps: 1e-6,
                norm_eps: 1e-6,
                q_lora: 1536,
                kv_lora: 512,
                head_k: 192,
                head_v: 128,
                conv: 4,
                kda_head_dim: 128,
                gate_lower_bound: 0.0,
                indexer: Indexer {
                    n_head: 32,
                    head_dim: 128,
                    top_k: 2048,
                    kpool: 4,
                },
                index_share_mtp: None,
                hc: Hc {
                    streams: 4,
                    sinkhorn: 20,
                    eps: 1e-6,
                },
                n_expert: 288,
                n_used: 8,
                expert_ff: 2048,
                n_shared: 1,
                shared_ff: 2048,
                dense_lead: 0,
                dense_ff: 0,
                weights_norm: true,
                weights_scale: 1.0,
                limit_exp: vec![0.0; LAYERS + 1],
                limit_shexp: vec![0.0; LAYERS + 1],
                kinds: vec![Kind::Kda, Kind::Kda, Kind::Kda, Kind::Latent, Kind::Latent],
                defaults: Vec::new(),
            }
        }

        /// One norm a layer, `n` layers.
        fn norms(n: usize) -> ModelTensors {
            ModelTensors {
                tensors: (0..n)
                    .map(|l| ModelTensor {
                        name: format!("blk.{l}.attn_norm.weight"),
                        shard: 0,
                        layer: Some(l),
                        role: Role::Attention,
                        ty: GgmlType::F32,
                        dims: vec![4096],
                        file_bytes: 4096 * 4,
                        gathered_rows: None,
                    })
                    .collect(),
                layers: n,
                experts: 288,
                experts_used: 8,
            }
        }

        /// A machine of one card running `layers` with room for every plan
        /// here, and a host with no bound.
        fn machine(layers: usize) -> Machine {
            Machine {
                cards: vec![Card {
                    name: "card".to_string(),
                    device: None,
                    usable_bytes: 1 << 40,
                    context_bytes: 1 << 20,
                    scratch_bytes: 2 << 20,
                    margin_bytes: 0,
                    granule_bytes: NonZeroU64::new(2 << 20).expect("2 MiB"),
                    free_bytes: None,
                    held_by: None,
                    layers: 0..layers,
                    head: true,
                    token_embedding: false,
                    reserves: vec![("draft".to_string(), 3 << 20)],
                }],
                tiers: Vec::new(),
                host: Host {
                    usable_bytes: u64::MAX,
                    reserves: Vec::new(),
                },
                unified: None,
            }
        }

        /// [`machine`] on one pool whose census read `pool`: the card's free
        /// reading and the host's figure are the pool, the host holding a
        /// pinned reserve so the pool binds before the card's own bound.
        fn unified(pool: u64) -> Machine {
            let mut m = machine(LAYERS);
            m.cards[0].free_bytes = Some(pool);
            m.host = Host {
                usable_bytes: pool,
                reserves: vec![("pinned".to_string(), 1 << 30)],
            };
            m.unified = Some(UnifiedPool { bytes: pool });
            m
        }

        /// The next-token layer: one norm, a routed down stack (Q8_0, 4,096
        /// rows of 2,048 an expert) the plan puts on the host, and a latent
        /// store.
        fn nextn() -> NextnInputs {
            let mut model = norms(1);
            let dims = vec![2048, 4096, 288];
            model.tensors.push(ModelTensor {
                name: "blk.0.ffn_down_exps.weight".to_string(),
                shard: 0,
                layer: Some(0),
                role: Role::RoutedExperts,
                ty: GgmlType::Q8_0,
                file_bytes: dims.iter().product::<u64>() / 32 * 34,
                dims,
                gathered_rows: None,
            });
            NextnInputs {
                index: LAYERS,
                model,
                machine: machine(1),
                kv: NextnKv {
                    row: row_bytes(512, 128),
                    pool_row: 256,
                    kpool: 4,
                },
            }
        }

        fn inputs() -> PlanInputs {
            let hp = hparams();
            let kv = KvLayout::of(&hp);
            PlanInputs {
                hp,
                model: norms(LAYERS),
                spec: ModelSpec {
                    arch: Arch::Glm5Next,
                    hidden: 4096,
                    vocab: 151_552,
                    ctx_train: 131_072,
                    rms_eps: 1e-6,
                    layers: Vec::new(),
                    mtp: Vec::new(),
                    hc: None,
                    engram: None,
                    chat: ChatSpec {
                        pre: String::new(),
                        template: None,
                        tools: None,
                        reasoning: None,
                    },
                },
                kv,
                room: (u64::MAX, HostRead::Given),
                arena: None,
            }
        }

        /// A dense layer's three tensors of types `tys` (gate, up, down) at layer `l`.
        fn dense_layer(l: usize, tys: [GgmlType; 3]) -> Vec<ModelTensor> {
            ["ffn_gate", "ffn_up", "ffn_down"]
                .into_iter()
                .zip(tys)
                .map(|(stem, ty)| ModelTensor {
                    name: format!("blk.{l}.{stem}.weight"),
                    shard: 0,
                    layer: Some(l),
                    role: Role::DenseFfn,
                    ty,
                    dims: vec![2048, 6144],
                    file_bytes: 0,
                    gathered_rows: None,
                })
                .collect()
        }

        /// The forms the front holds follow the dense tensors' types in the header, over every dense layer: Q8_0
        /// tensors the 32-value SwiGLU rows alone, K-quant gate and up the input's q8_1 blocks, a K-quant down the
        /// SwiGLU rows' q8_1 blocks, layers of both downs both rows; a shared expert's tensors and a file with no
        /// dense block add none.
        #[test]
        fn the_dense_forms_follow_the_headers_types() {
            use GgmlType::{Q5_K, Q6_K, Q8_0};
            let with = |layers: &[[GgmlType; 3]]| {
                let mut inputs = inputs();
                for (l, tys) in layers.iter().enumerate() {
                    inputs.model.tensors.extend(dense_layer(l, *tys));
                }
                inputs.model.tensors.push(ModelTensor {
                    name: "blk.9.ffn_down_shexp.weight".to_string(),
                    shard: 0,
                    layer: Some(9),
                    role: Role::SharedExpert,
                    ty: Q6_K,
                    dims: vec![2048, 4096],
                    file_bytes: 0,
                    gathered_rows: None,
                });
                (inputs.dense_forms(), inputs.front_widths())
            };
            assert_eq!(with(&[]).0, DenseForms::NONE);
            assert_eq!(with(&[[Q8_0; 3]]).0, DenseForms::Q8_0);
            let k = DenseForms {
                in128: true,
                rows32: false,
                rows128: true,
            };
            assert_eq!(with(&[[Q6_K, Q6_K, Q5_K]]).0, k);
            let (forms, widths) = with(&[[Q6_K, Q6_K, Q5_K], [Q8_0, Q8_0, Q8_0]]);
            assert_eq!(forms, DenseForms { rows32: true, ..k });
            assert_eq!(widths.forms, forms);
        }

        /// The card experts read every type the common card leg runs, and
        /// only those; a file whose routed stacks are all Q4_K or Q5_K (or
        /// host-only, as a Q6_K down) reserves the prompt batch's bytes it
        /// reserved before, and one with a routed stack the tiles do not run
        /// (an IQ3_S gate·up, an IQ4_XS down) reserves the card leg's
        /// buffers beside them (mutant: the reserve without the leg term).
        #[test]
        fn the_card_leg_reserve_follows_the_routed_types() {
            use GgmlType::{IQ3_S, IQ4_XS, Q4_K, Q5_K, Q6_K};
            for id in 0..64 {
                let ty = GgmlType::from_u32(id);
                assert_eq!(
                    card_routed(ty).is_some(),
                    crate::placement::kernels::card_runs(ty),
                    "{ty:?}"
                );
            }
            let routed = |tys: [GgmlType; 3]| {
                let mut i = inputs();
                for (stem, ty) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                    .into_iter()
                    .zip(tys)
                {
                    i.model.tensors.push(ModelTensor {
                        name: format!("blk.1.{stem}.weight"),
                        shard: 0,
                        layer: Some(1),
                        role: Role::RoutedExperts,
                        ty,
                        dims: vec![4096, 2048, 64],
                        file_bytes: 0,
                        gathered_rows: None,
                    });
                }
                i
            };
            let plain = prompt_reserve_bytes(&inputs(), CTX, false);
            for tys in [[Q4_K, Q4_K, Q5_K], [Q5_K, Q5_K, Q6_K], [Q4_K, Q4_K, Q4_K]] {
                let i = routed(tys);
                assert!(!card_leg_needed(&i.model), "{tys:?}");
                assert_eq!(prompt_reserve_bytes(&i, CTX, false), plain, "{tys:?}");
            }
            let leg = card_leg_bytes(4096, 2048, 8) as u64;
            // q8_1 of 8 rows of 4096, 64 slots' gate·up rows of 2048, their q8_1, their downs.
            assert_eq!(leg, 103_424 + 524_288 + 413_696 + 1_048_576);
            for tys in [[IQ3_S, IQ3_S, IQ4_XS], [Q4_K, Q4_K, IQ4_XS]] {
                let i = routed(tys);
                assert!(card_leg_needed(&i.model), "{tys:?}");
                assert_eq!(prompt_reserve_bytes(&i, CTX, false), plain + leg, "{tys:?}");
            }
        }

        /// A card's headroom and its terms: the plan's `headroom_bytes`, and
        /// the card's usable bytes less every term the plan counts on it,
        /// written from the fields.
        fn headroom(p: &Plan<'_>) -> (i128, i128) {
            let t = &p.cards[0];
            let terms = t.dense_bytes
                + t.expert_bytes
                + t.rounding_bytes
                + t.kv_bytes
                + t.scratch_bytes
                + t.context_bytes
                + t.reserve_bytes;
            (
                t.headroom_bytes,
                i128::from(p.usable_bytes(&p.machine.cards[0])) - i128::from(terms),
            )
        }

        /// The stage card's headroom past the bytes beside every slot's
        /// stores, which the plans add to its kv class after the totals, is
        /// its usable bytes less every term: on the plain plan and on the
        /// plan with the next-token layer, at two and three slots.
        #[test]
        fn slot_plans_keep_the_headroom_usable_less_every_term() {
            let (inputs, machine, nextn) = (inputs(), machine(LAYERS), nextn());
            let levers = PlanLevers::default();
            for n in [2, 3] {
                let plain = inputs
                    .plan_slots(&machine, CTX, &levers, KdaLanes::One, n)
                    .expect("the plan");
                let (got, want) = headroom(&plain);
                assert_eq!(got, want, "{n} slots: the plain plan");
                let with = inputs
                    .plan_nextn_slots(&machine, CTX, &levers, &nextn, n)
                    .expect("the plan with the next-token layer");
                let (got, want) = headroom(&with.plan);
                assert_eq!(got, want, "{n} slots: the plan with the next-token layer");
            }
        }

        /// On one pool a plan counts what its load holds beside the plan's own
        /// terms once on each side: the prompt front of a plain plan, the
        /// next-token layer's card bytes and arena, its routed experts on the
        /// host and the front on a plan with it. A pool of exactly the need the
        /// plan's host headroom counts plans clean, and one byte under it is
        /// refused once, by the pool's bound, naming those bytes beside the
        /// plan.
        #[test]
        fn slot_plans_on_one_pool_count_the_bytes_beside_once() {
            let (inputs, nextn) = (inputs(), nextn());
            let levers = PlanLevers::default();
            let front = prompt_reserve_bytes(&inputs, CTX, false);
            let ample: u64 = 1 << 40;
            let check = |at: &str, plan: &dyn Fn(u64) -> Result<i128, PlaceError>, beside: u64| {
                let room = plan(ample).unwrap_or_else(|e| panic!("{at}, an ample pool: {e}"));
                let need = ample - u64::try_from(room).expect("a headroom within the pool");
                if let Err(e) = plan(need) {
                    panic!("{at}, a pool of the host's need {need} B: {e}");
                }
                match plan(need - 1) {
                    Err(PlaceError::Broken(v)) => match v.as_slice() {
                        [Violation::PoolOver { beside: b, .. }] => {
                            assert_eq!(*b, beside, "{at}: the bytes beside the plan");
                        }
                        v => panic!("{at}: {v:?}, not one PoolOver"),
                    },
                    other => panic!("{at}, a pool one byte under the need: {other:?}"),
                }
            };
            for n in [1, 2] {
                check(
                    &format!("{n} slot(s)"),
                    &|pool| {
                        inputs
                            .plan_slots(&unified(pool), CTX, &levers, KdaLanes::One, n)
                            .map(|p| p.host.headroom_bytes)
                    },
                    front,
                );
                let on = unified(ample);
                let with = inputs
                    .plan_nextn_slots(&on, CTX, &levers, &nextn, n)
                    .expect("the plan with the next-token layer");
                assert!(
                    with.nextn.host.expert_bytes > 0,
                    "the draft's experts on the host"
                );
                let beside = with.nextn_card_bytes()
                    + with.arena_bytes
                    + front
                    + with.nextn.host.expert_bytes;
                check(
                    &format!("{n} slot(s) with the next-token layer"),
                    &|pool| {
                        inputs
                            .plan_nextn_slots(&unified(pool), CTX, &levers, &nextn, n)
                            .map(|p| p.host_headroom_bytes)
                    },
                    beside,
                );
            }
        }
    }
}
