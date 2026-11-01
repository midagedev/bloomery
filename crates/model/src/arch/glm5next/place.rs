//! A glm5next file planned onto a machine: the one path from the file's
//! headers to a plan that keeps its invariants, as `deepseek41::place` is for
//! V4.1. The hyperparameters, every tensor's role, the typed description and
//! each layer's KV bytes ([`KvLayout`]) are read once
//! ([`PlanInputs::read`]); a file with a feature the engine does not run is
//! refused there by the coverage check; then the placement
//! ([`PlanInputs::plan`]) runs the expert rule on the routed layers whose
//! stacks the card experts read ([`routed_rule`]), the rest of every layer's
//! experts on the host, and refuses a context past the deepest one a
//! reference set checks the token-pool selector at ([`ORACLE_POSITIONS`]).

use bloomery_placement::slots::{SeqTerms, SlotsOf, Stores};
use gguf::{GgmlType, Split};
use models::ModelSpec;
use runtime::prompt::group_sets;
use runtime::stores;

use super::hparams::{Hparams, Kind};
use super::{names, roles, spec};
use crate::arch::chat_of;
use crate::arch::coverage;
use crate::arch::names::routed_stacks;
use crate::arch::place_hybrid;
use crate::placement::accepts::{Program, leg_runs, routed_rule};
use crate::placement::ctx_ask::PlanRefusal;
use crate::placement::kernels::{self, DownAct, DownEntry, DownForms, GateUpEntry, q8_act_bytes};
use crate::placement::workstation::{self, GRANULE, HostRead, TierBatchBytes};
use crate::placement::{
    self, Card, CardTotals, Device, ExpertList, Format, Host, KvBytes, Machine, ModelTensor,
    ModelTensors, Node, PlacementError, Plan, PlanLevers, Role, Unimplemented, Violation,
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

/// Whether a card layer of gate·up type `gate` and down type `down` runs a
/// prompt batch on the grouped tile kernels (the Q4_K and Q5_K gate·up
/// entries, the Q4_K and Q5_K downs over `Q8Act`, either role, each bit for
/// bit its `_sel`); a card layer of any other types the card experts read
/// ([`routed_rule`]) runs it on the card leg in chunks of [`TIER_CHUNK`]
/// tokens ([`card_leg_bytes`]).
#[must_use]
pub const fn card_tiles_run(gate: GgmlType, down: GgmlType) -> bool {
    matches!(
        kernels::gate_up(gate),
        Some(GateUpEntry::Q4k | GateUpEntry::Q5k)
    ) && matches!(
        kernels::down(down),
        Some((DownEntry::Q4k | DownEntry::Q5k, DownAct::Q8Act))
    )
}

/// Card bytes of the card leg's buffers a prompt batch's card layers run on
/// when their stacks are not [`card_tiles_run`]'s: one run of [`TIER_CHUNK`]
/// tokens of `n` values through experts of `ff` rows, `used` slots a token,
/// the slots' columns in each form of `downs` ([`kernels::leg_bytes`], the
/// leg's buffers); none for no form, when the body makes no leg buffers. The
/// one formula of those bytes: the card body's scratch checks its made bytes
/// against it.
#[must_use]
pub fn card_leg_bytes(n: usize, ff: usize, used: usize, downs: DownForms) -> usize {
    if downs == DownForms::NONE {
        return 0;
    }
    let b = kernels::leg_bytes(TIER_CHUNK, used as u64, n as u64, ff as u64, downs);
    usize::try_from(b).unwrap_or(usize::MAX)
}

/// The routed layers of `model` the card experts run ([`leg_runs`] of the
/// body's program, its plan rule's arms), each with its stacks' `[gate, up,
/// down]` types, read from the headers alone.
fn carded_layers(model: &ModelTensors) -> impl Iterator<Item = (usize, [GgmlType; 3])> + '_ {
    let ty = |name: &str| {
        model
            .tensors
            .iter()
            .find(|t| t.role == Role::RoutedExperts && t.name == name)
            .map(|t| t.ty)
    };
    (0..model.layers)
        .filter_map(move |l| {
            let [g, u, d] = routed_stacks(l).map(|name| ty(&name));
            Some((l, [g?, u?, d?]))
        })
        .filter(|&(_, tys)| leg_runs(Program::Glm5nextBody, tys) == Some(true))
}

/// The down forms of the card leg's buffers a prompt batch runs the card
/// layers of `model` on: those of each layer the card experts run whose
/// stacks the grouped tiles do not ([`card_tiles_run`]); none when every card
/// layer is a tile layer, and the plan then reserves no leg buffers beside the
/// tiles' scratch ([`card_leg_bytes`]).
fn card_leg_forms(model: &ModelTensors) -> DownForms {
    down_forms(
        carded_layers(model)
            .filter(|&(_, [g, _, d])| !card_tiles_run(g, d))
            .map(|(_, [_, _, d])| d),
    )
}

/// The activation forms the down stacks of types `downs` read ([`kernels::down`]); a type with no
/// down entry adds none.
fn down_forms(downs: impl Iterator<Item = GgmlType>) -> DownForms {
    downs
        .filter_map(kernels::down)
        .fold(DownForms::NONE, |forms, (_, act)| forms.with(act))
}

/// Tokens a chunk of the tier's prompt-batch service runs at most: the
/// stage card's chunk of the card experts' launches, which the body binds
/// to its own constant.
pub const TIER_CHUNK: u64 = 8;

/// The expert tier's prompt-batch bytes for the selected program's routed
/// layers: staging, the common leg's candidate-dependent scratch and the
/// host rows and places. The load checks its allocations against the plan.
pub fn tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    let hp = &inputs.hp;
    workstation::try_tier_leg_batch_bytes_of(
        hp.n_embd as u64,
        hp.expert_ff as u64,
        hp.n_used as u64,
        (TIER_CHUNK, crate::moe::UNION_MAX_COLS as u64),
        crate::placement::accepts::leg_down_forms(Program::Glm5nextBody, &inputs.model),
        false,
    )
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

/// The inputs of the GEMM front a mixer's K-quant site reads as q8_1 blocks of 128 values, over the file's
/// mixers: each input holds the q8 blocks of 32 values beside them for its Q8_0 and BF16 sites. A file whose
/// mixer sites are all Q8_0 or BF16 holds none.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MixerForms {
    /// The normed input: a KDA qkv, gate and β, a latent stack.
    pub embd: bool,
    /// A KDA mixer's low-rank gate rows.
    pub low: bool,
    /// A KDA mixer's gated rows.
    pub gated: bool,
    /// A latent mixer's normed query low rank.
    pub q_low: bool,
    /// A latent mixer's heads' outputs.
    pub heads: bool,
}

impl MixerForms {
    /// No K-quant mixer site.
    pub const NONE: MixerForms = MixerForms {
        embd: false,
        low: false,
        gated: false,
        q_low: false,
        heads: false,
    };
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
    /// The mixer inputs a K-quant site reads: their q8_1 blocks of 128 values beside the q8 blocks of 32.
    pub mixer: MixerForms,
}

/// Card bytes of a GEMM front of `cols` token columns of widths `w`, four
/// bytes a word: the dense table — the slot list and its all-zero ids, a
/// column a word each, two words a tile (a tile every [`FRONT_TILE_COLS`]
/// columns, and one more) and the two counts — each quantized input's three
/// activation planes, twenty words a column's [`FRONT_STEP`]-value step
/// (sixteen of codes, two scales, two code sums), over the normed input, the
/// KDA gate's low-rank and gated rows, the latent query's low rank, the
/// latent heads' outputs and, for a Q8_0 down, the dense block's SwiGLU rows;
/// the dense block's K-quant forms ([`DenseForms`]) and the mixers' ([`MixerForms`])
/// in q8_1 blocks of 128 values, the q8_1 planes' own bytes ([`q8_act_bytes`],
/// over the width's whole super-blocks) over the normed input (one buffer for
/// either), a K-quant down's SwiGLU rows and each other input a K-quant mixer
/// site reads; and the f32 rows the front holds whole, a word a value: the latent
/// query heads, the heads' outputs, and the dense block's gate and up rows. A width of 0 makes nothing (the dense block's forms none at an `ff` of 0). The one formula of those
/// bytes: the card body's front checks its made bytes against it.
#[must_use]
pub fn front_bytes(cols: usize, w: FrontWidths) -> usize {
    let table = 2 * cols + 2 * (cols / FRONT_TILE_COLS + 1) + 2;
    let act = |k: usize| 20 * cols * k.div_ceil(FRONT_STEP);
    // The q8_1 planes of the K-quant forms, the common owner's count at the width's whole
    // super-blocks: a width that is not a whole number of them holds a padded last one.
    let act128 = |k: usize| {
        let padded = k.div_ceil(FRONT_SUPER) * FRONT_SUPER;
        q8_act_bytes(cols as u64, padded as u64)
    };
    let forms = if w.ff == 0 { DenseForms::NONE } else { w.forms };
    let rows32 = if forms.rows32 { w.ff } else { 0 };
    let acts: usize = [w.embd, w.low, w.gated, w.q_low, w.heads_v, rows32]
        .into_iter()
        .map(act)
        .sum();
    // The normed input's q8_1 blocks are one buffer for a K-quant dense gate and up and a K-quant mixer site.
    let m = w.mixer;
    let in128 = if forms.in128 || m.embd { w.embd } else { 0 };
    let rows128 = if forms.rows128 { w.ff } else { 0 };
    let mixer128 = [
        (m.low, w.low),
        (m.gated, w.gated),
        (m.q_low, w.q_low),
        (m.heads, w.heads_v),
    ]
    .into_iter()
    .filter_map(|(held, k)| held.then_some(k));
    let acts128: u64 = [in128, rows128]
        .into_iter()
        .chain(mixer128)
        .map(act128)
        .sum();
    let rows = cols * (w.q + w.heads_v + 2 * w.ff);
    usize::try_from(4 * (table + acts + rows) as u64 + acts128)
        .expect("the front's bytes fit a usize")
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
/// ([`card_tile_bytes`] at the front's columns, plus [`card_leg_bytes`] in the
/// forms [`card_leg_forms`] names) and the units a group of
/// [`PROMPT_GROUP`] holds past the first ([`runtime::prompt::group_sets`], [`unit_bytes`] at
/// the front's columns, the tier places when `tiered`). The first unit and
/// the other buffers the units share come out of the card's margin.
#[must_use]
pub fn prompt_reserve_bytes(inputs: &PlanInputs, ctx_max: u64, tiered: bool) -> u64 {
    let hp = &inputs.hp;
    let cols = prompt_cols(ctx_max);
    let unit = unit_bytes(hp.n_embd, cols, hp.hc.streams, hp.n_used, tiered);
    let tiles = card_tile_bytes(hp.n_embd, hp.expert_ff, hp.n_used, cols)
        + card_leg_bytes(
            hp.n_embd,
            hp.expert_ff,
            hp.n_used,
            card_leg_forms(&inputs.model),
        );
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
    /// The effective room is [`Plan::plan_at`]'s: the reading's plan room,
    /// capped by the machine's host room and unified pool.
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
    #[error("{0}")]
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

impl PlanRefusal for PlaceError {
    fn placement(&self) -> Option<&PlacementError> {
        match self {
            Self::Placement(e) => Some(e),
            _ => None,
        }
    }

    fn violations(&self) -> Option<&[Violation]> {
        match self {
            Self::Broken(v) => Some(v),
            _ => None,
        }
    }
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
    /// arena's given bytes, a room that cannot be read refused by name. The
    /// room is read after the file's description, through the same steps as
    /// [`PlanInputs::describe_in`].
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        Self::describe_with(split, || {
            workstation::host_available_read().map_err(PlacementError::HostRoom)
        })
    }

    /// The file's description with `room` as the host's room, no host read
    /// for it: another machine's plan checked from this one, `room` that
    /// machine's bytes and how they were read. The arena is still this
    /// process's `BLOOMERY_NVTIER_BYTES`, as in [`PlanInputs::describe`].
    pub fn describe_in(split: &Split, room: (u64, HostRead)) -> Result<PlanInputs, PlacementError> {
        Self::describe_with(split, || Ok(room))
    }

    /// The file's hyperparameters, roles, chat surface, spec and KV layout,
    /// then the room `room` gives, then the arena, the first that fails
    /// being the error.
    fn describe_with(
        split: &Split,
        room: impl FnOnce() -> Result<(u64, HostRead), PlacementError>,
    ) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let model = roles::classify(split, &hp)?;
        let chat = chat_of(split, spec::TOOLS, Some(models::ReasoningFormat::THINK))?;
        let spec = spec::spec_of(&hp, &model, chat)?;
        let kv = KvLayout::of(&hp);
        let room = room()?;
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
            mixer: self.mixer_forms(),
        }
    }

    /// The inputs of the front a K-quant mixer site reads ([`MixerForms`]), from the types of the trunk's
    /// sites in the header: the sites the prompt batch's GEMM front runs, each by the input it reads and
    /// the kind of its layer.
    #[must_use]
    pub fn mixer_forms(&self) -> MixerForms {
        let hp = &self.hp;
        let kquant: std::collections::HashSet<&str> = self
            .model
            .tensors
            .iter()
            .filter(|t| t.role == Role::Attention && is_kquant(t.ty))
            .map(|t| t.name.as_str())
            .collect();
        let held = |name: String| kquant.contains(name.as_str());
        let mut f = MixerForms::NONE;
        for (l, kind) in hp.kinds[..hp.n_trunk].iter().enumerate() {
            match kind {
                Kind::Kda => {
                    f.embd |= [names::attn_q(l), names::attn_k(l), names::attn_v(l)]
                        .into_iter()
                        .chain([names::ssm_g_a(l), names::ssm_beta(l)])
                        .any(held);
                    f.low |= held(names::ssm_g_b(l));
                    f.gated |= held(names::attn_output(l));
                }
                Kind::Latent => {
                    f.embd |= [
                        names::attn_q_a(l),
                        names::attn_kv_a_mqa(l),
                        names::indexer_attn_k(l),
                        names::indexer_compressor_gate(l),
                    ]
                    .into_iter()
                    .any(held);
                    f.q_low |= held(names::attn_q_b(l));
                    f.heads |= held(names::attn_output(l));
                }
            }
        }
        f
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
    /// expert rule on the layers whose routed stacks [`routed_rule`] carries,
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
        self.plan_slots_nodes(machine, ctx_max, levers, lanes, slots, None)
    }

    /// Lower explicit node-owned expert ranges through the common placement owner.
    pub fn plan_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        lanes: KdaLanes,
        owned: &[u64],
    ) -> Result<Plan<'a>, PlaceError> {
        self.plan_slots_nodes(machine, ctx_max, levers, lanes, 1, Some(owned))
    }

    fn plan_slots_nodes<'a>(
        &'a self,
        machine: &'a Machine,
        ctx_max: u64,
        levers: &PlanLevers,
        lanes: KdaLanes,
        slots: usize,
        owned: Option<&[u64]>,
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
        let front = prompt_reserve_bytes(self, ctx_max, machine.expert_tiers().next().is_some());
        let plan = placement::plan_in_pool(machine, levers, |levers| {
            let mut plan = placement::nodes::loaded_routed(
                &self.model,
                machine,
                ctx_max,
                &terms.slots_of(n),
                levers,
                routed_rule(Program::Glm5nextBody),
                front.saturating_add(beside),
                owned,
            )?;
            // The bytes beside the stores ride the stage card's KV term, out
            // of the budget reserved above, so the card's bound holds them.
            if let Some(t) = plan.cards.first_mut() {
                t.grow_kv(beside);
            }
            let room = plan.plan_at(&self.room);
            placement::expert_nvme_tier(&mut plan, room, self.arena)?;
            Ok(plan)
        })?;
        // The front stays beside the plan's terms (`refuse_reserve` holds it to
        // the card), so a unified pool counts it beside them.
        let plan = checked_beside(plan, front).map_err(PlaceError::Broken)?;
        if let (Some(card), Some(t)) = (machine.cards.first(), plan.cards.first()) {
            let limit = plan.usable_bytes(card).saturating_sub(card.margin_bytes);
            place_hybrid::refuse_reserve(
                card,
                place_hybrid::card_terms(t),
                limit,
                front,
                |card, front, left, short| PlaceError::FrontOver {
                    card,
                    front,
                    left,
                    short,
                },
            )?;
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
        if let Some(missing) = nextn_required().find(|stem| {
            let want = format!("{prefix}{stem}");
            !tensors.iter().any(|t| t.name == want)
        }) {
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
                bf16_words: |_| false,
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
            tier_leg: None,
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
            reserve_only: false,
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: Host {
                usable_bytes: u64::MAX,
                reserves: Vec::new(),
            },
            unified: None,
        }],
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
    /// fits; its totals are the target's alone. Its host headroom leaves the
    /// NextN layer's routed experts in, for the load to take out beside it.
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
    /// it; a room under them is refused by name
    /// ([`PlacementError::DraftHostRoom`]).
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
        let front = prompt_reserve_bytes(self, ctx_max, machine.expert_tiers().next().is_some());
        let nextn_card = nextn_card_bytes(&draft.cards[0]);
        let reserve = (nextn_card + arena + front).saturating_add(beside);
        // The draft layer's routed experts are the host's beside the trunk's,
        // a plan of their own: the trunk splits at the room they leave.
        let draft_host = draft.host.expert_bytes;
        let plan = placement::plan_in_pool(machine, levers, |levers| {
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
            let planned_at = plan.plan_at(&self.room);
            let room = planned_at
                .checked_sub(draft_host)
                .ok_or(PlacementError::DraftHostRoom {
                    room: planned_at,
                    draft: draft_host,
                })?;
            placement::expert_nvme_tier(&mut plan, room, self.arena)?;
            // A split plan's headroom is what the room it ran at left, a room
            // short of the draft's experts. An unsplit plan's headroom never
            // took them out; the split one's gets them back, so the load,
            // which hosts them beside the plan, takes them out once.
            if plan.host.nvme_expert_bytes > 0 {
                plan.host.headroom_bytes += i128::from(draft_host);
            }
            Ok(plan)
        })?;
        let total = place_hybrid::card_terms(&plan.cards[0]) + nextn_card + arena;
        let usable = plan.usable_bytes(card);
        let limit = usable.saturating_sub(card.margin_bytes);
        let host_headroom = plan.host.headroom_bytes - i128::from(draft_host);
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
            let usable = machine.host().usable_bytes;
            broken.push(Violation::HostOver {
                total: u64::try_from(i128::from(usable) - host_own).unwrap_or(u64::MAX),
                usable,
            });
        }
        if !broken.is_empty() {
            return Err(PlaceError::Broken(broken));
        }
        place_hybrid::refuse_reserve(card, total, limit, front, |card, front, left, short| {
            PlaceError::FrontOver {
                card,
                front,
                left,
                short,
            }
        })?;
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
        DenseForms, FRONT_STEP, FrontWidths, KdaLanes, Kind, KvBytes, KvLayout, MixerForms,
        NextnKv, PROMPT_GROUP, card_tile_bytes, front_bytes, group_sets, recurrent_bytes,
        row_bytes, seq_terms, stage_term, unit_bytes,
    };

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
            mixer: MixerForms::NONE,
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
        // A K-quant mixer site adds the q8_1 blocks of its input: 2048 as above, 8192 (sb 32): 256·16 + 256·8 +
        // 320 = 6,464 words, 1536 (sb 6): 256·3 + 256·2 + 60 = 1,340, 16,384 (sb 64): 256·32 + 256·16 + 640 =
        // 12,928; the normed input's blocks are one buffer for the dense block and the mixers.
        let with = |mixer, forms| front_bytes(cols, FrontWidths { mixer, ..w(forms) });
        let embd = MixerForms {
            embd: true,
            ..MixerForms::NONE
        };
        assert_eq!(with(embd, DenseForms::Q8_0), q8 + in128);
        assert_eq!(with(embd, all_k), front_bytes(cols, w(all_k)));
        let (gated, q_low, heads) = (4 * cols * 6_464, 4 * cols * 1_340, 4 * cols * 12_928);
        let rest = MixerForms {
            gated: true,
            q_low: true,
            heads: true,
            ..MixerForms::NONE
        };
        assert_eq!(with(rest, DenseForms::Q8_0), q8 + gated + q_low + heads);
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
            DenseForms, KdaLanes, Kind, KvLayout, MixerForms, NextnInputs, NextnKv, PlaceError,
            PlanInputs, card_leg_bytes, card_leg_forms, prompt_reserve_bytes, row_bytes,
        };
        use crate::placement::churn::ChurnPool;
        use crate::placement::workstation::{HostNeed, HostRead};
        use crate::placement::{
            Card, Host, Machine, ModelTensor, ModelTensors, Node, PlacementError, Plan, PlanLevers,
            PoolNeed, Role, UnifiedPool,
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
                bf16_words: |_| false,
            }
        }

        /// A machine of one card running `layers` with room for every plan
        /// here, and a host with no bound.
        fn machine(layers: usize) -> Machine {
            Machine {
                cards: vec![Card {
                    name: "card".to_string(),
                    tier_leg: None,
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
                    reserve_only: false,
                    reserves: vec![("draft".to_string(), 3 << 20)],
                }],
                tiers: Vec::new(),
                nodes: vec![Node {
                    host: Host {
                        usable_bytes: u64::MAX,
                        reserves: Vec::new(),
                    },
                    unified: None,
                }],
            }
        }

        /// [`machine`] on one pool whose census read `pool`: the card's free
        /// reading and the host's figure are the pool, the host holding a
        /// pinned reserve so the pool binds before the card's own bound.
        fn unified(pool: u64) -> Machine {
            let mut m = machine(LAYERS);
            m.cards[0].free_bytes = Some(pool);
            *m.host_mut() = Host {
                usable_bytes: pool,
                reserves: vec![("pinned".to_string(), 1 << 30)],
            };
            *m.unified_mut() = Some(UnifiedPool { bytes: pool });
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
                    encoder: None,
                    logit_epilogue: None,
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

        /// [`inputs`] with a routed layer of three stacks of types `layers[l]` (gate, up, down) at each layer `l`.
        fn routed_types(layers: &[[GgmlType; 3]]) -> PlanInputs {
            let mut i = inputs();
            for (l, tys) in layers.iter().enumerate() {
                for (stem, &ty) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                    .into_iter()
                    .zip(tys)
                {
                    i.model.tensors.push(ModelTensor {
                        name: format!("blk.{l}.{stem}.weight"),
                        shard: 0,
                        layer: Some(l),
                        role: Role::RoutedExperts,
                        ty,
                        dims: vec![4096, 2048, 64],
                        file_bytes: 0,
                        gathered_rows: None,
                    });
                }
            }
            i
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

        /// The mixer forms follow the header's types: a K-quant attention output reads the gated rows on a KDA
        /// layer and the heads' outputs on a latent one, a K-quant `attn_q`, `ssm_g_a` or `attn_kv_a_mqa` the
        /// normed input, `ssm_g_b` the low-rank rows, `attn_q_b` the query's low rank; Q8_0 sites and a K-quant
        /// absorbed pair (which the front does not run) hold none. Layer 0 is a KDA layer, layer 3 a latent one.
        #[test]
        fn the_mixer_forms_follow_the_headers_types() {
            use GgmlType::{Q5_K, Q6_K, Q8_0};
            let with = |sites: &[(usize, &str, GgmlType)]| {
                let mut inputs = inputs();
                for &(l, stem, ty) in sites {
                    inputs.model.tensors.push(ModelTensor {
                        name: format!("blk.{l}.{stem}.weight"),
                        shard: 0,
                        layer: Some(l),
                        role: Role::Attention,
                        ty,
                        dims: vec![4096, 4096],
                        file_bytes: 0,
                        gathered_rows: None,
                    });
                }
                (inputs.mixer_forms(), inputs.front_widths())
            };
            let none = MixerForms::NONE;
            assert_eq!(with(&[(0, "attn_q", Q8_0), (3, "attn_q_b", Q8_0)]).0, none);
            assert_eq!(
                with(&[(3, "attn_k_b", Q6_K), (3, "attn_v_b", Q6_K)]).0,
                none
            );
            let one = |l, stem, ty| with(&[(l, stem, ty)]).0;
            let embd = MixerForms { embd: true, ..none };
            assert_eq!(one(0, "attn_k", Q6_K), embd);
            assert_eq!(one(0, "ssm_g_a", Q5_K), embd);
            assert_eq!(one(3, "attn_kv_a_mqa", Q6_K), embd);
            assert_eq!(one(0, "ssm_g_b", Q5_K), MixerForms { low: true, ..none });
            assert_eq!(
                one(0, "attn_output", Q5_K),
                MixerForms {
                    gated: true,
                    ..none
                }
            );
            assert_eq!(
                one(3, "attn_output", Q5_K),
                MixerForms {
                    heads: true,
                    ..none
                }
            );
            assert_eq!(
                one(3, "attn_q_b", Q5_K),
                MixerForms {
                    q_low: true,
                    ..none
                }
            );
            let (forms, widths) = with(&[(0, "attn_output", Q5_K), (3, "attn_q_b", Q6_K)]);
            assert_eq!(
                forms,
                MixerForms {
                    gated: true,
                    q_low: true,
                    ..none
                }
            );
            assert_eq!(widths.mixer, forms);
        }

        /// An attention projection in bloomery's MXFP8 (`GgmlType::MXFP8`, as an
        /// FP4-Local checkpoint ships its attention) is placed on the stage
        /// card in its own two planes at the bytes the card loader allocates,
        /// beside a Q8_0 projection of the same shape in Q8_0's planes; one
        /// whose row is no whole number of scale words (49 blocks) is refused
        /// by the plan naming it.
        #[test]
        fn an_mxfp8_attention_projection_is_placed_in_its_planes() {
            use crate::placement::{CardFormat, Device, Format};
            let mxfp8 = GgmlType::MXFP8;
            let projection = |name: &str, ty: GgmlType, k: u64| {
                let rows = 16_384u64;
                let (blck, size) = (ty.blck_size().unwrap(), ty.type_size().unwrap());
                ModelTensor {
                    name: name.to_string(),
                    shard: 0,
                    layer: Some(3),
                    role: Role::Attention,
                    ty,
                    dims: vec![k, rows],
                    file_bytes: rows * k / blck * size,
                    gathered_rows: None,
                }
            };
            let mut i = inputs();
            i.model.tensors.extend([
                projection("blk.3.attn_q_b.weight", mxfp8, 1536),
                projection("blk.3.indexer.attn_q_b.weight", GgmlType::Q8_0, 1536),
            ]);
            let machine = machine(LAYERS);
            let plan = i
                .plan(&machine, CTX, &PlanLevers::default())
                .expect("the plan");
            for (name, ty, format) in [
                ("blk.3.attn_q_b.weight", mxfp8, CardFormat::Mxfp8Planes),
                (
                    "blk.3.indexer.attn_q_b.weight",
                    GgmlType::Q8_0,
                    CardFormat::Q8_0Planes,
                ),
            ] {
                let row = plan
                    .rows
                    .iter()
                    .find(|r| i.model.tensors[r.tensor].name == name)
                    .expect("its row");
                let want = 16_384 * (1536 / 32) * if ty == mxfp8 { 33 } else { 34 };
                let got: Vec<_> = row
                    .segments
                    .iter()
                    .map(|s| (s.device, s.format, s.resident_bytes))
                    .collect();
                assert_eq!(
                    got,
                    [(Device::Card(0), Format::Card(format), want)],
                    "{name}"
                );
            }
            let mut i = inputs();
            i.model
                .tensors
                .push(projection("blk.3.attn_q_b.weight", mxfp8, 1568));
            match i.plan(&machine, CTX, &PlanLevers::default()) {
                Err(e) => assert!(e.to_string().contains("blk.3.attn_q_b.weight"), "{e}"),
                Ok(_) => panic!("49 blocks a row placed in whole scale words"),
            }
        }

        /// The card leg's reserve follows the card layers' types: a file whose
        /// card layers are all tile layers (a Q4_K or Q5_K gate·up and down)
        /// reserves the prompt batch's bytes it reserved before; one with a
        /// card layer the tiles do not run reserves the card leg's buffers in
        /// its down's form beside them: an IQ3_S gate·up, an IQ4_XS down or a
        /// Q6_K down the 128-value form, an NVFP4 layer or a Q8_0 down the
        /// 32-value form, an IQ3_S layer and an NVFP4 layer together both.
        /// Mutants: the reserve without the leg term; the down arm without
        /// Q6_K (the Q6_K-down layers reserve nothing); the NVFP4 layer's form
        /// left out.
        // PIN(2026-10-22): a Q6_K down is a card layer the tiles do not run
        // (`card_down_runs` admits it over the 128-value form), so the Q5_K
        // and IQ3_S layers beside one reserve the leg's 128-value buffers. The
        // old pin kept both on the plain reserve while a Q6_K down was host-only.
        #[test]
        fn the_card_leg_reserve_follows_the_routed_types() {
            use crate::placement::kernels::{DownAct, DownForms};
            use GgmlType::{IQ3_S, IQ4_XS, NVFP4, Q4_K, Q5_K, Q6_K, Q8_0};
            let plain = prompt_reserve_bytes(&inputs(), CTX, false);
            for tys in [[Q4_K, Q4_K, Q5_K], [Q4_K, Q4_K, Q4_K]] {
                let i = routed_types(&[tys]);
                assert_eq!(card_leg_forms(&i.model), DownForms::NONE, "{tys:?}");
                assert_eq!(prompt_reserve_bytes(&i, CTX, false), plain, "{tys:?}");
            }
            let q8 = DownForms::of(DownAct::Q8Act);
            let b32 = DownForms::of(DownAct::Q8Blocks32);
            let leg = |forms| card_leg_bytes(4096, 2048, 8, forms) as u64;
            assert_eq!(card_leg_bytes(4096, 2048, 8, DownForms::NONE), 0);
            // q8_1 of 8 rows of 4096, 64 slots' gate·up rows of 2048, their q8_1, their downs.
            assert_eq!(leg(q8), 103_424 + 524_288 + 413_696 + 1_048_576);
            // The same with the slots' columns in 32-value blocks: 64 blocks a column.
            assert_eq!(leg(b32), 103_424 + 524_288 + 163_840 + 1_048_576);
            for (layers, forms) in [
                (vec![[IQ3_S, IQ3_S, IQ4_XS]], q8),
                (vec![[Q4_K, Q4_K, IQ4_XS]], q8),
                (vec![[Q5_K, Q5_K, Q6_K]], q8),
                (vec![[IQ3_S, IQ3_S, Q6_K]], q8),
                (vec![[NVFP4; 3]], b32),
                (vec![[Q4_K, Q4_K, Q8_0]], b32),
                (
                    vec![[IQ3_S, IQ3_S, IQ4_XS], [NVFP4; 3]],
                    q8.with(DownAct::Q8Blocks32),
                ),
            ] {
                let i = routed_types(&layers);
                assert_eq!(card_leg_forms(&i.model), forms, "{layers:?}");
                assert_eq!(
                    prompt_reserve_bytes(&i, CTX, false),
                    plain + leg(forms),
                    "{layers:?}"
                );
            }
        }

        /// A Q5_K gate and up with a Q6_K down takes every expert on a card
        /// with room for the stack; a Q6_K gate and up takes none.
        /// PIN(2026-10-22): the down arm admits Q6_K, so the layer is
        /// card-eligible. It was host-only while both arms read `card_runs`
        /// (those 288 experts stayed on the host). This card holds the whole
        /// stack; the A6000 serving count is the file's plan, not 288.
        #[test]
        fn a_q6k_down_layer_takes_the_cards_experts() {
            use GgmlType::{Q5_K, Q6_K};
            let mut i = inputs();
            let push = |model: &mut ModelTensors, l: usize, tys: [GgmlType; 3]| {
                for (stem, ty) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                    .into_iter()
                    .zip(tys)
                {
                    model.tensors.push(ModelTensor {
                        name: format!("blk.{l}.{stem}.weight"),
                        shard: 0,
                        layer: Some(l),
                        role: Role::RoutedExperts,
                        ty,
                        dims: vec![4096, 2048, 288],
                        file_bytes: 0,
                        gathered_rows: None,
                    });
                }
            };
            push(&mut i.model, 1, [Q5_K, Q5_K, Q6_K]);
            push(&mut i.model, 2, [Q6_K, Q6_K, Q6_K]);
            let machine = machine(LAYERS);
            let plan = i
                .plan(&machine, CTX, &PlanLevers::default())
                .expect("the plan");
            assert_eq!(plan.n_l[1], 288, "the Q6_K down");
            assert_eq!(plan.n_l[2], 0, "the Q6_K gate");
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
        /// refused by the host floor before pool validation.
        // PIN(2026-10-23): capped tier rooms refuse at the host floor before pool validation.
        #[test]
        fn slot_plans_on_one_pool_count_the_bytes_beside_once() {
            let (inputs, nextn) = (inputs(), nextn());
            let levers = PlanLevers::default();
            let front = prompt_reserve_bytes(&inputs, CTX, false);
            let ample: u64 = 1 << 40;
            let check = |at: &str,
                         plan: &dyn Fn(u64) -> Result<(i128, PoolNeed), PlaceError>,
                         beside: u64| {
                let (room, pool) =
                    plan(ample).unwrap_or_else(|e| panic!("{at}, an ample pool: {e}"));
                let need = ample - u64::try_from(room).expect("a headroom within the pool");
                let physical = i128::from(ample)
                    + i128::from(pool.cards)
                    + i128::from(pool.host)
                    + i128::from(beside)
                    - i128::from(pool.pool);
                assert_eq!(
                    i128::from(need),
                    physical,
                    "{at}: every beside byte counted once"
                );
                if let Err(e) = plan(need) {
                    panic!("{at}, a pool of the host's need {need} B: {e}");
                }
                match plan(need - 1) {
                    Err(PlaceError::Placement(PlacementError::PoolBudget { last, .. })) => {
                        match *last {
                            PlacementError::HostRoomFloor { room, floor, .. } => {
                                assert_eq!(floor - room, 1, "{at}: the exact one-byte shortfall");
                            }
                            e => panic!("{at}: {e:?}, not the host floor"),
                        }
                    }
                    other => panic!("{at}, a pool one byte under the need: {other:?}"),
                }
            };
            for n in [1, 2] {
                check(
                    &format!("{n} slot(s)"),
                    &|pool| {
                        inputs
                            .plan_slots(&unified(pool), CTX, &levers, KdaLanes::One, n)
                            .map(|p| {
                                (
                                    p.host.headroom_bytes,
                                    p.pool_need().expect("a unified pool"),
                                )
                            })
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
                            .map(|p| {
                                (
                                    p.host_headroom_bytes,
                                    p.plan.pool_need().expect("a unified pool"),
                                )
                            })
                    },
                    beside,
                );
            }
        }

        /// [`inputs`] at `room` as given, with Q4_K routed stacks (gate, up
        /// and down, 1,024 rows of 4,096 an expert) on every trunk layer.
        fn routed(room: u64) -> PlanInputs {
            let mut i = inputs();
            for l in 0..LAYERS {
                for (stem, dims) in [
                    ("ffn_gate_exps", [4096, 1024, 288]),
                    ("ffn_up_exps", [4096, 1024, 288]),
                    ("ffn_down_exps", [1024, 4096, 288]),
                ] {
                    i.model.tensors.push(ModelTensor {
                        name: format!("blk.{l}.{stem}.weight"),
                        shard: 0,
                        layer: Some(l),
                        role: Role::RoutedExperts,
                        ty: GgmlType::Q4_K,
                        file_bytes: dims.iter().product::<u64>() / 256 * 144,
                        dims: dims.to_vec(),
                        gathered_rows: None,
                    });
                }
            }
            i.room = (room, HostRead::Given);
            i
        }

        #[test]
        fn nvme_room_is_capped_by_the_unified_pool() {
            let levers = PlanLevers {
                card_budget_bytes: bloomery_levers::CardBudget::Every(1 << 30),
            };
            for slots in [1, 2, 4] {
                let ample = unified(1 << 40);
                let wide = routed(1 << 40);
                let whole = wide
                    .plan_slots(&ample, CTX, &levers, KdaLanes::One, slots)
                    .unwrap();
                let pool = HostNeed::of(&whole, 0).bytes() - 1;
                let m = unified(pool);
                let tight = routed(pool);
                let expected = tight
                    .plan_slots(&m, CTX, &levers, KdaLanes::One, slots)
                    .expect("the trunk fits with NVMe");
                let plan = wide
                    .plan_slots(&m, CTX, &levers, KdaLanes::One, slots)
                    .expect("a larger host reading keeps the same pool budget");
                assert!(plan.host.nvme_expert_bytes > 0);
                assert_eq!(plan.n_l, expected.n_l);
                assert_eq!(format!("{:?}", plan.rows), format!("{:?}", expected.rows));
                assert_eq!(plan.host.headroom_bytes, expected.host.headroom_bytes);
                assert!(
                    plan.violations_beside(prompt_reserve_bytes(&wide, CTX, false))
                        .is_empty()
                );
            }
            assert!(
                routed(1 << 40)
                    .plan_slots(&unified(1 << 30), CTX, &levers, KdaLanes::One, 1)
                    .is_err()
            );
        }

        #[test]
        fn nextn_nvme_room_is_capped_by_the_unified_pool() {
            let levers = PlanLevers {
                card_budget_bytes: bloomery_levers::CardBudget::Every(1 << 30),
            };
            let nextn = nextn();
            for slots in [1, 2, 4] {
                let ample = unified(1 << 40);
                let wide = routed(1 << 40);
                let whole = wide
                    .plan_nextn_slots(&ample, CTX, &levers, &nextn, slots)
                    .unwrap();
                let pool = HostNeed::of(&whole.plan, whole.nextn.host.expert_bytes).bytes() - 1;
                let m = unified(pool);
                let tight = routed(pool);
                let expected = tight
                    .plan_nextn_slots(&m, CTX, &levers, &nextn, slots)
                    .expect("the trunk and NextN fit with NVMe");
                let plan = wide
                    .plan_nextn_slots(&m, CTX, &levers, &nextn, slots)
                    .expect("NextN keeps the same pool budget");
                assert!(plan.plan.host.nvme_expert_bytes > 0);
                assert_eq!(plan.plan.n_l, expected.plan.n_l);
                assert_eq!(
                    format!("{:?}", plan.plan.rows),
                    format!("{:?}", expected.plan.rows)
                );
                assert_eq!(plan.host_headroom_bytes, expected.host_headroom_bytes);
                assert!(
                    plan.plan
                        .violations_beside(
                            plan.nextn_card_bytes()
                                + plan.arena_bytes
                                + prompt_reserve_bytes(&wide, CTX, false)
                                + plan.nextn.host.expert_bytes
                        )
                        .is_empty()
                );
            }
        }

        #[test]
        fn node_nvme_room_is_capped_by_the_unified_pool() {
            let levers = PlanLevers {
                card_budget_bytes: bloomery_levers::CardBudget::Every(1 << 30),
            };
            for ranks in [1, 2, 4] {
                let wide = routed(1 << 40);
                let owned = vec![wide.model.experts / ranks; ranks as usize];
                let mut m = unified(1 << 40);
                for rank in 1..ranks {
                    let mut tier = m.cards[0].clone();
                    tier.name = format!("tier-{rank}");
                    tier.head = false;
                    tier.layers = 0..0;
                    m.tiers.push(tier);
                    m.nodes.push(m.nodes[0].clone());
                }
                let whole = wide
                    .plan_nodes(&m, CTX, &levers, KdaLanes::Two, &owned)
                    .unwrap();
                let pool = HostNeed::of(&whole, 0).bytes() - 1;
                m.host_mut().usable_bytes = pool;
                *m.unified_mut() = Some(UnifiedPool { bytes: pool });
                m.cards[0].free_bytes = Some(pool);
                let tight = routed(pool);
                let expected = tight
                    .plan_nodes(&m, CTX, &levers, KdaLanes::Two, &owned)
                    .expect("the node-owned experts fit with NVMe");
                let plan = wide
                    .plan_nodes(&m, CTX, &levers, KdaLanes::Two, &owned)
                    .expect("each rank keeps the same local pool budget");
                assert!(plan.host.nvme_expert_bytes > 0, "{ranks} ranks");
                assert_eq!(plan.n_l, expected.n_l);
                assert_eq!(plan.tier_n_l, expected.tier_n_l);
                assert_eq!(plan.tier_n_l.len(), ranks as usize - 1);
                assert_eq!(format!("{:?}", plan.rows), format!("{:?}", expected.rows));
            }
        }

        /// The tier reserves precisely the selected layers' down forms,
        /// including a mixed 128/32-value load; a gate/up mismatch remains
        /// host-only and cannot add a form.
        #[test]
        fn tier_scratch_follows_selected_mixed_layers() {
            use super::super::TIER_CHUNK;
            use crate::placement::kernels::{self, DownAct, DownForms};
            use GgmlType::{IQ3_S, NVFP4, Q5_K, Q8_0};
            let mut inputs = routed(64 << 30);
            inputs
                .model
                .tensors
                .retain(|t| t.role != Role::RoutedExperts || t.layer.is_some_and(|l| l < 2));
            let set = |inputs: &mut PlanInputs, l, tys: [GgmlType; 3]| {
                for t in inputs
                    .model
                    .tensors
                    .iter_mut()
                    .filter(|t| t.role == Role::RoutedExperts && t.layer == Some(l))
                {
                    let part = match crate::placement::ExpertPart::of_name(&t.name) {
                        Some(crate::placement::ExpertPart::Gate) => 0,
                        Some(crate::placement::ExpertPart::Up) => 1,
                        Some(crate::placement::ExpertPart::Down) => 2,
                        None => unreachable!(),
                    };
                    t.ty = tys[part];
                }
            };
            for (layers, forms) in [
                (
                    [[IQ3_S, IQ3_S, Q5_K], [Q5_K, Q5_K, Q5_K]],
                    DownForms::of(DownAct::Q8Act),
                ),
                (
                    [[Q5_K, Q5_K, Q8_0], [Q5_K, Q5_K, Q8_0]],
                    DownForms::of(DownAct::Q8Blocks32),
                ),
                (
                    [[IQ3_S, IQ3_S, Q5_K], [NVFP4; 3]],
                    DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32),
                ),
                (
                    [[IQ3_S, IQ3_S, Q5_K], [Q5_K, Q5_K, Q8_0]],
                    DownForms::of(DownAct::Q8Act).with(DownAct::Q8Blocks32),
                ),
                (
                    [[IQ3_S, IQ3_S, Q5_K], [IQ3_S, Q5_K, Q8_0]],
                    DownForms::of(DownAct::Q8Act),
                ),
            ] {
                for (l, tys) in layers.into_iter().enumerate() {
                    set(&mut inputs, l, tys);
                }
                let bytes = super::super::tier_batch(&inputs).expect("tier bytes");
                let hp = &inputs.hp;
                assert_eq!(
                    bytes.scratch,
                    kernels::leg_bytes(
                        TIER_CHUNK,
                        hp.n_used as u64,
                        hp.n_embd as u64,
                        hp.expert_ff as u64,
                        forms,
                    ),
                    "{layers:?}",
                );
            }
        }

        #[test]
        fn node_ranges_use_the_common_loaded_assignment() {
            let inputs = routed(64 << 30);
            let mut machine = unified(64 << 30);
            let mut tier = machine.cards[0].clone();
            tier.name = "remote".into();
            tier.head = false;
            tier.layers = 0..0;
            machine.tiers.push(tier);
            let half = inputs.model.experts / 2;
            let owned = [half, inputs.model.experts - half];
            let plan = inputs
                .plan_nodes(&machine, CTX, &PlanLevers::default(), KdaLanes::Two, &owned)
                .unwrap();
            assert!(plan.violations().is_empty());
            assert_eq!(plan.tier_n_l, vec![vec![owned[1]; LAYERS]]);
            assert!(plan.n_l.iter().all(|&n| n <= owned[0]));
            assert_eq!(plan.cards[1].expert_bytes, 0);
            assert!(
                inputs
                    .plan_nodes(
                        &machine,
                        CTX,
                        &PlanLevers::default(),
                        KdaLanes::Two,
                        &[inputs.model.experts, 0]
                    )
                    .is_err()
            );
        }

        /// The load hosts the next-token layer's routed experts beside the
        /// plan and takes them out of its host headroom once
        /// (`ChurnPool::check_beside`, a pool of 0 B): what is left, and the
        /// plan's `host_headroom_bytes`, are the room less the host need with
        /// them, whether the trunk's experts fit an ample room or split through
        /// the NVMe tier at one byte under that need (mutants: a split plan's
        /// headroom short of the draft's experts, or given them back twice;
        /// an unsplit plan's given them again).
        #[test]
        fn a_nextn_plan_leaves_the_draft_experts_in_its_host_headroom() {
            let (m, nextn) = (machine(LAYERS), nextn());
            let levers = PlanLevers {
                card_budget_bytes: Some(1 << 30).into(),
            };
            let ample = routed(u64::MAX);
            let whole = ample
                .plan_nextn_slots(&m, CTX, &levers, &nextn, 1)
                .expect("the plan at an ample room");
            let short = HostNeed::of(&whole.plan, whole.nextn.host.expert_bytes).bytes() - 1;
            for (room, split) in [(u64::MAX, false), (short, true)] {
                let i = routed(room);
                let np = i
                    .plan_nextn_slots(&m, CTX, &levers, &nextn, 1)
                    .unwrap_or_else(|e| panic!("{room} B: {e}"));
                assert_eq!(np.plan.host.nvme_expert_bytes > 0, split, "{room} B");
                let (_, beside) = np.host_runs().expect("the draft's host runs");
                let want = i128::from(room) - i128::from(HostNeed::of(&np.plan, beside).bytes());
                let left = ChurnPool::of(&np.plan, 0, usize::MAX)
                    .and_then(|pool| pool.check_beside(&np.plan, beside))
                    .map_err(|e| e.to_string());
                assert_eq!((np.host_headroom_bytes, left), (want, Ok(want)), "{room} B");
            }
        }

        /// On one pool whose card fill leaves the host under the NVMe tier's
        /// floor, the plain plan and the plan with the next-token layer, at
        /// one and two slots, are made under a card budget lowered to the
        /// floor ([`crate::placement::plan_in_pool`]): the card keeps fewer
        /// experts, the rest page through the tier, and the budget is marked
        /// as the pool's. The same room under the card's whole bytes as a
        /// given budget is the floor's refusal, and a room under the
        /// next-token layer's routed experts is refused by name before any
        /// budget is tried.
        #[test]
        fn slot_plans_on_one_pool_lower_the_card_budget_to_the_tiers_floor() {
            let unset = PlanLevers::default();
            let whole = PlanLevers {
                card_budget_bytes: Some(u64::MAX).into(),
            };
            let nextn = nextn();
            let draft = nextn.model.tensors[1].file_bytes;
            let pool: u64 = 15 << 29;
            let lowered =
                |at: &str, plan: Result<Plan<'_>, PlaceError>, floor: &dyn Fn() -> Option<u64>| {
                    let plan = plan.unwrap_or_else(|e| panic!("{at}: {e}"));
                    assert!(plan.budget_from_pool, "{at}: the pool's budget");
                    assert!(plan.host.nvme_expert_bytes > 0, "{at}: paged experts");
                    assert!(plan.n_l.iter().any(|&n| n > 0), "{at}: card experts");
                    let budget = plan.card_budget.every().expect("a card budget");
                    assert!(budget < pool, "{at}: {budget} B, not under the pool");
                    assert!(
                        floor().is_some(),
                        "{at}: the whole budget's plan is not the floor's refusal"
                    );
                };
            let refused = |p: Result<u64, PlaceError>| match p {
                Err(PlaceError::Placement(PlacementError::HostRoomFloor {
                    room, floor, ..
                })) => {
                    assert!(floor > room);
                    Some(floor)
                }
                _ => None,
            };
            for n in [1, 2] {
                let m = unified(pool);
                let i = routed(pool);
                lowered(
                    &format!("{n} slot(s)"),
                    i.plan_slots(&m, CTX, &unset, KdaLanes::One, n),
                    &|| {
                        refused(
                            i.plan_slots(&m, CTX, &whole, KdaLanes::One, n)
                                .map(|p| p.ctx_max),
                        )
                    },
                );
                let m = unified(pool + draft);
                let i = routed(pool + draft);
                lowered(
                    &format!("{n} slot(s) with the next-token layer"),
                    i.plan_nextn_slots(&m, CTX, &unset, &nextn, n)
                        .map(|p| p.plan),
                    &|| {
                        refused(
                            i.plan_nextn_slots(&m, CTX, &whole, &nextn, n)
                                .map(|p| p.plan.ctx_max),
                        )
                    },
                );
                match routed(draft - 1).plan_nextn_slots(&m, CTX, &unset, &nextn, n) {
                    Err(PlaceError::Placement(PlacementError::DraftHostRoom {
                        room,
                        draft: d,
                    })) => assert_eq!((d, room), (draft, draft - 1), "the draft's own bytes"),
                    other => panic!(
                        "a room under the draft's experts: {:?}",
                        other.map(|p| p.plan.card_budget)
                    ),
                }
            }
        }
    }
}
