//! A muse-spark file planned onto a machine: the common hybrid owner's plan
//! ([`place_hybrid`]) over this family's reading, as `mimo2::place` is for MiMo.
//! The hyperparameters, every tensor's role and the typed description are read
//! here ([`PlanInputs::read`], which refuses a file with a feature no program
//! runs, every one listed); what each layer holds, the expert rule and the
//! refusals are the owner's. The family's two facts the owner takes as
//! parameters:
//!
//! - **the routed stacks' shape**: the experts run in the latent, 4096 wide in
//!   and out, three IQ2_XXS stacks a layer (`accepts::MUSESPARK_LEG`), so the rule's
//!   per-expert bytes come from the stacks' own dims and the plan needs no
//!   Muse-only sizing;
//! - **the stores' rule**: 46 sliding layers (window 2048) and 16 full ones,
//!   read from the typed description ([`KvLayout::of`]). The body keeps a plane
//!   of every position on every layer (its head-64 flash has no ring entry),
//!   so the plan counts planes ([`Rows::Plane`], [`KV_ROWS`]); the ring each
//!   sliding layer needs ([`Rows::Ring`], the owner's default) is a flash entry
//!   away, and the plan then frees the difference to the card's experts.
//!
//! The stage card runs every layer and the head, its card experts through the
//! family's `_sel` launches; a machine that names an expert tier card (plan
//! (b′): the A6000 the stage, the 3090 its tier) gives the tier the next ids of
//! each layer, which the common card leg runs (`iq2_xxs_gate_up_sel`,
//! `iq2_xxs_gemv_sel`), and the rest stays on the host tier
//! ([`PlanInputs::plan`]). The tier's prompt-batch bytes are [`tier_batch`] and
//! the stage card's tier places [`tier_places_bytes`]. The placed load's prompt
//! pass makes buffers beside the plan's terms; the plan keeps their bytes out of
//! the expert budget ([`prompt_reserve_bytes`]) by the one formula the load cuts
//! them by ([`pass_bytes`]).

use std::collections::BTreeMap;

use gguf::{GgmlType, Split};
use models::ModelSpec;

use super::hparams::Hparams;
use super::names::{
    ATTN_GATE, ATTN_K, ATTN_OUTPUT, ATTN_Q, ATTN_V, FFN_DOWN_EXPS, FFN_GATE_EXPS, FFN_LATENT_DOWN,
    FFN_LATENT_UP, FFN_UP_EXPS,
};
use super::spec;
use crate::arch::names::blk;
use crate::arch::place_hybrid::{self, RingForms, ring_pass_bytes_in};
use crate::moe::UNION_TAIL_MAX_COLS;
use crate::placement::kernels::{self, SiteAct, q8_act_bytes};
use crate::placement::workstation::{self, TierBatchBytes};
use crate::placement::{Machine, ModelTensors, PlacementError};

pub use place_hybrid::{KvLayout, PlaceError};
pub use runtime::stores::Rows;

/// What a plan of a muse-spark file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Muse Spark";

/// The rows a sliding layer's store holds in the plan: every position, as the
/// body allocates them.
pub const KV_ROWS: Rows = Rows::Plane;

/// Tokens one run of the tier's card leg covers in a prompt block: the stage
/// gemv pass the other card-leg tiers (GLM, Solar, Qwen) name too.
pub const TIER_CHUNK: u64 = 8;

/// The rows of a placed load's prompt pass at most, `n_used` routed slots a
/// row: every routed slot of a post-norm layer is a column of the host's union
/// call, so a pass of `rows` rows is `rows · n_used` columns, at most the
/// union tail's.
#[must_use]
pub const fn pass_rows_max(n_used: usize) -> usize {
    UNION_TAIL_MAX_COLS / if n_used == 0 { 1 } else { n_used }
}

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for blocks of a
/// placed pass's rows ([`pass_rows_max`]): the experts run in the latent, so
/// the tier's rows and its staging are the latent's width, through the card
/// leg in chunks of [`TIER_CHUNK`].
pub fn tier_batch(hp: &Hparams) -> Result<TierBatchBytes, PlacementError> {
    let ff = hp.expert_width();
    workstation::tier_leg_batch_bytes(
        hp.latent as u64,
        ff as u64,
        hp.n_used as u64,
        TIER_CHUNK,
        pass_rows_max(hp.n_used) as u64,
    )
}

/// The stage card of a tiered machine ([`workstation::plan_tiers`]) held as
/// the placed load holds it: the token embedding on the card, which the body
/// reads there; the tiered machine leaves it on the host. The scratch is the
/// m = 1 figure either way. A machine with no stage card is refused by name
/// ([`PlacementError::Stages`]).
pub fn stage_placed(machine: &mut Machine) -> Result<(), PlacementError> {
    let card = machine.cards.first_mut().ok_or_else(|| {
        PlacementError::Stages(
            "a machine with no stage card: the placed load reads the token embedding on it"
                .to_string(),
        )
    })?;
    card.token_embedding = true;
    Ok(())
}

/// Rows one head-64 attention launch of a pass takes: a pass runs its attention in runs of this
/// many rows. The body holds each of the constants below equal to the kernels' by assertion.
pub const PASS_ATTN_ROWS: usize = 32;
/// Words of a pass's record before its per-head row words: the first position.
pub const PASS_POS_WORDS: usize = 1;
/// Segments a windowless layer's row is cut into.
pub const FLASH_SEGMENTS: usize = 80;
/// Keys one segment of a windowed layer covers.
pub const SEGMENT_KEYS: usize = 64;
/// Columns one tile of the grouped GEMM holds.
pub const GEMM_COLS: usize = 64;
/// Values one step of the 32-value GEMM family's walk covers: a column of its form is padded to
/// whole steps.
pub const GEMM_STEP: usize = 64;

/// Segments a head-64 launch with `window` cuts a row into: [`FLASH_SEGMENTS`] without a window,
/// else whole [`SEGMENT_KEYS`]-key segments of the window.
const fn window_segments(window: usize) -> usize {
    if window == 0 {
        FLASH_SEGMENTS
    } else {
        window.div_ceil(SEGMENT_KEYS)
    }
}

/// The quantized forms one wide input width of a pass is held in ([`PassShape::wide`]): q8_1
/// blocks of 128 values for a K-quant site, q8 blocks of 32 for a Q8_0 or Q5_0 one. A site that
/// reads f32 rows holds none: the input's f32 rows are the caller's own buffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassForms {
    pub q128: bool,
    pub q32: bool,
}

impl PassForms {
    /// The forms either of `self` and `other` holds.
    #[must_use]
    pub const fn union(self, other: PassForms) -> PassForms {
        PassForms {
            q128: self.q128 | other.q128,
            q32: self.q32 | other.q32,
        }
    }

    /// The forms a projection site of file type `ty` reads (the kernel table's activation form of
    /// its launch); `None` for a type no site entry runs.
    fn of(ty: GgmlType) -> Option<PassForms> {
        let reads = kernels::dense(ty)?.reads();
        Some(PassForms {
            q128: reads == SiteAct::Q8x128,
            q32: matches!(reads, SiteAct::Q8x32 | SiteAct::Q8Blocks32),
        })
    }
}

/// Every width and form a placed load's prompt pass cuts its buffers from: the widths every layer
/// shares, the segments the widest cut of a row takes and, per input width of the sites, the forms
/// its sites read. The load collects the same from its layers, and the plan from the headers
/// ([`pass_shape`]); the body refuses a load whose shape is not the plan's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassShape {
    /// The hidden row.
    pub hidden: usize,
    /// The head width (the flash instance's).
    pub head: usize,
    pub n_head: usize,
    pub n_kv: usize,
    /// The latent the experts run in, the experts, those a token uses and one expert's width.
    pub latent: usize,
    pub n_expert: usize,
    pub top_k: usize,
    pub ff: usize,
    /// The segments the widest cut of a row takes: a windowless layer's [`FLASH_SEGMENTS`] or a
    /// window's whole segments.
    pub segs: usize,
    /// The forms each site input width is held in, by width.
    pub wide: BTreeMap<usize, PassForms>,
}

/// The card bytes of the buffers of a pass of `rows` rows over `s`, four bytes a value:
///
/// - f32 rows, per row: eight hidden (the layer's input and output rows, the rows between its
///   halves, the normed row, the delta, the post-norm row and three tiles), four query-wide (the
///   query, its norm, the gate and the attention), five key-wide (the key, its second form, the
///   value and the two head-major regroupings), four latent (the down's rows, their norm, the
///   sum and the host's sums), the router's scores, the top-k weights, the slots' up rows
///   (`top_k · ff`), two slots' latent rows (`top_k · latent` each) and the rope row; and once, a
///   run of [`PASS_ATTN_ROWS`] rows' query and output rows and the run's partials (`head + 2`
///   values a row, head and segment);
/// - u32 words: the ids and the slots' places, the record (the position word, a row word a key
///   head, the ids), the positions and live key counts and the run's live counts;
/// - the forms: the latent rows' q8_1, the slots' up rows' q8_1, each site width's wide forms
///   ([`PassShape::wide`]) and the grouped GEMM's one-expert table.
///
/// The one formula of those bytes: the plan keeps them out of the expert budget
/// ([`prompt_reserve_bytes`]) and the load's fit check and allocation count them by it.
#[must_use]
pub fn pass_bytes(s: &PassShape, rows: usize) -> u64 {
    let (f, u) = (size_of::<f32>(), size_of::<u32>());
    let q_width = s.n_head * s.head;
    let kv_width = s.n_kv * s.head;
    let f32s = rows
        * (8 * s.hidden
            + 4 * q_width
            + 5 * kv_width
            + 4 * s.latent
            + s.n_expert
            + s.top_k
            + s.top_k * s.ff
            + 2 * s.top_k * s.latent
            + s.head)
        + 2 * PASS_ATTN_ROWS * q_width
        + PASS_ATTN_ROWS * s.n_head * s.segs * (s.head + 2);
    let u32s = 2 * rows * s.top_k + PASS_POS_WORDS + s.n_kv + rows + 2 * rows + PASS_ATTN_ROWS;
    let act = |k: usize| q8_act_bytes(1, k as u64) as usize;
    let act32 = |k: usize| k.div_ceil(GEMM_STEP) * (16 + 2 + 2) * u;
    let wide: usize = s
        .wide
        .iter()
        .map(|(&k, w)| rows * (usize::from(w.q128) * act(k) + usize::from(w.q32) * act32(k)))
        .sum();
    // The one-expert table: the slots, two words a tile (a tile every GEMM_COLS slots, and one
    // more), its count words and the all-zero ids a column each.
    let tiles = rows / GEMM_COLS + rows.min(1);
    let route = (rows + 2 * tiles + 2 + rows) * u;
    let forms = rows * act(s.latent) + s.top_k * rows * act(s.ff) + wide + route;
    (f32s * f + u32s * u + forms) as u64
}

/// The shape of the prompt pass of the file `hp` and `model` describe, read from the headers
/// alone: the widths of the hyperparameters, the segments of the layers' windows and, per
/// input width of the seven projection sites a layer reads (the query, key, value, gate and
/// output, the latent down and up), the forms their file types read. A site tensor missing or of
/// a type no site entry runs is refused by name.
pub fn pass_shape(hp: &Hparams, model: &ModelTensors) -> Result<PassShape, PlacementError> {
    let ff = hp
        .expert_ff
        .first()
        .copied()
        .ok_or_else(|| PlacementError::Metadata {
            key: "expert_feed_forward_length".to_string(),
            detail: "names no layer's width".to_string(),
        })?;
    let (hidden, q_width) = (hp.n_embd, hp.n_head * hp.head_dim);
    let sites = [
        (ATTN_Q, hidden),
        (ATTN_K, hidden),
        (ATTN_V, hidden),
        (ATTN_GATE, hidden),
        (ATTN_OUTPUT, q_width),
        (FFN_LATENT_DOWN, hidden),
        (FFN_LATENT_UP, hp.latent),
    ];
    let mut wide: BTreeMap<usize, PassForms> = BTreeMap::new();
    for l in 0..hp.n_layer {
        for (stem, k) in sites {
            let name = blk(l, stem);
            let tensor = model
                .tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| PlacementError::Tensor {
                    name: name.clone(),
                    detail: "is not in the file: a pass's projection reads it".to_string(),
                })?;
            let forms = PassForms::of(tensor.ty).ok_or_else(|| PlacementError::Tensor {
                name,
                detail: format!(
                    "is {:?}: no projection site entry runs that type",
                    tensor.ty
                ),
            })?;
            let slot = wide.entry(k).or_default();
            *slot = slot.union(forms);
        }
    }
    let segs = hp
        .sliding
        .iter()
        .map(|&sliding| window_segments(if sliding { hp.window } else { 0 }))
        .max()
        .unwrap_or_else(|| window_segments(0));
    Ok(PassShape {
        hidden,
        head: hp.head_dim,
        n_head: hp.n_head,
        n_kv: hp.n_head_kv,
        latent: hp.latent,
        n_expert: hp.n_expert,
        top_k: hp.n_used,
        ff,
        segs,
        wide,
    })
}

/// The card bytes the plan keeps out of the expert budget for a placed load's prompt pass:
/// [`pass_bytes`] of the file's [`pass_shape`] at [`pass_rows_max`] rows, the most a pass takes,
/// whatever the context (a load takes fewer rows only where the card's room is short of them,
/// which this reserve keeps from happening).
pub fn prompt_reserve_bytes(hp: &Hparams, model: &ModelTensors) -> Result<u64, PlacementError> {
    Ok(pass_bytes(
        &pass_shape(hp, model)?,
        pass_rows_max(hp.n_used),
    ))
}

/// The stage card's bytes a tiered load makes beside the plan's terms: the
/// slot map's tier view (a place per expert per layer) and the tier places of
/// the step's slots and of a pass's ([`pass_rows_max`] rows of `n_used`), four
/// bytes each.
#[must_use]
pub fn tier_places_bytes(hp: &Hparams) -> u64 {
    4 * (hp.n_layer * hp.n_expert + (1 + pass_rows_max(hp.n_used)) * hp.n_used) as u64
}

/// The slots of each half of the expert stream's ring a streaming plan keeps
/// room for ([`stream_ring_bytes`]) [derived: at a pass of [`pass_rows_max`]
/// rows a hosted layer's host experts carry about sixteen columns each, and the
/// stream's rule moves experts until the layer's copies and card columns match
/// the union the host keeps, a little over half of a stage card's host experts
/// at plan (a); the half is the power of two above that].
pub const STREAM_HALF_SLOTS: u64 = 128;

/// The card bytes the prompt calls' expert stream takes (`BLOOMERY_XSTREAM=split`):
/// two halves of [`STREAM_HALF_SLOTS`] slots, a slot the widest layer's expert
/// at its three routed stacks' file bytes per expert (the ring holds the file's
/// blocks). The ring sizes itself from the card's free bytes past the stream's
/// keep-free, which the card's margin holds; a plan that keeps these bytes out
/// of the expert budget leaves them free for it
/// ([`PlanInputs::streaming`](place_hybrid::PlanInputs::streaming)).
/// Refused by name: a layer's routed stack missing, or one that is not a stack
/// of the file's experts.
pub fn stream_ring_bytes(hp: &Hparams, model: &ModelTensors) -> Result<u64, PlacementError> {
    place_hybrid::stream_ring_bytes(
        model,
        |l| [FFN_GATE_EXPS, FFN_UP_EXPS, FFN_DOWN_EXPS].map(|stem| blk(l, stem)),
        (hp.n_expert as u64, STREAM_HALF_SLOTS),
        0,
    )
}

/// The card bytes a streaming plan keeps out of the expert budget (`BLOOMERY_XSTREAM=split`): the
/// ring ([`stream_ring_bytes`]) beside the ring pass's route GEMM buffers at [`pass_rows_max`] rows
/// over a half of [`STREAM_HALF_SLOTS`] ([`ring_pass_bytes_in`], in the forms the file's routed
/// stacks read, [`RingForms::of`]). Refused by name: as [`stream_ring_bytes`] refuses, a model with
/// no expert width, and routed stacks of a type no route GEMM reads.
pub fn stream_bytes(hp: &Hparams, model: &ModelTensors) -> Result<u64, PlacementError> {
    let ring = stream_ring_bytes(hp, model)?;
    let ff = hp
        .expert_ff
        .first()
        .copied()
        .ok_or_else(|| PlacementError::Metadata {
            key: "expert_feed_forward_length".to_string(),
            detail: "names no layer's width".to_string(),
        })?;
    let ty = |stem: &str| {
        let name = blk(0, stem);
        model
            .tensors
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.ty)
            .ok_or_else(|| PlacementError::Tensor {
                name,
                detail: "is not in the file: the ring pass reads the routed stacks".to_string(),
            })
    };
    let (gate, down) = (ty(FFN_GATE_EXPS)?, ty(FFN_DOWN_EXPS)?);
    let forms = RingForms::of(gate, down).ok_or_else(|| PlacementError::Tensor {
        name: blk(0, FFN_GATE_EXPS),
        detail: format!("is {gate:?} beside a {down:?} down: no route GEMM reads the pair"),
    })?;
    let pass = ring_pass_bytes_in(
        pass_rows_max(hp.n_used),
        hp.n_used,
        hp.latent,
        ff,
        STREAM_HALF_SLOTS as usize,
        forms,
    );
    Ok(ring + pass as u64)
}

impl PlanInputs {
    /// `split`'s hyperparameters, then its tensors' roles, then its
    /// description, the first that fails being the error; then a file with a
    /// feature the engine does not run ([`PlacementError::Unimplemented`],
    /// every one listed).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// [`PlanInputs::read`] without the refusals: the file as it is, for a gate
    /// that plans or describes a file whatever the coverage list holds, and the
    /// host's room and the arena's given bytes, a room that cannot be read
    /// refused by name.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        inputs_of(hp, read.tensors, read.spec)
    }
}

/// The plan inputs of a file read as `hp`, `model` and `spec`: the prompt pass's buffers
/// ([`prompt_reserve_bytes`]) and the tier's places ([`tier_places_bytes`]) as the card's reserves,
/// and the stores as the body keeps them ([`KV_ROWS`]). The body's host run is the common one, so
/// the plan pages the host experts the room cannot hold ([`PlanInputs::with_nvme_tier`]).
fn inputs_of(
    hp: Hparams,
    model: ModelTensors,
    spec: ModelSpec,
) -> Result<PlanInputs, PlacementError> {
    let reserve = prompt_reserve_bytes(&hp, &model)?;
    let tier = tier_places_bytes(&hp);
    let mut inputs = PlanInputs::new(ENGINE, hp, model, spec)?
        .with_prompt_reserve(reserve)
        .with_tier_reserve(tier)
        .with_nvme_tier();
    inputs.kv = inputs.kv.with_rows(KV_ROWS);
    Ok(inputs)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::num::NonZeroU64;

    use gguf::GgmlType::{IQ2_XXS, Q6_K, Q8_0};
    use models::ChatSpec;

    use super::super::hparams::Hparams;
    use super::super::names::{
        ATTN_GATE, ATTN_K, ATTN_OUTPUT, ATTN_Q, ATTN_V, FFN_DOWN_EXPS, FFN_GATE_EXPS,
        FFN_LATENT_DOWN, FFN_LATENT_UP, FFN_UP_EXPS,
    };
    use super::super::spec::spec_of;
    use super::{
        PassForms, PassShape, PlanInputs, inputs_of, pass_bytes, pass_rows_max, pass_shape,
        prompt_reserve_bytes, stage_placed,
    };
    use crate::arch::names::blk;
    use crate::placement::workstation::HostRead;
    use crate::placement::{
        Card, Host, Machine, ModelTensor, ModelTensors, Node, PlacementError, PlanLevers, Role,
        accepts,
    };

    const LAYERS: usize = 4;
    const CTX: u64 = 256;

    /// The hyperparameters of `LAYERS` layers of 16 query heads over 2 key heads of 64, three
    /// sliding layers (window 2048) and a full one, eight of `experts` routed a token.
    fn hparams(embd: usize, latent: usize, ff: usize, experts: usize) -> Hparams {
        Hparams {
            n_layer: LAYERS,
            n_embd: embd,
            n_ctx: 4096,
            n_vocab: 1024,
            rms_eps: 1e-5,
            post_eps: 1e-8,
            n_head: 16,
            n_head_kv: 2,
            head_dim: 64,
            rope_dims: 64,
            rope_base: 5e5,
            window: 2048,
            sliding: vec![true, true, true, false],
            n_expert: experts,
            n_used: 8,
            expert_ff: vec![ff; LAYERS],
            latent,
            weights_norm: true,
            weights_scale: 1.0,
            logit_scale: 0.176_776_7,
            softcap: 20.0,
            defaults: Vec::new(),
        }
    }

    /// Every layer's seven projection sites at the types the real file's carry (Q6_K query and
    /// output, Q8_0 the rest) and its three IQ2_XXS routed stacks.
    fn tensors(hp: &Hparams) -> ModelTensors {
        let (e, lat, ff, x) = (
            hp.n_embd as u64,
            hp.latent as u64,
            hp.expert_ff[0] as u64,
            hp.n_expert as u64,
        );
        let (q, kv) = (
            (hp.n_head * hp.head_dim) as u64,
            (hp.n_head_kv * hp.head_dim) as u64,
        );
        let mut tensors = Vec::new();
        for l in 0..hp.n_layer {
            for (stem, role, ty, dims) in [
                (ATTN_Q, Role::Attention, Q6_K, vec![e, q]),
                (ATTN_K, Role::Attention, Q8_0, vec![e, kv]),
                (ATTN_V, Role::Attention, Q8_0, vec![e, kv]),
                (ATTN_GATE, Role::Attention, Q8_0, vec![e, q]),
                (ATTN_OUTPUT, Role::Attention, Q6_K, vec![q, e]),
                (FFN_LATENT_DOWN, Role::Router, Q8_0, vec![e, lat]),
                (FFN_LATENT_UP, Role::Router, Q8_0, vec![lat, e]),
                (
                    FFN_GATE_EXPS,
                    Role::RoutedExperts,
                    IQ2_XXS,
                    vec![lat, ff, x],
                ),
                (FFN_UP_EXPS, Role::RoutedExperts, IQ2_XXS, vec![lat, ff, x]),
                (
                    FFN_DOWN_EXPS,
                    Role::RoutedExperts,
                    IQ2_XXS,
                    vec![ff, lat, x],
                ),
            ] {
                let file_bytes = if role == Role::RoutedExperts {
                    dims.iter().product::<u64>() / 256 * 66
                } else {
                    0
                };
                tensors.push(ModelTensor {
                    name: blk(l, stem),
                    shard: 0,
                    layer: Some(l),
                    role,
                    ty,
                    dims,
                    file_bytes,
                    gathered_rows: None,
                });
            }
        }
        ModelTensors {
            tensors,
            layers: hp.n_layer,
            experts: hp.n_expert as u64,
            experts_used: hp.n_used as u64,
            bf16_words: |_| false,
        }
    }

    /// The pass's shape of the small file: hidden 256, 16 heads of 64 (1,024 wide), 2 key heads,
    /// latent 512, 16 experts of 256 rows, eight a token. The window layers cut a row into 2048 / 64
    /// = 32 segments, the full layer into 80: the larger. The query and output sites are Q6_K (q8_1
    /// blocks of 128), the key, value, gate and latent pair Q8_0 (q8 blocks of 32): the hidden
    /// width reads both forms, the output's 1,024 the first, the latent up's 512 the second.
    fn small() -> (Hparams, PassShape) {
        let hp = hparams(256, 512, 256, 16);
        let shape = PassShape {
            hidden: 256,
            head: 64,
            n_head: 16,
            n_kv: 2,
            latent: 512,
            n_expert: 16,
            top_k: 8,
            ff: 256,
            segs: 80,
            wide: BTreeMap::from([
                (
                    256,
                    PassForms {
                        q128: true,
                        q32: true,
                    },
                ),
                (
                    512,
                    PassForms {
                        q128: false,
                        q32: true,
                    },
                ),
                (
                    1024,
                    PassForms {
                        q128: true,
                        q32: false,
                    },
                ),
            ]),
        };
        (hp, shape)
    }

    /// The shape follows the headers: the widths of the hyperparameters, the larger segment cut, and
    /// the forms of each site width by its tensors' types; a layer missing a site tensor is refused
    /// by name.
    #[test]
    fn the_pass_shape_follows_the_headers() {
        let (hp, shape) = small();
        let mut model = tensors(&hp);
        assert_eq!(pass_shape(&hp, &model).expect("the small file"), shape);
        let name = blk(2, ATTN_V);
        model.tensors.retain(|t| t.name != name);
        match pass_shape(&hp, &model) {
            Err(PlacementError::Tensor { name: n, .. }) => assert_eq!(n, name),
            other => panic!("{other:?}, not the missing tensor's refusal"),
        }
    }

    /// A pass of 64 rows of the small shape holds [derived by hand from the buffers' widths]:
    /// f32 per row 8·256 + 4·1,024 + 5·128 + 4·512 + 16 + 8 + 8·256 + 2·8·512 + 64 = 19,160, times
    /// 64 rows, then the run's 2·32·1,024 and its partials 32·16·80·66 = 2,703,360, four bytes
    /// each; u32 words 2·64·8 + 1 + 2 + 64 + 2·64 + 32 = 1,251, four bytes each; the q8_1 forms
    /// 64 · 2,128 (the latent rows), 8 · 64 · 2,088 (the slots' up rows), the wide forms 64 ·
    /// (2,088 + 320) at 256, 64 · 640 at 512 and 64 · 3,232 at 1,024, and the one-expert table
    /// (64 + 2 · 2 + 2 + 64) · 4 = 536. Mutant: a term of the formula dropped.
    #[test]
    fn a_pass_holds_the_buffers_its_widths_name() {
        let (_, shape) = small();
        let f32s = 64 * 19_160 + 2 * 32 * 1024 + 2_703_360;
        let forms = 64 * 2128 + 8 * 64 * 2088 + 64 * (2088 + 320) + 64 * 640 + 64 * 3232 + 536;
        assert_eq!(pass_bytes(&shape, 64), 4 * f32s + 4 * 1251 + forms);
        assert_eq!(pass_bytes(&shape, 64), 17_593_252);
    }

    /// The machine of one card of `usable` bytes at a one-byte allocator granule, so an expert
    /// costs its bytes exactly, and a host with no bound.
    fn machine(usable: u64) -> Machine {
        Machine {
            cards: vec![Card {
                name: "card".to_string(),
                device: None,
                usable_bytes: usable,
                context_bytes: 0,
                scratch_bytes: 0,
                margin_bytes: 0,
                granule_bytes: NonZeroU64::new(1).expect("one byte"),
                free_bytes: None,
                held_by: None,
                layers: 0..LAYERS,
                head: true,
                token_embedding: false,
                reserve_only: false,
                tier_leg: None,
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

    /// The experts a plan puts on a card of `usable` bytes; `None` for a plan the card cannot
    /// hold.
    fn on_card(inputs: &PlanInputs, usable: u64) -> Option<u64> {
        let machine = machine(usable);
        let plan = inputs.plan(&machine, CTX, &PlanLevers::default());
        plan.ok().map(|p| p.n_l.iter().sum())
    }

    /// The smallest card bytes whose plan holds `n` experts or more.
    fn card_for(inputs: &PlanInputs, n: u64) -> u64 {
        let (mut lo, mut hi) = (0u64, 1u64 << 40);
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if on_card(inputs, mid).is_some_and(|c| c >= n) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        hi
    }

    /// The plan's reserve for the prompt pass is the load's formula over the file's shape at the
    /// most rows a pass takes, and the experts it leaves on the card are the ones the rest of the
    /// card holds. The file has 4 layers of 64 experts of 1,024 rows over a 1,024-value latent: an
    /// expert's three IQ2_XXS stacks take `e` bytes on the card (the stacks' own buffers, 811,008
    /// B), the allocator granule is one byte, so the experts a card holds are `(usable - fixed) / e`
    /// and a card sized to hold 252 with half an expert over holds `(252 · e + e / 2 - reserve) / e`
    /// with the reserve. Mutant: the reserve 0 (the plan keeps nothing for the pass: the figure
    /// differs from the formula's and all 252 experts stay).
    #[test]
    fn the_plan_keeps_the_passes_buffers_out_of_the_expert_budget() {
        let hp = hparams(2048, 1024, 1024, 64);
        let model = tensors(&hp);
        let reserve = pass_bytes(
            &pass_shape(&hp, &model).expect("the shape"),
            pass_rows_max(hp.n_used),
        );
        assert_eq!(
            prompt_reserve_bytes(&hp, &model).expect("the reserve"),
            reserve
        );
        let spec = spec_of(
            &hp,
            ChatSpec {
                pre: String::new(),
                template: None,
                tools: None,
                reasoning: None,
            },
        )
        .expect("the description");
        let mut inputs = inputs_of(hp.clone(), model, spec).expect("the inputs");
        inputs.room = (u64::MAX, HostRead::Given);
        inputs.arena = None;
        assert_eq!(inputs.prompt_reserve, reserve);
        // One expert's bytes on the card: its three stacks' rows at the card format.
        let e: u64 = [
            (hp.latent, hp.expert_ff[0]),
            (hp.latent, hp.expert_ff[0]),
            (hp.expert_ff[0], hp.latent),
        ]
        .into_iter()
        .map(|(k, rows)| {
            accepts::leg_card_format(IQ2_XXS)
                .buffer_bytes(IQ2_XXS, k as u64, rows as u64)
                .expect("a card buffer of the expert's rows")
                .iter()
                .sum::<u64>()
        })
        .sum();
        let held = (LAYERS * hp.n_expert) as u64 - 4;
        assert!(
            reserve < held * e,
            "the reserve {reserve} B passes {held} experts of {e} B"
        );
        let mut bare = inputs.clone();
        bare.prompt_reserve = 0;
        // The card that holds `held` experts with half of one to spare, plan without the reserve.
        let usable = card_for(&bare, held) + e / 2;
        let card = machine(usable);
        let plan = bare
            .plan(&card, CTX, &PlanLevers::default())
            .expect("the plan without the reserve");
        let n0: u64 = plan.n_l.iter().sum();
        assert_eq!(n0, held);
        assert_eq!(plan.cards[0].expert_bytes, n0 * e);
        let expected = (held * e + e / 2 - reserve) / e;
        assert!(expected < held);
        assert_eq!(on_card(&inputs, usable), Some(expected));
    }

    /// A machine with no stage card is refused by name where the placed load's embedding would
    /// be held, not left to a panic.
    #[test]
    fn stage_placed_refuses_a_machine_with_no_stage_card() {
        let mut machine = machine(1 << 30);
        stage_placed(&mut machine).expect("the stage card holds the embedding");
        assert!(machine.cards[0].token_embedding);
        machine.cards.clear();
        match stage_placed(&mut machine) {
            Err(PlacementError::Stages(why)) => assert!(why.contains("no stage card"), "{why}"),
            other => panic!("{other:?}, not the stage-card refusal"),
        }
    }
}
