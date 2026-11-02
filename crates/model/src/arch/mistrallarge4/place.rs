//! A Mistral Large 4 file planned onto a machine: the common hybrid owner's plan
//! ([`place_hybrid`]) over this family's reading. What each layer holds is read off the
//! description ([`KvLayout::of`]): a latent row a position on each of the layers that write their
//! own cache, none on the layers that read another's (`attention.kv_source_map`), and an index key
//! a position on each layer whose indexer writes keys of its own (`attention.indexer.types`). The
//! card experts' formats are the common leg's ([`Program::MistralLarge4Body`]'s rule is the one the
//! mimo2 and solar2 plans read): a layer whose gate and up are of one type the leg runs and whose
//! down it reads over its 128-value form is a card layer, and any other layer's experts stay on the
//! host. The body's host run is the common one, so the plan pages the host experts the room cannot
//! hold ([`PlanInputs::with_nvme_tier`]), and an expert tier card holds the next ids of each card
//! layer on the common leg.
//!
//! The prompt batch's card buffers are the plan's prompt reserve ([`prompt_reserve_bytes`] at
//! [`PROMPT_ROWS`] rows, a group's units, records and lists by [`configure_group`]), kept out of
//! the expert budget; the tier's places of a block of [`PROMPT_ROWS`] rows are its reserve.
//!
//! The body's scratch is a flat term of the machine and the indexer's scores over the context: a
//! row of scores a position on the step's scratch of each full layer and on the prompt batch's
//! one ([`index_scores_row_bytes`]). The plan counts those at the context it is fitted to, with
//! the stage's position rows ([`PlanInputs::with_position_scratch`]), so a context the card holds
//! its scores beside the other terms is the one the plan serves.

use gguf::Split;
use models::{Ffn, ModelSpec};

use super::hparams::Hparams;
use super::names as stems;
use super::spec;
use crate::arch::names::{self, blk};
use crate::arch::place_hybrid;
use crate::placement::accepts::{Program, leg_down_forms};
use crate::placement::kernels::attention::Forms;
use crate::placement::kernels::ffn::{
    PromptWalk, RoutedBatch, WideBlock, WideForms, dense_route_bytes,
};
use crate::placement::workstation::{self, TierBatchBytes, TierLegScratch};
use crate::placement::{ModelTensors, PlacementError, kernels};

pub use place_hybrid::{KvLayout, PlaceError};

/// What a plan of a Mistral Large 4 file is made from, read from its headers.
pub type PlanInputs = place_hybrid::PlanInputs<Hparams>;

/// The engine the plan's refusals name.
const ENGINE: &str = "Mistral Large 4";

/// The rows of a prompt block the expert tier's service is sized for: the host union's columns.
pub const PROMPT_ROWS: usize = crate::moe::UNION_MAX_COLS;

/// Tokens a chunk of the tier's service runs at most: the card leg's launch group, the figure every
/// tier of the common leg names.
pub const TIER_CHUNK: u64 = 8;

/// Rows one attention launch of the prompt batch reads: the partials' rows, which a unit's rows
/// walk a sub-block at a time.
pub const ATTN_ROWS: usize = 8;

/// Rows one indexer launch of the prompt batch scores. Their index queries and head weights are
/// read a token per column, the score pass's layout, so the reserve holds this many rows of each
/// in that layout; their scores over the context are the plan's position scratch
/// ([`index_scores_row_bytes`]).
pub const IDX_CHUNK: usize = 16;

/// Rows the step's indexer scores a launch: one token's scores over the context, on each full
/// layer's own scratch.
pub const IDX_STEP_ROWS: usize = 1;

/// Bytes of one score, an f32.
const SCORE_BYTES: u64 = 4;

/// The bytes of indexer scores a position of the context takes on the card: one score a row of
/// each scores scratch the body makes over the context, [`IDX_STEP_ROWS`] rows on each full layer
/// (the step's) and [`IDX_CHUNK`] rows on the prompt batch's one. The plan counts them at its
/// context ([`PlanInputs::with_position_scratch`]), one copy whatever the slots; the body holds
/// its scores buffers equal to them at its load.
#[must_use]
pub fn index_scores_row_bytes(hp: &Hparams) -> u64 {
    let full = hp.indexer_full.iter().filter(|&&f| f).count() as u64;
    SCORE_BYTES * (IDX_STEP_ROWS as u64 * full + IDX_CHUNK as u64)
}

/// [`index_scores_row_bytes`] over `ctx` positions: what the plan counts for the indexer's scores
/// at a context of `ctx`.
#[must_use]
pub fn index_scores_bytes(hp: &Hparams, ctx: u64) -> u64 {
    index_scores_row_bytes(hp).saturating_mul(ctx)
}

/// The selected latent flash's segments past one segment's keys, and a segment's keys: the
/// flash kernels' own (`flash::SEGMENTS`, `seg_keys`), which the body holds to this by name.
const FLASH_SEGMENTS: u64 = 80;
const SEG_KEYS: u64 = 64;

/// The partial segments a query row of the prompt batch's attention takes over lists of `top_k`
/// rows: the wider of the roped layers' selected flash (one segment up to [`SEG_KEYS`] keys,
/// else [`FLASH_SEGMENTS`]) and the NoPE layers' walk over its one-row ring and the list (a
/// segment for the ring, one per [`SEG_KEYS`] list rows).
#[must_use]
pub const fn attn_segments(top_k: u64) -> u64 {
    let flash = if top_k <= SEG_KEYS { 1 } else { FLASH_SEGMENTS };
    let walk = 1 + top_k.div_ceil(SEG_KEYS);
    if flash > walk { flash } else { walk }
}

fn overflow() -> PlacementError {
    PlacementError::Metadata {
        key: "mistral-large4 prompt reserve".into(),
        detail: "buffer bytes overflow u64".into(),
    }
}

/// The activation form the dense site `name` reads, by its type in the description.
fn reads(model: &ModelTensors, name: &str) -> Result<kernels::SiteAct, PlacementError> {
    let tensor = model
        .tensors
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| PlacementError::Tensor {
            name: name.into(),
            detail: "is not in the description".into(),
        })?;
    kernels::dense(tensor.ty)
        .map(|site| site.reads())
        .ok_or_else(|| PlacementError::Tensor {
            name: name.into(),
            detail: format!("type {:?} has no dense projection site", tensor.ty),
        })
}

/// The prompt batch's attention buffers over `rows` rows and `sets` units of a group, from the
/// widths the body cuts them at: the rows every front shares (the norms, the projections, the
/// absorbed queries, the key path, the index heads, the values and the residual's delta), a
/// sub-block's latent outputs, their q8_1 form, its ids and partials, an indexer launch's index
/// queries and head weights a token per column, the NoPE walk's sinks and ring, each unit's
/// record (live and selected counts, walk pairs, rope rows and their bits) and list, the wide
/// forms the sites read at their widths and the one-expert route table.
pub fn attention_bytes(
    hp: &Hparams,
    model: &ModelTensors,
    rows: u64,
    sets: u64,
) -> Result<u64, PlacementError> {
    let w = |v: usize| v as u64;
    let (hidden, heads, latent, rope) =
        (w(hp.n_embd), w(hp.n_head), w(hp.kv_lora), w(hp.rope_dims));
    let (qk, hv) = (heads * w(hp.head_k), heads * w(hp.head_v));
    let (idx_q, idx_d, top_k) = (
        w(hp.indexer_heads) * w(hp.indexer_d),
        w(hp.indexer_d),
        w(hp.indexer_top_k),
    );
    let row_values = 3 * hidden
        + 2 * qk
        + heads * (latent + rope)
        + 3 * (latent + rope)
        + 2 * idx_q
        + w(hp.indexer_heads)
        + 2 * idx_d
        + hv;
    let q_rows = w(ATTN_ROWS) * heads;
    let segs = attn_segments(top_k);
    let sub_values = q_rows * latent
        + q_rows
        + q_rows * segs * (latent + 2)
        + heads
        + w(IDX_CHUNK) * (idx_q + w(hp.indexer_heads));
    let record = 4 * rows + 1 + 2 * rows * rope + rows * top_k;
    let mut input = Forms::default();
    let mut output = Forms::default();
    for l in 0..hp.n_layer {
        let mut sites = vec![blk(l, stems::ATTN_Q)];
        if hp.owns_cache(l) {
            sites.push(blk(l, stems::ATTN_KV_A_MQA));
        }
        if hp.indexer_full[l] {
            sites.push(blk(l, stems::INDEXER_ATTN_Q_B));
            sites.push(blk(l, stems::INDEXER_ATTN_K));
            sites.push(blk(l, stems::INDEXER_PROJ));
        }
        for name in sites {
            input = input.with(reads(model, &name)?);
        }
        let o = reads(model, &blk(l, stems::ATTN_OUTPUT))?;
        // The output projection reads the heads' values: at the model's width its forms are
        // the input's, at any other its own.
        if hv == hidden {
            input = input.with(o);
        } else {
            output = output.with(o);
        }
    }
    let forms = input
        .bytes(rows, hidden)
        .zip(output.bytes(rows, hv))
        .and_then(|(i, o)| i.checked_add(o));
    let v_act = Forms {
        q128: true,
        q32: false,
    }
    .bytes(q_rows, latent);
    row_values
        .checked_mul(rows)
        .and_then(|v| v.checked_add(sub_values))
        .and_then(|v| v.checked_add(sets.checked_mul(record)?))
        .and_then(|v| v.checked_mul(4))
        .and_then(|b| b.checked_add(latent.checked_mul(2)?))
        .and_then(|b| b.checked_add(forms?))
        .and_then(|b| b.checked_add(v_act?))
        .and_then(|b| b.checked_add(dense_route_bytes(rows)?))
        .ok_or_else(overflow)
}

fn block_forms(
    model: &ModelTensors,
    names: [String; 3],
    block: &mut WideBlock,
) -> Result<(), PlacementError> {
    block.input = block
        .input
        .with(reads(model, &names[0])?)
        .with(reads(model, &names[1])?);
    block.down = block.down.with(reads(model, &names[2])?);
    Ok(())
}

/// The common routed and shared block's batch over `rows` rows: the router's logits rows beside
/// its scores (the family's router keeps them), the shared expert's and the dense lead's wide
/// forms by their types, and the card leg's chunk scratch, which a load reserves whenever a card
/// serves routed experts.
pub fn routed_batch(
    spec: &ModelSpec,
    model: &ModelTensors,
    rows: u64,
) -> Result<RoutedBatch, PlacementError> {
    let empty = WideBlock {
        hidden: u64::from(spec.hidden),
        ff: 0,
        input: WideForms::default(),
        down: WideForms::default(),
    };
    let (mut shared, mut dense) = (empty, empty);
    let (mut experts, mut used, mut expert_ff) = (0, 0, 0);
    for (l, layer) in spec.layers.iter().enumerate() {
        match &layer.ffn {
            Ffn::Dense { ff, .. } => {
                dense.ff = dense.ff.max(u64::from(*ff));
                block_forms(
                    model,
                    [names::ffn_gate(l), names::ffn_up(l), names::ffn_down(l)],
                    &mut dense,
                )?;
            }
            Ffn::Moe(moe) => {
                experts = experts.max(u64::from(moe.experts));
                used = used.max(u64::from(moe.top_k));
                expert_ff = expert_ff.max(u64::from(moe.expert_ff));
                let sh = moe
                    .shared
                    .as_ref()
                    .ok_or_else(|| PlacementError::Metadata {
                        key: format!("layer {l} shared block"),
                        detail: "Mistral Large 4 has one shared expert a routed layer".into(),
                    })?;
                shared.ff = shared.ff.max(u64::from(sh.ff));
                block_forms(
                    model,
                    [
                        names::ffn_gate_shexp(l),
                        names::ffn_up_shexp(l),
                        names::ffn_down_shexp(l),
                    ],
                    &mut shared,
                )?;
            }
        }
    }
    Ok(RoutedBatch {
        hidden: u64::from(spec.hidden),
        experts,
        used,
        shared,
        dense: (dense.ff > 0).then_some(dense),
        card: Some(TierLegScratch {
            n_embd: u64::from(spec.hidden),
            ff: expert_ff,
            n_used: used,
            chunk: rows,
            downs: leg_down_forms(Program::MistralLarge4Body, model),
            input_scales: false,
        }),
        tiered: false,
        router_logits: true,
    })
}

/// The prompt batch's card bytes over `rows` rows and `sets` units of a group: the attention's
/// ([`attention_bytes`]), the common walk's units ([`PromptWalk`]) and the routed block's batch
/// ([`routed_batch`]), from the same widths and types the body allocates by.
pub fn prompt_reserve_bytes(
    hp: &Hparams,
    spec: &ModelSpec,
    model: &ModelTensors,
    rows: u64,
    sets: u64,
) -> Result<u64, PlacementError> {
    let attention = attention_bytes(hp, model, rows, sets)?;
    let walk = PromptWalk {
        hidden: u64::from(spec.hidden),
        sets,
    }
    .bytes(rows)
    .ok_or_else(overflow)?;
    let ffn = routed_batch(spec, model, rows)?
        .bytes(rows)
        .ok_or_else(overflow)?;
    attention
        .checked_add(walk)
        .and_then(|b| b.checked_add(ffn))
        .ok_or_else(overflow)
}

/// Price the prompt group `group` before the plan borrows the inputs: a group's units, records
/// and lists multiply, the rest is shared. Refused by name for a group outside 1 to the lever's
/// most; the reserve is left as it was then.
pub fn configure_group(inputs: &mut PlanInputs, group: usize) -> Result<(), PlacementError> {
    runtime::prompt::check_group(group, bloomery_levers::PREFILL_GROUP_MAX as usize).map_err(
        |error| PlacementError::Metadata {
            key: runtime::prompt::GROUP_LEVER.into(),
            detail: error.to_string(),
        },
    )?;
    inputs.prompt_reserve = prompt_reserve_bytes(
        &inputs.hp,
        &inputs.spec,
        &inputs.model,
        PROMPT_ROWS as u64,
        runtime::prompt::group_sets(group) as u64,
    )?;
    Ok(())
}

/// The expert tier's prompt-batch bytes ([`TierBatchBytes`]) for a described file: the tier card's
/// staging, the leg's chunk scratch in the down forms the file's card layers read, and the host's
/// rows and places, for blocks of [`PROMPT_ROWS`] columns. The plan reserves them on the tier card
/// and the host.
pub fn tier_batch(inputs: &PlanInputs) -> Result<TierBatchBytes, PlacementError> {
    let hp = &inputs.hp;
    workstation::try_tier_leg_batch_bytes_of(
        hp.n_embd as u64,
        hp.expert_ff as u64,
        hp.n_used as u64,
        (TIER_CHUNK, PROMPT_ROWS as u64),
        leg_down_forms(Program::MistralLarge4Body, &inputs.model),
        false,
    )
}

impl PlanInputs {
    /// [`PlanInputs::describe`], then the file refused with every feature no row of the family's
    /// program covers ([`PlacementError::Unimplemented`]).
    pub fn read(split: &Split) -> Result<PlanInputs, PlacementError> {
        PlanInputs::describe(split)?.runnable()
    }

    /// The file as it is, for a gate that plans or describes it whatever the coverage list holds,
    /// and the host's room and the arena's given bytes, a room that cannot be read refused by name.
    /// The prompt reserve is the batch of one unit: a load that runs a prompt group of more units
    /// prices it by [`configure_group`] before the plan, or the group's buffers pass the reserve
    /// and the load refuses by name.
    pub fn describe(split: &Split) -> Result<PlanInputs, PlacementError> {
        let hp = Hparams::read(split)?;
        let read = spec::read(split)?;
        let tier = place_hybrid::tier_places_bytes(PROMPT_ROWS, hp.n_used);
        let reserve = prompt_reserve_bytes(&hp, &read.spec, &read.tensors, PROMPT_ROWS as u64, 1)?;
        let scores = index_scores_row_bytes(&hp);
        Ok(PlanInputs::new(ENGINE, hp, read.tensors, read.spec)?
            .with_position_scratch(scores)
            .with_prompt_reserve(reserve)
            .with_tier_reserve(tier)
            .with_nvme_tier())
    }
}
