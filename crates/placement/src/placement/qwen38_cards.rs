//! Qwen3.8-Flash-Next (`qwen4exp`) planned on cards larger than this
//! workstation's: the real planner over the file's own tensor table
//! (`tests/qwen38_q4_tensors.tsv`), with the machine and cache terms of
//! `bloomery-model`'s `arch::qwen35moe::place` restated here — that crate
//! does not build on the Mac. The restated terms are held to the box gate's
//! pins of the same file (`crates/model/tests/qwen4exp_meta.rs`) on the
//! A6000 first, so a term restated wrong fails here before any large-card
//! figure is read.

use bloomery_levers::{
    Residency38Pick, Residency38Why, UnhostedPool, residency38_at_plan, residency38_least_fewest,
};
use gguf::GgmlType;

use super::churn::ChurnPool;
use super::ctx_ask::{Probe, card_experts_floor, short_of};
use super::workstation::{
    A6000, ALIASES, CONTEXT, CONTEXT_SELF, CardSpec, DeviceInfo, HOST_DRIFT, HostNeed, HostRead,
    MARGIN, MIB, PLAN_ROOM, ROW_PAGE, RTX_3090, SCRATCH, TIER_BATCH_HOST_RESERVE,
    TIER_BATCH_RESERVE, census_usable, host, plan_room, resolve, tier_batch_host_bytes,
    tier_batch_staging_bytes,
};
use super::{
    ARENA_ALIGN, ARENA_LANE_SLOTS, ArenaDial, ArenaFloor, Card, Device, KvBytes, Machine,
    ModelTensor, ModelTensors, Node, PlacementError, Plan, PlanLevers, Role, UnifiedPool,
    arena_part_bytes, arena_window_bytes, nvme_arena_of, plan_routed_reserving,
};
use crate::accepts::{Program, routed_rule};
use crate::slots::{SeqTerms, Stores};

/// The UD-Q4_K_XL file's tensors, in its shards' order.
const Q4_TABLE: &str = include_str!("../../tests/qwen38_q4_tensors.tsv");

const LAYERS: usize = 48;
const EXPERTS: u64 = 512;

/// The tester's card as its driver names it [assumed: `nvidia-smi`'s
/// product name], and its `cuDeviceTotalMem` [assumed: a 97,887 MiB total
/// less the 548 MiB reserve measured on the A6000]. No figure of it is
/// measured, so it takes the census path ([`super::devices::spec_of_device`]).
const NAME_96: &str = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition";
const TOTAL_96: u64 = 97_339 * MIB;

/// `n` such cards, by ordinal, each read idle: its free bytes the usable
/// bytes less the primary context's own creation, as an idle A6000's census
/// reads ([`CONTEXT_SELF`]), which the plan's cap adds back.
fn census_96(n: usize) -> Vec<DeviceInfo> {
    (0..n)
        .map(|i| DeviceInfo {
            ordinal: u32::try_from(i).expect("small"),
            name: NAME_96.to_string(),
            total_bytes: TOTAL_96,
            free_bytes: census_usable(TOTAL_96) - CONTEXT_SELF,
            uuid: [u8::try_from(i).expect("small") + 0x30; 16],
            pci_bus: format!("0000:{:02x}:00.0", 0x41 + i),
            held_by: None,
            compute_capability: None,
            granule_bytes: None,
            integrated: false,
        })
        .collect()
}

/// The cards an alias resolves to on `census`.
fn picked(word: &str, census: &[DeviceInfo]) -> Vec<CardSpec> {
    let (_, picks) = ALIASES.iter().find(|(w, _)| *w == word).expect("an alias");
    resolve(picks, census).expect("resolved")
}

/// A qwen4exp tensor's role and layer by name (`qwen35moe::roles`).
fn role(name: &str) -> (Role, Option<usize>) {
    match name {
        "token_embd.weight" => return (Role::TokenEmbedding, None),
        "output.weight"
        | "output_hc_norm.weight"
        | "output_hc_down.weight"
        | "output_hc_up.weight" => return (Role::Head, None),
        // The PLE table belongs to the PLE site's layer.
        "per_layer_token_embd.weight" => return (Role::EngramTable, Some(1)),
        _ => {}
    }
    let (layer, stem) = name
        .strip_prefix("blk.")
        .and_then(|n| n.split_once('.'))
        .unwrap_or_else(|| panic!("{name}: no role"));
    let layer: usize = layer.parse().expect("a layer");
    let role = match stem {
        "hc_attn_norm.weight"
        | "hc_attn_down.weight"
        | "hc_attn_up.weight"
        | "hc_attn_inject.weight"
        | "hc_ffn_norm.weight"
        | "hc_ffn_down.weight"
        | "hc_ffn_up.weight"
        | "hc_ffn_inject.weight" => Role::HyperConnection,
        "ffn_gate_inp.weight" => Role::Router,
        "ffn_gate_inp_shexp.weight"
        | "ffn_gate_shexp.weight"
        | "ffn_up_shexp.weight"
        | "ffn_down_shexp.weight" => Role::SharedExpert,
        "ffn_gate_exps.weight" | "ffn_up_exps.weight" | "ffn_down_exps.weight" => {
            Role::RoutedExperts
        }
        "ple_key.weight" | "ple_value.weight" => Role::EngramDense,
        "ple_norm_key.weight"
        | "ple_norm_query.weight"
        | "ple_norm_conv.weight"
        | "ple_conv1d.weight" => Role::EngramGain,
        "attn_qkv.weight"
        | "attn_gate.weight"
        | "ssm_conv1d.weight"
        | "ssm_dt.bias"
        | "ssm_a"
        | "ssm_alpha.weight"
        | "ssm_beta.weight"
        | "ssm_norm.weight"
        | "ssm_out.weight"
        | "attn_q.weight"
        | "attn_k.weight"
        | "attn_v.weight"
        | "attn_q_norm.weight"
        | "attn_k_norm.weight"
        | "attn_output.weight"
        | "indexer.q_proj.weight"
        | "indexer.k_proj.weight"
        | "indexer.q_norm.weight"
        | "indexer.k_norm.weight" => Role::Attention,
        other => panic!("{name}: no role for {other}"),
    };
    (role, Some(layer))
}

/// The UD-Q3_K_XL type of a Q4 tensor: the routed gate and up IQ3_XXS
/// (IQ4_XS where the Q4 file has Q5_K), the Q5_1 downs IQ4_NL, the head
/// Q6_K; every other tensor as in the Q4 file. The Q3 file's own header
/// lists exactly these, in the same order and shapes.
fn q3_type(name: &str, ty: GgmlType) -> GgmlType {
    let gate_up = name.ends_with("ffn_gate_exps.weight") || name.ends_with("ffn_up_exps.weight");
    match ty {
        GgmlType::Q4_K if gate_up => GgmlType::IQ3_XXS,
        GgmlType::Q5_K if gate_up => GgmlType::IQ4_XS,
        GgmlType::Q5_1 if name.ends_with("ffn_down_exps.weight") => GgmlType::IQ4_NL,
        GgmlType::Q8_0 if name == "output.weight" => GgmlType::Q6_K,
        t => t,
    }
}

/// The file's tensors as the plan reads them: the Q4 file's table, or under
/// `q3` the Q3 file's types with each tensor's bytes from its type's blocks.
pub(super) fn model(q3: bool) -> ModelTensors {
    let tensors = Q4_TABLE
        .lines()
        .filter(|l| !l.starts_with('#'))
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            let [name, ty, dims, bytes] = f[..] else {
                panic!("{l}: not four columns");
            };
            let dims: Vec<u64> = dims.split(',').map(|d| d.parse().expect("dim")).collect();
            let mut ty = GgmlType::from_u32(ty.parse().expect("a type id"));
            let mut file_bytes: u64 = bytes.parse().expect("bytes");
            if q3 {
                ty = q3_type(name, ty);
                let values: u64 = dims.iter().product();
                file_bytes = values / ty.blck_size().expect("a block")
                    * ty.type_size().expect("a block's bytes");
            }
            let (role, layer) = role(name);
            ModelTensor {
                name: name.to_string(),
                shard: 0,
                layer,
                role,
                ty,
                dims,
                file_bytes,
                gathered_rows: match role {
                    Role::TokenEmbedding => Some(1),
                    // (ngram 3 − 1) · 8 heads an n-gram.
                    Role::EngramTable => Some(16),
                    _ => None,
                },
            }
        })
        .collect();
    ModelTensors {
        tensors,
        layers: LAYERS,
        experts: EXPERTS,
        experts_used: 10,
        bf16_words: indexer_projection,
    }
}

/// The file's BF16 matrices, the 24 indexer projections of the 12 selecting
/// layers, which the card keeps in their words (`qwen35moe::roles`'s
/// `bf16_words` names these among the stems `Body38` reads as BF16).
fn indexer_projection(t: &ModelTensor) -> bool {
    t.name.ends_with(".indexer.q_proj.weight") || t.name.ends_with(".indexer.k_proj.weight")
}

/// The file's cache bytes a layer (`qwen35moe::place::KvLayout`) [derived
/// from `runtime::stores`]: a delta layer's state in one lane of 48 heads
/// of 128 × 128 f32 with its u32 stamp, its verify record of
/// `DELTA_VERIFY_ROWS` rows of 48 · 3 · 128 f32, its conv ring of 11 rows
/// of 10,240 f32 and its carry of 132,612 f32, 4,421,652 B; the PLE layer's
/// ring beside it, 17 rows of 4 · 2,560 f32; a selecting layer, every
/// fourth, its K and V, 2,048 B a position, a pooled key, 256 B a pool of
/// four, and its raw index key ring of 11 rows of 128 f16, 2,816 B; and
/// beside the cache the stage's rope rows, the file's 64 rope dims of f32 a
/// position.
struct Kv38;

impl KvBytes for Kv38 {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let ple = if layer == 1 { 17 * 4 * 2560 * 4 } else { 0 };
        let own = if layer % 4 == 3 {
            ctx_max * 2048 + ctx_max.div_ceil(4) * 256 + 11 * 128 * 2
        } else {
            4_421_652
        };
        own + ple
    }

    fn stage_position_bytes(&self, ctx_max: u64) -> u64 {
        ctx_max * 64 * 4
    }
}

/// A sequence's card bytes beside its stores
/// (`qwen35moe::place::PlanInputs::seq_terms`): (1 + 8) rows of 4 · 2,560 f32
/// and a lane word, plus the replay table's row for each delta layer.
const BESIDE: u64 = ROWS + 4 + 36 * runtime::stores::DELTA_REPLAY_BYTES;
/// A sequence's step and pass arena rows.
const ROWS: u64 = 9 * 4 * 2560 * 4;

/// The most host bytes one saved sequence state of `ctx` positions holds, the
/// prompt cache's entry (`qwen35moe::place::PlanInputs::seq_state_bytes`,
/// which a box gate holds equal to the engine's `seq38_bytes`) [derived]:
/// the 12 selecting layers' positional bytes (2,048 B a position and a
/// 256 B pooled key a pool of four), a context- and budget-bounded number
/// of checkpoint copies of the 36 delta layers' committed state lane,
/// conv ring and carry, 4,126,736 B each, the PLE ring, 17 · 4 · 2,560 f32,
/// and the 12 raw index key rings, 11 rows of 128 f16 each. Under the draft,
/// its store (2,048 B a position), the arena rows ([`ROWS`]) and each saved
/// inner point's hidden row (4 · 2,560 f32).
fn state38(ctx: u64, drafts: bool) -> u64 {
    let attention = 12 * (ctx * 2048 + ctx.div_ceil(4) * 256);
    let points = runtime::seqstate::Price {
        point_bytes: CHECKPOINT_38,
        every: UBATCH_PLANNED as u32,
    }
    .saved_points(ctx);
    let draft = if drafts {
        ctx * 2048 + ROWS + points * 4 * 2560 * 4
    } else {
        0
    };
    attention + (1 + points) * CHECKPOINT_38 + draft
}

/// One checkpoint's host bytes: the 36 delta layers' committed state lane,
/// conv ring and carry, the PLE ring and the 12 selecting layers' raw index
/// key rings ([`state38`]'s term).
const CHECKPOINT_38: u64 = 36 * 4_126_736 + 17 * 4 * 2560 * 4 + 12 * 11 * 128 * 2;

/// The stage card of `qwen35moe::place::machine_for_experts(spec,
/// Stack::q8_blocks32(48), 4096, Experts::Card)`: every layer, the head and
/// the token embedding, its scratch the m = 1 scratch, the ubatch walk's and
/// its card route's at 4,096 positions (the box gate's `CARD_ROUTE_SCRATCH`
/// row).
fn stage(spec: CardSpec) -> Card {
    Card {
        name: spec.name.to_string(),
        tier_leg: None,
        device: spec.device,
        usable_bytes: spec.usable_bytes(),
        context_bytes: CONTEXT,
        // PIN(2026-10-25): the box gate's `CARD_ROUTE_SCRATCH` row at 4,096 positions, 3,130,740,736 B
        // (was 3,228,766,208): HC's BF16 bottleneck and refusal add 644 B per ubatch position and the
        // ubatch arena's flash rows leave 24,576 [derived: 4,096 · (644 − 24,576) = −98,025,472].
        // PIN(2026-10-28): 3,020,115,968 B: the ubatch arena's query, β·α and one-row indexer rows
        // leave 27,008 B per ubatch position [derived: 4,096 · 27,008 = 110,624,768].
        // PIN(2026-10-30): 2,942,128,128 B: the PLE site's NVFP4 planes add 1,440 B and the ubatch
        // arena's BF16 stream buffer takes 20,480 B per ubatch position [derived: 4,096 · (1,440 −
        // 20,480) = −77,987,840].
        scratch_bytes: 2_942_128_128,
        margin_bytes: spec.margin(),
        granule_bytes: spec.granule_bytes,
        free_bytes: spec.free_bytes,
        held_by: spec.held_by,
        layers: 0..LAYERS,
        head: true,
        token_embedding: true,
        reserve_only: false,
        reserves: Vec::new(),
    }
}

/// `--place a`: the stage card alone beside the host.
fn machine_a(spec: CardSpec) -> Machine {
    Machine {
        cards: vec![stage(spec)],
        tiers: Vec::new(),
        nodes: vec![Node {
            host: host(),
            unified: None,
        }],
    }
}

/// `--place bp` (`qwen35moe::place::machine_bp_on` at 4,096 positions, no
/// draft): the stage with the ubatch walk's tier join in its scratch,
/// 210,124,800 B; the tier card with its prompt batch reserved — the
/// staging, and the block route's 348,980,248 B (the box gate's `BP_PLAIN`
/// derivation) — and the host the batch's rows.
fn machine_bp(stage_spec: CardSpec, tier: CardSpec) -> Machine {
    let mut s = stage(stage_spec);
    s.scratch_bytes += 210_124_800;
    let mut h = host();
    h.reserves.push((
        TIER_BATCH_HOST_RESERVE.to_string(),
        tier_batch_host_bytes(2560, 10, 4096),
    ));
    Machine {
        cards: vec![s],
        tiers: vec![Card {
            name: tier.name.to_string(),
            tier_leg: None,
            device: tier.device,
            usable_bytes: tier.usable_bytes(),
            context_bytes: CONTEXT,
            scratch_bytes: SCRATCH,
            margin_bytes: tier.margin(),
            granule_bytes: tier.granule_bytes,
            free_bytes: tier.free_bytes,
            held_by: tier.held_by,
            layers: 0..0,
            head: false,
            token_embedding: false,
            reserve_only: false,
            reserves: vec![(
                TIER_BATCH_RESERVE.to_string(),
                tier_batch_staging_bytes(2560, 10, 4096) + 348_980_248,
            )],
        }],
        nodes: vec![Node {
            host: h,
            unified: None,
        }],
    }
}

/// The card rule's plan of `model` on `machine` at `ctx` positions a slot
/// for `slots` resident sequences (`PlanInputs::plan_with_slots` under
/// `Experts::Card`, no draft): every slot's stores, the bytes beside them of
/// every slot but the live one reserved out of the expert budget, then the
/// PLE table's tier by the host's `room` ([`super::row_table_tier`] for
/// calls of up to [`UBATCH_PLANNED`] positions, as `PlanInputs` places it).
/// A plan that breaks an invariant fails the test by name.
fn plan_at<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx: u64,
    slots: u64,
    room: u64,
) -> Result<Plan<'a>, PlacementError> {
    plan_with(model, machine, ctx, slots, room, true)
}

/// [`plan_at`], the NVMe expert tier ([`super::expert_nvme_tier`], as
/// `PlanInputs` runs it after the PLE table's) applied or not.
fn plan_with<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx: u64,
    slots: u64,
    room: u64,
    experts_to_nvme: bool,
) -> Result<Plan<'a>, PlacementError> {
    plan_given(
        model,
        machine,
        ctx,
        slots,
        room,
        experts_to_nvme.then_some(None),
    )
}

/// [`plan_with`] with the split's given arena (`BLOOMERY_NVTIER_BYTES`'s
/// value) chosen: `None` runs no split, `Some(given)` runs it with `given`.
fn plan_given<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx: u64,
    slots: u64,
    room: u64,
    split: Option<Option<u64>>,
) -> Result<Plan<'a>, PlacementError> {
    let plan = planned(model, machine, ctx, slots, room, split)?;
    let broken = plan.violations();
    assert!(broken.is_empty(), "ctx {ctx} slots {slots}: {broken:?}");
    Ok(plan)
}

/// [`plan_given`]'s plan before its invariants are read: what a seat's
/// probe does with a plan that breaks one (it is refused at that context).
fn planned<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx: u64,
    slots: u64,
    room: u64,
    split: Option<Option<u64>>,
) -> Result<Plan<'a>, PlacementError> {
    planned_drafting(model, machine, ctx, slots, room, split, None)
}

/// [`planned`] beside the MTP draft when `fixed` names its card bytes: those,
/// and its store, 2,048 B a position of every slot, reserved out of the
/// expert budget as the seat's drafted plan holds them, and the PLE table's
/// floor counting the draft's saved state ([`state38`]).
fn planned_drafting<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx: u64,
    slots: u64,
    room: u64,
    split: Option<Option<u64>>,
    fixed: Option<u64>,
) -> Result<Plan<'a>, PlacementError> {
    let terms = SeqTerms {
        layers: Stores {
            kv: &Kv38,
            count: LAYERS,
        },
        draft: None,
        beside: BESIDE,
    };
    let rows = terms.plan_beside(slots);
    let kv = terms.slots_of(slots);
    let levers = PlanLevers::default();
    let draft = fixed.map_or(0, |f| f + slots * 2048 * ctx);
    let mut plan = plan_routed_reserving(
        model,
        machine,
        ctx,
        &kv,
        &levers,
        routed_rule(Program::Qwen38Body),
        rows + draft,
    )?;
    super::row_table_tier(
        &mut plan,
        room,
        2 * state38(ctx, fixed.is_some()),
        runtime::stores::reader_columns(UBATCH_PLANNED.min(ctx)),
        ROW_PAGE,
        super::RowTableSource::Split,
    )?;
    if let Some(given) = split {
        plan.host.step_slots(
            usize::try_from(slots).expect("slots"),
            super::STEP_PORT_COLUMNS,
        )?;
        super::expert_nvme_tier(&mut plan, room, given)?;
    }
    plan.cards[0].grow_kv(rows);
    Ok(plan)
}

/// The most positions one PLE fill reads (`qwen35moe::place::UBATCH_PLANNED`,
/// the largest ubatch a load runs).
const UBATCH_PLANNED: u64 = 4096;

/// The room of the 64 GiB host the sitting holds a Qwen3.8 arm to
/// (`depth-qwen3moe`'s `mem=61G` scope: `MemoryMax` 61 GiB, nothing charged
/// yet); the box's own is [`PLAN_ROOM`].
const SCOPE_61G: u64 = 61 << 30;

/// The unset residency rule on `plan` (`residency38_at_plan`) with
/// `MemAvailable` ample, and its churn pool's bytes, on the Qwen3.8 body,
/// which carries an unhosted pool (`Body38::UNHOSTED_POOL`, a crate this one
/// does not see).
fn residency(plan: &Plan<'_>) -> (Residency38Pick, u64) {
    let pool = |pinned| ChurnPool::of(plan, 0, pinned).map(|p| p.bytes);
    let pick = residency38_at_plan(
        UnhostedPool::Carried,
        plan.n_l.iter().copied(),
        plan.host.experts,
        pool,
        plan.host.headroom_bytes,
        i128::MAX,
    )
    .expect("the churn pool");
    let bytes = pick.pinned.map_or(0, |p| pool(p).expect("the pool"));
    (pick, bytes)
}

/// The Q4 file's routed experts and the PLE table, in the file's bytes:
/// what the box gate pins as `HOST_EXPERTS` and `HOST_TABLES`.
const Q4_ROUTED: u64 = 77_017_907_200;
const PLE_TABLE: u64 = 28_800_138_240;
// PIN(2026-10-07): the PLE table's row room on the NVMe tier, which a plan
// sets aside in place of the 4 GiB engram row cache (was ROW_CACHE,
// 4,294,967,296, the reserve of every machine's host): a call of
// UBATCH_PLANNED 4,096 positions reads 16 rows a position, each 90 B
// IQ4_NL row on at most 1 + ⌈89 / 4,096⌉ = 2 pages of 4,096 B —
// 4,096 · 16 · 2 · 4,096 = 536,870,912 B (`row_room`).
const ROW_ROOM: u64 = 536_870_912;

/// One, two or four such cards: the unset `--place` of `generate_qwen3moe`
/// and of the Qwen3.8 seat is `a`, the stage on cuda0 — the lowest ordinal
/// among equal usable bytes — and no tier. On it, at 4,096 positions and one
/// sequence, the card holds every routed expert of all 48 layers, so the
/// host serves none and the host need is the PLE table alone; the unset
/// residency rule is `off`, no churn pool beside it: no
/// host expert is left to move to the card. The restated terms first give
/// an A6000 plan at the box gate's context and ubatch, its `CARD_PLANS` row:
/// 267 experts on the first 28 layers, 266 on the rest, 40,100,659,200 B of
/// experts and 524,808,704 B of rounding.
#[test]
fn q4_on_one_96gb_card_holds_every_routed_expert() {
    let q4 = model(false);
    let routed: u64 = q4
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts)
        .map(|t| t.file_bytes)
        .sum();
    assert_eq!(routed, Q4_ROUTED);
    let m = machine_a(A6000);
    let a6000 = plan_at(&q4, &m, 4096, 1, PLAN_ROOM).expect("the A6000 plan");
    // PIN(2026-10-21): 262 on the first 46 layers and 39,404,339,200 B; was 262 on the first 40 and
    // 39,385,907,200 B. The card keeps the 24 indexer projections in their 2-byte words
    // (`indexer_projection`), 18 granules (37,748,736 B) fewer, which the spread fills with 6
    // experts of 3,072,000 B and their granules [derived: the box gate's CARD_PLANS row].
    // PIN(2026-10-24): 262 on the first 43 layers and 39,395,123,200 B; was 262 on the first 46 and
    // 39,404,339,200 B. Each of the 36 GDN layers keeps a carry of 497,296 B
    // (`runtime::stores::delta_carry_bytes`), 17,902,656 B of the card, which the spread takes from 3
    // experts of 3,072,000 B and their granules [derived: the box gate's CARD_PLANS row].
    // PIN(2026-10-25): unchanged by the stage's HC scratch, 2,637,824 B at 4,096 [derived: this plan].
    // PIN(2026-10-25): 263 on the first 31 layers and 262 on the rest, 39,508,684,800 B; was 262 on the
    // first 43 and 39,395,123,200 B. The ubatch arena's flash rows leave the stage's scratch,
    // 100,663,296 B at 4,096, which the spread fills with experts and their granules [derived: this
    // plan].
    // PIN(2026-10-26): 263 on the first 30 layers, 39,505,100,800 B and 593,981,952 B of rounding,
    // the box gate's CARD_PLANS row; was 263 on the first 31 and 39,508,684,800 B. The stage's rope
    // rows (`Kv38::stage_position_bytes`, 1,048,576 B at 4,096) leave the expert budget, so layer
    // 30 gives up one expert of 3,584,000 B [derived: this plan].
    // PIN(2026-10-28): 264 on the first 17 layers and 263 on the rest, 39,615,590,400 B and
    // 592,544,256 B of rounding, the box gate's CARD_PLANS row; was 263 on the first 30. The ubatch
    // arena's query, β·α and one-row indexer rows leave the stage's scratch, 110,624,768 B at 4,096,
    // which the spread fills with experts and their granules [derived: this plan].
    // PIN(2026-10-29): 264 on the first 20 layers, 39,624,806,400 B and 595,911,168 B of rounding;
    // was 264 on the first 17 and 39,615,590,400 B. A selecting layer's raw index keys leave its
    // positional bytes for a ring of 11 rows (`Kv38`), 12 · (256 · 4,096 − 2,816) = 12,549,120 B at
    // 4,096, which the spread fills with 3 experts of 3,072,000 B [derived: this plan].
    // PIN(2026-10-30): 267 on the first 28 layers and 266 on the rest, 40,100,659,200 B and
    // 524,808,704 B of rounding, the box gate's CARD_PLANS row; was 264 on the first 20 and
    // 39,624,806,400 B. Each of the 36 delta layers keeps one state lane and a verify record
    // (`Kv38`, 4,388,500 B, was 13,530,784 B of four lanes), and the stage's scratch is the box
    // gate's CARD_ROUTE_SCRATCH row (`stage`), 407,110,064 B more room in all, which the spread
    // fills with experts and their granules [derived: this plan].
    let want: Vec<u64> = (0..LAYERS)
        .map(|l| if l < 28 { 267 } else { 266 })
        .collect();
    assert_eq!(
        (
            &a6000.n_l,
            a6000.cards[0].expert_bytes,
            a6000.cards[0].rounding_bytes
        ),
        (&want, 40_100_659_200, 524_808_704)
    );

    let one = picked("a", &census_96(1))[0];
    for n in [1, 2, 4] {
        let cards = picked("a", &census_96(n));
        assert_eq!(cards, vec![one], "{n} cards");
    }
    assert_eq!(
        (one.name, one.device.map(|d| d.ordinal), one.usable_bytes()),
        (
            "RTX_PRO_6000_Blackwell_Max-Q_Workstation_Edition",
            Some(0),
            TOTAL_96
        )
    );
    let m = machine_a(one);
    let plan = plan_at(&q4, &m, 4096, 1, PLAN_ROOM).expect("the 96 GB plan");
    assert_eq!(plan.n_l, vec![EXPERTS; LAYERS]);
    assert_eq!(
        (
            plan.cards[0].experts,
            plan.cards[0].expert_bytes,
            plan.host.experts,
            plan.host.expert_bytes
        ),
        (EXPERTS * LAYERS as u64, Q4_ROUTED, 0, 0)
    );
    // PIN(2026-10-07): the PLE table alone; was PLE_TABLE + 4 GiB, the engram
    // row cache every machine's host reserved. The room holds the table, so
    // the plan reads it from the host and sets no row reserve aside
    // (`row_table_tier`; `q3_routes_onto_cards_like_the_q4_file` pins the NVMe arm).
    assert_eq!(HostNeed::of(&plan, 0).bytes(), PLE_TABLE);
    // PIN(2026-10-06): `off` (NoHostExperts), no pool; was `mid-p256-s1`
    // with a pool of Q4_ROUTED / 2. plan.host.experts is 0 above, so a pool
    // of the card's experts past the pinned ones serves no host expert.
    let (pick, pool) = residency(&plan);
    assert_eq!(
        pick,
        Residency38Pick {
            pinned: None,
            why: Residency38Why::NoHostExperts
        }
    );
    assert_eq!((pick.word(), pool), ("off".to_string(), 0));
}

/// The Q3 file's routed gate and up are IQ3_XXS or IQ4_XS and its downs
/// IQ4_NL or Q8_0: all of them card types in their parts (`routed_rule`), so on a large
/// enough card every routed expert is the card's as the Q4 file's are — the
/// host need the PLE table alone, the residency rule
/// `off` for no host expert left. Under `--place bp` on two such cards both
/// plans are refused by name as a tier card that holds no expert: each
/// file's stage card holds every expert. On the gate card, at the 8,192
/// positions of the 64 GiB plan, the PLE table's tier follows the host's
/// room (`row_table_tier`): the box's room holds the plan with the table on
/// the host, where it stays with no row reserve; the 61 GiB scope's does
/// not, so the table stays on the NVMe tier and the plan sets its row room
/// aside instead — a need the scope holds. The rule's boundary is the host
/// arm's need itself.
// PIN(2026-10-06): the iq stacks joined `card_routed` (iqwire), so the Q3
// file plans onto cards like the Q4 file; was n_l 0 on every layer, the
// whole routed share host-side (`NoCardExperts`), and the bp refusal because
// no layer's stacks were card types — now because the stage holds them all.
#[test]
fn q3_routes_onto_cards_like_the_q4_file() {
    let q3 = model(true);
    let bytes = |keep: fn(&ModelTensor) -> bool| -> u64 {
        q3.tensors
            .iter()
            .filter(|t| keep(t))
            .map(|t| t.file_bytes)
            .sum()
    };
    // The UD-Q3_K_XL shards' tensor bytes [read from their headers: the
    // shards' 89,986,353,824 B less shard 1, which holds none, and the two
    // other headers rounded up to 32 B], and its routed share.
    assert_eq!(bytes(|_| true), 89_975_329_280);
    let routed = bytes(|t| t.role == Role::RoutedExperts);
    assert_eq!(routed, 55_823_564_800);
    let cards = picked("bp", &census_96(2));
    let m = machine_a(cards[0]);
    let plan = plan_at(&q3, &m, 4096, 1, PLAN_ROOM).expect("the Q3 plan");
    assert_eq!(plan.n_l, vec![EXPERTS; LAYERS]);
    assert_eq!(
        (plan.cards[0].experts, plan.cards[0].expert_bytes),
        (EXPERTS * LAYERS as u64, routed)
    );
    // PIN(2026-10-07): the PLE table alone; was PLE_TABLE + 4 GiB, the engram
    // row cache every machine's host reserved. The room holds the table, so
    // the plan reads it from the host and sets no row reserve aside
    // (`row_table_tier`; the gate card's NVMe arm below).
    assert_eq!(HostNeed::of(&plan, 0).bytes(), PLE_TABLE);
    assert_eq!(
        residency(&plan),
        (
            Residency38Pick {
                pinned: None,
                why: Residency38Why::NoHostExperts
            },
            0
        )
    );
    let gate = machine_a(RTX_3090);
    let at = |room: u64| plan_at(&q3, &gate, 8192, 1, room).expect("the gate card's Q3 plan");
    let host = at(PLAN_ROOM);
    let host_need = HostNeed::of(&host, 0).bytes();
    assert_eq!(
        (
            host.row_tier().expect("one tier"),
            host.nvme_bytes,
            host.host.table_bytes,
            host.host.row_reserve_bytes
        ),
        (Some(Device::Host), 0, PLE_TABLE, 0)
    );
    let nvme = at(SCOPE_61G);
    let nvme_need = HostNeed::of(&nvme, 0).bytes();
    assert_eq!(
        (
            nvme.row_tier().expect("one tier"),
            nvme.nvme_bytes,
            nvme.host.table_bytes,
            nvme.host.row_reserve_bytes
        ),
        (Some(Device::Nvme), PLE_TABLE, 0, ROW_ROOM)
    );
    // The host experts are the routed share less the 13,912,089,600 B the
    // 3090 holds; the host arm adds the table to them, the NVMe arm the row
    // room (the OS reserve is MemAvailable's own and leaves the need).
    // PIN(2026-10-21): 6 experts of this file's 2,176,000 B more on the card, 13,056,000 B off the
    // host and both needs; was 41,924,531,200, 70,724,669,440 and 42,461,402,112 B. The card keeps
    // the 24 indexer projections in their words (`indexer_projection`), 18 granules fewer
    // [derived: the same spread, 13,056,000 = 6 · 2,176,000].
    // PIN(2026-10-24): 3 experts of 2,176,000 B fewer on the card, 6,528,000 B onto the host and both
    // needs; was 41,911,475,200, 70,711,613,440 and 42,448,346,112 B. The GDN layers' carries take
    // 17,902,656 B of the card [derived: the same spread, 6,528,000 = 3 · 2,176,000].
    // PIN(2026-10-25): 1 expert of 2,176,000 B fewer on the card, onto the host and both needs; was
    // 41,918,003,200, 70,718,141,440 and 42,454,874,112 B. The stage's scratch holds HC's 644 B per
    // ubatch position, 2,637,824 B at 4,096 [derived: the same spread].
    // PIN(2026-10-25): 35,635,200 B more experts on the card, off the host and both needs; was
    // 41,920,179,200, 70,720,317,440 and 42,457,050,112 B. The ubatch arena's flash rows leave the
    // stage's scratch, 100,663,296 B at 4,096 [derived: the same spread].
    // PIN(2026-10-28): 196,300,800 B more experts on the card, off the host and both needs; was
    // 41,884,544,000, 70,684,682,240 and 42,421,414,912 B. The ubatch arena's query, β·α and one-row
    // indexer rows leave the stage's scratch, 110,624,768 B at 4,096 [derived: the same spread].
    // PIN(2026-10-29): 12 experts of 2,176,000 B more on the card, 26,112,000 B off the host and both
    // needs; was 41,688,243,200, 70,488,381,440 and 42,225,114,112 B. A selecting layer's raw index
    // keys leave its positional bytes for a ring of 11 rows (`Kv38`), 12,549,120 B of the card at
    // 4,096, which the spread fills with experts and their granules [derived: the same spread].
    // PIN(2026-10-30): 378,726,400 B more experts on the card, off the host and both needs; was
    // 41,662,131,200, 70,462,269,440 and 42,199,002,112 B. The delta layers' one state lane and
    // verify record (`Kv38`) and the box gate's stage scratch (`stage`) leave 407,110,064 B of the
    // card at 8,192, which the spread fills with experts and their granules [derived: the same
    // spread].
    assert_eq!(
        (host.host.expert_bytes, host_need, nvme_need),
        (41_283_404_800, 70_083_543_040, 41_820_275_712)
    );
    assert_eq!(nvme_need + PLE_TABLE, host_need + ROW_ROOM);
    assert!(nvme_need <= SCOPE_61G && SCOPE_61G < host_need);
    assert_eq!(
        at(host_need).row_tier().expect("one tier"),
        Some(Device::Host)
    );
    assert_eq!(
        at(host_need - 1).row_tier().expect("one tier"),
        Some(Device::Nvme)
    );
    // A table the rule already moved is not the NVMe tier's to place again,
    // and a table on no tier a reader takes is refused by name.
    let mut moved = at(PLAN_ROOM);
    assert!(matches!(
        super::row_table_tier(
            &mut moved,
            PLAN_ROOM,
            0,
            runtime::stores::reader_columns(UBATCH_PLANNED),
            ROW_PAGE,
            super::RowTableSource::Split,
        ),
        Err(PlacementError::Tensor { .. })
    ));
    let ple = moved
        .rows
        .iter()
        .position(|r| q3.tensors[r.tensor].role == Role::EngramTable)
        .expect("the PLE row");
    moved.rows[ple].segments[0].device = Device::Unused;
    assert!(matches!(
        moved.row_tier(),
        Err(PlacementError::Tensor { .. })
    ));
    let bp = machine_bp(cards[0], cards[1]);
    // PIN(2026-10-06): refused (IdleTier); was a plan whose tier_n_l is 0 on
    // every layer. The stage holds all 512 experts of every layer, so no
    // expert is left for the tier.
    assert!(matches!(
        plan_at(&q3, &bp, 4096, 1, PLAN_ROOM),
        Err(PlacementError::IdleTier { tier: 0, .. })
    ));
    let q4 = model(false);
    assert!(matches!(
        plan_at(&q4, &bp, 4096, 1, PLAN_ROOM),
        Err(PlacementError::IdleTier { tier: 0, .. })
    ));
}

/// The target matrices an MTP draft file borrows, `token_embd` and
/// `output`: the Q4 file's are Q8_0, and the Q3 file's `output` is the Q6_K
/// form the draft's head gemv also reads, so both files lend them
/// (`PlanInputs::mtp_borrows`); `token_embd` is Q8_0 in both.
// PIN(2026-10-06): `output` follows `mtp::head_kind`'s set (Q8_0 or Q6_K,
// iqwire), so the Q3 file's Q6_K head lends too; was Q8_0 alone, which
// turned the Q3 seat's unset draft off.
#[test]
fn both_files_lend_the_draft_their_output_matrices() {
    let types = |q3: bool| -> Vec<(String, GgmlType)> {
        model(q3)
            .tensors
            .into_iter()
            .filter(|t| t.name == "token_embd.weight" || t.name == "output.weight")
            .map(|t| (t.name, t.ty))
            .collect()
    };
    let want = |out: GgmlType| {
        vec![
            ("output.weight".to_string(), out),
            ("token_embd.weight".to_string(), GgmlType::Q8_0),
        ]
    };
    assert_eq!(types(false), want(GgmlType::Q8_0));
    assert_eq!(types(true), want(GgmlType::Q6_K));
}

/// The card bytes of the Qwen3.8 seat's default load, two resident
/// sequences and no draft, at `ctx` positions a slot on one such card: its
/// per-layer counts and its host experts beside them.
fn two_slots(ctx: u64) -> (u64, Vec<u64>, u64) {
    let q4 = model(false);
    let m = machine_a(picked("a", &census_96(1))[0]);
    let p = plan_at(&q4, &m, ctx, 2, PLAN_ROOM).expect("a two-slot plan");
    (p.cards[0].expert_bytes, p.n_l.clone(), p.host.experts)
}

/// The Qwen3.8 seat's context rule (`ctx38`, `margin38`) counts against
/// the plan at 4,096 positions a slot, which on one such card holds every
/// routed expert for both sequences.
#[test]
fn two_slots_at_4096_hold_every_routed_expert() {
    let (bytes, n_l, host) = two_slots(4096);
    assert_eq!((bytes, n_l, host), (Q4_ROUTED, vec![EXPERTS; LAYERS], 0));
}

/// 300,032 positions a slot on one such card, two sequences and no draft,
/// is the last step of 256 whose plan holds at most the plan's margin fewer
/// card expert bytes than the plan at 4,096
/// ([`two_slots_at_4096_hold_every_routed_expert`]: every routed byte); the
/// next step passes it. There 505 experts a layer stay on the card and 336
/// go to the host.
// PIN(2026-10-21): 252,928 positions (was 252,416), 505 or 506 experts a layer with one at 506 and
// 335 on the host (was 504 or 505 with 47 at 505 and 337). The card keeps the 24 indexer
// projections in their words (`indexer_projection`): 37,748,736 B of room, 664 positions of the two
// slots' cache (2 · 28,416 B a position), and the boundary moved two steps of 256 within the
// granules [derived: the plan's own boundary, 252,928 − 252,416 = 512].
// PIN(2026-10-24): 252,416 positions (was 252,928), 505 experts on every layer and 336 on the host
// (was 505 or 506 with one at 506 and 335). Each sequence's 36 GDN layers keep a carry of 497,296 B
// (`runtime::stores::delta_carry_bytes`): 35,805,312 B of the two slots' room, 630 positions of
// their cache, and the boundary moved two steps of 256 within the granules [derived: the plan's own
// boundary, 252,928 − 252,416 = 512].
// PIN(2026-10-25): 504 or 505 experts a layer with 46 at 505 and 338 on the host (was 505 on every
// layer and 336). The stage's scratch holds HC's 644 B per ubatch position, 2,637,824 B at 4,096,
// and the boundary stays at 252,416 [derived: the plan's own counts].
// PIN(2026-10-25): 254,208 positions (was 252,416), the same 504 or 505 experts a layer with 46 at
// 505 and 338 on the host. The ubatch arena's flash rows leave the stage's scratch, 100,663,296 B at
// 4,096, and the boundary moved seven steps of 256 within the granules [derived: the plan's own
// boundary, 254,208 − 252,416 = 1,792].
// PIN(2026-10-26): 252,928 positions (was 254,208), 505 experts on every layer and 336 on the host
// (was 504 or 505 with 46 at 505 and 338). The stage keeps its rope rows
// (`Kv38::stage_position_bytes`, 256 B a position, one copy for both slots): 65,077,248 B at
// 254,208, 1,140 positions of the two slots' 57,088 B, and the boundary moved five steps of 256
// within the granules [derived: the plan's own boundary, 254,208 − 252,928 = 1,280].
// PIN(2026-10-28): 254,976 positions (was 252,928), 504 or 505 experts a layer with 47 at 505 and
// 337 on the host (was 505 on every layer and 336). The ubatch arena's query, β·α and one-row
// indexer rows leave the stage's scratch, 110,624,768 B at 4,096, and the boundary moved eight
// steps of 256 within the granules [derived: the plan's own boundary, 254,976 − 252,928 = 2,048].
// PIN(2026-10-29): 285,696 positions (was 254,976), the same 504 or 505 experts a layer with 47 at
// 505 and 337 on the host. A selecting layer's raw index keys leave its positional bytes for a ring
// of 11 rows (`Kv38`): 3,072 B a position a slot, the two slots' cache and the stage's rope rows
// 50,944 B a position (was 57,088), and the boundary moved 120 steps of 256 [derived: the plan's
// own boundary, 285,696 − 254,976 = 30,720].
// PIN(2026-10-30): 300,032 positions (was 285,696), 505 experts on every layer and 336 on the host
// (was 504 or 505 with 47 at 505 and 337). Each sequence's 36 delta layers keep one state lane and
// a verify record (`Kv38`, 9,142,284 B less a layer) and the stage's scratch is the box gate's
// CARD_ROUTE_SCRATCH row (`stage`, 77,987,840 B less): 736,232,288 B of room, 14,451 positions of
// 50,944 B, and the boundary moved 56 steps of 256 [derived: the plan's own boundary, 300,032 −
// 285,696 = 14,336].
#[test]
fn two_slots_at_300032_stay_within_the_margin() {
    let (at, n_l, host) = two_slots(300_032);
    assert!(Q4_ROUTED - at <= MARGIN);
    let (next, _, _) = two_slots(300_032 + 256);
    assert!(Q4_ROUTED - next > MARGIN);
    let (lo, hi) = (
        *n_l.iter().min().expect("layers"),
        *n_l.iter().max().expect("layers"),
    );
    assert_eq!(
        (lo, hi, n_l.iter().filter(|&&n| n == hi).count(), host),
        (505, 505, 48, 336)
    );
}

/// The host-served routed experts of `layer` in `plan`: per routed stack, its
/// Host segment's expert count and its NVMe segment's, and the bytes of both
/// sides.
fn layer_split(plan: &Plan<'_>, layer: usize) -> (u64, u64, u64, u64) {
    let (mut host_n, mut nvme_n, mut host_b, mut nvme_b) = (0, 0, 0, 0);
    let mut first = true;
    for r in &plan.rows {
        let t = &plan.model.tensors[r.tensor];
        if t.role != Role::RoutedExperts || t.layer != Some(layer) {
            continue;
        }
        for s in &r.segments {
            let n = s.experts.as_ref().map_or(0, |l| l.len());
            match s.device {
                Device::Host => {
                    host_b += s.resident_bytes;
                    if first {
                        host_n += n;
                    }
                }
                Device::Nvme => {
                    nvme_b += s.resident_bytes;
                    if first {
                        nvme_n += n;
                    }
                }
                _ => {}
            }
        }
        first = false;
    }
    (host_n, nvme_n, host_b, nvme_b)
}

/// Bytes of one expert of `layer` across its routed stacks.
fn layer_unit(model: &ModelTensors, layer: usize) -> u64 {
    model
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts && t.layer == Some(layer))
        .map(|t| t.file_bytes / EXPERTS)
        .sum()
}

/// The split dial's terms at `room` as [`super::expert_nvme_tier`] reads
/// them off `twin`, the plan before the split: the host terms beside the
/// routed experts, the heaviest layer's host bytes, the host leg, the need
/// and the arena's floor at the served column.
fn dial_of(twin: &Plan<'_>, room: u64) -> ArenaDial {
    let need = HostNeed::of(twin, 0);
    ArenaDial {
        room,
        base: HostNeed { experts: 0, ..need }.bytes(),
        cards: need.cards,
        heavy: (0..LAYERS)
            .map(|l| layer_split(twin, l).2)
            .max()
            .expect("layers"),
        held: twin.host.expert_bytes,
        need: need.bytes(),
        least: ArenaFloor::of(twin).expect("the arena's floor"),
    }
}

/// What the tests derive of a plan on the gate card beside a room: the plan
/// with the NVMe expert tier, its unsplit twin at the same room, and the
/// numbers of the table.
struct NvmeArm {
    old_need: u64,
    room: u64,
    arena: u64,
    moved: u64,
    r_min: u64,
    r_max: u64,
    host_bytes: u64,
    floor: u64,
    max_unit: u64,
}

/// The gate card's Q4 plan at 4,096 positions beside `room` (the 3090: 92
/// or 93 experts a layer on the card, the rest host-served), every clause
/// both plans of the NVMe tier share, and the arm's numbers. `old_need` is
/// the need the unsplit plan has with the PLE table on the NVMe tier,
/// derived from the box plan: its need less the table plus the row room.
fn nvme_arm(room: u64) -> NvmeArm {
    let q4 = model(false);
    let gate = machine_a(RTX_3090);
    let box_plan = plan_at(&q4, &gate, 4096, 1, PLAN_ROOM).expect("the box plan");
    assert_eq!(
        box_plan.host.nvme_expert_bytes, 0,
        "the box room splits nothing"
    );
    let old_need = HostNeed::of(&box_plan, 0).bytes() - PLE_TABLE + ROW_ROOM;
    let split = plan_at(&q4, &gate, 4096, 1, room).expect("the split plan");
    let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
    assert_eq!(HostNeed::of(&twin, 0).bytes(), old_need);

    // The PLE table first: on the NVMe tier, the row room set aside.
    assert_eq!(split.row_tier().expect("one tier"), Some(Device::Nvme));
    assert_eq!(split.host.table_bytes, 0);
    assert_eq!(split.host.row_reserve_bytes, ROW_ROOM);

    // The split dial's own terms, from the unsplit plan's bytes: the floor,
    // then the arena it picks for this room, which the plan states.
    let layer_bytes = (0..LAYERS)
        .map(|l| layer_split(&twin, l).2)
        .max()
        .expect("layers");
    let base = HostNeed {
        experts: 0,
        ..HostNeed::of(&twin, 0)
    }
    .bytes();
    let floor = base + 3 * layer_bytes;
    let arena = super::nvme_arena_of(&dial_of(&twin, room), None).expect("the dial's default");
    assert_eq!(
        split.host.nvme_arena_bytes, arena,
        "the plan states the arena the dial chose"
    );

    let need = HostNeed::of(&split, 0).plan_bytes();
    let moved = split.host.nvme_expert_bytes;
    // The split plan reserves the prompt run-ahead's window, two of the
    // heaviest layer's host bytes, beside its host experts.
    let window = 2 * layer_bytes;
    let overflow = old_need + window - (room - arena);
    let max_unit = (0..LAYERS)
        .map(|l| layer_unit(&q4, l))
        .max()
        .expect("layers");
    assert!(
        need + arena <= room,
        "need {need} B and arena {arena} B pass the room {room} B"
    );
    assert_eq!(need + moved, old_need + window);
    // The NVMe segments are the overflow the arena and the window grew, to
    // within the one expert a layer's whole experts leave.
    assert!(
        overflow <= moved && moved < overflow + max_unit,
        "moved {moved} B for an overflow of {overflow} B (arena {arena} B)"
    );

    // Every expert once, the host's ids before the NVMe's; the sums of the
    // rows are the plan's fields; the card side is the unsplit plan's.
    let (mut host_sum, mut nvme_sum) = (0, 0);
    let (mut r_min, mut r_max) = (u64::MAX, 0);
    for l in 0..LAYERS {
        let (host_n, nvme_n, host_b, nvme_b) = layer_split(&split, l);
        let (t_host_n, t_nvme_n, ..) = layer_split(&twin, l);
        assert_eq!(nvme_n + host_n, t_host_n, "layer {l}");
        assert_eq!(t_nvme_n, 0);
        let unit = layer_unit(&q4, l);
        assert_eq!(
            (host_b, nvme_b),
            (host_n * unit, nvme_n * unit),
            "layer {l}"
        );
        host_sum += host_b;
        nvme_sum += nvme_b;
        r_min = r_min.min(host_n);
        r_max = r_max.max(host_n);
        for r in &split.rows {
            let t = &q4.tensors[r.tensor];
            if t.role != Role::RoutedExperts || t.layer != Some(l) {
                continue;
            }
            let ids: Vec<u32> = r
                .segments
                .iter()
                .filter(|s| matches!(s.device, Device::Host | Device::Nvme))
                .flat_map(|s| s.experts.as_ref().expect("a list").ids().to_vec())
                .collect();
            assert!(
                ids.windows(2).all(|w| w[0] < w[1]),
                "layer {l}: host ids, then NVMe's"
            );
            let on_host = r.segments.iter().find(|s| s.device == Device::Host);
            assert_eq!(
                on_host.map_or(0, |s| s.experts.as_ref().expect("a list").len()),
                host_n
            );
        }
    }
    assert_eq!(host_sum, split.host.expert_bytes);
    assert_eq!(nvme_sum, moved);
    assert_eq!(split.nvme_bytes, PLE_TABLE + moved);
    assert_eq!(split.host.expert_bytes + moved, twin.host.expert_bytes);
    assert_eq!(format!("{:?}", split.n_l), format!("{:?}", twin.n_l));
    assert_eq!(
        format!("{:?}", split.tier_n_l),
        format!("{:?}", twin.tier_n_l)
    );
    assert_eq!(format!("{:?}", split.cards), format!("{:?}", twin.cards));
    for (a, b) in split.rows.iter().zip(&twin.rows) {
        let card = |r: &super::Row| -> String {
            let on: Vec<_> = r
                .segments
                .iter()
                .filter(|s| matches!(s.device, Device::Card(_)))
                .collect();
            format!("{on:?}")
        };
        assert_eq!(card(a), card(b), "{}", q4.tensors[a.tensor].name);
    }

    // The floor, from the unsplit plan's own bytes: the host terms beside
    // the routed experts, and three of the heaviest layer's host-served
    // experts — the terms the dial's arena leaves over it.
    let arm = NvmeArm {
        old_need,
        room,
        arena,
        moved,
        r_min,
        r_max,
        host_bytes: split.host.expert_bytes,
        floor,
        max_unit,
    };
    println!(
        "room {} B | old need {} B | arena {} B | host experts {} B | NVMe experts {} B | PLE NVMe | r_l {}..{} | floor {} B",
        arm.room,
        arm.old_need,
        arm.arena,
        arm.host_bytes,
        arm.moved,
        arm.r_min,
        arm.r_max,
        arm.floor
    );
    arm
}

/// A host of 27 GiB (a 32 GB machine's room): the PLE table on the NVMe
/// tier first, then the split dial gives the tier the largest arena the room
/// leaves above the floor, and each layer's host experts split between the
/// host and the NVMe tier by the overflow the arena grew.
#[test]
fn a_27_gib_room_puts_the_overflow_on_the_nvme_tier() {
    let arm = nvme_arm(27 << 30);
    assert!(arm.moved > 0 && arm.r_min > 0 && arm.r_max < EXPERTS);
    // The deep split: the room cannot hold half the host leg, so the arena
    // takes what the floor leaves — and the host segments shrink to it.
    assert!(arm.arena > 0, "the dial gives the deep split an arena");
    assert_eq!(arm.arena, arm.room - arm.floor);
}

/// A split plan's headroom is what the room leaves past the arena and the
/// host need, so a churn pool is priced against the bytes the load really
/// holds: on the 27 GiB room the pool of every card expert (`mid-p0`) is
/// refused by name, and on both rooms the unset rule's pick keeps the host
/// need, the arena and its pool inside the room.
#[test]
fn a_split_plans_pool_fits_the_room_past_the_arena() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    for room in [27u64 << 30, 58 << 30] {
        let split = plan_at(&q4, &gate, 4096, 1, room).expect("the split plan");
        let need = HostNeed::of(&split, 0).plan_bytes();
        let arena = split.host.nvme_arena_bytes;
        assert!(split.host.nvme_expert_bytes > 0, "room {room} splits");
        assert_eq!(
            split.host.headroom_bytes,
            i128::from(room) - i128::from(arena) - i128::from(need),
            "room {room}: the headroom is the room's past the arena and the need"
        );
        let (pick, pool) = residency(&split);
        assert!(
            need + arena + pool <= room,
            "room {room}: {} holds need {need} B + arena {arena} B + pool {pool} B",
            pick.word()
        );
        // PIN(2026-10-11): the pick on a paged plan is the family's own
        // rule, `mid-p0-s1` over the empty pool its stacks leave
        // ([`a_paged_stack_adds_nothing_to_the_churn_pool`]); the removed
        // paged clause had answered `off` at the 27 GiB room.
        assert_eq!(pick.word(), "mid-p0-s1", "room {room}: the unset rule");
    }
    // The pool of every card expert, as the unsplit plan holds it.
    let split = plan_at(&q4, &gate, 4096, 1, 27 << 30).expect("the split plan");
    let twin = plan_with(&q4, &gate, 4096, 1, 27 << 30, false).expect("the unsplit plan");
    let all = ChurnPool::of(&twin, 0, 0).expect("the pool of every card expert");
    match all.check(&split) {
        Err(PlacementError::ResidencyOverHost(_)) => {}
        other => panic!("mid-p0's pool of {} B is not refused: {other:?}", all.bytes),
    }
}

/// The load's check counts the arena (`HostNeed::bytes`): a paged plan made
/// at a room that holds its need and arena, checked at load against an
/// available reading the room shrank to — above the need without the arena,
/// under it with — refuses by name with the arena's bytes as a term, the
/// reading that holds both passes, and a plan with no arena checks exactly
/// as before (the two sums one).
#[test]
fn a_paged_plans_load_check_counts_the_arena() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let deep = plan_at(&q4, &gate, 4096, 1, 27 << 30).expect("the split plan");
    let need = HostNeed::of(&deep, 0);
    assert!(need.arena > 0, "the deep split's dial gives an arena");
    assert_eq!(need.arena, deep.host.nvme_arena_bytes);
    // The shrunk room: the need without the arena fits, the load's bytes do
    // not.
    let shrunk = need.bytes() - 1;
    assert!(
        need.plan_bytes() <= shrunk,
        "the shrunk reading {shrunk} B is past the arena-free {} B",
        need.plan_bytes()
    );
    let text = need
        .check(&(shrunk, HostRead::MemAvailable))
        .expect_err("the load refuses")
        .to_string();
    for part in [
        format!("the {} B this load needs", need.bytes()),
        format!("+ NVMe tier arena {} B", need.arena),
        "BLOOMERY_NVTIER_BYTES=0".to_string(),
    ] {
        assert!(text.contains(&part), "{part:?} in {text}");
    }
    assert_eq!(need.check(&(need.bytes(), HostRead::MemAvailable)), Ok(()));
    // A plan with no arena checks exactly as before: the two sums are one.
    let wide = plan_at(&q4, &gate, 4096, 1, 58 << 30).expect("the wide split plan");
    let need = HostNeed::of(&wide, 0);
    assert_eq!(need.arena, 0, "the wide room's dial gives no arena");
    assert_eq!(need.bytes(), need.plan_bytes());
    assert_eq!(need.check(&(need.bytes(), HostRead::MemAvailable)), Ok(()));
    let text = need
        .check(&(need.bytes() - 1, HostRead::MemAvailable))
        .expect_err("one byte short")
        .to_string();
    assert!(!text.contains("NVMe tier arena"), "{text}");
}

/// A deep split fills its room, so a plan made at the host's reading itself
/// passes the load's check only while the reading holds. A plan made at
/// [`plan_room`] of a read room keeps the drift margin and passes the check
/// at a reading the process's own start has lowered by less than that
/// margin.
#[test]
fn a_read_rooms_plan_holds_at_a_lowered_reading() {
    const READ: u64 = 27_631_529_984;
    const LOWERED: u64 = READ - 132_071_424;
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let room = plan_room(&(READ, HostRead::MemAvailable));
    let plan = plan_at(&q4, &gate, 16384, 1, room).expect("the deep split plan");
    let need = HostNeed::of(&plan, 0);
    assert!(need.arena > 0, "the room is a deep split's");
    assert_eq!(need.check(&(LOWERED, HostRead::MemAvailable)), Ok(()));
}

/// A room the caller gave plans as given: the plan at [`plan_room`] of a
/// given room is the plan at that room.
#[test]
fn a_given_room_plans_as_given() {
    const GIVEN: u64 = 27_631_529_984;
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let room = plan_room(&(GIVEN, HostRead::Given));
    assert_eq!(room, GIVEN);
    let at = plan_at(&q4, &gate, 16384, 1, room).expect("the plan at the given room");
    let direct = plan_at(&q4, &gate, 16384, 1, GIVEN).expect("the plan at the room");
    assert_eq!(format!("{at:?}"), format!("{direct:?}"));
}

/// An unpaged plan keeps the drift margin too. A deeper context grows while
/// no expert goes to the NVMe tier, so a plan made at the reading itself
/// ends with its need at the reading,
/// and a load whose reading fell by a little less than the margin is
/// refused with no arena to shrink. A plan made at [`plan_room`] of the
/// reading pages the context whose need passes the margin, so its need ends
/// a margin under the reading.
#[test]
fn an_unpaged_plan_at_the_plan_room_ends_a_margin_under_the_reading() {
    // The reading the load took less than the plan's, as a second host
    // reported it.
    const SHORT: u64 = 162_668_544;
    // A room under the PLE table and over the experts: the table is on the
    // NVMe tier already, as in the reporter's plan (tables 0), and no
    // expert is.
    const WHOLE: u64 = 70 << 30;
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let whole = plan_given(&q4, &gate, 4096, 1, WHOLE, Some(None)).expect("the plan at 4096");
    assert_eq!(whole.host.table_bytes, 0, "the table is off the host");
    assert_eq!(whole.host.nvme_expert_bytes, 0, "the room pages no expert");
    let need = HostNeed::of(&whole, 0).bytes();
    assert!(need < WHOLE, "the room holds the experts whole");

    // A plan made at the reading itself fills it, and its load is refused.
    let at_reading =
        plan_given(&q4, &gate, 4096, 1, need, Some(None)).expect("the plan at its need");
    assert_eq!(
        at_reading.host.nvme_expert_bytes, 0,
        "the reading holds it whole"
    );
    let filled = HostNeed::of(&at_reading, 0);
    assert_eq!(filled.bytes(), need);
    assert!(
        filled
            .check(&(need - SHORT, HostRead::MemAvailable))
            .is_err(),
        "no arena to shrink"
    );

    // The plan at the reading's plan room pages, and its load passes the
    // lowered reading.
    let room = plan_room(&(need, HostRead::MemAvailable));
    let kept =
        plan_given(&q4, &gate, 4096, 1, room, Some(None)).expect("the plan at the plan room");
    assert!(
        kept.host.nvme_expert_bytes > 0,
        "the context passes the margin: it pages"
    );
    assert_eq!(
        HostNeed::of(&kept, 0).check(&(need - SHORT, HostRead::MemAvailable)),
        Ok(())
    );
}

/// A plan names the room it was made at: the reading and the plan room, a
/// read room's less the drift margin and a given room's as given; a plan
/// made with no host room names none and adds nothing to the load's refusal.
#[test]
fn a_plan_names_the_room_it_was_made_at() {
    const READ: u64 = 27_631_529_984;
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let mut plan = plan_given(&q4, &gate, 4096, 1, PLAN_ROOM, None).expect("the plan");
    assert_eq!(plan.host.room, None);
    assert_eq!(plan.host.room_clause(), "");
    let planned = plan.plan_at(&(READ, HostRead::MemAvailable));
    assert_eq!(planned, READ - HOST_DRIFT);
    assert_eq!(
        plan.host.room,
        Some((READ, (planned, HostRead::MemAvailable)))
    );
    assert_eq!(
        plan.host.room_clause(),
        format!("; the plan was made at a reading of {READ} B (plan room {planned} B)")
    );
    let given = plan.plan_at(&(READ, HostRead::Given));
    assert_eq!(plan.host.room, Some((READ, (given, HostRead::Given))));
    assert_eq!(given, READ);
}

/// The seat's prompt-cache budget counts the arena once: `HostNeed::bytes`
/// carries it, and `CacheRam::of_tier` no longer subtracts it by hand, so
/// the budget's terms are the same bytes as the old by-hand subtraction —
/// `available − the arena-free need − pool − arena`, the need holding the
/// checkpoints its machine reserves — on every plan, paged or not. (The budget itself lives in gpu-gates'
/// `CacheRam`, a device crate the Mac cannot test whose reading of the
/// host's available bytes has no unit-test seam; this pins the identity its
/// terms rest on.)
#[test]
fn the_seats_cache_budget_counts_the_arena_once() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    for room in [27u64 << 30, 58 << 30] {
        let split = plan_at(&q4, &gate, 4096, 1, room).expect("the split plan");
        let need = HostNeed::of(&split, 0);
        assert_eq!(
            need.bytes(),
            need.plan_bytes() + need.arena,
            "room {room}: the arena rides in bytes, once"
        );
    }
}

/// The split leaves the prompt run-ahead's `2 W` page-cache window, a host
/// reserve of the split plan, and keeps the host experts in what the room
/// leaves past the arena and the window: on both rooms the reserve grows by
/// the window and the need stays inside the room past the arena, and at the
/// largest arena (27 GiB) the host keeps at most `1 W` of experts.
#[test]
fn a_split_leaves_the_run_ahead_window() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    for room in [27u64 << 30, 58 << 30] {
        let split = plan_at(&q4, &gate, 4096, 1, room).expect("the split plan");
        let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
        let w = (0..LAYERS)
            .map(|l| layer_split(&twin, l).2)
            .max()
            .expect("layers");
        assert_eq!(
            split.host.reserve_bytes,
            twin.host.reserve_bytes + 2 * w,
            "room {room}: the window is the split plan's reserve"
        );
        assert!(HostNeed::of(&split, 0).plan_bytes() + split.host.nvme_arena_bytes <= room);
        if room == 27 << 30 {
            assert!(
                split.host.expert_bytes <= w,
                "the largest arena's host keeps {} B of experts, past 1 W = {w} B",
                split.host.expert_bytes
            );
        }
    }
}

/// A churn pool leaves out the stacks the plan pages to the NVMe tier, whose
/// victims the tier's arena serves: on both paged gate plans every layer
/// pages and the pool at P 0 is empty, while the unsplit twin keeps its whole
/// pool. The unset rule resolves `mid-p0-s1` over that empty pool on both:
/// the plan that pages through an arena (27 GiB) and the one whose tier has
/// none, the mapping path (58 GiB). No plan clause precedes the family's own
/// rule, since a promotion reads the arena's slots, not the file mapping.
#[test]
fn a_paged_stack_adds_nothing_to_the_churn_pool() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    for (room, arena) in [(27u64 << 30, true), (58 << 30, false)] {
        let split = plan_at(&q4, &gate, 4096, 1, room).expect("the split plan");
        let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
        assert!(
            (0..LAYERS).all(|l| layer_split(&split, l).1 > 0),
            "room {room}: every layer pages"
        );
        let pool = |p: &Plan<'_>| ChurnPool::of(p, 0, 0).expect("the pool").bytes;
        assert_eq!(pool(&split), 0, "room {room}: the paged plan's pool");
        assert!(pool(&twin) > 0, "room {room}: the unsplit twin's pool");
        assert_eq!(
            split.host.nvme_arena_bytes > 0,
            arena,
            "room {room}: the arena"
        );
        let (pick, bytes) = residency(&split);
        assert_eq!(
            (pick.word(), bytes),
            ("mid-p0-s1".to_string(), 0),
            "room {room}: the unset rule on the paged plan"
        );
        if arena {
            // PIN(2026-10-11): `plan (a)` over the plan's 92 fewest card
            // experts a layer (the 3090 gate plan at ctx 4096 and 27 GiB:
            // P 0 needs the pool empty, the headroom at least 0 and fewest
            // at least 2, the spares and one that moves).
            // PIN(2026-10-25): 93 (was 92): the ubatch arena's flash rows
            // leave the stage's scratch, 100,663,296 B at 4,096 [derived:
            // this plan].
            // PIN(2026-10-30): 95 (was 93): the delta layers' one state lane and verify record
            // (`Kv38`) and the box gate's stage scratch (`stage`) [derived: this plan].
            assert_eq!(
                pick,
                Residency38Pick {
                    pinned: Some(0),
                    why: Residency38Why::PlanA { fewest: 95 },
                },
                "room {room}: the unset rule on the plan that pages through an arena"
            );
        }
    }
}

/// The serving seats' paged rule reads the plan's RAM arena
/// (`bloomery_levers::paged_columns`): the 3090 gate plan at a 27 GiB room
/// pages its host experts through an arena at two slots and at one, and an
/// unset draft goes off with that count kept; at 40 GiB (over half the host
/// leg, under its need) the tier pages experts with no arena (the mapping
/// path), so the same unset draft is served as asked — the serve gate's two
/// rooms.
#[test]
fn the_paged_rule_reads_the_arena() {
    use bloomery_levers::{Paged, PagedAt, paged_columns};
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let arena = |slots: u64, room: u64| {
        let plan = plan_at(&q4, &gate, 4096, slots, room).expect("the plan");
        (plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes)
    };
    let at = |arena, draft| PagedAt {
        arena,
        draft,
        draft_set: false,
    };
    let (moved2, deep2) = arena(2, 27 << 30);
    let (moved1, deep1) = arena(1, 27 << 30);
    println!("27 GiB: two slots arena {deep2} B, one slot arena {deep1} B");
    assert!(moved2 > 0 && deep2 > 0 && moved1 > 0 && deep1 > 0);
    assert_eq!(paged_columns(&at(deep2, true)), Paged::Reduced);
    assert_eq!(paged_columns(&at(deep1, true)), Paged::Reduced);
    assert_eq!(paged_columns(&at(deep2, false)), Paged::AsAsked);
    let (moved, shallow) = arena(2, 40 << 30);
    println!("40 GiB: two slots NVMe experts {moved} B, arena {shallow} B");
    assert!(moved > 0, "the 40 GiB room pages experts");
    assert_eq!(shallow, 0, "through the mapping, no arena");
    assert_eq!(paged_columns(&at(shallow, true)), Paged::AsAsked);
}

/// The arena's floor counts a column a slot, to the step port's width: the
/// paged gate plan at plans of 1, 2, 3, 4, 8 and 9 slots holds 37, 37, 37,
/// 45, 85 and 85 slots a layer (`k = 10` ids a column beside the 5 lane
/// slots, against the union's `2 × 16 + 5`), the dial's arena at least the
/// floor.
#[test]
fn the_arena_floor_counts_the_plans_slots() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    for (slots, columns, want) in [
        (1_u64, 1_u64, 37_u64),
        (2, 2, 37),
        (3, 3, 37),
        (4, 4, 45),
        (8, 8, 85),
        (9, 8, 85),
    ] {
        let plan = plan_at(&q4, &gate, 1024, slots, 27 << 30).expect("the plan");
        let floor = ArenaFloor::of(&plan).expect("the floor");
        assert_eq!(
            (floor.columns, floor.slots),
            (columns, want),
            "{slots} slots: {floor}"
        );
        assert_eq!(floor.bytes, want * floor.slot_set, "{slots} slots");
        assert!(
            plan.host.nvme_arena_bytes >= floor.bytes,
            "{slots} slots: the arena {} B holds the floor {} B",
            plan.host.nvme_arena_bytes,
            floor.bytes
        );
    }
}

/// A host of 58 GiB (a 64 GB machine's room): the room covers over half the
/// host leg, so the dial keeps R1's split — no arena — and the overflow is
/// the smaller one.
#[test]
fn a_58_gib_room_puts_the_overflow_on_the_nvme_tier() {
    let big = nvme_arm(58 << 30);
    let small = nvme_arm(27 << 30);
    assert_eq!(big.arena, 0, "the dial keeps R1's split on a wide room");
    assert!(small.arena > 0);
    assert!(0 < big.moved && big.moved < small.moved);
    assert!(big.r_min > small.r_max);
}

/// A room that holds the plan's host need is no case for the tier: the plan
/// is the unsplit plan's, byte for byte, and one byte under the need moves
/// experts.
#[test]
fn a_room_that_holds_the_need_keeps_the_plan_unchanged() {
    let q4 = model(false);
    let gate = machine_a(RTX_3090);
    let box_plan = plan_at(&q4, &gate, 4096, 1, PLAN_ROOM).expect("the box plan");
    let need = HostNeed::of(&box_plan, 0).bytes() - PLE_TABLE + ROW_ROOM;
    for room in [PLAN_ROOM, need + 1, need] {
        let kept = plan_at(&q4, &gate, 4096, 1, room).expect("the plan");
        let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
        assert_eq!(format!("{kept:?}"), format!("{twin:?}"), "room {room}");
    }
    let under = plan_at(&q4, &gate, 4096, 1, need - 1).expect("the plan");
    assert!(under.host.nvme_expert_bytes > 0);
    assert!(HostNeed::of(&under, 0).bytes() < need);
}

/// A room under the floor is refused by name, with the room, the floor, the
/// need and the arena's least the floor counts; the floor itself is planned,
/// with the least arena.
// PIN(2026-10-09): the floor is `base + 3 W` plus the arena's least on a
// plan with a deep side (was `base + 3 W` alone, which planned arenas of one
// to ten slots a paged layer at the rooms just above it, overflowed by the
// first decode call's ten ids): `ArenaDial::room_floor`.
#[test]
fn a_room_under_the_floor_is_refused_by_name() {
    let arm = nvme_arm(27 << 30);
    let q4 = model(false);
    let gate = machine_a(RTX_3090);
    let twin = plan_with(&q4, &gate, 4096, 1, 27 << 30, false).expect("the unsplit plan");
    let least = dial_of(&twin, 0).least;
    let floor = arm.floor + least.bytes;
    match plan_at(&q4, &gate, 4096, 1, floor - 1) {
        Err(PlacementError::HostRoomFloor {
            room,
            floor: f,
            need,
            arena,
            ..
        }) => assert_eq!(
            (room, f, need, arena),
            (floor - 1, floor, arm.old_need, Some(least))
        ),
        other => panic!("not refused by name: {:?}", other.map(|p| p.host)),
    }
    let msg = plan_at(&q4, &gate, 4096, 1, floor - 1)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(msg.contains(&(floor - 1).to_string()) && msg.contains(&floor.to_string()));
    let at = nvme_arm(floor);
    assert!(at.r_min <= at.r_max && at.moved > 0);
    assert!(at.max_unit > 0);
    assert_eq!(
        at.arena, least.bytes,
        "the floor's room plans the least arena"
    );
}

/// The arena's floor counts a routed stack of NVFP4 rows at the windows a
/// fill of a checkpoint folder's two planes a matrix takes: its codes'
/// window and its block scales' ([`arena_part_bytes`]: the codes 8 of a
/// group's 9 bytes, the scales the ninth), as the arena carves it. Every
/// other type stays one window a part. FAIL-first: a planes stack counted at
/// one window over its bytes leaves the floor two or three pages a part
/// under the carve, and the carve at the floor's room under the floor's
/// slots.
#[test]
fn the_floor_counts_an_nvfp4_rows_stack_at_its_two_planes_windows() {
    let mut fp4 = model(false);
    for t in &mut fp4.tensors {
        if t.role == Role::RoutedExperts {
            t.ty = GgmlType::NVFP4_ROWS;
            t.file_bytes = t.dims.iter().product::<u64>() / 16 * 9;
        }
    }
    let gate = machine_a(RTX_3090);
    let twin = plan_with(&fp4, &gate, 4096, 1, 27 << 30, false).expect("the unsplit plan");
    let want: u64 = fp4
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts && t.layer.is_some())
        .map(|t| {
            let blocks = t.file_bytes / fp4.experts / 9;
            arena_part_bytes(8 * blocks, blocks)
        })
        .sum();
    let least = ArenaFloor::of(&twin).expect("the arena's floor");
    assert_eq!(least.slot_set, want);
    let one: u64 = fp4
        .tensors
        .iter()
        .filter(|t| t.role == Role::RoutedExperts && t.layer.is_some())
        .map(|t| arena_window_bytes(t.file_bytes / fp4.experts))
        .sum();
    assert!(least.slot_set > one, "two windows a part over one");
}

/// The arena's floor on the gate plan: the larger of ten ids a call (the
/// routed width, one column) with the residency lane's five slots
/// (`ARENA_LANE_SLOTS`: four open reads and a victim claim) beside them, and
/// a prompt union in batches (two batches of `UNION_BATCH_IDS` pinned at
/// once and the lane's slots), over one slot of
/// each of the 48 layers — every part of the Q4 file a whole number of
/// pages, so a slot set is the experts' bytes and two pages a part. A room
/// that leaves the arena fewer slots a paged layer — `k − 1`, or the floor's
/// own count less one — is refused at plan time by name, naming the slots,
/// the ids a call, the lane slots, the union's slots and the bytes, where
/// the old dial planned both and the first decode call overflowed them. The
/// floor's own room plans an arena that carves exactly its slots.
///
/// PIN(2026-10-12): the floor is 37 slots, not 33 (2026-10-11: 11) — a union
/// run in batches pins two batches while the next is filled and the lane
/// holds `LANE_THREADS + 1` slots, four open reads and a victim claim
/// (2 × 16 + 5 = 37 > 10 × 1 + 5 = 15): a carve of 15 to 36 slots can load,
/// then refuse a claim beside the pins (`TierError::BatchFloor`).
#[test]
fn a_room_that_leaves_under_the_floors_slots_is_refused_by_name() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let twin = plan_with(&q4, &gate, 4096, 1, 27 << 30, false).expect("the unsplit plan");
    let dial = dial_of(&twin, 0);
    let least = dial.least;
    assert_eq!(
        (
            least.k,
            least.columns,
            least.lane_slots,
            least.union,
            least.slots
        ),
        (10, 1, 5, 37, 37)
    );
    let units: u64 = (0..LAYERS).map(|l| layer_unit(&q4, l)).sum();
    assert_eq!(least.slot_set, units + LAYERS as u64 * 3 * 2 * ARENA_ALIGN);
    assert_eq!(least.bytes, 37 * least.slot_set);
    let floor = dial.floor().expect("the floor");
    for slots in [least.k - 1, least.slots - 1] {
        let room = floor + slots * least.slot_set;
        let planned = plan_at(&q4, &gate, 4096, 1, room).map(|p| p.host.nvme_arena_bytes);
        let e = match planned {
            Err(e) => e,
            Ok(arena) => panic!("{slots} slots' room planned an arena of {arena} B"),
        };
        match &e {
            PlacementError::HostRoomFloor {
                room: r,
                floor: f,
                arena: Some(a),
                ..
            } => assert_eq!((*r, *f, *a), (room, floor + least.bytes, least)),
            other => panic!("{slots} slots: not the floor's refusal: {other}"),
        }
        let text = e.to_string();
        for part in [
            "37 slots a paged layer".to_string(),
            "10 × 1 ids a call + 5 lane slots".to_string(),
            "37 for a union in batches".to_string(),
            format!("= {} B", least.bytes),
            "BLOOMERY_NVTIER_BYTES=0".to_string(),
        ] {
            assert!(text.contains(&part), "{part:?} in {text}");
        }
    }
    let at = plan_at(&q4, &gate, 4096, 1, floor + least.bytes).expect("the floor's own room");
    let carve = ArenaFloor::of(&at).expect("the split plan's floor");
    assert_eq!(at.host.nvme_arena_bytes, least.bytes);
    assert_eq!(
        carve, least,
        "every layer pages: the split plan's floor is the twin's"
    );
    assert_eq!(at.host.nvme_arena_bytes / carve.slot_set, least.slots);
}

/// A given `BLOOMERY_NVTIER_BYTES` under the arena's floor is refused by
/// name with the value and the floor; the floor itself and 0 are taken. A
/// given 0 pages through the file mapping at a room the dial refuses
/// unset (between the tier's floor and its least arena above it), and under
/// the tier's floor even a given 0 is refused, the floor then holding no
/// arena term.
#[test]
fn a_given_arena_under_the_floor_is_refused_by_name() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let twin = plan_with(&q4, &gate, 4096, 1, 27 << 30, false).expect("the unsplit plan");
    let dial = dial_of(&twin, 27 << 30);
    let least = dial.least;
    let floor = dial.floor().expect("the floor");
    let e = nvme_arena_of(&dial, Some(least.bytes - 1)).expect_err("under the floor");
    match &e {
        PlacementError::NvTierArenaUnder { arena, least: l } => {
            assert_eq!((*arena, *l), (least.bytes - 1, least));
        }
        other => panic!("not refused by name: {other}"),
    }
    let text = e.to_string();
    for part in [
        format!("{} B (BLOOMERY_NVTIER_BYTES)", least.bytes - 1),
        format!("set at least {} B", least.bytes),
        // PIN(2026-10-12): 37 slots, not 33: the lane holds five slots
        // (`ARENA_LANE_SLOTS`), the union's 2 × 16 + 5.
        "37 slots a paged layer".to_string(),
    ] {
        assert!(text.contains(&part), "{part:?} in {text}");
    }
    let take = |d: &ArenaDial, given| nvme_arena_of(d, given).expect("taken");
    assert_eq!(take(&dial, Some(least.bytes)), least.bytes);
    assert_eq!(take(&dial, Some(0)), 0);
    let low = ArenaDial {
        room: floor + least.bytes - 1,
        ..dial
    };
    assert_eq!(take(&low, Some(0)), 0, "a given 0: the mapping path");
    assert!(matches!(
        nvme_arena_of(&low, None),
        Err(PlacementError::HostRoomFloor { arena: Some(_), .. })
    ));
    let under = ArenaDial {
        room: floor - 1,
        ..dial
    };
    match nvme_arena_of(&under, Some(0)) {
        Err(PlacementError::HostRoomFloor {
            floor: f,
            arena: None,
            ..
        }) => assert_eq!(f, floor),
        other => panic!("under the tier's floor: {other:?}"),
    }
}

/// The split reads no environment: its arena is the `given` it is passed. The
/// test re-runs itself, marked by libtest's `--test-threads=1`, in a child
/// process whose `BLOOMERY_NVTIER_BYTES` is not bytes, which the lever's own
/// read refuses by name; there a plan with a given arena is planned as it is,
/// where a second read of the lever inside the split would refuse it or
/// replace the arena.
#[test]
fn the_split_reads_no_environment() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let room = 27 << 30;
    let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
    let least = dial_of(&twin, room).least.bytes;
    if std::env::args().any(|a| a == "--test-threads=1") {
        assert!(
            bloomery_levers::nvtier_levers().is_err(),
            "the lever read must refuse a value that is not bytes"
        );
        let split = plan_given(&q4, &gate, 4096, 1, room, Some(Some(least)))
            .expect("a given arena is planned whatever the environment holds");
        assert_eq!(split.host.nvme_arena_bytes, least);
        return;
    }
    let (_, path) = module_path!().split_once("::").expect("a crate path");
    let out = std::process::Command::new(std::env::current_exe().expect("the test binary"))
        .args([
            "--exact",
            &format!("{path}::the_split_reads_no_environment"),
        ])
        .arg("--test-threads=1")
        .env(bloomery_levers::NVTIER_BYTES, "not-bytes")
        .output()
        .expect("the child test run");
    assert!(
        out.status.success() && String::from_utf8_lossy(&out.stdout).contains("1 passed"),
        "the child run: {}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}

/// The dial's one step on the gate plan, through the planner: the last room
/// with an arena is `H/2 − 1`, where the arena is the deep side's largest,
/// `H/2 − 1 − F`, and at `H/2` the plan pages through the file mapping with
/// none.
#[test]
fn the_dials_one_step_is_at_half_the_host_leg() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let twin = plan_with(&q4, &gate, 4096, 1, 27 << 30, false).expect("the unsplit plan");
    let dial = dial_of(&twin, 0);
    let floor = dial.floor().expect("the floor");
    let last = dial
        .step_room()
        .expect("the step")
        .expect("the gate plan has a deep side");
    assert_eq!(last, twin.host.expert_bytes / 2 - 1);
    let at = |room| {
        let plan = plan_at(&q4, &gate, 4096, 1, room).expect("the plan");
        assert!(plan.host.nvme_expert_bytes > 0, "room {room} pages");
        plan.host.nvme_arena_bytes
    };
    assert_eq!(at(last), last - floor);
    assert_eq!(at(last + 1), 0);
    assert!(last - floor >= dial.least.bytes);
}

/// A unified card plans the arena a discrete one does beside the same card
/// bytes: the gate plan at host room `R`, and the same machine on a unified
/// pool at `R` plus the cards' share of it, take one arena at 4,096
/// positions for one slot and for two, the deep side's at 27 GiB and none
/// at 40 GiB.
#[test]
fn a_unified_card_reads_the_host_room_past_its_share() {
    let q4 = model(false);
    let discrete = machine_a(RTX_3090);
    let mut unified = machine_a(RTX_3090);
    *unified.unified_mut() = Some(UnifiedPool { bytes: 1 << 40 });
    for slots in [1, 2] {
        for (gib, deep) in [(27u64, true), (40, false)] {
            let room = gib << 30;
            let twin =
                plan_with(&q4, &unified, 4096, slots, room, false).expect("the unsplit plan");
            let cards = HostNeed::of(&twin, 0).cards;
            assert!(cards > 0, "the unified plan counts its card in the pool");
            let arena = |m, room| {
                plan_at(&q4, m, 4096, slots, room)
                    .expect("the split plan")
                    .host
                    .nvme_arena_bytes
            };
            let want = arena(&discrete, room);
            assert_eq!(want > 0, deep, "{slots} slots, {gib} GiB: {want} B");
            assert_eq!(
                arena(&unified, room + cards),
                want,
                "{slots} slots, {gib} GiB plus the cards' {cards} B"
            );
        }
    }
}

/// The dial at the gate plan's and the A6000 plan's terms (one slot, 4,096
/// positions), every room from the floor to the need on a 64 MiB grid and at
/// each breakpoint and its neighbours: refused only under the floor and its
/// least arena; never an arena under the least; on the deep side a byte of
/// room moves the arena by one byte, and the one step is at `H/2`, the deep
/// side's largest to none; the deep side's room the room past the floor
/// (27 GiB on the gate plan), and the A6000 at 27 GiB none.
#[test]
fn the_dial_moves_a_byte_a_byte_but_at_its_step() {
    let q4 = model(false);
    for (name, spec) in [("3090", RTX_3090), ("A6000", A6000)] {
        let m = machine_a(spec);
        let twin = plan_with(&q4, &m, 4096, 1, 27 << 30, false).expect("the unsplit plan");
        let dial = dial_of(&twin, 0);
        let least = dial.least;
        let floor = dial.floor().expect("the floor");
        let half = dial.held / 2;
        let last = dial.step_room().expect("the step").expect("a deep side");
        let arena = |room| nvme_arena_of(&ArenaDial { room, ..dial }, None).ok();
        let mut rooms: Vec<u64> = (floor..dial.need).step_by(64 << 20).collect();
        for at in [floor, floor + least.bytes, half, last] {
            rooms.extend([at - 1, at, at + 1]);
        }
        for room in rooms {
            let (here, next) = (arena(room), arena(room + 1));
            assert_eq!(
                here.is_none(),
                room < floor + least.bytes,
                "{name}: room {room} refused"
            );
            let Some(a) = here else { continue };
            assert!(a == 0 || a >= least.bytes, "{name}: room {room}: {a} B");
            let b = next.expect("a room past a planned one plans");
            if room == last {
                assert_eq!((a, b), (last - floor, 0), "{name}: the step");
            } else if room < last {
                assert_eq!(b, a + 1, "{name}: room {room}: {a} → {b}");
            } else {
                assert_eq!((a, b), (0, 0), "{name}: room {room} past the step");
            }
        }
        let gib27 = arena(27 << 30).expect("27 GiB plans");
        match name {
            "3090" => assert_eq!(gib27, (27 << 30) - floor),
            _ => assert_eq!(gib27, 0),
        }
    }
}

/// The split dial's own rule ([`super::nvme_arena_of`]) on round terms: the
/// room past the floor up to half the host leg, none from there; a value
/// the lever gives is taken as it is, one that pushes the host segments
/// under the floor is refused by name with the room, the floor and the
/// value.
#[test]
fn the_split_dial_picks_its_arena_or_refuses_a_too_large_one() {
    const G: u64 = 1 << 30;
    // PIN(2026-10-12): a slot set of G / 16, a floor of 37 slots = 2.3125 G
    // (the floor carries the batched union's 2 × 16 + 5 slots; 33 slots =
    // 2.0625 G before, 11 slots = 1.375 G at G / 8 earlier), and the
    // given arena below is 3 G, not 2 G: the test's subject is that a given
    // arena at or over the floor is taken as it is.
    let least = ArenaFloor::new(10, 1, ARENA_LANE_SLOTS, G / 16).expect("the floor");
    let dial = |room| ArenaDial {
        room,
        base: G,
        cards: 0,
        heavy: G,
        held: 60 * G,
        need: 61 * G,
        least,
    };
    let of = |room, given| nvme_arena_of(&dial(room), given).expect("the dial picks an arena");
    let floor = 4 * G;
    // The default: the deep split takes the largest arena that fits.
    assert_eq!(of(27 * G, None), 23 * G);
    // The boundary itself: exactly half the host leg takes no arena.
    assert_eq!(of(30 * G - 1, None), 26 * G - 1);
    assert_eq!(of(30 * G, None), 0);
    // A room that covers over half the host leg keeps the split's segments.
    assert_eq!(of(58 * G, None), 0);
    // A given arena is taken as it is, up to the largest that fits.
    assert_eq!(of(27 * G, Some(3 * G)), 3 * G);
    assert_eq!(of(27 * G, Some(23 * G)), 23 * G);
    // One that pushes the host segments under the floor is refused by name.
    let room = 27 * G;
    match nvme_arena_of(&dial(room), Some(room - floor + 1)) {
        Err(PlacementError::NvTierArena {
            arena,
            room: r,
            floor: f,
        }) => {
            assert_eq!((arena, r, f), (room - floor + 1, room, floor));
        }
        other => panic!("not refused by name: {:?}", other),
    }
    let msg = nvme_arena_of(&dial(room), Some(u64::MAX))
        .unwrap_err()
        .to_string();
    assert!(msg.contains(&room.to_string()) && msg.contains(&floor.to_string()));
}

/// `BLOOMERY_HOST_ROOM`'s reading: a set value is a room the caller gave; an
/// unset one leaves the machine's reading; anything else is refused naming
/// the lever and the value.
#[test]
fn the_host_room_lever_reads_bytes_or_is_refused_by_name() {
    use std::ffi::OsStr;
    let read = |v: Option<&str>| super::host_room_given("BLOOMERY_HOST_ROOM", v.map(OsStr::new));
    assert_eq!(read(None), Ok(None));
    assert_eq!(read(Some("27G")), Ok(Some((27 << 30, HostRead::Given))));
    assert_eq!(read(Some("512M")), Ok(Some((512 << 20, HostRead::Given))));
    assert_eq!(
        read(Some("28_991_029_248")),
        Ok(Some((28_991_029_248, HostRead::Given)))
    );
    for bad in [
        "",
        "abc",
        "-1",
        "1.5G",
        "27 G",
        "27g",
        "27GiB",
        "99999999999999999999G",
    ] {
        let e = read(Some(bad)).expect_err(bad);
        assert!(
            e.contains("BLOOMERY_HOST_ROOM") && e.contains(&format!("{bad:?}")),
            "{e}"
        );
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let e =
            super::host_room_given("BLOOMERY_HOST_ROOM", Some(OsStr::from_bytes(&[0xff, b'G'])))
                .expect_err("not UTF-8");
        assert!(
            e.contains("BLOOMERY_HOST_ROOM") && e.contains("UTF-8"),
            "{e}"
        );
    }
}

/// The Qwen3.8 seat's drafted plan (`fixed` the draft's card bytes, its
/// store 2,048 B a position) or its plain one (`fixed` none) on a DGX Spark
/// (`free` the census's pool, `room` `MemAvailable`) at `ctx` positions for
/// the default slot, the card rule's, then the PLE table's tier (kept on the
/// host only beside two saved states) and the NVMe expert tier: whether it
/// holds, or the memory it is short of.
fn spark_holds(
    model: &ModelTensors,
    (free, room): (u64, u64),
    ctx: u64,
    fixed: Option<u64>,
) -> Result<super::ctx_ask::Probe<()>, PlacementError> {
    let spec = super::workstation::gb10(free);
    let mut machine = machine_a(spec);
    *machine.host_mut() = super::workstation::host_of(&spec);
    *machine.unified_mut() = super::workstation::unified_of(&spec);
    machine
        .host_mut()
        .reserves
        .push(("checkpoints".to_string(), 4 << 30));
    let terms = SeqTerms {
        layers: Stores {
            kv: &Kv38,
            count: LAYERS,
        },
        draft: None,
        beside: BESIDE,
    };
    let rows = terms.plan_beside(1);
    let kv = terms.slots_of(1);
    let draft = fixed.map_or(0, |f| f + 2048 * ctx);
    let planned = plan_routed_reserving(
        model,
        &machine,
        ctx,
        &kv,
        &PlanLevers::default(),
        routed_rule(Program::Qwen38Body),
        rows + draft,
    )
    .and_then(|mut plan| {
        let floor = 2 * state38(ctx, fixed.is_some());
        super::row_table_tier(
            &mut plan,
            room,
            floor,
            runtime::stores::reader_columns(UBATCH_PLANNED.min(ctx)),
            ROW_PAGE,
            super::RowTableSource::Split,
        )?;
        super::expert_nvme_tier(&mut plan, room, None)?;
        plan.cards[0].grow_kv(rows);
        Ok(plan)
    });
    match planned {
        Ok(plan) if plan.violations().is_empty() => Ok(super::ctx_ask::Probe::Holds(())),
        Ok(plan) => panic!("ctx {ctx}: {:?}", plan.violations()),
        Err(e) => e.card_short().map(super::ctx_ask::Probe::Short).ok_or(e),
    }
}

/// The Qwen3.8 seat's default on a DGX Spark with no flag, both files at
/// their own readings (the census's free pool, `MemAvailable`): the default's
/// target, the trained 262,144 positions for the one default slot, with the draft on —
/// the drafted plan holds the target, so the draft costs no position and does
/// not yield. The prompt cache's budget does not lower it: the cache keeps
/// what its budget holds at that context.
///
/// PIN(2026-10-14): the default is the seats' ladder (ctx-first, the user approved
/// D1, D5 and D6). UD-Q4 was 131,072, the first rung whose saved state the
/// cache's default budget held; UD-Q3 stays 262,144.
#[test]
fn a_spark_defaults_to_the_ladders_target() {
    use super::ctx_ask::{Ask, DEFAULT_SLOTS, Limit, Rule, Steps, choose};
    // The draft's fixed card bytes close each file's need at 262,144
    // [derived: the need less every other term]; UD-Q3's is 68,736 B more, the
    // draft borrowing the target's output matrix.
    for (q3, free, room, fixed) in [
        (true, 125_309_718_528, 125_124_673_536, 2_839_606_572_u64),
        (false, 125_394_333_696, 125_111_492_608, 2_839_537_836),
    ] {
        let model = model(q3);
        let limit = Limit {
            positions: 262_144,
            source: "trained",
        };
        let steps = Steps {
            floor: 4096,
            step: 1024,
        };
        let pick = |fixed: Option<u64>| {
            choose(Ask::Default, DEFAULT_SLOTS, limit, steps, |c| {
                spark_holds(&model, (free, room), c, fixed)
            })
            .expect("a Spark default")
        };
        for fixed in [Some(fixed), None] {
            let chosen = pick(fixed);
            assert_eq!(
                (chosen.slot_ctx, chosen.rule),
                (262_144, Rule::Target),
                "q3 {q3}, drafted {}",
                fixed.is_some()
            );
        }
    }
}

/// The MTP draft's fixed card bytes [derived: 3,180,890,540 B, the draft's
/// card bytes in the gate's small-card row at 65,536 positions, less its
/// store's 2,048 B a position].
const MTP_FIXED: u64 = 3_046_672_812;

/// The Qwen3.8 seat's plan on the 3090 of `slots` resident sequences of `ctx`
/// positions a slot beside the MTP draft when `fixed` names its card bytes
/// (`planned_drafting`), as a default context's probe reads it: the card's
/// experts a layer when it holds, the memory it is short of when it does not,
/// and with `least` a plan that leaves a layer fewer card experts is short
/// too ([`card_experts_floor`]).
fn default_probe(
    model: &ModelTensors,
    (slots, ctx): (u64, u64),
    fixed: Option<u64>,
    least: Option<u64>,
) -> Result<Probe<Vec<u64>>, PlacementError> {
    let machine = machine_a(RTX_3090);
    let plan = match planned_drafting(model, &machine, ctx, slots, PLAN_ROOM, Some(None), fixed) {
        Ok(plan) => plan,
        Err(e) => return e.card_short().map(Probe::Short).ok_or(e),
    };
    let broken = plan.violations();
    if let Some(short) = short_of(&broken) {
        return Ok(Probe::Short(short));
    }
    assert!(broken.is_empty(), "ctx {ctx} slots {slots}: {broken:?}");
    match least.map(|least| card_experts_floor(&plan, 0, least)) {
        Some(Err(e)) => e.card_short().map(Probe::Short).ok_or(e),
        _ => Ok(Probe::Holds(plan.n_l)),
    }
}

/// An unset context leaves the unset residency its experts. On the 3090, at
/// one and two slots, the fit that reads [`card_experts_floor`] takes the
/// largest context whose plan leaves every layer
/// [`residency38_least_fewest`] card experts, and the plan one step above
/// does not. Where the plain default already leaves them (one slot, the
/// trained 262,144 positions at the target [measured: the gate's `ctx` line
/// on the 3090]) the choice is the same; where it spends the card on context
/// (two slots) the floor costs under a tenth of it [derived: about 3 % at two
/// slots with the draft, 173K to 168K].
/// FAIL-first: the fit without the floor (`least` dropped from the probe),
/// where two slots with the draft spend the card on context, leaves a layer
/// one card expert (the gate's `fewest (1)`) and turns the first assertion
/// red.
#[test]
fn a_default_context_on_the_3090_leaves_the_residency_floor() {
    use super::ctx_ask::{Ask, Limit, Rule, Steps, choose};
    let q4 = model(false);
    let least = residency38_least_fewest();
    let limit = Limit {
        positions: 262_144,
        source: "trained",
    };
    let steps = Steps {
        floor: 4096,
        step: 1024,
    };
    let fit = |slots: usize, fixed: Option<u64>, least: Option<u64>| {
        let n = u64::try_from(slots).expect("slots");
        choose(Ask::Default, slots, limit, steps, |c| {
            default_probe(&q4, (n, c), fixed, least)
        })
        .expect("a default")
    };
    let fewest = |n_l: &[u64]| n_l.iter().copied().filter(|&n| n > 0).min().unwrap_or(0);
    for (slots, fixed) in [(1, None), (2, None), (2, Some(MTP_FIXED))] {
        let (plain, floored) = (fit(slots, fixed, None), fit(slots, fixed, Some(least)));
        eprintln!(
            "3090, {slots} slot(s), draft {}: plain ctx {} (fewest {}, {:?}), floored ctx {} \
             (fewest {}, {:?})",
            fixed.is_some(),
            plain.slot_ctx,
            fewest(&plain.plan),
            plain.rule,
            floored.slot_ctx,
            fewest(&floored.plan),
            floored.rule,
        );
        assert!(
            floored.plan.iter().all(|&n| n >= least),
            "{slots} slot(s), draft {}: a layer under {least} card experts: {:?}",
            fixed.is_some(),
            floored.plan
        );
        assert!(
            floored.slot_ctx <= plain.slot_ctx,
            "the floor only takes context away"
        );
        // With the draft, this probe's plain fit spends a layer down to no
        // card expert, which the seat does not (the gate's `fewest (1)` at
        // 173,056), so its plain context is not the seat's to compare with.
        if fixed.is_none() {
            assert!(
                plain.slot_ctx - floored.slot_ctx <= plain.slot_ctx / 10,
                "the floor costs over a tenth of the context"
            );
        }
        if fewest(&plain.plan) >= least {
            assert_eq!(
                (floored.slot_ctx, floored.rule),
                (plain.slot_ctx, plain.rule),
                "a plan that leaves the experts needs no floor"
            );
        }
    }
    let one = fit(1, None, Some(least));
    assert_eq!(
        (one.slot_ctx, one.rule),
        (262_144, Rule::Target),
        "one slot and no draft keep the trained context"
    );
}

#[test]
fn saved_state_and_reader_count_the_load_shape() {
    let point = CHECKPOINT_38;
    assert_eq!(state38(1, false), 12 * (2048 + 256) + point);
    let positions: u64 = 16_384;
    let attention = 12 * (positions * 2048 + positions.div_ceil(4) * 256);
    assert_eq!(state38(positions, false), attention + 5 * point);
    assert_eq!(
        state38(positions, true),
        attention + 5 * point + positions * 2048 + ROWS + 4 * 4 * 2560 * 4
    );
    let mut model = model(false);
    let table = model
        .tensors
        .iter_mut()
        .find(|t| t.role == Role::EngramTable)
        .unwrap();
    table.ty = GgmlType::NVFP4_ROWS;
    table.dims[0] = 64;
    table.file_bytes = table.dims.iter().product::<u64>() / 16 * 9;
    let reserve = |ubatch| {
        super::row_room(
            &model,
            runtime::stores::reader_columns(ubatch),
            ROW_PAGE,
            super::RowTableSource::Parts,
        )
        .unwrap()
    };
    assert_eq!(reserve(4096) - reserve(512), 896 << 20);
    assert_eq!(reserve(1), reserve(8));
}
