//! Qwen3.8-Flash-Next (`qwen4exp`) planned on cards larger than this
//! workstation's: the real planner over the file's own tensor table
//! (`tests/qwen38_q4_tensors.tsv`), with the machine and cache terms of
//! `bloomery-model`'s `arch::qwen35moe::place` restated here — that crate
//! does not build on the Mac. The restated terms are held to the box gate's
//! pins of the same file (`crates/model/tests/qwen4exp_meta.rs`) on the
//! A6000 first, so a term restated wrong fails here before any large-card
//! figure is read.

use bloomery_levers::{Residency38Pick, Residency38Why, residency38_at_plan};
use gguf::GgmlType;

use super::churn::ChurnPool;
use super::workstation::{
    A6000, ALIASES, CONTEXT, CONTEXT_SELF, CardSpec, DeviceInfo, GRANULE, HOST_USABLE, HostNeed,
    HostRead, MARGIN, MIB, OS_OTHER, RTX_3090, SCRATCH, TIER_BATCH_HOST_RESERVE,
    TIER_BATCH_RESERVE, census_usable, host, resolve, tier_batch_host_bytes,
    tier_batch_staging_bytes,
};
use super::{
    Card, CardFormat, Device, KvBytes, Machine, ModelTensor, ModelTensors, PlacementError, Plan,
    PlanLevers, Role, plan_routed_reserving,
};
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
fn model(q3: bool) -> ModelTensors {
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
    }
}

/// The card leg's routed stacks (`qwen35moe::place::card_routed`): a Q4_K,
/// Q5_K, IQ3_XXS or IQ4_XS gate and up, a Q5_1, Q8_0 or IQ4_NL down, in the
/// file's blocks.
fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    match ty {
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q5_1
        | GgmlType::Q8_0
        | GgmlType::IQ3_XXS
        | GgmlType::IQ4_XS
        | GgmlType::IQ4_NL => Some(CardFormat::KQuant),
        _ => None,
    }
}

/// The file's cache bytes a layer (`qwen35moe::place::KvLayout`) [derived
/// from `runtime::stores`]: a delta layer's state in four lanes of 48 heads
/// of 128 × 128 f32 with a u32 stamp each and its conv ring of 11 rows of
/// 10,240 f32, 13,033,488 B; the PLE layer's ring beside it, 17 rows of
/// 4 · 2,560 f32; a selecting layer, every fourth, its K, V and raw index
/// key, 2,304 B a position, and a pooled key, 256 B a pool of four.
struct Kv38;

impl KvBytes for Kv38 {
    fn layer_bytes(&self, layer: usize, ctx_max: u64) -> u64 {
        let ple = if layer == 1 { 17 * 4 * 2560 * 4 } else { 0 };
        let own = if layer % 4 == 3 {
            ctx_max * 2304 + ctx_max.div_ceil(4) * 256
        } else {
            13_033_488
        };
        own + ple
    }
}

/// A sequence's card bytes beside its stores
/// (`qwen35moe::place::slot_resident_bytes`): (1 + 8) rows of 4 · 2,560 f32
/// and a lane word.
const BESIDE: u64 = 9 * 4 * 2560 * 4 + 4;

/// The stage card of `qwen35moe::place::machine_for_experts(spec, 48, 4096,
/// Experts::Card)`: every layer, the head and the token embedding, its
/// scratch the m = 1 scratch, the ubatch walk's and its card route's at
/// 4,096 positions (the box gate's `CARD_ROUTE_SCRATCH` row).
fn stage(spec: CardSpec) -> Card {
    Card {
        name: spec.name.to_string(),
        device: spec.device,
        usable_bytes: spec.usable_bytes(),
        context_bytes: CONTEXT,
        scratch_bytes: 3_228_766_208,
        margin_bytes: MARGIN,
        granule_bytes: GRANULE,
        free_bytes: spec.free_bytes,
        held_by: spec.held_by,
        layers: 0..LAYERS,
        head: true,
        token_embedding: true,
        reserves: Vec::new(),
    }
}

/// `--place a`: the stage card alone beside the host.
fn machine_a(spec: CardSpec) -> Machine {
    Machine {
        cards: vec![stage(spec)],
        tiers: Vec::new(),
        host: host(),
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
            device: tier.device,
            usable_bytes: tier.usable_bytes(),
            context_bytes: CONTEXT,
            scratch_bytes: SCRATCH,
            margin_bytes: MARGIN,
            granule_bytes: GRANULE,
            free_bytes: tier.free_bytes,
            held_by: tier.held_by,
            layers: 0..0,
            head: false,
            token_embedding: false,
            reserves: vec![(
                TIER_BATCH_RESERVE.to_string(),
                tier_batch_staging_bytes(2560, 10, 4096) + 348_980_248,
            )],
        }],
        host: h,
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
    let mut plan = plan_routed_reserving(model, machine, ctx, &kv, &levers, card_routed, rows)?;
    super::row_table_tier(&mut plan, room, UBATCH_PLANNED)?;
    if experts_to_nvme {
        super::expert_nvme_tier(&mut plan, room)?;
    }
    plan.cards[0].kv_bytes += rows;
    let broken = plan.violations();
    assert!(broken.is_empty(), "ctx {ctx} slots {slots}: {broken:?}");
    Ok(plan)
}

/// The most positions one PLE fill reads (`qwen35moe::place::UBATCH_PLANNED`,
/// the largest ubatch a load runs).
const UBATCH_PLANNED: u64 = 4096;

/// The box's room with nothing loaded [derived from the measured `free -b`
/// figures: `HOST_USABLE` less `OS_OTHER`], and the room of the 64 GiB host
/// the sitting holds a Qwen3.8 arm to (`depth-qwen3moe`'s `mem=61G` scope:
/// `MemoryMax` 61 GiB, nothing charged yet).
const BOX_ROOM: u64 = HOST_USABLE - OS_OTHER;
const SCOPE_61G: u64 = 61 << 30;

/// The unset residency rule on `plan` with `MemAvailable` ample
/// (`residency38_at_plan`), and its churn pool's bytes.
fn residency(plan: &Plan<'_>) -> (Residency38Pick, u64) {
    let pool = |pinned| ChurnPool::of(plan, 0, pinned).map(|p| p.bytes);
    let pick = residency38_at_plan(
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
/// the box gate's A6000
/// row at the same context and ubatch (`CARD_PLANS`: 262 experts on the
/// first 40 layers, 261 on the rest, 39,385,907,200 B).
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
    let a6000 = plan_at(&q4, &m, 4096, 1, BOX_ROOM).expect("the A6000 plan");
    let want: Vec<u64> = (0..LAYERS)
        .map(|l| if l < 40 { 262 } else { 261 })
        .collect();
    assert_eq!(
        (&a6000.n_l, a6000.cards[0].expert_bytes),
        (&want, 39_385_907_200)
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
    let plan = plan_at(&q4, &m, 4096, 1, BOX_ROOM).expect("the 96 GB plan");
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
/// IQ4_NL or Q8_0: all of them card types (`card_routed`), so on a large
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
    let plan = plan_at(&q3, &m, 4096, 1, BOX_ROOM).expect("the Q3 plan");
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
    let host = at(BOX_ROOM);
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
    // The host experts are the routed share less the 13,899,033,600 B the
    // 3090 holds (the iqplan's 64 GiB plan holds 10,967,959,552 B beside the
    // draft's 2,919,229,612 B reserve, within 12 MB of the same budget); the
    // host arm adds the table to them, the NVMe arm the row room (the OS
    // reserve is MemAvailable's own and leaves the need).
    assert_eq!(
        (host.host.expert_bytes, host_need, nvme_need),
        (41_924_531_200, 70_724_669_440, 42_461_402_112)
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
    let mut moved = at(BOX_ROOM);
    assert!(matches!(
        super::row_table_tier(&mut moved, BOX_ROOM, UBATCH_PLANNED),
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
        plan_at(&q3, &bp, 4096, 1, BOX_ROOM),
        Err(PlacementError::IdleTier { tier: 0, .. })
    ));
    let q4 = model(false);
    assert!(matches!(
        plan_at(&q4, &bp, 4096, 1, BOX_ROOM),
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
    let p = plan_at(&q4, &m, ctx, 2, BOX_ROOM).expect("a two-slot plan");
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

/// The seat's default context on one such card, two sequences and no
/// draft: 252,416 positions a slot, the last step of 256 whose plan holds
/// at most the plan's margin fewer card expert bytes than the plan at
/// 4,096 ([`two_slots_at_4096_hold_every_routed_expert`]: every routed
/// byte), the next step past it. There 504 or 505 experts a layer stay on
/// the card and 337 go to the host.
#[test]
fn two_slots_default_context_is_252416() {
    let (at, n_l, host) = two_slots(252_416);
    assert!(Q4_ROUTED - at <= MARGIN);
    let (next, _, _) = two_slots(252_416 + 256);
    assert!(Q4_ROUTED - next > MARGIN);
    let (lo, hi) = (
        *n_l.iter().min().expect("layers"),
        *n_l.iter().max().expect("layers"),
    );
    assert_eq!(
        (lo, hi, n_l.iter().filter(|&&n| n == hi).count(), host),
        (504, 505, 47, 337)
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
    let box_plan = plan_at(&q4, &gate, 4096, 1, BOX_ROOM).expect("the box plan");
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
    let arena = super::nvme_arena_of(room, twin.host.expert_bytes, floor, None)
        .expect("the dial's default");
    assert_eq!(
        split.host.nvme_arena_bytes, arena,
        "the plan states the arena the dial chose"
    );

    let need = HostNeed::of(&split, 0).bytes();
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
        let need = HostNeed::of(&split, 0).bytes();
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
        assert!(HostNeed::of(&split, 0).bytes() + split.host.nvme_arena_bytes <= room);
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
/// pages, the pool is empty and the unset rule resolves `mid-p0-s1`, while
/// the unsplit twin keeps its whole pool.
#[test]
fn a_paged_stack_adds_nothing_to_the_churn_pool() {
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    for room in [27u64 << 30, 58 << 30] {
        let split = plan_at(&q4, &gate, 4096, 1, room).expect("the split plan");
        let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
        assert!(
            (0..LAYERS).all(|l| layer_split(&split, l).1 > 0),
            "room {room}: every layer pages"
        );
        let pool = |p: &Plan<'_>| ChurnPool::of(p, 0, 0).expect("the pool").bytes;
        assert_eq!(pool(&split), 0, "room {room}: the paged plan's pool");
        assert!(pool(&twin) > 0, "room {room}: the unsplit twin's pool");
        let (pick, bytes) = residency(&split);
        assert_eq!(
            (pick.word(), bytes),
            ("mid-p0-s1".to_string(), 0),
            "room {room}: the unset rule on the paged plan"
        );
    }
}

/// The serving seats' one-column rule reads the plan's RAM arena
/// (`bloomery_levers::paged_columns`): the 3090 gate plan at a 27 GiB room
/// pages its host experts through an arena at the seat's default two slots
/// and at one, so the default falls to one column and `--parallel 2` is
/// refused; at 40 GiB (over half the host leg, under its need) the tier
/// pages experts with no arena (the mapping path), and the plan serves as
/// asked — the serve gate's two rooms.
#[test]
fn the_one_column_rule_reads_the_arena() {
    use bloomery_levers::{Paged, PagedAt, SlotsBy, paged_columns};
    let (q4, gate) = (model(false), machine_a(RTX_3090));
    let arena = |slots: u64, room: u64| {
        let plan = plan_at(&q4, &gate, 4096, slots, room).expect("the plan");
        (plan.host.nvme_expert_bytes, plan.host.nvme_arena_bytes)
    };
    let at = |arena, slots, slots_set: bool| PagedAt {
        arena,
        slots,
        slots_by: slots_set.then_some(SlotsBy::Parallel),
        draft: false,
        draft_set: false,
    };
    let (moved2, deep2) = arena(2, 27 << 30);
    let (moved1, deep1) = arena(1, 27 << 30);
    println!("27 GiB: two slots arena {deep2} B, one slot arena {deep1} B");
    assert!(moved2 > 0 && deep2 > 0 && moved1 > 0 && deep1 > 0);
    assert_eq!(paged_columns(&at(deep2, 2, false)), Ok(Paged::OneColumn));
    assert_eq!(paged_columns(&at(deep1, 1, false)), Ok(Paged::AsAsked));
    assert!(paged_columns(&at(deep2, 2, true)).is_err());
    let (moved, shallow) = arena(2, 40 << 30);
    println!("40 GiB: two slots NVMe experts {moved} B, arena {shallow} B");
    assert!(moved > 0, "the 40 GiB room pages experts");
    assert_eq!(shallow, 0, "through the mapping, no arena");
    assert_eq!(paged_columns(&at(shallow, 2, true)), Ok(Paged::AsAsked));
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
    let box_plan = plan_at(&q4, &gate, 4096, 1, BOX_ROOM).expect("the box plan");
    let need = HostNeed::of(&box_plan, 0).bytes() - PLE_TABLE + ROW_ROOM;
    for room in [BOX_ROOM, need + 1, need] {
        let kept = plan_at(&q4, &gate, 4096, 1, room).expect("the plan");
        let twin = plan_with(&q4, &gate, 4096, 1, room, false).expect("the unsplit plan");
        assert_eq!(format!("{kept:?}"), format!("{twin:?}"), "room {room}");
    }
    let under = plan_at(&q4, &gate, 4096, 1, need - 1).expect("the plan");
    assert!(under.host.nvme_expert_bytes > 0);
    assert!(HostNeed::of(&under, 0).bytes() < need);
}

/// A room under the floor is refused by name, with the room, the floor and
/// the need; the floor itself is planned.
#[test]
fn a_room_under_the_floor_is_refused_by_name() {
    let arm = nvme_arm(27 << 30);
    let q4 = model(false);
    let gate = machine_a(RTX_3090);
    match plan_at(&q4, &gate, 4096, 1, arm.floor - 1) {
        Err(PlacementError::HostRoomFloor {
            room, floor, need, ..
        }) => assert_eq!(
            (room, floor, need),
            (arm.floor - 1, arm.floor, arm.old_need)
        ),
        other => panic!("not refused by name: {:?}", other.map(|p| p.host)),
    }
    let msg = plan_at(&q4, &gate, 4096, 1, arm.floor - 1)
        .map(|_| ())
        .unwrap_err()
        .to_string();
    assert!(msg.contains(&(arm.floor - 1).to_string()) && msg.contains(&arm.floor.to_string()));
    let at = nvme_arm(arm.floor);
    assert!(at.r_min <= at.r_max && at.moved > 0);
    assert!(at.max_unit > 0);
}

/// The split dial's own rule ([`super::nvme_arena_of`]): the default is the
/// largest arena the room leaves above the floor on a room that cannot hold
/// half the host leg's bytes, 0 on one that covers it; a value the lever
/// gives is taken as it is and one that pushes the host segments under the
/// floor is refused by name, with the room, the floor and the value.
#[test]
fn the_split_dial_picks_its_arena_or_refuses_a_too_large_one() {
    use super::nvme_arena_of;
    let of = |room, held, floor, given| {
        nvme_arena_of(room, held, floor, given).expect("the dial picks an arena")
    };
    let (room, held, floor) = (27 << 30, 63_166_000_000u64, 5_560_000_000);
    // The default: the deep split takes the largest arena that fits.
    assert_eq!(of(room, held, floor, None), room - floor);
    // A room that covers over half the host leg keeps R1's split.
    assert_eq!(of(58 << 30, held, floor, None), 0);
    // The boundary itself: exactly half the host leg takes no arena.
    assert_eq!(of(held / 2, held, floor, None), 0);
    assert_eq!(of(held / 2 - 1, held, floor, None), held / 2 - 1 - floor);
    // A given arena is taken as it is, up to the largest that fits.
    assert_eq!(of(room, held, floor, Some(1 << 30)), 1 << 30);
    assert_eq!(of(room, held, floor, Some(room - floor)), room - floor);
    // One that pushes the host segments under the floor is refused by name.
    match nvme_arena_of(room, held, floor, Some(room - floor + 1)) {
        Err(PlacementError::NvTierArena {
            arena,
            room: r,
            floor: f,
        }) => {
            assert_eq!((arena, r, f), (room - floor + 1, room, floor));
        }
        other => panic!("not refused by name: {:?}", other),
    }
    let msg = nvme_arena_of(room, held, floor, Some(u64::MAX))
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
