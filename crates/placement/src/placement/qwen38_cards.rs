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
    A6000, ALIASES, CONTEXT, CONTEXT_SELF, CardSpec, DeviceInfo, GRANULE, HostNeed, MARGIN, MIB,
    SCRATCH, TIER_BATCH_HOST_RESERVE, TIER_BATCH_RESERVE, census_usable, host, resolve,
    tier_batch_host_bytes, tier_batch_staging_bytes,
};
use super::{
    Card, CardFormat, Device, Format, KvBytes, Machine, ModelTensor, ModelTensors, PlacementError,
    Plan, PlanLevers, Role, plan_routed_reserving,
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

/// The card leg's routed stacks (`qwen35moe::place::card_routed`): a Q4_K
/// or Q5_K gate and up, a Q5_1 or Q8_0 down, in the file's blocks.
fn card_routed(ty: GgmlType) -> Option<CardFormat> {
    match ty {
        GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q5_1 | GgmlType::Q8_0 => {
            Some(CardFormat::KQuant)
        }
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
/// PLE table moved from the NVMe tier to the host (`ple_on_host`). A plan
/// that breaks an invariant fails the test by name.
fn plan_at<'a>(
    model: &'a ModelTensors,
    machine: &'a Machine,
    ctx: u64,
    slots: u64,
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
    for r in &mut plan.rows {
        if model.tensors[r.tensor].role != Role::EngramTable {
            continue;
        }
        let s = &mut r.segments[0];
        assert_eq!((s.device, s.format), (Device::Nvme, Format::NvmeFile));
        plan.nvme_bytes -= s.resident_bytes;
        plan.host.table_bytes += s.resident_bytes;
        plan.host.headroom_bytes -= i128::from(s.resident_bytes);
        (s.device, s.format) = (Device::Host, Format::HostFile);
    }
    plan.cards[0].kv_bytes += rows;
    let broken = plan.violations();
    assert!(broken.is_empty(), "ctx {ctx} slots {slots}: {broken:?}");
    Ok(plan)
}

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

/// The Q4 file's routed experts and the PLE table, in the file's bytes, and
/// the host's engram row cache reserve: what the box gate pins as
/// `HOST_EXPERTS` and `HOST_TABLES`, and `workstation::ROW_CACHE`.
const Q4_ROUTED: u64 = 77_017_907_200;
const PLE_TABLE: u64 = 28_800_138_240;
const ROW_CACHE: u64 = 4_294_967_296;

/// One, two or four such cards: the unset `--place` of `generate_qwen3moe`
/// and of the Qwen3.8 seat is `a`, the stage on cuda0 — the lowest ordinal
/// among equal usable bytes — and no tier. On it, at 4,096 positions and one
/// sequence, the card holds every routed expert of all 48 layers, so the
/// host serves none and the host need is the PLE table and the row cache
/// alone; the unset residency rule is `off`, no churn pool beside it: no
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
    let a6000 = plan_at(&q4, &m, 4096, 1).expect("the A6000 plan");
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
    let plan = plan_at(&q4, &m, 4096, 1).expect("the 96 GB plan");
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
    assert_eq!(HostNeed::of(&plan, 0).bytes(), PLE_TABLE + ROW_CACHE);
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
/// IQ4_NL or Q8_0: no layer's three stacks are all card types, so on any
/// card every routed expert stays on the host, the host need is those, the
/// PLE table and the row cache, and the residency rule is `off`. Under
/// `--place bp` on two such cards both plans are refused by name as a tier
/// card that holds no expert: the Q3 file's because no layer's stacks are
/// card types, the Q4 file's because its stage card holds every expert.
#[test]
fn q3_keeps_every_routed_expert_on_the_host() {
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
    let plan = plan_at(&q3, &m, 4096, 1).expect("the Q3 plan");
    assert_eq!(plan.n_l, vec![0; LAYERS]);
    assert_eq!(
        (plan.host.experts, plan.host.expert_bytes),
        (EXPERTS * LAYERS as u64, routed)
    );
    assert_eq!(
        HostNeed::of(&plan, 0).bytes(),
        routed + PLE_TABLE + ROW_CACHE
    );
    assert_eq!(
        residency(&plan),
        (
            Residency38Pick {
                pinned: None,
                why: Residency38Why::NoCardExperts
            },
            0
        )
    );
    let bp = machine_bp(cards[0], cards[1]);
    // PIN(2026-10-06): refused (IdleTier); was a plan whose tier_n_l is 0 on
    // every layer. The tier's eligible layers are none, so no expert is
    // left for it.
    assert!(matches!(
        plan_at(&q3, &bp, 4096, 1),
        Err(PlacementError::IdleTier { tier: 0, .. })
    ));
    let q4 = model(false);
    assert!(matches!(
        plan_at(&q4, &bp, 4096, 1),
        Err(PlacementError::IdleTier { tier: 0, .. })
    ));
}

/// The target matrices an MTP draft file borrows, `token_embd` and
/// `output`, are Q8_0 in the Q4 file, the one format the draft reads them
/// in; the Q3 file's `output` is Q6_K, so its draft is refused or, unset on
/// the seat, off (`PlanInputs::mtp_borrows`).
#[test]
fn only_the_q4_file_lends_the_draft_q8_0_matrices() {
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
    let p = plan_at(&q4, &m, ctx, 2).expect("a two-slot plan");
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
