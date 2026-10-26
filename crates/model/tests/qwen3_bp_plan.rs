//! The bp plan's tier experts at the serving context, on synthetic Q8_0 files
//! of the shapes the headers and the role tables name. No file and no card.
//! The arena is 0: a serving plan adds the body's `placed_arena_bytes`, which
//! this crate does not size, on top of the m = 1 scratch.
//!
//! The vocabulary lengths are the ones the modelspec design records for these
//! files. A plan reads `vocab_size` from the file, and this crate has no
//! constant for either.

use gguf::GgmlType;
use model::arch::models::ChatSpec;
use model::arch::qwen3moe::hparams::{Experts, Hparams as Hparams3, Rope, RopeMode, Score};
use model::arch::qwen3moe::place::{self as q3, KvLayout as Kv3, PlanInputs as Plan3};
use model::arch::qwen35moe::hparams::{FfnKind, Hparams as Hparams35, Kind, SectionOrder, Variant};
use model::arch::qwen35moe::place::{self as q35, KvLayout as Kv35, PlaceError as Place35Error};
use model::placement::accepts::{Program, routed_rule};
use model::placement::workstation::{
    A6000, CardSpec, DeviceId, HostRead, RTX_3090, TierBatchBytes,
};
use model::placement::{Machine, ModelTensor, ModelTensors, PlacementError, PlanLevers, Role};

/// The serving context, the files' `context_length`.
const CTX: u64 = 262_144;
const Q3_LAYERS: usize = 48;
const Q3_EXPERTS: u64 = 128;
const Q3_VOCAB: u64 = 151_936;
const Q36_LAYERS: usize = 40;
const Q36_EXPERTS: u64 = 256;
const Q36_VOCAB: u64 = 248_320;
const HIDDEN: u64 = 2048;

fn file_bytes(ty: GgmlType, dims: &[u64]) -> u64 {
    let values: u64 = dims.iter().product();
    values / ty.blck_size().expect("a block") * ty.type_size().expect("bytes")
}

fn tensor(
    tensors: &mut Vec<ModelTensor>,
    name: String,
    layer: Option<usize>,
    role: Role,
    ty: GgmlType,
    dims: Vec<u64>,
    gathered_rows: Option<u64>,
) {
    if ty == GgmlType::Q8_0 {
        assert!(
            dims.first().is_some_and(|k| k.is_multiple_of(32)),
            "{name} dims {dims:?} are not a Q8_0 row"
        );
    }
    let file_bytes = file_bytes(ty, &dims);
    tensors.push(ModelTensor {
        name,
        shard: 0,
        layer,
        role,
        ty,
        dims,
        file_bytes,
        gathered_rows,
    });
}

fn q8(
    tensors: &mut Vec<ModelTensor>,
    name: String,
    layer: Option<usize>,
    role: Role,
    dims: Vec<u64>,
    gathered_rows: Option<u64>,
) {
    tensor(
        tensors,
        name,
        layer,
        role,
        GgmlType::Q8_0,
        dims,
        gathered_rows,
    );
}

fn f32(
    tensors: &mut Vec<ModelTensor>,
    name: String,
    layer: Option<usize>,
    role: Role,
    dims: Vec<u64>,
) {
    tensor(tensors, name, layer, role, GgmlType::F32, dims, None);
}

/// `plan_tiers` with the stage held as the placed load holds it, arena 0.
fn tiered(layers: usize, stage: CardSpec, tier: CardSpec, batch: TierBatchBytes) -> Machine {
    let mut machine =
        model::placement::workstation::plan_tiers(layers, stage, &[tier], None, batch)
            .expect("the stage and the tier are distinct devices");
    q3::stage_placed(&mut machine, 0);
    machine
}

/// Two 3090s are one device by name. A distinct device id is what a census
/// gives each card.
fn rtx(ordinal: u32) -> CardSpec {
    CardSpec {
        device: Some(DeviceId {
            ordinal,
            uuid: [ordinal as u8; 16],
        }),
        ..RTX_3090
    }
}

/// `(stage, host, tier)` experts, and the three cover the file.
fn accounted(plan: &model::placement::Plan<'_>, experts: u64) -> (u64, u64, u64) {
    assert_eq!(plan.tier_n_l.len(), 1, "one expert tier");
    let tier = plan.tier_n_l[0].iter().sum::<u64>();
    let card = plan.n_l.iter().sum::<u64>();
    assert_eq!(
        card + tier + plan.host.experts,
        experts,
        "stage {card} + tier {tier} + host {} leaves the file's {experts}",
        plan.host.experts
    );
    (card, plan.host.experts, tier)
}

fn qwen3_inputs() -> Plan3 {
    let hp = Hparams3 {
        n_layer: Q3_LAYERS,
        n_embd: HIDDEN as usize,
        n_head: 32,
        n_head_kv: 4,
        head_dim: 128,
        rope: Rope {
            mode: RopeMode::Neox,
            base: 1e7,
            dims: 128,
        },
        rms_eps: 1e-6,
        n_vocab: Q3_VOCAB as usize,
        n_ctx_train: CTX as usize,
        experts: Experts {
            n_expert: Q3_EXPERTS as usize,
            n_used: 8,
            ff: 768,
            score: Score::Softmax,
            weights_norm: true,
        },
    };
    let mut tensors = Vec::new();
    q8(
        &mut tensors,
        "token_embd.weight".to_string(),
        None,
        Role::TokenEmbedding,
        vec![HIDDEN, Q3_VOCAB],
        Some(1),
    );
    for l in 0..Q3_LAYERS {
        let blk = |stem: &str| format!("blk.{l}.{stem}");
        f32(
            &mut tensors,
            blk("attn_norm.weight"),
            Some(l),
            Role::Attention,
            vec![HIDDEN],
        );
        q8(
            &mut tensors,
            blk("attn_q.weight"),
            Some(l),
            Role::Attention,
            vec![HIDDEN, 32 * 128],
            None,
        );
        for stem in ["attn_k.weight", "attn_v.weight"] {
            q8(
                &mut tensors,
                blk(stem),
                Some(l),
                Role::Attention,
                vec![HIDDEN, 4 * 128],
                None,
            );
        }
        for stem in ["attn_q_norm.weight", "attn_k_norm.weight"] {
            f32(&mut tensors, blk(stem), Some(l), Role::Attention, vec![128]);
        }
        q8(
            &mut tensors,
            blk("attn_output.weight"),
            Some(l),
            Role::Attention,
            vec![32 * 128, HIDDEN],
            None,
        );
        f32(
            &mut tensors,
            blk("ffn_norm.weight"),
            Some(l),
            Role::FfnNorm,
            vec![HIDDEN],
        );
        f32(
            &mut tensors,
            blk("ffn_gate_inp.weight"),
            Some(l),
            Role::Router,
            vec![HIDDEN, Q3_EXPERTS],
        );
        for stem in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            q8(
                &mut tensors,
                blk(stem),
                Some(l),
                Role::RoutedExperts,
                vec![HIDDEN, 768, Q3_EXPERTS],
                None,
            );
        }
        q8(
            &mut tensors,
            blk("ffn_down_exps.weight"),
            Some(l),
            Role::RoutedExperts,
            vec![768, HIDDEN, Q3_EXPERTS],
            None,
        );
    }
    f32(
        &mut tensors,
        "output_norm.weight".to_string(),
        None,
        Role::Head,
        vec![HIDDEN],
    );
    q8(
        &mut tensors,
        "output.weight".to_string(),
        None,
        Role::Head,
        vec![HIDDEN, Q3_VOCAB],
        None,
    );
    let kv = Kv3::of(&hp);
    Plan3 {
        hp,
        model: ModelTensors {
            tensors,
            layers: Q3_LAYERS,
            experts: Q3_EXPERTS,
            experts_used: 8,
            bf16_words: |_| false,
        },
        kv,
    }
}

fn qwen36_kinds() -> Vec<Kind> {
    (0..Q36_LAYERS)
        .map(|l| {
            if (l + 1).is_multiple_of(4) {
                Kind::Attention
            } else {
                Kind::DeltaRule
            }
        })
        .collect()
}

fn qwen36_inputs() -> q35::PlanInputs {
    let kinds = qwen36_kinds();
    let hp = Hparams35 {
        variant: Variant::Qwen35Moe,
        n_layer: Q36_LAYERS,
        n_trunk: Q36_LAYERS,
        n_embd: HIDDEN as usize,
        n_head: 16,
        n_head_kv: 2,
        head_dim: 256,
        rope_dims: 64,
        rope_sections: [11, 11, 10, 0],
        rope_order: SectionOrder::Tyxz,
        rope_base: 1e7,
        rms_eps: 1e-6,
        n_vocab: Q36_VOCAB as usize,
        n_ctx_train: CTX as usize,
        n_expert: Q36_EXPERTS as usize,
        n_used: 8,
        expert_ff: 512,
        shared_ff: Some(512),
        ff: None,
        conv: 4,
        state: 128,
        v_heads: 32,
        k_heads: 16,
        interval: 4,
        ffns: vec![FfnKind::Routed; Q36_LAYERS],
        exp: None,
        defaults: Vec::new(),
        kinds,
    };
    let mut tensors = Vec::new();
    q8(
        &mut tensors,
        "token_embd.weight".to_string(),
        None,
        Role::TokenEmbedding,
        vec![HIDDEN, Q36_VOCAB],
        Some(1),
    );
    for (l, &kind) in hp.kinds.iter().enumerate().take(Q36_LAYERS) {
        let blk = |stem: &str| format!("blk.{l}.{stem}");
        f32(
            &mut tensors,
            blk("attn_norm.weight"),
            Some(l),
            Role::Attention,
            vec![HIDDEN],
        );
        if kind == Kind::DeltaRule {
            // qkv rows are the conv channels: 2·k_heads·state + v_heads·state.
            q8(
                &mut tensors,
                blk("attn_qkv.weight"),
                Some(l),
                Role::Attention,
                vec![HIDDEN, 2 * 16 * 128 + 32 * 128],
                None,
            );
            // The output gate z: v_heads · state rows of the hidden width.
            q8(
                &mut tensors,
                blk("attn_gate.weight"),
                Some(l),
                Role::Attention,
                vec![HIDDEN, 32 * 128],
                None,
            );
            f32(
                &mut tensors,
                blk("ssm_conv1d.weight"),
                Some(l),
                Role::Attention,
                vec![4, 2 * 16 * 128 + 32 * 128],
            );
            f32(
                &mut tensors,
                blk("ssm_dt.bias"),
                Some(l),
                Role::Attention,
                vec![32],
            );
            // VCols of one value a head: the width is the value-head count.
            f32(
                &mut tensors,
                blk("ssm_a"),
                Some(l),
                Role::Attention,
                vec![32],
            );
            for stem in ["ssm_alpha.weight", "ssm_beta.weight"] {
                f32(
                    &mut tensors,
                    blk(stem),
                    Some(l),
                    Role::Attention,
                    vec![HIDDEN, 32],
                );
            }
            f32(
                &mut tensors,
                blk("ssm_norm.weight"),
                Some(l),
                Role::Attention,
                vec![128],
            );
            q8(
                &mut tensors,
                blk("ssm_out.weight"),
                Some(l),
                Role::Attention,
                vec![32 * 128, HIDDEN],
                None,
            );
        } else {
            // Query rows are heads · head_dim · 2: the packed output gate.
            q8(
                &mut tensors,
                blk("attn_q.weight"),
                Some(l),
                Role::Attention,
                vec![HIDDEN, 16 * 256 * 2],
                None,
            );
            for stem in ["attn_k.weight", "attn_v.weight"] {
                q8(
                    &mut tensors,
                    blk(stem),
                    Some(l),
                    Role::Attention,
                    vec![HIDDEN, 2 * 256],
                    None,
                );
            }
            for stem in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                f32(&mut tensors, blk(stem), Some(l), Role::Attention, vec![256]);
            }
            q8(
                &mut tensors,
                blk("attn_output.weight"),
                Some(l),
                Role::Attention,
                vec![16 * 256, HIDDEN],
                None,
            );
        }
        f32(
            &mut tensors,
            blk("post_attention_norm.weight"),
            Some(l),
            Role::FfnNorm,
            vec![HIDDEN],
        );
        f32(
            &mut tensors,
            blk("ffn_gate_inp.weight"),
            Some(l),
            Role::Router,
            vec![HIDDEN, Q36_EXPERTS],
        );
        f32(
            &mut tensors,
            blk("ffn_gate_inp_shexp.weight"),
            Some(l),
            Role::SharedExpert,
            vec![HIDDEN],
        );
        for stem in ["ffn_gate_shexp.weight", "ffn_up_shexp.weight"] {
            q8(
                &mut tensors,
                blk(stem),
                Some(l),
                Role::SharedExpert,
                vec![HIDDEN, 512],
                None,
            );
        }
        q8(
            &mut tensors,
            blk("ffn_down_shexp.weight"),
            Some(l),
            Role::SharedExpert,
            vec![512, HIDDEN],
            None,
        );
        for stem in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            q8(
                &mut tensors,
                blk(stem),
                Some(l),
                Role::RoutedExperts,
                vec![HIDDEN, 512, Q36_EXPERTS],
                None,
            );
        }
        q8(
            &mut tensors,
            blk("ffn_down_exps.weight"),
            Some(l),
            Role::RoutedExperts,
            vec![512, HIDDEN, Q36_EXPERTS],
            None,
        );
    }
    f32(
        &mut tensors,
        "output_norm.weight".to_string(),
        None,
        Role::Head,
        vec![HIDDEN],
    );
    q8(
        &mut tensors,
        "output.weight".to_string(),
        None,
        Role::Head,
        vec![HIDDEN, Q36_VOCAB],
        None,
    );
    let roles = ModelTensors {
        tensors,
        layers: Q36_LAYERS,
        experts: Q36_EXPERTS,
        experts_used: 8,
        bf16_words: |_| false,
    };
    // The description the plan reads, made from the header facts above as a file's open makes it.
    let spec = model::arch::qwen35moe::spec::spec_of(
        &hp,
        &roles,
        ChatSpec {
            pre: String::new(),
            template: None,
            tools: None,
            reasoning: None,
        },
    )
    .expect("the synthetic Qwen3.6 header describes its layers");
    let kv = Kv35::of_spec(&spec).expect("every layer has a store the layout sizes");
    q35::PlanInputs {
        model: roles,
        spec,
        kv,
        room: (u64::MAX, HostRead::Given),
        row_source: model::placement::RowTableSource::Split,
        arena: None,
    }
}

fn qwen36_tier(stage: CardSpec, tier: CardSpec) -> Result<(u64, u64, u64), Place35Error> {
    let inputs = qwen36_inputs();
    let machine = tiered(Q36_LAYERS, stage, tier, q35::leg_tier_batch(&inputs.spec));
    let plan = inputs.plan_rule(
        &machine,
        CTX,
        &PlanLevers::default(),
        routed_rule(Program::Qwen35Body),
    )?;
    Ok(accounted(&plan, Q36_LAYERS as u64 * Q36_EXPERTS))
}

/// Qwen3-30B Q8_0 at the serving context, an A6000 stage beside a 3090 tier:
/// `(stage, host, tier)` experts. The count is the planner's, at arena 0.
#[test]
fn qwen3_30b_q8_bp_tier_experts() {
    let inputs = qwen3_inputs();
    let machine = tiered(Q3_LAYERS, A6000, RTX_3090, q3::tier_batch(&inputs.hp));
    let plan = inputs
        .plan(&machine, CTX, &PlanLevers::default())
        .expect("the A6000 stage plans a tier");
    assert_eq!(
        accounted(&plan, Q3_LAYERS as u64 * Q3_EXPERTS),
        (4_311, 0, 1_833),
        "stage, host, tier"
    );
}

/// Qwen3.6 Q8_0 at the serving context on an A6000 stage beside a 3090: the
/// stage holds every expert, so the tier is idle.
#[test]
fn qwen36_q8_bp_on_a6000_is_idle() {
    match qwen36_tier(A6000, RTX_3090) {
        Err(Place35Error::Placement(PlacementError::IdleTier { tier: 0, card })) => {
            assert_eq!(card, "3090");
        }
        Err(e) => panic!("not IdleTier on the 3090: {e}"),
        Ok(split) => panic!("the 3090 holds {split:?}"),
    }
}

/// Qwen3.6 Q8_0 at the serving context on two 3090s: `(stage, host, tier)`
/// experts. The count is the planner's, at arena 0.
#[test]
fn qwen36_q8_bp_on_two_3090s() {
    assert_eq!(
        qwen36_tier(rtx(0), rtx(1)).expect("two 3090s plan a tier"),
        (4_580, 0, 5_660),
        "stage, host, tier"
    );
}
