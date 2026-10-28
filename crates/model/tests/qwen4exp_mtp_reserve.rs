//! A drafted qwen4exp plan under the card experts keeps its own bound:
//! the expert rule spreads within the card's budget less the draft's card
//! bytes (`PlanInputs::plan_mtp_with`), so the sum the plan checks fits
//! whatever the card's size, and the draft's card bytes are one number —
//! the reserve a plan (b′) machine carries (`MtpInputs::card_bytes_of`) is
//! what the drafted plan counts (`MtpPlan::draft_card_bytes` and its arena).
//!
//! The card is sized so the fill leaves no slack: the Q8_0-head plan on a
//! wide card fills part of the routed experts, and the card shrunk by its
//! slack past the margin keeps that fill with exactly the margin left. A
//! Q6_K head's walk adds its activation (`mtp_head_act_bytes`) beside the
//! arena; on that card the fill must give an expert back for it.
//!
//! On one pool (GB10) the drafted plan counts the draft once on each side:
//! the host's need leaves the cards their share with the draft's card bytes
//! in it, under either expert rule, and the pool's bound holds the draft
//! beside the plan's own terms.
//!
//! No file, no card: the four-layer synthetic file of the place module's
//! slot tests, each layer's three routed stacks of types the card experts
//! read, and the shared MTP draft as its header states it.

use gguf::GgmlType;
use model::arch::models::headpick::HeadSet;
use model::arch::models::{
    Act, Arch, Borrows, ChatSpec, DeltaKind, DeltaRule, EngramSpec, Extra, Ffn, GdnGate, Gqa,
    HcKind, HcSpec, KHeadMap, LayerSpec, Mixer, ModelSpec, Moe, MtpDraft, MtpHeadNorm, MtpInput,
    MtpSource, NgramRule, OutGate, PoolRule, Residual, Rope, RopeMode, Router, Score, Selector,
    Shared,
};
use model::arch::qwen35moe::mtp::BorrowedHead;
use model::arch::qwen35moe::place::{
    Experts, FileTensor, KvLayout, MtpInputs, PlanInputs, mtp_head_act_bytes, mtp_tensors,
};
use model::placement::workstation::{CONTEXT, GRANULE, MARGIN, SCRATCH};
use model::placement::{
    Card, Host, Machine, ModelTensor, ModelTensors, Node, PlanLevers, Role, UnifiedPool,
};

const CTX: u64 = 4096;
const LAYERS: usize = 4;
const EXPERTS: u64 = 512;
/// The card the fill leaves part of the routed experts off: the draft's
/// ~2.85 GB, the margin, context and scratch, and about half the four
/// layers' 512 experts of 3,584,000 B.
const WIDE: u64 = 8 << 30;

/// The slot tests' file: three delta layers, the PLE site on the second, a
/// selecting attention layer; Qwen3.8's widths.
fn description() -> ModelSpec {
    let rope = Rope {
        mode: RopeMode::Imrope {
            sections: [11, 11, 10, 0],
        },
        dims: 64,
        base: 1e7,
        yarn: None,
        factors: false,
    };
    let gdn = Mixer::DeltaRule(DeltaRule {
        kind: DeltaKind::Gdn {
            khead_map: KHeadMap::Tiled,
            gate: GdnGate::Sigmoid,
        },
        k_heads: 16,
        v_heads: 48,
        d: 128,
        conv: 4,
    });
    let attention = Mixer::Gqa(Gqa {
        heads: 16,
        kv_heads: 2,
        head_dim: 256,
        value_dim: 256,
        rope,
        qk_norm: true,
        o_norm: false,
        out_gate: OutGate::Packed,
        select: Some(Selector::TokenPool {
            heads: 4,
            d: 128,
            top_k: 2048,
            pool: 4,
            rule: PoolRule::Mean { rope },
        }),
        window: None,
        sinks: false,
        value_scale: None,
    });
    let layer = |l: usize, mixer: Mixer| LayerSpec {
        mixer,
        ffn: Ffn::Moe(Moe {
            experts: 512,
            top_k: 10,
            expert_ff: 640,
            latent: None,
            pre_exp_norm: false,
            post_exp_norm: None,
            act: Act::SwiGlu { limit: None },
            router: Router {
                score: Score::Softmax,
                bias: false,
                norm: models::RouterNorm::SelectedSum,
                scale: 1.0,
                hash: false,
            },
            shared: Some(Shared {
                ff: 640,
                act: Act::SwiGlu { limit: None },
                sigmoid_gate: true,
            }),
        }),
        residual: Residual::Hc,
        extras: if l == 1 { vec![Extra::Ple] } else { Vec::new() },
    };
    ModelSpec {
        arch: Arch::Qwen4Exp,
        hidden: 2560,
        vocab: 248_320,
        ctx_train: 262_144,
        rms_eps: 1e-6,
        layers: vec![
            layer(0, gdn.clone()),
            layer(1, gdn.clone()),
            layer(2, gdn),
            layer(3, attention),
        ],
        mtp: Vec::new(),
        hc: Some(HcSpec {
            streams: 4,
            kind: HcKind::Gated { rank: 320 },
        }),
        engram: Some(EngramSpec {
            heads: 4,
            max_ngram: 3,
            key_length: 256,
            rule: NgramRule::Ple {
                eos: 0,
                image: None,
                conv: 4,
                dilation: 3,
            },
        }),
        encoder: None,
        logit_epilogue: None,
        chat: ChatSpec {
            pre: String::new(),
            template: None,
            tools: None,
            reasoning: None,
        },
    }
}

/// A layer's norm, and its routed stacks in types the card experts read: a
/// Q4_K gate and up of 640 rows of 2,560, a Q8_0 down of 2,560 rows of 640.
fn inputs() -> PlanInputs {
    let spec = description();
    let kv = KvLayout::of_spec(&spec).expect("the description's layout");
    let tensor = |l: usize, stem: &str, role: Role, ty: GgmlType, dims: Vec<u64>| {
        let values: u64 = dims.iter().product();
        ModelTensor {
            name: format!("blk.{l}.{stem}"),
            shard: 0,
            layer: Some(l),
            role,
            ty,
            dims,
            file_bytes: values / ty.blck_size().expect("a block") * ty.type_size().expect("bytes"),
            gathered_rows: None,
        }
    };
    let mut tensors = Vec::new();
    for l in 0..LAYERS {
        tensors.push(tensor(
            l,
            "attn_norm.weight",
            Role::Attention,
            GgmlType::F32,
            vec![2560],
        ));
        for stem in ["ffn_gate_exps.weight", "ffn_up_exps.weight"] {
            tensors.push(tensor(
                l,
                stem,
                Role::RoutedExperts,
                GgmlType::Q4_K,
                vec![2560, 640, EXPERTS],
            ));
        }
        tensors.push(tensor(
            l,
            "ffn_down_exps.weight",
            Role::RoutedExperts,
            GgmlType::Q8_0,
            vec![640, 2560, EXPERTS],
        ));
    }
    PlanInputs {
        model: ModelTensors {
            tensors,
            layers: LAYERS,
            experts: EXPERTS,
            experts_used: 10,
            bf16_words: |_| false,
        },
        spec,
        kv,
        // A synthetic file: a room no host need passes, given, not read.
        room: (u64::MAX, model::placement::workstation::HostRead::Given),
        row_source: model::placement::RowTableSource::Split,
        arena: None,
    }
}

/// Qwen3.8's MTP layer as `mtp_of` reads the shared file.
fn draft() -> MtpDraft {
    MtpDraft {
        source: MtpSource::File {
            first_shard: "/m/mtp.gguf".into(),
            bytes: 2_775_621_632,
            borrows: Borrows {
                embedding: true,
                head: true,
            },
        },
        layer: LayerSpec {
            mixer: Mixer::Gqa(Gqa {
                heads: 24,
                kv_heads: 2,
                head_dim: 256,
                value_dim: 256,
                rope: Rope {
                    mode: RopeMode::Imrope {
                        sections: [11, 11, 10, 0],
                    },
                    dims: 64,
                    base: 1e7,
                    yarn: None,
                    factors: false,
                },
                qk_norm: true,
                o_norm: false,
                out_gate: OutGate::Packed,
                select: None,
                window: None,
                sinks: false,
                value_scale: None,
            }),
            ffn: Ffn::Moe(Moe {
                experts: 512,
                top_k: 10,
                expert_ff: 640,
                latent: None,
                pre_exp_norm: false,
                post_exp_norm: None,
                act: Act::SwiGlu { limit: None },
                router: Router {
                    score: Score::Softmax,
                    bias: false,
                    norm: models::RouterNorm::SelectedSum,
                    scale: 1.0,
                    hash: false,
                },
                shared: Some(Shared {
                    ff: 640,
                    act: Act::SwiGlu { limit: None },
                    sigmoid_gate: true,
                }),
            }),
            residual: Residual::Hc,
            extras: Vec::new(),
        },
        index: 48,
        hidden: 2560,
        vocab: 248_320,
        rms_eps: 1e-6,
        hc: Some(HcSpec {
            streams: 4,
            kind: HcKind::Gated { rank: 320 },
        }),
        input: MtpInput::Streams,
        head_norm: MtpHeadNorm::HcHead,
        head_rows: HeadSet::Full,
    }
}

/// The shared draft file's tensors as its header states them: the
/// statement's forms, the indexer's projections BF16 and its norms F32.
fn file() -> Vec<FileTensor> {
    let d = draft();
    let mut out: Vec<FileTensor> = mtp_tensors(&d)
        .expect("the layer's statement")
        .into_iter()
        .map(|t| {
            let (ty, dims) = t.form.clone().unwrap_or(match t.stem {
                "indexer.q_proj.weight" => (GgmlType::BF16, vec![2560, 512]),
                "indexer.k_proj.weight" => (GgmlType::BF16, vec![2560, 128]),
                _ => (GgmlType::F32, vec![128]),
            });
            let values: u64 = dims.iter().product();
            let nbytes = match ty {
                GgmlType::Q8_0 => values / 32 * 34,
                GgmlType::BF16 => values * 2,
                _ => values * 4,
            };
            FileTensor {
                name: t.name(d.index),
                shard: 0,
                ty,
                dims,
                nbytes,
            }
        })
        .collect();
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

/// The draft, its borrowed head read as `head`.
fn mtp(head: BorrowedHead) -> MtpInputs {
    let mut m = MtpInputs::from_parts(draft(), &file()).expect("the draft's inputs");
    m.borrowed_head = head;
    m
}

/// One card of `usable` bytes running every layer and the head, with the
/// workstation's context, scratch, margin and granule; a host with no bound.
fn machine(usable: u64) -> Machine {
    Machine {
        cards: vec![Card {
            name: "card".to_string(),
            tier_leg: None,
            device: None,
            usable_bytes: usable,
            context_bytes: CONTEXT,
            scratch_bytes: SCRATCH,
            margin_bytes: MARGIN,
            granule_bytes: GRANULE,
            free_bytes: None,
            held_by: None,
            layers: 0..LAYERS,
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

#[test]
fn a_drafted_card_plan_keeps_its_bound_at_any_slack() {
    let inputs = inputs();
    let levers = PlanLevers::default();
    let (q8, q6) = (mtp(BorrowedHead::Q8_0), mtp(BorrowedHead::Q6K));
    assert_eq!(mtp_head_act_bytes(BorrowedHead::Q6K, 2560), 68_736);
    for slots in [1, 2] {
        let wide = machine(WIDE);
        let at = inputs
            .plan_mtp_with_slots(&wide, CTX, &levers, &q8, Experts::Card, slots)
            .expect("the Q8_0 head's plan on the wide card");
        let held = at.plan.cards[0].experts;
        assert!(
            0 < held && held < LAYERS as u64 * EXPERTS,
            "{slots} slot(s): the wide card holds {held} experts, not a part of them"
        );
        let slack = u64::try_from(at.headroom_bytes).expect("headroom") - MARGIN;
        let tight = machine(WIDE - slack);
        let q8_tight = inputs
            .plan_mtp_with_slots(&tight, CTX, &levers, &q8, Experts::Card, slots)
            .expect("the Q8_0 head's plan on the card of no slack");
        assert_eq!(
            (q8_tight.plan.cards[0].experts, q8_tight.headroom_bytes),
            (held, i128::from(MARGIN)),
            "{slots} slot(s): the card shrunk by the slack keeps the fill with the margin left"
        );
        let with = match inputs.plan_mtp_with_slots(&tight, CTX, &levers, &q6, Experts::Card, slots)
        {
            Ok(p) => p,
            Err(e) => panic!(
                "{slots} slot(s), a Q6_K head on the card of no slack ({} B): the plan the expert \
                 rule filled is refused: {e}",
                WIDE - slack
            ),
        };
        assert!(
            with.headroom_bytes >= i128::from(MARGIN),
            "{slots} slot(s): headroom {} B under the margin",
            with.headroom_bytes
        );
        assert_eq!(
            q6.card_bytes_of(CTX, slots)
                .expect("the draft's card bytes"),
            with.draft_card_bytes() + with.arena_bytes,
            "{slots} slot(s): a plan (b′) reserve is not what the drafted plan counts"
        );
    }
}

/// [`machine`] on one pool whose census read `pool`: the card's free reading
/// and the host's figure are the pool, with no OS reserve
/// (`workstation::host_of`).
fn unified(usable: u64, pool: u64) -> Machine {
    let mut m = machine(usable);
    m.cards[0].free_bytes = Some(pool);
    *m.host_mut() = Host {
        usable_bytes: pool,
        reserves: Vec::new(),
    };
    *m.unified_mut() = Some(UnifiedPool { bytes: pool });
    m
}

/// One routed expert's bytes across the three stacks: a Q4_K gate and up of
/// 640 rows of 2,560 and a Q8_0 down of 2,560 rows of 640.
const EXPERT_BYTES: u64 = 2 * (640 * 2560 / 256 * 144) + 2560 * 640 / 32 * 34;

/// On one pool the drafted plan counts the draft once on each side. A pool of
/// exactly the need the plan's host headroom counts plans clean with no
/// expert paged from the NVMe tier, and what the pool holds past the plan's
/// cards and host is the draft's card bytes. One byte under it, the tier's
/// room is a byte short (`Plan::plan_at` caps it at the pool), so the plan
/// pages some host experts instead of breaking: it stays clean, frees the
/// host an expert's bytes less that byte, and the pool still holds the
/// draft's card bytes beside it. The two sides count the cards, the draft and
/// the context's own creation cost alike: counting the draft twice would
/// split at the need, and not at all would refuse one byte under it.
// PIN(2026-10-25): one byte under the need was one `PoolOver` naming the
// draft's card bytes. The host room is now capped at the pool, so a pool short
// of the need splits the host's experts onto the NVMe tier down to the split's
// floor, and the break is a named placement refusal there, never `PoolOver`.
#[test]
fn a_drafted_plan_on_one_pool_counts_the_draft_once() {
    let inputs = inputs();
    let levers = PlanLevers::default();
    let q8 = mtp(BorrowedHead::Q8_0);
    // The plan's host headroom, its host experts paged from the NVMe tier and
    // what the pool holds past its cards and host, or its refusal.
    let plan = |experts, slots, pool| {
        inputs
            .plan_mtp_with_slots(&unified(WIDE, pool), CTX, &levers, &q8, experts, slots)
            .map(|p| {
                let held = p.plan.pool_need().expect("a unified machine's pool terms");
                (
                    p.plan.host.headroom_bytes,
                    p.plan.host.nvme_expert_bytes,
                    held.pool - held.cards - held.host,
                )
            })
    };
    let ample: u64 = 1 << 40;
    for experts in [Experts::Card, Experts::Host] {
        for slots in [1, 2] {
            let at = format!("{experts:?}, {slots} slot(s)");
            let beside = q8
                .card_bytes_of(CTX, slots)
                .expect("the draft's card bytes");
            let (room, paged, _) =
                plan(experts, slots, ample).unwrap_or_else(|e| panic!("{at}, an ample pool: {e}"));
            assert_eq!(paged, 0, "{at}: an ample pool pages no expert");
            let need = ample - u64::try_from(room).expect("a headroom within the pool");
            let (headroom, paged, held) = plan(experts, slots, need)
                .unwrap_or_else(|e| panic!("{at}, a pool of the host's need {need} B: {e}"));
            assert_eq!(
                (headroom, paged, held),
                (0, 0, beside),
                "{at}: the need leaves no headroom, pages no expert and holds the draft beside the plan"
            );
            let (headroom, paged, held) = plan(experts, slots, need - 1)
                .unwrap_or_else(|e| panic!("{at}, a pool one byte under the need: {e}"));
            assert!(paged > 0, "{at}: one byte under the need pages no expert");
            assert_eq!(
                headroom,
                i128::from(EXPERT_BYTES) - 1,
                "{at}: the split frees an expert's bytes"
            );
            assert_eq!(
                i128::from(held),
                i128::from(beside) + headroom,
                "{at}: the pool holds the draft's card bytes beside the plan"
            );
        }
    }
}
