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
//! No file, no card: the four-layer synthetic file of the place module's
//! slot tests, each layer's three routed stacks of types the card experts
//! read, and the shared MTP draft as its header states it.

use gguf::GgmlType;
use model::arch::models::{
    Act, Arch, Borrows, ChatSpec, Ffn, Gqa, HcKind, HcSpec, HeadRows, LayerSpec, Mixer, ModelSpec,
    Moe, MtpDraft, MtpHeadNorm, MtpInput, MtpSource, Residual, Rope, RopeMode, Router, Score,
    Shared,
};
use model::arch::qwen35moe::hparams::{Exp, FfnKind, Hparams, Kind, Ple, Variant};
use model::arch::qwen35moe::mtp::BorrowedHead;
use model::arch::qwen35moe::place::{
    Experts, FileTensor, KvLayout, MtpInputs, PlanInputs, mtp_head_act_bytes, mtp_tensors,
};
use model::placement::workstation::{CONTEXT, GRANULE, MARGIN, SCRATCH};
use model::placement::{Card, Host, Machine, ModelTensor, ModelTensors, PlanLevers, Role};

const CTX: u64 = 4096;
const LAYERS: usize = 4;
const EXPERTS: u64 = 512;
/// The card the fill leaves part of the routed experts off: the draft's
/// ~2.85 GB, the margin, context and scratch, and about half the four
/// layers' 512 experts of 3,584,000 B.
const WIDE: u64 = 8 << 30;

/// The slot tests' file: three delta layers, the PLE site on the second, a
/// selecting attention layer; Qwen3.8's widths.
fn hparams() -> Hparams {
    Hparams {
        variant: Variant::Qwen4Exp,
        n_layer: LAYERS,
        n_embd: 2560,
        n_head: 16,
        n_head_kv: 2,
        head_dim: 256,
        rope_dims: 64,
        rope_sections: [11, 11, 10, 0],
        rope_base: 1e7,
        rms_eps: 1e-6,
        n_vocab: 248_320,
        n_ctx_train: 262_144,
        n_expert: 512,
        n_used: 10,
        expert_ff: 640,
        shared_ff: Some(640),
        ff: None,
        conv: 4,
        state: 128,
        v_heads: 48,
        k_heads: 16,
        interval: LAYERS,
        kinds: vec![
            Kind::DeltaRule,
            Kind::DeltaRule,
            Kind::DeltaRule,
            Kind::Attention,
        ],
        ffns: vec![FfnKind::Routed; LAYERS],
        exp: Some(Exp {
            hc_streams: 4,
            hc_rank: 320,
            idx_heads: 4,
            idx_dim: 128,
            idx_top_k: 2048,
            ratios: vec![0, 0, 0, 4],
            ple: Some(Ple {
                layer: 1,
                ngram: 3,
                heads_per_ngram: 4,
                conv: 4,
                eos: 0,
                image: None,
                row: 256,
            }),
        }),
        defaults: Vec::new(),
    }
}

/// A layer's norm, and its routed stacks in types the card experts read: a
/// Q4_K gate and up of 640 rows of 2,560, a Q8_0 down of 2,560 rows of 640.
fn inputs() -> PlanInputs {
    let hp = hparams();
    let kv = KvLayout::of(&hp);
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
    let spec = ModelSpec {
        arch: Arch::Qwen4Exp,
        hidden: 2560,
        vocab: 248_320,
        ctx_train: 262_144,
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
    };
    PlanInputs {
        hp,
        model: ModelTensors {
            tensors,
            layers: LAYERS,
            experts: EXPERTS,
            experts_used: 10,
        },
        spec,
        kv,
        // A synthetic file: a room no host need passes, given, not read.
        room: (u64::MAX, model::placement::workstation::HostRead::Given),
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
                rope: Rope {
                    mode: RopeMode::Imrope {
                        sections: [11, 11, 10, 0],
                    },
                    dims: 64,
                    base: 1e7,
                    yarn: None,
                },
                qk_norm: true,
                out_gate: true,
                select: None,
            }),
            ffn: Ffn::Moe(Moe {
                experts: 512,
                top_k: 10,
                expert_ff: 640,
                act: Act::SwiGlu { limit: None },
                router: Router {
                    score: Score::Softmax,
                    bias: false,
                    norm: true,
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
        head_rows: HeadRows::Full,
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
            reserves: Vec::new(),
        }],
        tiers: Vec::new(),
        host: Host {
            usable_bytes: u64::MAX,
            reserves: Vec::new(),
        },
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
