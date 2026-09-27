//! The coverage check: what of a file no program in this tree runs, every
//! item listed at once ([`check`]). It joins the model's [`needs`] with a
//! hand-written table of what exists ([`AVAILABLE`], one row per kernel
//! instance or program step, each citing the constant or the function that
//! owns it), the card formats of the tensors' types ([`CardFormat`], the
//! format owner), the program's per-tensor type pins, the pre-tokenizers the
//! tokenizer knows and the chat parsers.
//!
//! The kernels' own file-against-constant checks stay behind it as a second
//! line: a file this check passes cannot trip them.
//!
//! An architecture no program runs yet ([`program_of`] is `None`) is listed
//! against the whole tree: a need any program's row covers, and a tensor type
//! any program's pin reads, is not an item; the program itself is one.

use gguf::GgmlType;
use models::{Arch, Extra, LatentUp, ModelSpec, Need, RopeMode, Score, needs};

use crate::placement::{CardFormat, ModelTensors, Role, Unimplemented};

/// A layer program: the chain that runs a family's layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Program {
    /// The V4.1 chain (`crates/gpu-deepseek41`: `chain/`, `body.rs`).
    Deepseek41Chain,
    /// The whole-card qwen3moe body (`crates/gpu/src/arch/qwen3moe`).
    Qwen3moeBody,
}

/// The program that runs `arch`'s layers; `None` when none does.
#[must_use]
pub fn program_of(arch: Arch) -> Option<Program> {
    match arch {
        Arch::Deepseek41 | Arch::Deepseek4 => Some(Program::Deepseek41Chain),
        Arch::Qwen3Moe | Arch::Qwen35Moe => Some(Program::Qwen3moeBody),
        Arch::Glm5Next | Arch::Qwen4Exp => None,
    }
}

/// One thing a program runs: the need it covers, and where it lives.
#[derive(Clone, Copy)]
pub struct Available {
    pub program: Program,
    /// The constant or function that owns the instance.
    pub at: &'static str,
    /// Whether this row covers `need`.
    pub runs: fn(&Need) -> bool,
}

/// Everything the programs run that a [`Need`] names. A need no row covers
/// is an item of [`check`]; a row removed makes its needs items.
pub const AVAILABLE: &[Available] = &[
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/attn.rs LATENT; chain/attn.rs rope dims",
        runs: |n| {
            matches!(
                n,
                Need::Latent {
                    latent: 512,
                    rope: 64,
                    up: LatentUp::KeqV
                }
            )
        },
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/compress.rs WIDTH",
        runs: |n| matches!(n, Need::Compress { width: 512 }),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/indexer.rs HEADS, HEAD_DIM; index_key.rs WIDTH",
        runs: |n| matches!(n, Need::StreamIndex { heads: 32, d: 128 }),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/router.rs N_EXPERT, N_USED",
        runs: |n| {
            matches!(
                n,
                Need::Router {
                    score: Score::SqrtSoftplus,
                    experts: 384,
                    top_k: 6,
                    bias: true,
                    norm: true
                }
            )
        },
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/chain/ffn.rs (the shared expert)",
        runs: |n| {
            matches!(
                n,
                Need::Shared {
                    sigmoid_gate: false,
                    ..
                }
            )
        },
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/hc.rs HC_STREAMS",
        runs: |n| matches!(n, Need::Hc { streams: 4 }),
    },
    Available {
        program: Program::Deepseek41Chain,
        at: "gpu-deepseek41/src/engram_gate.rs ROW",
        runs: |n| matches!(n, Need::Engram { row: 5120 }),
    },
    Available {
        program: Program::Qwen3moeBody,
        at: "gpu/src/flash_gqa.rs HEAD, GROUP",
        runs: |n| matches!(n, Need::Gqa { head: 128, pack: 8 }),
    },
    Available {
        program: Program::Qwen3moeBody,
        at: "gpu/src/rope_neox.rs HEAD",
        runs: |n| {
            matches!(
                n,
                Need::QkRope {
                    qk_norm: true,
                    head: 128,
                    mode: RopeMode::Neox,
                    dims: 128
                }
            )
        },
    },
    Available {
        program: Program::Qwen3moeBody,
        at: "gpu/src/arch/qwen3moe/router.rs N_EXPERT, N_USED",
        runs: |n| {
            matches!(
                n,
                Need::Router {
                    score: Score::Softmax,
                    experts: 128,
                    top_k: 8,
                    bias: false,
                    norm: true
                }
            )
        },
    },
];

/// The pre-tokenizers the tokenizer knows (`tokenizer::pretok::Pre::NAMES`,
/// which is crate-private there).
pub const PRE_TOKENIZERS: &[&str] = &["deepseek-v3", "hunyuan-dense", "joyai-llm", "qwen2"];

/// A tensor family whose type a program pins, and the types it reads.
struct TypePin {
    program: Program,
    role: Role,
    /// Only the quantized tensors of two or more dimensions (the matrices a
    /// gemv reads; an F32 parameter table is its layer's mixer's).
    matrices: bool,
    what: &'static str,
    reads: &'static [GgmlType],
}

/// The per-tensor type pins of the programs (the qwen3moe body's `kq_site`
/// calls in `Body::load`, the head's Q6_K gemv in `gpu/src/head.rs`, the
/// card embedding's `embed_rows_q4k`; the V4.1 chain's attention gemvs, whose
/// head-split sites in `chain/attn.rs` take q3_K or q8_0, the same head, and
/// the hyper-connection fn `hc.rs` reads as q3_K words, `hc_f32.rs` as f32).
const TYPE_PINS: &[TypePin] = &[
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Attention,
        matrices: true,
        what: "attention matrices (the body reads q4_K and q6_K)",
        reads: &[GgmlType::Q4_K, GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::Head,
        matrices: true,
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Qwen3moeBody,
        role: Role::TokenEmbedding,
        matrices: true,
        what: "token embedding (the card reads q4_K rows)",
        reads: &[GgmlType::Q4_K],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::Attention,
        matrices: true,
        what: "attention matrices (the chain reads q3_K and q8_0)",
        reads: &[GgmlType::Q3_K, GgmlType::Q8_0],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::Head,
        matrices: true,
        what: "output head (the head reads q6_K)",
        reads: &[GgmlType::Q6_K],
    },
    TypePin {
        program: Program::Deepseek41Chain,
        role: Role::HyperConnection,
        matrices: true,
        what: "hyper-connection fn (the chain reads q3_K and f32)",
        reads: &[GgmlType::Q3_K],
    },
];

/// Every part of `spec` (with its tensors `model`) that no program runs:
/// layer by layer in layer order, then the tensor formats, then the
/// model-wide ones. Empty for a file a program runs whole.
#[must_use]
pub fn check(spec: &ModelSpec, model: &ModelTensors) -> Vec<Unimplemented> {
    check_with(spec, model, AVAILABLE)
}

/// [`check`] against the table `available` instead of [`AVAILABLE`].
#[must_use]
pub fn check_with(
    spec: &ModelSpec,
    model: &ModelTensors,
    available: &[Available],
) -> Vec<Unimplemented> {
    let program = program_of(spec.arch);
    let mut out = Vec::new();
    let mut at = |layer: Option<usize>, need: Need| {
        let f = Unimplemented {
            layer,
            feature: need.to_string(),
        };
        if !out.contains(&f) {
            out.push(f);
        }
    };
    // With no program, the row of any program covers a need.
    let runs = |need: &Need| {
        available
            .iter()
            .any(|a| program.is_none_or(|p| a.program == p) && (a.runs)(need))
    };
    let (layered, wide): (Vec<_>, Vec<_>) = needs(spec).into_iter().partition(|(_, l)| l.is_some());
    for (need, l) in layered {
        if !runs(&need) {
            at(l.map(|l| l as usize), need);
        }
    }
    for t in &model.tensors {
        let card = match program {
            // The V4.1 chain's rule: a stack of a type no card format loads.
            Some(Program::Deepseek41Chain) => CardFormat::of(t.ty),
            // A whole-card program, or the one still to be written, needs a
            // card expert kernel for the type.
            Some(Program::Qwen3moeBody) | None => CardFormat::of_routed(t.ty),
        };
        if t.role == Role::RoutedExperts && card.is_none() {
            at(t.layer, Need::RoutedFormat(t.ty));
        }
        let matrix = t.dims.len() >= 2 && t.ty != GgmlType::F32;
        let pins: Vec<&TypePin> = TYPE_PINS
            .iter()
            .filter(|pin| {
                program.is_none_or(|p| pin.program == p)
                    && pin.role == t.role
                    && (!pin.matrices || matrix)
            })
            .collect();
        let refused: Vec<&TypePin> = match program {
            Some(_) => pins
                .into_iter()
                .filter(|pin| !pin.reads.contains(&t.ty))
                .collect(),
            // No program: an item only when no program's pin reads the type,
            // named by the first pin.
            None if pins.iter().all(|pin| !pin.reads.contains(&t.ty)) => {
                pins.into_iter().take(1).collect()
            }
            None => Vec::new(),
        };
        for pin in refused {
            at(
                t.layer,
                Need::WeightFormat {
                    what: pin.what,
                    ty: t.ty,
                },
            );
        }
    }
    for (need, _) in wide {
        if !runs(&need) {
            at(None, need);
        }
    }
    match program {
        None => at(None, Need::Program(spec.arch.name())),
        Some(Program::Deepseek41Chain)
            if !spec
                .layers
                .iter()
                .any(|l| l.extras.contains(&Extra::Engram)) =>
        {
            at(None, Need::NoEngram);
        }
        Some(_) => {}
    }
    if !PRE_TOKENIZERS.contains(&spec.chat.pre.as_str()) {
        at(None, Need::PreTokenizer(spec.chat.pre.clone()));
    }
    if spec.chat.template.is_some() && spec.chat.tools.is_none() {
        at(None, Need::ToolParser);
    }
    out
}
