//! Qwen3.8's layer plans (`qwen4exp`): what each layer of the Flash-Next
//! trunk runs, read once at load from the file's description and checked
//! against the one geometry the chain's kernels are built for ([`geo`]) —
//! every weight by name, type and shape — so the walk reads names and never
//! decides a layer's kind.
//!
//! Every layer wraps its two sub-layers in the gated-residual streams (a
//! mix site each, [`HcSite`]); its mixer is the sigmoid-gated delta rule
//! ([`GdnPlan`]) or the gated GQA whose positions the mean-pool selector
//! picks ([`QsaPlan`]); its feed-forward block is 512 routed experts, all
//! on the host tier, and the sigmoid-gated shared expert on the card
//! ([`FfnPlan`]). The PLE layer ([`PlePlan`]) adds the n-gram embedding to
//! the streams before its mix.

use super::router::RouterDims;
use crate::linear::{self, KHeadMap, LinearShape};
use crate::weights::{DevWeight, Weights};
use crate::{GpuError, flash_gqa, ple, q38, qsa};
use model::arch::coverage::turns_as_neox;
use model::arch::models::shape::MoeShape;
use model::arch::models::{
    Act, DeltaKind, Extra, Ffn, GdnGate, Gqa, HcKind, KHeadMap as SpecMap, Mixer, ModelSpec,
    NgramRule, PoolRule, Residual, Score, Selector,
};

/// What the plan's refusals name.
const WHAT: &str = "qwen4exp plan";

/// The geometry the chain's kernels are built for: Qwen3.8-Flash-Next's.
/// Each constant is held to the kernel that fixes it below.
pub(super) mod geo {
    /// Values of a residual stream (`embedding_length`).
    pub(in super::super) const HIDDEN: usize = 2560;
    /// Hyper-connection streams and the gate's bottleneck.
    pub(in super::super) const STREAMS: usize = 4;
    pub(in super::super) const RANK: usize = 320;
    /// Query heads, key heads and their values.
    pub(in super::super) const N_HEAD: usize = 24;
    pub(in super::super) const N_KV: usize = 2;
    pub(in super::super) const HEAD: usize = 256;
    /// Values the rope turns at the head of a query, a key and an indexer
    /// key.
    pub(in super::super) const ROPE: usize = 64;
    /// Delta-rule key and value heads.
    pub(in super::super) const K_HEADS: usize = 16;
    pub(in super::super) const V_HEADS: usize = 48;
    /// Routed and shared experts' feed-forward width, the routed count and
    /// the experts a token keeps.
    pub(in super::super) const FF: usize = 640;
    pub(in super::super) const EXPERTS: usize = 512;
    pub(in super::super) const N_USED: usize = 10;
    /// The selector's heads, key width, pool and the tokens it keeps.
    pub(in super::super) const IDX_HEADS: usize = 4;
    pub(in super::super) const IDX_DIM: usize = 128;
    pub(in super::super) const POOL: usize = 4;
    pub(in super::super) const TOP_K: usize = 2048;
    /// Pools a selecting row keeps.
    pub(in super::super) const KEPT: usize = TOP_K / POOL;
    /// The PLE site's conv.
    pub(in super::super) const PLE_TAPS: usize = 4;
    pub(in super::super) const PLE_DILATION: usize = 3;
    /// The query projection's rows a token: each head's `[q | gate]`.
    pub(in super::super) const Q_ROWS: usize = 2 * N_HEAD * HEAD;
    /// A token's attention rows, and a delta layer's gated-norm rows.
    pub(in super::super) const ATTN: usize = N_HEAD * HEAD;
    /// A token's key (and value) rows.
    pub(in super::super) const KV: usize = N_KV * HEAD;
}

// The kernels' own constants, which the geometry must equal.
const _: () = assert!(
    geo::HIDDEN == ple::ROW
        && geo::HEAD == flash_gqa::HEAD_256
        && geo::HEAD == q38::HEAD
        && geo::IDX_DIM == qsa::DIM
        && geo::IDX_DIM == q38::DIM
        && geo::IDX_HEADS == qsa::HEADS
        && geo::POOL == qsa::POOL
        && geo::ROPE == qsa::ROT
        && geo::PLE_TAPS == ple::TAPS
        && geo::PLE_DILATION == ple::DILATION
        && geo::N_HEAD.is_multiple_of(flash_gqa::PACK_4 * geo::N_KV)
        && geo::V_HEADS * linear::HEAD == geo::ATTN
        && geo::KEPT * geo::POOL == geo::TOP_K
);

/// The delta layers' launch shape.
pub(super) const GDN: LinearShape = LinearShape {
    n_k: geo::K_HEADS,
    n_v: geo::V_HEADS,
    map: KHeadMap::Tiled,
};

/// A gated-residual mix site's weights: the streams' grouped gain, the
/// gate's down and up, and the combine's inject (`None` for the head).
pub(super) struct HcSite {
    pub(super) norm: String,
    pub(super) down: String,
    pub(super) up: String,
    pub(super) inject: Option<String>,
}

/// A delta layer's weights: the q·k·v and `z` projections (Q8_0), β and α
/// joined into one F32 stack (β's rows first), the conv, the decay's bias
/// and `A`, the gated norm's gain and the output projection (Q8_0).
pub(super) struct GdnPlan {
    pub(super) qkv: String,
    pub(super) z: String,
    pub(super) beta_alpha: String,
    pub(super) conv: String,
    pub(super) dt_bias: String,
    pub(super) ssm_a: String,
    pub(super) ssm_norm: String,
    pub(super) ssm_out: String,
}

/// A selecting attention layer's weights: q (with the gates), k, v and the
/// output (Q8_0), the q/k norms, the indexer's key and query projections
/// (BF16 widened to F32 on the card) and their norms.
pub(super) struct QsaPlan {
    pub(super) q: String,
    pub(super) k: String,
    pub(super) v: String,
    pub(super) q_norm: String,
    pub(super) k_norm: String,
    pub(super) out: String,
    pub(super) idx_k: String,
    pub(super) idx_q: String,
    pub(super) idx_k_norm: String,
    pub(super) idx_q_norm: String,
}

/// What a layer mixes its tokens with.
pub(super) enum Mixer38 {
    Gdn(GdnPlan),
    Qsa(QsaPlan),
}

/// A layer's feed-forward block on the card: the router joined with the
/// shared expert's gate as its last row, and the shared expert's three
/// projections (Q8_0). The routed stacks are the host tier's.
pub(super) struct FfnPlan {
    pub(super) router: String,
    pub(super) gate_sh: String,
    pub(super) up_sh: String,
    pub(super) down_sh: String,
}

/// The PLE site's weights: the key and value projections (Q8_0), the three
/// gains and the conv.
pub(super) struct PlePlan {
    pub(super) key: String,
    pub(super) value: String,
    pub(super) norm_key: String,
    pub(super) norm_query: String,
    pub(super) norm_conv: String,
    pub(super) conv: String,
}

/// One layer's plan.
pub(super) struct Layer38 {
    pub(super) attn_hc: HcSite,
    pub(super) mixer: Mixer38,
    pub(super) ffn_hc: HcSite,
    pub(super) ffn: FfnPlan,
    pub(super) ple: Option<PlePlan>,
}

/// The joined name of layer `l`'s β and α stack.
pub(super) fn beta_alpha(l: usize) -> String {
    format!("derived.blk.{l}.ssm_beta_alpha")
}

/// The joined name of layer `l`'s router with the shared gate.
pub(super) fn router(l: usize) -> String {
    format!("derived.blk.{l}.ffn_gate_inp_sh")
}

/// The head's mix site.
pub(super) fn head_site() -> HcSite {
    HcSite {
        norm: model::arch::qwen35moe::names::output_hc_norm(),
        down: model::arch::qwen35moe::names::output_hc_down(),
        up: model::arch::qwen35moe::names::output_hc_up(),
        inject: None,
    }
}

fn site(l: usize, sub: model::arch::qwen35moe::names::Sub) -> HcSite {
    HcSite {
        norm: model::arch::qwen35moe::names::hc_norm(l, sub),
        down: model::arch::qwen35moe::names::hc_down(l, sub),
        up: model::arch::qwen35moe::names::hc_up(l, sub),
        inject: Some(model::arch::qwen35moe::names::hc_inject(l, sub)),
    }
}

/// What the description says layer `l` runs, checked against [`geo`]; the
/// description's other layers are the caller's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Kind38 {
    Gdn,
    Qsa,
}

/// The chain's shape a description gives ([`read`]).
pub(super) struct Shape38 {
    /// Every layer's kind.
    pub(super) kinds: Vec<Kind38>,
    /// The router's instance.
    pub(super) router: RouterDims,
    /// The rope base.
    pub(super) base: f32,
    /// The PLE site's layer.
    pub(super) ple: usize,
}

/// The chain's shape from `spec`. A layer, a width or a rule the kernels are
/// not built for is refused by name with its layer.
pub(super) fn read(spec: &ModelSpec) -> Result<Shape38, GpuError> {
    let hc_ok = spec.hc.is_some_and(|h| {
        h.streams as usize == geo::STREAMS
            && matches!(h.kind, HcKind::Gated { rank } if rank as usize == geo::RANK)
    });
    let ple_ok = spec.engram.is_some_and(|e| {
        (e.max_ngram as usize - 1) * e.heads as usize * e.key_length as usize == geo::HIDDEN
            && matches!(e.rule, NgramRule::Ple { conv, dilation, .. }
                if conv as usize == geo::PLE_TAPS && dilation as usize == geo::PLE_DILATION)
    });
    if spec.hidden as usize != geo::HIDDEN || !hc_ok || !ple_ok || !spec.mtp.is_empty() {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "hidden {} streams {:?} engram {:?} mtp layers {}; the kernels take {} values in \
                 {} gated streams of rank {}, a PLE site of {} values a token through a {}-tap \
                 conv {} apart, and no next-token layer",
                spec.hidden,
                spec.hc,
                spec.engram,
                spec.mtp.len(),
                geo::HIDDEN,
                geo::STREAMS,
                geo::RANK,
                geo::HIDDEN,
                geo::PLE_TAPS,
                geo::PLE_DILATION
            ),
        ));
    }
    let mut kinds = Vec::with_capacity(spec.layers.len());
    let (mut base, mut ple, mut router) = (None, None, None);
    for (l, s) in spec.layers.iter().enumerate() {
        let refuse = |d: String| GpuError::shape(WHAT, format!("layer {l}: {d}"));
        if s.residual != Residual::Hc {
            return Err(refuse(
                "a plain residual; every layer runs in the streams".into(),
            ));
        }
        match s.extras.as_slice() {
            [] => {}
            [Extra::Ple] if ple.is_none() && l > 0 => ple = Some(l),
            other => {
                return Err(refuse(format!(
                    "extras {other:?}; the program runs one PLE site, on a layer after the first"
                )));
            }
        }
        kinds.push(match &s.mixer {
            Mixer::DeltaRule(d) => {
                let fits = d.kind
                    == DeltaKind::Gdn {
                        khead_map: SpecMap::Tiled,
                        gate: GdnGate::Sigmoid,
                    }
                    && (d.k_heads as usize, d.v_heads as usize) == (GDN.n_k, GDN.n_v)
                    && d.d as usize == linear::HEAD
                    && d.conv as usize == linear::CONV_TAPS;
                if !fits {
                    return Err(refuse(format!(
                        "delta rule {d:?}; the kernels take a sigmoid-gated GDN of {}/{} heads \
                         of {} (tiled) with a {}-tap conv",
                        GDN.n_k,
                        GDN.n_v,
                        linear::HEAD,
                        linear::CONV_TAPS
                    )));
                }
                Kind38::Gdn
            }
            Mixer::Gqa(g) => {
                if !gqa_fits(g) {
                    return Err(refuse(format!(
                        "attention {g:?}; the kernels take {}/{} heads of {}, q/k norms, the \
                         output gate, the first {} values turned by NEOX (IMROPE sections \
                         covering them) and the mean-pool selector of {} heads of {}, pools of \
                         {} and {} tokens kept",
                        geo::N_HEAD,
                        geo::N_KV,
                        geo::HEAD,
                        geo::ROPE,
                        geo::IDX_HEADS,
                        geo::IDX_DIM,
                        geo::POOL,
                        geo::TOP_K
                    )));
                }
                match base {
                    None => base = Some(g.rope.base),
                    Some(b) if b.to_bits() == g.rope.base.to_bits() => {}
                    Some(b) => {
                        return Err(refuse(format!(
                            "rope base {} after {b}; the table is one base's",
                            g.rope.base
                        )));
                    }
                }
                Kind38::Qsa
            }
            Mixer::Latent(_) => {
                return Err(refuse(
                    "a latent-attention mixer, which no kernel here runs".into(),
                ));
            }
        });
        let Ffn::Moe(m) = &s.ffn else {
            return Err(refuse("a dense FFN; the program routes every layer".into()));
        };
        let shared_ok = m.shared.is_some_and(|sh| {
            sh.ff as usize == geo::FF && sh.sigmoid_gate && sh.act == Act::SwiGlu { limit: None }
        });
        let plain = m.router.score == Score::Softmax
            && m.router.norm
            && !m.router.bias
            && !m.router.hash
            && m.router.scale == 1.0;
        let dims = RouterDims::of(MoeShape::of(m)).map_err(|e| refuse(e.to_string()))?;
        let fits = m.experts as usize == geo::EXPERTS
            && m.top_k as usize == geo::N_USED
            && m.expert_ff as usize == geo::FF
            && m.act == Act::SwiGlu { limit: None }
            && plain
            && shared_ok
            && dims.gated()
            && dims.slots() == geo::N_USED + 1;
        if !fits {
            return Err(refuse(format!(
                "experts {m:?}; the program routes {}/{} SwiGLU experts of {} by a renormalized \
                 softmax at scale 1, with a sigmoid-gated shared expert of {}",
                geo::EXPERTS,
                geo::N_USED,
                geo::FF,
                geo::FF
            )));
        }
        router.get_or_insert(dims);
    }
    let base = base.ok_or(GpuError::shape(
        WHAT,
        "no attention layer to read the rope base from",
    ))?;
    let ple = ple.ok_or(GpuError::shape(WHAT, "no PLE site on any layer"))?;
    let router = router.ok_or(GpuError::shape(WHAT, "no layer"))?;
    Ok(Shape38 {
        kinds,
        router,
        base,
        ple,
    })
}

/// Whether `g` is the selecting gated attention the kernels run.
fn gqa_fits(g: &Gqa) -> bool {
    let rope_ok = |r: &model::arch::models::Rope| {
        turns_as_neox(r.mode, r.dims) && r.dims as usize == geo::ROPE && r.yarn.is_none()
    };
    let select_ok = match g.select {
        Some(Selector::TokenPool {
            heads,
            d,
            top_k,
            pool,
            rule: PoolRule::Mean { rope },
        }) => {
            (heads as usize, d as usize, top_k as usize, pool as usize)
                == (geo::IDX_HEADS, geo::IDX_DIM, geo::TOP_K, geo::POOL)
                && rope_ok(&rope)
                && rope.base.to_bits() == g.rope.base.to_bits()
        }
        _ => false,
    };
    (g.heads as usize, g.kv_heads as usize, g.head_dim as usize)
        == (geo::N_HEAD, geo::N_KV, geo::HEAD)
        && g.qk_norm
        && g.out_gate
        && rope_ok(&g.rope)
        && select_ok
}

/// Every layer's plan from its kind, `ple` the PLE site's layer, each weight
/// checked against the geometry: type (the q8f32 planes for Q8_0, F32 for
/// the gains, the joined stacks and the indexer's widened projections) and
/// shape.
pub(super) fn plans(w: &Weights, kinds: &[Kind38], ple: usize) -> Result<Vec<Layer38>, GpuError> {
    use geo::{ATTN, EXPERTS, FF, HEAD, HIDDEN, IDX_DIM, IDX_HEADS, KV, Q_ROWS, RANK, STREAMS};
    let wide = STREAMS * HIDDEN;
    let check_site = |s: &HcSite| -> Result<(), GpuError> {
        f32_len(w, &s.norm, wide)?;
        q8_shape(w, &s.down, RANK, wide)?;
        q8_shape(w, &s.up, wide, RANK)?;
        if let Some(inj) = &s.inject {
            f32_shape(w, inj, STREAMS, wide)?;
        }
        Ok(())
    };
    check_site(&head_site())?;
    kinds
        .iter()
        .enumerate()
        .map(|(l, kind)| {
            let mixer = match kind {
                Kind38::Gdn => {
                    let p = GdnPlan {
                        qkv: model::arch::qwen35moe::names::attn_qkv(l),
                        z: model::arch::qwen35moe::names::attn_gate(l),
                        beta_alpha: beta_alpha(l),
                        conv: model::arch::qwen35moe::names::ssm_conv1d(l),
                        dt_bias: model::arch::qwen35moe::names::ssm_dt_bias(l),
                        ssm_a: model::arch::qwen35moe::names::ssm_a(l),
                        ssm_norm: model::arch::qwen35moe::names::ssm_norm(l),
                        ssm_out: model::arch::qwen35moe::names::ssm_out(l),
                    };
                    let (c, nv) = (GDN.channels(), GDN.n_v);
                    q8_shape(w, &p.qkv, c, HIDDEN)?;
                    q8_shape(w, &p.z, nv * linear::HEAD, HIDDEN)?;
                    f32_shape(w, &p.beta_alpha, 2 * nv, HIDDEN)?;
                    f32_len(w, &p.conv, c * linear::CONV_TAPS)?;
                    f32_len(w, &p.dt_bias, nv)?;
                    f32_len(w, &p.ssm_a, nv)?;
                    f32_len(w, &p.ssm_norm, linear::HEAD)?;
                    q8_shape(w, &p.ssm_out, HIDDEN, nv * linear::HEAD)?;
                    Mixer38::Gdn(p)
                }
                Kind38::Qsa => {
                    let p = QsaPlan {
                        q: model::arch::qwen35moe::names::attn_q(l),
                        k: model::arch::qwen35moe::names::attn_k(l),
                        v: model::arch::qwen35moe::names::attn_v(l),
                        q_norm: model::arch::qwen35moe::names::attn_q_norm(l),
                        k_norm: model::arch::qwen35moe::names::attn_k_norm(l),
                        out: model::arch::qwen35moe::names::attn_output(l),
                        idx_k: model::arch::qwen35moe::names::indexer_k_proj(l),
                        idx_q: model::arch::qwen35moe::names::indexer_q_proj(l),
                        idx_k_norm: model::arch::qwen35moe::names::indexer_k_norm(l),
                        idx_q_norm: model::arch::qwen35moe::names::indexer_q_norm(l),
                    };
                    q8_shape(w, &p.q, Q_ROWS, HIDDEN)?;
                    q8_shape(w, &p.k, KV, HIDDEN)?;
                    q8_shape(w, &p.v, KV, HIDDEN)?;
                    f32_len(w, &p.q_norm, HEAD)?;
                    f32_len(w, &p.k_norm, HEAD)?;
                    q8_shape(w, &p.out, HIDDEN, ATTN)?;
                    f32_shape(w, &p.idx_k, IDX_DIM, HIDDEN)?;
                    f32_shape(w, &p.idx_q, IDX_HEADS * IDX_DIM, HIDDEN)?;
                    f32_len(w, &p.idx_k_norm, IDX_DIM)?;
                    f32_len(w, &p.idx_q_norm, IDX_DIM)?;
                    Mixer38::Qsa(p)
                }
            };
            let ffn = FfnPlan {
                router: router(l),
                gate_sh: model::arch::qwen35moe::names::ffn_gate_shexp(l),
                up_sh: model::arch::qwen35moe::names::ffn_up_shexp(l),
                down_sh: model::arch::qwen35moe::names::ffn_down_shexp(l),
            };
            f32_shape(w, &ffn.router, EXPERTS + 1, HIDDEN)?;
            q8_shape(w, &ffn.gate_sh, FF, HIDDEN)?;
            q8_shape(w, &ffn.up_sh, FF, HIDDEN)?;
            q8_shape(w, &ffn.down_sh, HIDDEN, FF)?;
            let ple = if l == ple {
                let p = PlePlan {
                    key: model::arch::qwen35moe::names::ple_key(l),
                    value: model::arch::qwen35moe::names::ple_value(l),
                    norm_key: model::arch::qwen35moe::names::ple_norm_key(l),
                    norm_query: model::arch::qwen35moe::names::ple_norm_query(l),
                    norm_conv: model::arch::qwen35moe::names::ple_norm_conv(l),
                    conv: model::arch::qwen35moe::names::ple_conv1d(l),
                };
                q8_shape(w, &p.key, wide, HIDDEN)?;
                q8_shape(w, &p.value, HIDDEN, HIDDEN)?;
                for g in [&p.norm_key, &p.norm_query, &p.norm_conv] {
                    f32_len(w, g, wide)?;
                }
                f32_len(w, &p.conv, geo::PLE_TAPS * wide)?;
                Some(p)
            } else {
                None
            };
            let (attn_hc, ffn_hc) = (
                site(l, model::arch::qwen35moe::names::Sub::Attn),
                site(l, model::arch::qwen35moe::names::Sub::Ffn),
            );
            check_site(&attn_hc)?;
            check_site(&ffn_hc)?;
            Ok(Layer38 {
                attn_hc,
                mixer,
                ffn_hc,
                ffn,
                ple,
            })
        })
        .collect()
}

/// The resident Q8_0 planes of `name`, `rows` rows of `k` values; any other
/// variant or shape is refused by name.
pub(super) fn q8_shape(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    match w.get(name) {
        Some(DevWeight::Q8_0 { d, k: got, .. }) if d.rows() == rows && *got == k => Ok(()),
        Some(DevWeight::Q8_0 { d, k: got, .. }) => Err(GpuError::shape(
            WHAT,
            format!(
                "{name} is {} rows of {got}, the kernels read {rows} of {k}",
                d.rows()
            ),
        )),
        Some(_) => Err(GpuError::tensor(WHAT, name, "Q8_0 (the q8f32 planes)")),
        None => Err(GpuError::tensor(WHAT, name, "resident")),
    }
}

/// The resident F32 plane of `name`, `rows` rows of `k`; refused otherwise.
fn f32_shape(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w: t, .. }) if t.rows() == rows && t.cols() == k => Ok(()),
        Some(DevWeight::F32 { w: t, .. }) => Err(GpuError::shape(
            WHAT,
            format!(
                "{name} is {} rows of {}, the kernels read {rows} of {k}",
                t.rows(),
                t.cols()
            ),
        )),
        Some(_) => Err(GpuError::tensor(WHAT, name, "F32")),
        None => Err(GpuError::tensor(WHAT, name, "resident")),
    }
}

/// The resident F32 plane of `name`, `len` values in all; refused otherwise.
fn f32_len(w: &Weights, name: &str, len: usize) -> Result<(), GpuError> {
    match w.get(name) {
        Some(DevWeight::F32 { w: t, .. }) if t.buf().len() == len => Ok(()),
        Some(DevWeight::F32 { w: t, .. }) => Err(GpuError::shape(
            WHAT,
            format!(
                "{name} holds {} values, the kernels read {len}",
                t.buf().len()
            ),
        )),
        Some(_) => Err(GpuError::tensor(WHAT, name, "F32")),
        None => Err(GpuError::tensor(WHAT, name, "resident")),
    }
}
