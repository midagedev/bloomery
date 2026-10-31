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

use crate::hc_gated::HcMatrix;
use crate::linear::{self, KHeadMap, LinearShape};
use crate::route_softmax::RouterDims;
use crate::site::{self as dense, SiteTy};
use crate::weights::{DevWeight, Weights};
use crate::{GpuError, flash_gqa, ple, q38, qsa};
use model::arch::coverage::turns_as_neox;
use model::arch::models::shape::MoeShape;
use model::arch::models::{
    Act, DeltaKind, Extra, Ffn, GdnGate, Gqa, HcKind, KHeadMap as SpecMap, Mixer, ModelSpec,
    NgramRule, OutGate, PoolRule, Residual, RouterNorm, Score, Selector,
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

/// The launches a layer's by-name projections ([`dense::ByName`]) add to the
/// one launch an f32-row site makes, from the resident types of its sites
/// ([`dense::by_name_extra`]): at one column and at more than one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ByLaunches {
    pub(super) one: usize,
    pub(super) rows: usize,
}

/// One layer's plan.
pub(super) struct Layer38 {
    pub(super) attn_hc: HcSite,
    pub(super) mixer: Mixer38,
    pub(super) ffn_hc: HcSite,
    pub(super) ffn: FfnPlan,
    pub(super) ple: Option<PlePlan>,
    /// The by-name projections' extra launches.
    pub(super) by: ByLaunches,
}

impl GdnPlan {
    /// The sites that read the mix's rows.
    pub(super) fn input_sites(&self) -> [&str; 3] {
        [&self.qkv, &self.z, &self.beta_alpha]
    }
}

impl QsaPlan {
    /// The sites that read the mix's rows.
    pub(super) fn input_sites(&self) -> [&str; 5] {
        [&self.q, &self.k, &self.v, &self.idx_k, &self.idx_q]
    }
}

impl FfnPlan {
    /// The shared expert's sites that read the block's input rows.
    pub(super) fn shared_inputs(&self) -> [&str; 2] {
        [&self.gate_sh, &self.up_sh]
    }
}

impl PlePlan {
    /// The sites that read the PLE rows.
    pub(super) fn input_sites(&self) -> [&str; 2] {
        [&self.key, &self.value]
    }
}

impl Layer38 {
    /// The layer's sites that read model-width rows: the mixer's, the shared
    /// expert's gate and up, and the PLE site's.
    pub(super) fn hidden_sites(&self) -> Vec<&str> {
        let mut sites = match &self.mixer {
            Mixer38::Gdn(p) => p.input_sites().to_vec(),
            Mixer38::Qsa(p) => p.input_sites().to_vec(),
        };
        sites.extend(self.ffn.shared_inputs());
        sites.extend(self.ple.iter().flat_map(PlePlan::input_sites));
        sites
    }

    /// The layer's site that reads attention-width rows: the mixer's output
    /// projection.
    pub(super) fn attn_site(&self) -> &str {
        match &self.mixer {
            Mixer38::Gdn(p) => &p.ssm_out,
            Mixer38::Qsa(p) => &p.out,
        }
    }

    /// The layer's dense sites a walk projects by name ([`dense::ByName`]):
    /// the mixer's, the shared expert's three and the PLE site's. The joined
    /// β·α stack is not among them: it reads the f32 rows, whole at every
    /// row count.
    pub(super) fn projected(&self) -> Vec<&str> {
        let mut sites: Vec<&str> = match &self.mixer {
            Mixer38::Gdn(p) => vec![&p.qkv, &p.z, &p.ssm_out],
            Mixer38::Qsa(p) => vec![&p.q, &p.k, &p.v, &p.out, &p.idx_k, &p.idx_q],
        };
        sites.extend([&*self.ffn.gate_sh, &*self.ffn.up_sh, &*self.ffn.down_sh]);
        sites.extend(self.ple.iter().flat_map(PlePlan::input_sites));
        sites
    }

    /// The launches the layer's by-name projections add to a walk over `m`
    /// columns ([`ByLaunches`]).
    pub(super) fn by_extra(&self, m: usize) -> usize {
        if m > 1 { self.by.rows } else { self.by.one }
    }
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
            other @ (Mixer::Latent(_) | Mixer::Mamba2(_) | Mixer::ShortConv(_) | Mixer::None) => {
                return Err(refuse(format!(
                    "{}, which no kernel here runs",
                    other.what()
                )));
            }
        });
        let Ffn::Moe(m) = &s.ffn else {
            return Err(refuse("a dense FFN; the program routes every layer".into()));
        };
        let shared_ok = m.shared.is_some_and(|sh| {
            sh.ff as usize == geo::FF && sh.sigmoid_gate && sh.act == Act::SwiGlu { limit: None }
        });
        let plain = m.router.score == Score::Softmax
            && m.router.norm == RouterNorm::SelectedSum
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
        turns_as_neox(r.mode, r.dims)
            && r.dims as usize == geo::ROPE
            && r.yarn.is_none()
            && !r.factors
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
        && g.out_gate == OutGate::Packed
        && rope_ok(&g.rope)
        && select_ok
}

/// Every layer's plan from its kind, `ple` the PLE site's layer, each weight
/// checked against the geometry: type and shape. A dense projection, the
/// indexer's two among them, is of a type the common site projects by name
/// (`site::PROJECTED`: the f32 rows' families, the K-quants and IQ4_XS), each
/// tensor's own: the walks pick its launch by the resident form. The joined
/// β·α stack is of an f32 rows' family (`site::BY_NAME`). An HC matrix is one
/// `HcMatrix::of` reads. The gains, the conv and the injects are F32 (the
/// load widens a BF16 one, `ModelTensors::bf16_words`); the router is F32 or
/// a BF16 one's words, which its kernels read either way.
pub(super) fn plans(w: &Weights, kinds: &[Kind38], ple: usize) -> Result<Vec<Layer38>, GpuError> {
    use geo::{ATTN, EXPERTS, FF, HEAD, HIDDEN, IDX_DIM, IDX_HEADS, KV, Q_ROWS, RANK, STREAMS};
    let wide = STREAMS * HIDDEN;
    let check_site = |s: &HcSite| -> Result<(), GpuError> {
        f32_len(w, &s.norm, wide)?;
        hc_shape(w, &s.down, RANK, wide)?;
        hc_shape(w, &s.up, wide, RANK)?;
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
                    projected_shape(w, &p.qkv, c, HIDDEN)?;
                    projected_shape(w, &p.z, nv * linear::HEAD, HIDDEN)?;
                    dense_shape(w, &p.beta_alpha, 2 * nv, HIDDEN)?;
                    f32_len(w, &p.conv, c * linear::CONV_TAPS)?;
                    f32_len(w, &p.dt_bias, nv)?;
                    f32_len(w, &p.ssm_a, nv)?;
                    f32_len(w, &p.ssm_norm, linear::HEAD)?;
                    projected_shape(w, &p.ssm_out, HIDDEN, nv * linear::HEAD)?;
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
                    projected_shape(w, &p.q, Q_ROWS, HIDDEN)?;
                    projected_shape(w, &p.k, KV, HIDDEN)?;
                    projected_shape(w, &p.v, KV, HIDDEN)?;
                    f32_len(w, &p.q_norm, HEAD)?;
                    f32_len(w, &p.k_norm, HEAD)?;
                    projected_shape(w, &p.out, HIDDEN, ATTN)?;
                    projected_shape(w, &p.idx_k, IDX_DIM, HIDDEN)?;
                    projected_shape(w, &p.idx_q, IDX_HEADS * IDX_DIM, HIDDEN)?;
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
            dense::resident_site(
                w,
                WHAT,
                &ffn.router,
                (EXPERTS + 1, HIDDEN),
                &[SiteTy::F32, SiteTy::Bf16],
            )?;
            projected_shape(w, &ffn.gate_sh, FF, HIDDEN)?;
            projected_shape(w, &ffn.up_sh, FF, HIDDEN)?;
            projected_shape(w, &ffn.down_sh, HIDDEN, FF)?;
            let ple = if l == ple {
                let p = PlePlan {
                    key: model::arch::qwen35moe::names::ple_key(l),
                    value: model::arch::qwen35moe::names::ple_value(l),
                    norm_key: model::arch::qwen35moe::names::ple_norm_key(l),
                    norm_query: model::arch::qwen35moe::names::ple_norm_query(l),
                    norm_conv: model::arch::qwen35moe::names::ple_norm_conv(l),
                    conv: model::arch::qwen35moe::names::ple_conv1d(l),
                };
                projected_shape(w, &p.key, wide, HIDDEN)?;
                projected_shape(w, &p.value, HIDDEN, HIDDEN)?;
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
            let mut layer = Layer38 {
                attn_hc,
                mixer,
                ffn_hc,
                ffn,
                ple,
                by: ByLaunches::default(),
            };
            layer.by = by_launches(w, &layer)?;
            Ok(layer)
        })
        .collect()
}

/// The dense sites of `plans` whose resident type reads q8_1 rows (the
/// K-quants, NVFP4): the sites a [`dense::ByName`] holds forms for. None on a
/// file whose sites all read the f32 rows or q8 blocks of 32.
pub(super) fn q8_1_sites<'p>(w: &Weights, plans: &'p [Layer38]) -> Result<Vec<&'p str>, GpuError> {
    let mut sites = Vec::new();
    for name in plans.iter().flat_map(Layer38::projected) {
        if dense::resident_type(w, WHAT, name)?.reads() == dense::Form::Q8x128 {
            sites.push(name);
        }
    }
    Ok(sites)
}

/// The extra launches of `layer`'s by-name projections from its sites'
/// resident types.
fn by_launches(w: &Weights, layer: &Layer38) -> Result<ByLaunches, GpuError> {
    let mut by = ByLaunches::default();
    for name in layer.projected() {
        let ty = dense::resident_type(w, WHAT, name)?;
        by.one += dense::by_name_extra(ty, 1);
        by.rows += dense::by_name_extra(ty, 2);
    }
    Ok(by)
}

/// The resident weight `name`, `rows` rows of `k` values, of a type the
/// common dense site launches by name on both walks (`site::BY_NAME`): the f32
/// rows' families, which the joined β·α stack reads whole.
fn dense_shape(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    dense::resident_site(w, WHAT, name, (rows, k), &dense::BY_NAME).map(drop)
}

/// Values of a K-quant super-block, which a q8_1 row holds whole.
const SUPER_BLOCK: usize = 256;

/// The resident weight `name`, `rows` rows of `k` values, of a type a
/// by-name projection launches (`site::PROJECTED`: every dense family but
/// Q5_0, the K-quants, IQ4_XS and NVFP4 over the q8_1 rows of their input); another
/// type or shape is refused by name.
fn projected_shape(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    let ty = dense::resident_site(w, WHAT, name, (rows, k), &dense::PROJECTED)?;
    if ty.reads() == dense::Form::Q8x128 && !k.is_multiple_of(SUPER_BLOCK) {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "{name} is {ty} over rows of {k} values: a q8_1 row is whole super-blocks of {SUPER_BLOCK}"
            ),
        ));
    }
    Ok(())
}

/// The resident HC matrix `name` in a form `HcMatrix::of` reads, `rows`
/// rows of `k` values; another form or shape is refused by name.
fn hc_shape(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    let got = HcMatrix::of(w, WHAT, name)?
        .shape()
        .map_err(|e| GpuError::shape(WHAT, format!("{name}: {e}")))?;
    if got == (rows, k) {
        Ok(())
    } else {
        Err(GpuError::shape(
            WHAT,
            format!(
                "{name} is {} rows of {}, the kernels read {rows} of {k}",
                got.0, got.1
            ),
        ))
    }
}

/// The resident F32 plane of `name`, `rows` rows of `k`; refused otherwise.
fn f32_shape(w: &Weights, name: &str, rows: usize, k: usize) -> Result<(), GpuError> {
    dense::site(w, WHAT, name, (rows, k), SiteTy::F32)
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
