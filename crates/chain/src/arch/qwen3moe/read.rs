//! A layer's plan read from the file's header: every weight's name, and the type of each site
//! checked against the shape the plan's kernels take. Nothing is uploaded and no tensor byte is
//! read; a layer the kernels do not run, a type no launch reads and a shape that disagrees are
//! refused by name with the layer.
//!
//! Each reader takes `what`, the loader's entry-point name its refusals are attributed to.

use super::plan::{
    DeltaPlan, FfnPlan, FfnRoute, Flash, GqaKind, GqaPlan, SharedPlan, SharedSites, SiteTy,
    down_entry, gate_up_entry, shared_entry,
};
use crate::ChainError;
use crate::linear::{self, LinearShape};
use crate::site::{file_site, shaped};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::{Ffn, LayerSpec};
use model::arch::names::blk;
use model::placement::accepts::{self, LegSplit};

/// The row counts a plan reads its sites at, from the file's shapes.
#[derive(Clone, Copy, Debug)]
pub struct PlanDims {
    /// Values of the residual stream.
    pub hidden: usize,
    /// A routed expert's values, or a dense FFN's.
    pub ff: usize,
    /// Rows the query projection writes a token: the attention rows, twice that when it writes a
    /// gate beside each head's query.
    pub q_rows: usize,
    /// Values of a token's key (or value) rows.
    pub kv_len: usize,
    /// Values of a token's attention rows.
    pub attn_len: usize,
}

/// The joined name of layer `l`'s stack `part` (`gate`, `up`, `down`).
pub fn joint(l: usize, part: &str) -> String {
    format!("derived.blk.{l}.ffn_{part}_exps_sh")
}

/// The joined name of layer `l`'s router weight with its shared gate as the last row.
pub fn joint_gate_inp(l: usize) -> String {
    format!("derived.blk.{l}.ffn_gate_inp_sh")
}

/// The types a projection site launches at: a projection of the normed rows or
/// an output projection, a K-quant or Q8_0.
const PROJ: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
];
/// [`PROJ`] and F32: the types β and α launch at, and those a routed layer's
/// stacks and its shared expert are read at, which the card kernel table's
/// entries then admit ([`plan_ffn`]); under a placed read the stacks of a layer
/// the dispatch's launches do not run are typed as the host tier's instead
/// ([`card_leg`]): a layer the common leg runs holds its stacks to the placed
/// side, which reads their types from the resident stacks.
const PROJ_F32: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The type of site `name` (`rows` rows of `k`) from `file`'s header.
fn ty_of(
    file: &Split,
    what: &'static str,
    name: &str,
    rows: usize,
    k: usize,
    allowed: &[SiteTy],
) -> Result<SiteTy, ChainError> {
    file_site(file, what, name, (rows, k), allowed)
}

/// What a read of the file holds a layer's routed stacks to.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Read35 {
    /// A whole load: every routed stack runs on the card, so each is read at
    /// a site type and admitted by the card kernel table's entries.
    Whole,
    /// A placed load: a layer whose stacks the card leg does not run is the
    /// host tier's, and its stacks are read as `HOSTED` — each stack still
    /// refused by name when no tier runs its type.
    Placed,
}

/// The type a placed plan holds a host layer's routed stacks as. No launch
/// reads them: the layer has no card expert (a placed load refuses card
/// experts of a layer its leg does not run), and an F32 site reads no
/// quantized form ([`SiteTy::reads`]), so no backend counts a form for them:
/// the layer's card work is its shared expert's alone.
pub(crate) const HOSTED: SiteTy = SiteTy::F32;

/// The file type of routed stack `name` of `rows` rows of `k` values, from
/// its header entry: the type itself, where `file_site` names only the types
/// a card site launches. A missing tensor or another shape is refused by
/// name.
fn stack_ty(
    file: &Split,
    what: &'static str,
    name: &str,
    shape: (usize, usize),
) -> Result<GgmlType, ChainError> {
    Ok(shaped(file, what, name, shape)?.ty)
}

/// The leg that runs layer `l`'s routed stacks (`n` experts of `ff` rows of
/// `h` values) on the placed card, with the stacks' file types, read from
/// `file`'s header before any upload: the planner's own verdict
/// ([`accepts::qwen35_leg`]), `None` for a layer the host tier runs. A stack of
/// a type that neither the card leg's arm for its part ([`accepts::leg_arms`])
/// nor the host tier's fused rows ([`model::ops::host_stack_runs`]) runs is
/// refused by name, with the layer: no plan can place it. Both qwen3moe
/// bodies' placed loads read their legs by it.
pub fn card_leg(
    file: &Split,
    what: &'static str,
    l: usize,
    (ff, h, n): (usize, usize, usize),
) -> Result<Option<(LegSplit, [GgmlType; 3])>, ChainError> {
    let program = accepts::Program::Qwen35Body;
    let arms = accepts::leg_arms(program)
        .ok_or_else(|| ChainError::shape(what, "the placed side's body runs no card leg"))?;
    let stack = |part: &str,
                 (rows, k): (usize, usize),
                 (kind, arm): (&str, fn(GgmlType) -> bool)|
     -> Result<GgmlType, ChainError> {
        let name = blk(l, &format!("ffn_{part}_exps.weight"));
        let ty = stack_ty(file, what, &name, (n * rows, k))?;
        if !arm(ty) {
            model::ops::host_stack_runs("the host tier", &name, ty, k).map_err(|e| {
                ChainError::shape(
                    what,
                    format!(
                        "layer {l}: the {part} stack is {ty}, which runs on no tier: the card \
                         leg's {kind} launches read none; {e}"
                    ),
                )
            })?;
        }
        Ok(ty)
    };
    let gate_up = ("gate·up", arms.gate_up);
    let tys = [
        stack("gate", (ff, h), gate_up)?,
        stack("up", (ff, h), gate_up)?,
        stack("down", (h, ff), ("down", arms.down))?,
    ];
    Ok(accepts::qwen35_leg(tys).map(|leg| (leg, tys)))
}

/// The types of routed part `part` of layer `l` from `file`'s header: its
/// experts' stack (`experts` experts of `rows` rows of `k` values; [`HOSTED`]
/// for a layer the host tier runs) and its shared expert's matrix (`rows` of
/// `k`).
fn part_tys(
    file: &Split,
    what: &'static str,
    l: usize,
    part: &str,
    (rows, k, experts): (usize, usize, usize),
    hosted: bool,
) -> Result<(SiteTy, SiteTy), ChainError> {
    let exps = blk(l, &format!("ffn_{part}_exps.weight"));
    let sh = blk(l, &format!("ffn_{part}_shexp.weight"));
    let routed = if hosted {
        HOSTED
    } else {
        ty_of(file, what, &exps, experts * rows, k, PROJ_F32)?
    };
    Ok((routed, ty_of(file, what, &sh, rows, k, PROJ_F32)?))
}

/// Layer `l`'s FFN by its description `spec`, every site's type from
/// `file`'s header. A routed one: its stacks' types admitted by the card
/// kernel table's entries (`gate_up_entry`, `down_entry`) and
/// its shared expert's by its dense sites (`shared_entry`), each
/// refused by name with the layer; the router joined with the shared gate
/// as row `n` (the routed count, which a backend's load joins); and the shared expert folded
/// into the stacks as expert `n` when its gate, up and down are each its
/// stack's type ([`SharedPlan::Joined`]), else kept apart at its own types
/// ([`SharedPlan::Apart`]). Under a placed read ([`Read35::Placed`]) a layer
/// whose stacks the dispatch's launches do not run ([`card_leg`]) is typed as
/// the host tier's: its stacks are read as `HOSTED` and no card entry is
/// asked of them, its shared expert kept apart at its own types, which the
/// card runs whatever the stacks are; the common leg's layers
/// ([`LegSplit::Common`]) are among them, their types the placed side's to
/// read. A dense one: its three matrices a stack of one expert on the
/// arena's fixed route.
pub fn plan_ffn(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    spec: &LayerSpec,
    l: usize,
    read: Read35,
) -> Result<FfnPlan, ChainError> {
    let (h, ff) = (d.hidden, d.ff);
    let ffn_norm = blk(l, "post_attention_norm.weight");
    match &spec.ffn {
        Ffn::Moe(m) => {
            let n = m.experts as usize;
            let hosted = read == Read35::Placed
                && !matches!(
                    card_leg(file, what, l, (ff, h, n))?,
                    Some((LegSplit::Dispatch, _))
                );
            let (gate_ty, sh_gate) = part_tys(file, what, l, "gate", (ff, h, n), hosted)?;
            let (up_ty, sh_up) = part_tys(file, what, l, "up", (ff, h, n), hosted)?;
            let (down_ty, sh_down) = part_tys(file, what, l, "down", (h, ff, n), hosted)?;
            let at = |e: ChainError| ChainError::shape(what, format!("layer {l}: {e}"));
            if !hosted {
                gate_up_entry(gate_ty, up_ty, what).map_err(at)?;
                down_entry(down_ty, what).map_err(at)?;
            }
            for ty in [sh_gate, sh_up, sh_down] {
                shared_entry(ty, what).map_err(at)?;
            }
            let routed = |part: &str| blk(l, &format!("ffn_{part}_exps.weight"));
            let (shared, [gate, up, down]) =
                if !hosted && (gate_ty, up_ty, down_ty) == (sh_gate, sh_up, sh_down) {
                    let stacks = ["gate", "up", "down"].map(|part| joint(l, part));
                    (SharedPlan::Joined, stacks)
                } else {
                    let sites = SharedSites {
                        gate: blk(l, "ffn_gate_shexp.weight"),
                        up: blk(l, "ffn_up_shexp.weight"),
                        down: blk(l, "ffn_down_shexp.weight"),
                        gate_ty: sh_gate,
                        up_ty: sh_up,
                        down_ty: sh_down,
                    };
                    let stacks = ["gate", "up", "down"].map(routed);
                    (SharedPlan::Apart(sites), stacks)
                };
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Router {
                    gate_inp: joint_gate_inp(l),
                    shared: Some(shared),
                },
                gate,
                up,
                down,
                gate_ty,
                up_ty,
                down_ty,
            })
        }
        Ffn::Dense { .. } => {
            let (gate, up, down) = (
                blk(l, "ffn_gate.weight"),
                blk(l, "ffn_up.weight"),
                blk(l, "ffn_down.weight"),
            );
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Dense,
                gate_ty: ty_of(file, what, &gate, ff, h, PROJ)?,
                up_ty: ty_of(file, what, &up, ff, h, PROJ)?,
                down_ty: ty_of(file, what, &down, h, ff, PROJ)?,
                gate,
                up,
                down,
            })
        }
    }
}

/// Layer `l`'s gated attention over the flash `flash` its group selects,
/// every site's type from `file`'s header.
pub fn plan_gqa(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    l: usize,
    flash: Flash,
) -> Result<GqaPlan, ChainError> {
    let (h, kv, att) = (d.hidden, d.kv_len, d.attn_len);
    let (q, k, v, o) = (
        blk(l, "attn_q.weight"),
        blk(l, "attn_k.weight"),
        blk(l, "attn_v.weight"),
        blk(l, "attn_output.weight"),
    );
    Ok(GqaPlan {
        kind: GqaKind::Gated256,
        flash,
        attn_norm: blk(l, "attn_norm.weight"),
        attn_q_norm: blk(l, "attn_q_norm.weight"),
        attn_k_norm: blk(l, "attn_k_norm.weight"),
        q_ty: ty_of(file, what, &q, d.q_rows, h, PROJ)?,
        k_ty: ty_of(file, what, &k, kv, h, PROJ)?,
        v_ty: ty_of(file, what, &v, kv, h, PROJ)?,
        o_ty: ty_of(file, what, &o, h, att, PROJ)?,
        attn_q: q,
        attn_k: k,
        attn_v: v,
        attn_output: o,
    })
}

/// Layer `l`'s delta rule, every site's type from `file`'s header.
pub fn plan_delta(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    shape: LinearShape,
    l: usize,
) -> Result<DeltaPlan, ChainError> {
    let (h, c, nv) = (d.hidden, shape.channels(), shape.n_v);
    let (qkv, gate, beta, alpha, out) = (
        blk(l, "attn_qkv.weight"),
        blk(l, "attn_gate.weight"),
        blk(l, "ssm_beta.weight"),
        blk(l, "ssm_alpha.weight"),
        blk(l, "ssm_out.weight"),
    );
    Ok(DeltaPlan {
        attn_norm: blk(l, "attn_norm.weight"),
        conv: blk(l, "ssm_conv1d.weight"),
        dt_bias: blk(l, "ssm_dt.bias"),
        ssm_a: blk(l, "ssm_a"),
        ssm_norm: blk(l, "ssm_norm.weight"),
        shape,
        qkv_ty: ty_of(file, what, &qkv, c, h, PROJ)?,
        gate_ty: ty_of(file, what, &gate, nv * linear::HEAD, h, PROJ)?,
        beta_ty: ty_of(file, what, &beta, nv, h, PROJ_F32)?,
        alpha_ty: ty_of(file, what, &alpha, nv, h, PROJ_F32)?,
        out_ty: ty_of(file, what, &out, h, nv * linear::HEAD, PROJ)?,
        qkv,
        gate,
        beta,
        alpha,
        ssm_out: out,
    })
}
