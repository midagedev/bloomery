//! A layer's plan read from the file's header: every weight's name, and the type of each site
//! checked against the shape the plan's kernels take. Nothing is uploaded and no tensor byte is
//! read; a layer the kernels do not run, a type no launch reads and a shape that disagrees are
//! refused by name with the layer.
//!
//! Each reader takes `what`, the loader's entry-point name its refusals are attributed to.

use super::plan::{
    DeltaPlan, FfnPlan, FfnRoute, Flash, GqaKind, GqaPlan, HeadGate, Kind35, LayerPlan, MixerPlan,
    ResidualPlan, SandwichPlan, SharedPlan, SharedSites, ShortConvPlan, SiteTy, layer_gqa, q35,
};
use crate::ChainError;
use crate::linear::{self, LinearShape, Ring};
use crate::site::{file_site, shaped};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::{Ffn, Gqa, LayerSpec, Mixer, ModelSpec, OutGate, Residual};
use model::arch::names::blk;
use model::placement::accepts::{self, LegSplit};

/// The row counts a plan reads its sites at, from the file's shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl PlanDims {
    /// The row counts a plan reads its sites at, from `spec`'s description alone: its first
    /// attention layer's head counts and width, the gate rows beside each query its
    /// [`OutGate`] writes (none for a gate with a projection of its own, per head), and its
    /// first layer's FFN width (a layer with another count is refused by name when its own sites
    /// are read at these rows).
    pub fn of(spec: &ModelSpec, what: &'static str) -> Result<PlanDims, ChainError> {
        let g = first_gqa(spec, what)?;
        let gate = match g.out_gate {
            OutGate::None | OutGate::ProjHead => 1,
            OutGate::Packed => 2,
            OutGate::Proj => {
                return Err(ChainError::shape(
                    what,
                    "an output gate of a value a channel projected on its own (`attn_gate`), \
                     which no attention site here reads",
                ));
            }
        };
        let ff = match spec.layers.first().map(|l| &l.ffn) {
            Some(Ffn::Moe(m)) => m.expert_ff,
            Some(Ffn::Dense { ff, .. }) => *ff,
            None => return Err(ChainError::shape(what, "no layer to size the FFN for")),
        };
        let (heads, kv_heads, head) = (g.heads as usize, g.kv_heads as usize, g.head_dim as usize);
        Ok(PlanDims {
            hidden: spec.hidden as usize,
            ff: ff as usize,
            q_rows: gate * heads * head,
            kv_len: kv_heads * head,
            attn_len: heads * head,
        })
    }

    /// The row counts the card's qwen35moe plan reads its sites at, from `spec`'s description:
    /// its first attention layer's head counts and the geometry the card's kernels are built
    /// for — [`q35::HEAD_DIM`] values an attention head, a routed expert's [`q35::EXPERT_FF`].
    pub fn q35(spec: &ModelSpec, what: &'static str) -> Result<PlanDims, ChainError> {
        let g = first_gqa(spec, what)?;
        let (n_head, n_kv) = (g.heads as usize, g.kv_heads as usize);
        let head = q35::HEAD_DIM as usize;
        Ok(PlanDims {
            hidden: spec.hidden as usize,
            ff: q35::EXPERT_FF as usize,
            q_rows: 2 * n_head * head,
            kv_len: n_kv * head,
            attn_len: n_head * head,
        })
    }
}

/// `spec`'s first attention layer's mixer, or the named refusal of a description with none.
fn first_gqa<'a>(spec: &'a ModelSpec, what: &'static str) -> Result<&'a Gqa, ChainError> {
    spec.layers
        .iter()
        .find_map(|l| match &l.mixer {
            Mixer::Gqa(g) => Some(g),
            _ => None,
        })
        .ok_or_else(|| ChainError::shape(what, "no attention layer to size the attention rows for"))
}

/// The joined name of layer `l`'s stack `part` (`gate`, `up`, `down`).
pub fn joint(l: usize, part: &str) -> String {
    format!("derived.blk.{l}.ffn_{part}_exps_sh")
}

/// The joined name of layer `l`'s router weight with its shared gate as the last row.
pub fn joint_gate_inp(l: usize) -> String {
    format!("derived.blk.{l}.ffn_gate_inp_sh")
}

/// Layer `l`'s routed `part` stack (`gate`, `up`, `down`), the file's own name.
fn exps(l: usize, part: &str) -> String {
    blk(l, &format!("ffn_{part}_exps.weight"))
}

/// Layer `l`'s shared expert's `part` (`gate`, `up`, `down`), the file's own name.
fn shexp(l: usize, part: &str) -> String {
    blk(l, &format!("ffn_{part}_shexp.weight"))
}

/// A routed layer's weights as the file holds them, beside the joined names
/// its plan carries ([`joint`], [`joint_gate_inp`]): the router weight and
/// the shared expert's gate row beside it, the routed stacks, and the shared
/// expert's three parts. A load that folds the shared expert in joins these
/// into the plan's derived names; a backend that runs it apart from the
/// stacks loads them as they are.
pub struct FileNames {
    /// The router weight (`ffn_gate_inp.weight`).
    pub router: String,
    /// The shared expert's gate row (`ffn_gate_inp_shexp.weight`).
    pub router_gate: String,
    /// The routed stacks' file names.
    pub gate: String,
    pub up: String,
    pub down: String,
    /// The shared expert's parts' file names.
    pub sh_gate: String,
    pub sh_up: String,
    pub sh_down: String,
}

/// Layer `l`'s routed FFN's [`FileNames`]: the file's own names, whatever
/// the plan folds into derived ones.
pub fn file_names(l: usize) -> FileNames {
    FileNames {
        router: blk(l, "ffn_gate_inp.weight"),
        router_gate: blk(l, "ffn_gate_inp_shexp.weight"),
        gate: exps(l, "gate"),
        up: exps(l, "up"),
        down: exps(l, "down"),
        sh_gate: shexp(l, "gate"),
        sh_up: shexp(l, "up"),
        sh_down: shexp(l, "down"),
    }
}

/// The types a card projection site launches at: a projection of the normed
/// rows or an output projection, a K-quant or Q8_0.
const PROJ: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
];
/// The types a card dense FFN's gate, up and down launch at: [`PROJ`] and
/// NVFP4, whose planes the gemv arm reads over the q8_1 rows and the wide
/// GEMM over the q8 blocks of 32. A mixer's projection reads no NVFP4.
const DENSE_FFN: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::Nvfp4,
];
/// [`PROJ`] and F32: the types the card's β and α launch at, and those a
/// routed layer's stacks and its shared expert are read at, which the card
/// kernel table's entries then admit ([`plan_ffn`]); under a placed read the
/// stacks of a layer the dispatch's launches do not run are typed as the host
/// tier's instead ([`card_leg`]): a layer the common leg runs holds its
/// stacks to the placed side, which reads their types from the resident
/// stacks.
const PROJ_F32: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q8_0,
    SiteTy::F32,
];

/// The site types a backend's launches read, this reader's one question of a
/// backend: stated as data by each backend's owner — the card's
/// [`Sites35::CARD`] here, Metal's beside its session — never a branch in
/// this reader. A type no list of the backend's holds is refused by name
/// with the site (`file_site`), the dispatch lists' refusal naming the list.
pub struct Sites35 {
    /// A projection of the normed rows or an output projection.
    pub proj: &'static [SiteTy],
    /// A dense FFN's gate, up and down.
    pub dense: &'static [SiteTy],
    /// The routed stacks, the shared expert's parts, β and α.
    pub stacks: &'static [SiteTy],
    /// The gate·up launch's types: one launch reads the gate and the up of
    /// one type, whatever the lists (a gate beside another type's up is
    /// refused whatever the backend).
    pub gate_up: &'static [SiteTy],
    /// The down launch's types.
    pub down: &'static [SiteTy],
    /// A shared expert kept apart: the types its own parts run at.
    pub shared: &'static [SiteTy],
}

impl Sites35 {
    /// The card's lists: the projections', the dense FFN's and the stacks' of
    /// today, and the FFN dispatch's three from the kernel tables
    /// (`model::arch::coverage`'s `qwen35_gate_up`, `qwen35_down` and
    /// `qwen35_shared`), each pinned to its table by test, the dense FFN's to
    /// the coverage pin (`Q35_DENSE_FFN`). A backend with launches of its own
    /// states its lists beside them (Metal's beside its session).
    pub const CARD: Sites35 = Sites35 {
        proj: PROJ,
        dense: DENSE_FFN,
        stacks: PROJ_F32,
        gate_up: &[SiteTy::Q4K, SiteTy::Q5K],
        down: &[SiteTy::Q4K, SiteTy::Q5K, SiteTy::Q6K],
        shared: &[
            SiteTy::Q3K,
            SiteTy::Q4K,
            SiteTy::Q5K,
            SiteTy::Q6K,
            SiteTy::Q8_0,
            SiteTy::F32,
            SiteTy::Mxfp8,
            SiteTy::Fp8B128,
        ],
    };
}

/// `ty` among `list`, else the named refusal that lists the types the
/// backend's launch reads.
fn launches(ty: SiteTy, list: &[SiteTy], kind: &str, what: &'static str) -> Result<(), ChainError> {
    if list.contains(&ty) {
        return Ok(());
    }
    Err(ChainError::shape(
        what,
        format!(
            "a {ty} {kind}; this backend launches {}",
            crate::site::list(list)
        ),
    ))
}

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

/// Layer `l`'s pre-FFN norm: the one of the two spellings the file holds, `ffn_norm.weight`
/// or `post_attention_norm.weight`. A file with both or neither is refused by name. A layer
/// with the sandwich residual ([`Residual::Sandwich`]) holds both by design: its
/// `post_attention_norm.weight` is the mixer's output norm, so its pre-FFN norm is
/// `ffn_norm.weight` alone.
fn ffn_norm_name(
    file: &Split,
    what: &'static str,
    l: usize,
    residual: Residual,
) -> Result<String, ChainError> {
    let [plain, post] = ["ffn_norm.weight", "post_attention_norm.weight"].map(|s| blk(l, s));
    if matches!(residual, Residual::Sandwich { .. }) {
        return if file.find(&plain).is_some() {
            Ok(plain)
        } else {
            Err(ChainError::tensor(what, plain, "in the file"))
        };
    }
    match (file.find(&plain).is_some(), file.find(&post).is_some()) {
        (true, false) => Ok(plain),
        (false, true) => Ok(post),
        (true, true) => Err(ChainError::shape(
            what,
            format!("layer {l}: the file holds both {plain} and {post}; the pre-FFN norm is one"),
        )),
        (false, false) => Err(ChainError::shape(
            what,
            format!("layer {l}: the file holds neither {plain} nor {post}"),
        )),
    }
}

/// Layer `l`'s FFN by its description `spec`, every site's type from
/// `file`'s header at a type of the backend's `sites`, its pre-FFN norm the
/// file's own spelling ([`ffn_norm_name`]). A routed one: its
/// stacks' types admitted by the backend's dispatch lists (`gate_up`, `down`
/// — one gate·up launch reading the gate and the up of one type) and, where
/// the description has a shared expert, its own (`shared`), each refused by
/// name with the layer. A shared expert comes with the router joined with
/// its gate as row `n` (the routed count, which a backend's load joins) and is
/// folded into the stacks as expert `n` when its gate, up and down are each
/// its stack's type ([`SharedPlan::Joined`]), else kept apart at its own
/// types ([`SharedPlan::Apart`]); with none, the router is the file's own
/// weight and the stacks the file's own. Under a placed read
/// ([`Read35::Placed`]) a layer whose stacks the dispatch's launches do not run ([`card_leg`]) is typed as
/// the host tier's: its stacks are read as `HOSTED` and no dispatch list is
/// asked of them, its shared expert kept apart at its own types, which the
/// card runs whatever the stacks are; the common leg's layers
/// ([`LegSplit::Common`]) are among them, their types the placed side's to
/// read. A dense one: its three matrices a stack of one expert on the
/// arena's fixed route, each at a type of the backend's `dense` list.
pub fn plan_ffn(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    spec: &LayerSpec,
    l: usize,
    sites: &Sites35,
    read: Read35,
) -> Result<FfnPlan, ChainError> {
    let (h, ff) = (d.hidden, d.ff);
    let ffn_norm = ffn_norm_name(file, what, l, spec.residual)?;
    match &spec.ffn {
        Ffn::Moe(m) => {
            let n = m.experts as usize;
            let hosted = read == Read35::Placed
                && !matches!(
                    card_leg(file, what, l, (ff, h, n))?,
                    Some((LegSplit::Dispatch, _))
                );
            let stack = |part: &str, (rows, k): (usize, usize)| {
                if hosted {
                    Ok(HOSTED)
                } else {
                    ty_of(file, what, &exps(l, part), n * rows, k, sites.stacks)
                }
            };
            let (gate_ty, up_ty, down_ty) = (
                stack("gate", (ff, h))?,
                stack("up", (ff, h))?,
                stack("down", (h, ff))?,
            );
            let part = |part: &str, (rows, k): (usize, usize)| {
                ty_of(file, what, &shexp(l, part), rows, k, sites.stacks)
            };
            let shared_tys = match m.shared {
                Some(_) => Some([
                    part("gate", (ff, h))?,
                    part("up", (ff, h))?,
                    part("down", (h, ff))?,
                ]),
                None => None,
            };
            let at = |e: ChainError| ChainError::shape(what, format!("layer {l}: {e}"));
            if !hosted {
                if gate_ty != up_ty {
                    return Err(at(ChainError::shape(
                        what,
                        format!(
                            "a {gate_ty} gate beside a {up_ty} up; one gate·up launch reads one \
                             type"
                        ),
                    )));
                }
                launches(gate_ty, sites.gate_up, "gate·up", what).map_err(at)?;
                launches(down_ty, sites.down, "down `_sel`", what).map_err(at)?;
            }
            for ty in shared_tys.iter().flatten() {
                launches(*ty, sites.shared, "shared expert part", what).map_err(at)?;
            }
            let routed = |part: &str| exps(l, part);
            let (shared, [gate, up, down]) = match shared_tys {
                Some(tys) if !hosted && tys == [gate_ty, up_ty, down_ty] => (
                    Some(SharedPlan::Joined),
                    ["gate", "up", "down"].map(|part| joint(l, part)),
                ),
                Some([sh_gate, sh_up, sh_down]) => {
                    let sites = SharedSites {
                        gate: shexp(l, "gate"),
                        up: shexp(l, "up"),
                        down: shexp(l, "down"),
                        gate_ty: sh_gate,
                        up_ty: sh_up,
                        down_ty: sh_down,
                    };
                    (
                        Some(SharedPlan::Apart(sites)),
                        ["gate", "up", "down"].map(routed),
                    )
                }
                None => (None, ["gate", "up", "down"].map(routed)),
            };
            let gate_inp = match shared {
                Some(_) => joint_gate_inp(l),
                None => blk(l, "ffn_gate_inp.weight"),
            };
            Ok(FfnPlan {
                ffn_norm,
                route: FfnRoute::Router { gate_inp, shared },
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
                gate_ty: ty_of(file, what, &gate, ff, h, sites.dense)?,
                up_ty: ty_of(file, what, &up, ff, h, sites.dense)?,
                down_ty: ty_of(file, what, &down, h, ff, sites.dense)?,
                gate,
                up,
                down,
            })
        }
    }
}

/// The model-level table of rope frequency factors, one F32 per pair of the rope's values.
const ROPE_FREQS: &str = "rope_freqs.weight";

/// Layer `l`'s attention of kind `kind` over the flash `flash` its group
/// selects, every site's type from `file`'s header at a type of the backend's
/// `sites`. The q and k norm names are planned for every kind; only a kind
/// with a head norm ([`GqaKind::head_norm`]) reads them, and a backend's
/// residency check asks the file for them only then. The query projection is
/// read at the rows `d` names, which hold the kind's gate rows. A kind that
/// keeps the layer's window and rope factors ([`GqaKind::head_gate`]) is
/// refused by name: its plan reads them from the description ([`plan_gqa_of`]).
/// A kind that keeps only rope factors ([`GqaKind::rope_factors`]) plans here as a
/// layer without them: the caller is the card's, whose [`kind35`] refuses a layer
/// that has them.
///
/// [`kind35`]: super::plan::kind35
pub fn plan_gqa(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    l: usize,
    (kind, flash): (GqaKind, Flash),
    sites: &Sites35,
) -> Result<GqaPlan, ChainError> {
    if kind.head_gate() {
        return Err(ChainError::shape(
            what,
            format!(
                "layer {l}: attention {kind:?} keeps the layer's window and rope factors in its \
                 description; plan it with plan_gqa_of"
            ),
        ));
    }
    gqa_sites(file, what, d, l, (kind, flash), sites)
}

/// [`plan_gqa`] for the layer's own description `g`: besides the sites every kind has, a kind
/// with an output gate of its own ([`GqaKind::head_gate`]) reads the gate's projection (one row
/// a query head) and the window `g` attends, and a kind that keeps rope factors
/// ([`GqaKind::rope_factors`]) reads, where `g`'s rope has them, the file's table (F32, one
/// value a pair of the rope's values). A kind that keeps none refuses a description with a
/// window or factors by name.
pub fn plan_gqa_of(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    l: usize,
    (kind, flash): (GqaKind, Flash),
    g: &Gqa,
    sites: &Sites35,
) -> Result<GqaPlan, ChainError> {
    let refuse = |detail: String| ChainError::shape(what, format!("layer {l}: {detail}"));
    let mut plan = gqa_sites(file, what, d, l, (kind, flash), sites)?;
    if (g.window.is_some() && !kind.head_gate()) || (g.rope.factors && !kind.rope_factors()) {
        return Err(refuse(format!(
            "attention {kind:?} takes no window and no rope factors; the description has \
             window {:?} and factors {}",
            g.window, g.rope.factors
        )));
    }
    if kind.head_gate() {
        if g.window == Some(0) {
            return Err(refuse("a window of no position".to_string()));
        }
        let name = blk(l, "attn_gate.weight");
        plan.gate = Some(HeadGate {
            ty: ty_of(file, what, &name, g.heads as usize, d.hidden, sites.stacks)?,
            name,
        });
        plan.window = g.window;
    }
    if g.rope.factors {
        let pairs = g.rope.dims as usize / 2;
        let table = shaped(file, what, ROPE_FREQS, (1, pairs))?;
        if table.ty != GgmlType::F32 {
            return Err(refuse(format!(
                "{ROPE_FREQS} is {}; the rope's factors are f32",
                table.ty
            )));
        }
        plan.rope_freqs = Some(ROPE_FREQS.to_string());
    }
    Ok(plan)
}

/// The sites of layer `l`'s attention of every kind, with no window, no gate of its own and no
/// rope factors: what [`plan_gqa`] and [`plan_gqa_of`] start from.
fn gqa_sites(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    l: usize,
    (kind, flash): (GqaKind, Flash),
    sites: &Sites35,
) -> Result<GqaPlan, ChainError> {
    let (h, kv, att) = (d.hidden, d.kv_len, d.attn_len);
    let (q, k, v, o) = (
        blk(l, "attn_q.weight"),
        blk(l, "attn_k.weight"),
        blk(l, "attn_v.weight"),
        blk(l, "attn_output.weight"),
    );
    Ok(GqaPlan {
        kind,
        flash,
        attn_norm: blk(l, "attn_norm.weight"),
        attn_q_norm: blk(l, "attn_q_norm.weight"),
        attn_k_norm: blk(l, "attn_k_norm.weight"),
        q_ty: ty_of(file, what, &q, d.q_rows, h, sites.proj)?,
        k_ty: ty_of(file, what, &k, kv, h, sites.proj)?,
        v_ty: ty_of(file, what, &v, kv, h, sites.proj)?,
        o_ty: ty_of(file, what, &o, h, att, sites.proj)?,
        attn_q: q,
        attn_k: k,
        attn_v: v,
        attn_output: o,
        window: None,
        gate: None,
        rope_freqs: None,
    })
}

/// Layer `l`'s gated short conv of `ring`, every projection's type from `file`'s header at a
/// type of the backend's `sites`; the filters are F32, `hidden` channels of the ring's taps.
pub fn plan_shortconv(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    ring: Ring,
    l: usize,
    sites: &Sites35,
) -> Result<ShortConvPlan, ChainError> {
    let h = d.hidden;
    let (conv_in, conv, conv_out) = (
        blk(l, "shortconv.in_proj.weight"),
        blk(l, "shortconv.conv.weight"),
        blk(l, "shortconv.out_proj.weight"),
    );
    let filters = shaped(file, what, &conv, (h, ring.taps()))?;
    if filters.ty != GgmlType::F32 {
        return Err(ChainError::shape(
            what,
            format!("layer {l}: {conv} is {}; the filters are f32", filters.ty),
        ));
    }
    Ok(ShortConvPlan {
        attn_norm: blk(l, "attn_norm.weight"),
        in_ty: ty_of(file, what, &conv_in, 3 * h, h, sites.proj)?,
        out_ty: ty_of(file, what, &conv_out, h, h, sites.proj)?,
        hidden: h,
        ring,
        conv_in,
        conv,
        conv_out,
    })
}

/// Layer `l`'s residual form from its description: [`Residual::Plain`] needs nothing of the
/// file; [`Residual::Sandwich`] reads the two output norms, F32 of `hidden` values, and holds
/// its epsilon to a finite non-negative value and its scales to finite ones. A form no walk
/// here runs is refused by name.
pub fn plan_residual(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    spec: &LayerSpec,
    l: usize,
) -> Result<ResidualPlan, ChainError> {
    match spec.residual {
        Residual::Plain => Ok(ResidualPlan::Plain),
        Residual::Sandwich { eps, add } => {
            if !(eps.is_finite() && eps >= 0.0 && add.iter().all(|a| a.is_finite())) {
                return Err(ChainError::shape(
                    what,
                    format!(
                        "layer {l}: the sandwich residual has epsilon {eps} and scales {add:?}; \
                         the epsilon is finite and non-negative, the scales finite"
                    ),
                ));
            }
            let norms = ["post_attention_norm.weight", "post_ffw_norm.weight"].map(|s| blk(l, s));
            for name in &norms {
                let gain = shaped(file, what, name, (1, d.hidden))?;
                if gain.ty != GgmlType::F32 {
                    return Err(ChainError::shape(
                        what,
                        format!("layer {l}: {name} is {}; a norm's gain is f32", gain.ty),
                    ));
                }
            }
            Ok(ResidualPlan::Sandwich(SandwichPlan { eps, add, norms }))
        }
        other @ (Residual::Hc | Residual::AlphaBeta { .. }) => Err(ChainError::shape(
            what,
            format!("layer {l}: the residual {other:?}, which no walk here runs"),
        )),
    }
}

/// Layer `l`'s delta rule, every site's type from `file`'s header at a type
/// of the backend's `sites`.
pub fn plan_delta(
    file: &Split,
    what: &'static str,
    d: &PlanDims,
    shape: LinearShape,
    l: usize,
    sites: &Sites35,
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
        qkv_ty: ty_of(file, what, &qkv, c, h, sites.proj)?,
        gate_ty: ty_of(file, what, &gate, nv * linear::HEAD, h, sites.proj)?,
        beta_ty: ty_of(file, what, &beta, nv, h, sites.stacks)?,
        alpha_ty: ty_of(file, what, &alpha, nv, h, sites.stacks)?,
        out_ty: ty_of(file, what, &out, h, nv * linear::HEAD, sites.proj)?,
        qkv,
        gate,
        beta,
        alpha,
        ssm_out: out,
    })
}

/// Every layer's plan of a qwen3moe-family chain, `kinds` its layers' kinds
/// ([`super::plan::kinds_of`] or the card's [`super::plan::kinds35`], one a
/// layer; an attention layer's kind carries the class its geometry selects), each
/// layer's mixer, FFN and residual form read from `file`'s header at the rows `pd` names and
/// at a type of the backend's `sites`, the routed stacks held as `read` says
/// ([`Read35`]): the loop a whole load runs, both backends' — a backend that
/// transforms the plans after the read (a placed load's `apart_plans`)
/// transforms the `Vec` this returns.
pub fn layer_plans(
    file: &Split,
    what: &'static str,
    pd: &PlanDims,
    kinds: &[Kind35],
    layers: &[LayerSpec],
    sites: &Sites35,
    read: Read35,
) -> Result<Vec<LayerPlan>, ChainError> {
    if kinds.len() != layers.len() {
        return Err(ChainError::shape(
            what,
            format!(
                "{} mixer kinds for {} layers; the plan reads one kind a layer",
                kinds.len(),
                layers.len()
            ),
        ));
    }
    kinds
        .iter()
        .zip(layers)
        .enumerate()
        .map(|(l, (kind, layer))| {
            let mixer = match *kind {
                Kind35::Gqa(kind, flash) => {
                    let (g, class) = layer_gqa(layer)
                        .map_err(|e| ChainError::shape(what, format!("layer {l}: {e}")))?;
                    if class != kind {
                        return Err(ChainError::shape(
                            what,
                            format!(
                                "layer {l}: the kind reads attention {kind:?}; the description's \
                                 geometry selects {class:?}"
                            ),
                        ));
                    }
                    MixerPlan::Gqa(plan_gqa_of(file, what, pd, l, (kind, flash), g, sites)?)
                }
                Kind35::Delta(shape) => {
                    MixerPlan::Delta(plan_delta(file, what, pd, shape, l, sites)?)
                }
                Kind35::ShortConv(ring) => {
                    match &layer.mixer {
                        Mixer::ShortConv(c) if c.taps as usize == ring.taps() => {}
                        other => {
                            return Err(ChainError::shape(
                                what,
                                format!(
                                    "layer {l}: the kind reads a short conv of {} taps; the \
                                     description's mixer is {}",
                                    ring.taps(),
                                    other.what()
                                ),
                            ));
                        }
                    }
                    MixerPlan::ShortConv(plan_shortconv(file, what, pd, ring, l, sites)?)
                }
            };
            Ok(LayerPlan {
                mixer,
                ffn: plan_ffn(file, what, pd, layer, l, sites, read)?,
                residual: plan_residual(file, what, pd, layer, l)?,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        FfnRoute, Flash, GqaKind, LayerPlan, MixerPlan, PlanDims, Read35, ResidualPlan, SharedPlan,
        SiteTy, Sites35, blk, layer_plans, plan_ffn, plan_gqa, plan_gqa_of,
    };
    use crate::ChainError;
    use crate::arch::qwen3moe::plan::fixtures::{
        delta, dense_layer, gqa, head128_gqa, layer, lfm3_conv, lfm3_gqa, lfm3_layer, q30_gqa,
        q30_layer, spec,
    };
    use crate::arch::qwen3moe::plan::{Kind35, kinds_of, kinds35};
    use crate::linear::Ring;
    use gguf::Split;
    use gguf::quant::GgmlType::{self, F16, F32, NVFP4, Q4_K, Q6_K, Q8_0};
    use gguf::write::{Layout, TensorDecl};
    use model::arch::coverage::{Q35_DENSE_FFN, qwen35_down, qwen35_gate_up, qwen35_shared};
    use model::arch::models::{Act, Ffn, LayerSpec, Mixer, ModelSpec, OutGate, Residual};
    use std::path::PathBuf;

    /// `name`, `dims` (`dims[0]` the contiguous axis) at `ty`, as a file header declares it.
    fn decl(name: String, dims: &[u64], ty: GgmlType) -> TensorDecl {
        let id = ty.as_u32();
        let (_, blck, size) = gguf::ggml_type_info(id).expect("a type the table sizes");
        TensorDecl {
            name,
            dims: dims.to_vec(),
            type_id: id,
            nbytes: gguf::tensor_bytes(blck, size, dims).expect("a size that fits"),
        }
    }

    /// A file's header of `decls` and no weight bytes: all a plan reads.
    fn header(decls: Vec<TensorDecl>) -> Split {
        let layout = Layout::new(&[], decls).unwrap_or_else(|e| panic!("{e}"));
        Split::header_only(PathBuf::from("synthetic.gguf"), &layout)
            .unwrap_or_else(|e| panic!("{e}"))
    }

    /// Layer `l`'s routed FFN as the file holds it: the pre-FFN norm under `norm`, the stacks of
    /// `n` experts of `ff` values over `h`, the down at `down`, and the shared expert's three
    /// parts at `shared` when given.
    fn ffn_header(
        l: usize,
        norm: &str,
        (h, ff, n): (u64, u64, u64),
        down: GgmlType,
        shared: Option<GgmlType>,
    ) -> Vec<TensorDecl> {
        let mut t = vec![
            decl(blk(l, norm), &[h], F32),
            decl(blk(l, "ffn_gate_exps.weight"), &[h, ff, n], Q4_K),
            decl(blk(l, "ffn_up_exps.weight"), &[h, ff, n], Q4_K),
            decl(blk(l, "ffn_down_exps.weight"), &[ff, h, n], down),
        ];
        if let Some(ty) = shared {
            t.extend([
                decl(blk(l, "ffn_gate_shexp.weight"), &[h, ff], ty),
                decl(blk(l, "ffn_up_shexp.weight"), &[h, ff], ty),
                decl(blk(l, "ffn_down_shexp.weight"), &[ff, h], ty),
            ]);
        }
        t
    }

    /// Layer `l`'s attention sites over `h` values: `q` rows of query (gate rows included), `kv`
    /// rows of key and of value (at `v`), and `att` values of attention output.
    fn gqa_header(l: usize, h: u64, (q, kv, att): (u64, u64, u64), v: GgmlType) -> Vec<TensorDecl> {
        vec![
            decl(blk(l, "attn_q.weight"), &[h, q], Q4_K),
            decl(blk(l, "attn_k.weight"), &[h, kv], Q4_K),
            decl(blk(l, "attn_v.weight"), &[h, kv], v),
            decl(blk(l, "attn_output.weight"), &[att, h], Q4_K),
        ]
    }

    fn site(name: &str, ty: SiteTy) -> String {
        format!("{name} {ty}")
    }

    /// Everything a layer's plan holds, one entry a name or a name with its type, in field order.
    fn describe(p: &LayerPlan) -> Vec<String> {
        let mut out = Vec::new();
        match &p.mixer {
            MixerPlan::Gqa(g) => {
                out.extend([
                    format!("gqa {:?} {:?}", g.kind, g.flash),
                    g.attn_norm.clone(),
                    g.attn_q_norm.clone(),
                    g.attn_k_norm.clone(),
                    site(&g.attn_q, g.q_ty),
                    site(&g.attn_k, g.k_ty),
                    site(&g.attn_v, g.v_ty),
                    site(&g.attn_output, g.o_ty),
                ]);
                if let Some(w) = g.window {
                    out.push(format!("window {w}"));
                }
                if let Some(gate) = &g.gate {
                    out.push(site(&gate.name, gate.ty));
                }
                if let Some(table) = &g.rope_freqs {
                    out.push(format!("factors {table}"));
                }
            }
            MixerPlan::ShortConv(c) => out.extend([
                format!(
                    "shortconv {} taps of {}, ring {} rows",
                    c.ring.taps(),
                    c.hidden,
                    c.ring.rows()
                ),
                c.attn_norm.clone(),
                c.conv.clone(),
                site(&c.conv_in, c.in_ty),
                site(&c.conv_out, c.out_ty),
            ]),
            MixerPlan::Delta(d) => out.extend([
                format!("delta {}/{}", d.shape.n_k, d.shape.n_v),
                d.attn_norm.clone(),
                d.conv.clone(),
                d.dt_bias.clone(),
                d.ssm_a.clone(),
                d.ssm_norm.clone(),
                site(&d.qkv, d.qkv_ty),
                site(&d.gate, d.gate_ty),
                site(&d.beta, d.beta_ty),
                site(&d.alpha, d.alpha_ty),
                site(&d.ssm_out, d.out_ty),
            ]),
        }
        out.push(p.ffn.ffn_norm.clone());
        match &p.ffn.route {
            FfnRoute::Router { gate_inp, shared } => {
                out.push(format!("router {gate_inp}"));
                out.push(match shared {
                    None => "no shared".to_string(),
                    Some(SharedPlan::Joined) => "joined".to_string(),
                    Some(SharedPlan::Apart(s)) => format!(
                        "apart {} / {} / {}",
                        site(&s.gate, s.gate_ty),
                        site(&s.up, s.up_ty),
                        site(&s.down, s.down_ty)
                    ),
                });
            }
            FfnRoute::Dense => out.push("dense".to_string()),
        }
        out.extend([
            site(&p.ffn.gate, p.ffn.gate_ty),
            site(&p.ffn.up, p.ffn.up_ty),
            site(&p.ffn.down, p.ffn.down_ty),
        ]);
        if let ResidualPlan::Sandwich(s) = &p.residual {
            out.push(format!(
                "sandwich eps {} add {:?} norms {} {}",
                s.eps, s.add, s.norms[0], s.norms[1]
            ));
        }
        out
    }

    /// The plans of `spec` over `file` at the card's lists, by the card's entry points
    /// (`kinds35`, `PlanDims::q35`) or by the chain's (`kinds_of`, `PlanDims::of`).
    fn plans(file: &Split, spec: &ModelSpec, card: bool) -> Vec<Vec<String>> {
        let (kinds, pd) = if card {
            (kinds35(&spec.layers), PlanDims::q35(spec, "t"))
        } else {
            (kinds_of(&spec.layers), PlanDims::of(spec, "t"))
        };
        let (kinds, pd) = (
            kinds.unwrap_or_else(|e| panic!("{e}")),
            pd.unwrap_or_else(|e| panic!("{e}")),
        );
        layer_plans(
            file,
            "t",
            &pd,
            &kinds,
            &spec.layers,
            &Sites35::CARD,
            Read35::Whole,
        )
        .unwrap_or_else(|e| panic!("{e}"))
        .iter()
        .map(describe)
        .collect()
    }

    /// Qwen3.6's description reads to the plans it always read, through the card's entry points and
    /// through the chain's alike: a delta layer and two attention layers (one with a Q6_K v), the
    /// shared expert folded into the stacks where its parts are their types and kept apart where
    /// they are not, the router joined with its gate, and the pre-FFN norm under the file's
    /// `post_attention_norm.weight`.
    #[test]
    fn qwen36_reads_to_the_plans_it_read_before() {
        let (h, ff, n) = (2048, 512, 256);
        let mut t = vec![
            decl(blk(0, "attn_qkv.weight"), &[h, 8192], Q4_K),
            decl(blk(0, "attn_gate.weight"), &[h, 4096], Q4_K),
            decl(blk(0, "ssm_beta.weight"), &[h, 32], F32),
            decl(blk(0, "ssm_alpha.weight"), &[h, 32], F32),
            decl(blk(0, "ssm_out.weight"), &[4096, h], Q4_K),
        ];
        t.extend(gqa_header(1, h, (8192, 512, 4096), Q6_K));
        t.extend(gqa_header(2, h, (8192, 512, 4096), Q4_K));
        let norm = "post_attention_norm.weight";
        t.extend(ffn_header(0, norm, (h, ff, n), Q4_K, Some(Q4_K)));
        t.extend(ffn_header(1, norm, (h, ff, n), Q4_K, Some(Q4_K)));
        t.extend(ffn_header(2, norm, (h, ff, n), Q6_K, Some(Q8_0)));
        let file = header(t);
        let spec = spec(vec![layer(delta()), layer(gqa()), layer(gqa())], h as u32);
        let want = [
            vec![
                "delta 16/32",
                "blk.0.attn_norm.weight",
                "blk.0.ssm_conv1d.weight",
                "blk.0.ssm_dt.bias",
                "blk.0.ssm_a",
                "blk.0.ssm_norm.weight",
                "blk.0.attn_qkv.weight Q4_K",
                "blk.0.attn_gate.weight Q4_K",
                "blk.0.ssm_beta.weight F32",
                "blk.0.ssm_alpha.weight F32",
                "blk.0.ssm_out.weight Q4_K",
                "blk.0.post_attention_norm.weight",
                "router derived.blk.0.ffn_gate_inp_sh",
                "joined",
                "derived.blk.0.ffn_gate_exps_sh Q4_K",
                "derived.blk.0.ffn_up_exps_sh Q4_K",
                "derived.blk.0.ffn_down_exps_sh Q4_K",
            ],
            vec![
                "gqa Gated256 Group",
                "blk.1.attn_norm.weight",
                "blk.1.attn_q_norm.weight",
                "blk.1.attn_k_norm.weight",
                "blk.1.attn_q.weight Q4_K",
                "blk.1.attn_k.weight Q4_K",
                "blk.1.attn_v.weight Q6_K",
                "blk.1.attn_output.weight Q4_K",
                "blk.1.post_attention_norm.weight",
                "router derived.blk.1.ffn_gate_inp_sh",
                "joined",
                "derived.blk.1.ffn_gate_exps_sh Q4_K",
                "derived.blk.1.ffn_up_exps_sh Q4_K",
                "derived.blk.1.ffn_down_exps_sh Q4_K",
            ],
            vec![
                "gqa Gated256 Group",
                "blk.2.attn_norm.weight",
                "blk.2.attn_q_norm.weight",
                "blk.2.attn_k_norm.weight",
                "blk.2.attn_q.weight Q4_K",
                "blk.2.attn_k.weight Q4_K",
                "blk.2.attn_v.weight Q4_K",
                "blk.2.attn_output.weight Q4_K",
                "blk.2.post_attention_norm.weight",
                "router derived.blk.2.ffn_gate_inp_sh",
                "apart blk.2.ffn_gate_shexp.weight Q8_0 / blk.2.ffn_up_shexp.weight Q8_0 / \
                 blk.2.ffn_down_shexp.weight Q8_0",
                "blk.2.ffn_gate_exps.weight Q4_K",
                "blk.2.ffn_up_exps.weight Q4_K",
                "blk.2.ffn_down_exps.weight Q6_K",
            ],
        ];
        for card in [true, false] {
            assert_eq!(plans(&file, &spec, card), want, "card entry points: {card}");
        }
        let pd = PlanDims::q35(&spec, "t").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(PlanDims::of(&spec, "t").ok(), Some(pd));
        let card = plan_gqa(
            &file,
            "t",
            &pd,
            1,
            (GqaKind::Gated256, Flash::Group),
            &Sites35::CARD,
        );
        assert_eq!(card.map(|g| g.kind).ok(), Some(GqaKind::Gated256));
    }

    /// Qwen3-30B-A3B's description reads to whole-head NEOX attention over the group flash with
    /// 4096 query rows (no gate rows), 128 routed experts of 768 with no shared expert, the
    /// router and the stacks under the file's own names, and the pre-FFN norm under the file's
    /// `ffn_norm.weight`.
    #[test]
    fn qwen3_30b_reads_to_neox_attention_and_no_shared_expert() {
        let (h, ff, n) = (2048, 768, 128);
        let mut t = Vec::new();
        t.extend(gqa_header(0, h, (4096, 512, 4096), Q4_K));
        t.extend(gqa_header(1, h, (4096, 512, 4096), Q6_K));
        t.extend(ffn_header(0, "ffn_norm.weight", (h, ff, n), Q4_K, None));
        t.extend(ffn_header(1, "ffn_norm.weight", (h, ff, n), Q6_K, None));
        let file = header(t);
        let spec = spec(vec![q30_layer(), q30_layer()], h as u32);
        let pd = PlanDims::of(&spec, "t").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (pd.hidden, pd.ff, pd.q_rows, pd.kv_len, pd.attn_len),
            (2048, 768, 4096, 512, 4096)
        );
        let want = [
            vec![
                "gqa Neox128 Group",
                "blk.0.attn_norm.weight",
                "blk.0.attn_q_norm.weight",
                "blk.0.attn_k_norm.weight",
                "blk.0.attn_q.weight Q4_K",
                "blk.0.attn_k.weight Q4_K",
                "blk.0.attn_v.weight Q4_K",
                "blk.0.attn_output.weight Q4_K",
                "blk.0.ffn_norm.weight",
                "router blk.0.ffn_gate_inp.weight",
                "no shared",
                "blk.0.ffn_gate_exps.weight Q4_K",
                "blk.0.ffn_up_exps.weight Q4_K",
                "blk.0.ffn_down_exps.weight Q4_K",
            ],
            vec![
                "gqa Neox128 Group",
                "blk.1.attn_norm.weight",
                "blk.1.attn_q_norm.weight",
                "blk.1.attn_k_norm.weight",
                "blk.1.attn_q.weight Q4_K",
                "blk.1.attn_k.weight Q4_K",
                "blk.1.attn_v.weight Q6_K",
                "blk.1.attn_output.weight Q4_K",
                "blk.1.ffn_norm.weight",
                "router blk.1.ffn_gate_inp.weight",
                "no shared",
                "blk.1.ffn_gate_exps.weight Q4_K",
                "blk.1.ffn_up_exps.weight Q4_K",
                "blk.1.ffn_down_exps.weight Q6_K",
            ],
        ];
        assert_eq!(plans(&file, &spec, false), want);
    }

    /// A kind that is not the description's is refused by name with the layer, never read as an
    /// attention class: an attention kind over a delta-rule layer.
    #[test]
    fn a_kind_the_description_does_not_hold_is_refused_by_name() {
        let spec = spec(vec![layer(delta())], 2048);
        let pd = PlanDims {
            hidden: 2048,
            ff: 512,
            q_rows: 8192,
            kv_len: 512,
            attn_len: 4096,
        };
        let kinds = [Kind35::Gqa(GqaKind::Gated256, Flash::Group)];
        let file = header(Vec::new());
        match layer_plans(
            &file,
            "t",
            &pd,
            &kinds,
            &spec.layers,
            &Sites35::CARD,
            Read35::Whole,
        ) {
            Err(e) => assert!(e.to_string().contains("layer 0:"), "{e}"),
            Ok(p) => panic!("read as {:?}", p.iter().map(describe).collect::<Vec<_>>()),
        }
    }

    /// The pre-FFN norm is the file's: a file that holds both spellings or neither is refused by
    /// name with the layer, never read as one of them.
    #[test]
    fn the_pre_ffn_norm_is_the_files_one_spelling() {
        let (h, ff, n) = (2048, 768, 128);
        let spec = spec(vec![q30_layer()], h as u32);
        let pd = PlanDims::of(&spec, "t").unwrap_or_else(|e| panic!("{e}"));
        let both = vec![
            decl(blk(0, "ffn_norm.weight"), &[h], F32),
            decl(blk(0, "post_attention_norm.weight"), &[h], F32),
        ];
        let neither = ffn_header(0, "ffn_norm.weight", (h, ff, n), Q4_K, None)[1..].to_vec();
        for (what, tensors, text) in [
            ("both", both, "holds both"),
            ("neither", neither, "neither"),
        ] {
            let file = header(tensors);
            let got = plan_ffn(
                &file,
                "t",
                &pd,
                &spec.layers[0],
                0,
                &Sites35::CARD,
                Read35::Whole,
            );
            match got {
                Err(e) => assert!(
                    e.to_string().contains("layer 0:") && e.to_string().contains(text),
                    "{what}: {e}"
                ),
                Ok(p) => panic!("{what}: read as {}", p.ffn_norm),
            }
        }
    }

    /// The card's dispatch lists are the kernel tables' families, each list
    /// exactly the families whose file type its table names an entry for: a
    /// table row that moves a type in or out moves its list with it, and
    /// this names the change. The projections' lists the tables admit not
    /// (a projection runs at whatever the table's dense rows name), so
    /// `proj` and `stacks` carry no such pin.
    #[test]
    fn the_card_dispatch_lists_are_the_tables_families() {
        let families = |table: fn(GgmlType) -> bool| -> Vec<SiteTy> {
            SiteTy::ALL
                .iter()
                .copied()
                .filter(|&s| table(s.ggml()))
                .collect()
        };
        assert_eq!(
            families(|ty| qwen35_gate_up(ty).is_some()),
            Sites35::CARD.gate_up
        );
        assert_eq!(families(|ty| qwen35_down(ty).is_some()), Sites35::CARD.down);
        assert_eq!(
            families(|ty| qwen35_shared(ty).is_some()),
            Sites35::CARD.shared
        );
    }

    /// An LFM3 file's header at a size a test can write: hidden 256, a dense FFN of 512, 16/2
    /// heads of 128, a conv of seven taps. Layer 0 is a short conv, layer 1 full attention
    /// (the gate's projection Q4_K, the rope's factors) and layer 2 sliding (the gate F32); every
    /// layer has both spellings of the post-norms the sandwich reads beside its pre-FFN norm.
    fn lfm3_tensors() -> Vec<TensorDecl> {
        let (h, ff, att, kv) = (256u64, 512u64, 2048u64, 512u64);
        let mut t = vec![decl("rope_freqs.weight".to_string(), &[64], F32)];
        for l in 0..3 {
            t.extend([
                decl(blk(l, "attn_norm.weight"), &[h], F32),
                decl(blk(l, "ffn_norm.weight"), &[h], F32),
                decl(blk(l, "post_attention_norm.weight"), &[h], F32),
                decl(blk(l, "post_ffw_norm.weight"), &[h], F32),
                decl(blk(l, "ffn_gate.weight"), &[h, ff], Q4_K),
                decl(blk(l, "ffn_up.weight"), &[h, ff], Q4_K),
                decl(blk(l, "ffn_down.weight"), &[ff, h], Q4_K),
            ]);
        }
        t.extend([
            decl(blk(0, "shortconv.in_proj.weight"), &[h, 3 * h], Q4_K),
            decl(blk(0, "shortconv.conv.weight"), &[7, h], F32),
            decl(blk(0, "shortconv.out_proj.weight"), &[h, h], Q4_K),
        ]);
        for (l, gate) in [(1, Q4_K), (2, F32)] {
            t.extend([
                decl(blk(l, "attn_q.weight"), &[h, att], Q4_K),
                decl(blk(l, "attn_k.weight"), &[h, kv], Q4_K),
                decl(blk(l, "attn_v.weight"), &[h, kv], Q4_K),
                decl(blk(l, "attn_output.weight"), &[att, h], Q4_K),
                decl(blk(l, "attn_gate.weight"), &[h, 16], gate),
                decl(blk(l, "attn_q_norm.weight"), &[128], F32),
                decl(blk(l, "attn_k_norm.weight"), &[128], F32),
            ]);
        }
        t
    }

    /// The description of [`lfm3_tensors`]: a short conv of seven taps, a full attention layer
    /// whose rope has the file's factors, and a sliding one of 1024 positions.
    fn lfm3_spec() -> ModelSpec {
        spec(
            vec![
                lfm3_layer(lfm3_conv(7)),
                lfm3_layer(Mixer::Gqa(lfm3_gqa(None, true))),
                lfm3_layer(Mixer::Gqa(lfm3_gqa(Some(1024), false))),
            ],
            256,
        )
    }

    /// The plans of `spec` over a file of `tensors`, by the chain's entry points.
    fn lfm3_plans(
        tensors: Vec<TensorDecl>,
        spec: &ModelSpec,
    ) -> Result<Vec<LayerPlan>, ChainError> {
        let file = header(tensors);
        let kinds = kinds_of(&spec.layers)?;
        let pd = PlanDims::of(spec, "t")?;
        layer_plans(
            &file,
            "t",
            &pd,
            &kinds,
            &spec.layers,
            &Sites35::CARD,
            Read35::Whole,
        )
    }

    /// `lfm3_plans` is refused, and its error holds every text of `texts`.
    fn lfm3_refused(what: &str, tensors: Vec<TensorDecl>, spec: &ModelSpec, texts: &[&str]) {
        match lfm3_plans(tensors, spec) {
            Err(e) => {
                let e = e.to_string();
                for text in texts {
                    assert!(e.contains(text), "{what}: {e}");
                }
            }
            Ok(_) => panic!("{what}: the plans were read"),
        }
    }

    /// `tensors` with `name` replaced by `with`.
    fn swapped(mut tensors: Vec<TensorDecl>, name: &str, with: TensorDecl) -> Vec<TensorDecl> {
        let at = tensors
            .iter()
            .position(|t| t.name == name)
            .unwrap_or_else(|| panic!("{name}: not in the header"));
        tensors[at] = with;
        tensors
    }

    /// `tensors` without `name`.
    fn without(mut tensors: Vec<TensorDecl>, name: &str) -> Vec<TensorDecl> {
        tensors.retain(|t| t.name != name);
        tensors
    }

    /// LFM3's description reads to a short conv over its ring of fourteen slots, a full attention
    /// layer with the gate's projection and the rope's factors, a sliding one with the same gate
    /// at another type and a window, each over a dense FFN under its `ffn_norm.weight` (though
    /// the file has `post_attention_norm.weight` too, the mixer's sandwich norm) and joined by the
    /// sandwich of both post-norms.
    #[test]
    fn lfm3_reads_to_a_conv_two_gated_heads_and_the_sandwich() {
        let spec = lfm3_spec();
        let pd = PlanDims::of(&spec, "t").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(
            (pd.hidden, pd.ff, pd.q_rows, pd.kv_len, pd.attn_len),
            (256, 512, 2048, 512, 2048)
        );
        let own = |lines: &[&str]| -> Vec<String> {
            lines.iter().map(|line| (*line).to_string()).collect()
        };
        let with_tail = |mut head: Vec<String>, l: usize| -> Vec<String> {
            head.extend([
                format!("blk.{l}.ffn_norm.weight"),
                "dense".to_string(),
                format!("blk.{l}.ffn_gate.weight Q4_K"),
                format!("blk.{l}.ffn_up.weight Q4_K"),
                format!("blk.{l}.ffn_down.weight Q4_K"),
                format!(
                    "sandwich eps 0.00001 add [1.0, 1.0] norms blk.{l}.post_attention_norm.weight \
                     blk.{l}.post_ffw_norm.weight"
                ),
            ]);
            head
        };
        let want = vec![
            with_tail(
                own(&[
                    "shortconv 7 taps of 256, ring 14 rows",
                    "blk.0.attn_norm.weight",
                    "blk.0.shortconv.conv.weight",
                    "blk.0.shortconv.in_proj.weight Q4_K",
                    "blk.0.shortconv.out_proj.weight Q4_K",
                ]),
                0,
            ),
            with_tail(
                own(&[
                    "gqa HeadGate128 Group",
                    "blk.1.attn_norm.weight",
                    "blk.1.attn_q_norm.weight",
                    "blk.1.attn_k_norm.weight",
                    "blk.1.attn_q.weight Q4_K",
                    "blk.1.attn_k.weight Q4_K",
                    "blk.1.attn_v.weight Q4_K",
                    "blk.1.attn_output.weight Q4_K",
                    "blk.1.attn_gate.weight Q4_K",
                    "factors rope_freqs.weight",
                ]),
                1,
            ),
            with_tail(
                own(&[
                    "gqa HeadGate128 Group",
                    "blk.2.attn_norm.weight",
                    "blk.2.attn_q_norm.weight",
                    "blk.2.attn_k_norm.weight",
                    "blk.2.attn_q.weight Q4_K",
                    "blk.2.attn_k.weight Q4_K",
                    "blk.2.attn_v.weight Q4_K",
                    "blk.2.attn_output.weight Q4_K",
                    "window 1024",
                    "blk.2.attn_gate.weight F32",
                ]),
                2,
            ),
        ];
        assert_eq!(plans(&header(lfm3_tensors()), &spec, false), want);
    }

    /// What the LFM3 plan reads from the file it refuses by name, with the tensor or the layer:
    /// a missing post-norm, one that is not f32, short-conv filters that are not f32 or are of
    /// another tap count, a missing gate projection, a factor table of another length or type,
    /// a sandwich layer with no `ffn_norm.weight` (its `post_attention_norm.weight` is the
    /// mixer's norm, never the pre-FFN one) and a residual form no walk runs.
    #[test]
    fn what_the_lfm3_plan_reads_wrongly_from_the_file_is_refused_by_name() {
        let spec = lfm3_spec();
        let tensors = lfm3_tensors;
        lfm3_refused(
            "no FFN post-norm",
            without(tensors(), "blk.1.post_ffw_norm.weight"),
            &spec,
            &["blk.1.post_ffw_norm.weight", "is not in the file"],
        );
        lfm3_refused(
            "an f16 mixer post-norm",
            swapped(
                tensors(),
                "blk.1.post_attention_norm.weight",
                decl(blk(1, "post_attention_norm.weight"), &[256], F16),
            ),
            &spec,
            &["blk.1.post_attention_norm.weight", "a norm's gain is f32"],
        );
        lfm3_refused(
            "f16 filters",
            swapped(
                tensors(),
                "blk.0.shortconv.conv.weight",
                decl(blk(0, "shortconv.conv.weight"), &[7, 256], F16),
            ),
            &spec,
            &["blk.0.shortconv.conv.weight", "the filters are f32"],
        );
        lfm3_refused(
            "filters of six taps",
            swapped(
                tensors(),
                "blk.0.shortconv.conv.weight",
                decl(blk(0, "shortconv.conv.weight"), &[6, 256], F32),
            ),
            &spec,
            &["blk.0.shortconv.conv.weight", "it takes 256 rows of 7"],
        );
        lfm3_refused(
            "no gate projection",
            without(tensors(), "blk.2.attn_gate.weight"),
            &spec,
            &["blk.2.attn_gate.weight", "is not in the file"],
        );
        lfm3_refused(
            "factors of 32",
            swapped(
                tensors(),
                "rope_freqs.weight",
                decl("rope_freqs.weight".to_string(), &[32], F32),
            ),
            &spec,
            &["rope_freqs.weight", "of 64"],
        );
        lfm3_refused(
            "f16 factors",
            swapped(
                tensors(),
                "rope_freqs.weight",
                decl("rope_freqs.weight".to_string(), &[64], F16),
            ),
            &spec,
            &["layer 1:", "the rope's factors are f32"],
        );
        lfm3_refused(
            "no pre-FFN norm",
            without(tensors(), "blk.0.ffn_norm.weight"),
            &spec,
            &["blk.0.ffn_norm.weight", "is not in the file"],
        );
        let mut hc = lfm3_spec();
        hc.layers[1].residual = Residual::Hc;
        let mut plain = tensors();
        plain.retain(|t| !t.name.starts_with("blk.1.post_"));
        lfm3_refused("a Hc residual", plain, &hc, &["layer 1:", "Hc"]);
        let mut negative = lfm3_spec();
        negative.layers[2].residual = Residual::Sandwich {
            eps: -1.0,
            add: [1.0, 1.0],
        };
        lfm3_refused(
            "a negative epsilon",
            tensors(),
            &negative,
            &["layer 2:", "epsilon"],
        );
    }

    /// What a description and its kinds, or a gate's rows, do not say alike is refused by name: a
    /// short-conv kind of other taps than the description's or over an attention layer, a window of
    /// no position, an output gate of a value a channel, the plain `plan_gqa` asked for the
    /// head with a gate of its own (whose window and factors only the description holds), and
    /// `plan_gqa_of` asked for a window under a kind with none.
    #[test]
    fn an_lfm3_description_and_its_kinds_must_agree() {
        let spec = lfm3_spec();
        let pd = PlanDims::of(&spec, "t").unwrap_or_else(|e| panic!("{e}"));
        let file = header(lfm3_tensors());
        let head_gate = Kind35::Gqa(GqaKind::HeadGate128, Flash::Group);
        for (what, kinds, text) in [
            (
                "a conv of five taps over seven",
                [Kind35::ShortConv(Ring::new(5)), head_gate, head_gate],
                "layer 0: the kind reads a short conv of 5 taps",
            ),
            (
                "a conv over attention",
                [
                    Kind35::ShortConv(Ring::new(7)),
                    Kind35::ShortConv(Ring::new(7)),
                    head_gate,
                ],
                "layer 1: the kind reads a short conv of 7 taps; the description's mixer is a GQA mixer",
            ),
        ] {
            let got = layer_plans(
                &file,
                "t",
                &pd,
                &kinds,
                &spec.layers,
                &Sites35::CARD,
                Read35::Whole,
            );
            match got {
                Err(e) => assert!(e.to_string().contains(text), "{what}: {e}"),
                Ok(_) => panic!("{what}: the plans were read"),
            }
        }
        let mut empty = lfm3_spec();
        empty.layers[2].mixer = Mixer::Gqa(lfm3_gqa(Some(0), false));
        lfm3_refused(
            "a window of no position",
            lfm3_tensors(),
            &empty,
            &["layer 2:", "a window of no position"],
        );
        let mut per_channel = lfm3_spec();
        if let Mixer::Gqa(g) = &mut per_channel.layers[1].mixer {
            g.out_gate = OutGate::Proj;
        }
        match PlanDims::of(&per_channel, "t") {
            Err(e) => assert!(e.to_string().contains("a value a channel"), "{e}"),
            Ok(pd) => panic!("a gate a channel gave {pd:?}"),
        }
        let plain = plan_gqa(
            &file,
            "t",
            &pd,
            1,
            (GqaKind::HeadGate128, Flash::Group),
            &Sites35::CARD,
        );
        match plain {
            Err(e) => assert!(
                e.to_string().contains("layer 1:") && e.to_string().contains("plan_gqa_of"),
                "{e}"
            ),
            Ok(_) => panic!("plan_gqa read the head with a gate of its own"),
        }
        let windowed = lfm3_gqa(Some(8), false);
        let got = plan_gqa_of(
            &file,
            "t",
            &pd,
            1,
            (GqaKind::Neox128, Flash::Group),
            &windowed,
            &Sites35::CARD,
        );
        match got {
            Err(e) => assert!(
                e.to_string().contains("layer 1:") && e.to_string().contains("takes no window"),
                "{e}"
            ),
            Ok(_) => panic!("a window under a kind with none"),
        }
        let neox = plan_gqa_of(
            &file,
            "t",
            &pd,
            1,
            (GqaKind::Neox128, Flash::Group),
            &q30_gqa(),
            &Sites35::CARD,
        );
        assert!(neox.is_ok(), "a plain description under its plain kind");
    }

    /// The card's dense FFN list is the coverage pin's types: a type the pin reads and the list
    /// lacks is covered, then refused at the plan; the reverse is a type the plan reads that
    /// coverage lists as an item.
    #[test]
    fn the_card_dense_ffn_list_is_the_coverage_pin() {
        let pinned: Vec<SiteTy> = SiteTy::ALL
            .iter()
            .copied()
            .filter(|s| Q35_DENSE_FFN.contains(&s.ggml()))
            .collect();
        assert_eq!(pinned, Sites35::CARD.dense);
    }

    /// A dense FFN reads NVFP4 at its gate, up and down; a card projection of a mixer does not,
    /// so an NVFP4 attention site is refused by name.
    #[test]
    fn a_dense_ffn_reads_nvfp4_and_a_mixer_projection_does_not() {
        let (h, ff): (u64, u64) = (512, 1024);
        let dense = LayerSpec {
            ffn: Ffn::Dense {
                ff: ff as u32,
                act: Act::SwiGlu { limit: None },
            },
            ..layer(gqa())
        };
        let spec = spec(vec![dense], h as u32);
        let pd = PlanDims::of(&spec, "t").unwrap_or_else(|e| panic!("{e}"));
        let mut t = gqa_header(0, h, (8192, 512, 4096), Q4_K);
        t.extend([
            decl(blk(0, "ffn_norm.weight"), &[h], F32),
            decl(blk(0, "ffn_gate.weight"), &[h, ff], NVFP4),
            decl(blk(0, "ffn_up.weight"), &[h, ff], NVFP4),
            decl(blk(0, "ffn_down.weight"), &[ff, h], NVFP4),
        ]);
        let plan = plan_ffn(
            &header(t.clone()),
            "t",
            &pd,
            &spec.layers[0],
            0,
            &Sites35::CARD,
            Read35::Whole,
        )
        .unwrap_or_else(|e| panic!("{e}"));
        assert!(matches!(plan.route, FfnRoute::Dense));
        assert_eq!([plan.gate_ty, plan.up_ty, plan.down_ty], [SiteTy::Nvfp4; 3]);
        t[0] = decl(blk(0, "attn_q.weight"), &[h, 8192], NVFP4);
        let kind = (GqaKind::Gated256, Flash::Group);
        match plan_gqa(&header(t), "t", &pd, 0, kind, &Sites35::CARD) {
            Err(e) => assert!(e.to_string().contains("blk.0.attn_q.weight"), "{e}"),
            Ok(g) => panic!("an NVFP4 attention site read as {}", g.q_ty),
        }
    }
    /// A head-128 NORM layer's file at a size a test can write: hidden 256, 16/4 heads of 128 (a
    /// group of four), a dense FFN of 512, no q/k norm tensors, and the rope's factors as `table`
    /// when given.
    fn head128_tensors(table: Option<TensorDecl>) -> Vec<TensorDecl> {
        let (h, ff, att, kv) = (256u64, 512u64, 2048u64, 512u64);
        let mut t: Vec<TensorDecl> = table.into_iter().collect();
        t.extend(gqa_header(0, h, (att, kv, att), Q4_K));
        t.extend([
            decl(blk(0, "attn_norm.weight"), &[h], F32),
            decl(blk(0, "ffn_norm.weight"), &[h], F32),
            decl(blk(0, "ffn_gate.weight"), &[h, ff], Q4_K),
            decl(blk(0, "ffn_up.weight"), &[h, ff], Q4_K),
            decl(blk(0, "ffn_down.weight"), &[ff, h], Q4_K),
        ]);
        t
    }

    /// The head-128 NORM attention reads the file's rope factors where its description says its
    /// rope has them, and none where it does not: the plan names the table (F32, one value a
    /// pair), a table of another length or type is refused by name with the layer, a description
    /// whose rope has factors over a file with no table is refused by name, and a kind that keeps
    /// no factors refuses a description that has them. Mutant: `plan_gqa_of` reading the table
    /// only for the kind with a gate (the first assertion), or not checking the table's type (the
    /// f16 table).
    #[test]
    fn a_head_128_norm_layer_reads_the_files_rope_factors() {
        let table = || decl("rope_freqs.weight".to_string(), &[64], F32);
        let plans = |tensors: Vec<TensorDecl>, factors: bool| {
            let spec = spec(
                vec![dense_layer(Mixer::Gqa(head128_gqa(16, 4, factors)))],
                256,
            );
            let kinds = kinds_of(&spec.layers)?;
            let pd = PlanDims::of(&spec, "t")?;
            layer_plans(
                &header(tensors),
                "t",
                &pd,
                &kinds,
                &spec.layers,
                &Sites35::CARD,
                Read35::Whole,
            )
        };
        let got = plans(head128_tensors(Some(table())), true).unwrap_or_else(|e| panic!("{e}"));
        let lines = describe(&got[0]);
        assert_eq!(
            lines[0],
            "gqa Head128 { pairs: Norm, head_norm: false } Group"
        );
        assert_eq!(lines[8], "factors rope_freqs.weight");
        let MixerPlan::Gqa(g) = &got[0].mixer else {
            panic!("an attention layer")
        };
        assert!(g.window.is_none() && g.gate.is_none());
        // A table in the file the description does not use is not read.
        let got = plans(head128_tensors(Some(table())), false).unwrap_or_else(|e| panic!("{e}"));
        let MixerPlan::Gqa(g) = &got[0].mixer else {
            panic!("an attention layer")
        };
        assert_eq!(g.rope_freqs, None);
        for (what, tensors, texts) in [
            (
                "no table",
                head128_tensors(None),
                &["rope_freqs.weight", "is not in the file"][..],
            ),
            (
                "a table of 32",
                head128_tensors(Some(decl("rope_freqs.weight".to_string(), &[32], F32))),
                &["rope_freqs.weight", "of 64"][..],
            ),
            (
                "an f16 table",
                head128_tensors(Some(decl("rope_freqs.weight".to_string(), &[64], F16))),
                &["layer 0:", "the rope's factors are f32"][..],
            ),
        ] {
            match plans(tensors, true) {
                Err(e) => {
                    let e = e.to_string();
                    for text in texts {
                        assert!(e.contains(text), "{what}: {e}");
                    }
                }
                Ok(_) => panic!("{what}: the plans were read"),
            }
        }
        // A kind that keeps no factors refuses a description that has them.
        let q30 = {
            let mut g = q30_gqa();
            g.rope.factors = true;
            g
        };
        let layer = dense_layer(Mixer::Gqa(head128_gqa(16, 4, false)));
        let pd = PlanDims::of(&spec(vec![layer], 256), "t").unwrap_or_else(|e| panic!("{e}"));
        match plan_gqa_of(
            &header(head128_tensors(Some(table()))),
            "t",
            &pd,
            0,
            (GqaKind::Neox128, Flash::Group),
            &q30,
            &Sites35::CARD,
        ) {
            Err(e) => assert!(
                e.to_string()
                    .contains("takes no window and no rope factors"),
                "{e}"
            ),
            Ok(_) => panic!("a NEOX layer read rope factors"),
        }
    }
}
