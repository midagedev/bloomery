//! What a qwen35moe file asks of this backend, read from its header and
//! tensor table alone: the one owner of every refusal the open makes before it
//! allocates a buffer.
//!
//! [`check`] reads the file's description (the architecture verdict), the
//! plan of each layer's two halves at the types Metal's mat-vecs launch, and
//! the arena's own shape, type and geometry rules over the tensors the plan
//! names. It returns every refusal it meets, in the order the open meets them
//! ([`Item`]), and, when there is none, the [`Checked`] file: the plans and
//! the arena's [`Layout`], each weight resolved to the name, the shape and the
//! kernel type the loader copies. The open fails on the first item; `--plan`
//! prints them all. No weight byte is read and no buffer is made, so a file
//! whose data section is cut short is checked as a whole one.
//!
//! The order is the open's: the description, the layers' plans (a layer's
//! mixer, then its FFN), then the arena's rules (the cache, the head and the
//! embedding, then each layer whose two plans read). A half whose plan the
//! launch lists refuse names its first refusal as the chain reader words it,
//! then every other site of that half the same lists refuse, so one file's
//! sites of a type no mat-vec runs are all listed and the first item is the
//! one the open names.

use bloomery_device::fault::FAULT_WORDS;
use chain::ChainError;
use chain::arch::qwen3moe::plan::{
    DeltaPlan, FfnRoute, GqaPlan, Kind35, LayerPlan, MixerPlan, ResidualPlan, SharedPlan,
    ShortConvPlan, SiteTy, kinds_of,
};
use chain::arch::qwen3moe::read::{
    PlanDims, Read35, Sites35, file_names, plan_delta, plan_ffn, plan_gqa_of, plan_residual,
    plan_shortconv,
};
use chain::linear::{LinearShape, Ring};
use chain::rope::{RopeSpec, RopeTable};
use chain::site::file_site;
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::{Act, Ffn, LayerSpec, Mixer as SpecMixer, ModelSpec};
use model::arch::names;

use super::arena::LANES;
use super::rows::RowsLens;
use super::session::{PROJECTED, SITES, SessionError, WHAT, pass_rows_max};
use crate::geom::Geometry;
use crate::instance::{self, Instance};
use crate::kernels::{HeadMap, MAX_SEGMENTS, Quant, segments_for_keys, shared_entry};

/// The arena's name in a refusal.
const ARENA: &str = "qwen35moe arena";

/// A pass-through page: Apple silicon's, the size a Metal buffer's bytes round
/// up to in the driver's count.
pub const PAGE: u64 = 16 * 1024;

/// Every site type the chain reader knows, the widest lists a read can take:
/// the read of a half that the launch lists refuse still names its sites.
const ANY: &[SiteTy] = &[
    SiteTy::Q3K,
    SiteTy::Q4K,
    SiteTy::Iq4Xs,
    SiteTy::Q5K,
    SiteTy::Q6K,
    SiteTy::Q5_0,
    SiteTy::Q8_0,
    SiteTy::F32,
    SiteTy::Nvfp4,
    SiteTy::Mxfp8,
    SiteTy::Fp8B128,
    SiteTy::Bf16,
    SiteTy::MlxA4,
    SiteTy::MlxA8,
    SiteTy::MlxA4h,
    SiteTy::MlxA8h,
];

/// [`ANY`] as the reader's six lists.
const WIDEST: Sites35 = Sites35 {
    proj: ANY,
    dense: ANY,
    stacks: ANY,
    gate_up: ANY,
    down: ANY,
    shared: ANY,
};

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: ARENA,
        detail: detail.into(),
    }
}

/// One refusal, in the words the open returns it in.
#[derive(Debug)]
pub struct Item {
    /// The layer the refusal was met at; `None` for the file's own.
    pub layer: Option<usize>,
    pub error: SessionError,
}

/// Where a check's items collect, and the layer they are met at: an arena
/// refusal of a layer names it, a plan's or the file's does not.
struct At<'a> {
    items: &'a mut Vec<Item>,
    layer: Option<usize>,
}

impl At<'_> {
    /// Refuse `error`, naming the layer if the check is a layer's.
    fn push(&mut self, error: SessionError) {
        let error = match (error, self.layer) {
            (SessionError::Shape { what, detail }, Some(l)) => SessionError::Shape {
                what,
                detail: format!("layer {l}: {detail}"),
            },
            (other, _) => other,
        };
        self.items.push(Item {
            layer: self.layer,
            error,
        });
    }

    /// `r`'s value, or `None` with its refusal collected.
    fn ok<T>(&mut self, r: Result<T, SessionError>) -> Option<T> {
        r.map_err(|e| self.push(e)).ok()
    }
}

/// The header's view of the tensors: a name's type and shape, judged by the
/// arena's rules at the launch geometry `geom` of the file's instance.
struct Table<'a> {
    split: &'a Split,
    geom: &'a Geometry,
}

impl Table<'_> {
    /// The type and dims of tensor `name`.
    fn info(&self, name: &str) -> Result<(GgmlType, &[u64]), SessionError> {
        let (_, info) = self
            .split
            .find(name)
            .ok_or_else(|| refuse(format!("the file has no tensor {name}")))?;
        Ok((info.ty, &info.dims))
    }

    /// F32 tensor `name` of `len` values.
    fn f32s(&self, name: &str, len: usize) -> Result<Values, SessionError> {
        let (ty, dims) = self.info(name)?;
        let n: u64 = dims.iter().product();
        if ty != GgmlType::F32 || usize::try_from(n).ok() != Some(len) {
            return Err(refuse(format!(
                "{name} is {ty:?} {dims:?}; the step reads {len} F32 values"
            )));
        }
        Ok(Values::new(name, len, ty))
    }

    /// F32 tensor `name` of `len` values, or MLX's affine 8-bit tensor of the
    /// same values — BF16- or F16-scaled (`MLX_A8`, `MLX_A8H`) — which the
    /// open dequantizes exactly: the router and the shared-expert gate never
    /// run a kernel, so no 8-bit path exists on the card.
    fn f32s_or_a8(&self, name: &str, len: usize) -> Result<Values, SessionError> {
        let (ty, dims) = self.info(name)?;
        let n: u64 = dims.iter().product();
        if usize::try_from(n).ok() != Some(len) {
            return Err(refuse(format!(
                "{name} is {ty:?} {dims:?}; the step reads {len} F32 values"
            )));
        }
        match ty {
            GgmlType::F32 | GgmlType::MLX_A8 | GgmlType::MLX_A8H => Ok(Values::new(name, len, ty)),
            other => Err(refuse(format!(
                "{name} is {other:?}; the step reads F32, MLX_A8 or MLX_A8H"
            ))),
        }
    }

    /// Tensor `name`: `rows` rows of `cols`, every axis past the first
    /// counted as rows, in a type every launch of a site reads
    /// ([`Quant::dense`]).
    fn site(&self, name: &str, cols: usize, rows: usize) -> Result<Site, SessionError> {
        self.site_of(name, cols, rows, Quant::dense)
    }

    /// A projection of a layer's P1: [`Table::site`], and F32 rows too, which
    /// the segmented mat-vec reads.
    fn p1_site(&self, name: &str, cols: usize, rows: usize) -> Result<Site, SessionError> {
        self.site_of(name, cols, rows, |_| true)
    }

    /// Tensor `name` as [`Table::site`] reads it, in a type `has` admits.
    fn site_of(
        &self,
        name: &str,
        cols: usize,
        rows: usize,
        has: fn(Quant) -> bool,
    ) -> Result<Site, SessionError> {
        let (ty, dims) = self.info(name)?;
        let got_rows: u64 = dims.iter().skip(1).product();
        let shape_ok = dims.first().and_then(|&c| usize::try_from(c).ok()) == Some(cols)
            && usize::try_from(got_rows).ok() == Some(rows);
        if !shape_ok {
            return Err(refuse(format!(
                "{name} is {dims:?}; the step reads {rows} rows of {cols}"
            )));
        }
        let quant = Quant::of_ggml(ty).filter(|&q| has(q)).ok_or_else(|| {
            refuse(format!(
                "{name} is {ty:?}, which no mat-vec of this backend runs ({})",
                runs(has).join(", ")
            ))
        })?;
        Ok(Site {
            name: name.to_owned(),
            cols,
            rows,
            quant,
        })
    }
}

/// A weight matrix the loader copies into a buffer of its own: `rows` rows of
/// `cols` values in the mat-vec type `quant`.
pub(super) struct Site {
    pub(super) name: String,
    pub(super) cols: usize,
    pub(super) rows: usize,
    pub(super) quant: Quant,
}

impl Site {
    /// The bytes of the tensor: `rows * cols / values_per_block()` blocks.
    pub(super) fn bytes(&self) -> u64 {
        count(self.rows)
            * count(self.cols / self.quant.values_per_block())
            * count(self.quant.block_bytes())
    }
}

/// A tensor the loader reads as `len` f32 values: F32 as it is, MLX's affine
/// 8-bit dequantized at open.
pub(super) struct Values {
    pub(super) name: String,
    pub(super) len: usize,
    pub(super) ty: GgmlType,
}

impl Values {
    fn new(name: &str, len: usize, ty: GgmlType) -> Values {
        Values {
            name: name.to_owned(),
            len,
            ty,
        }
    }
}

/// One dispatch of a layer's P1: the sites of one weight type, in one buffer.
pub(super) struct Dispatch {
    pub(super) quant: Quant,
    /// The P1's sites it holds, by place in [`P1Layout::sites`].
    pub(super) members: Vec<usize>,
    /// One past the last output value a member writes.
    pub(super) out_end: usize,
}

/// A layer's P1: the norm its load applies, and its sites in the order their
/// outputs sit, grouped into one dispatch per weight type.
pub(super) struct P1Layout {
    pub(super) norm: Values,
    pub(super) sites: Vec<Site>,
    pub(super) dispatches: Vec<Dispatch>,
}

impl P1Layout {
    /// Site `i`'s output range, the rows of the sites before it ahead.
    pub(super) fn range(&self, i: usize) -> core::ops::Range<usize> {
        let at: usize = self.sites[..i].iter().map(|s| s.rows).sum();
        at..at + self.sites[i].rows
    }
}

pub(super) struct DeltaLayout {
    pub(super) p1: P1Layout,
    pub(super) out: Site,
    pub(super) conv: Values,
    pub(super) dt_bias: Values,
    pub(super) ssm_a: Values,
    pub(super) ssm_norm: Values,
    pub(super) n_k: usize,
    pub(super) map: HeadMap,
    /// The elements of the two-lane state and of the conv ring.
    pub(super) state: usize,
    pub(super) ring: usize,
}

pub(super) struct AttnLayout {
    pub(super) p1: P1Layout,
    pub(super) out: Site,
    pub(super) q_norm: Values,
    pub(super) k_norm: Values,
    /// The elements of one cache plane.
    pub(super) cache: usize,
    /// The output gate of one value a query head is the P1's fourth segment,
    /// after the query, the key and the value.
    pub(super) gate: bool,
    /// The positions the layer attends; none: every position.
    pub(super) window: Option<u32>,
    /// The layer's rope table, by place in [`Layout::ropes`].
    pub(super) rope: usize,
}

/// A gated short conv layer: its input projection is one P1 segment of
/// `3 · hidden` rows (`B | C | x`), and the ring keeps the conv's inputs of
/// the last positions.
pub(super) struct ConvLayout {
    pub(super) p1: P1Layout,
    pub(super) out: Site,
    /// The filters, `[hidden][taps]` f32, the oldest tap first.
    pub(super) conv: Values,
    pub(super) taps: usize,
    /// The slots of the ring ([`Ring::rows`]).
    pub(super) ring_rows: usize,
    /// The elements of the ring: `ring_rows` rows of `hidden` ([`Ring::values`]).
    pub(super) ring: usize,
}

pub(super) enum MixerLayout {
    Delta(DeltaLayout),
    Attention(AttnLayout),
    ShortConv(ConvLayout),
}

/// A layer's sandwich: the norm each half's output goes through before it
/// joins the stream, `x += add[i] · gain[i] ⊙ rms_norm(out)`. The epsilon and
/// the scales are the plan's (`SandwichPlan`), read by the join.
pub(super) struct PostLayout {
    /// The mixer's gain, then the FFN's: F32, `hidden` values.
    pub(super) gains: [Values; 2],
}

/// A rope table's inputs: θ's base, and the file's factors the pair
/// frequencies divide by on a layer whose rope has them. The factors' values
/// are read at open, where the table is built.
pub(super) struct RopeLayout {
    pub(super) base: f32,
    pub(super) factors: Option<Values>,
}

/// A layer's FFN half: the router's values and the experts' sites, the
/// shared expert's apart from the routed stacks. A dense layer has no router:
/// its three matrices are the one expert of the sites, picked every token.
pub(super) struct FfnLayout {
    pub(super) norm: Values,
    /// The router's weight; none for a dense FFN.
    pub(super) gate_inp: Option<Values>,
    pub(super) gate_up_quant: Quant,
    pub(super) gate: Site,
    pub(super) up: Site,
    pub(super) down_quant: Quant,
    pub(super) down: Site,
    /// The shared expert, on a router that runs one.
    pub(super) shared: Option<SharedLayout>,
}

/// A shared expert: its router row and its three sites, each of its own type
/// (its stack's, or a pair of [`crate::kernels::APART_GATE_UP`] or
/// [`crate::kernels::APART_DOWN`]).
pub(super) struct SharedLayout {
    pub(super) gate_inp: Values,
    pub(super) gate: Site,
    pub(super) up: Site,
    pub(super) down: Site,
}

pub(super) struct LayerLayout {
    pub(super) mixer: MixerLayout,
    pub(super) ffn: FfnLayout,
    /// The sandwich norms; none on a layer that adds each half to the stream.
    pub(super) post: Option<PostLayout>,
}

/// What the arena holds, resolved: every tensor it copies, every length it
/// sizes a buffer by. The loader makes buffers from this and checks nothing.
pub(super) struct Layout {
    pub(super) hidden: usize,
    pub(super) vocab: usize,
    pub(super) ctx: usize,
    pub(super) eps: f32,
    pub(super) top_k: usize,
    pub(super) n_experts: usize,
    /// The rope tables, one for each distinct base and factors the attention
    /// layers have.
    pub(super) ropes: Vec<RopeLayout>,
    pub(super) embed: Site,
    /// The head: `output.weight`, or the token embedding again for a file with
    /// no `output.weight`.
    pub(super) head: Site,
    pub(super) head_norm: Values,
    pub(super) layers: Vec<LayerLayout>,
    /// The elements of one rope table, the attention's partials and output.
    pub(super) rope_len: usize,
    pub(super) attn_partials: usize,
    pub(super) attn_out: usize,
    /// The widest P1 output a layer writes.
    pub(super) proj: usize,
}

impl Layout {
    /// Whether a layer is a short conv: the arena holds the conv's output.
    pub(super) fn has_conv(&self) -> bool {
        self.layers
            .iter()
            .any(|l| matches!(l.mixer, MixerLayout::ShortConv(_)))
    }

    /// Whether a layer is a sandwich: the arena holds the buffer a half's
    /// out projection writes before its join.
    pub(super) fn has_join(&self) -> bool {
        self.layers.iter().any(|l| l.post.is_some())
    }
}

/// A file the open runs: the instance its shape selects and that instance's
/// launch geometry (every arena rule below was judged at it, and the arena and
/// the launches are sized by it), the plans of its layers and the arena's
/// layout.
pub struct Checked {
    pub(super) instance: &'static Instance,
    pub(super) geom: Geometry,
    pub(super) plans: Vec<LayerPlan>,
    pub(super) layout: Layout,
}

/// What the file's description says, once it reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub arch: String,
    pub layers: usize,
    pub delta: usize,
    pub conv: usize,
    pub attention: usize,
    pub hidden: usize,
    pub vocab: usize,
    pub experts: usize,
    pub picks: usize,
    pub ctx_train: usize,
}

/// What [`check`] found: the description, every refusal, and the checked file
/// when there is no refusal.
pub struct Outcome {
    pub spec: Option<Summary>,
    pub items: Vec<Item>,
    pub checked: Option<Checked>,
}

/// The layers a pass of `cap` rows asks of the rows arena: the delta layers'
/// key heads.
fn delta_heads(layers: &[LayerLayout]) -> impl Iterator<Item = usize> + '_ {
    layers.iter().filter_map(|l| match &l.mixer {
        MixerLayout::Delta(d) => Some(d.n_k),
        MixerLayout::Attention(_) | MixerLayout::ShortConv(_) => None,
    })
}

/// The facts the layers collect: the rope tables (one for each distinct base
/// and factors, in the order a layer first asks for them) and the MoE's
/// experts-and-picks shape, which every layer must share (the first layer that
/// has it sets it, every later layer holds to it).
#[derive(Default)]
struct SharedShapes {
    ropes: Vec<RopeLayout>,
    moe: Option<(usize, usize)>,
}

/// The rows an F32 projection's first output row is a multiple of (the pass's
/// f32 GEMM stores `float4`s).
const F32_ROW_ALIGN: usize = 4;

fn count(n: usize) -> u64 {
    u64::try_from(n).expect("a count fits u64")
}

/// The file types of the mat-vec families `has` admits, in [`Quant::ALL`]'s
/// order, named as a refusal prints them.
fn runs(has: fn(Quant) -> bool) -> Vec<String> {
    Quant::ALL
        .into_iter()
        .filter(|&q| has(q))
        .map(|q| q.ggml().to_string().to_ascii_uppercase())
        .collect()
}

/// [`runs`] as "A, B or C": the words a site's refusal names its types in.
fn reading(has: fn(Quant) -> bool) -> String {
    let names = runs(has);
    match names.split_last() {
        Some((last, [])) => last.clone(),
        Some((last, rest)) => format!("{} or {last}", rest.join(", ")),
        None => String::new(),
    }
}

impl Outcome {
    /// The checked file, or the first refusal: what the open returns.
    pub fn checked_or_first(self) -> Result<Checked, SessionError> {
        match self.checked {
            Some(checked) => Ok(checked),
            None => Err(self.items.into_iter().next().map_or_else(
                || super::session::refuse("the file was refused with no item named".to_owned()),
                |item| item.error,
            )),
        }
    }
}

/// The open's checks over `split`'s header and tensor table, at a cache of
/// `ctx` rows.
#[must_use]
pub fn check(split: &Split, ctx: usize) -> Outcome {
    let mut items = Vec::new();
    let mut summary = None;
    let checked = read(split, ctx, &mut items, &mut summary);
    Outcome {
        spec: summary,
        checked: if items.is_empty() { checked } else { None },
        items,
    }
}

/// The plan of layer `l`'s mixer of kind `kind`, at the lists `sites`: the
/// arm of the chain's `layer_plans` loop. An attention layer is planned from
/// its own description `layer`, which holds its window and its rope's factors.
fn plan_mixer(
    split: &Split,
    pd: &PlanDims,
    kind: Kind35,
    layer: &LayerSpec,
    l: usize,
    sites: &Sites35,
) -> Result<MixerPlan, ChainError> {
    Ok(match kind {
        Kind35::Gqa(kind, flash) => {
            let SpecMixer::Gqa(g) = &layer.mixer else {
                return Err(ChainError::shape(
                    WHAT,
                    format!(
                        "layer {l}: the kind reads attention; the description's mixer is {}",
                        layer.mixer.what()
                    ),
                ));
            };
            MixerPlan::Gqa(plan_gqa_of(split, WHAT, pd, l, (kind, flash), g, sites)?)
        }
        Kind35::Delta(shape) => MixerPlan::Delta(plan_delta(split, WHAT, pd, shape, l, sites)?),
        Kind35::ShortConv(ring) => {
            MixerPlan::ShortConv(plan_shortconv(split, WHAT, pd, ring, l, sites)?)
        }
    })
}

/// A half's plan, or its refusals: the strict read's, as the chain words it
/// (the open's own first refusal), then each other site of the half the
/// launch lists refuse (`sweep`), which a reader that stops at its first
/// refusal does not reach.
fn read_half<T>(
    items: &mut Vec<Item>,
    l: usize,
    strict: Result<T, ChainError>,
    sweep: impl FnOnce() -> Vec<SessionError>,
) -> Option<T> {
    let first = match strict {
        Ok(plan) => return Some(plan),
        Err(e) => SessionError::Plan(e),
    };
    let text = first.to_string();
    items.push(Item {
        layer: Some(l),
        error: first,
    });
    for extra in sweep() {
        if extra.to_string() != text {
            items.push(Item {
                layer: Some(l),
                error: extra,
            });
        }
    }
    None
}

/// Every site of layer `l`'s mixer the launch lists refuse, in the reader's
/// order. The sites are named by the plan read at the widest lists; a half
/// that read fails even there (a missing tensor, a shape, a type no family
/// holds) has only its strict refusal.
fn mixer_sweep(
    split: &Split,
    pd: &PlanDims,
    kind: Kind35,
    layer: &LayerSpec,
    l: usize,
) -> Vec<SessionError> {
    match plan_mixer(split, pd, kind, layer, l, &WIDEST) {
        Ok(MixerPlan::Delta(d)) => refused_sites(
            split,
            [d.qkv, d.gate, d.beta, d.alpha, d.ssm_out],
            PROJECTED,
        ),
        Ok(MixerPlan::Gqa(g)) => refused_sites(
            split,
            [g.attn_q, g.attn_k, g.attn_v, g.attn_output]
                .into_iter()
                .chain(g.gate.map(|h| h.name)),
            PROJECTED,
        ),
        Ok(MixerPlan::ShortConv(c)) => refused_sites(split, [c.conv_in, c.conv_out], PROJECTED),
        Err(_) => Vec::new(),
    }
}

/// Every site of layer `l`'s FFN the launch lists refuse, in the reader's
/// order: a routed layer's stacks and shared parts by the file's own names,
/// a dense layer's by its plan.
fn ffn_sweep(split: &Split, pd: &PlanDims, layer: &LayerSpec, l: usize) -> Vec<SessionError> {
    match &layer.ffn {
        Ffn::Moe(_) => {
            let n = file_names(l);
            refused_sites(
                split,
                [n.gate, n.sh_gate, n.up, n.sh_up, n.down, n.sh_down],
                PROJECTED,
            )
        }
        Ffn::Dense { .. } => match plan_ffn(split, WHAT, pd, layer, l, &WIDEST, Read35::Whole) {
            Ok(f) => refused_sites(split, [f.gate, f.up, f.down], PROJECTED),
            Err(_) => Vec::new(),
        },
    }
}

/// The sites among `names` that the chain's own site check refuses at the
/// launch list `list` (the reader's, [`PROJECTED`] for the types a read names
/// before the dispatch lists judge them): its words, so the item is the one a strict read
/// returns for the site. A site is judged by the shape the header gives it, so only its
/// type and presence are at stake.
fn refused_sites(
    split: &Split,
    names: impl IntoIterator<Item = String>,
    list: &[SiteTy],
) -> Vec<SessionError> {
    names
        .into_iter()
        .filter_map(|name| {
            let shape = split.find(&name).map_or((0, 0), |(_, t)| {
                let k = t.dims.first().copied().unwrap_or(0);
                let rows: u64 = t.dims.iter().skip(1).product();
                (
                    usize::try_from(rows).unwrap_or(usize::MAX),
                    usize::try_from(k).unwrap_or(usize::MAX),
                )
            });
            file_site(split, WHAT, &name, shape, list)
                .err()
                .map(SessionError::Plan)
        })
        .collect()
}

/// The facts of `spec` a plan prints first, before any check.
fn summarize(split: &Split, spec: &ModelSpec) -> Summary {
    let count = |is: fn(&SpecMixer) -> bool| spec.layers.iter().filter(|l| is(&l.mixer)).count();
    let (experts, picks) = spec
        .layers
        .iter()
        .find_map(|l| match &l.ffn {
            Ffn::Moe(m) => Some((m.experts as usize, m.top_k as usize)),
            Ffn::Dense { .. } => None,
        })
        .unwrap_or((0, 0));
    Summary {
        arch: split.architecture().unwrap_or("").to_owned(),
        layers: spec.layers.len(),
        delta: count(|m| matches!(m, SpecMixer::DeltaRule(_))),
        conv: count(|m| matches!(m, SpecMixer::ShortConv(_))),
        attention: count(|m| matches!(m, SpecMixer::Gqa(_))),
        hidden: spec.hidden as usize,
        vocab: spec.vocab as usize,
        experts,
        picks,
        ctx_train: spec.ctx_train as usize,
    }
}

fn read(
    split: &Split,
    ctx: usize,
    items: &mut Vec<Item>,
    summary: &mut Option<Summary>,
) -> Option<Checked> {
    let path = split
        .shard_path(0)
        .expect("a split holds a first shard")
        .display()
        .to_string();
    let spec = match model::arch::spec(split) {
        Ok(read) => read.spec,
        Err(e) => {
            At {
                items: &mut *items,
                layer: None,
            }
            .push(super::session::refuse(format!("{path}: {e}")));
            return None;
        }
    };
    read_spec(split, &spec, &path, ctx, items, summary)
}

/// [`read`] from the description `spec` the header gave: the instance, the
/// plans and the arena's rules. `path` names the file in a refusal.
fn read_spec(
    split: &Split,
    spec: &ModelSpec,
    path: &str,
    ctx: usize,
    items: &mut Vec<Item>,
    summary: &mut Option<Summary>,
) -> Option<Checked> {
    *summary = Some(summarize(split, spec));
    // The library a session of the file runs on: the instance its shape
    // selects (a registered one, or one made at open), a geometry no kernel
    // takes refused by name. The arena's rules below are judged at the
    // instance's launch geometry, so a shape with no instance has none of
    // them: its plans are read, and no more.
    let inst = At {
        items: &mut *items,
        layer: None,
    }
    .ok(instance::select(spec).map_err(|e| super::session::refuse(format!("{path}: {e}"))));
    let geom = inst.map(Geometry::of);
    let (kinds, pd) = match kinds_of(&spec.layers)
        .and_then(|kinds| PlanDims::of(spec, WHAT).map(|pd| (kinds, pd)))
    {
        Ok(read) => read,
        Err(e) => {
            At {
                items: &mut *items,
                layer: None,
            }
            .push(SessionError::Plan(e));
            return None;
        }
    };
    // Each layer's two halves, in the order `layer_plans` reads them.
    let mut plans: Vec<Option<LayerPlan>> = Vec::with_capacity(kinds.len());
    for (l, (kind, layer)) in kinds.iter().zip(&spec.layers).enumerate() {
        let mixer = read_half(
            items,
            l,
            plan_mixer(split, &pd, *kind, layer, l, &SITES),
            || mixer_sweep(split, &pd, *kind, layer, l),
        );
        let ffn = read_half(
            items,
            l,
            plan_ffn(split, WHAT, &pd, layer, l, &SITES, Read35::Whole),
            || ffn_sweep(split, &pd, layer, l),
        );
        let residual = read_half(
            items,
            l,
            plan_residual(split, WHAT, &pd, layer, l),
            Vec::new,
        );
        plans.push(
            mixer
                .zip(ffn)
                .zip(residual)
                .map(|((mixer, ffn), residual)| LayerPlan {
                    mixer,
                    ffn,
                    residual,
                }),
        );
    }
    let table = geom.as_ref().map(|geom| Table { split, geom });
    let head = table.as_ref().and_then(|table| {
        head_checks(
            &mut At {
                items: &mut *items,
                layer: None,
            },
            table,
            spec,
            ctx,
        )
    });
    let mut shared = SharedShapes::default();
    let hidden = spec.hidden as usize;
    let mut layouts: Vec<Option<LayerLayout>> = Vec::with_capacity(plans.len());
    for (l, (plan, layer)) in plans.iter().zip(&spec.layers).enumerate() {
        layouts.push(plan.as_ref().zip(table.as_ref()).and_then(|(plan, table)| {
            layer_checks(
                &mut At {
                    items: &mut *items,
                    layer: Some(l),
                },
                table,
                plan,
                layer,
                hidden,
                ctx,
                &mut shared,
            )
        }));
    }
    let mut file = At {
        items: &mut *items,
        layer: None,
    };
    // A layer that was refused sets nothing: the facts the layers share are
    // missing for that reason alone, and only a file with no refusal yet
    // can lack them for its own.
    let clean = file.items.is_empty();
    let moe = clean
        .then(|| file.ok(shared.moe.ok_or_else(|| refuse("a file of no layer"))))
        .flatten();
    let ropes = clean
        .then(|| {
            file.ok(if shared.ropes.is_empty() {
                Err(refuse("no attention layer to read the rope from"))
            } else {
                Ok(shared.ropes)
            })
        })
        .flatten()
        .and_then(|ropes| {
            // The table each layer's rope turns by: built at open, from the
            // base and the factors; a base or a width it cannot take is
            // refused here, and a factor it cannot divide by at the open.
            let rot = geom?.attn.rot;
            for r in &ropes {
                file.ok(RopeTable::new(&RopeSpec::window(r.base, rot))
                    .map_err(|e| refuse(format!("the rope table: {e}"))))?;
            }
            Some(ropes)
        });
    let layers: Option<Vec<LayerLayout>> = layouts.into_iter().collect();
    let proj = layers.as_ref().and_then(|layers| {
        file.ok(layers
            .iter()
            .flat_map(|l| match &l.mixer {
                MixerLayout::Delta(d) => &d.p1.dispatches,
                MixerLayout::Attention(a) => &a.p1.dispatches,
                MixerLayout::ShortConv(c) => &c.p1.dispatches,
            })
            .map(|d| d.out_end)
            .max()
            .ok_or_else(|| refuse("a file of no layer")))
    });
    let attn_out = geom.and_then(|g| {
        file.ok(g
            .attn
            .out_len()
            .ok_or_else(|| refuse("the attention output does not fit")))
    });
    let plans: Option<Vec<LayerPlan>> = plans.into_iter().collect();
    let (inst, geom, head, (n_experts, top_k), ropes, layers, proj, attn_out, plans) = (
        inst?, geom?, head?, moe?, ropes?, layers?, proj?, attn_out?, plans?,
    );
    Some(Checked {
        instance: inst,
        geom,
        plans,
        layout: Layout {
            hidden,
            vocab: spec.vocab as usize,
            ctx,
            eps: spec.rms_eps,
            top_k,
            n_experts,
            ropes,
            embed: head.embed,
            head: head.head,
            head_norm: head.head_norm,
            layers,
            rope_len: head.rope_len,
            attn_partials: head.attn_partials,
            attn_out,
            proj,
        },
    })
}

/// What the file-level checks resolve: the embedding, the head and the
/// lengths a cache of `ctx` rows sets.
struct HeadChecks {
    embed: Site,
    head: Site,
    head_norm: Values,
    rope_len: usize,
    attn_partials: usize,
}

/// The cache's rows and the lengths of them every buffer past the layers
/// holds at `g`: `ctx` within the file's trained context, and every length of
/// it fits.
fn cache_lens(g: &Geometry, ctx: usize, trained: usize) -> Result<(usize, usize), SessionError> {
    if ctx == 0 {
        return Err(refuse(format!(
            "a cache of {ctx} rows; a session holds at least one"
        )));
    }
    if ctx > trained {
        return Err(refuse(format!(
            "a cache of {ctx} rows; the file was trained to {trained}"
        )));
    }
    let word = u32::try_from(ctx)
        .map_err(|_| refuse(format!("a cache of {ctx} rows does not fit u32")))?;
    let _ = g
        .attn
        .f16_len(ctx)
        .ok_or_else(|| refuse(format!("the f16 cache of {ctx} rows does not fit")))?;
    let rope_len = ctx
        .checked_mul(g.attn.rot)
        .ok_or_else(|| refuse(format!("the rope table of {ctx} rows does not fit")))?;
    let segments = segments_for_keys(word) as usize;
    let attn_partials = g.attn.partials_len(segments).ok_or_else(|| {
        refuse(format!(
            "the attention partials of {segments} segments do not fit"
        ))
    })?;
    Ok((rope_len, attn_partials))
}

/// The tensor the head reads: `output.weight`, or the token embedding when the
/// file has no `output.weight` (a head tied to the embedding, which the loader
/// then makes a second buffer of). A file with neither is answered
/// `output.weight`, which the site check refuses by name as absent.
fn head_tensor(split: &Split) -> String {
    if split.find(&names::output()).is_none() && split.find(&names::token_embd()).is_some() {
        names::token_embd()
    } else {
        names::output()
    }
}

/// The cache, the hidden width, the vocabulary, the embedding and the head.
fn head_checks(at: &mut At<'_>, t: &Table<'_>, spec: &ModelSpec, ctx: usize) -> Option<HeadChecks> {
    let lens = at.ok(cache_lens(t.geom, ctx, spec.ctx_train as usize));
    let hidden = spec.hidden as usize;
    let vocab = spec.vocab as usize;
    let mut ok = true;
    if hidden != t.geom.hidden {
        at.push(refuse(format!(
            "a hidden width of {hidden}; the router reads {}",
            t.geom.hidden
        )));
        ok = false;
    }
    if vocab == 0 {
        at.push(refuse(
            "a vocabulary of 0; the head reads at least one row".to_string(),
        ));
        ok = false;
    }
    let embed = at
        .ok(t.site(&names::token_embd(), hidden, vocab))
        .filter(|s| {
            let fits = s.quant.embeds();
            if !fits {
                at.push(refuse(format!(
                    "{} is {:?}; the embedding reads {}",
                    names::token_embd(),
                    s.quant,
                    reading(Quant::embeds)
                )));
            }
            fits
        });
    let head_name = head_tensor(t.split);
    let head = at.ok(t.site(&head_name, hidden, vocab)).filter(|s| {
        let fits = s.quant.heads();
        if !fits {
            at.push(refuse(format!(
                "{head_name} is {:?}; the head reads {}",
                s.quant,
                reading(Quant::heads)
            )));
        }
        fits
    });
    let head_norm = at.ok(t.f32s(&names::output_norm(), hidden));
    let (rope_len, attn_partials) = lens?;
    ok.then_some(HeadChecks {
        embed: embed?,
        head: head?,
        head_norm: head_norm?,
        rope_len,
        attn_partials,
    })
}

/// A layer's P1: `parts` (each a tensor and its rows, all of `cols`) normed on
/// load by `norm`, one dispatch per weight type.
fn p1_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    norm: &str,
    parts: &[(&str, usize)],
    cols: usize,
) -> Option<P1Layout> {
    let mut sites = Vec::with_capacity(parts.len());
    let mut ok = true;
    for &(name, rows) in parts {
        match at.ok(t.p1_site(name, cols, rows)) {
            Some(site) => {
                if !cols.is_multiple_of(site.quant.values_per_block()) {
                    at.push(refuse(format!(
                        "{norm}'s P1 reads {cols} columns, not whole blocks of {}",
                        site.quant.values_per_block()
                    )));
                    ok = false;
                }
                sites.push(site);
            }
            None => ok = false,
        }
    }
    let mut dispatches = Vec::new();
    if ok {
        let mut at_row = Vec::with_capacity(sites.len());
        let mut next = 0usize;
        for s in &sites {
            at_row.push(next);
            next += s.rows;
        }
        // The pass runs an F32 segment as a GEMM whose output window starts
        // at the segment's first row, four floats at least from a 16-byte
        // boundary.
        for (s, &row) in sites.iter().zip(&at_row) {
            if s.quant == Quant::F32 && !row.is_multiple_of(F32_ROW_ALIGN) {
                at.push(refuse(format!(
                    "{}: an f32 projection's output starts at row {row}, not a multiple of {F32_ROW_ALIGN}",
                    s.name
                )));
                ok = false;
            }
        }
        for quant in Quant::ALL {
            let members: Vec<usize> = (0..sites.len())
                .filter(|&i| sites[i].quant == quant)
                .collect();
            if members.is_empty() {
                continue;
            }
            if members.len() > MAX_SEGMENTS {
                at.push(refuse(format!(
                    "{norm}'s P1 has {} {quant:?} tensors; a dispatch takes {MAX_SEGMENTS}",
                    members.len()
                )));
                ok = false;
                continue;
            }
            let out_end = members
                .iter()
                .map(|&i| at_row[i] + sites[i].rows)
                .max()
                .unwrap_or(0);
            dispatches.push(Dispatch {
                quant,
                members,
                out_end,
            });
        }
    }
    let norm = at.ok(t.f32s(norm, cols));
    (ok).then_some(P1Layout {
        norm: norm?,
        sites,
        dispatches,
    })
}

fn delta_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    d: &DeltaPlan,
    hidden: usize,
) -> Option<DeltaLayout> {
    let LinearShape { n_k, n_v, map } = d.shape;
    let g = &t.geom.delta;
    if n_v != g.value_heads {
        at.push(refuse(format!(
            "{n_v} value heads; the delta step runs {}",
            g.value_heads
        )));
        return None;
    }
    let Some(ch) = g.channels(n_k) else {
        at.push(refuse(format!("{n_k} key heads do not fit")));
        return None;
    };
    let z_len = g.value_heads * g.head_cols;
    let p1 = p1_checks(
        at,
        t,
        &d.attn_norm,
        &[
            (&d.qkv, ch),
            (&d.gate, z_len),
            (&d.beta, g.value_heads),
            (&d.alpha, g.value_heads),
        ],
        hidden,
    );
    let out = at.ok(t.site(&d.ssm_out, z_len, hidden)).filter(|s| {
        let fits = s.quant.headnorms();
        if !fits {
            at.push(refuse(format!(
                "{} is {:?}; the head-norm projection reads {}",
                d.ssm_out,
                s.quant,
                reading(Quant::headnorms)
            )));
        }
        fits
    });
    let state = at.ok(g
        .state_elems(LANES)
        .ok_or_else(|| refuse("the delta state does not fit")));
    let ring = at.ok(g
        .ring_elems(n_k)
        .ok_or_else(|| refuse("the conv ring does not fit")));
    let conv = at.ok(t.f32s(&d.conv, g.conv_taps * ch));
    let dt_bias = at.ok(t.f32s(&d.dt_bias, g.value_heads));
    let ssm_a = at.ok(t.f32s(&d.ssm_a, g.value_heads));
    let ssm_norm = at.ok(t.f32s(&d.ssm_norm, g.head_cols));
    Some(DeltaLayout {
        p1: p1?,
        out: out?,
        conv: conv?,
        dt_bias: dt_bias?,
        ssm_a: ssm_a?,
        ssm_norm: ssm_norm?,
        n_k,
        map: match map {
            chain::linear::KHeadMap::Tiled => HeadMap::Tiled,
            chain::linear::KHeadMap::Grouped => HeadMap::Grouped,
        },
        state: state?,
        ring: ring?,
    })
}

fn attention_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    g: &GqaPlan,
    hidden: usize,
    ctx: usize,
    rope: Option<usize>,
) -> Option<AttnLayout> {
    let a = &t.geom.attn;
    let q_rows = at.ok(a
        .qg_len()
        .ok_or_else(|| refuse("the query rows do not fit")));
    let kv_rows = a.kv_heads * a.head;
    let p1 = q_rows.and_then(|q_rows| {
        // The output gate of one value a query head, when the layer has one,
        // is the segment after the value's.
        let mut parts = vec![
            (g.attn_q.as_str(), q_rows),
            (g.attn_k.as_str(), kv_rows),
            (g.attn_v.as_str(), kv_rows),
        ];
        if let Some(gate) = &g.gate {
            parts.push((gate.name.as_str(), a.query_heads));
        }
        p1_checks(at, t, &g.attn_norm, &parts, hidden)
    });
    let att = at.ok(a
        .out_len()
        .ok_or_else(|| refuse("the attention rows do not fit")));
    let out = att.and_then(|att| at.ok(t.site(&g.attn_output, att, hidden)));
    let cache = at.ok(a
        .f16_len(ctx)
        .ok_or_else(|| refuse("the cache does not fit")));
    let q_norm = at.ok(t.f32s(&g.attn_q_norm, a.head));
    let k_norm = at.ok(t.f32s(&g.attn_k_norm, a.head));
    Some(AttnLayout {
        p1: p1?,
        out: out?,
        q_norm: q_norm?,
        k_norm: k_norm?,
        cache: cache?,
        gate: g.gate.is_some(),
        window: g.window,
        rope: rope?,
    })
}

/// A short conv layer: its input projection as the one segment of its P1, its
/// output projection, its filters and the ring `taps` taps take. The plan's
/// ring is held to the ring those taps make, so a plan and a description that
/// disagree on the conv are refused by name here.
fn conv_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    c: &ShortConvPlan,
    taps: usize,
    hidden: usize,
) -> Option<ConvLayout> {
    if taps == 0 {
        at.push(refuse("a short conv of no tap"));
        return None;
    }
    let ring = Ring::new(taps);
    if c.ring != ring {
        at.push(refuse(format!(
            "the plan's conv ring is {} taps; the layer's conv has {taps}",
            c.ring.taps()
        )));
        return None;
    }
    let p1 = p1_checks(
        at,
        t,
        &c.attn_norm,
        &[(c.conv_in.as_str(), 3 * hidden)],
        hidden,
    );
    let out = at.ok(t.site(&c.conv_out, hidden, hidden));
    let conv = at.ok(t.f32s(&c.conv, hidden * taps));
    Some(ConvLayout {
        p1: p1?,
        out: out?,
        conv: conv?,
        taps,
        ring_rows: ring.rows(),
        ring: ring.values(hidden),
    })
}

/// The place in [`SharedShapes::ropes`] of the rope table a layer turns by:
/// θ's `base` and the factors tensor `freqs` names, if any. An earlier
/// layer's table when it has the same pair, else a new one whose factors the
/// header must give as F32, one for each pair of the values the rope turns.
fn rope_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    shared: &mut SharedShapes,
    base: f32,
    freqs: Option<&str>,
) -> Option<usize> {
    let same = |r: &RopeLayout| {
        r.base.to_bits() == base.to_bits() && r.factors.as_ref().map(|v| v.name.as_str()) == freqs
    };
    if let Some(place) = shared.ropes.iter().position(same) {
        return Some(place);
    }
    let factors = match freqs {
        Some(name) => Some(at.ok(t.f32s(name, t.geom.attn.rot / 2))?),
        None => None,
    };
    shared.ropes.push(RopeLayout { base, factors });
    Some(shared.ropes.len() - 1)
}

/// A layer's residual form. The plain add has nothing to read. A sandwich
/// reads its two gains, F32 of `hidden` values, and keeps the plan's epsilon
/// and scales. The join runs after the out projection of an attention or a
/// short conv layer and after a dense FFN's down; a sandwich on a gated delta
/// layer or over a routed FFN is refused by name, since the delta's out
/// projection and the routed fold add into the stream. `None`: refused.
fn post_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    plan: &LayerPlan,
    hidden: usize,
) -> Option<Option<PostLayout>> {
    let ResidualPlan::Sandwich(s) = &plan.residual else {
        return Some(None);
    };
    let mut ok = true;
    if matches!(plan.mixer, MixerPlan::Delta(_)) {
        at.push(refuse(
            "a sandwich residual on a gated delta layer; the delta step's out projection adds \
             into the stream",
        ));
        ok = false;
    }
    if matches!(plan.ffn.route, FfnRoute::Router { .. }) {
        at.push(refuse(
            "a sandwich residual over a routed FFN; the routed fold adds into the stream",
        ));
        ok = false;
    }
    let [mixer, ffn] = s.norms.each_ref().map(|name| at.ok(t.f32s(name, hidden)));
    let (mixer, ffn) = (mixer?, ffn?);
    ok.then_some(Some(PostLayout {
        gains: [mixer, ffn],
    }))
}

fn ffn_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    plan: &LayerPlan,
    hidden: usize,
    (n_experts, top_k): (usize, usize),
    l: usize,
) -> Option<FfnLayout> {
    // A shared expert, in its stacks' types (`Joined`) or kept apart at its
    // own: the launches read it as the router's last group either way, and
    // `ok` below holds the type pairs they have an entry for.
    let joined = match &plan.ffn.route {
        FfnRoute::Router {
            shared: Some(SharedPlan::Joined | SharedPlan::Apart(_)),
            ..
        } => true,
        FfnRoute::Router { shared: None, .. } => false,
        FfnRoute::Dense => return dense_ffn_checks(at, t, plan, hidden),
    };
    let (router, experts) = (&t.geom.router, &t.geom.experts);
    if experts.dense {
        at.push(refuse(
            "a routed FFN, and the instance's experts run one dense slot",
        ));
        return None;
    }
    if joined != (router.shared == 1) {
        at.push(refuse(if joined {
            "a shared expert, which the instance's router and experts run none of"
        } else {
            "a router with no shared expert, and the instance's router runs one"
        }));
        return None;
    }
    if n_experts != router.experts || top_k != experts.routed_ids {
        at.push(refuse(format!(
            "{n_experts} experts picking {top_k}; the router and the experts run \
             {} picking {}",
            router.experts, experts.routed_ids
        )));
        return None;
    }
    // The file's tensors, under the plan's file-side names (chain's one owner
    // of the naming): the plan names the joined stacks, which this backend
    // reads as the routed stack and the shared expert apart.
    let file = file_names(l);
    let gate = at.ok(t.site(&file.gate, hidden, experts.k * n_experts));
    let up = at.ok(t.site(&file.up, hidden, experts.k * n_experts));
    let shexp_gate = joined.then(|| at.ok(t.site(&file.sh_gate, hidden, experts.k)));
    let shexp_up = joined.then(|| at.ok(t.site(&file.sh_up, hidden, experts.k)));
    let mut ok = true;
    match (&gate, &up, &shexp_gate, &shexp_up) {
        (Some(g), Some(u), Some(Some(sg)), Some(Some(su)))
            if u.quant != g.quant || su.quant != sg.quant =>
        {
            at.push(refuse(format!(
                "gate {:?}, up {:?}, shared gate {:?}, shared up {:?}; the pair runs one type \
                 for its stacks and one for its shared expert",
                g.quant, u.quant, sg.quant, su.quant
            )));
            ok = false;
        }
        (Some(g), Some(_), Some(Some(sg)), Some(Some(_)))
            if !shared_entry(g.quant, sg.quant, true) =>
        {
            at.push(refuse(format!(
                "gate and up {:?}, shared gate and up {:?}; the pair has no entry for the types",
                g.quant, sg.quant
            )));
            ok = false;
        }
        (Some(g), Some(u), None, None) if u.quant != g.quant => {
            at.push(refuse(format!(
                "gate {:?}, up {:?}; the pair runs one type",
                g.quant, u.quant
            )));
            ok = false;
        }
        _ => {}
    }
    let down = at.ok(t.site(&file.down, experts.k, hidden * n_experts));
    let shexp_down = joined.then(|| at.ok(t.site(&file.sh_down, experts.k, hidden)));
    if let (Some(d), Some(Some(sd))) = (&down, &shexp_down)
        && !shared_entry(d.quant, sd.quant, false)
    {
        at.push(refuse(format!(
            "down {:?}, shared down {:?}; the fold has no entry for the types",
            d.quant, sd.quant
        )));
        ok = false;
    }
    let norm = at.ok(t.f32s(&plan.ffn.ffn_norm, hidden));
    // The router and the shared-expert gate: F32 in a GGUF file, MLX's
    // affine 8-bit in an MLX file, dequantized exactly at open.
    let gate_inp = at.ok(t.f32s_or_a8(&file.router, n_experts * hidden));
    let gate_inp_shexp = joined.then(|| at.ok(t.f32s_or_a8(&file.router_gate, hidden)));
    let shared = match (gate_inp_shexp, shexp_gate, shexp_up, shexp_down) {
        (Some(gate_inp), Some(gate), Some(up), Some(down)) => Some(SharedLayout {
            gate_inp: gate_inp?,
            gate: gate?,
            up: up?,
            down: down?,
        }),
        _ => None,
    };
    let (gate, up, down) = (gate?, up?, down?);
    ok.then_some(FfnLayout {
        norm: norm?,
        gate_inp: Some(gate_inp?),
        gate_up_quant: gate.quant,
        down_quant: down.quant,
        gate,
        up,
        down,
        shared,
    })
}

/// A dense layer's FFN: its gate, up and down are the one expert of the
/// instance's one-expert stacks, at the instance's width, the pair's two in one
/// type. The launches read it as the experts' one slot, id 0 at weight 1, and no
/// router weight is read.
fn dense_ffn_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    plan: &LayerPlan,
    hidden: usize,
) -> Option<FfnLayout> {
    let experts = &t.geom.experts;
    if !experts.dense {
        at.push(refuse(
            "a dense FFN, and the instance's experts run routed slots",
        ));
        return None;
    }
    let f = &plan.ffn;
    let gate = at.ok(t.site(&f.gate, hidden, experts.k));
    let up = at.ok(t.site(&f.up, hidden, experts.k));
    let down = at.ok(t.site(&f.down, experts.k, hidden));
    let mut ok = true;
    if let (Some(g), Some(u)) = (&gate, &up)
        && u.quant != g.quant
    {
        at.push(refuse(format!(
            "gate {:?}, up {:?}; the pair runs one type",
            g.quant, u.quant
        )));
        ok = false;
    }
    let norm = at.ok(t.f32s(&f.ffn_norm, hidden));
    let (gate, up, down) = (gate?, up?, down?);
    ok.then_some(FfnLayout {
        norm: norm?,
        gate_inp: None,
        gate_up_quant: gate.quant,
        down_quant: down.quant,
        gate,
        up,
        down,
        shared: None,
    })
}

/// Layer `l`'s layout: the mixer its plan and description agree on, the FFN
/// of the expert shape every layer shares, the facts the layers must share
/// read from and set on `shared` (the first layer that has one sets each).
fn layer_checks(
    at: &mut At<'_>,
    t: &Table<'_>,
    plan: &LayerPlan,
    layer: &LayerSpec,
    hidden: usize,
    ctx: usize,
    shared: &mut SharedShapes,
) -> Option<LayerLayout> {
    let l = at.layer.expect("a layer's check names its layer");
    let mixer = match (&plan.mixer, &layer.mixer) {
        (MixerPlan::Delta(d), SpecMixer::DeltaRule(_)) => {
            delta_checks(at, t, d, hidden).map(MixerLayout::Delta)
        }
        (MixerPlan::Gqa(g), SpecMixer::Gqa(spec_g)) => {
            let mut ok = true;
            if spec_g.rope.dims as usize != t.geom.attn.rot {
                at.push(refuse(format!(
                    "a rope over {} values; the attention turns {}",
                    spec_g.rope.dims, t.geom.attn.rot
                )));
                ok = false;
            }
            let rope = rope_checks(at, t, shared, spec_g.rope.base, g.rope_freqs.as_deref());
            let attn = attention_checks(at, t, g, hidden, ctx, rope);
            attn.filter(|_| ok).map(MixerLayout::Attention)
        }
        (MixerPlan::ShortConv(c), SpecMixer::ShortConv(spec_c)) => {
            conv_checks(at, t, c, spec_c.taps as usize, hidden).map(MixerLayout::ShortConv)
        }
        _ => {
            at.push(refuse("the plan's mixer and the description's differ"));
            None
        }
    };
    // A dense FFN is the one expert of a one-expert stack, picked by every
    // token: the shape the layers share, and the width the experts run.
    let (shape, width) = match &layer.ffn {
        Ffn::Moe(m) => ((m.experts as usize, m.top_k as usize), m.expert_ff as usize),
        Ffn::Dense { ff, act } => {
            if !matches!(act, Act::SwiGlu { limit: None }) {
                at.push(refuse(format!(
                    "a dense FFN of {}; the pair runs SwiGLU",
                    act.what()
                )));
                return None;
            }
            ((1, 1), *ff as usize)
        }
    };
    let mut ok = true;
    match shared.moe {
        None => shared.moe = Some(shape),
        Some(s) if s == shape => {}
        Some(s) => {
            at.push(refuse(format!(
                "{shape:?} experts and picks, another layer's are {s:?}"
            )));
            ok = false;
        }
    }
    if width != t.geom.experts.k {
        at.push(refuse(if matches!(layer.ffn, Ffn::Dense { .. }) {
            format!(
                "a dense FFN of {width} values; the experts run {}",
                t.geom.experts.k
            )
        } else {
            format!("an expert of {width}; the experts run {}", t.geom.experts.k)
        }));
        ok = false;
    }
    let ffn = ffn_checks(at, t, plan, hidden, shape, l);
    let post = post_checks(at, t, plan, hidden);
    ok.then_some(LayerLayout {
        mixer: mixer?,
        ffn: ffn?,
        post: post?,
    })
}

/// One class of buffers the open makes: how many, the bytes asked of the
/// driver, and the bytes its pages count.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Tally {
    pub buffers: u64,
    pub requested: u64,
    pub paged: u64,
}

impl Tally {
    fn add(&mut self, bytes: u64) {
        self.buffers += 1;
        self.requested += bytes;
        self.paged += bytes.div_ceil(PAGE) * PAGE;
    }
}

/// The bytes the open makes buffers of, by class: the file's weights, the
/// attention caches, the delta layers' state and rings, the token arena's
/// other buffers, and the rows arena.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Bytes {
    pub weights: Tally,
    pub kv: Tally,
    pub state: Tally,
    pub scratch: Tally,
    pub rows: Tally,
}

impl Bytes {
    /// All classes together.
    #[must_use]
    pub fn total(&self) -> Tally {
        [self.weights, self.kv, self.state, self.scratch, self.rows]
            .iter()
            .fold(Tally::default(), |a, t| Tally {
                buffers: a.buffers + t.buffers,
                requested: a.requested + t.requested,
                paged: a.paged + t.paged,
            })
    }
}

impl Checked {
    /// The bytes the open makes buffers of: every buffer the arena makes, and
    /// the rows arena of the pass the session holds, from the lengths the
    /// open sizes them by.
    pub fn bytes(&self) -> Result<Bytes, SessionError> {
        let (g, geom) = (&self.layout, &self.geom);
        let mut b = Bytes::default();
        let f32s = |n: usize| count(n) * 4;
        let weights = &mut b.weights;
        for s in [&g.embed, &g.head] {
            weights.add(s.bytes());
        }
        weights.add(f32s(g.head_norm.len));
        let p1_weights = |t: &mut Tally, p1: &P1Layout| {
            for d in &p1.dispatches {
                t.add(d.members.iter().map(|&i| p1.sites[i].bytes()).sum());
            }
            t.add(f32s(p1.norm.len));
        };
        for layer in &g.layers {
            match &layer.mixer {
                MixerLayout::Delta(d) => {
                    p1_weights(&mut b.weights, &d.p1);
                    b.weights.add(d.out.bytes());
                    for v in [&d.conv, &d.dt_bias, &d.ssm_a, &d.ssm_norm] {
                        b.weights.add(f32s(v.len));
                    }
                    b.state.add(f32s(d.state));
                    b.state.add(f32s(d.ring));
                }
                MixerLayout::Attention(a) => {
                    p1_weights(&mut b.weights, &a.p1);
                    b.weights.add(a.out.bytes());
                    for v in [&a.q_norm, &a.k_norm] {
                        b.weights.add(f32s(v.len));
                    }
                    b.kv.add(count(a.cache) * 2);
                    b.kv.add(count(a.cache) * 2);
                }
                MixerLayout::ShortConv(c) => {
                    p1_weights(&mut b.weights, &c.p1);
                    b.weights.add(c.out.bytes());
                    b.weights.add(f32s(c.conv.len));
                    b.state.add(f32s(c.ring));
                }
            }
            if let Some(post) = &layer.post {
                for gain in &post.gains {
                    b.weights.add(f32s(gain.len));
                }
            }
            let f = &layer.ffn;
            let sh = f.shared.as_ref();
            for s in [
                Some(&f.gate),
                Some(&f.up),
                sh.map(|s| &s.gate),
                sh.map(|s| &s.up),
                Some(&f.down),
                sh.map(|s| &s.down),
            ]
            .into_iter()
            .flatten()
            {
                b.weights.add(s.bytes());
            }
            for v in [Some(&f.norm), f.gate_inp.as_ref(), sh.map(|s| &s.gate_inp)]
                .into_iter()
                .flatten()
            {
                b.weights.add(f32s(v.len));
            }
            // The picks and their weights, and a router's logits, partials
            // and ticket: a dense layer has the fixed route alone.
            let slots = geom.experts.slots;
            let words = if f.gate_inp.is_some() {
                vec![geom.router.rows, slots, slots, geom.router.partials, 1]
            } else {
                vec![slots, slots]
            };
            for n in words {
                b.scratch.add(f32s(n));
            }
        }
        // The rope tables, one a distinct base and factors; the conv's output
        // and the sandwich's join buffer, where a layer has them; and the
        // hand-on buffer, the reader's input and one scale word.
        for _ in &g.ropes {
            b.scratch.add(f32s(g.rope_len));
        }
        for held in [g.has_conv(), g.has_join()] {
            if held {
                b.scratch.add(f32s(g.hidden));
            }
        }
        b.scratch.add(f32s(g.hidden + 1));
        for words in [
            g.hidden,
            g.proj,
            geom.delta.value_heads * geom.delta.head_cols,
            geom.delta.partials_stride,
            g.attn_out,
            g.attn_partials,
            geom.attn.kv_heads,
            geom.experts.slots * geom.experts.k,
            g.vocab,
            super::arena::RECORD_SLOTS,
            1,
            LANES,
        ] {
            b.scratch.add(f32s(words));
        }
        // The fault word of each step record slot.
        for _ in 0..super::arena::RECORD_SLOTS {
            b.scratch.add(count(FAULT_WORDS) * 4);
        }
        let lens = RowsLens::of(
            geom,
            pass_rows_max(geom).min(g.ctx),
            g.hidden,
            g.proj,
            delta_heads(&g.layers),
        )?;
        for bytes in lens.buffer_bytes() {
            b.rows.add(bytes);
        }
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use chain::arch::qwen3moe::plan::SandwichPlan;
    use chain::arch::qwen3moe::read::layer_plans;
    use gguf::Value;
    use gguf::write::{Layout as FileLayout, TensorDecl};
    use model::arch::models::Residual;

    use super::*;
    use crate::instance::{FfnAxes, MoeAxes};

    /// A file's header and tensor table, in memory, at a registered row's
    /// shapes (or `lfm3_instance`'s): two layers (the Qwen3.6 row's a delta
    /// layer, then an attention one: `full_attention_interval` 2), every
    /// weight Q4_K but the head's Q6_K, and no weight byte anywhere. A test
    /// edits it and reads it back as a `Split`.
    struct Synthetic {
        kvs: Vec<(String, Value)>,
        tensors: Vec<(String, Vec<u64>, GgmlType)>,
        /// The first attention layer's place.
        attn_layer: usize,
    }

    /// The Qwen3.6 row's residual width, for the tests that pin its words.
    const HIDDEN: u64 = 2048;
    const VOCAB: u64 = 248_320;
    /// LFM3-1.7B's layers: a short conv (c), a full attention layer (F) or a
    /// sliding one (S).
    const LFM3_KINDS: &str = "ccFcScFcScFcScFcScFcScFc";
    const LFM3_VOCAB: u64 = 128_000;
    /// A sliding layer's window, and the taps of every short conv.
    const LFM3_WINDOW: u32 = 256;
    const LFM3_TAPS: u64 = 7;
    /// The dense FFN of LFM3-1.7B's every layer.
    const LFM3_FF: u32 = 7680;

    /// The instance of an LFM3-1.7B file: no row registers its dense FFN, so
    /// it is the one made at open for its key, at the attention axes of the
    /// kernel gate's `LFM3_ATTN` row.
    fn lfm3_instance() -> &'static Instance {
        let key = instance::Key {
            hidden: u32::try_from(HIDDEN).expect("the width fits u32"),
            ffn: FfnAxes::Dense { ff: LFM3_FF },
            attn: instance::LFM3_ATTN.key.attn,
            delta: None,
        };
        instance::open_key(key).expect("every kernel takes LFM3's geometry")
    }
    /// The key heads of the delta layer's conv channels: a file's own count,
    /// no axis of an instance.
    const KEY_HEADS: u32 = 16;

    impl Synthetic {
        /// The Qwen3.6-35B-A3B table: [`Synthetic::of`] at its row.
        fn new() -> Synthetic {
            Synthetic::of(&instance::QWEN35MOE_A3B)
        }

        /// The table of a file of `inst`'s shape, its header and tensors at
        /// the row's axes: two layers, the last of them attention (a gated
        /// delta layer ahead of it when the row has a delta axis).
        fn of(inst: &Instance) -> Synthetic {
            let key = &inst.key;
            if key.delta.is_none() && key.is_dense() {
                return Synthetic::lfm3(inst);
            }
            let moe = key.routed();
            let (arch, pre, delta) = match (key.delta, moe.map(|m| m.router.rule.gated)) {
                (Some(d), Some(true)) => ("qwen35moe", "qwen35", Some(d)),
                (None, Some(false)) => ("qwen3moe", "qwen2", None),
                // A dense qwen35 file: the delta net beside the attention, each
                // layer's FFN one SwiGLU block.
                (Some(d), None) => ("qwen35", "qwen35", Some(d)),
                _ => panic!("{}: no synthetic file is written for this shape", inst.name),
            };
            let u = |k: &str, v: u32| (format!("{arch}.{k}"), Value::U32(v));
            let a = &key.attn;
            let ff = match key.ffn {
                FfnAxes::Routed(m) => m.expert_ff,
                FfnAxes::Dense { ff } => ff,
            };
            let mut kvs = vec![
                (
                    gguf::GENERAL_ARCHITECTURE.to_owned(),
                    Value::String(arch.to_owned()),
                ),
                (
                    "tokenizer.ggml.pre".to_owned(),
                    Value::String(pre.to_owned()),
                ),
                u("block_count", 2),
                u(
                    "vocab_size",
                    u32::try_from(VOCAB).expect("the vocabulary fits u32"),
                ),
                u("context_length", 262_144),
                u("embedding_length", key.hidden),
                u("attention.head_count", a.n_head),
                u("attention.head_count_kv", a.n_kv),
                (
                    format!("{arch}.attention.layer_norm_rms_epsilon"),
                    Value::F32(1e-6),
                ),
                u("attention.key_length", a.row.head),
                u("attention.value_length", a.row.value),
                (format!("{arch}.rope.freq_base"), Value::F32(10_000_000.0)),
                u("rope.dimension_count", a.rot),
            ];
            match moe {
                Some(m) => kvs.extend([
                    u("expert_count", m.router.experts),
                    u("expert_used_count", m.top_k),
                    u("expert_feed_forward_length", ff),
                ]),
                None => kvs.push(u("feed_forward_length", ff)),
            }
            if let Some(d) = delta {
                if moe.is_some() {
                    kvs.push(u("expert_shared_feed_forward_length", ff));
                }
                kvs.extend([
                    u("ssm.conv_kernel", d.conv_taps),
                    u("ssm.state_size", d.head_cols),
                    u("ssm.group_count", KEY_HEADS),
                    u("ssm.time_step_rank", d.value_heads),
                    u("ssm.inner_size", d.value_heads * d.head_cols),
                    u("full_attention_interval", 2),
                    (
                        format!("{arch}.rope.dimension_sections"),
                        Value::Array(vec![
                            Value::I32(11),
                            Value::I32(11),
                            Value::I32(10),
                            Value::I32(0),
                        ]),
                    ),
                ]);
            }
            let mut t = Synthetic {
                kvs,
                tensors: Vec::new(),
                attn_layer: 1,
            };
            let hidden = u64::from(key.hidden);
            let (ff, experts) = (
                u64::from(ff),
                moe.map_or(1, |m| u64::from(m.router.experts)),
            );
            let q = GgmlType::Q4_K;
            t.add("token_embd.weight", &[hidden, VOCAB], q);
            t.add("output.weight", &[hidden, VOCAB], GgmlType::Q6_K);
            t.add("output_norm.weight", &[hidden], GgmlType::F32);
            let (heads, kv, head) = (
                u64::from(a.n_head),
                u64::from(a.n_kv),
                u64::from(a.row.head),
            );
            for l in 0..2 {
                let attention = delta.is_none() || l == 1;
                let b = |s: &str| format!("blk.{l}.{s}");
                t.add(&b("attn_norm.weight"), &[hidden], GgmlType::F32);
                if attention {
                    let gate = 1 + u64::from(a.q_gate);
                    t.add(&b("attn_q.weight"), &[hidden, heads * head * gate], q);
                    t.add(&b("attn_k.weight"), &[hidden, kv * head], q);
                    t.add(&b("attn_v.weight"), &[hidden, kv * head], q);
                    t.add(&b("attn_output.weight"), &[heads * head, hidden], q);
                    t.add(&b("attn_q_norm.weight"), &[head], GgmlType::F32);
                    t.add(&b("attn_k_norm.weight"), &[head], GgmlType::F32);
                } else if let Some(d) = delta {
                    let (vh, hc) = (u64::from(d.value_heads), u64::from(d.head_cols));
                    let channels = (2 * u64::from(KEY_HEADS) + vh) * hc;
                    t.add(&b("attn_qkv.weight"), &[hidden, channels], q);
                    t.add(&b("attn_gate.weight"), &[hidden, vh * hc], q);
                    t.add(&b("ssm_beta.weight"), &[hidden, vh], q);
                    t.add(&b("ssm_alpha.weight"), &[hidden, vh], q);
                    t.add(
                        &b("ssm_conv1d.weight"),
                        &[u64::from(d.conv_taps), channels],
                        GgmlType::F32,
                    );
                    t.add(&b("ssm_dt.bias"), &[vh], GgmlType::F32);
                    t.add(&b("ssm_a"), &[vh], GgmlType::F32);
                    t.add(&b("ssm_norm.weight"), &[hc], GgmlType::F32);
                    t.add(&b("ssm_out.weight"), &[vh * hc, hidden], q);
                }
                let norm = if delta.is_some() {
                    "post_attention_norm.weight"
                } else {
                    "ffn_norm.weight"
                };
                t.add(&b(norm), &[hidden], GgmlType::F32);
                if moe.is_none() {
                    t.add(&b("ffn_gate.weight"), &[hidden, ff], q);
                    t.add(&b("ffn_up.weight"), &[hidden, ff], q);
                    t.add(&b("ffn_down.weight"), &[ff, hidden], q);
                    continue;
                }
                t.add(&b("ffn_gate_inp.weight"), &[hidden, experts], GgmlType::F32);
                if delta.is_some() {
                    t.add(&b("ffn_gate_inp_shexp.weight"), &[hidden], GgmlType::F32);
                }
                t.add(&b("ffn_gate_exps.weight"), &[hidden, ff, experts], q);
                t.add(&b("ffn_up_exps.weight"), &[hidden, ff, experts], q);
                t.add(&b("ffn_down_exps.weight"), &[ff, hidden, experts], q);
                if delta.is_some() {
                    t.add(&b("ffn_gate_shexp.weight"), &[hidden, ff], q);
                    t.add(&b("ffn_up_shexp.weight"), &[hidden, ff], q);
                    t.add(&b("ffn_down_shexp.weight"), &[ff, hidden], q);
                }
            }
            t
        }

        /// The table of an LFM3 file of `inst`'s shape (`lfm3`): the
        /// [`LFM3_KINDS`] layers, a dense FFN and the sandwich's two norms in
        /// each, the rope's factors, and a head tied to the embedding (no
        /// `output.weight`), whose type, Q5_K, both the embedding and the head
        /// read. Every weight is Q4_K but the embedding.
        fn lfm3(inst: &Instance) -> Synthetic {
            let key = &inst.key;
            let a = &key.attn;
            let FfnAxes::Dense { ff } = key.ffn else {
                panic!("{}: the LFM3 table has a dense FFN", inst.name);
            };
            let u = |k: &str, v: u32| (format!("lfm3.{k}"), Value::U32(v));
            let kv_heads: Vec<Value> = LFM3_KINDS
                .chars()
                .map(|c| {
                    Value::I32(if c == 'c' {
                        0
                    } else {
                        i32::try_from(a.n_kv).expect("fits")
                    })
                })
                .collect();
            let sliding: Vec<Value> = LFM3_KINDS.chars().map(|c| Value::Bool(c == 'S')).collect();
            let kvs = vec![
                (
                    gguf::GENERAL_ARCHITECTURE.to_owned(),
                    Value::String("lfm3".to_owned()),
                ),
                (
                    "tokenizer.ggml.pre".to_owned(),
                    Value::String("lfm2".to_owned()),
                ),
                u(
                    "block_count",
                    u32::try_from(LFM3_KINDS.len()).expect("the layer count fits u32"),
                ),
                u("embedding_length", key.hidden),
                u("context_length", 128_000),
                u(
                    "vocab_size",
                    u32::try_from(LFM3_VOCAB).expect("the vocabulary fits u32"),
                ),
                u("feed_forward_length", ff),
                u("attention.head_count", a.n_head),
                (
                    "lfm3.attention.head_count_kv".to_owned(),
                    Value::Array(kv_heads),
                ),
                u("attention.key_length", a.row.head),
                u("attention.value_length", a.row.value),
                (
                    "lfm3.attention.layer_norm_rms_epsilon".to_owned(),
                    Value::F32(1e-5),
                ),
                u("attention.sliding_window", LFM3_WINDOW),
                (
                    "lfm3.attention.sliding_window_pattern".to_owned(),
                    Value::Array(sliding),
                ),
                ("lfm3.rope.freq_base".to_owned(), Value::F32(1e6)),
                ("lfm3.rope.freq_base_swa".to_owned(), Value::F32(1e4)),
                u("rope.dimension_count", a.rot),
                u("shortconv.l_cache", 7),
            ];
            let mut t = Synthetic {
                kvs,
                tensors: Vec::new(),
                attn_layer: LFM3_KINDS.find(['F', 'S']).expect("an attention layer"),
            };
            let e = u64::from(key.hidden);
            let (heads, kv, head) = (
                u64::from(a.n_head),
                u64::from(a.n_kv),
                u64::from(a.row.head),
            );
            let ff = u64::from(ff);
            let q = GgmlType::Q4_K;
            t.add("token_embd.weight", &[e, LFM3_VOCAB], GgmlType::Q5_K);
            t.add("output_norm.weight", &[e], GgmlType::F32);
            t.add("rope_freqs.weight", &[u64::from(a.rot) / 2], GgmlType::F32);
            for (l, kind) in LFM3_KINDS.chars().enumerate() {
                let b = |s: &str| format!("blk.{l}.{s}.weight");
                for norm in [
                    "attn_norm",
                    "post_attention_norm",
                    "ffn_norm",
                    "post_ffw_norm",
                ] {
                    t.add(&b(norm), &[e], GgmlType::F32);
                }
                t.add(&b("ffn_gate"), &[e, ff], q);
                t.add(&b("ffn_up"), &[e, ff], q);
                t.add(&b("ffn_down"), &[ff, e], q);
                if kind == 'c' {
                    t.add(&b("shortconv.in_proj"), &[e, 3 * e], q);
                    t.add(&b("shortconv.conv"), &[LFM3_TAPS, e], GgmlType::F32);
                    t.add(&b("shortconv.out_proj"), &[e, e], q);
                } else {
                    t.add(&b("attn_q"), &[e, heads * head], q);
                    t.add(&b("attn_k"), &[e, kv * head], q);
                    t.add(&b("attn_v"), &[e, kv * head], q);
                    t.add(&b("attn_output"), &[heads * head, e], q);
                    t.add(&b("attn_gate"), &[e, heads], q);
                    t.add(&b("attn_q_norm"), &[head], GgmlType::F32);
                    t.add(&b("attn_k_norm"), &[head], GgmlType::F32);
                }
            }
            t
        }

        fn add(&mut self, name: &str, dims: &[u64], ty: GgmlType) {
            self.tensors.push((name.to_owned(), dims.to_vec(), ty));
        }

        /// Tensor `name` retyped.
        fn ty(mut self, name: &str, ty: GgmlType) -> Synthetic {
            self.tensor(name).2 = ty;
            self
        }

        /// Tensor `name` of other dims.
        fn dims(mut self, name: &str, dims: &[u64]) -> Synthetic {
            self.tensor(name).1 = dims.to_vec();
            self
        }

        /// Tensor `name` absent.
        fn drop(mut self, name: &str) -> Synthetic {
            self.tensors.retain(|t| t.0 != name);
            self
        }

        fn kv(mut self, key: &str, v: Value) -> Synthetic {
            self.kvs.iter_mut().find(|(k, _)| k == key).expect(key).1 = v;
            self
        }

        fn tensor(&mut self, name: &str) -> &mut (String, Vec<u64>, GgmlType) {
            self.tensors
                .iter_mut()
                .find(|t| t.0 == name)
                .unwrap_or_else(|| panic!("the synthetic file has no {name}"))
        }

        /// The bytes of every tensor the table holds.
        fn tensor_bytes(&self) -> u64 {
            self.decls().iter().map(|d| d.nbytes).sum()
        }

        /// The bytes of the weight buffers the arena makes of the table: every
        /// tensor but the rope's factors (they make a table of the arena's
        /// own), and the embedding a second time when the head is tied to it.
        fn weight_bytes(&self) -> u64 {
            let of = |name: &str| {
                self.decls()
                    .iter()
                    .find(|d| d.name == name)
                    .map_or(0, |d| d.nbytes)
            };
            let tied = of("output.weight") == 0;
            self.tensor_bytes() - of("rope_freqs.weight")
                + if tied { of("token_embd.weight") } else { 0 }
        }

        fn decls(&self) -> Vec<TensorDecl> {
            self.tensors
                .iter()
                .map(|(name, dims, ty)| {
                    let id = ty.as_u32();
                    let (_, blck, tsz) = gguf::ggml_type_info(id).expect("a sized type");
                    TensorDecl {
                        name: name.clone(),
                        dims: dims.clone(),
                        type_id: id,
                        nbytes: gguf::tensor_bytes(blck, tsz, dims).expect("a size"),
                    }
                })
                .collect()
        }

        fn layout(&self) -> FileLayout {
            FileLayout::new(&self.kvs, self.decls()).expect("the synthetic header lays out")
        }

        fn split(&self) -> Split {
            Split::header_only(PathBuf::from("synthetic.gguf"), &self.layout())
                .expect("the synthetic header opens")
        }
    }

    const CTX: usize = 4096;

    /// The items' texts, in order.
    fn texts(split: &Split, ctx: usize) -> Vec<String> {
        check(split, ctx)
            .items
            .iter()
            .map(|i| i.error.to_string())
            .collect()
    }

    /// The error the open met before the collector: `layer_plans` over the
    /// same header, the call the open made.
    fn old_plan_error(split: &Split) -> String {
        let spec = model::arch::spec(split)
            .expect("the description reads")
            .spec;
        let kinds = kinds_of(&spec.layers).expect("the kinds read");
        let pd = PlanDims::of(&spec, WHAT).expect("the dims read");
        let err = layer_plans(
            split,
            WHAT,
            &pd,
            &kinds,
            &spec.layers,
            &SITES,
            Read35::Whole,
        )
        .err()
        .expect("the plan is refused");
        SessionError::Plan(err).to_string()
    }

    /// Every registered row's file is judged at that row's launch geometry: a
    /// clean table of the row opens at its instance and geometry, its weights
    /// are the file's tensors byte for byte, and a norm of another width than
    /// the row's head is refused by name at the row's own width.
    #[test]
    fn each_registered_instance_is_judged_at_its_own_geometry() {
        for inst in instance::REGISTERED.iter().filter(|i| i.opens) {
            let name = inst.name;
            let t = Synthetic::of(inst);
            let out = check(&t.split(), CTX);
            assert!(out.items.is_empty(), "{name}: {:?}", out.items);
            let checked = out.checked.expect("a checked file");
            assert_eq!(*checked.instance, *inst, "{name}: the instance");
            assert_eq!(checked.geom, Geometry::of(inst), "{name}: the geometry");
            let bytes = checked.bytes().expect("the bytes");
            assert_eq!(
                bytes.weights.requested,
                t.weight_bytes(),
                "{name}: the weights"
            );
            assert!(bytes.total().paged >= bytes.total().requested, "{name}");

            let head = inst.key.attn.row.head;
            let wrong = u64::from(head) + 32;
            let l = t.attn_layer;
            let t = t.dims(&format!("blk.{l}.attn_q_norm.weight"), &[wrong]);
            let got = texts(&t.split(), CTX);
            assert_eq!(
                got,
                [format!(
                    "qwen35moe arena: layer {l}: blk.{l}.attn_q_norm.weight is F32 [{wrong}]; the \
                     step reads {head} F32 values"
                )],
                "{name}"
            );
        }
    }

    /// One class of buffers from the byte size of each: how many, the bytes
    /// asked of the driver and the bytes its pages count.
    fn class(sizes: &[u64]) -> Tally {
        Tally {
            buffers: count(sizes.len()),
            requested: sizes.iter().sum(),
            paged: sizes.iter().map(|s| s.div_ceil(PAGE) * PAGE).sum(),
        }
    }

    /// `Checked::bytes` over the synthetic table, class by class: the buffers
    /// `Arena::open` and `RowsArena::open` make, listed here by field from the
    /// table's shape (layer 0 delta with 16 key heads, layer 1 attention, 8
    /// picks, a cache of `CTX` rows), so a buffer dropped from or added to one
    /// list moves one class.
    #[test]
    fn the_bytes_pin_each_class_of_buffers() {
        use super::super::arena::RECORD_SLOTS;
        use crate::kernels::route_table_words;

        const N_K: usize = KEY_HEADS as usize;
        const TOP_K: usize = 8;
        let g = Geometry::of(&instance::QWEN35MOE_A3B);
        let (value_heads, head_cols) = (g.delta.value_heads, g.delta.head_cols);
        let hidden = HIDDEN as usize;
        let vocab = VOCAB as usize;
        let t = Synthetic::new();
        let checked = check(&t.split(), CTX).checked.expect("a checked file");
        let bytes = checked.bytes().expect("the bytes");
        let f32s = |n: usize| count(n) * 4;

        // The K and the V cache of the attention layer, f16.
        let cache = count(g.attn.kv_heads * CTX * g.attn.head) * 2;
        assert_eq!(bytes.kv, class(&[cache, cache]));

        // The delta layer's state lanes and its conv ring.
        let channels = (2 * N_K + value_heads) * head_cols;
        assert_eq!(
            bytes.state,
            class(&[
                f32s(LANES * value_heads * head_cols * head_cols),
                f32s(g.delta.ring_rows * channels),
            ])
        );

        // Each layer's router (logits, picks, weights, partials, ticket), then
        // the token arena's buffers in field order.
        let router = [g.router.rows, TOP_K + 1, TOP_K + 1, g.router.partials, 1];
        let proj = channels + value_heads * head_cols + 2 * value_heads;
        let attn_partials = g
            .attn
            .partials_len(segments_for_keys(CTX as u32) as usize)
            .expect("the partials fit");
        let mut scratch: Vec<u64> = [router, router].concat().into_iter().map(f32s).collect();
        scratch.extend(
            [
                hidden,                        // x
                proj,                          // proj
                value_heads * head_cols,       // g
                g.delta.partials_stride,       // delta_partials
                value_heads * head_cols,       // attn
                attn_partials,                 // attn_partials
                g.attn.kv_heads,               // attn_ticket
                CTX * g.attn.rot,              // rope
                g.experts.slots * g.experts.k, // h
                vocab,                         // logits
            ]
            .map(f32s),
        );
        scratch.extend([RECORD_SLOTS, 1, LANES].map(f32s)); // step_id, id_in, lane
        scratch.extend([f32s(FAULT_WORDS); RECORD_SLOTS]); // faults
        // PIN(2026-10-30): the hand-on buffer (`Arena::xn`), held by every
        // file: `hidden` values of the stream times the reader's gain and one
        // scale word.
        scratch.push(f32s(hidden + 1));
        assert_eq!(bytes.scratch, class(&scratch));

        // The rows arena of a pass of `cap` rows: ids, residual, normed input,
        // P1 output, mixer rows, delta partials and prepared rows, logits, four
        // slot lists, the route table, the pair's and the down's outputs.
        let cap = pass_rows_max(&g).min(CTX);
        let slots = cap * g.experts.slots;
        let prepared = g.delta.prepared_elems(N_K).expect("the prepared row fits");
        let mut rows = [
            cap,
            cap * hidden,
            cap * hidden,
            cap * proj.next_multiple_of(4),
            cap * value_heads * head_cols,
            cap * g.delta.partials_stride,
            cap * prepared,
            cap * g.router.logit_stride,
            slots,
            slots,
            slots,
            slots,
            route_table_words(slots, g.router.rows),
            slots * g.experts.k,
            slots * hidden,
        ]
        .map(f32s)
        .to_vec();
        // PIN(2026-10-29): the half rows the dense GEMMs read
        // (`RowsArena::xh`), `cap` rows of the mixer's width, two bytes a value.
        rows.push(count(cap * value_heads * head_cols) * 2);
        assert_eq!(bytes.rows, class(&rows));
    }

    /// A dense file's FFN is an instance's axis: its layers are the experts'
    /// one slot, the arena holds no router weight, and the plan names the
    /// fixed route.
    #[test]
    fn a_dense_file_opens_at_a_dense_instance_made_for_it() {
        let mut inst = instance::QWEN35MOE_A3B;
        inst.key.ffn = FfnAxes::Dense { ff: 512 };
        let t = Synthetic::of(&inst);
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let summary = out.spec.clone().expect("the description reads");
        assert_eq!((summary.experts, summary.picks), (0, 0));
        let checked = out.checked.expect("a checked file");
        assert_eq!(checked.instance.name, instance::UNREGISTERED);
        assert_eq!(checked.instance.key, inst.key, "the instance");
        assert!(checked.geom.experts.dense);
        assert_eq!((checked.layout.n_experts, checked.layout.top_k), (1, 1));
        for (layer, plan) in checked.layout.layers.iter().zip(&checked.plans) {
            assert!(layer.ffn.gate_inp.is_none() && layer.ffn.shared.is_none());
            assert!(matches!(plan.ffn.route, FfnRoute::Dense));
        }
        let bytes = checked.bytes().expect("the bytes");
        assert_eq!(bytes.weights.requested, t.tensor_bytes(), "the weights");
    }

    /// `Checked::bytes` of a dense file: a layer's fixed route (one id, one
    /// weight) and no router buffers, and a rows arena with no routed buffers
    /// but the gate's and the up's rows of the FFN's width.
    #[test]
    fn the_bytes_of_a_dense_file_hold_the_fixed_route_and_two_ffn_rows() {
        use super::super::arena::RECORD_SLOTS;

        const N_K: usize = KEY_HEADS as usize;
        let mut inst = instance::QWEN35MOE_A3B;
        inst.key.ffn = FfnAxes::Dense { ff: 512 };
        let g = Geometry::of(&inst);
        let (value_heads, head_cols) = (g.delta.value_heads, g.delta.head_cols);
        let (hidden, vocab) = (HIDDEN as usize, VOCAB as usize);
        let checked = check(&Synthetic::of(&inst).split(), CTX)
            .checked
            .expect("a checked file");
        let bytes = checked.bytes().expect("the bytes");
        let f32s = |n: usize| count(n) * 4;

        let channels = (2 * N_K + value_heads) * head_cols;
        let proj = channels + value_heads * head_cols + 2 * value_heads;
        let attn_partials = g
            .attn
            .partials_len(segments_for_keys(CTX as u32) as usize)
            .expect("the partials fit");
        let mut scratch: Vec<u64> = [1usize; 4].into_iter().map(f32s).collect();
        scratch.extend(
            [
                hidden,                        // x
                proj,                          // proj
                value_heads * head_cols,       // g
                g.delta.partials_stride,       // delta_partials
                value_heads * head_cols,       // attn
                attn_partials,                 // attn_partials
                g.attn.kv_heads,               // attn_ticket
                CTX * g.attn.rot,              // rope
                g.experts.slots * g.experts.k, // h
                vocab,                         // logits
            ]
            .map(f32s),
        );
        scratch.extend([RECORD_SLOTS, 1, LANES].map(f32s)); // step_id, id_in, lane
        scratch.extend([f32s(FAULT_WORDS); RECORD_SLOTS]); // faults
        // PIN(2026-10-30): the hand-on buffer (`Arena::xn`), held by every
        // file: `hidden` values of the stream times the reader's gain and one
        // scale word.
        scratch.push(f32s(hidden + 1));
        assert_eq!(bytes.scratch, class(&scratch));

        let cap = pass_rows_max(&g).min(CTX);
        let prepared = g.delta.prepared_elems(N_K).expect("the prepared row fits");
        let mut rows = [
            cap,
            cap * hidden,
            cap * hidden,
            cap * proj.next_multiple_of(4),
            cap * value_heads * head_cols,
            cap * g.delta.partials_stride,
            cap * prepared,
            cap * g.experts.k, // the gate's rows
            cap * g.experts.k, // the up's rows
        ]
        .map(f32s)
        .to_vec();
        // PIN(2026-10-29): the half rows the dense GEMMs read
        // (`RowsArena::xh`), `cap` rows of the wider of the mixer and the FFN's
        // activated row, two bytes a value.
        rows.push(count(cap * (value_heads * head_cols).max(g.experts.k)) * 2);
        assert_eq!(bytes.rows, class(&rows));
    }

    /// A dense layer the pair cannot run is refused by name at its layer.
    #[test]
    fn a_dense_layer_the_launches_cannot_run_is_refused_by_name() {
        let mut inst = instance::QWEN35MOE_A3B;
        inst.key.ffn = FfnAxes::Dense { ff: 512 };
        // The gate and the up in two types: the pair launch reads one.
        let t = Synthetic::of(&inst).ty("blk.0.ffn_up.weight", GgmlType::Q6_K);
        let got = texts(&t.split(), CTX);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(
            got[0].ends_with("layer 0: gate Q4K, up Q6K; the pair runs one type"),
            "{got:?}"
        );
        // A down of another width than the experts run.
        let t = Synthetic::of(&inst).dims("blk.1.ffn_down.weight", &[1024, 2048]);
        let got = texts(&t.split(), CTX);
        assert!(
            got.iter().any(|e| e.contains("blk.1.ffn_down.weight")),
            "{got:?}"
        );
        // F32 rows have no pair or fold entry.
        let t = Synthetic::of(&inst).ty("blk.0.ffn_gate.weight", GgmlType::F32);
        let got = texts(&t.split(), CTX);
        assert!(
            got.iter().any(|e| e.contains("blk.0.ffn_gate.weight")),
            "{got:?}"
        );
    }

    #[test]
    fn the_architecture_verdict_is_the_one_item() {
        // An architecture no description reader reads.
        let t = Synthetic::new().kv(
            gguf::GENERAL_ARCHITECTURE,
            Value::String("deepseek2".to_owned()),
        );
        let got = texts(&t.split(), CTX);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(
            got[0].starts_with("qwen35moe session: synthetic.gguf: ")
                && got[0].contains("unsupported architecture \"deepseek2\""),
            "{got:?}"
        );
    }

    #[test]
    fn a_geometry_no_kernel_takes_is_the_opens_first_item() {
        // An expert width the fold's blocks do not divide.
        let t = Synthetic::new().kv("qwen35moe.expert_feed_forward_length", Value::U32(700));
        let split = t.split();
        let got = texts(&split, CTX);
        assert!(
            got.first().is_some_and(|e| e
                .starts_with("qwen35moe session: synthetic.gguf: no instance is built for ")
                && e.contains("expert ff 700")
                && e.ends_with(
                    "experts: expert ff is 700; it must be a multiple of 256 \
                     (the fold maps whole blocks)"
                )),
            "{got:?}"
        );
        assert!(check(&split, CTX).checked.is_none());
    }

    /// A shape no row registers, whose geometry every kernel takes, is judged
    /// at an instance made for it: its arena and launches are sized by that
    /// geometry, and the file's weights are its tensors byte for byte.
    #[test]
    fn a_geometry_no_row_registers_is_judged_at_its_own_instance() {
        let mut inst = instance::QWEN35MOE_A3B;
        inst.key.hidden = 3072;
        inst.key.ffn = FfnAxes::Routed(MoeAxes {
            expert_ff: 1024,
            ..inst.key.routed().expect("a routed row")
        });
        let t = Synthetic::of(&inst);
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let checked = out.checked.expect("a checked file");
        assert_eq!(checked.instance.name, instance::UNREGISTERED);
        assert_eq!(checked.instance.key, inst.key, "the instance");
        assert_eq!(checked.geom, Geometry::of(checked.instance), "the geometry");
        assert_eq!(checked.geom.hidden, 3072);
        assert_eq!(checked.geom.experts.k, 1024);
        let bytes = checked.bytes().expect("the bytes");
        assert_eq!(bytes.weights.requested, t.tensor_bytes(), "the weights");
    }

    #[test]
    fn a_site_type_no_mat_vec_runs_is_the_chain_words_and_the_opens_first() {
        let t = Synthetic::new().ty("blk.0.attn_qkv.weight", GgmlType::Q3_K);
        let split = t.split();
        let got = texts(&split, CTX);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0], old_plan_error(&split));
        assert_eq!(
            got[0],
            "qwen35moe session: the plan: qwen35moe session: blk.0.attn_qkv.weight is q3_K in the \
             file; this site launches Q4_K, Q6_K, MLX_A4, MLX_A8, MLX_A4H, MLX_A8H, Q8_0, Q5_K, F32"
        );
    }

    /// `t` with each of `names` retyped to `ty`.
    fn retyped(mut t: Synthetic, names: &[&str], ty: GgmlType) -> Synthetic {
        for name in names {
            t = t.ty(name, ty);
        }
        t
    }

    /// The tensors of the two-layer table that a mat-vec, the embedding, the
    /// head or the head-norm projection reads, and no routed stack or shared
    /// expert: the sites a UD file holds in Q8_0 beside its stacks.
    const DENSE: [&str; 11] = [
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.attn_gate.weight",
        "blk.0.ssm_beta.weight",
        "blk.0.ssm_alpha.weight",
        "blk.0.ssm_out.weight",
        "blk.1.attn_q.weight",
        "blk.1.attn_k.weight",
        "blk.1.attn_v.weight",
        "blk.1.attn_output.weight",
    ];

    /// Q8_0 at the embedding, the head, both layers' P1 (one dispatch each),
    /// the delta layer's head-norm projection and the attention layer's out.
    #[test]
    fn q8_0_sites_open_at_every_dense_site() {
        let t = retyped(Synthetic::new(), &DENSE, GgmlType::Q8_0);
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let layout = out.checked.expect("a checked file").layout;
        assert_eq!(layout.embed.quant, Quant::Q8_0);
        assert_eq!(layout.head.quant, Quant::Q8_0);
        let [delta, attention] = &layout.layers[..] else {
            panic!("two layers");
        };
        let MixerLayout::Delta(d) = &delta.mixer else {
            panic!("layer 0 is a delta layer");
        };
        assert_eq!(d.out.quant, Quant::Q8_0);
        let quants: Vec<Quant> = d.p1.dispatches.iter().map(|x| x.quant).collect();
        assert_eq!(quants, [Quant::Q8_0]);
        assert_eq!(d.p1.dispatches[0].members, [0, 1, 2, 3]);
        let MixerLayout::Attention(a) = &attention.mixer else {
            panic!("layer 1 is an attention layer");
        };
        assert_eq!(a.out.quant, Quant::Q8_0);
        let quants: Vec<Quant> = a.p1.dispatches.iter().map(|x| x.quant).collect();
        assert_eq!(quants, [Quant::Q8_0]);
    }

    /// The same sites in Q5_K.
    #[test]
    fn q5_k_sites_open_at_every_dense_site() {
        let t = retyped(Synthetic::new(), &DENSE, GgmlType::Q5_K);
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let layout = out.checked.expect("a checked file").layout;
        assert_eq!(layout.embed.quant, Quant::Q5K);
        assert_eq!(layout.head.quant, Quant::Q5K);
    }

    /// The routed stacks and the shared expert in Q8_0 and Q5_K: the pair
    /// reads one type and the fold one, the shared expert in each its
    /// stack's.
    #[test]
    fn q8_0_and_q5_k_stacks_open_with_a_shared_expert_of_their_type() {
        for (gate_up, down) in [
            (GgmlType::Q8_0, GgmlType::Q8_0),
            (GgmlType::Q5_K, GgmlType::Q5_K),
            (GgmlType::Q4_K, GgmlType::Q5_K),
        ] {
            let mut t = Synthetic::new();
            for l in 0..2 {
                t = retyped(
                    t,
                    &[
                        &format!("blk.{l}.ffn_gate_exps.weight"),
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        &format!("blk.{l}.ffn_gate_shexp.weight"),
                        &format!("blk.{l}.ffn_up_shexp.weight"),
                    ],
                    gate_up,
                );
                t = retyped(
                    t,
                    &[
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        &format!("blk.{l}.ffn_down_shexp.weight"),
                    ],
                    down,
                );
            }
            let out = check(&t.split(), CTX);
            assert!(
                out.items.is_empty(),
                "{gate_up:?} {down:?}: {:?}",
                out.items
            );
            let layout = out.checked.expect("a checked file").layout;
            for layer in &layout.layers {
                assert_eq!(layer.ffn.gate_up_quant.ggml(), gate_up);
                assert_eq!(layer.ffn.down_quant.ggml(), down);
            }
        }
    }

    /// Every MLX affine kind opens at each site its kernels read: the dense
    /// sites a mat-vec reads ([`DENSE`], among them the embedding, the head
    /// and the head-norm projection), the routed stacks and the shared
    /// expert. The router, which the host dequantizes at open, takes the 8-bit
    /// kinds and refuses the 4-bit one by name.
    #[test]
    fn mlx_affine_kinds_open_at_every_site_their_kernels_read() {
        for kind in [
            GgmlType::MLX_A4,
            GgmlType::MLX_A8,
            GgmlType::MLX_A4H,
            GgmlType::MLX_A8H,
        ] {
            let quant = Quant::of_ggml(kind).expect("the kernels read the kind");
            let mut t = retyped(Synthetic::new(), &DENSE, kind);
            for l in 0..2 {
                t = retyped(
                    t,
                    &[
                        &format!("blk.{l}.ffn_gate_exps.weight"),
                        &format!("blk.{l}.ffn_up_exps.weight"),
                        &format!("blk.{l}.ffn_gate_shexp.weight"),
                        &format!("blk.{l}.ffn_up_shexp.weight"),
                    ],
                    kind,
                );
                t = retyped(
                    t,
                    &[
                        &format!("blk.{l}.ffn_down_exps.weight"),
                        &format!("blk.{l}.ffn_down_shexp.weight"),
                    ],
                    kind,
                );
            }
            let out = check(&t.split(), CTX);
            assert!(out.items.is_empty(), "{kind:?}: {:?}", out.items);
            let checked = out.checked.expect("a checked file");
            let layout = checked.layout;
            assert_eq!(layout.embed.quant, quant);
            assert_eq!(layout.head.quant, quant);
            for layer in &layout.layers {
                assert_eq!(layer.ffn.gate_up_quant, quant, "{kind:?}");
                assert_eq!(layer.ffn.down_quant, quant, "{kind:?}");
                let out = match &layer.mixer {
                    MixerLayout::Delta(d) => &d.out,
                    MixerLayout::Attention(a) => &a.out,
                    MixerLayout::ShortConv(_) => {
                        panic!("{kind:?}: the synthetic file has no short-conv layer")
                    }
                };
                assert_eq!(out.quant, quant, "{kind:?}");
            }
        }
        // The router and the shared-expert gate are values the open
        // dequantizes: the 8-bit kinds open, the 4-bit ones are refused.
        for (router, opens) in [
            (GgmlType::MLX_A8, true),
            (GgmlType::MLX_A8H, true),
            (GgmlType::MLX_A4, false),
        ] {
            let t = Synthetic::new()
                .ty("blk.0.ffn_gate_inp.weight", router)
                .ty("blk.1.ffn_gate_inp.weight", router);
            let out = check(&t.split(), CTX);
            assert_eq!(out.items.is_empty(), opens, "{router:?}: {:?}", out.items);
        }
    }

    /// A P1 of tensors of several types is one dispatch a type, in the
    /// order of [`Quant::ALL`] whatever order the tensors are in.
    #[test]
    fn a_p1_of_several_types_is_one_dispatch_a_type_in_the_kernels_order() {
        let t = Synthetic::new()
            .ty("blk.1.attn_q.weight", GgmlType::Q8_0)
            .ty("blk.1.attn_k.weight", GgmlType::Q5_K)
            .ty("blk.1.attn_v.weight", GgmlType::Q6_K);
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let layout = out.checked.expect("a checked file").layout;
        let MixerLayout::Attention(a) = &layout.layers[1].mixer else {
            panic!("layer 1 is an attention layer");
        };
        let got: Vec<(Quant, &[usize])> =
            a.p1.dispatches
                .iter()
                .map(|d| (d.quant, d.members.as_slice()))
                .collect();
        assert_eq!(
            got,
            [
                (Quant::Q6K, &[2usize][..]),
                (Quant::Q8_0, &[0][..]),
                (Quant::Q5K, &[1][..]),
            ]
        );
    }

    /// The delta net's alpha and beta in F32, as a UD file holds them: a
    /// dispatch of their own, last in the kernels' order, beside the Q8_0
    /// projections.
    #[test]
    fn f32_projections_are_a_dispatch_of_their_own_at_a_p1() {
        let t = retyped(Synthetic::new(), &DENSE, GgmlType::Q8_0)
            .ty("blk.0.ssm_beta.weight", GgmlType::F32)
            .ty("blk.0.ssm_alpha.weight", GgmlType::F32);
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let layout = out.checked.expect("a checked file").layout;
        let MixerLayout::Delta(d) = &layout.layers[0].mixer else {
            panic!("layer 0 is a delta layer");
        };
        let got: Vec<(Quant, &[usize])> =
            d.p1.dispatches
                .iter()
                .map(|x| (x.quant, x.members.as_slice()))
                .collect();
        assert_eq!(
            got,
            [(Quant::Q8_0, &[0usize, 1][..]), (Quant::F32, &[2, 3][..])]
        );
        assert_eq!(d.p1.sites[2].bytes(), 4 * 32 * 2048);
    }

    /// F32 is a type of the P1's projections only: every other site that
    /// reads it refuses it by name.
    #[test]
    fn f32_is_refused_by_name_beyond_a_p1_projection() {
        // A site the arena judges names its tensor.
        for name in [
            "blk.0.ssm_out.weight",
            "blk.1.attn_output.weight",
            "token_embd.weight",
            "output.weight",
        ] {
            let t = Synthetic::new().ty(name, GgmlType::F32);
            let got = texts(&t.split(), CTX);
            assert!(
                got.iter().any(|g| g.contains(name)),
                "{name} in F32 is not an item: {got:?}"
            );
        }
        // A stack or a shared part is judged by the dispatch lists.
        for name in ["blk.0.ffn_gate_exps.weight", "blk.1.ffn_down_shexp.weight"] {
            let t = Synthetic::new().ty(name, GgmlType::F32);
            let got = texts(&t.split(), CTX);
            assert!(
                got.iter().any(|g| g.to_lowercase().contains("f32")),
                "{name} in F32 is not an item: {got:?}"
            );
        }
    }

    /// A shared expert of a pair of the tables opens apart (the Q8_0 shared
    /// expert of a UD file beside its Q4_K, Q5_K and Q6_K stacks), with each
    /// part of the type it is read in; a pair outside them is refused.
    #[test]
    fn a_shared_expert_of_a_pair_of_the_tables_opens_apart() {
        for (gate_up, down) in [
            (GgmlType::Q4_K, GgmlType::Q5_K),
            (GgmlType::Q5_K, GgmlType::Q6_K),
        ] {
            let mut t = Synthetic::new();
            for l in 0..2 {
                for name in ["gate", "up"] {
                    t = t
                        .ty(&format!("blk.{l}.ffn_{name}_exps.weight"), gate_up)
                        .ty(&format!("blk.{l}.ffn_{name}_shexp.weight"), GgmlType::Q8_0);
                }
                t = t
                    .ty(&format!("blk.{l}.ffn_down_exps.weight"), down)
                    .ty(&format!("blk.{l}.ffn_down_shexp.weight"), GgmlType::Q8_0);
            }
            let out = check(&t.split(), CTX);
            assert!(
                out.items.is_empty(),
                "{gate_up:?} {down:?}: {:?}",
                out.items
            );
            let layout = out.checked.expect("a checked file").layout;
            for layer in &layout.layers {
                let shared = layer.ffn.shared.as_ref().expect("a shared expert");
                assert_eq!(layer.ffn.gate_up_quant.ggml(), gate_up);
                assert_eq!(layer.ffn.down_quant.ggml(), down);
                assert_eq!(
                    [shared.gate.quant, shared.up.quant, shared.down.quant],
                    [Quant::Q8_0; 3]
                );
            }
        }
        // Q4_K gate and up beside a Q6_K shared pair: a pair of no entry.
        let mut t = Synthetic::new();
        for l in 0..2 {
            t = t
                .ty(&format!("blk.{l}.ffn_gate_shexp.weight"), GgmlType::Q6_K)
                .ty(&format!("blk.{l}.ffn_up_shexp.weight"), GgmlType::Q6_K);
        }
        let got = texts(&t.split(), CTX);
        assert_eq!(
            got[0],
            "qwen35moe arena: layer 0: gate and up Q4K, shared gate and up Q6K; the pair has no \
             entry for the types"
        );
        // The shared gate and up in two types: the launch takes one.
        let t = Synthetic::new().ty("blk.0.ffn_up_shexp.weight", GgmlType::Q8_0);
        let got = texts(&t.split(), CTX);
        assert!(
            got[0].contains("one type for its stacks and one for its shared expert"),
            "{got:?}"
        );
    }

    #[test]
    fn every_site_of_a_half_the_lists_refuse_is_an_item_in_the_readers_order() {
        let t = Synthetic::new()
            .ty("blk.0.attn_qkv.weight", GgmlType::Q3_K)
            .ty("blk.0.ssm_beta.weight", GgmlType::BF16)
            .ty("blk.0.attn_gate.weight", GgmlType::Q5_0);
        let split = t.split();
        let got = texts(&split, CTX);
        assert_eq!(got[0], old_plan_error(&split));
        let names: Vec<&str> = got
            .iter()
            .map(|g| {
                g.split(" is ")
                    .next()
                    .expect("a name")
                    .rsplit(' ')
                    .next()
                    .expect("a name")
            })
            .collect();
        // The reader's order: qkv, gate, beta.
        assert_eq!(
            names,
            [
                "blk.0.attn_qkv.weight",
                "blk.0.attn_gate.weight",
                "blk.0.ssm_beta.weight"
            ],
            "{got:?}"
        );
    }

    #[test]
    fn the_shared_parts_and_the_stacks_of_an_ffn_are_listed_in_the_readers_order() {
        let t = Synthetic::new()
            .ty("blk.1.ffn_down_shexp.weight", GgmlType::Q3_K)
            .ty("blk.1.ffn_down_exps.weight", GgmlType::Q5_0)
            .ty("blk.1.ffn_gate_shexp.weight", GgmlType::Q3_K);
        let split = t.split();
        let got = texts(&split, CTX);
        assert_eq!(got[0], old_plan_error(&split));
        let tensors: Vec<&str> = got
            .iter()
            .map(|g| {
                g.split_whitespace()
                    .find(|w| w.starts_with("blk."))
                    .expect("a tensor")
            })
            .collect();
        // The routed stacks (gate, up, down), then the shared parts (gate,
        // up, down): `plan_ffn`'s order.
        assert_eq!(
            tensors,
            [
                "blk.1.ffn_down_exps.weight",
                "blk.1.ffn_gate_shexp.weight",
                "blk.1.ffn_down_shexp.weight"
            ],
            "{got:?}"
        );
    }

    #[test]
    fn a_missing_tensor_and_a_shape_the_plan_does_not_take_are_one_item_each() {
        let split = Synthetic::new().drop("blk.0.attn_gate.weight").split();
        let got = texts(&split, CTX);
        assert_eq!(
            got,
            [
                "qwen35moe session: the plan: qwen35moe session: blk.0.attn_gate.weight is not in the file"
            ]
        );
        let split = Synthetic::new()
            .dims("blk.0.attn_gate.weight", &[HIDDEN, 4000])
            .split();
        let got = texts(&split, CTX);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(
            got[0].contains(
                "blk.0.attn_gate.weight is [2048, 4000] in the file; it takes 4096 rows of 2048"
            ),
            "{got:?}"
        );
    }

    #[test]
    fn a_gate_beside_another_types_up_is_one_item() {
        let split = Synthetic::new()
            .ty("blk.0.ffn_up_exps.weight", GgmlType::Q6_K)
            .split();
        let got = texts(&split, CTX);
        assert_eq!(got, [old_plan_error(&split)]);
        assert!(
            got[0].contains(
                "layer 0: qwen35moe session: a Q4_K gate beside a Q6_K up; one gate·up launch \
                 reads one type"
            ),
            "{got:?}"
        );
    }

    #[test]
    fn the_cache_is_judged_at_the_asked_rows() {
        let split = Synthetic::new().split();
        assert_eq!(
            texts(&split, 0),
            ["qwen35moe arena: a cache of 0 rows; a session holds at least one"]
        );
        assert_eq!(
            texts(&split, 262_145),
            ["qwen35moe arena: a cache of 262145 rows; the file was trained to 262144"]
        );
    }

    #[test]
    fn the_head_and_the_embedding_are_judged_by_their_kernels() {
        let got = texts(
            &Synthetic::new()
                .ty("token_embd.weight", GgmlType::Q6_K)
                .split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: token_embd.weight is Q6K; the embedding reads Q4_K, MLX_A4, \
                 MLX_A8, MLX_A4H, MLX_A8H, Q8_0 or Q5_K"
            ]
        );
        let got = texts(
            &Synthetic::new().ty("output.weight", GgmlType::Q4_K).split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: output.weight is Q4K; the head reads Q6_K, MLX_A4, MLX_A8, \
                 MLX_A4H, MLX_A8H, Q8_0 or Q5_K"
            ]
        );
    }

    #[test]
    fn a_vocabulary_of_the_file_opens_and_zero_is_one_item() {
        // MiniCPM-V's 248,144 rows: the head's last group is partial.
        let t = Synthetic::new()
            .kv("qwen35moe.vocab_size", Value::U32(248_144))
            .dims("token_embd.weight", &[HIDDEN, 248_144])
            .dims("output.weight", &[HIDDEN, 248_144]);
        assert!(texts(&t.split(), CTX).is_empty());
        // A key of 0 beside 248,320-row tensors: the head's rule, then each
        // tensor's own.
        let t = Synthetic::new().kv("qwen35moe.vocab_size", Value::U32(0));
        assert_eq!(
            texts(&t.split(), CTX),
            [
                "qwen35moe arena: a vocabulary of 0; the head reads at least one row",
                "qwen35moe arena: token_embd.weight is [2048, 248320]; the step reads 0 rows of 2048",
                "qwen35moe arena: output.weight is [2048, 248320]; the step reads 0 rows of 2048",
            ]
        );
    }

    #[test]
    fn the_arena_rules_of_a_layer_name_the_layer() {
        // A shared expert of another type than its stacks, a pair of no entry:
        // the plan reads it apart, the arena's fold has no entry for it.
        let got = texts(
            &Synthetic::new()
                .ty("blk.0.ffn_down_shexp.weight", GgmlType::Q6_K)
                .split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: layer 0: down Q4K, shared down Q6K; the fold has no entry for \
                 the types"
            ]
        );
        // The router in a type the host does not dequantize.
        let got = texts(
            &Synthetic::new()
                .ty("blk.1.ffn_gate_inp.weight", GgmlType::Q4_K)
                .split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: layer 1: blk.1.ffn_gate_inp.weight is Q4_K; the step reads F32, \
                 MLX_A8 or MLX_A8H"
            ]
        );
        // A norm in a type the step does not read.
        let got = texts(
            &Synthetic::new()
                .ty("blk.0.attn_norm.weight", GgmlType::F16)
                .split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: layer 0: blk.0.attn_norm.weight is F16 [2048]; the step reads 2048 F32 values"
            ]
        );
    }

    #[test]
    fn two_faults_give_both_the_plans_before_the_arenas() {
        let t = Synthetic::new()
            .ty("output.weight", GgmlType::Q4_K)
            .ty("blk.0.attn_qkv.weight", GgmlType::Q3_K);
        let split = t.split();
        let got = texts(&split, CTX);
        assert_eq!(got.len(), 2, "{got:?}");
        assert_eq!(got[0], old_plan_error(&split));
        assert!(
            got[1].contains("output.weight is Q4K; the head reads"),
            "{got:?}"
        );
    }

    #[test]
    fn a_refused_layer_does_not_hide_the_next_and_the_first_is_the_opens() {
        // Layer 0's FFN, then layer 1's mixer: the order `layer_plans` reads.
        let t = Synthetic::new()
            .ty("blk.1.attn_q.weight", GgmlType::Q3_K)
            .ty("blk.0.ffn_down_exps.weight", GgmlType::Q3_K);
        let split = t.split();
        let out = check(&split, CTX);
        let layers: Vec<Option<usize>> = out.items.iter().map(|i| i.layer).collect();
        assert_eq!(layers, [Some(0), Some(1)], "{:?}", out.items);
        assert_eq!(out.items[0].error.to_string(), old_plan_error(&split));
        assert!(out.checked.is_none());
    }

    #[test]
    fn a_refused_file_is_not_a_file_of_no_layer() {
        // Every layer refused sets none of the facts the layers share; the
        // plan does not add a refusal of its own for that.
        let t = Synthetic::new()
            .ty("blk.0.attn_qkv.weight", GgmlType::Q3_K)
            .ty("blk.1.attn_q.weight", GgmlType::Q3_K);
        let got = texts(&t.split(), CTX);
        assert!(
            got.iter()
                .all(|g| !g.contains("no layer") && !g.contains("no attention layer")),
            "{got:?}"
        );
    }

    /// A file of `bytes`, named for `tag`, removed when the test ends.
    struct TempFile(PathBuf);

    impl TempFile {
        fn new(tag: &str, bytes: &[u8]) -> TempFile {
            let path = std::env::temp_dir().join(format!(
                "bloomery-metal-check-{tag}-{}.gguf",
                std::process::id()
            ));
            std::fs::write(&path, bytes).expect("the temp file is written");
            TempFile(path)
        }
    }

    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn a_file_cut_after_its_tensor_table_is_read_as_its_header() {
        let t = Synthetic::new();
        let layout = t.layout();
        let file = TempFile::new("cut", layout.header_bytes());
        let header = super::super::plan::open_header(&file.0).expect("the header reads");
        assert_eq!(
            header.data,
            super::super::plan::Data::Cut {
                file_len: layout.data_base(),
                need: layout.file_len(),
            }
        );
        assert_eq!(header.split().tensor_count(), t.tensors.len());
        let out = check(header.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        // The open of the same file stays a named error.
        let e = Split::open(&file.0).err().expect("the open refuses");
        assert!(e.to_string().contains("exceeds the file"), "{e}");
    }

    #[test]
    fn a_header_the_plan_cannot_read_is_a_named_error() {
        let file = TempFile::new("junk", b"this is not a gguf file at all");
        let e = super::super::plan::open_header(&file.0)
            .err()
            .expect("refused");
        assert!(e.contains("not a GGUF file"), "{e}");
        // A header cut inside its own tensor table is not a plan either.
        let layout = Synthetic::new().layout();
        let file = TempFile::new("short", &layout.header_bytes()[..1500]);
        let e = super::super::plan::open_header(&file.0)
            .err()
            .expect("refused");
        assert!(e.contains("truncated file"), "{e}");
    }

    #[test]
    fn the_widest_lists_hold_every_family_the_reader_knows() {
        for family in SiteTy::ALL
            .iter()
            .copied()
            .chain(chain::site::BACKEND_SITES.iter().map(|r| r.1))
        {
            assert!(ANY.contains(&family), "{family:?} is in no widest list");
        }
    }

    /// An LFM3-1.7B file opens the instance made at open for its key, at the
    /// attention row of `LFM3_ATTN`, with a layout of each of its kinds: thirteen short convs of seven taps over a ring of fourteen
    /// slots, eleven attention layers whose P1 holds the output gate as the
    /// fourth segment, a window on the sliding layers only, two rope tables (the
    /// full layers' θ with the file's factors, the sliding layers' without),
    /// a sandwich on every layer, and a head that is the embedding again.
    /// Mutant: `conv_checks` or `rope_checks` or `post_checks` is a refusal, or
    /// the gate is left out of the P1's parts.
    #[test]
    fn an_lfm3_file_opens_an_instance_made_at_open_with_a_layout_of_each_kind() {
        let inst = lfm3_instance();
        let out = check(&Synthetic::of(inst).split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let summary = out.spec.clone().expect("the description reads");
        assert_eq!(
            (
                summary.layers,
                summary.delta,
                summary.conv,
                summary.attention
            ),
            (24, 0, 13, 11)
        );
        assert_eq!((summary.experts, summary.picks), (0, 0));
        let checked = out.checked.expect("a checked file");
        assert_eq!(*checked.instance, *inst);
        assert_eq!(inst.name, instance::UNREGISTERED);
        assert_eq!(inst.key.attn, instance::LFM3_ATTN.key.attn);
        assert_eq!(checked.geom, Geometry::of(inst), "the geometry");
        let layout = &checked.layout;
        let table = |i: usize| {
            let r = &layout.ropes[i];
            (r.base, r.factors.as_ref().map(|v| (v.name.as_str(), v.len)))
        };
        assert_eq!(layout.ropes.len(), 2);
        assert_eq!(table(0), (1e6, Some(("rope_freqs.weight", 64))));
        assert_eq!(table(1), (1e4, None));
        for (l, (kind, layer)) in LFM3_KINDS.chars().zip(&layout.layers).enumerate() {
            match (kind, &layer.mixer) {
                ('c', MixerLayout::ShortConv(c)) => {
                    assert_eq!(
                        (c.taps, c.ring_rows, c.ring),
                        (7, 14, 14 * HIDDEN as usize),
                        "layer {l}"
                    );
                    assert_eq!(c.p1.dispatches.len(), 1, "layer {l}");
                    assert_eq!(c.p1.dispatches[0].members, [0], "layer {l}");
                    assert_eq!(c.p1.range(0), 0..3 * HIDDEN as usize, "layer {l}");
                }
                ('F' | 'S', MixerLayout::Attention(a)) => {
                    let want = if kind == 'F' {
                        (None, 0)
                    } else {
                        (Some(LFM3_WINDOW), 1)
                    };
                    assert_eq!((a.window, a.rope), want, "layer {l}");
                    assert!(a.gate, "layer {l}: the head gate");
                    assert_eq!(a.p1.dispatches.len(), 1, "layer {l}");
                    assert_eq!(a.p1.dispatches[0].members, [0, 1, 2, 3], "layer {l}");
                    assert_eq!(a.p1.range(3).len(), 16, "layer {l}: a row a query head");
                }
                _ => panic!("layer {l} is not a {kind}"),
            }
            assert!(layer.post.is_some(), "layer {l}: a sandwich");
        }
        assert_eq!(layout.head.name, "token_embd.weight");
        assert_eq!(layout.embed.name, "token_embd.weight");
    }

    /// `Checked::bytes` of an LFM3-1.7B file, class by class: thirteen conv
    /// rings of 114,688 B in the state, the eleven attention layers' caches,
    /// two rope tables, a conv output, a join buffer and a hand-on buffer
    /// among the arena's buffers, the gate rows beside the pass's, and the
    /// embedding twice in the weights (the tied head is its second buffer).
    /// Mutant: the head or the ring count, a rope table, `conv_y`, `join`,
    /// `xn` or the gate rows is left out of `Checked::bytes` or of `RowsLens`.
    #[test]
    fn the_bytes_of_an_lfm3_file_pin_the_rings_the_tables_the_gate_rows_and_the_head() {
        use super::super::arena::RECORD_SLOTS;

        let inst = lfm3_instance();
        let g = Geometry::of(inst);
        let t = Synthetic::of(inst);
        let checked = check(&t.split(), CTX).checked.expect("a checked file");
        let bytes = checked.bytes().expect("the bytes");
        let f32s = |n: usize| count(n) * 4;
        let (hidden, vocab) = (HIDDEN as usize, LFM3_VOCAB as usize);

        // The tensors but the factors, and the embedding again: the head. The
        // buffers are the head's three, a conv layer's ten (P1, its norm, out,
        // filters, two gains, gate, up, down, the FFN's norm) and an attention
        // layer's eleven (P1, its norm, out, two head norms, the same six).
        assert_eq!(bytes.weights.requested, t.weight_bytes());
        assert_eq!(bytes.weights.buffers, 3 + 13 * 10 + 11 * 11);

        // The K and the V cache of each attention layer, f16.
        let cache = count(g.attn.kv_heads * CTX * g.attn.head) * 2;
        assert_eq!(bytes.kv, class(&[cache; 22]));

        // A conv layer's ring: `rows` rows of the hidden width, f32.
        let ring = f32s(Ring::new(7).values(hidden));
        assert_eq!(ring, 114_688);
        assert_eq!(bytes.state, class(&[ring; 13]));

        // Each layer's fixed route (one id, one weight), the two rope tables,
        // the conv's output and the join buffer, then the token arena's
        // buffers in field order.
        let proj = 3 * hidden;
        let attn_out = g.attn.out_len().expect("the attention rows fit");
        let attn_partials = g
            .attn
            .partials_len(segments_for_keys(CTX as u32) as usize)
            .expect("the partials fit");
        let mut scratch: Vec<u64> = vec![f32s(1); 2 * 24];
        scratch.extend([CTX * g.attn.rot; 2].map(f32s));
        scratch.extend([hidden; 2].map(f32s));
        // PIN(2026-10-29): the hand-on buffer (`Arena::xn`), `hidden` values
        // of the joined stream times the reader's gain and one scale word.
        scratch.push(f32s(hidden + 1));
        scratch.extend(
            [
                hidden,                                  // x
                proj,                                    // proj
                g.delta.value_heads * g.delta.head_cols, // g
                g.delta.partials_stride,                 // delta_partials
                attn_out,                                // attn
                attn_partials,                           // attn_partials
                g.attn.kv_heads,                         // attn_ticket
                g.experts.slots * g.experts.k,           // h
                vocab,                                   // logits
                RECORD_SLOTS,                            // step_id
                1,                                       // id_in
                LANES,                                   // lane
            ]
            .map(f32s),
        );
        scratch.extend([f32s(FAULT_WORDS); RECORD_SLOTS]); // faults
        assert_eq!(bytes.scratch, class(&scratch));

        // The rows arena of a pass of `cap` rows: ids, residual, normed input,
        // P1 output, mixer rows, delta partials and prepared rows, a dense
        // FFN's gate and up rows, and the output gate, a row of the query heads.
        let cap = pass_rows_max(&g).min(CTX);
        let mix = (g.delta.value_heads * g.delta.head_cols)
            .max(g.attn.query_heads * g.attn.head)
            .max(hidden);
        let mut rows = [
            cap,
            cap * hidden,
            cap * hidden,
            cap * proj.next_multiple_of(4),
            cap * mix,
            cap * g.delta.partials_stride,
            cap,
            cap * g.experts.k,
            cap * g.experts.k,
            cap * g.attn.query_heads,
        ]
        .map(f32s)
        .to_vec();
        // PIN(2026-10-29): the half rows the dense GEMMs read
        // (`RowsArena::xh`), `cap` rows of the wider of the mixer and the FFN's
        // activated row, two bytes a value.
        rows.push(count(cap * mix.max(g.experts.k)) * 2);
        assert_eq!(bytes.rows, class(&rows));
    }

    /// A file with no `output.weight` reads its head from the token embedding;
    /// a file with its own head reads that; a file with neither refuses the
    /// embedding and the head each by name; and a tied embedding the head
    /// cannot read is refused in the head's words. Mutant: `head_tensor`
    /// answers the embedding without finding it, or never answers it.
    #[test]
    fn a_head_with_no_output_weight_is_the_embedding_again() {
        let inst = lfm3_instance();
        let split = Synthetic::of(inst).split();
        let checked = check(&split, CTX).checked.expect("a tied file opens");
        assert_eq!(checked.layout.head.name, "token_embd.weight");
        assert_eq!(checked.layout.head.quant, Quant::Q5K);

        let mut own = Synthetic::of(inst);
        own.add("output.weight", &[HIDDEN, LFM3_VOCAB], GgmlType::Q6_K);
        let checked = check(&own.split(), CTX)
            .checked
            .expect("an untied file opens");
        assert_eq!(checked.layout.head.name, "output.weight");
        assert_eq!(checked.layout.embed.name, "token_embd.weight");

        // Neither: the description reads the intact file, the tables are judged
        // on the file with the embedding dropped.
        let spec = model::arch::spec(&split)
            .expect("the description reads")
            .spec;
        let bare = Synthetic::of(inst).drop("token_embd.weight").split();
        let geom = Geometry::of(inst);
        let mut items = Vec::new();
        let got = head_checks(
            &mut At {
                items: &mut items,
                layer: None,
            },
            &Table {
                split: &bare,
                geom: &geom,
            },
            &spec,
            CTX,
        );
        assert!(got.is_none());
        let said: Vec<String> = items.iter().map(|i| i.error.to_string()).collect();
        assert_eq!(
            said,
            [
                "qwen35moe arena: the file has no tensor token_embd.weight",
                "qwen35moe arena: the file has no tensor output.weight"
            ]
        );

        let got = texts(
            &Synthetic::of(inst)
                .ty("token_embd.weight", GgmlType::Q4_K)
                .split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: token_embd.weight is Q4K; the head reads Q6_K, MLX_A4, MLX_A8, \
                 MLX_A4H, MLX_A8H, Q8_0 or Q5_K"
            ]
        );
    }

    /// Layer `l` of `t` as the layer check meets it with its residual replaced
    /// by a sandwich of the file's own `attn_norm` (F32, `hidden` values): its
    /// items, and whether it has a layout.
    fn sandwiched(t: &Synthetic, l: usize) -> (Vec<String>, bool) {
        let split = t.split();
        let spec = model::arch::spec(&split)
            .expect("the description reads")
            .spec;
        let kinds = kinds_of(&spec.layers).expect("the kinds read");
        let pd = PlanDims::of(&spec, WHAT).expect("the dims read");
        let mut plans = layer_plans(
            &split,
            WHAT,
            &pd,
            &kinds,
            &spec.layers,
            &SITES,
            Read35::Whole,
        )
        .expect("the plans read");
        let norm = format!("blk.{l}.attn_norm.weight");
        plans[l].residual = ResidualPlan::Sandwich(SandwichPlan {
            eps: 1e-5,
            add: [1.0, 1.0],
            norms: [norm.clone(), norm],
        });
        let inst = instance::select(&spec).expect("the instance");
        let geom = Geometry::of(inst);
        let table = Table {
            split: &split,
            geom: &geom,
        };
        let mut items = Vec::new();
        let layout = layer_checks(
            &mut At {
                items: &mut items,
                layer: Some(l),
            },
            &table,
            &plans[l],
            &spec.layers[l],
            spec.hidden as usize,
            CTX,
            &mut SharedShapes::default(),
        );
        (
            items.iter().map(|i| i.error.to_string()).collect(),
            layout.is_some(),
        )
    }

    /// A sandwich is run after an attention or a short conv layer's out
    /// projection and after a dense FFN's down; on a gated delta layer, or over
    /// a routed FFN, it is refused by name at the layer. Mutant: either refusal
    /// is deleted from `post_checks`, or the sandwich is refused whatever the
    /// layer.
    #[test]
    fn a_sandwich_is_refused_by_name_on_a_delta_layer_and_over_a_routed_ffn() {
        let mut dense_inst = instance::QWEN35MOE_A3B;
        dense_inst.key.ffn = FfnAxes::Dense { ff: 512 };
        let dense = Synthetic::of(&dense_inst);
        // Layer 1 is attention over a dense FFN: the sandwich runs.
        assert_eq!(sandwiched(&dense, 1), (Vec::new(), true));
        // Layer 0 is a delta layer.
        assert_eq!(
            sandwiched(&dense, 0),
            (
                vec![
                    "qwen35moe arena: layer 0: a sandwich residual on a gated delta layer; the \
                     delta step's out projection adds into the stream"
                        .to_owned()
                ],
                false
            )
        );
        // Qwen3.6's layer 1 is attention over a routed FFN; its layer 0 is both.
        let routed = Synthetic::new();
        let fold = "qwen35moe arena: layer 1: a sandwich residual over a routed FFN; the routed \
                    fold adds into the stream";
        assert_eq!(sandwiched(&routed, 1), (vec![fold.to_owned()], false));
        let (items, laid_out) = sandwiched(&routed, 0);
        assert_eq!(items.len(), 2, "{items:?}");
        assert!(!laid_out && items[0].contains("a gated delta layer"));
        assert!(items[1].ends_with("over a routed FFN; the routed fold adds into the stream"));
    }

    /// A rope table is one for each base and factors the layers ask for, and
    /// the factors are read as F32, a value for each pair of the values the
    /// rope turns: a table of another length or type is refused by name at
    /// the layer that asks, and makes no table. Mutant: `rope_checks` reads
    /// the factors' tensor without `f32s`, or matches a table by its base
    /// alone.
    #[test]
    fn a_rope_table_is_one_a_base_and_factors_and_a_factor_table_is_held_to_f32_pairs() {
        let inst = lfm3_instance();
        let geom = Geometry::of(inst);
        for (case, t, want) in [
            (
                "length",
                Synthetic::of(inst).dims("rope_freqs.weight", &[31]),
                "rope_freqs.weight is F32 [31]; the step reads 64 F32 values",
            ),
            (
                "type",
                Synthetic::of(inst).ty("rope_freqs.weight", GgmlType::F16),
                "rope_freqs.weight is F16 [64]; the step reads 64 F32 values",
            ),
        ] {
            let split = t.split();
            let mut items = Vec::new();
            let mut shared = SharedShapes::default();
            let got = rope_checks(
                &mut At {
                    items: &mut items,
                    layer: Some(2),
                },
                &Table {
                    split: &split,
                    geom: &geom,
                },
                &mut shared,
                1e6,
                Some("rope_freqs.weight"),
            );
            assert_eq!(got, None, "{case}");
            assert!(shared.ropes.is_empty(), "{case}: no table was made");
            let said: Vec<String> = items.iter().map(|i| i.error.to_string()).collect();
            assert_eq!(
                said,
                [format!("qwen35moe arena: layer 2: {want}")],
                "{case}"
            );
        }

        let split = Synthetic::of(inst).split();
        let table = Table {
            split: &split,
            geom: &geom,
        };
        let mut shared = SharedShapes::default();
        let mut items = Vec::new();
        let mut place = |base: f32, freqs: Option<&str>| {
            rope_checks(
                &mut At {
                    items: &mut items,
                    layer: Some(2),
                },
                &table,
                &mut shared,
                base,
                freqs,
            )
        };
        assert_eq!(place(1e6, Some("rope_freqs.weight")), Some(0));
        assert_eq!(place(1e4, None), Some(1));
        assert_eq!(place(1e6, Some("rope_freqs.weight")), Some(0), "one table");
        assert_eq!(place(1e6, None), Some(2), "the factors are the table's own");
        assert_eq!(place(1e4, None), Some(1));
        assert!(items.is_empty(), "{items:?}");
    }

    /// The plan's conv ring is held to the ring the layer's taps make: a plan
    /// and a description that disagree on the conv are refused by name.
    /// Mutant: `conv_checks` does not compare the ring.
    #[test]
    fn a_conv_plan_ring_that_differs_from_the_layers_is_refused_by_name() {
        let inst = lfm3_instance();
        let split = Synthetic::of(inst).split();
        let geom = Geometry::of(inst);
        let spec = model::arch::spec(&split)
            .expect("the description reads")
            .spec;
        let pd = PlanDims::of(&spec, WHAT).expect("the dims read");
        let mut plan =
            plan_shortconv(&split, WHAT, &pd, Ring::new(7), 0, &SITES).expect("the plan reads");
        let table = Table {
            split: &split,
            geom: &geom,
        };
        let run = |plan: &ShortConvPlan| {
            let mut items = Vec::new();
            let got = conv_checks(
                &mut At {
                    items: &mut items,
                    layer: Some(0),
                },
                &table,
                plan,
                7,
                HIDDEN as usize,
            );
            let said: Vec<String> = items.iter().map(|i| i.error.to_string()).collect();
            (got.is_some(), said)
        };
        assert_eq!(run(&plan), (true, Vec::new()));
        plan.ring = Ring::new(5);
        assert_eq!(
            run(&plan),
            (
                false,
                vec![
                    "qwen35moe arena: layer 0: the plan's conv ring is 5 taps; the layer's conv \
                     has 7"
                        .to_owned()
                ]
            )
        );
    }

    /// A description whose residual is neither the plain add nor the sandwich
    /// is refused by name at its layer, from the description alone, and the
    /// check returns none. A layer that is not a sandwich reads one pre-FFN
    /// norm, so the file's two norms at the layer refuse its FFN half beside
    /// the residual's own refusal: both are at the layer, the residual's once.
    /// Mutant: chain's `plan_residual` plans `Residual::Hc` or
    /// `Residual::AlphaBeta` as the plain add.
    #[test]
    fn a_residual_the_session_does_not_run_is_refused_by_name_at_check() {
        let inst = lfm3_instance();
        let split = Synthetic::of(inst).split();
        for residual in [Residual::Hc, Residual::AlphaBeta { eps: 1e-5 }] {
            let mut spec = model::arch::spec(&split)
                .expect("the description reads")
                .spec;
            spec.layers[3].residual = residual;
            let mut items = Vec::new();
            let mut summary = None;
            let checked = read_spec(
                &split,
                &spec,
                "synthetic.gguf",
                CTX,
                &mut items,
                &mut summary,
            );
            assert!(checked.is_none(), "{residual:?}");
            let said: Vec<String> = items.iter().map(|i| i.error.to_string()).collect();
            assert!(
                items.iter().all(|i| i.layer == Some(3)),
                "{residual:?}: {said:?}"
            );
            let refused = said
                .iter()
                .filter(|s| s.contains("layer 3: the residual") && s.contains("no walk here runs"))
                .count();
            assert_eq!(refused, 1, "{residual:?}: {said:?}");
            assert!(
                said.iter().any(|s| s.contains("the pre-FFN norm is one")),
                "{residual:?}: {said:?}"
            );
        }
    }
}
