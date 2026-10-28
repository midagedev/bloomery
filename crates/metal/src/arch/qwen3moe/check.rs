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
    DeltaPlan, FfnRoute, GqaPlan, Kind35, LayerPlan, MixerPlan, SharedPlan, SiteTy, kinds_of,
};
use chain::arch::qwen3moe::read::{
    PlanDims, Read35, Sites35, file_names, plan_delta, plan_ffn, plan_gqa, plan_residual,
    plan_shortconv,
};
use chain::linear::LinearShape;
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
use crate::kernels::{HeadMap, MAX_SEGMENTS, Quant, head_groups, segments_for_keys, shared_entry};

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
];

/// [`ANY`] as the reader's five lists.
const WIDEST: Sites35 = Sites35 {
    proj: ANY,
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
    /// same values, which the open dequantizes exactly: the router and the
    /// shared-expert gate never run a kernel, so no 8-bit path exists on the
    /// card.
    fn f32s_or_a8(&self, name: &str, len: usize) -> Result<Values, SessionError> {
        let (ty, dims) = self.info(name)?;
        let n: u64 = dims.iter().product();
        if usize::try_from(n).ok() != Some(len) {
            return Err(refuse(format!(
                "{name} is {ty:?} {dims:?}; the step reads {len} F32 values"
            )));
        }
        match ty {
            GgmlType::F32 | GgmlType::MLX_A8 => Ok(Values::new(name, len, ty)),
            other => Err(refuse(format!(
                "{name} is {other:?}; the step reads F32 or MLX_A8"
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
}

pub(super) enum MixerLayout {
    Delta(DeltaLayout),
    Attention(AttnLayout),
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
    pub(super) rope: RopeTable,
    pub(super) embed: Site,
    pub(super) head: Site,
    pub(super) head_norm: Values,
    pub(super) layers: Vec<LayerLayout>,
    /// The elements of the rope table, the attention's partials and output.
    pub(super) rope_len: usize,
    pub(super) attn_partials: usize,
    pub(super) attn_out: usize,
    /// The widest P1 output a layer writes.
    pub(super) proj: usize,
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
        MixerLayout::Attention(_) => None,
    })
}

/// The facts the layers must share, each set by the first layer that has one
/// and held by every later layer: the rope table's base (one table serves
/// one) and the MoE's experts-and-picks shape.
#[derive(Default)]
struct SharedShapes {
    rope_base: Option<f32>,
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
/// arm of the chain's `layer_plans` loop.
fn plan_mixer(
    split: &Split,
    pd: &PlanDims,
    kind: Kind35,
    l: usize,
    sites: &Sites35,
) -> Result<MixerPlan, ChainError> {
    Ok(match kind {
        Kind35::Gqa(kind, flash) => {
            MixerPlan::Gqa(plan_gqa(split, WHAT, pd, l, (kind, flash), sites)?)
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
fn mixer_sweep(split: &Split, pd: &PlanDims, kind: Kind35, l: usize) -> Vec<SessionError> {
    match plan_mixer(split, pd, kind, l, &WIDEST) {
        Ok(MixerPlan::Delta(d)) => refused_sites(
            split,
            [d.qkv, d.gate, d.beta, d.alpha, d.ssm_out],
            PROJECTED,
        ),
        Ok(MixerPlan::Gqa(g)) => refused_sites(
            split,
            [g.attn_q, g.attn_k, g.attn_v, g.attn_output],
            PROJECTED,
        ),
        Ok(MixerPlan::ShortConv(_)) | Err(_) => Vec::new(),
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
    let delta = spec
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, SpecMixer::DeltaRule(_)))
        .count();
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
        delta,
        attention: spec.layers.len() - delta,
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
    *summary = Some(summarize(split, &spec));
    // The library a session of the file runs on: the instance its shape
    // selects (a registered one, or one made at open), a geometry no kernel
    // takes refused by name. The arena's rules below are judged at the
    // instance's launch geometry, so a shape with no instance has none of
    // them: its plans are read, and no more.
    let inst = At {
        items: &mut *items,
        layer: None,
    }
    .ok(instance::select(&spec).map_err(|e| super::session::refuse(format!("{path}: {e}"))));
    let geom = inst.map(Geometry::of);
    let (kinds, pd) = match kinds_of(&spec.layers)
        .and_then(|kinds| PlanDims::of(&spec, WHAT).map(|pd| (kinds, pd)))
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
        let mixer = read_half(items, l, plan_mixer(split, &pd, *kind, l, &SITES), || {
            mixer_sweep(split, &pd, *kind, l)
        });
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
            &spec,
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
    let rope = clean
        .then(|| {
            file.ok(shared
                .rope_base
                .ok_or_else(|| refuse("no attention layer to read the rope from")))
        })
        .flatten()
        .and_then(|base| {
            let rot = geom?.attn.rot;
            file.ok(RopeTable::new(&RopeSpec::window(base, rot))
                .map_err(|e| refuse(format!("the rope table: {e}"))))
        });
    let layers: Option<Vec<LayerLayout>> = layouts.into_iter().collect();
    let proj = layers.as_ref().and_then(|layers| {
        file.ok(layers
            .iter()
            .flat_map(|l| match &l.mixer {
                MixerLayout::Delta(d) => &d.p1.dispatches,
                MixerLayout::Attention(a) => &a.p1.dispatches,
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
    let (inst, geom, head, (n_experts, top_k), rope, layers, proj, attn_out, plans) = (
        inst?, geom?, head?, moe?, rope?, layers?, proj?, attn_out?, plans?,
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
            rope,
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
    let head = at.ok(t.site(&names::output(), hidden, vocab)).filter(|s| {
        let fits = s.quant.heads();
        if !fits {
            at.push(refuse(format!(
                "{} is {:?}; the head reads {}",
                names::output(),
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
) -> Option<AttnLayout> {
    let a = &t.geom.attn;
    let q_rows = at.ok(a
        .qg_len()
        .ok_or_else(|| refuse("the query rows do not fit")));
    let kv_rows = a.kv_heads * a.head;
    let p1 = q_rows.and_then(|q_rows| {
        p1_checks(
            at,
            t,
            &g.attn_norm,
            &[
                (&g.attn_q, q_rows),
                (&g.attn_k, kv_rows),
                (&g.attn_v, kv_rows),
            ],
            hidden,
        )
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
    })
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
            match shared.rope_base {
                None => shared.rope_base = Some(spec_g.rope.base),
                Some(b) if b == spec_g.rope.base => {}
                Some(b) => {
                    at.push(refuse(format!(
                        "a rope base of {}, another layer's is {b}; one table serves one",
                        spec_g.rope.base
                    )));
                    ok = false;
                }
            }
            let attn = attention_checks(at, t, g, hidden, ctx);
            attn.filter(|_| ok).map(MixerLayout::Attention)
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
    ok.then_some(LayerLayout {
        mixer: mixer?,
        ffn: ffn?,
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
        for words in [
            g.rope_len,
            g.hidden,
            g.proj,
            geom.delta.value_heads * geom.delta.head_cols,
            geom.delta.partials_stride,
            g.attn_out,
            g.attn_partials,
            1,
            geom.experts.slots * geom.experts.k,
            g.vocab,
            1,
            super::arena::RECORD_SLOTS,
            1,
            LANES,
        ] {
            b.scratch.add(f32s(words));
        }
        b.scratch.add(count(head_groups(g.vocab)) * 8);
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

    use chain::arch::qwen3moe::read::layer_plans;
    use gguf::Value;
    use gguf::write::{Layout as FileLayout, TensorDecl};

    use super::*;
    use crate::instance::{FfnAxes, MoeAxes};

    /// A file's header and tensor table, in memory, at a registered row's
    /// shapes: two layers (the Qwen3.6 row's a delta layer, then an attention
    /// one: `full_attention_interval` 2), every weight Q4_K but the head's
    /// Q6_K, and no weight byte anywhere. A test edits it and reads it back as
    /// a `Split`.
    struct Synthetic {
        kvs: Vec<(String, Value)>,
        tensors: Vec<(String, Vec<u64>, GgmlType)>,
    }

    /// The Qwen3.6 row's residual width, for the tests that pin its words.
    const HIDDEN: u64 = 2048;
    const VOCAB: u64 = 248_320;
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
                t.tensor_bytes(),
                "{name}: the weights"
            );
            assert!(bytes.total().paged >= bytes.total().requested, "{name}");

            let head = inst.key.attn.row.head;
            let wrong = u64::from(head) + 32;
            let t = t.dims("blk.1.attn_q_norm.weight", &[wrong]);
            let got = texts(&t.split(), CTX);
            assert_eq!(
                got,
                [format!(
                    "qwen35moe arena: layer 1: blk.1.attn_q_norm.weight is F32 [{wrong}]; the step \
                     reads {head} F32 values"
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
                1,                             // attn_ticket
                CTX * g.attn.rot,              // rope
                g.experts.slots * g.experts.k, // h
                vocab,                         // logits
            ]
            .map(f32s),
        );
        scratch.push(count(head_groups(vocab)) * 8); // candidates
        scratch.extend([1, RECORD_SLOTS, 1, LANES].map(f32s)); // head_ticket, step_id, id_in, lane
        scratch.extend([f32s(FAULT_WORDS); RECORD_SLOTS]); // faults
        assert_eq!(bytes.scratch, class(&scratch));

        // The rows arena of a pass of `cap` rows: ids, residual, normed input,
        // P1 output, mixer rows, delta partials and prepared rows, logits, four
        // slot lists, the route table, the pair's and the down's outputs.
        let cap = pass_rows_max(&g).min(CTX);
        let slots = cap * g.experts.slots;
        let prepared = g.delta.prepared_elems(N_K).expect("the prepared row fits");
        let rows = [
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
        .map(f32s);
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
                1,                             // attn_ticket
                CTX * g.attn.rot,              // rope
                g.experts.slots * g.experts.k, // h
                vocab,                         // logits
            ]
            .map(f32s),
        );
        scratch.push(count(head_groups(vocab)) * 8); // candidates
        scratch.extend([1, RECORD_SLOTS, 1, LANES].map(f32s)); // head_ticket, step_id, id_in, lane
        scratch.extend([f32s(FAULT_WORDS); RECORD_SLOTS]); // faults
        assert_eq!(bytes.scratch, class(&scratch));

        let cap = pass_rows_max(&g).min(CTX);
        let prepared = g.delta.prepared_elems(N_K).expect("the prepared row fits");
        let rows = [
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
        .map(f32s);
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
             file; this site launches Q4_K, Q6_K, MLX_A4, Q8_0, Q5_K, F32"
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
                 Q8_0 or Q5_K"
            ]
        );
        let got = texts(
            &Synthetic::new().ty("output.weight", GgmlType::Q4_K).split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: output.weight is Q4K; the head reads Q6_K, MLX_A4, Q8_0 or \
                 Q5_K"
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
                "qwen35moe arena: layer 1: blk.1.ffn_gate_inp.weight is Q4_K; the step reads F32 or MLX_A8"
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
}
