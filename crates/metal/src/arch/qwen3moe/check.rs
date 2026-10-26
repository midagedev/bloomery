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
//! `Q8_0` sites are all listed and the first item is the one the open names.

use bloomery_device::fault::FAULT_WORDS;
use chain::ChainError;
use chain::arch::qwen3moe::plan::{
    DeltaPlan, FfnRoute, GqaPlan, Kind35, LayerPlan, MixerPlan, SharedPlan, SiteTy, kinds35,
};
use chain::arch::qwen3moe::read::{
    PlanDims, Read35, Sites35, file_names, plan_delta, plan_ffn, plan_gqa,
};
use chain::linear::LinearShape;
use chain::rope::{RopeSpec, RopeTable};
use chain::site::file_site;
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::{Ffn, LayerSpec, Mixer as SpecMixer, ModelSpec};
use model::arch::names;

use super::arena::LANES;
use super::rows::RowsLens;
use super::session::{LAUNCHED, SITES, SessionError, WHAT, pass_rows_max};
use crate::kernels::{
    CONV_TAPS, EXPERT_K, EXPERT_SLOTS, HEAD_COLS, HEAD_DIM, HEAD_K, HeadMap, KV_HEADS,
    MAX_SEGMENTS, PARTIALS_PER_HEAD, Quant, ROT, ROUTED_IDS, VALUE_HEADS, channels, f16_len,
    head_groups, out_len, partials_len, qg_len, ring_elems, router, segments_for_keys, state_elems,
};

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
/// arena's rules.
struct Table<'a> {
    split: &'a Split,
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
    /// counted as rows, in a type a mat-vec runs.
    fn site(&self, name: &str, cols: usize, rows: usize) -> Result<Site, SessionError> {
        let (ty, dims) = self.info(name)?;
        let got_rows: u64 = dims.iter().skip(1).product();
        let shape_ok = dims.first().and_then(|&c| usize::try_from(c).ok()) == Some(cols)
            && usize::try_from(got_rows).ok() == Some(rows);
        if !shape_ok {
            return Err(refuse(format!(
                "{name} is {dims:?}; the step reads {rows} rows of {cols}"
            )));
        }
        let quant = Quant::of_ggml(ty).ok_or_else(|| {
            refuse(format!(
                "{name} is {ty:?}, which no mat-vec of this backend runs (Q4_K, Q6_K, MLX_A4)"
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
/// shared expert's apart from the routed stacks.
pub(super) struct FfnLayout {
    pub(super) norm: Values,
    pub(super) gate_inp: Values,
    pub(super) gate_inp_shexp: Values,
    pub(super) gate_up_quant: Quant,
    pub(super) gate: Site,
    pub(super) up: Site,
    pub(super) shexp_gate: Site,
    pub(super) shexp_up: Site,
    pub(super) down_quant: Quant,
    pub(super) down: Site,
    pub(super) shexp_down: Site,
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

/// A file the open runs: the plans of its layers and the arena's layout.
pub struct Checked {
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

fn count(n: usize) -> u64 {
    u64::try_from(n).expect("a count fits u64")
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
        Ok(MixerPlan::Delta(d)) => {
            refused_sites(split, [d.qkv, d.gate, d.beta, d.alpha, d.ssm_out])
        }
        Ok(MixerPlan::Gqa(g)) => {
            refused_sites(split, [g.attn_q, g.attn_k, g.attn_v, g.attn_output])
        }
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
            refused_sites(split, [n.gate, n.sh_gate, n.up, n.sh_up, n.down, n.sh_down])
        }
        Ffn::Dense { .. } => match plan_ffn(split, WHAT, pd, layer, l, &WIDEST, Read35::Whole) {
            Ok(f) => refused_sites(split, [f.gate, f.up, f.down]),
            Err(_) => Vec::new(),
        },
    }
}

/// The sites among `names` that the chain's own site check refuses at the
/// launch list: its words, so the item is the one a strict read returns for
/// the site. A site is judged by the shape the header gives it, so only its
/// type and presence are at stake.
fn refused_sites(split: &Split, names: impl IntoIterator<Item = String>) -> Vec<SessionError> {
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
            file_site(split, WHAT, &name, shape, LAUNCHED)
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
    let spec = match model::arch::qwen35moe::spec::read(split) {
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
    let (kinds, pd) = match kinds35(&spec.layers)
        .and_then(|kinds| PlanDims::q35(&spec, WHAT).map(|pd| (kinds, pd)))
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
        plans.push(mixer.zip(ffn).map(|(mixer, ffn)| LayerPlan { mixer, ffn }));
    }
    let table = Table { split };
    let head = head_checks(
        &mut At {
            items: &mut *items,
            layer: None,
        },
        &table,
        &spec,
        ctx,
    );
    let mut shared = SharedShapes::default();
    let hidden = spec.hidden as usize;
    let mut layouts: Vec<Option<LayerLayout>> = Vec::with_capacity(plans.len());
    for (l, (plan, layer)) in plans.iter().zip(&spec.layers).enumerate() {
        layouts.push(plan.as_ref().and_then(|plan| {
            layer_checks(
                &mut At {
                    items: &mut *items,
                    layer: Some(l),
                },
                &table,
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
            file.ok(RopeTable::new(&RopeSpec::window(base, ROT))
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
    let attn_out = file.ok(out_len().ok_or_else(|| refuse("the attention output does not fit")));
    let plans: Option<Vec<LayerPlan>> = plans.into_iter().collect();
    let (head, (n_experts, top_k), rope, layers, proj, attn_out, plans) =
        (head?, moe?, rope?, layers?, proj?, attn_out?, plans?);
    Some(Checked {
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
/// holds: `ctx` within the file's trained context, and every length of it
/// fits.
fn cache_lens(ctx: usize, trained: usize) -> Result<(usize, usize), SessionError> {
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
    let _ =
        f16_len(ctx).ok_or_else(|| refuse(format!("the f16 cache of {ctx} rows does not fit")))?;
    let rope_len = ctx
        .checked_mul(ROT)
        .ok_or_else(|| refuse(format!("the rope table of {ctx} rows does not fit")))?;
    let segments = segments_for_keys(word) as usize;
    let attn_partials = partials_len(segments).ok_or_else(|| {
        refuse(format!(
            "the attention partials of {segments} segments do not fit"
        ))
    })?;
    Ok((rope_len, attn_partials))
}

/// The cache, the hidden width, the vocabulary, the embedding and the head.
fn head_checks(at: &mut At<'_>, t: &Table<'_>, spec: &ModelSpec, ctx: usize) -> Option<HeadChecks> {
    let lens = at.ok(cache_lens(ctx, spec.ctx_train as usize));
    let hidden = spec.hidden as usize;
    let vocab = spec.vocab as usize;
    let mut ok = true;
    if hidden != HEAD_K || hidden != router::K {
        at.push(refuse(format!(
            "a hidden width of {hidden}; the head and the router read {HEAD_K}"
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
            let fits = matches!(s.quant, Quant::Q4K | Quant::A4);
            if !fits {
                at.push(refuse(format!(
                    "{} is {:?}; the embedding reads Q4_K or MLX_A4",
                    names::token_embd(),
                    s.quant
                )));
            }
            fits
        });
    let head = at.ok(t.site(&names::output(), hidden, vocab)).filter(|s| {
        let fits = matches!(s.quant, Quant::Q6K | Quant::A4);
        if !fits {
            at.push(refuse(format!(
                "{} is {:?}; the head reads Q6_K or MLX_A4",
                names::output(),
                s.quant
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
        match at.ok(t.site(name, cols, rows)) {
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
        for quant in [Quant::Q4K, Quant::Q6K, Quant::A4] {
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
    if n_v != VALUE_HEADS {
        at.push(refuse(format!(
            "{n_v} value heads; the delta step runs {VALUE_HEADS}"
        )));
        return None;
    }
    let Some(ch) = channels(n_k) else {
        at.push(refuse(format!("{n_k} key heads do not fit")));
        return None;
    };
    let z_len = VALUE_HEADS * HEAD_COLS;
    let p1 = p1_checks(
        at,
        t,
        &d.attn_norm,
        &[
            (&d.qkv, ch),
            (&d.gate, z_len),
            (&d.beta, VALUE_HEADS),
            (&d.alpha, VALUE_HEADS),
        ],
        hidden,
    );
    let out = at.ok(t.site(&d.ssm_out, z_len, hidden)).filter(|s| {
        let fits = matches!(s.quant, Quant::Q4K | Quant::A4);
        if !fits {
            at.push(refuse(format!(
                "{} is {:?}; the head-norm projection reads Q4_K or MLX_A4",
                d.ssm_out, s.quant
            )));
        }
        fits
    });
    let state = at.ok(state_elems(LANES).ok_or_else(|| refuse("the delta state does not fit")));
    let ring = at.ok(ring_elems(n_k).ok_or_else(|| refuse("the conv ring does not fit")));
    let conv = at.ok(t.f32s(&d.conv, CONV_TAPS * ch));
    let dt_bias = at.ok(t.f32s(&d.dt_bias, VALUE_HEADS));
    let ssm_a = at.ok(t.f32s(&d.ssm_a, VALUE_HEADS));
    let ssm_norm = at.ok(t.f32s(&d.ssm_norm, HEAD_COLS));
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
    let q_rows = at.ok(qg_len().ok_or_else(|| refuse("the query rows do not fit")));
    let kv_rows = KV_HEADS * HEAD_DIM;
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
    let att = at.ok(out_len().ok_or_else(|| refuse("the attention rows do not fit")));
    let out = att.and_then(|att| at.ok(t.site(&g.attn_output, att, hidden)));
    let cache = at.ok(f16_len(ctx).ok_or_else(|| refuse("the cache does not fit")));
    let q_norm = at.ok(t.f32s(&g.attn_q_norm, HEAD_DIM));
    let k_norm = at.ok(t.f32s(&g.attn_k_norm, HEAD_DIM));
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
    match &plan.ffn.route {
        FfnRoute::Router {
            shared: Some(SharedPlan::Joined),
            ..
        } => {}
        FfnRoute::Router {
            shared: Some(SharedPlan::Apart(_)),
            ..
        } => {
            at.push(refuse(
                "a shared expert of another type than its stacks, which the fold does not run",
            ));
            return None;
        }
        FfnRoute::Router { shared: None, .. } => {
            at.push(refuse(
                "a router with no shared expert, which the router does not run",
            ));
            return None;
        }
        FfnRoute::Dense => {
            at.push(refuse(
                "a dense FFN, which this backend's experts do not run",
            ));
            return None;
        }
    }
    if n_experts != router::EXPERTS || top_k != ROUTED_IDS {
        at.push(refuse(format!(
            "{n_experts} experts picking {top_k}; the router and the experts run \
             {} picking {ROUTED_IDS}",
            router::EXPERTS
        )));
        return None;
    }
    // The file's tensors, under the plan's file-side names (chain's one owner
    // of the naming): the plan names the joined stacks, which this backend
    // reads as the routed stack and the shared expert apart.
    let file = file_names(l);
    let gate = at.ok(t.site(&file.gate, hidden, EXPERT_K * n_experts));
    let up = at.ok(t.site(&file.up, hidden, EXPERT_K * n_experts));
    let shexp_gate = at.ok(t.site(&file.sh_gate, hidden, EXPERT_K));
    let shexp_up = at.ok(t.site(&file.sh_up, hidden, EXPERT_K));
    let mut ok = true;
    if let (Some(g), Some(u), Some(sg), Some(su)) = (&gate, &up, &shexp_gate, &shexp_up)
        && [u.quant, sg.quant, su.quant].iter().any(|&q| q != g.quant)
    {
        at.push(refuse(format!(
            "gate {:?}, up {:?}, shared gate {:?}, shared up {:?}; the pair runs one type",
            g.quant, u.quant, sg.quant, su.quant
        )));
        ok = false;
    }
    let down = at.ok(t.site(&file.down, EXPERT_K, hidden * n_experts));
    let shexp_down = at.ok(t.site(&file.sh_down, EXPERT_K, hidden));
    if let (Some(d), Some(sd)) = (&down, &shexp_down)
        && sd.quant != d.quant
    {
        at.push(refuse(format!(
            "down {:?}, shared down {:?}; the fold runs one type",
            d.quant, sd.quant
        )));
        ok = false;
    }
    let norm = at.ok(t.f32s(&plan.ffn.ffn_norm, hidden));
    // The router and the shared-expert gate: F32 in a GGUF file, MLX's
    // affine 8-bit in an MLX file, dequantized exactly at open.
    let gate_inp = at.ok(t.f32s_or_a8(&file.router, n_experts * hidden));
    let gate_inp_shexp = at.ok(t.f32s_or_a8(&file.router_gate, hidden));
    let (gate, up, shexp_gate, shexp_up, down, shexp_down) =
        (gate?, up?, shexp_gate?, shexp_up?, down?, shexp_down?);
    ok.then_some(FfnLayout {
        norm: norm?,
        gate_inp: gate_inp?,
        gate_inp_shexp: gate_inp_shexp?,
        gate_up_quant: gate.quant,
        down_quant: down.quant,
        gate,
        up,
        shexp_gate,
        shexp_up,
        down,
        shexp_down,
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
            if spec_g.rope.dims as usize != ROT {
                at.push(refuse(format!(
                    "a rope over {} values; the attention turns {ROT}",
                    spec_g.rope.dims
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
    let Ffn::Moe(m) = &layer.ffn else {
        at.push(refuse(
            "a dense FFN, which this backend's experts do not run",
        ));
        return None;
    };
    let shape = (m.experts as usize, m.top_k as usize);
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
    if m.expert_ff as usize != EXPERT_K {
        at.push(refuse(format!(
            "an expert of {}; the experts run {EXPERT_K}",
            m.expert_ff
        )));
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
        let g = &self.layout;
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
            for s in [
                &f.gate,
                &f.up,
                &f.shexp_gate,
                &f.shexp_up,
                &f.down,
                &f.shexp_down,
            ] {
                b.weights.add(s.bytes());
            }
            for v in [&f.norm, &f.gate_inp, &f.gate_inp_shexp] {
                b.weights.add(f32s(v.len));
            }
            // The router's logits, picks, weights, partials and ticket.
            for words in [router::ROWS, g.top_k + 1, g.top_k + 1, router::PARTIALS, 1] {
                b.scratch.add(f32s(words));
            }
        }
        for words in [
            g.rope_len,
            g.hidden,
            g.proj,
            VALUE_HEADS * HEAD_COLS,
            VALUE_HEADS * PARTIALS_PER_HEAD,
            g.attn_out,
            g.attn_partials,
            1,
            EXPERT_SLOTS * EXPERT_K,
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
            pass_rows_max().min(g.ctx),
            g.hidden,
            g.proj,
            g.n_experts,
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

    /// A qwen35moe file's header and tensor table, in memory: two layers (a
    /// delta layer, then an attention one: `full_attention_interval` 2) at the
    /// shapes the kernels run, every weight Q4_K but the head's Q6_K, and no
    /// weight byte anywhere. A test edits it and reads it back as a `Split`.
    struct Synthetic {
        kvs: Vec<(String, Value)>,
        tensors: Vec<(String, Vec<u64>, GgmlType)>,
    }

    const HIDDEN: u64 = 2048;
    const VOCAB: u64 = 248_320;
    const EXPERTS: u64 = 256;
    const FF: u64 = 512;

    impl Synthetic {
        fn new() -> Synthetic {
            let u = |k: &str, v: u32| (format!("qwen35moe.{k}"), Value::U32(v));
            let kvs = vec![
                (
                    gguf::GENERAL_ARCHITECTURE.to_owned(),
                    Value::String("qwen35moe".to_owned()),
                ),
                (
                    "tokenizer.ggml.pre".to_owned(),
                    Value::String("qwen35".to_owned()),
                ),
                u("block_count", 2),
                u("vocab_size", 248_320),
                u("context_length", 262_144),
                u("embedding_length", 2048),
                u("attention.head_count", 16),
                u("attention.head_count_kv", 2),
                (
                    "qwen35moe.attention.layer_norm_rms_epsilon".to_owned(),
                    Value::F32(1e-6),
                ),
                u("expert_count", 256),
                u("expert_used_count", 8),
                u("attention.key_length", 256),
                u("attention.value_length", 256),
                u("expert_feed_forward_length", 512),
                u("expert_shared_feed_forward_length", 512),
                u("ssm.conv_kernel", 4),
                u("ssm.state_size", 128),
                u("ssm.group_count", 16),
                u("ssm.time_step_rank", 32),
                u("ssm.inner_size", 4096),
                u("full_attention_interval", 2),
                (
                    "qwen35moe.rope.dimension_sections".to_owned(),
                    Value::Array(vec![
                        Value::I32(11),
                        Value::I32(11),
                        Value::I32(10),
                        Value::I32(0),
                    ]),
                ),
                (
                    "qwen35moe.rope.freq_base".to_owned(),
                    Value::F32(10_000_000.0),
                ),
                u("rope.dimension_count", 64),
            ];
            let mut t = Synthetic {
                kvs,
                tensors: Vec::new(),
            };
            let q = GgmlType::Q4_K;
            t.add("token_embd.weight", &[HIDDEN, VOCAB], q);
            t.add("output.weight", &[HIDDEN, VOCAB], GgmlType::Q6_K);
            t.add("output_norm.weight", &[HIDDEN], GgmlType::F32);
            for (l, attention) in [false, true].into_iter().enumerate() {
                let b = |s: &str| format!("blk.{l}.{s}");
                t.add(&b("attn_norm.weight"), &[HIDDEN], GgmlType::F32);
                if attention {
                    t.add(&b("attn_q.weight"), &[HIDDEN, 8192], q);
                    t.add(&b("attn_k.weight"), &[HIDDEN, 512], q);
                    t.add(&b("attn_v.weight"), &[HIDDEN, 512], q);
                    t.add(&b("attn_output.weight"), &[4096, HIDDEN], q);
                    t.add(&b("attn_q_norm.weight"), &[256], GgmlType::F32);
                    t.add(&b("attn_k_norm.weight"), &[256], GgmlType::F32);
                } else {
                    t.add(&b("attn_qkv.weight"), &[HIDDEN, 8192], q);
                    t.add(&b("attn_gate.weight"), &[HIDDEN, 4096], q);
                    t.add(&b("ssm_beta.weight"), &[HIDDEN, 32], q);
                    t.add(&b("ssm_alpha.weight"), &[HIDDEN, 32], q);
                    t.add(&b("ssm_conv1d.weight"), &[4, 8192], GgmlType::F32);
                    t.add(&b("ssm_dt.bias"), &[32], GgmlType::F32);
                    t.add(&b("ssm_a"), &[32], GgmlType::F32);
                    t.add(&b("ssm_norm.weight"), &[128], GgmlType::F32);
                    t.add(&b("ssm_out.weight"), &[4096, HIDDEN], q);
                }
                t.add(&b("post_attention_norm.weight"), &[HIDDEN], GgmlType::F32);
                t.add(&b("ffn_gate_inp.weight"), &[HIDDEN, EXPERTS], GgmlType::F32);
                t.add(&b("ffn_gate_inp_shexp.weight"), &[HIDDEN], GgmlType::F32);
                t.add(&b("ffn_gate_exps.weight"), &[HIDDEN, FF, EXPERTS], q);
                t.add(&b("ffn_up_exps.weight"), &[HIDDEN, FF, EXPERTS], q);
                t.add(&b("ffn_down_exps.weight"), &[FF, HIDDEN, EXPERTS], q);
                t.add(&b("ffn_gate_shexp.weight"), &[HIDDEN, FF], q);
                t.add(&b("ffn_up_shexp.weight"), &[HIDDEN, FF], q);
                t.add(&b("ffn_down_shexp.weight"), &[FF, HIDDEN], q);
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
        let spec = model::arch::qwen35moe::spec::read(split)
            .expect("the description reads")
            .spec;
        let kinds = kinds35(&spec.layers).expect("the kinds read");
        let pd = PlanDims::q35(&spec, WHAT).expect("the dims read");
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

    #[test]
    fn a_clean_table_opens() {
        let t = Synthetic::new();
        let out = check(&t.split(), CTX);
        assert!(out.items.is_empty(), "{:?}", out.items);
        let checked = out.checked.expect("a checked file");
        // The weights the plan counts are the file's tensors, byte for byte.
        let bytes = checked.bytes().expect("the bytes");
        assert_eq!(bytes.weights.requested, t.tensor_bytes());
        assert!(bytes.total().paged >= bytes.total().requested);
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
        use crate::kernels::{LOGIT_STRIDE, RING_ROWS, prepared_elems, route_table_words};

        const N_K: usize = 16;
        const TOP_K: usize = 8;
        let hidden = HIDDEN as usize;
        let vocab = VOCAB as usize;
        let t = Synthetic::new();
        let checked = check(&t.split(), CTX).checked.expect("a checked file");
        let bytes = checked.bytes().expect("the bytes");
        let f32s = |n: usize| count(n) * 4;

        // The K and the V cache of the attention layer, f16.
        let cache = count(KV_HEADS * CTX * HEAD_DIM) * 2;
        assert_eq!(bytes.kv, class(&[cache, cache]));

        // The delta layer's state lanes and its conv ring.
        let channels = (2 * N_K + VALUE_HEADS) * HEAD_COLS;
        assert_eq!(
            bytes.state,
            class(&[
                f32s(LANES * VALUE_HEADS * HEAD_COLS * HEAD_COLS),
                f32s(RING_ROWS * channels),
            ])
        );

        // Each layer's router (logits, picks, weights, partials, ticket), then
        // the token arena's buffers in field order.
        let router = [router::ROWS, TOP_K + 1, TOP_K + 1, router::PARTIALS, 1];
        let proj = channels + VALUE_HEADS * HEAD_COLS + 2 * VALUE_HEADS;
        let attn_partials =
            partials_len(segments_for_keys(CTX as u32) as usize).expect("the partials fit");
        let mut scratch: Vec<u64> = [router, router].concat().into_iter().map(f32s).collect();
        scratch.extend(
            [
                hidden,                          // x
                proj,                            // proj
                VALUE_HEADS * HEAD_COLS,         // g
                VALUE_HEADS * PARTIALS_PER_HEAD, // delta_partials
                VALUE_HEADS * HEAD_COLS,         // attn
                attn_partials,                   // attn_partials
                1,                               // attn_ticket
                CTX * ROT,                       // rope
                EXPERT_SLOTS * EXPERT_K,         // h
                vocab,                           // logits
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
        let cap = pass_rows_max().min(CTX);
        let slots = cap * EXPERT_SLOTS;
        let prepared = prepared_elems(N_K).expect("the prepared row fits");
        let rows = [
            cap,
            cap * hidden,
            cap * hidden,
            cap * proj.next_multiple_of(4),
            cap * VALUE_HEADS * HEAD_COLS,
            cap * VALUE_HEADS * PARTIALS_PER_HEAD,
            cap * prepared,
            cap * LOGIT_STRIDE,
            slots,
            slots,
            slots,
            slots,
            route_table_words(slots, router::EXPERTS),
            slots * EXPERT_K,
            slots * hidden,
        ]
        .map(f32s);
        assert_eq!(bytes.rows, class(&rows));
    }

    #[test]
    fn the_architecture_verdict_is_the_one_item() {
        let t = Synthetic::new().kv(
            gguf::GENERAL_ARCHITECTURE,
            Value::String("qwen3moe".to_owned()),
        );
        let got = texts(&t.split(), CTX);
        assert_eq!(got.len(), 1, "{got:?}");
        assert!(
            got[0].starts_with("qwen35moe session: synthetic.gguf: ")
                && got[0].contains(gguf::GENERAL_ARCHITECTURE)
                && got[0].contains("\"qwen3moe\""),
            "{got:?}"
        );
    }

    #[test]
    fn a_site_type_no_mat_vec_runs_is_the_chain_words_and_the_opens_first() {
        let t = Synthetic::new().ty("blk.0.attn_qkv.weight", GgmlType::Q8_0);
        let split = t.split();
        let got = texts(&split, CTX);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0], old_plan_error(&split));
        assert_eq!(
            got[0],
            "qwen35moe session: the plan: qwen35moe session: blk.0.attn_qkv.weight is q8_0 in the \
             file; this site launches Q4_K, Q6_K, MLX_A4"
        );
    }

    #[test]
    fn every_site_of_a_half_the_lists_refuse_is_an_item_in_the_readers_order() {
        let t = Synthetic::new()
            .ty("blk.0.attn_qkv.weight", GgmlType::Q8_0)
            .ty("blk.0.ssm_beta.weight", GgmlType::F32)
            .ty("blk.0.attn_gate.weight", GgmlType::Q5_K);
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
            .ty("blk.1.ffn_down_shexp.weight", GgmlType::Q8_0)
            .ty("blk.1.ffn_down_exps.weight", GgmlType::Q5_K)
            .ty("blk.1.ffn_gate_shexp.weight", GgmlType::Q8_0);
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
        // gate, shared gate, up, shared up, down, shared down.
        assert_eq!(
            tensors,
            [
                "blk.1.ffn_gate_shexp.weight",
                "blk.1.ffn_down_exps.weight",
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
            ["qwen35moe arena: token_embd.weight is Q6K; the embedding reads Q4_K or MLX_A4"]
        );
        let got = texts(
            &Synthetic::new().ty("output.weight", GgmlType::Q4_K).split(),
            CTX,
        );
        assert_eq!(
            got,
            ["qwen35moe arena: output.weight is Q4K; the head reads Q6_K or MLX_A4"]
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
        // A shared expert of another type than its stacks: the plan reads it
        // apart, the arena's fold does not run it.
        let got = texts(
            &Synthetic::new()
                .ty("blk.0.ffn_down_shexp.weight", GgmlType::Q6_K)
                .split(),
            CTX,
        );
        assert_eq!(
            got,
            [
                "qwen35moe arena: layer 0: a shared expert of another type than its stacks, which \
              the fold does not run"
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
            .ty("blk.0.attn_qkv.weight", GgmlType::Q8_0);
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
            .ty("blk.1.attn_q.weight", GgmlType::Q8_0)
            .ty("blk.0.ffn_down_exps.weight", GgmlType::Q5_K);
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
            .ty("blk.0.attn_qkv.weight", GgmlType::Q8_0)
            .ty("blk.1.attn_q.weight", GgmlType::Q8_0);
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
        assert_eq!(header.split.tensor_count(), t.tensors.len());
        let out = check(&header.split, CTX);
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
