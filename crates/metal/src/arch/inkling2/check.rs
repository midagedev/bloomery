//! What an Inkling 2 file asks of this backend, read from its header and
//! tensor table alone: the one owner of every refusal the open makes before it
//! allocates a buffer, and of the bytes the open makes buffers of.
//!
//! [`check`] reads the file's description (`model::arch::inkling2::spec`), its
//! visits, and every tensor a visit, the ends and the loop read, each resolved
//! to its name, shape and launch type. It returns every refusal it meets, in
//! the order the open meets them, and the [`Checked`] file when there is none.
//! A GGUF file and an MLX folder read through the same names (the folder's map
//! writes the GGUF twin's header): a dense projection launches in its stored
//! type (Q8_0, the K-quants, MLX's affine blocks); a projection stored BF16 or
//! F32 (the sink and dynamic-tap rows) is widened to f32 at the load, exactly;
//! a routed stack is NVFP4 (the GGUF's blocks, the folder's planes), Q8_0 (the
//! GGUF coda) or MXFP8 (the folder's coda). Any other type is refused by name.

use gguf::quant::GgmlType;
use gguf::{Place, Split};
use model::arch::models::{Act, Ffn, Mixer, ModelSpec};

use crate::inkling2_geom::MIX_VECS;
use crate::kernels::{NvfpLayout, Quant};
use crate::plan::Item;
use crate::session::room::{Bytes, Tally};

/// The family's name in a refusal.
pub const WHAT: &str = "inkling2";

/// The epsilon of the gainless RMS norms (the mix's, the controllers', the
/// value rows'): the reference's fixed `1e-6`.
pub const EPS: f32 = 1.0e-6;

/// The stored layers a block of the attention residual spans: a layer whose
/// index is a multiple of it closes a block onto the bank.
pub const BLOCK_LAYERS: usize = 6;

/// The conv ring's channels a visit: the k and v convs' 1024 each, then the
/// attention's and the FFN's post-convs at the hidden width.
pub const CONV_CHANNELS: usize = 2 * 1024 + 2 * 2304;

/// A refusal of the check, in the session's words.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal(pub String);

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{WHAT}: {}", self.0)
    }
}

/// A dense projection: its tensor, its launch type and its shape.
#[derive(Clone, Debug, PartialEq)]
pub struct Site {
    pub name: String,
    /// The launch type; [`Quant::F32`] for a BF16 or F32 tensor the load
    /// widens.
    pub quant: Quant,
    /// The type the file stores it in.
    pub stored: GgmlType,
    pub rows: usize,
    pub cols: usize,
}

impl Site {
    /// The blocks of one row in the launch type.
    #[must_use]
    pub fn row_blocks(&self) -> usize {
        self.cols / self.quant.values_per_block()
    }

    /// The bytes of `rows` rows in the launch type.
    #[must_use]
    pub fn bytes_of(&self, rows: usize) -> usize {
        rows * self.row_blocks() * self.quant.block_bytes()
    }

    /// The bytes the load makes of it.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.bytes_of(self.rows)
    }
}

/// A routed stack: its tensor, its form, its experts' rows and width.
#[derive(Clone, Debug, PartialEq)]
pub struct Stack {
    pub name: String,
    pub layout: NvfpLayout,
    /// The folder's planes, which the load keeps as two planes an expert
    /// (NVFP4) or interleaves into rows (MXFP8); the GGUF's stacks load as
    /// stored.
    pub planes: bool,
    pub experts: usize,
    pub rows: usize,
    pub k: usize,
    /// The per-expert scale tensor of a GGUF NVFP4 stack; a folder's scales
    /// ride in its place, and the other forms have none.
    pub scale: Option<String>,
}

impl Stack {
    /// The bytes one expert's rows take in the launch's layout.
    #[must_use]
    pub fn expert_bytes(&self) -> usize {
        self.layout
            .bytes(self.rows, self.k)
            .expect("the check refused a width the layout does not take")
    }
}

/// One stored layer's tensors.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPlan {
    pub windowed: bool,
    /// The layer is in the loop's core: its four controllers are read.
    pub core: bool,
    /// The input norms of the q, k and v paths (one gain thrice on a windowed
    /// layer).
    pub norms: [String; 3],
    /// The q, relative and gate projections: one launch over one type.
    pub qrg: [Site; 3],
    pub sink: Site,
    pub k: Site,
    pub v: Site,
    /// The dynamic taps' rows of a windowless layer's k and v convs.
    pub k_taps: Option<Site>,
    pub v_taps: Option<Site>,
    pub wo: Site,
    /// The attention's mix queries and their count; `None` at layer 0, which
    /// reads the stream as it is.
    pub attn_query: Option<(String, usize)>,
    pub ffn_query: String,
    pub ffn_norm: String,
    pub q_clip: String,
    pub rel_clip: String,
    pub rel_bias: String,
    /// The k, v, attention-output and FFN-output convs' taps.
    pub convs: [String; 4],
    pub router: String,
    pub router_bias: String,
    pub gate_up: Stack,
    pub down: Stack,
}

/// One visit of the walk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisitPlan {
    pub layer: usize,
    pub pass: usize,
    pub windowed: bool,
    /// The stored layer closes a block onto the bank.
    pub boundary: bool,
    /// A core visit of a pass past the first: it merges the loop's messages.
    pub message: bool,
    /// The core layer's index among the core's layers.
    pub core_index: Option<usize>,
    /// The last core visit of a pass the loop runs again.
    pub pass_end: bool,
}

/// A file every check passed: what the open loads and walks.
#[derive(Clone, Debug)]
pub struct Checked {
    pub layers: Vec<LayerPlan>,
    pub visits: Vec<VisitPlan>,
    pub hidden: usize,
    pub vocab: usize,
    /// The vocabulary rows the head computes and picks over: the file's cut.
    pub cut: usize,
    pub embed: Site,
    pub head: Site,
    pub ctx: usize,
    pub core: (usize, usize),
    pub passes: usize,
    /// The blocks the layers before the core close.
    pub prelude_blocks: usize,
    /// The blocks a walk holds at most: the stream's row is one more.
    pub max_banks: usize,
    /// The blocks a pass closes past its first: the loop's message bank holds
    /// them and the stream's row.
    pub message_rows: usize,
    pub window: usize,
    pub ring_rows: usize,
    pub taps: usize,
    pub rms_eps: f32,
    pub qk_clip: f32,
    pub rel_clip: f32,
    pub log_floor: u32,
    pub log_alpha: f32,
    pub route_scale: f32,
    pub gate_beta: f32,
    pub value_beta: f32,
    pub softcap: f32,
    pub ff: usize,
    pub top_k: usize,
    pub experts: usize,
    /// The description's arch line, for the plan.
    pub arch_line: String,
}

/// The check's verdict: the description's line, every refusal, and the
/// checked file when there is none.
#[derive(Debug)]
pub struct Outcome {
    pub spec_line: Option<String>,
    pub items: Vec<Item<Refusal>>,
    pub checked: Option<Checked>,
}

impl Outcome {
    /// The checked file, or the first refusal.
    ///
    /// # Errors
    /// The first refusal the check met.
    pub fn checked_or_first(self) -> Result<Checked, Refusal> {
        match (self.checked, self.items.into_iter().next()) {
            (_, Some(first)) => Err(first.error),
            (Some(c), None) => Ok(c),
            (None, None) => Err(Refusal("the check found no file and no refusal".into())),
        }
    }
}

/// The tensor `name`, or a refusal naming it.
fn find<'s>(split: &'s Split, name: &str) -> Result<&'s gguf::TensorInfo, Refusal> {
    split
        .find(name)
        .map(|(_, t)| t)
        .ok_or_else(|| Refusal(format!("the file has no tensor {name}")))
}

/// `name` as a dense projection of `rows` rows over `cols` columns.
fn site(split: &Split, name: String, rows: usize, cols: usize) -> Result<Site, Refusal> {
    let t = find(split, &name)?;
    let quant = match t.ty {
        GgmlType::BF16 | GgmlType::F32 => Quant::F32,
        ty => Quant::of_ggml(ty).ok_or_else(|| {
            Refusal(format!(
                "{name} is {ty:?}; a projection launches Q8_0, a K-quant or MLX's affine blocks, \
                 or widens BF16 and F32 to f32"
            ))
        })?,
    };
    let want = [cols as u64, rows as u64];
    if t.dims != want {
        return Err(Refusal(format!(
            "{name} is {:?}; the family reads {want:?}",
            t.dims
        )));
    }
    if !cols.is_multiple_of(quant.values_per_block()) {
        return Err(Refusal(format!(
            "{name}: {cols} columns are not whole blocks of {}",
            quant.values_per_block()
        )));
    }
    Ok(Site {
        name,
        quant,
        stored: t.ty,
        rows,
        cols,
    })
}

/// `name` as an f32 tensor of `dims`.
fn values(split: &Split, name: String, dims: &[u64]) -> Result<String, Refusal> {
    let t = find(split, &name)?;
    if t.ty != GgmlType::F32 {
        return Err(Refusal(format!(
            "{name} is {:?}; the family reads it as F32",
            t.ty
        )));
    }
    if t.dims != dims {
        return Err(Refusal(format!(
            "{name} is {:?}; the family reads {dims:?}",
            t.dims
        )));
    }
    Ok(name)
}

/// `name` as a routed stack of `experts` experts of `rows` rows over `k`; an
/// MXFP8 stack of the folder's coda is in the layout `mx8`.
fn stack(
    split: &Split,
    name: String,
    experts: usize,
    rows: usize,
    k: usize,
    mx8: NvfpLayout,
) -> Result<Stack, Refusal> {
    let t = find(split, &name)?;
    let planes = matches!(split.place(&name), Some(Place::Stack { .. }));
    let layout = match (t.ty, planes) {
        (GgmlType::NVFP4, false) => NvfpLayout::Blocks,
        (GgmlType::NVFP4_ROWS, true) => NvfpLayout::Planes,
        (GgmlType::Q8_0, false) => NvfpLayout::Q8_0,
        (GgmlType::MXFP8, true) => mx8,
        (ty, _) => {
            return Err(Refusal(format!(
                "{name} is {ty:?}{}; a routed stack launches NVFP4 (the GGUF's blocks or the \
                 folder's planes), the GGUF's Q8_0 or the folder's MXFP8",
                if planes { " in planes" } else { "" }
            )));
        }
    };
    let want = [k as u64, rows as u64, experts as u64];
    if t.dims != want {
        return Err(Refusal(format!(
            "{name} is {:?}; the family reads {want:?}",
            t.dims
        )));
    }
    if layout.bytes(rows, k).is_none() {
        return Err(Refusal(format!(
            "{name}: {k} values a row are not whole groups of {}",
            layout.step()
        )));
    }
    let scale = match layout {
        NvfpLayout::Blocks => Some(values(
            split,
            name.replace(".weight", ".scale"),
            &[experts as u64],
        )?),
        _ => None,
    };
    Ok(Stack {
        name,
        layout,
        planes,
        experts,
        rows,
        k,
        scale,
    })
}

/// Every refusal of the file `split` at a cache of `ctx` rows, in the order
/// the open meets them, and the checked file when there is none: its MXFP8
/// coda in the quads layout.
#[must_use]
pub fn check(split: &Split, ctx: usize) -> Outcome {
    check_with(split, ctx, NvfpLayout::Mxfp8Quads)
}

/// [`check`] with the MXFP8 coda in the layout `mx8`: [`NvfpLayout::Mxfp8Quads`]
/// or [`NvfpLayout::Mxfp8Rows`]. Another layout is a refusal.
#[must_use]
pub fn check_with(split: &Split, ctx: usize, mx8: NvfpLayout) -> Outcome {
    let mut items = Vec::new();
    if !matches!(mx8, NvfpLayout::Mxfp8Quads | NvfpLayout::Mxfp8Rows) {
        items.push(Item {
            layer: None,
            error: Refusal(format!(
                "the MXFP8 coda's layout is {mx8:?}: the quads or the rows"
            )),
        });
        return Outcome {
            spec_line: None,
            items,
            checked: None,
        };
    }
    let spec = match model::arch::inkling2::spec::read(split) {
        Ok(read) => read.spec,
        Err(e) => {
            items.push(Item {
                layer: None,
                error: Refusal(format!("the description: {e}")),
            });
            return Outcome {
                spec_line: None,
                items,
                checked: None,
            };
        }
    };
    let checked = read(split, &spec, ctx, mx8, &mut items);
    Outcome {
        spec_line: Some(arch_line(&spec)),
        checked: checked.filter(|_| items.is_empty()),
        items,
    }
}

/// [`check`], then the room rule's refusal appended when the session's bytes
/// pass `free` (the device's free bytes, or why there is no reading).
#[must_use]
pub fn check_room(split: &Split, ctx: usize, free: Result<u64, String>) -> Outcome {
    let mut outcome = check(split, ctx);
    if let (Some(checked), Ok(free)) = (&outcome.checked, free) {
        let held = checked.bytes().held().requested;
        if held > free {
            outcome.items.push(Item {
                layer: None,
                error: Refusal(format!(
                    "the session asks {held} bytes of the device, which has {free} free"
                )),
            });
            outcome.checked = None;
        }
    }
    outcome
}

/// The description's line in the plan.
fn arch_line(spec: &ModelSpec) -> String {
    let (cs, ce, passes) = spec
        .loop_
        .map_or((0, 0, 1), |l| (l.core_start, l.core_end, l.passes));
    format!(
        "arch=inkling2 layers={} visits={} (core {cs}..{ce} x{passes}) hidden={} vocab={} \
         trained_ctx={}",
        spec.layers.len(),
        spec.visits().len(),
        spec.hidden,
        spec.vocab,
        spec.ctx_train
    )
}

/// The visits of a walk over `windowed` stored layers with the loop's core
/// `core` run `passes` times: the boundary, message and pass-end marks each
/// visit carries.
#[must_use]
pub fn visit_plans(
    visits: &[model::arch::models::Visit],
    windowed: &[bool],
    core: (usize, usize),
    passes: usize,
) -> Vec<VisitPlan> {
    visits
        .iter()
        .map(|v| {
            let layer = v.layer as usize;
            let pass = v.pass as usize;
            let in_core = (core.0..core.1).contains(&layer);
            VisitPlan {
                layer,
                pass,
                windowed: windowed[layer],
                boundary: layer.is_multiple_of(BLOCK_LAYERS),
                message: in_core && pass > 0,
                core_index: in_core.then(|| layer - core.0),
                pass_end: in_core && layer + 1 == core.1 && pass + 1 < passes,
            }
        })
        .collect()
}

/// The checked file of `spec`, its refusals pushed onto `items`.
fn read(
    split: &Split,
    spec: &ModelSpec,
    ctx: usize,
    mx8: NvfpLayout,
    items: &mut Vec<Item<Refusal>>,
) -> Option<Checked> {
    let refuse = |items: &mut Vec<Item<Refusal>>, detail: String| {
        items.push(Item {
            layer: None,
            error: Refusal(detail),
        });
    };
    let hidden = spec.hidden as usize;
    let vocab = spec.vocab as usize;
    let Some(loop_) = spec.loop_ else {
        refuse(
            items,
            "the file states no loop: this backend runs the looped Mini".into(),
        );
        return None;
    };
    let (cs, ce) = (loop_.core_start as usize, loop_.core_end as usize);
    if !cs.is_multiple_of(BLOCK_LAYERS) {
        refuse(
            items,
            format!(
                "the loop's core starts at layer {cs}, inside a block: the walk keeps the \
                 prelude's stream as the block the core's first layer closes"
            ),
        );
        return None;
    }
    if ctx == 0 || u32::try_from(ctx).is_err() {
        refuse(
            items,
            format!("a cache of {ctx} rows: a session holds 1 to u32 rows"),
        );
        return None;
    }
    let epi = spec.logit_epilogue;
    let cut = epi.and_then(|e| e.cut).map_or(vocab, |c| c as usize);
    let softcap = epi.map_or(0.0, |e| e.softcap);
    let (Some(Mixer::RelBias(a0)), Some(Ffn::Moe(moe))) = (
        spec.layers.first().map(|l| &l.mixer),
        spec.layers.first().map(|l| &l.ffn),
    ) else {
        refuse(
            items,
            "layer 0 is not the relative-bias attention over a routed FFN".into(),
        );
        return None;
    };
    let Act::Situ {
        gate_beta,
        value_beta,
    } = moe.act
    else {
        refuse(
            items,
            format!(
                "the experts' activation is {}; the family runs situ",
                moe.act.what()
            ),
        );
        return None;
    };
    let (experts, ff, top_k) = (
        moe.experts as usize,
        moe.expert_ff as usize,
        moe.top_k as usize,
    );
    if (experts, top_k, hidden) != (128, 8, 2304) {
        refuse(
            items,
            format!(
                "{experts} experts at the top {top_k} over a hidden of {hidden}; the family's \
                 kernels hold 128 at the top 8 over 2304"
            ),
        );
        return None;
    }
    let mut window = 0usize;
    let mut layers = Vec::with_capacity(spec.layers.len());
    for (l, layer) in spec.layers.iter().enumerate() {
        let Mixer::RelBias(a) = &layer.mixer else {
            items.push(Item {
                layer: Some(l),
                error: Refusal(format!("layer {l} is not the relative-bias attention")),
            });
            continue;
        };
        let shape = (a.heads, a.kv_heads, a.head_dim, a.d_rel, a.extent, a.taps);
        if shape != (24, 8, 128, 16, 1024, 4) {
            items.push(Item {
                layer: Some(l),
                error: Refusal(format!(
                    "layer {l}: (heads, key heads, head, relative, extent, taps) {shape:?}; the \
                     kernels hold (24, 8, 128, 16, 1024, 4)"
                )),
            });
            continue;
        }
        if let Some(w) = a.window {
            window = w as usize;
        }
        let core = (cs..ce).contains(&l);
        match layer_plan(
            split,
            l,
            a.window.is_some(),
            core,
            (hidden, ff, experts),
            mx8,
        ) {
            Ok(p) => layers.push(p),
            Err(e) => items.push(Item {
                layer: Some(l),
                error: e,
            }),
        }
    }
    let globals = (|| {
        let embed = site(split, "token_embd.weight".into(), vocab, hidden)?;
        let head = site(split, "output.weight".into(), vocab, hidden)?;
        if matches!(head.quant, Quant::F32 | Quant::Q4K) {
            return Err(Refusal(format!(
                "output.weight is {:?}; the head launches Q8_0, Q5_K, Q6_K or MLX's affine blocks",
                head.stored
            )));
        }
        if matches!(embed.quant, Quant::F32) && embed.stored != GgmlType::F32 {
            return Err(Refusal(format!(
                "token_embd.weight is {:?}; the embedding reads a quantized or F32 table",
                embed.stored
            )));
        }
        let h = hidden as u64;
        for n in [
            "token_embd_norm.weight",
            "output_norm.weight",
            "output_res_query.weight",
        ] {
            values(split, n.into(), &[h])?;
        }
        let rows = 2 * (ce - cs) as u64;
        for n in ["loop_res_query.weight", "loop_kv_res_query.weight"] {
            values(split, n.into(), &[h, rows])?;
        }
        Ok((embed, head))
    })();
    let (embed, head) = match globals {
        Ok(g) => g,
        Err(e) => {
            refuse(items, e.0);
            return None;
        }
    };
    if !items.is_empty() {
        return None;
    }
    let windowed: Vec<bool> = layers.iter().map(|l| l.windowed).collect();
    let passes = loop_.passes as usize;
    let visits = visit_plans(&spec.visits(), &windowed, (cs, ce), passes);
    let blocks = |r: std::ops::Range<usize>| r.filter(|l| l.is_multiple_of(BLOCK_LAYERS)).count();
    let prelude_blocks = blocks(0..cs);
    let pass_blocks = blocks(cs..ce);
    let max_banks = prelude_blocks + pass_blocks + blocks(ce..spec.layers.len());
    if max_banks + 1 > MIX_VECS {
        refuse(
            items,
            format!("a walk banks {max_banks} blocks and the stream; the mix holds {MIX_VECS}"),
        );
        return None;
    }
    let layout = super::layout::Layout::of(spec);
    Some(Checked {
        layers,
        visits,
        hidden,
        vocab,
        cut,
        embed,
        head,
        ctx,
        core: (cs, ce),
        passes,
        prelude_blocks,
        max_banks,
        message_rows: pass_blocks,
        window,
        ring_rows: usize::try_from(layout.ring_rows(ctx as u64)).unwrap_or(ctx),
        taps: a0.taps as usize,
        rms_eps: spec.rms_eps,
        qk_clip: a0.qk_clip,
        rel_clip: a0.rel_clip,
        log_floor: a0.log_floor,
        log_alpha: a0.log_alpha,
        route_scale: moe.router.scale,
        gate_beta,
        value_beta,
        softcap,
        ff,
        top_k,
        experts,
        arch_line: arch_line(spec),
    })
}

/// Layer `l`'s tensors: `dims` the hidden, the experts' ff and the expert
/// count, `mx8` the layout of an MXFP8 coda stack.
fn layer_plan(
    split: &Split,
    l: usize,
    windowed: bool,
    core: bool,
    dims: (usize, usize, usize),
    mx8: NvfpLayout,
) -> Result<LayerPlan, Refusal> {
    let (hidden, ff, experts) = dims;
    let n = |s: &str| format!("blk.{l}.{s}.weight");
    let h = hidden as u64;
    let norms = if windowed {
        let g = values(split, n("attn_norm"), &[h])?;
        [g.clone(), g.clone(), g]
    } else {
        [
            values(split, n("attn_q_inp_norm"), &[h])?,
            values(split, n("attn_k_inp_norm"), &[h])?,
            values(split, n("attn_v_inp_norm"), &[h])?,
        ]
    };
    let qrg = [
        site(split, n("attn_q"), 3072, hidden)?,
        site(split, n("attn_rel"), 384, hidden)?,
        site(split, n("attn_gate"), 3072, hidden)?,
    ];
    if qrg.iter().any(|s| s.quant != qrg[0].quant) {
        return Err(Refusal(format!(
            "layer {l}: the q, relative and gate projections are {:?}, {:?} and {:?}; their one \
             launch reads one type",
            qrg[0].stored, qrg[1].stored, qrg[2].stored
        )));
    }
    let taps = |s: &str| {
        (!windowed)
            .then(|| site(split, n(s), 32, hidden))
            .transpose()
    };
    let attn_query = if l == 0 {
        None
    } else if windowed {
        Some((values(split, n("attn_res_query"), &[h])?, 1))
    } else {
        Some((values(split, n("attn_res_query"), &[h, 3])?, 3))
    };
    if core {
        for m in ["attn", "k", "v", "mlp"] {
            values(split, n(&format!("loop_{m}_down")), &[2 * h, 64])?;
            values(split, n(&format!("loop_{m}_gain")), &[64, h])?;
            values(split, n(&format!("loop_{m}_forget")), &[64, h])?;
        }
    }
    Ok(LayerPlan {
        windowed,
        core,
        norms,
        qrg,
        sink: site(split, n("attn_sink_proj"), 24, hidden)?,
        k: site(split, n("attn_k"), 1024, hidden)?,
        v: site(split, n("attn_v"), 1024, hidden)?,
        k_taps: taps("attn_k_conv_proj")?,
        v_taps: taps("attn_v_conv_proj")?,
        wo: site(split, n("attn_output"), hidden, 3072)?,
        attn_query,
        ffn_query: values(split, n("ffn_res_query"), &[h])?,
        ffn_norm: values(split, n("ffn_norm"), &[h])?,
        q_clip: values(split, n("attn_q_clip"), &[128, 24])?,
        rel_clip: values(split, n("attn_rel_clip"), &[16, 24])?,
        rel_bias: values(split, n("attn_rel_bias"), &[16, 1024])?,
        convs: [
            values(split, n("attn_k_conv"), &[4, 1024])?,
            values(split, n("attn_v_conv"), &[4, 1024])?,
            values(split, n("attn_post_conv"), &[4, h])?,
            values(split, n("ffn_post_conv"), &[4, h])?,
        ],
        router: values(split, n("ffn_gate_inp"), &[h, experts as u64])?,
        router_bias: values(
            split,
            format!("blk.{l}.exp_probs_b.bias"),
            &[experts as u64],
        )?,
        gate_up: stack(split, n("ffn_gate_up_exps"), experts, 2 * ff, hidden, mx8)?,
        down: stack(split, n("ffn_down_exps"), experts, hidden, ff, mx8)?,
    })
}

impl Checked {
    /// The rows a visit's K and V planes hold: the ring's for a windowed
    /// visit, the cache's for a windowless one.
    #[must_use]
    pub fn plane_rows(&self, windowed: bool) -> usize {
        if windowed { self.ring_rows } else { self.ctx }
    }

    /// The bytes the open makes buffers of, by class, one [`Tally::add`] a
    /// buffer the arena makes (`arena.rs`, in the same order), and the prompt
    /// pass's buffers as the rows (`pass.rs`).
    #[must_use]
    pub fn bytes(&self) -> Bytes {
        let n = self.hidden;
        let f32s = |count: usize| (count * 4) as u64;
        let mut w = Tally::default();
        for l in &self.layers {
            // The q path's weights, the sink, k, v, the taps' rows, wo.
            w.add(l.qrg.iter().map(|s| s.bytes() as u64).sum());
            for s in [
                Some(&l.sink),
                Some(&l.k),
                Some(&l.v),
                l.k_taps.as_ref(),
                l.v_taps.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                w.add(s.bytes() as u64);
            }
            w.add(l.wo.bytes() as u64);
            // The three norms (a windowed layer's one gain once), the queries,
            // the clips, the bias table, the four convs' taps, the router.
            let norms = if l.windowed { 1 } else { 3 };
            for _ in 0..norms {
                w.add(f32s(n));
            }
            if let Some((_, nq)) = &l.attn_query {
                w.add(f32s(n * nq));
            }
            w.add(f32s(n));
            w.add(f32s(n));
            w.add(f32s(24 * 128));
            w.add(f32s(24 * 16));
            w.add(f32s(16 * 1024));
            for d in [1024, 1024, n, n] {
                w.add(f32s(d * self.taps));
            }
            w.add(f32s(128 * n));
            w.add(f32s(128));
            // Each stack, then its scale row.
            for st in [&l.gate_up, &l.down] {
                w.add((st.experts * st.expert_bytes()) as u64);
                w.add(f32s(st.experts));
            }
            if l.core {
                for _ in 0..4 {
                    w.add(f32s(64 * 2 * n));
                    w.add(f32s(n * 64));
                    w.add(f32s(n * 64));
                }
            }
        }
        w.add(self.embed.bytes() as u64);
        w.add(self.head.bytes_of(self.cut) as u64);
        for _ in 0..2 {
            w.add(f32s(n));
        }
        w.add(f32s((self.core.1 - self.core.0) * 4 * n));
        let mut kv = Tally::default();
        let mut state = Tally::default();
        for v in &self.visits {
            let plane = (8 * self.plane_rows(v.windowed) * 128 * 2) as u64;
            kv.add(plane);
            kv.add(plane);
            state.add(f32s((self.taps - 1) * CONV_CHANNELS));
        }
        let mut s = Tally::default();
        for count in self.scratch_f32s() {
            s.add(f32s(count));
        }
        // The router's ticket, its ids, the combine's identity map, the pick
        // and the fault word.
        for words in [1, 8, 8, 1, bloomery_device::fault::FAULT_WORDS] {
            s.add((words * 4) as u64);
        }
        Bytes {
            weights: w,
            kv,
            state,
            scratch: s,
            rows: super::pass::tally(self),
            ..Bytes::default()
        }
    }

    /// The message banks the step keeps: one a pass the loop runs again, and
    /// at least one.
    #[must_use]
    pub fn message_banks(&self) -> usize {
        self.visits.iter().filter(|v| v.pass_end).count().max(1)
    }

    /// The f32 scratch buffers the arena makes, in its order: the bank's rows,
    /// the message banks, the mixes, the controllers' outputs and down rows,
    /// the paths' outputs, the attention's partials and output, the router's
    /// partials, logits and weights, the experts' slots, the visits' expert
    /// sums and the logits.
    #[must_use]
    pub fn scratch_f32s(&self) -> Vec<usize> {
        let n = self.hidden;
        let segments = crate::attention_geom::SEGMENT_CAP;
        vec![
            (self.max_banks + 1) * n,
            self.message_banks() * self.message_rows * n,
            3 * n,
            4 * n,
            3 * n,
            4 * crate::inkling2_geom::CONTROLLER_RANK,
            3072 + 384 + 3072,
            24,
            1024,
            1024,
            32,
            32,
            1024,
            1024,
            3072,
            3072,
            crate::kernels::rel_partials_len(segments).unwrap_or(0),
            n,
            n,
            n,
            n,
            n,
            128,
            128,
            8,
            self.top_k * 2 * self.ff,
            self.top_k * self.ff,
            self.top_k * n,
            self.visits.len() * n,
            n,
            n,
            self.cut,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Mini's walk: 48 visits, a block closed at each stored layer that is
    /// a multiple of six (eight of the visits), the later passes' 24 core
    /// visits merging messages, and two pass ends, the third pass running into
    /// the coda.
    ///
    /// Mutant: `pass_end` without its `pass + 1 < passes` term marks a third
    /// end, after the last pass.
    #[test]
    fn the_minis_walk_marks_its_boundaries_messages_and_pass_ends() {
        let spec = super::super::layout::mini();
        let lp = spec.loop_.expect("the Mini loops");
        let windowed: Vec<bool> = spec
            .layers
            .iter()
            .map(|l| matches!(&l.mixer, Mixer::RelBias(a) if a.window.is_some()))
            .collect();
        let plans = visit_plans(
            &spec.visits(),
            &windowed,
            (lp.core_start as usize, lp.core_end as usize),
            lp.passes as usize,
        );
        assert_eq!(plans.len(), 48);
        assert_eq!(plans.iter().filter(|v| v.boundary).count(), 8);
        assert_eq!(plans.iter().filter(|v| v.message).count(), 24);
        let ends: Vec<usize> = (0..plans.len()).filter(|&i| plans[i].pass_end).collect();
        assert_eq!(ends, [17, 29]);
    }
}
