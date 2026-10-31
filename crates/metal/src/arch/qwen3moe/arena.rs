//! The qwen35moe arena: every buffer one token touches, allocated when the
//! session opens and never in a step.
//!
//! The weights are copied from the file into buffers of their own. A layer's
//! P1 projection is one buffer per weight type, each the segments of one
//! dispatch. The load runs on [`LOAD_WORKERS`] threads at most, each taking
//! the next of the embedding, the head and the layers in the file's order, so
//! the drive serves several reads at once while the host repacks and copies
//! what came in; every buffer holds what a one-thread load writes, and a
//! refusal is the first in that order.
//!
//! The activations, the caches, the recurrent stores, the routers'
//! and the attention's tickets, the host's id slot, the step record slots and
//! the fault words are the session's. Every name, type and length comes from the
//! layout [`super::check`] resolved from the file's header (the plan, the
//! file's shapes and a launch's geometry): a shape or a type no launch runs was
//! refused there, by name, before any buffer was allocated, so the loader
//! checks nothing but that the model file gives a tensor's bytes.
//!
//! Two tokens may be committed at once, the second reading what the first
//! writes. What the second would overwrite before the first is done is held
//! twice: the step record (the head's id and the fault word) has a slot per
//! position parity, and so does the delta state, whose lanes a token reads by
//! its position's parity and writes by the next position's.

use core::ops::Range;
use core::sync::atomic::{AtomicUsize, Ordering};
use std::borrow::Cow;

use bloomery_device::fault::LAYER_HEAD;
use bloomery_device::{Elem, Elems};
use chain::rope::{Direction, RopeSpec, RopeTable};
use gguf::quant::GgmlType;
use model::part::Part;
use model_hf::Weights;
use model_hf::model_file::Error as WeightError;

use super::check::{
    AttnLayout, ConvLayout, DeltaLayout, FfnLayout, LayerLayout, Layout, MixerLayout, P1Layout,
    PostLayout, RopeLayout, Site, Values,
};
use super::count::{FfnKind, GroupBytes, GroupKind, Layer, Projection, Residual};
use super::ops::Half;
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::fault::MetalFaultWord;
use crate::geom::Geometry;
use crate::kernels::{HeadMap, Quant, Segment, segments_for_window};
use crate::residency::Alloc;

const WHAT: &str = "qwen35moe arena";

/// The most threads one open loads with: enough reads in flight to keep the
/// drive at its streaming rate, each beside its own repack and copy.
const LOAD_WORKERS: usize = 8;

/// Lanes of the delta state: the token at `pos` reads lane [`lane_of`]`(pos)`
/// and writes the next position's, so a token that ran and was not taken
/// leaves the lane its replacement reads as it was.
pub(super) const LANES: usize = 2;

/// The lane words: word `i` names lane `i`.
const LANE_WORDS: [u32; LANES] = [0, 1];

/// Step record slots: the token at `pos` writes its id and raises its fault
/// word in slot [`slot_of`]`(pos)`, so the host reads one token's record
/// while the next token writes the other.
pub(super) const RECORD_SLOTS: usize = 2;

/// The step record slot of the token at `pos`.
pub(super) fn slot_of(pos: u32) -> usize {
    pos as usize % RECORD_SLOTS
}

/// The step record slot of the token before `pos`: the slot whose id a token
/// fed on the device embeds.
pub(super) fn slot_before(pos: u32) -> usize {
    (pos as usize + RECORD_SLOTS - 1) % RECORD_SLOTS
}

/// The delta lane the token at `pos` reads; it writes `lane_of(pos + 1)`.
pub(super) fn lane_of(pos: u32) -> usize {
    pos as usize % LANES
}

/// One P1 dispatch: the segments of one weight type, in one buffer.
pub(super) struct P1Part {
    pub(super) quant: Quant,
    pub(super) weights: MetalBuffer<u8>,
    pub(super) segments: Vec<Segment>,
    /// One past the last output value a segment writes.
    pub(super) out_end: usize,
}

/// A layer's P1: the norm its load applies, and one dispatch per weight type
/// of its projections, all in one group.
pub(super) struct P1 {
    pub(super) norm: MetalBuffer<f32>,
    pub(super) parts: Vec<P1Part>,
}

impl P1 {
    /// The count's projection: the dispatches of the group.
    fn projection(&self) -> Projection {
        Projection::of(self.parts.len()).expect("a P1 holds at least one dispatch")
    }
}

/// A gated delta layer's buffers.
pub(super) struct DeltaBufs {
    pub(super) p1: P1,
    /// Where the conv input, z, β and α sit in the P1 output.
    pub(super) qkv: Range<usize>,
    pub(super) z: Range<usize>,
    pub(super) beta: Range<usize>,
    pub(super) alpha: Range<usize>,
    pub(super) conv: MetalBuffer<f32>,
    pub(super) dt_bias: MetalBuffer<f32>,
    pub(super) ssm_a: MetalBuffer<f32>,
    pub(super) ssm_norm: MetalBuffer<f32>,
    /// `ssm_out`, in a type the head-norm projection reads.
    pub(super) out: MetalBuffer<u8>,
    /// The out projection's type: the head-norm launch reads it.
    pub(super) out_quant: Quant,
    pub(super) state: MetalBuffer<f32>,
    pub(super) ring: MetalBuffer<f32>,
    pub(super) n_k: usize,
    pub(super) map: HeadMap,
}

/// A gated attention layer's buffers.
pub(super) struct AttnBufs {
    pub(super) p1: P1,
    /// Where the query with its gate, the key and the value sit in the P1
    /// output.
    pub(super) q: Range<usize>,
    pub(super) k: Range<usize>,
    pub(super) v: Range<usize>,
    /// Where the output gate of one value a query head sits in the P1 output,
    /// after the value, on a layer that has one.
    pub(super) gate: Option<Range<usize>>,
    /// The positions the layer attends; none: every position.
    pub(super) window: Option<u32>,
    /// The layer's rope table, by place in [`Arena::ropes`].
    pub(super) rope: usize,
    pub(super) q_norm: MetalBuffer<f32>,
    pub(super) k_norm: MetalBuffer<f32>,
    pub(super) out: MetalBuffer<u8>,
    pub(super) out_quant: Quant,
    pub(super) k_cache: MetalBuffer<u16>,
    pub(super) v_cache: MetalBuffer<u16>,
}

/// A gated short conv layer's buffers: its input projection is one segment of
/// the P1, and the ring keeps the conv's inputs of the last positions.
pub(super) struct ConvBufs {
    pub(super) p1: P1,
    /// Where `B | C | x`, `3 · hidden` values, sit in the P1 output.
    pub(super) bcx: Range<usize>,
    /// The filters, `[hidden][taps]`, the oldest tap first.
    pub(super) conv: MetalBuffer<f32>,
    /// `out_proj`, in a type the mat-vec reads.
    pub(super) out: MetalBuffer<u8>,
    /// The out projection's type: the launch reads it.
    pub(super) out_quant: Quant,
    /// The ring: `ring_rows` rows of `hidden` f32.
    pub(super) ring: MetalBuffer<f32>,
    pub(super) taps: usize,
    pub(super) ring_rows: usize,
}

/// A layer's mixer.
pub(super) enum MixerBufs {
    Delta(DeltaBufs),
    Attention(AttnBufs),
    ShortConv(ConvBufs),
}

impl MixerBufs {
    /// The mixer's P1.
    pub(super) fn p1(&self) -> &P1 {
        match self {
            Self::Delta(d) => &d.p1,
            Self::Attention(a) => &a.p1,
            Self::ShortConv(c) => &c.p1,
        }
    }
}

/// A layer's sandwich: the gain of the norm each half's output goes through
/// before it joins the stream, `x += add[i] · gain[i] ⊙ rms_norm(out)`, the
/// mixer's half first (the order of `SandwichPlan`'s `norms`). The epsilon and
/// the scales are the plan's (`SandwichPlan`).
pub(super) struct PostBufs {
    pub(super) gains: [MetalBuffer<f32>; 2],
}

/// A shared expert's router row and weights, each of the type the launches
/// read it in: its stacks', or a pair of the experts' `Apart` tables.
pub(super) struct SharedBufs {
    pub(super) gate_inp: MetalBuffer<f32>,
    /// The type of the gate and the up.
    pub(super) gate_up_quant: Quant,
    /// The type of the down.
    pub(super) down_quant: Quant,
    pub(super) gate: MetalBuffer<u8>,
    pub(super) up: MetalBuffer<u8>,
    pub(super) down: MetalBuffer<u8>,
}

/// A routed layer's router: its weight and the buffers it selects through.
pub(super) struct RouterBufs {
    pub(super) gate_inp: MetalBuffer<f32>,
    pub(super) logits: MetalBuffer<f32>,
    pub(super) partials: MetalBuffer<f32>,
    pub(super) ticket: MetalBuffer<u32>,
}

/// The id of a dense FFN's one slot: expert 0 of its one-expert stacks.
const DENSE_ID: u32 = 0;

/// The route weight of a dense FFN's one slot.
const DENSE_WEIGHT: f32 = 1.0;

/// A layer's FFN half: the router and its outputs, and the experts. Each layer
/// keeps its own router outputs, so every layer's picks are readable after a
/// step. A dense layer has no router: its slot's id and weight are the fixed
/// route [`DENSE_ID`] and [`DENSE_WEIGHT`], laid once at the open, and its gate,
/// up and down are the one expert of the stacks.
pub(super) struct FfnBufs {
    pub(super) norm: MetalBuffer<f32>,
    /// The router; none for a dense FFN.
    pub(super) router: Option<RouterBufs>,
    pub(super) gate_up_quant: Quant,
    pub(super) gate: MetalBuffer<u8>,
    pub(super) up: MetalBuffer<u8>,
    pub(super) down_quant: Quant,
    pub(super) down: MetalBuffer<u8>,
    /// The shared expert, on a router that runs one.
    pub(super) shared: Option<SharedBufs>,
    /// The slots' ids and route weights: the router's picks, or the dense
    /// FFN's fixed route.
    pub(super) ids: MetalBuffer<u32>,
    pub(super) weights: MetalBuffer<f32>,
}

/// One layer's buffers.
pub(super) struct LayerBufs {
    pub(super) mixer: MixerBufs,
    pub(super) ffn: FfnBufs,
    /// The sandwich; none on a layer that adds each half to the stream.
    pub(super) post: Option<PostBufs>,
}

/// Every buffer of the session.
pub(super) struct Arena {
    /// The launch geometry of the instance the file selects: every length
    /// below that is not the file's own is derived from it.
    pub(super) geom: Geometry,
    pub(super) hidden: usize,
    pub(super) vocab: usize,
    pub(super) ctx: usize,
    pub(super) eps: f32,
    pub(super) top_k: usize,
    pub(super) n_experts: usize,
    pub(super) layers: Vec<LayerBufs>,
    /// The embedding table's type: the embed launch reads it.
    pub(super) embed_quant: Quant,
    pub(super) embed: MetalBuffer<u8>,
    /// The head's type: the head launch reads it.
    pub(super) head_quant: Quant,
    pub(super) head_w: MetalBuffer<u8>,
    pub(super) head_norm: MetalBuffer<f32>,
    /// The residual stream, updated in place by every layer.
    pub(super) x: MetalBuffer<f32>,
    /// The P1 output of the layer running.
    pub(super) proj: MetalBuffer<f32>,
    /// The delta step's gated numerator and its partial sums of squares.
    pub(super) g: MetalBuffer<f32>,
    pub(super) delta_partials: MetalBuffer<f32>,
    /// The attention output after its gate, and its segment partials.
    pub(super) attn: MetalBuffer<f32>,
    pub(super) attn_partials: MetalBuffer<f32>,
    /// The attention fold's ticket, one word: the flash's last threadgroup
    /// resets it, so every launch finds it zero.
    pub(super) attn_ticket: MetalBuffer<u32>,
    /// A short conv's output, `hidden` values; none for a file with no short
    /// conv layer.
    pub(super) conv_y: Option<MetalBuffer<f32>>,
    /// What a half's out projection writes before its join, `hidden` values;
    /// none for a file with no sandwich.
    pub(super) join: Option<MetalBuffer<f32>>,
    /// What a join that hands on ([`Arena::hands_on`]) writes for the launch
    /// after it: the new stream times that launch's norm gain, `hidden`
    /// values, then the norm's scale; none for a file with no sandwich.
    pub(super) xn: Option<MetalBuffer<f32>>,
    /// The rope tables, one for each distinct base and factors the attention
    /// layers have: `ctx` rows of `AttnGeom::rot` interleaved `(cos, sin)`.
    pub(super) ropes: Vec<MetalBuffer<f32>>,
    /// The experts' activations, one row of `ExpertsGeom::k` per slot.
    pub(super) h: MetalBuffer<f32>,
    pub(super) logits: MetalBuffer<f32>,
    /// The head's pick, one per step record slot — or, for a token whose
    /// pick nobody reads, the id the host writes there for the token ahead
    /// to embed.
    pub(super) step_id: MetalBuffer<u32>,
    /// The id a host-fed token embeds, written by the host.
    pub(super) id_in: MetalBuffer<u32>,
    /// The delta step's lane words, [`LANE_WORDS`].
    pub(super) lane: MetalBuffer<u32>,
    /// The fault word of each step record slot.
    pub(super) faults: [MetalFaultWord; RECORD_SLOTS],
}

/// The fault word's layer of model layer `l`.
pub(super) fn layer_word(l: usize) -> u32 {
    u32::try_from(l).expect("a layer index fits u32")
}

/// A launch that norms the stream on load right after a sandwich join.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Reader {
    /// Layer `l`'s P1, after layer `l − 1`'s FFN join.
    P1(usize),
    /// Layer `l`'s gate-up pair, after its mixer join.
    Pair(usize),
    /// The head's logits, after the last layer's FFN join.
    Head,
}

impl Reader {
    /// The reader after layer `l`'s join of `half`, in a model of `layers`
    /// layers.
    pub(super) fn after(layers: usize, l: usize, half: Half) -> Self {
        match half {
            Half::Mixer => Self::Pair(l),
            Half::Ffn if l + 1 < layers => Self::P1(l + 1),
            Half::Ffn => Self::Head,
        }
    }

    /// The fault word's layer the reader's norm raises at.
    pub(super) fn layer(self) -> u32 {
        match self {
            Self::P1(l) | Self::Pair(l) => layer_word(l),
            Self::Head => LAYER_HEAD,
        }
    }
}

/// `reader`'s weight type and norm gain, when it is one launch of a type with
/// a scaled entry: a P1 of one part, a pair whose shared expert (if any) is
/// its stacks' type, the head; none of F32.
pub(super) fn reader_norm<'a>(
    layers: &'a [LayerBufs],
    head: (Quant, &'a MetalBuffer<f32>),
    reader: Reader,
) -> Option<(Quant, &'a MetalBuffer<f32>)> {
    let (quant, gain) = match reader {
        Reader::P1(l) => {
            let p1 = layers.get(l)?.mixer.p1();
            let [part] = p1.parts.as_slice() else {
                return None;
            };
            (part.quant, &p1.norm)
        }
        Reader::Pair(l) => {
            let f = &layers.get(l)?.ffn;
            if f.shared
                .as_ref()
                .is_some_and(|s| s.gate_up_quant != f.gate_up_quant)
            {
                return None;
            }
            (f.gate_up_quant, &f.norm)
        }
        Reader::Head => head,
    };
    (quant != Quant::F32).then_some((quant, gain))
}

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// Refusal `e` of layer `l`, naming the layer.
fn at_layer(l: usize, e: SessionError) -> SessionError {
    match e {
        SessionError::Shape { what, detail } => SessionError::Shape {
            what,
            detail: format!("layer {l}: {detail}"),
        },
        other => other,
    }
}

/// The file's tensors, copied into buffers on `device`. Every tensor named
/// here was judged against the header by [`super::check`]; what the loader
/// still refuses is a tensor whose bytes the model file cannot give.
struct Loader<'a> {
    device: Alloc<'a>,
    weights: &'a Weights<'a>,
    geom: &'a Geometry,
}

/// `spans` joined into one run: one span as it is, several copied together.
fn joined<'s>(spans: &[&'s [u8]]) -> Cow<'s, [u8]> {
    match spans {
        [one] => Cow::Borrowed(*one),
        many => Cow::Owned(many.concat()),
    }
}

/// `spans` written one after another into `buffer` from byte `at`; returns the
/// byte after the last.
fn put(buffer: &mut MetalBuffer<u8>, at: usize, spans: &[&[u8]]) -> Result<usize, SessionError> {
    let mut at = at;
    for span in spans {
        buffer.write_at(Elems(at), span)?;
        at += span.len();
    }
    Ok(at)
}

impl<'a> Loader<'a> {
    /// The part of tensor `name`: its bytes in the type its header declares,
    /// whether the file is a GGUF split (the mapped bytes, lent) or a
    /// checkpoint folder (the bytes its source makes of the stored planes).
    fn part(&self, name: &str) -> Result<Part<'a>, SessionError> {
        self.weights.bytes(name).map_err(|e| match e {
            WeightError::Unknown { .. } => refuse(format!("the file has no tensor {name}")),
            e => refuse(format!("{name}: {e}")),
        })
    }

    /// The f32 values of `v`: an F32 tensor as it is, or MLX's affine 8-bit
    /// tensor of the same values — BF16- or F16-scaled (`MLX_A8`, `MLX_A8H`)
    /// — dequantized exactly at open through the host reader (`s·q + b` in
    /// f32, the pair's product exact). The reader's own refusals (a
    /// non-finite scale or bias) name the tensor and its block.
    fn values(&self, v: &Values) -> Result<MetalBuffer<f32>, SessionError> {
        let values = self.host_values(v)?;
        let mut buffer = self.device.buffer(Elems(v.len))?;
        buffer.write(&values)?;
        Ok(buffer)
    }

    /// The f32 values of `v`, as [`Loader::values`] reads them, kept on the
    /// host: what a table the arena builds from them is made of.
    fn host_values(&self, v: &Values) -> Result<Vec<f32>, SessionError> {
        let part = self.part(&v.name)?;
        let bytes = joined(&part.spans());
        let mut values = vec![0f32; v.len];
        match v.ty {
            GgmlType::F32 => {
                for (x, b) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *x = f32::from_le_bytes(*b);
                }
            }
            GgmlType::MLX_A8 | GgmlType::MLX_A8H => {
                gguf::quant::dequant_row(v.ty, &bytes, &mut values)
                    .map_err(|e| refuse(format!("{}: {e}", v.name)))?;
            }
            other => {
                return Err(refuse(format!(
                    "{} is {other:?}; the step reads F32, MLX_A8 or MLX_A8H",
                    v.name
                )));
            }
        }
        Ok(values)
    }

    /// The rope table `r` describes, `ctx` positions of `rot` values: the
    /// window rope of its base, dividing by the file's factors when it has
    /// them. A factor the rope cannot divide by is refused by name here, where
    /// the values are read.
    fn rope(&self, r: &RopeLayout, ctx: usize, rot: usize) -> Result<Vec<f32>, SessionError> {
        let mut spec = RopeSpec::window(r.base, rot);
        if let Some(factors) = &r.factors {
            spec = spec
                .with_factors(&self.host_values(factors)?)
                .map_err(|e| refuse(format!("{}: {e}", factors.name)))?;
        }
        let table = RopeTable::new(&spec).map_err(|e| refuse(format!("the rope table: {e}")))?;
        Ok(rope_rows(&table, ctx, rot))
    }

    /// Site `s` as its own buffer, written from its part.
    fn site(&self, s: &Site) -> Result<MetalBuffer<u8>, SessionError> {
        let part = self.part(&s.name)?;
        let spans = part.spans();
        let mut buffer = self
            .device
            .buffer(Elems(spans.iter().map(|b| b.len()).sum()))?;
        put(&mut buffer, 0, &spans)?;
        Ok(buffer)
    }

    /// The P1 of `p`: one dispatch per weight type, its segments in the order
    /// the layout gives, each site's output at the offset its place gives and
    /// each site's bytes written straight into the dispatch's buffer. Returns
    /// the P1 and every site's output range.
    fn p1(&self, p: &P1Layout) -> Result<(P1, Vec<Range<usize>>), SessionError> {
        let mut parts: Vec<P1Part> = Vec::with_capacity(p.dispatches.len());
        for d in &p.dispatches {
            let held: Vec<Part<'a>> = d
                .members
                .iter()
                .map(|&i| self.part(&p.sites[i].name))
                .collect::<Result<_, _>>()?;
            let spans: Vec<Vec<&[u8]>> = held.iter().map(Part::spans).collect();
            let total: usize = spans.iter().flatten().map(|b| b.len()).sum();
            let mut weights = self.device.buffer(Elems(total))?;
            let mut segments = Vec::with_capacity(d.members.len());
            let mut at = 0;
            for (&i, s) in d.members.iter().zip(&spans) {
                let site = &p.sites[i];
                segments.push(Segment {
                    cols: site.cols,
                    rows: site.rows,
                    weight_blocks: at / d.quant.block_bytes(),
                    out_offset: p.range(i).start,
                });
                at = put(&mut weights, at, s)?;
            }
            parts.push(P1Part {
                quant: d.quant,
                weights,
                segments,
                out_end: d.out_end,
            });
        }
        let norm = self.values(&p.norm)?;
        Ok((
            P1 { norm, parts },
            (0..p.sites.len()).map(|i| p.range(i)).collect(),
        ))
    }

    /// One layer's buffers, from its layout.
    fn layer(&self, layer: &LayerLayout) -> Result<LayerBufs, SessionError> {
        let mixer = match &layer.mixer {
            MixerLayout::Delta(d) => MixerBufs::Delta(delta(self, d)?),
            MixerLayout::Attention(a) => MixerBufs::Attention(attention(self, a)?),
            MixerLayout::ShortConv(c) => MixerBufs::ShortConv(short_conv(self, c)?),
        };
        Ok(LayerBufs {
            mixer,
            ffn: ffn(self, self.device, &layer.ffn)?,
            post: layer.post.as_ref().map(|p| post(self, p)).transpose()?,
        })
    }

    /// Job `j` of the load: 0 the embedding, 1 the head's weights, then
    /// layer `j - 2`.
    fn job(&self, g: &Layout, j: usize) -> Result<Loaded, SessionError> {
        match j {
            0 => self.site(&g.embed).map(Loaded::Site),
            1 => self.site(&g.head).map(Loaded::Site),
            j => {
                let l = j - 2;
                self.layer(&g.layers[l])
                    .map(|b| Loaded::Layer(Box::new(b)))
                    .map_err(|e| at_layer(l, e))
            }
        }
    }

    /// The embedding, the head's weights and every layer, loaded by up to
    /// [`LOAD_WORKERS`] threads ([`in_order`]): the refusal returned is the
    /// first a load in order meets, and each buffer holds the bytes its
    /// tensor gives, whichever thread wrote it.
    fn weights(&self, g: &Layout) -> Result<Loads, SessionError> {
        let workers = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(LOAD_WORKERS);
        let mut loaded = in_order(g.layers.len() + 2, workers, |j| self.job(g, j))?.into_iter();
        let mut site = || match loaded.next() {
            Some(Loaded::Site(b)) => Ok(b),
            _ => Err(refuse(
                "the load's first two jobs are the embedding and the head",
            )),
        };
        let (embed, head) = (site()?, site()?);
        let layers = loaded
            .map(|r| match r {
                Loaded::Layer(b) => Ok(*b),
                Loaded::Site(_) => Err(refuse("a layer job loaded a site")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Loads {
            embed,
            head,
            layers,
        })
    }
}

/// The weights the load's jobs made: the embedding, the head's and each
/// layer's, in layer order.
struct Loads {
    embed: MetalBuffer<u8>,
    head: MetalBuffer<u8>,
    layers: Vec<LayerBufs>,
}

/// Jobs `0..jobs` run by up to `workers` threads, each taking the next job in
/// order; their results in job order, or the error of the earliest job that
/// failed. A worker takes no job past the earliest failure, and every job
/// before it was taken, so the error is the one a run in order on one thread
/// returns.
fn in_order<T: Send, E: Send>(
    jobs: usize,
    workers: usize,
    job: impl Fn(usize) -> Result<T, E> + Sync,
) -> Result<Vec<T>, E> {
    let next = AtomicUsize::new(0);
    let failed = AtomicUsize::new(usize::MAX);
    let work = || {
        let mut done = Vec::new();
        loop {
            let j = next.fetch_add(1, Ordering::Relaxed);
            if j >= jobs || j > failed.load(Ordering::Relaxed) {
                return done;
            }
            let r = job(j);
            if r.is_err() {
                failed.fetch_min(j, Ordering::Relaxed);
            }
            done.push((j, r));
        }
    };
    let mut slots: Vec<Option<Result<T, E>>> = std::iter::repeat_with(|| None).take(jobs).collect();
    std::thread::scope(|s| {
        let threads: Vec<_> = (1..workers.clamp(1, jobs.max(1)))
            .map(|_| s.spawn(work))
            .collect();
        let mut runs = vec![work()];
        for t in threads {
            runs.push(t.join().unwrap_or_else(|p| std::panic::resume_unwind(p)));
        }
        for (j, r) in runs.into_iter().flatten() {
            slots[j] = Some(r);
        }
    });
    // Every job up to the earliest failure ran; the first `None` is past it.
    let mut out = Vec::with_capacity(jobs);
    for r in slots.into_iter().map_while(|r| r) {
        out.push(r?);
    }
    assert_eq!(
        out.len(),
        jobs,
        "in_order: a job past no failure did not run"
    );
    Ok(out)
}

/// What one job of the load made.
enum Loaded {
    Site(MetalBuffer<u8>),
    Layer(Box<LayerBufs>),
}

/// A buffer of `n` elements, zeroed.
fn zeroed<T: bloomery_device::Elem>(
    device: Alloc<'_>,
    n: usize,
) -> Result<MetalBuffer<T>, SessionError> {
    let mut buffer = device.buffer(Elems(n))?;
    buffer.zero();
    Ok(buffer)
}

/// `n` as the byte rule's u64, a count the arena already holds.
fn count_u64(n: usize, what: &str) -> u64 {
    u64::try_from(n).unwrap_or_else(|_| panic!("the byte rule: {what} do not fit u64"))
}

/// The bytes `n` elements of `T` take.
fn bytes_of_len<T: Elem>(n: usize) -> u64 {
    count_u64(n, "elements")
        * u64::try_from(core::mem::size_of::<T>()).expect("an element's size fits u64")
}

/// The bytes buffer `b` holds.
fn bytes_of<T: Elem>(b: &MetalBuffer<T>) -> u64 {
    bytes_of_len::<T>(b.pages().len().get())
}

/// Layer `layer`'s attention buffers for the byte rule, or the named
/// refusal for a layer of the other mixer.
fn attn_of(layer: &LayerBufs, l: usize, kind: GroupKind) -> Result<&AttnBufs, SessionError> {
    match &layer.mixer {
        MixerBufs::Attention(a) => Ok(a),
        MixerBufs::Delta(_) | MixerBufs::ShortConv(_) => {
            Err(refuse(format!("{kind:?}: layer {l} is no attention layer")))
        }
    }
}

/// Layer `layer`'s delta buffers for the byte rule, or the named refusal.
fn delta_of(layer: &LayerBufs, l: usize, kind: GroupKind) -> Result<&DeltaBufs, SessionError> {
    match &layer.mixer {
        MixerBufs::Delta(d) => Ok(d),
        MixerBufs::Attention(_) | MixerBufs::ShortConv(_) => {
            Err(refuse(format!("{kind:?}: layer {l} is no delta layer")))
        }
    }
}

/// Layer `layer`'s short conv buffers for the byte rule, or the named
/// refusal.
fn conv_of(layer: &LayerBufs, l: usize, kind: GroupKind) -> Result<&ConvBufs, SessionError> {
    match &layer.mixer {
        MixerBufs::ShortConv(c) => Ok(c),
        MixerBufs::Delta(_) | MixerBufs::Attention(_) => Err(refuse(format!(
            "{kind:?}: layer {l} is no short conv layer"
        ))),
    }
}

/// The rope table of `ctx` positions: per position, `rot` / 2 pairs
/// `(cos θ, sin θ)` with θ = `pos · base^(−2i/rot)`, θ advanced by repeated
/// multiplication in f32 as ggml's `rope_yarn` does with no YaRN — chain's
/// [`RopeTable`] over a window spec, the one owner of the recipe both
/// backends' tables come from.
fn rope_rows(table: &RopeTable, ctx: usize, rot: usize) -> Vec<f32> {
    let positions = u32::try_from(ctx).expect("the check made the cache's rows fit u32");
    let mut out = Vec::with_capacity(ctx * rot);
    for pos in 0..positions {
        table.push(pos, Direction::Forward, &mut out);
    }
    out
}

/// The stream-sized buffers a layer's out projection touches: the residual,
/// read and written (2), or on a sandwich the join buffer, written (1).
fn stream_of(layer: &LayerBufs) -> u64 {
    if layer.post.is_some() { 1 } else { 2 }
}

impl Arena {
    /// The arena of `g` (the layout [`super::check`] resolved from the model
    /// file's header, at the launch geometry `geom` of the file's instance):
    /// each weight copied from `weights` into a buffer of its own, and every
    /// buffer a token touches made, all zeroed but the weights, the rope table
    /// and the lane words.
    pub(super) fn open(
        device: Alloc<'_>,
        weights: &Weights<'_>,
        geom: &Geometry,
        g: &Layout,
    ) -> Result<Arena, SessionError> {
        let (hidden, vocab, ctx) = (g.hidden, g.vocab, g.ctx);
        let load = Loader {
            device,
            weights,
            geom,
        };
        let Loads {
            embed,
            head: head_w,
            layers,
        } = load.weights(g)?;
        let head_norm = load.values(&g.head_norm)?;
        let mut ropes = Vec::with_capacity(g.ropes.len());
        for r in &g.ropes {
            let mut table = device.buffer(Elems(g.rope_len))?;
            table.write(&load.rope(r, ctx, geom.attn.rot)?)?;
            ropes.push(table);
        }
        let mut lane = device.buffer(Elems(LANES))?;
        lane.write(&LANE_WORDS)?;
        Ok(Arena {
            geom: *geom,
            hidden,
            vocab,
            ctx,
            eps: g.eps,
            top_k: g.top_k,
            n_experts: g.n_experts,
            layers,
            embed_quant: g.embed.quant,
            embed,
            head_quant: g.head.quant,
            head_w,
            head_norm,
            x: zeroed(device, hidden)?,
            proj: zeroed(device, g.proj)?,
            g: zeroed(device, geom.delta.value_heads * geom.delta.head_cols)?,
            delta_partials: zeroed(device, geom.delta.partials_stride)?,
            attn: zeroed(device, g.attn_out)?,
            attn_partials: zeroed(device, g.attn_partials)?,
            attn_ticket: zeroed(device, 1)?,
            conv_y: g.has_conv().then(|| zeroed(device, hidden)).transpose()?,
            join: g.has_join().then(|| zeroed(device, hidden)).transpose()?,
            xn: g
                .has_join()
                .then(|| zeroed(device, hidden + 1))
                .transpose()?,
            ropes,
            h: zeroed(device, geom.experts.slots * geom.experts.k)?,
            logits: zeroed(device, vocab)?,
            step_id: zeroed(device, RECORD_SLOTS)?,
            id_in: zeroed(device, 1)?,
            lane,
            faults: [
                MetalFaultWord::registered(device)?,
                MetalFaultWord::registered(device)?,
            ],
        })
    }

    /// The token as the count models it: each layer's kind and whether its
    /// P1 is two dispatches.
    pub(super) fn token_layers(&self) -> Vec<Layer> {
        self.layers
            .iter()
            .map(|l| {
                let ffn = if l.ffn.router.is_some() {
                    FfnKind::Routed
                } else {
                    FfnKind::Dense
                };
                let residual = if l.post.is_some() {
                    Residual::Sandwich
                } else {
                    Residual::Plain
                };
                match &l.mixer {
                    MixerBufs::Delta(d) => Layer::Delta {
                        qkv: d.p1.projection(),
                        ffn,
                        residual,
                    },
                    MixerBufs::Attention(a) => Layer::Attention {
                        v: a.p1.projection(),
                        ffn,
                        residual,
                    },
                    MixerBufs::ShortConv(c) => Layer::ShortConv {
                        in_proj: c.p1.projection(),
                        ffn,
                        residual,
                    },
                }
            })
            .collect()
    }

    /// Whether layer `l`'s joins hand their readers the normed input: a
    /// sandwich layer whose two readers ([`Reader::after`]) each have a
    /// scaled entry ([`reader_norm`]). The join writes the new stream times
    /// the reader's gain and the reader's scale into [`Arena::xn`], and the
    /// reader takes them in place of its norm on load. A layer's two joins
    /// hand on together; the join and its reader both ask here.
    pub(super) fn hands_on(&self, l: usize) -> bool {
        let Some(layer) = self.layers.get(l) else {
            return false;
        };
        let head = (self.head_quant, &self.head_norm);
        layer.post.is_some()
            && self.xn.is_some()
            && [Half::Mixer, Half::Ffn].into_iter().all(|half| {
                reader_norm(
                    &self.layers,
                    head,
                    Reader::after(self.layers.len(), l, half),
                )
                .is_some()
            })
    }

    /// The scale word's bytes `reader` reads besides its input: 4 when it
    /// takes its join's, else none.
    fn scale_word(&self, reader: Reader) -> u64 {
        if self.reads_scaled(reader) { 4 } else { 0 }
    }

    /// Whether `reader` takes the input its join handed on.
    pub(super) fn reads_scaled(&self, reader: Reader) -> bool {
        match reader {
            Reader::P1(0) => false,
            Reader::P1(l) => self.hands_on(l - 1),
            Reader::Pair(l) => self.hands_on(l),
            Reader::Head => self
                .layers
                .len()
                .checked_sub(1)
                .is_some_and(|l| self.hands_on(l)),
        }
    }

    /// The bytes the group `kind` of layer `layer` of the token at `pos`
    /// moves: the one owner of the per-group byte rule. `ids` is that
    /// layer's router ids as the step left them (the routed picks, then the
    /// shared expert's); the expert groups read them, every other group
    /// takes none and is refused a slice.
    ///
    /// What is counted, each side once: the weights the group's launches
    /// read (the embedding's one row, each routed pick's slice of its
    /// stack, every other weight whole); the activations the group reads
    /// and writes (the residual, the projections' output, the experts'
    /// rows, the norms, the logits and the router's small outputs); and the
    /// state the group reads and writes (the attention's key and value
    /// rows — the live keys of its window read, the token's own row written
    /// — the delta state's read and written lanes with the conv ring, and a
    /// short conv's ring rows).
    pub(super) fn group_bytes(
        &self,
        kind: GroupKind,
        layer: usize,
        pos: u32,
        ids: &[u32],
    ) -> Result<GroupBytes, SessionError> {
        let f32s = |n: usize| bytes_of_len::<f32>(n);
        let no_ids = || {
            (ids.is_empty()).then_some(()).ok_or_else(|| {
                refuse(format!(
                    "{kind:?}: router ids belong to the expert groups, which read them"
                ))
            })
        };
        let bufs = || {
            self.layers
                .get(layer)
                .ok_or_else(|| refuse(format!("{kind:?}: no layer {layer}")))
        };
        match kind {
            GroupKind::Embed => {
                no_ids()?;
                let row = bytes_of(&self.embed) / count_u64(self.vocab, "the embedding's rows");
                if row * count_u64(self.vocab, "the embedding's rows") != bytes_of(&self.embed) {
                    return Err(refuse(
                        "the embedding table holds no whole row a vocabulary shares",
                    ));
                }
                Ok(GroupBytes {
                    weights: row,
                    // The id slot read, the residual written.
                    acts: 4 + f32s(self.hidden),
                    state: 0,
                })
            }
            GroupKind::AttnP1 | GroupKind::DeltaP1 | GroupKind::ConvP1 => {
                no_ids()?;
                let l = bufs()?;
                let p1 = match kind {
                    GroupKind::AttnP1 => &attn_of(l, layer, kind)?.p1,
                    GroupKind::ConvP1 => &conv_of(l, layer, kind)?.p1,
                    _ => &delta_of(l, layer, kind)?.p1,
                };
                let out_end = p1.parts.iter().map(|p| p.out_end).max().unwrap_or(0);
                Ok(GroupBytes {
                    // The norm's gain with its layer, read here or by the join
                    // that hands on to this P1.
                    weights: p1.parts.iter().map(|p| bytes_of(&p.weights)).sum::<u64>()
                        + bytes_of(&p1.norm),
                    // The residual read (or the join's normed input and its
                    // scale), the projection's output written.
                    acts: f32s(self.hidden) + f32s(out_end) + self.scale_word(Reader::P1(layer)),
                    state: 0,
                })
            }
            GroupKind::AttnFlash => {
                no_ids()?;
                let a = attn_of(bufs()?, layer, kind)?;
                // The keys the layer attends: the last `window` of the live
                // keys, or all of them.
                let live = pos + 1;
                let n_keys = u64::from(a.window.map_or(live, |w| live.min(w)));
                let attn = &self.geom.attn;
                let kv_row = count_u64(attn.kv_heads * attn.head, "a cache row") * 2;
                let segments = segments_for_window(live, a.window);
                let parts = attn
                    .partials_len(segments as usize)
                    .ok_or_else(|| refuse("the attention partials do not fit"))?;
                // The flash writes the gated output within one segment;
                // past it, the partials its own fold then reads back.
                let wrote = if segments == 1 {
                    bytes_of(&self.attn)
                } else {
                    f32s(parts) + bytes_of(&self.attn)
                };
                Ok(GroupBytes {
                    // The query's and the key's head norms.
                    weights: 2 * f32s(attn.head),
                    // The query with its gate, the output gate of one value a
                    // query head where the layer has one, the rope row, and the
                    // flash's output.
                    acts: f32s(a.q.len())
                        + a.gate.as_ref().map_or(0, |g| f32s(g.len()))
                        + f32s(attn.rot)
                        + wrote,
                    // The live keys and values read, this token's row written.
                    state: 2 * (n_keys * kv_row + kv_row),
                })
            }
            GroupKind::AttnOut => {
                no_ids()?;
                let l = bufs()?;
                let a = attn_of(l, layer, kind)?;
                Ok(GroupBytes {
                    weights: bytes_of(&a.out),
                    // The attention output read; the residual read and
                    // written, or the join buffer written on a sandwich.
                    acts: bytes_of(&self.attn) + stream_of(l) * f32s(self.hidden),
                    state: 0,
                })
            }
            GroupKind::ConvCore => {
                no_ids()?;
                let c = conv_of(bufs()?, layer, kind)?;
                Ok(GroupBytes {
                    weights: bytes_of(&c.conv),
                    // The `B | C | x` rows read, the conv's output written.
                    acts: f32s(c.bcx.len()) + f32s(self.hidden),
                    // The taps − 1 predecessor rows of the ring read and the
                    // token's own row written.
                    state: count_u64(c.taps, "the conv's taps") * f32s(self.hidden),
                })
            }
            GroupKind::ConvOut => {
                no_ids()?;
                let l = bufs()?;
                let c = conv_of(l, layer, kind)?;
                Ok(GroupBytes {
                    weights: bytes_of(&c.out),
                    // The conv's output read; the residual read and written,
                    // or the join buffer written on a sandwich.
                    acts: f32s(self.hidden) + stream_of(l) * f32s(self.hidden),
                    state: 0,
                })
            }
            GroupKind::Join => {
                no_ids()?;
                let Some(post) = &bufs()?.post else {
                    return Err(refuse(format!(
                        "{kind:?}: layer {layer} adds its halves to the stream and runs no join"
                    )));
                };
                // A join that hands on writes its reader's input and scale; the
                // reader's gain it reads is counted with the reader.
                let handed = if self.hands_on(layer) {
                    f32s(self.hidden + 1)
                } else {
                    0
                };
                Ok(GroupBytes {
                    // A gain of either half: `hidden` values.
                    weights: bytes_of(&post.gains[0]),
                    // The half's output and the residual read, the residual
                    // written.
                    acts: 3 * f32s(self.hidden) + handed,
                    state: 0,
                })
            }
            GroupKind::DeltaCore => {
                no_ids()?;
                let d = delta_of(bufs()?, layer, kind)?;
                let lane = self
                    .geom
                    .delta
                    .state_elems(1)
                    .ok_or_else(|| refuse("a state lane does not fit"))?;
                Ok(GroupBytes {
                    weights: bytes_of(&d.conv)
                        + bytes_of(&d.dt_bias)
                        + bytes_of(&d.ssm_a)
                        + bytes_of(&d.ssm_norm),
                    // The P1 outputs read, the gated numerator and its
                    // partials written, the ring read and written, the lane
                    // word read.
                    acts: f32s(d.qkv.len())
                        + f32s(d.z.len())
                        + f32s(d.beta.len())
                        + f32s(d.alpha.len())
                        + bytes_of(&self.g)
                        + bytes_of(&self.delta_partials)
                        + 2 * bytes_of(&d.ring)
                        + 4,
                    // The lane the token reads and the lane it writes.
                    state: 2 * f32s(lane),
                })
            }
            GroupKind::DeltaOut => {
                no_ids()?;
                let d = delta_of(bufs()?, layer, kind)?;
                Ok(GroupBytes {
                    weights: bytes_of(&d.out),
                    // The numerator and its partials read, the residual read
                    // and written.
                    acts: bytes_of(&self.g)
                        + bytes_of(&self.delta_partials)
                        + 2 * f32s(self.hidden),
                    state: 0,
                })
            }
            GroupKind::Router => {
                no_ids()?;
                let f = &bufs()?.ffn;
                let Some(r) = &f.router else {
                    return Err(refuse(format!(
                        "{kind:?}: layer {layer} has a dense FFN and no router"
                    )));
                };
                Ok(GroupBytes {
                    weights: bytes_of(&r.gate_inp)
                        + f.shared.as_ref().map_or(0, |s| bytes_of(&s.gate_inp))
                        + bytes_of(&f.norm),
                    // The residual read; the logits, the ids, the weights,
                    // the partials and the ticket written.
                    acts: f32s(self.hidden)
                        + bytes_of(&r.logits)
                        + bytes_of(&f.ids)
                        + bytes_of(&f.weights)
                        + bytes_of(&r.partials)
                        + 4,
                    state: 0,
                })
            }
            GroupKind::GateUp | GroupKind::Down => {
                let l = bufs()?;
                let f = &l.ffn;
                let layer_joins = l.post.is_some();
                let picks = self.top_k;
                let shared = usize::from(self.geom.experts.has_shared);
                if ids.len() != picks + shared {
                    return Err(refuse(format!(
                        "{kind:?}: {} router ids; the picks, then any shared expert's, are {}",
                        ids.len(),
                        picks + shared
                    )));
                }
                let n = count_u64(self.n_experts, "the experts");
                let expert = |stack: &MetalBuffer<u8>| -> Result<u64, SessionError> {
                    let whole = bytes_of(stack);
                    let slice = whole / n;
                    if slice * n != whole {
                        return Err(refuse(
                            "an expert stack holds no whole slice an expert takes",
                        ));
                    }
                    Ok(slice)
                };
                // A pick reads its expert's slice of every stack it runs; the
                // shared expert's parts run with the picks'.
                let picks = count_u64(picks, "the picks");
                // The FFN norm is a weight of the router's group; a dense FFN
                // has no router, and the pair's group holds it.
                let norm = bytes_of(&f.norm);
                let (norm_weight, norm_read) = if f.router.is_none() {
                    (norm, 0)
                } else {
                    (0, norm)
                };
                let (weights, acts) = match kind {
                    GroupKind::GateUp => (
                        picks * (expert(&f.gate)? + expert(&f.up)?)
                            + f.shared
                                .as_ref()
                                .map_or(0, |s| bytes_of(&s.gate) + bytes_of(&s.up))
                            + norm_weight,
                        // The residual normed as it is loaded (or the join's
                        // normed input and its scale), the ids read, the
                        // experts' rows written.
                        f32s(self.hidden)
                            + norm_read
                            + self.scale_word(Reader::Pair(layer))
                            + 4 * count_u64(ids.len(), "ids")
                            + bytes_of(&self.h),
                    ),
                    _ if layer_joins => (
                        picks * expert(&f.down)?,
                        // The one slot's row read, the join buffer written: a
                        // sandwich's down stores its projection, and no route
                        // weight scales it.
                        bytes_of(&self.h) + f32s(self.hidden),
                    ),
                    _ => (
                        picks * expert(&f.down)?
                            + f.shared.as_ref().map_or(0, |s| bytes_of(&s.down)),
                        // The experts' rows and their weights read, the
                        // residual read and written.
                        bytes_of(&self.h) + bytes_of(&f.weights) + 2 * f32s(self.hidden),
                    ),
                };
                Ok(GroupBytes {
                    weights,
                    acts,
                    state: 0,
                })
            }
            GroupKind::Head => {
                no_ids()?;
                Ok(GroupBytes {
                    weights: bytes_of(&self.head_w) + bytes_of(&self.head_norm),
                    // The residual read (or the join's normed input and its
                    // scale) and the logits written.
                    acts: f32s(self.hidden)
                        + bytes_of(&self.logits)
                        + self.scale_word(Reader::Head),
                    state: 0,
                })
            }
            GroupKind::HeadPick => {
                no_ids()?;
                Ok(GroupBytes {
                    weights: 0,
                    // The logits read and the pick written.
                    acts: bytes_of(&self.logits) + 4,
                    state: 0,
                })
            }
        }
    }

    /// Every store a token sequence writes back to its state at rest: the
    /// caches, the recurrent states (both lanes) and rings, the tickets, the
    /// id slots, the lane words. The fault words are the session's to clear,
    /// on its stream.
    pub(super) fn clear_stores(&mut self) -> Result<(), SessionError> {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                MixerBufs::Delta(d) => {
                    d.state.zero();
                    d.ring.zero();
                }
                MixerBufs::Attention(a) => {
                    a.k_cache.zero();
                    a.v_cache.zero();
                }
                MixerBufs::ShortConv(c) => c.ring.zero(),
            }
            if let Some(router) = &mut layer.ffn.router {
                router.ticket.zero();
            }
        }
        self.attn_ticket.zero();
        self.step_id.zero();
        self.id_in.zero();
        self.lane.write(&LANE_WORDS)?;
        Ok(())
    }
}

fn delta(load: &Loader<'_>, d: &DeltaLayout) -> Result<DeltaBufs, SessionError> {
    let (p1, ranges) = load.p1(&d.p1)?;
    let out = load.site(&d.out)?;
    let device = load.device;
    let state = zeroed(device, d.state)?;
    let ring = zeroed(device, d.ring)?;
    Ok(DeltaBufs {
        qkv: ranges[0].clone(),
        z: ranges[1].clone(),
        beta: ranges[2].clone(),
        alpha: ranges[3].clone(),
        p1,
        conv: load.values(&d.conv)?,
        dt_bias: load.values(&d.dt_bias)?,
        ssm_a: load.values(&d.ssm_a)?,
        ssm_norm: load.values(&d.ssm_norm)?,
        out,
        out_quant: d.out.quant,
        state,
        ring,
        n_k: d.n_k,
        map: d.map,
    })
}

fn short_conv(load: &Loader<'_>, c: &ConvLayout) -> Result<ConvBufs, SessionError> {
    let (p1, ranges) = load.p1(&c.p1)?;
    let out = load.site(&c.out)?;
    Ok(ConvBufs {
        bcx: ranges[0].clone(),
        p1,
        conv: load.values(&c.conv)?,
        out,
        out_quant: c.out.quant,
        ring: zeroed(load.device, c.ring)?,
        taps: c.taps,
        ring_rows: c.ring_rows,
    })
}

/// The sandwich norms' gains of a layer, each a buffer of its own.
fn post(load: &Loader<'_>, p: &PostLayout) -> Result<PostBufs, SessionError> {
    let [mixer, ffn] = &p.gains;
    Ok(PostBufs {
        gains: [load.values(mixer)?, load.values(ffn)?],
    })
}

fn attention(load: &Loader<'_>, a: &AttnLayout) -> Result<AttnBufs, SessionError> {
    let (p1, ranges) = load.p1(&a.p1)?;
    let out = load.site(&a.out)?;
    let device = load.device;
    Ok(AttnBufs {
        q: ranges[0].clone(),
        k: ranges[1].clone(),
        v: ranges[2].clone(),
        gate: a.gate.then(|| ranges[3].clone()),
        window: a.window,
        rope: a.rope,
        p1,
        q_norm: load.values(&a.q_norm)?,
        k_norm: load.values(&a.k_norm)?,
        out,
        out_quant: a.out.quant,
        k_cache: zeroed(device, a.cache)?,
        v_cache: zeroed(device, a.cache)?,
    })
}

fn ffn(load: &Loader<'_>, device: Alloc<'_>, f: &FfnLayout) -> Result<FfnBufs, SessionError> {
    let geom = load.geom;
    // The buffers are made in the order the arena always made them: the
    // weights, then the router's logits, the picks, the router's partials and
    // its ticket. A dense FFN makes the picks alone.
    let norm = load.values(&f.norm)?;
    // The router and the shared-expert gate: F32 in a GGUF file, MLX's
    // affine 8-bit in an MLX file, dequantized exactly at open.
    let gate_inp = f.gate_inp.as_ref().map(|v| load.values(v)).transpose()?;
    let gate = load.site(&f.gate)?;
    let up = load.site(&f.up)?;
    let down = load.site(&f.down)?;
    let shared = f
        .shared
        .as_ref()
        .map(|s| {
            Ok::<_, SessionError>(SharedBufs {
                gate_inp: load.values(&s.gate_inp)?,
                gate_up_quant: s.gate.quant,
                down_quant: s.down.quant,
                gate: load.site(&s.gate)?,
                up: load.site(&s.up)?,
                down: load.site(&s.down)?,
            })
        })
        .transpose()?;
    let logits: Option<MetalBuffer<f32>> = gate_inp
        .as_ref()
        .map(|_| zeroed(device, geom.router.rows))
        .transpose()?;
    let mut ids: MetalBuffer<u32> = zeroed(device, geom.experts.slots)?;
    let mut weights: MetalBuffer<f32> = zeroed(device, geom.experts.slots)?;
    let partials: Option<MetalBuffer<f32>> = gate_inp
        .as_ref()
        .map(|_| zeroed(device, geom.router.partials))
        .transpose()?;
    let ticket: Option<MetalBuffer<u32>> =
        gate_inp.as_ref().map(|_| zeroed(device, 1)).transpose()?;
    let router = match (gate_inp, logits, partials, ticket) {
        (Some(gate_inp), Some(logits), Some(partials), Some(ticket)) => Some(RouterBufs {
            gate_inp,
            logits,
            partials,
            ticket,
        }),
        _ => {
            // A dense FFN's route is fixed: its one slot is expert 0 at
            // weight 1, laid once.
            ids.write(&[DENSE_ID])?;
            weights.write(&[DENSE_WEIGHT])?;
            None
        }
    };
    Ok(FfnBufs {
        norm,
        router,
        gate_up_quant: f.gate_up_quant,
        gate,
        up,
        down_quant: f.down_quant,
        down,
        shared,
        ids,
        weights,
    })
}

#[cfg(test)]
mod in_order_tests {
    use super::in_order;

    /// The results come back in job order whatever thread ran each, and the
    /// error is the earliest failing job's, however the threads interleave:
    /// a later job that fails first does not win. Mutant: the error that
    /// came first in time returned (job 9 here, with two workers or more).
    #[test]
    fn jobs_return_in_order_and_the_earliest_failure_wins() {
        for workers in [1, 2, 3, 8] {
            let got = in_order(20, workers, |j| Ok::<_, usize>(j * 3)).unwrap();
            assert_eq!(got, (0..20).map(|j| j * 3).collect::<Vec<_>>(), "{workers}");
            let err = in_order(20, workers, |j| {
                if j == 9 || j == 4 {
                    // The earlier failure finishes last.
                    if j == 4 {
                        std::thread::sleep(std::time::Duration::from_millis(30));
                    }
                    return Err(j);
                }
                Ok(j)
            })
            .unwrap_err();
            assert_eq!(err, 4, "{workers} workers");
        }
        assert_eq!(in_order(0, 4, Ok::<_, ()>).unwrap(), Vec::<usize>::new());
    }
}

#[cfg(test)]
mod tests {
    use chain::arch::qwen3moe::plan::SiteTy;
    use chain::site::BACKEND_SITES;

    use crate::arch::qwen3moe::session::{LAUNCHED, PROJECTED};
    use crate::kernels::Quant;

    /// The session's site lists and the mat-vecs' quant set name the same
    /// families, over the card's families and the backends' rows: a family
    /// the plan admits has a launch, and a launch the plan never admits is
    /// dead. A projection may be any type with a launch, the rest only the
    /// types with every launch.
    #[test]
    fn launched_sites_are_the_arena_quants() {
        let families = SiteTy::ALL
            .iter()
            .copied()
            .chain(BACKEND_SITES.iter().map(|r| r.1));
        for s in families {
            let quant = Quant::of_ggml(s.ggml());
            assert_eq!(
                PROJECTED.contains(&s),
                quant.is_some(),
                "{s:?}: the session's PROJECTED and Quant::of_ggml disagree"
            );
            assert_eq!(
                LAUNCHED.contains(&s),
                quant.is_some_and(Quant::dense),
                "{s:?}: the session's LAUNCHED and Quant::dense disagree"
            );
        }
    }
}
