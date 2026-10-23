//! The qwen35moe arena: every buffer one token touches, allocated when the
//! session opens and never in a step.
//!
//! The weights are copied from the file into buffers of their own. A layer's
//! P1 projection is one buffer per weight type, each the segments of one
//! dispatch. The activations, the caches, the recurrent stores, the router's
//! and the head's tickets, the host's id slot, the step record slots and the
//! fault words are the session's. Every length comes from the plan, the
//! file's shapes or a launch's geometry. A shape no launch runs is refused by
//! name before the buffers past it are allocated.
//!
//! Two tokens may be committed at once, the second reading what the first
//! writes. What the second would overwrite before the first is done is held
//! twice: the step record (the head's id and the fault word) has a slot per
//! position parity, and so does the delta state, whose lanes a token reads by
//! its position's parity and writes by the next position's.

use core::ops::Range;

use bloomery_device::{Elem, Elems};
use chain::arch::qwen3moe::plan::{FfnRoute, LayerPlan, MixerPlan, SharedPlan};
use chain::arch::qwen3moe::read::file_names;
use chain::linear::{KHeadMap, LinearShape};
use chain::rope::{Direction, RopeSpec, RopeTable};
use gguf::Split;
use gguf::quant::GgmlType;
use model::arch::models::{Ffn, LayerSpec, Mixer as SpecMixer, ModelSpec};
use model::arch::names;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLDevice;

use super::count::{GroupBytes, GroupKind, Layer, Projection};
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::fault::MetalFaultWord;
use crate::kernels::{
    BLOCK_VALUES, EXPERT_K, EXPERT_SLOTS, HEAD_COLS, HEAD_DIM, HEAD_GROUPS, HEAD_K, HEAD_VOCAB,
    HeadMap, KV_HEADS, MAX_SEGMENTS, PARTIALS_PER_HEAD, Quant, ROT, ROUTED_IDS, Segment,
    VALUE_HEADS, attention_dispatches, channels, f16_len, out_len, partials_len, qg_len,
    ring_elems, router, segments_for_keys, state_elems,
};

const WHAT: &str = "qwen35moe arena";

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
    /// The count's projection: one dispatch, or two in the group.
    fn projection(&self) -> Projection {
        if self.parts.len() == 2 {
            Projection::Q6
        } else {
            Projection::Q4
        }
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
    /// `ssm_out`, Q4_K: the head-norm projection's only type.
    pub(super) out: MetalBuffer<u8>,
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
    pub(super) q_norm: MetalBuffer<f32>,
    pub(super) k_norm: MetalBuffer<f32>,
    pub(super) out: MetalBuffer<u8>,
    pub(super) out_quant: Quant,
    pub(super) k_cache: MetalBuffer<u16>,
    pub(super) v_cache: MetalBuffer<u16>,
}

/// A layer's mixer.
pub(super) enum MixerBufs {
    Delta(DeltaBufs),
    Attention(AttnBufs),
}

/// A layer's FFN half: the router's weights and outputs, and the experts.
/// Each layer keeps its own router outputs, so every layer's picks are
/// readable after a step.
pub(super) struct FfnBufs {
    pub(super) norm: MetalBuffer<f32>,
    pub(super) gate_inp: MetalBuffer<f32>,
    pub(super) gate_inp_shexp: MetalBuffer<f32>,
    pub(super) gate_up_quant: Quant,
    pub(super) gate: MetalBuffer<u8>,
    pub(super) up: MetalBuffer<u8>,
    pub(super) shexp_gate: MetalBuffer<u8>,
    pub(super) shexp_up: MetalBuffer<u8>,
    pub(super) down_quant: Quant,
    pub(super) down: MetalBuffer<u8>,
    pub(super) shexp_down: MetalBuffer<u8>,
    pub(super) logits: MetalBuffer<f32>,
    pub(super) ids: MetalBuffer<u32>,
    pub(super) weights: MetalBuffer<f32>,
    pub(super) partials: MetalBuffer<f32>,
    pub(super) ticket: MetalBuffer<u32>,
}

/// One layer's buffers.
pub(super) struct LayerBufs {
    pub(super) mixer: MixerBufs,
    pub(super) ffn: FfnBufs,
}

/// Every buffer of the session.
pub(super) struct Arena {
    pub(super) hidden: usize,
    pub(super) vocab: usize,
    pub(super) ctx: usize,
    pub(super) eps: f32,
    pub(super) top_k: usize,
    pub(super) n_experts: usize,
    pub(super) layers: Vec<LayerBufs>,
    pub(super) embed: MetalBuffer<u8>,
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
    /// `ctx` rows of [`ROT`] interleaved `(cos, sin)`.
    pub(super) rope: MetalBuffer<f32>,
    /// The experts' activations, one row of [`EXPERT_K`] per slot.
    pub(super) h: MetalBuffer<f32>,
    pub(super) logits: MetalBuffer<f32>,
    pub(super) candidates: MetalBuffer<u64>,
    pub(super) head_ticket: MetalBuffer<u32>,
    /// The head's pick, one per step record slot.
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

fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// The shape facts every layer must share, each set by the first layer that
/// has one and held by every later layer: the rope table's base (one table
/// serves one) and the MoE's experts-and-picks shape.
#[derive(Default)]
struct SharedShapes {
    rope_base: Option<f32>,
    moe: Option<(usize, usize)>,
}

/// The launch quant of file type `ty`, or `None` for a type no mat-vec runs.
fn quant_of(ty: GgmlType) -> Option<Quant> {
    match ty {
        GgmlType::Q4_K => Some(Quant::Q4K),
        GgmlType::Q6_K => Some(Quant::Q6K),
        _ => None,
    }
}

/// The file's tensors, copied into buffers on `device`.
struct Loader<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    split: &'a Split,
}

impl Loader<'_> {
    /// The type, shape and bytes of tensor `name`.
    fn tensor(&self, name: &str) -> Result<(GgmlType, Vec<u64>, &[u8]), SessionError> {
        let (shard, info) = self
            .split
            .find(name)
            .ok_or_else(|| refuse(format!("the file has no tensor {name}")))?;
        let file = self
            .split
            .shard(shard)
            .ok_or_else(|| refuse(format!("{name}: shard {shard} is not open")))?;
        let bytes = file
            .data(info)
            .map_err(|e| refuse(format!("{name}: {e}")))?;
        Ok((info.ty, info.dims.clone(), bytes))
    }

    /// F32 tensor `name` of `len` values.
    fn f32s(&self, name: &str, len: usize) -> Result<MetalBuffer<f32>, SessionError> {
        let (ty, dims, bytes) = self.tensor(name)?;
        let n: u64 = dims.iter().product();
        if ty != GgmlType::F32 || usize::try_from(n).ok() != Some(len) {
            return Err(refuse(format!(
                "{name} is {ty:?} {dims:?}; the step reads {len} F32 values"
            )));
        }
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let mut buffer = MetalBuffer::new(self.device, Elems(len))?;
        buffer.write(&values)?;
        Ok(buffer)
    }

    /// The quant and bytes of tensor `name`: `rows` rows of `cols`, every
    /// axis past the first counted as rows, in a type a mat-vec runs.
    fn rows(&self, name: &str, cols: usize, rows: usize) -> Result<(Quant, &[u8]), SessionError> {
        let (ty, dims, bytes) = self.tensor(name)?;
        let got_rows: u64 = dims.iter().skip(1).product();
        let shape_ok = dims.first().and_then(|&c| usize::try_from(c).ok()) == Some(cols)
            && usize::try_from(got_rows).ok() == Some(rows);
        if !shape_ok {
            return Err(refuse(format!(
                "{name} is {dims:?}; the step reads {rows} rows of {cols}"
            )));
        }
        let quant = quant_of(ty).ok_or_else(|| {
            refuse(format!(
                "{name} is {ty:?}, which no mat-vec of this backend runs (Q4_K, Q6_K)"
            ))
        })?;
        Ok((quant, bytes))
    }

    /// Tensor `name` as its own buffer, with its quant.
    fn quant(
        &self,
        name: &str,
        cols: usize,
        rows: usize,
    ) -> Result<(Quant, MetalBuffer<u8>), SessionError> {
        let (quant, bytes) = self.rows(name, cols, rows)?;
        let mut buffer = MetalBuffer::new(self.device, Elems(bytes.len()))?;
        buffer.write(bytes)?;
        Ok((quant, buffer))
    }

    /// The P1 of `parts` (each a tensor and its rows, all of `cols`), normed
    /// on load by `norm`: one dispatch per weight type, its segments in the
    /// order given, each part's output at the offset its place in `parts`
    /// gives. Returns the P1 and every part's output range.
    fn p1(
        &self,
        norm: &str,
        parts: &[(&str, usize)],
        cols: usize,
    ) -> Result<(P1, Vec<Range<usize>>), SessionError> {
        if !cols.is_multiple_of(BLOCK_VALUES) {
            return Err(refuse(format!(
                "{norm}'s P1 reads {cols} columns, not whole blocks of {BLOCK_VALUES}"
            )));
        }
        let mut ranges = Vec::with_capacity(parts.len());
        let mut tensors = Vec::with_capacity(parts.len());
        let mut at = 0usize;
        for &(name, rows) in parts {
            let (quant, bytes) = self.rows(name, cols, rows)?;
            ranges.push(at..at + rows);
            tensors.push((quant, rows, bytes, at));
            at += rows;
        }
        let mut dispatches: Vec<P1Part> = Vec::new();
        for quant in [Quant::Q4K, Quant::Q6K] {
            let mine: Vec<_> = tensors.iter().filter(|t| t.0 == quant).collect();
            if mine.is_empty() {
                continue;
            }
            if mine.len() > MAX_SEGMENTS {
                return Err(refuse(format!(
                    "{norm}'s P1 has {} {quant:?} tensors; a dispatch takes {MAX_SEGMENTS}",
                    mine.len()
                )));
            }
            let bytes: usize = mine.iter().map(|t| t.2.len()).sum();
            let mut host = Vec::with_capacity(bytes);
            let mut segments = Vec::with_capacity(mine.len());
            let mut out_end = 0;
            for &&(_, rows, b, out_at) in &mine {
                segments.push(Segment {
                    cols,
                    rows,
                    weight_blocks: host.len() / quant.block_bytes(),
                    out_offset: out_at,
                });
                host.extend_from_slice(b);
                out_end = out_end.max(out_at + rows);
            }
            let mut weights = MetalBuffer::new(self.device, Elems(host.len()))?;
            weights.write(&host)?;
            dispatches.push(P1Part {
                quant,
                weights,
                segments,
                out_end,
            });
        }
        let norm = self.f32s(norm, cols)?;
        Ok((
            P1 {
                norm,
                parts: dispatches,
            },
            ranges,
        ))
    }

    /// Layer `l`'s buffers: the mixer its plan and description agree on, the
    /// FFN of the expert shape every layer shares, the facts the layers must
    /// share read from and set on `shared` (the first layer that has one sets
    /// each). A refusal of layer `l` names it.
    fn layer(
        &self,
        plan: &LayerPlan,
        layer: &LayerSpec,
        l: usize,
        hidden: usize,
        ctx: usize,
        shared: &mut SharedShapes,
    ) -> Result<LayerBufs, SessionError> {
        let at = |e: SessionError| match e {
            SessionError::Shape { what, detail } => SessionError::Shape {
                what,
                detail: format!("layer {l}: {detail}"),
            },
            other => other,
        };
        let mixer = match (&plan.mixer, &layer.mixer) {
            (MixerPlan::Delta(d), SpecMixer::DeltaRule(_)) => {
                MixerBufs::Delta(delta(self, d, hidden).map_err(at)?)
            }
            (MixerPlan::Gqa(g), SpecMixer::Gqa(spec_g)) => {
                if spec_g.rope.dims as usize != ROT {
                    return Err(at(refuse(format!(
                        "a rope over {} values; the attention turns {ROT}",
                        spec_g.rope.dims
                    ))));
                }
                match shared.rope_base {
                    None => shared.rope_base = Some(spec_g.rope.base),
                    Some(b) if b == spec_g.rope.base => {}
                    Some(b) => {
                        return Err(at(refuse(format!(
                            "a rope base of {}, another layer's is {b}; one table serves one",
                            spec_g.rope.base
                        ))));
                    }
                }
                MixerBufs::Attention(attention(self, g, hidden, ctx).map_err(at)?)
            }
            _ => {
                return Err(at(refuse(
                    "the plan's mixer and the description's differ".to_string(),
                )));
            }
        };
        let Ffn::Moe(m) = &layer.ffn else {
            return Err(at(refuse(
                "a dense FFN, which this backend's experts do not run".to_string(),
            )));
        };
        let shape = (m.experts as usize, m.top_k as usize);
        match shared.moe {
            None => shared.moe = Some(shape),
            Some(s) if s == shape => {}
            Some(s) => {
                return Err(at(refuse(format!(
                    "{shape:?} experts and picks, another layer's are {s:?}"
                ))));
            }
        }
        if m.expert_ff as usize != EXPERT_K {
            return Err(at(refuse(format!(
                "an expert of {}; the experts run {EXPERT_K}",
                m.expert_ff
            ))));
        }
        let ffn = ffn(self, self.device, plan, hidden, shape, l).map_err(at)?;
        Ok(LayerBufs { mixer, ffn })
    }
}

/// A buffer of `n` elements, zeroed.
fn zeroed<T: bloomery_device::Elem>(
    device: &ProtocolObject<dyn MTLDevice>,
    n: usize,
) -> Result<MetalBuffer<T>, SessionError> {
    let mut buffer = MetalBuffer::new(device, Elems(n))?;
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
        MixerBufs::Delta(_) => Err(refuse(format!("{kind:?}: layer {l} is no attention layer"))),
    }
}

/// Layer `layer`'s delta buffers for the byte rule, or the named refusal.
fn delta_of(layer: &LayerBufs, l: usize, kind: GroupKind) -> Result<&DeltaBufs, SessionError> {
    match &layer.mixer {
        MixerBufs::Delta(d) => Ok(d),
        MixerBufs::Attention(_) => Err(refuse(format!("{kind:?}: layer {l} is no delta layer"))),
    }
}

/// The rope table of `ctx` positions: per position, [`ROT`] / 2 pairs
/// `(cos θ, sin θ)` with θ = `pos · base^(−2i/ROT)`, θ advanced by repeated
/// multiplication in f32 as ggml's `rope_yarn` does with no YaRN — chain's
/// [`RopeTable`] over a window spec, the one owner of the recipe both
/// backends' tables come from.
fn rope_rows(ctx: usize, base: f32) -> Result<Vec<f32>, SessionError> {
    let table = RopeTable::new(&RopeSpec::window(base, ROT))
        .map_err(|e| refuse(format!("the rope table: {e}")))?;
    let positions = u32::try_from(ctx).expect("the arena checked the cache's rows fit u32");
    let mut out = Vec::with_capacity(ctx * ROT);
    for pos in 0..positions {
        table.push(pos, Direction::Forward, &mut out);
    }
    Ok(out)
}

impl Arena {
    /// The arena of `plans` (read from `split`, whose description is
    /// `spec`) at `ctx` positions, past one segment of keys or within it. A
    /// `ctx` of no rows or past the file's trained context, a length one of
    /// the session's buffers does not fit, a shape no launch runs, or a type
    /// no launch reads is refused by name.
    pub(super) fn open(
        device: &ProtocolObject<dyn MTLDevice>,
        split: &Split,
        spec: &ModelSpec,
        plans: &[LayerPlan],
        ctx: usize,
    ) -> Result<Arena, SessionError> {
        if ctx == 0 {
            return Err(refuse(format!(
                "a cache of {ctx} rows; a session holds at least one"
            )));
        }
        let trained = spec.ctx_train as usize;
        if ctx > trained {
            return Err(refuse(format!(
                "a cache of {ctx} rows; the file was trained to {trained}"
            )));
        }
        // Every length the session holds of `ctx` rows, before any of them
        // is allocated.
        let word = ctx_word(ctx)?;
        let _ = f16_len(ctx)
            .ok_or_else(|| refuse(format!("the f16 cache of {ctx} rows does not fit")))?;
        let rope_len = ctx
            .checked_mul(ROT)
            .ok_or_else(|| refuse(format!("the rope table of {ctx} rows does not fit")))?;
        let segments = segments_for_keys(word) as usize;
        let attn_partials = partials_len(segments).ok_or_else(|| {
            refuse(format!(
                "the attention partials of {segments} segments do not fit"
            ))
        })?;
        let hidden = spec.hidden as usize;
        let vocab = spec.vocab as usize;
        if hidden != HEAD_K || hidden != router::K {
            return Err(refuse(format!(
                "a hidden width of {hidden}; the head and the router read {HEAD_K}"
            )));
        }
        if vocab != HEAD_VOCAB {
            return Err(refuse(format!(
                "a vocabulary of {vocab}; the head reads {HEAD_VOCAB}"
            )));
        }
        if plans.len() != spec.layers.len() {
            return Err(refuse(format!(
                "{} plans for {} layers",
                plans.len(),
                spec.layers.len()
            )));
        }
        let load = Loader { device, split };
        let (embed_quant, embed) = load.quant(&names::token_embd(), hidden, vocab)?;
        if embed_quant != Quant::Q4K {
            return Err(refuse(format!(
                "{} is {embed_quant:?}; the embedding reads Q4_K",
                names::token_embd()
            )));
        }
        let (head_quant, head_w) = load.quant(&names::output(), hidden, vocab)?;
        if head_quant != Quant::Q6K {
            return Err(refuse(format!(
                "{} is {head_quant:?}; the head reads Q6_K",
                names::output()
            )));
        }
        let head_norm = load.f32s(&names::output_norm(), hidden)?;
        let mut shared = SharedShapes::default();
        let mut layers = Vec::with_capacity(plans.len());
        for (l, (plan, layer)) in plans.iter().zip(&spec.layers).enumerate() {
            layers.push(load.layer(plan, layer, l, hidden, ctx, &mut shared)?);
        }
        let (n_experts, top_k) = shared.moe.ok_or_else(|| refuse("a file of no layer"))?;
        let base = shared
            .rope_base
            .ok_or_else(|| refuse("no attention layer to read the rope from"))?;
        let mut rope = MetalBuffer::new(device, Elems(rope_len))?;
        rope.write(&rope_rows(ctx, base)?)?;
        let proj = layers
            .iter()
            .flat_map(|l| match &l.mixer {
                MixerBufs::Delta(d) => &d.p1.parts,
                MixerBufs::Attention(a) => &a.p1.parts,
            })
            .map(|p| p.out_end)
            .max()
            .ok_or_else(|| refuse("a file of no layer"))?;
        let mut lane = MetalBuffer::new(device, Elems(LANES))?;
        lane.write(&LANE_WORDS)?;
        Ok(Arena {
            hidden,
            vocab,
            ctx,
            eps: spec.rms_eps,
            top_k,
            n_experts,
            layers,
            embed,
            head_w,
            head_norm,
            x: zeroed(device, hidden)?,
            proj: zeroed(device, proj)?,
            g: zeroed(device, VALUE_HEADS * HEAD_COLS)?,
            delta_partials: zeroed(device, VALUE_HEADS * PARTIALS_PER_HEAD)?,
            attn: zeroed(
                device,
                out_len().ok_or_else(|| refuse("the attention output does not fit"))?,
            )?,
            attn_partials: zeroed(device, attn_partials)?,
            rope,
            h: zeroed(device, EXPERT_SLOTS * EXPERT_K)?,
            logits: zeroed(device, vocab)?,
            candidates: zeroed(device, HEAD_GROUPS)?,
            head_ticket: zeroed(device, 1)?,
            step_id: zeroed(device, RECORD_SLOTS)?,
            id_in: zeroed(device, 1)?,
            lane,
            faults: [MetalFaultWord::new(device)?, MetalFaultWord::new(device)?],
        })
    }

    /// The token as the count models it: each layer's kind and whether its
    /// P1 is two dispatches.
    pub(super) fn token_layers(&self) -> Vec<Layer> {
        self.layers
            .iter()
            .map(|l| match &l.mixer {
                MixerBufs::Delta(d) => Layer::Delta {
                    qkv: d.p1.projection(),
                },
                MixerBufs::Attention(a) => Layer::Attention {
                    v: a.p1.projection(),
                },
            })
            .collect()
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
    /// rows — the live keys read, the token's own row written — and the
    /// delta state's read and written lanes with the conv ring).
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
            GroupKind::AttnP1 | GroupKind::DeltaP1 => {
                no_ids()?;
                let l = bufs()?;
                let p1 = match kind {
                    GroupKind::AttnP1 => &attn_of(l, layer, kind)?.p1,
                    _ => &delta_of(l, layer, kind)?.p1,
                };
                let out_end = p1.parts.iter().map(|p| p.out_end).max().unwrap_or(0);
                Ok(GroupBytes {
                    weights: p1.parts.iter().map(|p| bytes_of(&p.weights)).sum::<u64>()
                        + bytes_of(&p1.norm),
                    // The residual read, the projection's output written.
                    acts: f32s(self.hidden) + f32s(out_end),
                    state: 0,
                })
            }
            GroupKind::AttnFlash => {
                no_ids()?;
                let a = attn_of(bufs()?, layer, kind)?;
                let n_keys = u64::from(pos) + 1;
                let kv_row = count_u64(KV_HEADS * HEAD_DIM, "a cache row") * 2;
                let segments = segments_for_keys(pos + 1);
                let parts = partials_len(segments as usize)
                    .ok_or_else(|| refuse("the attention partials do not fit"))?;
                // Within one segment the flash writes the output itself;
                // past it, it writes the partials the merge folds.
                let wrote = if attention_dispatches(segments) == 1 {
                    bytes_of(&self.attn)
                } else {
                    f32s(parts)
                };
                Ok(GroupBytes {
                    // The query's and the key's head norms.
                    weights: 2 * f32s(HEAD_DIM),
                    // The query with its gate, the rope row, and the flash's
                    // output.
                    acts: f32s(a.q.len()) + f32s(ROT) + wrote,
                    // The live keys and values read, this token's row written.
                    state: 2 * (n_keys * kv_row + kv_row),
                })
            }
            GroupKind::AttnMerge => {
                no_ids()?;
                let segments = segments_for_keys(pos + 1);
                let parts = partials_len(segments as usize)
                    .ok_or_else(|| refuse("the attention partials do not fit"))?;
                Ok(GroupBytes {
                    weights: 0,
                    // The partials read, the attention output written.
                    acts: f32s(parts) + bytes_of(&self.attn),
                    state: 0,
                })
            }
            GroupKind::AttnOut => {
                no_ids()?;
                let a = attn_of(bufs()?, layer, kind)?;
                Ok(GroupBytes {
                    weights: bytes_of(&a.out),
                    // The attention output read, the residual read and written.
                    acts: bytes_of(&self.attn) + 2 * f32s(self.hidden),
                    state: 0,
                })
            }
            GroupKind::DeltaCore => {
                no_ids()?;
                let d = delta_of(bufs()?, layer, kind)?;
                let lane = state_elems(1).ok_or_else(|| refuse("a state lane does not fit"))?;
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
                Ok(GroupBytes {
                    weights: bytes_of(&f.gate_inp)
                        + bytes_of(&f.gate_inp_shexp)
                        + bytes_of(&f.norm),
                    // The residual read; the logits, the ids, the weights,
                    // the partials and the ticket written.
                    acts: f32s(self.hidden)
                        + bytes_of(&f.logits)
                        + bytes_of(&f.ids)
                        + bytes_of(&f.weights)
                        + bytes_of(&f.partials)
                        + 4,
                    state: 0,
                })
            }
            GroupKind::GateUp | GroupKind::Down => {
                let f = &bufs()?.ffn;
                let picks = self.top_k;
                if ids.len() != picks + 1 {
                    return Err(refuse(format!(
                        "{kind:?}: {} router ids; the picks, then the shared expert's, are {}",
                        ids.len(),
                        picks + 1
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
                let (weights, acts) = match kind {
                    GroupKind::GateUp => (
                        picks * (expert(&f.gate)? + expert(&f.up)?)
                            + bytes_of(&f.shexp_gate)
                            + bytes_of(&f.shexp_up),
                        // The residual normed as it is loaded, the ids read,
                        // the experts' rows written.
                        f32s(self.hidden)
                            + bytes_of(&f.norm)
                            + 4 * count_u64(ids.len(), "ids")
                            + bytes_of(&self.h),
                    ),
                    _ => (
                        picks * expert(&f.down)? + bytes_of(&f.shexp_down),
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
                    // The residual read, the logits, the candidates and the
                    // pick written.
                    acts: f32s(self.hidden)
                        + bytes_of(&self.logits)
                        + bytes_of(&self.candidates)
                        + 4,
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
            }
            layer.ffn.ticket.zero();
        }
        self.head_ticket.zero();
        self.step_id.zero();
        self.id_in.zero();
        self.lane.write(&LANE_WORDS)?;
        Ok(())
    }
}

/// `ctx` as the launches' word.
fn ctx_word(ctx: usize) -> Result<u32, SessionError> {
    u32::try_from(ctx).map_err(|_| refuse(format!("a cache of {ctx} rows does not fit u32")))
}

/// The value-head map the delta launch reads.
fn head_map(map: KHeadMap) -> HeadMap {
    match map {
        KHeadMap::Tiled => HeadMap::Tiled,
        KHeadMap::Grouped => HeadMap::Grouped,
    }
}

fn delta(
    load: &Loader<'_>,
    d: &chain::arch::qwen3moe::plan::DeltaPlan,
    hidden: usize,
) -> Result<DeltaBufs, SessionError> {
    let LinearShape { n_k, n_v, map } = d.shape;
    if n_v != VALUE_HEADS {
        return Err(refuse(format!(
            "{n_v} value heads; the delta step runs {VALUE_HEADS}"
        )));
    }
    let ch = channels(n_k).ok_or_else(|| refuse(format!("{n_k} key heads do not fit")))?;
    let z_len = VALUE_HEADS * HEAD_COLS;
    let (p1, ranges) = load.p1(
        &d.attn_norm,
        &[
            (&d.qkv, ch),
            (&d.gate, z_len),
            (&d.beta, VALUE_HEADS),
            (&d.alpha, VALUE_HEADS),
        ],
        hidden,
    )?;
    let (out_quant, out) = load.quant(&d.ssm_out, z_len, hidden)?;
    if out_quant != Quant::Q4K {
        return Err(refuse(format!(
            "{} is {out_quant:?}; the head-norm projection reads Q4_K",
            d.ssm_out
        )));
    }
    let device = load.device;
    let state = zeroed(
        device,
        state_elems(LANES).ok_or_else(|| refuse("the delta state does not fit"))?,
    )?;
    let ring = zeroed(
        device,
        ring_elems(n_k).ok_or_else(|| refuse("the conv ring does not fit"))?,
    )?;
    Ok(DeltaBufs {
        qkv: ranges[0].clone(),
        z: ranges[1].clone(),
        beta: ranges[2].clone(),
        alpha: ranges[3].clone(),
        p1,
        conv: load.f32s(&d.conv, crate::kernels::CONV_TAPS * ch)?,
        dt_bias: load.f32s(&d.dt_bias, VALUE_HEADS)?,
        ssm_a: load.f32s(&d.ssm_a, VALUE_HEADS)?,
        ssm_norm: load.f32s(&d.ssm_norm, HEAD_COLS)?,
        out,
        state,
        ring,
        n_k,
        map: head_map(map),
    })
}

fn attention(
    load: &Loader<'_>,
    g: &chain::arch::qwen3moe::plan::GqaPlan,
    hidden: usize,
    ctx: usize,
) -> Result<AttnBufs, SessionError> {
    let q_rows = qg_len().ok_or_else(|| refuse("the query rows do not fit"))?;
    let kv_rows = KV_HEADS * HEAD_DIM;
    let (p1, ranges) = load.p1(
        &g.attn_norm,
        &[
            (&g.attn_q, q_rows),
            (&g.attn_k, kv_rows),
            (&g.attn_v, kv_rows),
        ],
        hidden,
    )?;
    let att = out_len().ok_or_else(|| refuse("the attention rows do not fit"))?;
    let (out_quant, out) = load.quant(&g.attn_output, att, hidden)?;
    let cache = f16_len(ctx).ok_or_else(|| refuse("the cache does not fit"))?;
    let device = load.device;
    Ok(AttnBufs {
        q: ranges[0].clone(),
        k: ranges[1].clone(),
        v: ranges[2].clone(),
        p1,
        q_norm: load.f32s(&g.attn_q_norm, HEAD_DIM)?,
        k_norm: load.f32s(&g.attn_k_norm, HEAD_DIM)?,
        out,
        out_quant,
        k_cache: zeroed(device, cache)?,
        v_cache: zeroed(device, cache)?,
    })
}

fn ffn(
    load: &Loader<'_>,
    device: &ProtocolObject<dyn MTLDevice>,
    plan: &LayerPlan,
    hidden: usize,
    (n_experts, top_k): (usize, usize),
    l: usize,
) -> Result<FfnBufs, SessionError> {
    match &plan.ffn.route {
        FfnRoute::Router {
            shared: Some(SharedPlan::Joined),
            ..
        } => {}
        FfnRoute::Router {
            shared: Some(SharedPlan::Apart(_)),
            ..
        } => {
            return Err(refuse(
                "a shared expert of another type than its stacks, which the fold does not run",
            ));
        }
        FfnRoute::Router { shared: None, .. } => {
            return Err(refuse(
                "a router with no shared expert, which the router does not run",
            ));
        }
        FfnRoute::Dense => {
            return Err(refuse(
                "a dense FFN, which this backend's experts do not run",
            ));
        }
    }
    if n_experts != router::EXPERTS || top_k != ROUTED_IDS {
        return Err(refuse(format!(
            "{n_experts} experts picking {top_k}; the router and the experts run \
             {} picking {ROUTED_IDS}",
            router::EXPERTS
        )));
    }
    // The file's tensors, under the plan's file-side names (chain's one
    // owner of the naming): the plan names the joined stacks, which this
    // backend reads as the routed stack and the shared expert apart.
    let file = file_names(l);
    let (gq, gate) = load.quant(&file.gate, hidden, EXPERT_K * n_experts)?;
    let (uq, up) = load.quant(&file.up, hidden, EXPERT_K * n_experts)?;
    let (sgq, shexp_gate) = load.quant(&file.sh_gate, hidden, EXPERT_K)?;
    let (suq, shexp_up) = load.quant(&file.sh_up, hidden, EXPERT_K)?;
    if [uq, sgq, suq].iter().any(|&q| q != gq) {
        return Err(refuse(format!(
            "gate {gq:?}, up {uq:?}, shared gate {sgq:?}, shared up {suq:?}; the pair runs \
             one type"
        )));
    }
    let (dq, down) = load.quant(&file.down, EXPERT_K, hidden * n_experts)?;
    let (sdq, shexp_down) = load.quant(&file.sh_down, EXPERT_K, hidden)?;
    if sdq != dq {
        return Err(refuse(format!(
            "down {dq:?}, shared down {sdq:?}; the fold runs one type"
        )));
    }
    Ok(FfnBufs {
        norm: load.f32s(&plan.ffn.ffn_norm, hidden)?,
        gate_inp: load.f32s(&file.router, n_experts * hidden)?,
        gate_inp_shexp: load.f32s(&file.router_gate, hidden)?,
        gate_up_quant: gq,
        gate,
        up,
        shexp_gate,
        shexp_up,
        down_quant: dq,
        down,
        shexp_down,
        logits: zeroed(device, router::ROWS)?,
        ids: zeroed(device, top_k + 1)?,
        weights: zeroed(device, top_k + 1)?,
        partials: zeroed(device, router::PARTIALS)?,
        ticket: zeroed(device, 1)?,
    })
}
