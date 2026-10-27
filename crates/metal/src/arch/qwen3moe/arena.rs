//! The qwen35moe arena: every buffer one token touches, allocated when the
//! session opens and never in a step.
//!
//! The weights are copied from the file into buffers of their own. A layer's
//! P1 projection is one buffer per weight type, each the segments of one
//! dispatch. The activations, the caches, the recurrent stores, the router's
//! and the head's tickets, the host's id slot, the step record slots and the
//! fault words are the session's. Every name, type and length comes from the
//! layout [`super::check`] resolved from the file's header (the plan, the
//! file's shapes and a launch's geometry): a shape or a type no launch runs was
//! refused there, by name, before any buffer was allocated, so the loader
//! checks nothing but that the split gives a tensor's bytes.
//!
//! Two tokens may be committed at once, the second reading what the first
//! writes. What the second would overwrite before the first is done is held
//! twice: the step record (the head's id and the fault word) has a slot per
//! position parity, and so does the delta state, whose lanes a token reads by
//! its position's parity and writes by the next position's.

use core::ops::Range;

use bloomery_device::{Elem, Elems};
use chain::rope::{Direction, RopeTable};
use gguf::Split;
use gguf::quant::GgmlType;

use super::check::{
    AttnLayout, DeltaLayout, FfnLayout, LayerLayout, Layout, MixerLayout, P1Layout, Site, Values,
};
use super::count::{GroupBytes, GroupKind, Layer, Projection};
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::fault::MetalFaultWord;
use crate::geom::Geometry;
use crate::kernels::{HeadMap, Quant, Segment, head_groups, segments_for_keys};
use crate::residency::Alloc;

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

/// A shared expert's router row and weights.
pub(super) struct SharedBufs {
    pub(super) gate_inp: MetalBuffer<f32>,
    pub(super) gate: MetalBuffer<u8>,
    pub(super) up: MetalBuffer<u8>,
    pub(super) down: MetalBuffer<u8>,
}

/// A layer's FFN half: the router's weights and outputs, and the experts.
/// Each layer keeps its own router outputs, so every layer's picks are
/// readable after a step.
pub(super) struct FfnBufs {
    pub(super) norm: MetalBuffer<f32>,
    pub(super) gate_inp: MetalBuffer<f32>,
    pub(super) gate_up_quant: Quant,
    pub(super) gate: MetalBuffer<u8>,
    pub(super) up: MetalBuffer<u8>,
    pub(super) down_quant: Quant,
    pub(super) down: MetalBuffer<u8>,
    /// The shared expert, on a router that runs one.
    pub(super) shared: Option<SharedBufs>,
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
    /// `ctx` rows of `AttnGeom::rot` interleaved `(cos, sin)`.
    pub(super) rope: MetalBuffer<f32>,
    /// The experts' activations, one row of `ExpertsGeom::k` per slot.
    pub(super) h: MetalBuffer<f32>,
    pub(super) logits: MetalBuffer<f32>,
    pub(super) candidates: MetalBuffer<u64>,
    pub(super) head_ticket: MetalBuffer<u32>,
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
/// still refuses is a tensor whose bytes the split cannot give.
struct Loader<'a> {
    device: Alloc<'a>,
    split: &'a Split,
    geom: &'a Geometry,
}

impl Loader<'_> {
    /// The bytes of tensor `name`.
    fn bytes(&self, name: &str) -> Result<&[u8], SessionError> {
        let (shard, info) = self
            .split
            .find(name)
            .ok_or_else(|| refuse(format!("the file has no tensor {name}")))?;
        let file = self
            .split
            .shard(shard)
            .ok_or_else(|| refuse(format!("{name}: shard {shard} is not open")))?;
        file.data(info).map_err(|e| refuse(format!("{name}: {e}")))
    }

    /// The f32 values of `v`: an F32 tensor as it is, or MLX's affine 8-bit
    /// tensor of the same values, dequantized exactly at open through the host
    /// reader (`s·q + b` in f32, the pair's product exact). The reader's own
    /// refusals (a non-finite scale or bias) name the tensor and its block.
    fn values(&self, v: &Values) -> Result<MetalBuffer<f32>, SessionError> {
        let bytes = self.bytes(&v.name)?;
        let mut values = vec![0f32; v.len];
        match v.ty {
            GgmlType::F32 => {
                for (x, b) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
                    *x = f32::from_le_bytes(*b);
                }
            }
            GgmlType::MLX_A8 => {
                gguf::quant::dequant_row(GgmlType::MLX_A8, bytes, &mut values)
                    .map_err(|e| refuse(format!("{}: {e}", v.name)))?;
            }
            other => {
                return Err(refuse(format!(
                    "{} is {other:?}; the step reads F32 or MLX_A8",
                    v.name
                )));
            }
        }
        let mut buffer = self.device.buffer(Elems(v.len))?;
        buffer.write(&values)?;
        Ok(buffer)
    }

    /// Site `s` as its own buffer.
    fn site(&self, s: &Site) -> Result<MetalBuffer<u8>, SessionError> {
        let bytes = self.bytes(&s.name)?;
        let mut buffer = self.device.buffer(Elems(bytes.len()))?;
        buffer.write(bytes)?;
        Ok(buffer)
    }

    /// The P1 of `p`: one dispatch per weight type, its segments in the order
    /// the layout gives, each site's output at the offset its place gives.
    /// Returns the P1 and every site's output range.
    fn p1(&self, p: &P1Layout) -> Result<(P1, Vec<Range<usize>>), SessionError> {
        let mut parts: Vec<P1Part> = Vec::with_capacity(p.dispatches.len());
        for d in &p.dispatches {
            let bytes: Vec<&[u8]> = d
                .members
                .iter()
                .map(|&i| self.bytes(&p.sites[i].name))
                .collect::<Result<_, _>>()?;
            let mut host = Vec::with_capacity(bytes.iter().map(|b| b.len()).sum());
            let mut segments = Vec::with_capacity(d.members.len());
            for (&i, b) in d.members.iter().zip(&bytes) {
                let site = &p.sites[i];
                segments.push(Segment {
                    cols: site.cols,
                    rows: site.rows,
                    weight_blocks: host.len() / d.quant.block_bytes(),
                    out_offset: p.range(i).start,
                });
                host.extend_from_slice(b);
            }
            let mut weights = self.device.buffer(Elems(host.len()))?;
            weights.write(&host)?;
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
    fn layer(&self, layer: &LayerLayout, top_k: usize) -> Result<LayerBufs, SessionError> {
        let mixer = match &layer.mixer {
            MixerLayout::Delta(d) => MixerBufs::Delta(delta(self, d)?),
            MixerLayout::Attention(a) => MixerBufs::Attention(attention(self, a)?),
        };
        Ok(LayerBufs {
            mixer,
            ffn: ffn(self, self.device, &layer.ffn, top_k)?,
        })
    }
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

impl Arena {
    /// The arena of `g` (the layout [`super::check`] resolved from `split`'s
    /// header, at the launch geometry `geom` of the file's instance): each
    /// weight copied from `split` into a buffer of its own, and every buffer a
    /// token touches made, all zeroed but the weights, the rope table and the
    /// lane words.
    pub(super) fn open(
        device: Alloc<'_>,
        split: &Split,
        geom: &Geometry,
        g: &Layout,
    ) -> Result<Arena, SessionError> {
        let (hidden, vocab, ctx) = (g.hidden, g.vocab, g.ctx);
        let load = Loader {
            device,
            split,
            geom,
        };
        let embed = load.site(&g.embed)?;
        let head_w = load.site(&g.head)?;
        let head_norm = load.values(&g.head_norm)?;
        let mut layers = Vec::with_capacity(g.layers.len());
        for (l, layer) in g.layers.iter().enumerate() {
            layers.push(load.layer(layer, g.top_k).map_err(|e| at_layer(l, e))?);
        }
        let mut rope = device.buffer(Elems(g.rope_len))?;
        rope.write(&rope_rows(&g.rope, ctx, geom.attn.rot))?;
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
            rope,
            h: zeroed(device, geom.experts.slots * geom.experts.k)?,
            logits: zeroed(device, vocab)?,
            candidates: zeroed(device, head_groups(vocab))?,
            head_ticket: zeroed(device, 1)?,
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
                let attn = &self.geom.attn;
                let kv_row = count_u64(attn.kv_heads * attn.head, "a cache row") * 2;
                let segments = segments_for_keys(pos + 1);
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
                    // The query with its gate, the rope row, and the flash's
                    // output.
                    acts: f32s(a.q.len()) + f32s(attn.rot) + wrote,
                    // The live keys and values read, this token's row written.
                    state: 2 * (n_keys * kv_row + kv_row),
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
                Ok(GroupBytes {
                    weights: bytes_of(&f.gate_inp)
                        + f.shared.as_ref().map_or(0, |s| bytes_of(&s.gate_inp))
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
                let shared = self.geom.router.shared;
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
                let (weights, acts) = match kind {
                    GroupKind::GateUp => (
                        picks * (expert(&f.gate)? + expert(&f.up)?)
                            + f.shared
                                .as_ref()
                                .map_or(0, |s| bytes_of(&s.gate) + bytes_of(&s.up)),
                        // The residual normed as it is loaded, the ids read,
                        // the experts' rows written.
                        f32s(self.hidden)
                            + bytes_of(&f.norm)
                            + 4 * count_u64(ids.len(), "ids")
                            + bytes_of(&self.h),
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

fn attention(load: &Loader<'_>, a: &AttnLayout) -> Result<AttnBufs, SessionError> {
    let (p1, ranges) = load.p1(&a.p1)?;
    let out = load.site(&a.out)?;
    let device = load.device;
    Ok(AttnBufs {
        q: ranges[0].clone(),
        k: ranges[1].clone(),
        v: ranges[2].clone(),
        p1,
        q_norm: load.values(&a.q_norm)?,
        k_norm: load.values(&a.k_norm)?,
        out,
        out_quant: a.out.quant,
        k_cache: zeroed(device, a.cache)?,
        v_cache: zeroed(device, a.cache)?,
    })
}

fn ffn(
    load: &Loader<'_>,
    device: Alloc<'_>,
    f: &FfnLayout,
    top_k: usize,
) -> Result<FfnBufs, SessionError> {
    Ok(FfnBufs {
        norm: load.values(&f.norm)?,
        // The router and the shared-expert gate: F32 in a GGUF file, MLX's
        // affine 8-bit in an MLX file, dequantized exactly at open.
        gate_inp: load.values(&f.gate_inp)?,
        gate_up_quant: f.gate_up_quant,
        gate: load.site(&f.gate)?,
        up: load.site(&f.up)?,
        down_quant: f.down_quant,
        down: load.site(&f.down)?,
        shared: f
            .shared
            .as_ref()
            .map(|s| {
                Ok::<_, SessionError>(SharedBufs {
                    gate_inp: load.values(&s.gate_inp)?,
                    gate: load.site(&s.gate)?,
                    up: load.site(&s.up)?,
                    down: load.site(&s.down)?,
                })
            })
            .transpose()?,
        logits: zeroed(device, load.geom.router.rows)?,
        ids: zeroed(device, top_k + load.geom.router.shared)?,
        weights: zeroed(device, top_k + load.geom.router.shared)?,
        partials: zeroed(device, load.geom.router.partials)?,
        ticket: zeroed(device, 1)?,
    })
}

#[cfg(test)]
mod tests {
    use chain::arch::qwen3moe::plan::SiteTy;
    use chain::site::BACKEND_SITES;

    use crate::arch::qwen3moe::session::LAUNCHED;
    use crate::kernels::Quant;

    /// The session's site list and the mat-vecs' quant set name the same
    /// families, over the card's families and the backends' rows: a family
    /// the plan admits has a launch, and a launch the plan never admits is
    /// dead.
    #[test]
    fn launched_sites_are_the_arena_quants() {
        let families = SiteTy::ALL
            .iter()
            .copied()
            .chain(BACKEND_SITES.iter().map(|r| r.1));
        for s in families {
            assert_eq!(
                LAUNCHED.contains(&s),
                Quant::of_ggml(s.ggml()).is_some(),
                "{s:?}: the session's LAUNCHED and Quant::of_ggml disagree"
            );
        }
    }
}
