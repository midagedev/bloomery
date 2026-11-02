//! The verify arms of the ops: a pass of two to [`lanes::VERIFY_ROWS`] rows
//! whose row `r` is the bits of a one-row step at position `from + r`.
//!
//! What a pass of more than a step's window runs (`rows.rs`) is GEMMs over
//! 64-row tiles and a prefill flash held to the decode flash by a band; a
//! verify of two rows would pay the tile's 64 and part from the step by
//! rounding. A verify runs the kernels whose rows are the one-row kernels'
//! bits instead, one group an op as the step's walk names them:
//!
//! - the projections of a layer (P1), the attention's out projection, the FFN's
//!   down and the head: the M-row A4 mat-vecs (`kernels/matvec_rows.rs`), the
//!   weights read once for every row;
//! - the dense FFN's gate-up pair with its SwiGLU: the M-row pair;
//! - the hand-ons: a layer whose readers each have a scaled entry
//!   ([`Arena::hands_on`], every layer of a file the pass serves) hands them
//!   their input as the step does, one launch over the rows after each half's
//!   out projection ([`Kernels::norm_next_m`]), and the P1, the pair and the
//!   head read it scaled, each row the step's bits; the first layer's P1 reads
//!   the embed's rows and norms on the load;
//! - the attention: the decode flash's rows launch (`Kernels::attention_rows_layer`),
//!   each row at its own position over its own live keys. The launch holds one
//!   split-K segment count for its rows, and the decode launch of a row takes
//!   the count of its own live keys (1 up to [`ONE_PASS_KEYS`] keys, then the
//!   split's), so a verify whose rows straddle that boundary is two launches
//!   ([`chunks`]);
//! - the delta rule: the rows launch that leaves the lane ring
//!   (`Kernels::delta_rows_ring`), row `r`'s end state in the lane of position
//!   `from + r + 1` ([`lanes::Verify`]);
//! - the delta layer's out projection: the one-row head-normed mat-vec once a
//!   row, since the family has no M-row entry of it. Its weights are read once
//!   a row.
//!
//! The rows sit in the rows arena at the stride the layer's P1 writes: the
//! segmented M-row mat-vec stores row `r`'s outputs `out_end` values after row
//! `r - 1`'s (the end of its last segment), and every reader of the P1 rows
//! reads them at that stride. The mixers' outputs are dense rows of the width
//! the out projection reads.
//!
//! What the pass does not serve is refused by name before it encodes anything
//! ([`layer_refusal`], [`head_refusal`]): a routed FFN, a short conv layer, a
//! sandwich residual, a per-head output gate beside the P1 rows, and any
//! weight type but A4 where the family has an entry.

use bloomery_device::fault::LAYER_HEAD;
use bloomery_device::{Buffer, Elems};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{DeltaPlan, FfnPlan, FfnRoute, GqaPlan};

use super::super::arena::{Arena, MixerBufs, P1, P1Part, Reader, layer_word, reader_norm};
use super::super::count::GroupKind;
use super::super::lanes;
use super::super::rows::RowsArena;
use super::super::session::SessionError;
use super::{Half, MetalOps, RowsCtx, WHAT, all, all_mut, gains, refuse, rope_of};
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::kernels::{
    AttentionRowsParams, AttnCache, AttnLayer, BiasRowsParams, DeltaRowStrides, HEAD_SIMD_GROUPS,
    HeadMParams, HeadNormParams, Kernels, LoadNorm, MatvecMParams, NormNextMParams, PairMParams,
    Quant, SegMParams, segments_for_window,
};
use bloomery_device::Window;

/// The weight type the M-row family holds.
const SERVED: Quant = Quant::A4;

/// One launch of the attention rows kernels: `rows` rows from row `first`
/// of the verify, over `segments` split-K segments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Chunk {
    pub(super) first: usize,
    pub(super) rows: usize,
    pub(super) segments: u32,
}

/// The launches of the attention rows kernels a verify of `m` rows from
/// position `from` takes over a layer that attends the last `window` of its
/// keys (or all of them): row `r` has `from + r + 1` live keys, and the decode
/// launch of a row takes the segment count of its own keys
/// ([`segments_for_window`]), which a rows launch holds for all its rows. Rows
/// of one count are one launch; the count never falls as the keys grow, so the
/// launches come in order.
pub(super) fn chunks(from: u32, m: usize, window: Option<u32>) -> impl Iterator<Item = Chunk> {
    let segments = move |r: usize| {
        let row = u32::try_from(r).expect("a verify's rows fit u32");
        segments_for_window(from + row + 1, window)
    };
    let mut first = 0;
    std::iter::from_fn(move || {
        if first >= m {
            return None;
        }
        let at = segments(first);
        let mut end = first + 1;
        while end < m && segments(end) == at {
            end += 1;
        }
        let chunk = Chunk {
            first,
            rows: end - first,
            segments: at,
        };
        first = end;
        Some(chunk)
    })
}

/// What a layer is to the verify pass's decision whether to serve it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MixerKind {
    Delta,
    /// An attention layer; `head_gate`: its output gate is a value a query
    /// head in a projection of its own.
    Attention {
        head_gate: bool,
    },
    ShortConv,
}

/// What the verify pass reads of a layer to serve it or refuse it.
pub(super) struct LayerFacts {
    pub(super) layer: usize,
    pub(super) mixer: MixerKind,
    /// The layer's P1 dispatches, and the first one's type and the output
    /// values it ends at.
    pub(super) p1_parts: usize,
    pub(super) p1: (Quant, usize),
    /// The mixer's out projection's type.
    pub(super) out: Quant,
    pub(super) routed: bool,
    pub(super) gate_up: Quant,
    pub(super) down: Quant,
    /// The shared expert's gate-up type, on a layer that has one.
    pub(super) shared: Option<Quant>,
    pub(super) sandwich: bool,
}

/// Why the verify pass does not serve layer `f`, or `None`. The first reason
/// in the order the walk meets them.
pub(super) fn layer_refusal(f: &LayerFacts) -> Option<String> {
    let l = f.layer;
    if f.sandwich {
        return Some(format!(
            "layer {l}: a sandwich residual, whose join has no M-row entry"
        ));
    }
    match f.mixer {
        MixerKind::ShortConv => {
            return Some(format!(
                "layer {l}: a short conv layer, which has no verify rows"
            ));
        }
        MixerKind::Attention { head_gate: true } => {
            return Some(format!(
                "layer {l}: an output gate of a value a query head in a projection of its own: \
                 the flash's rows launch reads that gate contiguous, not at the P1 rows' stride"
            ));
        }
        MixerKind::Attention { head_gate: false } | MixerKind::Delta => {}
    }
    if f.p1_parts != 1 {
        return Some(format!(
            "layer {l}: a P1 of {} dispatches; the M-row segmented mat-vec writes one stride",
            f.p1_parts
        ));
    }
    let (quant, out_end) = f.p1;
    if quant != SERVED {
        return Some(format!(
            "layer {l}: a P1 of {}, which has no M-row mat-vec (the family holds {})",
            quant.name(),
            SERVED.name()
        ));
    }
    if !out_end.is_multiple_of(4) {
        return Some(format!(
            "layer {l}: a P1 of {out_end} values a row; the rows after the first start on a \
             16-byte boundary only at a multiple of four"
        ));
    }
    if f.mixer == MixerKind::Delta && matches!(f.out, Quant::Q6K | Quant::F32) {
        return Some(format!(
            "layer {l}: a delta out projection of {}, which has no head-normed mat-vec",
            f.out.name()
        ));
    }
    if matches!(f.mixer, MixerKind::Attention { .. }) && f.out != SERVED {
        return Some(format!(
            "layer {l}: an attention out projection of {}, which has no M-row mat-vec (the \
             family holds {})",
            f.out.name(),
            SERVED.name()
        ));
    }
    if f.routed {
        return Some(format!(
            "layer {l}: a routed FFN, whose experts have no M-slot entry"
        ));
    }
    for (what, quant) in [("gate and up", f.gate_up), ("down", f.down)] {
        if quant != SERVED {
            return Some(format!(
                "layer {l}: a dense FFN's {what} of {}, which has no M-row mat-vec (the family \
                 holds {})",
                quant.name(),
                SERVED.name()
            ));
        }
    }
    if let Some(shared) = f.shared
        && shared != f.gate_up
    {
        return Some(format!(
            "layer {l}: a shared expert of {} beside stacks of {}: the M-row pair holds one type",
            shared.name(),
            f.gate_up.name()
        ));
    }
    None
}

/// Why the verify pass does not serve a head of `quant` over `vocab` ids, or
/// `None`.
pub(super) fn head_refusal(quant: Quant, vocab: usize) -> Option<String> {
    if quant != SERVED {
        return Some(format!(
            "the head is {}, which has no M-row mat-vec (the family holds {})",
            quant.name(),
            SERVED.name()
        ));
    }
    if !vocab.is_multiple_of(4) {
        return Some(format!(
            "a vocabulary of {vocab} ids: the M-row head's argmax reads whole float4 quads"
        ));
    }
    None
}

/// The facts of layer `l` of `arena`.
fn facts(arena: &Arena, l: usize, sandwich: bool) -> LayerFacts {
    let layer = &arena.layers[l];
    let p1: &P1 = layer.mixer.p1();
    let first = p1
        .parts
        .first()
        .map_or((Quant::F32, 0), |p| (p.quant, p.out_end));
    let (mixer, out) = match &layer.mixer {
        MixerBufs::Delta(d) => (MixerKind::Delta, d.out_quant),
        MixerBufs::Attention(a) => (
            MixerKind::Attention {
                head_gate: a.gate.is_some(),
            },
            a.out_quant,
        ),
        MixerBufs::ShortConv(c) => (MixerKind::ShortConv, c.out_quant),
    };
    LayerFacts {
        layer: l,
        mixer,
        p1_parts: p1.parts.len(),
        p1: first,
        out,
        routed: layer.ffn.router.is_some(),
        gate_up: layer.ffn.gate_up_quant,
        down: layer.ffn.down_quant,
        shared: layer.ffn.shared.as_ref().map(|s| s.gate_up_quant),
        sandwich,
    }
}

/// Whether the verify pass serves every layer and the head of `arena`: the
/// first thing it cannot is refused by name. Nothing is encoded.
pub(in super::super) fn serves(arena: &Arena) -> Result<(), SessionError> {
    for l in 0..arena.layers.len() {
        let f = facts(arena, l, arena.layers[l].post.is_some());
        if let Some(why) = layer_refusal(&f) {
            return Err(unserved(why));
        }
    }
    match head_refusal(arena.head_quant, arena.vocab) {
        Some(why) => Err(unserved(why)),
        None => Ok(()),
    }
}

/// The refusal of a verify over what the pass does not serve.
fn unserved(why: String) -> SessionError {
    refuse(format!(
        "a verify: the pass does not serve this file ({why}); it serves attention and delta \
         layers with a dense A4 FFN and an A4 head"
    ))
}

/// The refusal of a verify arm over ops built with no rows arena.
fn no_rows<T>() -> Result<T, SessionError> {
    Err(ChainError::shape(WHAT, "a verify arm of ops built with no rows arena").into())
}

/// The refusal of a verify arm before the first verify opened its buffers.
fn no_buffers<T>() -> Result<T, SessionError> {
    Err(ChainError::shape(WHAT, "a verify arm with no verify buffers open").into())
}

/// The layer's single P1 dispatch, which the preflight judged served.
fn single(l: usize, p1: &P1) -> Result<&P1Part, SessionError> {
    match p1.parts.as_slice() {
        [part] if part.quant == SERVED => Ok(part),
        _ => Err(refuse(format!(
            "layer {l}: a verify over a P1 that is not one A4 dispatch"
        ))),
    }
}

/// The rows a hand-on wrote and their scale words, as windows.
type Handed<'b> = (Window<'b, f32, MetalDevice>, Window<'b, f32, MetalDevice>);

/// The rows the hand-on wrote over the verify's `m` rows and their `m` scale
/// words: what a reader that takes its input scaled reads.
fn handed_rows<'b>(
    handed: Option<&'b MetalBuffer<f32>>,
    hidden: usize,
    m: usize,
) -> Result<Handed<'b>, SessionError> {
    let Some(xn) = handed else {
        return no_buffers();
    };
    Ok((
        xn.window(Elems(0), Elems(m * hidden))?,
        xn.window(Elems(m * hidden), Elems(m))?,
    ))
}

/// The refusal of a reader that is not handed its input: the M-row pair and
/// head read it scaled only.
fn not_handed<T>(reader: Reader) -> Result<T, SessionError> {
    Err(refuse(format!(
        "{reader:?} takes no hand-on's rows, and the M-row entry of it reads them scaled only"
    )))
}

/// The input a segmented reader takes over the verify's `m` rows: the rows the
/// hand-on wrote and their `m` scale words when the reader takes them scaled,
/// else the residual rows with the reader's gain.
fn reader_rows<'b>(
    scaled: bool,
    x: &'b MetalBuffer<f32>,
    handed: Option<&'b MetalBuffer<f32>>,
    gain: &'b MetalBuffer<f32>,
    eps: f32,
    hidden: usize,
    m: usize,
) -> Result<(Window<'b, f32, MetalDevice>, LoadNorm<'b>), SessionError> {
    if scaled {
        let (rows, scale) = handed_rows(handed, hidden, m)?;
        return Ok((rows, LoadNorm::Scaled { scale }));
    }
    Ok((
        x.window(Elems(0), Elems(m * hidden))?,
        LoadNorm::Gain {
            gain: all(gain)?,
            eps,
        },
    ))
}

impl MetalOps<'_, '_> {
    /// Layer `l`'s P1 over the verify's rows: the norm on load, or the input
    /// the layer before handed on, and one M-row segmented mat-vec, rows
    /// `out_end` values apart in the rows arena.
    fn verify_p1(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.rows_slot(m);
        let scaled = self.arena.reads_scaled(Reader::P1(l));
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            ..
        } = &mut *self.arena;
        let p1 = layers[l].mixer.p1();
        let part = single(l, p1)?;
        let (x, norm) = reader_rows(
            scaled,
            &rows.x,
            rows.verify.as_ref().map(|v| &v.xn),
            &p1.norm,
            *eps,
            *hidden,
            m,
        )?;
        self.kernels.matvec_seg_m(
            self.enc,
            SegMParams {
                quant: part.quant,
                m,
                segments: &part.segments,
                weights: all(&part.weights)?,
                x,
                y: rows.proj.window_mut(Elems(0), Elems(m * part.out_end))?,
                norm,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s q, k and v biases added to the verify's P1 rows, one group.
    fn verify_attn_bias(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let MixerBufs::Attention(a) = &self.arena.layers[l].mixer else {
            return Err(refuse(format!("layer {l}: a bias op on another mixer")));
        };
        let bias = a.bias.as_ref().ok_or_else(|| {
            refuse(format!(
                "layer {l}: the plan has q, k and v biases and the layer's buffers hold none"
            ))
        })?;
        let stride = single(l, &a.p1)?.out_end;
        let cols = bias.len().get();
        self.kernels.bias_rows(
            self.enc,
            BiasRowsParams {
                rows: m,
                cols,
                y: rows
                    .proj
                    .window_mut(Elems(a.q.start), Elems((m - 1) * stride + cols))?,
                y_stride: stride,
                bias: all(bias)?,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s attention over the verify's rows: the decode flash's rows
    /// launch, row `r` at position `from + r` over its own live keys, its
    /// output a dense row of the width the out projection reads. One launch
    /// per segment count the rows take ([`chunks`]), each closed by a barrier:
    /// the next launch reads the keys this one appended.
    fn verify_attn_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.rows_slot(m);
        let from = self.pos;
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Some(verify) = rows.verify.as_mut() else {
            return no_buffers();
        };
        let Arena {
            layers,
            ropes,
            faults,
            eps,
            ctx: cache_rows,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Attention(a) = &mut layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: an attention op on a layer of another mixer"),
            )
            .into());
        };
        let rope = rope_of(ropes, l, a.rope)?;
        let stride = single(l, &a.p1)?.out_end;
        let out_cols = geom
            .attn
            .out_len()
            .ok_or_else(|| ChainError::shape(WHAT, "the attention output does not fit"))?;
        let kv_heads = geom.attn.kv_heads;
        let window = a.window;
        for chunk in chunks(from, m, window) {
            let partials = geom
                .attn
                .partials_len(chunk.segments as usize)
                .and_then(|row| row.checked_mul(chunk.rows))
                .ok_or_else(|| refuse("a verify's partials do not fit"))?;
            // A window of a part of a row, from the chunk's first row: it
            // ends where the chunk's last row's part does.
            let at = |range_start: usize| {
                (
                    Elems(chunk.first * stride + range_start),
                    Elems(chunk.rows * stride - range_start),
                )
            };
            let (q_at, q_len) = at(a.q.start);
            let (k_at, k_len) = at(a.k.start);
            let (v_at, v_len) = at(a.v.start);
            self.kernels.attention_rows_layer(
                self.enc,
                AttentionRowsParams {
                    head_dim: geom.attn.head,
                    query_heads: geom.attn.query_heads,
                    kv_heads,
                    segments: chunk.segments as usize,
                    ctx: *cache_rows,
                    eps: *eps,
                    layer: layer_word(l),
                    from: from + u32::try_from(chunk.first).expect("a verify's rows fit u32"),
                    rows: chunk.rows,
                    q: rows.proj.window(q_at, q_len)?,
                    q_stride: stride,
                    k: rows.proj.window(k_at, k_len)?,
                    k_stride: stride,
                    v: rows.proj.window(v_at, v_len)?,
                    v_stride: stride,
                    q_gain: gains(&a.q_norm)?,
                    k_gain: gains(&a.k_norm)?,
                    rope: all(rope)?,
                    cache: AttnCache::F16 {
                        k: all_mut(&mut a.k_cache)?,
                        v: all_mut(&mut a.v_cache)?,
                    },
                    partials: verify.partials.window_mut(Elems(0), Elems(partials))?,
                    out: rows
                        .mix
                        .window_mut(Elems(chunk.first * out_cols), Elems(chunk.rows * out_cols))?,
                    out_stride: out_cols,
                    fault: &faults[slot],
                    ticket: verify
                        .ticket
                        .window_mut(Elems(0), Elems(chunk.rows * kv_heads))?,
                },
                AttnLayer {
                    window,
                    head_gate: None,
                },
            )?;
            self.enc.memory_barrier();
        }
        Ok(())
    }

    /// Layer `l`'s attention out projection of the verify's rows, added into
    /// the residual rows: one M-row residual mat-vec.
    fn verify_attn_out(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Attention(a) = &layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: an attention op on a layer of another mixer"),
            )
            .into());
        };
        let out_cols = geom
            .attn
            .out_len()
            .ok_or_else(|| ChainError::shape(WHAT, "the attention output does not fit"))?;
        self.kernels.matvec_m(
            self.enc,
            MatvecMParams {
                quant: a.out_quant,
                m,
                rows: *hidden,
                cols: out_cols,
                weights: all(&a.out)?,
                x: rows.mix.window(Elems(0), Elems(m * out_cols))?,
                y: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s delta step over the verify's rows: the rows launch that
    /// leaves the lane ring, row `r`'s end state in the lane of position
    /// `from + r + 1`, and the conv ring's commit. The numerators and the
    /// partial sums land in the rows arena as the pass's do, the P1 rows the
    /// layer's own `out_end` values apart.
    fn verify_delta_core(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let plan = lanes::Verify::at(self.pos, m).map_err(|e| refuse(e.to_string()))?;
        let stride = single(l, self.arena.layers[l].mixer.p1())?.out_end;
        self.delta_rows_launch(
            l,
            m,
            stride,
            (plan.lane_at(), plan.lane_to()),
            Kernels::delta_rows_ring,
        )
    }

    /// Layer `l`'s delta out projection of the verify's rows: the head-normed
    /// residual mat-vec a row at a time, each over its own numerator and
    /// partial sums, into its own residual row. The rows write disjoint
    /// outputs, so one barrier closes them all.
    fn verify_delta_out(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        for r in 0..m {
            self.verify_delta_out_row(l, m, r)?;
        }
        self.enc.memory_barrier();
        Ok(())
    }

    /// Row `r` of layer `l`'s delta out projection: the one-row head-normed
    /// residual mat-vec over the row's numerator and partial sums, into the
    /// row's residual row. No barrier: the rows write disjoint outputs.
    fn verify_delta_out_row(&mut self, l: usize, m: usize, r: usize) -> Result<(), SessionError> {
        let slot = self.rows_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let MixerBufs::Delta(d) = &layers[l].mixer else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {l}: a delta op on a layer of another mixer"),
            )
            .into());
        };
        let dense = DeltaRowStrides::of(&geom.delta, d.n_k).ok_or_else(|| SessionError::Shape {
            what: WHAT,
            detail: format!("{} key heads do not fit", d.n_k),
        })?;
        let cols = geom.delta.value_heads * geom.delta.head_cols;
        self.kernels.headnorm_resid(
            self.enc,
            HeadNormParams {
                quant: d.out_quant,
                rows: *hidden,
                cols,
                weights: all(&d.out)?,
                g: rows.mix.window(Elems(r * dense.g), Elems(cols))?,
                partials: rows
                    .delta_partials
                    .window(Elems(r * dense.partials), Elems(geom.delta.partials_stride))?,
                y: rows.x.window_mut(Elems(r * *hidden), Elems(*hidden))?,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        Ok(())
    }

    /// Layer `l`'s dense FFN over the verify's rows: the M-row gate-up pair
    /// with its SwiGLU, then the M-row residual mat-vec of the down
    /// projection. A dense slot's route weight is 1, so the down is the plain
    /// mat-vec the step runs.
    fn verify_dense_ffn(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        self.verify_pair(l, m)?;
        self.verify_down(l, m)
    }

    /// Layer `l`'s dense gate-up pair over the verify's rows: the input the
    /// layer's mixer half handed on and the SwiGLU, each row's activations the
    /// slots' whole span after the row before, closed by a barrier.
    fn verify_pair(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let slot = self.rows_slot(m);
        if !self.arena.reads_scaled(Reader::Pair(l)) {
            return not_handed(Reader::Pair(l));
        }
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            faults,
            hidden,
            n_experts,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        if f.router.is_some() {
            return Err(refuse(format!(
                "layer {l}: a routed FFN over a verify, whose experts have no M-slot entry"
            )));
        }
        let (slots, k) = (geom.experts.slots, geom.experts.k);
        let ids = f
            .ids
            .window(Elems(0), Elems(geom.experts.routed_ids))?
            .cast::<i32>()?;
        // The pair's rows sit the slots' whole span apart, the last ending at
        // its own last slot's outputs; a dense FFN has one slot of `k` values.
        let y_len = (slots - 1) * k + k + (m - 1) * slots * k;
        let (x, scale) = handed_rows(rows.verify.as_ref().map(|v| &v.xn), *hidden, m)?;
        self.kernels.pair_m(
            self.enc,
            PairMParams {
                quant: f.gate_up_quant,
                shared_quant: f
                    .shared
                    .as_ref()
                    .map_or(f.gate_up_quant, |s| s.gate_up_quant),
                m,
                slots,
                k,
                cols: *hidden,
                n_experts: *n_experts,
                ids,
                gate: all(&f.gate)?,
                up: all(&f.up)?,
                shared_gate: f.shared.as_ref().map(|s| all(&s.gate)).transpose()?,
                shared_up: f.shared.as_ref().map(|s| all(&s.up)).transpose()?,
                x,
                scale,
                y: rows.h.window_mut(Elems(0), Elems(y_len))?,
                out_stride: k,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s dense down projection over the verify's rows: the M-row
    /// residual mat-vec over the pair's activations, closed by a barrier.
    fn verify_down(&mut self, l: usize, m: usize) -> Result<(), SessionError> {
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Arena {
            layers,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        if f.router.is_some() {
            return Err(refuse(format!(
                "layer {l}: a routed FFN over a verify, whose experts have no M-slot entry"
            )));
        }
        let k = geom.experts.k;
        self.kernels.matvec_m(
            self.enc,
            MatvecMParams {
                quant: f.down_quant,
                m,
                rows: *hidden,
                cols: k,
                weights: all(&f.down)?,
                x: rows.h.window(Elems(0), Elems(m * k))?,
                y: rows.x.window_mut(Elems(0), Elems(m * *hidden))?,
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// Layer `l`'s hand-on group of `half` over the verify's rows, on a layer
    /// that hands on ([`Arena::hands_on`]): the stream the half's out projection
    /// added into, each row times the next reader's gain, and that reader's
    /// scale of each row, into the verify's hand-on rows. One launch for the
    /// rows, each row the step's hand-on bits ([`MetalOps::hand_on`]). A layer
    /// that does not hand on encodes nothing.
    fn verify_hand_on(&mut self, l: usize, m: usize, half: Half) -> Result<(), SessionError> {
        if !self.arena.hands_on(l) {
            return Ok(());
        }
        let slot = self.rows_slot(m);
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Some(verify) = rows.verify.as_mut() else {
            return no_buffers();
        };
        let Arena {
            layers,
            faults,
            eps,
            hidden,
            head_quant,
            head_norm,
            ..
        } = &*self.arena;
        let reader = Reader::after(layers.len(), l, half);
        let (quant, gain) =
            reader_norm(layers, (*head_quant, head_norm), reader).ok_or_else(|| {
                refuse(format!(
                    "layer {l}: a half hands on to {reader:?}, which has no scaled entry"
                ))
            })?;
        self.kernels.norm_next_m(
            self.enc,
            NormNextMParams {
                quant,
                m,
                x: rows.x.window(Elems(0), Elems(m * *hidden))?,
                gain: all(gain)?,
                eps: *eps,
                layer: reader.layer(),
                out: verify.xn.window_mut(Elems(0), Elems(m * *hidden + m))?,
                fault: &faults[slot],
            },
        )?;
        self.enc.memory_barrier();
        Ok(())
    }

    /// The head over the verify's rows: the M-row logits over the input the
    /// last layer handed on, and one argmax a row, each row's pick in its own
    /// word.
    fn verify_head(&mut self, m: usize) -> Result<(), SessionError> {
        let slot = self.rows_slot(m);
        if !self.arena.reads_scaled(Reader::Head) {
            return not_handed(Reader::Head);
        }
        let Some(ctx) = self.rows.as_mut() else {
            return no_rows();
        };
        let rows = &mut *ctx.arena;
        let Some(verify) = rows.verify.as_mut() else {
            return no_buffers();
        };
        let Arena {
            head_w,
            head_quant,
            faults,
            hidden,
            vocab,
            ..
        } = &mut *self.arena;
        let (x, scale) = handed_rows(Some(&verify.xn), *hidden, m)?;
        self.kernels.head_m(
            self.enc,
            HeadMParams {
                quant: *head_quant,
                m,
                rows: *vocab,
                k: *hidden,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: all(head_w)?,
                x,
                scale,
                logits: verify.logits.window_mut(Elems(0), Elems(m * *vocab))?,
                step_ids: verify.picks.window_mut(Elems(0), Elems(m))?,
                fault: &faults[slot],
                layer: LAYER_HEAD,
            },
        )?;
        Ok(())
    }
}

/// Which part of its layer a group of the verify's walk is: the group runs
/// whole, one row of the rows it runs alone (the delta layers' out
/// projection), or the hand-on after one half.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum Part {
    Whole,
    Row(usize),
    Half(Half),
}

/// The ops of a verify of `m` rows from `from`: [`MetalOps`] over the rows
/// arena with the verify's arms in place of the pass's. The ops a verify does
/// not serve keep [`ChainOps`]' default bodies, which refuse by name.
pub(in super::super) struct VerifyOps<'a, 'e> {
    ops: MetalOps<'a, 'e>,
}

impl<'a, 'e> VerifyOps<'a, 'e> {
    /// The ops of a verify whose first row is at `from`, on `enc`.
    pub(in super::super) fn new(
        enc: &'a mut ComputeEncoder<'e>,
        arena: &'a mut Arena,
        rows: &'a mut RowsArena,
        kernels: &'a Kernels,
        from: u32,
    ) -> Self {
        Self {
            ops: MetalOps::rows(
                enc,
                arena,
                RowsCtx {
                    arena: rows,
                    after: &[],
                },
                kernels,
                from,
                true,
            ),
        }
    }
}

impl VerifyOps<'_, '_> {
    /// One group of the verify's walk on the encoder open: the ops above split
    /// as [`GroupKind`] names the groups of a token, so a profile can give
    /// each its own command buffer. A group is the dispatches the walk's op
    /// runs between two barriers; `row` is the row of a group a row runs
    /// alone (the delta layer's out projection) or the half a hand-on follows,
    /// as `part` names it, and ignored by the others, and `layer` by the embed
    /// and the head. The groups of the walk's ops, run in the walk's order,
    /// encode what [`ChainOps`] below encodes.
    pub(in super::super) fn group(
        &mut self,
        kind: GroupKind,
        layer: usize,
        m: usize,
        part: Part,
    ) -> Result<(), SessionError> {
        let o = &mut self.ops;
        match kind {
            GroupKind::Embed => o.rows_embed(m),
            GroupKind::AttnP1 | GroupKind::DeltaP1 => o.verify_p1(layer, m),
            GroupKind::AttnBias => o.verify_attn_bias(layer, m),
            GroupKind::AttnFlash => o.verify_attn_core(layer, m),
            GroupKind::AttnOut => o.verify_attn_out(layer, m),
            GroupKind::DeltaCore => o.verify_delta_core(layer, m),
            GroupKind::DeltaOut => {
                let Part::Row(row) = part else {
                    return Err(refuse(
                        "a delta out group of a verify names no row".to_string(),
                    ));
                };
                o.verify_delta_out_row(layer, m, row)?;
                o.enc.memory_barrier();
                Ok(())
            }
            GroupKind::HandOn => {
                let Part::Half(half) = part else {
                    return Err(refuse(
                        "a hand-on group of a verify names no half".to_string(),
                    ));
                };
                o.verify_hand_on(layer, m, half)
            }
            GroupKind::GateUp => o.verify_pair(layer, m),
            GroupKind::Down => o.verify_down(layer, m),
            GroupKind::Head => o.verify_head(m),
            other => Err(refuse(format!(
                "{other:?}: no group of a verify's walk runs it"
            ))),
        }
    }
}

impl ChainOps for VerifyOps<'_, '_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        self.ops.rows_embed(m)
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, g: &GqaPlan) -> Result<(), SessionError> {
        self.ops.verify_p1(l.layer, l.m)?;
        if g.bias.is_some() {
            self.ops.verify_attn_bias(l.layer, l.m)?;
        }
        Ok(())
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        self.ops.verify_attn_core(l.layer, l.m)
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::VerifyOps::attn_out")?;
        self.ops.verify_attn_out(l.layer, l.m)?;
        self.ops.verify_hand_on(l.layer, l.m, Half::Mixer)
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        self.ops.verify_p1(l.layer, l.m)
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        self.ops.verify_delta_core(l.layer, l.m)
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::VerifyOps::delta_out")?;
        self.ops.verify_delta_out(l.layer, l.m)?;
        self.ops.verify_hand_on(l.layer, l.m, Half::Mixer)
    }

    fn route(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        match f.route {
            // A dense FFN has no router: its one slot is the fixed route the
            // open laid, and its norm is the gate-up's.
            FfnRoute::Dense => Ok(()),
            FfnRoute::Router { .. } => Err(refuse(format!(
                "layer {}: a routed FFN over a verify, whose experts have no M-slot entry",
                l.layer
            ))),
        }
    }

    fn experts(&mut self, l: LayerCtx<'_>, f: &FfnPlan) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::VerifyOps::experts")?;
        match f.route {
            FfnRoute::Dense => {
                self.ops.verify_dense_ffn(l.layer, l.m)?;
                self.ops.verify_hand_on(l.layer, l.m, Half::Ffn)
            }
            FfnRoute::Router { .. } => Err(refuse(format!(
                "layer {}: a routed FFN over a verify, whose experts have no M-slot entry",
                l.layer
            ))),
        }
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        self.ops.verify_head(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{ONE_PASS_KEYS, SEGMENT_CAP};

    fn list(from: u32, m: usize, window: Option<u32>) -> Vec<(usize, usize, u32)> {
        chunks(from, m, window)
            .map(|c| (c.first, c.rows, c.segments))
            .collect()
    }

    /// Rows well inside one segment count are one launch, whatever their
    /// number.
    #[test]
    fn rows_of_one_segment_count_are_one_launch() {
        assert_eq!(list(0, 2, None), [(0, 2, 1)]);
        assert_eq!(list(16, 4, None), [(0, 4, 1)]);
        let cap = SEGMENT_CAP as u32;
        assert_eq!(list(2000, 2, None), [(0, 2, cap)]);
    }

    /// The rows that straddle the one-pass ceiling are two launches, each
    /// with the count the decode launch of its rows takes: row `r` has
    /// `from + r + 1` live keys. The last row with [`ONE_PASS_KEYS`] keys is
    /// the one-pass flash's and the first with one more is the split's.
    ///
    /// Mutant: `chunks` takes the last row's count for every row, or cuts at
    /// the row with exactly [`ONE_PASS_KEYS`] keys.
    #[test]
    fn rows_that_straddle_the_one_pass_ceiling_are_two_launches() {
        let ceiling = u32::try_from(ONE_PASS_KEYS).expect("the ceiling fits u32");
        let cap = SEGMENT_CAP as u32;
        // `ceiling - 1` is the row with `ceiling` keys, the last of the first
        // count.
        assert_eq!(list(ceiling - 1, 2, None), [(0, 1, 1), (1, 1, cap)]);
        assert_eq!(list(ceiling - 2, 2, None), [(0, 2, 1)]);
        assert_eq!(list(ceiling, 2, None), [(0, 2, cap)]);
        assert_eq!(list(ceiling - 2, 4, None), [(0, 2, 1), (2, 2, cap)]);
        assert_eq!(list(ceiling - 1, 4, None), [(0, 1, 1), (1, 3, cap)]);
        // The rows tile the verify exactly, in order.
        for from in [0, 1, ceiling - 3, ceiling - 1, ceiling, ceiling + 7] {
            for m in 2..=4 {
                let mut next = 0;
                for c in chunks(from, m, None) {
                    assert_eq!(c.first, next, "from {from}, {m} rows");
                    assert!(c.rows > 0);
                    next += c.rows;
                }
                assert_eq!(next, m, "from {from}, {m} rows");
            }
        }
    }

    /// A windowed layer's count follows the keys the window holds, not the
    /// live ones: a window under the ceiling never splits.
    #[test]
    fn a_windowed_layer_counts_the_keys_of_its_window() {
        let ceiling = u32::try_from(ONE_PASS_KEYS).expect("the ceiling fits u32");
        assert_eq!(list(ceiling - 1, 2, Some(256)), [(0, 2, 1)]);
        assert_eq!(list(4000, 4, Some(256)), [(0, 4, 1)]);
    }

    fn served() -> LayerFacts {
        LayerFacts {
            layer: 7,
            mixer: MixerKind::Delta,
            p1_parts: 1,
            p1: (Quant::A4, 16480),
            out: Quant::A4,
            routed: false,
            gate_up: Quant::A4,
            down: Quant::A4,
            shared: None,
            sandwich: false,
        }
    }

    fn says(f: &LayerFacts, needle: &str) {
        match layer_refusal(f) {
            Some(why) => {
                assert!(why.contains(needle), "{why} does not say {needle}");
                assert!(why.starts_with("layer 7"), "{why} does not name the layer");
            }
            None => panic!("the layer is served, a refusal naming {needle} is expected"),
        }
    }

    /// A dense A4 layer of either mixer is served, and every other kind is
    /// refused by name with its layer, at the first thing it has the pass
    /// cannot run.
    ///
    /// Mutant: a refusal arm of `layer_refusal` removed.
    #[test]
    fn what_the_pass_does_not_serve_is_refused_by_name() {
        assert_eq!(layer_refusal(&served()), None);
        let attention = LayerFacts {
            mixer: MixerKind::Attention { head_gate: false },
            p1: (Quant::A4, 8216),
            ..served()
        };
        assert_eq!(layer_refusal(&attention), None);
        says(
            &LayerFacts {
                routed: true,
                ..served()
            },
            "routed FFN",
        );
        says(
            &LayerFacts {
                mixer: MixerKind::ShortConv,
                ..served()
            },
            "short conv",
        );
        says(
            &LayerFacts {
                sandwich: true,
                ..served()
            },
            "sandwich",
        );
        says(
            &LayerFacts {
                mixer: MixerKind::Attention { head_gate: true },
                ..served()
            },
            "output gate",
        );
        says(
            &LayerFacts {
                p1_parts: 2,
                ..served()
            },
            "2 dispatches",
        );
        says(
            &LayerFacts {
                p1: (Quant::A8, 16480),
                ..served()
            },
            "a8",
        );
        says(
            &LayerFacts {
                p1: (Quant::A4H, 16480),
                ..served()
            },
            "a4h",
        );
        says(
            &LayerFacts {
                p1: (Quant::A4, 16482),
                ..served()
            },
            "16482 values a row",
        );
        says(
            &LayerFacts {
                mixer: MixerKind::Attention { head_gate: false },
                out: Quant::Q6K,
                ..attention
            },
            "attention out projection",
        );
        says(
            &LayerFacts {
                out: Quant::Q6K,
                ..served()
            },
            "head-normed",
        );
        says(
            &LayerFacts {
                gate_up: Quant::A8,
                ..served()
            },
            "gate and up",
        );
        says(
            &LayerFacts {
                down: Quant::Q6K,
                ..served()
            },
            "down",
        );
        says(
            &LayerFacts {
                shared: Some(Quant::A8),
                ..served()
            },
            "shared expert",
        );
    }

    /// The head: A4 over a vocabulary of whole quads is served.
    #[test]
    fn the_head_is_a4_over_whole_quads() {
        assert_eq!(head_refusal(Quant::A4, 248_320), None);
        assert!(head_refusal(Quant::A8, 248_320).unwrap().contains("a8"));
        assert!(head_refusal(Quant::A4, 248_322).unwrap().contains("quads"));
    }
}
