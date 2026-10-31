//! The qwen35moe chain's ops on Metal: each op encodes its kernels on the
//! token's one encoder, in the token DAG's order, and closes its group with a
//! barrier.
//!
//! The DAG, one group per op past the embed: `E | P1 | mixer | out | R | G |
//! W` for every layer, then `H | K`, the head's logits and its pick; a layer
//! of a dense FFN has no `R`. P1 is one dispatch per weight type of the
//! layer's projections in one group. The flash folds its segments' partials in
//! its last threadgroup, so an attention mixer is one group at any depth. A
//! layer whose residual is a sandwich has a join group `J` after the mixer's
//! out and after the down: the out and the down store into the join buffer,
//! and `J` norms that buffer into the stream ([`MetalOps::join`]). A layer
//! whose readers each have a scaled entry hands on ([`Arena::hands_on`]): its
//! joins, or on a plain residual a hand-on group `N` after the mixer's out
//! and after the down ([`MetalOps::hand_on`]), write the next reader's input,
//! the stream times that reader's gain and then the norm's scale, which the
//! reader takes in place of its norm on load. The FFN
//! norm has no group of its own: the router (R) and the gate-up pair (G) each
//! norm the residual as they load it. The head's pick is the token's last
//! group and closes no barrier; a token whose pick nobody reads encodes none
//! of the head. Every op only encodes: no wait, no allocation past the
//! launches' own argument lists, no host round trip.
//!
//! The token at `pos` binds the step record slot of its position: the head
//! writes its id there and every kernel raises that slot's fault word. A
//! host-fed token embeds the host's id slot; a token fed on the device embeds
//! the id the token before it wrote, in that token's slot: its head's pick,
//! or the id the host wrote there for a token ahead. The delta step reads the
//! lane of `pos` and writes the lane of `pos + 1`.

use bloomery_device::fault::LAYER_HEAD;
use bloomery_device::{Buffer, DeviceError, Elem, Elems, Window, WindowMut};
use chain::ChainError;
use chain::arch::qwen3moe::ops::{ChainOps, LayerCtx};
use chain::arch::qwen3moe::plan::{
    DeltaPlan, FfnPlan, FfnRoute, GqaPlan, ResidualPlan, ShortConvPlan,
};

use super::arena::{
    Arena, ConvBufs, LANES, LayerBufs, MixerBufs, Reader, lane_of, layer_word, reader_norm,
    slot_before, slot_of,
};
use super::rows::RowsArena;
use super::session::SessionError;
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::kernels::router::RouterParams;
use crate::kernels::{
    AttentionParams, AttnCache, AttnLayer, DeltaParams, EmbedParams, FoldParams, HEAD_SIMD_GROUPS,
    HeadNormParams, HeadParams, Kernels, LoadNorm, MatvecParams, Mode, NormAddNextParams,
    NormAddRowsParams, NormNextParams, NormRowsParams, PASS_ROWS, PairParams, Quant, SegParams,
    SharedPlan, ShortConvStepParams, segments_for_window,
};

/// The half of a layer a sandwich joins: the mixer's, then the FFN's, the
/// order of `SandwichPlan`'s gains and scales.
#[derive(Clone, Copy, Debug)]
pub(super) enum Half {
    Mixer,
    Ffn,
}

impl Half {
    /// The half's index into `SandwichPlan`'s `norms` and `add`.
    pub(super) fn index(self) -> usize {
        self as usize
    }
}

mod rows;

const WHAT: &str = "qwen35moe ops";

/// A request this arena or this arm does not run, refused by name.
fn refuse(detail: impl Into<String>) -> SessionError {
    SessionError::Shape {
        what: WHAT,
        detail: detail.into(),
    }
}

/// The whole of `b`.
fn all<T: Elem>(b: &MetalBuffer<T>) -> Result<Window<'_, T, MetalDevice>, DeviceError> {
    b.window(Elems(0), b.len())
}

/// The whole of `b`, written.
fn all_mut<T: Elem>(b: &mut MetalBuffer<T>) -> Result<WindowMut<'_, T, MetalDevice>, DeviceError> {
    let len = b.len();
    b.window_mut(Elems(0), len)
}

/// Values `r` of `b`.
fn part<'b, T: Elem>(
    b: &'b MetalBuffer<T>,
    r: &core::ops::Range<usize>,
) -> Result<Window<'b, T, MetalDevice>, DeviceError> {
    b.window(Elems(r.start), Elems(r.len()))
}

/// The head's windows over `arena`, for the step record slot `slot`: the
/// weights and the norm's gain whole, the logits whole, the slot's pick and
/// fault word. `x` is the residual the head norms: the arena's own for a
/// token (or the input the last join handed on, [`Arena::reads_scaled`]),
/// the last row of the rows arena for a pass.
fn head_params<'a>(
    arena: &'a mut Arena,
    x: Option<Window<'a, f32, MetalDevice>>,
    slot: usize,
) -> Result<HeadParams<'a>, SessionError> {
    let scaled = x.is_none() && arena.reads_scaled(Reader::Head);
    let Arena {
        head_w,
        head_quant,
        head_norm,
        x: residual,
        xn,
        logits,
        step_id,
        faults,
        eps,
        hidden,
        vocab,
        ..
    } = arena;
    let (x, norm) = match x {
        Some(x) => (
            x,
            LoadNorm::Gain {
                gain: all(head_norm)?,
                eps: *eps,
            },
        ),
        None if scaled => handed_on(xn.as_ref(), *hidden)?,
        None => (
            residual.window(Elems(0), Elems(*hidden))?,
            LoadNorm::Gain {
                gain: all(head_norm)?,
                eps: *eps,
            },
        ),
    };
    Ok(HeadParams {
        quant: *head_quant,
        rows: *vocab,
        k: *hidden,
        simdgroups: HEAD_SIMD_GROUPS,
        weights: all(head_w)?,
        x,
        norm,
        logits: all_mut(logits)?,
        step_id: step_id.window_mut(Elems(slot), Elems(1))?,
        fault: &faults[slot],
        layer: LAYER_HEAD,
    })
}

/// The input a join handed on and its scale word: `hidden` values of the
/// stream times the reader's gain, then the norm's scale.
fn handed_on(
    xn: Option<&MetalBuffer<f32>>,
    hidden: usize,
) -> Result<(Window<'_, f32, MetalDevice>, LoadNorm<'_>), SessionError> {
    let xn = xn.ok_or_else(|| refuse("a scaled reader over an arena with no hand-on buffer"))?;
    Ok((
        xn.window(Elems(0), Elems(hidden))?,
        LoadNorm::Scaled {
            scale: xn.window(Elems(hidden), Elems(1))?,
        },
    ))
}

/// The norm a reader of the stream applies as it loads it: `gain` and the
/// arena's epsilon over the residual, or the input its join handed on.
fn reader_input<'b>(
    scaled: bool,
    x: &'b MetalBuffer<f32>,
    xn: Option<&'b MetalBuffer<f32>>,
    gain: &'b MetalBuffer<f32>,
    eps: f32,
    hidden: usize,
) -> Result<(Window<'b, f32, MetalDevice>, LoadNorm<'b>), SessionError> {
    if scaled {
        return handed_on(xn, hidden);
    }
    Ok((
        x.window(Elems(0), Elems(hidden))?,
        LoadNorm::Gain {
            gain: all(gain)?,
            eps,
        },
    ))
}

/// Where a token's embed finds its id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Feed {
    /// The host's id slot, which the host wrote before the commit.
    Host,
    /// The step record slot of the token before it: that token's pick, or
    /// the id the host wrote there for a token ahead whose pick nobody
    /// reads.
    Device,
}

/// What a pass adds to a step's ops: the rows arena, and the rows after
/// which the delta layers leave copies of their stores (none for a pass that
/// takes none).
pub(super) struct RowsCtx<'a> {
    pub(super) arena: &'a mut RowsArena,
    pub(super) after: &'a [u32],
}

/// The arm a unit of rows takes.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Arm {
    /// One row: the token's mat-vecs.
    Step,
    /// More than a step's window: the GEMM arm over the rows arena.
    Rows,
}

/// The ops of one token over the session's arena, encoding on `enc`. `pos`
/// is the token's position: its cache row, its conv ring slot, its step
/// record slot and its delta lanes. `head` names whether the token encodes
/// the head: a step whose pick nobody reads skips it.
///
/// A pass over rows ([`MetalOps::rows`]) takes the rows arms: `pos` is the
/// position of its first row, its step record slot is its last row's, and
/// its delta lane words are the first row's lane and the lane after the last.
pub(super) struct MetalOps<'a, 'e> {
    enc: &'a mut ComputeEncoder<'e>,
    arena: &'a mut Arena,
    kernels: &'a Kernels,
    pos: u32,
    feed: Feed,
    head: bool,
    rows: Option<RowsCtx<'a>>,
}

impl<'a, 'e> MetalOps<'a, 'e> {
    pub(super) fn new(
        enc: &'a mut ComputeEncoder<'e>,
        arena: &'a mut Arena,
        kernels: &'a Kernels,
        pos: u32,
        feed: Feed,
        head: bool,
    ) -> Self {
        Self {
            enc,
            arena,
            kernels,
            pos,
            feed,
            head,
            rows: None,
        }
    }

    /// The ops of a pass whose first row is at `from`, over `rows`.
    pub(super) fn rows(
        enc: &'a mut ComputeEncoder<'e>,
        arena: &'a mut Arena,
        rows: RowsCtx<'a>,
        kernels: &'a Kernels,
        from: u32,
        head: bool,
    ) -> Self {
        Self {
            enc,
            arena,
            kernels,
            pos: from,
            feed: Feed::Host,
            head,
            rows: Some(rows),
        }
    }

    /// The arm a unit of `m` rows takes: a step runs one row, a pass more
    /// than a step's window; any other count is refused by name.
    fn arm(&self, m: usize) -> Result<Arm, SessionError> {
        match (m, self.rows.is_some()) {
            (1, false) => Ok(Arm::Step),
            (m, true) if m > PASS_ROWS => Ok(Arm::Rows),
            _ => Err(ChainError::shape(
                WHAT,
                format!("a unit of {m} rows; a step runs one and a pass more than {PASS_ROWS}"),
            )
            .into()),
        }
    }

    /// The group closes: the next op's dispatches see what this one wrote.
    fn close(&mut self) {
        self.enc.memory_barrier();
    }

    /// Layer `l`'s P1: the norm on load (or the input the join before it
    /// handed on), one dispatch per weight type, one group.
    fn p1(&mut self, l: usize) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let scaled = self.arena.reads_scaled(Reader::P1(l));
        let Arena {
            x,
            xn,
            proj,
            layers,
            faults,
            eps,
            hidden,
            ..
        } = &mut *self.arena;
        let p1 = layers[l].mixer.p1();
        for part in &p1.parts {
            let (x, norm) = reader_input(scaled, x, xn.as_ref(), &p1.norm, *eps, *hidden)?;
            self.kernels.matvec_seg(
                self.enc,
                SegParams {
                    quant: part.quant,
                    segments: &part.segments,
                    weights: all(&part.weights)?,
                    x,
                    y: proj.window_mut(Elems(0), Elems(part.out_end))?,
                    norm,
                    fault: &faults[slot],
                    layer: layer_word(l),
                },
            )?;
        }
        self.close();
        Ok(())
    }
}

/// The layer's delta buffers, or a named refusal for a layer of another mixer.
macro_rules! delta_of {
    ($layers:expr, $l:expr) => {
        match &mut $layers[$l].mixer {
            MixerBufs::Delta(d) => d,
            MixerBufs::Attention(_) => {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {}: a delta op on an attention layer", $l),
                )
                .into());
            }
            MixerBufs::ShortConv(_) => {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {}: a delta op on a short conv layer", $l),
                )
                .into());
            }
        }
    };
}

/// The layer's attention buffers, or a named refusal for a layer of another
/// mixer.
macro_rules! attention_of {
    ($layers:expr, $l:expr) => {
        match &mut $layers[$l].mixer {
            MixerBufs::Attention(a) => a,
            MixerBufs::Delta(_) => {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {}: an attention op on a delta layer", $l),
                )
                .into());
            }
            MixerBufs::ShortConv(_) => {
                return Err(ChainError::shape(
                    WHAT,
                    format!("layer {}: an attention op on a short conv layer", $l),
                )
                .into());
            }
        }
    };
}

/// Layer `l`'s short conv buffers, or the named refusal for a layer of another
/// mixer.
fn conv_bufs(layers: &mut [LayerBufs], l: usize) -> Result<&mut ConvBufs, SessionError> {
    match &mut layers[l].mixer {
        MixerBufs::ShortConv(c) => Ok(c),
        MixerBufs::Delta(_) | MixerBufs::Attention(_) => Err(ChainError::shape(
            WHAT,
            format!("layer {l}: a short conv op on a layer of another mixer"),
        )
        .into()),
    }
}

/// Where the out projection of one half of layer `l` writes, and how: a plain
/// residual adds into the stream in the residual epilogue; a sandwich stores
/// into the join buffer, which [`MetalOps::join`] then norms into the stream.
fn half_out<'b>(
    l: LayerCtx<'_>,
    x: &'b mut MetalBuffer<f32>,
    join: &'b mut Option<MetalBuffer<f32>>,
    hidden: usize,
) -> Result<(WindowMut<'b, f32, MetalDevice>, Mode<'b>), SessionError> {
    match &l.plan.residual {
        ResidualPlan::Plain => Ok((x.window_mut(Elems(0), Elems(hidden))?, Mode::Residual)),
        ResidualPlan::Sandwich(_) => {
            let join = join.as_mut().ok_or_else(|| no_join(l.layer))?;
            Ok((all_mut(join)?, Mode::Store))
        }
    }
}

/// What layer `l`'s half `half` hands on to: the reader, its weight type and
/// gain, and the hand-on buffer the writer fills.
struct HandOnTo<'a> {
    reader: Reader,
    quant: Quant,
    gain: &'a MetalBuffer<f32>,
    xn: &'a mut MetalBuffer<f32>,
}

/// The reader after layer `l`'s half `half` and the hand-on buffer; a reader
/// with no scaled entry or an arena with no hand-on buffer is refused by name.
fn hand_on_to<'a>(
    layers: &'a [LayerBufs],
    head: (Quant, &'a MetalBuffer<f32>),
    xn: &'a mut Option<MetalBuffer<f32>>,
    l: usize,
    half: Half,
) -> Result<HandOnTo<'a>, SessionError> {
    let reader = Reader::after(layers.len(), l, half);
    let (quant, gain) = reader_norm(layers, head, reader).ok_or_else(|| {
        refuse(format!(
            "layer {l}: a half hands on to {reader:?}, which has no scaled entry"
        ))
    })?;
    let xn = xn.as_mut().ok_or_else(|| {
        refuse(format!(
            "layer {l}: a half hands on over an arena with no hand-on buffer"
        ))
    })?;
    Ok(HandOnTo {
        reader,
        quant,
        gain,
        xn,
    })
}

/// The refusal of a sandwich layer over an arena with no join buffer.
fn no_join(l: usize) -> SessionError {
    refuse(format!(
        "layer {l}: a sandwich residual over an arena with no join buffer"
    ))
}

/// The refusal of a sandwich over a routed FFN: its combine only adds into the
/// stream.
fn routed_sandwich(l: usize) -> SessionError {
    refuse(format!(
        "layer {l}: a sandwich residual over a routed FFN, whose combine only adds into the stream"
    ))
}

/// The refusal of a short conv layer over an arena with no conv output buffer.
fn no_conv_y(l: usize) -> SessionError {
    refuse(format!(
        "layer {l}: a short conv layer over an arena with no conv output buffer"
    ))
}

/// Layer `l`'s sandwich norm gain of `half`, or the refusal of a layer the
/// arena holds none for.
fn post_gain(
    layers: &[LayerBufs],
    l: usize,
    half: Half,
) -> Result<&MetalBuffer<f32>, SessionError> {
    let post = layers
        .get(l)
        .and_then(|layer| layer.post.as_ref())
        .ok_or_else(|| {
            refuse(format!(
                "layer {l}: a sandwich residual over a layer with no norm gains"
            ))
        })?;
    Ok(&post.gains[half.index()])
}

impl MetalOps<'_, '_> {
    /// The attention launch of layer `l` at `pos`: the one shape both its
    /// dispatches read, built per dispatch (the windows it holds are `Copy`,
    /// the parameters are not), and what the layer adds to it: its window of
    /// keys and its per-head gate.
    fn attn_params(
        arena: &mut Arena,
        l: usize,
        pos: u32,
        n_keys: u32,
        segments: u32,
    ) -> Result<(AttentionParams<'_>, AttnLayer<'_>), SessionError> {
        let slot = slot_of(pos);
        let Arena {
            proj,
            layers,
            ropes,
            attn,
            attn_partials,
            attn_ticket,
            faults,
            eps,
            ctx,
            geom,
            ..
        } = &mut *arena;
        let a = attention_of!(layers, l);
        let rope = rope_of(ropes, l, a.rope)?;
        // The arena holds the partials of the cache's whole segment count;
        // the launch reads and writes the prefix this token's keys take.
        let parts =
            geom.attn
                .partials_len(segments as usize)
                .ok_or_else(|| SessionError::Shape {
                    what: WHAT,
                    detail: format!("the partials of {segments} segments do not fit"),
                })?;
        let head_gate = match &a.gate {
            Some(g) => Some(part(proj, g)?),
            None => None,
        };
        let layer = AttnLayer {
            window: a.window,
            head_gate,
        };
        let params = AttentionParams {
            head_dim: geom.attn.head,
            query_heads: geom.attn.query_heads,
            kv_heads: geom.attn.kv_heads,
            segments: segments as usize,
            ctx: *ctx,
            eps: *eps,
            layer: layer_word(l),
            pos,
            n_keys,
            q: part(proj, &a.q)?,
            k: part(proj, &a.k)?,
            v: part(proj, &a.v)?,
            q_gain: all(&a.q_norm)?,
            k_gain: all(&a.k_norm)?,
            rope: all(rope)?,
            cache: AttnCache::F16 {
                k: all_mut(&mut a.k_cache)?,
                v: all_mut(&mut a.v_cache)?,
            },
            partials: attn_partials.window_mut(Elems(0), Elems(parts))?,
            out: all_mut(attn)?,
            fault: &faults[slot],
            ticket: attn_ticket.window_mut(Elems(0), Elems(1))?,
        };
        Ok((params, layer))
    }
}

/// The rope table `at` of layer `l`, or the refusal of an index the arena's
/// tables do not reach.
fn rope_of(
    ropes: &[MetalBuffer<f32>],
    l: usize,
    at: usize,
) -> Result<&MetalBuffer<f32>, SessionError> {
    ropes.get(at).ok_or_else(|| {
        refuse(format!(
            "layer {l}: rope table {at} of an arena with {}",
            ropes.len()
        ))
    })
}

impl MetalOps<'_, '_> {
    /// Layer `l`'s flash group: the attention over the keys the layer attends
    /// (the last `window` of the `pos + 1` live keys, or all of them), in the
    /// segments those keys take, this position's keys stored first. Live keys
    /// past one segment fold inside the flash: its last threadgroup merges the
    /// partials and writes the gated output. The one place the segment rule
    /// meets the layer's window, for the step and the profiled step alike.
    pub(super) fn attn_flash(&mut self, l: usize) -> Result<(), SessionError> {
        let n_keys = self.pos + 1;
        let window = attention_of!(self.arena.layers, l).window;
        let segments = segments_for_window(n_keys, window);
        let (p, layer) = Self::attn_params(&mut *self.arena, l, self.pos, n_keys, segments)?;
        self.kernels.attention_flash_layer(self.enc, p, layer)?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s gate-up group: the routed picks' and the shared expert's
    /// gate and up, the residual normed as it is loaded.
    pub(super) fn experts_pair(&mut self, l: usize) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let scaled = self.arena.reads_scaled(Reader::Pair(l));
        let Arena {
            layers,
            x,
            xn,
            h,
            faults,
            hidden,
            eps,
            n_experts,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let ids = f
            .ids
            .window(Elems(0), Elems(geom.experts.routed_ids))?
            .cast::<i32>()?;
        let (x, norm) = reader_input(scaled, x, xn.as_ref(), &f.norm, *eps, *hidden)?;
        self.kernels.pair(
            self.enc,
            PairParams {
                quant: f.gate_up_quant,
                slots: geom.experts.slots,
                k: geom.experts.k,
                cols: *hidden,
                n_experts: *n_experts,
                ids,
                gate: all(&f.gate)?,
                up: all(&f.up)?,
                shared_quant: f
                    .shared
                    .as_ref()
                    .map_or(f.gate_up_quant, |s| s.gate_up_quant),
                shared_gate: f.shared.as_ref().map(|s| all(&s.gate)).transpose()?,
                shared_up: f.shared.as_ref().map(|s| all(&s.up)).transpose()?,
                x,
                norm,
                y: all_mut(h)?,
                out_stride: geom.experts.k,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s down group: the experts' down projections, the routed
    /// picks' and the shared expert's, folded into the residual.
    pub(super) fn experts_fold(&mut self, l: usize) -> Result<(), SessionError> {
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            x,
            h,
            faults,
            hidden,
            n_experts,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l].ffn;
        let ids = f
            .ids
            .window(Elems(0), Elems(geom.experts.routed_ids))?
            .cast::<i32>()?;
        let y = x.window_mut(Elems(0), Elems(*hidden))?;
        self.kernels.fold(
            self.enc,
            FoldParams {
                quant: f.down_quant,
                shared: match &f.shared {
                    Some(s) if s.down_quant == f.down_quant => SharedPlan::Same(all(&s.down)?),
                    Some(s) => SharedPlan::Apart {
                        quant: s.down_quant,
                        weights: all(&s.down)?,
                    },
                    None => SharedPlan::None,
                },
                rows: *hidden,
                slots: geom.experts.slots,
                k: geom.experts.k,
                n_experts: *n_experts,
                ids,
                down: all(&f.down)?,
                h: all(h)?,
                w: all(&f.weights)?,
                y: &y,
                fault: &faults[slot],
                layer: layer_word(l),
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s down group: a plain residual folds the routed experts'
    /// down projections into the stream. A dense FFN's one down projection
    /// is the plain mat-vec, into the output [`half_out`] picks: added into
    /// the stream by the residual mat-vec under a plain residual (a dense
    /// slot's route weight is 1 and the mat-vec walks the fold's lane order,
    /// so the stream takes the fold's bits), stored into the join buffer for
    /// [`MetalOps::join`] under a sandwich. A sandwich over a routed FFN is
    /// refused by name: the combine only adds.
    pub(super) fn experts_down(&mut self, l: LayerCtx<'_>) -> Result<(), SessionError> {
        match (&l.plan.ffn.route, &l.plan.residual) {
            (FfnRoute::Router { .. }, ResidualPlan::Plain) => return self.experts_fold(l.layer),
            (FfnRoute::Router { .. }, ResidualPlan::Sandwich(_)) => {
                return Err(routed_sandwich(l.layer));
            }
            (FfnRoute::Dense, _) => {}
        }
        let Arena {
            layers,
            h,
            x,
            join,
            hidden,
            geom,
            ..
        } = &mut *self.arena;
        let f = &layers[l.layer].ffn;
        let (y, mode) = half_out(l, x, join, *hidden)?;
        self.kernels.matvec(
            self.enc,
            MatvecParams {
                quant: f.down_quant,
                rows: *hidden,
                cols: geom.experts.k,
                weights: all(&f.down)?,
                x: h.window(Elems(0), Elems(geom.experts.k))?,
                y,
                mode,
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s delta out group: the head-norm projection of the gated
    /// numerator, added into the stream. The projection has the residual
    /// epilogue only: a sandwich over the delta rule is refused by name, at
    /// the open and here.
    pub(super) fn delta_project(&mut self, l: LayerCtx<'_>) -> Result<(), SessionError> {
        l.plain_only("qwen3moe::MetalOps::delta_out")?;
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            g,
            delta_partials,
            x,
            faults,
            eps,
            hidden,
            ..
        } = &mut *self.arena;
        let d = delta_of!(layers, l.layer);
        self.kernels.headnorm_resid(
            self.enc,
            HeadNormParams {
                quant: d.out_quant,
                rows: *hidden,
                cols: g.len().get(),
                weights: all(&d.out)?,
                g: all(g)?,
                partials: all(delta_partials)?,
                y: x.window_mut(Elems(0), Elems(*hidden))?,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s attention out group: the output projection of the flash's
    /// gated output, added into the stream (plain) or stored into the join
    /// buffer (sandwich).
    pub(super) fn attn_project(&mut self, l: LayerCtx<'_>) -> Result<(), SessionError> {
        let Arena {
            layers,
            attn,
            x,
            join,
            hidden,
            ..
        } = &mut *self.arena;
        let a = attention_of!(layers, l.layer);
        let (y, mode) = half_out(l, x, join, *hidden)?;
        self.kernels.matvec(
            self.enc,
            MatvecParams {
                quant: a.out_quant,
                rows: *hidden,
                cols: attn.len().get(),
                weights: all(&a.out)?,
                x: all(attn)?,
                y,
                mode,
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s short conv group: the conv over the layer's ring for the
    /// token at `pos`, `C ⊙ conv(B ⊙ x)` into the conv output buffer.
    fn conv_step(&mut self, l: usize) -> Result<(), SessionError> {
        let pos = [self.pos];
        let slot = slot_of(self.pos);
        let Arena {
            proj,
            layers,
            conv_y,
            faults,
            hidden,
            ..
        } = &mut *self.arena;
        let c = conv_bufs(layers, l)?;
        let y = conv_y.as_mut().ok_or_else(|| no_conv_y(l))?;
        self.kernels.shortconv_step(
            self.enc,
            ShortConvStepParams {
                bcx: part(proj, &c.bcx)?,
                bcx_stride: c.bcx.len(),
                conv_w: all(&c.conv)?,
                ring: all_mut(&mut c.ring)?,
                y: all_mut(y)?,
                y_stride: *hidden,
                faults: &faults[slot],
                hidden: *hidden,
                taps: c.taps,
                ring_rows: c.ring_rows,
                layer: layer_word(l),
                pos: &pos,
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s conv out group: the output projection of the conv's output,
    /// added into the stream (plain) or stored into the join buffer
    /// (sandwich).
    pub(super) fn conv_project(&mut self, l: LayerCtx<'_>) -> Result<(), SessionError> {
        let Arena {
            layers,
            conv_y,
            x,
            join,
            hidden,
            ..
        } = &mut *self.arena;
        let c = conv_bufs(layers, l.layer)?;
        let y = conv_y.as_ref().ok_or_else(|| no_conv_y(l.layer))?;
        let (out, mode) = half_out(l, x, join, *hidden)?;
        self.kernels.matvec(
            self.enc,
            MatvecParams {
                quant: c.out_quant,
                rows: *hidden,
                cols: *hidden,
                weights: all(&c.out)?,
                x: all(y)?,
                y: out,
                mode,
            },
        )?;
        self.close();
        Ok(())
    }

    /// Layer `l`'s join group of `half` on a sandwich: the half's output in
    /// the join buffer normed by the half's gain and added into the stream at
    /// the half's scale, `x += add · gain ⊙ rms_norm(out)`. A layer that
    /// hands on ([`Arena::hands_on`]) also writes its reader's normed input
    /// into the hand-on buffer. A plain residual has no join group (its out
    /// and down added into the stream): one that hands on runs its hand-on
    /// group instead ([`MetalOps::hand_on`]), and one that does not encodes
    /// nothing.
    pub(super) fn join(&mut self, l: LayerCtx<'_>, half: Half) -> Result<(), SessionError> {
        let ResidualPlan::Sandwich(s) = &l.plan.residual else {
            return self.hand_on(l.layer, half);
        };
        let slot = slot_of(self.pos);
        let hands_on = self.arena.hands_on(l.layer);
        let Arena {
            layers,
            x,
            xn,
            join,
            faults,
            hidden,
            eps,
            head_quant,
            head_norm,
            ..
        } = &mut *self.arena;
        let out = join.as_ref().ok_or_else(|| no_join(l.layer))?;
        let joined = NormAddRowsParams {
            norm: NormRowsParams {
                rows: 1,
                cols: *hidden,
                x: all(out)?,
                x_stride: *hidden,
                gain: all(post_gain(layers, l.layer, half)?)?,
                y: x.window_mut(Elems(0), Elems(*hidden))?,
                y_stride: *hidden,
                eps: s.eps,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
            add: s.add[half.index()],
        };
        if hands_on {
            let to = hand_on_to(layers, (*head_quant, &*head_norm), xn, l.layer, half)?;
            self.kernels.norm_add_next(
                self.enc,
                NormAddNextParams {
                    join: joined,
                    quant: to.quant,
                    gain: all(to.gain)?,
                    eps: *eps,
                    layer: to.reader.layer(),
                    out: all_mut(to.xn)?,
                },
            )?;
        } else {
            self.kernels.norm_add_rows(self.enc, joined)?;
        }
        self.close();
        Ok(())
    }

    /// Layer `l`'s hand-on group of `half` on a plain residual that hands on
    /// ([`Arena::hands_on`]): the stream the half's out projection added
    /// into, times the next reader's gain, and that reader's scale, into the
    /// hand-on buffer. A layer that does not hand on encodes nothing.
    fn hand_on(&mut self, l: usize, half: Half) -> Result<(), SessionError> {
        if !self.arena.hands_on(l) {
            return Ok(());
        }
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            x,
            xn,
            faults,
            hidden,
            eps,
            head_quant,
            head_norm,
            ..
        } = &mut *self.arena;
        let to = hand_on_to(layers, (*head_quant, &*head_norm), xn, l, half)?;
        self.kernels.norm_next(
            self.enc,
            NormNextParams {
                quant: to.quant,
                x: x.window(Elems(0), Elems(*hidden))?,
                gain: all(to.gain)?,
                eps: *eps,
                layer: to.reader.layer(),
                out: all_mut(to.xn)?,
                fault: &faults[slot],
            },
        )?;
        self.close();
        Ok(())
    }

    /// The head's logits group of a step: the norm on load and the output
    /// projection, closed by a barrier so the pick reads what it wrote. The
    /// pass's head is [`MetalOps::rows_head`], which encodes both whole.
    pub(super) fn head_logits(&mut self, m: usize) -> Result<(), SessionError> {
        self.head_step(m, Kernels::head_logits)?;
        self.close();
        Ok(())
    }

    /// The head's pick group of a step: the argmax over the logits. It is the
    /// token's last group and closes no barrier.
    pub(super) fn head_pick(&mut self, m: usize) -> Result<(), SessionError> {
        self.head_step(m, Kernels::head_pick)
    }

    /// One of the head's launches over a step's windows. A pass is refused by
    /// name: its head is [`MetalOps::rows_head`].
    fn head_step(
        &mut self,
        m: usize,
        launch: fn(&Kernels, &mut ComputeEncoder<'_>, &HeadParams<'_>) -> Result<(), DeviceError>,
    ) -> Result<(), SessionError> {
        if self.arm(m)? != Arm::Step {
            return Err(refuse(format!(
                "a head group of {m} rows; a pass encodes its head whole"
            )));
        }
        let p = head_params(&mut *self.arena, None, slot_of(self.pos))?;
        launch(self.kernels, self.enc, &p)?;
        Ok(())
    }
}

impl ChainOps for MetalOps<'_, '_> {
    type Error = SessionError;

    fn embed(&mut self, m: usize) -> Result<(), SessionError> {
        if self.arm(m)? == Arm::Rows {
            return self.rows_embed(m);
        }
        let slot = slot_of(self.pos);
        let Arena {
            embed,
            embed_quant,
            vocab,
            hidden,
            id_in,
            step_id,
            x,
            faults,
            ..
        } = &mut *self.arena;
        let ids = match self.feed {
            Feed::Host => all(id_in)?,
            Feed::Device => step_id.window(Elems(slot_before(self.pos)), Elems(1))?,
        };
        self.kernels.embed(
            self.enc,
            EmbedParams {
                quant: *embed_quant,
                table: all(embed)?,
                vocab: *vocab,
                cols: *hidden,
                ids,
                out: x.window_mut(Elems(0), Elems(*hidden))?,
                fault: &faults[slot],
                layer: layer_word(0),
            },
        )?;
        self.close();
        Ok(())
    }

    fn attn_in(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        match self.arm(l.m)? {
            Arm::Step => self.p1(l.layer),
            Arm::Rows => self.rows_p1(l.layer, l.m),
        }
    }

    fn attn_core(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        if self.arm(l.m)? == Arm::Rows {
            return self.rows_attn_core(l.layer, l.m);
        }
        self.attn_flash(l.layer)
    }

    fn attn_out(&mut self, l: LayerCtx<'_>, _: &GqaPlan) -> Result<(), SessionError> {
        if self.arm(l.m)? == Arm::Rows {
            return self.rows_attn_out(l, l.m);
        }
        self.attn_project(l)?;
        self.join(l, Half::Mixer)
    }

    fn delta_in(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        match self.arm(l.m)? {
            Arm::Step => self.p1(l.layer),
            Arm::Rows => self.rows_p1(l.layer, l.m),
        }
    }

    fn delta_core(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        if self.arm(l.m)? == Arm::Rows {
            return self.rows_delta_core(l.layer, l.m);
        }
        let pos = [self.pos];
        let slot = slot_of(self.pos);
        let Arena {
            proj,
            layers,
            g,
            delta_partials,
            lane,
            faults,
            eps,
            geom,
            ..
        } = &mut *self.arena;
        let d = delta_of!(layers, l.layer);
        self.kernels.delta(
            self.enc,
            DeltaParams {
                x: part(proj, &d.qkv)?,
                conv_w: all(&d.conv)?,
                b_raw: part(proj, &d.beta)?,
                a_raw: part(proj, &d.alpha)?,
                dt_bias: all(&d.dt_bias)?,
                ssm_a: all(&d.ssm_a)?,
                z: part(proj, &d.z)?,
                norm_w: all(&d.ssm_norm)?,
                lane: all(lane)?,
                state: all_mut(&mut d.state)?,
                ring: all_mut(&mut d.ring)?,
                partials: all_mut(delta_partials)?,
                g: all_mut(g)?,
                faults: &faults[slot],
                n_v: geom.delta.value_heads,
                head: geom.delta.head_cols,
                taps: geom.delta.conv_taps,
                n_k: d.n_k,
                map: d.map,
                lanes: LANES,
                lane_at: lane_of(self.pos),
                lane_to: lane_of(self.pos + 1),
                layer: layer_word(l.layer),
                pos: &pos,
                eps: *eps,
            },
        )?;
        self.close();
        Ok(())
    }

    fn delta_out(&mut self, l: LayerCtx<'_>, _: &DeltaPlan) -> Result<(), SessionError> {
        if self.arm(l.m)? == Arm::Rows {
            l.plain_only("qwen3moe::MetalOps::delta_out")?;
            return self.rows_delta_out(l.layer, l.m);
        }
        self.delta_project(l)?;
        self.join(l, Half::Mixer)
    }

    fn conv_in(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), SessionError> {
        match self.arm(l.m)? {
            Arm::Step => self.p1(l.layer),
            Arm::Rows => self.rows_p1(l.layer, l.m),
        }
    }

    fn conv_core(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), SessionError> {
        if self.arm(l.m)? == Arm::Rows {
            return self.rows_conv_core(l.layer, l.m);
        }
        self.conv_step(l.layer)
    }

    fn conv_out(&mut self, l: LayerCtx<'_>, _: &ShortConvPlan) -> Result<(), SessionError> {
        if self.arm(l.m)? == Arm::Rows {
            return self.rows_conv_out(l, l.m);
        }
        self.conv_project(l)?;
        self.join(l, Half::Mixer)
    }

    fn route(&mut self, l: LayerCtx<'_>, plan: &FfnPlan) -> Result<(), SessionError> {
        let arm = self.arm(l.m)?;
        if matches!(plan.route, FfnRoute::Dense) {
            // A dense FFN has no router: its one slot is the fixed route the
            // open laid, and its norm is the gate-up's.
            return Ok(());
        }
        if arm == Arm::Rows {
            return self.rows_route(l.layer, l.m);
        }
        let slot = slot_of(self.pos);
        let Arena {
            layers,
            x,
            faults,
            eps,
            hidden,
            top_k,
            geom,
            ..
        } = &mut *self.arena;
        let f = &mut layers[l.layer].ffn;
        let Some(r) = f.router.as_mut() else {
            return Err(ChainError::shape(
                WHAT,
                format!("layer {}: a router launch on a layer with none", l.layer),
            )
            .into());
        };
        self.kernels.router(
            self.enc,
            RouterParams {
                rows: geom.router.rows,
                k: *hidden,
                top_k: *top_k,
                routed: all(&r.gate_inp)?,
                shared: f.shared.as_ref().map(|s| all(&s.gate_inp)).transpose()?,
                x: x.window(Elems(0), Elems(*hidden))?,
                gain: all(&f.norm)?,
                partials: all_mut(&mut r.partials)?,
                ticket: all_mut(&mut r.ticket)?,
                logits: all_mut(&mut r.logits)?,
                ids: all_mut(&mut f.ids)?,
                weights: all_mut(&mut f.weights)?,
                eps: *eps,
                fault: &faults[slot],
                layer: layer_word(l.layer),
            },
        )?;
        self.close();
        Ok(())
    }

    fn experts(&mut self, l: LayerCtx<'_>, plan: &FfnPlan) -> Result<(), SessionError> {
        let arm = self.arm(l.m)?;
        if let (FfnRoute::Router { .. }, ResidualPlan::Sandwich(_)) =
            (&plan.route, &l.plan.residual)
        {
            return Err(routed_sandwich(l.layer));
        }
        if arm == Arm::Rows {
            return match &plan.route {
                FfnRoute::Dense => self.rows_dense_ffn(l, l.m),
                FfnRoute::Router { .. } => self.rows_experts(l.layer, l.m),
            };
        }
        self.experts_pair(l.layer)?;
        self.experts_down(l)?;
        self.join(l, Half::Ffn)
    }

    fn head(&mut self, m: usize) -> Result<(), SessionError> {
        let arm = self.arm(m)?;
        if !self.head {
            // The call said nobody reads this token's pick: the head's
            // dispatches are the whole of what is skipped, and no later op
            // depends on them — the head closes no barrier past its own and
            // writes nothing but its own outputs.
            return Ok(());
        }
        if arm == Arm::Rows {
            return self.rows_head(m);
        }
        self.head_logits(m)?;
        self.head_pick(m)
    }
}
