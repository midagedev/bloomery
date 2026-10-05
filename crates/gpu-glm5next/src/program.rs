//! The GLM layer program of the one-token step, of the verify's two rows and
//! of a pass of slots' rows ([`StepProgram`]) and the port its host legs
//! go through ([`HostLeg`]): [`runtime::sched::walk`] over the point `(1, 1,
//! Step)`, `(2, 1, Step)` for the verify, or a pass's point ([`slots_point`]:
//! [`runtime::sched::slot_lanes`]' for a plain pass's slots, a row each,
//! `(m, 1, Step)` for a NextN load's slots' `m` verify rows), whose rows run
//! one layer apart, each row's launches the step's on its own buffers and
//! its own sequence's stores (`crate::body::Parts::at_row`). Each layer's
//! parts, by its programs (`runtime::layer::Layer`), never its number:
//!
//! - the front: the mixer sub-layer whole — `hc_pre`, the fold, the KDA or
//!   latent mixer, `hc_post` — then the feed-forward sub-layer's `hc_pre`
//!   and fold, and either the dense block and its `hc_post` or the routed
//!   block up to its go;
//! - the shadow (a routed layer): its card experts, when the slot map puts
//!   any on the card, and the shared expert;
//! - the back (a routed layer): the wait, the sum and `hc_post`, then the
//!   host tier told the layer is enqueued.
//!
//! A layer's streams go to its tap, where a gate armed one, once its last
//! `hc_post` is enqueued. The begin starts from stream buffer 0, which the
//! embedding wrote; the end is the streams' mean into the head, and the
//! head.
//!
//! Launches per layer: 3 a sub-layer for the streams (`hc_pre`, the fold,
//! `hc_post`), plus the mixer's ([`crate::kda::LAUNCHES`],
//! [`crate::mla::LAUNCHES`]) and the block's ([`crate::ffn::DENSE_LAUNCHES`],
//! [`crate::ffn::MOE_LAUNCHES`], and [`crate::ffn::CARD_LAUNCHES`] more on a
//! layer with card experts); the head adds the mean and its own three.

use bloomery_gpu::head::Head;
use bloomery_gpu::hybrid::Hybrid;
use bloomery_gpu::weights::Weights;
use bloomery_gpu::{Gpu, GpuError};
use bloomery_gpu_deepseek41::hc::{HcPostArgs, HcQ8Params, HcQ8PreArgs};
use cuda_core::CudaStream;
use model::arch::glm5next::names::Sub;
use runtime::layer::{FfnKind, MixerKind};
use runtime::sched::{self, At, LayerProgram, Overlap, Port, PortKind, Refused};

use crate::body::{LANES, LOAD_ROWS, Parts, f32v, q8};
use crate::host::GlmHost;
use crate::tensors::LayerNames;
use crate::{ffn, kda, mla};

/// What the walk's errors name.
const WHAT: &str = "glm5next Body::enqueue_chain";

/// The launches of the step's captured chain: every layer's
/// ([`layer_launches`], `cards[l]` whether layer `l` has card experts) and
/// the head's four.
#[must_use]
pub fn step_launches(mixers: &[MixerKind], ffns: &[FfnKind], cards: &[bool]) -> usize {
    mixers
        .iter()
        .zip(ffns)
        .zip(cards)
        .map(|((&m, &f), &c)| layer_launches(m, f, c))
        .sum::<usize>()
        + 4
}

/// One layer's launches: two sub-layers' three for the streams, its mixer's
/// and its block's, with its card experts' when `card`.
#[must_use]
pub fn layer_launches(mixer: MixerKind, ffn: FfnKind, card: bool) -> usize {
    let mix = match mixer {
        MixerKind::DeltaRule => kda::LAUNCHES,
        MixerKind::Latent => mla::LAUNCHES,
        MixerKind::Gqa => 0,
    };
    let block = match ffn {
        FfnKind::Dense => ffn::DENSE_LAUNCHES,
        FfnKind::Moe if card => ffn::MOE_LAUNCHES + ffn::CARD_LAUNCHES,
        FfnKind::Moe => ffn::MOE_LAUNCHES,
    };
    2 * 3 + mix + block
}

/// The stream buffer a row's streams end in after a walk of `layers`
/// layers from buffer 0: each layer's two sub-layers flip it once each
/// (`hc_post` writes the other buffer), the step's walk and a prompt
/// batch's alike.
#[must_use]
pub(crate) const fn final_streams(layers: usize) -> usize {
    (2 * layers) % 2
}

/// The step's point, `(1, 1, Step)`.
const STEP: Overlap = Overlap {
    units: 1,
    cols: 1,
    port: PortKind::Step,
};

/// Walk the step over `parts`, the host tier opened for it first, into
/// `head`.
pub(crate) fn walk_step(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    head: &mut Head,
) -> Result<(), GpuError> {
    walk(gpu, w, parts, hybrid, STEP, [Some(head), None, None, None])
}

/// Walk the verify's two rows over `parts`, the host tier opened for them
/// first, row `r` into `heads[r]`.
pub(crate) fn walk_pair(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    heads: [&mut Head; LANES],
) -> Result<(), GpuError> {
    let o = Overlap {
        units: LANES,
        ..STEP
    };
    let [a, b] = heads;
    walk(gpu, w, parts, hybrid, o, [Some(a), Some(b), None, None])
}

/// The point a pass of `slots` busy slots' `rows` rows takes: a plain
/// pass's slots, a row each, at [`sched::slot_lanes`]' point (the step for
/// one, the pair for two); a NextN load's slots, each its verify's rows, a
/// row a unit of one column, `(rows, 1, Step)` — the pair for one slot, the
/// quad for two. A pass of no slot is refused by name.
pub(crate) fn slots_point(slots: usize, rows: usize) -> Result<Overlap, GpuError> {
    if rows == slots {
        return Ok(sched::slot_lanes(slots).map_err(HostLeg::refused)?.overlap);
    }
    Ok(Overlap {
        units: rows,
        ..STEP
    })
}

/// Walk a pass of `slots` busy slots' `heads.len()` rows over `parts`,
/// whose rows are bound each to its own slot's sequence, the host tier
/// opened for them first, row `r` into `heads[r]`, at [`slots_point`]. A
/// point of other than one column a row, a unit a row, is refused by name
/// before anything is enqueued.
pub(crate) fn walk_slots(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    slots: usize,
    heads: &mut [Head],
) -> Result<(), GpuError> {
    let n = heads.len();
    let o = slots_point(slots, n)?;
    let refused = || GpuError::Shape {
        what: WHAT,
        detail: format!(
            "a pass of {slots} slots' {n} rows at the point {o:?}: the walk runs one, two or \
             {LOAD_ROWS} rows, one column a row"
        ),
    };
    if o.units != n || o.cols != 1 {
        return Err(refused());
    }
    let heads = match heads {
        [a] => [Some(a), None, None, None],
        [a, b] => [Some(a), Some(b), None, None],
        [a, b, c, d] => [Some(a), Some(b), Some(c), Some(d)],
        _ => return Err(refused()),
    };
    walk(gpu, w, parts, hybrid, o, heads)
}

/// Walk one row per head in `heads` at the point `o` (the step's one row,
/// the verify's two, a pass of slots' one, two or four).
fn walk(
    gpu: &Gpu,
    w: &Weights,
    parts: Parts<'_>,
    hybrid: &mut Hybrid<GlmHost>,
    o: Overlap,
    heads: [Option<&mut Head>; LOAD_ROWS],
) -> Result<(), GpuError> {
    let layers = parts.cfg.len();
    let mut port = HostLeg {
        stream: gpu.stream(),
        hybrid,
    };
    let mut prog = StepProgram {
        gpu,
        w,
        parts,
        cur: [0; LOAD_ROWS],
        heads,
    };
    sched::walk(o, layers, &mut port, &mut prog)
}

/// The step's host leg through the body's host tier.
pub(crate) struct HostLeg<'a> {
    stream: &'a CudaStream,
    hybrid: &'a mut Hybrid<GlmHost>,
}

impl Port for HostLeg<'_> {
    type Error = GpuError;
    const KIND: PortKind = PortKind::Step;

    /// Opens the host tier's step port on the point's rows
    /// ([`Hybrid::open_step`]): one row of one column, two for the verify or
    /// a pass of two slots' rows, four for a NextN load's two slots' verify
    /// rows (the boundary must carry them); any other point is refused by
    /// name.
    fn open(&mut self, o: Overlap) -> Result<(), GpuError> {
        self.hybrid.open_step(self.stream, o.units, o.cols)
    }

    fn refused(why: Refused) -> GpuError {
        GpuError::Shape {
            what: WHAT,
            detail: why.to_string(),
        }
    }
}

/// The GLM program over one walk: the body's parts, each row's stream
/// buffer its next sub-layer reads, each row's head.
struct StepProgram<'s, 'w> {
    gpu: &'w Gpu,
    w: &'w Weights,
    parts: Parts<'s>,
    cur: [usize; LOAD_ROWS],
    heads: [Option<&'s mut Head>; LOAD_ROWS],
}

impl StepProgram<'_, '_> {
    /// The launches after this are row `row`'s, on its buffers.
    fn at(&mut self, row: usize) -> Result<(), GpuError> {
        self.parts.at_row(row)
    }

    /// Sub-layer `sub` of layer `l`'s input into the current row's `x`
    /// ([`hc_in`]).
    fn hc_in(&mut self, l: usize, sub: Sub) -> Result<(), GpuError> {
        let cur = self.cur[self.parts.row];
        hc_in(self.gpu, self.w, &mut self.parts, cur, l, sub)
    }

    /// The current row's sub-layer output into its next streams
    /// ([`hc_out`]).
    fn hc_out(&mut self) -> Result<(), GpuError> {
        let row = self.parts.row;
        self.cur[row] = hc_out(self.gpu, &mut self.parts, self.cur[row])?;
        Ok(())
    }
}

impl<'s> LayerProgram for StepProgram<'s, '_> {
    type Port = HostLeg<'s>;

    /// The row's stream buffer 0, which its embedding wrote.
    fn begin(&mut self, unit: usize) -> Result<(), GpuError> {
        *self.cur.get_mut(unit).ok_or_else(|| no_row(unit))? = 0;
        Ok(())
    }

    /// The mixer sub-layer, then the block's input and the dense block with
    /// its `hc_post`, or the routed block up to its go.
    fn front(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        self.at(at.unit)?;
        let (gpu, w, l) = (self.gpu, self.w, at.layer);
        let kind = self.parts.cfg[l].kind;
        self.hc_in(l, Sub::Attn)?;
        match kind.mixer {
            MixerKind::DeltaRule => kda::kda(gpu, w, &mut self.parts, l)?,
            MixerKind::Latent => mla::mla(gpu, w, &mut self.parts, l)?,
            MixerKind::Gqa => {
                return Err(GpuError::Shape {
                    what: WHAT,
                    detail: format!("layer {l}: a GQA mixer, which glm5next has none of"),
                });
            }
        }
        self.hc_out()?;
        self.hc_in(l, Sub::Ffn)?;
        match kind.ffn {
            FfnKind::Dense => {
                ffn::dense(gpu, w, &mut self.parts, l)?;
                self.hc_out()?;
                self.parts.tap(gpu, l, self.cur[at.unit])
            }
            FfnKind::Moe => ffn::front(gpu, w, &mut self.parts, port.hybrid, l),
        }
    }

    /// A routed layer's card experts and shared expert under its host leg.
    fn shadow(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        self.at(at.unit)?;
        let l = at.layer;
        if !self.parts.cfg[l].kind.host_leg() {
            return Ok(());
        }
        ffn::shadow(self.gpu, self.w, &mut self.parts, port.hybrid.boundary(), l)
    }

    /// A routed layer's wait, sum and `hc_post`, its tap, and the host tier
    /// told the layer is enqueued (an eager chain is served there).
    fn back(&mut self, port: &mut HostLeg<'s>, at: At) -> Result<(), GpuError> {
        let (gpu, l) = (self.gpu, at.layer);
        if !self.parts.cfg[l].kind.host_leg() {
            return Ok(());
        }
        self.at(at.unit)?;
        ffn::back(gpu, &mut self.parts, port.hybrid, l)?;
        self.hc_out()?;
        self.parts.tap(gpu, l, self.cur[at.unit])?;
        port.hybrid.row_enqueued(l, at.unit)
    }

    /// The row's streams' mean into its head, and the head.
    fn end(&mut self, unit: usize) -> Result<(), GpuError> {
        self.at(unit)?;
        let (gpu, w) = (self.gpu, self.w);
        let n = self.parts.d.embd;
        let head = self
            .heads
            .get_mut(unit)
            .and_then(Option::as_deref_mut)
            .ok_or_else(|| no_row(unit))?;
        let s = &self.parts.s.streams[self.cur[unit]];
        self.parts
            .k
            .hc
            .enqueue_mean(gpu.stream(), s, n, 0, head.input_mut())?;
        head.enqueue(gpu, w)
    }
}

/// A unit the walk has no row or head for, by name.
fn no_row(unit: usize) -> GpuError {
    GpuError::Shape {
        what: WHAT,
        detail: format!("row {unit} of a walk of {LOAD_ROWS} rows at most, or one with no head"),
    }
}

/// Sub-layer `sub` of layer `l`'s input from stream buffer `cur`: its own
/// mix of the streams (`hc_pre_q8_0`) and their fold by it, into `x` — the
/// step program's launches.
pub(crate) fn hc_in(
    gpu: &Gpu,
    w: &Weights,
    p: &mut Parts<'_>,
    cur: usize,
    l: usize,
    sub: Sub,
) -> Result<(), GpuError> {
    let stream = gpu.stream();
    let d = *p.d;
    let names: &[LayerNames] = p.names;
    let n = names
        .get(l)
        .ok_or_else(|| GpuError::Shape {
            what: WHAT,
            detail: format!("layer {l} past the {} named", names.len()),
        })?
        .hc(sub);
    let (qs, dd) = q8(w, &n.fn_)?;
    let params = HcQ8Params {
        qs,
        d: dd,
        scale: f32v(w, &n.scale)?,
        base: f32v(w, &n.base)?,
        eps: d.hc_eps,
        iters: d.hc_iters,
    };
    let hc = &p.k.hc;
    let s = &mut *p.s;
    let streams = &s.streams[cur];
    hc.enqueue_pre_q8_0(
        stream,
        &HcQ8PreArgs {
            params: &params,
            x: streams,
            tokens: 1,
            rms_eps: d.rms_eps,
        },
        1,
        &mut s.hc_scratch,
        &mut s.mixes,
        &mut s.hc,
    )?;
    hc.enqueue_fold(stream, streams, &s.hc, d.embd, 1, &mut s.x)
}

/// The sub-layer's output into the other stream buffer by its mix
/// (`hc_post`), residual from `cur`; returns the buffer written — the step
/// program's launch.
pub(crate) fn hc_out(gpu: &Gpu, p: &mut Parts<'_>, cur: usize) -> Result<usize, GpuError> {
    let n = p.d.embd;
    let hc = &p.k.hc;
    let s = &mut *p.s;
    let [a, b] = &mut s.streams;
    let (res, next) = if cur == 0 { (&*a, b) } else { (&*b, a) };
    hc.enqueue_post(
        gpu.stream(),
        &HcPostArgs {
            x: &s.out,
            res,
            hc: &s.hc,
            n_embd: n,
            tokens: 1,
        },
        next,
        &mut s.fold,
    )?;
    Ok(cur ^ 1)
}
