//! The Inkling 2 prompt pass: a prompt's ids as passes of rows, every visit's
//! dispatches over the pass's rows at once, in the step walk's order
//! ([`super::ops`]), each mixer in its row form:
//! 1. the attention's inputs: the mix of the blocks and the stream over the
//!    rows (a launch a query row), the loop's messages and the controllers
//!    over the rows;
//! 2. each path's rows normed by its gain, then the q and relative rows, the
//!    gate rows, the k and v rows as GEMMs over them, the sink rows and a
//!    windowless visit's taps as GEMMs in their stored type;
//! 3. the k and v convs over the rows, then the clips and the append of rows
//!    `pos .. pos + m`;
//! 4. the rel-bias attention over the rows (`rel_prefill_f16`), its gate,
//!    `wo` and the post-conv;
//! 5. the FFN's mix, its message, its norm and the router over the rows;
//! 6. the route table over the rows' picks, the experts as two grouped GEMMs
//!    over it — the fused gate·up stack with the SITU epilogue, the down
//!    scaled by the route weight — the slots' sum, the post-conv and its add.
//!
//! A conv reads each row's predecessors from the pass's own rows and, before
//! the pass, from the visit's ring, and stores the pass's last `taps − 1`
//! inputs there; the clips append every row's key and value. So the pass
//! leaves every visit's K and V rows and conv ring where the steps of its ids
//! would, and the head's logits of its last row. The bank, the message banks
//! and the controllers' rows are a position's own, as in the step, and carry
//! nothing to the next one.
//!
//! A windowed visit's ring holds the window and the pass's rows less one: row
//! `t` reads the window before it while the rows after it have appended
//! theirs, so a pass of `m` rows over a ring narrower than the cache needs
//! `window + m − 1` ring rows ([`pass_rows`]); a longer pass is refused by
//! name.
//!
//! Each visit encodes on a command buffer of its own, committed unwaited (the
//! host encodes the next visit while the card runs this one), the last one
//! settled once. A routed pass (the gates') settles at every visit's router,
//! hands its scores, kept ids and weights to its hook and runs the experts at
//! the route the hook gives.

use bloomery_device::{Buffer, DeviceError, Elems, Window, WindowMut};
use runtime::stores::PROMPT_BATCH;

use super::arena::{Arena, LayerBufs};
use super::check::{CONV_CHANNELS, Checked, EPS, VisitPlan};
use super::ops::{
    BASE_ATTN, BASE_FFN, BASE_K, BASE_V, KV_ROWS, Q_ROWS, RANK, REL_ROWS, TAP_HEADS, Walk, tau, win,
};
use super::session::{SessionError, refuse};
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::{ComputeEncoder, EncodeCount};
use crate::fault::MetalFaultWord;
use crate::kernels::{
    BiasRowsParams, CombineStoreParams, ConvResParams, F32GemmParams, GemmMode, GemmParams,
    GemmSegParams, GemmX, HEAD_SIMD_GROUPS, HeadParams, Ink2RoutedParams, Ink2TableParams, Kernels,
    LoadNorm, LoopControlParams, NormRowsParams, NvfpLayout, Operand, OutGateParams,
    QkClipRowsParams, Quant, RelPrefillParams, ResMixParams, RouterRowsParams, Segment,
    route_table_words,
};
use crate::residency::Alloc;
use crate::session::pass::LayerBuffers;
use crate::session::room::Tally;
use crate::stream::MetalStream;

/// The q path's row: the query rows, then the relative rows.
const QR_ROWS: usize = Q_ROWS + REL_ROWS;
/// The sink rows: one a query head.
const SINKS: usize = 24;
/// The operand every GEMM of the pass computes in.
const OPERAND: Operand = Operand::F16;

/// The route a routed pass takes at a visit ([`super::session::Session::pass_routed`]):
/// the visit, the router's scores (`rows · experts`), kept ids and weights
/// (`rows · top_k` each) in; each row's ids and weights out, `rows · top_k`
/// each.
pub type PassRoutes<'a> =
    &'a mut dyn FnMut(usize, &[f32], &[u32], &[f32]) -> Result<(Vec<u32>, Vec<f32>), String>;

/// The rows a pass of `c` holds at most: a prompt batch, no more than the
/// cache's rows, and, where a windowed visit's ring is narrower than the
/// cache, no more than the ring holds beside the window (`window + m − 1`
/// rows).
#[must_use]
pub(super) fn pass_rows(c: &Checked) -> usize {
    let batch = usize::try_from(PROMPT_BATCH)
        .expect("a prompt batch is a usize")
        .min(c.ctx);
    if c.ring_rows >= c.ctx {
        batch
    } else {
        batch.min((c.ring_rows + 1).saturating_sub(c.window).max(1))
    }
}

/// Whether the pass serves `c`: every routed stack in a form the grouped
/// GEMM reads (NVFP4 planes or MXFP8 rows) and the q, k, v and output projections in
/// a type the quantized GEMM takes. A file it does not serve prefills a step an
/// id.
#[must_use]
pub(super) fn serves(c: &Checked) -> bool {
    let routed = |l: NvfpLayout| matches!(l, NvfpLayout::Planes | NvfpLayout::Mxfp8Rows);
    c.layers.iter().all(|l| {
        routed(l.gate_up.layout)
            && routed(l.down.layout)
            && [&l.qrg[0], &l.k, &l.v, &l.wo]
                .iter()
                .all(|s| s.quant != Quant::F32)
    })
}

/// A dense projection of the normed rows: `rows` weight rows of `cols`
/// columns over `acts` rows, stored as `quant`.
struct Dense<'w> {
    quant: Quant,
    rows: usize,
    cols: usize,
    acts: usize,
    weights: &'w MetalBuffer<u8>,
    x: Window<'w, f32, MetalDevice>,
    y: WindowMut<'w, f32, MetalDevice>,
    faults: &'w MetalFaultWord,
    layer: u32,
}

/// Encode `d` in its stored type, the decode's sink and tap launches over
/// rows: the f32 GEMM for rows the load widened (a GGUF's F32 sink and taps),
/// the quantized GEMM over the f16 operand for the rest (an MLX folder's
/// affine blocks). The outputs are `rows` wide, `acts` rows apart.
fn dense_rows(k: &Kernels, enc: &mut ComputeEncoder<'_>, d: Dense<'_>) -> Result<(), DeviceError> {
    if d.quant == Quant::F32 {
        return k.gemm_f32(
            enc,
            F32GemmParams {
                rows: d.rows,
                cols: d.cols,
                acts: d.acts,
                weights: d
                    .weights
                    .whole()?
                    .cast::<f32>()?
                    .sub(Elems(0), Elems(d.rows * d.cols))?,
                x: d.x,
                x_stride: d.cols,
                y: d.y,
                y_stride: d.rows,
                fault: d.faults,
                layer: d.layer,
            },
        );
    }
    k.gemm(
        enc,
        GemmParams {
            quant: d.quant,
            rows: d.rows,
            cols: d.cols,
            acts: d.acts,
            weights: d.weights.whole()?,
            x: GemmX::F32 {
                x: d.x,
                stride: d.cols,
            },
            y: d.y,
            y_stride: d.rows,
            mode: GemmMode::Store,
            operand: OPERAND,
            fault: d.faults,
            layer: d.layer,
        },
    )
}

/// The f32 buffers of a pass of `m` rows, in [`PassBufs::open`]'s order.
fn f32_sizes(c: &Checked, m: usize) -> [usize; 30] {
    let n = c.hidden;
    let slot = m * n;
    let taps = TAP_HEADS * c.taps;
    let picks = m * c.top_k;
    [
        (c.max_banks + 1) * slot,
        c.message_banks() * c.message_rows * slot,
        3 * slot,
        4 * slot,
        3 * slot,
        4 * m * RANK,
        3 * slot,
        m * QR_ROWS,
        m * Q_ROWS,
        m * SINKS,
        m * KV_ROWS,
        m * KV_ROWS,
        m * taps,
        m * taps,
        m * KV_ROWS,
        m * KV_ROWS,
        m * Q_ROWS,
        m * Q_ROWS,
        slot,
        slot,
        slot,
        slot,
        slot,
        m * c.experts,
        picks,
        picks * c.ff,
        picks * n,
        slot,
        slot,
        m,
    ]
}

/// The u32 buffers of a pass of `m` rows, in [`PassBufs::open`]'s order.
fn u32_sizes(c: &Checked, m: usize) -> [usize; 4] {
    let picks = m * c.top_k;
    [picks, picks, picks, route_table_words(picks, c.experts)]
}

/// The bytes the pass's buffers take, one [`Tally::add`] a buffer, in
/// [`PassBufs::open`]'s order.
#[must_use]
pub(super) fn tally(c: &Checked) -> Tally {
    let m = pass_rows(c);
    let mut t = Tally::default();
    for count in f32_sizes(c, m).into_iter().chain(u32_sizes(c, m)) {
        t.add((count * 4) as u64);
    }
    t
}

/// The pass's buffers, each holding [`pass_rows`] rows: a row-indexed buffer
/// of several vectors a row (the bank's slots, the mixes, the messages, the
/// controllers' and the paths' rows) holds each vector's rows together, the
/// rows of vector `i` from `i · rows · width`.
pub(super) struct PassBufs {
    /// The rows a buffer holds.
    cap: usize,
    /// The bank's blocks, then the stream: slot `i`'s rows from `i · cap · n`.
    rows: MetalBuffer<f32>,
    /// The message banks, one a pass the loop runs again, `message_rows`
    /// slots each.
    message: MetalBuffer<f32>,
    amix: MetalBuffer<f32>,
    msgs: MetalBuffer<f32>,
    xin: MetalBuffer<f32>,
    lc_h: MetalBuffer<f32>,
    /// The paths' normed inputs.
    xn: MetalBuffer<f32>,
    qr: MetalBuffer<f32>,
    gate: MetalBuffer<f32>,
    sinks: MetalBuffer<f32>,
    k_raw: MetalBuffer<f32>,
    v_raw: MetalBuffer<f32>,
    k_taps: MetalBuffer<f32>,
    v_taps: MetalBuffer<f32>,
    k: MetalBuffer<f32>,
    v: MetalBuffer<f32>,
    att: MetalBuffer<f32>,
    gated: MetalBuffer<f32>,
    ao: MetalBuffer<f32>,
    ac: MetalBuffer<f32>,
    fmix: MetalBuffer<f32>,
    fin: MetalBuffer<f32>,
    fnorm: MetalBuffer<f32>,
    probs: MetalBuffer<f32>,
    route_w: MetalBuffer<f32>,
    /// Each slot's SITU rows and its weighted down rows, a row a place of
    /// the route table.
    act: MetalBuffer<f32>,
    down: MetalBuffer<f32>,
    moe: MetalBuffer<f32>,
    moe_conv: MetalBuffer<f32>,
    /// Each row's position factor of a windowless visit's q and relative rows.
    taus: MetalBuffer<f32>,
    ids: MetalBuffer<u32>,
    sorted: MetalBuffer<u32>,
    inv: MetalBuffer<u32>,
    tiles: MetalBuffer<u32>,
}

impl PassBufs {
    /// The pass's buffers of the checked file `c`.
    ///
    /// # Errors
    /// A buffer the driver refuses, by name.
    pub fn open(alloc: Alloc<'_>, c: &Checked) -> Result<PassBufs, SessionError> {
        let cap = pass_rows(c);
        let f = f32_sizes(c, cap);
        let mut f = f.iter().map(|&s| alloc.buffer::<f32>(Elems(s)));
        let mut nf = || {
            f.next()
                .expect("the size list names every f32 buffer the pass makes")
                .map_err(SessionError::from)
        };
        let u = u32_sizes(c, cap);
        let mut u = u.iter().map(|&s| alloc.buffer::<u32>(Elems(s)));
        let mut nu = || {
            u.next()
                .expect("the size list names every u32 buffer the pass makes")
                .map_err(SessionError::from)
        };
        Ok(PassBufs {
            cap,
            rows: nf()?,
            message: nf()?,
            amix: nf()?,
            msgs: nf()?,
            xin: nf()?,
            lc_h: nf()?,
            xn: nf()?,
            qr: nf()?,
            gate: nf()?,
            sinks: nf()?,
            k_raw: nf()?,
            v_raw: nf()?,
            k_taps: nf()?,
            v_taps: nf()?,
            k: nf()?,
            v: nf()?,
            att: nf()?,
            gated: nf()?,
            ao: nf()?,
            ac: nf()?,
            fmix: nf()?,
            fin: nf()?,
            fnorm: nf()?,
            probs: nf()?,
            route_w: nf()?,
            act: nf()?,
            down: nf()?,
            moe: nf()?,
            moe_conv: nf()?,
            taus: nf()?,
            ids: nu()?,
            sorted: nu()?,
            inv: nu()?,
            tiles: nu()?,
        })
    }

    /// The rows a pass holds at most.
    #[must_use]
    pub fn cap(&self) -> usize {
        self.cap
    }
}

/// Where a path's input rows lie: an element offset into the bank's slots,
/// the attention's mixes or the controllers' rows.
#[derive(Clone, Copy, PartialEq, Eq)]
enum In {
    Rows(usize),
    Amix(usize),
    Xin(usize),
}

/// `len` elements of the buffer `s` names (the bank's slots, the mixes or the
/// controllers' rows), from its offset.
fn src<'b>(
    rows: &'b MetalBuffer<f32>,
    amix: &'b MetalBuffer<f32>,
    xin: &'b MetalBuffer<f32>,
    s: In,
    len: usize,
) -> Result<Window<'b, f32, MetalDevice>, DeviceError> {
    match s {
        In::Rows(at) => win(rows, at, len),
        In::Amix(at) => win(amix, at, len),
        In::Xin(at) => win(xin, at, len),
    }
}

/// The attention module `i`'s controller over `m` rows of `input`.
#[allow(clippy::too_many_arguments)]
fn controller<'b>(
    c: &Checked,
    lb: &'b LayerBufs,
    p: &'b mut PassBufs,
    input: In,
    i: usize,
    m: usize,
    faults: &'b MetalFaultWord,
    layer: u32,
) -> Result<LoopControlParams<'b>, SessionError> {
    let n = c.hidden;
    let slot = p.cap * n;
    let [down, gain, forget] = &lb.controllers[i];
    let PassBufs {
        rows,
        amix,
        msgs,
        xin,
        lc_h,
        cap,
        ..
    } = p;
    Ok(LoopControlParams {
        n,
        rank: RANK,
        m,
        cur: match input {
            In::Rows(at) => win(rows, at, m * n)?,
            In::Amix(at) => win(amix, at, m * n)?,
            In::Xin(_) => {
                return Err(refuse(
                    "a controller's input is another controller's output",
                ));
            }
        },
        msg: win(msgs, i * slot, m * n)?,
        down: down.whole()?,
        gain: gain.whole()?,
        forget: forget.whole()?,
        y: xin.window_mut(Elems(i * slot), Elems(m * n))?,
        h: lc_h.window_mut(Elems(i * *cap * RANK), Elems(m * RANK))?,
        eps: EPS,
        faults,
        layer,
    })
}

/// One pass's walk over the session's buffers.
pub(super) struct Pass<'a> {
    pub c: &'a Checked,
    pub a: &'a mut Arena,
    pub p: &'a mut PassBufs,
    pub k: &'a Kernels,
    pub stream: &'a MetalStream,
}

impl Pass<'_> {
    /// `ids` at positions `pos ..`: the head's pick over the last row when
    /// `head` is set (else the last id), and what the buffers encoded. With
    /// `routes`, every visit's experts run at the route it gives.
    ///
    /// # Errors
    /// A pass past the rows its buffers or the rings hold, a route the hook
    /// refuses or one past the experts, or the device's error.
    pub fn run(
        &mut self,
        ids: &[u32],
        pos: u32,
        head: bool,
        mut routes: Option<PassRoutes<'_>>,
    ) -> Result<(u32, EncodeCount), SessionError> {
        let (c, k, stream) = (self.c, self.k, self.stream);
        let m = ids.len();
        if m == 0 || m > self.p.cap {
            return Err(refuse(format!(
                "a pass of {m} rows; the pass holds 1 to {}",
                self.p.cap
            )));
        }
        let ring = c.plane_rows(true);
        if ring < c.ctx && ring + 1 < c.window + m {
            return Err(refuse(format!(
                "a pass of {m} rows over a ring of {ring} rows: the window of {} and the rows \
                 after the first need {}",
                c.window,
                c.window + m - 1
            )));
        }
        let n = c.hidden;
        let mut h0 = Vec::with_capacity(m * n);
        {
            let walk = Walk {
                c,
                a: &mut *self.a,
                k,
                stream,
            };
            for &id in ids {
                h0.extend(walk.embedded(id)?);
            }
        }
        self.p.rows.write_at(Elems(0), &h0)?;
        if c.visits.iter().any(|v| !v.windowed) {
            let taus: Vec<f32> = (0..m as u32).map(|t| tau(c, pos + t)).collect();
            self.p.taus.write_at(Elems(0), &taus)?;
        }
        // Each pass's end stores into its own bank, which starts at zero.
        self.p.message.zero();
        let mut buffers = LayerBuffers::new(stream);
        let mut banks = 0usize;
        let mut passes = 0usize;
        for (vi, v) in c.visits.iter().enumerate() {
            buffers
                .at(Some(vi))?
                .encoder(|enc| self.front(enc, vi, v, m, banks, pos, passes))?;
            if v.boundary {
                banks += 1;
            }
            if let Some(r) = routes.as_mut() {
                buffers.finish()?;
                if let Some(s) = buffers.last.take() {
                    stream.settle(s)?;
                }
                self.reroute(vi, m, r)?;
            }
            buffers
                .at(Some(vi))?
                .encoder(|enc| self.back(enc, vi, v, m, banks, pos, passes))?;
            if v.pass_end {
                banks = c.prelude_blocks;
                passes += 1;
            }
        }
        if head {
            buffers
                .at(Some(c.visits.len()))?
                .encoder(|enc| self.head(enc, m, banks))?;
        }
        buffers.finish()?;
        if let Some(s) = buffers.last.take() {
            stream.settle(s)?;
        }
        let pick = if head {
            let mut pick = [0u32];
            self.a
                .scratch
                .step_id
                .pages()
                .read_at(Elems(0), &mut pick)?;
            pick[0]
        } else {
            ids[m - 1]
        };
        Ok((pick, buffers.count))
    }

    /// The route `routes` gives visit `vi`'s `m` rows, after the router's
    /// buffer settled, written where the route table reads it.
    fn reroute(
        &mut self,
        vi: usize,
        m: usize,
        routes: &mut PassRoutes<'_>,
    ) -> Result<(), SessionError> {
        let (experts, top_k) = (self.c.experts, self.c.top_k);
        let p = &mut *self.p;
        let mut probs = vec![0.0f32; m * experts];
        p.probs.pages().read_at(Elems(0), &mut probs)?;
        let mut kept = vec![0u32; m * top_k];
        p.ids.pages().read_at(Elems(0), &mut kept)?;
        let mut weights = vec![0.0f32; m * top_k];
        p.route_w.pages().read_at(Elems(0), &mut weights)?;
        let (ids, w) = routes(vi, &probs, &kept, &weights).map_err(refuse)?;
        if ids.len() != m * top_k || w.len() != m * top_k {
            return Err(refuse(format!(
                "visit {vi}: a route of {} ids and {} weights for {m} rows of {top_k}",
                ids.len(),
                w.len()
            )));
        }
        if let Some(bad) = ids.iter().find(|&&e| e as usize >= experts) {
            return Err(refuse(format!("visit {vi}: a route names expert {bad}")));
        }
        p.ids.write_at(Elems(0), &ids)?;
        p.route_w.write_at(Elems(0), &w)?;
        Ok(())
    }

    /// Steps 1 to 5 of visit `vi` over `m` rows, the stream at slot `banks`,
    /// after `passes` of the loop's passes ended.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn front(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        vi: usize,
        v: &VisitPlan,
        m: usize,
        banks: usize,
        pos: u32,
        passes: usize,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let n = c.hidden;
        let layer = vi as u32;
        let lp = &c.layers[v.layer];
        let Arena {
            layers,
            visits,
            faults,
            loop_queries,
            ..
        } = &mut *self.a;
        let faults: &MetalFaultWord = faults;
        let lb: &LayerBufs = &layers[v.layer];
        let vb = &mut visits[vi];
        let pb = &mut *self.p;
        let cap = pb.cap;
        let slot = cap * n;
        let rows_n = m * n;
        let scale = 1.0 / (n as f32).sqrt();
        let stream_at = banks * slot;
        // 1. The attention's inputs: its mixes and the loop's messages, which
        // read nothing the other writes, then the controllers.
        let mut inputs = [In::Rows(stream_at); 3];
        let mut mixed = false;
        if let (Some((_, nq)), Some(query)) = (&lp.attn_query, &lb.attn_query) {
            for q in 0..*nq {
                k.res_mix(
                    enc,
                    ResMixParams {
                        n,
                        nq: 1,
                        m,
                        n_banks: banks,
                        stride: slot,
                        banks: (banks > 0)
                            .then(|| win(&pb.rows, 0, (banks - 1) * slot + rows_n))
                            .transpose()?,
                        cur: win(&pb.rows, stream_at, rows_n)?,
                        queries: win(query, q * n, n)?,
                        y: pb.amix.window_mut(Elems(q * slot), Elems(rows_n))?,
                        eps: EPS,
                        scale,
                        faults,
                        layer,
                    },
                )?;
            }
            mixed = true;
            inputs = if *nq == 1 {
                [In::Amix(0); 3]
            } else {
                [In::Amix(0), In::Amix(slot), In::Amix(2 * slot)]
            };
        }
        if v.message {
            let ci = v.core_index.expect("a message visit is a core visit");
            let mr = c.message_rows;
            // The bank of the pass that ended last.
            let bank = passes
                .checked_sub(1)
                .ok_or_else(|| refuse("a message visit before any pass ended"))?
                * mr
                * slot;
            for i in 0..4 {
                k.res_mix(
                    enc,
                    ResMixParams {
                        n,
                        nq: 1,
                        m,
                        n_banks: mr - 1,
                        stride: slot,
                        banks: (mr > 1)
                            .then(|| win(&pb.message, bank, (mr - 2) * slot + rows_n))
                            .transpose()?,
                        cur: win(&pb.message, bank + (mr - 1) * slot, rows_n)?,
                        queries: win(loop_queries, (ci * 4 + i) * n, n)?,
                        y: pb.msgs.window_mut(Elems(i * slot), Elems(rows_n))?,
                        eps: EPS,
                        scale,
                        faults,
                        layer,
                    },
                )?;
            }
            mixed = true;
        }
        if mixed {
            // Hazard: the mixes and the messages, the controllers' and the
            // norms' inputs.
            enc.memory_barrier();
        }
        if v.message {
            for (i, input) in inputs.iter().enumerate() {
                k.loop_control_down(enc, &controller(c, lb, pb, *input, i, m, faults, layer)?)?;
            }
            // Hazard: the down rows, the merges' input.
            enc.memory_barrier();
            for (i, input) in inputs.iter_mut().enumerate() {
                k.loop_control_merge(enc, &controller(c, lb, pb, *input, i, m, faults, layer)?)?;
                *input = In::Xin(i * slot);
            }
            // Hazard: the controllers' rows, the norms' inputs.
            enc.memory_barrier();
        }
        // 2. Each path's rows normed by its gain (a windowed visit's paths
        // share one), then the projections over them.
        let gains = lb.norms.len();
        let mut normed = [0usize; 3];
        let mut done: Vec<(In, usize, usize)> = Vec::with_capacity(3);
        for i in 0..3 {
            let g = i.min(gains - 1);
            if let Some(&(_, _, at)) = done.iter().find(|d| d.0 == inputs[i] && d.1 == g) {
                normed[i] = at;
                continue;
            }
            let at = done.len() * slot;
            k.norm_rows(
                enc,
                NormRowsParams {
                    rows: m,
                    cols: n,
                    x: src(&pb.rows, &pb.amix, &pb.xin, inputs[i], rows_n)?,
                    x_stride: n,
                    gain: lb.norms[g].whole()?,
                    y: pb.xn.window_mut(Elems(at), Elems(rows_n))?,
                    y_stride: n,
                    eps: c.rms_eps,
                    fault: faults,
                    layer,
                },
            )?;
            done.push((inputs[i], g, at));
            normed[i] = at;
        }
        // Hazard: the normed rows, the projections' inputs.
        enc.memory_barrier();
        let rb = lp.qrg[0].row_blocks();
        let row_bytes = rb * lp.qrg[0].quant.block_bytes();
        let xq = || -> Result<GemmX<'_>, DeviceError> {
            Ok(GemmX::F32 {
                x: win(&pb.xn, normed[0], rows_n)?,
                stride: n,
            })
        };
        k.gemm_seg(
            enc,
            GemmSegParams {
                quant: lp.qrg[0].quant,
                segments: &[
                    Segment {
                        cols: n,
                        rows: Q_ROWS,
                        weight_blocks: 0,
                        out_offset: 0,
                    },
                    Segment {
                        cols: n,
                        rows: REL_ROWS,
                        weight_blocks: Q_ROWS * rb,
                        out_offset: Q_ROWS,
                    },
                ],
                weights: lb.qrg.window(Elems(0), Elems(QR_ROWS * row_bytes))?,
                x: xq()?,
                acts: m,
                y: pb.qr.window_mut(Elems(0), Elems(m * QR_ROWS))?,
                y_stride: QR_ROWS,
                mode: GemmMode::Store,
                operand: OPERAND,
                fault: faults,
                layer,
            },
        )?;
        k.gemm_seg(
            enc,
            GemmSegParams {
                quant: lp.qrg[0].quant,
                segments: &[Segment {
                    cols: n,
                    rows: Q_ROWS,
                    weight_blocks: QR_ROWS * rb,
                    out_offset: 0,
                }],
                weights: lb.qrg.whole()?,
                x: xq()?,
                acts: m,
                y: pb.gate.window_mut(Elems(0), Elems(m * Q_ROWS))?,
                y_stride: Q_ROWS,
                mode: GemmMode::Store,
                operand: OPERAND,
                fault: faults,
                layer,
            },
        )?;
        dense_rows(
            k,
            enc,
            Dense {
                quant: lp.sink.quant,
                rows: SINKS,
                cols: n,
                acts: m,
                weights: &lb.sink,
                x: win(&pb.xn, normed[0], rows_n)?,
                y: pb.sinks.window_mut(Elems(0), Elems(m * SINKS))?,
                faults,
                layer,
            },
        )?;
        for (i, (site, w, out)) in [(&lp.k, &lb.k, &mut pb.k_raw), (&lp.v, &lb.v, &mut pb.v_raw)]
            .into_iter()
            .enumerate()
        {
            k.gemm(
                enc,
                GemmParams {
                    quant: site.quant,
                    rows: KV_ROWS,
                    cols: n,
                    acts: m,
                    weights: w.whole()?,
                    x: GemmX::F32 {
                        x: win(&pb.xn, normed[i + 1], rows_n)?,
                        stride: n,
                    },
                    y: out.window_mut(Elems(0), Elems(m * KV_ROWS))?,
                    y_stride: KV_ROWS,
                    mode: GemmMode::Store,
                    operand: OPERAND,
                    fault: faults,
                    layer,
                },
            )?;
        }
        let taps = TAP_HEADS * c.taps;
        for (i, site, w, out) in [
            (1, &lp.k_taps, &lb.k_taps, &mut pb.k_taps),
            (2, &lp.v_taps, &lb.v_taps, &mut pb.v_taps),
        ] {
            if let (Some(site), Some(w)) = (site, w) {
                dense_rows(
                    k,
                    enc,
                    Dense {
                        quant: site.quant,
                        rows: taps,
                        cols: n,
                        acts: m,
                        weights: w,
                        x: win(&pb.xn, normed[i], rows_n)?,
                        y: out.window_mut(Elems(0), Elems(m * taps))?,
                        faults,
                        layer,
                    },
                )?;
            }
        }
        // Hazard: the k and v rows and their taps, the convs' inputs.
        enc.memory_barrier();
        // 3. The k and v convs, then the clips and the append.
        let plane = c.plane_rows(v.windowed);
        for (base, w, x, taps_buf, y) in [
            (BASE_K, &lb.convs[0], &pb.k_raw, &pb.k_taps, &mut pb.k),
            (BASE_V, &lb.convs[1], &pb.v_raw, &pb.v_taps, &mut pb.v),
        ] {
            k.conv_res(
                enc,
                ConvResParams {
                    d: KV_ROWS,
                    taps: c.taps,
                    heads: if v.windowed { 0 } else { TAP_HEADS },
                    rows: c.taps - 1,
                    ch: CONV_CHANNELS,
                    base,
                    m,
                    pos,
                    x: win(x, 0, m * KV_ROWS)?,
                    w: w.whole()?,
                    dyn_taps: (!v.windowed)
                        .then(|| win(taps_buf, 0, m * taps))
                        .transpose()?,
                    y: y.window_mut(Elems(0), Elems(m * KV_ROWS))?,
                    ring: vb.ring.whole_mut()?,
                    faults,
                    layer,
                },
            )?;
        }
        // Hazard: the convs' k and v rows, which the clips norm and append.
        enc.memory_barrier();
        k.qk_clip_rows(
            enc,
            QkClipRowsParams {
                rows: m,
                qr: pb.qr.window_mut(Elems(0), Elems(m * QR_ROWS))?,
                qr_stride: QR_ROWS,
                r_at: Q_ROWS,
                k: pb.k.window_mut(Elems(0), Elems(m * KV_ROWS))?,
                v: pb.v.window_mut(Elems(0), Elems(m * KV_ROWS))?,
                gq: lb.q_clip.whole()?,
                gr: lb.rel_clip.whole()?,
                cache_k: vb.cache_k.whole_mut()?,
                cache_v: vb.cache_v.whole_mut()?,
                ctx: plane,
                pos,
                taus: (!v.windowed).then(|| win(&pb.taus, 0, m)).transpose()?,
                v_eps: EPS,
                qk_thresh: c.qk_clip,
                r_thresh: c.rel_clip,
                faults,
                layer,
            },
        )?;
        // Hazard: the clipped q and relative rows and the cache's new rows.
        enc.memory_barrier();
        // 4. The attention, its gate, wo and the post-conv.
        k.rel_prefill(
            enc,
            RelPrefillParams {
                rows: m,
                from: pos,
                plane,
                ctx: c.ctx,
                window: if v.windowed { c.window } else { 0 },
                scale: 1.0 / (128.0f32).sqrt(),
                q: win(&pb.qr, 0, (m - 1) * QR_ROWS + Q_ROWS)?,
                r: win(&pb.qr, Q_ROWS, (m - 1) * QR_ROWS + REL_ROWS)?,
                q_stride: QR_ROWS,
                bias: lb.rel_bias.whole()?,
                sinks: win(&pb.sinks, 0, m * SINKS)?,
                sink_stride: SINKS,
                cache_k: vb.cache_k.whole()?,
                cache_v: vb.cache_v.whole()?,
                out: pb.att.window_mut(Elems(0), Elems(m * Q_ROWS))?,
                out_stride: Q_ROWS,
                faults,
                layer,
            },
        )?;
        // Hazard: the attention's output, the gate's input.
        enc.memory_barrier();
        k.out_gate(
            enc,
            OutGateParams {
                n: m * Q_ROWS,
                kqv: win(&pb.att, 0, m * Q_ROWS)?,
                g: win(&pb.gate, 0, m * Q_ROWS)?,
                y: pb.gated.window_mut(Elems(0), Elems(m * Q_ROWS))?,
                faults,
                layer,
            },
        )?;
        // Hazard: the gated rows, wo's input.
        enc.memory_barrier();
        k.gemm(
            enc,
            GemmParams {
                quant: lp.wo.quant,
                rows: n,
                cols: Q_ROWS,
                acts: m,
                weights: lb.wo.whole()?,
                x: GemmX::F32 {
                    x: win(&pb.gated, 0, m * Q_ROWS)?,
                    stride: Q_ROWS,
                },
                y: pb.ao.window_mut(Elems(0), Elems(rows_n))?,
                y_stride: n,
                mode: GemmMode::Store,
                operand: OPERAND,
                fault: faults,
                layer,
            },
        )?;
        // Hazard: wo's rows, the post-conv's input.
        enc.memory_barrier();
        // At a boundary the stream's rows close as a block and the post-conv's
        // output is the residual, the next slot's stream.
        let y = if v.boundary {
            pb.rows.window_mut(Elems(stream_at + slot), Elems(rows_n))?
        } else {
            pb.ac.window_mut(Elems(0), Elems(rows_n))?
        };
        k.conv_res(
            enc,
            ConvResParams {
                d: n,
                taps: c.taps,
                heads: 0,
                rows: c.taps - 1,
                ch: CONV_CHANNELS,
                base: BASE_ATTN,
                m,
                pos,
                x: win(&pb.ao, 0, rows_n)?,
                w: lb.convs[2].whole()?,
                dyn_taps: None,
                y,
                ring: vb.ring.whole_mut()?,
                faults,
                layer,
            },
        )?;
        // Hazard: the post-conv's rows, the stream's add (or, at a boundary,
        // the new stream the FFN's mix reads).
        enc.memory_barrier();
        if !v.boundary {
            k.bias_rows(
                enc,
                BiasRowsParams {
                    rows: 1,
                    cols: rows_n,
                    y: pb.rows.window_mut(Elems(stream_at), Elems(rows_n))?,
                    y_stride: rows_n,
                    bias: win(&pb.ac, 0, rows_n)?,
                },
            )?;
            // Hazard: the stream's rows, the FFN's mix's input.
            enc.memory_barrier();
        }
        // 5. The FFN's mix, its message, its norm and the router.
        let banks = banks + usize::from(v.boundary);
        let stream_at = banks * slot;
        k.res_mix(
            enc,
            ResMixParams {
                n,
                nq: 1,
                m,
                n_banks: banks,
                stride: slot,
                banks: Some(win(&pb.rows, 0, (banks - 1) * slot + rows_n)?),
                cur: win(&pb.rows, stream_at, rows_n)?,
                queries: lb.ffn_query.whole()?,
                y: pb.fmix.window_mut(Elems(0), Elems(rows_n))?,
                eps: EPS,
                scale,
                faults,
                layer,
            },
        )?;
        // Hazard: the FFN's mix, its controller's (or its norm's) input.
        enc.memory_barrier();
        let ffn_in = if v.message {
            let [down, gain, forget] = &lb.controllers[3];
            k.loop_control(
                enc,
                LoopControlParams {
                    n,
                    rank: RANK,
                    m,
                    cur: win(&pb.fmix, 0, rows_n)?,
                    msg: win(&pb.msgs, 3 * slot, rows_n)?,
                    down: down.whole()?,
                    gain: gain.whole()?,
                    forget: forget.whole()?,
                    y: pb.fin.window_mut(Elems(0), Elems(rows_n))?,
                    h: pb.lc_h.window_mut(Elems(3 * cap * RANK), Elems(m * RANK))?,
                    eps: EPS,
                    faults,
                    layer,
                },
            )?;
            // Hazard: the controller's rows, the norm's input.
            enc.memory_barrier();
            &pb.fin
        } else {
            &pb.fmix
        };
        k.norm_rows(
            enc,
            NormRowsParams {
                rows: m,
                cols: n,
                x: win(ffn_in, 0, rows_n)?,
                x_stride: n,
                gain: lb.ffn_norm.whole()?,
                y: pb.fnorm.window_mut(Elems(0), Elems(rows_n))?,
                y_stride: n,
                eps: c.rms_eps,
                fault: faults,
                layer,
            },
        )?;
        // Hazard: the normed rows, the router's input.
        enc.memory_barrier();
        k.router_sigmoid_rows(
            enc,
            RouterRowsParams {
                rows: m,
                top_k: c.top_k,
                weights: lb.router.whole()?,
                x: win(&pb.fnorm, 0, rows_n)?,
                bias: lb.router_bias.whole()?,
                probs: pb.probs.window_mut(Elems(0), Elems(m * c.experts))?,
                ids: pb.ids.window_mut(Elems(0), Elems(m * c.top_k))?,
                route_w: pb.route_w.window_mut(Elems(0), Elems(m * c.top_k))?,
                scale: c.route_scale,
                faults,
                layer,
            },
        )?;
        Ok(())
    }

    /// Step 6 of visit `vi` over `m` rows, the stream at slot `banks`, and a
    /// pass's end: the route table over the rows' picks (or the hook's), the
    /// experts over it, their sum, the post-conv and its add.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn back(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        vi: usize,
        v: &VisitPlan,
        m: usize,
        banks: usize,
        pos: u32,
        passes: usize,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let (n, ff, top_k, experts) = (c.hidden, c.ff, c.top_k, c.experts);
        let layer = vi as u32;
        let lp = &c.layers[v.layer];
        let Arena {
            layers,
            visits,
            faults,
            ..
        } = &mut *self.a;
        let faults: &MetalFaultWord = faults;
        let lb = &layers[v.layer];
        let pb = &mut *self.p;
        let slot = pb.cap * n;
        let rows_n = m * n;
        let slots = m * top_k;
        let words = route_table_words(slots, experts);
        // Hazard: the router's ids and weights (or the host's route) and the
        // normed rows, the table's and the experts' inputs.
        enc.memory_barrier();
        k.ink2_route_table(
            enc,
            Ink2TableParams {
                slots,
                ids: win(&pb.ids, 0, slots)?,
                sorted: pb.sorted.window_mut(Elems(0), Elems(slots))?,
                inv: pb.inv.window_mut(Elems(0), Elems(slots))?,
                tiles: pb.tiles.window_mut(Elems(0), Elems(words))?,
                fault: faults,
                layer,
            },
        )?;
        // Hazard: the table, the grouped GEMMs' tiles.
        enc.memory_barrier();
        k.gemm_routed_ink2(
            enc,
            Ink2RoutedParams {
                layout: lp.gate_up.layout,
                pair: true,
                cols: n,
                rows: ff,
                stack: lb.gate_up.whole()?,
                scales: lb.gate_up_scale.whole()?,
                x: win(&pb.fnorm, 0, rows_n)?,
                x_rows: m,
                x_stride: n,
                gather: true,
                y: pb.act.window_mut(Elems(0), Elems(slots * ff))?,
                y_stride: ff,
                slots,
                per_row: top_k,
                sorted: win(&pb.sorted, 0, slots)?,
                route_w: win(&pb.route_w, 0, slots)?,
                tiles: win(&pb.tiles, 0, words)?,
                gate_beta: c.gate_beta,
                value_beta: c.value_beta,
                faults,
                layer,
            },
        )?;
        // Hazard: the places' SITU rows, the downs' input.
        enc.memory_barrier();
        k.gemm_routed_ink2(
            enc,
            Ink2RoutedParams {
                layout: lp.down.layout,
                pair: false,
                cols: ff,
                rows: n,
                stack: lb.down.whole()?,
                scales: lb.down_scale.whole()?,
                x: win(&pb.act, 0, slots * ff)?,
                x_rows: slots,
                x_stride: ff,
                gather: false,
                y: pb.down.window_mut(Elems(0), Elems(slots * n))?,
                y_stride: n,
                slots,
                per_row: top_k,
                sorted: win(&pb.sorted, 0, slots)?,
                route_w: win(&pb.route_w, 0, slots)?,
                tiles: win(&pb.tiles, 0, words)?,
                gate_beta: c.gate_beta,
                value_beta: c.value_beta,
                faults,
                layer,
            },
        )?;
        // Hazard: the places' weighted rows, the sum's input.
        enc.memory_barrier();
        k.combine_rows_store(
            enc,
            CombineStoreParams {
                rows: m,
                per_row: top_k,
                cols: n,
                y: win(&pb.down, 0, slots * n)?,
                y_stride: n,
                inv: win(&pb.inv, 0, slots)?,
                x: pb.moe.window_mut(Elems(0), Elems(rows_n))?,
                x_stride: n,
            },
        )?;
        // Hazard: the experts' sums, the post-conv's input.
        enc.memory_barrier();
        k.conv_res(
            enc,
            ConvResParams {
                d: n,
                taps: c.taps,
                heads: 0,
                rows: c.taps - 1,
                ch: CONV_CHANNELS,
                base: BASE_FFN,
                m,
                pos,
                x: win(&pb.moe, 0, rows_n)?,
                w: lb.convs[3].whole()?,
                dyn_taps: None,
                y: pb.moe_conv.window_mut(Elems(0), Elems(rows_n))?,
                ring: visits[vi].ring.whole_mut()?,
                faults,
                layer,
            },
        )?;
        // Hazard: the post-conv's rows, the stream's add.
        enc.memory_barrier();
        k.bias_rows(
            enc,
            BiasRowsParams {
                rows: 1,
                cols: rows_n,
                y: pb.rows.window_mut(Elems(banks * slot), Elems(rows_n))?,
                y_stride: rows_n,
                bias: win(&pb.moe_conv, 0, rows_n)?,
            },
        )?;
        // Hazard: the stream's rows, the next visit's (or the head's) input.
        enc.memory_barrier();
        if v.pass_end {
            let (mr, pb_blocks) = (c.message_rows, c.prelude_blocks);
            if banks.checked_sub(pb_blocks) != Some(mr) {
                return Err(refuse(format!(
                    "a pass ends at bank slot {banks}; its message is {mr} slots past the \
                     prelude's {pb_blocks}"
                )));
            }
            // The pass's blocks past its first and its stream, every slot
            // whole, into its own bank.
            k.bias_rows(
                enc,
                BiasRowsParams {
                    rows: 1,
                    cols: mr * slot,
                    y: pb
                        .message
                        .window_mut(Elems(passes * mr * slot), Elems(mr * slot))?,
                    y_stride: mr * slot,
                    bias: win(&pb.rows, (pb_blocks + 1) * slot, mr * slot)?,
                },
            )?;
            // Hazard: the next pass rewrites the slots the bank was read from.
            enc.memory_barrier();
        }
        Ok(())
    }

    /// The head over the last of `m` rows: the mix of the bank's last rows and
    /// the stream's, normed on load, its logits over the cut and the pick.
    fn head(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        m: usize,
        banks: usize,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let n = c.hidden;
        let Arena {
            scratch: sc,
            faults,
            head,
            output_norm,
            output_query,
            ..
        } = &mut *self.a;
        let faults: &MetalFaultWord = faults;
        let pb = &*self.p;
        let slot = pb.cap * n;
        let last = (m - 1) * n;
        let layer = c.visits.len() as u32;
        k.res_mix(
            enc,
            ResMixParams {
                n,
                nq: 1,
                m: 1,
                n_banks: banks,
                stride: slot,
                banks: (banks > 0)
                    .then(|| win(&pb.rows, last, (banks - 1) * slot + n))
                    .transpose()?,
                cur: win(&pb.rows, banks * slot + last, n)?,
                queries: output_query.whole()?,
                y: sc.head_mix.whole_mut()?,
                eps: EPS,
                scale: 1.0 / (n as f32).sqrt(),
                faults,
                layer,
            },
        )?;
        // Hazard: the head's mix, its input.
        enc.memory_barrier();
        k.head(
            enc,
            HeadParams {
                quant: c.head.quant,
                rows: c.cut,
                k: n,
                simdgroups: HEAD_SIMD_GROUPS,
                weights: head.whole()?,
                x: sc.head_mix.whole()?,
                norm: LoadNorm::Gain {
                    gain: output_norm.whole()?,
                    eps: c.rms_eps,
                },
                logits: sc.logits.whole_mut()?,
                step_id: sc.step_id.whole_mut()?,
                fault: faults,
                layer,
            },
        )?;
        Ok(())
    }
}
