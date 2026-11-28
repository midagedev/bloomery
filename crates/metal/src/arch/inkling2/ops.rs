//! The walk of one Inkling 2 token over its visits: the dispatches of a
//! visit in the reference's order (`mlx_lm/models/inkling2/layers.py`,
//! `DecoderLayer.__call__`), all of them on concurrent encoders of
//! three command buffers committed as the host encodes them (the first after
//! two visits, the second halfway), one wait a token. A barrier stands where a
//! dispatch reads what an earlier one wrote (or writes what it read), and
//! each one names that hazard.
//!
//! A visit, at stream row `s` of the bank's rows (the blocks closed so far,
//! then the stream):
//! 1. the attention's inputs — the normed mix of the blocks and the stream
//!    (one query on a windowed visit, three on a windowless one; layer 0 reads
//!    the stream as it is), each merged with its loop message on a later
//!    pass's core visit;
//! 2. the q path (q, relative and gate rows in one launch, the sink rows in
//!    another) and the k and v paths, each normed on load by its gain;
//! 3. the k and v short convs (a windowless visit's taps from its own rows)
//!    and, in the same launch, the clips, the value norm and the cache append;
//! 4. the rel-bias attention, its output gate, `wo` and the post-conv; at a
//!    block boundary the stream row closes as a block and the post-conv's
//!    output is the next row's stream, else the post-conv's launch adds its
//!    output into the stream;
//! 5. the FFN's mix, its loop message, and the router, which norms its input
//!    (the experts' input too) and leaves its picks' ids and weights on the
//!    card;
//! 6. every pick's gate·up rows in one launch and its down rows in another,
//!    each slot reading its expert's id there (the route weight folded into
//!    the down's scale), then one launch of their sum, the post-conv and its
//!    add into the stream.
//!
//! A pass the loop runs again ends by adding its blocks past its first and
//! its stream into its own message bank, on the card, and the next pass
//! starts again at the prelude's stream row, which the core's first visit
//! closed and nothing since has written. The head mixes the bank and the
//! stream, norms the mix on load and picks; the host reads the pick.
//!
//! A step that hands the host its nodes (taps) or takes its routes from the
//! host commits and waits after a visit's router, and a tapped one after its
//! experts too, so the host reads settled buffers or writes the route the
//! experts then read.

use bloomery_device::{Buffer, DeviceError, Elems, Stream, Window};

use super::arena::{Arena, LayerBufs, Scratch};
use super::check::{CONV_CHANNELS, Checked, EPS, VisitPlan};
use super::clock::Trace;
use super::count::Kind;
use super::session::{Inputs, Routes, SessionError, Tap, TapSink, refuse};
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::{ComputeEncoder, EncodeCount};
use crate::fault::MetalFaultWord;
use crate::kernels::{
    BiasRowsParams, CombineConvJoinParams, ConvResJoinParams, ConvResParams, HEAD_SIMD_GROUPS,
    HeadParams, Kernels, KvConvParams, LoadNorm, LoopControlParams, MatvecParams, Mode,
    Nvfp4Params, OutGateParams, QkClipConvParams, QkClipParams, RelFlashParams, ResMixParams,
    RouterSigmoidNormParams, RouterSigmoidParams, SegParams, Segment, SituParams, TailConv,
    segments_for_window,
};
use crate::stream::{MetalStream, Recording};

/// The query, key and relative rows' widths of a visit.
pub(super) const Q_ROWS: usize = 3072;
pub(super) const KV_ROWS: usize = 1024;
pub(super) const REL_ROWS: usize = 384;
/// The heads of a windowless conv's dynamic taps, and the controllers' rank.
pub(super) const TAP_HEADS: usize = 8;
pub(super) const RANK: usize = 64;
/// The conv ring's channel bases: k, v, the attention's post-conv, the FFN's.
pub(super) const BASE_K: usize = 0;
pub(super) const BASE_V: usize = 1024;
pub(super) const BASE_ATTN: usize = 2048;
pub(super) const BASE_FFN: usize = 2048 + 2304;

/// Where a path's input lies.
#[derive(Clone, Copy)]
enum Src {
    /// The bank's rows, at element `at`.
    Rows(usize),
    Amix(usize),
    Xin(usize),
}

/// The visits of a token's first command buffer: few enough that the card
/// starts while the host encodes the rest, enough that the host's encode of
/// the rest ends before the card runs out of them.
const FIRST_CHUNK_VISITS: usize = 2;

/// Whether a plain step commits the command buffer it holds after visit
/// `vi` of `visits`: after the first chunk, and halfway, so the host's
/// encode of the token's tail overlaps the card's run of its head.
fn chunk_ends(vi: usize, visits: usize) -> bool {
    vi + 1 == FIRST_CHUNK_VISITS || vi + 1 == visits / 2
}

/// One token's walk over the session's buffers.
pub struct Walk<'a> {
    pub c: &'a Checked,
    pub a: &'a mut Arena,
    pub k: &'a Kernels,
    pub stream: &'a MetalStream,
    /// The step clock's record of the token, when the clock runs.
    pub trace: Option<&'a mut Trace>,
}

/// `len` elements of `b` from `at`.
pub(super) fn win<T: bloomery_device::Elem>(
    b: &MetalBuffer<T>,
    at: usize,
    len: usize,
) -> Result<Window<'_, T, MetalDevice>, DeviceError> {
    b.window(Elems(at), Elems(len))
}

/// The input `s` names, `n` values, out of the bank's rows, the attention's
/// mixes and the controllers' outputs.
fn src<'b>(
    rows: &'b MetalBuffer<f32>,
    amix: &'b MetalBuffer<f32>,
    xin: &'b MetalBuffer<f32>,
    n: usize,
    s: Src,
) -> Result<Window<'b, f32, MetalDevice>, DeviceError> {
    match s {
        Src::Rows(at) => win(rows, at, n),
        Src::Amix(at) => win(amix, at, n),
        Src::Xin(at) => win(xin, at, n),
    }
}

/// The log-scaled position factor of a windowless visit's q and relative rows.
pub(super) fn tau(c: &Checked, pos: u32) -> f32 {
    let ratio = (f64::from(pos) + 1.0) / f64::from(c.log_floor);
    (1.0 + f64::from(c.log_alpha) * ratio.max(1.0).ln()) as f32
}

/// Attention module `i`'s controller over the input `input`: its weights, its
/// message row, its output row of the paths' inputs and its down rows.
fn controller<'b>(
    c: &Checked,
    lb: &'b LayerBufs,
    sc: &'b mut Scratch,
    input: Src,
    i: usize,
    faults: &'b MetalFaultWord,
    layer: u32,
) -> Result<LoopControlParams<'b>, SessionError> {
    let n = c.hidden;
    let [down, gain, forget] = &lb.controllers[i];
    let Scratch {
        rows,
        amix,
        msgs,
        xin,
        lc_h,
        ..
    } = sc;
    Ok(LoopControlParams {
        n,
        rank: RANK,
        m: 1,
        cur: match input {
            Src::Rows(at) => win(rows, at, n)?,
            Src::Amix(at) => win(amix, at, n)?,
            Src::Xin(_) => {
                return Err(refuse(
                    "a controller's input is another controller's output",
                ));
            }
        },
        msg: win(msgs, i * n, n)?,
        down: down.whole()?,
        gain: gain.whole()?,
        forget: forget.whole()?,
        y: xin.window_mut(Elems(i * n), Elems(n))?,
        h: lc_h.window_mut(Elems(i * RANK), Elems(RANK))?,
        eps: EPS,
        faults,
        layer,
    })
}

impl Walk<'_> {
    /// The token `id` at position `pos`: its pick, every visit's nodes handed
    /// to `taps` when it is given.
    pub fn token(
        &mut self,
        id: u32,
        pos: u32,
        mut taps: Option<TapSink<'_>>,
        mut routes: Option<Routes<'_>>,
        mut inputs: Option<Inputs<'_>>,
    ) -> Result<(u32, EncodeCount), SessionError> {
        if inputs.is_some() && taps.is_none() {
            return Err(refuse(
                "a step whose visits start from given rows reads every visit back",
            ));
        }
        if let Some(t) = self.trace.as_deref_mut() {
            t.begin();
        }
        let h0 = self.embed(id)?;
        if let Some(t) = taps.as_mut() {
            t(0, Tap::Embed, &h0);
        }
        self.a.scratch.rows.write_at(Elems(0), &h0)?;
        // Each visit's experts add into their own sum row and each pass's end
        // into its own message bank: both start the step at zero.
        self.a.scratch.moe.zero();
        self.a.scratch.message.zero();
        let stream = self.stream;
        let split = taps.is_some() || routes.is_some();
        let mut banks = 0usize;
        let mut passes = 0usize;
        // What every command buffer of the token encoded, summed.
        let mut count = EncodeCount::default();
        let mut tr = stream.begin()?.open_token_encoder()?;
        for (vi, v) in self.c.visits.iter().enumerate() {
            // The tapped walk settled every buffer before this visit.
            if let Some(f) = inputs.as_mut() {
                self.start_from(f, vi, banks)?;
            }
            tr.encoder(|enc| self.front(enc, vi, v, banks, pos, passes))?;
            if v.boundary {
                banks += 1;
            }
            if split {
                let (rec, encoded) = tr.finish();
                count += encoded;
                self.flush(rec)?;
                if let Some(t) = taps.as_mut() {
                    self.front_taps(vi, v, banks, t)?;
                }
                if let Some(r) = routes.as_mut() {
                    let (ids, weights) = r(vi).map_err(refuse)?;
                    self.admit(&ids)?;
                    self.a.scratch.route_ids.write_at(Elems(0), &ids)?;
                    self.a.scratch.route_weights.write_at(Elems(0), &weights)?;
                }
                tr = stream.begin()?.open_token_encoder()?;
            }
            tr.encoder(|enc| self.experts(enc, vi, v, banks, pos))?;
            if v.pass_end {
                tr.encoder(|enc| self.message_end(enc, banks, passes))?;
            }
            if let Some(t) = taps.as_mut() {
                let (rec, encoded) = tr.finish();
                count += encoded;
                self.flush(rec)?;
                self.back_taps(vi, banks, t)?;
                tr = stream.begin()?.open_token_encoder()?;
            } else if !split && chunk_ends(vi, self.c.visits.len()) {
                // The card runs the token's first visits while the host
                // encodes the rest: the chunk signals the token boundary at
                // its end, and the next chunk's encoder waits for it.
                let (rec, encoded) = tr.finish();
                count += encoded;
                let stamps = self.trace.as_ref().map(|_| rec.stamps());
                // The token's closing sync reads every chunk's failure.
                drop(stream.submit_signalled(rec)?);
                if let (Some(t), Some(stamps)) = (self.trace.as_deref_mut(), stamps) {
                    t.committed(stamps);
                }
                tr = stream.begin()?.open_token_encoder()?;
            }
            if v.pass_end {
                banks = self.c.prelude_blocks;
                passes += 1;
            }
        }
        if let Some(f) = inputs.as_mut() {
            self.start_from(f, self.c.visits.len(), banks)?;
        }
        tr.encoder(|enc| self.head(enc, banks))?;
        let (rec, encoded) = tr.finish();
        count += encoded;
        let stamps = self.trace.as_ref().map(|_| rec.stamps());
        stream.submit(rec)?;
        if let (Some(t), Some(stamps)) = (self.trace.as_deref_mut(), stamps) {
            t.committed(stamps);
        }
        stream.sync()?;
        if let Some(t) = self.trace.as_deref_mut() {
            t.woke();
        }
        let mut pick = [0u32];
        self.a
            .scratch
            .step_id
            .pages()
            .read_at(Elems(0), &mut pick)?;
        Ok((pick[0], count))
    }

    /// Every visit's groups and the head's on one encoder, in the order
    /// [`Walk::token`] encodes them (a plain step's, whose command buffers
    /// end between visits): the profile's walk, which holds all but one
    /// group back from Metal and runs the host side whole.
    pub(super) fn encode_visits(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        pos: u32,
    ) -> Result<(), SessionError> {
        let mut banks = 0usize;
        let mut passes = 0usize;
        for (vi, v) in self.c.visits.iter().enumerate() {
            self.front(enc, vi, v, banks, pos, passes)?;
            if v.boundary {
                banks += 1;
            }
            self.experts(enc, vi, v, banks, pos)?;
            if v.pass_end {
                self.message_end(enc, banks, passes)?;
                banks = self.c.prelude_blocks;
                passes += 1;
            }
        }
        self.head(enc, banks)
    }

    /// The bank's rows and the stream visit `vi` (the head at the visit
    /// count) starts from, where `inputs` gives them for its `banks` blocks.
    fn start_from(
        &mut self,
        inputs: &mut Inputs<'_>,
        vi: usize,
        banks: usize,
    ) -> Result<(), SessionError> {
        let Some(rows) = inputs(vi, banks).map_err(refuse)? else {
            return Ok(());
        };
        let want = (banks + 1) * self.c.hidden;
        if rows.len() != want {
            return Err(refuse(format!(
                "visit {vi} starts from {} values; its {banks} blocks and the stream are {want}",
                rows.len()
            )));
        }
        self.a.scratch.rows.write_at(Elems(0), &rows)?;
        Ok(())
    }

    /// Submit `rec` and wait for it.
    fn flush(&self, rec: Recording<'_>) -> Result<(), SessionError> {
        self.stream.submit(rec)?;
        self.stream.sync()?;
        Ok(())
    }

    /// The embedded and normed row of `id`, on the host.
    fn embed(&self, id: u32) -> Result<Vec<f32>, SessionError> {
        let e = &self.c.embed;
        let row = e.bytes_of(1);
        let mut bytes = vec![0u8; row];
        self.a
            .embed
            .pages()
            .read_at(Elems(id as usize * row), &mut bytes)?;
        let mut x = vec![0.0f32; e.cols];
        let ty = if e.quant == crate::kernels::Quant::F32 {
            gguf::quant::GgmlType::F32
        } else {
            e.stored
        };
        gguf::quant::dequant_row(ty, &bytes, &mut x)
            .map_err(|err| refuse(format!("the embedding of id {id}: {err}")))?;
        let ms = x.iter().map(|v| f64::from(*v) * f64::from(*v)).sum::<f64>() / x.len() as f64;
        let r = 1.0 / (ms + f64::from(self.c.rms_eps)).sqrt();
        for (v, g) in x.iter_mut().zip(&self.a.embed_norm) {
            *v = (f64::from(*v) * r) as f32 * g;
        }
        Ok(x)
    }

    /// A host's route the experts may read: each id under the expert count.
    fn admit(&self, ids: &[u32; 8]) -> Result<(), SessionError> {
        match ids.iter().find(|&&e| e as usize >= self.c.experts) {
            Some(bad) => Err(refuse(format!("a route names expert {bad}"))),
            None => Ok(()),
        }
    }

    /// Steps 1 to 5 of visit `vi`, the stream at row `banks`, after `passes`
    /// of the loop's passes ended.
    fn front(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        vi: usize,
        v: &VisitPlan,
        banks: usize,
        pos: u32,
        passes: usize,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let n = c.hidden;
        let layer = vi as u32;
        let p = &c.layers[v.layer];
        let Arena {
            layers,
            visits,
            scratch: sc,
            faults,
            loop_queries,
            ..
        } = &mut *self.a;
        let faults: &MetalFaultWord = faults;
        let lb: &LayerBufs = &layers[v.layer];
        let vb = &mut visits[vi];
        let scale = 1.0 / (n as f32).sqrt();
        let stream_at = banks * n;
        // 1. The attention's inputs: its mix and the loop's messages, which
        // read nothing the other writes, then the controllers.
        enc.tag(Kind::Mix);
        let mut inputs = [Src::Rows(stream_at); 3];
        let mut mixed = false;
        if let (Some((_, nq)), Some(query)) = (&p.attn_query, &lb.attn_query) {
            k.res_mix(
                enc,
                ResMixParams {
                    n,
                    nq: *nq,
                    m: 1,
                    n_banks: banks,
                    stride: n,
                    banks: (banks > 0)
                        .then(|| win(&sc.rows, 0, banks * n))
                        .transpose()?,
                    cur: win(&sc.rows, stream_at, n)?,
                    queries: query.whole()?,
                    y: sc.amix.window_mut(Elems(0), Elems(nq * n))?,
                    eps: EPS,
                    scale,
                    faults,
                    layer,
                },
            )?;
            mixed = true;
            inputs = if *nq == 1 {
                [Src::Amix(0); 3]
            } else {
                [Src::Amix(0), Src::Amix(n), Src::Amix(2 * n)]
            };
        }
        if v.message {
            let ci = v.core_index.expect("a message visit is a core visit");
            let m = c.message_rows;
            // The bank of the pass that ended last.
            let bank = passes
                .checked_sub(1)
                .ok_or_else(|| refuse("a message visit before any pass ended"))?
                * m
                * n;
            k.res_mix(
                enc,
                ResMixParams {
                    n,
                    nq: 4,
                    m: 1,
                    n_banks: m - 1,
                    stride: n,
                    banks: (m > 1)
                        .then(|| win(&sc.message, bank, (m - 1) * n))
                        .transpose()?,
                    cur: win(&sc.message, bank + (m - 1) * n, n)?,
                    queries: win(loop_queries, ci * 4 * n, 4 * n)?,
                    y: sc.msgs.window_mut(Elems(0), Elems(4 * n))?,
                    eps: EPS,
                    scale,
                    faults,
                    layer,
                },
            )?;
            mixed = true;
        }
        if mixed {
            // Hazard: the mixes and the messages, the controllers' and the
            // projections' inputs.
            enc.memory_barrier();
        }
        if v.message {
            // The three attention modules' down rows together, then their
            // merges together.
            enc.tag(Kind::LoopDown);
            for (i, input) in inputs.iter().enumerate() {
                k.loop_control_down(enc, &controller(c, lb, sc, *input, i, faults, layer)?)?;
            }
            // Hazard: the down rows, the merges' input.
            enc.memory_barrier();
            enc.tag(Kind::LoopMerge);
            for (i, input) in inputs.iter_mut().enumerate() {
                k.loop_control_merge(enc, &controller(c, lb, sc, *input, i, faults, layer)?)?;
                *input = Src::Xin(i * n);
            }
            // Hazard: the controllers' rows, the projections' inputs.
            enc.memory_barrier();
        }
        // 2. The projections, each path normed on load by its own gain.
        let norm = |i: usize| -> Result<LoadNorm<'_>, DeviceError> {
            Ok(LoadNorm::Gain {
                gain: lb.norms[i.min(lb.norms.len() - 1)].whole()?,
                eps: c.rms_eps,
            })
        };
        let rb = p.qrg[0].row_blocks();
        let qrg_segments = [
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
            Segment {
                cols: n,
                rows: Q_ROWS,
                weight_blocks: (Q_ROWS + REL_ROWS) * rb,
                out_offset: Q_ROWS + REL_ROWS,
            },
        ];
        enc.tag(Kind::Proj);
        k.matvec_seg(
            enc,
            SegParams {
                quant: p.qrg[0].quant,
                segments: &qrg_segments,
                weights: lb.qrg.whole()?,
                x: src(&sc.rows, &sc.amix, &sc.xin, n, inputs[0])?,
                y: sc.qrg.whole_mut()?,
                norm: norm(0)?,
                fault: faults,
                layer,
            },
        )?;
        let one = |rows: usize| {
            [Segment {
                cols: n,
                rows,
                weight_blocks: 0,
                out_offset: 0,
            }]
        };
        k.matvec_seg(
            enc,
            SegParams {
                quant: p.sink.quant,
                segments: &one(24),
                weights: lb.sink.whole()?,
                x: src(&sc.rows, &sc.amix, &sc.xin, n, inputs[0])?,
                y: sc.sinks.whole_mut()?,
                norm: norm(0)?,
                fault: faults,
                layer,
            },
        )?;
        for (i, (site, w, out)) in [(&p.k, &lb.k, &mut sc.k_raw), (&p.v, &lb.v, &mut sc.v_raw)]
            .into_iter()
            .enumerate()
        {
            k.matvec_seg(
                enc,
                SegParams {
                    quant: site.quant,
                    segments: &one(KV_ROWS),
                    weights: w.whole()?,
                    x: src(&sc.rows, &sc.amix, &sc.xin, n, inputs[i + 1])?,
                    y: out.whole_mut()?,
                    norm: norm(i + 1)?,
                    fault: faults,
                    layer,
                },
            )?;
        }
        for (i, (site, w, out)) in [
            (&p.k_taps, &lb.k_taps, &mut sc.k_taps),
            (&p.v_taps, &lb.v_taps, &mut sc.v_taps),
        ]
        .into_iter()
        .enumerate()
        {
            if let (Some(site), Some(w)) = (site, w) {
                k.matvec_seg(
                    enc,
                    SegParams {
                        quant: site.quant,
                        segments: &one(TAP_HEADS * c.taps),
                        weights: w.whole()?,
                        x: src(&sc.rows, &sc.amix, &sc.xin, n, inputs[i + 1])?,
                        y: out.whole_mut()?,
                        norm: norm(i + 1)?,
                        fault: faults,
                        layer,
                    },
                )?;
            }
        }
        // Hazard: the k and v rows and their taps, the convs' inputs.
        enc.memory_barrier();
        // 3. The k and v convs run in the tail of the clips and the append,
        // one launch.
        let rows = c.plane_rows(v.windowed);
        enc.tag(Kind::QkClip);
        let (q, r) = sc
            .qrg
            .window_mut(Elems(0), Elems(Q_ROWS + REL_ROWS))?
            .split_at(Elems(Q_ROWS))?;
        k.qk_clip_conv_append(
            enc,
            QkClipConvParams {
                clip: QkClipParams {
                    q,
                    k: sc.k.whole_mut()?,
                    v: sc.v.whole_mut()?,
                    r,
                    gq: lb.q_clip.whole()?,
                    gr: lb.rel_clip.whole()?,
                    cache_k: vb.cache_k.whole_mut()?,
                    cache_v: vb.cache_v.whole_mut()?,
                    ctx: rows,
                    pos: pos % rows as u32,
                    v_eps: EPS,
                    qk_thresh: c.qk_clip,
                    r_thresh: c.rel_clip,
                    tau: if v.windowed { 1.0 } else { tau(c, pos) },
                    faults,
                    layer,
                },
                conv: KvConvParams {
                    k_raw: sc.k_raw.whole()?,
                    v_raw: sc.v_raw.whole()?,
                    wk: lb.convs[0].whole()?,
                    wv: lb.convs[1].whole()?,
                    dyn_k: (!v.windowed).then(|| sc.k_taps.whole()).transpose()?,
                    dyn_v: (!v.windowed).then(|| sc.v_taps.whole()).transpose()?,
                    taps: c.taps,
                    ring: vb.ring.whole_mut()?,
                    ring_rows: c.taps - 1,
                    ch: CONV_CHANNELS,
                    base_k: BASE_K,
                    base_v: BASE_V,
                    pos,
                },
            },
        )?;
        // Hazard: the clipped q and relative rows and the cache's new row.
        enc.memory_barrier();
        // 4. The attention, its gate, wo and the post-conv.
        let window = v.windowed.then_some(c.window as u32);
        let segments = segments_for_window(pos + 1, window) as usize;
        let parts = crate::kernels::rel_partials_len(segments)
            .ok_or_else(|| refuse("the attention's partials overflow"))?;
        enc.tag(Kind::Flash);
        k.rel_flash(
            enc,
            RelFlashParams {
                segments,
                ctx: c.ctx,
                rows,
                keys: pos as usize + 1,
                window: if v.windowed { c.window } else { 0 },
                scale: 1.0 / (128.0f32).sqrt(),
                q: win(&sc.qrg, 0, Q_ROWS)?,
                cache_k: vb.cache_k.whole()?,
                cache_v: vb.cache_v.whole()?,
                r: win(&sc.qrg, Q_ROWS, REL_ROWS)?,
                bias: lb.rel_bias.whole()?,
                sinks: sc.sinks.whole()?,
                partials: sc.partials.window_mut(Elems(0), Elems(parts))?,
                out: sc.att.whole_mut()?,
                faults,
                layer,
            },
        )?;
        // Hazard: the attention's output, the gate's input.
        enc.memory_barrier();
        enc.tag(Kind::Gate);
        k.out_gate(
            enc,
            OutGateParams {
                n: Q_ROWS,
                kqv: sc.att.whole()?,
                g: win(&sc.qrg, Q_ROWS + REL_ROWS, Q_ROWS)?,
                y: sc.gated.whole_mut()?,
                faults,
                layer,
            },
        )?;
        // Hazard: the gated rows, wo's input.
        enc.memory_barrier();
        enc.tag(Kind::Wo);
        k.matvec(
            enc,
            MatvecParams {
                quant: p.wo.quant,
                rows: n,
                cols: Q_ROWS,
                weights: lb.wo.whole()?,
                x: sc.gated.whole()?,
                y: sc.ao.whole_mut()?,
                mode: Mode::Store,
            },
        )?;
        // Hazard: wo's row, the post-conv's input.
        enc.memory_barrier();
        // At a boundary the stream row closes as a block and the post-conv's
        // output is the residual, the next row's stream.
        enc.tag(Kind::ConvAttn);
        if v.boundary {
            // The stream row closes as a block and the post-conv's output is
            // the residual, the next row's stream.
            k.conv_res(
                enc,
                ConvResParams {
                    d: n,
                    taps: c.taps,
                    heads: 0,
                    rows: c.taps - 1,
                    ch: CONV_CHANNELS,
                    base: BASE_ATTN,
                    m: 1,
                    pos,
                    x: sc.ao.whole()?,
                    w: lb.convs[2].whole()?,
                    dyn_taps: None,
                    y: sc.rows.window_mut(Elems(stream_at + n), Elems(n))?,
                    ring: vb.ring.whole_mut()?,
                    faults,
                    layer,
                },
            )?;
            // Hazard: the post-conv's row, the new stream row the FFN's mix
            // reads.
            enc.memory_barrier();
        } else {
            // The post-conv's output adds into the stream in the same launch.
            k.conv_res_join(
                enc,
                ConvResJoinParams {
                    cols: n,
                    x: sc.ao.whole()?,
                    conv: TailConv {
                        w: lb.convs[2].whole()?,
                        taps: c.taps,
                        ring: vb.ring.whole_mut()?,
                        ring_rows: c.taps - 1,
                        ch: CONV_CHANNELS,
                        base: BASE_ATTN,
                        pos,
                    },
                    y: sc.ac.whole_mut()?,
                    stream: sc.rows.window_mut(Elems(stream_at), Elems(n))?,
                    faults,
                    layer,
                },
            )?;
            // Hazard: the stream row, the FFN's mix's input.
            enc.memory_barrier();
        }
        // 5. The FFN's mix, its message, its norm and the router.
        let banks = banks + usize::from(v.boundary);
        let stream_at = banks * n;
        enc.tag(Kind::FfnMix);
        k.res_mix(
            enc,
            ResMixParams {
                n,
                nq: 1,
                m: 1,
                n_banks: banks,
                stride: n,
                banks: Some(win(&sc.rows, 0, banks * n)?),
                cur: win(&sc.rows, stream_at, n)?,
                queries: lb.ffn_query.whole()?,
                y: sc.fmix.whole_mut()?,
                eps: EPS,
                scale,
                faults,
                layer,
            },
        )?;
        // Hazard: the FFN's mix, its controller's (or its norm's) input.
        enc.memory_barrier();
        let ffn_in = if v.message {
            enc.tag(Kind::LoopFfn);
            let [down, gain, forget] = &lb.controllers[3];
            k.loop_control(
                enc,
                LoopControlParams {
                    n,
                    rank: RANK,
                    m: 1,
                    cur: sc.fmix.whole()?,
                    msg: win(&sc.msgs, 3 * n, n)?,
                    down: down.whole()?,
                    gain: gain.whole()?,
                    forget: forget.whole()?,
                    y: sc.fin.whole_mut()?,
                    h: sc.lc_h.window_mut(Elems(3 * RANK), Elems(RANK))?,
                    eps: EPS,
                    faults,
                    layer,
                },
            )?;
            // Hazard: the controller's row, the norm's input.
            enc.memory_barrier();
            &sc.fin
        } else {
            &sc.fmix
        };
        // The norm runs in the router's front: its row is the experts' input.
        enc.tag(Kind::Router);
        k.router_sigmoid_norm(
            enc,
            RouterSigmoidNormParams {
                router: RouterSigmoidParams {
                    k: n,
                    top_k: c.top_k,
                    rows: lb.router.whole()?,
                    x: ffn_in.whole()?,
                    bias: lb.router_bias.whole()?,
                    partials: sc.router_partials.whole_mut()?,
                    ticket: sc.ticket.whole_mut()?,
                    logits: sc.router_logits.whole_mut()?,
                    ids: sc.route_ids.whole_mut()?,
                    weights: sc.route_weights.whole_mut()?,
                    scale: c.route_scale,
                    faults,
                    layer,
                },
                gain: lb.ffn_norm.whole()?,
                normed: sc.fnorm.whole_mut()?,
                eps: c.rms_eps,
            },
        )?;
        Ok(())
    }

    /// Step 6 of visit `vi`, the stream at row `banks`: the router's picks
    /// (or the host's route) read on the card by every launch of the slots.
    fn experts(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        vi: usize,
        v: &VisitPlan,
        banks: usize,
        pos: u32,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let (n, ff, slots) = (c.hidden, c.ff, c.top_k);
        let layer = vi as u32;
        let p = &c.layers[v.layer];
        let Arena {
            layers,
            visits,
            scratch: sc,
            faults,
            ..
        } = &mut *self.a;
        let faults: &MetalFaultWord = faults;
        let lb = &layers[v.layer];
        // Hazard: the router's ids and weights and the normed row, the
        // slots' inputs.
        enc.memory_barrier();
        // Every slot's gate rows first, then every slot's up rows.
        enc.tag(Kind::GateUp);
        k.nvfp4_matvec(
            enc,
            Nvfp4Params {
                layout: p.gate_up.layout,
                rows: 2 * ff,
                k: n,
                weights: lb.gate_up.whole()?,
                ids: win(&sc.route_ids, 0, slots)?,
                scales: lb.gate_up_scale.whole()?,
                route_weights: None,
                x: sc.fnorm.whole()?,
                x_per_slot: false,
                y: sc.gate_up.window_mut(Elems(0), Elems(slots * 2 * ff))?,
                plane: ff,
                faults,
                layer,
            },
        )?;
        // Hazard: the slots' gate and up rows, the activation's input.
        enc.memory_barrier();
        enc.tag(Kind::Pair);
        k.pair_situ(
            enc,
            SituParams {
                n: slots * ff,
                gate: win(&sc.gate_up, 0, slots * ff)?,
                up: win(&sc.gate_up, slots * ff, slots * ff)?,
                y: sc.act.window_mut(Elems(0), Elems(slots * ff))?,
                gate_beta: c.gate_beta,
                value_beta: c.value_beta,
                faults,
                layer,
            },
        )?;
        // Hazard: the slots' activations, the downs' inputs.
        enc.memory_barrier();
        enc.tag(Kind::Down);
        k.nvfp4_matvec(
            enc,
            Nvfp4Params {
                layout: p.down.layout,
                rows: n,
                k: ff,
                weights: lb.down.whole()?,
                ids: win(&sc.route_ids, 0, slots)?,
                scales: lb.down_scale.whole()?,
                route_weights: Some(win(&sc.route_weights, 0, slots)?),
                x: win(&sc.act, 0, slots * ff)?,
                x_per_slot: true,
                y: sc.down.window_mut(Elems(0), Elems(slots * n))?,
                plane: n,
                faults,
                layer,
            },
        )?;
        // Hazard: the slots' weighted rows, the combine's input.
        enc.memory_barrier();
        // The sum, the post-conv of it and the stream's add, one launch.
        enc.tag(Kind::Combine);
        k.combine_conv_join(
            enc,
            CombineConvJoinParams {
                cols: n,
                slots,
                y: sc.down.whole()?,
                y_stride: n,
                x: sc.moe.window_mut(Elems(vi * n), Elems(n))?,
                conv: TailConv {
                    w: lb.convs[3].whole()?,
                    taps: c.taps,
                    ring: visits[vi].ring.whole_mut()?,
                    ring_rows: c.taps - 1,
                    ch: CONV_CHANNELS,
                    base: BASE_FFN,
                    pos,
                },
                conv_out: sc.moe_conv.whole_mut()?,
                stream: sc.rows.window_mut(Elems(banks * n), Elems(n))?,
                faults,
                layer,
            },
        )?;
        // Hazard: the stream row, the next visit's (or the head's) input.
        enc.memory_barrier();
        Ok(())
    }

    /// The end of a pass the loop runs again, the stream at row `banks`: its
    /// blocks past its first and its stream added into message bank `bank`,
    /// which the step zeroed and the next pass's message visits read.
    fn message_end(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        banks: usize,
        bank: usize,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let (n, m, pb) = (c.hidden, c.message_rows, c.prelude_blocks);
        if banks.checked_sub(pb) != Some(m) {
            return Err(refuse(format!(
                "a pass ends at bank row {banks}; its message is {m} rows past the prelude's {pb}"
            )));
        }
        let sc = &mut self.a.scratch;
        enc.tag(Kind::MsgEnd);
        k.bias_rows(
            enc,
            BiasRowsParams {
                rows: 1,
                cols: m * n,
                y: sc.message.window_mut(Elems(bank * m * n), Elems(m * n))?,
                y_stride: m * n,
                bias: win(&sc.rows, (pb + 1) * n, m * n)?,
            },
        )?;
        // Hazard: the next pass rewrites the rows the bank was read from.
        enc.memory_barrier();
        Ok(())
    }

    /// The head: the mix of the bank and the stream, normed on load, its
    /// logits over the cut and the pick.
    fn head(&mut self, enc: &mut ComputeEncoder<'_>, banks: usize) -> Result<(), SessionError> {
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
        let layer = c.visits.len() as u32;
        enc.tag(Kind::HeadMix);
        k.res_mix(
            enc,
            ResMixParams {
                n,
                nq: 1,
                m: 1,
                n_banks: banks,
                stride: n,
                banks: (banks > 0)
                    .then(|| win(&sc.rows, 0, banks * n))
                    .transpose()?,
                cur: win(&sc.rows, banks * n, n)?,
                queries: output_query.whole()?,
                y: sc.head_mix.whole_mut()?,
                eps: EPS,
                scale: 1.0 / (n as f32).sqrt(),
                faults,
                layer,
            },
        )?;
        enc.memory_barrier();
        enc.tag(Kind::Head);
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

    /// The values of `len` elements of `b` from `at`.
    fn read(b: &MetalBuffer<f32>, at: usize, len: usize) -> Result<Vec<f32>, SessionError> {
        let mut v = vec![0.0f32; len];
        b.pages().read_at(Elems(at), &mut v)?;
        Ok(v)
    }

    /// The nodes of visit `vi` the front's buffer settled, the stream at row
    /// `banks` (after a boundary's close).
    fn front_taps(
        &self,
        vi: usize,
        v: &VisitPlan,
        banks: usize,
        t: &mut TapSink<'_>,
    ) -> Result<(), SessionError> {
        let n = self.c.hidden;
        let sc = &self.a.scratch;
        t(vi, Tap::Q, &Self::read(&sc.qrg, 0, Q_ROWS)?);
        t(vi, Tap::R, &Self::read(&sc.qrg, Q_ROWS, REL_ROWS)?);
        t(vi, Tap::K, &Self::read(&sc.k, 0, KV_ROWS)?);
        t(vi, Tap::V, &Self::read(&sc.v, 0, KV_ROWS)?);
        t(vi, Tap::Kqv, &Self::read(&sc.att, 0, Q_ROWS)?);
        if !v.boundary {
            t(vi, Tap::AttnOut, &Self::read(&sc.ac, 0, n)?);
        }
        t(vi, Tap::FfnInp, &Self::read(&sc.rows, banks * n, n)?);
        if v.message {
            t(vi, Tap::LoopQ, &Self::read(&sc.xin, 0, n)?);
            t(vi, Tap::LoopFfn, &Self::read(&sc.fin, 0, n)?);
        }
        t(vi, Tap::FfnNorm, &Self::read(&sc.fnorm, 0, n)?);
        t(
            vi,
            Tap::Probs,
            &Self::read(&sc.router_logits, 0, self.c.experts)?,
        );
        let mut ids = vec![0u32; self.c.top_k];
        sc.route_ids.pages().read_at(Elems(0), &mut ids)?;
        let ids: Vec<f32> = ids.iter().map(|&e| e as f32).collect();
        t(vi, Tap::TopK, &ids);
        t(
            vi,
            Tap::Weights,
            &Self::read(&sc.route_weights, 0, self.c.top_k)?,
        );
        Ok(())
    }

    /// The nodes of visit `vi` the experts' buffer settled.
    fn back_taps(&self, vi: usize, banks: usize, t: &mut TapSink<'_>) -> Result<(), SessionError> {
        let n = self.c.hidden;
        let sc = &self.a.scratch;
        t(vi, Tap::MoeOut, &Self::read(&sc.moe, vi * n, n)?);
        t(vi, Tap::FfnOut, &Self::read(&sc.moe_conv, 0, n)?);
        t(vi, Tap::LOut, &Self::read(&sc.rows, banks * n, n)?);
        Ok(())
    }
}

impl Walk<'_> {
    /// The embedded and normed row of `id` a prompt pass's row starts from:
    /// the step's own.
    pub(super) fn embedded(&self, id: u32) -> Result<Vec<f32>, SessionError> {
        self.embed(id)
    }
}
