//! The walk of one Inkling 2 token over its visits: the dispatches of a
//! visit in the reference's order (`mlx_lm/models/inkling2/layers.py`,
//! `DecoderLayer.__call__`), the command buffers the routed experts and the
//! loop's passes cut, and the host's reads between them.
//!
//! A visit, at stream row `s` of the bank's rows (the blocks closed so far,
//! then the stream):
//! 1. the attention's inputs — the normed mix of the blocks and the stream
//!    (one query on a windowed visit, three on a windowless one; layer 0 reads
//!    the stream as it is), each merged with its loop message on a later
//!    pass's core visit;
//! 2. the q path (q, relative and gate rows in one launch, the sink rows in
//!    another) and the k and v paths, each normed on load by its gain;
//! 3. the k and v short convs (a windowless visit's taps from its own rows),
//!    then the clips, the value norm and the cache append;
//! 4. the rel-bias attention, its output gate, `wo` and the post-conv; at a
//!    block boundary the stream row closes as a block and the post-conv's
//!    output is the next row's stream, else it adds into the stream;
//! 5. the FFN's mix, its loop message, its norm and the router — the buffer
//!    ends here and the host reads the picks;
//! 6. each pick's gate·up rows, situ and down rows (the route weight folded
//!    into the down's scale), their sum, the post-conv and its add.
//!
//! A pass the loop runs again ends its buffer after its last visit: the host
//! copies the pass's blocks past its first and its stream into the message
//! bank, and the next pass starts again at the prelude's stream row, which
//! the core's first visit closed and nothing since has written. The head mixes
//! the bank and the stream, norms the mix on load and picks.

use bloomery_device::{Buffer, DeviceError, Elems, Stream, Window};

use super::arena::{Arena, LayerBufs};
use super::check::{CONV_CHANNELS, Checked, EPS, VisitPlan};
use super::session::{Inputs, Routes, SessionError, Tap, TapSink, refuse};
use crate::buffer::MetalBuffer;
use crate::device::MetalDevice;
use crate::encode::ComputeEncoder;
use crate::fault::MetalFaultWord;
use crate::kernels::{
    BiasRowsParams, CombineParams, ConvResParams, HEAD_SIMD_GROUPS, HeadParams, Kernels, LoadNorm,
    LoopControlParams, MatvecParams, Mode, NormRowsParams, Nvfp4Params, OutGateParams,
    QkClipParams, RelFlashParams, ResMixParams, RouterSigmoidParams, SegParams, Segment,
    SituParams, segments_for_window,
};
use crate::stream::{MetalStream, Recording};

/// The query, key and relative rows' widths of a visit.
const Q_ROWS: usize = 3072;
const KV_ROWS: usize = 1024;
const REL_ROWS: usize = 384;
/// The heads of a windowless conv's dynamic taps, and the controllers' rank.
const TAP_HEADS: usize = 8;
const RANK: usize = 64;
/// The conv ring's channel bases: k, v, the attention's post-conv, the FFN's.
const BASE_K: usize = 0;
const BASE_V: usize = 1024;
const BASE_ATTN: usize = 2048;
const BASE_FFN: usize = 2048 + 2304;

/// Where a path's input lies.
#[derive(Clone, Copy)]
enum Src {
    /// The bank's rows, at element `at`.
    Rows(usize),
    Amix(usize),
    Xin(usize),
}

/// One token's walk over the session's buffers.
pub struct Walk<'a> {
    pub c: &'a Checked,
    pub a: &'a mut Arena,
    pub k: &'a Kernels,
    pub stream: &'a MetalStream,
}

/// `len` elements of `b` from `at`.
fn win<T: bloomery_device::Elem>(
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
fn tau(c: &Checked, pos: u32) -> f32 {
    let ratio = (f64::from(pos) + 1.0) / f64::from(c.log_floor);
    (1.0 + f64::from(c.log_alpha) * ratio.max(1.0).ln()) as f32
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
    ) -> Result<u32, SessionError> {
        let n = self.c.hidden;
        if inputs.is_some() && taps.is_none() {
            return Err(refuse(
                "a step whose visits start from given rows reads every visit back",
            ));
        }
        let h0 = self.embed(id)?;
        if let Some(t) = taps.as_mut() {
            t(0, Tap::Embed, &h0);
        }
        self.a.scratch.rows.write_at(Elems(0), &h0)?;
        let stream = self.stream;
        let mut banks = 0usize;
        let mut rec = stream.begin()?;
        for (vi, v) in self.c.visits.iter().enumerate() {
            // The tapped walk settled every buffer before this visit.
            if let Some(f) = inputs.as_mut() {
                self.start_from(f, vi, banks)?;
            }
            {
                let mut enc = rec.compute_encoder()?;
                let r = self.front(&mut enc, vi, v, banks, pos);
                enc.end();
                r?;
            }
            if v.boundary {
                banks += 1;
            }
            self.flush(rec)?;
            if let Some(t) = taps.as_mut() {
                self.front_taps(vi, v, banks, t)?;
            }
            let (ids, weights) = match routes.as_mut() {
                Some(r) => {
                    let (ids, weights) = r(vi).map_err(refuse)?;
                    self.admit(&ids)?;
                    (ids, weights)
                }
                None => self.picks()?,
            };
            self.a.scratch.moe.zero();
            rec = stream.begin()?;
            {
                let mut enc = rec.compute_encoder()?;
                let r = self.experts(&mut enc, vi, v, banks, &ids, &weights, pos);
                enc.end();
                r?;
            }
            if v.pass_end || taps.is_some() {
                self.flush(rec)?;
                if let Some(t) = taps.as_mut() {
                    self.back_taps(vi, banks, t)?;
                }
                if v.pass_end {
                    let pb = self.c.prelude_blocks;
                    let mut rows = vec![0.0f32; (banks - pb) * n];
                    self.a
                        .scratch
                        .rows
                        .pages()
                        .read_at(Elems((pb + 1) * n), &mut rows)?;
                    self.a.scratch.message.write_at(Elems(0), &rows)?;
                    banks = pb;
                }
                rec = stream.begin()?;
            }
        }
        if let Some(f) = inputs.as_mut() {
            self.start_from(f, self.c.visits.len(), banks)?;
        }
        {
            let mut enc = rec.compute_encoder()?;
            let r = self.head(&mut enc, banks);
            enc.end();
            r?;
        }
        self.flush(rec)?;
        let mut pick = [0u32];
        self.a
            .scratch
            .step_id
            .pages()
            .read_at(Elems(0), &mut pick)?;
        Ok(pick[0])
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

    /// The router's picks and their weights, read on the host.
    fn picks(&self) -> Result<([u32; 8], [f32; 8]), SessionError> {
        let mut ids = [0u32; 8];
        let mut w = [0.0f32; 8];
        self.a
            .scratch
            .route_ids
            .pages()
            .read_at(Elems(0), &mut ids)?;
        self.a
            .scratch
            .route_weights
            .pages()
            .read_at(Elems(0), &mut w)?;
        self.admit(&ids)?;
        Ok((ids, w))
    }

    /// Kept ids the experts' launches may index: each under the expert count.
    fn admit(&self, ids: &[u32; 8]) -> Result<(), SessionError> {
        match ids.iter().find(|&&e| e as usize >= self.c.experts) {
            Some(bad) => Err(refuse(format!("a route names expert {bad}"))),
            None => Ok(()),
        }
    }

    /// Steps 1 to 5 of visit `vi`, the stream at row `banks`.
    fn front(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        vi: usize,
        v: &VisitPlan,
        banks: usize,
        pos: u32,
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
        // 1. The attention's inputs.
        let mut inputs = [Src::Rows(stream_at); 3];
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
            enc.memory_barrier();
            inputs = if *nq == 1 {
                [Src::Amix(0); 3]
            } else {
                [Src::Amix(0), Src::Amix(n), Src::Amix(2 * n)]
            };
        }
        if v.message {
            let ci = v.core_index.expect("a message visit is a core visit");
            let m = c.message_rows;
            k.res_mix(
                enc,
                ResMixParams {
                    n,
                    nq: 4,
                    m: 1,
                    n_banks: m - 1,
                    stride: n,
                    banks: (m > 1)
                        .then(|| win(&sc.message, 0, (m - 1) * n))
                        .transpose()?,
                    cur: win(&sc.message, (m - 1) * n, n)?,
                    queries: win(loop_queries, ci * 4 * n, 4 * n)?,
                    y: sc.msgs.window_mut(Elems(0), Elems(4 * n))?,
                    eps: EPS,
                    scale,
                    faults,
                    layer,
                },
            )?;
            enc.memory_barrier();
            for (i, input) in inputs.iter_mut().enumerate() {
                let [down, gain, forget] = &lb.controllers[i];
                k.loop_control(
                    enc,
                    LoopControlParams {
                        n,
                        rank: RANK,
                        m: 1,
                        cur: match *input {
                            Src::Rows(at) => win(&sc.rows, at, n)?,
                            Src::Amix(at) => win(&sc.amix, at, n)?,
                            Src::Xin(_) => {
                                return Err(refuse(
                                    "a controller's input is another controller's output",
                                ));
                            }
                        },
                        msg: win(&sc.msgs, i * n, n)?,
                        down: down.whole()?,
                        gain: gain.whole()?,
                        forget: forget.whole()?,
                        y: sc.xin.window_mut(Elems(i * n), Elems(n))?,
                        eps: EPS,
                        faults,
                        layer,
                    },
                )?;
                *input = Src::Xin(i * n);
            }
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
        enc.memory_barrier();
        // 3. The k and v convs, then the clips and the append.
        let rows = c.plane_rows(v.windowed);
        for (base, w, x, taps_buf, y) in [
            (BASE_K, &lb.convs[0], &sc.k_raw, &sc.k_taps, &mut sc.k),
            (BASE_V, &lb.convs[1], &sc.v_raw, &sc.v_taps, &mut sc.v),
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
                    m: 1,
                    pos,
                    x: x.whole()?,
                    w: w.whole()?,
                    dyn_taps: (!v.windowed).then(|| taps_buf.whole()).transpose()?,
                    y: y.whole_mut()?,
                    ring: vb.ring.whole_mut()?,
                    faults,
                    layer,
                },
            )?;
        }
        enc.memory_barrier();
        let (q, r) = sc
            .qrg
            .window_mut(Elems(0), Elems(Q_ROWS + REL_ROWS))?
            .split_at(Elems(Q_ROWS))?;
        k.qk_clip_append(
            enc,
            QkClipParams {
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
        )?;
        enc.memory_barrier();
        // 4. The attention, its gate, wo and the post-conv.
        let window = v.windowed.then_some(c.window as u32);
        let segments = segments_for_window(pos + 1, window) as usize;
        let parts = crate::kernels::rel_partials_len(segments)
            .ok_or_else(|| refuse("the attention's partials overflow"))?;
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
        enc.memory_barrier();
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
        enc.memory_barrier();
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
        enc.memory_barrier();
        // At a boundary the stream row closes as a block and the post-conv's
        // output is the residual, the next row's stream.
        let y = if v.boundary {
            sc.rows.window_mut(Elems(stream_at + n), Elems(n))?
        } else {
            sc.ac.whole_mut()?
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
                m: 1,
                pos,
                x: sc.ao.whole()?,
                w: lb.convs[2].whole()?,
                dyn_taps: None,
                y,
                ring: vb.ring.whole_mut()?,
                faults,
                layer,
            },
        )?;
        enc.memory_barrier();
        if !v.boundary {
            k.bias_rows(
                enc,
                BiasRowsParams {
                    rows: 1,
                    cols: n,
                    y: sc.rows.window_mut(Elems(stream_at), Elems(n))?,
                    y_stride: n,
                    bias: sc.ac.whole()?,
                },
            )?;
            enc.memory_barrier();
        }
        // 5. The FFN's mix, its message, its norm and the router.
        let banks = banks + usize::from(v.boundary);
        let stream_at = banks * n;
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
        enc.memory_barrier();
        let ffn_in = if v.message {
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
                    eps: EPS,
                    faults,
                    layer,
                },
            )?;
            enc.memory_barrier();
            &sc.fin
        } else {
            &sc.fmix
        };
        k.norm_rows(
            enc,
            NormRowsParams {
                rows: 1,
                cols: n,
                x: ffn_in.whole()?,
                x_stride: n,
                gain: lb.ffn_norm.whole()?,
                y: sc.fnorm.whole_mut()?,
                y_stride: n,
                eps: c.rms_eps,
                fault: faults,
                layer,
            },
        )?;
        enc.memory_barrier();
        k.router_sigmoid(
            enc,
            RouterSigmoidParams {
                k: n,
                top_k: c.top_k,
                rows: lb.router.whole()?,
                x: sc.fnorm.whole()?,
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
        )?;
        Ok(())
    }

    /// Step 6 of visit `vi` at the picks `ids` and their weights, the stream at
    /// row `banks`.
    #[allow(clippy::too_many_arguments)]
    fn experts(
        &mut self,
        enc: &mut ComputeEncoder<'_>,
        vi: usize,
        v: &VisitPlan,
        banks: usize,
        ids: &[u32; 8],
        weights: &[f32; 8],
        pos: u32,
    ) -> Result<(), SessionError> {
        let (c, k) = (self.c, self.k);
        let (n, ff) = (c.hidden, c.ff);
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
        let (gu_per, down_per) = (p.gate_up.expert_bytes(), p.down.expert_bytes());
        for (s, &e) in ids.iter().enumerate() {
            let e = e as usize;
            k.nvfp4_matvec(
                enc,
                Nvfp4Params {
                    layout: p.gate_up.layout,
                    rows: 2 * ff,
                    k: n,
                    weights: win(&lb.gate_up, e * gu_per, gu_per)?,
                    x: sc.fnorm.whole()?,
                    y: sc.gate_up.window_mut(Elems(s * 2 * ff), Elems(2 * ff))?,
                    w_scale: lb.gate_up_scale[e],
                    faults,
                    layer,
                },
            )?;
        }
        enc.memory_barrier();
        for s in 0..ids.len() {
            k.pair_situ(
                enc,
                SituParams {
                    n: ff,
                    gate: win(&sc.gate_up, s * 2 * ff, ff)?,
                    up: win(&sc.gate_up, s * 2 * ff + ff, ff)?,
                    y: sc.act.window_mut(Elems(s * ff), Elems(ff))?,
                    gate_beta: c.gate_beta,
                    value_beta: c.value_beta,
                    faults,
                    layer,
                },
            )?;
        }
        enc.memory_barrier();
        for (s, (&e, &w)) in ids.iter().zip(weights).enumerate() {
            let e = e as usize;
            k.nvfp4_matvec(
                enc,
                Nvfp4Params {
                    layout: p.down.layout,
                    rows: n,
                    k: ff,
                    weights: win(&lb.down, e * down_per, down_per)?,
                    x: win(&sc.act, s * ff, ff)?,
                    y: sc.down.window_mut(Elems(s * n), Elems(n))?,
                    w_scale: lb.down_scale[e] * w,
                    faults,
                    layer,
                },
            )?;
        }
        enc.memory_barrier();
        k.combine_rows(
            enc,
            CombineParams {
                rows: 1,
                per_row: ids.len(),
                cols: n,
                y: sc.down.whole()?,
                y_rows: ids.len(),
                y_stride: n,
                inv: sc.slots.whole()?,
                x: sc.moe.whole_mut()?,
                x_stride: n,
            },
        )?;
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
                m: 1,
                pos,
                x: sc.moe.whole()?,
                w: lb.convs[3].whole()?,
                dyn_taps: None,
                y: sc.moe_conv.whole_mut()?,
                ring: visits[vi].ring.whole_mut()?,
                faults,
                layer,
            },
        )?;
        enc.memory_barrier();
        k.bias_rows(
            enc,
            BiasRowsParams {
                rows: 1,
                cols: n,
                y: sc.rows.window_mut(Elems(banks * n), Elems(n))?,
                y_stride: n,
                bias: sc.moe_conv.whole()?,
            },
        )?;
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
        t(vi, Tap::MoeOut, &Self::read(&sc.moe, 0, n)?);
        t(vi, Tap::FfnOut, &Self::read(&sc.moe_conv, 0, n)?);
        t(vi, Tap::LOut, &Self::read(&sc.rows, banks * n, n)?);
        Ok(())
    }
}
