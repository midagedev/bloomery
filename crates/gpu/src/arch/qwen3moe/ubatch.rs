//! qwen3moe's GEMM prefill: a prompt fed in ubatches of up to [`UBATCH`]
//! tokens, every weight read once per ubatch. The dense projections run
//! through the grouped int8 GEMM over a one-expert table built once per
//! ubatch; the routed experts through one route table per layer, gate and up
//! reading each token's column (`GemmInput::Shared`), down each slot's own.
//! The glue between them is the chain's own kernels at `T` rows: the
//! embedding rows, `rms_norm` then the q8_1 quantizer (the bytes
//! `norm_quant` writes), the per-head norm with the rope and the cache
//! append, the grouped-query flash with each row's own live key count, the
//! residual add, the router in two launches, SwiGLU with the down's
//! quantizer in one launch (`GemmKernels::enqueue_swiglu_quant`), and the
//! combine of the down rows with the router weights and the residual.
//!
//! Numeric class. Every launch computes a token's values from that token's
//! inputs alone, so a token's bits do not depend on the ubatch it lands in or
//! the tokens beside it — the GEMM accumulates each (slot, row) on its own,
//! the norm, quantizer, router and combine work per column, and the flash
//! walks and merges a row's live segments in one order whatever the grid.
//! Against the one-token path the dense and expert products sum in another
//! order (128-value integer blocks into one f32 accumulator, where the gemv
//! sums lane partials through a warp tree), so the two paths agree to the
//! error of those sums and are not bit-equal.
//!
//! Buffers. The arena holds `min(UBATCH, ctx)` rows of every intermediate
//! and the prompt image room for a prompt as long as the cache: token ids,
//! positions, live key counts and rope rows, each an array over the prompt,
//! written and copied to the card once per prompt; a ubatch reads windows of
//! it. Nothing is allocated per prompt or per ubatch. The ubatches run eager
//! in both step modes.

use super::body::{Kernels, Kq, LayerNames};
use super::experts::CombineArgs;
use super::router::{N_EXPERT, N_USED, RouterOut};
use super::scratch::{Dims, KvPlanes, f32_view, param_view};
use crate::flash_gqa::{GqaArgs, HEAD, segments_for};
use crate::gemm::{GEMM_MAX_SLOTS, GemmAct, GemmInput, GemmRoute, GemmWeight};
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::NeoxArgs;
use crate::rope_table::{Direction, RopeTable};
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::qwen3moe::names::token_embd;
use std::mem::ManuallyDrop;

/// Tokens one ubatch takes: the grouped GEMM's slot cap at top-8.
pub const UBATCH: usize = GEMM_MAX_SLOTS / N_USED;

/// Row-segment pairs one flash launch of a ubatch covers at most, the size
/// its partials are cut for: a ubatch's rows are walked in chunks whose row
/// count times the segments of their deepest row stays within it (or a
/// single row, when one row's segments pass it).
const FLASH_ROW_SEGS: usize = 2048;

const WHAT: &str = "qwen3moe::ubatch";

/// The GEMM's type for a qwen3moe projection.
fn gemm_ty(kq: Kq) -> GemmWeight {
    match kq {
        Kq::Q4K => GemmWeight::Q4K,
        Kq::Q6K => GemmWeight::Q6K,
    }
}

/// The ubatch arena, in chain order: every intermediate of one layer for up
/// to `rows` tokens, token-major (slot-major for the expert rows), shared by
/// all layers.
pub(super) struct UbArena {
    dims: Dims,
    rows: usize,
    /// The layer's input residual (the embedding rows for layer 0), and its
    /// output: the combine writes the next layer's input here.
    x: DeviceBuffer<f32>,
    /// The normed rows of either norm: the quantizer's input, and the
    /// router's.
    normed: DeviceBuffer<f32>,
    /// q8_1 of `normed`: the q·k·v input, then the gate·up input.
    act_hid: GemmAct,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    part_v: DeviceBuffer<f32>,
    part_ms: DeviceBuffer<f32>,
    /// Row-segment pairs the partials hold.
    row_segs: usize,
    /// The attention rows, `n_head · HEAD` per token.
    attn: DeviceBuffer<f32>,
    act_attn: GemmAct,
    /// The output projection's rows; the residual add reads them.
    attn_o: DeviceBuffer<f32>,
    /// The FFN's input residual: `x` plus the attention output.
    ffn_inp: DeviceBuffer<f32>,
    /// The router's outputs, slot `t · N_USED + j` token `t`'s slot `j`.
    route: RouterOut,
    /// The one-expert table every dense projection of a ubatch reads.
    dense: GemmRoute,
    /// The layer's expert table over `t · N_USED` slots.
    moe: GemmRoute,
    /// Gate and up rows, per slot `ff` values.
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    /// q8_1 of their SwiGLU, one column per slot: the down's input.
    act_h: GemmAct,
    /// The down rows, per slot `hidden` values.
    down: DeviceBuffer<f32>,
}

impl UbArena {
    /// The arena for `d` and up to `rows` tokens. Load-time only.
    fn new(stream: &CudaStream, d: Dims, rows: usize) -> Result<UbArena, GpuError> {
        let q_len = d.n_head * HEAD;
        let kv_len = d.n_kv * HEAD;
        let slots = rows * N_USED;
        let row_segs = FLASH_ROW_SEGS.max(segments_for(d.ctx));
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(UbArena {
            x: f(rows * d.hidden)?,
            normed: f(rows * d.hidden)?,
            act_hid: GemmAct::new(stream, rows, d.hidden)?,
            q: f(rows * q_len)?,
            k: f(rows * kv_len)?,
            v: f(rows * kv_len)?,
            part_v: f(row_segs * d.n_head * HEAD)?,
            part_ms: f(row_segs * d.n_head * 2)?,
            row_segs,
            attn: f(rows * q_len)?,
            act_attn: GemmAct::new(stream, rows, q_len)?,
            attn_o: f(rows * d.hidden)?,
            ffn_inp: f(rows * d.hidden)?,
            route: RouterOut::for_ubatch(stream, rows)?,
            dense: GemmRoute::new(stream, rows, 1)?,
            moe: GemmRoute::new(stream, slots, N_EXPERT)?,
            gate: f(slots * d.ff)?,
            up: f(slots * d.ff)?,
            act_h: GemmAct::new(stream, slots, d.ff)?,
            down: f(slots * d.hidden)?,
            dims: d,
            rows,
        })
    }

    /// Device bytes of the arena.
    fn bytes(&self) -> usize {
        let bufs = [
            &self.x,
            &self.normed,
            &self.q,
            &self.k,
            &self.v,
            &self.part_v,
            &self.part_ms,
            &self.attn,
            &self.attn_o,
            &self.ffn_inp,
            &self.gate,
            &self.up,
            &self.down,
        ];
        bufs.iter().map(|b| b.num_bytes()).sum::<usize>()
            + [&self.act_hid, &self.act_attn, &self.act_h]
                .iter()
                .map(|a| a.bytes())
                .sum::<usize>()
            + self.route.bytes()
            + self.dense.bytes()
            + self.moe.bytes()
    }
}

/// Element offsets into the prompt image for a capacity of `cap` tokens:
/// the ids, the positions and the live key counts (u32 each), then the rope
/// rows ([`HEAD`] f32 as bits per token).
fn img_off(cap: usize) -> (usize, usize, usize, usize) {
    (0, cap, 2 * cap, 3 * cap)
}

/// The prompt image: one array per input over the prompt's tokens.
struct UbImage {
    buf: DeviceBuffer<u32>,
    host: Vec<u32>,
    /// One position's rope row, reused for every token of a prompt.
    cs_host: Vec<f32>,
    /// Tokens the image has room for.
    cap: usize,
    /// Tokens of the prompt the image holds, and the position of its first.
    len: usize,
    pos0: u32,
}

/// One ubatch's (or one flash chunk's) windows onto the image.
struct UbIo {
    tokens: ManuallyDrop<DeviceBuffer<u32>>,
    pos: ManuallyDrop<DeviceBuffer<u32>>,
    n_keys: ManuallyDrop<DeviceBuffer<u32>>,
    cs: ManuallyDrop<DeviceBuffer<f32>>,
}

impl UbImage {
    fn new(stream: &CudaStream, cap: usize) -> Result<UbImage, GpuError> {
        Ok(UbImage {
            buf: DeviceBuffer::zeroed(stream, cap * (3 + HEAD))?,
            host: Vec::with_capacity(cap * (3 + HEAD)),
            cs_host: Vec::with_capacity(HEAD),
            cap,
            len: 0,
            pos0: 0,
        })
    }

    /// Write the image of `tokens` at positions `pos0 ..` and copy it to the
    /// card. Synchronizes (the copy of a borrowed host slice does).
    fn write(
        &mut self,
        stream: &CudaStream,
        rope: &RopeTable,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        let n = tokens.len();
        if n == 0 || n > self.cap {
            return Err(GpuError::shape(
                WHAT,
                format!("{n} tokens into an image of {}", self.cap),
            ));
        }
        let (o_tok, o_pos, o_nk, o_cs) = img_off(self.cap);
        self.host.clear();
        self.host.resize(self.cap * (3 + HEAD), 0);
        for (i, (&token, pos)) in tokens.iter().zip(pos0..).enumerate() {
            self.host[o_tok + i] = token;
            self.host[o_pos + i] = pos;
            self.host[o_nk + i] = pos + 1;
            self.cs_host.clear();
            rope.push(pos, Direction::Forward, &mut self.cs_host);
            for (dst, v) in self.host[o_cs + i * HEAD..][..HEAD]
                .iter_mut()
                .zip(&self.cs_host)
            {
                *dst = v.to_bits();
            }
        }
        self.buf.copy_from_host(stream, &self.host)?;
        self.len = n;
        self.pos0 = pos0;
        Ok(())
    }

    /// Windows onto tokens `s .. s + t` of the prompt in the image.
    fn io(&self, s: usize, t: usize) -> Result<UbIo, GpuError> {
        if t == 0 || s + t > self.len {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "tokens {s}..{} are not inside the {}-token prompt in the image",
                    s + t,
                    self.len
                ),
            ));
        }
        let (o_tok, o_pos, o_nk, o_cs) = img_off(self.cap);
        // SAFETY: s + t <= len <= cap, so each window lies inside its array
        // of the image (`img_off`), which lies inside `buf`; the windows live
        // for one ubatch, while the image stays in place (it is reallocated
        // only at load).
        unsafe {
            Ok(UbIo {
                tokens: param_view::<u32>(&self.buf, o_tok + s, t),
                pos: param_view::<u32>(&self.buf, o_pos + s, t),
                n_keys: param_view::<u32>(&self.buf, o_nk + s, t),
                cs: param_view::<f32>(&self.buf, o_cs + s * HEAD, t * HEAD),
            })
        }
    }
}

/// The GEMM prefill's resident state: the arena and the prompt image.
pub(super) struct Ubatch {
    a: UbArena,
    img: UbImage,
}

impl Ubatch {
    /// The arena for `min(UBATCH, ctx_max)` rows of `d` and an image for a
    /// prompt of `ctx_max` tokens. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims, ctx_max: usize) -> Result<Ubatch, GpuError> {
        Ok(Ubatch {
            a: UbArena::new(stream, d, UBATCH.min(ctx_max))?,
            img: UbImage::new(stream, ctx_max)?,
        })
    }

    /// Write the image of the tokens the ubatches of a prompt take, from
    /// position `pos0`. Synchronizes.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        rope: &RopeTable,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        self.img.write(stream, rope, tokens, pos0)
    }

    /// Device bytes of the arena and the image.
    pub(super) fn bytes(&self) -> usize {
        self.a.bytes() + self.img.buf.num_bytes()
    }

    /// Enqueue the ubatch of the image's tokens `s .. s + t`, standing at
    /// position `pos` (the image's position for token `s`, else refused):
    /// the embedding rows, the one-expert table, then every layer. The last
    /// layer's output rows stay in the arena's `x`.
    pub(super) fn enqueue(
        &mut self,
        c: &UbCtx<'_>,
        kv: &mut [KvPlanes],
        s: usize,
        t: usize,
        pos: u32,
    ) -> Result<(), GpuError> {
        if t > self.a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a ubatch of {t} tokens on a {}-row arena", self.a.rows),
            ));
        }
        let want = u32::try_from(s)
            .ok()
            .and_then(|s| self.img.pos0.checked_add(s));
        if want != Some(pos) {
            return Err(GpuError::state(
                WHAT,
                "the ubatch's first position is the image's position for its first token",
            ));
        }
        let io = self.img.io(s, t)?;
        let (gpu, a) = (c.gpu, &mut self.a);
        let stream = gpu.stream();
        gpu.elem().enqueue_embed_rows_q4k(
            stream,
            kq_weight(c.w, &token_embd())?,
            &io.tokens,
            &mut a.x,
        )?;
        c.k.gemm
            .enqueue_route_dense(stream, t, &mut a.dense, gpu.unlabelled_sink())?;
        for (slot, (n, kv)) in c.names.iter().zip(kv.iter_mut()).enumerate() {
            let sink = gpu.layer_sink(slot)?;
            attention(c, n, kv, a, &self.img, &io, s, t, pos as usize, sink)?;
            ffn(c, n, a, t, sink)?;
        }
        Ok(())
    }

    /// Row `t − 1` of the arena's residual: the last token's output after
    /// the last layer, the head's input.
    pub(super) fn last_row(&self, t: usize) -> Result<ManuallyDrop<DeviceBuffer<f32>>, GpuError> {
        let h = self.a.dims.hidden;
        if t == 0 || t > self.a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("row {t} of a {}-row arena", self.a.rows),
            ));
        }
        // SAFETY: row t − 1 < rows spans `hidden` values inside `x`
        // (rows · hidden), which stays in place while the window lives (one
        // copy).
        Ok(unsafe { f32_view(&self.a.x, (t - 1) * h, h) })
    }
}

/// What every launch of a ubatch reads besides its arena and the cache.
pub(super) struct UbCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) names: &'a [LayerNames],
    pub(super) k: &'a Kernels,
    pub(super) mma: bool,
    pub(super) eps: f32,
}

/// The flash's row chunks of a ubatch of `t` rows whose row `i` sees `p0 +
/// i + 1` keys: `(c0, c1, max_keys)` in row order, the widest chunks whose
/// row count times `segments_for(max_keys)` stays within `row_segs`. The
/// arena's `row_segs` covers one row at the cache's height, so every chunk
/// holds a row; one that did not fit would be refused by the flash's own
/// partials check.
struct FlashChunks {
    p0: usize,
    t: usize,
    row_segs: usize,
    c0: usize,
}

impl Iterator for FlashChunks {
    type Item = (usize, usize, usize);

    fn next(&mut self) -> Option<(usize, usize, usize)> {
        if self.c0 >= self.t {
            return None;
        }
        let fits = |c1: usize| (c1 - self.c0) * segments_for(self.p0 + c1) <= self.row_segs;
        // The cost grows with c1, so the widest chunk is the last c1 that fits.
        let (mut lo, mut hi) = (self.c0 + 1, self.t);
        if fits(hi) {
            lo = hi;
        }
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            if fits(mid) { lo = mid } else { hi = mid - 1 }
        }
        let c = (self.c0, lo, self.p0 + lo);
        self.c0 = lo;
        Some(c)
    }
}

/// The attention half at `t` rows: `x` in, `ffn_inp = x + attn_output(attn(x))`
/// out; the ubatch's K/V rows appended to the layer's planes.
#[allow(
    clippy::too_many_arguments,
    reason = "the chain's context, the layer, its arena, image and windows, the ubatch's shape"
)]
fn attention(
    c: &UbCtx<'_>,
    n: &LayerNames,
    kv: &mut KvPlanes,
    a: &mut UbArena,
    img: &UbImage,
    io: &UbIo,
    s: usize,
    t: usize,
    p0: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = a.dims;
    let (q_len, kv_len) = (d.n_head * HEAD, d.n_kv * HEAD);
    gpu.elem().enqueue_rms_norm(
        stream,
        &a.x,
        f32_gain(w, &n.attn_norm)?,
        c.eps,
        d.hidden,
        t,
        &mut a.normed,
    )?;
    gpu.enqueue_quantize_gemm(&a.normed, t, &mut a.act_hid, sink)?;
    for (name, ty, rows, y) in [
        (&n.attn_q, Kq::Q4K, q_len, &mut a.q),
        (&n.attn_k, Kq::Q4K, kv_len, &mut a.k),
        (&n.attn_v, n.v_ty, kv_len, &mut a.v),
    ] {
        k.gemm.enqueue_gemm(
            stream,
            gemm_ty(ty),
            kq_weight(w, name)?,
            rows,
            &a.act_hid,
            &a.dense,
            GemmInput::PerSlot,
            y,
        )?;
    }
    k.neox.enqueue_head_norm_neox_append(
        stream,
        NeoxArgs {
            q: &mut a.q,
            k: &mut a.k,
            v: &a.v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            cs: &io.cs,
            pos: &io.pos,
            eps: c.eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m: t,
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
    let chunks = FlashChunks {
        p0,
        t,
        row_segs: a.row_segs,
        c0: 0,
    };
    for (c0, c1, max_keys) in chunks {
        let m = c1 - c0;
        let rows = img.io(s + c0, m)?;
        // SAFETY: rows c0 .. c1 <= t <= rows of the arena lie inside `q` and
        // `attn` (rows · q_len each), which stay in place while the windows
        // live (this launch's enqueue).
        let (qw, mut yw) = unsafe {
            (
                f32_view(&a.q, c0 * q_len, m * q_len),
                f32_view(&a.attn, c0 * q_len, m * q_len),
            )
        };
        k.flash.enqueue_pass_upto(
            stream,
            GqaArgs {
                q: &qw,
                kc: &kv.k,
                vc: &kv.v,
                n_keys: &rows.n_keys,
                scale: 1.0 / (HEAD as f32).sqrt(),
                n_kv: d.n_kv,
                ctx: d.ctx,
                m,
                part_v: &mut a.part_v,
                part_ms: &mut a.part_ms,
                y: &mut yw,
            },
            c.mma,
            max_keys,
        )?;
    }
    gpu.enqueue_quantize_gemm(&a.attn, t, &mut a.act_attn, sink)?;
    k.gemm.enqueue_gemm(
        stream,
        GemmWeight::Q4K,
        kq_weight(w, &n.attn_output)?,
        d.hidden,
        &a.act_attn,
        &a.dense,
        GemmInput::PerSlot,
        &mut a.attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, &a.x, &a.attn_o, t * d.hidden, &mut a.ffn_inp)
}

/// The routed FFN half at `t` rows: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's eight experts out, into
/// `x`.
fn ffn(
    c: &UbCtx<'_>,
    n: &LayerNames,
    a: &mut UbArena,
    t: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = a.dims;
    let slots = t * N_USED;
    gpu.elem().enqueue_rms_norm(
        stream,
        &a.ffn_inp,
        f32_gain(w, &n.ffn_norm)?,
        c.eps,
        d.hidden,
        t,
        &mut a.normed,
    )?;
    gpu.enqueue_quantize_gemm(&a.normed, t, &mut a.act_hid, sink)?;
    k.router.enqueue_ubatch(
        stream,
        f32_tensor(w, &n.ffn_gate_inp)?,
        &a.normed,
        t,
        &mut a.route,
    )?;
    k.gemm
        .enqueue_route(stream, &a.route.ids, slots, &mut a.moe, sink)?;
    for (name, y) in [(&n.ffn_gate_exps, &mut a.gate), (&n.ffn_up_exps, &mut a.up)] {
        k.gemm.enqueue_gemm(
            stream,
            GemmWeight::Q4K,
            kq_weight(w, name)?,
            d.ff,
            &a.act_hid,
            &a.moe,
            GemmInput::Shared { top_k: N_USED },
            y,
        )?;
    }
    k.gemm
        .enqueue_swiglu_quant(stream, &a.gate, &a.up, slots, &mut a.act_h, sink)?;
    k.gemm.enqueue_gemm(
        stream,
        gemm_ty(n.down_ty),
        kq_weight(w, &n.ffn_down_exps)?,
        d.hidden,
        &a.act_h,
        &a.moe,
        GemmInput::PerSlot,
        &mut a.down,
    )?;
    k.experts.enqueue_combine_tokens(
        stream,
        CombineArgs {
            down: &a.down,
            w: &a.route.weights,
            resid: &a.ffn_inp,
            rows: d.hidden,
            n_slots: N_USED,
            m: t,
            y: &mut a.x,
        },
    )
}
