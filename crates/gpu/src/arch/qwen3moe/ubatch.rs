//! qwen3moe's GEMM prefill: a prompt fed in ubatches of up to `U` tokens,
//! every weight read once per ubatch, `U` the model's ubatch size — set at
//! load ([`ubatch_size`]: `BLOOMERY_QWEN3_UBATCH`, else [`UBATCH`]), at most
//! [`UBATCH`]. The dense projections run through the grouped int8 GEMM over
//! a one-expert table built once per ubatch; the routed experts through one
//! route table per layer, gate and up reading each token's column
//! (`GemmInput::Shared`), down each slot's own. The glue between them is the
//! chain's own kernels at `T` rows: the embedding rows, `rms_norm` then the
//! q8_1 quantizer (the bytes `norm_quant` writes), the per-head norm with
//! the rope and the cache append, the prefill flash (`flash_gqa_prefill`: each row over its own
//! live key count, the cache's key tiles staged once per block), the
//! residual add, the router in two launches, SwiGLU with the down's
//! quantizer in one launch (`GemmKernels::enqueue_swiglu_quant`), and the
//! combine of the down rows with the router weights and the residual.
//!
//! Numeric class. Every launch computes a token's values from that token's
//! inputs alone, so a token's bits do not depend on the ubatch it lands in or
//! the tokens beside it — the GEMM accumulates each (slot, row) on its own,
//! the norm, quantizer, router and combine work per column, and the flash
//! walks a row's keys in fixed 64-key tiles whatever the grid or the rows
//! beside it. Against the one-token path the dense and expert products sum
//! in another order (128-value integer blocks into one f32 accumulator, where
//! the gemv sums lane partials through a warp tree) and the attention weighs
//! the values with f16 weights on the tensor cores, so the two paths agree to
//! the error of those sums and roundings and are not bit-equal.
//!
//! Buffers. The arena holds `min(U, ctx)` rows of every intermediate. The
//! prompt image has room for a prompt as long as the cache: its token ids,
//! positions and live key counts, three arrays back to back over the
//! prompt's tokens, written and copied to the card once per prompt as far as
//! the prompt reaches. The rope table holds every cache position's rope row
//! (`RopeTable::push`), made and copied once at load; a position's row does
//! not depend on the prompt. A ubatch reads windows of the image and of the
//! table. Nothing is allocated per prompt or per ubatch; a new `U`
//! ([`Ubatch::resize`]) reallocates the arena alone. The ubatches run eager
//! in both step modes, so no captured graph holds an arena address.
//!
//! Sizes. No kernel of a ubatch needs more of `U` than `1..=UBATCH`: the
//! route and the GEMM take any slot count up to [`GEMM_MAX_SLOTS`], the
//! router logits cut the tokens into blocks of 32 and the prefill flash into
//! groups of eight, each guarding the last, and every other launch works per
//! token or per slot.

use super::body::{Body, Kernels, Kq, LayerNames};
use super::experts::CombineArgs;
use super::router::{N_EXPERT, N_USED, RouterOut};
use super::scratch::{Dims, KvPlanes, f32_view, param_view};
use crate::flash_gqa::HEAD;
use crate::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs};
use crate::gemm::{GEMM_MAX_SLOTS, GemmAct, GemmInput, GemmRoute, GemmWeight};
use crate::model::GpuModel;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::NeoxArgs;
use crate::rope_table::{Direction, RopeTable};
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use model::arch::qwen3moe::names::token_embd;
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ops::Range;
use std::time::{Duration, Instant};

/// The most tokens one ubatch takes: the grouped GEMM's slot cap at top-8.
/// Also the default ubatch size: the one that reads each weight the fewest
/// times per prompt and gives each expert's GEMM the most columns.
pub const UBATCH: usize = GEMM_MAX_SLOTS / N_USED;

/// The environment variable a load reads the ubatch size from.
pub const UBATCH_ENV: &str = "BLOOMERY_QWEN3_UBATCH";

const WHAT: &str = "qwen3moe::ubatch";

/// `u` as a ubatch size, or an error unless it is in `1..=UBATCH`.
fn ubatch_of(u: usize) -> Result<NonZeroUsize, GpuError> {
    NonZeroUsize::new(u)
        .filter(|u| u.get() <= UBATCH)
        .ok_or_else(|| GpuError::shape(WHAT, format!("a ubatch of {u} tokens (1..={UBATCH})")))
}

/// The ubatch size a load takes: [`UBATCH_ENV`] if set, else [`UBATCH`].
/// Read once per process; a value that is set but not a size in
/// `1..=UBATCH` is an error that names it, at every load.
pub fn ubatch_size() -> Result<usize, GpuError> {
    static SIZE: std::sync::OnceLock<Result<usize, String>> = std::sync::OnceLock::new();
    SIZE.get_or_init(|| match std::env::var(UBATCH_ENV) {
        Err(std::env::VarError::NotPresent) => Ok(UBATCH),
        Err(e) => Err(format!("{UBATCH_ENV}: {e}")),
        Ok(v) => match v.parse::<usize>() {
            Ok(u) if (1..=UBATCH).contains(&u) => Ok(u),
            _ => Err(format!(
                "{UBATCH_ENV}={v} is not a ubatch size in 1..={UBATCH}"
            )),
        },
    })
    .clone()
    .map_err(|e| GpuError::shape(WHAT, e))
}

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
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(UbArena {
            x: f(rows * d.hidden)?,
            normed: f(rows * d.hidden)?,
            act_hid: GemmAct::new(stream, rows, d.hidden)?,
            q: f(rows * q_len)?,
            k: f(rows * kv_len)?,
            v: f(rows * kv_len)?,
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

/// Element offsets into the prompt image for a prompt of `n` tokens: its
/// ids, its positions and its live key counts (u32 each), back to back.
fn img_off(n: usize) -> (usize, usize, usize) {
    (0, n, 2 * n)
}

/// What the GEMM prefill spends on the host outside its launches
/// ([`GpuModel::ubatch_prologue`]).
#[derive(Clone, Copy, Debug)]
pub struct UbPrologue {
    /// Rows of the rope table — the cache's positions — and the host time
    /// that computed them at load: per row, the `RopeTable::push` of one
    /// position.
    pub table_rows: usize,
    pub table_build: Duration,
    /// The last prompt image written; `None` before a prompt took a ubatch.
    pub last: Option<ImageWrite>,
}

/// One prompt image's write: its tokens, the bytes copied to the card, the
/// host's fill of them and the copy, which synchronizes the stream.
#[derive(Clone, Copy, Debug)]
pub struct ImageWrite {
    pub tokens: usize,
    pub bytes: usize,
    pub fill: Duration,
    pub copy: Duration,
}

/// The prompt image and the rope table.
struct UbImage {
    /// The three arrays of the prompt the image holds, at `img_off(len)`;
    /// room for a prompt of `cap` tokens.
    buf: DeviceBuffer<u32>,
    /// `buf`'s host side at its full length, filled at load (the value is
    /// never read) so that a prompt's writes fault no page; a prompt of `n`
    /// tokens writes and copies its first `3n` words.
    host: Vec<u32>,
    /// Position `p`'s rope row, [`HEAD`] f32 at `p · HEAD`, for every `p <
    /// cap`: `RopeTable::push`'s bits, written once at load.
    rope: DeviceBuffer<f32>,
    /// Positions the image and the table have room for: the cache's rows.
    cap: usize,
    /// Tokens of the prompt the image holds (0 while it holds none), and the
    /// position of its first.
    len: usize,
    pos0: u32,
    /// The table's build at load and the last prompt's write.
    stat: UbPrologue,
}

/// One ubatch's windows onto the image and the table.
struct UbIo {
    tokens: ManuallyDrop<DeviceBuffer<u32>>,
    pos: ManuallyDrop<DeviceBuffer<u32>>,
    n_keys: ManuallyDrop<DeviceBuffer<u32>>,
    cs: ManuallyDrop<DeviceBuffer<f32>>,
}

impl UbImage {
    /// The image and the rope table for a cache of `cap` rows, the table's
    /// rows `rope`'s. Load-time only.
    fn new(stream: &CudaStream, cap: usize, rope: &RopeTable) -> Result<UbImage, GpuError> {
        let positions = u32::try_from(cap).map_err(|_| {
            GpuError::shape(
                WHAT,
                format!("a cache of {cap} rows: positions and live key counts are u32"),
            )
        })?;
        if rope.n_dims() != HEAD {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a rope table of {} values a position; a row is {HEAD}",
                    rope.n_dims()
                ),
            ));
        }
        let t0 = Instant::now();
        let mut table = Vec::with_capacity(cap * HEAD);
        for pos in 0..positions {
            rope.push(pos, Direction::Forward, &mut table);
        }
        let table_build = t0.elapsed();
        Ok(UbImage {
            buf: DeviceBuffer::zeroed(stream, 3 * cap)?,
            host: vec![u32::MAX; 3 * cap],
            rope: DeviceBuffer::from_host(stream, &table)?,
            cap,
            len: 0,
            pos0: 0,
            stat: UbPrologue {
                table_rows: cap,
                table_build,
                last: None,
            },
        })
    }

    /// Write the ids, positions and live key counts of `tokens` at positions
    /// `pos0 ..` into the image's first `3n` words and copy those to the
    /// card; the rope rows are the table's. Synchronizes (the copy of a
    /// borrowed host slice does). A prompt whose positions pass the table is
    /// refused: this check keeps [`UbImage::io`]'s table window inside it.
    fn write(&mut self, stream: &CudaStream, tokens: &[u32], pos0: u32) -> Result<(), GpuError> {
        self.len = 0;
        let n = tokens.len();
        let end = pos0 as usize + n;
        if n == 0 || end > self.cap {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{n} tokens at positions {pos0}..{end}: the image and the rope table hold \
                     positions 0..{}",
                    self.cap
                ),
            ));
        }
        let t0 = Instant::now();
        let (o_tok, o_pos, o_nk) = img_off(n);
        let words = &mut self.host[..3 * n];
        words[o_tok..o_pos].copy_from_slice(tokens);
        for (i, pos) in (pos0..).take(n).enumerate() {
            words[o_pos + i] = pos;
            words[o_nk + i] = pos + 1;
        }
        let t1 = Instant::now();
        // SAFETY: 3n <= 3·cap = buf.len() (`end <= cap` above); the window
        // lives for this copy alone.
        let mut dst = unsafe { param_view::<u32>(&self.buf, 0, 3 * n) };
        dst.copy_from_host(stream, words)?;
        let t2 = Instant::now();
        self.len = n;
        self.pos0 = pos0;
        self.stat.last = Some(ImageWrite {
            tokens: n,
            bytes: 3 * n * size_of::<u32>(),
            fill: t1 - t0,
            copy: t2 - t1,
        });
        Ok(())
    }

    /// Windows onto tokens `s .. s + t` of the prompt in the image, and onto
    /// their rope rows in the table: rows `pos0 + s ..`.
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
        let (o_tok, o_pos, o_nk) = img_off(self.len);
        // SAFETY: s + t <= len, so each window lies inside its array of the
        // prompt (`img_off(len)`), and 3·len <= 3·cap = buf.len(); the
        // windows live for one ubatch, while the image stays in place (it is
        // allocated only at load).
        let (tokens, pos, n_keys) = unsafe {
            (
                param_view::<u32>(&self.buf, o_tok + s, t),
                param_view::<u32>(&self.buf, o_pos + s, t),
                param_view::<u32>(&self.buf, o_nk + s, t),
            )
        };
        let row = self.pos0 as usize + s;
        // SAFETY: rows `row .. row + t` end at or below pos0 + len <= cap
        // (`write` refused a prompt past the table), inside the table's cap
        // rows of HEAD values; the table stays in place for the model's life.
        let cs = unsafe { f32_view(&self.rope, row * HEAD, t * HEAD) };
        Ok(UbIo {
            tokens,
            pos,
            n_keys,
            cs,
        })
    }

    /// Rows `positions` of the rope table, copied to the host. Synchronizes.
    fn rope_rows(
        &self,
        stream: &CudaStream,
        positions: Range<usize>,
    ) -> Result<Vec<f32>, GpuError> {
        if positions.is_empty() || positions.end > self.cap {
            return Err(GpuError::shape(
                WHAT,
                format!("rope rows {positions:?} of a {}-row table", self.cap),
            ));
        }
        // SAFETY: rows `positions` end at or below cap, inside the table; the
        // window lives for this copy alone.
        let rows = unsafe { f32_view(&self.rope, positions.start * HEAD, positions.len() * HEAD) };
        Ok(rows.to_host_vec(stream)?)
    }
}

/// The GEMM prefill's resident state: the arena, the prompt image with the
/// rope table, and the prefill flash.
pub(super) struct Ubatch {
    a: UbArena,
    img: UbImage,
    flash: FlashGqaPrefill,
    /// The ubatch size `U`; the arena holds `min(U, ctx)` rows.
    size: NonZeroUsize,
}

impl Ubatch {
    /// The arena for `min(u, ctx_max)` rows of `d`, an image for a prompt
    /// of `ctx_max` tokens, the rope table of the `ctx_max` positions by
    /// `rope`, and the prefill flash's module; `u` in `1..=UBATCH`, else
    /// refused. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        d: Dims,
        ctx_max: usize,
        u: usize,
        rope: &RopeTable,
    ) -> Result<Ubatch, GpuError> {
        let size = ubatch_of(u)?;
        Ok(Ubatch {
            a: UbArena::new(stream, d, u.min(ctx_max))?,
            img: UbImage::new(stream, ctx_max, rope)?,
            flash: FlashGqaPrefill::load(stream.context())?,
            size,
        })
    }

    /// The ubatch size.
    pub(super) fn size(&self) -> NonZeroUsize {
        self.size
    }

    /// Reallocate the arena for ubatches of up to `u` (`1..=UBATCH`, else
    /// refused) tokens. The new arena is allocated before the old one is
    /// freed, so a refusal or a failed allocation leaves the old size in
    /// place. Load-time allocation.
    pub(super) fn resize(&mut self, stream: &CudaStream, u: usize) -> Result<(), GpuError> {
        let size = ubatch_of(u)?;
        let d = self.a.dims;
        self.a = UbArena::new(stream, d, u.min(d.ctx))?;
        self.size = size;
        Ok(())
    }

    /// Write the image of the tokens the ubatches of a prompt take, from
    /// position `pos0`. Synchronizes.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        self.img.write(stream, tokens, pos0)
    }

    /// Device bytes of the arena, the image and the rope table.
    pub(super) fn bytes(&self) -> usize {
        self.a.bytes() + self.img.buf.num_bytes() + self.img.rope.num_bytes()
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
            attention(c, n, kv, a, &self.flash, &io, t, pos as usize, sink)?;
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

impl GpuModel<Body> {
    /// The GEMM prefill's host prologue: the rope table's build at load and
    /// the last prompt image's write ([`UbPrologue`]).
    pub fn ubatch_prologue(&self) -> Result<UbPrologue, GpuError> {
        Ok(self.body("qwen3moe::ubatch_prologue")?.ub.img.stat)
    }

    /// Rows `positions` of the rope table the ubatches read, [`HEAD`] f32 a
    /// row. Synchronizes; gate use.
    pub fn rope_rows(&self, positions: Range<usize>) -> Result<Vec<f32>, GpuError> {
        let img = &self.body("qwen3moe::rope_rows")?.ub.img;
        img.rope_rows(self.stage_stream()?, positions)
    }

    /// The rope rows the ubatch of the image's tokens `first .. first +
    /// tokens` reads — the window `Ubatch::enqueue` hands the rope kernel —
    /// [`HEAD`] f32 a token. Synchronizes; gate use.
    pub fn ubatch_rope_window(&self, first: usize, tokens: usize) -> Result<Vec<f32>, GpuError> {
        let io = self
            .body("qwen3moe::ubatch_rope_window")?
            .ub
            .img
            .io(first, tokens)?;
        Ok(io.cs.to_host_vec(self.stage_stream()?)?)
    }
}

/// What every launch of a ubatch reads besides its arena and the cache.
pub(super) struct UbCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) names: &'a [LayerNames],
    pub(super) k: &'a Kernels,
    pub(super) eps: f32,
}

/// The attention half at `t` rows: `x` in, `ffn_inp = x + attn_output(attn(x))`
/// out; the ubatch's K/V rows appended to the layer's planes.
#[allow(
    clippy::too_many_arguments,
    reason = "the chain's context, the layer, its arena, the flash, the windows, the ubatch's shape"
)]
fn attention(
    c: &UbCtx<'_>,
    n: &LayerNames,
    kv: &mut KvPlanes,
    a: &mut UbArena,
    flash: &FlashGqaPrefill,
    io: &UbIo,
    t: usize,
    p0: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = a.dims;
    if p0 + t > d.ctx {
        return Err(GpuError::shape(
            WHAT,
            format!("rows {p0}..{} past the {}-row cache", p0 + t, d.ctx),
        ));
    }
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
            fault: sink,
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
    flash.enqueue(
        stream,
        GqaPrefillArgs {
            q: &a.q,
            kc: &kv.k,
            vc: &kv.v,
            n_keys: &io.n_keys,
            scale: 1.0 / (HEAD as f32).sqrt(),
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            t,
            fault: sink,
            y: &mut a.attn,
        },
    )?;
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
        sink,
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
