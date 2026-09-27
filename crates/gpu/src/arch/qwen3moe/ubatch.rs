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
//! Buffers. The arena holds `min(U, ctx)` rows of every intermediate, the
//! rows' positions and live key counts among them. The prompt image is one
//! input record with room for a prompt as long as the cache — the position
//! of the prompt's first token, then its ids — written into pinned host
//! words and copied to the card once per prompt, asynchronously, as far as
//! the prompt reaches. A ubatch reads a window of the ids; its embedding
//! launch writes its rows' positions (the image's position plus the
//! window's offset plus the row) and live key counts, and the rope turns
//! each row by the body's table at that position. Nothing is allocated per
//! prompt or per ubatch; a new `U` ([`Ubatch::resize`]) reallocates the
//! arena alone. The ubatches run eager in both step modes, so no captured
//! graph holds an arena address.
//!
//! Sizes. No kernel of a ubatch needs more of `U` than `1..=UBATCH`: the
//! route and the GEMM take any slot count up to [`GEMM_MAX_SLOTS`], the
//! router logits cut the tokens into blocks of 32 and the prefill flash into
//! groups of eight, each guarding the last, and every other launch works per
//! token or per slot.

use super::body::{ATTN_SCALE, Body, Kernels, Kq, LayerNames};
use super::experts::CombineArgs;
use super::router::{N_EXPERT, N_USED, RouterOut};
use super::scratch::{Dims, IN_IDS, IN_POS0, Inbox, KvPlanes, f32_view, param_view, put_input};
use crate::elem::EmbedRowsArgs;
use crate::flash_gqa::HEAD;
use crate::flash_gqa_prefill::{FlashGqaPrefill, GqaPrefillArgs};
use crate::gemm::{GEMM_MAX_SLOTS, GemmAct, GemmArgs, GemmInput, GemmRoute, GemmWeight};
use crate::model::GpuModel;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::NeoxArgs;
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
    /// Row `t`'s position and live key count, which the embedding launch
    /// writes: the rope turns by the table's row `pos[t]` and appends there,
    /// the prefill flash attends over `n_keys[t]` keys.
    pos: DeviceBuffer<u32>,
    n_keys: DeviceBuffer<u32>,
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
            pos: DeviceBuffer::zeroed(stream, rows)?,
            n_keys: DeviceBuffer::zeroed(stream, rows)?,
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
            + self.pos.num_bytes()
            + self.n_keys.num_bytes()
            + [&self.act_hid, &self.act_attn, &self.act_h]
                .iter()
                .map(|a| a.bytes())
                .sum::<usize>()
            + self.route.bytes()
            + self.dense.bytes()
            + self.moe.bytes()
    }
}

/// What the GEMM prefill spends on the host outside its launches
/// ([`GpuModel::ubatch_prologue`]).
#[derive(Clone, Copy, Debug)]
pub struct UbPrologue {
    /// Rows of the rope table every path reads — the cache's positions —
    /// and the host time that computed them at load: per row, the
    /// `RopeTable::push` of one position.
    pub table_rows: usize,
    pub table_build: Duration,
    /// The last prompt image written; `None` before a prompt took a ubatch.
    pub last: Option<ImageWrite>,
}

/// One prompt image's write: its tokens, the bytes copied to the card, the
/// host's fill of them and the enqueue of their copy, which does not wait
/// for it: the launches behind it do.
#[derive(Clone, Copy, Debug)]
pub struct ImageWrite {
    pub tokens: usize,
    pub bytes: usize,
    pub fill: Duration,
    pub copy: Duration,
}

/// The prompt image: one input record, the prompt's first position and its
/// ids.
struct UbImage {
    /// The record, with room for a prompt of `cap` tokens; a prompt of `n`
    /// tokens writes and copies its first `n + 1` words.
    inbox: Inbox,
    /// Tokens the image has room for, and positions the rope table holds:
    /// the cache's rows.
    cap: usize,
    /// Tokens of the prompt the image holds (0 while it holds none), and the
    /// position of its first.
    len: usize,
    pos0: u32,
    /// The last prompt's write.
    last: Option<ImageWrite>,
}

/// One ubatch's windows onto the image: its ids and the prompt's position
/// word, and the ids' offset in the prompt.
struct UbIo {
    ids: ManuallyDrop<DeviceBuffer<u32>>,
    pos0: ManuallyDrop<DeviceBuffer<u32>>,
    first: usize,
}

impl UbImage {
    /// The image for a cache of `cap` rows. Load-time only.
    fn new(stream: &CudaStream, cap: usize) -> Result<UbImage, GpuError> {
        Ok(UbImage {
            inbox: Inbox::new(stream, IN_IDS + cap)?,
            cap,
            len: 0,
            pos0: 0,
            last: None,
        })
    }

    /// Write the record of `tokens` at positions `pos0 ..` — its first
    /// `n + 1` words — and enqueue their copy to the card. Asynchronous: the
    /// ubatches behind it read it. A prompt whose positions pass the cache's
    /// rows, the rope table's too, is refused.
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
        put_input(self.inbox.host_mut()?, tokens, pos0)?;
        let t1 = Instant::now();
        self.inbox.upload(stream, IN_IDS + n)?;
        let t2 = Instant::now();
        self.len = n;
        self.pos0 = pos0;
        self.last = Some(ImageWrite {
            tokens: n,
            bytes: (IN_IDS + n) * size_of::<u32>(),
            fill: t1 - t0,
            copy: t2 - t1,
        });
        Ok(())
    }

    /// Windows onto the prompt's `tokens` in the image and its position
    /// word: row `t` of the ubatch is position `pos0 + tokens.start + t`.
    fn io(&self, tokens: Range<usize>) -> Result<UbIo, GpuError> {
        if tokens.is_empty() || tokens.end > self.len {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "tokens {tokens:?} are not inside the {}-token prompt in the image",
                    self.len
                ),
            ));
        }
        // SAFETY: tokens.end <= len <= cap, so the ids lie inside the
        // record's `IN_IDS + cap` words, and so does its position word; the
        // windows live for one ubatch, while the image stays in place (it is
        // allocated only at load).
        let (ids, pos0) = unsafe {
            (
                param_view::<u32>(self.inbox.dev(), IN_IDS + tokens.start, tokens.len()),
                param_view::<u32>(self.inbox.dev(), IN_POS0, 1),
            )
        };
        Ok(UbIo {
            ids,
            pos0,
            first: tokens.start,
        })
    }
}

/// The GEMM prefill's resident state: the arena, the prompt image and the
/// prefill flash.
pub(super) struct Ubatch {
    a: UbArena,
    img: UbImage,
    flash: FlashGqaPrefill,
    /// The ubatch size `U`; the arena holds `min(U, ctx)` rows.
    size: NonZeroUsize,
}

impl Ubatch {
    /// The arena for `min(u, d.ctx)` rows of `d`, an image for a prompt of
    /// `d.ctx` tokens, and the prefill flash's module; `u` in `1..=UBATCH`,
    /// else refused. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: Dims, u: usize) -> Result<Ubatch, GpuError> {
        let size = ubatch_of(u)?;
        Ok(Ubatch {
            a: UbArena::new(stream, d, u.min(d.ctx))?,
            img: UbImage::new(stream, d.ctx)?,
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
    /// position `pos0`, and enqueue its copy. Asynchronous.
    pub(super) fn write(
        &mut self,
        stream: &CudaStream,
        tokens: &[u32],
        pos0: u32,
    ) -> Result<(), GpuError> {
        self.img.write(stream, tokens, pos0)
    }

    /// Device bytes of the arena and the image.
    pub(super) fn bytes(&self) -> usize {
        self.a.bytes() + self.img.inbox.bytes()
    }

    /// Enqueue the ubatch of the image's `tokens`, standing at position `pos`
    /// (the image's position for its first token, else refused): the
    /// embedding rows with their positions and live key counts, the
    /// one-expert table, then every layer. The last layer's output rows stay
    /// in the arena's `x`.
    pub(super) fn enqueue(
        &mut self,
        c: &UbCtx<'_>,
        kv: &mut [KvPlanes],
        tokens: Range<usize>,
        pos: u32,
    ) -> Result<(), GpuError> {
        let t = tokens.len();
        if t > self.a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a ubatch of {t} tokens on a {}-row arena", self.a.rows),
            ));
        }
        let want = u32::try_from(tokens.start)
            .ok()
            .and_then(|s| self.img.pos0.checked_add(s));
        if want != Some(pos) {
            return Err(GpuError::state(
                WHAT,
                "the ubatch's first position is the image's position for its first token",
            ));
        }
        let io = self.img.io(tokens)?;
        let (gpu, a) = (c.gpu, &mut self.a);
        let stream = gpu.stream();
        gpu.elem().enqueue_embed_rows_q4k(
            stream,
            EmbedRowsArgs {
                w: kq_weight(c.w, &token_embd())?,
                ids: &io.ids,
                pos0: &io.pos0,
                first: io.first,
                y: &mut a.x,
                pos: &mut a.pos,
                n_keys: &mut a.n_keys,
            },
        )?;
        c.k.gemm
            .enqueue_route_dense(stream, t, &mut a.dense, gpu.unlabelled_sink())?;
        for (slot, (n, kv)) in c.names.iter().zip(kv.iter_mut()).enumerate() {
            let sink = gpu.layer_sink(slot)?;
            attention(c, n, kv, a, &self.flash, t, pos as usize, sink)?;
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
    /// The GEMM prefill's host prologue: the build at load of the rope table
    /// every path reads, and the last prompt image's write ([`UbPrologue`]).
    pub fn ubatch_prologue(&self) -> Result<UbPrologue, GpuError> {
        let body = self.body("qwen3moe::ubatch_prologue")?;
        Ok(UbPrologue {
            table_rows: body.rope.rows(),
            table_build: body.rope.build,
            last: body.ub.img.last,
        })
    }
}

/// What every launch of a ubatch reads besides its arena and the cache.
pub(super) struct UbCtx<'a> {
    pub(super) gpu: &'a Gpu,
    pub(super) w: &'a Weights,
    pub(super) names: &'a [LayerNames],
    pub(super) k: &'a Kernels,
    pub(super) eps: f32,
    /// The rope table the rope launches read by position.
    pub(super) table: &'a DeviceBuffer<f32>,
}

/// The attention half at `t` rows: `x` in, `ffn_inp = x + attn_output(attn(x))`
/// out; the ubatch's K/V rows appended to the layer's planes at the rows'
/// positions.
#[allow(
    clippy::too_many_arguments,
    reason = "the chain's context, the layer, its arena, the flash, the ubatch's shape"
)]
fn attention(
    c: &UbCtx<'_>,
    n: &LayerNames,
    kv: &mut KvPlanes,
    a: &mut UbArena,
    flash: &FlashGqaPrefill,
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
            GemmArgs {
                ty: gemm_ty(ty),
                w: kq_weight(w, name)?,
                rows_per_expert: rows,
                act: &a.act_hid,
                route: &a.dense,
                input: GemmInput::PerSlot,
                y,
            },
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
            table: c.table,
            pos: &a.pos,
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
            n_keys: &a.n_keys,
            scale: ATTN_SCALE,
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
        GemmArgs {
            ty: GemmWeight::Q4K,
            w: kq_weight(w, &n.attn_output)?,
            rows_per_expert: d.hidden,
            act: &a.act_attn,
            route: &a.dense,
            input: GemmInput::PerSlot,
            y: &mut a.attn_o,
        },
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
            GemmArgs {
                ty: GemmWeight::Q4K,
                w: kq_weight(w, name)?,
                rows_per_expert: d.ff,
                act: &a.act_hid,
                route: &a.moe,
                input: GemmInput::Shared { top_k: N_USED },
                y,
            },
        )?;
    }
    k.gemm
        .enqueue_swiglu_quant(stream, &a.gate, &a.up, slots, &mut a.act_h, sink)?;
    k.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty: gemm_ty(n.down_ty),
            w: kq_weight(w, &n.ffn_down_exps)?,
            rows_per_expert: d.hidden,
            act: &a.act_h,
            route: &a.moe,
            input: GemmInput::PerSlot,
            y: &mut a.down,
        },
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
