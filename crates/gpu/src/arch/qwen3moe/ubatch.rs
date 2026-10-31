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
//! combine of the down rows with the router weights and the residual. The
//! attention half is the wide arm's head-128 half ([`wide::neox_128_rows`])
//! over this arena's buffers.
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
//!
//! A placed load holds the wide arm's arena (`scratch::Arena`) at the placed
//! units' rows instead ([`Ubatch::placed`]): its units run through the host
//! tier's batch walk (`placed::BatchWalk`) over that arena, reading the same
//! prompt image.

use super::body::{Body, Kernels};
use super::dispatch::embed_into;
use super::experts::CombineArgs;
pub use super::image::ImageWrite;
use super::image::{ImageWindows, PromptImage};
use super::plan::{FfnPlan, FfnRoute, GqaPlan, LayerPlan, MixerPlan, SiteTy};
use super::scratch::{Arena, Dims, KvPlanes, cells_view};
use super::wide::{self, Neox128Rows};
use crate::flash_gqa::HEAD;
use crate::gemm::GemmAct32;
use crate::gemm::{GEMM_MAX_SLOTS, GemmAct, GemmArgs, GemmInput, GemmRoute, GemmWeight};
use crate::model::GpuModel;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::route_softmax::RouterOut;
use crate::site::{Form, WideActsMut};
use crate::tensor::WindowMut;
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::{CudaStream, DeviceBuffer};
use std::num::NonZeroUsize;
use std::ops::Range;
use std::time::Duration;

pub use chain::arch::qwen3moe::router::UBATCH;

/// The environment variable a load reads the ubatch size from.
pub const UBATCH_ENV: &str = "BLOOMERY_QWEN3_UBATCH";

const WHAT: &str = "qwen3moe::ubatch";

/// `u` as a ubatch size, or an error unless it is in `1..=UBATCH`.
fn ubatch_of(u: usize) -> Result<NonZeroUsize, GpuError> {
    NonZeroUsize::new(u)
        .filter(|u| u.get() <= UBATCH)
        .ok_or_else(|| GpuError::shape(WHAT, format!("a ubatch of {u} tokens (1..={UBATCH})")))
}

/// The ubatch size [`UBATCH_ENV`] asks, `None` when it is unset. Read once
/// per process; a value that is set but not a size in `1..=UBATCH` is an
/// error that names it, at every load.
pub fn ubatch_asked() -> Result<Option<usize>, GpuError> {
    static ASKED: std::sync::OnceLock<Result<Option<usize>, String>> = std::sync::OnceLock::new();
    ASKED
        .get_or_init(|| match std::env::var(UBATCH_ENV) {
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(e) => Err(format!("{UBATCH_ENV}: {e}")),
            Ok(v) => match v.parse::<usize>() {
                Ok(u) if (1..=UBATCH).contains(&u) => Ok(Some(u)),
                _ => Err(format!(
                    "{UBATCH_ENV}={v} is not a ubatch size in 1..={UBATCH}"
                )),
            },
        })
        .clone()
        .map_err(|e| GpuError::shape(WHAT, e))
}

/// The ubatch size a load takes: [`UBATCH_ENV`] if set ([`ubatch_asked`]),
/// else [`UBATCH`].
pub fn ubatch_size() -> Result<usize, GpuError> {
    Ok(ubatch_asked()?.unwrap_or(UBATCH))
}

/// The ubatch size a load of a cache of `ctx` positions runs: [`ubatch_size`]
/// clipped to the cache. A qwen4exp caller reads it once and hands the one
/// value to the plan's machine (`place::machine`) and to the load
/// (`Body38::open_placed`).
pub fn ubatch_for(ctx: usize) -> Result<usize, GpuError> {
    Ok(ubatch_size()?.min(ctx))
}

// The default ubatch at nine slots a token (eight routed experts and the
// shared one) fits one GEMM route.
const _: () = assert!(UBATCH * 9 <= GEMM_MAX_SLOTS);

/// The GEMM's type for a qwen3moe projection, a K-quant (the qwen3moe load
/// admits no other); another type is refused by name.
fn gemm_ty(ty: SiteTy) -> Result<GemmWeight, GpuError> {
    crate::site::gemm_weight(ty).ok_or_else(|| {
        GpuError::shape(
            "qwen3moe::ubatch",
            format!("a {ty} projection; the qwen3moe ubatch runs K-quants"),
        )
    })
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
    /// q8 blocks of 32 of `normed` for the q·k·v of a Q8_0 site; `None` when
    /// no site of the chain reads that form ([`Q32`]).
    act_hid32: Option<GemmAct32>,
    q: DeviceBuffer<f32>,
    k: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    /// The attention rows, `n_head · HEAD` per token.
    attn: DeviceBuffer<f32>,
    act_attn: GemmAct,
    /// q8 blocks of 32 of `attn` for a Q8_0 output projection ([`Q32`]).
    act_attn32: Option<GemmAct32>,
    /// The output projection's rows; the residual add reads them.
    attn_o: DeviceBuffer<f32>,
    /// The FFN's input residual: `x` plus the attention output.
    ffn_inp: DeviceBuffer<f32>,
    /// The router's outputs, slot `t · k + j` token `t`'s slot `j`.
    route: RouterOut,
    /// The one-expert table every dense projection of a ubatch reads.
    dense: GemmRoute,
    /// The layer's expert table over `t · k` slots.
    moe: GemmRoute,
    /// Gate and up rows, per slot `ff` values.
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    /// q8_1 of their SwiGLU, one column per slot: the down's input.
    act_h: GemmAct,
    /// The down rows, per slot `hidden` values.
    down: DeviceBuffer<f32>,
}

/// Whether any attention site of the chain reads q8 blocks of 32 values (a
/// Q8_0 site) from the normed rows and from the attention rows: the
/// ubatch holds those forms only then.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Q32 {
    hid: bool,
    attn: bool,
}

impl Q32 {
    /// What the attention sites of `plans` read.
    pub(super) fn of(plans: &[LayerPlan]) -> Q32 {
        let reads = |tys: &[SiteTy]| tys.iter().any(|t| t.reads() == Form::Q8x32);
        let mut q = Q32::default();
        for p in plans {
            if let MixerPlan::Gqa(g) = &p.mixer {
                q.hid |= reads(&[g.q_ty, g.k_ty, g.v_ty]);
                q.attn |= reads(&[g.o_ty]);
            }
        }
        q
    }
}

impl UbArena {
    /// The arena for `d` and up to `rows` tokens, holding the q8 blocks of 32
    /// `q32` names. Load-time only.
    fn new(stream: &CudaStream, d: Dims, rows: usize, q32: Q32) -> Result<UbArena, GpuError> {
        let q_len = d.n_head * HEAD;
        let kv_len = d.n_kv * HEAD;
        let slots = rows * d.slots();
        if slots > GEMM_MAX_SLOTS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "a ubatch of {rows} tokens at {} slots a token is {slots} slots; the grouped \
                     GEMM's route table holds {GEMM_MAX_SLOTS}",
                    d.slots()
                ),
            ));
        }
        let r = d.routed(WHAT)?;
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(UbArena {
            x: f(rows * d.hidden)?,
            pos: DeviceBuffer::zeroed(stream, rows)?,
            n_keys: DeviceBuffer::zeroed(stream, rows)?,
            normed: f(rows * d.hidden)?,
            act_hid: GemmAct::new(stream, rows, d.hidden)?,
            act_hid32: q32
                .hid
                .then(|| GemmAct32::new(stream, rows, d.hidden))
                .transpose()?,
            q: f(rows * q_len)?,
            k: f(rows * kv_len)?,
            v: f(rows * kv_len)?,
            attn: f(rows * q_len)?,
            act_attn: GemmAct::new(stream, rows, q_len)?,
            act_attn32: q32
                .attn
                .then(|| GemmAct32::new(stream, rows, q_len))
                .transpose()?,
            attn_o: f(rows * d.hidden)?,
            ffn_inp: f(rows * d.hidden)?,
            route: RouterOut::for_ubatch(stream, r, rows)?,
            dense: GemmRoute::new(stream, rows, 1)?,
            moe: GemmRoute::new(stream, slots, r.experts())?,
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
            + [&self.act_hid32, &self.act_attn32]
                .into_iter()
                .flatten()
                .map(GemmAct32::bytes)
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

/// Err unless `pos` is the image's position for the first of `tokens`.
fn check_start(img: &PromptImage, tokens: &Range<usize>, pos: u32) -> Result<(), GpuError> {
    let want = u32::try_from(tokens.start)
        .ok()
        .and_then(|s| img.pos0.checked_add(s));
    if want != Some(pos) {
        return Err(GpuError::state(
            WHAT,
            "the ubatch's first position is the image's position for its first token",
        ));
    }
    Ok(())
}

/// The arena of the GEMM prefill: the whole-card load's own, or a placed
/// load's units' (the wide arm's, `wide`).
enum UbRows {
    /// The whole-card GEMM prefill's arena.
    Whole(Box<UbArena>),
    /// A placed load's arena of the placed units' rows, which `placed`'s
    /// batch walk runs over.
    Placed(Box<Arena>),
}

/// The GEMM prefill's resident state: the arena and the prompt image (the
/// prefill flash is the chain's, [`Kernels`]).
pub(super) struct Ubatch {
    a: UbRows,
    img: PromptImage,
    /// The ubatch size `U`; a whole-card arena holds `min(U, ctx)` rows.
    size: NonZeroUsize,
    /// The q8 blocks of 32 the arena holds ([`Q32`]), kept for a resize.
    q32: Q32,
}

impl Ubatch {
    /// The arena for `min(u, d.ctx)` rows of `d` (with the q8 blocks of 32
    /// `q32` names) and an image for a prompt of `d.ctx` tokens; `u` in
    /// `1..=UBATCH`, else refused. Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        d: Dims,
        u: usize,
        q32: Q32,
    ) -> Result<Ubatch, GpuError> {
        let size = ubatch_of(u)?;
        Ok(Ubatch {
            a: UbRows::Whole(Box::new(UbArena::new(stream, d, u.min(d.ctx), q32)?)),
            img: PromptImage::new(stream, d.ctx, false)?,
            size,
            q32,
        })
    }

    /// A placed load's state: the wide arena `arena` its walks run over (the
    /// plan's count held, `placed::counted_arena`), ubatches of `u` tokens
    /// (`1..=UBATCH`, else refused) and an image for a prompt as long as the
    /// cache. Load-time only.
    pub(super) fn placed(stream: &CudaStream, arena: Arena, u: usize) -> Result<Ubatch, GpuError> {
        let size = ubatch_of(u)?;
        let ctx = arena.dims.ctx;
        Ok(Ubatch {
            a: UbRows::Placed(Box::new(arena)),
            img: PromptImage::new(stream, ctx, false)?,
            size,
            q32: Q32::default(),
        })
    }

    /// The ubatch size.
    pub(super) fn size(&self) -> NonZeroUsize {
        self.size
    }

    /// Reallocate the arena for ubatches of up to `u` (`1..=UBATCH`, else
    /// refused) tokens. The new arena is allocated before the old one is
    /// freed, so a refusal or a failed allocation leaves the old size in
    /// place. Load-time allocation. A placed load's arena is the plan's
    /// count and is refused by name.
    pub(super) fn resize(&mut self, stream: &CudaStream, u: usize) -> Result<(), GpuError> {
        let size = ubatch_of(u)?;
        let UbRows::Whole(a) = &self.a else {
            return Err(GpuError::state(
                WHAT,
                "a whole-card load (a placed load's arena is the plan's count)",
            ));
        };
        let d = a.dims;
        self.a = UbRows::Whole(Box::new(UbArena::new(stream, d, u.min(d.ctx), self.q32)?));
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
        let arena = match &self.a {
            UbRows::Whole(a) => a.bytes(),
            UbRows::Placed(a) => a.bytes(),
        };
        arena + self.img.bytes()
    }

    /// Enqueue the ubatch of the image's `tokens`, standing at position `pos`
    /// (the image's position for its first token, else refused): the
    /// embedding rows with their positions and live key counts, the
    /// one-expert table, then every layer. The last layer's output rows stay
    /// in the arena's `x`. A placed load's units run through its batch walk
    /// ([`Ubatch::placed_unit`]); this is refused by name there.
    pub(super) fn enqueue(
        &mut self,
        c: &UbCtx<'_>,
        kv: &mut [KvPlanes],
        tokens: Range<usize>,
        pos: u32,
    ) -> Result<(), GpuError> {
        let UbRows::Whole(a) = &mut self.a else {
            return Err(GpuError::state(
                WHAT,
                "a whole-card load (a placed load's units run through its batch walk)",
            ));
        };
        let t = tokens.len();
        if t > a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a ubatch of {t} tokens on a {}-row arena", a.rows),
            ));
        }
        check_start(&self.img, &tokens, pos)?;
        let win = self.img.windows(tokens)?;
        let io = win.io();
        let gpu = c.gpu;
        let stream = gpu.stream();
        embed_into(gpu, c.w, c.k, &io, (&mut a.x, &mut a.pos, &mut a.n_keys))?;
        c.k.gemm
            .enqueue_route_dense(stream, t, &mut a.dense, gpu.unlabelled_sink())?;
        for (slot, (p, kv)) in c.plans.iter().zip(kv.iter_mut()).enumerate() {
            let sink = gpu.layer_sink(slot)?;
            attention(c, p.gqa(WHAT, slot)?, kv, a, t, pos as usize, sink)?;
            ffn(c, &p.ffn, a, t, sink)?;
        }
        Ok(())
    }

    /// A placed load's unit of the image's `tokens`, standing at position
    /// `pos`: the arena the batch walk runs over and the unit's windows onto
    /// the image. The unit's first position is the image's position for its
    /// first token, else refused; a unit past the arena's rows, and a
    /// whole-card load, are refused by name.
    pub(super) fn placed_unit(
        &mut self,
        tokens: Range<usize>,
        pos: u32,
    ) -> Result<(&mut Arena, ImageWindows), GpuError> {
        let UbRows::Placed(a) = &mut self.a else {
            return Err(GpuError::state(WHAT, "a placed load's arena"));
        };
        let t = tokens.len();
        if t > a.rows {
            return Err(GpuError::shape(
                WHAT,
                format!("a unit of {t} tokens on a {}-row arena", a.rows),
            ));
        }
        check_start(&self.img, &tokens, pos)?;
        let win = self.img.windows(tokens)?;
        Ok((&mut **a, win))
    }

    /// Row `t − 1` of the arena's residual: the last token's output after
    /// the last layer, the head's input.
    pub(super) fn last_row(&self, t: usize) -> Result<WindowMut<'static, f32>, GpuError> {
        let (x, rows, h) = match &self.a {
            UbRows::Whole(a) => (&a.x, a.rows, a.dims.hidden),
            UbRows::Placed(a) => (&a.x, a.rows, a.dims.hidden),
        };
        if t == 0 || t > rows {
            return Err(GpuError::shape(
                WHAT,
                format!("row {t} of a {rows}-row arena"),
            ));
        }
        // SAFETY: row t − 1 < rows spans `hidden` values inside `x` (rows ·
        // hidden), which stays in place while the window lives (one copy).
        Ok(unsafe { cells_view(x, (t - 1) * h, h) })
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
    pub(super) plans: &'a [LayerPlan],
    pub(super) k: &'a Kernels,
    pub(super) eps: f32,
    /// The rope table the rope launches read by position.
    pub(super) table: &'a DeviceBuffer<f32>,
}

/// The attention half at `t` rows: `x` in, `ffn_inp = x + attn_output(attn(x))`
/// out; the ubatch's K/V rows appended to the layer's planes at the rows'
/// positions. The wide arm's head-128 half over this arena's rows
/// ([`wide::neox_128_rows`]): each projection runs the launch its type
/// picks, the rows are quantized into the forms those sites read.
fn attention(
    c: &UbCtx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    a: &mut UbArena,
    t: usize,
    p0: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    let d = a.dims;
    if p0 + t > d.ctx {
        return Err(GpuError::shape(
            WHAT,
            format!("rows {p0}..{} past the {}-row cache", p0 + t, d.ctx),
        ));
    }
    let UbArena {
        x,
        pos,
        n_keys,
        normed,
        act_hid,
        act_hid32,
        q,
        k,
        v,
        attn,
        act_attn,
        act_attn32,
        attn_o,
        ffn_inp,
        dense,
        ..
    } = a;
    wide::neox_128_rows(
        (c.gpu, c.w, c.k),
        (c.table, c.eps, sink),
        (n, kv),
        (&d, t),
        Neox128Rows {
            x,
            pos,
            n_keys,
            normed,
            q,
            k,
            v,
            attn,
            attn_o,
            ffn_inp,
            hid: WideActsMut {
                q128: Some(act_hid),
                q32: act_hid32.as_mut(),
            },
            out: WideActsMut {
                q128: Some(act_attn),
                q32: act_attn32.as_mut(),
            },
            dense,
        },
    )
}

/// The routed FFN half at `t` rows: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's `k` experts out, into
/// `x`.
fn ffn(
    c: &UbCtx<'_>,
    n: &FfnPlan,
    a: &mut UbArena,
    t: usize,
    sink: FaultSink,
) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::ubatch::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = a.dims;
    let slots = t * d.slots();
    let FfnRoute::Router { gate_inp, .. } = &n.route else {
        return Err(GpuError::shape(
            WHAT_FFN,
            "a dense FFN (Qwen3's ubatch routes every layer)",
        ));
    };
    let used = d.routed(WHAT_FFN)?.used();
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
        f32_tensor(w, gate_inp)?,
        &a.normed,
        t,
        sink,
        &mut a.route,
    )?;
    k.gemm
        .enqueue_route(stream, &a.route.ids, slots, &mut a.moe, sink)?;
    for (ty, name, y) in [
        (n.gate_ty, &n.gate, &mut a.gate),
        (n.up_ty, &n.up, &mut a.up),
    ] {
        k.gemm.enqueue_gemm(
            stream,
            GemmArgs {
                ty: gemm_ty(ty)?,
                w: kq_weight(w, name)?,
                rows_per_expert: d.ff,
                act: &a.act_hid,
                route: &a.moe,
                input: GemmInput::Shared { top_k: used },
                y,
            },
        )?;
    }
    k.gemm
        .enqueue_swiglu_quant(stream, &a.gate, &a.up, slots, &mut a.act_h, sink)?;
    k.gemm.enqueue_gemm(
        stream,
        GemmArgs {
            ty: gemm_ty(n.down_ty)?,
            w: kq_weight(w, &n.down)?,
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
            n_slots: d.slots(),
            m: t,
            y: &mut a.x,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::Q32;
    use crate::arch::qwen3moe::plan::fixtures::layer;
    use crate::arch::qwen3moe::plan::{GqaKind, SiteTy};

    fn q32_of(tys: [SiteTy; 4]) -> Q32 {
        Q32::of(&[layer(GqaKind::Neox128, tys)])
    }

    fn q32(hid: bool, attn: bool) -> Q32 {
        Q32 { hid, attn }
    }

    /// The ubatch holds q8 blocks of 32 values only for the forms a Q8_0 site
    /// reads: the normed rows for q, k and v, the attention rows for the
    /// output projection. K-quant and F32 sites read other forms.
    #[test]
    fn the_ubatch_holds_32_value_blocks_for_a_q8_0_site_only() {
        use SiteTy::{F32, Q4K, Q5K, Q8_0};
        assert_eq!(q32_of([Q4K; 4]), q32(false, false));
        assert_eq!(q32_of([Q5K; 4]), q32(false, false));
        assert_eq!(q32_of([F32; 4]), q32(false, false));
        assert_eq!(q32_of([Q8_0, Q4K, Q4K, Q4K]), q32(true, false));
        assert_eq!(q32_of([Q4K, Q8_0, Q4K, Q4K]), q32(true, false));
        assert_eq!(q32_of([Q4K, Q4K, Q8_0, Q4K]), q32(true, false));
        assert_eq!(q32_of([Q4K, Q4K, Q4K, Q8_0]), q32(false, true));
        assert_eq!(q32_of([Q8_0; 4]), q32(true, true));
    }
}
