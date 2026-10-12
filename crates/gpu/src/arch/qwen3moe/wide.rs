//! The wide arm of each layer op: a unit of more than [`GEMV_COLS`] rows.
//! Each op of `dispatch` (and the delta mixer's input and output halves)
//! takes its first line from here when `m > GEMV_COLS` and keeps its gemv
//! body below it — one layer walk, the op choosing its kernel family from
//! `m`, as ik's `mul_mat` picks its matrix-vector kernel for at most eight
//! columns.
//!
//! A wide op reads each weight once for the unit: every projection runs
//! through the GEMM its file type picks ([`gemm_site`], `crate::site::gemm`: a
//! K-quant the grouped int8 GEMM over q8_1 blocks of 128, a Q8_0 the 32-value
//! GEMM over q8 blocks of 32, an F32 the F32 tile) over
//! a one-expert table
//! ([`route_dense`], filled once per unit after its embedding), the routed
//! experts through one table per layer (gate and up reading each token's
//! column, down each slot's own), a dense FFN's three matrices through the
//! one-expert table (its one slot a token is the token's column). A unit's
//! rows are quantized into exactly the forms the sites reading them take;
//! the norms, the quantizers, the rope with the cache append,
//! the prefill flash, the router's two launches, the SwiGLU quantizer and
//! the combine run over the unit's rows. The conv, the delta step and the
//! gated norm are the gemv arm's own launches: they already take any row
//! count.
//!
//! The attention half splits by the plan's kind, as the gemv arm does: head
//! 256 with its gate ([`GqaKind::Gated256`]) and head 128
//! ([`GqaKind::Neox128`]), whose half ([`neox_128_rows`]) runs over buffer
//! views, so the arena of this arm and the whole-card GEMM prefill's
//! (`ubatch.rs`) both lend it their rows.
//!
//! Numeric class. A wide op computes a token's values from that token's
//! inputs alone — the GEMM accumulates each (slot, row) on its own, the
//! flash walks a row's keys in fixed tiles whatever the rows beside it — so
//! a token's bits do not depend on the unit it lands in. Against the gemv arm
//! the products sum in another order (128-value integer blocks into one f32
//! accumulator, where the gemv sums lane partials through a warp tree) and
//! the attention weighs the values with f16 weights, so the two arms agree
//! to the error of those sums and are not bit-equal.

use super::body::{ATTN_SCALE, ATTN_SCALE_256, Kernels};
use super::body35::Body35;
use super::dispatch::Ctx;
use super::experts::CombineArgs;
use super::hostleg::{HostCombineArgs, HostLegKernels};
use super::plan::{DeltaPlan, FfnPlan, FfnRoute, Form, GqaKind, GqaPlan, SharedPlan, SiteTy};
use super::router::MAX_TOKENS;
use super::scratch::{
    ApartForms, Append128, Append256, Arena, Dims, Forms, GdnArena, KvPlanes, PrefillFlash, Route,
    Wants,
};
use super::taps35::Ffn35Run;
use crate::flash_gqa::{HEAD, HEAD_256, partials_ms_len, partials_v_len, partials_v_len_256};
use crate::gated_quant::GateLayout;
use crate::gemm::{
    GEMM_BN, GEMM_MAX_SLOTS, GEMM32_STEP, Gemm32Kernels, GemmAct, GemmAct32, GemmInput, GemmRoute,
};
use crate::host::xstream::RingLayer;
use crate::hybrid::HOST;
use crate::kquant::Act;
use crate::linear::{self, LinearShape};
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::model::{GpuModel, MAX_PASS_ROWS};
use crate::q38::OutGateArgs;
use crate::site::{self, Gated, GatedIo, SiteWeight, SwigluRows, WideIn, WideKernels};
use crate::tensor::{DeviceTensor, WindowMut};
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError};
use cuda_core::{CudaContext, CudaEvent, CudaStream, DeviceBuffer};
use std::mem::ManuallyDrop;
use std::sync::Arc;

/// The most rows an op runs through its gemv arm — ik's matrix-vector cut
/// (`ne[1] <= 8`), and the rows of a captured pass. Past it, the wide arm.
pub(super) const GEMV_COLS: usize = MAX_PASS_ROWS;
const _: () = assert!(GEMV_COLS == MAX_TOKENS);

const WHAT: &str = "qwen3moe::wide";

/// One wide input's quantized forms, each held when a site reads it
/// ([`Wants`]): q8_1 blocks of 128 for a K-quant site, q8 blocks of 32 for a
/// Q8_0 site.
struct Acts {
    q128: Option<GemmAct>,
    q32: Option<GemmAct32>,
}

impl Acts {
    fn new(stream: &CudaStream, want: Wants, cols: usize, k: usize) -> Result<Acts, GpuError> {
        Ok(Acts {
            q128: want
                .q128
                .then(|| GemmAct::new(stream, cols, k))
                .transpose()?,
            q32: want
                .q32
                .then(|| GemmAct32::new(stream, cols, k))
                .transpose()?,
        })
    }

    fn bytes(&self) -> usize {
        self.q128.as_ref().map_or(0, GemmAct::bytes) + self.q32.as_ref().map_or(0, GemmAct32::bytes)
    }

    /// The forms `x`'s first `m` columns are quantized into for sites of
    /// types `tys`: each form one of them reads, and nothing else.
    fn quantize(
        &mut self,
        c: &Ctx<'_>,
        x: &DeviceBuffer<f32>,
        m: usize,
        tys: &[SiteTy],
    ) -> Result<(), GpuError> {
        self.views_mut().quantize((c.gpu, c.k), x, m, tys, c.sink)
    }

    /// The forms as views.
    pub(super) fn views_mut(&mut self) -> ActsMut<'_> {
        ActsMut {
            q128: self.q128.as_mut(),
            q32: self.q32.as_mut(),
        }
    }

    /// The forms beside the f32 rows `f32` a site picks from.
    fn input<'a>(&'a self, f32: &'a DeviceBuffer<f32>) -> WideIn<'a> {
        WideIn {
            q128: self.q128.as_ref(),
            q32: self.q32.as_ref(),
            f32: Some(f32),
        }
    }

    /// The forms a gated block's SwiGLU writes beside the f32 rows `f32`, if
    /// its down reads them.
    fn swiglu_rows<'a>(&'a mut self, f32: Option<&'a mut DeviceBuffer<f32>>) -> SwigluRows<'a> {
        SwigluRows {
            q128: self.q128.as_mut(),
            q32: self.q32.as_mut(),
            f32,
        }
    }
}

/// One wide input's quantized forms as views: those of an [`Acts`], or of an
/// arena that holds them as plain buffers (the whole-card GEMM prefill's), so
/// one launch sequence ([`neox_128_rows`]) runs over either.
pub(super) struct ActsMut<'a> {
    pub(super) q128: Option<&'a mut GemmAct>,
    pub(super) q32: Option<&'a mut GemmAct32>,
}

impl ActsMut<'_> {
    /// The forms `x`'s first `m` columns are quantized into for sites of
    /// types `tys`: each form one of them reads, and nothing else.
    fn quantize(
        &mut self,
        (gpu, k): (&Gpu, &Kernels),
        x: &DeviceBuffer<f32>,
        m: usize,
        tys: &[SiteTy],
        sink: FaultSink,
    ) -> Result<(), GpuError> {
        const WHAT_Q: &str = "qwen3moe::wide::quantize";
        let reads = |f: Form| tys.iter().any(|t| t.reads() == f);
        if reads(Form::Q8x128) {
            let a = self.q128.as_deref_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the unit's q8_1 blocks of 128 (a K-quant site reads them)",
            ))?;
            gpu.enqueue_quantize_gemm(x, m, a, sink)?;
        }
        if reads(Form::Q8x32) {
            let a = self.q32.as_deref_mut().ok_or(GpuError::state(
                WHAT_Q,
                "the unit's q8 blocks of 32 (a Q8_0 site reads them)",
            ))?;
            k.q35(WHAT_Q)?
                .g32
                .enqueue_quantize_gemm32(gpu.stream(), x, m, a, sink)?;
        }
        Ok(())
    }

    /// The forms beside the f32 rows `f32` a site picks from.
    fn input<'a>(&'a self, f32: &'a DeviceBuffer<f32>) -> WideIn<'a> {
        WideIn {
            q128: self.q128.as_deref(),
            q32: self.q32.as_deref(),
            f32: Some(f32),
        }
    }
}

/// The wide part of an arena of `rows > GEMV_COLS` rows: the GEMMs'
/// activations, the second SwiGLU operand, the output projection's rows and
/// the two route tables.
pub(super) struct Wide {
    /// The normed rows (`hidden` a token): the mixer's projections' input,
    /// then the experts' gate·up input.
    act_hid: Acts,
    /// The attention rows (or a delta layer's gated norm): the output
    /// projection's input.
    act_attn: Acts,
    /// The slots' SwiGLU, one column per slot: the down's input.
    act_h: Acts,
    /// The up rows, per slot `ff` values; the gate rows are the arena's `h`.
    up: DeviceBuffer<f32>,
    /// The output projection's rows; the residual add reads them.
    attn_o: DeviceBuffer<f32>,
    /// The one-expert table every projection of a unit reads.
    dense: GemmRoute,
    /// The layer's expert table over `t · slots` slots, `slots` the
    /// router's per token (a folded shared expert's among them), over the
    /// joined stacks' `logits()` experts; `None` for a dense chain, whose
    /// FFN reads `dense`.
    moe: Option<GemmRoute>,
    /// What a layer that keeps its shared expert apart reads
    /// (`Forms::apart`); `None` on a chain whose every layer joins it.
    apart: Option<WideApart>,
}

/// The wide part's buffers of a layer that keeps its shared expert apart
/// from its routed stacks: the routed table and its map, the slots' card
/// list the combine reads, the shared expert's gate, up and SwiGLU rows with
/// the SwiGLU's forms its downs read, its output and the zero host sum
/// (`scratch::ApartRows`, the gemv arm's twin).
struct WideApart {
    /// Expert `e`'s place in the routed stacks: `e` for each routed id,
    /// [`HOST`] for the shared expert's (the routed count), so the routed
    /// table lists no shared slot.
    map: DeviceBuffer<u32>,
    /// The routed experts' table over `t · slots` slots of the file's
    /// routed stacks.
    moe: GemmRoute,
    sel: DeviceBuffer<u32>,
    g: DeviceBuffer<f32>,
    u: DeviceBuffer<f32>,
    h: DeviceBuffer<f32>,
    act_h: Acts,
    y: DeviceBuffer<f32>,
    zero: DeviceBuffer<f32>,
}

impl WideApart {
    /// The buffers for `rows` tokens of `d`, whose router routes, and the
    /// shared downs' forms `forms`, the map written and every other buffer
    /// zero. Load-time only.
    fn new(
        stream: &CudaStream,
        d: &Dims,
        rows: usize,
        forms: ApartForms,
    ) -> Result<WideApart, GpuError> {
        let n = d.routed(WHAT)?.experts();
        let slots = rows * d.slots();
        let host = u32::try_from(n).map_err(|_| GpuError::shape(WHAT, "an expert id past u32"))?;
        let map: Vec<u32> = (0..host).chain([HOST]).collect();
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(WideApart {
            map: DeviceBuffer::from_host(stream, &map)?,
            moe: GemmRoute::new(stream, slots, n)?,
            sel: DeviceBuffer::zeroed(stream, slots)?,
            g: f(rows * d.ff)?,
            u: f(rows * d.ff)?,
            h: f(rows * d.ff)?,
            act_h: Acts::new(stream, forms.h, rows, d.ff)?,
            y: f(rows * d.hidden)?,
            zero: f(rows * d.hidden)?,
        })
    }

    fn bytes(&self) -> usize {
        [&self.g, &self.u, &self.h, &self.y, &self.zero]
            .iter()
            .map(|b| b.num_bytes())
            .sum::<usize>()
            + self.map.num_bytes()
            + self.moe.bytes()
            + self.sel.num_bytes()
            + self.act_h.bytes()
    }
}

impl Wide {
    /// The wide part for `rows` tokens of `d` and the sites' forms `forms`,
    /// or a named refusal when their slots pass the GEMM's route table.
    /// Load-time only.
    pub(super) fn new(
        stream: &CudaStream,
        d: &Dims,
        rows: usize,
        forms: Forms,
    ) -> Result<Wide, GpuError> {
        let slots = rows * d.slots();
        if slots > GEMM_MAX_SLOTS {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{rows} rows at {} slots a token are {slots} slots; the grouped GEMM's route \
                     table holds {GEMM_MAX_SLOTS}",
                    d.slots()
                ),
            ));
        }
        let f = |n: usize| DeviceBuffer::<f32>::zeroed(stream, n);
        Ok(Wide {
            act_hid: Acts::new(stream, forms.hid, rows, d.hidden)?,
            act_attn: Acts::new(stream, forms.attn, rows, d.attn_len())?,
            act_h: Acts::new(stream, forms.h, slots, d.ff)?,
            up: f(slots * d.ff)?,
            attn_o: f(rows * d.hidden)?,
            dense: GemmRoute::new(stream, rows, 1)?,
            moe: d
                .router
                .map(|r| GemmRoute::new(stream, slots, r.logits()))
                .transpose()?,
            apart: forms
                .apart
                .map(|a| WideApart::new(stream, d, rows, a))
                .transpose()?,
        })
    }

    /// Device bytes.
    pub(super) fn bytes(&self) -> usize {
        self.act_hid.bytes()
            + self.act_attn.bytes()
            + self.act_h.bytes()
            + self.up.num_bytes()
            + self.attn_o.num_bytes()
            + self.dense.bytes()
            + self.moe.as_ref().map_or(0, GemmRoute::bytes)
            + self.apart.as_ref().map_or(0, WideApart::bytes)
    }
}

/// Bytes of one q8_1 activation column of `k` values, a `Q8Act`'s or a
/// `GemmAct`'s: the q3 pairs (u64), the q4 and q6 permutations, the 32-value
/// code sums and the 128-value scales.
pub(super) fn act_col_bytes(k: usize) -> usize {
    let n_sb = k / 256;
    64 * n_sb.div_ceil(2) * 8
        + (256 * n_sb.div_ceil(4) + 128 * n_sb.div_ceil(2)) * 4
        + 10 * n_sb * 4
}

/// Bytes of one `GemmAct32` column of `k` values: per 64-value step sixteen
/// code words, two scales and two sums.
fn act32_col_bytes(k: usize) -> usize {
    k.div_ceil(GEMM32_STEP) * (16 + 2 + 2) * 4
}

/// Bytes of one wide input's forms `want` over `cols` columns of `k`.
fn wants_bytes(want: Wants, cols: usize, k: usize) -> usize {
    cols * (usize::from(want.q128) * act_col_bytes(k) + usize::from(want.q32) * act32_col_bytes(k))
}

/// Bytes of a route table for `slots` slots over `experts` experts.
pub(super) fn route_bytes(slots: usize, experts: usize) -> usize {
    let tiles = slots / GEMM_BN + experts.min(slots);
    let zeros = if experts == 1 { slots } else { 0 };
    (slots + 2 * tiles + 2 + zeros) * 4
}

/// Bytes of a delta layer's intermediates for `rows` tokens
/// (`GdnArena::new`): the four projection blocks, the conv, β, the decay,
/// the delta output, and past one row the Q6_K projection's row-major copy
/// of at most [`GEMV_COLS`] tokens.
fn gdn_bytes(s: LinearShape, rows: usize) -> usize {
    let (c, zl, nv) = (s.channels(), s.n_v * linear::HEAD, s.n_v);
    let cols = if rows > 1 { rows.min(GEMV_COLS) * c } else { 0 };
    (rows * (c + zl + 2 * nv) + rows * c + 2 * rows * nv + rows * zl + cols) * 4
}

/// The device bytes `Arena::with(d, rows, forms)` allocates, from `d`, `rows`
/// and `forms` alone: what a load checks against the card's free bytes
/// before it allocates the arena, and what the arena it then allocates must
/// hold.
pub(super) fn arena_bytes(d: &Dims, rows: usize, forms: Forms) -> usize {
    let n = rows.min(GEMV_COLS);
    let (q_len, kv_len, att, slots) = (d.q_rows, d.kv_len(), d.attn_len(), d.slots());
    let part_v = if d.head == HEAD_256 {
        partials_v_len_256(n, d.n_head)
    } else {
        partials_v_len(n, d.n_head)
    };
    let f32s = 3 * rows * d.hidden
        + rows * (q_len + 2 * kv_len)
        + if q_len == att { 0 } else { rows * att }
        + if rows > 1 { n * kv_len } else { 0 }
        + part_v
        + partials_ms_len(n, d.n_head)
        + rows * att
        + rows * slots * (d.ff + d.hidden)
        + if forms.glu { 2 * n * slots * d.ff } else { 0 }
        + if rows > 1 { n * forms.cols } else { 0 };
    let u32s = 2 * rows;
    let acts: usize = (1..=n)
        .map(|m| {
            m * (2 * act_col_bytes(d.hidden) + act_col_bytes(att)) + m * slots * act_col_bytes(d.ff)
        })
        .sum();
    let route = match d.router {
        Some(r) => (rows * (r.logits() + r.experts() + 2 * slots) + 1) * 4,
        None => 2 * rows * 4,
    };
    let wide = if rows > GEMV_COLS {
        let s = rows * slots;
        wants_bytes(forms.hid, rows, d.hidden)
            + wants_bytes(forms.attn, rows, att)
            + wants_bytes(forms.h, s, d.ff)
            + (s * d.ff + rows * d.hidden) * 4
            + route_bytes(rows, 1)
            + d.router.map_or(0, |r| route_bytes(s, r.logits()))
    } else {
        0
    };
    // A layer that keeps its shared expert apart: the gemv arm's card list,
    // the shared gate, up and SwiGLU rows with their q8_1, its output and
    // the zero host sum; past GEMV_COLS rows the wide part's twins, its map
    // and its routed table.
    let apart = match (forms.apart, d.router) {
        (Some(a), Some(r)) => {
            let narrow = (n * slots + 3 * n * d.ff + 2 * n * d.hidden) * 4
                + (1..=n).map(|m| m * act_col_bytes(d.ff)).sum::<usize>();
            let wide = if rows > GEMV_COLS {
                let s = rows * slots;
                (s + r.experts() + 1 + 3 * rows * d.ff + 2 * rows * d.hidden) * 4
                    + wants_bytes(a.h, rows, d.ff)
                    + route_bytes(s, r.experts())
            } else {
                0
            };
            narrow + wide
        }
        (Some(_), None) | (None, _) => 0,
    };
    (f32s + u32s) * 4 + acts + route + d.lin.map_or(0, |s| gdn_bytes(s, rows)) + wide + apart
}

/// The arena's wide part, or a named refusal (an arena of at most
/// [`GEMV_COLS`] rows has none).
fn wide_of<'a>(w: &'a mut Option<Wide>, what: &'static str) -> Result<&'a mut Wide, GpuError> {
    w.as_mut().ok_or(GpuError::state(
        what,
        "the arena's wide part (an arena of more than GEMV_COLS rows)",
    ))
}

/// Enqueue `y = W · x` for `rows` rows of weight `name` (of type `ty`) over
/// the columns `route` holds, `input` picking each slot's column of the
/// forms `x` holds, `m` the unit's rows.
#[allow(
    clippy::too_many_arguments,
    reason = "one site's context, weight, rows, inputs, table, width and output (rust-quality R8)"
)]
fn gemm(
    c: &Ctx<'_>,
    (ty, name): (SiteTy, &str),
    rows: usize,
    x: &WideIn<'_>,
    (route, input): (&GemmRoute, GemmInput),
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    gemm_site((c.gpu, c.w, c.k), (ty, name), rows, x, (route, input), m, y)
}

/// The GEMM kernels a wide site launches: the grouped GEMM always, the
/// 32-value kernels when the load holds them (a load whose sites are all
/// K-quants holds none; `site` refuses a site that needs them by name).
fn wide_kernels(k: &Kernels) -> WideKernels<'_> {
    WideKernels {
        gemm: &k.gemm,
        g32: k.q35.as_ref().map(|q| &q.g32),
    }
}

/// [`gemm`] over the pieces it reads, through `site::gemm` with the load's
/// [`wide_kernels`].
#[allow(
    clippy::too_many_arguments,
    reason = "one site's device, weights, kernels, weight, rows, inputs, table, width and output (rust-quality R8)"
)]
pub(super) fn gemm_site(
    (gpu, w, k): (&Gpu, &Weights, &Kernels),
    (ty, name): (SiteTy, &str),
    rows: usize,
    x: &WideIn<'_>,
    (route, input): (&GemmRoute, GemmInput),
    m: usize,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    site::gemm(
        gpu,
        &wide_kernels(k),
        (ty, w, SiteWeight::Resident(name)),
        rows,
        x,
        (route, input),
        m,
        y,
    )
}

/// The one-expert table of a unit of `m` rows: `m` slots on expert 0, slot
/// `s` reading column `s`, which every projection of the unit reads. Once per
/// unit, behind its embedding.
pub(super) fn route_dense(c: &Ctx<'_>, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    let w = wide_of(&mut s.wide, "qwen3moe::wide::route_dense")?;
    c.k.gemm
        .enqueue_route_dense(c.gpu.stream(), m, &mut w.dense, c.gpu.unlabelled_sink())
}

/// The attention half at `m` rows: `x` in, `ffn_inp` out, the unit's K/V
/// rows appended to the layer's planes at the rows' positions; the plan's
/// kind picks the half ([`gated_256`], [`neox_128`]).
pub(super) fn attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    match n.kind {
        GqaKind::Gated256 => gated_256(c, n, kv, s, m),
        GqaKind::Neox128 => neox_128(c, n, kv, s, m),
    }
}

/// The head-256 attention half ([`GqaKind::Gated256`]): `ffn_inp = x + W_o ·
/// (flash ⊙ σ(gate))`. A K-quant `attn_output` reads the gated rows quantized
/// in one launch; any other type reads them as f32 rows (the out gate into
/// the free query buffer), quantized as it takes them.
fn gated_256(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_ATTN: &str = "qwen3moe::wide::attention";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let q35 = k.q35(WHAT_ATTN)?;
    let stream = gpu.stream();
    let d = s.dims;
    let Arena {
        x,
        normed,
        pos,
        n_keys,
        q,
        k: kr,
        v,
        q_out,
        attn,
        ffn_inp,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_ATTN)?;
    let q_out = q_out.as_mut().ok_or(GpuError::state(
        WHAT_ATTN,
        "the arena's buffer of gated queries",
    ))?;
    gpu.elem().enqueue_rms_norm(
        stream,
        x,
        f32_gain(w, &n.attn_norm)?,
        c.eps,
        d.hidden,
        m,
        normed,
    )?;
    wd.act_hid
        .quantize(c, normed, m, &[n.q_ty, n.k_ty, n.v_ty])?;
    let table = (&wd.dense, GemmInput::PerSlot);
    let hid = wd.act_hid.input(normed);
    gemm(c, (n.q_ty, &n.attn_q), d.q_rows, &hid, table, m, q)?;
    gemm(c, (n.k_ty, &n.attn_k), d.kv_len(), &hid, table, m, kr)?;
    gemm(c, (n.v_ty, &n.attn_v), d.kv_len(), &hid, table, m, v)?;
    kv.append_256(
        &k.neox,
        stream,
        Append256 {
            qg: q,
            q: q_out,
            k: kr,
            v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            table: c.table,
            pos,
            eps: c.eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m,
            fault: c.sink,
        },
    )?;
    kv.prefill_256(
        &k.prefill,
        stream,
        PrefillFlash {
            q: q_out,
            n_keys,
            scale: ATTN_SCALE_256,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            t: m,
            fault: c.sink,
            y: attn,
        },
        n.flash,
    )?;
    let Wide {
        act_attn,
        dense,
        attn_o,
        ..
    } = wd;
    let gated: &DeviceBuffer<f32> = if n.o_ty.kquant() {
        let a = act_attn.q128.as_mut().ok_or(GpuError::state(
            WHAT_ATTN,
            "the wide part's q8_1 blocks of the attention rows",
        ))?;
        q35.gated.enqueue_gemm(
            stream,
            (attn, q),
            GateLayout {
                head: d.head,
                head_stride: 2 * d.head,
                offset: d.head,
                col_stride: d.q_rows,
            },
            a,
            m,
            c.sink,
        )?;
        attn
    } else {
        // The flash has read the queries: their buffer takes the gated rows.
        q35.q38.enqueue_out_gate(
            stream,
            OutGateArgs {
                attn,
                qg: q,
                n_head: d.n_head,
                m,
                fault: c.sink,
                y: q_out,
            },
        )?;
        act_attn.quantize(c, q_out, m, &[n.o_ty])?;
        q_out
    };
    let input = act_attn.input(gated);
    gemm(
        c,
        (n.o_ty, &n.attn_output),
        d.hidden,
        &input,
        (dense, GemmInput::PerSlot),
        m,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, m * d.hidden, ffn_inp)
}

/// The head-128 attention half ([`GqaKind::Neox128`]) of the arena's `m`
/// rows: `ffn_inp = x + W_o · flash(x)`, over [`neox_128_rows`].
fn neox_128(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_ATTN: &str = "qwen3moe::wide::neox_128";
    let d = s.dims;
    let Arena {
        x,
        normed,
        pos,
        n_keys,
        q,
        k,
        v,
        attn,
        ffn_inp,
        wide,
        ..
    } = s;
    let Wide {
        act_hid,
        act_attn,
        dense,
        attn_o,
        ..
    } = wide_of(wide, WHAT_ATTN)?;
    neox_128_rows(
        (c.gpu, c.w, c.k),
        (c.table, c.eps, c.sink),
        (n, kv),
        (&d, m),
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
            hid: act_hid.views_mut(),
            out: act_attn.views_mut(),
            dense,
        },
    )
}

/// The buffers of a head-128 attention half ([`neox_128_rows`]) as views, each
/// holding `t` rows: the unit's residual `x` and its rows' positions and live
/// key counts, the normed rows with the forms the q·k·v projections read
/// (`hid`), the projections' rows `q`, `k` and `v`, the flash output `attn`
/// with the forms the output projection reads (`out`), that projection's rows
/// `attn_o`, the half's output `ffn_inp`, and the unit's one-expert table.
pub(super) struct Neox128Rows<'a> {
    pub(super) x: &'a DeviceBuffer<f32>,
    pub(super) pos: &'a DeviceBuffer<u32>,
    pub(super) n_keys: &'a DeviceBuffer<u32>,
    pub(super) normed: &'a mut DeviceBuffer<f32>,
    pub(super) q: &'a mut DeviceBuffer<f32>,
    pub(super) k: &'a mut DeviceBuffer<f32>,
    pub(super) v: &'a mut DeviceBuffer<f32>,
    pub(super) attn: &'a mut DeviceBuffer<f32>,
    pub(super) attn_o: &'a mut DeviceBuffer<f32>,
    pub(super) ffn_inp: &'a mut DeviceBuffer<f32>,
    pub(super) hid: ActsMut<'a>,
    pub(super) out: ActsMut<'a>,
    pub(super) dense: &'a GemmRoute,
}

/// The head-128 attention half at `t` rows over `b`'s views: `x` in, `ffn_inp
/// = x + attn_output(attn(x))` out; the unit's K/V rows appended to the
/// layer's planes at the rows' positions. Each projection runs the launch
/// its type picks ([`gemm_site`]) over the one-expert table; the rows are
/// quantized into the forms those sites read. The one launch sequence of the
/// wide arm and of the whole-card GEMM prefill.
pub(super) fn neox_128_rows(
    (gpu, w, k): (&Gpu, &Weights, &Kernels),
    (table, eps, sink): (&DeviceBuffer<f32>, f32, FaultSink),
    (n, kv): (&GqaPlan, &mut KvPlanes),
    (d, t): (&Dims, usize),
    b: Neox128Rows<'_>,
) -> Result<(), GpuError> {
    const WHAT_N: &str = "qwen3moe::wide::neox_128_rows";
    if d.head != HEAD {
        return Err(GpuError::shape(
            WHAT_N,
            format!("a head of {} values in a head-{HEAD} attention", d.head),
        ));
    }
    let stream = gpu.stream();
    let (q_len, kv_len) = (d.n_head * HEAD, d.n_kv * HEAD);
    let Neox128Rows {
        x,
        pos,
        n_keys,
        normed,
        q,
        k: kr,
        v,
        attn,
        attn_o,
        ffn_inp,
        mut hid,
        mut out,
        dense,
    } = b;
    gpu.elem().enqueue_rms_norm(
        stream,
        x,
        f32_gain(w, &n.attn_norm)?,
        eps,
        d.hidden,
        t,
        normed,
    )?;
    hid.quantize((gpu, k), normed, t, &[n.q_ty, n.k_ty, n.v_ty], sink)?;
    let input = hid.input(normed);
    for (name, ty, rows, y) in [
        (&n.attn_q, n.q_ty, q_len, &mut *q),
        (&n.attn_k, n.k_ty, kv_len, &mut *kr),
        (&n.attn_v, n.v_ty, kv_len, &mut *v),
    ] {
        let one = (dense, GemmInput::PerSlot);
        gemm_site((gpu, w, k), (ty, name), rows, &input, one, t, y)?;
    }
    kv.append_128(
        &k.neox,
        stream,
        Append128 {
            q: &mut *q,
            k: &mut *kr,
            v: &*v,
            gq: f32_gain(w, &n.attn_q_norm)?,
            gk: f32_gain(w, &n.attn_k_norm)?,
            table,
            pos,
            eps,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            m: t,
            fault: sink,
        },
    )?;
    kv.prefill_128(
        &k.prefill,
        stream,
        PrefillFlash {
            q: &*q,
            n_keys,
            scale: ATTN_SCALE,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            t,
            fault: sink,
            y: &mut *attn,
        },
    )?;
    out.quantize((gpu, k), attn, t, &[n.o_ty], sink)?;
    let input = out.input(attn);
    gemm_site(
        (gpu, w, k),
        (n.o_ty, &n.attn_output),
        d.hidden,
        &input,
        (dense, GemmInput::PerSlot),
        t,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, t * d.hidden, ffn_inp)
}

/// A delta layer's input half at `m` rows: `x` normed and quantized into the
/// forms its projections read, then its four projections — `attn_qkv`,
/// `attn_gate`, β and α — into the arena's `[x | z | b | a]` blocks, each a
/// GEMM of its type.
pub(super) fn delta_in(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_IN: &str = "qwen3moe::wide::delta_in";
    let (gpu, w) = (c.gpu, c.w);
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let Arena {
        x,
        normed,
        gdn,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_IN)?;
    let g: &mut GdnArena = gdn
        .as_mut()
        .ok_or(GpuError::state(WHAT_IN, "the arena's delta intermediates"))?;
    gpu.elem().enqueue_rms_norm(
        stream,
        x,
        f32_gain(w, &d.attn_norm)?,
        c.eps,
        hidden,
        m,
        normed,
    )?;
    wd.act_hid
        .quantize(c, normed, m, &[d.qkv_ty, d.gate_ty, d.beta_ty, d.alpha_ty])?;
    let (a, t) = (wd.act_hid.input(normed), (&wd.dense, GemmInput::PerSlot));
    let (ch, zl, nv) = (d.shape.channels(), d.shape.n_v * linear::HEAD, d.shape.n_v);
    gemm(c, (d.qkv_ty, &d.qkv), ch, &a, t, m, &mut g.x)?;
    gemm(c, (d.gate_ty, &d.gate), zl, &a, t, m, &mut g.z)?;
    gemm(c, (d.beta_ty, &d.beta), nv, &a, t, m, &mut g.b)?;
    gemm(c, (d.alpha_ty, &d.alpha), nv, &a, t, m, &mut g.a)
}

/// A delta layer's output half at `m` rows: the gated norm's rows (the
/// arena's `attn`) quantized into the form `ssm_out` reads, `ssm_out`, then
/// `ffn_inp = x + ssm_out(·)`.
pub(super) fn delta_out(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_OUT: &str = "qwen3moe::wide::delta_out";
    let gpu = c.gpu;
    let stream = gpu.stream();
    let hidden = s.dims.hidden;
    let Arena {
        x,
        attn,
        ffn_inp,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_OUT)?;
    wd.act_attn.quantize(c, attn, m, &[d.out_ty])?;
    let Wide {
        act_attn,
        dense,
        attn_o,
        ..
    } = wd;
    gemm(
        c,
        (d.out_ty, &d.ssm_out),
        hidden,
        &act_attn.input(attn),
        (dense, GemmInput::PerSlot),
        m,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, m * hidden, ffn_inp)
}

/// The FFN half at `m` rows: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's slots out, into `out`
/// (`None`: into `x`). The gated router's slots (the shared expert the last
/// of each token's, its id the joined stacks' last expert) through the
/// layer's expert table, or a dense FFN's one slot a token through the
/// unit's one-expert table at the arena's fixed weight 1; the SwiGLU's
/// quantizer is the one the down's type reads. A layer that keeps its shared
/// expert apart ([`FfnPlan::apart`]) routes its routed slots through a table
/// over the file's routed stacks that lists no shared slot, runs the shared
/// expert's gate, up, SwiGLU and down on the unit's one-expert table at
/// their own types, and adds it in the host-sum combine (`hostleg`):
/// `((routed + 0) + resid) + σ · shared`, the host's routed sum zero on a
/// whole-card load. That arm is three parts ([`ffn_front`], [`ffn_card`],
/// [`ffn_combine`]) with the slots' card list between the first two
/// ([`card_list`]). A plain router (no shared expert) takes its wide rows
/// through the whole-card GEMM prefill (`ubatch.rs`) or a placed load's walk
/// (the three parts with the card route and host sums of `placed`); this
/// whole-card entry refuses it by name.
pub(super) fn ffn(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    if let FfnRoute::Router { shared: None, .. } = n.route {
        return Err(GpuError::shape(
            "qwen3moe::wide::ffn",
            format!(
                "layer {}: the plain router at {m} rows on the whole-card wide arm (it runs \
                 through ubatch.rs, or a placed load's walk)",
                c.layer
            ),
        ));
    }
    ffn_front(c, n, s, m)?;
    if n.apart().is_none() {
        return ffn_joined(c, n, s, m, out);
    }
    card_list(c, s, m)?;
    ffn_card(c, n, s, m, None)?;
    ffn_combine(c, n, s, m, None, out)
}

/// The FFN's front at `m` rows ([`ffn`]): `ffn_inp` RMS-normed into `normed`,
/// its forms every site of the layer reads, and the router over the normed
/// rows (ids and weights in the arena's route): the gated router with a
/// shared expert, the plain one without.
pub(super) fn ffn_front(c: &Ctx<'_>, n: &FfnPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::wide::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let slots = d.slots();
    let used = d.router.map_or(0, |r| r.used());
    if slots != c.p.slots(used) {
        return Err(GpuError::shape(
            WHAT_FFN,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(used)
            ),
        ));
    }
    let Arena {
        normed,
        ffn_inp,
        route,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_FFN)?;
    gpu.elem().enqueue_rms_norm(
        stream,
        ffn_inp,
        f32_gain(w, &n.ffn_norm)?,
        c.eps,
        d.hidden,
        m,
        normed,
    )?;
    let sh = n.apart();
    let hid_tys = [
        n.gate_ty,
        n.up_ty,
        sh.map_or(n.gate_ty, |a| a.gate_ty),
        sh.map_or(n.up_ty, |a| a.up_ty),
    ];
    wd.act_hid.quantize(c, normed, m, &hid_tys)?;
    if let FfnRoute::Router { gate_inp, shared } = &n.route {
        let gate = f32_tensor(w, gate_inp)?;
        match shared {
            Some(_) => k.q35(WHAT_FFN)?.router.enqueue_ubatch(
                stream,
                gate,
                normed,
                m,
                c.sink,
                route.gated(WHAT_FFN)?,
            )?,
            None => {
                k.router
                    .enqueue_ubatch(stream, gate, normed, m, c.sink, route.plain(WHAT_FFN)?)?
            }
        }
    }
    Ok(())
}

/// The rest of [`ffn`] for a layer whose shared expert is folded into the
/// stacks (or a dense FFN) after [`ffn_front`]: the slots through the
/// layer's one table and the combine of their rows.
fn ffn_joined(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::wide::ffn";
    let (gpu, k) = (c.gpu, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let slots = d.slots();
    let Arena {
        x,
        normed,
        ffn_inp,
        route,
        h,
        down,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_FFN)?;
    let Wide {
        act_hid,
        act_h,
        up,
        dense,
        moe,
        ..
    } = wd;
    let hid = act_hid.input(normed);
    let rows = (act_h, &mut *h, &mut *up, &mut *down);
    let y = match out {
        Some(y) => y,
        None => x,
    };
    let table = match &n.route {
        FfnRoute::Router { .. } => {
            let moe = moe.as_mut().ok_or(GpuError::state(
                WHAT_FFN,
                "the wide part's expert table (a routed chain's)",
            ))?;
            k.gemm
                .enqueue_route(stream, route.ids(), m * slots, moe, c.sink)?;
            (&*moe, GemmInput::Shared { top_k: slots })
        }
        FfnRoute::Dense => (&*dense, GemmInput::PerSlot),
    };
    slots_gemm(c, (n, None), &d, table, &hid, rows, m)?;
    k.experts.enqueue_combine_tokens(
        stream,
        CombineArgs {
            down,
            w: route.weights(),
            resid: ffn_inp,
            rows: d.hidden,
            n_slots: slots,
            m,
            y,
        },
    )
}

/// The slots' card list the combine reads ([`ffn`]'s apart arm) for the
/// routed ids of `m` rows: the shared slot's id, the routed count, is
/// [`HOST`].
pub(super) fn card_list(c: &Ctx<'_>, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    const WHAT_C: &str = "qwen3moe::wide::card_list";
    let d = s.dims;
    let n_routed = d.routed(WHAT_C)?.experts();
    let Arena { route, wide, .. } = s;
    let wd = wide_of(wide, WHAT_C)?;
    let apart = wd.apart.as_mut().ok_or(GpuError::state(
        WHAT_C,
        "the wide part's buffers of a shared expert kept apart",
    ))?;
    c.gpu.elem().enqueue_card_sel(
        c.gpu.stream(),
        route.ids(),
        m * d.slots(),
        n_routed,
        n_routed + 1,
        c.sink,
        &mut apart.sel,
    )
}

/// Err unless `n` routes its experts with no shared expert folded into their
/// stacks: a router with none (the plain one's) or one kept apart. The apart
/// arm's card route and combine ([`ffn_card`], [`ffn_combine`]) run neither a
/// folded shared expert nor a dense FFN.
fn refuse_folded(n: &FfnPlan, what: &'static str) -> Result<(), GpuError> {
    match &n.route {
        FfnRoute::Router {
            shared: None | Some(SharedPlan::Apart(_)),
            ..
        } => Ok(()),
        FfnRoute::Router {
            shared: Some(SharedPlan::Joined),
            ..
        }
        | FfnRoute::Dense => Err(GpuError::state(
            what,
            "a router with no shared expert, or one kept apart (the plan folds it into the \
             stacks, or is a dense FFN)",
        )),
    }
}

/// A layer whose routed experts the card holds only in part (a placed
/// load's, `placed`), for [`ffn_card`]: the stacks the card holds are the
/// layer's `n_card` experts, in the slot map's order.
pub(super) struct CardRoute<'a> {
    /// Expert `e`'s place in the card's stacks, [`HOST`] for an expert the
    /// host tier computes, and a last entry [`HOST`] for the shared expert's
    /// id (the routed count): `n_routed + 1` entries.
    pub(super) map: &'a DeviceBuffer<u32>,
    /// The routed table over the card's stacks (`n_card` experts, slots for
    /// the unit's rows); `None` on a layer with no card expert, which
    /// launches no routed GEMM.
    pub(super) table: Option<&'a mut GemmRoute>,
    /// The kernels that build the table from the ids through the map.
    pub(super) remap: &'a Gemm32Kernels,
}

/// The shared expert's rows in the wide part of a layer that keeps it apart:
/// its gate, up and SwiGLU rows with the SwiGLU's forms its down reads, and
/// its output.
struct SharedRows<'a> {
    g: &'a mut DeviceBuffer<f32>,
    u: &'a mut DeviceBuffer<f32>,
    h: &'a mut DeviceBuffer<f32>,
    act_h: &'a mut Acts,
    y: &'a mut DeviceBuffer<f32>,
}

/// The card's part of the apart arm of [`ffn`] over `m` rows, after
/// [`ffn_front`] and (on the whole-card arm) [`card_list`]: the routed table
/// over the file's routed stacks, the routed slots' gate, up, SwiGLU and down
/// into `down`, and — on a plan with a shared expert kept apart — the shared
/// expert's gate, up, SwiGLU and down on the unit's one-expert table into its
/// own output. `card` is the layer's split of its routed experts between the
/// card and a host tier; `None` runs every routed expert on the card, through
/// the arena's own map and table, and needs the shared expert's buffers. A
/// plan with no shared expert (the plain router's) runs only with a `card`.
pub(super) fn ffn_card(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    card: Option<CardRoute<'_>>,
) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::wide::ffn_card";
    refuse_folded(n, WHAT_FFN)?;
    let (gpu, k) = (c.gpu, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let slots = d.slots();
    let Arena {
        normed,
        route,
        h,
        down,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_FFN)?;
    let Wide {
        act_hid,
        act_h,
        up,
        dense,
        apart,
        ..
    } = wd;
    let hid = act_hid.input(normed);
    let rows = (act_h, &mut *h, &mut *up, &mut *down);
    let own = apart.as_mut().map(|a| {
        let WideApart {
            map,
            moe,
            g,
            u,
            h: sh_h,
            act_h: sh_act,
            y,
            ..
        } = a;
        let shared = SharedRows {
            g,
            u,
            h: sh_h,
            act_h: sh_act,
            y,
        };
        (&*map, moe, shared)
    });
    let (map, table, remap, shared) = match (card, own) {
        (Some(CardRoute { map, table, remap }), own) => {
            (map, table, remap, own.map(|(_, _, shared)| shared))
        }
        (None, Some((map, moe, shared))) => (map, Some(moe), &k.q35(WHAT_FFN)?.g32, Some(shared)),
        (None, None) => {
            return Err(GpuError::state(
                WHAT_FFN,
                "a card route, or the wide part's buffers of a shared expert kept apart",
            ));
        }
    };
    // The routed table over the file's routed stacks, which lists no shared
    // slot.
    if let Some(table) = table {
        remap.enqueue_route_remap(stream, route.ids(), map, m * slots, table, c.sink)?;
        let list = (&*table, GemmInput::Shared { top_k: slots });
        slots_gemm(c, (n, None), &d, list, &hid, rows, m)?;
    }
    let Some(sh) = n.apart() else {
        return Ok(());
    };
    let SharedRows {
        g,
        u,
        h: sh_h,
        act_h: sh_act,
        y: sh_y,
    } = shared.ok_or(GpuError::state(
        WHAT_FFN,
        "the wide part's buffers of a shared expert kept apart",
    ))?;
    let shared = Gated {
        w: c.w,
        route: dense,
        gate: (sh.gate_ty, SiteWeight::Resident(&sh.gate)),
        up: (sh.up_ty, SiteWeight::Resident(&sh.up)),
        down: (sh.down_ty, SiteWeight::Resident(&sh.down)),
        ff: d.ff,
        hidden: d.hidden,
        rule: Act::SiluMul,
        input: GemmInput::PerSlot,
    };
    let io = GatedIo {
        x: &hid,
        gate: g,
        up: u,
        rows: sh_act.swiglu_rows(Some(sh_h)),
        y: sh_y,
    };
    site::gated(gpu, &wide_kernels(c.k), &shared, (m, m), io, c.sink)
}

/// What the combine of a layer split between the card and a host tier reads
/// beside the card's rows ([`ffn_combine`]).
pub(super) struct HostSide<'a> {
    /// Each slot's place in the card's stacks, [`HOST`] for the host's
    /// (`hostleg`'s places), `pitch` a token.
    pub(super) sel: &'a DeviceBuffer<u32>,
    /// The host tier's routed sums, `hidden` a token.
    pub(super) hsum: &'a DeviceBuffer<f32>,
    /// The layer's experts on the card.
    pub(super) n_card: usize,
    /// The combine's kernels.
    pub(super) leg: &'a HostLegKernels,
}

/// The combine over `m` rows, after [`ffn_card`], into `out` (`None`: into
/// `x`): the card's slots, the host's routed sum, the residual and — on a plan
/// with a shared expert kept apart — the shared expert. `host` is the places
/// and the host sums of a layer split with a host tier; `None` runs every
/// routed slot on the card, so the host's sum is the arena's zero rows, and
/// needs the shared expert's buffers (a plan with none runs only with a
/// `host`).
pub(super) fn ffn_combine(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    host: Option<HostSide<'_>>,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::wide::ffn_combine";
    refuse_folded(n, WHAT_FFN)?;
    let d = s.dims;
    let used = d.router.map_or(0, |r| r.used());
    let n_routed = d.routed(WHAT_FFN)?.experts();
    let Arena {
        x,
        ffn_inp,
        route,
        down,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_FFN)?;
    let apart = wd.apart.as_ref();
    let shared = match (n.apart(), apart) {
        (Some(_), Some(a)) => Some((&a.y, used)),
        (None, _) => None,
        (Some(_), None) => {
            return Err(GpuError::state(
                WHAT_FFN,
                "the wide part's buffers of a shared expert kept apart",
            ));
        }
    };
    let (sel, hsum, n_card, leg) = match (host, apart) {
        (
            Some(HostSide {
                sel,
                hsum,
                n_card,
                leg,
            }),
            _,
        ) => (sel, hsum, n_card, leg),
        (None, Some(a)) => (&a.sel, &a.zero, n_routed, &c.k.q35(WHAT_FFN)?.ffn.leg),
        (None, None) => {
            return Err(GpuError::state(
                WHAT_FFN,
                "a host side, or the wide part's buffers of a shared expert kept apart",
            ));
        }
    };
    let y = match out {
        Some(y) => y,
        None => x,
    };
    leg.enqueue_combine(
        c.gpu.stream(),
        HostCombineArgs {
            down,
            w: route.weights(),
            sel,
            hsum,
            resid: ffn_inp,
            shared,
            rows: d.hidden,
            pitch: d.slots(),
            n_card,
            m,
            fault: c.sink,
            y,
        },
    )
}

/// A half of the expert stream's ring as the routed GEMMs read a layer's
/// stacks ([`ffn_ring`]): its gate, up and down, each a stack of the half's
/// slots at the layer's own types and row words.
pub(super) struct RingStacks<'a> {
    pub(super) gate: &'a DeviceTensor<u32>,
    pub(super) up: &'a DeviceTensor<u32>,
    pub(super) down: &'a DeviceTensor<u32>,
}

/// Layer `r`'s stream on the card for the length of `f`: its union map of
/// `n_expert` words, and its half's gate, up and down ([`RingStacks`]),
/// `half_slots` experts of `rows[p]` rows of `cols[p]` words in each part
/// `p`; the windows are given back after `f` on every path.
pub(super) fn with_ring_windows<T>(
    r: &RingLayer,
    ctx: &Arc<CudaContext>,
    n_expert: usize,
    rows: [usize; 3],
    cols: [usize; 3],
    f: impl FnOnce(&DeviceBuffer<u32>, &RingStacks<'_>) -> Result<T, GpuError>,
) -> Result<T, GpuError> {
    // SAFETY: the stream's union row is `n_expert` words of its mapped host
    // rows, and its half holds `half_slots` experts of the layer's part bytes
    // in each part from the half's first slot — its gate and up rows of the
    // card stacks' words, its down rows likewise (`XStream::ring_layer`); all
    // stay allocated while the stream lives, past `f`'s launches, and the
    // windows are given back below.
    let (union_map, gate, up, down) = unsafe {
        (
            crate::tensor::window::<u32>(r.union_map, n_expert, ctx),
            DeviceTensor::<u32>::window(r.parts[0], r.half_slots * rows[0], cols[0], ctx),
            DeviceTensor::<u32>::window(r.parts[1], r.half_slots * rows[1], cols[1], ctx),
            DeviceTensor::<u32>::window(r.parts[2], r.half_slots * rows[2], cols[2], ctx),
        )
    };
    let ran = f(
        &union_map,
        &RingStacks {
            gate: &gate,
            up: &up,
            down: &down,
        },
    );
    drop(ManuallyDrop::into_inner(union_map).into_raw_parts());
    for w in [gate, up, down] {
        DeviceTensor::release(w);
    }
    ran
}

/// The ring's part of the apart arm of [`ffn`] over `m` rows, after
/// [`ffn_card`] and before [`ffn_combine`]: one GEMM batch of a layer's
/// stream — the routed table over the ring's half from the batch's map
/// (`map`: the stream's row, a streamed expert's slot in the half, [`HOST`]
/// for every other expert and for the shared expert's id, the routed count),
/// then the engine stream waiting the batch's landed event (`wait`), then
/// the routed slots the table lists through the half's stacks into the rows
/// [`ffn_card`]'s slots write. The SwiGLU runs every slot's column, so a
/// slot of the card's stacks gets its form again from its unchanged rows:
/// the same bits. Only K-quant stacks route; another type is refused by
/// name before the batch's GEMMs ([`site::gated`]).
#[allow(
    clippy::too_many_arguments,
    reason = "the layer's context, plan, arena and width, the batch's map, table and wait, and the half's stacks (rust-quality R8)"
)]
pub(super) fn ffn_ring(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    map: &DeviceBuffer<u32>,
    table: &mut GemmRoute,
    wait: Option<&CudaEvent>,
    ring: &RingStacks<'_>,
) -> Result<(), GpuError> {
    const WHAT_R: &str = "qwen3moe::wide::ffn_ring";
    let stream = c.gpu.stream();
    let d = s.dims;
    let slots = d.slots();
    let Arena {
        normed,
        route,
        h,
        down,
        wide,
        ..
    } = s;
    let Wide {
        act_hid, act_h, up, ..
    } = wide_of(wide, WHAT_R)?;
    c.k.q35(WHAT_R)?
        .g32
        .enqueue_route_remap(stream, route.ids(), map, m * slots, table, c.sink)?;
    if let Some(e) = wait {
        stream.wait(e)?;
    }
    let hid = act_hid.input(normed);
    let list = (&*table, GemmInput::Shared { top_k: slots });
    slots_gemm(c, (n, Some(ring)), &d, list, &hid, (act_h, h, up, down), m)
}

/// The slots' half of the FFN at `m` rows: the gated block over `table`
/// (each slot's column of `hid` as its input picks it) — gate and up, their
/// SwiGLU into the form the down's type reads, and the down over the same
/// table, slot by slot, into `down` — the `m · d.slots()` slots' SwiGLU
/// forms, gate (`h`), up and down rows. The stacks are the plan's `n`, or
/// with `ring` a half of the expert stream's ring at the plan's types. A
/// slot the table lists in no tile keeps its rows.
fn slots_gemm(
    c: &Ctx<'_>,
    (n, ring): (&FfnPlan, Option<&RingStacks<'_>>),
    d: &Dims,
    (table, input): (&GemmRoute, GemmInput),
    hid: &WideIn<'_>,
    (act_h, h, up, down): (
        &mut Acts,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
        &mut DeviceBuffer<f32>,
    ),
    m: usize,
) -> Result<(), GpuError> {
    let slots = Gated {
        w: c.w,
        route: table,
        gate: slot_site((n.gate_ty, &n.gate), ring.map(|r| r.gate)),
        up: slot_site((n.up_ty, &n.up), ring.map(|r| r.up)),
        down: slot_site((n.down_ty, &n.down), ring.map(|r| r.down)),
        ff: d.ff,
        hidden: d.hidden,
        rule: Act::SiluMul,
        input,
    };
    let io = GatedIo {
        x: hid,
        gate: h,
        up,
        rows: act_h.swiglu_rows(None),
        y: down,
    };
    site::gated(
        c.gpu,
        &wide_kernels(c.k),
        &slots,
        (m, m * d.slots()),
        io,
        c.sink,
    )
}

/// Site `name` of type `ty` as [`slots_gemm`]'s block reads its weight: the
/// resident stack, or with `window` the ring's words of the same type.
fn slot_site<'a>(
    (ty, name): (SiteTy, &'a str),
    window: Option<&'a DeviceTensor<u32>>,
) -> (SiteTy, SiteWeight<'a>) {
    let src = match window {
        Some(w) => SiteWeight::Words { name, w },
        None => SiteWeight::Resident(name),
    };
    (ty, src)
}

impl GpuModel<Body35> {
    /// Run layer `l`'s FFN half alone eagerly at the rows of `ffn_inp` (past
    /// [`GEMV_COLS`] rows, at most the ubatch arena's: the wide arm) on the
    /// ubatch arena, the unit's one-expert table first, and read back what it
    /// left: the router's logits, ids and weights and the output residual.
    /// Synchronizes; gate use.
    pub fn ffn_rows_wide(&mut self, l: usize, ffn_inp: &[f32]) -> Result<Ffn35Run, GpuError> {
        const WHAT: &str = "qwen35moe::ffn_rows_wide";
        let hidden = self.body(WHAT)?.hidden();
        let slot = self.layer_slot(l, WHAT)?;
        let (gpu, w, b) = self.body_parts(WHAT)?;
        let m = ffn_inp.len() / hidden.max(1);
        if m * hidden != ffn_inp.len() || m <= GEMV_COLS || m > b.u.rows {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "{} values: {}..={} rows of {hidden}",
                    ffn_inp.len(),
                    GEMV_COLS + 1,
                    b.u.rows
                ),
            ));
        }
        let stream = gpu.stream();
        // The window lives for this copy alone.
        WindowMut::<f32>::of_mut(&mut b.u.ffn_inp, 0, m * hidden)?
            .copy_from_host(stream, ffn_inp)?;
        let c = Ctx::new(
            gpu,
            w,
            (&b.plans[slot], slot),
            &b.k,
            b.mma,
            b.eps,
            &b.rope.table,
        )?;
        route_dense(&c, &mut b.u, m)?;
        ffn(&c, &b.plans[slot].ffn, &mut b.u, m, None)?;
        let Route::Gated(r) = &b.u.route else {
            return Err(GpuError::state(WHAT, "the gated router's buffers"));
        };
        let slots = b.u.dims.slots();
        let head = |v: &DeviceBuffer<f32>, n: usize| -> Result<Vec<f32>, GpuError> {
            let mut v = v.to_host_vec(stream)?;
            v.truncate(n);
            Ok(v)
        };
        let mut ids = r.ids.to_host_vec(stream)?;
        ids.truncate(m * slots);
        Ok(Ffn35Run {
            logits: head(&r.logits, m * r.dims().logits())?,
            ids,
            weights: head(&r.weights, m * slots)?,
            l_out: head(&b.u.x, m * hidden)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::GEMV_COLS;
    use crate::flash_gqa::{HEAD, HEAD_256, partials_ms_len, partials_v_len, partials_v_len_256};
    use model::placement::workstation::SCRATCH;

    /// The decode flash's partials a load's step arena (one row) and pass
    /// arena ([`GEMV_COLS`] rows) hold fit the m = 1 scratch a plan's card
    /// counts for them (`workstation::SCRATCH`), at either of the family's
    /// head layouts: Qwen3-30B's 32 heads of [`HEAD`], Qwen3.6's 16 of
    /// [`HEAD_256`]. The partials do not grow with the context, so no
    /// context term carries them.
    #[test]
    fn the_flash_partials_fit_the_planned_scratch() {
        let rows = 1 + GEMV_COLS;
        for (n_head, head) in [(32, HEAD), (16, HEAD_256)] {
            let v = if head == HEAD_256 {
                partials_v_len_256(rows, n_head)
            } else {
                partials_v_len(rows, n_head)
            };
            let bytes = (v + partials_ms_len(rows, n_head)) * std::mem::size_of::<f32>();
            assert!(
                bytes as u64 <= SCRATCH,
                "{n_head} heads of {head}: {bytes} B of partials past the {SCRATCH} B scratch"
            );
        }
    }
}
