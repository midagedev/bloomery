//! The wide arm of each layer op: a unit of more than [`GEMV_COLS`] rows.
//! Each op the chain's walk runs (attention's input, middle and output, the
//! delta mixer's input and output, the FFN's route and experts) takes its
//! first line from here when `m > GEMV_COLS` and keeps its gemv body below
//! it — one walk for every unit, the op choosing its kernel family from `m`,
//! as ik's `mul_mat` picks its matrix-vector kernel for at most eight
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
//! ([`GqaKind::Neox128`], [`GqaKind::Head128`]), whose three parts run over
//! buffer views ([`Neox128Rows`]), so the arena of this arm and the
//! whole-card GEMM prefill's (`ubatch.rs`, through [`neox_128_rows`]) both
//! lend them their rows.
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
use super::dispatch::{Ctx, PassCtx};
use super::experts::CombineArgs;
use super::hostleg::{HostCombineArgs, HostLegKernels};
use super::plan::{DeltaPlan, FfnPlan, FfnRoute, GqaKind, GqaPlan, Pairs, SharedPlan, SiteTy};
use super::program::{Program, Tail};
use super::scratch::{
    ApartForms, Append128, Append256, AppendNorm, Arena, Dims, Forms, GdnArena, KvPlanes,
    PrefillFlash, Route,
};
use super::taps35::Ffn35Run;
use crate::flash_gqa::{HEAD, HEAD_256, partials_ms_len, partials_v_len, partials_v_len_256};
use crate::gated_quant::GateLayout;
use crate::gemm::{GEMM_MAX_SLOTS, Gemm32Kernels, GemmInput, GemmRoute};
pub(super) use crate::host::xstream::{RingStacks, with_ring_windows};
use crate::hybrid::HOST;
use crate::kquant::Act;
use crate::linear::{self, LinearShape};
use crate::model::fit::{act_col_bytes, route_bytes};
use crate::model::lookup::{f32_gain, f32_tensor};
use crate::model::{GpuModel, MAX_PASS_ROWS};
use crate::q38::OutGateArgs;
use crate::route_softmax::MAX_TOKENS;
use crate::site::{self, Gated, GatedIo, SiteWeight, WideActs, WideActsMut, WideIn, WideKernels};
use crate::tensor::{DeviceTensor, WindowMut};
use crate::weights::Weights;
use crate::{FaultSink, Gpu, GpuError};
use chain::arch::qwen3moe::walk;
use cuda_core::{CudaEvent, CudaStream, DeviceBuffer};

/// The most rows an op runs through its gemv arm — ik's matrix-vector cut
/// (`ne[1] <= 8`), and the rows of a captured pass. Past it, the wide arm.
pub(super) const GEMV_COLS: usize = MAX_PASS_ROWS;
const _: () = assert!(GEMV_COLS == MAX_TOKENS);

const WHAT: &str = "qwen3moe::wide";

/// The wide part of an arena of `rows > GEMV_COLS` rows: the GEMMs'
/// activations, the second SwiGLU operand, the output projection's rows and
/// the two route tables.
pub(super) struct Wide {
    /// The normed rows (`hidden` a token): the mixer's projections' input,
    /// then the experts' gate·up input.
    act_hid: WideActs,
    /// The attention rows (or a delta layer's gated norm): the output
    /// projection's input.
    act_attn: WideActs,
    /// The slots' SwiGLU, one column per slot: the down's input.
    act_h: WideActs,
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
/// list the combine reads, the shared expert's gate and up rows, its
/// SwiGLU's forms its downs read (the f32 rows among them), its output and
/// the zero host sum (`scratch::ApartRows`, the gemv arm's twin).
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
    act_h: WideActs,
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
            act_h: WideActs::new(stream, forms.h, rows, d.ff)?,
            y: f(rows * d.hidden)?,
            zero: f(rows * d.hidden)?,
        })
    }

    fn bytes(&self) -> usize {
        [&self.g, &self.u, &self.y, &self.zero]
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
            act_hid: WideActs::new(stream, forms.hid, rows, d.hidden)?,
            act_attn: WideActs::new(stream, forms.attn, rows, d.attn_len())?,
            act_h: WideActs::new(stream, forms.h, slots, d.ff)?,
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
        + if forms.out_gate { rows * att } else { 0 }
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
        forms.hid.bytes(rows, d.hidden)
            + forms.attn.bytes(rows, att)
            + forms.h.bytes(s, d.ff)
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
                (s + r.experts() + 1 + 2 * rows * d.ff + 2 * rows * d.hidden) * 4
                    + a.h.bytes(rows, d.ff)
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

/// The attention half's input side at `m` rows (`dispatch::attn_in`'s wide
/// arm): `x` RMS-normed into `normed`, its forms the q·k·v sites read, and
/// the three projections, each the GEMM of its type over the unit's
/// one-expert table; the plan's kind picks the half ([`gated_256_in`],
/// [`neox_128_in`] over the arena's views).
pub(super) fn attn_in(c: &Ctx<'_>, n: &GqaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    match n.kind {
        GqaKind::Gated256 => gated_256_in(c, n, s, m),
        GqaKind::Neox128 | GqaKind::Head128 { .. } => {
            let d = s.dims;
            let env = (c.gpu, c.w, c.k);
            neox_128_in(env, (c.eps, c.sink), n, (&d, m), &mut neox_rows(s)?)
        }
        GqaKind::HeadGate128 => Err(head_gate_refused()),
    }
}

/// The attention half's middle at `m` rows (`dispatch::attn_core`'s wide
/// arm), after [`attn_in`]: the q/k norm and rope with the unit's K/V rows
/// appended to the layer's planes at the rows' positions, the prefill flash,
/// and the output projection's input — the flash rows (times σ(gate) at head
/// 256) in the forms that projection reads.
pub(super) fn attn_core(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    match n.kind {
        GqaKind::Gated256 => gated_256_core(c, n, kv, s, m),
        GqaKind::Neox128 | GqaKind::Head128 { .. } => {
            let d = s.dims;
            let env = (c.gpu, c.w, c.k);
            let at = (c.table, c.eps, c.sink);
            neox_128_core(env, at, (n, kv), (&d, m), &mut neox_rows(s)?)
        }
        GqaKind::HeadGate128 => Err(head_gate_refused()),
    }
}

/// The attention half's output side at `m` rows (`dispatch::attn_out`'s
/// wide arm), after [`attn_core`]: the output projection's GEMM, then the
/// residual add, `ffn_inp = x + W_o · rows`.
pub(super) fn attn_out(c: &Ctx<'_>, n: &GqaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    match n.kind {
        GqaKind::Gated256 => gated_256_out(c, n, s, m),
        GqaKind::Neox128 | GqaKind::Head128 { .. } => {
            let d = s.dims;
            neox_128_out((c.gpu, c.w, c.k), n, (&d, m), &mut neox_rows(s)?)
        }
        GqaKind::HeadGate128 => Err(head_gate_refused()),
    }
}

const WHAT_GATED: &str = "qwen3moe::wide::attention";

/// The refusal of the head-128 attention with an output gate of its own, which no launch here
/// serves.
fn head_gate_refused() -> GpuError {
    GpuError::shape(
        WHAT_GATED,
        "a head-128 attention with an output gate of its own, which no launch here serves",
    )
}

/// The head-256 half's input side ([`GqaKind::Gated256`], [`attn_in`]): the
/// `[q | gate]` rows, k and v of the normed rows.
fn gated_256_in(c: &Ctx<'_>, n: &GqaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    let d = s.dims;
    let Arena {
        x,
        normed,
        q,
        k: kr,
        v,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_GATED)?;
    c.gpu.elem().enqueue_rms_norm(
        c.gpu.stream(),
        x,
        f32_gain(c.w, &n.attn_norm)?,
        c.eps,
        d.hidden,
        m,
        normed,
    )?;
    wd.act_hid.quantize(
        c.gpu,
        &wide_kernels(c.k),
        c.sink,
        normed,
        m,
        &[n.q_ty, n.k_ty, n.v_ty],
    )?;
    let table = (&wd.dense, GemmInput::PerSlot);
    let hid = wd.act_hid.input(normed);
    gemm(c, (n.q_ty, &n.attn_q), d.q_rows, &hid, table, m, q)?;
    gemm(c, (n.k_ty, &n.attn_k), d.kv_len(), &hid, table, m, kr)?;
    gemm(c, (n.v_ty, &n.attn_v), d.kv_len(), &hid, table, m, v)
}

/// The head-256 half's middle ([`attn_core`]): the q/k norm and the turn of
/// the first 64 values with the cache append, the queries read out of the
/// `[q | gate]` rows into their own buffer; the prefill flash; then the flash
/// rows times σ(gate) — quantized in one launch for a K-quant `attn_output`,
/// for any other type the out gate into the arena's gated rows (`q_out` keeps
/// the queries the flash read), quantized as that projection takes them.
fn gated_256_core(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let q35 = k.q35(WHAT_GATED)?;
    let stream = gpu.stream();
    let d = s.dims;
    let Arena {
        pos,
        n_keys,
        q,
        k: kr,
        v,
        q_out,
        gated_rows,
        attn,
        wide,
        ..
    } = s;
    let wd = wide_of(wide, WHAT_GATED)?;
    let q_out = q_out.as_mut().ok_or(GpuError::state(
        WHAT_GATED,
        "the arena's buffer of gated queries",
    ))?;
    // SAFETY: the layer program embeds consecutive positions before attention;
    // saturation is outside the cache. Each call uses one sequence's row window
    // and its own cache planes, not the combined positions of different slots.
    unsafe {
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
        )
    }?;
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
    let act_attn = &mut wd.act_attn;
    if n.o_ty.kquant() {
        let a = act_attn.q128.as_mut().ok_or(GpuError::state(
            WHAT_GATED,
            "the wide part's q8_1 blocks of the attention rows",
        ))?;
        return q35.gated.enqueue_gemm(
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
        );
    }
    let gated = gated_rows
        .as_mut()
        .ok_or(GpuError::state(WHAT_GATED, "the arena's gated rows"))?;
    q35.q38.enqueue_out_gate(
        stream,
        OutGateArgs {
            attn,
            qg: q,
            n_head: d.n_head,
            m,
            fault: c.sink,
            y: gated,
        },
    )?;
    act_attn.quantize(gpu, &wide_kernels(k), c.sink, gated, m, &[n.o_ty])
}

/// The head-256 half's output side ([`attn_out`]): `ffn_inp = x + W_o ·
/// (flash ⊙ σ(gate))`, the projection reading the gated rows' forms
/// [`gated_256_core`] wrote — the flash rows' for a K-quant, the arena's
/// gated rows' for any other type.
fn gated_256_out(c: &Ctx<'_>, n: &GqaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    let d = s.dims;
    let Arena {
        x,
        attn,
        gated_rows,
        ffn_inp,
        wide,
        ..
    } = s;
    let Wide {
        act_attn,
        dense,
        attn_o,
        ..
    } = wide_of(wide, WHAT_GATED)?;
    let gated: &DeviceBuffer<f32> = if n.o_ty.kquant() {
        attn
    } else {
        gated_rows
            .as_ref()
            .ok_or(GpuError::state(WHAT_GATED, "the arena's gated rows"))?
    };
    gemm(
        c,
        (n.o_ty, &n.attn_output),
        d.hidden,
        &act_attn.input(gated),
        (dense, GemmInput::PerSlot),
        m,
        attn_o,
    )?;
    c.gpu
        .elem()
        .enqueue_add(c.gpu.stream(), x, attn_o, m * d.hidden, ffn_inp)
}

/// The arena's buffers as a head-128 attention half's views
/// ([`GqaKind::Neox128`], [`GqaKind::Head128`], [`Neox128Rows`]).
fn neox_rows(s: &mut Arena) -> Result<Neox128Rows<'_>, GpuError> {
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
    } = wide_of(wide, "qwen3moe::wide::neox_128")?;
    Ok(Neox128Rows {
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
    })
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
    pub(super) hid: WideActsMut<'a>,
    pub(super) out: WideActsMut<'a>,
    pub(super) dense: &'a GemmRoute,
}

/// The head-128 attention half at `t` rows over `b`'s views: `x` in, `ffn_inp
/// = x + attn_output(attn(x))` out; the unit's K/V rows appended to the
/// layer's planes at the rows' positions — [`neox_128_in`],
/// [`neox_128_core`], [`neox_128_out`] in order. The whole-card GEMM
/// prefill's half (`ubatch.rs`, whose walk is its own); the wide arm's ops run
/// the three parts.
pub(super) fn neox_128_rows(
    env: (&Gpu, &Weights, &Kernels),
    (table, eps, sink): (&DeviceBuffer<f32>, f32, FaultSink),
    (n, kv): (&GqaPlan, &mut KvPlanes),
    (d, t): (&Dims, usize),
    mut b: Neox128Rows<'_>,
) -> Result<(), GpuError> {
    neox_128_in(env, (eps, sink), n, (d, t), &mut b)?;
    neox_128_core(env, (table, eps, sink), (n, kv), (d, t), &mut b)?;
    neox_128_out(env, n, (d, t), &mut b)
}

/// The head-128 half's input side at `t` rows over `b`'s views: `x` normed
/// and quantized into the forms its projections read, then q, k and v, each
/// the launch its type picks ([`gemm_site`]) over the one-expert table. A
/// head of other than [`HEAD`] values is refused by name.
fn neox_128_in(
    (gpu, w, k): (&Gpu, &Weights, &Kernels),
    (eps, sink): (f32, FaultSink),
    n: &GqaPlan,
    (d, t): (&Dims, usize),
    b: &mut Neox128Rows<'_>,
) -> Result<(), GpuError> {
    if d.head != HEAD {
        return Err(GpuError::shape(
            "qwen3moe::wide::neox_128_rows",
            format!("a head of {} values in a head-{HEAD} attention", d.head),
        ));
    }
    let (q_len, kv_len) = (d.n_head * HEAD, d.n_kv * HEAD);
    gpu.elem().enqueue_rms_norm(
        gpu.stream(),
        b.x,
        f32_gain(w, &n.attn_norm)?,
        eps,
        d.hidden,
        t,
        b.normed,
    )?;
    b.hid.quantize(
        gpu,
        &wide_kernels(k),
        sink,
        b.normed,
        t,
        WideActs::wants_of(&[n.q_ty, n.k_ty, n.v_ty]),
    )?;
    let input = b.hid.input(b.normed);
    for (name, ty, rows, y) in [
        (&n.attn_q, n.q_ty, q_len, &mut *b.q),
        (&n.attn_k, n.k_ty, kv_len, &mut *b.k),
        (&n.attn_v, n.v_ty, kv_len, &mut *b.v),
    ] {
        let one = (b.dense, GemmInput::PerSlot);
        gemm_site((gpu, w, k), (ty, name), rows, &input, one, t, y)?;
    }
    Ok(())
}

/// The head-128 half's middle at `t` rows over `b`'s views: the turn with the
/// K/V rows appended to the layer's planes, the prefill flash at the layer's
/// layout (`n.flash`), and the flash rows quantized into the forms the output
/// projection reads. The turn follows the plan's kind: the head norm with the
/// NEOX turn ([`GqaKind::Neox128`]) or the NORM turn with no head norm
/// ([`GqaKind::Head128`]); any other kind is refused by name.
fn neox_128_core(
    (gpu, w, k): (&Gpu, &Weights, &Kernels),
    (table, eps, sink): (&DeviceBuffer<f32>, f32, FaultSink),
    (n, kv): (&GqaPlan, &mut KvPlanes),
    (d, t): (&Dims, usize),
    b: &mut Neox128Rows<'_>,
) -> Result<(), GpuError> {
    const WHAT_N: &str = "qwen3moe::wide::neox_128_core";
    let stream = gpu.stream();
    match n.kind {
        GqaKind::Neox128 => {
            // SAFETY: the sequence uploads consecutive positions for its rows; different sequences use disjoint planes.
            unsafe {
                kv.append_128(
                    &k.neox,
                    stream,
                    Append128 {
                        q: &mut *b.q,
                        k: &mut *b.k,
                        v: &*b.v,
                        gq: f32_gain(w, &n.attn_q_norm)?,
                        gk: f32_gain(w, &n.attn_k_norm)?,
                        table,
                        pos: b.pos,
                        eps,
                        n_head: d.n_head,
                        n_kv: d.n_kv,
                        ctx: d.ctx,
                        m: t,
                        fault: sink,
                    },
                )
            }?
        }
        GqaKind::Head128 {
            pairs: Pairs::Norm,
            head_norm: false,
        } => kv.append_norm(
            &k.q35(WHAT_N)?.norm,
            stream,
            AppendNorm {
                q: &mut *b.q,
                k: &mut *b.k,
                v: &*b.v,
                table,
                pos: b.pos,
                n_head: d.n_head,
                n_kv: d.n_kv,
                ctx: d.ctx,
                m: t,
                fault: sink,
            },
        )?,
        other => {
            return Err(GpuError::shape(
                WHAT_N,
                format!("the {other:?} kind in a head-128 attention half"),
            ));
        }
    }
    kv.prefill_head128(
        &k.prefill,
        stream,
        PrefillFlash {
            q: &*b.q,
            n_keys: b.n_keys,
            scale: ATTN_SCALE,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            t,
            fault: sink,
            y: &mut *b.attn,
        },
        n.flash,
    )?;
    b.out.quantize(
        gpu,
        &wide_kernels(k),
        sink,
        b.attn,
        t,
        WideActs::wants_of(&[n.o_ty]),
    )
}

/// The head-128 half's output side at `t` rows over `b`'s views: the output
/// projection over the flash rows' forms, then the residual add into
/// `ffn_inp`.
fn neox_128_out(
    (gpu, w, k): (&Gpu, &Weights, &Kernels),
    n: &GqaPlan,
    (d, t): (&Dims, usize),
    b: &mut Neox128Rows<'_>,
) -> Result<(), GpuError> {
    let input = b.out.input(b.attn);
    gemm_site(
        (gpu, w, k),
        (n.o_ty, &n.attn_output),
        d.hidden,
        &input,
        (b.dense, GemmInput::PerSlot),
        t,
        b.attn_o,
    )?;
    gpu.elem()
        .enqueue_add(gpu.stream(), b.x, b.attn_o, t * d.hidden, b.ffn_inp)
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
    wd.act_hid.quantize(
        c.gpu,
        &wide_kernels(c.k),
        c.sink,
        normed,
        m,
        &[d.qkv_ty, d.gate_ty, d.beta_ty, d.alpha_ty],
    )?;
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
    wd.act_attn
        .quantize(c.gpu, &wide_kernels(c.k), c.sink, attn, m, &[d.out_ty])?;
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

/// The FFN's front at `m` rows (`dispatch::route`'s wide arm): [`ffn_front`]
/// on a layer the whole-card wide arm runs. A plain router (no shared
/// expert) takes its wide rows through the whole-card GEMM prefill
/// (`ubatch.rs`) or a placed load's walk (the three parts with the card route
/// and host sums of `placed`); this whole-card entry refuses it by name.
pub(super) fn route(c: &Ctx<'_>, n: &FfnPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
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
    ffn_front(c, n, s, m)
}

/// The FFN's rest at `m` rows after [`route`] (`dispatch::experts`'s wide
/// arm): `ffn_inp + Σ_s w_s · down_s(swiglu(gate_s, up_s))` over each token's
/// slots, into `out` (`None`: into `x`). The gated router's slots (the shared
/// expert the last of each token's, its id the joined stacks' last expert)
/// through the layer's expert table, or a dense FFN's one slot a token
/// through the unit's one-expert table at the arena's fixed weight 1; the
/// SwiGLU's quantizer is the one the down's type reads. A layer that keeps
/// its shared expert apart ([`FfnPlan::apart`]) routes its routed slots
/// through a table over the file's routed stacks that lists no shared slot,
/// runs the shared expert's gate, up, SwiGLU and down on the unit's
/// one-expert table at their own types, and adds it in the host-sum combine
/// (`hostleg`): `((routed + 0) + resid) + σ · shared`, the host's routed sum
/// zero on a whole-card load — [`card_list`], [`ffn_card`], [`ffn_combine`].
pub(super) fn experts(
    c: &Ctx<'_>,
    n: &FfnPlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    if n.apart().is_none() {
        return ffn_joined(c, n, s, m, out);
    }
    card_list(c, s, m)?;
    ffn_card(c, n, s, m, None)?;
    ffn_combine(c, n, s, m, None, out)
}

/// The FFN's front at `m` rows ([`route`]): `ffn_inp` RMS-normed into `normed`,
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
    wd.act_hid
        .quantize(c.gpu, &wide_kernels(c.k), c.sink, normed, m, &hid_tys)?;
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

/// The rest of [`experts`] for a layer whose shared expert is folded into the
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

/// The slots' card list the combine reads ([`experts`]' apart arm) for the
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
/// its gate and up rows, the SwiGLU's forms its down reads, and its output.
struct SharedRows<'a> {
    g: &'a mut DeviceBuffer<f32>,
    u: &'a mut DeviceBuffer<f32>,
    act_h: &'a mut WideActs,
    y: &'a mut DeviceBuffer<f32>,
}

/// The card's part of the apart arm of [`experts`] over `m` rows, after
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
            act_h: sh_act,
            y,
            ..
        } = a;
        let shared = SharedRows {
            g,
            u,
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
        rows: sh_act.swiglu_rows(),
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

/// The ring's part of the apart arm of [`experts`] over `m` rows, after
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
        &mut WideActs,
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
        rows: act_h.swiglu_rows(),
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
        let pc = PassCtx {
            gpu,
            w,
            plans: &b.plans,
            k: &b.k,
            mma: b.mma,
            eps: b.eps,
            table: &b.rope.table,
        };
        // The FFN half reads no input record: the step's stands in.
        Program {
            c: &pc,
            stores: b.stores.as_mut_slice(),
            s: &mut b.u,
            io: &b.sp.io(),
            m,
            tail: Tail::Pass,
        }
        .alone(slot, walk::ffn)?;
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
    use super::{Dims, FfnRoute, Forms, GEMV_COLS, GqaKind, SiteTy, arena_bytes};
    use crate::arch::qwen3moe::plan::fixtures::layer;
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

    /// A dense layer's wide arena holds the forms its sites read on the wide arm
    /// ([`crate::site::reads_wide`]): an NVFP4 FFN reads the q8 blocks of 32, so beside Q8_0
    /// attention its arena is a Q8_0 FFN's, and a K-quant FFN's, which also reads the q8_1 blocks
    /// of 128, is larger. The bytes are what a plan counts for the card (`arena_bytes`).
    #[test]
    fn a_dense_nvfp4_ffn_holds_the_q32_forms_its_wide_sites_read() {
        let d = Dims {
            hidden: 5120,
            n_head: 32,
            n_kv: 4,
            head: HEAD,
            q_rows: 32 * HEAD,
            ff: 17408,
            router: None,
            lin: None,
            ctx: 4096,
        };
        let bytes = |ffn: SiteTy| {
            let mut p = layer(GqaKind::Neox128, [SiteTy::Q8_0; 4]);
            p.ffn.route = FfnRoute::Dense;
            [p.ffn.gate_ty, p.ffn.up_ty, p.ffn.down_ty] = [ffn; 3];
            arena_bytes(&d, 2 * GEMV_COLS, Forms::of(&[p], &d))
        };
        assert_eq!(bytes(SiteTy::Nvfp4), bytes(SiteTy::Q8_0));
        assert!(bytes(SiteTy::Q5K) > bytes(SiteTy::Nvfp4));
    }
}
