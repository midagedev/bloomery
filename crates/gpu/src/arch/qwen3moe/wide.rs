//! The wide arm of each layer op: a unit of more than [`GEMV_COLS`] rows.
//! Each op of `dispatch` (and the delta mixer's input and output halves)
//! takes its first line from here when `m > GEMV_COLS` and keeps its gemv
//! body below it — one layer walk, the op choosing its kernel family from
//! `m`, as ik's `mul_mat` picks its matrix-vector kernel for at most eight
//! columns.
//!
//! A wide op reads each weight once for the unit: every projection runs
//! through the grouped int8 GEMM over a one-expert table ([`route_dense`],
//! filled once per unit after its embedding), the routed experts through one
//! table per layer (gate and up reading each token's column, down each
//! slot's own); the norms, the quantizers, the rope with the cache append,
//! the prefill flash, the router's two launches, the SwiGLU quantizer and
//! the combine run over the unit's rows. The conv, the delta step and the
//! gated norm are the gemv arm's own launches: they already take any row
//! count.
//!
//! Numeric class. A wide op computes a token's values from that token's
//! inputs alone — the GEMM accumulates each (slot, row) on its own, the
//! flash walks a row's keys in fixed tiles whatever the rows beside it — so
//! a token's bits do not depend on the unit it lands in. Against the gemv arm
//! the products sum in another order (128-value integer blocks into one f32
//! accumulator, where the gemv sums lane partials through a warp tree) and
//! the attention weighs the values with f16 weights, so the two arms agree
//! to the error of those sums and are not bit-equal.

use super::body::ATTN_SCALE_256;
use super::dispatch::Ctx;
use super::experts::CombineArgs;
use super::plan::{DeltaPlan, GqaKind, GqaPlan, Kq, MoePlan};
use super::router::MAX_TOKENS;
use super::scratch::{Arena, Dims, GdnArena, KvPlanes};
use crate::GpuError;
use crate::flash_gqa::{HEAD_256, partials_ms_len, partials_v_len, partials_v_len_256};
use crate::flash_gqa_prefill::GqaPrefillArgs;
use crate::gated_quant::GateLayout;
use crate::gemm::{GEMM_BN, GEMM_MAX_SLOTS, GemmAct, GemmArgs, GemmInput, GemmRoute, GemmWeight};
use crate::linear::{self, LinearShape};
use crate::model::MAX_PASS_ROWS;
use crate::model::lookup::{f32_gain, f32_tensor, kq_weight};
use crate::rope_neox::PartialNeoxArgs;
use cuda_core::{CudaStream, DeviceBuffer};

/// The most rows an op runs through its gemv arm — ik's matrix-vector cut
/// (`ne[1] <= 8`), and the rows of a captured pass. Past it, the wide arm.
pub(super) const GEMV_COLS: usize = MAX_PASS_ROWS;
const _: () = assert!(GEMV_COLS == MAX_TOKENS);

const WHAT: &str = "qwen3moe::wide";

/// The wide part of an arena of `rows > GEMV_COLS` rows: the GEMMs'
/// activations, the second SwiGLU operand, the output projection's rows and
/// the two route tables.
pub(super) struct Wide {
    /// q8_1 of the normed rows (`hidden` a token): the mixer's projections'
    /// input, then the experts' gate·up input.
    act_hid: GemmAct,
    /// q8_1 of the attention rows (or a delta layer's gated norm): the
    /// output projection's input.
    act_attn: GemmAct,
    /// q8_1 of the slots' SwiGLU, one column per slot: the down's input.
    act_h: GemmAct,
    /// The up rows, per slot `ff` values; the gate rows are the arena's `h`.
    up: DeviceBuffer<f32>,
    /// The output projection's rows; the residual add reads them.
    attn_o: DeviceBuffer<f32>,
    /// The one-expert table every projection of a unit reads.
    dense: GemmRoute,
    /// The layer's expert table over `t · slots` slots, `slots` the
    /// router's per token (a folded shared expert's among them), over the
    /// joined stacks' `logits()` experts.
    moe: GemmRoute,
}

impl Wide {
    /// The wide part for `rows` tokens of `d`, or a named refusal when their
    /// slots pass the GEMM's route table. Load-time only.
    pub(super) fn new(stream: &CudaStream, d: &Dims, rows: usize) -> Result<Wide, GpuError> {
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
            act_hid: GemmAct::new(stream, rows, d.hidden)?,
            act_attn: GemmAct::new(stream, rows, d.attn_len())?,
            act_h: GemmAct::new(stream, slots, d.ff)?,
            up: f(slots * d.ff)?,
            attn_o: f(rows * d.hidden)?,
            dense: GemmRoute::new(stream, rows, 1)?,
            moe: GemmRoute::new(stream, slots, d.router.logits())?,
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
            + self.moe.bytes()
    }
}

/// Bytes of one q8_1 activation column of `k` values, a `Q8Act`'s or a
/// `GemmAct`'s: the q3 pairs (u64), the q4 and q6 permutations, the 32-value
/// code sums and the 128-value scales.
fn act_col_bytes(k: usize) -> usize {
    let n_sb = k / 256;
    64 * n_sb.div_ceil(2) * 8
        + (256 * n_sb.div_ceil(4) + 128 * n_sb.div_ceil(2)) * 4
        + 10 * n_sb * 4
}

/// Bytes of a route table for `slots` slots over `experts` experts.
fn route_bytes(slots: usize, experts: usize) -> usize {
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

/// The device bytes `Arena::new(d, rows)` allocates, from `d` and `rows`
/// alone: what a load checks against the card's free bytes before it
/// allocates the arena, and what the arena it then allocates must hold.
pub(super) fn arena_bytes(d: &Dims, rows: usize) -> usize {
    let n = rows.min(GEMV_COLS);
    let (q_len, kv_len, att, slots) = (d.q_rows, d.kv_len(), d.attn_len(), d.slots());
    let part_v = if d.head == HEAD_256 {
        partials_v_len_256(n, d.n_head, d.ctx)
    } else {
        partials_v_len(n, d.n_head, d.ctx)
    };
    let f32s = 3 * rows * d.hidden
        + rows * (q_len + 2 * kv_len)
        + if q_len == att { 0 } else { rows * att }
        + if rows > 1 { n * kv_len } else { 0 }
        + part_v
        + partials_ms_len(n, d.n_head, d.ctx)
        + rows * att
        + rows * slots * (d.ff + d.hidden);
    let u32s = 2 * rows;
    let acts: usize = (1..=n)
        .map(|m| {
            m * (2 * act_col_bytes(d.hidden) + act_col_bytes(att)) + m * slots * act_col_bytes(d.ff)
        })
        .sum();
    let r = d.router;
    let route = (rows * (r.logits() + r.experts() + 2 * slots) + 1) * 4;
    let wide = if rows > GEMV_COLS {
        let s = rows * slots;
        rows * (act_col_bytes(d.hidden) + act_col_bytes(att))
            + s * act_col_bytes(d.ff)
            + (s * d.ff + rows * d.hidden) * 4
            + route_bytes(rows, 1)
            + route_bytes(s, r.logits())
    } else {
        0
    };
    (f32s + u32s) * 4 + acts + route + d.lin.map_or(0, |s| gdn_bytes(s, rows)) + wide
}

/// The GEMM's type for a projection of `kq`.
fn gemm_ty(kq: Kq) -> GemmWeight {
    match kq {
        Kq::Q4K => GemmWeight::Q4K,
        Kq::Q6K => GemmWeight::Q6K,
    }
}

/// The arena's wide part, or a named refusal (an arena of at most
/// [`GEMV_COLS`] rows has none).
fn wide_of<'a>(w: &'a mut Option<Wide>, what: &'static str) -> Result<&'a mut Wide, GpuError> {
    w.as_mut().ok_or(GpuError::state(
        what,
        "the arena's wide part (an arena of more than GEMV_COLS rows)",
    ))
}

/// Enqueue `y = W · x` for `rows` rows of weight `name` (of type `kq`) over
/// the columns `route` holds, `input` picking each slot's column of `act`.
fn gemm(
    c: &Ctx<'_>,
    (kq, name): (Kq, &str),
    rows: usize,
    act: &GemmAct,
    route: &GemmRoute,
    input: GemmInput,
    y: &mut DeviceBuffer<f32>,
) -> Result<(), GpuError> {
    c.k.gemm.enqueue_gemm(
        c.gpu.stream(),
        GemmArgs {
            ty: gemm_ty(kq),
            w: kq_weight(c.w, name)?,
            rows_per_expert: rows,
            act,
            route,
            input,
            y,
        },
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

/// The attention half at `m` rows: `x` in, `ffn_inp = x + W_o · (flash ⊙
/// σ(gate))` out, the unit's K/V rows appended to the layer's planes at the
/// rows' positions. Qwen3's head-128 attention takes its wide rows through
/// `ubatch.rs` and is refused here by name.
pub(super) fn attention(
    c: &Ctx<'_>,
    n: &GqaPlan,
    kv: &mut KvPlanes,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT_ATTN: &str = "qwen3moe::wide::attention";
    if n.kind != GqaKind::Gated256 {
        return Err(GpuError::shape(
            WHAT_ATTN,
            format!(
                "layer {}: head-128 attention at {m} rows (Qwen3 runs more than GEMV_COLS rows \
                 through ubatch.rs)",
                c.layer
            ),
        ));
    }
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
    gpu.enqueue_quantize_gemm(normed, m, &mut wd.act_hid, c.sink)?;
    let dense = GemmInput::PerSlot;
    gemm(
        c,
        (Kq::Q4K, &n.attn_q),
        d.q_rows,
        &wd.act_hid,
        &wd.dense,
        dense,
        q,
    )?;
    gemm(
        c,
        (Kq::Q4K, &n.attn_k),
        d.kv_len(),
        &wd.act_hid,
        &wd.dense,
        dense,
        kr,
    )?;
    gemm(
        c,
        (n.v_ty, &n.attn_v),
        d.kv_len(),
        &wd.act_hid,
        &wd.dense,
        dense,
        v,
    )?;
    k.neox.enqueue_head_norm_neox_append_256(
        stream,
        PartialNeoxArgs {
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
            cache_k: &mut kv.k,
            cache_v: &mut kv.v,
        },
    )?;
    k.prefill.enqueue_256(
        stream,
        GqaPrefillArgs {
            q: q_out,
            kc: &kv.k,
            vc: &kv.v,
            n_keys,
            scale: ATTN_SCALE_256,
            n_head: d.n_head,
            n_kv: d.n_kv,
            ctx: d.ctx,
            t: m,
            fault: c.sink,
            y: attn,
        },
    )?;
    q35.gated.enqueue_gemm(
        stream,
        (attn, q),
        GateLayout {
            head: d.head,
            head_stride: 2 * d.head,
            offset: d.head,
            col_stride: d.q_rows,
        },
        &mut wd.act_attn,
        m,
        c.sink,
    )?;
    let Wide {
        act_attn,
        dense: table,
        attn_o,
        ..
    } = wd;
    gemm(
        c,
        (Kq::Q4K, &n.attn_output),
        d.hidden,
        act_attn,
        table,
        dense,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, m * d.hidden, ffn_inp)
}

/// A delta layer's input half at `m` rows: `x` normed and quantized, then
/// its four projections — `attn_qkv`, `attn_gate`, β and α — into the
/// arena's `[x | z | b | a]` blocks, each a GEMM of its own.
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
    gpu.enqueue_quantize_gemm(normed, m, &mut wd.act_hid, c.sink)?;
    let (a, t, p) = (&wd.act_hid, &wd.dense, GemmInput::PerSlot);
    let (ch, zl, nv) = (d.shape.channels(), d.shape.n_v * linear::HEAD, d.shape.n_v);
    gemm(c, (d.qkv_ty, &d.qkv), ch, a, t, p, &mut g.x)?;
    gemm(c, (Kq::Q4K, &d.gate), zl, a, t, p, &mut g.z)?;
    gemm(c, (Kq::Q4K, &d.beta), nv, a, t, p, &mut g.b)?;
    gemm(c, (Kq::Q4K, &d.alpha), nv, a, t, p, &mut g.a)
}

/// A delta layer's output half at `m` rows: the gated norm's rows (the
/// arena's `attn`) quantized, `ssm_out`, then `ffn_inp = x + ssm_out(·)`.
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
    gpu.enqueue_quantize_gemm(attn, m, &mut wd.act_attn, c.sink)?;
    let Wide {
        act_attn,
        dense,
        attn_o,
        ..
    } = wd;
    gemm(
        c,
        (Kq::Q4K, &d.ssm_out),
        hidden,
        act_attn,
        dense,
        GemmInput::PerSlot,
        attn_o,
    )?;
    gpu.elem()
        .enqueue_add(stream, x, attn_o, m * hidden, ffn_inp)
}

/// The routed FFN half at `m` rows: `ffn_inp` in, `ffn_inp + Σ_s w_s ·
/// down_s(swiglu(gate_s, up_s))` over each token's slots out, into `out`
/// (`None`: into `x`). The gated router's slots only (the shared expert the
/// last of each token's, its id the joined stacks' last expert); Qwen3's
/// plain router takes its wide rows through `ubatch.rs` and is refused here
/// by name.
pub(super) fn ffn(
    c: &Ctx<'_>,
    n: &MoePlan,
    s: &mut Arena,
    m: usize,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<(), GpuError> {
    const WHAT_FFN: &str = "qwen3moe::wide::ffn";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let d = s.dims;
    let slots = d.slots();
    if n.shared.is_none() {
        return Err(GpuError::shape(
            WHAT_FFN,
            format!(
                "layer {}: the plain router at {m} rows (Qwen3 runs more than GEMV_COLS rows \
                 through ubatch.rs)",
                c.layer
            ),
        ));
    }
    if slots != c.p.slots(d.router.used()) {
        return Err(GpuError::shape(
            WHAT_FFN,
            format!(
                "layer {}: the arena is cut for {slots} slots a token, the plan routes {}",
                c.layer,
                c.p.slots(d.router.used())
            ),
        ));
    }
    let q35 = k.q35(WHAT_FFN)?;
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
    gpu.elem().enqueue_rms_norm(
        stream,
        ffn_inp,
        f32_gain(w, &n.ffn_norm)?,
        c.eps,
        d.hidden,
        m,
        normed,
    )?;
    gpu.enqueue_quantize_gemm(normed, m, &mut wd.act_hid, c.sink)?;
    q35.router.enqueue_ubatch(
        stream,
        f32_tensor(w, &n.ffn_gate_inp)?,
        normed,
        m,
        c.sink,
        route.gated(WHAT_FFN)?,
    )?;
    k.gemm
        .enqueue_route(stream, route.ids(), m * slots, &mut wd.moe, c.sink)?;
    let shared = GemmInput::Shared { top_k: slots };
    let (a, t) = (&wd.act_hid, &wd.moe);
    gemm(c, (Kq::Q4K, &n.ffn_gate_exps), d.ff, a, t, shared, h)?;
    gemm(c, (Kq::Q4K, &n.ffn_up_exps), d.ff, a, t, shared, &mut wd.up)?;
    k.gemm
        .enqueue_swiglu_quant(stream, h, &wd.up, m * slots, &mut wd.act_h, c.sink)?;
    gemm(
        c,
        (n.down_ty, &n.ffn_down_exps),
        d.hidden,
        &wd.act_h,
        &wd.moe,
        GemmInput::PerSlot,
        down,
    )?;
    let y = match out {
        Some(y) => y,
        None => x,
    };
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
