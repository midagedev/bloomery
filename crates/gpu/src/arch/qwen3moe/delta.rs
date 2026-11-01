//! The gated-delta-rule mixer of a Qwen3.6 layer at `m` rows: `x` in,
//! `ffn_inp = x + W_out · (RMSNorm(o) ⊙ SiLU(z))` out, the layer's
//! recurrent store read and written in place.
//!
//! The launches, in order:
//! 1. the fused norm and q8_1 quantizer of `x` (`attn_norm`);
//! 2. the four input projections into the arena's `[x | z | b | a]`
//!    blocks: with Q4_K `attn_gate`, β and α in two launches — a Q4_K
//!    `attn_qkv` beside `attn_gate` in one three-matrix launch and β with α
//!    in another; a Q6_K `attn_qkv` in its own gemv (and at more than one
//!    row its token-major copy) and `attn_gate`, β and α in one launch —
//!    and with any other types each projection by its type
//!    (`dispatch::site_gemv`); past [`GEMV_COLS`] rows `attn_gate` waits
//!    for 5;
//! 3. the conv and prep ([`crate::linear::conv`]) over the layer's conv ring
//!    by the rows' positions;
//! 4. the delta step over the store's lane the record's lane word names
//!    (`linear::seqs::enqueue_gdn`): in place for a step, a pass and a
//!    pass-path prompt unit, the chunked prompt form for a wide prompt unit
//!    (past [`GEMV_COLS`] rows, its work in the q·k·v rows the conv read, its
//!    last chunk kept in the carry on a closing call's last unit,
//!    `Arena::keep_last`), a
//!    verify recorded into the store's record on a pass of a load
//!    that verifies ([`GdnArena::verify`](super::scratch::GdnArena));
//! 5. the gated norm ([`crate::linear::norm_gate`]) into the arena's
//!    attention rows; past [`GEMV_COLS`] rows with `attn_gate` before it and
//!    the rows' forms `ssm_out` reads, through the shared site owner
//!    (`wide::delta_gate`): a norm whose reader takes one q8 form writes it;
//! 6. a Q4_K output projection: the rows' q8_1 quantizer and the
//!    projection with the residual add; another K-quant: the quantizer, the
//!    projection and the add; any other type: the projection on the f32 rows
//!    and the add (`dispatch::out_resid`).
//!
//! Every launch computes each row the way its one-row launch does — the
//! projections take the one-column body per column, the conv and the delta
//! step walk the rows in order through the ring and the state — so a pass
//! of `m` rows leaves the state, the ring and the outputs of `m` one-row
//! steps.
//!
//! The chain's walk runs them as three ops: [`delta_in`] (1–2),
//! [`delta_core`] (3–5) and [`delta_out`] (6).
//!
//! Past [`GEMV_COLS`] rows the input half (1–2) and the output half (6) are
//! the wide arm's (`wide::delta_in`, `wide::delta_out`: the norm and the
//! quantizer, each projection a GEMM, the residual add a launch of its own);
//! the conv and the delta step (3–4) are the same launches, and the gated
//! norm (5) follows the gate projection (`wide::delta_gate`).

use super::dispatch::{self, Ctx, site_launches};
use super::plan::{DeltaPlan, SiteTy, input_fused, out_fused};
use super::proj::{OResidArgs, QkvArgs};
use super::scratch::{Arena, Io, RecStore};
use super::wide::{self, GEMV_COLS};
use crate::GpuError;
use crate::linear;
use crate::linear::conv::{ConvArgs, ConvRows};
use crate::linear::norm_gate::{GdnGate, NormGateArgs, NormGateTail};
use crate::linear::seqs::{GdnCall, GdnRun, enqueue_gdn};
use crate::model::lookup::{f32_gain, kq_weight};
use crate::tensor::WindowMut;

/// The launches [`delta`] makes at `m` rows: eight, plus a Q6_K q·k·v
/// projection's token-major copy at more than one row; projections of other
/// types each their own ([`site_launches`]); an output projection not of
/// Q4_K its quantizer (a K-quant), the projection and the add. Past
/// [`GEMV_COLS`] rows: the norm, one quantizer for each form the four
/// projections read, the four, the five middle launches (the conv, the
/// prompt form's three and the gated norm writing the one q8 form the output
/// projection reads), the output projection and the residual add — thirteen
/// for K-quants.
pub(super) fn launches(d: &DeltaPlan, m: usize) -> usize {
    if m > GEMV_COLS {
        return 1
            + dispatch::wide_quants(&[d.qkv_ty, d.gate_ty, d.beta_ty, d.alpha_ty])
            + 4
            + 5
            + 2;
    }
    let input = if input_fused(d) {
        2 + usize::from(d.qkv_ty == SiteTy::Q6K && m > 1)
    } else {
        [d.qkv_ty, d.gate_ty, d.beta_ty, d.alpha_ty]
            .iter()
            .map(|t| site_launches(*t, m))
            .sum()
    };
    let output = if out_fused(d) {
        2
    } else {
        usize::from(d.out_ty.kquant()) + dispatch::out_resid_launches(d.out_ty, m)
    };
    1 + input + 3 + output
}

/// The mixer's input half at `m` rows (module doc, 1–2): the arena checked
/// against the plan's shape, then the wide arm's or the gemv arm's.
pub(super) fn delta_in(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::delta";
    // The load's Qwen3.6 kernels, which the mixer's three parts launch: a
    // load without them is refused here by name, ahead of any launch.
    c.k.q35(WHAT)?;
    let g = s
        .gdn
        .as_ref()
        .ok_or(GpuError::state(WHAT, "the arena's delta intermediates"))?;
    if g.shape != d.shape {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: the arena is cut for {:?}, the plan runs {:?}",
                c.layer, g.shape, d.shape
            ),
        ));
    }
    if m > GEMV_COLS {
        wide::delta_in(c, d, s, m)
    } else {
        project(c, d, s, m)
    }
}

/// The mixer's middle at `m` rows (module doc, 3–5), the same launches at
/// every `m`: the conv and prep over the layer's ring by the rows' positions
/// (the arena's `pos`, which the embedding launch wrote), the delta step over
/// the store's lane the word `io.lane` names, in place, in the prompt form
/// on an arena with the prompt rule's work, or a verify (`GdnArena::verify`),
/// the gated norm into the arena's attention rows.
pub(super) fn delta_core(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    r: &mut RecStore,
    s: &mut Arena,
    io: &Io<'_>,
    m: usize,
) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::delta";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let q35 = k.q35(WHAT)?;
    let stream = gpu.stream();
    let Arena {
        pos,
        attn,
        gdn,
        normed,
        wide,
        ..
    } = s;
    let g = gdn
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the arena's delta intermediates"))?;
    let lin = &q35.linear;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x: ConvRows::F32(&g.x),
            b_raw: &g.b,
            a_raw: &g.a,
            ba_pitch: d.shape.n_v,
            w: f32_gain(w, &d.conv)?,
            dt_bias: f32_gain(w, &d.dt_bias)?,
            ssm_a: f32_gain(w, &d.ssm_a)?,
            pos,
            shape: d.shape,
            eps: c.eps,
            m,
            fault: c.sink,
            y: &mut g.conv,
            beta: &mut g.beta,
            decay: &mut g.decay,
            ring: &mut r.ring,
        },
    )?;
    let lane = io.lane.ok_or(GpuError::state(
        WHAT,
        "the record's lane word (a chain with delta layers carries one)",
    ))?;
    // A wide unit (past `GEMV_COLS` rows) runs the chunked prompt form, its
    // chunk records in the q·k·v rows the conv has read; a pass-path unit and
    // a step keep the step's sequential bits.
    let len = g.x.len();
    let mut work;
    let run = if g.verify {
        GdnRun::Verify
    } else if m > GEMV_COLS {
        work = WindowMut::<u32>::of_mut(&mut *g.x, 0, len)?;
        GdnRun::Prompt {
            work: &mut work,
            keep: g.keep,
        }
    } else {
        GdnRun::Step
    };
    enqueue_gdn(
        lin,
        stream,
        GdnCall {
            qkv: &g.conv,
            beta: &g.beta,
            decay: &g.decay,
            lane,
            pos,
            shape: d.shape,
            m,
            fault: c.sink,
            o: &mut g.o,
        },
        r,
        run,
    )?;
    let gain = f32_gain(w, &d.ssm_norm)?;
    if m > GEMV_COLS {
        let tail = NormGateTail {
            o: &g.o,
            w: gain,
            gate: GdnGate::Silu,
            eps: c.eps,
            fault: c.sink,
            y: attn,
        };
        return wide::delta_gate(c, d, (normed, wide), (&lin.norm_gate, &mut g.z, tail), m);
    }
    lin.norm_gate.enqueue_norm_gate(
        stream,
        NormGateArgs {
            o: &g.o,
            z: &g.z,
            w: gain,
            gate: GdnGate::Silu,
            eps: c.eps,
            n_v: d.shape.n_v,
            m,
            fault: c.sink,
            y: attn,
        },
    )
}

/// The mixer's output half at `m` rows (module doc, 6): the wide arm's or
/// the gemv arm's.
pub(super) fn delta_out(
    c: &Ctx<'_>,
    d: &DeltaPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    if m > GEMV_COLS {
        wide::delta_out(c, d, s, m)
    } else {
        output(c, d, s, m)
    }
}

/// The gemv arm's input half at `m <= GEMV_COLS` rows: the fused norm and
/// q8_1 quantizer of `x`, then the four input projections in two launches
/// (module doc, 1–2).
pub(super) fn project(c: &Ctx<'_>, d: &DeltaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    const WHAT: &str = "qwen3moe::delta::project";
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let stream = gpu.stream();
    let i = s.col(m)?;
    let Arena {
        x,
        normed,
        act_x,
        gdn,
        cols,
        ..
    } = s;
    let g = gdn
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the arena's delta intermediates"))?;
    gpu.fused().enqueue_norm_quant(
        stream,
        x,
        f32_gain(w, &d.attn_norm)?,
        c.eps,
        &mut act_x[i],
        normed,
        c.sink,
    )?;
    let act = &act_x[i];
    if !input_fused(d) {
        let (ch, zl, nv) = (d.shape.channels(), d.shape.n_v * linear::HEAD, d.shape.n_v);
        let x = (act, &*normed);
        let mut cols = cols.as_mut();
        dispatch::site_gemv(
            c,
            (d.qkv_ty, &d.qkv),
            ch,
            x,
            m,
            cols.as_deref_mut(),
            &mut g.x,
        )?;
        dispatch::site_gemv(
            c,
            (d.gate_ty, &d.gate),
            zl,
            x,
            m,
            cols.as_deref_mut(),
            &mut g.z,
        )?;
        dispatch::site_gemv(
            c,
            (d.beta_ty, &d.beta),
            nv,
            x,
            m,
            cols.as_deref_mut(),
            &mut g.b,
        )?;
        return dispatch::site_gemv(c, (d.alpha_ty, &d.alpha), nv, x, m, cols, &mut g.a);
    }
    let (wqkv, wz) = (kq_weight(w, &d.qkv)?, kq_weight(w, &d.gate)?);
    let (wb, wa) = (kq_weight(w, &d.beta)?, kq_weight(w, &d.alpha)?);
    let (b_in_z, a_in_z, a_in_b) = g.tail_offsets();
    let (x_len, z_len, a_len) = (g.x.len(), g.z.len(), g.a.len());
    if d.qkv_ty == SiteTy::Q4K {
        k.proj.enqueue_qkv(
            stream,
            QkvArgs {
                wq: wqkv,
                wk: wz,
                wv: None,
                act,
                y: &mut g.proj,
                off_k: x_len,
                off_v: x_len + z_len,
            },
        )?;
        return k.proj.enqueue_qkv(
            stream,
            QkvArgs {
                wq: wb,
                wk: wa,
                wv: None,
                act,
                y: &mut g.from_b,
                off_k: a_in_b,
                off_v: a_in_b + a_len,
            },
        );
    }
    match g.x_cols.as_mut().filter(|_| m > 1) {
        Some(cols) => {
            gpu.enqueue_gemv_q6k(wqkv, act, cols)?;
            k.proj
                .token_major
                .enqueue(stream, cols, d.shape.channels(), m, &mut g.x)?;
        }
        None => gpu.enqueue_gemv_q6k(wqkv, act, &mut g.x)?,
    }
    k.proj.enqueue_qkv(
        stream,
        QkvArgs {
            wq: wz,
            wk: wb,
            wv: Some(wa),
            act,
            y: &mut g.from_z,
            off_k: b_in_z,
            off_v: a_in_z,
        },
    )
}

/// The gemv arm's output half at `m <= GEMV_COLS` rows (module doc, 6): a
/// Q4_K projection with the residual add folded in, any other through
/// `dispatch::out_resid` — a K-quant on the gated norm's rows quantized, any
/// other type on them as f32.
pub(super) fn output(c: &Ctx<'_>, d: &DeltaPlan, s: &mut Arena, m: usize) -> Result<(), GpuError> {
    let (gpu, w, k) = (c.gpu, c.w, c.k);
    let i = s.col(m)?;
    let hidden = s.dims.hidden;
    let Arena {
        x,
        normed,
        attn,
        act_attn,
        ffn_inp,
        cols,
        ..
    } = s;
    if d.out_ty.kquant() {
        gpu.enqueue_quantize_q8_1_layer(attn, &mut act_attn[i], c.layer)?;
    }
    if out_fused(d) {
        return k.proj.enqueue_o_resid(
            gpu.stream(),
            OResidArgs {
                w: kq_weight(w, &d.ssm_out)?,
                act: &act_attn[i],
                x,
                y: ffn_inp,
            },
        );
    }
    dispatch::out_resid(
        c,
        (d.out_ty, &d.ssm_out),
        hidden,
        (&act_attn[i], attn),
        m,
        (x, normed, cols.as_mut()),
        ffn_inp,
    )
}
