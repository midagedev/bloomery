//! The gated-delta-rule mixer of a Qwen3.6 layer at `m` rows: `x` in,
//! `ffn_inp = x + W_out · (RMSNorm(o) ⊙ SiLU(z))` out, the layer's
//! recurrent store read and written in place.
//!
//! The launches, in order:
//! 1. the fused norm and q8_1 quantizer of `x` (`attn_norm`);
//! 2. the four input projections in two launches into the arena's
//!    `[x | z | b | a]` blocks: a Q4_K `attn_qkv` beside `attn_gate` in one
//!    three-matrix launch and β with α in another; a Q6_K `attn_qkv` in its
//!    own gemv (and at more than one row its token-major copy) and
//!    `attn_gate`, β and α in one launch;
//! 3. the conv and prep ([`crate::linear::conv`]) over the layer's conv ring
//!    by the rows' positions;
//! 4. the delta step ([`crate::linear::delta`]) over the state lane the
//!    record's lane word names;
//! 5. the gated norm ([`crate::linear::norm_gate`]) into the arena's
//!    attention rows;
//! 6. their q8_1 quantizer and the output projection with the residual add.
//!
//! Every launch computes each row the way its one-row launch does — the
//! projections take the one-column body per column, the conv and the delta
//! step walk the rows in order through the ring and the state — so a pass
//! of `m` rows leaves the state, the ring and the outputs of `m` one-row
//! steps.

use super::dispatch::Ctx;
use super::plan::{DeltaPlan, Kq};
use super::proj::{OResidArgs, QkvArgs};
use super::scratch::{Arena, Io, RecStore};
use crate::GpuError;
use crate::linear::conv::ConvArgs;
use crate::linear::delta::DeltaArgs;
use crate::linear::norm_gate::NormGateArgs;
use crate::model::lookup::{f32_gain, kq_weight};

/// The launches [`delta`] makes at `m` rows: eight, plus a Q6_K q·k·v
/// projection's token-major copy at more than one row.
pub(super) fn launches(d: &DeltaPlan, m: usize) -> usize {
    8 + usize::from(d.qkv_ty == Kq::Q6K && m > 1)
}

/// Enqueue layer `c.layer`'s delta-rule mixer at `m` rows over its store
/// `r` (module doc). The rows' positions are the arena's `pos` (the
/// embedding launch wrote them), the state lane the word `io.lane` holds.
pub(super) fn delta(
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
    let i = m - 1;
    let lane = io.lane.ok_or(GpuError::state(
        WHAT,
        "the record's lane word (a chain with delta layers carries one)",
    ))?;
    let Arena {
        x,
        normed,
        act_x,
        pos,
        attn,
        act_attn,
        ffn_inp,
        gdn,
        ..
    } = s;
    let g = gdn
        .as_mut()
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
    let (wqkv, wz) = (kq_weight(w, &d.qkv)?, kq_weight(w, &d.gate)?);
    let (wb, wa) = (kq_weight(w, &d.beta)?, kq_weight(w, &d.alpha)?);
    let (b_in_z, a_in_z, a_in_b) = g.tail_offsets();
    let (x_len, z_len, a_len) = (g.x.len(), g.z.len(), g.a.len());
    match d.qkv_ty {
        Kq::Q4K => {
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
            k.proj.enqueue_qkv(
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
            )?;
        }
        Kq::Q6K => {
            match g.x_cols.as_mut().filter(|_| m > 1) {
                Some(cols) => {
                    gpu.enqueue_gemv_q6k(wqkv, act, cols)?;
                    k.proj
                        .enqueue_token_major(stream, cols, d.shape.channels(), m, &mut g.x)?;
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
            )?;
        }
    }
    let lin = &q35.linear;
    lin.conv.enqueue_conv_prep(
        stream,
        ConvArgs {
            x: &g.x,
            b_raw: &g.b,
            a_raw: &g.a,
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
    lin.delta.enqueue_delta(
        stream,
        DeltaArgs {
            qkv: &g.conv,
            beta: &g.beta,
            decay: &g.decay,
            lane,
            lane_at: 0,
            lanes: r.lanes,
            shape: d.shape,
            m,
            fault: c.sink,
            o: &mut g.o,
            state: &mut r.state,
        },
    )?;
    lin.norm_gate.enqueue_norm_gate(
        stream,
        NormGateArgs {
            o: &g.o,
            z: &g.z,
            w: f32_gain(w, &d.ssm_norm)?,
            eps: c.eps,
            n_v: d.shape.n_v,
            m,
            fault: c.sink,
            y: attn,
        },
    )?;
    gpu.enqueue_quantize_q8_1_layer(attn, &mut act_attn[i], c.layer)?;
    k.proj.enqueue_o_resid(
        stream,
        OResidArgs {
            w: kq_weight(w, &d.ssm_out)?,
            act: &act_attn[i],
            x,
            y: ffn_inp,
        },
    )
}
