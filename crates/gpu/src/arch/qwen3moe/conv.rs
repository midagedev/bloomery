//! The gated short-conv mixer of an LFM3 layer at `m <= GEMV_COLS` rows: `x`
//! in, `ffn_inp = x ⊕ W_out · (C ⊙ conv(B ⊙ x'))` out, where `B | C | x'`
//! are the three chunks of `W_in · RMSNorm(x)` and `⊕` is the layer's join
//! (`dispatch::join`: the plain add, or the sandwich's norm and add). The
//! layer's conv ring is read and written in place.
//!
//! The launches, in order:
//! 1. the fused norm and q8_1 quantizer of `x` (`attn_norm`);
//! 2. the input projection into the arena's `B | C | x'` rows
//!    (`dispatch::site_gemv`, with a row-major K-quant's token-major copy);
//! 3. the short conv ([`crate::linear::shortconv`]) over the layer's ring by
//!    the rows' positions, into `normed`;
//! 4. a K-quant output projection's q8_1 of those rows;
//! 5. the output projection into `down`, and the join into `ffn_inp`.
//!
//! The conv walks the rows in order through the ring, so a pass of `m` rows
//! leaves the ring a run of `m` one-row steps leaves. Past [`GEMV_COLS`] rows
//! every op refuses by name: the wide arm has no short-conv launches.

use super::dispatch::{self, Ctx, site_gemv, site_launches};
use super::plan::{Half, LayerPlan, ShortConvPlan};
use super::scratch::Arena;
use super::wide::GEMV_COLS;
use crate::GpuError;
use crate::linear::shortconv::ShortConvArgs;
use crate::model::lookup::f32_gain;
use cuda_core::DeviceBuffer;

const WHAT: &str = "qwen3moe::short_conv";

/// The launches the mixer makes at `m` rows (module doc): the norm, the
/// input projection, the conv, the output's quantizer (a K-quant), the
/// output projection and the join ([`dispatch::join_launches`]).
pub(super) fn launches(n: &ShortConvPlan, p: &LayerPlan, m: usize) -> usize {
    1 + site_launches(n.in_ty, m)
        + 1
        + usize::from(n.out_ty.kquant())
        + site_launches(n.out_ty, m)
        + dispatch::join_launches(p)
}

/// The refusal of `m` rows past the gemv arm, and of a load without the
/// Qwen3.6 kernels, ahead of any launch.
fn gemv_arm(c: &Ctx<'_>, m: usize) -> Result<(), GpuError> {
    c.k.q35(WHAT)?;
    if m > GEMV_COLS {
        return Err(GpuError::shape(
            WHAT,
            format!(
                "layer {}: {m} rows of a short conv, past the {GEMV_COLS} the gemv arm takes; \
                 the wide arm has no short-conv launches",
                c.layer
            ),
        ));
    }
    Ok(())
}

/// The mixer's input side (module doc, 1–2).
pub(super) fn conv_in(
    c: &Ctx<'_>,
    n: &ShortConvPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    gemv_arm(c, m)?;
    let gpu = c.gpu;
    let d = s.dims;
    let i = s.col(m)?;
    gpu.fused().enqueue_norm_quant(
        gpu.stream(),
        &s.x,
        f32_gain(c.w, &n.attn_norm)?,
        c.eps,
        &mut s.act_x[i],
        &mut s.normed,
        c.sink,
    )?;
    let Arena {
        normed,
        act_x,
        cols,
        bcx,
        ..
    } = s;
    let bcx = bcx
        .as_mut()
        .ok_or(GpuError::state(WHAT, "the arena's B | C | x rows"))?;
    site_gemv(
        c,
        (n.in_ty, &n.conv_in),
        3 * d.hidden,
        (&act_x[i], normed),
        m,
        cols.as_mut(),
        bcx,
    )
}

/// The mixer's middle (module doc, 3–4): the conv over `ring` by the rows'
/// positions (the arena's `pos`, which the embedding launch wrote) into
/// `normed`, then a K-quant output projection's q8_1 of those rows.
pub(super) fn conv_core(
    c: &Ctx<'_>,
    n: &ShortConvPlan,
    ring: &mut DeviceBuffer<f32>,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    gemv_arm(c, m)?;
    let q35 = c.k.q35(WHAT)?;
    let gpu = c.gpu;
    let d = s.dims;
    let i = s.col(m)?;
    let bcx = s
        .bcx
        .as_ref()
        .ok_or(GpuError::state(WHAT, "the arena's B | C | x rows"))?;
    q35.short_conv.enqueue_short_conv(
        gpu.stream(),
        ShortConvArgs {
            ring: n.ring,
            d: d.hidden,
            m,
            bcx,
            weights: f32_gain(c.w, &n.conv)?,
            pos: &s.pos,
            fault: c.sink,
            y: &mut s.normed,
            state: ring,
        },
    )?;
    if n.out_ty.kquant() {
        gpu.enqueue_quantize_q8_1_layer(&s.normed, &mut s.act_x[i], c.layer)?;
    }
    Ok(())
}

/// The mixer's output side (module doc, 5): the output projection of the
/// conv rows into `down`, then the join of it and `x` into `ffn_inp`.
pub(super) fn conv_out(
    c: &Ctx<'_>,
    n: &ShortConvPlan,
    s: &mut Arena,
    m: usize,
) -> Result<(), GpuError> {
    gemv_arm(c, m)?;
    let d = s.dims;
    let i = s.col(m)?;
    let Arena {
        x,
        normed,
        act_x,
        cols,
        down,
        ffn_inp,
        ..
    } = s;
    site_gemv(
        c,
        (n.out_ty, &n.conv_out),
        d.hidden,
        (&act_x[i], normed),
        m,
        cols.as_mut(),
        down,
    )?;
    dispatch::join(c, Half::Mixer, (down, normed), x, (d.hidden, m), ffn_inp)
}
