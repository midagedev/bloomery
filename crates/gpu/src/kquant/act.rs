//! The activation rules a gate·up kernel of this family applies to a row's
//! two sums: `silu_mul` (`elem::silu_mul`, the reference's scalar SwiGLU
//! with the device `expf`) and ik's clamped SwiGLU [`swiglu_clamp`], whose
//! exponential is a transcription of ik's AVX2 `v_expf` and so rounds the
//! same on the device and the host. The two differ in their exponential, so
//! they are two rules, picked per launch by [`Act`].
//!
//! [`expf_ik`], [`silu_ik`] and [`swiglu_clamp`] are the text of
//! `bloomery_gpu_deepseek41::experts`'s, which the V4.1 kernels call.

use crate::elem::silu_mul;

/// A gate·up kernel's activation rule, a launch argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// `silu(g) · u` with the device `expf` (`elem::silu_mul`).
    SiluMul,
    /// ik's clamped SwiGLU at `limit` ([`swiglu_clamp`]; `limit <= 1e-6`
    /// clamps nothing, as ik's own test).
    SwigluClamp { limit: f32 },
}

/// The device code of [`Act::SiluMul`].
pub const ACT_SILU_MUL: u32 = 0;
/// The device code of [`Act::SwigluClamp`].
pub const ACT_SWIGLU_CLAMP: u32 = 1;

impl Act {
    /// The kernel's `(act, limit)` arguments.
    #[must_use]
    pub fn code(self) -> (u32, f32) {
        match self {
            Act::SiluMul => (ACT_SILU_MUL, 0.0),
            Act::SwigluClamp { limit } => (ACT_SWIGLU_CLAMP, limit),
        }
    }
}

/// The rule of code `act` on a row's gate sum `g` and up sum `u`: the
/// launcher passes one of the two codes of [`Act::code`], and `limit` only
/// with [`ACT_SWIGLU_CLAMP`].
#[inline(always)]
#[must_use]
pub fn apply(act: u32, limit: f32, g: f32, u: f32) -> f32 {
    if act == ACT_SWIGLU_CLAMP {
        swiglu_clamp(g, u, limit)
    } else {
        silu_mul(g, u)
    }
}

/// ik's AVX2 `v_expf` (`iqk_utils.h`), one lane, op for op: `x` split as
/// `n·ln2 + b` by the `0x1.8p23` shift (two fused multiply-adds for `b`), a
/// degree-5 polynomial in `b` by fused multiply-adds, and the scale `2^n` put
/// into the exponent bits — with ik's two escape paths for `|n| > 126` (split
/// scale) and `|n| > 192` (overflow to infinity, underflow to zero). Every
/// product that meets an add is an explicit `mul_add`, so the device and the
/// host round the same operations; the two plain products of the escape paths
/// scale by a power of two, exact unless the result leaves the normal range,
/// where adding 1 rounds to the same value fused or not.
#[inline(always)]
pub(crate) fn expf_ik(x: f32) -> f32 {
    const SHIFT: f32 = f32::from_bits(0x4b40_0000); // 0x1.8p23
    const LOG2E: f32 = f32::from_bits(0x3fb8_aa3b); // 0x1.715476p+0
    const LN2_HI: f32 = f32::from_bits(0x3f31_7200); // 0x1.62e4p-1
    const LN2_LO: f32 = f32::from_bits(0x35bf_be8e); // 0x1.7f7d1cp-20
    const C0: f32 = f32::from_bits(0x3f7f_fff6); // 0x1.ffffecp-1
    const C1: f32 = f32::from_bits(0x3eff_fedb); // 0x1.fffdb6p-2
    const C2: f32 = f32::from_bits(0x3e2a_af33); // 0x1.555e66p-3
    const C3: f32 = f32::from_bits(0x3d2b_9f17); // 0x1.573e2ep-5
    const C4: f32 = f32::from_bits(0x3c07_2010); // 0x1.0e4020p-7
    let z = x.mul_add(LOG2E, SHIFT);
    let n = z - SHIFT;
    let b = (-n).mul_add(LN2_LO, (-n).mul_add(LN2_HI, x));
    let e = z.to_bits() << 23;
    let k = f32::from_bits(e.wrapping_add(0x3f80_0000));
    let u = b * b;
    let j = C4
        .mul_add(b, C3)
        .mul_add(u, C2.mul_add(b, C1))
        .mul_add(u, C0 * b);
    // An ordered compare, as ik's: a NaN `n` takes the main path.
    if n.abs() > 126.0 {
        let g: u32 = if n <= 0.0 { 0x8200_0000 } else { 0 };
        let s1 = f32::from_bits(g.wrapping_add(0x7f00_0000));
        let s2 = f32::from_bits(e.wrapping_sub(g));
        return if n.abs() > 192.0 {
            s1 * s1
        } else {
            s2.mul_add(j, s2) * s1
        };
    }
    j.mul_add(k, k)
}

/// ik's AVX2 `v_silu`: `x / (1 + expf_ik(0 - x))`.
#[inline(always)]
pub fn silu_ik(x: f32) -> f32 {
    x / (1.0 + expf_ik(0.0 - x))
}

/// ik's clamped SwiGLU for one row: `min(silu(g), limit) · max(-limit,
/// min(limit, u))`, or `silu(g) · u` when `limit <= 1e-6`, each `min`/`max`
/// spelled as the `std::min`/`std::max` comparison it compiles from (a NaN
/// `silu(g)` passes the first; a NaN `u` becomes `limit`).
#[inline(always)]
pub fn swiglu_clamp(g: f32, u: f32, limit: f32) -> f32 {
    let mut s = silu_ik(g);
    let mut uc = u;
    if limit > 1e-6 {
        s = if limit < s { limit } else { s };
        uc = if u < limit { u } else { limit };
        uc = if -limit < uc { uc } else { -limit };
    }
    uc * s
}
