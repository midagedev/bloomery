//! The activation rules a gate·up kernel of this family applies to a row's
//! two sums: `silu_mul` (`elem::silu_mul`, the reference's scalar SwiGLU
//! with the device `expf`), ik's clamped SwiGLU [`swiglu_clamp`] and the
//! gpt-oss / MiniMax SwiGLU [`swiglu_oai`], whose exponentials are
//! transcriptions of ik's AVX2 `v_expf` and so round the same on the device
//! and the host. `silu_mul` differs from the other two in its exponential,
//! and the clamp from the OAI form in where the limit bites, so they are
//! three rules, picked per launch by [`Act`].
//!
//! V4.1 kernels share these rules and the non-finite sum check. Gate-up
//! entries use [`apply_checked`] before writing their rows.

use crate::GpuError;
use crate::elem::silu_mul;
use crate::fault::{FaultSink, FaultSite};
use crate::linear::expf_ik;
pub use crate::linear::silu as silu_ik;

/// A gate·up kernel's activation rule, a launch argument.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    /// `silu(g) · u` with the device `expf` (`elem::silu_mul`).
    SiluMul,
    /// ik's clamped SwiGLU at `limit` ([`swiglu_clamp`]; `limit <= 1e-6`
    /// clamps nothing, as ik's own test).
    SwigluClamp { limit: f32 },
    /// The gpt-oss / MiniMax SwiGLU at `limit` ([`swiglu_oai`]). The device
    /// rule's `alpha` is [`SWIGLU_OAI_ALPHA`], the one value the references
    /// run; any other is refused by [`Act::code`].
    SwigluOai { alpha: f32, limit: f32 },
}

/// The device code of [`Act::SiluMul`].
pub const ACT_SILU_MUL: u32 = 0;
/// The device code of [`Act::SwigluClamp`].
pub const ACT_SWIGLU_CLAMP: u32 = 1;
/// The device code of [`Act::SwigluOai`].
pub const ACT_SWIGLU_OAI: u32 = 2;
/// The largest rule code a kernel's launch contract admits.
pub const ACT_MAX: u32 = ACT_SWIGLU_OAI;

/// `alpha` of [`swiglu_oai`]: the reference graph's `constexpr float alpha =
/// 1.702f` (llama.cpp `build_moe_ffn`, ik's `k_swiglu_oai_alpha`).
pub const SWIGLU_OAI_ALPHA: f32 = 1.702;

impl Act {
    /// The kernel's `(act, limit)` arguments, for the launcher `what`: a rule
    /// the device cannot run is refused by name — a limit that is not finite
    /// ([`swiglu_clamp`] and [`swiglu_oai`] compare against it, and a NaN
    /// bites on nothing), a negative [`Act::SwigluOai`] limit (its clamp of
    /// the up sum would have `lo > hi`), and an `alpha` other than
    /// [`SWIGLU_OAI_ALPHA`], which the kernels do not take.
    ///
    /// # Errors
    ///
    /// [`GpuError::Shape`] naming `what` and the rule.
    pub fn code(self, what: &'static str) -> Result<(u32, f32), GpuError> {
        match self {
            Act::SiluMul => Ok((ACT_SILU_MUL, 0.0)),
            Act::SwigluClamp { limit } if limit.is_finite() => Ok((ACT_SWIGLU_CLAMP, limit)),
            Act::SwigluClamp { limit } => Err(GpuError::shape(
                what,
                format!("the clamp limit must be finite, got {limit}"),
            )),
            Act::SwigluOai { alpha, limit }
                if alpha.to_bits() == SWIGLU_OAI_ALPHA.to_bits()
                    && limit.is_finite()
                    && limit >= 0.0 =>
            {
                Ok((ACT_SWIGLU_OAI, limit))
            }
            Act::SwigluOai { alpha, limit } => Err(GpuError::shape(
                what,
                format!(
                    "swigluoai runs at alpha {SWIGLU_OAI_ALPHA} and a finite limit of at least \
                     0, got alpha {alpha}, limit {limit}"
                ),
            )),
        }
    }
}

/// The rule of code `act` on a row's gate sum `g` and up sum `u`: the
/// launcher passes one of the codes of [`Act::code`], and `limit` only with
/// [`ACT_SWIGLU_CLAMP`] and [`ACT_SWIGLU_OAI`].
#[inline(always)]
#[must_use]
pub fn apply(act: u32, limit: f32, g: f32, u: f32) -> f32 {
    if act == ACT_SWIGLU_CLAMP {
        swiglu_clamp(g, u, limit)
    } else if act == ACT_SWIGLU_OAI {
        swiglu_oai(g, u, limit)
    } else {
        silu_mul(g, u)
    }
}

/// Apply the rule only to finite gate and up sums. Otherwise raise
/// [`FaultSite::F32Product`] on `fault` and return NaN before a clamp can
/// hide the non-finite sum.
#[inline(always)]
#[must_use]
pub fn apply_checked(act: u32, limit: f32, g: f32, u: f32, fault: FaultSink) -> f32 {
    if g.is_finite() && u.is_finite() {
        apply(act, limit, g, u)
    } else {
        fault.raise(FaultSite::F32Product);
        f32::NAN
    }
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

/// The gpt-oss / MiniMax SwiGLU for one row: with `x = min(g, limit)` and
/// `y = clamp(u, -limit, limit)`, `x / (1 + e^(−α·x)) · (y + 1)` at `α =`
/// [`SWIGLU_OAI_ALPHA`] — the reference's `ggml_compute_forward_swiglu_oai`
/// term for term (the gate is clamped from above only, before the sigmoid; the
/// up sum carries the unit offset), the exponential ik's `v_expf` like
/// [`swiglu_clamp`]'s. The `min` and the clamp are the `std::min` and
/// `std::clamp` comparisons they compile from (a NaN `g` passes the first, a
/// NaN `u` the second); the launchers check the sums first.
#[inline(always)]
pub fn swiglu_oai(g: f32, u: f32, limit: f32) -> f32 {
    let x = if limit < g { limit } else { g };
    let y = if u < -limit {
        -limit
    } else if limit < u {
        limit
    } else {
        u
    };
    x / (1.0 + expf_ik(SWIGLU_OAI_ALPHA * (0.0 - x))) * (y + 1.0)
}
