//! The normed mix of Inkling 2's block attention residual: one threadgroup a
//! mix point scores the banked block inputs and the running stream against
//! `nq` query rows and writes the softmax-weighted sums of the RMS-normed
//! states (`kernels/res_mix.metal`).
//!
//! The vectors are the `n_banks` banks, bank `i`'s row `t` at
//! `banks[i · stride + t · n ..]`, and the stream `cur`, the last vector. A
//! launch with no banks (the stream alone) passes none.

use bloomery_device::{DeviceError, Window, WindowMut};

use crate::device::MetalDevice;
use crate::encode::{Binding, Bindings, ComputeEncoder};
use crate::fault::MetalFaultWord;
use crate::inkling2_geom::{MIX_NQ, MIX_THREADS, MIX_VECS};

use super::{Kernels, check_window_len, pack, range, word};

const WHAT: &str = "res mix";

/// One mix launch.
pub struct ResMixParams<'w> {
    /// Values a vector: the hidden width.
    pub n: usize,
    /// Query rows scored, `1..=`[`MIX_NQ`].
    pub nq: usize,
    /// Mix points (rows).
    pub m: usize,
    /// Banked vectors; the stream is one more, [`MIX_VECS`] at most in all.
    pub n_banks: usize,
    /// Elements between banks' rows of one row index: at least `m · n`.
    pub stride: usize,
    /// The banks, `(n_banks − 1) · stride + m · n` elements at least; held
    /// exactly when `n_banks` is not zero.
    pub banks: Option<Window<'w, f32, MetalDevice>>,
    /// The running stream, `m · n`.
    pub cur: Window<'w, f32, MetalDevice>,
    /// The query rows, `nq · n`, shared by every mix point.
    pub queries: Window<'w, f32, MetalDevice>,
    /// The mixes, `m · nq · n`: row `t`'s query `q` at `(t · nq + q) · n`.
    pub y: WindowMut<'w, f32, MetalDevice>,
    /// The RMS epsilon of the vectors' norm.
    pub eps: f32,
    /// The softmax's score scale, `1/√n` the caller derives.
    pub scale: f32,
    /// The fault word a raise lands in.
    pub faults: &'w MetalFaultWord,
    /// The layer a raise is recorded in.
    pub layer: u32,
}

fn refuse(detail: impl Into<String>) -> DeviceError {
    range(WHAT, detail)
}

impl Kernels {
    /// Encode the mix: one dispatch, a threadgroup of [`MIX_THREADS`] threads a
    /// mix point. The shape and every window are checked first; a refused
    /// launch encodes nothing.
    #[must_use = "a refused launch encoded nothing"]
    pub fn res_mix(
        &self,
        encoder: &mut ComputeEncoder<'_>,
        p: ResMixParams<'_>,
    ) -> Result<(), DeviceError> {
        if p.n == 0 || p.m == 0 {
            return Err(refuse(format!("{} values of {} rows", p.n, p.m)));
        }
        if !(1..=MIX_NQ).contains(&p.nq) {
            return Err(refuse(format!(
                "{} query rows, the kernel takes 1 to {MIX_NQ}",
                p.nq
            )));
        }
        if p.n_banks + 1 > MIX_VECS {
            return Err(refuse(format!(
                "{} vectors, the kernel takes at most {MIX_VECS}",
                p.n_banks + 1
            )));
        }
        if !(p.eps.is_finite() && p.eps > 0.0) || !(p.scale.is_finite() && p.scale > 0.0) {
            return Err(refuse(format!(
                "eps {} and scale {} are not finite and positive",
                p.eps, p.scale
            )));
        }
        let row =
            p.m.checked_mul(p.n)
                .ok_or_else(|| refuse(format!("{} rows of {} values wrap", p.m, p.n)))?;
        match (&p.banks, p.n_banks) {
            (None, 0) => {}
            (Some(banks), n_banks) if n_banks > 0 => {
                if p.stride < row {
                    return Err(refuse(format!(
                        "stride {} is under the {row} values of a bank",
                        p.stride
                    )));
                }
                let need = (n_banks - 1)
                    .checked_mul(p.stride)
                    .and_then(|v| v.checked_add(row))
                    .ok_or_else(|| refuse("the banks wrap".to_string()))?;
                if banks.len().get() < need {
                    return Err(refuse(format!(
                        "the banks window holds {} elements, the shape needs {need}",
                        banks.len().get()
                    )));
                }
            }
            (Some(_), _) => return Err(refuse("banks given for no banks".to_string())),
            (None, n_banks) => return Err(refuse(format!("{n_banks} banks and no window"))),
        }
        check_window_len(WHAT, "cur", p.cur.len(), row)?;
        check_window_len(WHAT, "queries", p.queries.len(), p.nq * p.n)?;
        check_window_len(WHAT, "y", p.y.len(), row * p.nq)?;
        let args = [
            word(WHAT, "n", p.n)?,
            word(WHAT, "nq", p.nq)?,
            word(WHAT, "m", p.m)?,
            word(WHAT, "banks", p.n_banks)?,
            word(WHAT, "stride", p.stride)?,
            p.layer,
            p.eps.to_bits(),
            p.scale.to_bits(),
        ]
        .map(u32::to_ne_bytes);
        encoder.set_pipeline(&self.ink2.res_mix);
        let mut slot = Bindings::new(encoder);
        slot.set(Binding::of(p.banks.as_ref().unwrap_or(&p.cur))?)?;
        slot.set(Binding::of(&p.cur)?)?;
        slot.set(Binding::of(&p.queries)?)?;
        slot.set(Binding::of_mut(&p.y)?)?;
        slot.set(Binding::bytes(pack(&args))?)?;
        slot.set(p.faults.binding())?;
        slot.dispatch_groups(p.m, 1, MIX_THREADS)
    }
}
