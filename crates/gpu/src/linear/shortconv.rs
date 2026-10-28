//! The gated short conv, `y = C ⊙ conv(B ⊙ x)`, over `m` tokens of `d` channels: one launch a
//! call. The input projection writes `bcx`, three chunks of `d` values a
//! token, `B | C | x` in that order; the conv is causal and depthwise, one filter of `taps` values
//! a channel, the oldest tap first, with no bias and no activation, over the products `B ⊙ x` of
//! the token's last `taps` positions. The projections around it are the tree's dense gemv/GEMM and
//! are not here. The host rule is [`chain::shortconv::short_conv`]; the card equals it bit for bit.
//!
//! Layouts, all f32 and token-major:
//! - `bcx` `[m][3·d]`; the filters `w` `[d][taps]`; `y` `[m][d]`;
//! - the conv ring `[rows][d]` ([`Ring`], `rows = taps − 1 + `[`super::PASS_ROWS`]): the product
//!   of position `p` in slot `p mod rows`, so the ring holds the products of the last `rows`
//!   positions a call wrote.
//!
//! Positions: token `t` is position `pos[t]`, `pos[t] = pos[0] + t`. Its predecessor `back`
//! positions before (1 to `taps − 1`) is the call's row `t − back` when `back <= t`; otherwise it
//! lies before the call: zero when `pos[t] < back` (the sequence start, whatever the ring holds),
//! else ring slot `(pos[t] − back) mod rows` ([`ring_input`], the indexing the delta layers' conv
//! and Mamba-2 read through too). The call writes the product of each of its last `min(m, rows)`
//! tokens into slot `pos[t] mod rows`, after every read of the channel's ring.
//!
//! Geometry: one thread a (unit, channel), `units(m, taps)·d` threads with the channel the fast
//! index, [`BLOCK`] threads a block, no shared memory and no barrier. Unit 0 takes the tokens that
//! read the ring, the first `min(m, taps − 1)`, in order, and afterwards stores the ring; unit
//! `u >= 1` takes the one token `taps − 2 + u`, whose predecessors are all the call's rows. The
//! ring is touched by unit 0 alone, each channel by one thread, its reads before its writes.
//! Nothing is kept between tokens: a token's taps read the call's rows (`B ⊙ x` formed again, the
//! same bits the ring holds) or the ring, so the filter width is a launch argument and nothing is
//! indexed by it in a register array.
//!
//! Numeric contract (the host rule is this list): the product `B ⊙ x` rounded to f32; `acc =
//! x₀·w₀` rounded, then `acc = fma(xⱼ, wⱼ, acc)` for the later taps, the oldest input first;
//! `y = C · acc` rounded. The multiply and the fused multiply-adds are explicit, so the device and
//! the host round the same operations.
//!
//! No silent failure: an output that is not finite (an input or a product was not), or a token
//! whose position is not `pos[0] + t`, raises [`FaultSite::LinearConv`] and the value written
//! there is NaN. The ring is stored as the products are, where the host rule refuses the call and
//! leaves it as it was: a product that is not finite stays in the ring and every later call that
//! reads it raises again, until the model resets.

use super::{BLOCK, ring_input};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use chain::linear::Ring;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The widest filter the entry is launched with; the host refuses a wider one by name. The
/// kernel's loops run over `taps` and no table is sized by it.
pub const MAX_TAPS: usize = 16;
// The entry's `launch_contract` bounds are literals.
const _: () = assert!(BLOCK == 128 && MAX_TAPS == 16);

/// The units of a call of `m` tokens at `taps` taps (module doc): unit 0, and one for each token
/// past the first `taps − 1`.
#[inline(always)]
#[must_use]
pub fn units(m: usize, taps: usize) -> usize {
    1 + m.saturating_sub(taps - 1)
}

/// The product `B ⊙ x` of row `t`, channel `c`, rounded to f32.
///
/// # Safety
///
/// `t < m`, `c < d`, and `bcx` holds `m·3·d` values.
#[inline(always)]
unsafe fn product(bcx: &[f32], d: usize, t: usize, c: usize) -> f32 {
    // SAFETY: t·3d + 2d + c < (t + 1)·3d <= m·3d <= bcx.len() by the contract.
    let (b, x) = unsafe {
        (
            *bcx.get_unchecked(t * 3 * d + c),
            *bcx.get_unchecked(t * 3 * d + 2 * d + c),
        )
    };
    mul_rn_f32(b, x)
}

/// The input `back` positions before token `t` (position `p`) of channel `c` of `d`: the call's
/// row `t − back` when it has one, else the ring's.
///
/// # Safety
///
/// [`product`]'s contract for the row `t − back`; `ring` addresses `rows·d` readable f32s that no
/// other thread writes while this one runs, `c < d` and `rows > back >= 1` when `back > t`.
#[allow(
    clippy::too_many_arguments,
    reason = "one token's coordinates and the call's inputs, forwarded from the entry"
)]
#[inline(always)]
unsafe fn tap_input(
    bcx: &[f32],
    ring: *const f32,
    d: usize,
    rows: usize,
    t: usize,
    p: usize,
    back: usize,
    c: usize,
) -> f32 {
    if back <= t {
        // SAFETY: the fn's contract, row t − back < m.
        unsafe { product(bcx, d, t - back, c) }
    } else {
        // SAFETY: the fn's contract.
        unsafe { ring_input(ring, rows, p, back, d, c) }
    }
}

#[cuda_module]
mod short_conv_kernels {
    use super::*;

    /// The short conv of `m` tokens (module doc): thread `i` is unit `i / d`, channel `i % d`. A
    /// non-finite output or a position that is not `pos[0] + t` raises [`FaultSite::LinearConv`]
    /// and the value is written NaN.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (
            bcx.len() >= m * 3 * d,
            w.len() >= d * taps,
            pos.len() >= m,
            m >= 1,
            d >= 1,
            taps >= 1,
            taps <= 16,
            rows >= taps,
            y.len() >= m * d,
            ring.len() >= rows * d
        )
    )]
    pub fn short_conv(
        bcx: &[f32],
        w: &[f32],
        pos: &[u32],
        d: u32,
        taps: u32,
        rows: u32,
        m: u32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let (d, taps, rows, m) = (d as usize, taps as usize, rows as usize, m as usize);
        let reach = taps - 1;
        if i >= units(m, taps) * d {
            return;
        }
        let (u, c) = (i / d, i % d);
        let (t0, t1) = if u == 0 {
            (0, m.min(reach))
        } else {
            (reach + u - 1, reach + u)
        };
        let ring = ring.as_mut_ptr();
        // SAFETY: pos holds m >= 1 words by the contract.
        let first_pos = unsafe { *pos.get_unchecked(0) };
        let p0 = u64::from(first_pos);
        let mut ok = true;
        let mut t = t0;
        while t < t1 {
            // SAFETY: t < m <= pos.len().
            let p = unsafe { *pos.get_unchecked(t) };
            // Tap j reads the input `taps − 1 − j` positions back, the oldest first.
            // SAFETY: t < m and c < d; w holds d·taps values, so c·taps + j < w.len(); the slot
            // (p − back) mod rows of the ring is a row of it (rows >= taps > back); the ring is
            // read only by unit 0, whose tokens these are when back > t, and written below by
            // the same thread after these reads.
            let acc = unsafe {
                let first = tap_input(bcx, ring.cast_const(), d, rows, t, p as usize, reach, c);
                let mut acc = mul_rn_f32(first, *w.get_unchecked(c * taps));
                let mut j = 1usize;
                while j < taps {
                    let back = reach - j;
                    let x = tap_input(bcx, ring.cast_const(), d, rows, t, p as usize, back, c);
                    acc = fma_rn_f32(x, *w.get_unchecked(c * taps + j), acc);
                    j += 1;
                }
                acc
            };
            // SAFETY: t < m and c < d: the chunk C of row t is inside bcx (m·3d values).
            let gate_c = unsafe { *bcx.get_unchecked(t * 3 * d + d + c) };
            let out = mul_rn_f32(gate_c, acc);
            let fin = (u64::from(p) == p0 + t as u64) & out.is_finite();
            ok &= fin;
            // SAFETY: t·d + c < m·d <= y.len() by the contract; (t, c) is this thread's alone.
            unsafe { *y.get_unchecked_mut(t * d + c) = if fin { out } else { f32::NAN } };
            t += 1;
        }
        if u == 0 {
            let mut t = m.saturating_sub(rows);
            while t < m {
                // SAFETY: t < m <= pos.len().
                let slot = unsafe { *pos.get_unchecked(t) } as usize % rows;
                // SAFETY: slot < rows and c < d, so slot·d + c < rows·d <= ring.len(), and row t
                // of bcx is inside it; channel c of the ring is this thread's alone, its reads
                // done above.
                unsafe { *ring.add(slot * d + c) = product(bcx, d, t, c) };
                t += 1;
            }
        }
        if !ok {
            fault.raise(FaultSite::LinearConv);
        }
    }
}

/// [`ShortConvKernels::enqueue_short_conv`]'s arguments: `m` tokens of the projection's output
/// `bcx` (`[m][3·d]`, `B | C | x`), the filters `weights` (`[d][ring.taps()]`, oldest tap first),
/// the tokens' positions `pos` (`[m]`, consecutive: the words the rope reads), the buffers the
/// call writes: the output `y` (`[m][d]`) and the conv ring `state` (`[ring.rows()][d]`), read
/// and written in place.
pub struct ShortConvArgs<'a> {
    pub ring: Ring,
    pub d: usize,
    pub m: usize,
    pub bcx: &'a DeviceBuffer<f32>,
    pub weights: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub fault: FaultSink,
    pub y: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ShortConvKernels {
    module: short_conv_kernels::LoadedModule,
}

impl ShortConvKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ShortConvKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(short_conv_kernels, ctx)? };
        Ok(ShortConvKernels { module })
    }

    /// Enqueue the short conv of `args.m` tokens: `⌈units(m, taps)·d / 128⌉` blocks of 128
    /// threads, the ring read and written in place. Refuses by name a call of no channel or no token, a filter of no tap
    /// or wider than [`MAX_TAPS`], and a buffer shorter than the launch needs; positions that are
    /// not consecutive are on the device (`pos`) and raise [`FaultSite::LinearConv`] there.
    /// Asynchronous, allocation-free, capturable.
    pub fn enqueue_short_conv(
        &self,
        stream: &CudaStream,
        args: ShortConvArgs<'_>,
    ) -> Result<(), GpuError> {
        const WHAT: &str = "enqueue_short_conv";
        let ShortConvArgs {
            ring,
            d,
            m,
            bcx,
            weights,
            pos,
            fault,
            y,
            state,
        } = args;
        let (taps, rows) = (ring.taps(), ring.rows());
        if d == 0 || m == 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("need d >= 1 and a call of at least one token, got d={d} m={m}"),
            ));
        }
        if taps > MAX_TAPS {
            return Err(GpuError::shape(
                WHAT,
                format!("taps={taps}: the entry takes at most {MAX_TAPS}"),
            ));
        }
        crate::short(
            WHAT,
            &[
                ("bcx", bcx.len(), m * 3 * d),
                ("weights", weights.len(), d * taps),
                ("pos", pos.len(), m),
                ("y", y.len(), m * d),
                ("state", state.len(), ring.values(d)),
            ],
        )?;
        let grid = launch_u32(WHAT, "grid", (units(m, taps) * d).div_ceil(BLOCK as usize))?;
        let d = launch_u32(WHAT, "d", d)?;
        let taps = launch_u32(WHAT, "taps", taps)?;
        let rows = launch_u32(WHAT, "rows", rows)?;
        let m = launch_u32(WHAT, "m", m)?;
        let prep = self
            .module
            .prepare_short_conv(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.short_conv(
            stream, &prep, bcx, weights, pos, d, taps, rows, m, fault, y, state,
        )?;
        Ok(())
    }
}
