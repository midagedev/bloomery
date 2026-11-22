//! The residual short conv of a looped block-attention layer (Inkling 2, llama.cpp PR #32695's
//! `src/models/inkling2.cpp` `convolve`, its static sum `ggml-cpu/ops.cpp`'s `ssm_conv`): a causal
//! depthwise conv over the last `taps` inputs of each channel, the current input added as a
//! residual, the whole scaled by `1 / √(taps + 1)` — `1 / √(2 · taps + 1)` where the conv is
//! dynamic — and, on the windowless layers, a second set of input-dependent taps beside the static
//! ones. The projections around it (the projection whose output is the conv's input, the dynamic
//! taps' projection) are the tree's dense gemv/GEMM and are not here.
//!
//! The rule, per token `t` of the call and channel `c` of `d` (the reference's op order, each tap
//! loop its own so the bits match term for term):
//!
//! ```text
//! acc  = w[0, c] · x(t, taps−1)                       the oldest static tap first
//! acc  = fma(w[k, c], x(t, taps−1−k), acc)            k = 1 .. taps−1, w[taps−1] the current
//! acc  = fma(dyn[t, h(c), lag], x(t, lag), acc)       the dynamic taps, lag = 0 the current
//! y    = (x(t, 0) + acc) · scale                      the residual cur-first, the scale last
//! ```
//!
//! with `h(c)` the channel's head of `heads` (`d / heads` channels a head) and `scale` the launch
//! argument the caller derives from `taps` and whether the conv is dynamic. **The static and
//! dynamic taps index opposite ends of the window** exactly as the reference spells them
//! (`inkling2.cpp:271`'s offset `n_conv − 1 − lag`): static `w[0]` pairs the oldest input, dynamic
//! `dyn[·, ·, 0]` the current one. The two sums are separate accumulations in that order (the
//! reference's `ssm_conv` dot, then one `ggml_add` a dynamic lag), the residual joins after both,
//! the scale last.
//!
//! Layouts, all f32 and token-major:
//! - the input `x` `[m][d]` — the projection's raw output rows, what the ring stores;
//! - the static filters `w` `[d][taps]` as the file lays it (`{taps, d}` in ggml's order: channel
//!   `c`'s taps contiguous at `c · taps`, the oldest first);
//! - the dynamic taps `dyn` `[m][heads][taps]` as the projection's output lays it (token-major,
//!   head, then the taps contiguous, `lag` 0 the current input); empty when the conv is static;
//! - `y` `[m][d]`;
//! - the conv ring `[rows][ch]` over the visit's whole channel width: the four convs of a visit
//!   read and write disjoint channel ranges of the one ring (the reference's single recurrent
//!   buffer), so the launch takes the ring's row width `ch` and this conv's channel base in it
//!   (`base + d <= ch`), and keeps the ring's indexing ([`super::ring_input`], the one owner):
//!   the input of position `p` in slot `p mod rows`.
//!
//! Positions: token `t` is position `pos[t]`, `pos[t] = pos[0] + t`. Its predecessor `back`
//! positions before (1 to `taps − 1`) is the call's row `t − back` when `back <= t`; otherwise it
//! lies before the call: zero when `pos[t] < back` (the sequence start, whatever the ring holds),
//! else ring slot `(pos[t] − back) mod rows`. The call writes the input of each of its last
//! `min(m, rows)` tokens into slot `pos[t] mod rows`, after every read of the channel's ring.
//!
//! Geometry: one thread a (unit, channel), the channel the fast index, [`super::BLOCK`] threads a
//! block, no shared memory and no barrier — the unit structure of [`super::shortconv`]: unit 0 the
//! first `min(m, taps − 1)` tokens (the ones that read the ring), in order, and afterwards the
//! ring store; unit `u >= 1` the one token `taps − 2 + u`, whose predecessors are all the call's
//! rows. The ring is touched by unit 0 alone, each channel by one thread, its reads before its
//! writes.
//!
//! Numeric contract: the first tap's product rounds alone (`mul_rn`), the rest fuse
//! (`fma_rn`), each loop ascending its own index as written above; the residual and the scale are
//! one rounded add and one rounded multiply.
//!
//! No silent failure: an output that is not finite (an input or a tap product was not), or a token
//! whose position is not `pos[0] + t`, raises [`FaultSite::LinearConv`] and the value written
//! there is NaN. The ring is stored as the inputs are: an input that is not finite stays in the
//! ring and every later call that reads it raises again, until the model resets.

use super::{BLOCK, ring_input, shortconv::units};
use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use chain::linear::Ring;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::{add_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// The widest filter the entry is launched with; the host refuses a wider one by name. The
/// kernel's loops run over `taps` and no table is sized by it.
pub const MAX_TAPS: usize = 16;
// The entry's `launch_contract` bounds are literals.
const _: () = assert!(BLOCK == 128 && MAX_TAPS == 16);

/// The input `back` positions before token `t` (position `p`) of channel `c` of `d`: the call's
/// row `t − back` when it has one, else the ring's.
///
/// # Safety
///
/// `t < m` and `c < d` with `x` holding `m · d` values; `ring` addresses `rows · ch` readable
/// f32s that no other thread writes while this one runs, `c + base < ch` and `rows > back >= 1`
/// when `back > t`.
#[allow(
    clippy::too_many_arguments,
    reason = "one token's coordinates and the call's inputs, forwarded from the entry"
)]
#[inline(always)]
unsafe fn tap_input(
    x: &[f32],
    ring: *const f32,
    d: usize,
    ch: usize,
    base: usize,
    rows: usize,
    t: usize,
    p: usize,
    back: usize,
    c: usize,
) -> f32 {
    if back <= t {
        // SAFETY: the fn's contract, row t − back < m.
        unsafe { *x.get_unchecked((t - back) * d + c) }
    } else {
        // SAFETY: the fn's contract, at this conv's channel of the ring's row.
        unsafe { ring_input(ring, rows, p, back, ch, base + c) }
    }
}

#[cuda_module]
mod conv_res_kernels {
    use super::*;

    /// The residual short conv of `m` tokens (module doc): thread `i` is unit `i / d`, channel
    /// `i % d`. `heads = 0` marks a static conv and its `dyn_` may be empty. A non-finite output
    /// or a position that is not `pos[0] + t` raises [`FaultSite::LinearConv`] and the value is
    /// written NaN.
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
            x.len() >= m * d,
            w.len() >= d * taps,
            dyn_.len() >= m * heads * taps,
            pos.len() >= m,
            m >= 1,
            d >= 1,
            heads >= 0,
            taps >= 1,
            taps <= 16,
            rows >= taps,
            base + d <= ch,
            y.len() >= m * d,
            ring.len() >= rows * ch
        )
    )]
    pub fn conv_res(
        x: &[f32],
        w: &[f32],
        dyn_: &[f32],
        pos: &[u32],
        d: u32,
        taps: u32,
        heads: u32,
        rows: u32,
        ch: u32,
        base: u32,
        m: u32,
        scale: f32,
        fault: FaultSink,
        mut y: DisjointSlice<f32>,
        mut ring: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        let (d, taps, heads, rows, ch, base, m) = (
            d as usize,
            taps as usize,
            heads as usize,
            rows as usize,
            ch as usize,
            base as usize,
            m as usize,
        );
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
        let per_head = d / heads.max(1);
        let mut ok = true;
        let mut t = t0;
        while t < t1 {
            // SAFETY: t < m <= pos.len().
            let p = unsafe { *pos.get_unchecked(t) };
            // The static taps, the oldest first: tap k reads the input `taps − 1 − k` positions
            // back. SAFETY: t < m and c < d; w holds d · taps values, so c · taps + k < w.len();
            // the ring is read only by unit 0, whose tokens these are when back > t, and written
            // below by the same thread after these reads.
            let acc = unsafe {
                let first = tap_input(
                    x,
                    ring.cast_const(),
                    d,
                    ch,
                    base,
                    rows,
                    t,
                    p as usize,
                    reach,
                    c,
                );
                let mut acc = mul_rn_f32(first, *w.get_unchecked(c * taps));
                let mut k = 1usize;
                while k < taps {
                    let back = reach - k;
                    let v = tap_input(
                        x,
                        ring.cast_const(),
                        d,
                        ch,
                        base,
                        rows,
                        t,
                        p as usize,
                        back,
                        c,
                    );
                    acc = fma_rn_f32(*w.get_unchecked(c * taps + k), v, acc);
                    k += 1;
                }
                // The dynamic taps, the current input first: lag reads the input `lag` positions
                // back, its head's own row of the call.
                if heads > 0 {
                    let head = c / per_head;
                    let mut lag = 0usize;
                    while lag < taps {
                        let tap = *dyn_.get_unchecked((t * heads + head) * taps + lag);
                        let v = tap_input(
                            x,
                            ring.cast_const(),
                            d,
                            ch,
                            base,
                            rows,
                            t,
                            p as usize,
                            lag,
                            c,
                        );
                        acc = fma_rn_f32(tap, v, acc);
                        lag += 1;
                    }
                }
                acc
            };
            // SAFETY: t < m and c < d, so t · d + c < m · d <= x.len() by the contract.
            let cur = unsafe { *x.get_unchecked(t * d + c) };
            let out = mul_rn_f32(add_rn_f32(cur, acc), scale);
            let fin = (u64::from(p) == p0 + t as u64) & out.is_finite();
            ok &= fin;
            // SAFETY: t · d + c < m · d <= y.len() by the contract; (t, c) is this thread's alone.
            unsafe { *y.get_unchecked_mut(t * d + c) = if fin { out } else { f32::NAN } };
            t += 1;
        }
        if u == 0 {
            let mut t = m.saturating_sub(rows);
            while t < m {
                // SAFETY: t < m <= pos.len().
                let slot = unsafe { *pos.get_unchecked(t) } as usize % rows;
                // SAFETY: slot < rows and base + c < ch, so slot · ch + base + c < rows · ch <=
                // ring.len(); row t of x is inside it; this conv's channel of the ring is this
                // thread's alone, its reads done above.
                unsafe { *ring.add(slot * ch + base + c) = *x.get_unchecked(t * d + c) };
                t += 1;
            }
        }
        if !ok {
            fault.raise(FaultSite::LinearConv);
        }
    }
}

/// [`ConvResKernels::enqueue`]'s arguments: `m` tokens of the conv's input `x` (`[m][d]`, the
/// projection's raw output rows), the static filters `weights` (`[d][taps]`, the oldest tap
/// first), the dynamic taps `dynamic` (`[m][heads][taps]`, lag 0 the current input; `heads = 0`
/// marks a static conv and the buffer may be empty), the tokens' positions `pos` (`[m]`,
/// consecutive: the words the rope reads), the ring `state` (`[ring.rows()][ch]`) this conv's
/// channels sit at `base` of, read and written in place, and the buffers the call writes: the
/// output `y` (`[m][d]`).
pub struct ConvResArgs<'a> {
    pub ring: Ring,
    /// The conv's channels `d`.
    pub d: usize,
    /// The conv's channels' base in the ring's rows (`base + d <= ch`).
    pub base: usize,
    /// The ring's row width `ch` (the visit's whole channel width).
    pub ch: usize,
    /// The dynamic taps' heads: the k and v convs' key-value head count, 0 a static conv.
    pub heads: usize,
    pub m: usize,
    pub x: &'a DeviceBuffer<f32>,
    pub weights: &'a DeviceBuffer<f32>,
    pub dynamic: &'a DeviceBuffer<f32>,
    pub pos: &'a DeviceBuffer<u32>,
    pub fault: FaultSink,
    /// The output scale: `1 / √(taps + 1)`, or `1 / √(2 · taps + 1)` a dynamic conv.
    pub scale: f32,
    pub y: &'a mut DeviceBuffer<f32>,
    pub state: &'a mut DeviceBuffer<f32>,
}

/// The loaded module. Owns no stream: each enqueue takes the engine stream.
pub struct ConvResKernels {
    module: conv_res_kernels::LoadedModule,
}

impl ConvResKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<ConvResKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(conv_res_kernels, ctx)? };
        Ok(ConvResKernels { module })
    }

    /// Enqueue the residual short conv of `args.m` tokens: `⌈units(m, taps) · d / 128⌉` blocks of
    /// 128 threads, the ring read and written in place. Refuses by name a call of no channel or no
    /// token, a filter of no tap or wider than [`MAX_TAPS`], a channel count the heads do not
    /// divide, a channel range outside the ring's rows, a scale that is not finite, and a buffer
    /// shorter than the launch needs; positions that are not consecutive are on the device
    /// (`pos`) and raise [`FaultSite::LinearConv`] there. Asynchronous, allocation-free,
    /// capturable.
    pub fn enqueue(&self, stream: &CudaStream, args: ConvResArgs<'_>) -> Result<(), GpuError> {
        const WHAT: &str = "conv_res::enqueue";
        let ConvResArgs {
            ring,
            d,
            base,
            ch,
            heads,
            m,
            x,
            weights,
            dynamic,
            pos,
            fault,
            scale,
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
        if heads > d || d % heads.max(1) != 0 {
            return Err(GpuError::shape(
                WHAT,
                format!("heads={heads} over d={d}: the head count divides the channels"),
            ));
        }
        if base + d > ch {
            return Err(GpuError::shape(
                WHAT,
                format!(
                    "channels {base}..{} of a ring row of {ch}: the conv's range is inside it",
                    base + d
                ),
            ));
        }
        if !scale.is_finite() {
            return Err(GpuError::shape(
                WHAT,
                format!("scale {scale}: a finite one"),
            ));
        }
        crate::short(
            WHAT,
            &[
                ("x", x.len(), m * d),
                ("weights", weights.len(), d * taps),
                ("dynamic", dynamic.len(), m * heads * taps),
                ("pos", pos.len(), m),
                ("y", y.len(), m * d),
                ("state", state.len(), rows * ch),
            ],
        )?;
        let grid = launch_u32(WHAT, "grid", (units(m, taps) * d).div_ceil(BLOCK as usize))?;
        let d = launch_u32(WHAT, "d", d)?;
        let taps = launch_u32(WHAT, "taps", taps)?;
        let heads = launch_u32(WHAT, "heads", heads)?;
        let rows = launch_u32(WHAT, "rows", rows)?;
        let ch = launch_u32(WHAT, "ch", ch)?;
        let base = launch_u32(WHAT, "base", base)?;
        let m = launch_u32(WHAT, "m", m)?;
        let prep = self
            .module
            .prepare_conv_res(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.conv_res(
            stream, &prep, x, weights, dynamic, pos, d, taps, heads, rows, ch, base, m, scale,
            fault, y, state,
        )?;
        Ok(())
    }
}
