//! The head-axis mix of a gated attention's output (ERNIE 5.5's VHA, every
//! attention layer): `o[h] ← Σ_g m[h, g] · o[g]` over the head axis of the
//! `[tokens][heads][width]` f32 rows `o`, with `m = I + V · Uᵀ` a
//! `[heads][heads]` matrix folded once at load ([`fold`]) and held transposed,
//! `mt[g·heads + h] = m[h][g]`. In place: every head's row of a column is read
//! before any of that column is written.
//!
//! Geometry: [`THREADS`] a block — `heads` rows by `cols = THREADS / heads`
//! columns, one thread per (head, column) — and one block per (token, column
//! group of `cols`), `tokens · width / cols` blocks. A block first stages its
//! columns' input values, head `h`'s from thread `h`, into [`THREADS`] f32 of
//! shared memory; after the barrier every thread of a column holds what it
//! reads. `heads` is a multiple of 32 that divides [`THREADS`]; `width` a
//! positive multiple of `cols`.
//!
//! Numeric contract (the host rule [`mix_host`] is this): a thread's value is
//! the sum over `g` ascending of `fma(mt[g·heads + h], o[g][c], acc)` from
//! `+0`, one fused multiply-add a step — this order is the gate. With the
//! identity matrix (`U = V = 0`) every value is its input bit for bit: the
//! own term is `fma(1, v, +0)` and every other `fma(±0, v, acc)`, which adds
//! a signed zero to `acc`.
//!
//! No silent failure: a non-finite input, or a value whose sum is not finite,
//! raises [`FaultSite::F32Product`] and that value is written NaN.

use crate::GpuError;
use crate::fault::{FaultSink, FaultSite};
use crate::launch_u32;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::float::fma_rn_f32;
use cuda_device::{DisjointSlice, SharedArray, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block: `heads` rows by `THREADS / heads` columns.
pub const THREADS: u32 = 256;
// The entry's `launch_bounds` and `launch_contract` block are literals.
const _: () = assert!(THREADS == 256);

/// The head-axis mix `m = I + V · Uᵀ` of `heads` heads at rank `rank`, folded
/// in f64 and rounded once to f32, transposed for the kernel: `out[g·heads +
/// h] = δ(h, g) + Σ_k V[h·rank + k] · U[g·rank + k]`. `U` and `V` hold
/// `heads` rows of `rank` values each, one row a head; another length, or no
/// head or rank, is refused by name.
pub fn fold(
    u: &[f32],
    v: &[f32],
    heads: usize,
    rank: usize,
    what: &'static str,
) -> Result<Vec<f32>, GpuError> {
    let want = heads
        .checked_mul(rank)
        .ok_or_else(|| GpuError::shape(what, "heads x rank overflows usize"))?;
    if heads == 0 || rank == 0 || u.len() != want || v.len() != want {
        return Err(GpuError::shape(
            what,
            format!(
                "U holds {} and V {} values; the head mix of {heads} heads at rank {rank} takes {want} of each",
                u.len(),
                v.len()
            ),
        ));
    }
    let mut out = vec![0.0f32; heads * heads];
    for g in 0..heads {
        for h in 0..heads {
            let mut acc = f64::from(u32::from(h == g));
            for k in 0..rank {
                acc += f64::from(v[h * rank + k]) * f64::from(u[g * rank + k]);
            }
            out[g * heads + h] = acc as f32;
        }
    }
    Ok(out)
}

/// The host rule of [`VhaKernels::enqueue_mix`]: `o` after the mix, in f64
/// over the same sum order — `fma(mt[g·heads + h], o[g][c], acc)` for `g`
/// ascending from `+0` — rounded once to f32.
#[must_use]
pub fn mix_host(o: &[f32], mt: &[f32], heads: usize, width: usize, tokens: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; tokens * heads * width];
    for t in 0..tokens {
        for c in 0..width {
            for h in 0..heads {
                let mut acc = 0.0f64;
                for g in 0..heads {
                    let v = f64::from(o[(t * heads + g) * width + c]);
                    acc = f64::mul_add(f64::from(mt[g * heads + h]), v, acc);
                }
                out[(t * heads + h) * width + c] = acc as f32;
            }
        }
    }
    out
}

#[cuda_module]
mod vha_kernels {
    use super::*;

    /// The mix of `tokens` tokens of `heads` heads of `width` values in
    /// place (module doc). Block `b` is token `b / groups`'s column group
    /// `b % groups`, `groups = width / cols` and `cols = THREADS / heads`;
    /// thread `i` is head `i / cols`, column `c0 + i % cols`. The block
    /// stages its `heads × cols` values into shared memory, then each thread
    /// sums its column over the heads in order. A non-finite input, or a sum
    /// that is not finite, raises [`FaultSite::F32Product`] and the value is
    /// written NaN.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (mt.len() >= heads * heads, o.len() >= tokens * heads * width)
    )]
    pub fn vha_mix(
        mt: &[f32],
        heads: u32,
        width: u32,
        tokens: u32,
        fault: FaultSink,
        mut o: DisjointSlice<f32>,
    ) {
        static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;

        let cols = THREADS / heads;
        let groups = width / cols;
        let b = thread::blockIdx_x() as usize;
        if b >= tokens as usize * groups as usize {
            return; // block-uniform
        }
        let (t, c0) = (b / groups as usize, (b % groups as usize) * cols as usize);
        let i = thread::threadIdx_x() as usize;
        let (h, cc) = (i / cols as usize, i % cols as usize);
        let c = c0 + cc;
        // SAFETY: TILE is this block's own shared allocation; the raw form is
        // the only way to reach it without a reference to a `static mut`.
        let tile = unsafe { SharedArray::as_raw_mut_ptr(&raw mut TILE) };
        // SAFETY: t < tokens, h < heads and c < width, so the read is inside
        // tokens·heads·width <= o.len() (launch contract); one thread per
        // (head, column) of the block's column group.
        let v = unsafe { *o.get_unchecked_mut((t * heads as usize + h) * width as usize + c) };
        // SAFETY: i < THREADS bounds the store; one thread per slot.
        unsafe { *tile.add(i) = v };
        thread::sync_threads();

        let mut acc = 0.0f32;
        let mut finite = true;
        let mut g = 0usize;
        while g < heads as usize {
            // SAFETY: g < heads and cc < cols, so the tile slot is this
            // block's, written before the barrier; g·heads + h < heads² <=
            // mt.len() by the launch contract.
            let (x, m) = unsafe {
                (
                    *tile.add(g * cols as usize + cc),
                    *mt.get_unchecked(g * heads as usize + h),
                )
            };
            finite &= x.is_finite() & m.is_finite();
            acc = fma_rn_f32(m, x, acc);
            g += 1;
        }
        finite &= acc.is_finite();
        if !finite {
            fault.raise(FaultSite::F32Product);
        }
        // SAFETY: the position of the read above, this thread's alone in the
        // launch; every read of this column by any thread of the block is
        // done (the barrier, then the loop over the tile, which holds them).
        unsafe {
            *o.get_unchecked_mut((t * heads as usize + h) * width as usize + c) =
                if finite { acc } else { f32::NAN };
        }
    }
}

/// The loaded head-mix module. Owns no stream: each enqueue takes the engine
/// stream, so its launch orders with the rest of the step and captures.
pub struct VhaKernels {
    module: vha_kernels::LoadedModule,
}

impl VhaKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<VhaKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(vha_kernels, ctx)? };
        Ok(VhaKernels { module })
    }

    /// Enqueue the mix of `tokens` tokens of `heads` heads of `width` values
    /// in place in `o` under the folded, transposed matrix `mt` ([`fold`]'s
    /// `heads²` values): `tokens · width / cols` blocks of [`THREADS`]
    /// threads. Refused before the launch: a head count that is not a
    /// positive multiple of 32 dividing [`THREADS`], a `width` that is not a
    /// positive multiple of `cols`, no token, and a short `mt` or `o`.
    /// Asynchronous, allocation-free, capturable.
    #[allow(
        clippy::too_many_arguments,
        reason = "host launcher: one argument per kernel argument it binds (rust-quality R8)"
    )]
    pub fn enqueue_mix(
        &self,
        stream: &CudaStream,
        mt: &DeviceBuffer<f32>,
        o: &mut DeviceBuffer<f32>,
        heads: usize,
        width: usize,
        tokens: usize,
        fault: FaultSink,
    ) -> Result<(), GpuError> {
        let what = "enqueue_mix";
        if heads == 0 || !heads.is_multiple_of(32) || !(THREADS as usize).is_multiple_of(heads) {
            return Err(GpuError::shape(
                what,
                format!(
                    "heads must be a positive multiple of 32 that divides {THREADS}, got {heads}"
                ),
            ));
        }
        let cols = THREADS as usize / heads;
        if width == 0 || !width.is_multiple_of(cols) {
            return Err(GpuError::shape(
                what,
                format!(
                    "width must be a positive multiple of {cols} ({THREADS} threads over {heads} \
                     heads), got {width}"
                ),
            ));
        }
        if tokens == 0 {
            return Err(GpuError::shape(what, "need tokens >= 1, got 0"));
        }
        let (mt_len, o_len) = (heads * heads, tokens * heads * width);
        if mt.len() < mt_len || o.len() < o_len {
            return Err(GpuError::shape(
                what,
                format!(
                    "mt.len() {} (need {mt_len}), o.len() {} (need {o_len})",
                    mt.len(),
                    o.len()
                ),
            ));
        }
        let grid = launch_u32(what, "grid", tokens * (width / cols))?;
        let heads = launch_u32(what, "heads", heads)?;
        let width = launch_u32(what, "width", width)?;
        let tokens = launch_u32(what, "tokens", tokens)?;
        let prep = self
            .module
            .prepare_vha_mix(LaunchConfig1D::new(grid, THREADS, 0))?;
        self.module
            .vha_mix(stream, &prep, mt, heads, width, tokens, fault, o)?;
        Ok(())
    }
}
