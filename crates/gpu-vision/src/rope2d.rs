//! The ViT's 2D rotary embedding (`vision.py` `get_vision_cos_sin`, `apply_rotary`): every query
//! and key head of 64 values is split into halves `x1 = x[0..32]`, `x2 = x[32..64]` and pair
//! `(x1[j], x2[j])` turns by angle `θ_j` of its patch. Patch `(h, w)` of the row-major grid has
//! `θ_j = h · f_j` for `j < 16` and `θ_j = w · f_(j−16)` for `j >= 16`, with
//! `f_i = 1 / theta^(2i/32)` — rows take the first sixteen angles, columns the last sixteen, and
//! the rotation covers the whole head.
//!
//! The table ([`RopeTable`]) is host work, one row of 32 `(cos, sin)` per patch; the kernel
//! applies it. Numeric rule, per pair, the reference's op order with every op rounded on its own
//! (torch runs `x1 * cos`, `x2 * sin` and the difference as separate f32 kernels):
//! `y1 = (x1·c) − (x2·s)`, `y2 = (x2·c) + (x1·s)`, from the bf16 inputs widened exactly, each
//! result rounded once to bf16. The kernel uses the `_rn` forms, which are never contracted, so
//! kernel and host rule ([`rope_ref`]) agree bit for bit on the same table.

use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{bf16_to_f32, f32_to_bf16_rne};
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Values in one attention head.
pub const HEAD_DIM: usize = 64;
/// Pairs one head turns: half the head, each pair `(x[j], x[j + 32])`.
pub const PAIRS: usize = HEAD_DIM / 2;
/// Angles per axis: the rows take `PAIRS / 2`, the columns the rest.
const PER_AXIS: usize = PAIRS / 2;
const BLOCK: u32 = 256;

/// Per patch, the `(cos, sin)` of its [`PAIRS`] angles, patch-major: pair `j` of patch `p` at
/// `cs[2·(p·PAIRS + j) ..]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable {
    pub n_h: usize,
    pub n_w: usize,
    pub cs: Vec<f32>,
}

impl RopeTable {
    /// The table of an `n_h × n_w` patch grid at RoPE base `theta`, as the reference builds it
    /// in f32: `f_i = 1.0 / theta^(i / 16)` — the exponent `arange(0, 32, 2) / 32` is exact in
    /// f32, the power correctly rounded (the device's `powf` the reference calls is within an
    /// ulp of it), the reciprocal one f32 division; the angle the position times `f_i` in f32;
    /// `cos` and `sin` of that f32 angle, correctly rounded.
    #[must_use]
    pub fn new(n_h: usize, n_w: usize, theta: f32) -> RopeTable {
        let freq: Vec<f32> = (0..PER_AXIS)
            .map(|i| {
                let e = (2 * i) as f32 / PAIRS as f32;
                let p = f64::from(theta).powf(f64::from(e)) as f32;
                1.0 / p
            })
            .collect();
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS);
        for h in 0..n_h {
            for w in 0..n_w {
                for (j, pos) in (0..PAIRS).map(|j| (j, if j < PER_AXIS { h } else { w })) {
                    let a = pos as f32 * freq[j % PER_AXIS];
                    cs.push(f64::from(a).cos() as f32);
                    cs.push(f64::from(a).sin() as f32);
                }
            }
        }
        RopeTable { n_h, n_w, cs }
    }

    /// Patches the table covers.
    #[must_use]
    pub fn patches(&self) -> usize {
        self.n_h * self.n_w
    }
}

/// One pair on the host: [`crate::rope2d`]'s rule on bf16 bits.
#[must_use]
pub fn rope_pair_ref(x1: u16, x2: u16, c: f32, s: f32) -> (u16, u16) {
    let (a, b) = (crate::bf16_f32(x1), crate::bf16_f32(x2));
    (
        crate::f32_bf16(a * c - b * s),
        crate::f32_bf16(b * c + a * s),
    )
}

/// The host rule over a whole `[patches, row_width]` bf16 buffer, in place: every head `h` of
/// `n_heads` at column `col0 + h·64` of every row turned by its patch's table row.
pub fn rope_ref(x: &mut [u16], row_width: usize, col0: usize, n_heads: usize, table: &RopeTable) {
    for p in 0..table.patches() {
        for h in 0..n_heads {
            let base = p * row_width + col0 + h * HEAD_DIM;
            for j in 0..PAIRS {
                let (c, s) = (
                    table.cs[2 * (p * PAIRS + j)],
                    table.cs[2 * (p * PAIRS + j) + 1],
                );
                let (y1, y2) = rope_pair_ref(x[base + j], x[base + j + PAIRS], c, s);
                x[base + j] = y1;
                x[base + j + PAIRS] = y2;
            }
        }
    }
}

#[cuda_module]
mod rope2d_kernels {
    use super::*;

    /// Turn the query and key heads of a `[n, row_width]` bf16 buffer in place: the `n_heads`
    /// heads at columns `q0 + h·64` and those at `k0 + h·64` of every row, pair `j` of patch `p`
    /// by table entry `cs[2·(p·32 + j) ..]`. One thread per (patch, q-or-k, head, pair); the
    /// two halves of a head are the pair's two values, so each thread owns its two positions.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= n * row_width, cs.len() >= n * 64)
    )]
    pub fn vis_rope2d(
        cs: &[f32],
        n: u32,
        row_width: u32,
        q0: u32,
        k0: u32,
        n_heads: u32,
        mut x: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        let per_patch = 2 * n_heads as usize * PAIRS;
        if i >= n as usize * per_patch {
            return;
        }
        let p = i / per_patch;
        let r = i - p * per_patch;
        let which = r / (n_heads as usize * PAIRS); // 0 query, 1 key
        let r = r - which * n_heads as usize * PAIRS;
        let h = r / PAIRS;
        let j = r - h * PAIRS;
        let col0 = if which == 0 { q0 } else { k0 } as usize;
        let a = p * row_width as usize + col0 + h * HEAD_DIM + j;
        let t = 2 * (p * PAIRS + j);
        // SAFETY: p < n and j < 32 put t + 1 below 64·n <= cs.len() (contract).
        let (c, s) = unsafe { (*cs.get_unchecked(t), *cs.get_unchecked(t + 1)) };
        // SAFETY: p < n and col0 + h·64 + 64 <= row_width (host-checked) bound a and a + 32
        // below n·row_width <= x.len(); this thread is the only one touching them.
        let (x1, x2) = unsafe {
            (
                bf16_to_f32(*x.get_unchecked_mut(a)),
                bf16_to_f32(*x.get_unchecked_mut(a + PAIRS)),
            )
        };
        let y1 = add_rn_f32(mul_rn_f32(x1, c), -mul_rn_f32(x2, s));
        let y2 = add_rn_f32(mul_rn_f32(x2, c), mul_rn_f32(x1, s));
        // SAFETY: the same two positions as the reads.
        unsafe {
            *x.get_unchecked_mut(a) = f32_to_bf16_rne(y1);
            *x.get_unchecked_mut(a + PAIRS) = f32_to_bf16_rne(y2);
        }
    }
}

/// [`RopeKernels::enqueue`]'s arguments: `x` holds `n` rows of `row_width` bf16, the query heads
/// at columns `q0 ..`, the key heads at `k0 ..`, `n_heads` of each; `cs` is the uploaded
/// [`RopeTable::cs`] of the same `n` patches.
pub struct RopeArgs<'a> {
    pub cs: &'a DeviceBuffer<f32>,
    pub n: usize,
    pub row_width: usize,
    pub q0: usize,
    pub k0: usize,
    pub n_heads: usize,
    pub x: &'a mut DeviceBuffer<u16>,
}

/// The loaded RoPE module.
pub struct RopeKernels {
    module: rope2d_kernels::LoadedModule,
}

impl RopeKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<RopeKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { rope2d_kernels::load(ctx)? };
        Ok(RopeKernels { module })
    }

    /// Enqueue the 2D RoPE of the query and key heads in place. Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: RopeArgs<'_>) -> Result<(), GpuError> {
        let what = "RopeKernels::enqueue";
        let RopeArgs {
            cs,
            n,
            row_width,
            q0,
            k0,
            n_heads,
            x,
        } = args;
        let span = n_heads * HEAD_DIM;
        if n == 0 || q0 + span > row_width || k0 + span > row_width {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_heads} heads of {HEAD_DIM} at columns {q0} and {k0} do not fit rows of {row_width} (n={n})"
                ),
            });
        }
        if x.len() < n * row_width || cs.len() < n * 2 * PAIRS {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "x.len() {} (need {}), cs.len() {} (need {})",
                    x.len(),
                    n * row_width,
                    cs.len(),
                    n * 2 * PAIRS
                ),
            });
        }
        let grid = launch_u32(
            what,
            "grid",
            (n * 2 * n_heads * PAIRS).div_ceil(BLOCK as usize),
        )?;
        let prep = self
            .module
            .prepare_vis_rope2d(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_rope2d(
            stream,
            &prep,
            cs,
            launch_u32(what, "n", n)?,
            launch_u32(what, "row_width", row_width)?,
            launch_u32(what, "q0", q0)?,
            launch_u32(what, "k0", k0)?,
            launch_u32(what, "n_heads", n_heads)?,
            x,
        )?;
        Ok(())
    }
}
