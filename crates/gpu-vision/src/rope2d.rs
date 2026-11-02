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

/// The patches `(row, column)` of an `n_h × n_w` grid in raster order.
pub fn raster(n_h: usize, n_w: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n_h).flat_map(move |h| (0..n_w).map(move |w| (h, w)))
}

/// The patches `(row, column)` of an `n_h × n_w` grid in merge order: groups of `merge × merge`
/// patches in raster order, the patches of a group in raster order inside it — the order of the
/// vision crate's `PatchLayout` with a `merge` above one.
pub fn merge_order(n_h: usize, n_w: usize, merge: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n_h / merge).flat_map(move |gy| {
        (0..n_w / merge).flat_map(move |gx| {
            (0..merge)
                .flat_map(move |dy| (0..merge).map(move |dx| (gy * merge + dy, gx * merge + dx)))
        })
    })
}

/// Per patch, the `(cos, sin)` of its [`PAIRS`] angles, patch-major: pair `j` of patch `p` at
/// `cs[2·(p·PAIRS + j) ..]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable {
    pub n_h: usize,
    pub n_w: usize,
    pub cs: Vec<f32>,
}

impl RopeTable {
    /// Gemma4's two NEOX half-head turns, applied through this module's
    /// whole-head pairing after the q/k rows and gains are permuted at load.
    #[must_use]
    pub(crate) fn neox_halves(n_h: usize, n_w: usize, theta: f32) -> RopeTable {
        let step = f64::from(theta).powf(f64::from(-2.0f32 / PAIRS as f32)) as f32;
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS);
        for (y, x) in raster(n_h, n_w) {
            for pos in [x, y] {
                let mut angle = pos as f32;
                for _ in 0..PER_AXIS {
                    cs.push(f64::from(angle).cos() as f32);
                    cs.push(f64::from(angle).sin() as f32);
                    angle *= step;
                }
            }
        }
        RopeTable { n_h, n_w, cs }
    }

    /// The table of an `n_h × n_w` patch grid at RoPE base `theta`, as the reference builds it
    /// in f32: `f_i = 1.0 / theta^(i / 16)` — the exponent `arange(0, 32, 2) / 32` is exact in
    /// f32, the power correctly rounded (the device's `powf` the reference calls is within an
    /// ulp of it), the reciprocal one f32 division; the angle the position times `f_i` in f32;
    /// `cos` and `sin` of that f32 angle, correctly rounded.
    #[must_use]
    pub fn new(n_h: usize, n_w: usize, theta: f32) -> RopeTable {
        RopeTable::in_order(n_h, n_w, raster(n_h, n_w), theta)
    }

    /// The table of the patches `coords` (each `(row, column)` of an `n_h × n_w` grid, in the
    /// order the rows are wanted) at RoPE base `theta`, by [`RopeTable::new`]'s rule.
    #[must_use]
    pub fn in_order(
        n_h: usize,
        n_w: usize,
        coords: impl IntoIterator<Item = (usize, usize)>,
        theta: f32,
    ) -> RopeTable {
        let freq: Vec<f32> = (0..PER_AXIS)
            .map(|i| {
                let e = (2 * i) as f32 / PAIRS as f32;
                let p = f64::from(theta).powf(f64::from(e)) as f32;
                1.0 / p
            })
            .collect();
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS);
        for (h, w) in coords {
            for (j, pos) in (0..PAIRS).map(|j| (j, if j < PER_AXIS { h } else { w })) {
                let a = pos as f32 * freq[j % PER_AXIS];
                cs.push(f64::from(a).cos() as f32);
                cs.push(f64::from(a).sin() as f32);
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

    /// A second head width is its own entry (`vis_rope2d_h72`), not this body made generic: a
    /// kernel body moved into an inline helper changes its PTX (docs/upstream/nvlabs-ledger.md
    /// row 39) and this entry's md5 is held.
    ///
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
        let module = unsafe { bloomery_gpu::shared_module!(rope2d_kernels, ctx)? };
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

// ------------------------------------------------------- heads of 72 (the Qwen3-VL tower)

/// Values in one attention head of the Qwen3-VL tower.
pub const HEAD_DIM_72: usize = 72;
/// Pairs one such head turns: half the head, each pair `(x[j], x[j + 36])`.
pub const PAIRS_72: usize = HEAD_DIM_72 / 2;
/// Angles per axis: the rows take `PAIRS_72 / 2`, the columns the rest.
const PER_AXIS_72: usize = PAIRS_72 / 2;

/// Per patch, the `(cos, sin)` of its [`PAIRS_72`] angles, in the order of the patches the table
/// was built over: pair `j` of patch `p` at `cs[2·(p·PAIRS_72 + j) ..]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable72 {
    pub n_h: usize,
    pub n_w: usize,
    pub cs: Vec<f32>,
}

impl RopeTable72 {
    /// The table of the patches `coords` (each `(row, column)` of an `n_h × n_w` grid) at RoPE
    /// base `theta`, as llama.cpp's vision RoPE builds its cache in f32
    /// (`ggml_mrope_cache_init`): `theta_scale = theta^(−2/36)`; the angle of pair `k` of an axis
    /// is the position times `theta_scale^k`, formed by multiplying the running angle by
    /// `theta_scale` once per pair (the first eighteen pairs take the row, the next eighteen the
    /// column, each axis starting again from its own position); `cos` and `sin` of
    /// that f32 angle, correctly rounded.
    #[must_use]
    pub fn in_order(
        n_h: usize,
        n_w: usize,
        coords: impl IntoIterator<Item = (usize, usize)>,
        theta: f32,
    ) -> RopeTable72 {
        let theta_scale = f64::from(theta).powf(f64::from(-2.0f32 / PAIRS_72 as f32)) as f32;
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS_72);
        for (y, x) in coords {
            for pos in [y, x] {
                let mut angle = pos as f32;
                for _ in 0..PER_AXIS_72 {
                    cs.push(f64::from(angle).cos() as f32);
                    cs.push(f64::from(angle).sin() as f32);
                    angle *= theta_scale;
                }
            }
        }
        RopeTable72 { n_h, n_w, cs }
    }

    /// Patches the table covers.
    #[must_use]
    pub fn patches(&self) -> usize {
        self.cs.len() / (2 * PAIRS_72)
    }
}

/// The host rule over a whole `[patches, row_width]` bf16 buffer, in place, for heads of 72:
/// every head `h` of `n_heads` at column `col0 + h·72` of every row turned by its patch's table row.
pub fn rope_72_ref(
    x: &mut [u16],
    row_width: usize,
    col0: usize,
    n_heads: usize,
    table: &RopeTable72,
) {
    for p in 0..table.patches() {
        for h in 0..n_heads {
            let base = p * row_width + col0 + h * HEAD_DIM_72;
            for j in 0..PAIRS_72 {
                let (c, s) = (
                    table.cs[2 * (p * PAIRS_72 + j)],
                    table.cs[2 * (p * PAIRS_72 + j) + 1],
                );
                let (y1, y2) = rope_pair_ref(x[base + j], x[base + j + PAIRS_72], c, s);
                x[base + j] = y1;
                x[base + j + PAIRS_72] = y2;
            }
        }
    }
}

#[cuda_module]
mod rope2d_72_kernels {
    use super::*;

    /// `vis_rope2d`'s rule for heads of 72 values (36 pairs):
    /// turn the query and key heads of a `[n, row_width]` bf16 buffer in place, the `n_heads`
    /// heads at columns `q0 + h·72` and those at `k0 + h·72` of every row, pair `j` of patch `p`
    /// by table entry `cs[2·(p·36 + j) ..]`. One thread per (patch, q-or-k, head, pair). A new
    /// entry, not the V4.1 body with a wider constant: moving that body into a shared inline
    /// helper changes its PTX (docs/upstream/nvlabs-ledger.md row 39), and its md5 is held.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= n * row_width, cs.len() >= n * 72)
    )]
    pub fn vis_rope2d_h72(
        cs: &[f32],
        n: u32,
        row_width: u32,
        q0: u32,
        k0: u32,
        n_heads: u32,
        mut x: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        let per_patch = 2 * n_heads as usize * PAIRS_72;
        if i >= n as usize * per_patch {
            return;
        }
        let p = i / per_patch;
        let r = i - p * per_patch;
        let which = r / (n_heads as usize * PAIRS_72); // 0 query, 1 key
        let r = r - which * n_heads as usize * PAIRS_72;
        let h = r / PAIRS_72;
        let j = r - h * PAIRS_72;
        let col0 = if which == 0 { q0 } else { k0 } as usize;
        let a = p * row_width as usize + col0 + h * HEAD_DIM_72 + j;
        let t = 2 * (p * PAIRS_72 + j);
        // SAFETY: p < n and j < 36 put t + 1 below 72·n <= cs.len() (contract).
        let (c, s) = unsafe { (*cs.get_unchecked(t), *cs.get_unchecked(t + 1)) };
        // SAFETY: p < n and col0 + h·72 + 72 <= row_width (host-checked) bound a and a + 36
        // below n·row_width <= x.len(); this thread is the only one touching them.
        let (x1, x2) = unsafe {
            (
                bf16_to_f32(*x.get_unchecked_mut(a)),
                bf16_to_f32(*x.get_unchecked_mut(a + PAIRS_72)),
            )
        };
        let y1 = add_rn_f32(mul_rn_f32(x1, c), -mul_rn_f32(x2, s));
        let y2 = add_rn_f32(mul_rn_f32(x2, c), mul_rn_f32(x1, s));
        // SAFETY: the same two positions as the reads.
        unsafe {
            *x.get_unchecked_mut(a) = f32_to_bf16_rne(y1);
            *x.get_unchecked_mut(a + PAIRS_72) = f32_to_bf16_rne(y2);
        }
    }
}

/// The loaded RoPE module for heads of 72.
pub struct Rope72Kernels {
    module: rope2d_72_kernels::LoadedModule,
}

impl Rope72Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Rope72Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(rope2d_72_kernels, ctx)? };
        Ok(Rope72Kernels { module })
    }

    /// Enqueue the 2D RoPE of the query and key heads of 72 values in place ([`RopeArgs`], its
    /// `cs` the uploaded [`RopeTable72::cs`]). Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: RopeArgs<'_>) -> Result<(), GpuError> {
        let what = "Rope72Kernels::enqueue";
        let RopeArgs {
            cs,
            n,
            row_width,
            q0,
            k0,
            n_heads,
            x,
        } = args;
        let span = n_heads * HEAD_DIM_72;
        if n == 0 || q0 + span > row_width || k0 + span > row_width {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_heads} heads of {HEAD_DIM_72} at columns {q0} and {k0} do not fit rows of {row_width} (n={n})"
                ),
            });
        }
        if x.len() < n * row_width || cs.len() < n * 2 * PAIRS_72 {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "x.len() {} (need {}), cs.len() {} (need {})",
                    x.len(),
                    n * row_width,
                    cs.len(),
                    n * 2 * PAIRS_72
                ),
            });
        }
        let grid = launch_u32(
            what,
            "grid",
            (n * 2 * n_heads * PAIRS_72).div_ceil(BLOCK as usize),
        )?;
        let prep = self
            .module
            .prepare_vis_rope2d_h72(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_rope2d_h72(
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

// ------------------------------------------------------ heads of 128 (the Pixtral tower)

pub use crate::attn_h128::HEAD_DIM_128;
/// Pairs one such head turns: half the head, each pair `(x[j], x[j + 64])`.
pub const PAIRS_128: usize = HEAD_DIM_128 / 2;
/// Pairs per axis: the rows take the first `PER_AXIS_128`, the columns the rest.
const PER_AXIS_128: usize = PAIRS_128 / 2;

/// The value of the file's head of 128 that value `r` of the card's head holds. llama.cpp's
/// `build_rope_2d` turns a head in two NORM-mode halves, adjacent pairs `(2i, 2i + 1)` of values
/// `0..64` by the row and of values `64..128` by the column; the kernel turns pairs
/// `(j, j + 64)`. So the card's value `j` holds the first of a pair and `j + 64` the second, the
/// row half's pair `j` for `j < 32` and the column half's pair `j − 32` above. The query and key
/// rows are permuted by it at load (the same permutation on both leaves `q · k` unchanged, and
/// the value rows are not touched), and [`RopeTable128::interleaved`] orders its pairs to match.
#[must_use]
pub const fn rope_source_128(r: usize) -> usize {
    let (pair, second) = (r % PAIRS_128, r / PAIRS_128);
    let (half, i) = (pair / PER_AXIS_128, pair % PER_AXIS_128);
    half * PAIRS_128 + 2 * i + second
}

/// Per patch, the `(cos, sin)` of its [`PAIRS_128`] angles, in the order of the patches the table
/// was built over: pair `j` of patch `p` at `cs[2·(p·PAIRS_128 + j) ..]`.
#[derive(Clone, Debug, PartialEq)]
pub struct RopeTable128 {
    pub n_h: usize,
    pub n_w: usize,
    pub cs: Vec<f32>,
}

impl RopeTable128 {
    /// The table of the patches `coords` (each `(row, column)` of an `n_h × n_w` grid) at RoPE
    /// base `theta`, as llama.cpp's `build_rope_2d` with interleaved frequencies builds it on the
    /// CPU in f32: two `ggml_rope_ext` calls of 64 values each (`ggml_rope_cache_init`),
    /// `theta_scale = theta^(−2/64)`; the angle of pair `k` of an axis is the running angle (the
    /// position, multiplied by `theta_scale` once per pair) times the call's `freq_scale` in f32 —
    /// 1 for the row's half, `theta^(−2/128)` for the column's (the odd frequencies); `cos` and
    /// `sin` of that f32 angle, correctly rounded. Pair `j < 32` is the row's pair `j`, pair
    /// `32 + j` the column's pair `j`: [`rope_source_128`]'s order.
    #[must_use]
    pub fn interleaved(
        n_h: usize,
        n_w: usize,
        coords: impl IntoIterator<Item = (usize, usize)>,
        theta: f32,
    ) -> RopeTable128 {
        let theta_scale = f64::from(theta).powf(f64::from(-2.0f32 / PAIRS_128 as f32)) as f32;
        let odd = f64::from(theta).powf(f64::from(-2.0f32 / HEAD_DIM_128 as f32)) as f32;
        let mut cs = Vec::with_capacity(n_h * n_w * 2 * PAIRS_128);
        for (y, x) in coords {
            for (pos, freq_scale) in [(y, 1.0f32), (x, odd)] {
                let mut angle = pos as f32;
                for _ in 0..PER_AXIS_128 {
                    let a = freq_scale * angle;
                    cs.push(f64::from(a).cos() as f32);
                    cs.push(f64::from(a).sin() as f32);
                    angle *= theta_scale;
                }
            }
        }
        RopeTable128 { n_h, n_w, cs }
    }

    /// Patches the table covers.
    #[must_use]
    pub fn patches(&self) -> usize {
        self.cs.len() / (2 * PAIRS_128)
    }
}

/// The host rule over a whole `[patches, row_width]` bf16 buffer, in place, for heads of 128:
/// every head `h` of `n_heads` at column `col0 + h·128` of every row turned by its patch's table
/// row, pairs `(j, j + 64)`.
pub fn rope_128_ref(
    x: &mut [u16],
    row_width: usize,
    col0: usize,
    n_heads: usize,
    table: &RopeTable128,
) {
    for p in 0..table.patches() {
        for h in 0..n_heads {
            let base = p * row_width + col0 + h * HEAD_DIM_128;
            for j in 0..PAIRS_128 {
                let (c, s) = (
                    table.cs[2 * (p * PAIRS_128 + j)],
                    table.cs[2 * (p * PAIRS_128 + j) + 1],
                );
                let (y1, y2) = rope_pair_ref(x[base + j], x[base + j + PAIRS_128], c, s);
                x[base + j] = y1;
                x[base + j + PAIRS_128] = y2;
            }
        }
    }
}

#[cuda_module]
mod rope2d_128_kernels {
    use super::*;

    /// `vis_rope2d`'s rule for heads of 128 values (64 pairs): turn the query and key heads of a
    /// `[n, row_width]` bf16 buffer in place, the `n_heads` heads at columns `q0 + h·128` and
    /// those at `k0 + h·128` of every row, pair `j` of patch `p` by table entry
    /// `cs[2·(p·64 + j) ..]`. One thread per (patch, q-or-k, head, pair). A new entry for the fact
    /// no existing one covers — heads of 128 — not a parameter of `vis_rope2d_h72`, whose PTX is
    /// the Qwen3-VL tower's and is held (docs/upstream/nvlabs-ledger.md row 39).
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (x.len() >= n * row_width, cs.len() >= n * 128)
    )]
    pub fn vis_rope2d_h128(
        cs: &[f32],
        n: u32,
        row_width: u32,
        q0: u32,
        k0: u32,
        n_heads: u32,
        mut x: DisjointSlice<u16>,
    ) {
        let i = thread::index_1d().get();
        let per_patch = 2 * n_heads as usize * PAIRS_128;
        if i >= n as usize * per_patch {
            return;
        }
        let p = i / per_patch;
        let r = i - p * per_patch;
        let which = r / (n_heads as usize * PAIRS_128); // 0 query, 1 key
        let r = r - which * n_heads as usize * PAIRS_128;
        let h = r / PAIRS_128;
        let j = r - h * PAIRS_128;
        let col0 = if which == 0 { q0 } else { k0 } as usize;
        let a = p * row_width as usize + col0 + h * HEAD_DIM_128 + j;
        let t = 2 * (p * PAIRS_128 + j);
        // SAFETY: p < n and j < 64 put t + 1 below 128·n <= cs.len() (contract).
        let (c, s) = unsafe { (*cs.get_unchecked(t), *cs.get_unchecked(t + 1)) };
        // SAFETY: p < n and col0 + h·128 + 128 <= row_width (host-checked) bound a and a + 64
        // below n·row_width <= x.len(); this thread is the only one touching them.
        let (x1, x2) = unsafe {
            (
                bf16_to_f32(*x.get_unchecked_mut(a)),
                bf16_to_f32(*x.get_unchecked_mut(a + PAIRS_128)),
            )
        };
        let y1 = add_rn_f32(mul_rn_f32(x1, c), -mul_rn_f32(x2, s));
        let y2 = add_rn_f32(mul_rn_f32(x2, c), mul_rn_f32(x1, s));
        // SAFETY: the same two positions as the reads.
        unsafe {
            *x.get_unchecked_mut(a) = f32_to_bf16_rne(y1);
            *x.get_unchecked_mut(a + PAIRS_128) = f32_to_bf16_rne(y2);
        }
    }
}

/// The loaded RoPE module for heads of 128.
pub struct Rope128Kernels {
    module: rope2d_128_kernels::LoadedModule,
}

impl Rope128Kernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Rope128Kernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(rope2d_128_kernels, ctx)? };
        Ok(Rope128Kernels { module })
    }

    /// Enqueue the 2D RoPE of the query and key heads of 128 values in place ([`RopeArgs`], its
    /// `cs` the uploaded [`RopeTable128::cs`]). Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: RopeArgs<'_>) -> Result<(), GpuError> {
        let what = "Rope128Kernels::enqueue";
        let RopeArgs {
            cs,
            n,
            row_width,
            q0,
            k0,
            n_heads,
            x,
        } = args;
        let span = n_heads * HEAD_DIM_128;
        if n == 0 || q0 + span > row_width || k0 + span > row_width {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_heads} heads of {HEAD_DIM_128} at columns {q0} and {k0} do not fit rows of {row_width} (n={n})"
                ),
            });
        }
        if x.len() < n * row_width || cs.len() < n * 2 * PAIRS_128 {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "x.len() {} (need {}), cs.len() {} (need {})",
                    x.len(),
                    n * row_width,
                    cs.len(),
                    n * 2 * PAIRS_128
                ),
            });
        }
        let grid = launch_u32(
            what,
            "grid",
            (n * 2 * n_heads * PAIRS_128).div_ceil(BLOCK as usize),
        )?;
        let prep = self
            .module
            .prepare_vis_rope2d_h128(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.vis_rope2d_h128(
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

#[cfg(test)]
mod tests_128 {
    use super::{HEAD_DIM_128, PAIRS_128, RopeTable128, rope_128_ref, rope_source_128};

    /// The permutation is a bijection of a head's 128 values that sends each NORM-mode pair
    /// `(2i, 2i + 1)` of either half to one kernel pair `(j, j + 64)`.
    #[test]
    fn the_permutation_maps_ggmls_pairs_onto_the_kernels() {
        let mut seen = [false; HEAD_DIM_128];
        for r in 0..HEAD_DIM_128 {
            let src = rope_source_128(r);
            assert!(!seen[src], "value {src} twice");
            seen[src] = true;
        }
        for j in 0..PAIRS_128 {
            let (a, b) = (rope_source_128(j), rope_source_128(j + PAIRS_128));
            assert_eq!((a % 2, b, a / 64), (0, a + 1, j / 32), "pair {j}");
        }
    }

    /// Turning the permuted head with the table is ggml's NORM-mode turn of the file's head
    /// (`x0·c − x1·s`, `x0·s + x1·c`, each product and sum rounded in f32), bit for bit: the row's
    /// half at the row's angles, the column's at the column's.
    #[test]
    fn the_permuted_turn_is_ggmls_turn() {
        let (y, x) = (3usize, 5usize);
        let table = RopeTable128::interleaved(4, 6, [(y, x)], 10_000.0);
        let file: Vec<u16> = (0..HEAD_DIM_128)
            .map(|i| crate::f32_bf16(((i * 37 % 101) as f32 - 50.0) / 16.0))
            .collect();
        let mut card: Vec<u16> = (0..HEAD_DIM_128)
            .map(|r| file[rope_source_128(r)])
            .collect();
        rope_128_ref(&mut card, HEAD_DIM_128, 0, 1, &table);
        let theta_scale = 10_000f64.powf(f64::from(-2.0f32 / 64.0)) as f32;
        let odd = 10_000f64.powf(f64::from(-2.0f32 / 128.0)) as f32;
        let mut want = file.clone();
        for (half, pos, freq_scale) in [(0usize, y, 1.0f32), (1, x, odd)] {
            let mut angle = pos as f32;
            for i in 0..32 {
                let a = freq_scale * angle;
                let (c, s) = (f64::from(a).cos() as f32, f64::from(a).sin() as f32);
                let (x0, x1) = (
                    crate::bf16_f32(file[half * 64 + 2 * i]),
                    crate::bf16_f32(file[half * 64 + 2 * i + 1]),
                );
                want[half * 64 + 2 * i] = crate::f32_bf16(x0 * c - x1 * s);
                want[half * 64 + 2 * i + 1] = crate::f32_bf16(x0 * s + x1 * c);
                angle *= theta_scale;
            }
        }
        for (r, &got) in card.iter().enumerate() {
            assert_eq!(got, want[rope_source_128(r)], "value {r}");
        }
    }
}
