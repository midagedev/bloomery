//! The dense projections of the V4.1 step — attention, the shared expert,
//! `engram_wkv` — in the format the file carries them in. The kernel of a
//! projection follows from its resident weight's format ([`Dense::of`]),
//! resolved when the step is enqueued (at capture, once per graph); a step
//! never branches on it.
//!
//! - Q8_0: `q8f32`'s gemv, f32 activations.
//! - Q3_K and Q4_K: `bloomery_gpu`'s gemvs against the q8_1 form of the
//!   activation (`Gpu::enqueue_gemv_q3k`, `Gpu::enqueue_gemv_q4k`), the
//!   rule every K-quant site of the card runs — `cores::q3k_row_dot`,
//!   `cores::q4k_row_dot`, ik's CUDA `mmvq` shape. The caller quantizes the
//!   activation once for every projection that reads it; a Q3_K row may have
//!   an odd super-block count.
//! - Q5_K: [`DenseKernels`]'s f32-activation gemv, the one Q5_K dense site
//!   (two layers' shared down projection): each weight dequantized as
//!   `gguf::quant::dequant_row` does it, then one fused multiply-add.
//!
//! Two shapes of their own, where the file's format is a K-quant:
//! - `attn_output_a`'s block diagonal ([`DenseKernels::enqueue_q3k_heads`]):
//!   group `g`'s rows dot column `g` of a q8_1 activation of `groups`
//!   columns — each row bit for bit what `q3k_gemv` computes for it on that
//!   column alone;
//! - the shared expert's gate·up·SwiGLU
//!   ([`DenseKernels::enqueue_shexp_gate_up_q3k`]): both Q3_K dots against the
//!   one q8_1 column, then `experts::swiglu_clamp` — the routed experts'
//!   `ds41_expert_gate_up` on one expert of its own.
//!
//! Numeric contract of the Q5_K gemv: one warp per row; lane `L` walks the
//! row's super-blocks in order, and in each its eight values `64j + L` and
//! `64j + 32 + L` for `j` ascending, the pair in that order; a value is
//! `fma(q, d·sc, −(dmin·m))` with the two products rounded first
//! (`dequant_q5_k`'s op order), and `acc = fma(value, x, acc)` from 0. The
//! 32 lane sums then go through `warp::reduce_sum_f32`.

use bloomery_gpu::cores::q3k_row_dot;
use bloomery_gpu::weights::{DevWeight, Weights};
use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::{cvt_f32_f16x2_hi, cvt_f32_f16x2_lo};
use cuda_device::float::{fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use gguf::quant::GgmlType;
use std::sync::Arc;

use crate::experts::swiglu_clamp;

const WHAT: &str = "deepseek41 dense";

/// Threads per block of every kernel here: eight warps, a warp per row.
const BLOCK: u32 = 256;

/// Bytes, and u32 words, of one Q5_K super-block of 256 values: `d` and
/// `dmin` (f16), 12 scale bytes, 32 high-bit bytes, 128 low-nibble bytes.
const Q5K_BYTES: usize = 176;
const Q5K_WORDS: usize = Q5K_BYTES / 4;

/// Byte `i` of the words from `base`.
///
/// # Safety
///
/// `base + i / 4 < w.len()`.
#[inline(always)]
unsafe fn byte(w: &[u32], base: usize, i: usize) -> u32 {
    // SAFETY: the covering word is inside `w` by this function's contract.
    (unsafe { *w.get_unchecked(base + i / 4) } >> (8 * (i % 4))) & 0xff
}

/// `get_scale_min_k4(j, scales)` of the super-block whose scale bytes start
/// at word `sc` (byte 4 of the super-block): the 6-bit scale and min of
/// sub-block `j` (0..8).
///
/// # Safety
///
/// `sc + 3 <= w.len()`.
#[inline(always)]
unsafe fn scale_min(w: &[u32], sc: usize, j: usize) -> (u32, u32) {
    // SAFETY: every byte index below is under 12, inside the three words
    // from `sc` by this function's contract.
    unsafe {
        if j < 4 {
            (byte(w, sc, j) & 63, byte(w, sc, j + 4) & 63)
        } else {
            (
                (byte(w, sc, j + 4) & 0x0f) | ((byte(w, sc, j - 4) >> 6) << 4),
                (byte(w, sc, j + 4) >> 4) | ((byte(w, sc, j) >> 6) << 4),
            )
        }
    }
}

/// Lane `lane`'s partial sum of Q5_K row `row` (`n_sb` super-blocks, word
/// aligned: 176 bytes each) against the f32 column `x`, in the module's
/// contract order.
///
/// # Safety
///
/// `w.len() >= (row + 1) · 44 · n_sb`, `x.len() >= 256 · n_sb`, `lane < 32`.
#[inline(always)]
unsafe fn q5k_lane_partial(w: &[u32], x: &[f32], n_sb: usize, row: usize, lane: usize) -> f32 {
    let mut acc = 0.0f32;
    let mut sb = 0usize;
    while sb < n_sb {
        let base = (row * n_sb + sb) * Q5K_WORDS;
        // SAFETY: base + 44 <= w.len() for sb < n_sb by this function's
        // contract; every word read below is one of those 44.
        let (dm, qh) = unsafe { (*w.get_unchecked(base), byte(w, base + 4, lane)) };
        let (d, dmin) = (cvt_f32_f16x2_lo(dm), cvt_f32_f16x2_hi(dm));
        let mut j = 0usize;
        while j < 4 {
            // SAFETY: the scale words base+1 .. base+4 and the low-nibble
            // byte 32j + lane of the 128 from word base + 12 are inside the
            // super-block.
            let ((sc1, m1), (sc2, m2), ql) = unsafe {
                (
                    scale_min(w, base + 1, 2 * j),
                    scale_min(w, base + 1, 2 * j + 1),
                    byte(w, base + 12, 32 * j + lane),
                )
            };
            let d1 = mul_rn_f32(d, sc1 as f32);
            let n1 = mul_rn_f32(dmin, m1 as f32);
            let d2 = mul_rn_f32(d, sc2 as f32);
            let n2 = mul_rn_f32(dmin, m2 as f32);
            let q1 = (ql & 0x0f) + 16 * ((qh >> (2 * j)) & 1);
            let q2 = (ql >> 4) + 16 * ((qh >> (2 * j + 1)) & 1);
            let v1 = fma_rn_f32(q1 as f32, d1, -n1);
            let v2 = fma_rn_f32(q2 as f32, d2, -n2);
            let at = 256 * sb + 64 * j + lane;
            // SAFETY: at + 32 < 256 · n_sb <= x.len() by this function's
            // contract.
            let (x1, x2) = unsafe { (*x.get_unchecked(at), *x.get_unchecked(at + 32)) };
            acc = fma_rn_f32(v1, x1, acc);
            acc = fma_rn_f32(v2, x2, acc);
            j += 1;
        }
        sb += 1;
    }
    acc
}

#[cuda_module]
mod dense_kernels {
    use super::*;

    /// `y[r] = W[r] · x` for a Q5_K weight of `n_rows` rows of `n_sb`
    /// super-blocks and one f32 column `x`: a warp per row, eight rows per
    /// block, the module's contract order.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            w.len() >= n_rows * 44 * n_sb,
            x.len() >= 256 * n_sb,
            y.len() >= n_rows
        )
    )]
    pub fn ds41_q5k_gemv_f32(
        w: &[u32],
        x: &[f32],
        n_rows: u32,
        n_sb: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // SAFETY: row < n_rows puts the row's words inside w, and x holds
        // 256·n_sb values, by the launch contract; lane < 32.
        let f = unsafe { q5k_lane_partial(w, x, n_sb as usize, row, lane) };
        let s = warp::reduce_sum_f32(f);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len(); lane 0 of the row's warp is
            // its only writer.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// The block diagonal of a Q3_K weight: row `r` of `n_rows` belongs to
    /// group `r / rows_per_head` and dots that column of the q8_1 activation
    /// (`groups` columns of `n_sb` super-blocks), `cores::q3k_row_dot` at
    /// one column; `y[r]` from lane 0.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * w.len() >= n_rows * 110 * n_sb,
            q.len() >= groups * 64 * iters,
            d8.len() >= groups * 2 * n_sb,
            groups * rows_per_head >= n_rows,
            y.len() >= n_rows
        )
    )]
    pub fn ds41_q3k_gemv_heads(
        w: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        rows_per_head: u32,
        groups: u32,
        n_sb: u32,
        iters: u32,
        mut y: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        let col = row / rows_per_head as usize;
        // The second test is the contract's, warp-uniform like the first.
        if row >= n_rows as usize || col >= groups as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        // q3k_row_dot's contract: the row inside w, and column col (< groups,
        // as row < n_rows <= groups·rows_per_head) inside q and d8, all by the
        // launch contract; every lane of the warp calls it.
        let f = q3k_row_dot(w, q, d8, n_sb as usize, iters, row, col, 1, lane);
        let s = warp::reduce_sum_f32(f[0]);
        if lane == 0 {
            // SAFETY: row < n_rows <= y.len(); lane 0 of the row's warp is
            // its only writer.
            unsafe { *y.get_unchecked_mut(row) = s };
        }
    }

    /// The shared expert's gate·up·SwiGLU for one token, Q3_K weights: row
    /// `r` (a warp per row) dots gate row `r` and up row `r` against the one
    /// q8_1 column, reduces each with the warp tree and stores
    /// `h[r] = swiglu_clamp(g, u, limit)` from lane 0.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            4 * wg.len() >= n_rows * 110 * n_sb,
            4 * wu.len() >= n_rows * 110 * n_sb,
            q.len() >= 64 * iters,
            d8.len() >= 2 * n_sb,
            h.len() >= n_rows
        )
    )]
    pub fn ds41_shexp_gate_up_q3k(
        wg: &[u32],
        wu: &[u32],
        q: &[u64],
        d8: &[f32],
        n_rows: u32,
        n_sb: u32,
        iters: u32,
        limit: f32,
        mut h: DisjointSlice<f32>,
    ) {
        let t = thread::index_1d().get() % 256;
        let row = (thread::index_1d().get() / 256) * 8 + t / 32;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let fg = q3k_row_dot(wg, q, d8, n_sb as usize, iters, row, 0, 1, lane);
        let fu = q3k_row_dot(wu, q, d8, n_sb as usize, iters, row, 0, 1, lane);
        let g = warp::reduce_sum_f32(fg[0]);
        let u = warp::reduce_sum_f32(fu[0]);
        if lane == 0 {
            let v = swiglu_clamp(g, u, limit);
            // SAFETY: row < n_rows <= h.len(); lane 0 of the row's warp is
            // its only writer.
            unsafe { *h.get_unchecked_mut(row) = v };
        }
    }
}

/// A dense projection's resident weight, by its file format.
#[derive(Clone, Copy)]
pub enum Dense<'w> {
    Q8_0 {
        qs: &'w DeviceTensor<u32>,
        d: &'w DeviceTensor<u16>,
    },
    Q3K(&'w DeviceTensor<u32>),
    Q4K(&'w DeviceTensor<u32>),
    Q5K(&'w DeviceTensor<u32>),
}

impl<'w> Dense<'w> {
    /// `name`'s resident weight, refused unless it projects `k` values onto
    /// `rows` rows in a format this module runs.
    pub fn of(w: &'w Weights, name: &str, k: usize, rows: usize) -> Result<Dense<'w>, GpuError> {
        let shape = |got_k: usize, got_rows: usize| GpuError::Shape {
            what: WHAT,
            detail: format!("{name} is {got_rows} rows of {got_k} values, want {rows} of {k}"),
        };
        match w.get(name) {
            Some(DevWeight::Q8_0 { qs, d, k: wk }) => {
                if *wk != k || d.rows() != rows {
                    return Err(shape(*wk, d.rows()));
                }
                Ok(Dense::Q8_0 { qs, d })
            }
            Some(DevWeight::KQuant { ty, w: t, k: wk }) => {
                if *wk != k || t.rows() != rows {
                    return Err(shape(*wk, t.rows()));
                }
                match ty {
                    GgmlType::Q3_K => Ok(Dense::Q3K(t)),
                    GgmlType::Q4_K => Ok(Dense::Q4K(t)),
                    GgmlType::Q5_K => Ok(Dense::Q5K(t)),
                    _ => Err(GpuError::Tensor {
                        what: WHAT,
                        name: name.to_string(),
                        need: "Q8_0, Q3_K, Q4_K or Q5_K",
                    }),
                }
            }
            found => Err(GpuError::Tensor {
                what: WHAT,
                name: name.to_string(),
                need: if found.is_some() {
                    "Q8_0, Q3_K, Q4_K or Q5_K"
                } else {
                    "resident"
                },
            }),
        }
    }

    /// Whether the projection reads its input in q8_1.
    #[must_use]
    pub fn reads_q8_1(&self) -> bool {
        matches!(self, Dense::Q3K(_) | Dense::Q4K(_))
    }
}

/// The loaded module. Owns no context and no stream — every enqueue takes
/// the engine stream, so the launches order with the rest of the step and
/// are capturable.
pub struct DenseKernels {
    module: dense_kernels::LoadedModule,
}

impl DenseKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<DenseKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; every launcher checks its launch contract.
        let module = unsafe { dense_kernels::load(ctx)? };
        Ok(DenseKernels { module })
    }

    /// Enqueue `y = W · x` for one token: `x` its f32 input, `act` that
    /// input's q8_1 form when `d` reads one ([`Dense::reads_q8_1`]; the
    /// caller quantized it), `y` one f32 per row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue(
        &self,
        gpu: &Gpu,
        d: Dense<'_>,
        x: &DeviceBuffer<f32>,
        act: Option<&Q8Act>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let stream = gpu.stream();
        let act = || {
            act.ok_or_else(|| GpuError::Shape {
                what: WHAT,
                detail: "a K-quant projection without its input's q8_1 form".to_string(),
            })
        };
        match d {
            Dense::Q8_0 { qs, d } => gpu.q8f32().enqueue_q8_0_gemv(stream, qs, d, x, 1, y),
            Dense::Q3K(w) => gpu.enqueue_gemv_q3k(w, act()?, y),
            Dense::Q4K(w) => gpu.enqueue_gemv_q4k(w, act()?, y),
            Dense::Q5K(w) => self.enqueue_q5k(stream, w, x, y),
        }
    }

    /// Enqueue the Q5_K f32-activation gemv of `w` (rows of whole
    /// super-blocks, word aligned) over the one column `x`.
    fn enqueue_q5k(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        x: &DeviceBuffer<f32>,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_q5k";
        let n_rows = w.rows();
        if w.cols() == 0 || !w.cols().is_multiple_of(Q5K_WORDS) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "Q5_K rows of {} words: not whole super-blocks of {Q5K_WORDS}",
                    w.cols()
                ),
            });
        }
        let n_sb = w.cols() / Q5K_WORDS;
        if x.len() < 256 * n_sb || y.len() < n_rows {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "x.len() {} < {} or y.len() {} < {n_rows}",
                    x.len(),
                    256 * n_sb,
                    y.len()
                ),
            });
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_ds41_q5k_gemv_f32(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module
            .ds41_q5k_gemv_f32(stream, &prep, w.buf(), x, n_rows, n_sb, y)?;
        Ok(())
    }

    /// Enqueue the block diagonal of Q3_K weight `w` (`w.rows()` rows of
    /// `act.n_sb()` super-blocks): row `r` against column `r /
    /// rows_per_head` of `act`, which holds one column per group, the groups
    /// covering every row. `y` takes one f32 per row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_q3k_heads(
        &self,
        stream: &CudaStream,
        w: &DeviceTensor<u32>,
        act: &Q8Act,
        rows_per_head: usize,
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_q3k_heads";
        let (n_rows, groups, n_sb) = (w.rows(), act.m(), act.n_sb());
        if rows_per_head == 0 || n_rows != groups * rows_per_head {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "{n_rows} rows are not {groups} groups of {rows_per_head}: one q8_1 column \
                     per group"
                ),
            });
        }
        kquant_rows(what, w, 110, n_sb)?;
        if y.len() < n_rows {
            return Err(GpuError::Shape {
                what,
                detail: format!("y.len() {} < {n_rows}", y.len()),
            });
        }
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let rows_per_head = launch_u32(what, "rows_per_head", rows_per_head)?;
        let groups = launch_u32(what, "groups", groups)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_ds41_q3k_gemv_heads(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_q3k_gemv_heads(
            stream,
            &prep,
            w.buf(),
            act.q3(),
            act.d8(),
            n_rows,
            rows_per_head,
            groups,
            n_sb,
            n_sb.div_ceil(2),
            y,
        )?;
        Ok(())
    }

    /// Enqueue the shared expert's gate·up·SwiGLU for one token from Q3_K
    /// `gate` and `up` (the same rows of `act.n_sb()` super-blocks) and the
    /// token's q8_1 column `act`; `h` takes one f32 per row. Asynchronous,
    /// allocation-free, capturable.
    pub fn enqueue_shexp_gate_up_q3k(
        &self,
        stream: &CudaStream,
        gate: &DeviceTensor<u32>,
        up: &DeviceTensor<u32>,
        act: &Q8Act,
        limit: f32,
        h: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "DenseKernels::enqueue_shexp_gate_up_q3k";
        let (n_rows, n_sb) = (gate.rows(), act.n_sb());
        if act.m() != 1 || up.rows() != n_rows || h.len() < n_rows {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "one q8_1 column, gate and up of one height and h of it: m {}, gate {} \
                     rows, up {}, h.len() {}",
                    act.m(),
                    n_rows,
                    up.rows(),
                    h.len()
                ),
            });
        }
        kquant_rows(what, gate, 110, n_sb)?;
        kquant_rows(what, up, 110, n_sb)?;
        let grid = launch_u32(what, "grid", n_rows.div_ceil(8))?;
        let n_rows = launch_u32(what, "n_rows", n_rows)?;
        let n_sb = launch_u32(what, "n_sb", n_sb)?;
        let prep = self
            .module
            .prepare_ds41_shexp_gate_up_q3k(LaunchConfig1D::new(grid, BLOCK, 0))?;
        self.module.ds41_shexp_gate_up_q3k(
            stream,
            &prep,
            gate.buf(),
            up.buf(),
            act.q3(),
            act.d8(),
            n_rows,
            n_sb,
            n_sb.div_ceil(2),
            limit,
            h,
        )?;
        Ok(())
    }
}

/// Refuse a K-quant tensor whose rows are not `n_sb` super-blocks of
/// `sb_bytes`: its flat word stream is the upload's (`CardFormat::KQuant`,
/// zero-padded at its end to whole words per row).
fn kquant_rows(
    what: &'static str,
    w: &DeviceTensor<u32>,
    sb_bytes: usize,
    n_sb: usize,
) -> Result<(), GpuError> {
    let rows = w.rows();
    let words = (rows * sb_bytes * n_sb).div_ceil(4).div_ceil(rows.max(1));
    if rows == 0 || n_sb == 0 || w.cols() != words {
        return Err(GpuError::Shape {
            what,
            detail: format!(
                "{rows} rows of {} words: not rows of {n_sb} super-blocks of {sb_bytes} bytes \
                 ({words} words a row)",
                w.cols()
            ),
        });
    }
    Ok(())
}
