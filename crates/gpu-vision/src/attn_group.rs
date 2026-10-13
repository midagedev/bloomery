//! Windowed attention for heads of 72 values: the rows are cut into consecutive groups of `g`
//! (`g` = 4 for MiniCPM-V's vit_merger, whose rows lie in merge order, so a 2×2 window is four
//! consecutive rows), and every row attends to the `g` rows of its own group only —
//! `softmax(scale · q·kᵀ)` over the group's keys, scale `1/√72`, no mask.
//!
//! Reads the fused bf16 `qkv` rows as `vis_attn_h72` does ([`QkvLayout`]: query, key and value at
//! columns `q0 + h·72`, `k0 + h·72`, `v0 + h·72`) and writes head `h` of row `i` at
//! `out[i·out_width + h·72 ..]` as bf16, the layout of that kernel's output, so the out-projection
//! GEMM reads it unchanged. A new entry (`vis_attn_group`), not a parameter of `vis_attn_h72`:
//! that kernel's PTX is the Qwen3-VL tower's and is held.
//!
//! Numeric rule, per (row, head): the bf16 inputs widened exactly; the score of each key an f32
//! fused-multiply-add chain over the lane's 18 values, the four lanes' partial sums added by an
//! xor butterfly, then `× scale`; an online softmax over the group's keys in order (`m' =
//! max(m, s)`, `c = exp(m − m')`, `p = exp(s − m')`, `l = l·c + p`, `o = o·c + p·v`, all f32);
//! the output `o / l` rounded once to bf16. [`attn_group_ref`] is the exact attention (f64) of
//! the same inputs on the group alone, the rule of `attn_72_ref`; the kernel is held inside
//! [`attn_bound`](crate::attn::attn_bound) at `n = g`.
//!
//! Geometry: four lanes per (row, head) — each owns 18 of the head's 72 values, nine u32 words of
//! a row — so one thread block of [`THREADS`] covers 64 (row, head) pairs, and a launch is one
//! block per 64 pairs for every group and head at once: `4·n·heads` threads. The heads of a row
//! are adjacent in a warp (eight of them), so a key row's loads are contiguous across the warp,
//! and the group's other rows read the same key and value words from the cache. State per
//! thread: the 18 query values and 18 output accumulators, which ptxas holds in 64 registers with
//! no spill and no shared memory, so four blocks of 256 threads fit an SM (32 warps resident).
//! The score reduction is a warp collective, so no thread leaves early: a thread past the last
//! pair works on pair 0 and only its store is skipped. For `n = 1024` with 16 heads a launch is
//! 65,536 threads in 256 blocks, against 328 resident blocks on a card of 82 SMs and 336 on one
//! of 84: one wave.

use crate::attn::{HEAD_DIM_72, QkvLayout, attn_72_ref};
use bloomery_gpu::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::convert::f32_to_bf16_rne;
use cuda_device::float::{add_rn_f32, div_rn_f32, fma_rn_f32, mul_rn_f32};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block.
pub const THREADS: usize = 256;
const THREADS_U32: u32 = THREADS as u32;
/// Lanes that share one (row, head): the head's values split four ways.
const LANES: usize = 4;
/// u32 words (bf16 pairs) of a head a lane owns.
const LANE_WORDS: usize = HEAD_DIM_72 / 2 / LANES;
/// Values of a head a lane owns.
const LANE_DIMS: usize = HEAD_DIM_72 / LANES;

const _: () = assert!(LANE_WORDS * LANES * 2 == HEAD_DIM_72 && LANE_DIMS == 2 * LANE_WORDS);
const _: () = assert!(THREADS.is_multiple_of(32) && LANES == 4);

/// The rule's reference for one (head `h`, row `i`): [`attn_72_ref`] on the group of `group` rows
/// that holds `i`, alone. Returns `o[d]`, `Σ_j p_j |v_j[d]|` and the row's largest
/// `scale · Σ_d |q_i[d] k_j[d]|`, the inputs of [`attn_bound`](crate::attn::attn_bound) at
/// `n = group`. `group` divides the row count.
#[must_use]
pub fn attn_group_ref(
    qkv: &[u16],
    lay: QkvLayout,
    group: usize,
    scale: f64,
    h: usize,
    i: usize,
) -> ([f64; HEAD_DIM_72], [f64; HEAD_DIM_72], f64) {
    let first = i - i % group;
    let rows = &qkv[first * lay.row_width..(first + group) * lay.row_width];
    attn_72_ref(rows, lay, group, scale, h, i - first)
}

#[cuda_module]
mod attn_group_kernels {
    use super::*;

    /// The module doc's windowed attention over `n` rows of `qkv` (rows of `row_width` bf16;
    /// heads at columns `q0`, `k0`, `v0` plus `h·72`), groups of `group` consecutive rows,
    /// `n_heads` heads, writing head `h` of row `i` at `out[i·out_width + h·72 ..]`. Thread `t`
    /// is lane `t % 4` of the pair `t / 4`, pair `p` being row `p / n_heads`, head `p % n_heads`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (qkv.len() >= n * row_width, out.len() >= n * out_width)
    )]
    pub fn vis_attn_group(
        qkv: &[u16],
        n: u32,
        group: u32,
        row_width: u32,
        q0: u32,
        k0: u32,
        v0: u32,
        n_heads: u32,
        scale: f32,
        out_width: u32,
        mut out: DisjointSlice<u16>,
    ) {
        let t = thread::index_1d().get();
        let heads = n_heads as usize;
        let pairs = n as usize * heads;
        let live = t / LANES < pairs;
        // A thread past the last pair works on pair 0: the butterfly below needs every lane.
        let pair = if live { t / LANES } else { 0 };
        let lane = t % LANES;
        let row = pair / heads;
        let h = pair % heads;
        let g = group as usize;
        let first = row - row % g;

        let src = qkv.as_ptr().cast::<u32>();
        let rw = (row_width / 2) as usize; // words per qkv row
        let lane_word = lane * LANE_WORDS;
        let (qw, kw, vw) = (
            row * rw + (q0 as usize + h * HEAD_DIM_72) / 2 + lane_word,
            first * rw + (k0 as usize + h * HEAD_DIM_72) / 2 + lane_word,
            first * rw + (v0 as usize + h * HEAD_DIM_72) / 2 + lane_word,
        );

        let mut q = [0.0f32; LANE_DIMS];
        let mut w = 0usize;
        while w < LANE_WORDS {
            cuda_device::thread::__unroll_config::<0>();
            // SAFETY: row < n and the column layout (host-checked) keep the word inside `qkv`
            // (contract), aligned: the base is, and every offset is whole words.
            let word = unsafe { *src.add(qw + w) };
            q[2 * w] = f32::from_bits(word << 16);
            q[2 * w + 1] = f32::from_bits(word & 0xffff_0000);
            w += 1;
        }

        let mut m = f32::NEG_INFINITY;
        let mut l = 0.0f32;
        let mut o = [0.0f32; LANE_DIMS];
        let mut j = 0usize;
        while j < g {
            let mut dot = 0.0f32;
            let mut w = 0usize;
            while w < LANE_WORDS {
                cuda_device::thread::__unroll_config::<0>();
                // SAFETY: first + j < n (the group lies inside the rows, host-checked), so the
                // word is inside `qkv` as the query's is.
                let word = unsafe { *src.add(kw + j * rw + w) };
                dot = fma_rn_f32(q[2 * w], f32::from_bits(word << 16), dot);
                dot = fma_rn_f32(q[2 * w + 1], f32::from_bits(word & 0xffff_0000), dot);
                w += 1;
            }
            dot = add_rn_f32(dot, warp::shuffle_xor_f32(dot, 1));
            dot = add_rn_f32(dot, warp::shuffle_xor_f32(dot, 2));
            let s = mul_rn_f32(dot, scale);
            let m_new = m.max(s);
            let c = (m - m_new).exp();
            let p = (s - m_new).exp();
            l = add_rn_f32(mul_rn_f32(l, c), p);
            let mut w = 0usize;
            while w < LANE_WORDS {
                cuda_device::thread::__unroll_config::<0>();
                // SAFETY: as the key word above, in the value columns.
                let word = unsafe { *src.add(vw + j * rw + w) };
                o[2 * w] = fma_rn_f32(p, f32::from_bits(word << 16), mul_rn_f32(o[2 * w], c));
                o[2 * w + 1] = fma_rn_f32(
                    p,
                    f32::from_bits(word & 0xffff_0000),
                    mul_rn_f32(o[2 * w + 1], c),
                );
                w += 1;
            }
            m = m_new;
            j += 1;
        }

        if live {
            let ow = out_width as usize;
            let mut d = 0usize;
            while d < LANE_DIMS {
                cuda_device::thread::__unroll_config::<0>();
                let y = f32_to_bf16_rne(div_rn_f32(o[d], l));
                // SAFETY: row < n and h·72 + 72 <= n_heads·72 <= out_width (host-checked) keep the
                // index below n·out_width <= out.len() (contract); one owner lane per value.
                unsafe {
                    *out.get_unchecked_mut(row * ow + h * HEAD_DIM_72 + lane * LANE_DIMS + d) = y;
                }
                d += 1;
            }
        }
    }
}

/// [`AttnGroupKernels::enqueue`]'s arguments: `n` rows of `qkv` in `layout`, cut into groups of
/// `group` consecutive rows (`group` divides `n`), the output rows of `out_width` bf16 (the
/// heads side by side from column 0).
pub struct AttnGroupArgs<'a> {
    pub qkv: &'a DeviceBuffer<u16>,
    pub layout: QkvLayout,
    pub n: usize,
    pub group: usize,
    pub scale: f32,
    pub out_width: usize,
    pub out: &'a mut DeviceBuffer<u16>,
}

/// The loaded windowed-attention module.
pub struct AttnGroupKernels {
    module: attn_group_kernels::LoadedModule,
}

impl AttnGroupKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<AttnGroupKernels, GpuError> {
        // SAFETY: this crate owns the embedded device bundle produced for the module above; the
        // launcher checks its launch contract.
        let module = unsafe { bloomery_gpu::shared_module!(attn_group_kernels, ctx)? };
        Ok(AttnGroupKernels { module })
    }

    /// Enqueue the windowed attention ([`AttnGroupArgs`]). A `group` that does not divide `n` is
    /// refused by name. Asynchronous.
    pub fn enqueue(&self, stream: &CudaStream, args: AttnGroupArgs<'_>) -> Result<(), GpuError> {
        let what = "AttnGroupKernels::enqueue";
        let AttnGroupArgs {
            qkv,
            layout: lay,
            n,
            group,
            scale,
            out_width,
            out,
        } = args;
        if group == 0 || n == 0 || !n.is_multiple_of(group) {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n {n} is not a whole number of windows of {group} rows (n % group must be 0)"
                ),
            });
        }
        let span = lay.n_heads * HEAD_DIM_72;
        let fits = |c0: usize| c0.is_multiple_of(2) && c0 + span <= lay.row_width;
        if lay.n_heads == 0
            || !lay.row_width.is_multiple_of(2)
            || !fits(lay.q0)
            || !fits(lay.k0)
            || !fits(lay.v0)
            || span > out_width
            || qkv.len() < n * lay.row_width
            || out.len() < n * out_width
        {
            return Err(GpuError::Shape {
                what,
                detail: format!(
                    "n {n}, group {group}, {lay:?}, out_width {out_width}: qkv.len() {}, out.len() {}",
                    qkv.len(),
                    out.len()
                ),
            });
        }
        // The launch contract multiplies in u32: both products must fit it.
        launch_u32(what, "n * row_width", n * lay.row_width)?;
        launch_u32(what, "n * out_width", n * out_width)?;
        let grid = launch_u32(what, "grid", (n * lay.n_heads * LANES).div_ceil(THREADS))?;
        let prep = self
            .module
            .prepare_vis_attn_group(LaunchConfig1D::new(grid, THREADS_U32, 0))?;
        self.module.vis_attn_group(
            stream,
            &prep,
            qkv,
            launch_u32(what, "n", n)?,
            launch_u32(what, "group", group)?,
            launch_u32(what, "row_width", lay.row_width)?,
            launch_u32(what, "q0", lay.q0)?,
            launch_u32(what, "k0", lay.k0)?,
            launch_u32(what, "v0", lay.v0)?,
            launch_u32(what, "n_heads", lay.n_heads)?,
            scale,
            launch_u32(what, "out_width", out_width)?,
            out,
        )?;
        Ok(())
    }
}
