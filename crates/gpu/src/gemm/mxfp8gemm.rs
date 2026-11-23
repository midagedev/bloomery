//! The 32-value GEMM's MXFP8 arm, `gemm_mxfp8p`: an MXFP8 stack in its two
//! planes ([`Mxfp8Stack`], `CardFormat::Mxfp8Planes`) times [`GemmAct32`]
//! columns over a route table, the launch geometry and route rules of the
//! family's other entries (`gemm32.rs`): one block per (tile,
//! [`GEMM_BM`]-row slab), slot `s` reading activation column `s / slot_div`
//! and writing `y[s · rows + r]`, the blocks of tile 0 writing NaN over their
//! slab's rows of every slot the route refused.
//!
//! Numeric contract: the routed `_sel` pair's integer rule
//! ([`crate::mxfp8_sel`]): a block's `isum` the exact i32 dot of its value
//! bytes as `2^9`-scaled integers with the column's int8 codes, entering lane
//! `ℓ`'s partial (blocks `ℓ, ℓ + 32, ..`) as `acc + (f32(isum) · 2^(E − 136))
//! · d_a`, each operation rounded once, `d_a` the column's 32-value block
//! scale; the warp tree reduces the partials. An output is a function of its
//! weight row and its column's codes and scales only, so the bits are the
//! `_sel` entries' for the same codes and scales. Each value's integer
//! (`|M| < 2^18`) enters as three signed 7-bit digits, `M = s · (A + 2^7 B +
//! 2^14 C)`, each digit plane one `dp4a` a word, `isum = I_A + 2^7 I_B + 2^14
//! I_C` exact in i32. An output past f32's range is the infinity or NaN that
//! rule gives, never a finite value.
//!
//! Geometry: [`THREADS`] threads, eight warps of sixteen rows of the slab;
//! each warp walks its rows one after the other and a row's tile columns
//! [`COLS`] at a time, a lane's digit words of a block held across the
//! columns.

use super::act32::GemmAct32;
use super::grouped::GEMM_BM;
use super::route::{GemmRoute, tile_parts};
use crate::mxfp8_sel::{Mxfp8Stack, block_scale, e4m3_int};
use crate::{GpuError, launch_u32};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer, LaunchConfig1D};
use cuda_device::dotprod::dp4a_s32;
use cuda_device::float::{add_rn_f32, mul_rn_f32};
use cuda_device::vector::{U32x4, as_vectors};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;
use std::sync::Arc;

/// Threads per block: eight warps, sixteen slab rows each.
const THREADS: u32 = 256;
/// Tile columns a warp's pass over a row carries.
const COLS: usize = 8;

// The entry's launch attributes spell the block as a literal; a slab is
// eight warps of sixteen rows.
const _: () = assert!(THREADS == 256 && GEMM_BM == 8 * 16);

/// The three signed 7-bit digit words of a word of four E4M3 bytes: byte `j`
/// of digit word `p` is `s · ((|M| >> 7p) & 127)` of byte `j`'s integer `M =
/// s · |M|` ([`e4m3_int`]), as an int8. `|M| < 2^18`, so `M = Σ_p 2^(7p) ·
/// digit_p` exactly.
#[inline(always)]
#[must_use]
pub fn digits(w: u32) -> [u32; 3] {
    let mut d = [0u32; 3];
    let mut j = 0;
    while j < 4 {
        let m = e4m3_int((w >> (8 * j)) & 0xFF);
        let a = m.unsigned_abs();
        let parts = [a & 127, (a >> 7) & 127, a >> 14];
        let mut p = 0;
        while p < 3 {
            let v = if m < 0 {
                parts[p].wrapping_neg()
            } else {
                parts[p]
            };
            d[p] |= (v & 0xFF) << (8 * j);
            p += 1;
        }
        j += 1;
    }
    d
}

#[cuda_module]
mod mxfp8_gemm_kernels {
    use super::*;

    /// The MXFP8 arm of the 32-value GEMM (module doc): `qs` is `n_experts ·
    /// rows` rows of `8 · kb` value words, `e` their `e_cols` scale words; the
    /// activations a `GemmAct32` of `act_cols` columns (`16 · steps` code
    /// words and `2 · steps` scales each); the route table (`cols`, `tiles`,
    /// `n_tiles`) built for `n_slots` slots of this stack, slot `s` reading
    /// column `s / slot_div`. Block `b` is tile `b / row_tiles`, row slab `b %
    /// row_tiles`.
    #[allow(
        clippy::too_many_arguments,
        reason = "kernel entry: the device ABI takes the arguments flat (rust-quality R8)"
    )]
    #[kernel]
    #[launch_bounds(256, 2)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        requires = (
            qs.len() >= n_experts * rows * 8 * kb,
            e.len() >= n_experts * rows * e_cols,
            4 * e_cols >= kb,
            2 * steps >= kb,
            q.len() >= act_cols * 16 * steps,
            d.len() >= act_cols * 2 * steps,
            cols.len() >= n_slots,
            tiles.len() >= 2 * max_tiles,
            n_tiles.len() >= 2,
            slot_div >= 1,
            y.len() >= n_slots * rows
        )
    )]
    pub fn gemm_mxfp8p(
        qs: &[u32],
        e: &[u32],
        q: &[u32],
        d: &[f32],
        cols: &[u32],
        tiles: &[u32],
        n_tiles: &[u32],
        n_experts: u32,
        rows: u32,
        kb: u32,
        e_cols: u32,
        steps: u32,
        act_cols: u32,
        n_slots: u32,
        max_tiles: u32,
        slot_div: u32,
        row_tiles: u32,
        mut y: DisjointSlice<f32>,
    ) {
        // Read by the launch contract only.
        let _ = (n_experts, act_cols, max_tiles);
        let b = thread::blockIdx_x() as usize;
        let rt_n = row_tiles as usize;
        let tile = b / rt_n;
        let rt = b - tile * rt_n;
        let tid = thread::threadIdx_x() as usize;
        let rows_n = rows as usize;
        if tile == 0 {
            // SAFETY: n_tiles.len() >= 2 by the launch contract.
            let refused = unsafe { *n_tiles.get_unchecked(1) } as usize;
            // Block-uniform: every thread read the same word.
            if refused > 0 {
                let first = n_slots as usize - refused;
                let r0 = rt * GEMM_BM;
                let mut u = tid;
                while u < refused * GEMM_BM {
                    let r = r0 + u % GEMM_BM;
                    if r < rows_n {
                        // SAFETY: the route wrote `refused` slots below
                        // n_slots at cols[first..n_slots], inside cols; the
                        // slot's row r is below n_slots · rows <= y.len(), and
                        // (slot, r) is this thread's alone.
                        unsafe {
                            let slot = *cols.get_unchecked(first + u / GEMM_BM) as usize;
                            *y.get_unchecked_mut(slot * rows_n + r) = f32::NAN;
                        }
                    }
                    u += THREADS as usize;
                }
            }
        }
        // SAFETY: n_tiles.len() >= 2 by the launch contract.
        if tile >= unsafe { *n_tiles.get_unchecked(0) } as usize {
            return; // block-uniform: the grid's bound is the table's worst case
        }
        // SAFETY: tile < n_tiles <= max_tiles, and tiles.len() >= 2 · max_tiles.
        let (ex, packed) = unsafe {
            (
                *tiles.get_unchecked(2 * tile) as usize,
                *tiles.get_unchecked(2 * tile + 1),
            )
        };
        let (start, len) = tile_parts(packed);
        let (start, len) = (start as usize, len as usize);
        // The launcher checks 16-byte bases; a row is `2 · kb` vectors and a
        // column `4 · steps`.
        let (Some(qv), Some(av)) = (as_vectors::<U32x4>(qs), as_vectors::<U32x4>(q)) else {
            return;
        };
        let lane = warp::lane_id() as usize;
        let row0 = rt * GEMM_BM + 16 * (tid / 32);
        let (kb, e_cols, steps) = (kb as usize, e_cols as usize, steps as usize);
        let (slot_div, col_d) = (slot_div as usize, 2 * steps);
        let mut i = 0usize;
        while i < 16 {
            let r = row0 + i;
            if r >= rows_n {
                break; // warp-uniform
            }
            let row_abs = ex * rows_n + r;
            let mut c0 = 0usize;
            while c0 < len {
                let nc = (len - c0).min(COLS);
                // The pass's slots and their activation columns; a column past
                // the tile's length repeats the pass's first.
                let mut slot = [0usize; COLS];
                let mut acol = [0usize; COLS];
                for j in 0..COLS {
                    ::cuda_device::thread::__unroll_config::<0>();
                    // SAFETY: start + c0 + j < start + len <= n_slots <=
                    // cols.len() for a live j, and start + c0 for the rest.
                    let s = unsafe { *cols.get_unchecked(start + c0 + if j < nc { j } else { 0 }) };
                    slot[j] = s as usize;
                    acol[j] = s as usize / slot_div;
                }
                let mut acc = [0.0f32; COLS];
                let mut blk = lane;
                while blk < kb {
                    // SAFETY: row_abs < n_experts · rows and blk < kb bound the
                    // block's two vectors and its scale word inside the planes
                    // (launch contract).
                    let (lo, hi, ew) = unsafe {
                        (
                            qv.get_unchecked(2 * (row_abs * kb + blk)).0,
                            qv.get_unchecked(2 * (row_abs * kb + blk) + 1).0,
                            *e.get_unchecked(row_abs * e_cols + blk / 4),
                        )
                    };
                    let sc = block_scale((ew >> (8 * (blk & 3))) & 0xFF);
                    let dg = [
                        digits(lo[0]),
                        digits(lo[1]),
                        digits(lo[2]),
                        digits(lo[3]),
                        digits(hi[0]),
                        digits(hi[1]),
                        digits(hi[2]),
                        digits(hi[3]),
                    ];
                    for j in 0..COLS {
                        ::cuda_device::thread::__unroll_config::<0>();
                        if j < nc {
                            let c = acol[j];
                            // SAFETY: c < act_cols (the route's columns), so its
                            // block blk's two code vectors (vectors 4 · steps · c
                            // + 2 · blk) and its scale are inside the planes.
                            let (a0, a1, da) = unsafe {
                                (
                                    av.get_unchecked(4 * steps * c + 2 * blk).0,
                                    av.get_unchecked(4 * steps * c + 2 * blk + 1).0,
                                    *d.get_unchecked(c * col_d + blk),
                                )
                            };
                            let a = [a0[0], a0[1], a0[2], a0[3], a1[0], a1[1], a1[2], a1[3]];
                            let (mut ia, mut ib, mut ic) = (0i32, 0i32, 0i32);
                            for t in 0..8 {
                                ::cuda_device::thread::__unroll_config::<0>();
                                ia = dp4a_s32(dg[t][0], a[t], ia);
                                ib = dp4a_s32(dg[t][1], a[t], ib);
                                ic = dp4a_s32(dg[t][2], a[t], ic);
                            }
                            let isum = ia + (ib << 7) + (ic << 14);
                            let v = mul_rn_f32(isum as f32, sc);
                            acc[j] = add_rn_f32(acc[j], mul_rn_f32(v, da));
                        }
                    }
                    blk += 32;
                }
                for j in 0..COLS {
                    ::cuda_device::thread::__unroll_config::<0>();
                    let s = warp::reduce_sum_f32(acc[j]);
                    if lane == 0 && j < nc {
                        // SAFETY: slot[j] < n_slots and r < rows, so the cell is
                        // inside y; the slot is in this tile alone and the row
                        // in this warp's slab rows alone.
                        unsafe { *y.get_unchecked_mut(slot[j] * rows_n + r) = s };
                    }
                }
                c0 += COLS;
            }
            i += 1;
        }
    }
}

/// The loaded MXFP8 GEMM module ([`Mxfp8GemmKernels::enqueue`]).
pub struct Mxfp8GemmKernels {
    module: mxfp8_gemm_kernels::LoadedModule,
}

/// A checked launch's scalars, in the entry's order (`gemm32.rs`'s
/// `Gemm32Launch`).
pub(super) struct Mxfp8Launch {
    pub(super) grid: u32,
    pub(super) n_experts: u32,
    pub(super) rows: u32,
    pub(super) kb: u32,
    pub(super) steps: u32,
    pub(super) act_cols: u32,
    pub(super) n_slots: u32,
    pub(super) max_tiles: u32,
    pub(super) slot_div: u32,
    pub(super) row_tiles: u32,
}

impl Mxfp8GemmKernels {
    /// Load this file's device bundle into `ctx`. Load-time only.
    pub fn load(ctx: &Arc<CudaContext>) -> Result<Mxfp8GemmKernels, GpuError> {
        // SAFETY: this package owns the embedded device bundle produced for
        // the module above; the launcher checks its launch contract.
        let module = unsafe { crate::shared_module!(mxfp8_gemm_kernels, ctx)? };
        Ok(Mxfp8GemmKernels { module })
    }

    /// Check `w`'s planes for `n_experts · rows` rows of `k` values (the
    /// activations' width): the two planes' widths, whole experts, and the
    /// code plane's 16-byte base. A named error otherwise.
    pub(super) fn check(
        what: &'static str,
        w: Mxfp8Stack<'_>,
        k: usize,
        rows: usize,
    ) -> Result<(), GpuError> {
        w.experts(what, k, rows)?;
        crate::tensor::check_alignment::<U32x4, _>(what, "MXFP8 code plane", w.qs.buf())
    }

    /// Enqueue `gemm_mxfp8p` over the checked launch `g`: the stack `w`
    /// ([`Mxfp8GemmKernels::check`]ed), the activations `act` and `route`'s
    /// last fill, into `y`. Asynchronous, allocation-free, capturable.
    pub(super) fn enqueue(
        &self,
        stream: &CudaStream,
        g: &Mxfp8Launch,
        (w, act, route): (Mxfp8Stack<'_>, &GemmAct32, &GemmRoute),
        y: &mut DeviceBuffer<f32>,
    ) -> Result<(), GpuError> {
        let what = "Gemm32Kernels::enqueue_gemm32";
        crate::tensor::check_alignment::<U32x4, _>(what, "GemmAct32 code plane", &act.q)?;
        let e_cols = launch_u32(what, "e_cols", w.e.cols())?;
        let prep = self
            .module
            .prepare_gemm_mxfp8p(LaunchConfig1D::new(g.grid, THREADS, 0))?;
        self.module.gemm_mxfp8p(
            stream,
            &prep,
            w.qs.buf(),
            w.e.buf(),
            &act.q,
            &act.d,
            &route.cols,
            &route.tiles,
            &route.n_tiles,
            g.n_experts,
            g.rows,
            g.kb,
            e_cols,
            g.steps,
            g.act_cols,
            g.n_slots,
            g.max_tiles,
            g.slot_div,
            g.row_tiles,
            y,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gpu;
    use crate::gemm::{Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmInput, GemmKernels};
    use crate::mxfp8_sel::rule_dot;
    use crate::mxfp8_sel::tests::{EXPERTS, f64_band, file_bytes, next, planes, row_of};

    /// `gemm_mxfp8p` over a route of three tokens of four slots over five
    /// experts (duplicates within and across tokens), K 1120 (an odd block
    /// count, a last scale word of three blocks), 48 rows an expert: every
    /// output is [`rule_dot`] over the `GemmAct32`'s own codes and scales bit
    /// for bit, and lies within the q8 band of the f64 dot of `gguf`'s
    /// dequantized row with the unquantized column.
    #[test]
    #[ignore = "needs a CUDA device; `just gate-gpu-lib` runs it"]
    fn hw_mxfp8_gemm_is_the_sel_rule() {
        const K: usize = 1120;
        const ROWS: usize = 48;
        const TOP_K: usize = 4;
        const M: usize = 3;
        let ids: [u32; M * TOP_K] = [3, 0, 1, 4, 0, 3, 1, 1, 4, 2, 3, 0];
        let gpu = Gpu::new().expect("card");
        let stream = gpu.stream();
        let g32 = Gemm32Kernels::load(gpu.context()).expect("the 32-value GEMM");
        let gemm = GemmKernels::load(gpu.context()).expect("the route");
        let bytes = file_bytes(EXPERTS * ROWS, K, 0x61);
        let (qs, e) = planes(stream, &bytes, EXPERTS * ROWS, K);
        let mut state = 0x99u64;
        let x: Vec<f32> = (0..M * K)
            .map(|_| ((next(&mut state) >> 33) % 2001) as f32 / 1000.0 - 1.0)
            .collect();
        let xb = DeviceBuffer::from_host(stream, &x).expect("x");
        let mut act = GemmAct32::new(stream, M, K).expect("act");
        g32.enqueue_quantize_gemm32(stream, &xb, M, &mut act, gpu.unlabelled_sink())
            .expect("quantize");
        let sel = DeviceBuffer::from_host(stream, &ids).expect("ids");
        let mut route = GemmRoute::new(stream, ids.len(), EXPERTS).expect("route");
        gemm.enqueue_route(stream, &sel, ids.len(), &mut route, gpu.unlabelled_sink())
            .expect("fill");
        let mut y = DeviceBuffer::zeroed(stream, ids.len() * ROWS).expect("y");
        g32.enqueue_gemm32(
            stream,
            Gemm32Args {
                w: Gemm32Weight::Mxfp8Planes(Mxfp8Stack { qs: &qs, e: &e }),
                rows_per_expert: ROWS,
                act: &act,
                route: &route,
                input: GemmInput::Shared { top_k: TOP_K },
                y: &mut y,
            },
        )
        .expect("gemm");
        assert_eq!(gpu.take_fault().expect("fault word"), None);
        let (y, q, d) = (
            y.to_host_vec(stream).expect("y"),
            act.q.to_host_vec(stream).expect("q"),
            act.d.to_host_vec(stream).expect("d"),
        );
        let steps = K.div_ceil(64);
        let mut failures = Vec::new();
        for (s, &id) in ids.iter().enumerate() {
            let c = s / TOP_K;
            let a: Vec<i8> = q[c * 16 * steps..][..K / 4]
                .iter()
                .flat_map(|w| w.to_le_bytes().map(|b| b as i8))
                .collect();
            let dc = &d[c * 2 * steps..][..K / 32];
            for r in 0..ROWS {
                let row = id as usize * ROWS + r;
                let (codes, scales) = row_of(&bytes, row, K);
                let want = rule_dot(&codes, &scales, &a, dc, 0);
                let got = y[s * ROWS + r];
                let (dot, band) = f64_band(&bytes, (row, K), &x[c * K..][..K], |i| dc[i / 32]);
                if got.to_bits() != want.to_bits() || (f64::from(want) - dot).abs() > band {
                    failures.push(format!(
                        "slot {s} row {r}: {got} against the rule's {want} and f64 {dot} ± {band}"
                    ));
                }
            }
        }
        println!(
            "mxfp8_gemm: {} failures; first: {:?}",
            failures.len(),
            failures.first()
        );
        assert!(failures.is_empty(), "\n{}", failures.join("\n"));
    }
}
