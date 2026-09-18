#![feature(f16)]

//! mulle stage 0 — Q3_K × f32 gemv/gemm on the RTX 3090, cuda-oxide.
//!
//! Kernel: one warp per output row. The row's 8 super-blocks (256 weights
//! each) split across lanes: lane owns the 4 consecutive 16-weight
//! sub-blocks starting at sub-block `lane*4` (i.e. super-block `lane/4`).
//! The decode is a port of ggml's `dequantize_row_q3_K` (ggml-quants.c in
//! ik_llama.cpp) — for weight w of the row:
//!
//!   q byte   = qs[32*(w>>7) + (w&31)]     low 2 bits  : (byte >> shift) & 3
//!   hmask    = hmask[w&31], bit (w>>5)&7  high bit    : set -> 0, clear -> -4
//!   value    = d * (scale6(w>>4) - 32) * ((int)(q&3) - (hm_bit ? 0 : 4))
//!
//! The 110-byte block stride makes field offsets odd-aligned, so every
//! access goes through `u32_at` (aligned word loads combined in registers).
//!
//! The decode loops are `#[unroll]`-annotated `while` loops with u32
//! counters — the one form the `#[kernel]` macro reads (attributes on `for`
//! loops are expression attributes and rejected; `#[unroll]` inside plain
//! helper functions never reaches the MIR pass). With constant trip counts
//! every index folds to a literal, so the window words and the eight
//! accumulators stay in registers. Before this the sm_86 PTX carried the
//! loops rolled with per-weight 64-bit index math (~1 weight/cycle/SM,
//! measured 2026-09-19: 62.7 GB/s on `stack` m=1).
//!
//! Host: loads /root/mulle-data written by tools/ref/q3k_ref.cpp, runs the
//! four stage-0 shapes (expert0 = [2048×1408], stack = [2048×90112],
//! M ∈ {1,8}) with 20 warm-up + 200 timed launches, and checks against the
//! CPU f32 reference files.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{DisjointSlice, kernel, launch_bounds, launch_contract, thread, warp};
use cuda_host::cuda_module;

const K: usize = 2048; // activation length, also the k dimension of the weight

#[cuda_module]
mod kernels {
    use super::*;

    // ---- Q3_K layout (ggml-common.h): hmask[32] qs[64] scales[12] d(f16) ----
    // byte offsets within a block: hmask 0, qs 32, scales 96, d 108; 110 B/block
    const BLOCK_B: usize = 110;
    const ROW_B: usize = 880; // 8 blocks of 256 weights per row

    /// Unaligned u32 at byte offset `off` (always even in this layout) from
    /// two aligned word loads. Callers guarantee off + 4 <= w.len() * 4.
    #[inline]
    fn u32_at(w: &[u32], off: usize) -> u32 {
        let i = off >> 2;
        if off & 2 == 0 {
            w[i]
        } else {
            (w[i] >> 16) | (w[i + 1] << 16)
        }
    }

    /// The f16 `d` field at byte offset `off` (off % 2 == 0) as f32, from the
    /// single aligned word containing it (never reads past the tensor).
    #[inline]
    fn f16_at(w: &[u32], off: usize) -> f32 {
        let word = w[off >> 2];
        let bits = if off & 2 == 0 {
            (word & 0xFFFF) as u16
        } else {
            (word >> 16) as u16
        };
        f16::from_bits(bits) as f32
    }

    /// 12 packed 6-bit scales at block_base+96 expanded to 16 signed bytes,
    /// the exact `aux[]` shuffle ggml's dequantize_row_q3_K performs.
    /// Returns the four words whose bytes are scale indices 0..3, 4..7, ...
    #[inline]
    fn expand_scales(w: &[u32], b: usize) -> (u32, u32, u32, u32) {
        const KMASK1: u32 = 0x0303_0303;
        const KMASK2: u32 = 0x0f0f_0f0f;
        let a0 = u32_at(w, b + 96);
        let a1 = u32_at(w, b + 100);
        let tmp = u32_at(w, b + 104);
        let n2 = ((a0 >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
        let n3 = ((a1 >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
        let n0 = (a0 & KMASK2) | ((tmp & KMASK1) << 4);
        let n1 = (a1 & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
        (n0, n1, n2, n3)
    }

    /// (scale[i] as int8 - 32) as f32 for expanded scale byte i.
    #[inline]
    fn scale_f32(n: (u32, u32, u32, u32), i: usize) -> f32 {
        let word = match i >> 2 {
            0 => n.0,
            1 => n.1,
            2 => n.2,
            _ => n.3,
        };
        let b = ((word >> (8 * (i & 3))) & 0xFF) as u8 as i8;
        (b as i32 - 32) as f32
    }

    /// Per-lane state for one output row: lane owns sub-blocks 4*lane..4*lane+4
    /// of super-block `lane/4`. Mirrors the closed form of the ggml decode.
    #[inline]
    fn lane_params(lane: usize) -> (usize, usize) {
        // (super-block index, sub-block group q within the block)
        (lane >> 2, lane & 3)
    }

    /// Select among 8 register-held window words. A `match` over scalars
    /// lowers to register selects (and folds away entirely once loops
    /// unroll), where indexing a `[u32; 8]` with a runtime value forces the
    /// array into local memory (measured: 32 st.local in the sm_86 PTX).
    #[inline]
    fn sel8(
        i: usize, v0: u32, v1: u32, v2: u32, v3: u32, v4: u32, v5: u32, v6: u32, v7: u32,
    ) -> u32 {
        match i {
            0 => v0,
            1 => v1,
            2 => v2,
            3 => v3,
            4 => v4,
            5 => v5,
            6 => v6,
            _ => v7,
        }
    }

    /// The lane's qs and hmask windows as unaligned u32s, loaded once per
    /// block. Elements are only read back with literal indices at the call
    /// sites, so the arrays never materialize in memory.
    #[inline]
    fn load_windows(w: &[u32], b: usize, q: usize) -> ([u32; 8], [u32; 8]) {
        let qs_off = b + 32 + 32 * (q >> 1);
        let mut qw = [0u32; 8];
        let mut hw = [0u32; 8];
        for i in 0..8 {
            qw[i] = u32_at(w, qs_off + 4 * i);
            hw[i] = u32_at(w, b + 4 * i);
        }
        (qw, hw)
    }

    /// One warp per output row, 8 warps (256 threads) per block.
    ///
    /// sub-block r in 0..4: qs/hm word half h4 = 4*(r&1),
    /// shift s = 2*(r>>1) + 4*(q&1), hmask bit = 2*q + (r>>1),
    /// scale index = 4*q + r.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn q3k_gemv_m1(
        w: &[u32],
        x: &[f32],
        mut y: DisjointSlice<f32>,
        n_rows: u32,
    ) {
        let gid = thread::index_1d();
        let row = (gid.get() as usize) >> 5;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let (blk, q) = lane_params(lane);
        let b = row * ROW_B + blk * BLOCK_B;
        let d_all = f16_at(w, b + 108);
        let scales = expand_scales(w, b);
        let (qw, hw) = load_windows(w, b, q);

        let mut acc = 0.0f32;
        let mut r: u32 = 0;
        #[unroll]
        while r < 4 {
            let ru = r as usize;
            let h4 = 4 * (ru & 1);
            let s = 2 * (ru >> 1) + 4 * (q & 1);
            let mbit = 1u32 << (2 * q + (ru >> 1));
            let dl = d_all * scale_f32(scales, 4 * q + ru);
            let k0 = 256 * blk + 16 * (4 * q + ru);
            let mut l: u32 = 0;
            #[unroll]
            while l < 16 {
                let lu = l as usize;
                let i = h4 + (lu >> 2);
                let sh = 8 * (lu & 3);
                let qword = sel8(i, qw[0], qw[1], qw[2], qw[3], qw[4], qw[5], qw[6], qw[7]);
                let hword = sel8(i, hw[0], hw[1], hw[2], hw[3], hw[4], hw[5], hw[6], hw[7]);
                let low2 = ((qword >> sh >> s) & 3) as i32;
                // hm bit set -> -0, clear -> -4, branchless: the per-weight
                // `if` lowered to ISETP+BRA+BPT per weight in the sm_86 SASS
                // (measured 2026-09-19: 171 BRA / 80 BPT in the m1 kernel)
                let hb = ((hword >> sh) & mbit != 0) as i32;
                let qv = low2 - 4 + 4 * hb;
                acc += (dl * qv as f32) * x[k0 + lu];
                l += 1;
            }
            r += 1;
        }
        let total = warp::reduce_sum_f32(acc);
        if lane == 0 {
            // SAFETY: row < n_rows == y.len(), single writer per row.
            unsafe {
                *y.get_unchecked_mut(row) = total;
            }
        }
    }

    /// Same decode for M = 8: one weight decode feeds eight accumulators.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1))]
    pub fn q3k_gemm_m8(
        w: &[u32],
        x: &[f32],
        mut y: DisjointSlice<f32>,
        n_rows: u32,
    ) {
        let gid = thread::index_1d();
        let row = (gid.get() as usize) >> 5;
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;
        let (blk, q) = lane_params(lane);
        let b = row * ROW_B + blk * BLOCK_B;
        let d_all = f16_at(w, b + 108);
        let scales = expand_scales(w, b);
        let (qw, hw) = load_windows(w, b, q);

        let mut acc = [0.0f32; 8];
        let mut r: u32 = 0;
        #[unroll]
        while r < 4 {
            let ru = r as usize;
            let h4 = 4 * (ru & 1);
            let s = 2 * (ru >> 1) + 4 * (q & 1);
            let mbit = 1u32 << (2 * q + (ru >> 1));
            let dl = d_all * scale_f32(scales, 4 * q + ru);
            let k0 = 256 * blk + 16 * (4 * q + ru);
            let mut l: u32 = 0;
            #[unroll]
            while l < 16 {
                let lu = l as usize;
                let i = h4 + (lu >> 2);
                let sh = 8 * (lu & 3);
                let qword = sel8(i, qw[0], qw[1], qw[2], qw[3], qw[4], qw[5], qw[6], qw[7]);
                let hword = sel8(i, hw[0], hw[1], hw[2], hw[3], hw[4], hw[5], hw[6], hw[7]);
                let low2 = ((qword >> sh >> s) & 3) as i32;
                // hm bit set -> -0, clear -> -4, branchless (see q3k_gemv_m1)
                let hb = ((hword >> sh) & mbit != 0) as i32;
                let qv = low2 - 4 + 4 * hb;
                let wv = dl * qv as f32;
                let k = k0 + lu;
                let mut m: u32 = 0;
                #[unroll]
                while m < 8 {
                    acc[m as usize] += wv * x[m as usize * K + k];
                    m += 1;
                }
                l += 1;
            }
            r += 1;
        }
        let mut m: u32 = 0;
        #[unroll]
        while m < 8 {
            let total = warp::reduce_sum_f32(acc[m as usize]);
            if lane == 0 {
                // SAFETY: row < n_rows, y.len() == n_rows * 8, one writer.
                unsafe {
                    *y.get_unchecked_mut(row * 8 + m as usize) = total;
                }
            }
            m += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// host

fn read_f32(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{path}: length {} not a multiple of 4", bytes.len()).into());
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn run_cmd(cmd: &str) {
    match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
        Ok(out) => {
            print!("{}", String::from_utf8_lossy(&out.stdout));
            let err = String::from_utf8_lossy(&out.stderr);
            if !err.is_empty() {
                eprint!("[stderr] {err}");
            }
        }
        Err(e) => println!("[witness cmd failed: {cmd}: {e}]"),
    }
}

/// quiet-machine witness: both cards, loadavg, io pressure (avg10)
fn witnesses(when: &str) {
    println!("[witness {when}]");
    run_cmd("nvidia-smi --query-gpu=name,memory.used,utilization.gpu,power.draw --format=csv");
    run_cmd("cat /proc/loadavg");
    run_cmd("grep -E '^(some|full)' /proc/pressure/io");
}

struct Shape {
    name: &'static str,
    rows: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = "/root/mulle-data";
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // raw q3_K bytes viewed as little-endian u32 words for the kernel
    let w_bytes = std::fs::read(format!("{data}/gate.q3k"))?;
    assert_eq!(w_bytes.len(), 90112 * 880, "unexpected gate.q3k size");
    let w_words: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let w_dev = DeviceBuffer::from_host(&stream, &w_words)?;

    let x1 = read_f32(&format!("{data}/x_m1.f32"))?;
    let x8 = read_f32(&format!("{data}/x_m8.f32"))?;
    assert_eq!(x1.len(), K);
    assert_eq!(x8.len(), 8 * K);

    let shapes = [
        Shape { name: "expert0", rows: 1408 },
        Shape { name: "stack", rows: 90112 },
    ];
    let y_ref: Vec<Vec<f32>> = shapes
        .iter()
        .map(|s| read_f32(&format!("{data}/y_ref_{}_m1.f32", s.name)))
        .collect::<Result<_, _>>()?;
    let y_ref8: Vec<Vec<f32>> = shapes
        .iter()
        .map(|s| read_f32(&format!("{data}/y_ref_{}_m8.f32", s.name)))
        .collect::<Result<_, _>>()?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };

    let x1_dev = DeviceBuffer::from_host(&stream, &x1)?;
    let x8_dev = DeviceBuffer::from_host(&stream, &x8)?;

    witnesses("before mulle timing");
    for (si, shape) in shapes.iter().enumerate() {
        // M = 1
        {
            let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, shape.rows)?;
            let grid = (shape.rows as u32).div_ceil(8); // 8 warps per block
            let prepared = module
                .prepare_q3k_gemv_m1(LaunchConfig1D::new(grid, 256, 0))?;
            for _ in 0..20 {
                module.q3k_gemv_m1(&stream, &prepared, &w_dev, &x1_dev, &mut y_dev, shape.rows as u32)?;
            }
            ctx.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                module.q3k_gemv_m1(&stream, &prepared, &w_dev, &x1_dev, &mut y_dev, shape.rows as u32)?;
            }
            ctx.synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
            let y = y_dev.to_host_vec(&stream)?;
            let err = max_rel_err(&y, &y_ref[si]);
            let bytes = (shape.rows * 880) as f64;
            println!(
                "RESULT engine=mulle shape={} m=1 us_per_launch={:.3} gbps={:.1} max_rel_err={:.3e}",
                shape.name,
                us,
                bytes / (us * 1e-6) / 1e9,
                err
            );
        }
        // M = 8
        {
            let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, shape.rows * 8)?;
            let grid = (shape.rows as u32).div_ceil(8);
            let prepared = module
                .prepare_q3k_gemm_m8(LaunchConfig1D::new(grid, 256, 0))?;
            for _ in 0..20 {
                module.q3k_gemm_m8(&stream, &prepared, &w_dev, &x8_dev, &mut y_dev, shape.rows as u32)?;
            }
            ctx.synchronize()?;
            let t0 = std::time::Instant::now();
            for _ in 0..200 {
                module.q3k_gemm_m8(&stream, &prepared, &w_dev, &x8_dev, &mut y_dev, shape.rows as u32)?;
            }
            ctx.synchronize()?;
            let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;
            let y = y_dev.to_host_vec(&stream)?;
            let err = max_rel_err(&y, &y_ref8[si]);
            let bytes = (shape.rows * 880) as f64;
            println!(
                "RESULT engine=mulle shape={} m=8 us_per_launch={:.3} gbps={:.1} max_rel_err={:.3e}",
                shape.name,
                us,
                bytes / (us * 1e-6) / 1e9,
                err
            );
        }
    }
    witnesses("after mulle timing");

    Ok(())
}

fn max_rel_err(y: &[f32], y_ref: &[f32]) -> f64 {
    let mut max_ref = 0.0f64;
    let mut max_err = 0.0f64;
    for (got, ref_) in y.iter().zip(y_ref.iter()) {
        let r = *ref_ as f64;
        let g = *got as f64;
        max_ref = max_ref.max(r.abs());
        max_err = max_err.max((g - r).abs());
    }
    max_err / max_ref
}
