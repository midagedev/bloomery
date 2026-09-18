use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, kernel, launch_bounds, launch_contract, thread, warp,
};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// IEEE-754 half to float, integer-only so no `f16` feature gate is
    /// needed on either side of the unified compilation.
    fn half_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) as u32) << 31;
        let exp = ((bits >> 10) & 0x1f) as u32;
        let mant = (bits & 0x3ff) as u32;
        let mag = if exp == 0x1f {
            0x7f800000 | (mant << 13)
        } else if exp == 0 {
            if mant == 0 {
                0
            } else {
                let mut e = 127 - 14;
                let mut m = mant;
                while m & 0x400 == 0 {
                    m <<= 1;
                    e -= 1;
                }
                (e << 23) | ((m & 0x3ff) << 13)
            }
        } else {
            ((exp + 112) << 23) | (mant << 13)
        };
        f32::from_bits(sign | mag)
    }

    /// Q3_K (K=2048, 8 super-blocks per row) times f32 activations, M in {1,8}.
    ///
    /// One block of 1024 threads (32 warps) handles 32 output rows. The block
    /// first stages the shared K*M activation tile in dynamic shared memory
    /// — without this every row re-reads x from HBM because the 79 MB weight
    /// stream evicts x from L2; staging once per 32 rows (not 8) quarters
    /// that traffic to 23 MB on the stack shape. Each warp then owns one row:
    /// the 32 lanes split the row's 8 super-blocks in lockstep (one
    /// super-block at a time, lane -> eighth of it), dequantizing in
    /// registers — low 2 bits from `qs`, high bit from `hmask`, 6-bit scales
    /// from `scales[12]` — using exactly the decode of ggml's
    /// `dequantize_row_q3_K` (ggml/src/ggml-quants.c), loaded with u32 vector
    /// loads, and FMAing against all M columns, so the weight is read once
    /// per launch for all M. A warp-wide butterfly sum per column finishes
    /// the dot; lane 0 writes the M outputs.
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1, block = (1024, 1, 1), dynamic_shared_range = (8192, 65536), requires = (w.len() >= n_rows * 880, x.len() >= m_cols * 2048, y.len() >= n_rows * m_cols))]
    pub fn q3k_gemv(w: &[u8], x: &[f32], n_rows: u32, m_cols: u32, mut y: DisjointSlice<f32>) {
        const K: usize = 2048;
        const ROW_BYTES: usize = 880; // 8 super-blocks * 110 B
        const BLOCK: usize = 1024;
        const ROWS_PER_BLOCK: usize = 32;

        let tid = thread::index_1d().get();
        let t = tid % BLOCK;
        let m = m_cols as usize;
        let xk = K * m;

        // Stage x in dynamic shared memory: one copy per 32-row block.
        // Host sizes it to exactly K*m f32s.
        let sx: *mut f32 = DynamicSharedArray::<f32>::get();
        let mut i = t;
        while i < xk {
            // SAFETY: i < K*m by loop bound; host allocates K*m f32s of
            // dynamic shared memory; all 1024 threads write disjoint i.
            unsafe {
                *sx.add(i) = x[i];
            }
            i += BLOCK;
        }
        thread::sync_threads();

        let row = (tid / BLOCK) * ROWS_PER_BLOCK + (t / 32);
        if row >= n_rows as usize {
            return;
        }
        let lane = warp::lane_id() as usize;

        // u32 vector-load helper: 4 consecutive little-endian bytes.
        let w32 = |off: usize| {
            w[off] as u32
                | ((w[off + 1] as u32) << 8)
                | ((w[off + 2] as u32) << 16)
                | ((w[off + 3] as u32) << 24)
        };

        // Scalar accumulators: acc[c] with a runtime c would put the array
        // in local memory (global-backed). m is launch-uniform, so guarded
        // scalars stay in registers with no divergence.
        let mut a0 = 0.0f32;
        let mut a1 = 0.0f32;
        let mut a2 = 0.0f32;
        let mut a3 = 0.0f32;
        let mut a4 = 0.0f32;
        let mut a5 = 0.0f32;
        let mut a6 = 0.0f32;
        let mut a7 = 0.0f32;
        let mut sb = 0usize;
        while sb < 8 {
            let base = row * ROW_BYTES + sb * 110;

            // 12 scale bytes -> 16 int8 sub-block scales, verbatim port of
            // the aux[] shuffle in dequantize_row_q3_K (3 vector loads).
            // Only one aux word is needed per lane (word s>>2); the if-chain
            // keeps constant indices so no array lives in local memory.
            let a0w = w32(base + 96);
            let a1w = w32(base + 100);
            let a2w = w32(base + 104);
            let kmask1 = 0x03030303u32;
            let kmask2 = 0x0f0f0f0fu32;
            let t0 = ((a0w >> 4) & kmask2) | (((a2w >> 4) & kmask1) << 4);
            let t1 = ((a1w >> 4) & kmask2) | (((a2w >> 6) & kmask1) << 4);
            let t2 = (a0w & kmask2) | ((a2w & kmask1) << 4);
            let t3 = (a1w & kmask2) | (((a2w >> 2) & kmask1) << 4);

            let d_bits = w[base + 108] as u16 | ((w[base + 109] as u16) << 8);
            let d_all = half_to_f32(d_bits);

            // This super-block's 256 weights: lane -> (sub-block s, half `part`).
            // Each lane's 8 qs/hmask bytes are consecutive and 4B-aligned,
            // so two u32 loads cover each.
            let s = lane >> 1;
            let part = lane & 1;
            let chunk = s >> 3;
            let j = (s & 7) >> 1;
            let half = s & 1;
            let shift = (j * 2) as u32;
            let mb: u32 = 1 << ((chunk << 2) + j);
            let auxw = if s < 4 {
                t2
            } else if s < 8 {
                t3
            } else if s < 12 {
                t0
            } else {
                t1
            };
            let sc = ((auxw >> (8 * (s & 3))) & 0xff) as u8 as i8;
            let dl = d_all * (sc as f32 - 32.0);
            let qb0 = w32(base + 32 + (chunk << 5) + (half << 4) + (part << 3));
            let qb1 = w32(base + 32 + (chunk << 5) + (half << 4) + (part << 3) + 4);
            // NOTE: hmask is NOT advanced per chunk: the same 32 bytes serve
            // both 128-weight halves, bit (chunk*4+j) selects.
            let hb0 = w32(base + (half << 4) + (part << 3));
            let hb1 = w32(base + (half << 4) + (part << 3) + 4);
            let mut o = 0usize;
            while o < 8 {
                let word = if o < 4 { qb0 } else { qb1 };
                let qb = (word >> (8 * (o & 3))) & 0xff;
                let hword = if o < 4 { hb0 } else { hb1 };
                let hb = (hword >> (8 * (o & 3))) & 0xff;
                let low = ((qb >> shift) & 3) as i8;
                let qv = low - if (hb & mb) != 0 { 0 } else { 4 };
                let wv = dl * qv as f32;
                let k = (sb << 8) + (chunk << 7) + (j << 5) + (half << 4) + (part << 3) + o;
                // SAFETY: m <= 8 and k < 2048 by construction; smem holds
                // K*m f32s, fully populated before the barrier. The m
                // guards are launch-uniform (no divergence).
                let xv0 = unsafe { *sx.add(k) };
                a0 += wv * xv0;
                if m > 1 {
                    let xv1 = unsafe { *sx.add(K + k) };
                    let xv2 = unsafe { *sx.add(2 * K + k) };
                    let xv3 = unsafe { *sx.add(3 * K + k) };
                    let xv4 = unsafe { *sx.add(4 * K + k) };
                    let xv5 = unsafe { *sx.add(5 * K + k) };
                    let xv6 = unsafe { *sx.add(6 * K + k) };
                    let xv7 = unsafe { *sx.add(7 * K + k) };
                    a1 += wv * xv1;
                    a2 += wv * xv2;
                    a3 += wv * xv3;
                    a4 += wv * xv4;
                    a5 += wv * xv5;
                    a6 += wv * xv6;
                    a7 += wv * xv7;
                }
                o += 1;
            }
            sb += 1;
        }

        // Warp-uniform: m is a launch-wide constant, so every lane takes the
        // same path and the butterfly sums stay converged.
        let s0 = warp::reduce_sum_f32(a0);
        if m == 1 {
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                // row < n_rows checked above; y has n_rows*m elements.
                unsafe {
                    *y.get_unchecked_mut(row) = s0;
                }
            }
        } else {
            let s1 = warp::reduce_sum_f32(a1);
            let s2 = warp::reduce_sum_f32(a2);
            let s3 = warp::reduce_sum_f32(a3);
            let s4 = warp::reduce_sum_f32(a4);
            let s5 = warp::reduce_sum_f32(a5);
            let s6 = warp::reduce_sum_f32(a6);
            let s7 = warp::reduce_sum_f32(a7);
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes, and warp `row`
                // owns the disjoint segment y[row*m .. row*m+m]. This is the
                // warp-collective store pattern the DisjointSlice docs bless
                // for lane-0 writes, extended to M slots in one segment.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = s0;
                    *y.get_unchecked_mut(b + 1) = s1;
                    *y.get_unchecked_mut(b + 2) = s2;
                    *y.get_unchecked_mut(b + 3) = s3;
                    *y.get_unchecked_mut(b + 4) = s4;
                    *y.get_unchecked_mut(b + 5) = s5;
                    *y.get_unchecked_mut(b + 6) = s6;
                    *y.get_unchecked_mut(b + 7) = s7;
                }
            }
        }
    }
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert!(b.len() % 4 == 0, "odd size for {path}");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data = "/root/mulle-data-muse";
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let w_host = std::fs::read(format!("{data}/gate.q3k"))?;
    let x_m1 = read_f32(&format!("{data}/x_m1.f32"));
    let x_m8 = read_f32(&format!("{data}/x_m8.f32"));
    assert_eq!(x_m1.len(), 2048);
    assert_eq!(x_m8.len(), 2048 * 8);

    let w_dev = DeviceBuffer::from_host(&stream, &w_host)?;
    let x1_dev = DeviceBuffer::from_host(&stream, &x_m1)?;
    let x8_dev = DeviceBuffer::from_host(&stream, &x_m8)?;

    // SAFETY: this package owns the embedded device bundle produced for the
    // kernels module above.
    let module = unsafe { kernels::load(&ctx)? };

    struct Shape {
        name: &'static str,
        n: usize,
        m: usize,
        wbytes: usize,
    }
    let shapes = [
        Shape { name: "expert0_m1", n: 1408, m: 1, wbytes: 1239040 },
        Shape { name: "expert0_m8", n: 1408, m: 8, wbytes: 1239040 },
        Shape { name: "stack_m1", n: 90112, m: 1, wbytes: 79298560 },
        Shape { name: "stack_m8", n: 90112, m: 8, wbytes: 79298560 },
    ];

    let mut all_ok = true;
    for sh in &shapes {
        let x_dev = if sh.m == 1 { &x1_dev } else { &x8_dev };
        let mut y_dev = DeviceBuffer::<f32>::zeroed(&stream, sh.n * sh.m)?;
        let blocks = sh.n.div_ceil(32) as u32;
        let smem = (2048 * sh.m * 4) as u32;
        let prepared =
            module.prepare_q3k_gemv(LaunchConfig1D::new(blocks, 1024, smem))?;
        for _ in 0..20 {
            module.q3k_gemv(
                &stream,
                &prepared,
                &w_dev,
                x_dev,
                sh.n as u32,
                sh.m as u32,
                &mut y_dev,
            )?;
        }
        stream.synchronize()?;
        let t0 = std::time::Instant::now();
        for _ in 0..200 {
            module.q3k_gemv(
                &stream,
                &prepared,
                &w_dev,
                x_dev,
                sh.n as u32,
                sh.m as u32,
                &mut y_dev,
            )?;
        }
        stream.synchronize()?;
        let us = t0.elapsed().as_secs_f64() * 1e6 / 200.0;

        let y = y_dev.to_host_vec(&stream)?;
        let yref = read_f32(&format!("{data}/y_ref_{}.f32", sh.name));
        let denom = yref.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let maxerr = y
            .iter()
            .zip(yref.iter())
            .fold(0.0f32, |a, (&g, &r)| a.max((g - r).abs()));
        let rel = maxerr / denom;
        let gbs = sh.wbytes as f64 / (us * 1e-6) / 1e9;
        println!(
            "shape {:<10} N={:>6} M={} weight_bytes={:>9} us={:>9.2} GB/s={:>7.2} max_rel_err={:.3e}",
            sh.name, sh.n, sh.m, sh.wbytes, us, gbs, rel
        );
        if rel > 1e-4 {
            eprintln!("FAIL: {} rel err {rel:.3e} exceeds 1e-4", sh.name);
            all_ok = false;
        }
    }
    if !all_ok {
        std::process::exit(1);
    }
    println!("PASSED: all 4 shapes within 1e-4");
    Ok(())
}
