use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, dotprod, kernel, launch_bounds, launch_contract, thread,
    warp,
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

    /// Q3_K (K=2048, 8 super-blocks per row) times q8_1 activations, M in {1,8}.
    ///
    /// One block of 1024 threads (32 warps) handles 32 output rows. The block
    /// first quantizes the shared K*M f32 activation tile to q8_1 in dynamic
    /// shared memory (64*M blocks of 32, absmax scale each) — one copy per
    /// 32-row block, so the 79 MB weight stream cannot evict x from L2. Each
    /// warp then owns one row and walks its 8 super-blocks in lockstep.
    ///
    /// Weight-load design (the mmvq lesson): a Q3_K super-block is 110 bytes
    /// whose per-lane bytes scatter, so per-lane byte loads turn into ~960
    /// scalar, mutually redundant global loads per warp per super-block.
    /// Instead the warp cooperatively loads the whole super-block once: 28
    /// aligned u32 words covering bytes `[B-(B&3), B+110)` where `B = sb*110`
    /// (`B&3` is 0 for even `sb`, 2 for odd `sb`, so 28 words always suffice
    /// and never read past the row end), one coalesced 4-byte `LDG` per lane,
    /// into the warp's private 112-byte slot in dynamic shared memory.
    ///
    /// Dot-product design: 4 consecutive weights always occupy 4 consecutive
    /// `qs` bytes (same bit-pair `j = (k>>5)&3`, divisible-by-4 base, never
    /// crossing a 32- or 128-boundary) plus 4 consecutive `hmask` bytes (same
    /// bit `j`), matching 4 consecutive q8 activation bytes. Each lane owns 2
    /// such groups (weights `8*lane .. 8*lane+8`, sub-block `lane>>1`) and
    /// evaluates them with two `dp4a`s against one packed q8 word each. The
    /// signed 3-bit values come from a branchless per-byte combine
    /// (`vil | (mask * 0xFC)`); per-16 scales use the exact aux shuffle of
    /// ggml's `dequantize_row_q3_K`. The weight is read once per launch for
    /// all M (the packed `vi` words are reused across columns). A warp-wide
    /// butterfly sum per column finishes the dot; lane 0 writes the M outputs.
    ///
    /// Accuracy: q8_1 activations quantize x, so the gate for this design is
    /// 1e-2 max rel err (ggml's own mmvq error is ~4e-3 for the same reason).
    #[kernel]
    #[launch_bounds(1024)]
    #[launch_contract(domain = 1, block = (1024, 1, 1), dynamic_shared_range = (5888, 22016), requires = (w.len() >= n_rows * 880, x.len() >= m_cols * 2048, y.len() >= n_rows * m_cols))]
    pub fn q3k_gemv(w: &[u8], x: &[f32], n_rows: u32, m_cols: u32, mut y: DisjointSlice<f32>) {
        const K: usize = 2048;
        const ROW_BYTES: usize = 880; // 8 super-blocks * 110 B
        const BLOCK: usize = 1024;
        const ROWS_PER_BLOCK: usize = 32;
        const SLOT_WORDS: usize = 28; // 28 u32 cover any 110 B super-block
        const SLOT_BYTES: usize = 112; // 28 * 4
        const QBLOCKS: usize = 64; // K / 32 q8 blocks per column

        let tid = thread::index_1d().get();
        let t = tid % BLOCK;
        let m = m_cols as usize;
        let warp_in_block = t / 32;
        let lane = warp::lane_id() as usize;

        // Shared layout: qx int8 tile [K*m], q8 scales [64*m] f32,
        // then 32 warp slots of 112 B. K*m is a multiple of 128, 64*m*4 a
        // multiple of 4, so the f32 and u32 views stay aligned.
        // SAFETY: host allocates K*m + 256*m + 32*112 dynamic-shared bytes.
        let sq: *mut u8 = DynamicSharedArray::<u8>::get();
        let ssc: *mut f32 = unsafe { sq.add(K * m) as *mut f32 };
        let slot: *mut u8 =
            unsafe { sq.add(K * m + QBLOCKS * m * 4 + warp_in_block * SLOT_BYTES) };
        let slotw: *mut u32 = slot as *mut u32;

        // Fused x quantize: 64*m blocks of 32; warp `b % 32` owns block b.
        // Block b covers column b/64, elements (b%64)*32 + lane.
        // SAFETY: all x indices below are < K*m by construction; x holds
        // K*m f32s per the launch contract.
        let xp = x.as_ptr();
        let mut b = warp_in_block;
        while b < QBLOCKS * m {
            let col = b / QBLOCKS;
            let seg = b % QBLOCKS;
            let v = unsafe { *xp.add(col * K + seg * 32 + lane) };
            let mut mx = if v >= 0.0 { v } else { -v };
            let o16 = warp::shuffle_xor_f32(mx, 16);
            let a16 = if o16 >= 0.0 { o16 } else { -o16 };
            if a16 > mx {
                mx = a16;
            }
            let o8 = warp::shuffle_xor_f32(mx, 8);
            let a8 = if o8 >= 0.0 { o8 } else { -o8 };
            if a8 > mx {
                mx = a8;
            }
            let o4 = warp::shuffle_xor_f32(mx, 4);
            let a4 = if o4 >= 0.0 { o4 } else { -o4 };
            if a4 > mx {
                mx = a4;
            }
            let o2 = warp::shuffle_xor_f32(mx, 2);
            let a2 = if o2 >= 0.0 { o2 } else { -o2 };
            if a2 > mx {
                mx = a2;
            }
            let o1 = warp::shuffle_xor_f32(mx, 1);
            let a1 = if o1 >= 0.0 { o1 } else { -o1 };
            if a1 > mx {
                mx = a1;
            }
            // Every lane holds the block absmax now.
            let d = if mx == 0.0 { 0.0 } else { mx / 127.0 };
            let qf = if d == 0.0 { 0.0 } else { v / d };
            let mut q = (qf + if qf >= 0.0 { 0.5 } else { -0.5 }) as i32;
            if q > 127 {
                q = 127;
            }
            if q < -127 {
                q = -127;
            }
            // SAFETY: disjoint bytes per (col, seg, lane); qx tile holds K*m.
            unsafe {
                *sq.add(col * K + seg * 32 + lane) = q as u8;
                if lane == 0 {
                    // SAFETY: only lane 0 writes; one scale per block.
                    *ssc.add(col * QBLOCKS + seg) = d;
                }
            }
            b += BLOCK / 32;
        }
        thread::sync_threads();

        let row = (tid / BLOCK) * ROWS_PER_BLOCK + (t / 32);
        if row >= n_rows as usize {
            return;
        }

        // Raw word view of the global weight bytes. Row starts are multiples
        // of 880 bytes, hence 4-byte aligned; the per-super-block word base
        // below is floored to 4-byte alignment, so every word load is
        // aligned. Raw loads also skip slice bounds checks (host guarantees
        // w.len() >= n_rows*880 via the launch contract).
        // SAFETY: all indices are < n_rows*220 by the row/sb/lane bounds.
        let wwords = w.as_ptr() as *const u32;
        let row_base_w = row * ROW_BYTES / 4;

        // Integer accumulators per column (dp4a) plus float sums.
        // m is launch-uniform, so guarded scalars stay in registers.
        let mut i0 = 0i32;
        let mut i1 = 0i32;
        let mut i2 = 0i32;
        let mut i3 = 0i32;
        let mut i4 = 0i32;
        let mut i5 = 0i32;
        let mut i6 = 0i32;
        let mut i7 = 0i32;
        let mut s0 = 0.0f32;
        let mut s1 = 0.0f32;
        let mut s2 = 0.0f32;
        let mut s3 = 0.0f32;
        let mut s4 = 0.0f32;
        let mut s5 = 0.0f32;
        let mut s6 = 0.0f32;
        let mut s7 = 0.0f32;

        let mut sb: u32 = 0;
        while sb < 8 {
            let sbs = sb as usize;
            // floor(sb*110/4): even sb -> exact, odd sb -> (sb*110-2)/4.
            let sb_base_w = row_base_w + sbs * 27 + sbs / 2;
            let odd = (sbs & 1) != 0;
            if lane < SLOT_WORDS {
                // SAFETY: sb_base_w + lane < row_base_w + 220 by the sb/lane
                // bounds; 4-byte aligned by construction; this warp's 28
                // lanes write disjoint slot words. Lanes 28..31 skip the
                // store but still reach the sync below.
                unsafe {
                    *slotw.add(lane) = *wwords.add(sb_base_w + lane);
                }
            }
            warp::sync_mask(0xffffffff);

            // Word at super-block byte offset `off` (off % 4 == 0): direct
            // for even sb; composed from two slot words for odd sb (slot
            // covers [B-2, B+110), so sb bytes off..off+3 are slot bytes
            // off+2..off+5).
            let qw = |off: usize| {
                // SAFETY: slot reads within this warp's 112-byte slot,
                // populated before the sync above. Even: off/4+1 <= 27 by
                // the field offsets (max 108). Odd: same bound shifted.
                unsafe {
                    if !odd {
                        *slotw.add(off / 4)
                    } else {
                        (*slotw.add(off / 4) >> 16) | (*slotw.add(off / 4 + 1) << 16)
                    }
                }
            };

            // This super-block's d (shared by all lanes) and the 12 scale
            // bytes -> 16 int8 sub-block scales, verbatim port of the aux[]
            // shuffle in dequantize_row_q3_K. Only one aux word is needed per
            // lane (word s>>2); the if-chain keeps constant indices so no
            // array lives in local memory.
            // d is read as bytes, not via qw(108): for odd sb that word
            // would compose slot words 27 and 28, and 28 is past the slot.
            let sh = (sbs & 1) * 2;
            // SAFETY: 109+sh <= 111 < 112; slot populated before the sync.
            let d_bits =
                unsafe { *slot.add(108 + sh) as u16 | ((*slot.add(109 + sh) as u16) << 8) };
            let d_all = half_to_f32(d_bits);
            let s = lane >> 1;
            let a0w = qw(96);
            let a1w = qw(100);
            let a2w = qw(104);
            let kmask1 = 0x03030303u32;
            let kmask2 = 0x0f0f0f0fu32;
            let t0 = ((a0w >> 4) & kmask2) | (((a2w >> 4) & kmask1) << 4);
            let t1 = ((a1w >> 4) & kmask2) | (((a2w >> 6) & kmask1) << 4);
            let t2 = (a0w & kmask2) | ((a2w & kmask1) << 4);
            let t3 = (a1w & kmask2) | (((a2w >> 2) & kmask1) << 4);
            let auxw = if s < 4 {
                t2
            } else if s < 8 {
                t3
            } else if s < 12 {
                t0
            } else {
                t1
            };
            let sc = ((auxw >> (8 * (s & 3))) & 0xff) as u8 as i8 as f32 - 32.0;

            // Lane's 8 weights: 8*lane .. 8*lane+8 of the super-block, two
            // groups of 4 consecutive weights. 4 consecutive weights =
            // 4 consecutive qs bytes (pair j) + 4 consecutive hmask bytes
            // (bit j), matching 4 consecutive q8 bytes in one word.
            let gbase = 8 * lane;
            let half_sb = (gbase >> 7) * 32; // 0 or 32 (128-weight half)
            let bl = gbase & 31; // byte base within the half, %4 == 0
            let j = (gbase >> 5) & 3; // shared by both groups
            // hmask bit selects (chunk*4+j): the mask bit keeps shifting
            // across the two 128-weight halves (as in dequantize_row_q3_K,
            // where only `shift` resets per half, not `m`).
            let jj = ((gbase >> 7) << 2) + j;
            let sh2 = (j * 2) as u32;
            // qs bytes at 32+half_sb+bl .. +bl+7 (64 bytes per super-block).
            // hmask is only 32 bytes (256 bits) shared by both halves: byte
            // bl .. bl+7 with bit (chunk*4+j); only qs advances per half.
            let q0 = qw(32 + half_sb + bl);
            let h0 = qw(bl);
            let q1 = qw(32 + half_sb + bl + 4);
            let h1 = qw(bl + 4);
            // Signed 3-bit values, 4 per word, branchless: low pairs minus
            // (bit clear ? 4 : 0); byte b with clear bit = vil_b | 0xFC.
            let vil0 = (q0 >> sh2) & 0x03030303;
            let msk0 = ((!h0 >> jj) & 0x01010101).wrapping_mul(0xFC);
            let vi0 = vil0 | msk0;
            let vil1 = (q1 >> sh2) & 0x03030303;
            let msk1 = ((!h1 >> jj) & 0x01010101).wrapping_mul(0xFC);
            let vi1 = vil1 | msk1;

            // q8 words for these 8 weights, per column. Weights are
            // consecutive, so each group is one word.
            // SAFETY: sbk+8 < K and col < m by construction; qx
            // tile holds K*m bytes, fully populated before the barrier.
            let sbk = sbs * 256 + gbase;
            let qx0 = unsafe { *(sq.add(sbk) as *const u32) };
            let qx1 = unsafe { *((sq as *const u32).add(sbk / 4 + 1)) };
            i0 = dotprod::dp4a_s32(vi0, qx0, dotprod::dp4a_s32(vi1, qx1, i0));
            if m > 1 {
                let mut c: u32 = 1;
                while c < 8 {
                    let cu = unsafe {
                        *((sq as *const u32).add(c as usize * 512 + sbk / 4))
                    };
                    let cu1 = unsafe {
                        *((sq as *const u32).add(c as usize * 512 + sbk / 4 + 1))
                    };
                    let dp = dotprod::dp4a_s32(vi0, cu, dotprod::dp4a_s32(vi1, cu1, 0));
                    if c == 1 {
                        i1 = dp;
                    } else if c == 2 {
                        i2 = dp;
                    } else if c == 3 {
                        i3 = dp;
                    } else if c == 4 {
                        i4 = dp;
                    } else if c == 5 {
                        i5 = dp;
                    } else if c == 6 {
                        i6 = dp;
                    } else {
                        i7 = dp;
                    }
                    c += 1;
                }
            }

            // Sub-block scale apply: acc(int) * sc * d3 * d8, d8 = this
            // lane's 8 weights share one q8 block (sbk>>5).
            // SAFETY: qb < 64*m; scales populated before the barrier.
            let d8 = unsafe { *ssc.add(sbk / 32) };
            let d38 = d_all * sc * d8;
            // i0 holds only this super-block's dp4a sum (zeroed below after
            // each consumption), so fold it into the float sum and reset.
            s0 += i0 as f32 * d38;
            i0 = 0;
            if m > 1 {
                // SAFETY: same bounds with the column stride.
                let d81 = unsafe { *ssc.add(64 + sbk / 32) };
                let d82 = unsafe { *ssc.add(128 + sbk / 32) };
                let d83 = unsafe { *ssc.add(192 + sbk / 32) };
                let d84 = unsafe { *ssc.add(256 + sbk / 32) };
                let d85 = unsafe { *ssc.add(320 + sbk / 32) };
                let d86 = unsafe { *ssc.add(384 + sbk / 32) };
                let d87 = unsafe { *ssc.add(448 + sbk / 32) };
                s1 += i1 as f32 * (d_all * sc * d81);
                s2 += i2 as f32 * (d_all * sc * d82);
                s3 += i3 as f32 * (d_all * sc * d83);
                s4 += i4 as f32 * (d_all * sc * d84);
                s5 += i5 as f32 * (d_all * sc * d85);
                s6 += i6 as f32 * (d_all * sc * d86);
                s7 += i7 as f32 * (d_all * sc * d87);
                i1 = 0;
                i2 = 0;
                i3 = 0;
                i4 = 0;
                i5 = 0;
                i6 = 0;
                i7 = 0;
            }
            sb += 1;
        }

        // Warp-uniform: m is a launch-wide constant, so every lane takes the
        // same path and the butterfly sums stay converged.
        let r0 = warp::reduce_sum_f32(s0);
        if m == 1 {
            if lane == 0 {
                // SAFETY: only lane 0 writes; warp `row` owns y[row].
                // row < n_rows checked above; y has n_rows*m elements.
                unsafe {
                    *y.get_unchecked_mut(row) = r0;
                }
            }
        } else {
            let r1 = warp::reduce_sum_f32(s1);
            let r2 = warp::reduce_sum_f32(s2);
            let r3 = warp::reduce_sum_f32(s3);
            let r4 = warp::reduce_sum_f32(s4);
            let r5 = warp::reduce_sum_f32(s5);
            let r6 = warp::reduce_sum_f32(s6);
            let r7 = warp::reduce_sum_f32(s7);
            if lane == 0 {
                // SAFETY: only lane 0 of each warp writes, and warp `row`
                // owns the disjoint segment y[row*m .. row*m+m]. This is the
                // warp-collective store pattern the DisjointSlice docs bless
                // for lane-0 writes, extended to M slots in one segment.
                unsafe {
                    let b = row * m;
                    *y.get_unchecked_mut(b) = r0;
                    *y.get_unchecked_mut(b + 1) = r1;
                    *y.get_unchecked_mut(b + 2) = r2;
                    *y.get_unchecked_mut(b + 3) = r3;
                    *y.get_unchecked_mut(b + 4) = r4;
                    *y.get_unchecked_mut(b + 5) = r5;
                    *y.get_unchecked_mut(b + 6) = r6;
                    *y.get_unchecked_mut(b + 7) = r7;
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
    let data = "/root/mulle-data-s0r-muse";
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
        // q8_1 design: K*m int8 + 64*m f32 scales + 32 warp slots of 112 B.
        // Accuracy gate for this design: 1e-2 (x is quantized, as in mmvq).
        let smem = (2048 * sh.m + 256 * sh.m + 32 * 112) as u32;
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
        if rel > 1e-2 {
            eprintln!("FAIL: {} rel err {rel:.3e} exceeds 1e-2", sh.name);
            all_ok = false;
        }
    }
    if !all_ok {
        std::process::exit(1);
    }
    println!("PASSED: all 4 shapes within 1e-2");
    Ok(())
}
