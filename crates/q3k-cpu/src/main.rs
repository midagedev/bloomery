// q3k-cpu — mulle stage 2 pre-study (MUL-3): a Q3_K × Q8_K gemv on the host
// CPU (Zen 3, AVX2+FMA3, 32 cores / 4 CCDs) against ik_llama.cpp's CPU
// backend (timed by tools/ref/q3k_cpu_ref.cpp).
//
// Ports, no invented decode:
//   * the activation quantizer is ggml's quantize_row_q8_K_ref
//     (ggml-quants.c:3974) into the 296-byte block_q8_K layout of the
//     vendored ik fork — d f32 @0, sum f32 @4 (ik-only field, the ggml ref
//     leaves it unwritten; we zero it), s8 qs[256] @8, s16 bsums[16] @264;
//   * the row dot is the __AVX2__ branch of ggml_vec_dot_q3_K_q8_K
//     (ggml-quants.c:6482), extended from one activation column to M <= 8:
//     the per-super-block weight decode (hmask shift, q3l/q3h fields, scale
//     shuffle) is computed once and the maddubs chain runs per column, so
//     decode cost amortizes over M exactly like the round-3 GPU kernel.
//     Block geometry comments verified against dequantize_row_q3_K at 1e-7
//     in stage 0: super-block = hmask[32] @+0, qs[64] @+32, scales[12] @+96,
//     f16 d @+108; weight k = 128c+32f+l reads qs byte 32c+l field f, its
//     high bit is hmask byte l bit 4c+f, its sub-block scale is 8c+2f+l/16.
//
// Threading: fixed CCD partition. /sys topology groups physical cores by the
// L3 they share (4 CCDs × 8 cores here); threads are spread over the CCDs
// first (T/4 per CCD, physical cores before their SMT siblings), pinned with
// sched_setaffinity, and each CCD's threads own one contiguous quarter of
// the rows so a CCD streams a contiguous weight range.
//
// Levers (RESULTS.md has the before/after for each):
const WEIGHTS_HUGEPAGE: bool = false; // 2 MiB-aligned mmap + MADV_HUGEPAGE for the weight buffer
const PREFETCH_ROWS: usize = 0; // software-prefetch this many rows ahead (0 = off)

use std::arch::x86_64::*;
use std::io::Read as _;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

const K: usize = 2048;
const NB: usize = K / 256; // Q3_K super-blocks per row = 8
const ROW_BYTES: usize = 110 * NB; // 880
const Q8K_STRIDE: usize = 296; // sizeof(block_q8_K) in the vendored ik fork
const WARMUP: usize = 5;
const TIMED: usize = 50;
const THREADS: [usize; 4] = [8, 16, 32, 64];

// --------------------------------------------------------------------- fp16

/// IEEE-754 half to float, integer-only (same routine the stage-0 GPU kernel
/// verified against ggml's GGML_FP16_TO_FP32 at 1e-7).
#[inline(always)]
fn half_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let mag = if exp == 0x1f {
        0x7f80_0000 | (mant << 13)
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

// ------------------------------------------------------------- q8_K encode

/// ggml's nearest_int (ggml-quants.c:1726): round-to-nearest via the
/// 2^23 + 2^22 f32 mantissa trick.
#[inline(always)]
fn nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    let i = f32::to_bits(val);
    ((i & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Port of quantize_row_q8_K_ref (ggml-quants.c:3974) over one 256-value
/// block. `out` is Q8K_STRIDE bytes: d @0, sum @4 (zeroed; the C ref never
/// writes it and nothing in the q3_K dot reads it), qs @8, bsums @264.
///
/// # Safety
/// `out` must point at Q8K_STRIDE writable bytes; `x` at 256 readable f32.
unsafe fn quantize_q8k_block(x: &[f32], out: *mut u8) {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    // SAFETY: plain stores inside the Q8K_STRIDE-byte block.
    unsafe {
        *out.add(4) = 0;
        if amax == 0.0 {
            std::ptr::write_bytes(out, 0, Q8K_STRIDE);
            return;
        }
        let iscale = -127.0f32 / max;
        let qs = out.add(8) as *mut i8;
        for j in 0..256 {
            let v = nearest_int(iscale * x[j]);
            *qs.add(j) = v.min(127) as i8;
        }
        let bsums = out.add(264) as *mut i16;
        for j in 0..16 {
            let mut sum = 0i32;
            for ii in 0..16 {
                sum += *qs.add(16 * j + ii) as i32;
            }
            *bsums.add(j) = sum as i16;
        }
        *(out as *mut f32) = 1.0 / iscale;
    }
}

// --------------------------------------------------------------- q3_K dot

/// get_scale_shuffle_q3k's table (ggml-quants.c:4041), verbatim. Vector f
/// broadcasts scale pair (2f, 2f+1) in the low 16 bytes and (2f+2, 2f+3) in
/// the high 16: the two 16-weight sub-blocks of a 32-value field.
static K_SHUFFLE_Q3K: [u8; 128] = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, //
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, //
    4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, //
    6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, //
    8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, //
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, //
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, //
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, //
];

/// Horizontal i32 sum of an __m256i — the integer counterpart of ggml's
/// hsum_float_8, applied before the i32→f32 conversion so the reduction
/// stays exact (|sumi| ≤ ~4.2e6 per super-block, exact in f32 either way).
#[inline(always)]
unsafe fn hsum_i32(v: __m256i) -> i32 {
    unsafe {
        // fold the two 128-bit halves, then lane-pair twice:
        // 0x4E = perm [2,3,0,1], 0x8D = perm [1,0,3,2]
        let lo = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x4E));
        let lo = _mm_add_epi32(lo, _mm_shuffle_epi32(lo, 0x8D));
        _mm_cvtsi128_si32(lo)
    }
}

/// One field of one 128-value half: decode the shared weight bytes and dot
/// them against every column's matching Q8_K span. SHIFT = 2f (q3bits shift),
/// BIT = 4j+f (hmask bit), q8off = 128j + 32f — const args because the shift
/// intrinsics need compile-time immediates; the C kernel's 2×4 loop is
/// unrolled by its compiler the same way.
///
/// # Safety
/// `w_row`/`cols` bounds as for `dot_row`; `sumi` has M lanes.
#[inline(always)]
unsafe fn field_dot<const SHIFT: i32, const BIT: i32, const M: usize>(
    sb: usize,
    cols: *const u8,
    q8off: usize,
    hbits: __m256i,
    q3bits: __m256i,
    scales_j: __m256i,
    shuf_f: __m256i,
    m3: __m256i,
    mone: __m256i,
    sumi: &mut [__m256i; M],
) {
    unsafe {
        let q3l = _mm256_and_si256(_mm256_srli_epi16::<SHIFT>(q3bits), m3);
        let q3h = _mm256_slli_epi16::<2>(
            _mm256_srli_epi16::<BIT>(_mm256_andnot_si256(hbits, _mm256_slli_epi16::<BIT>(mone))),
        );
        let sc = _mm256_shuffle_epi8(scales_j, shuf_f);
        for c in 0..M {
            // qs of Q8_K block `sb` of column c: block at
            // c*(NB*Q8K_STRIDE) + sb*Q8K_STRIDE, quants at +8.
            // SAFETY: unaligned load inside the column buffer.
            let q8f = _mm256_loadu_si256(
                cols.add(c * NB * Q8K_STRIDE + sb * Q8K_STRIDE + 8 + q8off) as *const __m256i,
            );
            // q3l (u8 0..3) and q3h (u8 {0,4}) both sit inside maddubs'
            // s16 saturation (2*4*127 = 1016 < 32767); the madd product
            // stays inside s16 too (31*762 = 23622).
            let q8s = _mm256_maddubs_epi16(q3h, q8f);
            let p = _mm256_maddubs_epi16(q3l, q8f);
            let p = _mm256_sub_epi16(p, q8s);
            let p = _mm256_madd_epi16(sc, p);
            sumi[c] = _mm256_add_epi32(sumi[c], p);
        }
    }
}

/// The row dot: port of the __AVX2__ branch of ggml_vec_dot_q3_K_q8_K,
/// one 880-byte Q3_K row (8 super-blocks) against M activation columns
/// held as NB Q8_K blocks each. Writes y[0..M].
///
/// # Safety
/// `w` must point at ROW_BYTES readable bytes, `cols` at M*NB*Q8K_STRIDE
/// readable bytes, `y` at M writable f32.
unsafe fn dot_row<const M: usize>(w: *const u8, cols: *const u8, y: *mut f32) {
    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let mone = _mm256_set1_epi8(1);
        let m32 = _mm_set1_epi8(32);
        // SAFETY: four 32-byte loads inside the 128-byte static table.
        let shuf: [__m256i; 4] = std::array::from_fn(|f| {
            _mm256_loadu_si256(K_SHUFFLE_Q3K.as_ptr().add(32 * f) as *const __m256i)
        });
        let kmask1 = 0x0303_0303u32;
        let kmask2 = 0x0f0f_0f0fu32;

        let mut acc = [0.0f32; M];
        for sb in 0..NB {
            let base = w.add(110 * sb);
            // hmask[32] @ +0
            // SAFETY: unaligned load inside the super-block.
            let hbits = _mm256_loadu_si256(base as *const __m256i);
            // scales[12] @ +96 decoded with the aux[] shuffle, verbatim.
            // SAFETY: three unaligned u32 reads inside the super-block.
            let aux = [
                (base.add(96) as *const u32).read_unaligned(),
                (base.add(100) as *const u32).read_unaligned(),
                (base.add(104) as *const u32).read_unaligned(),
            ];
            let s128 = _mm_set_epi32(
                (((aux[1] >> 4) & kmask2) | (((aux[2] >> 6) & kmask1) << 4)) as i32,
                (((aux[0] >> 4) & kmask2) | (((aux[2] >> 4) & kmask1) << 4)) as i32,
                ((aux[1] & kmask2) | (((aux[2] >> 2) & kmask1) << 4)) as i32,
                ((aux[0] & kmask2) | (((aux[2] >> 0) & kmask1) << 4)) as i32,
            );
            let s128 = _mm_sub_epi8(s128, m32);
            let all_scales = _mm256_cvtepi8_epi16(s128);
            let l = _mm256_extracti128_si256(all_scales, 0);
            let h = _mm256_extracti128_si256(all_scales, 1);
            let scales = [_mm256_set_m128i(l, l), _mm256_set_m128i(h, h)];

            // Super-block scale d: f16 at +108.
            // SAFETY: one unaligned u16 read inside the super-block.
            let d = half_to_f32((base.add(108) as *const u16).read_unaligned());

            let mut sumi = [_mm256_setzero_si256(); M];
            for j in 0..2 {
                // qs[64] @ +32, 32 bytes per half j.
                // SAFETY: unaligned load inside the super-block.
                let q3bits = _mm256_loadu_si256(base.add(32 + 32 * j) as *const __m256i);
                let scj = scales[j];
                // j=0: fields 0..3 → (SHIFT, BIT, q8off) = (2f, f, 32f)
                // j=1: fields 0..3 → (2f, 4+f, 128+32f)
                if j == 0 {
                    field_dot::<0, 0, M>(sb, cols, 0, hbits, q3bits, scj, shuf[0], m3, mone, &mut sumi);
                    field_dot::<2, 1, M>(sb, cols, 32, hbits, q3bits, scj, shuf[1], m3, mone, &mut sumi);
                    field_dot::<4, 2, M>(sb, cols, 64, hbits, q3bits, scj, shuf[2], m3, mone, &mut sumi);
                    field_dot::<6, 3, M>(sb, cols, 96, hbits, q3bits, scj, shuf[3], m3, mone, &mut sumi);
                } else {
                    field_dot::<0, 4, M>(sb, cols, 128, hbits, q3bits, scj, shuf[0], m3, mone, &mut sumi);
                    field_dot::<2, 5, M>(sb, cols, 160, hbits, q3bits, scj, shuf[1], m3, mone, &mut sumi);
                    field_dot::<4, 6, M>(sb, cols, 192, hbits, q3bits, scj, shuf[2], m3, mone, &mut sumi);
                    field_dot::<6, 7, M>(sb, cols, 224, hbits, q3bits, scj, shuf[3], m3, mone, &mut sumi);
                }
            }
            for c in 0..M {
                // d of Q8_K block sb of column c sits at the block start.
                // SAFETY: unaligned f32 read inside the column buffer.
                let dcol =
                    (cols.add(c * NB * Q8K_STRIDE + sb * Q8K_STRIDE) as *const f32).read_unaligned();
                acc[c] += dcol * d * hsum_i32(sumi[c]) as f32;
            }
        }
        for c in 0..M {
            // SAFETY: y has M writable f32 for this row.
            *y.add(c) = acc[c];
        }
    }
}

// ------------------------------------------------------------- topology

fn parse_cpu_list(s: &str) -> Vec<u32> {
    let mut out = Vec::new();
    for part in s.trim().split(',') {
        if let Some((a, b)) = part.split_once('-') {
            for c in a.trim().parse::<u32>().unwrap()..=b.trim().parse::<u32>().unwrap() {
                out.push(c);
            }
        } else {
            out.push(part.trim().parse().unwrap());
        }
    }
    out
}

fn read_sys(path: &str) -> Option<String> {
    std::fs::read_to_string(path).ok()
}

/// One CCD = the logical cpus sharing one L3 (Zen 3: 8 cores × SMT). Returns
/// groups sorted by primary core, each group listing the primary hyperthread
/// of each core first, then the siblings in the same core order.
fn ccd_topology() -> Vec<Vec<u32>> {
    let mut groups: Vec<(String, Vec<u32>)> = Vec::new(); // (l3 key, primaries)
    let mut cpu = 0;
    while let Some(sib) = read_sys(&format!(
        "/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    )) {
        let siblings = parse_cpu_list(&sib);
        let primary = *siblings.iter().min().unwrap();
        if primary == cpu {
            let l3 = read_sys(&format!(
                "/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
            ))
            .unwrap_or_default();
            match groups.iter_mut().find(|(k, _)| *k == l3) {
                Some((_, v)) => v.push(cpu),
                None => groups.push((l3, vec![cpu])),
            }
        }
        cpu += 1;
    }
    if groups.is_empty() {
        return vec![(0..cpu as u32).collect()];
    }
    groups.sort_by_key(|(_, v)| v[0]);
    groups
        .into_iter()
        .map(|(_, primaries)| {
            let mut full = primaries.clone();
            for &p in &primaries {
                let sib = parse_sys(&format!(
                    "/sys/devices/system/cpu/cpu{p}/topology/thread_siblings_list"
                ));
                for s in sib {
                    if s != p {
                        full.push(s);
                    }
                }
            }
            full
        })
        .collect()
}

fn parse_sys(path: &str) -> Vec<u32> {
    parse_cpu_list(&read_sys(path).unwrap_or_default())
}

// ------------------------------------------------------------- pool

/// Sense-reversing spin barrier: pinned workers park nowhere, the pause
/// loop keeps release latency at ~µs scale (a futex barrier would add
/// tens of µs per iteration on the small shapes).
struct Barrier {
    n: usize,
    count: std::sync::atomic::AtomicUsize,
    sense: AtomicBool,
}

impl Barrier {
    fn new(n: usize) -> Self {
        Self { n, count: std::sync::atomic::AtomicUsize::new(0), sense: AtomicBool::new(false) }
    }

    fn wait(&self, local: &mut bool) {
        let c = self.count.fetch_add(1, Ordering::SeqCst) + 1;
        if c == self.n {
            self.count.store(0, Ordering::SeqCst);
            self.sense.store(!*local, Ordering::Release);
            *local = !*local;
        } else {
            while self.sense.load(Ordering::Acquire) == *local {
                std::hint::spin_loop();
            }
            *local = !*local;
        }
    }
}

/// SAFETY: the raw pointers inside are partitioned disjointly per thread
/// (rows and q8 blocks never overlap between Assign entries); the shared
/// weight/x buffers are read-only during the run.
struct Job {
    w: *const u8,
    x: *const f32,
    cols: *mut u8,
    y: *mut f32,
    rows: std::ops::Range<usize>,
    qblocks: std::ops::Range<usize>,
    m: usize,
}
unsafe impl Send for Job {}

/// Pin the calling thread to one cpu.
///
/// # Safety
/// `cpu` must exist; affinity syscalls are the sanctioned unsafe use here.
unsafe fn pin(cpu: u32) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_SET(cpu as usize, &mut set);
        let r = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        assert_eq!(r, 0, "sched_setaffinity({cpu}) failed");
    }
}

/// Re-quantize this thread's share of the activation columns (phase A of
/// every iteration, inside the timed window — ggml's mul_mat quantizes per
/// graph compute too).
///
/// # Safety
/// Job bounds as per the pool protocol.
unsafe fn phase_a(job: &Job) {
    unsafe {
        for b in job.qblocks.clone() {
            let col = b / NB;
            let blk = b % NB;
            let x = std::slice::from_raw_parts(job.x.add(col * K + blk * 256), 256);
            quantize_q8k_block(x, job.cols.add(b * Q8K_STRIDE));
        }
    }
}

/// Dot this thread's contiguous row range (phase B).
///
/// # Safety
/// Job bounds as per the pool protocol.
unsafe fn phase_b(job: &Job) {
    unsafe {
        for r in job.rows.clone() {
            if PREFETCH_ROWS > 0 {
                let ahead = (r + PREFETCH_ROWS).min(job.rows.end - 1);
                let wp = job.w.add(ahead * ROW_BYTES);
                for l in 0..14 {
                    _mm_prefetch(wp.add(64 * l) as *const i8, _MM_HINT_T0);
                }
            }
            let w = job.w.add(r * ROW_BYTES);
            let y = job.y.add(r * job.m);
            match job.m {
                1 => dot_row::<1>(w, job.cols, y),
                8 => dot_row::<8>(w, job.cols, y),
                _ => unreachable!("sweep uses m in {{1, 8}}"),
            }
        }
    }
}

/// CCD-partitioned assignments: thread t of T goes to CCD t/(T/4), slot
/// t%(T/4) of that CCD's cpu list (physical cores first), owning the matching
/// contiguous slice of that CCD's quarter of the rows, plus a contiguous
/// share of the m*NB activation blocks for phase A.
fn assignments(
    n_rows: usize,
    m: usize,
    threads: usize,
    topo: &[Vec<u32>],
) -> Vec<(std::ops::Range<usize>, std::ops::Range<usize>, u32)> {
    let nccd = topo.len();
    assert!(
        threads % nccd == 0,
        "thread sweep ({threads}) must be a multiple of the CCD count ({nccd})"
    );
    let tpc = threads / nccd;
    let mut out = Vec::with_capacity(threads);
    for t in 0..threads {
        let ccd = t / tpc;
        let slot = t % tpc;
        let cpu = topo[ccd][slot];
        let per_ccd = n_rows / nccd;
        let start = ccd * per_ccd;
        let ccd_rows = if ccd == nccd - 1 { n_rows - start } else { per_ccd };
        let sub = ccd_rows / tpc;
        let r0 = start + slot * sub;
        let r1 = if slot == tpc - 1 { start + ccd_rows } else { r0 + sub };
        let tot = m * NB;
        let qb = tot / threads;
        let b0 = t * qb;
        let b1 = if t == threads - 1 { tot } else { b0 + qb };
        out.push((r0..r1, b0..b1, cpu));
    }
    out
}

/// One timed configuration: WARMUP + TIMED barrier-synchronised iterations
/// over `threads` pinned workers, main thread as the last participant.
/// Returns (µs per iteration, y) for the error check.
fn run_config(
    w: *const u8,
    x: *const f32,
    n: usize,
    m: usize,
    threads: usize,
    topo: &[Vec<u32>],
) -> (f64, Vec<f32>) {
    let mut cols = vec![0u8; m * NB * Q8K_STRIDE];
    let mut y = vec![0.0f32; n * m];
    let bar = Barrier::new(threads);
    let done = AtomicBool::new(false);
    let assign = assignments(n, m, threads, topo);
    let cols_p = cols.as_mut_ptr();
    let y_p = y.as_mut_ptr();
    let main_t = threads - 1;
    let mut local = false;
    let us = std::thread::scope(|s| {
        for t in 0..main_t {
            let (rows, qblocks, cpu) = assign[t].clone();
            let job = Job { w, x, cols: cols_p, y: y_p, rows, qblocks, m };
            let bar = &bar;
            let done = &done;
            s.spawn(move || {
                // SAFETY: cpu exists per the topology read.
                unsafe { pin(cpu) };
                let mut local = false;
                loop {
                    bar.wait(&mut local);
                    if done.load(Ordering::Relaxed) {
                        return;
                    }
                    // SAFETY: this thread's ranges are disjoint from the
                    // other participants'.
                    unsafe { phase_a(&job) };
                    bar.wait(&mut local);
                    unsafe { phase_b(&job) };
                    bar.wait(&mut local);
                }
            });
        }
        // SAFETY: main takes the last slot's assignment.
        unsafe { pin(assign[main_t].2) };
        let (rows, qblocks, _) = assign[main_t].clone();
        let job = Job { w, x, cols: cols_p, y: y_p, rows, qblocks, m };
        let it = |local: &mut bool| {
            bar.wait(local);
            unsafe { phase_a(&job) };
            bar.wait(local);
            unsafe { phase_b(&job) };
            bar.wait(local);
        };
        for _ in 0..WARMUP {
            it(&mut local);
        }
        let t0 = Instant::now();
        for _ in 0..TIMED {
            it(&mut local);
        }
        let us = t0.elapsed().as_secs_f64() * 1e6 / TIMED as f64;
        done.store(true, Ordering::Relaxed);
        bar.wait(&mut local);
        us
    });
    (us, y)
}

/// Weight buffer. Baseline: a plain Vec. With WEIGHTS_HUGEPAGE the bytes go
/// into a 2 MiB-aligned anonymous mapping advised MADV_HUGEPAGE (THP is in
/// madvise mode on this box — this is the only route to huge pages for the
/// weight stream). The mapping intentionally outlives the run un-freed; the
/// process exits right after the sweep.
fn load_weights(path: &str, n: usize) -> (*const u8, Vec<u8>) {
    if WEIGHTS_HUGEPAGE {
        // SAFETY: mmap/madvise sized for n + one huge page; the read fills
        // exactly the aligned n-byte span.
        unsafe {
            const HP: usize = 2 * 1024 * 1024;
            let map = libc::mmap(
                std::ptr::null_mut(),
                n + HP,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert!(map != libc::MAP_FAILED, "mmap({n}) failed");
            let aligned = (map as usize + HP - 1) & !(HP - 1);
            assert_eq!(
                libc::madvise(aligned as *mut _, n, libc::MADV_HUGEPAGE),
                0,
                "madvise(MADV_HUGEPAGE) failed"
            );
            let mut f = std::fs::File::open(path)
                .unwrap_or_else(|e| panic!("open {path}: {e}"));
            let mut dst = std::slice::from_raw_parts_mut(aligned as *mut u8, n);
            f.read_exact(&mut dst)
                .unwrap_or_else(|e| panic!("read {path}: {e}"));
            (aligned as *const u8, Vec::new())
        }
    } else {
        let v = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        assert_eq!(v.len(), n, "{path} size mismatch");
        let p = v.as_ptr();
        (p, v)
    }
}

fn read_f32(path: &str, n: usize) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(b.len(), n * 4, "{path} size mismatch");
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn main() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("FATAL: AVX2 not detected");
        std::process::exit(1);
    }
    let data = std::env::var("MULLE_DATA").unwrap_or_else(|_| "/root/mulle-data".to_string());

    const GATE_BYTES: usize = 90_112 * ROW_BYTES; // 79,298,560
    const BIG_BYTES: usize = 360_448 * ROW_BYTES; // 317,194,240
    let (gate, gate_keep) = load_weights(&format!("{data}/gate.q3k"), GATE_BYTES);
    let (big, big_keep) = load_weights(&format!("{data}/big.q3k"), BIG_BYTES);
    let x1 = read_f32(&format!("{data}/x_m1.f32"), K);
    let x8 = read_f32(&format!("{data}/x_m8.f32"), K * 8);

    let topo = ccd_topology();
    for (i, g) in topo.iter().enumerate() {
        println!("TOPO ccd{i} cpus={g:?}");
    }

    struct Shape {
        name: &'static str,
        w: *const u8,
        n: usize,
        wbytes: usize,
    }
    let shapes = [
        Shape { name: "expert0", w: gate, n: 1408, wbytes: 1408 * ROW_BYTES },
        Shape { name: "stack", w: gate, n: 90_112, wbytes: GATE_BYTES },
        Shape { name: "big", w: big, n: 360_448, wbytes: BIG_BYTES },
    ];

    let mut all_ok = true;
    for sh in &shapes {
        for m in [1usize, 8] {
            let x = if m == 1 { &x1 } else { &x8 };
            let yref = read_f32(&format!("{data}/y_ref_{}_m{}.f32", sh.name, m), sh.n * m);
            let denom = yref.iter().fold(0.0f32, |a, v| a.max(v.abs())) as f64;
            for threads in THREADS {
                let (us, y) = run_config(sh.w, x.as_ptr(), sh.n, m, threads, &topo);
                let gbs = sh.wbytes as f64 / (us * 1e-6) / 1e9;
                let mut maxerr = 0.0f64;
                for r in 0..sh.n {
                    for c in 0..m {
                        let d = ((y[r * m + c] - yref[r * m + c]) as f64).abs();
                        if d > maxerr {
                            maxerr = d;
                        }
                    }
                }
                let rel = maxerr / denom;
                println!(
                    "RUST shape={} m={} threads={} us={:.2} GB/s={:.2} max_rel_err={:.3e}",
                    sh.name, m, threads, us, gbs, rel
                );
                if rel > 1e-2 {
                    eprintln!(
                        "FAIL: {} m={} threads={} rel err {rel:.3e} exceeds 1e-2",
                        sh.name, m, threads
                    );
                    all_ok = false;
                }
            }
        }
    }
    // The weight buffers' owners (plain-Vec baseline) live to this point.
    drop(gate_keep);
    drop(big_keep);
    if !all_ok {
        std::process::exit(1);
    }
    println!("PASSED: all 24 configs within 1e-2 (q8_K activation design)");
}
