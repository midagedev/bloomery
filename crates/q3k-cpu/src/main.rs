// q3k-cpu — Q3_K x Q8_K host-expert-tier pre-study kernel (MUL-3).
//
// Port of the arithmetic of ggml's AVX2 `ggml_vec_dot_q3_K_q8_K`
// (ik_llama.cpp c10fbbcc, ggml/src/ggml-quants.c) plus a scalar port of
// `quantize_row_q8_K_ref`: no invented decode. Rows are spread over a pinned
// std-thread pool; each CCD's threads take contiguous row ranges.
//
// Protocol: one line per timed row, same units as tools/ref/q3k_cpu_ref.cpp:
//   RUST shape=<s> m=<M> threads=<t> us=<..> GB/s=<..> max_rel_err=<..>
// Diagnostics use the `RUST-note` prefix. Exit status is 1 if any row misses
// the 1e-2 error bound (all rows still run, so the table stays complete).

use std::arch::x86_64::*;
use std::fs;
use std::sync::{Arc, Barrier, Mutex};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Layout constants (Q3_K: QK_K = 256 values per super-block, 110 B each)
// ---------------------------------------------------------------------------

const K: usize = 2048;
const NB: usize = 8; // super-blocks per row
const SB_BYTES: usize = 110;
const ROW_BYTES: usize = NB * SB_BYTES; // 7040
const QK_K: usize = 256;

// Levers. Defaults are the final kept configuration (see RESULTS.md);
// every flag below overrides its default for ablation runs.
const DEFAULT_HUGEPAGE: bool = false;
const DEFAULT_PREFETCH: bool = false;
const DEFAULT_UNROLL2: bool = false;

const WARMUP: usize = 5;
const TIMED: usize = 50;
const ERR_BOUND: f64 = 1e-2;

// ---------------------------------------------------------------------------
// Q8_K activation blocks (compact: only d and qs feed the dot product)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct Q8KBlock {
    d: f32,
    qs: [i8; QK_K],
}

/// Scalar port of ggml's `quantize_row_q8_K_ref` for one 256-value block.
/// Deviation: the C version leaves `bsums` untouched when amax == 0; this
/// port has no bsums field at all. Outputs are identical (d == 0 zeroes the
/// dot either way).
fn quantize_q8k_block(x: &[f32], out: &mut Q8KBlock) {
    assert_eq!(x.len(), QK_K);
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        let ax = v.abs();
        if ax > amax {
            amax = ax;
            max = v;
        }
    }
    if amax == 0.0 {
        out.d = 0.0;
        out.qs = [0; QK_K];
        return;
    }
    let iscale = -127.0 / max;
    for (j, &v) in x.iter().enumerate() {
        let q = nearest_int(iscale * v).min(127);
        debug_assert!((-128..=127).contains(&q));
        out.qs[j] = q as i8;
    }
    out.d = 1.0 / iscale;
}

/// Exact port of ggml's `nearest_int` (the 12582912.f magic-constant trick).
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(fval <= 4194303.0);
    let val = fval + 12582912.0;
    ((val.to_bits() & 0x007fffff) as i32) - 0x00400000
}

// ---------------------------------------------------------------------------
// AVX2 Q3_K x Q8_K dot product — port of ggml_vec_dot_q3_K_q8_K's AVX2 branch
// ---------------------------------------------------------------------------

/// Shuffle table copied from ggml's `get_scale_shuffle_q3k` (4 x 32 B).
static SCALE_SHUFFLE_Q3K: [u8; 128] = [
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
    6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
    8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
    10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
    12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
    14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
];

/// IEEE-754 half to float, integer-only (same routine as the q3k-gemv
/// sibling crate, whose Q3_K decode verified against ggml at 1e-7).
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

/// Horizontal sum of 8 packed floats.
#[inline(always)]
unsafe fn hsum_float_8(v: __m256) -> f32 {
    // SAFETY: pure register arithmetic, no memory access.
    unsafe {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps::<1>(v);
        let sum = _mm_add_ps(lo, hi);
        let shuf = _mm_movehdup_ps(sum);
        let sum = _mm_add_ps(sum, shuf);
        let shuf = _mm_movehl_ps(shuf, sum);
        let sum = _mm_add_ss(sum, shuf);
        _mm_cvtss_f32(sum)
    }
}

/// Per-super-block integer accumulator shared by the 1-row and 2-row kernels.
/// Mirrors the AVX2 branch of `ggml_vec_dot_q3_K_q8_K`, including the
/// low-2-bits/high-bit split that feeds `vpmaddubsw` (`_mm256_maddubs_epi16`).
///
/// The one implementation-level difference from the C: `_mm256_slli_epi16`
/// / `_mm256_srli_epi16` take a compile-time count in Rust's std::arch but
/// the C passes a runtime `bit` (0..7). Per 16-bit lane the C computes
/// `(((~h) & M) >> bit) << 2` with `M = set1(1<<bit)`; the shift pairs cancel
/// bit-exactly to "4 where the hmask bit is clear, 0 where set" (each byte of
/// `M` is `1<<bit`, so `>>bit` yields 0/1 per byte with no cross-byte carry
/// for bit <= 7), which is what the compare/andnot sequence below computes.
/// The decode itself — which bits mean what — is unchanged from ggml.
#[inline(always)]
unsafe fn block_sumi(
    q3: *const u8,
    mut q8: *const i8,
    scales: &[__m256i; 2],
    hbits: __m256i,
    m3: __m256i,
    four: __m256i,
    zero: __m256i,
) -> __m256i {
    // SAFETY: caller guarantees q3 points at 64 readable bytes (one block's
    // qs) and q8 at 256 readable bytes (one Q8_K block's qs); all loads are
    // unaligned. Register-only otherwise.
    unsafe {
        let mut sumi = _mm256_setzero_si256();
        let mut q3p = q3;
        let mut bit: u32 = 0;
        for j in 0..2 {
            let q3bits = _mm256_loadu_si256(q3p as *const __m256i);
            q3p = q3p.add(32);

            // Four (low-2-bits, high-bit) pairs at shifts 0,2,4,6 / bits b..b+3.
            macro_rules! pair {
                ($shift:literal) => {{
                    let q3l = _mm256_and_si256(_mm256_srli_epi16::<$shift>(q3bits), m3);
                    let m = _mm256_set1_epi8(1u8.wrapping_shl(bit) as i8);
                    bit += 1;
                    let t = _mm256_andnot_si256(hbits, m);
                    let is_zero = _mm256_cmpeq_epi8(t, zero);
                    let q3h = _mm256_andnot_si256(is_zero, four);
                    let q8v = _mm256_loadu_si256(q8 as *const __m256i);
                    q8 = q8.add(32);
                    let p = _mm256_sub_epi16(
                        _mm256_maddubs_epi16(q3l, q8v),
                        _mm256_maddubs_epi16(q3h, q8v),
                    );
                    p
                }};
            }
            // NOTE: one statement per pair; `bit` advances 0..3 then 4..7,
            // exactly like the C loop.
            let p0 = pair!(0);
            let p1 = pair!(2);
            let p2 = pair!(4);
            let p3 = pair!(6);

            let s = |i: usize| {
                _mm256_loadu_si256(SCALE_SHUFFLE_Q3K.as_ptr().add(i * 32) as *const __m256i)
            };
            let p0 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], s(0)), p0);
            let p1 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], s(1)), p1);
            let p2 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], s(2)), p2);
            let p3 = _mm256_madd_epi16(_mm256_shuffle_epi8(scales[j], s(3)), p3);

            let a01 = _mm256_add_epi32(p0, p1);
            let a23 = _mm256_add_epi32(p2, p3);
            sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(a01, a23));
        }
        sumi
    }
}

/// Unpack one super-block's d, sign-extended scales and hmask.
/// Returns (d_combined, scales, hbits).
#[inline(always)]
unsafe fn block_setup(sb: *const u8, q8d: f32, m32: __m128i) -> (f32, [__m256i; 2], __m256i) {
    // SAFETY: caller guarantees sb points at a readable 110 B super-block.
    unsafe {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;
        let dh = std::ptr::read_unaligned(sb.add(108) as *const u16);
        let d = q8d * half_to_f32(dh);
        let mut aux = [0u32; 3];
        std::ptr::copy_nonoverlapping(sb.add(96), aux.as_mut_ptr() as *mut u8, 12);
        let scales128 = _mm_set_epi32(
            (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
            (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
            ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
            ((aux[0] & KMASK2) | (((aux[2] >> 0) & KMASK1) << 4)) as i32,
        );
        let scales128 = _mm_sub_epi8(scales128, m32);
        let all = _mm256_cvtepi8_epi16(scales128);
        let l = _mm256_extracti128_si256::<0>(all);
        let h = _mm256_extracti128_si256::<1>(all);
        let scales = [_mm256_set_m128i(l, l), _mm256_set_m128i(h, h)];
        let hbits = _mm256_loadu_si256(sb as *const __m256i);
        (d, scales, hbits)
    }
}

/// Dot of one weight row (8 super-blocks) with one quantized column.
#[inline(always)]
unsafe fn row_dot(w_row: *const u8, xq_col: *const Q8KBlock) -> f32 {
    // SAFETY: caller guarantees w_row points at NB readable super-blocks and
    // xq_col at NB readable Q8_K blocks. Requires AVX2+FMA (checked at startup).
    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let m32 = _mm_set1_epi8(32);
        let four = _mm256_set1_epi8(4);
        let zero = _mm256_setzero_si256();
        let mut acc = _mm256_setzero_ps();
        for b in 0..NB {
            let qb = &*xq_col.add(b);
            let sb = w_row.add(b * SB_BYTES);
            let (d, scales, hbits) = block_setup(sb, qb.d, m32);
            let sumi = block_sumi(sb.add(32), qb.qs.as_ptr(), &scales, hbits, m3, four, zero);
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

/// Two-row version sharing the Q8 activation loads (the --unroll2 lever).
#[inline(always)]
unsafe fn row_dot_2(w0: *const u8, w1: *const u8, xq_col: *const Q8KBlock) -> (f32, f32) {
    // SAFETY: same contract as row_dot, for two consecutive readable rows.
    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let m32 = _mm_set1_epi8(32);
        let four = _mm256_set1_epi8(4);
        let zero = _mm256_setzero_si256();
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();
        for b in 0..NB {
            let qb = &*xq_col.add(b);
            let sb0 = w0.add(b * SB_BYTES);
            let sb1 = w1.add(b * SB_BYTES);
            let (d0, sc0, hb0) = block_setup(sb0, qb.d, m32);
            let (d1, sc1, hb1) = block_setup(sb1, qb.d, m32);
            let s0 = block_sumi(sb0.add(32), qb.qs.as_ptr(), &sc0, hb0, m3, four, zero);
            let s1 = block_sumi(sb1.add(32), qb.qs.as_ptr(), &sc1, hb1, m3, four, zero);
            acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d0), _mm256_cvtepi32_ps(s0), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&d1), _mm256_cvtepi32_ps(s1), acc1);
        }
        (hsum_float_8(acc0), hsum_float_8(acc1))
    }
}

// ---------------------------------------------------------------------------
// Topology: order CPUs spread across L3 (CCD) groups, cores before SMT sibs
// ---------------------------------------------------------------------------

struct Topo {
    /// All CPUs, spread round-robin across CCD groups: physical cores first,
    /// then SMT siblings in the same spread order.
    order: Vec<usize>,
    /// CCD group index per cpu id (position in sorted group-key order).
    ccd_of: Vec<usize>,
    /// Group keys (L3 shared_cpu_list strings) in sorted order, for reporting.
    groups: Vec<String>,
}

fn read_topo() -> Topo {
    let mut cpus: Vec<usize> = fs::read_dir("/sys/devices/system/cpu")
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("cpu") && n[3..].chars().all(|c| c.is_ascii_digit()))
        .map(|n| n[3..].parse().unwrap())
        .collect();
    cpus.sort_unstable();
    let read_one = |cpu: usize, rel: &str| {
        fs::read_to_string(format!("/sys/devices/system/cpu/cpu{cpu}/{rel}"))
            .unwrap()
            .trim()
            .to_string()
    };
    // L3 group key per cpu: shared_cpu_list of the cache index with level 3.
    let mut l3key = Vec::new();
    for &cpu in &cpus {
        let mut key = String::new();
        for idx in 0..8 {
            let lvl = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{idx}/level");
            if fs::read_to_string(&lvl).map(|s| s.trim() == "3").unwrap_or(false) {
                key = read_one(cpu, &format!("cache/index{idx}/shared_cpu_list"));
                break;
            }
        }
        assert!(!key.is_empty(), "no L3 found for cpu{cpu}");
        l3key.push(key);
    }
    let mut groups: Vec<String> = l3key.clone();
    groups.sort();
    groups.dedup();
    let ccd_of_vec: Vec<usize> = l3key
        .iter()
        .map(|k| groups.iter().position(|g| g == k).unwrap())
        .collect();
    // One representative (lowest cpu) per thread_siblings_list key = cores.
    let mut sibs: Vec<(String, usize)> =
        cpus.iter().map(|&c| (read_one(c, "topology/thread_siblings_list"), c)).collect();
    sibs.sort();
    let mut core_rep = vec![false; cpus.len()];
    let mut last = String::new();
    for (key, cpu) in &sibs {
        if *key != last {
            last = key.clone();
            core_rep[cpus.iter().position(|&c| c == *cpu).unwrap()] = true;
        }
    }
    // Spread order: round-robin over CCD groups, cores then siblings.
    let mut by_ccd: Vec<Vec<usize>> = vec![Vec::new(); groups.len()];
    let mut sib_by_ccd: Vec<Vec<usize>> = vec![Vec::new(); groups.len()];
    for (i, &cpu) in cpus.iter().enumerate() {
        if core_rep[i] {
            by_ccd[ccd_of_vec[i]].push(cpu);
        } else {
            sib_by_ccd[ccd_of_vec[i]].push(cpu);
        }
    }
    let mut order = Vec::with_capacity(cpus.len());
    for round in 0.. {
        let mut moved = false;
        for g in &by_ccd {
            if let Some(&c) = g.get(round) {
                order.push(c);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    for round in 0.. {
        let mut moved = false;
        for g in &sib_by_ccd {
            if let Some(&c) = g.get(round) {
                order.push(c);
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    // ccd_of indexed by cpu id (assumes dense 0..max).
    let maxcpu = *cpus.iter().max().unwrap();
    let mut ccd_of = vec![0usize; maxcpu + 1];
    for (i, &cpu) in cpus.iter().enumerate() {
        ccd_of[cpu] = ccd_of_vec[i];
    }
    Topo { order, ccd_of, groups }
}

/// Pin the calling thread to `cpu` and verify the affinity took.
fn pin_self(cpu: usize) {
    // SAFETY: sched_setaffinity/sched_getaffinity on pid 0 (self) with a
    // correctly-sized cpu_set_t; CPU_SET/CPU_ISSET/CPU_ZERO are the libc
    // helpers for that type.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let rc = libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set as *const libc::cpu_set_t,
        );
        assert_eq!(rc, 0, "sched_setaffinity({cpu}) failed");
        let mut got: libc::cpu_set_t = std::mem::zeroed();
        let rc = libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut got as *mut libc::cpu_set_t,
        );
        assert_eq!(rc, 0, "sched_getaffinity failed");
        assert!(libc::CPU_ISSET(cpu, &got), "pin to cpu{cpu} did not stick");
    }
}

// ---------------------------------------------------------------------------
// Thread pool: persistent pinned workers, barrier-timed iterations
// ---------------------------------------------------------------------------

/// One timed configuration. Raw pointers are valid for the whole pool run:
/// main owns the buffers and joins the pool before dropping them.
#[derive(Clone)]
struct Job {
    w: *const u8,
    xq: *const Q8KBlock,
    y: *mut f32,
    nrows: usize,
    m: usize,
    unroll2: bool,
    prefetch: bool,
}

// SAFETY: Job only crosses threads behind a Mutex + Barrier rendezvous, and
// the pointed-to buffers are mutably-disjoint per worker (rows) or read-only.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

/// Even row split: chunk p of n over t (first n%t chunks get one extra row).
fn chunk_range(n: usize, t: usize, p: usize) -> (usize, usize) {
    let base = n / t;
    let rem = n % t;
    let start = p * base + p.min(rem);
    (start, start + base + (p < rem) as usize)
}

unsafe fn compute_chunk(job: &Job, r0: usize, r1: usize) {
    // SAFETY: r0..r1 is this worker's disjoint row range; w/xq/y come from
    // main's live buffers sized for nrows/m. Unaligned AVX loads only.
    unsafe {
        const AHEAD: usize = 2;
        let w = job.w;
        let xq = job.xq;
        let y = job.y;
        let m = job.m;
        if job.unroll2 {
            let mut r = r0;
            while r + 1 < r1 {
                if job.prefetch && r + AHEAD + 1 < r1 {
                    for k in 0..4 {
                        _mm_prefetch(
                            w.add((r + AHEAD) * ROW_BYTES + k * 2048) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
                for c in 0..m {
                    let (v0, v1) =
                        row_dot_2(w.add(r * ROW_BYTES), w.add((r + 1) * ROW_BYTES), xq.add(c * NB));
                    *y.add(r * m + c) = v0;
                    *y.add((r + 1) * m + c) = v1;
                }
                r += 2;
            }
            if r < r1 {
                for c in 0..m {
                    *y.add(r * m + c) = row_dot(w.add(r * ROW_BYTES), xq.add(c * NB));
                }
            }
        } else {
            for r in r0..r1 {
                if job.prefetch && r + AHEAD < r1 {
                    for k in 0..4 {
                        _mm_prefetch(
                            w.add((r + AHEAD) * ROW_BYTES + k * 2048) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }
                for c in 0..m {
                    *y.add(r * m + c) = row_dot(w.add(r * ROW_BYTES), xq.add(c * NB));
                }
            }
        }
    }
}

struct Pool {
    barrier: Arc<Barrier>,
    job: Arc<Mutex<Job>>,
    done: Arc<std::sync::atomic::AtomicBool>,
    /// Chunk slot per worker: workers sorted by (ccd, cpu) so each CCD's
    /// threads take contiguous row ranges.
    slot: Vec<usize>,
    nrows: usize,
    handles: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn spawn(cpus: &[usize], ccd_of: &[usize]) -> Pool {
        let t = cpus.len();
        let barrier = Arc::new(Barrier::new(t + 1));
        let job = Arc::new(Mutex::new(Job {
            w: std::ptr::null(),
            xq: std::ptr::null(),
            y: std::ptr::null_mut(),
            nrows: 0,
            m: 0,
            unroll2: false,
            prefetch: false,
        }));
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut idx: Vec<usize> = (0..t).collect();
        idx.sort_by_key(|&i| (ccd_of[cpus[i]], cpus[i]));
        let mut slot = vec![0usize; t];
        for (rank, &i) in idx.iter().enumerate() {
            slot[i] = rank;
        }
        let mut handles = Vec::with_capacity(t);
        for i in 0..t {
            let barrier = barrier.clone();
            let job = job.clone();
            let done = done.clone();
            let cpu = cpus[i];
            let my_slot = slot[i];
            handles.push(
                std::thread::Builder::new()
                    .name(format!("q3k-w{i}"))
                    .spawn(move || {
                        pin_self(cpu);
                        loop {
                            barrier.wait(); // start
                            if done.load(std::sync::atomic::Ordering::Acquire) {
                                break;
                            }
                            let j = job.lock().unwrap().clone();
                            let (r0, r1) = chunk_range(j.nrows, t, my_slot);
                            unsafe { compute_chunk(&j, r0, r1) };
                            barrier.wait(); // end
                        }
                    })
                    .unwrap(),
            );
        }
        Pool { barrier, job, done, slot, nrows: 0, handles }
    }

    /// Time one configuration: main quantizes x (per-iteration work, as in
    /// ggml's graph_compute), then releases the workers. Returns avg µs.
    fn run_timed(
        &mut self,
        w: &[u8],
        x_cols: &[Vec<f32>],
        xq: &mut [Q8KBlock],
        y: &mut [f32],
        flags: (bool, bool),
    ) -> f64 {
        let (unroll2, prefetch) = flags;
        let nrows = w.len() / ROW_BYTES;
        let m = x_cols.len();
        assert_eq!(xq.len(), m * NB);
        assert_eq!(y.len(), nrows * m);
        *self.job.lock().unwrap() = Job {
            w: w.as_ptr(),
            xq: xq.as_ptr(),
            y: y.as_mut_ptr(),
            nrows,
            m,
            unroll2,
            prefetch,
        };
        self.nrows = nrows;
        let mut total = std::time::Duration::ZERO;
        for it in 0..WARMUP + TIMED {
            let t0 = Instant::now();
            // Per-iteration activation quantize on main, inside the timed
            // window: ggml re-quantizes x inside every graph_compute, and in
            // serving x is new every token, so this is real per-token work.
            for (c, col) in x_cols.iter().enumerate() {
                for b in 0..NB {
                    quantize_q8k_block(&col[b * QK_K..(b + 1) * QK_K], &mut xq[c * NB + b]);
                }
            }
            self.barrier.wait(); // start
            self.barrier.wait(); // end
            let dt = t0.elapsed();
            if it >= WARMUP {
                total += dt;
            }
        }
        // Publish happened-before is via the start barrier + the Mutex write
        // above (workers lock it after the start rendezvous).
        total.as_secs_f64() * 1e6 / TIMED as f64
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.done.store(true, std::sync::atomic::Ordering::Release);
        self.barrier.wait(); // release workers to see done
        for h in self.handles.drain(..) {
            h.join().unwrap();
        }
    }
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

fn read_f32s(path: &str) -> Vec<f32> {
    let b = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(b.len() % 4, 0, "{path}: size not a multiple of 4");
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn max_abs(v: &[f32]) -> f64 {
    v.iter().map(|&f| (f as f64).abs()).fold(0.0, f64::max)
}

fn max_rel_err(y: &[f32], yref: &[f32]) -> f64 {
    assert_eq!(y.len(), yref.len());
    let denom = max_abs(yref);
    let mut m: f64 = 0.0;
    for (a, b) in y.iter().zip(yref.iter()) {
        m = m.max(((*a as f64) - (*b as f64)).abs());
    }
    m / denom
}

/// Parse /proc/self/smaps for AnonHugePages (THP witness for --hugepage).
fn anon_hugepages_kb() -> u64 {
    let mut kb = 0u64;
    if let Ok(s) = fs::read_to_string("/proc/self/smaps") {
        for line in s.lines() {
            if let Some(v) = line.strip_prefix("AnonHugePages:") {
                kb += v.split_whitespace().next().unwrap_or("0").parse::<u64>().unwrap_or(0);
            }
        }
    }
    kb
}

// ---------------------------------------------------------------------------
// Main: full sweep, no args needed
// ---------------------------------------------------------------------------

fn main() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        eprintln!("q3k-cpu FATAL: AVX2+FMA required");
        std::process::exit(2);
    }

    let mut hugepage = DEFAULT_HUGEPAGE;
    let mut prefetch = DEFAULT_PREFETCH;
    let mut unroll2 = DEFAULT_UNROLL2;
    let mut dump_topo = false;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "--hugepage" => hugepage = true,
            "--no-hugepage" => hugepage = false,
            "--prefetch" => prefetch = true,
            "--no-prefetch" => prefetch = false,
            "--unroll2" => unroll2 = true,
            "--no-unroll2" => unroll2 = false,
            "--dump-topology" => dump_topo = true,
            _ => {
                eprintln!("unknown arg: {a}");
                std::process::exit(2);
            }
        }
    }

    let topo = read_topo();
    if dump_topo {
        println!("RUST-note ccd_groups={:?}", topo.groups);
        println!("RUST-note cpu_order={:?}", topo.order);
        return;
    }

    let data = std::env::var("MULLE_DATA").unwrap_or("/root/mulle-data-cpu-muse".to_string());
    let mut w_gate = fs::read(format!("{data}/gate.q3k")).expect("read gate.q3k");
    let mut w_big = fs::read(format!("{data}/big.q3k")).expect("read big.q3k");
    assert_eq!(w_gate.len() % ROW_BYTES, 0);
    assert_eq!(w_big.len() % ROW_BYTES, 0);
    let x_m1 = read_f32s(&format!("{data}/x_m1.f32"));
    let x_m8 = read_f32s(&format!("{data}/x_m8.f32"));
    assert_eq!(x_m1.len(), K);
    assert_eq!(x_m8.len(), K * 8);

    if hugepage {
        // SAFETY: madvise hint on live Vec storage; out-of-range advice is a
        // no-op verified by return code, never UB.
        for w in [&mut w_gate, &mut w_big] {
            let rc = unsafe {
                libc::madvise(
                    w.as_mut_ptr() as *mut libc::c_void,
                    w.len(),
                    libc::MADV_HUGEPAGE,
                )
            };
            if rc != 0 {
                eprintln!("RUST-note madvise(HUGEPAGE) failed: {}", std::io::Error::last_os_error());
            }
        }
    }
    println!(
        "RUST-note flags hugepage={hugepage} prefetch={prefetch} unroll2={unroll2} k={K} row_bytes={ROW_BYTES}"
    );
    println!("RUST-note ccd_groups={:?}", topo.groups);

    // Column views of x (copied once; quantize reads them per iteration).
    let x_cols_m1: Vec<Vec<f32>> = vec![x_m1.clone()];
    let x_cols_m8: Vec<Vec<f32>> = (0..8).map(|c| x_m8[c * K..(c + 1) * K].to_vec()).collect();

    struct Shape<'a> {
        name: &'a str,
        w: &'a [u8],
        nrows: usize,
    }
    let shapes = [
        Shape { name: "expert0", w: &w_gate[..1408 * ROW_BYTES], nrows: 1408 },
        Shape { name: "stack", w: &w_gate, nrows: w_gate.len() / ROW_BYTES },
        Shape { name: "big", w: &w_big, nrows: w_big.len() / ROW_BYTES },
    ];
    println!(
        "RUST-note rows expert0={} stack={} big={}",
        shapes[0].nrows, shapes[1].nrows, shapes[2].nrows
    );

    let mut xq = vec![Q8KBlock { d: 0.0, qs: [0; QK_K] }; 8 * NB];
    let max_y = shapes.iter().map(|s| s.nrows * 8).max().unwrap();
    let mut y = vec![0.0f32; max_y];

    // Quantize cost witness (single-threaded scalar, outside the timed loop).
    {
        let t0 = Instant::now();
        for _ in 0..100 {
            for b in 0..NB {
                quantize_q8k_block(&x_m1[b * QK_K..(b + 1) * QK_K], &mut xq[b]);
            }
        }
        println!("RUST-note quantize_m1_us_per_iter={:.2}", t0.elapsed().as_secs_f64() * 1e6 / 100.0);
    }

    let mut failed = false;
    for &t in &[8usize, 16, 32, 64] {
        let cpus = &topo.order[..t];
        println!("RUST-note threads={t} cpus={cpus:?}");
        let mut pool = Pool::spawn(cpus, &topo.ccd_of);
        // Chunk-slot witness: worker order -> (ccd, cpu, rows) for big.
        {
            let n = shapes[2].nrows;
            let desc: Vec<String> = (0..t)
                .map(|i| {
                    let (r0, r1) = chunk_range(n, t, pool.slot[i]);
                    format!("w{i}->ccd{}:cpu{}:[{r0},{r1})", topo.ccd_of[cpus[i]], cpus[i])
                })
                .collect();
            println!("RUST-note big_slots t={t} {}", desc.join(" "));
        }
        for sh in &shapes {
            for (x_cols, m) in [(&x_cols_m1, 1usize), (&x_cols_m8, 8)] {
                let yref = read_f32s(&format!("{data}/y_ref_{}_{}.f32", sh.name, if m == 1 { "m1" } else { "m8" }));
                let yy = &mut y[..sh.nrows * m];
                let us = pool.run_timed(sh.w, x_cols, &mut xq[..m * NB], yy, (unroll2, prefetch));
                let wbytes = (sh.nrows * ROW_BYTES) as f64;
                let gbs = wbytes / (us * 1e-6) / 1e9;
                let err = max_rel_err(yy, &yref);
                if err > ERR_BOUND {
                    failed = true;
                }
                println!(
                    "RUST shape={} m={m} threads={t} us={us:.2} GB/s={gbs:.2} max_rel_err={err:.3e}",
                    sh.name
                );
            }
        }
    }
    if hugepage {
        println!("RUST-note anon_hugepages_kb={}", anon_hugepages_kb());
    }
    if failed {
        eprintln!("q3k-cpu FAIL: a row missed the {ERR_BOUND:.0e} error bound");
        std::process::exit(1);
    }
}
