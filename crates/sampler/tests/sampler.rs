//! Gate for the sampler: every stage of the chain against an oracle written
//! from the reference's definition, the draw against its softmax, the seed
//! against itself, the allocator against zero, and the one-scan top-k chain
//! against a frozen copy of the selection chain (same kept set and id).
//!
//! Pure host code — no box resources, so no `hw_` prefix and no `#[ignore]`.

use sampler::{Sampler, SamplerParams};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

// ---------------------------------------------------------------- allocator

static CALLS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // const-initialized and destructor-free, so reading it inside the allocator cannot recurse.
    static COUNTING: Cell<bool> = const { Cell::new(false) };
}

struct Counting;

fn count() {
    if COUNTING.with(Cell::get) {
        CALLS.fetch_add(1, Relaxed);
    }
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller meets `alloc`'s contract for `l`, and `System` receives it unchanged.
        unsafe { System.alloc(l) }
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller meets `alloc_zeroed`'s contract for `l`, and `System` receives it
        // unchanged.
        unsafe { System.alloc_zeroed(l) }
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: `p` came from this allocator with layout `l` — so from `System`, as every
        // allocating method forwards there — and the caller meets `realloc`'s contract for
        // `new_size`; `System` receives all three unchanged.
        unsafe { System.realloc(p, l, new_size) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        // SAFETY: `p` came from this allocator with layout `l`, so from `System`.
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

// ---------------------------------------------------------------- inputs

/// splitmix64: the test's own input generator, independent of the sampler's.
struct Gen(u64);

impl Gen {
    fn u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^ (x >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.u64() % n
    }
    fn unit(&mut self) -> f64 {
        (self.u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Roughly normal with standard deviation `sd` (sum of four uniforms).
    fn normal(&mut self, sd: f64) -> f32 {
        let s: f64 = (0..4).map(|_| self.unit()).sum::<f64>() - 2.0;
        (s * sd * 3f64.sqrt()) as f32
    }
    /// A row where ties, signed zeros, infinities and NaN are common.
    fn nasty_row(&mut self, n: usize) -> Vec<f32> {
        const LEVELS: [f32; 6] = [f32::NEG_INFINITY, -1.0, -0.0, 0.0, 0.5, 2.0];
        // Mostly finite levels; +inf and NaN rarely, so most rows keep a finite max.
        (0..n)
            .map(|_| match self.below(100) {
                0 => f32::INFINITY,
                1..=3 => f32::NAN,
                x => LEVELS[(x % 6) as usize],
            })
            .collect()
    }
    fn normal_row(&mut self, n: usize, sd: f64) -> Vec<f32> {
        (0..n).map(|_| self.normal(sd)).collect()
    }
    /// Uniform in `(0, 1)`.
    fn open_unit(&mut self) -> f64 {
        ((self.u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    /// Normal with standard deviation `sd` (Box-Muller).
    fn gauss(&mut self, sd: f64) -> f32 {
        let r = (-2.0 * self.open_unit().ln()).sqrt();
        (sd * r * (std::f64::consts::TAU * self.open_unit()).cos()) as f32
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

// ---------------------------------------------------------------- oracles

/// The engine's greedy rule, `argmax_take` in `crates/gpu/src/elem.rs`, as a
/// sequential fold from its `(-inf, 0)` sentinel: the kernel's result is this
/// fold's for any reduction tree, because the rule is a total order.
fn engine_argmax(row: &[f32]) -> u32 {
    let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
    for (i, &v) in (0u32..).zip(row) {
        if v > bv || (v == bv && i < bi) {
            bi = i;
            bv = v;
        }
    }
    bi
}

/// Finite row, best first: logit descending, id ascending.
fn best_first(row: &[f32]) -> Vec<(u32, f32)> {
    let mut v: Vec<(u32, f32)> = (0u32..).zip(row.iter().copied()).collect();
    v.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    v
}

fn ids(s: &Sampler) -> Vec<u32> {
    s.candidates().iter().map(|c| c.id).collect()
}

fn sampler(p: SamplerParams) -> Sampler {
    Sampler::new(p).expect("the gate's parameters are valid")
}

/// Temperature 1 with every filter off: the draw is the plain softmax.
fn open(seed: u64) -> SamplerParams {
    SamplerParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        repeat_last_n: 0,
        seed,
    }
}

// ---------------------------------------------------------------- gates

#[test]
fn greedy_is_the_engine_argmax() {
    let mut g = Gen(1);
    let mut s = sampler(SamplerParams::greedy());
    // A negative temperature is greedy too, as in the reference; a window of
    // recent tokens with penalty 1 penalizes nothing.
    let mut s_neg = sampler(SamplerParams {
        temperature: -1.0,
        repeat_last_n: 64,
        ..SamplerParams::greedy()
    });
    let recent: Vec<u32> = (0..64).collect();
    for r in 0..1000 {
        let n = 1 + g.below(512) as usize;
        let row = if r % 2 == 0 {
            g.nasty_row(n)
        } else {
            g.normal_row(n, 3.0)
        };
        let want = engine_argmax(&row);
        assert_eq!(s.sample(&row, &[]), want, "row {r} (n = {n})");
        assert_eq!(
            s_neg.sample(&row, &recent),
            want,
            "row {r} (n = {n}), temperature -1"
        );
    }
    assert_eq!(
        s.sample(&[], &[]),
        0,
        "an empty row answers the sentinel's 0"
    );
    println!("greedy: 1000 rows, engine argmax on every one");
}

#[test]
fn top_k_one_is_greedy() {
    let mut g = Gen(2);
    let mut s = sampler(SamplerParams {
        top_k: 1,
        ..SamplerParams::default()
    });
    for r in 0..1000 {
        let n = 1 + g.below(512) as usize;
        let row = if r % 2 == 0 {
            g.nasty_row(n)
        } else {
            g.normal_row(n, 3.0)
        };
        assert_eq!(
            s.sample(&row, &[]),
            engine_argmax(&row),
            "row {r} (n = {n})"
        );
    }
    println!("top_k = 1: 1000 rows, engine argmax on every one");
}

#[test]
fn draws_follow_the_softmax() {
    const DRAWS: u32 = 100_000;
    let row = [1.0f32, 0.5, 0.0, -0.5, -2.0];
    let e: Vec<f64> = row.iter().map(|&l| f64::from(l).exp()).collect();
    let z: f64 = e.iter().sum();
    let mut s = sampler(open(7));
    let mut hits = [0u32; 5];
    for _ in 0..DRAWS {
        hits[s.sample(&row, &[]) as usize] += 1;
    }
    let mut worst = 0.0f64;
    for (i, (&h, &w)) in hits.iter().zip(&e).enumerate() {
        let (f, p) = (f64::from(h) / f64::from(DRAWS), w / z);
        println!(
            "softmax: id {i} p = {p:.5} freq = {f:.5} |diff| = {:.5}",
            (f - p).abs()
        );
        worst = worst.max((f - p).abs());
    }
    assert!(
        worst < 0.01,
        "worst |freq - p| = {worst} over {DRAWS} draws"
    );
}

#[test]
fn top_p_keeps_the_smallest_prefix_reaching_p() {
    let mut g = Gen(3);
    let (mut checked, mut ambiguous) = (0, 0);
    let (mut top_p_shorter, mut min_p_shorter) = (0, 0);
    for r in 0..1000 {
        let n = 2 + g.below(300) as usize;
        let row = g.normal_row(n, 2.0);
        let top_p = g.unit() as f32;
        // A third of the rows run top-k first: top-p then normalizes over its survivors.
        let top_k = if r % 3 == 0 {
            1 + g.below(n as u64) as u32
        } else {
            0
        };
        // Another third run min-p after top-p, as the chain does: the answer is the
        // shorter of the two best-first cuts, top-p's mass still normalized over all of
        // the top-k survivors.
        let min_p = if r % 3 == 1 {
            (g.unit() as f32).max(1e-3)
        } else {
            0.0
        };
        let mut sorted = best_first(&row);
        if top_k > 0 {
            sorted.truncate(top_k as usize);
        }
        let max = sorted[0].1;
        let w: Vec<f64> = sorted
            .iter()
            .map(|&(_, l)| f64::from(l - max).exp())
            .collect();
        let z: f64 = w.iter().sum();
        let (mut mass, mut keep, mut margin) = (0.0, sorted.len(), f64::INFINITY);
        for (i, wi) in w.iter().enumerate() {
            mass += wi / z;
            margin = margin.min((mass - f64::from(top_p)).abs());
            if mass >= f64::from(top_p) && keep == sorted.len() {
                keep = i + 1;
            }
        }
        if margin < 1e-9 {
            ambiguous += 1;
            continue;
        }
        if min_p > 0.0 {
            let floor = max + min_p.ln();
            let m = sorted.iter().take_while(|c| c.1 >= floor).count();
            if m < keep {
                min_p_shorter += 1;
            } else {
                top_p_shorter += 1;
            }
            keep = keep.min(m);
        }
        let want: Vec<u32> = sorted[..keep].iter().map(|&(id, _)| id).collect();
        let mut s = sampler(SamplerParams {
            top_k,
            top_p,
            min_p,
            ..open(r)
        });
        let _ = s.sample(&row, &[]);
        assert_eq!(
            ids(&s),
            want,
            "row {r} (n = {n}, top_k = {top_k}, top_p = {top_p}, min_p = {min_p})"
        );
        checked += 1;
    }
    println!("top-p: {checked} rows exact, {ambiguous} skipped within 1e-9 of a boundary");
    println!(
        "top-p + min-p: top-p cut shorter or equal on {top_p_shorter} rows, min-p on {min_p_shorter}"
    );
    assert!(
        top_p_shorter > 0 && min_p_shorter > 0,
        "the combined rows must exercise both cuts"
    );
    assert!(checked >= 990, "only {checked} rows were unambiguous");
}

#[test]
fn min_p_drops_below_its_fraction_of_the_max() {
    let mut g = Gen(4);
    let mut near = 0;
    for r in 0..1000 {
        let n = 2 + g.below(300) as usize;
        let row = g.normal_row(n, 2.0);
        let min_p = (g.unit() as f32).max(1e-3);
        let sorted = best_first(&row);
        let max = sorted[0].1;
        // The reference's cut, in its own arithmetic: logit >= max + ln(min_p), in f32.
        let floor = max + min_p.ln();
        let want: Vec<u32> = sorted
            .iter()
            .filter(|c| c.1 >= floor)
            .map(|c| c.0)
            .collect();
        let mut s = sampler(SamplerParams { min_p, ..open(r) });
        let _ = s.sample(&row, &[]);
        let got = ids(&s);
        assert_eq!(got, want, "row {r} (n = {n}, min_p = {min_p})");
        // And the rule it stands for: p_i >= min_p · p_max, away from rounding.
        for &(id, l) in &sorted {
            let ratio = f64::from(l - max).exp() / f64::from(min_p);
            let kept = got.contains(&id);
            if (ratio - 1.0).abs() < 1e-5 {
                near += 1;
            } else {
                assert_eq!(
                    kept,
                    ratio > 1.0,
                    "row {r} id {id}: p/p_max = {ratio} · min_p"
                );
            }
        }
    }
    println!(
        "min-p: 1000 rows exact; {near} candidates within 1e-5 of the cut not judged by ratio"
    );
}

/// llama.cpp's penalty on one logit (`src/llama-sampler.cpp:2970-2976`,
/// `llama_sampler_penalties_apply`; ik's `llama_sample_repetition_penalties_impl`
/// is the same): a token counted `count > 0` times in the window has its
/// logit divided by `r` when positive, multiplied by it otherwise, then
/// `float(count) * freq + float(count > 0) * present` subtracted, in `f32`.
fn llama_penalty(logit: f32, count: u32, r: f32, freq: f32, present: f32) -> f32 {
    if count == 0 {
        return logit;
    }
    let l = if logit <= 0.0 { logit * r } else { logit / r };
    l - (count as f32 * freq + f32::from(u8::from(count > 0)) * present)
}

#[test]
fn penalties_follow_the_reference_formula() {
    const R: f32 = 1.3;
    const FREQ: f32 = 0.37;
    const PRESENT: f32 = 1.5;
    let row: [f32; 16] = [
        2.0, -1.5, 0.0, 3.0, -0.25, 1.0, 0.75, -3.0, 0.1, -0.1, 4.0, -4.0, 0.5, -0.5, 1.5, -2.5,
    ];
    // Window = the last 9: [3, 1, 3, 1000, 2, 3, 0, 5, 1]. Token 4 is older
    // than the window; 3 occurs three times and 1 twice, so frequency and
    // presence part; 1000 is past the vocabulary.
    let recent = [4u32, 3, 1, 3, 1000, 2, 3, 0, 5, 1];
    let count = |id: u32| recent[1..].iter().filter(|&&t| t == id).count() as u32;
    let arms = [
        ("repeat", R, 0.0, 0.0),
        ("frequency", 1.0, FREQ, 0.0),
        ("presence", 1.0, 0.0, PRESENT),
        ("all three", R, FREQ, PRESENT),
        ("negative frequency and presence", R, -FREQ, -PRESENT),
    ];
    for (name, r, freq, present) in arms {
        let mut s = sampler(SamplerParams {
            repeat_penalty: r,
            frequency_penalty: freq,
            presence_penalty: present,
            repeat_last_n: 9,
            ..open(0)
        });
        let _ = s.sample(&row, &recent);
        assert_eq!(s.candidates().len(), row.len(), "{name}");
        for c in s.candidates() {
            let want = llama_penalty(row[c.id as usize], count(c.id), r, freq, present);
            assert_eq!(
                c.logit.to_bits(),
                (want + 0.0).to_bits(),
                "{name}: id {} (count {})",
                c.id,
                count(c.id)
            );
        }
    }
    // Greedy sees the penalized logits, as llama.cpp's temperature 0 takes the
    // argmax after the penalties (`src/llama-sampler.cpp:270-285`).
    let mut greedy = sampler(SamplerParams {
        repeat_penalty: 2.0,
        repeat_last_n: 1,
        ..SamplerParams::greedy()
    });
    assert_eq!(greedy.sample(&[5.0, 4.0], &[0]), 1, "5 / 2 < 4");
    assert_eq!(greedy.sample(&[-1.0, -1.5], &[0]), 1, "-1 · 2 < -1.5");
    assert_eq!(
        greedy.sample(&[5.0, 4.0], &[1]),
        0,
        "the other token penalized"
    );
    let mut present = sampler(SamplerParams {
        presence_penalty: 1.5,
        repeat_last_n: 64,
        ..SamplerParams::greedy()
    });
    assert_eq!(present.sample(&[5.0, 4.0], &[0, 0]), 1, "5 - 1.5 < 4");
    println!(
        "penalty: 16 logits bit-exact on five arms, greedy flips on both signs and on presence"
    );
}

#[test]
fn window_is_the_last_repeat_last_n_tokens() {
    // Each id occurs once, at its own position: id i is `recent[i]`, so the
    // window of the last n holds exactly the ids `len - n .. len`.
    const LEN: u32 = 12;
    let recent: Vec<u32> = (0..LEN).collect();
    let row = [1.0f32; LEN as usize];
    for n in [0usize, 1, 5, 11, 12, 13, 1 << 20] {
        let mut s = sampler(SamplerParams {
            presence_penalty: 1.0,
            repeat_last_n: n,
            ..open(0)
        });
        let _ = s.sample(&row, &recent);
        let first_in = LEN.saturating_sub(u32::try_from(n).unwrap_or(u32::MAX));
        for c in s.candidates() {
            let want = if c.id >= first_in { 0.0 } else { 1.0 };
            assert_eq!(c.logit, want, "repeat_last_n {n}: id {}", c.id);
        }
    }
    println!(
        "window: the last n of {LEN} tokens for n in 0, 1, 5, 11, 12, 13 and 2^20 (a window past the history covers all of it)"
    );
}

#[test]
fn neutral_penalties_leave_the_chain_unchanged() {
    let mut g = Gen(9);
    // `-0.0 == 0.0`, so a negative zero is neutral too, as in llama.cpp.
    let neutral = SamplerParams {
        repeat_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: -0.0,
        repeat_last_n: 64,
        ..SamplerParams::default()
    };
    let greedy = SamplerParams {
        presence_penalty: -0.0,
        repeat_last_n: 64,
        ..SamplerParams::greedy()
    };
    for p in [neutral, greedy] {
        let mut a = sampler(p);
        let mut b = sampler(SamplerParams {
            repeat_last_n: 0,
            ..p
        });
        for r in 0..1000 {
            let n = 1 + g.below(512) as usize;
            let row = if r % 2 == 0 {
                g.nasty_row(n)
            } else {
                g.normal_row(n, 3.0)
            };
            let recent: Vec<u32> = (0..64).map(|_| g.below(n as u64) as u32).collect();
            assert_eq!(a.sample(&row, &recent), b.sample(&row, &[]), "row {r}");
            let bits = |s: &Sampler| -> Vec<(u32, u32)> {
                s.candidates()
                    .iter()
                    .map(|c| (c.id, c.logit.to_bits()))
                    .collect()
            };
            assert_eq!(bits(&a), bits(&b), "row {r}: candidates");
        }
    }
    println!(
        "neutral: 2000 rows, the same ids and candidate bits with and without a 64-token window"
    );
}

#[test]
fn same_seed_same_tokens() {
    const STEPS: usize = 10_000;
    let mut g = Gen(5);
    let rows: Vec<Vec<f32>> = (0..16).map(|_| g.normal_row(1000, 2.0)).collect();
    let p = SamplerParams {
        top_p: 0.95,
        ..open(42)
    };
    let run = |p: SamplerParams| -> Vec<u32> {
        let mut s = sampler(p);
        (0..STEPS)
            .map(|i| s.sample(&rows[i % rows.len()], &[]))
            .collect()
    };
    let (a, b) = (run(p), run(p));
    assert_eq!(a, b, "two samplers with seed 42 disagree");
    let other = run(SamplerParams { seed: 43, ..p });
    let differ = a.iter().zip(&other).filter(|(x, y)| x != y).count();
    let mut distinct = a.clone();
    distinct.sort_unstable();
    distinct.dedup();
    println!(
        "seed: {STEPS} ids identical for seed 42 twice, {} distinct; seed 43 differs at {differ}",
        distinct.len()
    );
    assert!(distinct.len() > 100, "the draw is degenerate");
    assert!(differ > STEPS / 2, "the seed barely matters");
}

#[test]
fn no_allocation_after_the_first_call() {
    const N_VOCAB: usize = 129_280;
    const CALLS_PER_ARM: usize = 20;
    let mut g = Gen(6);
    let rows: Vec<Vec<f32>> = (0..4).map(|_| g.normal_row(N_VOCAB, 3.0)).collect();
    let recent: Vec<u32> = (0..256).map(|_| g.below(N_VOCAB as u64) as u32).collect();
    let penalty = SamplerParams {
        repeat_penalty: 1.1,
        ..SamplerParams::default()
    };
    let arms = [
        ("default chain", SamplerParams::default()),
        ("penalty + default chain", penalty),
        (
            "penalty, top-p over the whole vocabulary",
            SamplerParams {
                top_k: 0,
                min_p: 0.0,
                ..penalty
            },
        ),
        ("greedy", SamplerParams::greedy()),
        (
            "greedy + penalty",
            SamplerParams {
                repeat_penalty: 1.1,
                repeat_last_n: 64,
                ..SamplerParams::greedy()
            },
        ),
        (
            "frequency + presence + default chain",
            SamplerParams {
                frequency_penalty: 0.5,
                presence_penalty: 1.5,
                ..SamplerParams::default()
            },
        ),
    ];
    for (name, p) in arms {
        let mut s = sampler(p);
        let _ = s.sample(&rows[0], &recent);
        CALLS.store(0, Relaxed);
        COUNTING.with(|c| c.set(true));
        for i in 0..CALLS_PER_ARM {
            let _ = s.sample(&rows[i % rows.len()], &recent);
        }
        COUNTING.with(|c| c.set(false));
        let calls = CALLS.load(Relaxed);
        println!("alloc: {name}: {calls} allocator calls over {CALLS_PER_ARM} warm calls");
        assert_eq!(calls, 0, "{name}");
    }
}

// ---------------------------------------------------------------- the chain against its frozen form

/// The chain as it ran before the top-k scan: every row materialized as candidates,
/// penalized in place, one selection for top-k, then top-p, min-p and the draw. It
/// shares no code with the crate, so a change to the crate's chain cannot move it.
mod frozen {
    use sampler::SamplerParams;
    use std::cmp::Ordering;

    #[derive(Clone, Copy)]
    pub struct Cand {
        pub id: u32,
        pub logit: f32,
    }

    struct Rng {
        s: [u64; 4],
    }

    impl Rng {
        fn seeded(seed: u64) -> Self {
            let mut z = seed;
            let mut next = || {
                z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut x = z;
                x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^ (x >> 31)
            };
            Self {
                s: [next(), next(), next(), next()],
            }
        }

        fn next_f64(&mut self) -> f64 {
            let s = &mut self.s;
            let out = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
            let t = s[1] << 17;
            s[2] ^= s[0];
            s[3] ^= s[1];
            s[1] ^= s[2];
            s[0] ^= s[3];
            s[2] ^= t;
            s[3] = s[3].rotate_left(45);
            (out >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        }
    }

    fn ids(logits: &[f32]) -> impl Iterator<Item = (u32, f32)> + '_ {
        (0..=u32::MAX).zip(logits.iter().copied())
    }

    fn argmax(xs: impl Iterator<Item = (u32, f32)>) -> (u32, f32) {
        let (mut best_i, mut best_v) = (0u32, f32::NEG_INFINITY);
        for (i, v) in xs {
            if v > best_v || (v == best_v && i < best_i) {
                best_i = i;
                best_v = v;
            }
        }
        (best_i, best_v)
    }

    fn sanitize(v: f32) -> f32 {
        if v.is_nan() {
            f32::NEG_INFINITY
        } else {
            v + 0.0
        }
    }

    fn best_first(a: &Cand, b: &Cand) -> Ordering {
        b.logit.total_cmp(&a.logit).then(a.id.cmp(&b.id))
    }

    fn weight(logit: f32, max: f32, inv_t: f64) -> f64 {
        (f64::from(logit - max) * inv_t).exp()
    }

    pub struct Chain {
        p: SamplerParams,
        rng: Rng,
        /// What the last call drew from.
        pub cand: Vec<Cand>,
        cum: Vec<f64>,
        count: Vec<u32>,
    }

    impl Chain {
        pub fn new(p: SamplerParams) -> Self {
            Self {
                p,
                rng: Rng::seeded(p.seed),
                cand: Vec::new(),
                cum: Vec::new(),
                count: Vec::new(),
            }
        }

        pub fn sample(&mut self, logits: &[f32], recent: &[u32]) -> u32 {
            let p = self.p;
            self.cand.clear();
            if logits.is_empty() {
                return 0;
            }
            let window = &recent[recent.len().saturating_sub(p.repeat_last_n)..];
            let penalize = !window.is_empty()
                && (p.repeat_penalty != 1.0
                    || p.frequency_penalty != 0.0
                    || p.presence_penalty != 0.0);
            if p.temperature <= 0.0 && !penalize {
                let (id, logit) = argmax(ids(logits));
                self.cand.push(Cand { id, logit });
                return id;
            }
            self.cand.extend(ids(logits).map(|(id, v)| Cand {
                id,
                logit: sanitize(v),
            }));
            if penalize {
                self.penalize(window);
            }
            if p.temperature <= 0.0 {
                let (id, logit) = argmax(self.cand.iter().map(|c| (c.id, c.logit)));
                self.cand.clear();
                self.cand.push(Cand { id, logit });
                return id;
            }
            let k = p.top_k as usize;
            if k > 0 && k < self.cand.len() {
                self.cand.select_nth_unstable_by(k - 1, best_first);
                self.cand.truncate(k);
            }
            self.top_p_min_p();
            self.draw()
        }

        fn penalize(&mut self, window: &[u32]) {
            let p = self.p;
            let n = self.cand.len();
            if self.count.len() < n {
                self.count.resize(n, 0);
            }
            for &t in window {
                let i = t as usize;
                if i < n {
                    self.count[i] += 1;
                }
            }
            for &t in window {
                let i = t as usize;
                if i < n && self.count[i] > 0 {
                    let k = self.count[i] as f32;
                    self.count[i] = 0;
                    let c = &mut self.cand[i];
                    let l = c.logit;
                    let l = if l <= 0.0 {
                        l * p.repeat_penalty
                    } else {
                        l / p.repeat_penalty
                    };
                    c.logit = sanitize(l - (k * p.frequency_penalty + p.presence_penalty));
                }
            }
        }

        fn top_p_min_p(&mut self) {
            let p = self.p;
            let max = self
                .cand
                .iter()
                .fold(f32::NEG_INFINITY, |m, c| m.max(c.logit));
            if !max.is_finite() {
                let best = self.cand.iter().copied().min_by(best_first);
                self.cand.clear();
                self.cand.extend(best);
                return;
            }
            let top_p = p.top_p < 1.0;
            let norm: f64 = if top_p {
                self.cand.iter().map(|c| weight(c.logit, max, 1.0)).sum()
            } else {
                0.0
            };
            if p.min_p > 0.0 {
                let floor = max + p.min_p.ln();
                self.cand.retain(|c| c.logit >= floor);
            }
            self.cand.sort_unstable_by(best_first);
            if top_p {
                let target = f64::from(p.top_p);
                let mut mass = 0.0;
                let mut keep = self.cand.len();
                for (i, c) in self.cand.iter().enumerate() {
                    mass += weight(c.logit, max, 1.0) / norm;
                    if mass >= target {
                        keep = i + 1;
                        break;
                    }
                }
                self.cand.truncate(keep);
            }
        }

        fn draw(&mut self) -> u32 {
            let [first, rest @ ..] = self.cand.as_slice() else {
                return 0;
            };
            if rest.is_empty() {
                return first.id;
            }
            let max = first.logit;
            let inv_t = 1.0 / f64::from(self.p.temperature);
            self.cum.clear();
            let mut total = 0.0;
            for c in &self.cand {
                total += weight(c.logit, max, inv_t);
                self.cum.push(total);
            }
            let u = self.rng.next_f64() * total;
            let i = self
                .cum
                .partition_point(|&c| c <= u)
                .min(self.cand.len() - 1);
            self.cand[i].id
        }
    }
}

/// The rows `examples/sample_bench.rs` times, built the same way from the same seeds.
fn bench_row(peaked: bool, n: usize, seed: u64) -> Vec<f32> {
    let mut g = Gen(seed);
    if peaked {
        let mut row: Vec<f32> = (0..n).map(|_| g.gauss(2.0)).collect();
        let body_max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let at = (g.u64() % n as u64) as usize;
        row[at] = body_max + 8.0 + 4.0 * g.open_unit() as f32;
        row
    } else {
        (0..n).map(|_| g.gauss(1.0)).collect()
    }
}

/// A row of one of the shapes the chain's order rules are about: normal, with ties,
/// signed zeros, infinities and NaN, on few levels (a tie at every boundary), constant,
/// and ascending or descending (the top-k scan's buffer fills over and over).
fn shaped_row(g: &mut Gen, n: usize, shape: u64) -> Vec<f32> {
    match shape {
        0 | 1 => {
            let sd = 0.5 + 3.0 * g.unit();
            g.normal_row(n, sd)
        }
        2 => g.nasty_row(n),
        3 => (0..n).map(|_| g.below(8) as f32 * 0.5 - 1.0).collect(),
        4 => vec![1.5; n],
        5 => (0..n).map(|i| (i / 7) as f32 * 0.01).collect(),
        _ => (0..n).rev().map(|i| (i / 3) as f32 * 0.01).collect(),
    }
}

/// A row whose logits above the floor are the maximum and a few entries sitting on
/// `max + ln(min_p)` in `f32`, the way the chain computes it, with the whole body below.
fn floor_row(g: &mut Gen, n: usize, min_p: f32) -> Vec<f32> {
    let max = 5.0f32;
    let floor = max + min_p.ln();
    let mut row: Vec<f32> = (0..n)
        .map(|_| floor - 0.25 - 3.0 * g.unit() as f32)
        .collect();
    let mut at: Vec<usize> = Vec::new();
    while at.len() < 4 + g.below(4) as usize {
        let i = g.below(n as u64) as usize;
        if !at.contains(&i) {
            at.push(i);
        }
    }
    row[at[0]] = max;
    for &i in &at[1..] {
        row[i] = floor;
    }
    row
}

fn random_params(g: &mut Gen, n: usize) -> SamplerParams {
    let n32 = n as u32;
    SamplerParams {
        temperature: if g.below(10) == 0 {
            0.0
        } else {
            0.2 + 1.8 * g.unit() as f32
        },
        top_k: match g.below(8) {
            0 => 0,
            1 => 1,
            2 => 2 + g.below(4) as u32,
            3 => 40,
            4 => n32,
            5 => n32 + 1,
            6 => 1 + g.below(n as u64) as u32,
            _ => 20,
        },
        top_p: match g.below(6) {
            0 => 1.0,
            1 => 0.95,
            2 => 0.9,
            3 => 0.5,
            4 => 0.0,
            _ => g.unit() as f32,
        },
        min_p: match g.below(6) {
            0 => 0.0,
            1 => 0.05,
            2 => 0.3,
            3 => 1.0,
            4 => 0.01,
            _ => g.unit() as f32,
        },
        repeat_penalty: g.pick(&[1.0, 1.0, 1.1, 1.3, 0.8, 2.0]),
        frequency_penalty: g.pick(&[0.0, 0.0, 0.3]),
        presence_penalty: g.pick(&[0.0, 0.0, 0.7]),
        repeat_last_n: g.pick(&[0, 4, 64, 1000]),
        seed: g.u64(),
    }
}

/// Each call of `s` answers what the frozen chain answers: the same id and the same
/// survivors, bit for bit and in the same order.
fn same_as_frozen(
    s: &mut Sampler,
    f: &mut frozen::Chain,
    row: &[f32],
    recent: &[u32],
    what: impl Fn() -> String,
) {
    let (got, want) = (s.sample(row, recent), f.sample(row, recent));
    let kept: Vec<(u32, u32)> = s
        .candidates()
        .iter()
        .map(|c| (c.id, c.logit.to_bits()))
        .collect();
    let want_kept: Vec<(u32, u32)> = f.cand.iter().map(|c| (c.id, c.logit.to_bits())).collect();
    assert_eq!(kept, want_kept, "kept set: {}", what());
    assert_eq!(got, want, "drawn id: {}", what());
}

#[test]
fn the_chain_is_its_frozen_form() {
    const RANDOM_ROWS: u64 = 1500;
    const FLOOR_ROWS: u64 = 300;
    let mut g = Gen(7);
    let mut calls = 0usize;

    // The rows the bench times, at the vocabularies it times, under the bench's parameters.
    let defaults = SamplerParams::default();
    let bench_sets = [
        defaults,
        SamplerParams {
            top_p: 1.0,
            min_p: 0.0,
            ..defaults
        },
        SamplerParams {
            top_k: 0,
            ..defaults
        },
        SamplerParams {
            top_k: 0,
            top_p: 1.0,
            ..defaults
        },
    ];
    for n in [151_936, 154_880] {
        for peaked in [true, false] {
            let row = bench_row(peaked, n, 0xB100_0000);
            for (i, p) in bench_sets.iter().enumerate() {
                let (mut s, mut f) = (sampler(*p), frozen::Chain::new(*p));
                for _ in 0..3 {
                    same_as_frozen(&mut s, &mut f, &row, &[], || {
                        format!("bench row (V = {n}, peaked = {peaked}), set {i}")
                    });
                    calls += 1;
                }
            }
        }
    }

    // Random rows under random parameters, penalties included; a second row of another
    // length goes through the same sampler to carry its scratch over.
    for r in 0..RANDOM_ROWS {
        let n = match g.below(4) {
            0 => 1 + g.below(40) as usize,
            1 => 1 + g.below(1500) as usize,
            _ => 1 + g.below(6000) as usize,
        };
        let p = random_params(&mut g, n);
        let shape = g.below(7);
        let row = shaped_row(&mut g, n, shape);
        let n2 = 1 + g.below(6000) as usize;
        let shape2 = g.below(7);
        let row2 = shaped_row(&mut g, n2, shape2);
        let recent: Vec<u32> = (0..g.below(200))
            .map(|_| g.below(n as u64 + n as u64 / 4 + 1) as u32)
            .collect();
        let (mut s, mut f) = (sampler(p), frozen::Chain::new(p));
        same_as_frozen(&mut s, &mut f, &row, &recent, || {
            format!("random row {r} (n = {n}, shape {shape}) under {p:?}")
        });
        same_as_frozen(&mut s, &mut f, &row2, &recent, || {
            format!("random row {r}, second row (n = {n2}, shape {shape2}) under {p:?}")
        });
        calls += 2;
    }

    // Entries exactly on the min-p floor and the whole body below it: they stay under
    // `>=` and go under `>`, with top-k keeping them and top-p keeping everything.
    for r in 0..FLOOR_ROWS {
        let n = 50 + g.below(2000) as usize;
        let min_p = g.pick(&[0.05, 0.1, 0.3]);
        let row = floor_row(&mut g, n, min_p);
        let p = SamplerParams {
            temperature: 0.2 + 1.8 * g.unit() as f32,
            top_k: g.pick(&[0, 40]),
            top_p: 1.0,
            min_p,
            repeat_last_n: 0,
            seed: g.u64(),
            ..defaults
        };
        let (mut s, mut f) = (sampler(p), frozen::Chain::new(p));
        same_as_frozen(&mut s, &mut f, &row, &[], || {
            format!("floor row {r} (n = {n}) under {p:?}")
        });
        assert!(
            s.candidates().len() > 3,
            "floor row {r}: the entries on the floor were dropped"
        );
        calls += 1;
    }
    println!(
        "frozen chain: {calls} calls over bench rows, {RANDOM_ROWS} random rows (penalties, ties, \
         NaN, ascending and descending) and {FLOOR_ROWS} floor rows: same kept set and id on every one"
    );
}
