//! Gate for the sampler: every stage of the chain against an oracle written
//! from the reference's definition, the draw against its softmax, the seed
//! against itself, and the allocator against zero.
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
