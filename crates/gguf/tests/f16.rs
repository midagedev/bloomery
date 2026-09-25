//! The engine's one f32 → f16 rounding, `gguf::quant::f32_to_f16_bits`, against
//! the CPU's own IEEE converter (F16C `vcvtps2ph`, round to nearest even) over
//! every f32 bit pattern: every input that is not a NaN rounds to the
//! converter's bits, and every NaN comes out a NaN of its own sign. The first
//! half is what every KV-cache write and tensor-core query row depends on, on
//! the host and on the card, so this enumeration is its proof; the second is
//! the refusal contract — a NaN is never turned into an infinity or a number.

use gguf::quant::f32_to_f16_bits;

/// Threads the 2^32 patterns are split over, at most.
const MAX_THREADS: usize = 32;

/// f32 NaN patterns: an all-ones exponent with a nonzero mantissa, under
/// either sign.
const NAN_PATTERNS: u64 = 2 * ((1 << 23) - 1);

/// The CPU's round-to-nearest-even conversion of `x` to f16 bits.
///
/// # Safety
/// F16C must be available on the CPU; being `#[inline(always)]`, it takes the
/// feature from its caller's.
#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn f16c_rne(x: f32) -> u16 {
    use std::arch::x86_64::{
        _MM_FROUND_TO_NEAREST_INT, _mm_cvtps_ph, _mm_cvtsi128_si32, _mm_set_ss,
    };
    // SAFETY: register-only intrinsics, no memory access; F16C by this fn's
    // contract.
    unsafe { _mm_cvtsi128_si32(_mm_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(_mm_set_ss(x))) as u16 }
}

/// What one chunk of patterns found.
#[derive(Default)]
struct Tally {
    checked: u64,
    nans: u64,
    /// Non-NaN inputs whose bits differ from the converter's, and the first.
    finite_bad: u64,
    first_finite_bad: Option<u32>,
    /// NaN inputs whose output is not a NaN of the input's sign, and the first.
    nan_bad: u64,
    first_nan_bad: Option<u32>,
}

impl Tally {
    fn add(&mut self, o: Tally) {
        self.checked += o.checked;
        self.nans += o.nans;
        self.finite_bad += o.finite_bad;
        self.nan_bad += o.nan_bad;
        self.first_finite_bad = min_some(self.first_finite_bad, o.first_finite_bad);
        self.first_nan_bad = min_some(self.first_nan_bad, o.first_nan_bad);
    }
}

fn min_some(a: Option<u32>, b: Option<u32>) -> Option<u32> {
    match (a, b) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (a, None) => a,
        (None, b) => b,
    }
}

/// Patterns `lo .. hi` (as u64, so the last chunk reaches `u32::MAX`).
///
/// # Safety
/// F16C must be available on the CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "f16c")]
unsafe fn scan(lo: u64, hi: u64) -> Tally {
    let mut t = Tally::default();
    for p in lo..hi {
        let bits = p as u32;
        let x = f32::from_bits(bits);
        let got = f32_to_f16_bits(x);
        t.checked += 1;
        if x.is_nan() {
            t.nans += 1;
            let is_nan = got & 0x7c00 == 0x7c00 && got & 0x03ff != 0;
            let same_sign = u32::from(got >> 15) == bits >> 31;
            if !(is_nan && same_sign) {
                t.nan_bad += 1;
                t.first_nan_bad = min_some(t.first_nan_bad, Some(bits));
            }
        } else {
            // SAFETY: F16C by this fn's contract.
            let want = unsafe { f16c_rne(x) };
            if got != want {
                t.finite_bad += 1;
                t.first_finite_bad = min_some(t.first_finite_bad, Some(bits));
            }
        }
    }
    t
}

#[test]
#[ignore = "hw: needs an x86-64 CPU with F16C; `just gate-1-1` runs it on the box"]
fn hw_f32_to_f16_bits_is_ieee_rne_on_every_input() {
    #[cfg(not(target_arch = "x86_64"))]
    panic!("the enumeration's reference is F16C: it runs on x86-64 only");
    #[cfg(target_arch = "x86_64")]
    {
        assert!(
            std::arch::is_x86_feature_detected!("f16c"),
            "the enumeration's reference is F16C, and this CPU has none"
        );
        let threads = std::thread::available_parallelism().map_or(8, |n| n.get().min(MAX_THREADS));
        let total: u64 = 1 << 32;
        let per = total.div_ceil(threads as u64);
        let t0 = std::time::Instant::now();
        let mut all = Tally::default();
        std::thread::scope(|s| {
            let handles: Vec<_> = (0..threads as u64)
                .map(|i| {
                    let (lo, hi) = (i * per, ((i + 1) * per).min(total));
                    // SAFETY: F16C was detected above.
                    s.spawn(move || unsafe { scan(lo, hi) })
                })
                .collect();
            for h in handles {
                all.add(h.join().expect("a scan thread panicked"));
            }
        });
        let show = |b: Option<u32>| match b {
            Some(b) => {
                let x = f32::from_bits(b);
                // SAFETY: F16C was detected above.
                let want = unsafe { f16c_rne(x) };
                format!(
                    "0x{b:08x} ({x:e}) -> 0x{:04x}, F16C 0x{want:04x}",
                    f32_to_f16_bits(x)
                )
            }
            None => "none".to_owned(),
        };
        println!(
            "f32_to_f16_bits: {} inputs on {threads} threads in {:.1} s, {} NaN; non-NaN inputs off \
             F16C's round-to-nearest-even bits {} (first {}); NaN inputs not a NaN of their sign {} \
             (first {})",
            all.checked,
            t0.elapsed().as_secs_f64(),
            all.nans,
            all.finite_bad,
            show(all.first_finite_bad),
            all.nan_bad,
            show(all.first_nan_bad)
        );
        assert_eq!(all.checked, total, "every f32 pattern is checked once");
        assert_eq!(all.nans, NAN_PATTERNS, "every NaN pattern is met");
        assert_eq!(
            all.finite_bad, 0,
            "a non-NaN input left IEEE round-to-nearest-even"
        );
        assert_eq!(
            all.nan_bad, 0,
            "a NaN input came out as something other than a NaN of its sign"
        );
    }
}
