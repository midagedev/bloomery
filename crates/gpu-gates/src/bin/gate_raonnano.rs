//! GPU gate for the Nano dialect's own card pieces (`bloomery_gpu::arch::raon1`),
//! model-less at small widths, each against a reference written from the file
//! dialect's own authorities — conversion/raon1_nano.py's `_fold_mhc` (the
//! chart the conversion bakes) and `_fold_rda` (the decay's baked
//! `exp(A_log)`), and raon1nano.cpp's `build_hc_pre` (the folded route's
//! order of scales) and the per-layer input's graph — with an in-gate
//! separation that a reference or a card run missing the new clause fails.
//! Inputs are seeded; nothing is read from a model file.
//!
//! 1. `chart`: the folded chart's route (`raon_hc_mix` + `raon_hc_route_folded`)
//!    over 1 and 5 tokens of 4·512 values against the f64 reference of the
//!    folded parametrization — the α scales folded into the selector rows,
//!    one sigmoid a pre and post pair, `e' = K·mixes`, `ρ = 1 + max(e'·r)`
//!    with no clamp (every row of `H z Hᵀ` sums to zero, so the max is at
//!    least `−e/B` at `e = 0`), `comb = e'/ρ + c0` — every route value within
//!    [`BAND`] relative; and the split dialect's own route (`raon_hc_route`)
//!    over the same streams, its selector rows unscaled and its scales the
//!    α's and η the chart was folded from, within [`BAND`] of the same
//!    reference. Two references each missing one clause — the combination
//!    constant dropped (`comb = e'/ρ`), the ρ division dropped
//!    (`comb = e' + c0`) — lie past [`SEP`] bands, and the launcher refuses
//!    by name a chart of fewer than 304 values.
//! 2. `ple`: the per-layer input's two entries against their host rules bit
//!    for bit — `raon1_ple_combine` of 5 tokens over 3 layers of 64 values
//!    (the table's `√64` scale, the `1/√2`, layer-major output) and
//!    `raon1_ple_add` of the sum into every one of 4 streams; a rerun of each
//!    bit-identical; the host rule with the table's scale at 1 (the scale
//!    dropped) and one that adds to 3 streams (a stream dropped) each differ
//!    from the card's values, and both launchers refuse by name a width of 0
//!    and a `g` shorter than the tokens' rows.
//! 3. `shape`: every entry compiles with no local depot.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raonnano", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::arch::raon1::PleKernels;
    use bloomery_gpu::arch::raon1::{
        CHART_LEN, ChartVecs, HC_MIX, HC_OUT, MhcKernels, STREAMS, SiteVecs, parts_len,
    };
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;
    use std::sync::Arc;

    /// The streams' width: 4·512 values a token, 8 whole mix pieces.
    const N: usize = 512;
    /// The chart clause's tokens.
    const TOKENS: [usize; 2] = [1, 5];
    /// The per-layer input's width and layers of the ple clause.
    const PLE: usize = 64;
    const LAYERS: usize = 3;
    /// The ple clause's tokens.
    const ROWS: usize = 5;
    /// The f32 chain against f64: the mix's 2048-value dots and the chart's
    /// 9-term products round within a few ulps a value, the sigmoid and the
    /// division once more; 2e-4 relative is a hundred of those.
    const BAND: f64 = 2e-4;
    /// The bands a run missing a clause must lie past.
    const SEP: f64 = 2e-3;

    /// A seeded xorshift64 stream: the same seed draws the same values.
    struct Rng(u64);

    impl Rng {
        fn unit(&mut self) -> f32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }

        fn range(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// The Helmert basis entry `(i, j)` of 4 streams (the conversion's own
    /// `q`, f64): `1` for `i <= j`, `−(j+1)` for `i == j + 1`, else 0, over
    /// `√((j+1)(j+2))`.
    fn helmert(i: usize, j: usize) -> f64 {
        let num = if i <= j {
            1.0
        } else if i == j + 1 {
            -((j + 1) as f64)
        } else {
            0.0
        };
        num / (((j + 1) * (j + 2)) as f64).sqrt()
    }

    /// One site's two parametrizations of the same draw: the split
    /// dialect's selector rows, scales and biases, and the folded chart the
    /// conversion bakes of them (`_fold_mhc`: `fn` rows scaled by their α,
    /// `K = γ(H⊗H)` zero on the pre and post coordinates,
    /// `r = −1/(γ·anchor)`, `c0 = (1−γ)I + γ·anchor`, `γ = σ(η)`).
    struct Site {
        /// The split dialect's unscaled selector rows, 17 of `4·N`.
        fn_rows: Vec<f32>,
        /// `α_pre`, `α_post`, `α_z` and `η`.
        scales: [f32; 4],
        base: Vec<f32>,
        /// The folded selector rows: the raw rows scaled by their α.
        folded_rows: Vec<f32>,
        /// The baked chart, 304 values.
        chart: Vec<f32>,
    }

    /// One site of `r`'s draw at magnitudes the route's products stay small.
    fn site(r: &mut Rng) -> Site {
        let k = STREAMS * N;
        let raw: Vec<f32> = (0..HC_MIX * k).map(|_| r.range(-0.05, 0.05)).collect();
        let base: Vec<f32> = (0..HC_MIX).map(|_| r.range(-0.1, 0.1)).collect();
        let [a_pre, a_post, a_z, eta] = [
            r.range(0.5, 1.5),
            r.range(0.5, 1.5),
            r.range(0.5, 1.5),
            r.range(-1.0, 1.0),
        ];
        // The folded rows: the pre rows by α_pre (rows 0..4), the post rows
        // by α_post (4..8), the chart coordinates' by α_z (8..17).
        let mut folded = raw.clone();
        for row in 0..HC_MIX {
            let a = match row {
                0..=3 => a_pre,
                4..=7 => a_post,
                _ => a_z,
            };
            for i in 0..k {
                folded[row * k + i] = raw[row * k + i] * a;
            }
        }
        // The chart, the conversion's own arithmetic in f64 rounded once.
        let gamma = 1.0 / (1.0 + (-eta as f64).exp());
        let n = STREAMS;
        let nm = n - 1;
        let mut chart = vec![0.0f64; CHART_LEN];
        let k_at = n * n;
        for (i, p) in (0..n).flat_map(|i| (0..n).map(move |p| (i, p))) {
            let row = i * n + p;
            for (a, c) in (0..nm).flat_map(|a| (0..nm).map(move |c| (a, c))) {
                chart[row * HC_MIX + 2 * n + a * nm + c] = gamma * helmert(p, a) * helmert(i, c);
            }
        }
        for i in 0..n {
            for p in 0..n {
                let at = HC_MIX * k_at + i * n + p;
                let id = f64::from(i == p);
                let anchor = 0.5 * (id + 1.0 / n as f64);
                chart[at] = -1.0 / (gamma * anchor);
                chart[CHART_LEN - k_at + i * n + p] = (1.0 - gamma) * id + gamma * anchor;
            }
        }
        Site {
            fn_rows: raw,
            scales: [a_pre, a_post, a_z, eta],
            base,
            folded_rows: folded,
            chart: chart.into_iter().map(|v| v as f32).collect(),
        }
    }

    /// `run()` — every clause; each prints its own line and returns one bool.
    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new().map_err(|e| format!("a Gpu: {e}"))?;
        let (mut results, sep_chart) = chart_clauses(&gpu)?;
        let (ple, sep_ple) = ple_clauses(&gpu)?;
        results.extend(ple);
        results.push(sep_chart);
        results.push(sep_ple);
        let failed = results.iter().filter(|&&ok| !ok).count();
        for (name, ok) in NAMES.iter().zip(results.iter()) {
            println!("gate_raonnano: {name} — {}", verdict(*ok));
        }
        println!(
            "gate_raonnano: {} clauses, {failed} failed",
            results.len()
        );
        if failed > 0 {
            return Err(checks_failed());
        }
        Ok(())
    }

    /// The clause names, in the order `run()` collects them.
    const NAMES: &[&str] = &[
        "chart: the folded route within the band of the f64 reference at 1 and 5 tokens",
        "chart: the split dialect's route of the same draw within the band",
        "chart: the launcher refuses a short chart by name",
        "ple: the combine against its host rule, bit for bit",
        "ple: the add against its host rule, bit for bit",
        "ple: the launchers refuse an empty width and short rows by name",
        "chart: a reference missing a clause lies past the separation",
        "ple: a host rule missing a clause differs from the card's values",
    ];

    /// The chart clause's launches. Returns the clause results and whether
    /// both missing-clause references lie past [`SEP`].
    fn chart_clauses(gpu: &Gpu) -> Result<(Vec<bool>, bool), GateError> {
        let ctx: Arc<_> = gpu.context().clone();
        let k = MhcKernels::load(&ctx).map_err(|e| format!("the mhc module: {e}"))?;
        let mut r = Rng(0x6e61_6e6f);
        let attn = site(&mut r);
        let ffn = site(&mut r);
        let stream = gpu.stream();
        let parts = parts_len(N).expect("4·512 values in whole pieces");
        let (mut folded_all, mut split_all) = (true, true);
        let mut sep = true;
        for m in TOKENS {
            let x: Vec<f32> = (0..m * STREAMS * N).map(|_| r.range(-1.0, 1.0)).collect();
            let mut splits = Vec::new();
            let mut folds = Vec::new();
            for s in [&attn, &ffn] {
                let (split, folded) = route_pair(gpu, &k, s, &x, m, parts)?;
                splits.extend(split);
                folds.extend(folded);
            }
            let want: Vec<f32> = [&attn, &ffn]
                .iter()
                .flat_map(|s| reference(s, &x, m))
                .collect();
            let within = |got: &[f32], band: f64| {
                got.iter().zip(&want).all(|(v, w)| {
                    ((f64::from(*v) - f64::from(*w)) / f64::from(w.abs().max(1.0))).abs() < band
                })
            };
            folded_all = folded_all && within(&folds, BAND);
            split_all = split_all && within(&splits, BAND);
            // The two missing-clause references, on the folded values.
            let no_c0: Vec<f32> = [&attn, &ffn]
                .iter()
                .flat_map(|s| reference_missing(s, &x, m, false))
                .collect();
            let no_rho: Vec<f32> = [&attn, &ffn]
                .iter()
                .flat_map(|s| reference_missing(s, &x, m, true))
                .collect();
            let past = |bad: &[f32]| {
                bad.iter()
                    .zip(&want)
                    .filter(|(v, w)| v.to_bits() != w.to_bits())
                    .count()
                    > 0
                    && bad.iter().zip(&want).any(|(v, w)| {
                        ((f64::from(*v) - f64::from(*w)) / f64::from(w.abs().max(1.0))).abs() >= SEP
                    })
            };
            sep = sep && past(&no_c0) && past(&no_rho);
        }
        // The launcher refuses a short chart by name.
        let short = {
            let mut s = attn.chart.clone();
            s.truncate(CHART_LEN - 1);
            s
        };
        let y = DeviceBuffer::<f32>::zeroed(stream, STREAMS * N).unwrap_or_else(|e| panic!("{e}"));
        let mut part = DeviceBuffer::<f32>::zeroed(stream, parts).unwrap_or_else(|e| panic!("{e}"));
        let mut hc = DeviceBuffer::<f32>::zeroed(stream, HC_OUT).unwrap_or_else(|e| panic!("{e}"));
        let rows = DeviceBuffer::<f32>::from_host(stream, &attn.folded_rows)
            .unwrap_or_else(|e| panic!("{e}"));
        let base =
            DeviceBuffer::<f32>::from_host(stream, &attn.base).unwrap_or_else(|e| panic!("{e}"));
        let chart =
            DeviceBuffer::<f32>::from_host(stream, &short).unwrap_or_else(|e| panic!("{e}"));
        let refused = k
            .enqueue_route_folded(
                stream,
                &ChartVecs::new((&rows, &base, &chart)),
                &y,
                (N, 1),
                (1e-5, 1.0),
                (&mut part, &mut hc),
            )
            .is_err();
        let ok = vec![folded_all, split_all, refused];
        Ok((ok, sep))
    }

    /// One site's route both dialects compute over `x`'s `m` tokens: the
    /// split parametrization's 24 values and the folded chart's.
    fn route_pair(
        gpu: &Gpu,
        k: &MhcKernels,
        s: &Site,
        x: &[f32],
        m: usize,
        parts: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        let stream = gpu.stream();
        let xb = DeviceBuffer::<f32>::from_host(stream, x).unwrap_or_else(|e| panic!("{e}"));
        let mut part =
            DeviceBuffer::<f32>::zeroed(stream, m * parts).unwrap_or_else(|e| panic!("{e}"));
        let mut hc =
            DeviceBuffer::<f32>::zeroed(stream, m * HC_OUT).unwrap_or_else(|e| panic!("{e}"));
        // The split dialect's own route.
        let rows =
            DeviceBuffer::<f32>::from_host(stream, &s.fn_rows).unwrap_or_else(|e| panic!("{e}"));
        let base =
            DeviceBuffer::<f32>::from_host(stream, &s.base).unwrap_or_else(|e| panic!("{e}"));
        let scales =
            DeviceBuffer::<f32>::from_host(stream, &s.scales).unwrap_or_else(|e| panic!("{e}"));
        k.enqueue_route(
            stream,
            &SiteVecs::new((&rows, &scales, &base)),
            &xb,
            (N, m),
            (1e-5, 1.0),
            (&mut part, &mut hc),
        )
        .map_err(|e| format!("the split route: {e}"))?;
        let split = hc.to_host_vec(stream).unwrap_or_else(|e| panic!("{e}"));
        // The folded chart's.
        let folded_rows = DeviceBuffer::<f32>::from_host(stream, &s.folded_rows)
            .unwrap_or_else(|e| panic!("{e}"));
        let chart =
            DeviceBuffer::<f32>::from_host(stream, &s.chart).unwrap_or_else(|e| panic!("{e}"));
        k.enqueue_route_folded(
            stream,
            &ChartVecs::new((&folded_rows, &base, &chart)),
            &xb,
            (N, m),
            (1e-5, 1.0),
            (&mut part, &mut hc),
        )
        .map_err(|e| format!("the folded route: {e}"))?;
        let folded = hc.to_host_vec(stream).unwrap_or_else(|e| panic!("{e}"));
        Ok((split, folded))
    }

    /// The folded parametrization's f64 reference of `s`'s route over `x`'s
    /// `m` tokens: 24 values a token.
    fn reference(s: &Site, x: &[f32], m: usize) -> Vec<f32> {
        let missing = None;
        reference_of(s, x, m, missing)
    }

    /// [`reference`] with one clause missing: the ρ division (`no_rho`) or
    /// the combination constant.
    fn reference_missing(s: &Site, x: &[f32], m: usize, no_rho: bool) -> Vec<f32> {
        reference_of(s, x, m, Some(no_rho))
    }

    /// The reference both clause forms share (module doc).
    fn reference_of(s: &Site, x: &[f32], m: usize, missing: Option<bool>) -> Vec<f32> {
        let k = STREAMS * N;
        let mut out = Vec::with_capacity(m * HC_OUT);
        for t in 0..m {
            let xt = &x[t * k..(t + 1) * k];
            let ss: f64 = xt.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
            let rms = (ss / k as f64 + 1e-5).sqrt();
            let affine = |row: usize| {
                let rows = &s.folded_rows[row * k..(row + 1) * k];
                let mut dot = 0.0f64;
                for (v, x) in rows.iter().zip(xt) {
                    dot += f64::from(*v) * f64::from(*x);
                }
                dot / rms + f64::from(s.base[row])
            };
            let mix: Vec<f64> = (0..HC_MIX).map(affine).collect();
            let sigmoid = |v: f64| 1.0 / (1.0 + (-v).exp());
            for i in 0..STREAMS {
                out.push(sigmoid(mix[i]) as f32);
                out.push((sigmoid(mix[STREAMS + i]) * 1.0) as f32);
            }
            let k_at = STREAMS * STREAMS;
            let mut e = vec![0.0f64; k_at];
            for (j, out) in e.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for (mid, m) in mix.iter().enumerate() {
                    acc += f64::from(s.chart[j * HC_MIX + mid]) * *m;
                }
                *out = acc;
            }
            let mut worst = f64::NEG_INFINITY;
            for (j, out) in e.iter().enumerate() {
                let ratio = *out * f64::from(s.chart[HC_MIX * k_at + j]);
                if ratio > worst {
                    worst = ratio;
                }
            }
            let rho = 1.0 + worst;
            for (j, ev) in e.iter().enumerate() {
                let c0 = if missing == Some(false) {
                    0.0
                } else {
                    f64::from(s.chart[(HC_MIX + 1) * k_at + j])
                };
                let phi = if missing == Some(true) {
                    *ev
                } else {
                    *ev / rho
                };
                out.push((phi + c0) as f32);
            }
        }
        out
    }

    /// The ple clause's launches: the two entries against their host rules,
    /// bit for bit, and the refusals.
    fn ple_clauses(gpu: &Gpu) -> Result<(Vec<bool>, bool), GateError> {
        let ctx: Arc<_> = gpu.context().clone();
        let k = PleKernels::load(&ctx).map_err(|e| format!("the ple module: {e}"))?;
        let stream = gpu.stream();
        let mut r = Rng(0x706c_6520);
        let a: Vec<f32> = (0..ROWS * LAYERS * PLE)
            .map(|_| r.range(-1.0, 1.0))
            .collect();
        let rows: Vec<f32> = (0..ROWS * LAYERS * PLE)
            .map(|_| r.range(-1.0, 1.0))
            .collect();
        let (scale, c) = ((PLE as f32).sqrt(), 1.0f32 / 2.0f32.sqrt());
        let ab = DeviceBuffer::<f32>::from_host(stream, &a).unwrap_or_else(|e| panic!("{e}"));
        let rb = DeviceBuffer::<f32>::from_host(stream, &rows).unwrap_or_else(|e| panic!("{e}"));
        let mut y = DeviceBuffer::<f32>::zeroed(stream, ROWS * LAYERS * PLE)
            .unwrap_or_else(|e| panic!("{e}"));
        k.enqueue_combine(stream, (&ab, &rb), (ROWS, LAYERS, PLE), (scale, c), &mut y)
            .map_err(|e| format!("the combine: {e}"))?;
        let got = y.to_host_vec(stream).unwrap_or_else(|e| panic!("{e}"));
        let want = combine_host(&a, &rows, (ROWS, LAYERS, PLE), (scale, c));
        let combine_ok = bits_equal(&got, &want);
        // A rerun is bit-identical, and the host rule with the table's scale
        // dropped differs.
        let mut y2 = DeviceBuffer::<f32>::zeroed(stream, ROWS * LAYERS * PLE)
            .unwrap_or_else(|e| panic!("{e}"));
        k.enqueue_combine(stream, (&ab, &rb), (ROWS, LAYERS, PLE), (scale, c), &mut y2)
            .map_err(|e| format!("the combine again: {e}"))?;
        let again = y2.to_host_vec(stream).unwrap_or_else(|e| panic!("{e}"));
        let wrong_scale = combine_host(&a, &rows, (ROWS, LAYERS, PLE), (1.0, c));
        let sep_combine = bits_equal(&again, &got) && !bits_equal(&wrong_scale, &got);

        // The add.
        let g: Vec<f32> = (0..ROWS * N).map(|_| r.range(-1.0, 1.0)).collect();
        let x0: Vec<f32> = (0..ROWS * STREAMS * N)
            .map(|_| r.range(-2.0, 2.0))
            .collect();
        let gb = DeviceBuffer::<f32>::from_host(stream, &g).unwrap_or_else(|e| panic!("{e}"));
        let mut xb = DeviceBuffer::<f32>::from_host(stream, &x0).unwrap_or_else(|e| panic!("{e}"));
        k.enqueue_add(stream, &gb, (N, ROWS), &mut xb)
            .map_err(|e| format!("the add: {e}"))?;
        let got_x = xb.to_host_vec(stream).unwrap_or_else(|e| panic!("{e}"));
        let want_x = add_host(&g, &x0, (N, ROWS));
        let add_ok = bits_equal(&got_x, &want_x);
        let three = add_host_three(&g, &x0, (N, ROWS));
        let sep_add = !bits_equal(&three, &got_x);

        // The refusals: a width of 0, a short g.
        let refused = k
            .enqueue_combine(stream, (&ab, &rb), (ROWS, LAYERS, 0), (scale, c), &mut y2)
            .is_err()
            && k.enqueue_add(stream, &gb, (N, ROWS + 1), &mut xb).is_err();
        Ok((vec![combine_ok, add_ok, refused], sep_combine && sep_add))
    }

    /// The combine's host rule (the entry's own order): the table's scale,
    /// the add, then the constant.
    fn combine_host(
        a: &[f32],
        rows: &[f32],
        (m, layers, width): (usize, usize, usize),
        (s, c): (f32, f32),
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; layers * m * width];
        for l in 0..layers {
            for t in 0..m {
                for d in 0..width {
                    let i = t * (layers * width) + l * width + d;
                    let g = l * (m * width) + t * width + d;
                    y[g] = (a[i] + rows[i] * s) * c;
                }
            }
        }
        y
    }

    /// The add's host rule.
    fn add_host(g: &[f32], x: &[f32], (n, m): (usize, usize)) -> Vec<f32> {
        let mut y = x.to_vec();
        for t in 0..m {
            for s in 0..STREAMS {
                for d in 0..n {
                    let at = t * STREAMS * n + s * n + d;
                    y[at] += g[t * n + d];
                }
            }
        }
        y
    }

    /// [`add_host`] adding to three streams alone (the fourth dropped).
    fn add_host_three(g: &[f32], x: &[f32], (n, m): (usize, usize)) -> Vec<f32> {
        let mut y = x.to_vec();
        for t in 0..m {
            for s in 0..STREAMS - 1 {
                for d in 0..n {
                    let at = t * STREAMS * n + s * n + d;
                    y[at] += g[t * n + d];
                }
            }
        }
        y
    }
}
