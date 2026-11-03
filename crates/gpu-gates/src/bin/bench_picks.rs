//! `bench_picks` — every candidate of a kernel pick, forced on the same shape and timed against its partner.
//!
//! The pairs (docs/research/kernel-picks.md §3.4), each on the shapes the Qwen3.8 files carry:
//!
//! - `mxfp8_<shape>_m<m>`: the MXFP8 gemv at `m` columns. The launcher takes the staged walk (`m` = 4) or the wide stage
//!   (`m` = 5..8) for a 16-byte aligned `x` and `k` a multiple of 512, so the plain walk at that `k` is a launch only a
//!   hook in `q8f32.rs` can make (an unaligned `x` runs the plain walk's scalar fallback, no candidate). At `m` = 4..8 the
//!   pair is a proxy: the plain walk at `k - 128` against the engine's walk at `k`, the same rows, compared per 128-value
//!   step by the model tool. At `m` = 1..3 the row is the engine's walk alone (`kcal single`).
//! - `q8_0p_<shape>_t<t>`: the Q8_0-plane GEMM, `gemm_q8_0p` against `gemm_q8_0p_slab` (`enqueue_gemm32_entry`), on the
//!   dense prompt projections, the stacks around the slab entry's wave, and two routed 512-expert stacks.
//! - `f32_<rows>_m<m>`, `bf16_<rows>_m<m>`: the exact tile against `ceil(m / 8)` launches of the gemv.
//!
//! Every pair but the MXFP8 proxy gives both candidates the same bits (checked on their first launch, a mismatch is a named
//! error, an all-zero result too). `--check` stops there and needs no timing lease. `--time` (lead-only, through
//! `tools/ref/bench-picks.sh`, which owns the lease and the witness blocks) then times each candidate's eager burst and
//! graph replay with `shared/bench.rs`'s instrument (`bench op=` rows) and runs [`PAIR_ROUNDS`] rounds in which the
//! candidates and a second capture of the first (the A/A arm, the instrument's own spread) take turns in rotating
//! order; the `kcal pair` row carries each arm's mean and least µs a launch and the ratio of every arm to the first,
//! per round. Weights are cycled through copies up to 512 MiB so a launch reads them from memory.
//!
//!     bench_picks (--check | --time) [--arm PREFIX]
//!
//! `--arm` keeps the arms whose label starts with PREFIX. `tools/ref/kernel-model.py` reads the `kcal` rows.

#[path = "shared/bench.rs"]
mod bench_arm;

fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("bench_picks", picks::run())
}

mod picks {
    use crate::bench_arm::{
        GREPS, N, ROUNDS, burst, fill_pattern, fill_pattern_f32, print_arm, replay,
    };
    use bloomery_gpu::gemm::{
        GEMM_BN, Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput, GemmKernels,
        GemmRoute, Q8_0pEntry,
    };
    use bloomery_gpu::q8f32::{
        E4m3GemvArgs, E4m3Scales, E4m3Weight, GemvOut, Q8F32Kernels, e4m3_planes,
    };
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Window, WindowMut};
    use bloomery_gpu_gates::{GateError, GateRng};
    use cuda_core::{CudaStream, DeviceBuffer};
    use std::fmt::Write as _;
    use std::time::Instant;

    /// Rounds of the paired timing: every arm takes one launch group a round, in an order that rotates.
    const PAIR_ROUNDS: usize = 9;
    /// Target bytes of weight copies, subject to the per-graph copy cap [`N`].
    const SPAN: u64 = 512 << 20;
    /// Columns one `f32_gemv` or `bf16_gemv` launch takes.
    const GEMV_COLS: usize = 8;
    /// Rows one Q8_0-plane tile covers (`GEMM_BM`).
    const TILE_ROWS: usize = 128;

    /// One enqueue of a candidate: the stream, which weight copy, the output.
    type Enq<'a> =
        Box<dyn Fn(&CudaStream, usize, &mut DeviceBuffer<f32>) -> Result<(), GpuError> + 'a>;

    /// How a candidate lays its outputs out, for the bit comparison.
    #[derive(Clone, Copy)]
    enum Layout {
        /// `y[c * rows + r]`: the GEMM families' slot-major and the gemv's token-major.
        Col,
        /// Launches of [`GEMV_COLS`] columns, each row-major with its own columns a row, back to back.
        Chunked,
    }

    struct Cand<'a> {
        name: &'static str,
        layout: Layout,
        enq: Enq<'a>,
    }

    /// One configuration: the arm's label, the fields every row prints, and what the timing needs of it.
    struct Cfg {
        label: String,
        family: &'static str,
        /// `key=value` words of the shape, in the row.
        fields: String,
        rows: usize,
        m: usize,
        copies: usize,
        bytes: u64,
        /// The candidate the engine's rule takes: its name, `-` when none of the candidates.
        rule: &'static str,
        /// Whether the candidates give one answer, which the first launch checks: false for a proxy pair of two shapes.
        compare: bool,
    }

    struct Opts {
        time: bool,
        arm: Option<String>,
    }

    impl Opts {
        fn keeps(&self, label: &str) -> bool {
            self.arm.as_deref().is_none_or(|p| label.starts_with(p))
        }
    }

    fn opts() -> Result<Opts, GateError> {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut o = Opts {
            time: false,
            arm: None,
        };
        let mut mode = None;
        let mut it = args.iter();
        while let Some(a) = it.next() {
            match a.as_str() {
                "--check" | "--time" if mode.is_none() => {
                    o.time = a == "--time";
                    mode = Some(());
                }
                "--arm" if o.arm.is_none() => {
                    let p = it.next().ok_or("bench_picks: --arm wants a label prefix")?;
                    o.arm = Some(p.clone());
                }
                _ => {
                    return Err(format!(
                        "bench_picks: want one of --check, --time and optionally --arm PREFIX; got {args:?}"
                    )
                    .into());
                }
            }
        }
        if mode.is_none() {
            return Err("bench_picks: want one of --check, --time".into());
        }
        Ok(o)
    }

    /// Copies of a `bytes` weight to reach [`SPAN`], capped at [`N`] launches a graph.
    fn copies(bytes: u64) -> usize {
        (SPAN.div_ceil(bytes) as usize).clamp(1, N)
    }

    fn mean_sd(xs: &[f64]) -> (f64, f64) {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0).max(1.0);
        (mean, var.sqrt())
    }

    /// `y` in `[column][row]` order as bits, for the comparison.
    fn canon(layout: Layout, y: &[f32], rows: usize, m: usize) -> Vec<u32> {
        match layout {
            Layout::Col => y.iter().map(|v| v.to_bits()).collect(),
            Layout::Chunked => {
                let mut out = vec![0u32; rows * m];
                let mut c0 = 0;
                while c0 < m {
                    let mj = (m - c0).min(GEMV_COLS);
                    for r in 0..rows {
                        for c in 0..mj {
                            out[(c0 + c) * rows + r] = y[c0 * rows + r * mj + c].to_bits();
                        }
                    }
                    c0 += mj;
                }
                out
            }
        }
    }

    /// The card's facts the model reads, one `kcal card` row.
    fn card(gpu: &Gpu, g32: &Gemm32Kernels) -> Result<(), GateError> {
        let ctx = gpu.context();
        let (major, minor) = ctx.compute_capability()?;
        let l2 = cuda_core::Device::new(ctx.ordinal())?.l2_cache_size_bytes()?;
        let rates = match bloomery_gpu::host::costs::card_rates(ctx) {
            Ok(r) => format!(
                "dram_b_per_us={:.1} int8_ops_per_us={:.1}",
                r.dram_b_per_us, r.int8_ops_per_us
            ),
            Err(e) => format!("rates_unpriced={:?}", e.to_string()),
        };
        println!(
            "kcal card name={:?} cc={major}.{minor} sms={} l2_bytes={l2} {rates} slab_wave={} dense_mma={}",
            gpu.device_name()?,
            ctx.multiprocessor_count()?,
            g32.q8_0p_wave(),
            bloomery_gpu::dense_mma(ctx)?,
        );
        println!(
            "kcal bench n={N} greps={GREPS} rounds={ROUNDS} pair_rounds={PAIR_ROUNDS} span_bytes={SPAN}"
        );
        Ok(())
    }

    /// Check `cands` give one answer and, under `--time`, time them: the `bench op=` rows, then the paired rounds.
    fn run_cfg(gpu: &Gpu, o: &Opts, cfg: &Cfg, cands: &[Cand<'_>]) -> Result<(), GateError> {
        let stream = gpu.stream();
        let y_len = cfg.rows * cfg.m;
        let mut ys = (0..=cands.len())
            .map(|_| DeviceBuffer::<f32>::zeroed(stream, y_len))
            .collect::<Result<Vec<_>, _>>()?;
        let mut first: Option<Vec<u32>> = None;
        for (j, c) in cands.iter().enumerate() {
            (c.enq)(stream, 0, &mut ys[j])?;
            stream.synchronize()?;
            let got = canon(c.layout, &ys[j].to_host_vec(stream)?, cfg.rows, cfg.m);
            if got.iter().all(|&b| b == 0) {
                return Err(format!(
                    "bench_picks: {}: {} wrote only zero bits, so the launch did not run",
                    cfg.label, c.name
                )
                .into());
            }
            match &first {
                None => first = Some(got),
                Some(_) if !cfg.compare => {}
                Some(want) => {
                    if let Some(i) = (0..want.len()).find(|&i| want[i] != got[i]) {
                        return Err(format!(
                            "bench_picks: {}: {} and {} differ at output {i}: {:#010x} against {:#010x}",
                            cfg.label, cands[0].name, c.name, want[i], got[i]
                        )
                        .into());
                    }
                }
            }
        }
        if !o.time {
            println!(
                "kcal check label={} family={} {} cands={} bits={}",
                cfg.label,
                cfg.family,
                cfg.fields,
                cands.iter().map(|c| c.name).collect::<Vec<_>>().join(","),
                if cfg.compare { "equal" } else { "unchecked" }
            );
            return Ok(());
        }
        // The graphs: one per candidate, and the first again from another copy, the A/A arm.
        let mut graphs = Vec::with_capacity(cands.len() + 1);
        for j in 0..=cands.len() {
            let (c, skew) = if j < cands.len() {
                (&cands[j], 0)
            } else {
                (&cands[0], cfg.copies / 2)
            };
            let y = &mut ys[j];
            let mut at = skew;
            let mut enq = |s: &CudaStream| {
                let r = (c.enq)(s, at % cfg.copies, y);
                at += 1;
                r
            };
            let eager = if j < cands.len() {
                Some(burst(stream, &mut enq)?)
            } else {
                None
            };
            let g = gpu.capture(|s| (0..N).try_for_each(|_| enq(s)))?;
            if let Some(e) = eager {
                let r = replay(stream, &g)?;
                print_arm(
                    &format!("{}_{}", cfg.label, c.name),
                    cfg.rows,
                    cfg.bytes,
                    g.node_count(),
                    e,
                    r,
                );
            }
            graphs.push(g);
        }
        let arms = graphs.len();
        for g in &graphs {
            for _ in 0..2 {
                g.launch(stream)?;
            }
        }
        stream.synchronize()?;
        // Launches of a graph a round: one when a graph is long enough to time by itself.
        let t0 = Instant::now();
        graphs[0].launch(stream)?;
        stream.synchronize()?;
        let reps = if t0.elapsed().as_secs_f64() > 0.02 {
            1
        } else {
            GREPS
        };
        let mut us = vec![Vec::with_capacity(PAIR_ROUNDS); arms];
        for r in 0..PAIR_ROUNDS {
            for j in 0..arms {
                let a = (r + j) % arms;
                let t0 = Instant::now();
                for _ in 0..reps {
                    graphs[a].launch(stream)?;
                }
                stream.synchronize()?;
                us[a].push(t0.elapsed().as_secs_f64() * 1e6 / (N * reps) as f64);
            }
        }
        let least = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let mut row = String::new();
        for (a, v) in us.iter().enumerate() {
            let name = if a < cands.len() { cands[a].name } else { "aa" };
            let _ = write!(
                row,
                " {name}_us_mean={:.3} {name}_us_min={:.3}",
                mean_sd(v).0,
                least(v)
            );
        }
        // Per round, every arm over the first.
        for (a, v) in us.iter().enumerate().skip(1) {
            let ratios: Vec<f64> = v.iter().zip(&us[0]).map(|(x, y)| x / y).collect();
            let (mean, sd) = mean_sd(&ratios);
            let name = if a < cands.len() { cands[a].name } else { "aa" };
            let _ = write!(
                row,
                " {name}_over_{}={mean:.4} {name}_over_{}_sd={sd:.4}",
                cands[0].name, cands[0].name
            );
        }
        let kind = if cands.len() == 1 { "single" } else { "pair" };
        println!(
            "kcal {kind} label={} family={} {} m={} a={} b={} rule={} bytes={} copies={} reps={reps} rounds={PAIR_ROUNDS}{row}",
            cfg.label,
            cfg.family,
            cfg.fields,
            cfg.m,
            cands[0].name,
            cands.get(1).map_or("-", |c| c.name),
            cfg.rule,
            cfg.bytes,
            cfg.copies,
        );
        Ok(())
    }

    /// The MXFP8 planes' source bytes: 33 bytes a 32 values, a scale byte of 2^-15..2^9 and values from the 254 finite E4M3 bytes.
    fn mxfp8_bytes(rows: usize, k: usize, seed: u64) -> Vec<u8> {
        let mut rng = GateRng(seed);
        let values: Vec<u8> = (0..=u8::MAX).filter(|b| b & 0x7F != 0x7F).collect();
        let mut out = Vec::with_capacity(rows * k / 32 * 33);
        for _ in 0..rows * k / 32 {
            out.push(112 + rng.below(25) as u8);
            for _ in 0..32 {
                out.push(values[rng.below(values.len())]);
            }
        }
        out
    }

    /// Qwen3.8's MXFP8 dense verify shapes `(label, rows, k)`: the GDN output, the qkvz, the attention query; then a
    /// latent attention's `attn_q_b` and `attn_output`, the short row and the long row `gate_mxfp8` runs.
    const MXFP8_SHAPES: [(&str, usize, usize); 5] = [
        ("2560x6144", 2560, 6144),
        ("10240x2560", 10_240, 2560),
        ("12288x2560", 12_288, 2560),
        ("16384x1536", 16_384, 1536),
        ("4096x16384", 4096, 16_384),
    ];

    /// The MXFP8 gemv of `ws[c % ws.len()]` against the `m` columns of `x`, token-major: the launch as the engine makes it.
    fn mxfp8_enq<'a>(
        q8: &'a Q8F32Kernels,
        ws: &'a [(DeviceTensor<u32>, DeviceTensor<u32>)],
        x: &'a DeviceBuffer<f32>,
        m: usize,
    ) -> Enq<'a> {
        Box::new(move |s, c, y| {
            let w = &ws[c % ws.len()];
            q8.enqueue_e4m3_gemv(
                s,
                E4m3GemvArgs {
                    w: E4m3Weight {
                        qs: &w.0,
                        e: &w.1,
                        scales: E4m3Scales::Mxfp8,
                    },
                    x,
                    m,
                    out: GemvOut::TokenMajor,
                    y,
                },
            )
        })
    }

    /// Weight planes of `rows` by `k` MXFP8 values, with the copies a graph cycles through.
    fn mxfp8_planes(
        gpu: &Gpu,
        (rows, k): (usize, usize),
        seed: u64,
    ) -> Result<Vec<(DeviceTensor<u32>, DeviceTensor<u32>)>, GateError> {
        let stream = gpu.stream();
        let (qs, e) = e4m3_planes(E4m3Scales::Mxfp8, &mxfp8_bytes(rows, k, seed), rows, k)?;
        (0..copies((rows * k / 32 * 33) as u64))
            .map(|_| {
                Ok((
                    DeviceTensor::upload(stream, &qs, rows, k / 4)?,
                    DeviceTensor::upload(stream, &e, rows, k / 128)?,
                ))
            })
            .collect::<Result<Vec<_>, GpuError>>()
            .map_err(Into::into)
    }

    /// The MXFP8 gemv at one to eight columns. The launcher takes the staged walk (four columns) or the wide stage (five
    /// to eight) for an `x` from a 16-byte aligned base when `k` is a multiple of 512, and the plain walk below four
    /// columns or at any other `k`; the plain walk at a multiple of 512 and four or more columns is a launch only a hook
    /// in `q8f32.rs` can make, and an unaligned `x` runs the plain walk's scalar fallback, which is no candidate. So
    /// the pair at four or more columns is a proxy: the plain walk at `k - 128` (not a multiple of 512, aligned `x`)
    /// against the engine's walk at `k`, each a launch the engine makes, on the same rows; the tool compares them per
    /// 128-value step. Below four columns the row is the engine's plain walk alone.
    fn mxfp8(gpu: &Gpu, o: &Opts) -> Result<(), GateError> {
        let stream = gpu.stream();
        let q8 = gpu.q8f32();
        for (i, (shape, rows, k)) in MXFP8_SHAPES.into_iter().enumerate() {
            if !(1..=GEMV_COLS).any(|m| o.keeps(&format!("mxfp8_{shape}_m{m}"))) {
                continue;
            }
            let seed = 0x5eed_0900 + i as u64;
            let bytes = (rows * k / 32 * 33) as u64;
            let ws = mxfp8_planes(gpu, (rows, k), seed)?;
            let k_plain = k - 128;
            let ws_plain = mxfp8_planes(gpu, (rows, k_plain), seed)?;
            let x_host = fill_pattern_f32(GEMV_COLS * k);
            for m in 1..=GEMV_COLS {
                let label = format!("mxfp8_{shape}_m{m}");
                if !o.keeps(&label) {
                    continue;
                }
                let x = DeviceBuffer::from_host(stream, &x_host[..m * k])?;
                let x_plain = DeviceBuffer::from_host(stream, &x_host[..m * k_plain])?;
                if !x.cu_deviceptr().is_multiple_of(16)
                    || !x_plain.cu_deviceptr().is_multiple_of(16)
                {
                    return Err(format!(
                        "bench_picks: {label}: the arms need an x at a 16-byte aligned base"
                    )
                    .into());
                }
                // The engine's walk at this width: the single-column kernel, the plain walk below four columns, the
                // staged walk at four and the wide stage past.
                let engine = match m {
                    1 => "mxfp8_gemv",
                    2 | 3 => "plain",
                    4 => "staged",
                    _ => "staged_wide",
                };
                let proxy = m >= 4;
                let cfg = Cfg {
                    label,
                    family: "mxfp8",
                    fields: format!(
                        "shape={shape} rows={rows} k={k} k_plain={} experts=1 slots={m}",
                        if proxy { k_plain } else { k }
                    ),
                    rows,
                    m,
                    copies: ws.len().max(ws_plain.len()),
                    bytes,
                    rule: engine,
                    compare: false,
                };
                let mut cands = vec![Cand {
                    name: engine,
                    layout: Layout::Col,
                    enq: mxfp8_enq(q8, &ws, &x, m),
                }];
                if proxy {
                    cands.insert(
                        0,
                        Cand {
                            name: "plain",
                            layout: Layout::Col,
                            enq: mxfp8_enq(q8, &ws_plain, &x_plain, m),
                        },
                    );
                }
                run_cfg(gpu, o, &cfg, &cands)?;
            }
        }
        Ok(())
    }

    /// One Q8_0-plane stack: `experts` of `rows` rows by `k`, the activations `cols` columns, `top_k` slots a token
    /// (1 for a dense weight), the slots reading one column each or a token's column in common.
    struct Q8Stack<'a> {
        label: &'a str,
        rows: usize,
        k: usize,
        experts: usize,
        top_k: usize,
        tokens: usize,
        per_slot: bool,
    }

    /// Expert ids for `t` tokens of `top_k` distinct experts out of `n_exp`, drawn from a fixed LCG.
    fn route_ids(t: usize, top_k: usize, n_exp: usize) -> Vec<u32> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut ids = Vec::with_capacity(t * top_k);
        for _ in 0..t {
            let mut pick: Vec<u32> = Vec::with_capacity(top_k);
            while pick.len() < top_k {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let e = ((s >> 33) % n_exp as u64) as u32;
                if !pick.contains(&e) {
                    pick.push(e);
                }
            }
            ids.extend(pick);
        }
        ids
    }

    /// The code and scale planes of a Q8_0 stack, one pair a copy.
    type Q8Planes = Vec<(DeviceTensor<u32>, DeviceTensor<u16>)>;

    /// Q8_0-plane planes for `stack` with the copies a graph cycles through.
    fn q8_planes(gpu: &Gpu, rows: usize, k: usize) -> Result<Q8Planes, GateError> {
        let stream = gpu.stream();
        let kb = k / 32;
        let words = fill_pattern(rows * k / 4);
        // Finite f16 scales in [2^-7, 2^-6): the pattern's low bits as the mantissa.
        let d_bits: Vec<u16> = fill_pattern(rows * kb)
            .into_iter()
            .map(|w| 0x2000 | (w & 0x03ff) as u16)
            .collect();
        let bytes = (rows * (k + 2 * kb)) as u64;
        (0..copies(bytes))
            .map(|_| {
                Ok((
                    DeviceTensor::upload(stream, &words, rows, k / 4)?,
                    DeviceTensor::upload(stream, &d_bits, rows, kb)?,
                ))
            })
            .collect::<Result<Vec<_>, GpuError>>()
            .map_err(Into::into)
    }

    fn q8_arm(
        gpu: &Gpu,
        o: &Opts,
        (gk, g32): (&GemmKernels, &Gemm32Kernels),
        s: &Q8Stack<'_>,
        ws: &[(DeviceTensor<u32>, DeviceTensor<u16>)],
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        let (rows, k, t) = (s.rows, s.k, s.tokens);
        let n_slots = t * s.top_k;
        let label = format!("q8_0p_{}_t{t}", s.label);
        if !o.keeps(&label) {
            return Ok(());
        }
        let cols = if s.per_slot { n_slots } else { t };
        let xs = DeviceBuffer::<f32>::from_host(stream, &fill_pattern_f32(cols * k))?;
        let mut act = GemmAct32::new(stream, cols, k)?;
        g32.enqueue_quantize_gemm32(stream, &xs, cols, &mut act, sink)?;
        let mut route = GemmRoute::new(stream, n_slots, s.experts)?;
        let input = if s.experts == 1 {
            gk.enqueue_route_dense(stream, n_slots, &mut route, sink)?;
            GemmInput::PerSlot
        } else {
            let ids = DeviceBuffer::<u32>::from_host(stream, &route_ids(t, s.top_k, s.experts))?;
            gk.enqueue_route(stream, &ids, n_slots, &mut route, sink)?;
            if s.per_slot {
                GemmInput::PerSlot
            } else {
                GemmInput::Shared { top_k: s.top_k }
            }
        };
        let grid = rows.div_ceil(TILE_ROWS) * (n_slots / GEMM_BN + s.experts.min(n_slots));
        let slab_rule = g32.q8_0p_slab(grid as u32, n_slots as u32, s.experts as u32);
        let bytes = (s.experts * rows * (k + 2 * (k / 32))) as u64;
        let enq = |force: Q8_0pEntry| -> Enq<'_> {
            let (act, route) = (&act, &route);
            Box::new(
                move |st: &CudaStream, c: usize, y: &mut DeviceBuffer<f32>| {
                    g32.enqueue_gemm32_entry(
                        st,
                        Gemm32Args {
                            w: Gemm32Weight::Q8_0Plane {
                                qs: &ws[c].0,
                                d: &ws[c].1,
                            },
                            rows_per_expert: rows,
                            act,
                            route,
                            input,
                            y,
                        },
                        Some(force),
                    )
                },
            )
        };
        let cands = [
            Cand {
                name: "plain",
                layout: Layout::Col,
                enq: enq(Q8_0pEntry::Plain),
            },
            Cand {
                name: "slab",
                layout: Layout::Col,
                enq: enq(Q8_0pEntry::Slab),
            },
        ];
        let cfg = Cfg {
            label,
            family: "q8_0p",
            fields: format!(
                "shape={} rows={rows} k={k} experts={} slots={n_slots} tokens={t} grid={grid} per_slot={}",
                s.label, s.experts, s.per_slot
            ),
            rows,
            m: n_slots,
            copies: ws.len(),
            bytes,
            rule: if slab_rule { "slab" } else { "plain" },
            compare: true,
        };
        run_cfg(gpu, o, &cfg, &cands)
    }

    /// The dense prompt projections of Qwen3.8 `(label, rows, k)`, at the columns of a short unit, a unit past the
    /// rule's 64, and the prompt's ubatches; then the stacks around one wave of the slab entry; then the routed stacks.
    fn q8(gpu: &Gpu, o: &Opts, g32: &Gemm32Kernels) -> Result<(), GateError> {
        let gk = GemmKernels::load(gpu.context())?;
        let wave = usize::try_from(g32.q8_0p_wave())?;
        let wave2 = wave.div_ceil(2);
        let ts = &[16usize, 65, 256, 512, 2048][..];
        let dense: Vec<(String, usize, usize, Vec<usize>)> = [
            ("qkv", 10_240usize, 2560usize),
            ("z", 6144, 2560),
            ("out", 2560, 6144),
            ("q", 12_288, 2560),
            ("kv", 512, 2560),
            ("shgu", 640, 2560),
            ("shdown", 2560, 640),
            ("hcdown", 320, 10_240),
            ("hcup", 10_240, 320),
        ]
        .into_iter()
        .map(|(l, r, k)| (l.to_string(), r, k, ts.to_vec()))
        .chain([
            ("wavem1".to_string(), TILE_ROWS * (wave - 1), 2560, vec![16]),
            ("wave".to_string(), TILE_ROWS * wave, 2560, vec![16]),
            ("wavep1".to_string(), TILE_ROWS * (wave + 1), 2560, vec![16]),
            (
                "wave2m".to_string(),
                TILE_ROWS * (wave2 - 1),
                2560,
                vec![65],
            ),
            ("wave2".to_string(), TILE_ROWS * wave2, 2560, vec![65]),
            (
                "wave2p".to_string(),
                TILE_ROWS * (wave2 + 1),
                2560,
                vec![65],
            ),
        ])
        .collect();
        for (label, rows, k, ts) in &dense {
            if !ts.iter().any(|t| o.keeps(&format!("q8_0p_{label}_t{t}"))) {
                continue;
            }
            let ws = q8_planes(gpu, *rows, *k)?;
            for &tokens in ts {
                let s = Q8Stack {
                    label,
                    rows: *rows,
                    k: *k,
                    experts: 1,
                    top_k: 1,
                    tokens,
                    per_slot: true,
                };
                q8_arm(gpu, o, (&gk, g32), &s, &ws)?;
            }
        }
        // Routed stacks of Qwen3.8's 512 experts, top-10: the down (each slot its own column) and a gate-shaped stack
        // (a token's column in common).
        for (label, rows, k, per_slot) in [
            ("exps_down", 2560usize, 640usize, true),
            ("exps_gate", 640, 2560, false),
        ] {
            let toks = [64usize, 256, 1024, 4096];
            if !toks.iter().any(|t| o.keeps(&format!("q8_0p_{label}_t{t}"))) {
                continue;
            }
            let ws = q8_planes(gpu, 512 * rows, k)?;
            for tokens in toks {
                let s = Q8Stack {
                    label,
                    rows,
                    k,
                    experts: 512,
                    top_k: 10,
                    tokens,
                    per_slot,
                };
                q8_arm(gpu, o, (&gk, g32), &s, &ws)?;
            }
        }
        Ok(())
    }

    /// Whether the F32 (or BF16) wide product is the tile or the gemv repeated: F32 weights `(rows)` of the UD-Q4_K_XL
    /// file (the router and the linear attention's β·α), BF16 weights of the selecting attention's indexer.
    fn wide(gpu: &Gpu, o: &Opts, g32: &Gemm32Kernels) -> Result<(), GateError> {
        let stream = gpu.stream();
        let q8 = gpu.q8f32();
        let k = 2560usize;
        let mma = bloomery_gpu::dense_mma(gpu.context())?;
        for (family, rows_set, ms) in [
            ("f32", &[512usize, 48][..], &[16usize, 64, 512, 2048][..]),
            ("bf16", &[512, 128][..], &[2, 4, 8, 16, 64, 512, 2048][..]),
        ] {
            for &rows in rows_set {
                if !ms.iter().any(|m| o.keeps(&format!("{family}_{rows}_m{m}"))) {
                    continue;
                }
                let is_f32 = family == "f32";
                let vals = fill_pattern_f32(rows * k);
                let bytes = (rows * k * if is_f32 { 4 } else { 2 }) as u64;
                let n_copies = copies(bytes);
                let wf: Vec<DeviceTensor<f32>> = if is_f32 {
                    (0..n_copies)
                        .map(|_| DeviceTensor::upload(stream, &vals, rows, k))
                        .collect::<Result<_, _>>()?
                } else {
                    Vec::new()
                };
                let words: Vec<u32> = vals
                    .chunks(2)
                    .map(|p| (p[0].to_bits() >> 16) | (p[1].to_bits() & 0xFFFF_0000))
                    .collect();
                let wb: Vec<DeviceTensor<u32>> = if is_f32 {
                    Vec::new()
                } else {
                    (0..n_copies)
                        .map(|_| DeviceTensor::upload(stream, &words, rows, k / 2))
                        .collect::<Result<_, _>>()?
                };
                for &m in ms {
                    let label = format!("{family}_{rows}_m{m}");
                    if !o.keeps(&label) {
                        continue;
                    }
                    let x = DeviceBuffer::from_host(stream, &fill_pattern_f32(m * k))?;
                    let (wf, wb, x) = (&wf, &wb, &x);
                    let gemv: Enq<'_> = Box::new(move |st, c, y| {
                        let mut c0 = 0;
                        while c0 < m {
                            let mj = (m - c0).min(GEMV_COLS);
                            let xw = Window::<f32>::of(x, c0 * k * 4, mj * k)?;
                            let mut yw =
                                WindowMut::<f32>::of_mut(&mut *y, c0 * rows * 4, mj * rows)?;
                            if is_f32 {
                                q8.enqueue_f32_gemv(st, &wf[c], &xw, mj, &mut yw)?;
                            } else {
                                q8.enqueue_bf16_gemv(st, &wb[c], &xw, mj, &mut yw)?;
                            }
                            c0 += mj;
                        }
                        Ok(())
                    });
                    let tile: Enq<'_> = Box::new(move |st, c, y| {
                        if is_f32 {
                            g32.enqueue_f32_tile(st, &wf[c], x, m, y)
                        } else {
                            q8.enqueue_bf16_tile(st, &wb[c], x, m, y)
                        }
                    });
                    let cfg = Cfg {
                        label,
                        family: if is_f32 { "f32" } else { "bf16" },
                        fields: format!("shape={rows}x{k} rows={rows} k={k} experts=1 slots={m}"),
                        rows,
                        m,
                        copies: n_copies,
                        bytes,
                        // The tile is the engine's wide product; on a BF16 tensor-core device the prompt takes the MMA
                        // tile, which no candidate here is.
                        rule: match (is_f32, m <= GEMV_COLS, mma) {
                            (_, true, _) => "gemv",
                            (false, false, true) => "-",
                            _ => "tile",
                        },
                        compare: true,
                    };
                    let cands = [
                        Cand {
                            name: "gemv",
                            layout: Layout::Chunked,
                            enq: gemv,
                        },
                        Cand {
                            name: "tile",
                            layout: Layout::Col,
                            enq: tile,
                        },
                    ];
                    run_cfg(gpu, o, &cfg, &cands)?;
                }
            }
        }
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let o = opts()?;
        let gpu = Gpu::new()?;
        let g32 = Gemm32Kernels::load(gpu.context())?;
        card(&gpu, &g32)?;
        mxfp8(&gpu, &o)?;
        q8(&gpu, &o, &g32)?;
        wide(&gpu, &o, &g32)?;
        Ok(())
    }
}
