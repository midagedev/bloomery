//! `bench_picks` — every candidate of a kernel pick, forced on the same shape and timed against its partner.
//!
//! The pairs (docs/research/kernel-picks.md §3.4), each on the shapes the Qwen3.8 files carry:
//!
//! - `mxfp8_<shape>_m<m>`: the MXFP8 gemv at `m` columns. At `m` = 4..8 every way the load-time picker races there
//!   (`autotune::Mcol::racing`, through `enqueue_e4m3_gemv_way`): the engine's walk (the staged walk at 4, the wide stage
//!   past 4), the plain walk, and past 4 the split (four columns staged, then the rest). At `m` = 1..3 the row is the
//!   engine's walk alone (`kcal single`).
//! - `q8_0p_<shape>_t<t>`: the Q8_0-plane GEMM, `gemm_q8_0p` against `gemm_q8_0p_slab` (`enqueue_gemm32_entry`), on the
//!   dense prompt projections, the stacks around the slab entry's wave, and two routed 512-expert stacks.
//! - `f32_<rows>_m<m>`, `bf16_<rows>_m<m>`: the exact tile against `ceil(m / 8)` launches of the gemv.
//! - `fewrow_<type>_<what>_<rows>x<k>`: what one verify pass of a dense MTP model asks of a matrix at one, two, four and
//!   eight columns, every width on the same weights in one row: the engine's walk (`eng_m<m>`: the per-slot `_sel` entries
//!   the dense FFN launches with `n_slots = m`, the `m` one-column launches of the head, the read-once m-column cores
//!   elsewhere) and, where the engine's walk is not a read-once core, that core on the same weights (`mcol_m<m>`). The
//!   `kcal few` row carries each arm's µs a launch and its ratio to the same walk's m = 1 (`<arm>_vs_<m1 arm>`), plus every
//!   arm over `eng_m1`. `--check` holds every width's column c bit-equal to the one-column launch of x column c.
//!
//! Every candidate of a configuration gives the same bits (checked on their first launch, a mismatch is a named error, an
//! all-zero result too; a `fewrow` group's arms of unequal widths are held to the one-column oracle, non-finite bits and
//! columns that do not differ being named errors). `--check` stops there and needs no timing lease. `--time` (lead-only, through
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
    use bloomery_gpu::arch::qwen3moe::experts::{ExpertKernels, GateUpArgs};
    use bloomery_gpu::autotune::Mcol;
    use bloomery_gpu::gemm::{
        GEMM_BN, Gemm32Args, Gemm32Kernels, Gemm32Weight, GemmAct32, GemmInput, GemmKernels,
        GemmRoute, Q8_0pEntry,
    };
    use bloomery_gpu::q6k_sel::Q6kSelKernels;
    use bloomery_gpu::q8f32::{
        E4m3GemvArgs, E4m3Scales, E4m3Weight, GemvOut, Q8F32Kernels, e4m3_planes,
    };
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError, Q8Act, Window, WindowMut};
    use bloomery_gpu_gates::{GateError, GateRng};
    use cuda_core::{CudaStream, DeviceBuffer};
    use std::cell::RefCell;
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

    /// Reads a candidate's output back in its own layout when the output is not one buffer `y`.
    type Readback<'a> =
        Box<dyn Fn(&CudaStream, &DeviceBuffer<f32>) -> Result<Vec<f32>, GpuError> + 'a>;

    /// The reference every width of a `fewrow` group is held to: `[column][row]` bits of the one-column launches of the
    /// group's columns.
    type Oracle<'a> = Box<dyn Fn(&CudaStream) -> Result<Vec<u32>, GpuError> + 'a>;

    struct Cand<'a> {
        name: &'static str,
        layout: Layout,
        enq: Enq<'a>,
        /// Columns the candidate writes; 0 is the configuration's `m`.
        cols: usize,
        /// The candidate whose time this one's ratio is also taken against; empty when none.
        base: &'static str,
        readback: Option<Readback<'a>>,
    }

    impl<'a> Cand<'a> {
        fn new(name: &'static str, layout: Layout, enq: Enq<'a>) -> Self {
            Cand {
                name,
                layout,
                enq,
                cols: 0,
                base: "",
                readback: None,
            }
        }
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
        run_cfg_with(gpu, o, cfg, cands, None)
    }

    /// [`run_cfg`] with an `oracle` the candidates' columns are held to in place of the first candidate's: each
    /// candidate's `[column][row]` bits must be a prefix of it, and it must be finite with its columns told apart.
    fn run_cfg_with(
        gpu: &Gpu,
        o: &Opts,
        cfg: &Cfg,
        cands: &[Cand<'_>],
        oracle: Option<&Oracle<'_>>,
    ) -> Result<(), GateError> {
        let stream = gpu.stream();
        let cols_of = |j: usize| match cands[j.min(cands.len() - 1)].cols {
            0 => cfg.m,
            n => n,
        };
        let mut ys = (0..=cands.len())
            .map(|j| {
                DeviceBuffer::<f32>::zeroed(
                    stream,
                    cfg.rows * cols_of(if j < cands.len() { j } else { 0 }),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let reference = match oracle {
            Some(f) => {
                let want = f(stream)?;
                if let Some(i) = want.iter().position(|&b| !f32::from_bits(b).is_finite()) {
                    return Err(format!(
                        "bench_picks: {}: the one-column oracle holds a non-finite value at output {i}",
                        cfg.label
                    )
                    .into());
                }
                if want.len() > cfg.rows && want[..cfg.rows] == want[cfg.rows..2 * cfg.rows] {
                    return Err(format!(
                        "bench_picks: {}: the oracle's first two columns are equal, so a wrong column would pass",
                        cfg.label
                    )
                    .into());
                }
                Some(want)
            }
            None => None,
        };
        let mut first: Option<Vec<u32>> = None;
        for (j, c) in cands.iter().enumerate() {
            (c.enq)(stream, 0, &mut ys[j])?;
            stream.synchronize()?;
            let host = match &c.readback {
                Some(f) => f(stream, &ys[j])?,
                None => ys[j].to_host_vec(stream)?,
            };
            let got = canon(c.layout, &host, cfg.rows, cols_of(j));
            if got.iter().all(|&b| b == 0) {
                return Err(format!(
                    "bench_picks: {}: {} wrote only zero bits, so the launch did not run",
                    cfg.label, c.name
                )
                .into());
            }
            if reference.is_some() && got.iter().any(|&b| !f32::from_bits(b).is_finite()) {
                return Err(format!(
                    "bench_picks: {}: {} wrote a non-finite value",
                    cfg.label, c.name
                )
                .into());
            }
            if reference.is_none() && first.is_none() {
                first = Some(got);
                continue;
            }
            let (want_name, want) = match &reference {
                Some(r) => ("the one-column oracle", r.as_slice()),
                None => (cands[0].name, first.as_deref().unwrap_or_default()),
            };
            if got.len() > want.len() {
                return Err(format!(
                    "bench_picks: {}: {} writes {} values, the reference {want_name} holds {}",
                    cfg.label,
                    c.name,
                    got.len(),
                    want.len()
                )
                .into());
            }
            if let Some(i) = (0..got.len()).find(|&i| want[i] != got[i]) {
                return Err(format!(
                    "bench_picks: {}: {want_name} and {} differ at output {i}: {:#010x} against {:#010x}",
                    cfg.label, c.name, want[i], got[i]
                )
                .into());
            }
        }
        if !o.time {
            println!(
                "kcal check label={} family={} {} cands={} bits=equal{}",
                cfg.label,
                cfg.family,
                cfg.fields,
                cands.iter().map(|c| c.name).collect::<Vec<_>>().join(","),
                if reference.is_some() {
                    " against=one_column_oracle"
                } else {
                    ""
                },
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
        // Each width over the same walk's one-column launch.
        for (a, c) in cands.iter().enumerate().filter(|(_, c)| !c.base.is_empty()) {
            let b = cands.iter().position(|x| x.name == c.base).ok_or_else(|| {
                format!(
                    "bench_picks: {}: {} names no base {}",
                    cfg.label, c.name, c.base
                )
            })?;
            let ratios: Vec<f64> = us[a].iter().zip(&us[b]).map(|(x, y)| x / y).collect();
            let (mean, sd) = mean_sd(&ratios);
            let _ = write!(
                row,
                " {}_vs_{}={mean:.4} {}_vs_{}_sd={sd:.4}",
                c.name, c.base, c.name, c.base
            );
        }
        // A `fewrow` group's row is its own kind: `tools/ref/kernel-model.py` reads `pair` and `single` rows by family.
        let kind = if cfg.family == "fewrow" {
            "few"
        } else if cands.len() == 1 {
            "single"
        } else {
            "pair"
        };
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
    /// latent attention's `attn_q_b` and `attn_output`, the short row and the long row `gate_mxfp8` runs; then Qwen3.8's
    /// other verify sites (the 6,144-row, 640-, 512-, 128- and 96-row projections and the head) and a 1,152-row
    /// projection at `k` 5,120.
    const MXFP8_SHAPES: [(&str, usize, usize); 12] = [
        ("2560x6144", 2560, 6144),
        ("10240x2560", 10_240, 2560),
        ("12288x2560", 12_288, 2560),
        ("16384x1536", 16_384, 1536),
        ("4096x16384", 4096, 16_384),
        ("6144x2560", 6144, 2560),
        ("640x2560", 640, 2560),
        ("512x2560", 512, 2560),
        ("128x2560", 128, 2560),
        ("96x2560", 96, 2560),
        ("248320x2560", 248_320, 2560),
        ("1152x5120", 1152, 5120),
    ];

    /// One MXFP8 weight copy: its value plane and its scale plane.
    type Mxfp8Planes = (DeviceTensor<u32>, DeviceTensor<u32>);

    /// The MXFP8 gemv of `ws[c % ws.len()]` against the `m` columns of `x`, token-major, by `way`: [`Mcol::Rule`] is the
    /// launch as the engine makes it.
    fn mxfp8_enq<'a>(
        q8: &'a Q8F32Kernels,
        ws: &'a [Mxfp8Planes],
        x: &'a DeviceBuffer<f32>,
        m: usize,
        way: Mcol,
    ) -> Enq<'a> {
        Box::new(move |s, c, y| {
            let w = &ws[c % ws.len()];
            q8.enqueue_e4m3_gemv_way(
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
                way,
            )
        })
    }

    /// Weight planes of `rows` by `k` MXFP8 values, with the copies a graph cycles through.
    fn mxfp8_planes(
        gpu: &Gpu,
        (rows, k): (usize, usize),
        seed: u64,
    ) -> Result<Vec<Mxfp8Planes>, GateError> {
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

    /// The MXFP8 gemv at one to eight columns, each candidate on the same planes and the same `x` from a 16-byte aligned
    /// base: at four to eight columns every way the picker races there, the engine's walk first; below four the engine's
    /// plain walk alone.
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
            let x_host = fill_pattern_f32(GEMV_COLS * k);
            for m in 1..=GEMV_COLS {
                let label = format!("mxfp8_{shape}_m{m}");
                if !o.keeps(&label) {
                    continue;
                }
                let x = DeviceBuffer::from_host(stream, &x_host[..m * k])?;
                if !x.cu_deviceptr().is_multiple_of(16) {
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
                let ways = match Mcol::racing(m) {
                    [] => &[Mcol::Rule][..],
                    racing => racing,
                };
                let cands: Vec<Cand<'_>> = ways
                    .iter()
                    .map(|&way| {
                        Cand::new(
                            if way == Mcol::Rule {
                                engine
                            } else {
                                way.word()
                            },
                            Layout::Col,
                            mxfp8_enq(q8, &ws, &x, m, way),
                        )
                    })
                    .collect();
                let cfg = Cfg {
                    label,
                    family: "mxfp8",
                    fields: format!("shape={shape} rows={rows} k={k} experts=1 slots={m}"),
                    rows,
                    m,
                    copies: ws.len(),
                    bytes,
                    rule: engine,
                };
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
            Cand::new("plain", Layout::Col, enq(Q8_0pEntry::Plain)),
            Cand::new("slab", Layout::Col, enq(Q8_0pEntry::Slab)),
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
                    };
                    let cands = [
                        Cand::new("gemv", Layout::Chunked, gemv),
                        Cand::new("tile", Layout::Col, tile),
                    ];
                    run_cfg(gpu, o, &cfg, &cands)?;
                }
            }
        }
        Ok(())
    }

    /// Columns of the activation every `fewrow` arm draws from, and the widths it times.
    const FEW_COLS: usize = 8;
    const FEW_MS: [usize; 4] = [1, 2, 4, 8];
    const ENG: [&str; 4] = ["eng_m1", "eng_m2", "eng_m4", "eng_m8"];
    const MCOL: [&str; 4] = ["mcol_m1", "mcol_m2", "mcol_m4", "mcol_m8"];

    /// What a `fewrow` group's matrix is and which entry the engine launches it with.
    #[derive(Clone, Copy)]
    enum Few {
        /// The dense FFN's Q4_K gate and up: the gate·up·SwiGLU `_sel` entry over one expert, a slot a column.
        GateUp,
        /// The dense FFN's Q4_K down: `q4k_gemv_sel`, a slot a column.
        DownQ4k,
        /// The dense FFN's Q6_K down: `q6k_gemv_sel`, a slot a column.
        DownQ6k,
        /// The Q6_K head: one one-column launch a verify row.
        HeadQ6k,
        /// The delta layers' Q6_K qkv: `q6k_gemv`, one m-column launch.
        QkvQ6k,
        /// A Q8_0 attention projection: `q8_0_gemv`, `q8_0_gemv_mcol` past one column.
        AttnQ8,
    }

    /// The groups `(label, kind, rows, k)`: the Qwen3.8-27B Q4_K_M file's FFN, head and delta qkv, and K-EXAONE's Q8_0
    /// attention query, the largest of their types by bytes a pass.
    const FEW_GROUPS: [(&str, Few, usize, usize); 6] = [
        ("q4k_gateup_17408x5120", Few::GateUp, 17_408, 5120),
        ("q4k_down_5120x17408", Few::DownQ4k, 5120, 17_408),
        ("q6k_down_5120x17408", Few::DownQ6k, 5120, 17_408),
        ("q6k_head_248320x5120", Few::HeadQ6k, 248_320, 5120),
        ("q6k_qkv_10240x5120", Few::QkvQ6k, 10_240, 5120),
        ("q8_0_attnq_8192x6144", Few::AttnQ8, 8192, 6144),
    ];

    /// A finite f16 scale in [2^-7, 2^-6): the pattern's low bits as the mantissa.
    fn f16_scale(w: u32) -> u32 {
        0x2000 | (w & 0x03ff)
    }

    /// `rows` Q4_K rows of `k` values: the pattern with each super-block's `d` and `dmin` finite.
    fn q4k_words(rows: usize, k: usize, salt: u32) -> Vec<u32> {
        let mut w = fill_pattern(rows * 36 * (k / 256));
        for (i, v) in w.iter_mut().enumerate() {
            *v ^= salt;
            if i % 36 == 0 {
                *v = f16_scale(*v) | (f16_scale(*v >> 3) << 16);
            }
        }
        w
    }

    /// `rows` Q6_K rows of `k` values (`k / 256` even): the pattern with each super-block's `d` finite.
    fn q6k_words(rows: usize, k: usize) -> Vec<u32> {
        let n_sb = k / 256;
        let mut w = fill_pattern(rows * 210 * n_sb / 4);
        for sb in 0..rows * n_sb {
            // `d` is the last two bytes of the 210-byte super-block, so on a 2-byte boundary of a word.
            let at = sb * 210 + 208;
            let (i, shift) = (at / 4, (at % 4) * 8);
            w[i] = (w[i] & !(0xFFFF << shift)) | (f16_scale(w[i].rotate_left(7)) << shift);
        }
        w
    }

    /// The activation's `FEW_COLS` columns on the card, its q8_1 form at each width of [`FEW_MS`] and column by column.
    struct FewIn {
        x: Vec<DeviceBuffer<f32>>,
        by_m: Vec<Q8Act>,
        by_col: Vec<Q8Act>,
    }

    impl FewIn {
        /// Columns of `k` values each: `x[i]` is the first `FEW_MS[i]` columns, `col[c]` column `c`; a K-quant `k` (a
        /// multiple of 256) also gets the quantized forms.
        fn new(
            gpu: &Gpu,
            k: usize,
            quantize: bool,
        ) -> Result<(FewIn, Vec<DeviceBuffer<f32>>), GateError> {
            let stream = gpu.stream();
            let host = fill_pattern_f32(FEW_COLS * k);
            let x = FEW_MS
                .iter()
                .map(|&m| DeviceBuffer::from_host(stream, &host[..m * k]))
                .collect::<Result<Vec<_>, _>>()?;
            let col = (0..FEW_COLS)
                .map(|c| DeviceBuffer::from_host(stream, &host[c * k..(c + 1) * k]))
                .collect::<Result<Vec<_>, _>>()?;
            let (mut by_m, mut by_col) = (Vec::new(), Vec::new());
            if quantize {
                for (&m, xm) in FEW_MS.iter().zip(&x) {
                    let mut a = Q8Act::with_k(stream, m, k)?;
                    gpu.enqueue_quantize_q8_1(xm, &mut a)?;
                    by_m.push(a);
                }
                for xc in &col {
                    let mut a = Q8Act::with_k(stream, 1, k)?;
                    gpu.enqueue_quantize_q8_1(xc, &mut a)?;
                    by_col.push(a);
                }
                stream.synchronize()?;
            }
            Ok((FewIn { x, by_m, by_col }, col))
        }
    }

    /// The oracle of `rows` output rows: `one(stream, c, y)` launches the walk's one-column form on column `c`.
    fn few_oracle<'a>(
        rows: usize,
        one: impl Fn(&CudaStream, usize, &mut DeviceBuffer<f32>) -> Result<(), GpuError> + 'a,
    ) -> Oracle<'a> {
        Box::new(move |s| {
            let mut out = Vec::with_capacity(rows * FEW_COLS);
            for c in 0..FEW_COLS {
                let mut y = DeviceBuffer::<f32>::zeroed(s, rows)?;
                one(s, c, &mut y)?;
                s.synchronize()?;
                out.extend(y.to_host_vec(s)?.iter().map(|v| v.to_bits()));
            }
            Ok(out)
        })
    }

    /// The head's engine arm at `m` columns: `m` one-column launches, each into a buffer of its own as each verify row's
    /// head has, read back as the `m` columns.
    fn few_head_arm<'a>(
        gpu: &'a Gpu,
        ws: &'a [DeviceTensor<u32>],
        cols: &'a [Q8Act],
        (rows, m): (usize, usize),
    ) -> Result<Cand<'a>, GateError> {
        let scratch = std::rc::Rc::new(RefCell::new(
            (0..m)
                .map(|_| DeviceBuffer::<f32>::zeroed(gpu.stream(), rows))
                .collect::<Result<Vec<_>, _>>()?,
        ));
        let (run, read) = (scratch.clone(), scratch);
        let mut cand = Cand::new(
            ENG[FEW_MS.iter().position(|&x| x == m).unwrap_or(0)],
            Layout::Col,
            Box::new(move |s, c, _| {
                let mut ys = run.borrow_mut();
                for (j, y) in ys.iter_mut().enumerate() {
                    gpu.enqueue_gemv_q6k_on(s, &ws[c % ws.len()], &cols[j], y)?;
                }
                Ok(())
            }),
        );
        cand.cols = m;
        cand.readback = Some(Box::new(move |s, _| {
            let mut out = Vec::with_capacity(rows * m);
            for y in read.borrow().iter() {
                out.extend(y.to_host_vec(s)?);
            }
            Ok(out)
        }));
        Ok(cand)
    }

    /// The few-column probe: each group of [`FEW_GROUPS`] on one set of weights at one, two, four and eight columns.
    fn fewrow(gpu: &Gpu, o: &Opts) -> Result<(), GateError> {
        let stream = gpu.stream();
        let sink = gpu.unlabelled_sink();
        for (label, kind, rows, k) in FEW_GROUPS {
            let label = format!("fewrow_{label}");
            if !o.keeps(&label) {
                continue;
            }
            let n_sb = k / 256;
            let (inp, cols) = FewIn::new(gpu, k, !matches!(kind, Few::AttnQ8))?;
            let sel = DeviceBuffer::<u32>::from_host(stream, &[0u32; FEW_COLS])?;
            let upload = |words: &[u32], wpr: usize| -> Result<DeviceTensor<u32>, GpuError> {
                DeviceTensor::upload(stream, words, rows, wpr)
            };
            let (ek, q6) = (
                ExpertKernels::load(gpu.context())?,
                Q6kSelKernels::load(gpu.context(), gpu.fault_word())?,
            );
            let (ek, q6, sel, inp, cols) = (&ek, &q6, &sel, &inp, &cols);
            // The weights are declared before the arms built over them, so they drop after.
            let (gu, q4, q6w, q8w);
            let q8 = gpu.q8f32();
            let mut cands: Vec<Cand<'_>> = Vec::new();
            let (oracle, bytes, copies_n, walk): (Oracle<'_>, u64, usize, &str);
            match kind {
                Few::GateUp => {
                    let b = (2 * rows * 144 * n_sb) as u64;
                    let host = (
                        q4k_words(rows, k, 0x1111_1111),
                        q4k_words(rows, k, 0x2222_2222),
                    );
                    gu = (0..copies(b))
                        .map(|_| Ok((upload(&host.0, 36 * n_sb)?, upload(&host.1, 36 * n_sb)?)))
                        .collect::<Result<Vec<_>, GpuError>>()?;
                    let launch = move |s: &CudaStream,
                                       w: &(DeviceTensor<u32>, DeviceTensor<u32>),
                                       mi: Option<usize>,
                                       c: usize,
                                       y: &mut DeviceBuffer<f32>| {
                        let (act, n_slots) = match mi {
                            Some(i) => (&inp.by_m[i], FEW_MS[i]),
                            None => (&inp.by_col[c], 1),
                        };
                        ek.enqueue_gate_up(
                            s,
                            GateUpArgs {
                                wg: &w.0,
                                wu: &w.1,
                                act,
                                sel,
                                n_slots,
                                rows_per_expert: rows,
                                fault: sink,
                                h: y,
                            },
                        )
                    };
                    let g = &gu;
                    for (i, name) in ENG.into_iter().enumerate() {
                        let mut c = Cand::new(
                            name,
                            Layout::Col,
                            Box::new(move |s, n, y| launch(s, &g[n % g.len()], Some(i), 0, y)),
                        );
                        c.cols = FEW_MS[i];
                        c.base = if i == 0 { "" } else { ENG[0] };
                        cands.push(c);
                    }
                    oracle = few_oracle(rows, move |s, c, y| launch(s, &g[0], None, c, y));
                    (bytes, copies_n, walk) = (b, gu.len(), "sel_gate_up");
                }
                Few::DownQ4k | Few::DownQ6k => {
                    let is4 = matches!(kind, Few::DownQ4k);
                    let b = (rows * if is4 { 144 * n_sb } else { 210 * n_sb }) as u64;
                    if is4 {
                        let host = q4k_words(rows, k, 0x3333_3333);
                        q4 = (0..copies(b))
                            .map(|_| upload(&host, 36 * n_sb))
                            .collect::<Result<Vec<_>, _>>()?;
                        q6w = Vec::new();
                    } else {
                        let host = q6k_words(rows, k);
                        q6w = (0..copies(b))
                            .map(|_| upload(&host, 210 * n_sb / 4))
                            .collect::<Result<Vec<_>, _>>()?;
                        q4 = Vec::new();
                    }
                    let ws = if is4 { &q4 } else { &q6w };
                    // The engine's slot-a-column entry, and the plain m-column gemv beside it.
                    let sel_launch =
                        move |s: &CudaStream,
                              w: &DeviceTensor<u32>,
                              act: &Q8Act,
                              n: usize,
                              y: &mut DeviceBuffer<f32>| {
                            if is4 {
                                gpu.q4k_sel()
                                    .enqueue_gemv_q4k_sel(s, w, act, sel, n, rows, y)
                            } else {
                                q6.enqueue_gemv_q6k_sel(s, w, act, sel, n, rows, y)
                            }
                        };
                    let mcol_launch =
                        move |s: &CudaStream,
                              w: &DeviceTensor<u32>,
                              act: &Q8Act,
                              y: &mut DeviceBuffer<f32>| {
                            if is4 {
                                gpu.enqueue_gemv_q4k_on(s, w, act, y)
                            } else {
                                gpu.enqueue_gemv_q6k_on(s, w, act, y)
                            }
                        };
                    for i in 0..FEW_MS.len() {
                        let m = FEW_MS[i];
                        let mut c = Cand::new(
                            ENG[i],
                            Layout::Col,
                            Box::new(move |s, n, y| {
                                sel_launch(s, &ws[n % ws.len()], &inp.by_m[i], m, y)
                            }),
                        );
                        c.cols = m;
                        c.base = if i == 0 { "" } else { ENG[0] };
                        cands.push(c);
                    }
                    for i in 0..FEW_MS.len() {
                        let mut c = Cand::new(
                            MCOL[i],
                            Layout::Chunked,
                            Box::new(move |s, n, y| {
                                mcol_launch(s, &ws[n % ws.len()], &inp.by_m[i], y)
                            }),
                        );
                        c.cols = FEW_MS[i];
                        c.base = if i == 0 { "" } else { MCOL[0] };
                        cands.push(c);
                    }
                    oracle = few_oracle(rows, move |s, c, y| {
                        mcol_launch(s, &ws[0], &inp.by_col[c], y)
                    });
                    (bytes, copies_n, walk) = (
                        b,
                        ws.len(),
                        if is4 {
                            "sel_q4k_and_mcol"
                        } else {
                            "sel_q6k_and_mcol"
                        },
                    );
                }
                Few::HeadQ6k | Few::QkvQ6k => {
                    let b = (rows * 210 * n_sb) as u64;
                    let host = q6k_words(rows, k);
                    q6w = (0..copies(b))
                        .map(|_| upload(&host, 210 * n_sb / 4))
                        .collect::<Result<Vec<_>, _>>()?;
                    let ws = &q6w;
                    let head = matches!(kind, Few::HeadQ6k);
                    for i in 0..FEW_MS.len() {
                        let m = FEW_MS[i];
                        let mut c = if head {
                            few_head_arm(gpu, ws, &inp.by_col, (rows, m))?
                        } else {
                            let mut c = Cand::new(
                                ENG[i],
                                Layout::Chunked,
                                Box::new(move |s, n, y| {
                                    gpu.enqueue_gemv_q6k_on(s, &ws[n % ws.len()], &inp.by_m[i], y)
                                }),
                            );
                            c.cols = m;
                            c
                        };
                        c.base = if i == 0 { "" } else { ENG[0] };
                        cands.push(c);
                    }
                    if head {
                        for i in 0..FEW_MS.len() {
                            let mut c = Cand::new(
                                MCOL[i],
                                Layout::Chunked,
                                Box::new(move |s, n, y| {
                                    gpu.enqueue_gemv_q6k_on(s, &ws[n % ws.len()], &inp.by_m[i], y)
                                }),
                            );
                            c.cols = FEW_MS[i];
                            c.base = if i == 0 { "" } else { MCOL[0] };
                            cands.push(c);
                        }
                    }
                    oracle = few_oracle(rows, move |s, c, y| {
                        gpu.enqueue_gemv_q6k_on(s, &ws[0], &inp.by_col[c], y)
                    });
                    (bytes, copies_n, walk) = (
                        b,
                        ws.len(),
                        if head {
                            "one_column_launches_and_mcol"
                        } else {
                            "mcol"
                        },
                    );
                }
                Few::AttnQ8 => {
                    q8w = q8_planes(gpu, rows, k)?;
                    let ws = &q8w;
                    let b = (rows * (k + 2 * (k / 32))) as u64;
                    for i in 0..FEW_MS.len() {
                        let m = FEW_MS[i];
                        let mut c = Cand::new(
                            ENG[i],
                            Layout::Chunked,
                            Box::new(move |s, n, y| {
                                let w = &ws[n % ws.len()];
                                q8.enqueue_q8_0_gemv(s, &w.0, &w.1, &inp.x[i], m, y)
                            }),
                        );
                        c.cols = m;
                        c.base = if i == 0 { "" } else { ENG[0] };
                        cands.push(c);
                    }
                    oracle = few_oracle(rows, move |s, c, y| {
                        q8.enqueue_q8_0_gemv(s, &ws[0].0, &ws[0].1, &cols[c], 1, y)
                    });
                    (bytes, copies_n, walk) = (b, ws.len(), "mcol");
                }
            }
            let cfg = Cfg {
                label,
                family: "fewrow",
                fields: format!(
                    "shape={} rows={rows} k={k} experts=1 slots={FEW_COLS} ms=1,2,4,8 walk={walk}",
                    FEW_GROUPS
                        .iter()
                        .find(|g| g.2 == rows
                            && g.3 == k
                            && std::mem::discriminant(&g.1) == std::mem::discriminant(&kind))
                        .map_or("", |g| g.0),
                ),
                rows,
                m: FEW_COLS,
                copies: copies_n,
                bytes,
                rule: "-",
            };
            run_cfg_with(gpu, o, &cfg, &cands, Some(&oracle))?;
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
        fewrow(&gpu, &o)?;
        Ok(())
    }
}
