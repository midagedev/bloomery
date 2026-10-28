//! LFM3's attention kernel, model-free: the head-128 flash at a group of four
//! query heads a key head (16 query heads over 4 key heads), with a window of
//! 256 and with none, on a plane (`rows == ctx`) and on a ring (`rows < ctx`),
//! through the binding's launch (`attention_h128::Kernels::enqueue_attention`),
//! with no sink term: a group of four folds none.
//!
//! The flash is not bit-identical to an f64 softmax; it is held to the
//! first-order band of `flash_band::scalar_bound` (the scalar pass's, the one
//! `gate_qwen3moe_flash` derives) around `flash_band::exact_attn`: the exact
//! attention in f64 of the same f16 keys and values and the f32 query, over
//! the keys of the gate's own window — the last `window` of a row's live
//! count, or all of them — not the kernel's `window_cut`, so a mutant of the
//! cut moves the kernel and not the reference. The keys sit in a plane or a
//! ring the gate lays out from its logical planes; every row the launch must
//! not read (past the count on a plane, not resident on a ring) is an f16
//! NaN, so a read of one poisons its output.
//!
//! - `flash_h128_group4_window`: every head of every row inside its band, at
//!   counts around the tile (32), segment (64), window (256) and ring (768)
//!   edges for one row a launch, and at consecutive counts for a 64-row prompt
//!   chunk that crosses a window edge or the ring's wrap; no fault word.
//! - `flash_h128_group4_is_group8`: a group-of-four launch over `2 · n_kv` key
//!   heads that hold one plane twice equals the group-of-eight launch over that
//!   plane bit for bit, with no sinks, on the plane and the ring: the per-head
//!   arithmetic is the group's alone, so the block's head indexing is the only
//!   thing that differs.
//! - `flash_h128_no_sink_term`: a launch with no sinks equals one whose sinks
//!   are `-inf` (neutral), at the group of eight.
//! - `flash_h128_group_refusal`: a group other than eight and four is refused
//!   by name on the plane and on the ring before any launch.
//! - `flash_h128_group4_sinks_refusal`: a group of four with sinks is refused
//!   by name on the plane and on the ring before any launch.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_lfm3_kernels", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::attention_h128::Kernels;
    use bloomery_gpu::fault::Fault;
    use bloomery_gpu::flash_gqa::{HEAD, KEY_TILE, SEG_KEYS, seg_span, window_segments};
    use bloomery_gpu::flash_gqa_h128_swa::{GROUP_4, H128SwaArgs};
    use bloomery_gpu_gates::flash_band::{exact_attn, scalar_bound};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, par_map, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};

    /// An f16 NaN: the rows a launch must not read.
    const NAN16: u16 = 0x7e00;
    /// The seeded query's scale over [`activations`]' [-1, 1): logits of a few
    /// units, so the softmax weights spread over several orders of magnitude.
    const Q_SCALE: f32 = 3.0;
    /// LFM3-1.7B's heads: 16 query heads over 4 key heads.
    const N_HEAD: usize = 16;
    const N_KV: usize = 4;
    /// A prompt chunk's rows in the band clause, and in the group clause.
    const CHUNK: usize = 64;
    const SMALL_CHUNK: usize = 8;

    /// A layer kind and the cache it runs on. `rows == ctx` is a plane; a
    /// smaller `rows` is the ring a window layer holds past its window plus a
    /// prompt batch (768 rows at a window of 256).
    struct Kind {
        name: &'static str,
        window: usize,
        ctx: usize,
        rows: usize,
        /// First positions of the prompt chunks.
        starts: &'static [usize],
    }

    const KINDS: [Kind; 3] = [
        Kind {
            name: "full, plane",
            window: 0,
            ctx: 1300,
            rows: 1300,
            starts: &[0, 700, 1236],
        },
        Kind {
            name: "sliding 256, plane",
            window: 256,
            ctx: 600,
            rows: 600,
            starts: &[0, 224, 536],
        },
        Kind {
            name: "sliding 256, ring 768",
            window: 256,
            ctx: 3000,
            rows: 768,
            starts: &[0, 760, 1500, 2936],
        },
    ];

    fn score_scale() -> f32 {
        1.0 / (HEAD as f32).sqrt()
    }

    /// The window's first key by the gate's own rule: the last `window` of
    /// `count` live keys, or all of them with no window or a shorter row.
    fn first_of(count: usize, window: usize) -> usize {
        if window == 0 || count < window {
            0
        } else {
            count - window
        }
    }

    /// Logical key and value planes, `[n_kv][ctx][HEAD]`: f16 bits and the
    /// f32 values the reference reads.
    struct Planes {
        ctx: usize,
        k: Vec<u16>,
        v: Vec<u16>,
        kf: Vec<f32>,
        vf: Vec<f32>,
    }

    fn planes(n_kv: usize, ctx: usize, seed: u32) -> Planes {
        let bits = |s: u32| -> Vec<u16> {
            activations(HEAD, n_kv * ctx, s)
                .into_iter()
                .map(f32_to_f16_bits)
                .collect()
        };
        let (k, v) = (bits(seed), bits(seed + 1));
        let wide = |b: &[u16]| -> Vec<f32> { b.iter().copied().map(half_to_f32).collect() };
        Planes {
            ctx,
            kf: wide(&k),
            vf: wide(&v),
            k,
            v,
        }
    }

    /// Positions `[lo, hi)` of every key head of `bits` as a device plane of
    /// `rows` rows a key head, position `p` in row `p mod rows`; every other
    /// row is [`NAN16`].
    fn layout(
        bits: &[u16],
        n_kv: usize,
        ctx: usize,
        rows: usize,
        (lo, hi): (usize, usize),
    ) -> Vec<u16> {
        let mut out = vec![NAN16; n_kv * rows * HEAD];
        for kh in 0..n_kv {
            for p in lo..hi {
                let src = (kh * ctx + p) * HEAD;
                let dst = (kh * rows + p % rows) * HEAD;
                out[dst..dst + HEAD].copy_from_slice(&bits[src..src + HEAD]);
            }
        }
        out
    }

    /// The positions a launch whose last live count is `hi` holds: the whole
    /// prefix on a plane, the last `rows` on a ring.
    fn resident(hi: usize, rows: usize) -> (usize, usize) {
        (hi.saturating_sub(rows), hi)
    }

    /// One launch: `counts.len()` rows of `n_head` query heads over `n_kv` key
    /// heads, the planes `[n_kv][rows][HEAD]`.
    struct Case<'a> {
        n_kv: usize,
        n_head: usize,
        ctx: usize,
        rows: usize,
        window: usize,
        q: &'a [f32],
        k: &'a [u16],
        v: &'a [u16],
        counts: &'a [u32],
        sinks: Option<&'a [f32]>,
    }

    /// The launch's output and the fault word it left.
    fn launch(
        gpu: &Gpu,
        kernels: &Kernels,
        c: &Case<'_>,
    ) -> Result<(Vec<f32>, Option<Fault>), GateError> {
        let s = gpu.stream();
        let m = c.counts.len();
        let segs = window_segments(c.window);
        let q = DeviceBuffer::from_host(s, c.q)?;
        let kc = DeviceBuffer::from_host(s, c.k)?;
        let vc = DeviceBuffer::from_host(s, c.v)?;
        let n_keys = DeviceBuffer::from_host(s, c.counts)?;
        let sinks = c.sinks.map(|h| DeviceBuffer::from_host(s, h)).transpose()?;
        let mut part_v = DeviceBuffer::<f32>::zeroed(s, m * c.n_head * segs * HEAD)?;
        let mut part_ms = DeviceBuffer::<f32>::zeroed(s, m * c.n_head * segs * 2)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, m * c.n_head * HEAD)?;
        gpu.take_fault()?;
        kernels.enqueue_attention(
            s,
            H128SwaArgs {
                q: &q,
                kc: &kc,
                vc: &vc,
                n_keys: &n_keys,
                scale: score_scale(),
                n_kv: c.n_kv,
                ctx: c.ctx,
                m,
                window: c.window,
                sinks: sinks.as_ref(),
                part_v: &mut part_v,
                part_ms: &mut part_ms,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
            c.n_head,
            c.rows,
        )?;
        s.synchronize()?;
        let out = y.to_host_vec(s)?;
        Ok((out, gpu.take_fault()?))
    }

    /// The counts of one row a launch, around the tile, segment, window, ring
    /// and cache edges.
    fn counts_of(k: &Kind) -> Vec<usize> {
        let mut c = vec![
            1, 2, 31, 32, 33, 63, 64, 65, 255, 256, 257, 320, 511, 512, 513, 1000,
        ];
        c.extend([
            k.rows - 1,
            k.rows,
            k.rows + 1,
            2 * k.rows - 1,
            2 * k.rows,
            2 * k.rows + 1,
        ]);
        c.extend([k.ctx - 1, k.ctx]);
        c.retain(|&n| (1..=k.ctx).contains(&n));
        c.sort_unstable();
        c.dedup();
        c
    }

    /// Every (row, head) of `y` inside its band of the exact attention:
    /// whether all are, and the worst measured over bound.
    fn band(c: &Case<'_>, p: &Planes, y: &[f32]) -> (bool, f64) {
        let group = c.n_head / c.n_kv;
        let units = par_map(c.counts.len() * c.n_head, |i| {
            let (t, h) = (i / c.n_head, i % c.n_head);
            let count = c.counts[t] as usize;
            let first = first_of(count, c.window);
            let n = count - first;
            let span = seg_span(n, window_segments(c.window), SEG_KEYS);
            let base = ((h / group) * p.ctx + first) * HEAD;
            let (kf, vf) = (&p.kf[base..base + n * HEAD], &p.vf[base..base + n * HEAD]);
            let ex = exact_attn(
                &c.q[i * HEAD..(i + 1) * HEAD],
                kf,
                vf,
                HEAD,
                n,
                score_scale(),
            );
            let bound = scalar_bound(&ex, vf, HEAD, KEY_TILE, span);
            let (mut ok, mut worst) = (true, 0.0f64);
            for ((&got, &want), &b) in y[i * HEAD..(i + 1) * HEAD].iter().zip(&ex.o).zip(&bound) {
                let err = (f64::from(got) - want).abs();
                // NaN fails the comparison and reads as an infinite ratio.
                ok &= err <= b;
                worst = worst.max(if err.is_nan() { f64::INFINITY } else { err / b });
            }
            (ok, worst)
        });
        units
            .into_iter()
            .fold((true, 0.0f64), |(ok, w), (o, x)| (ok && o, w.max(x)))
    }

    fn query(m: usize, n_head: usize, seed: u32) -> Vec<f32> {
        activations(HEAD, m * n_head, seed)
            .into_iter()
            .map(|x| x * Q_SCALE)
            .collect()
    }

    /// `flash_h128_group4_window`.
    fn group4_window(gpu: &Gpu, kernels: &Kernels) -> Result<bool, GateError> {
        let mut all = true;
        for kind in &KINDS {
            let p = planes(N_KV, kind.ctx, 11);
            let (mut ok, mut worst, mut launches) = (true, 0.0f64, 0usize);
            let mut run = |counts: Vec<u32>, seed: u32| -> Result<(), GateError> {
                let hi = counts.iter().copied().max().map_or(0, |n| n as usize);
                let (k, v) = (
                    layout(&p.k, N_KV, kind.ctx, kind.rows, resident(hi, kind.rows)),
                    layout(&p.v, N_KV, kind.ctx, kind.rows, resident(hi, kind.rows)),
                );
                let q = query(counts.len(), N_HEAD, seed);
                let c = Case {
                    n_kv: N_KV,
                    n_head: N_HEAD,
                    ctx: kind.ctx,
                    rows: kind.rows,
                    window: kind.window,
                    q: &q,
                    k: &k,
                    v: &v,
                    counts: &counts,
                    sinks: None,
                };
                let (y, fault) = launch(gpu, kernels, &c)?;
                let (in_band, w) = band(&c, &p, &y);
                let clean = fault.is_none();
                if !(in_band && clean) {
                    println!(
                        "flash_h128_group4_window {}: counts {:?}..: band={in_band} (measured/bound \
                         {w:.3e}) fault={fault:?} FAIL",
                        kind.name,
                        &counts[..counts.len().min(3)]
                    );
                }
                ok &= in_band && clean;
                worst = worst.max(w);
                launches += 1;
                Ok(())
            };
            for n in counts_of(kind) {
                run(vec![n as u32], 3 + n as u32)?;
            }
            for &start in kind.starts {
                let counts: Vec<u32> = (start + 1..=start + CHUNK).map(|n| n as u32).collect();
                run(counts, 7 + start as u32)?;
            }
            println!(
                "flash_h128_group4_window {} 16/4 window {}: {launches} launches, band (measured/bound \
                 {worst:.3e}), no fault {}",
                kind.name,
                kind.window,
                verdict(ok)
            );
            all &= ok;
        }
        Ok(all)
    }

    /// The same plane `twice`: key head 0 then key head 1, for a launch of
    /// `2 · n_kv` key heads that holds one plane in both.
    fn twice(plane: &[u16]) -> Vec<u16> {
        plane.iter().chain(plane).copied().collect()
    }

    /// `flash_h128_group4_is_group8`: eight query heads over one key head at a
    /// group of eight, and the same eight over a plane held twice at a group
    /// of four, both with no sinks.
    fn group4_is_group8(gpu: &Gpu, kernels: &Kernels) -> Result<bool, GateError> {
        const HEADS: usize = 2 * GROUP_4;
        let mut all = true;
        for kind in &KINDS {
            let p = planes(1, kind.ctx, 19);
            let (mut ok, mut launches) = (true, 0usize);
            let mut starts: Vec<usize> = kind.starts.to_vec();
            starts.push(kind.rows - 1);
            for start in starts {
                let counts: Vec<u32> = (start + 1..=start + SMALL_CHUNK)
                    .filter(|&n| n <= kind.ctx)
                    .map(|n| n as u32)
                    .collect();
                let hi = counts.iter().copied().max().map_or(0, |n| n as usize);
                let one = (
                    layout(&p.k, 1, kind.ctx, kind.rows, resident(hi, kind.rows)),
                    layout(&p.v, 1, kind.ctx, kind.rows, resident(hi, kind.rows)),
                );
                let q = query(counts.len(), HEADS, 23 + start as u32);
                let case = |n_kv: usize, k: &[u16], v: &[u16]| -> Result<Vec<f32>, GateError> {
                    let (y, fault) = launch(
                        gpu,
                        kernels,
                        &Case {
                            n_kv,
                            n_head: HEADS,
                            ctx: kind.ctx,
                            rows: kind.rows,
                            window: kind.window,
                            q: &q,
                            k,
                            v,
                            counts: &counts,
                            sinks: None,
                        },
                    )?;
                    if fault.is_some() {
                        return Err(format!("fault {fault:?} at start {start}").into());
                    }
                    Ok(y)
                };
                let g8 = case(1, &one.0, &one.1)?;
                let g4 = case(2, &twice(&one.0), &twice(&one.1))?;
                let same = bits_equal(&g8, &g4) && g8.iter().all(|x| x.is_finite());
                if !same {
                    println!(
                        "flash_h128_group4_is_group8 {} start {start}: group of four != group of eight FAIL",
                        kind.name
                    );
                }
                ok &= same;
                launches += 1;
            }
            println!(
                "flash_h128_group4_is_group8 {}: {launches} launches, 2 x 4 heads == 8 heads bit \
                 for bit {}",
                kind.name,
                verdict(ok)
            );
            all &= ok;
        }
        Ok(all)
    }

    /// `flash_h128_no_sink_term`: no sinks equals sinks of `-inf`, at the group
    /// of eight (16 heads over 2 key heads).
    fn no_sink_term(gpu: &Gpu, kernels: &Kernels) -> Result<bool, GateError> {
        // PIN(2026-10-29): the group-of-four arm (16 heads over 4 key heads) is removed, because
        // a group of four takes no sinks and its launch with `-inf` sinks is refused
        // (`flash_h128_group4_sinks_refusal`). The merges that arm ran take `(n_head, m)` alone,
        // so the group-of-eight arm runs the same entries over the same 16 heads, and
        // `flash_h128_group4_is_group8` holds the no-sink walk of four equal to eight's.
        const N_KV8: usize = N_KV / 2;
        let mut all = true;
        for kind in &KINDS {
            let p = planes(N_KV8, kind.ctx, 29);
            let counts: Vec<u32> = (kind.ctx - SMALL_CHUNK + 1..=kind.ctx)
                .map(|n| n as u32)
                .collect();
            let hi = kind.ctx;
            let k = layout(&p.k, N_KV8, kind.ctx, kind.rows, resident(hi, kind.rows));
            let v = layout(&p.v, N_KV8, kind.ctx, kind.rows, resident(hi, kind.rows));
            let q = query(counts.len(), N_HEAD, 31);
            let neg_inf = vec![f32::NEG_INFINITY; N_HEAD];
            let run = |sinks: Option<&[f32]>| -> Result<Vec<f32>, GateError> {
                let (y, fault) = launch(
                    gpu,
                    kernels,
                    &Case {
                        n_kv: N_KV8,
                        n_head: N_HEAD,
                        ctx: kind.ctx,
                        rows: kind.rows,
                        window: kind.window,
                        q: &q,
                        k: &k,
                        v: &v,
                        counts: &counts,
                        sinks,
                    },
                )?;
                if fault.is_some() {
                    return Err(format!("fault {fault:?}").into());
                }
                Ok(y)
            };
            let none = run(None)?;
            let neutral = run(Some(&neg_inf))?;
            // A neutral sink changes nothing but the sign of a zero.
            let same = none.len() == neutral.len()
                && none.iter().zip(&neutral).all(|(a, b)| a == b)
                && none.iter().all(|x| x.is_finite());
            println!(
                "flash_h128_no_sink_term {} group {}: no sinks == sinks -inf {}",
                kind.name,
                N_HEAD / N_KV8,
                verdict(same)
            );
            all &= same;
        }
        Ok(all)
    }

    /// Whether `r` is a refusal naming `phrase`, printed under `clause`.
    fn refused(
        clause: &str,
        label: &str,
        r: Result<(Vec<f32>, Option<Fault>), GateError>,
        phrase: &str,
    ) -> bool {
        let pass = match &r {
            Err(e) => e.to_string().contains(phrase),
            Ok(_) => false,
        };
        println!(
            "{clause} {label}: {} {}",
            r.map_or_else(|e| e.to_string(), |_| "accepted".to_string()),
            verdict(pass)
        );
        pass
    }

    /// `flash_h128_group_refusal`: five query heads over one key head, on a
    /// plane and on a ring.
    fn group_refusal(gpu: &Gpu, kernels: &Kernels) -> Result<bool, GateError> {
        const HEADS: usize = 5;
        let phrase = "8 or 4 query heads a key head";
        let mut ok = true;
        for (label, window, ctx, rows) in
            [("plane", 0usize, 8usize, 8usize), ("ring", 256, 1024, 512)]
        {
            let k = vec![0u16; rows * HEAD];
            let q = vec![0.0f32; HEADS * HEAD];
            let counts = [1u32];
            let r = launch(
                gpu,
                kernels,
                &Case {
                    n_kv: 1,
                    n_head: HEADS,
                    ctx,
                    rows,
                    window,
                    q: &q,
                    k: &k,
                    v: &k,
                    counts: &counts,
                    sinks: None,
                },
            );
            ok &= refused("flash_h128_group_refusal", label, r, phrase);
        }
        Ok(ok)
    }

    /// `flash_h128_group4_sinks_refusal`: four query heads over one key head
    /// with per-head sinks, on a plane and on a ring.
    fn group4_sinks_refusal(gpu: &Gpu, kernels: &Kernels) -> Result<bool, GateError> {
        const HEADS: usize = GROUP_4;
        let phrase = "folds no per-head sinks";
        let mut ok = true;
        for (label, window, ctx, rows) in
            [("plane", 0usize, 8usize, 8usize), ("ring", 256, 1024, 512)]
        {
            let k = vec![0u16; rows * HEAD];
            let q = vec![0.0f32; HEADS * HEAD];
            let sinks = vec![0.0f32; HEADS];
            let counts = [1u32];
            let r = launch(
                gpu,
                kernels,
                &Case {
                    n_kv: 1,
                    n_head: HEADS,
                    ctx,
                    rows,
                    window,
                    q: &q,
                    k: &k,
                    v: &k,
                    counts: &counts,
                    sinks: Some(&sinks),
                },
            );
            ok &= refused("flash_h128_group4_sinks_refusal", label, r, phrase);
        }
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        println!("gate_lfm3_kernels: device {}", gpu.device_name()?);
        let kernels = Kernels::load(&gpu)?;
        let mut ok = group4_window(&gpu, &kernels)?;
        // PIN(2026-10-29): the sinks arm of `flash_h128_group4_is_group8` is removed with the group
        // of four's sinks: a group of four takes none, so there is no sinks launch of four to
        // compare with eight's. `flash_h128_group4_sinks_refusal` holds the refusal; the sinks
        // merge is the group of eight's, held against its reference by `gate_flint_attn`.
        ok &= group4_is_group8(&gpu, &kernels)?;
        ok &= no_sink_term(&gpu, &kernels)?;
        ok &= group_refusal(&gpu, &kernels)?;
        ok &= group4_sinks_refusal(&gpu, &kernels)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
