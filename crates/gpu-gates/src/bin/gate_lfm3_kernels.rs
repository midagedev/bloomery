//! LFM3's kernels, model-free: the head-128 flash at a group of four
//! query heads a key head (16 query heads over 4 key heads), with a window of
//! 256 and with none, on a plane (`rows == ctx`) and on a ring (`rows < ctx`),
//! through the binding's launch (`attention_h128::Kernels::enqueue_attention`),
//! with no sink term (a group of four folds none); the gated short conv and the per-head output
//! gate.
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
//!
//! The short conv is held to its host rule bit for bit (`chain::shortconv::short_conv`,
//! one arithmetic order), the per-head gate to the card's own sigmoid rule and within a band
//! to the chain's, whose `exp` is the host libm's.
//!
//! - `short_conv_twin`: the kernel against `short_conv` bit for bit, outputs and ring after every call,
//!   at d = 2048 and the taps of LFM3 (7) and of LFM2.5 (3): one decode token at position 0; decode
//!   calls on past the ring's wrap; a prompt chunk of 37 from 0, decode calls after it, a chunk
//!   shorter than the conv's reach, and a second chunk of 37 from the middle of the sequence. The
//!   rings start NaN, so a read of a slot nothing wrote poisons the output.
//! - `short_conv_fault`: a NaN in `x`, a NaN in `C` and a position that is not consecutive each raise
//!   the fault word; the NaN outputs are exactly the tokens that read the value, every other
//!   output equals the twin's bit for bit.
//! - `out_gate_head_twin`: `attn_out_gate_sigmoid_head` at 16 heads of 128, one token and 37, against
//!   the card's host rule over the gate spread to every value of its head, bit for bit, and against
//!   `chain::out_gate` within a band; a gate kind the entry does not take is refused by name; a
//!   non-finite gate makes its whole head NaN and a non-finite value only its own.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_lfm3_kernels", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::Gpu;
    use bloomery_gpu::attention_h128::Kernels;
    use bloomery_gpu::fault::{Fault, FaultSite, LAYER_NONE};
    use bloomery_gpu::flash_gqa::{HEAD, KEY_TILE, SEG_KEYS, seg_span, window_segments};
    use bloomery_gpu::flash_gqa_h128_swa::{GROUP_4, H128SwaArgs};
    use bloomery_gpu::linear::out_gate::{OutGateArgs, OutGateKernels, out_gate_host};
    use bloomery_gpu::linear::shortconv::{ShortConvArgs, ShortConvKernels};
    use bloomery_gpu_gates::flash_band::{exact_attn, scalar_bound};
    use bloomery_gpu_gates::{GateError, activations, bits_equal, checks_failed, par_map, verdict};
    use chain::linear::Ring;
    use chain::out_gate::out_gate;
    use chain::shortconv::short_conv;
    use cuda_core::DeviceBuffer;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use model::arch::models::OutGate;

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
    /// LFM3-1.7B's hidden width, `lfm3.embedding_length`: the short conv's channels.
    const D: usize = 2048;
    /// LFM3-1.7B's `lfm3.shortconv.l_cache`: the taps of the short conv's filter.
    const TAPS: usize = 7;
    /// LFM2.5-24B-A2B's taps, the short conv's second customer, at the same width.
    const TAPS_3: usize = 3;

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

    /// What one conv launch left: its output rows and the fault word.
    struct Launched {
        y: Vec<f32>,
        fault: Option<Fault>,
    }

    /// One sequence's short conv on both sides: the filters `[D][taps]`, and the ring as the
    /// twin keeps it (`host`) and as the card does (`dev`).
    struct Conv {
        ring: Ring,
        w: Vec<f32>,
        wd: DeviceBuffer<f32>,
        host: Vec<f32>,
        dev: DeviceBuffer<f32>,
    }

    impl Conv {
        /// A conv of `taps` taps whose ring holds `ring0` (`[rows][D]`) on both sides.
        fn new(gpu: &Gpu, taps: usize, ring0: &[f32]) -> Result<Conv, GateError> {
            let w = activations(taps, D, 5 + taps as u32);
            let s = gpu.stream();
            Ok(Conv {
                ring: Ring::new(taps),
                wd: DeviceBuffer::from_host(s, &w)?,
                w,
                host: ring0.to_vec(),
                dev: DeviceBuffer::from_host(s, ring0)?,
            })
        }

        /// The twin's call: the rows and, on success, the ring.
        fn twin(&mut self, bcx: &[f32], pos: &[u32]) -> Result<Vec<f32>, GateError> {
            let mut y = vec![0.0f32; pos.len() * D];
            short_conv(self.ring, D, bcx, &self.w, pos, &mut self.host, &mut y)?;
            Ok(y)
        }

        /// The kernel's call on the card, with a clean fault word before it.
        fn launch(
            &mut self,
            gpu: &Gpu,
            kernels: &ShortConvKernels,
            bcx: &[f32],
            pos: &[u32],
        ) -> Result<Launched, GateError> {
            let s = gpu.stream();
            let m = pos.len();
            let bd = DeviceBuffer::from_host(s, bcx)?;
            let pd = DeviceBuffer::from_host(s, pos)?;
            let mut y = DeviceBuffer::<f32>::zeroed(s, m * D)?;
            gpu.take_fault()?;
            kernels.enqueue_short_conv(
                s,
                ShortConvArgs {
                    ring: self.ring,
                    d: D,
                    m,
                    bcx: &bd,
                    weights: &self.wd,
                    pos: &pd,
                    fault: gpu.unlabelled_sink(),
                    y: &mut y,
                    state: &mut self.dev,
                },
            )?;
            s.synchronize()?;
            Ok(Launched {
                y: y.to_host_vec(s)?,
                fault: gpu.take_fault()?,
            })
        }

        /// The card's ring.
        fn ring_on_card(&self, gpu: &Gpu) -> Result<Vec<f32>, GateError> {
            Ok(self.dev.to_host_vec(gpu.stream())?)
        }
    }

    /// The positions `from .. from + m` as the words the rope reads.
    fn positions(from: usize, m: usize) -> Vec<u32> {
        (from..from + m).map(|p| p as u32).collect()
    }

    /// A ring of `rows` slots every value of which is NaN: a slot nothing wrote, whose read
    /// poisons the output.
    fn nan_ring(taps: usize) -> Vec<f32> {
        vec![f32::NAN; Ring::new(taps).values(D)]
    }

    /// One case of `short_conv_twin`: `calls` of `(first position, tokens)` one after another on
    /// `conv`, each held to the twin's rows and ring bit for bit with no fault. Whether all were.
    fn twin_case(
        gpu: &Gpu,
        kernels: &ShortConvKernels,
        (conv, name): (&mut Conv, &str),
        calls: &[(usize, usize)],
        seed: u32,
    ) -> Result<bool, GateError> {
        let taps = conv.ring.taps();
        let mut ok = true;
        for (n, &(from, m)) in calls.iter().enumerate() {
            let bcx = activations(3 * D, m, seed + n as u32);
            let pos = positions(from, m);
            let want = conv.twin(&bcx, &pos)?;
            let got = conv.launch(gpu, kernels, &bcx, &pos)?;
            let (rows_ok, ring_ok) = (
                bits_equal(&got.y, &want),
                bits_equal(&conv.ring_on_card(gpu)?, &conv.host),
            );
            let clean = got.fault.is_none();
            if !(rows_ok && ring_ok && clean) {
                println!(
                    "short_conv_twin taps {taps} {name}: call {n} (from {from}, {m} tokens): rows \
                     {rows_ok} ring {ring_ok} fault {:?} FAIL",
                    got.fault
                );
            }
            ok &= rows_ok && ring_ok && clean;
        }
        println!(
            "short_conv_twin taps {taps} d {D} {name}: {} calls, rows and ring == twin bit for bit, no \
             fault {}",
            calls.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// `short_conv_twin`. Mutants that turn it red: the taps read newest first (the filters are
    /// asymmetric); `B ⊙ x` taken as `x`, or `C` left out; the ring read at the sequence start
    /// (its slots are NaN) or at the slot `pos mod taps`, or with `rows = taps`; the ring written at
    /// the slot `t mod rows`, or one token short (`m.saturating_sub(rows) + 1`); the ring written
    /// before a token that reads it (the chunk of 37 and the chunk from the middle read it at their
    /// first tokens and overwrite it at their last), or by a unit that does not read it; the units
    /// split at the wrong token (at 7 taps the chunk of 3 is one unit, the chunks of 37 are thirty-two).
    fn short_conv_twin(gpu: &Gpu, kernels: &ShortConvKernels) -> Result<bool, GateError> {
        let mut all = true;
        for taps in [TAPS, TAPS_3] {
            let rows = Ring::new(taps).rows();
            // Decode calls from position 0, on past the ring's first wrap.
            let mut a = Conv::new(gpu, taps, &nan_ring(taps))?;
            all &= twin_case(
                gpu,
                kernels,
                (&mut a, "decode at position 0"),
                &[(0, 1)],
                100,
            )?;
            let on: Vec<(usize, usize)> = (1..rows + 6).map(|p| (p, 1)).collect();
            all &= twin_case(gpu, kernels, (&mut a, "decode past the wrap"), &on, 200)?;
            // A prompt chunk, decode after it, a chunk shorter than the conv's reach, and a second
            // chunk from the middle of the sequence.
            let mut b = Conv::new(gpu, taps, &nan_ring(taps))?;
            all &= twin_case(
                gpu,
                kernels,
                (&mut b, "prompt chunk of 37 from 0"),
                &[(0, 37)],
                300,
            )?;
            let after: Vec<(usize, usize)> = (37..37 + rows + 2).map(|p| (p, 1)).collect();
            let from = 37 + rows + 2;
            all &= twin_case(
                gpu,
                kernels,
                (&mut b, "decode after the chunk"),
                &after,
                400,
            )?;
            all &= twin_case(gpu, kernels, (&mut b, "chunk of 3"), &[(from, 3)], 500)?;
            all &= twin_case(
                gpu,
                kernels,
                (&mut b, "chunk of 37 from mid-sequence"),
                &[(from + 3, 37)],
                600,
            )?;
        }
        Ok(all)
    }

    /// `short_conv_fault`: the call over 20 tokens from position 30 at LFM3's taps, its ring
    /// holding finite values, planted with a fault. `nan` lists the `(token, channel)` outputs
    /// that must be NaN, every output that reads the value; every other output is the clean
    /// twin's bit for bit, and the fault word is the conv's site alone. The tokens straddle the
    /// units of the launch (token 9 is past the first six). Mutants that turn it red: the finite
    /// test on the accumulator where the product with `C` is (the `C` plant), the position check
    /// left out, a NaN written as the plain value, the fault raised by unit 0 alone.
    fn short_conv_fault(gpu: &Gpu, kernels: &ShortConvKernels) -> Result<bool, GateError> {
        const M: usize = 20;
        const FROM: usize = 30;
        const T0: usize = 9;
        const C0: usize = 100;
        let reach = TAPS - 1;
        let ring0 = activations(D, Ring::new(TAPS).rows(), 9);
        let clean_bcx = activations(3 * D, M, 77);
        let clean_pos = positions(FROM, M);
        let clean_y = Conv::new(gpu, TAPS, &ring0)?.twin(&clean_bcx, &clean_pos)?;

        type Patch = fn(&mut [f32], &mut [u32]);
        struct Plant {
            what: &'static str,
            bad: Patch,
            nan: Vec<(usize, usize)>,
        }
        let plants = [
            Plant {
                what: "NaN in x",
                bad: |bcx, _| bcx[T0 * 3 * D + 2 * D + C0] = f32::NAN,
                // The product of row T0 is read by the tokens T0 ..= T0 + reach.
                nan: (T0..=T0 + reach).map(|t| (t, C0)).collect(),
            },
            Plant {
                what: "NaN in C",
                bad: |bcx, _| bcx[T0 * 3 * D + D + C0] = f32::NAN,
                nan: vec![(T0, C0)],
            },
            Plant {
                what: "position not consecutive",
                bad: |_, pos| pos[T0] += 1,
                nan: (0..D).map(|c| (T0, c)).collect(),
            },
        ];
        let mut all = true;
        for plant in plants {
            let (mut bcx, mut pos) = (clean_bcx.clone(), clean_pos.clone());
            (plant.bad)(&mut bcx, &mut pos);
            let got = Conv::new(gpu, TAPS, &ring0)?.launch(gpu, kernels, &bcx, &pos)?;
            let want_fault = Fault::at(LAYER_NONE, FaultSite::LinearConv);
            let fault_ok = got.fault == Some(want_fault);
            let mut wrong = 0usize;
            for t in 0..M {
                for c in 0..D {
                    let (g, w) = (got.y[t * D + c], clean_y[t * D + c]);
                    let right = if plant.nan.contains(&(t, c)) {
                        g.is_nan()
                    } else {
                        g.to_bits() == w.to_bits()
                    };
                    wrong += usize::from(!right);
                }
            }
            let ok = fault_ok && wrong == 0;
            println!(
                "short_conv_fault {}: fault {:?} (want {want_fault:?}), {wrong} outputs wrong {}",
                plant.what,
                got.fault,
                verdict(ok)
            );
            all &= ok;
        }
        Ok(all)
    }

    /// One launch of the per-head out gate (`ProjHead`, or the kind `kind`) over `m` tokens of
    /// `N_HEAD` heads: the output and the fault word.
    fn head_gate_launch(
        gpu: &Gpu,
        kernels: &OutGateKernels,
        kind: OutGate,
        (attn, gate): (&[f32], &[f32]),
        m: usize,
    ) -> Result<Launched, GateError> {
        let s = gpu.stream();
        let ad = DeviceBuffer::from_host(s, attn)?;
        let gd = DeviceBuffer::from_host(s, gate)?;
        let mut y = DeviceBuffer::<f32>::zeroed(s, attn.len())?;
        gpu.take_fault()?;
        kernels.enqueue_out_gate(
            s,
            OutGateArgs {
                kind,
                attn: &ad,
                gate: &gd,
                n_head: N_HEAD,
                m,
                fault: gpu.unlabelled_sink(),
                y: &mut y,
            },
        )?;
        s.synchronize()?;
        Ok(Launched {
            y: y.to_host_vec(s)?,
            fault: gpu.take_fault()?,
        })
    }

    /// The gate of value `i` of a `[m][N_HEAD][HEAD]` tensor, one value a head: the gate spread
    /// to a value each.
    fn spread(gate: &[f32]) -> Vec<f32> {
        (0..gate.len() * HEAD).map(|i| gate[i / HEAD]).collect()
    }

    /// `out_gate_head_twin`. Mutants that turn it red: the gate index `i % N_HEAD`, `i / HEAD %
    /// N_HEAD` (the token dropped) or `i` (one gate a value, reading past the gate), the sigmoid of
    /// `-g`, and the finite test on `g` left out (the infinite gate passes the value through).
    fn out_gate_head_twin(gpu: &Gpu, kernels: &OutGateKernels) -> Result<bool, GateError> {
        let mut all = true;
        for m in [1usize, 37] {
            let n = m * N_HEAD * HEAD;
            let attn = activations(HEAD, m * N_HEAD, 41 + m as u32);
            let gate: Vec<f32> = activations(1, m * N_HEAD, 43 + m as u32)
                .into_iter()
                .map(|g| g * 3.0)
                .collect();
            let got = head_gate_launch(gpu, kernels, OutGate::ProjHead, (&attn, &gate), m)?;
            let want = out_gate_host(&attn, &spread(&gate));
            let exact = bits_equal(&got.y, &want);
            // The chain's rule takes the host libm's `exp`, so the card is held to it within a
            // band [derived]: the card's polynomial `exp` is a few ulp off, the sum, the division
            // and the product round half an ulp each, on both sides.
            let mut chain_y = vec![0.0f32; n];
            out_gate(OutGate::ProjHead, &attn, &gate, N_HEAD, HEAD, &mut chain_y)?;
            let in_band = got
                .y
                .iter()
                .zip(&chain_y)
                .all(|(&g, &c)| (g - c).abs() <= 1e-6 * c.abs() + 1e-30);
            let clean = got.fault.is_none();
            println!(
                "out_gate_head_twin m {m} n_head {N_HEAD}: {n} values == card rule bit for bit {}, \
                 within 1e-6 of the chain's {}, no fault {}",
                verdict(exact),
                verdict(in_band),
                verdict(clean)
            );
            all &= exact && in_band && clean;
        }

        // A gate kind the entry does not take is refused before any launch.
        for kind in [OutGate::None, OutGate::Packed] {
            let r = head_gate_launch(
                gpu,
                kernels,
                kind,
                (&[0.0; HEAD * N_HEAD], &[0.0; N_HEAD]),
                1,
            );
            let pass = r
                .as_ref()
                .err()
                .is_some_and(|e| e.to_string().contains("is not a gate of its own"));
            println!(
                "out_gate_head_twin refusal {kind:?}: {} {}",
                r.map_or_else(|e| e.to_string(), |_| "accepted".to_string()),
                verdict(pass)
            );
            all &= pass;
        }

        // Faults: a non-finite gate makes the whole head NaN, a non-finite value only itself.
        const M: usize = 2;
        let attn = activations(HEAD, M * N_HEAD, 59);
        let gate = activations(1, M * N_HEAD, 61);
        let want = out_gate_host(&attn, &spread(&gate));
        let head = |t: usize, h: usize| (t * N_HEAD + h) * HEAD;
        for (what, bad_attn, bad_gate, nan) in [
            (
                "NaN gate",
                None,
                Some((head(1, 3) / HEAD, f32::NAN)),
                head(1, 3)..head(1, 3) + HEAD,
            ),
            (
                "infinite gate",
                None,
                Some((head(0, 15) / HEAD, f32::INFINITY)),
                head(0, 15)..head(0, 15) + HEAD,
            ),
            (
                "NaN value",
                Some((head(1, 7) + 5, f32::NAN)),
                None,
                head(1, 7) + 5..head(1, 7) + 6,
            ),
        ] {
            let (mut a, mut g) = (attn.clone(), gate.clone());
            if let Some((i, v)) = bad_attn {
                a[i] = v;
            }
            if let Some((i, v)) = bad_gate {
                g[i] = v;
            }
            let got = head_gate_launch(gpu, kernels, OutGate::ProjHead, (&a, &g), M)?;
            let fault_ok = got.fault == Some(Fault::at(LAYER_NONE, FaultSite::F32Product));
            let wrong = got
                .y
                .iter()
                .zip(&want)
                .enumerate()
                .filter(|&(i, (g, w))| {
                    if nan.contains(&i) {
                        !g.is_nan()
                    } else {
                        g.to_bits() != w.to_bits()
                    }
                })
                .count();
            let pass = fault_ok && wrong == 0;
            println!(
                "out_gate_head_twin fault {what}: fault {:?}, {wrong} outputs wrong {}",
                got.fault,
                verdict(pass)
            );
            all &= pass;
        }
        Ok(all)
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
        let short = ShortConvKernels::load(gpu.context())?;
        let gate = OutGateKernels::load(gpu.context())?;
        ok &= short_conv_twin(&gpu, &short)?;
        ok &= short_conv_fault(&gpu, &short)?;
        ok &= out_gate_head_twin(&gpu, &gate)?;
        if ok { Ok(()) } else { Err(checks_failed()) }
    }
}
