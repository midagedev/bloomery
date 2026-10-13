//! GPU gate for GLM-5.3-Flash's MLA layer on the card, with no model:
//! synthetic rows from fixed seeds at the file's shapes (64 heads, a
//! 512-value latent, the 128-value indexer key and gate, a stacked
//! projection row of 1536 + 512 + 128 + 128 values). Two new kernels
//! (`bloomery_gpu::latent`) and the V4.1 attention (`ds41_attn_seg`,
//! `ds41_attn_merge`) reused with no window, no sinks and the scale
//! `1/√256`: the latent cache is the compressed source, read as a prefix.
//!
//! Clauses:
//! 1. `latent`: the latent append of 1, 8 and 512 tokens at scattered
//!    positions into a cache of NaN rows — the f16 cache, every row, bit
//!    for bit against [`latent_append_host`], and no fault. At
//!    [`JUDGED_M`] tokens each rule mutant (the norm's ε, no gain, the next
//!    row, the latent one column over) must move at least one cached bit;
//!    an ε mutant moves about one cached value in a hundred or fewer, too
//!    few for one token's row.
//! 2. `index`: the index-key append the same way against
//!    [`index_key_append_host`]; mutants ε, no bias, no weight, gate and
//!    key swapped, the next row.
//! 3. `mqa`: the absorbed attention — 64 query heads over one latent cache,
//!    window 0, the cache a compressed prefix of `pos + 1` rows, sinks −∞,
//!    scale 1/16 — for one token at position 39 (one segment), eight tokens
//!    at positions 100 … 107 and one token at 2,050 (33 segments, the last
//!    of 3 keys), against the rule (the f16 query, the exact f64 dot rounded
//!    to f32 and scaled, the softmax and value sum in f64, no sink) within
//!    a band derived per case from the kernel's error model ([`band`]); a
//!    rerun bit-identical. On the first two cases each rule mutant (scale
//!    `1/√512`, the token's own key dropped, the query kept in f32) must
//!    sit past the band.
//! 4. `sink`: on the one-segment case the merge's fold of a −∞ sink is
//!    nothing: the output equals, bit for bit, the no-sink fold of the
//!    segment's partials read back from the card; the same launch with a
//!    finite sink (0) must not.
//! 5. `fault`: a NaN latent, a latent whose squares overflow, a position
//!    past the cache, a NaN key, a key whose variance overflows, a gate past
//!    f16's range and a position past the index cache — each raises its
//!    site in its layer (`CacheValue`, `CachePos`) alone, the cache equals
//!    the host rule's (the planted row non-finite, or untouched), and the
//!    word is clean after a clean launch.
//! 6. `shape`: the two new entries compile with no local depot.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!("gate_glm_mla: built without the `deepseek41` feature; see `just gate-gpu-glm-mla`.");
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_glm_mla", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::latent::{
        INDEX_HEAD, INDEX_ROW, IndexKeyArgs, LATENT, LatentAppendArgs, LatentKernels, Rows,
        index_key_append_host, latent_append_host,
    };
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu};
    use bloomery_gpu_deepseek41::attn::{self, AttnArgs, AttnKernels};
    use bloomery_gpu_gates::{
        GateError, NAN_F16, bits_equal, checks_failed, max_rel_err, no_local_depot, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::quant::{f32_to_f16_bits, half_to_f32};

    const _: () = assert!(LATENT == attn::LATENT);

    /// The stacked projection row: `attn_q_a` 1536, `attn_kv_a_mqa` 512,
    /// `indexer.attn_k` 128, `indexer_compressor_gate` 128.
    const STRIDE: usize = 1536 + LATENT + 2 * INDEX_HEAD;
    const KV_OFF: usize = 1536;
    const K_OFF: usize = KV_OFF + LATENT;
    const G_OFF: usize = K_OFF + INDEX_HEAD;
    /// The file's `layer_norm_rms_epsilon` (the latent's RMS norm) and
    /// `layer_norm_epsilon` (the indexer key's LayerNorm).
    const RMS_EPS: f32 = 1e-5;
    const LN_EPS: f32 = 1e-6;
    /// Rows of every cache here: the dense limit's 2,051 positions, rounded
    /// up to whole segments.
    const ROWS: usize = 33 * attn::SEG_KEYS;
    /// The token count at which the append clauses judge their mutants.
    const JUDGED_M: usize = 512;
    /// Query heads.
    const HEADS: usize = 64;
    /// `1/√key_length_mla`, `key_length_mla` = 256.
    const SCALE: f32 = 0.0625;
    /// f32's unit roundoff, 2^-24.
    const U: f64 = f32::EPSILON as f64 / 2.0;

    /// A 64-bit LCG (Knuth's MMIX constants); the high 24 bits as a unit
    /// float.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 40) as f32 / (1u64 << 24) as f32
        }
        fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| lo + (hi - lo) * self.unit()).collect()
        }
    }

    /// Indices where two f16 caches differ.
    fn diff_u16(a: &[u16], b: &[u16]) -> usize {
        a.iter().zip(b).filter(|(x, y)| x != y).count() + a.len().abs_diff(b.len())
    }

    /// The stacked rows of `m` tokens: the latent in [−4, 4), the key in
    /// [1, 5) (a mean the LayerNorm must remove), the gate in [−8, 8), the
    /// q_a part in [−1, 1) (read by no clause).
    fn stacked(m: usize, seed: u64) -> Vec<f32> {
        let mut r = Lcg(seed);
        let mut x = r.fill(m * STRIDE, -1.0, 1.0);
        for t in 0..m {
            let row = &mut x[t * STRIDE..(t + 1) * STRIDE];
            row[KV_OFF..K_OFF].copy_from_slice(&r.fill(LATENT, -4.0, 4.0));
            row[K_OFF..G_OFF].copy_from_slice(&r.fill(INDEX_HEAD, 1.0, 5.0));
            row[G_OFF..].copy_from_slice(&r.fill(INDEX_HEAD, -8.0, 8.0));
        }
        x
    }

    /// Scattered positions: token `t` at `3t + 5`.
    fn scattered(m: usize) -> Vec<u32> {
        (0..m).map(|t| (3 * t + 5) as u32).collect()
    }

    /// The layer's parameters: the latent norm's gain in [0.25, 2), the
    /// LayerNorm's weight in [0.5, 1.5) and bias in [−0.5, 0.5).
    struct Params {
        gain: Vec<f32>,
        w: Vec<f32>,
        b: Vec<f32>,
    }

    impl Params {
        fn new(seed: u64) -> Params {
            let mut r = Lcg(seed);
            Params {
                gain: r.fill(LATENT, 0.25, 2.0),
                w: r.fill(INDEX_HEAD, 0.5, 1.5),
                b: r.fill(INDEX_HEAD, -0.5, 0.5),
            }
        }
    }

    struct Ctx<'a> {
        gpu: &'a Gpu,
        lk: &'a LatentKernels,
        ak: &'a AttnKernels,
    }

    impl Ctx<'_> {
        fn stream(&self) -> &CudaStream {
            self.gpu.stream()
        }

        /// One latent append on the card into a cache of `init` rows,
        /// read back.
        fn latent(
            &self,
            sink: FaultSink,
            x: &[f32],
            p: &Params,
            pos: &[u32],
            init: &[u16],
        ) -> Result<Vec<u16>, GateError> {
            let s = self.stream();
            let xb = DeviceBuffer::from_host(s, x)?;
            let gain = DeviceBuffer::from_host(s, &p.gain)?;
            let posb = DeviceBuffer::from_host(s, pos)?;
            let mut cache = DeviceTensor::upload(s, init, init.len() / LATENT, LATENT)?;
            self.lk.enqueue_latent_append(
                s,
                LatentAppendArgs {
                    rows: Rows {
                        x: &xb,
                        stride: STRIDE,
                        m: pos.len(),
                    },
                    off: KV_OFF,
                    gain: &gain,
                    pos: &posb,
                    eps: RMS_EPS,
                    fault: sink,
                    cache: &mut cache,
                },
            )?;
            s.synchronize()?;
            Ok(cache.buf().to_host_vec(s)?)
        }

        /// One index-key append on the card into a cache of `init` rows,
        /// read back.
        fn index(
            &self,
            sink: FaultSink,
            x: &[f32],
            p: &Params,
            pos: &[u32],
            init: &[u16],
        ) -> Result<Vec<u16>, GateError> {
            let s = self.stream();
            let xb = DeviceBuffer::from_host(s, x)?;
            let w = DeviceBuffer::from_host(s, &p.w)?;
            let b = DeviceBuffer::from_host(s, &p.b)?;
            let posb = DeviceBuffer::from_host(s, pos)?;
            let mut cache = DeviceTensor::upload(s, init, init.len() / INDEX_ROW, INDEX_ROW)?;
            self.lk.enqueue_index_key_append(
                s,
                IndexKeyArgs {
                    rows: Rows {
                        x: &xb,
                        stride: STRIDE,
                        m: pos.len(),
                    },
                    k_off: K_OFF,
                    g_off: G_OFF,
                    w: &w,
                    b: &b,
                    pos: &posb,
                    eps: LN_EPS,
                    fault: sink,
                    cache: &mut cache,
                },
            )?;
            s.synchronize()?;
            Ok(cache.buf().to_host_vec(s)?)
        }
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let lk = LatentKernels::load(gpu.context())?;
        let ak = AttnKernels::load(gpu.context())?;
        let cx = Ctx {
            gpu: &gpu,
            lk: &lk,
            ak: &ak,
        };
        println!(
            "gate_glm_mla: device {} — heads={HEADS} latent={LATENT} index_row={INDEX_ROW} \
             stride={STRIDE} cache_rows={ROWS} scale={SCALE}, synthetic",
            gpu.device_name()?
        );
        gpu.clear_fault()?;
        let mut failed = 0u32;
        let mut clauses = 0u32;
        let mut tally = |pass: bool| {
            clauses += 1;
            failed += u32::from(!pass);
        };
        for m in [1usize, 8, 512] {
            tally(latent_case(&cx, m)?);
        }
        for m in [1usize, 8, 512] {
            tally(index_case(&cx, m)?);
        }
        for case in MQA_CASES {
            tally(mqa_case(&cx, case)?);
        }
        tally(sink_case(&cx)?);
        for pass in fault_cases(&cx)? {
            tally(pass);
        }
        tally(no_local_depot(&[
            "latent_rms_append",
            "index_key_ln_append",
        ])?);
        let pass = failed == 0;
        println!(
            "gate_glm_mla: {clauses} clauses, {failed} failed — {}",
            verdict(pass)
        );
        if !pass {
            return Err(checks_failed());
        }
        Ok(())
    }

    // ------------------------------------------------------------ appends

    /// Clause 1 at `m` tokens.
    fn latent_case(cx: &Ctx<'_>, m: usize) -> Result<bool, GateError> {
        let x = stacked(m, 0x6c61_7400 + m as u64);
        let p = Params::new(0x7061_7200 + m as u64);
        let pos = scattered(m);
        let init = vec![NAN_F16; ROWS * LATENT];
        let card = cx.latent(cx.gpu.unlabelled_sink(), &x, &p, &pos, &init)?;
        let word = cx.gpu.take_fault()?;
        let host = |x: &[f32], off: usize, gain: &[f32], pos: &[u32], eps: f32| {
            let mut c = init.clone();
            let sites = latent_append_host(x, STRIDE, off, gain, pos, eps, &mut c);
            (c, sites)
        };
        let (want, sites) = host(&x, KV_OFF, &p.gain, &pos, RMS_EPS);
        let differ = diff_u16(&card, &want);
        let next: Vec<u32> = pos.iter().map(|&q| q + 1).collect();
        let mutants = [
            ("eps1e-6", host(&x, KV_OFF, &p.gain, &pos, 1e-6).0),
            ("no_gain", host(&x, KV_OFF, &[1.0; LATENT], &pos, RMS_EPS).0),
            ("next_row", host(&x, KV_OFF, &p.gain, &next, RMS_EPS).0),
            ("column+1", host(&x, KV_OFF + 1, &p.gain, &pos, RMS_EPS).0),
        ];
        let mut killed = true;
        let judged = m == JUDGED_M;
        let moved: Vec<String> = mutants
            .iter()
            .map(|(name, c)| {
                let n = diff_u16(&card, c);
                killed &= !judged || n > 0;
                format!("{name}={n}")
            })
            .collect();
        let pass = differ == 0 && sites == 0 && word.is_none() && killed;
        println!(
            "latent m={m} cache bits differ={differ}/{} host_sites={sites:#x} fault={} \
             mutants(moved bits{}): {} {}",
            card.len(),
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            if judged { ", each > 0" } else { ", reported" },
            moved.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause 2 at `m` tokens.
    fn index_case(cx: &Ctx<'_>, m: usize) -> Result<bool, GateError> {
        let x = stacked(m, 0x6964_7800 + m as u64);
        let p = Params::new(0x7069_7800 + m as u64);
        let pos = scattered(m);
        let init = vec![NAN_F16; ROWS * INDEX_ROW];
        let card = cx.index(cx.gpu.unlabelled_sink(), &x, &p, &pos, &init)?;
        let word = cx.gpu.take_fault()?;
        let host = |w: &[f32], b: &[f32], pos: &[u32], eps: f32| {
            let mut c = init.clone();
            let sites = index_key_append_host(&x, STRIDE, K_OFF, G_OFF, w, b, pos, eps, &mut c);
            (c, sites)
        };
        let (want, sites) = host(&p.w, &p.b, &pos, LN_EPS);
        let differ = diff_u16(&card, &want);
        let next: Vec<u32> = pos.iter().map(|&q| q + 1).collect();
        let mut swapped = want.clone();
        for row in swapped.chunks_mut(INDEX_ROW) {
            row.rotate_left(INDEX_HEAD);
        }
        let mutants = [
            ("eps1e-5", host(&p.w, &p.b, &pos, 1e-5).0),
            ("no_bias", host(&p.w, &[0.0; INDEX_HEAD], &pos, LN_EPS).0),
            ("no_weight", host(&[1.0; INDEX_HEAD], &p.b, &pos, LN_EPS).0),
            ("gate_first", swapped),
            ("next_row", host(&p.w, &p.b, &next, LN_EPS).0),
        ];
        let mut killed = true;
        let judged = m == JUDGED_M;
        let moved: Vec<String> = mutants
            .iter()
            .map(|(name, c)| {
                let n = diff_u16(&card, c);
                killed &= !judged || n > 0;
                format!("{name}={n}")
            })
            .collect();
        let pass = differ == 0 && sites == 0 && word.is_none() && killed;
        println!(
            "index m={m} cache bits differ={differ}/{} host_sites={sites:#x} fault={} \
             mutants(moved bits{}): {} {}",
            card.len(),
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            if judged { ", each > 0" } else { ", reported" },
            moved.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    // ---------------------------------------------------------- attention

    /// A case of clause 3: `tokens` tokens at positions `first ..`, and
    /// whether the rule mutants are judged on it.
    #[derive(Clone, Copy)]
    struct MqaCase {
        name: &'static str,
        tokens: usize,
        first: usize,
        mutants: bool,
    }

    const MQA_CASES: [MqaCase; 3] = [
        MqaCase {
            name: "one_segment",
            tokens: 1,
            first: 39,
            mutants: true,
        },
        MqaCase {
            name: "batch",
            tokens: 8,
            first: 100,
            mutants: true,
        },
        MqaCase {
            name: "depth",
            tokens: 1,
            first: 2050,
            mutants: false,
        },
    ];

    /// A case's host inputs: the latent cache (`ROWS` rows, f16 bits; the
    /// rows past the last position NaN), the query rows (`tokens · HEADS`
    /// of [`LATENT`]) and each token's visible count `pos + 1`.
    struct MqaInputs {
        cache: Vec<u16>,
        q: Vec<f32>,
        counts: Vec<usize>,
    }

    impl MqaInputs {
        /// Keys of unit RMS (uniform in [−√3, √3)), queries uniform in
        /// [−2.4, 2.4) (RMS 1.39): scaled logits of σ ≈ 2 [derived: 1.39 ·
        /// √512 / 16].
        fn new(c: MqaCase) -> MqaInputs {
            let n = c.first + c.tokens;
            let mut r = Lcg(0x6d71_6100 + c.first as u64);
            let s3 = 3.0f32.sqrt();
            let mut cache = vec![NAN_F16; ROWS * LATENT];
            for (h, v) in cache.iter_mut().zip(r.fill(n * LATENT, -s3, s3)) {
                *h = f32_to_f16_bits(v);
            }
            MqaInputs {
                cache,
                q: r.fill(c.tokens * HEADS * LATENT, -2.4, 2.4),
                counts: (0..c.tokens).map(|t| c.first + t + 1).collect(),
            }
        }
    }

    /// What the rule saw, for [`band`]: the RMS of the unscaled logits and
    /// the mean |value| over the visible keys.
    struct RuleStats {
        logit_rms: f64,
        mean_abs_v: f64,
    }

    /// Our rule for a launch: row `t·HEADS + h` sees the first `counts[t]`
    /// cache rows; the query rounded to f16 when `f16q`; each logit the
    /// exact f64 dot rounded to f32 and scaled in f32; the softmax and the
    /// value sum in f64, no sink.
    fn rule(
        q: &[f32],
        keys: &[f32],
        counts: &[usize],
        scale: f32,
        f16q: bool,
    ) -> (Vec<f32>, RuleStats) {
        let mut out = Vec::with_capacity(q.len());
        let (mut sq, mut nl) = (0.0f64, 0usize);
        for (t, &n) in counts.iter().enumerate() {
            for h in 0..HEADS {
                let row = &q[(t * HEADS + h) * LATENT..][..LATENT];
                let qr: Vec<f64> = row
                    .iter()
                    .map(|&v| {
                        f64::from(if f16q {
                            half_to_f32(f32_to_f16_bits(v))
                        } else {
                            v
                        })
                    })
                    .collect();
                let logits: Vec<f64> = (0..n)
                    .map(|j| {
                        let k = &keys[j * LATENT..][..LATENT];
                        let dot: f64 = k.iter().zip(&qr).map(|(&a, &b)| f64::from(a) * b).sum();
                        sq += dot * dot;
                        nl += 1;
                        f64::from((dot as f32) * scale)
                    })
                    .collect();
                let mx = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut denom = 0.0f64;
                let mut num = vec![0.0f64; LATENT];
                for (j, &s) in logits.iter().enumerate() {
                    let w = (s - mx).exp();
                    denom += w;
                    for (a, &v) in num.iter_mut().zip(&keys[j * LATENT..][..LATENT]) {
                        *a += w * f64::from(v);
                    }
                }
                out.extend(num.iter().map(|&a| (a / denom) as f32));
            }
        }
        let n = counts.iter().copied().max().unwrap_or(0);
        let mean_abs_v = keys[..n * LATENT]
            .iter()
            .map(|&v| f64::from(v.abs()))
            .sum::<f64>()
            / (n * LATENT) as f64;
        let stats = RuleStats {
            logit_rms: (sq / nl.max(1) as f64).sqrt(),
            mean_abs_v,
        };
        (out, stats)
    }

    /// The band on `max_rel_err(kernel, rule)` for a case [derived]. Per
    /// output, the kernel's distance from the rule is the sum of three
    /// terms, each a unit roundoff u times its operation's count and the
    /// size of what it rounds, E|v| the values' mean magnitude:
    /// - the logit: the tensor cores add the 512 exact f16 products to an
    ///   f32 accumulator in blocks of 8, and each block may drop up to an ulp
    ///   of the running sum toward zero — a bias, not a walk, so the bound
    ///   takes all `LATENT / 8` of them: `u·64·rms(q·k)`, scaled; it moves
    ///   the output by that times E|v − y| ≤ E|v|;
    /// - the weights: `ex2.approx` at 2^-22 relative, times E|v|;
    /// - the value sums: f32 FMAs over a segment's 64 keys, then one fold a
    ///   segment in the merge, as a walk: `u·(√64 + √segs)·E|v|`.
    ///
    /// The largest of the launch's outputs sits about four of these above
    /// the typical one (the tail of ~10^4–10^5 draws), and the metric
    /// divides by the rule's largest magnitude. The logit term dominates;
    /// were the accumulator's errors a walk of 2u a block, it would be a
    /// quarter of this, which is where the kernel is expected to sit.
    fn band(st: &RuleStats, segs: usize, y_max: f64) -> f64 {
        let blocks = (LATENT / 8) as f64;
        let logit = U * blocks * f64::from(SCALE) * st.logit_rms * st.mean_abs_v;
        let exp = 2f64.powi(-22) * st.mean_abs_v;
        let values = U * ((attn::SEG_KEYS as f64).sqrt() + (segs as f64).sqrt()) * st.mean_abs_v;
        4.0 * (logit + exp + values) / y_max
    }

    /// One attention launch's device buffers: window 0 (a zero-row view of
    /// the cache), the cache the compressed prefix, NaN partials and output.
    struct Mqa {
        q: DeviceBuffer<f32>,
        cache: DeviceTensor<u16>,
        vis: DeviceBuffer<u32>,
        sinks: DeviceBuffer<f32>,
        part_v: DeviceBuffer<f32>,
        part_ms: DeviceBuffer<f32>,
        y: DeviceBuffer<f32>,
        tokens: usize,
    }

    impl Mqa {
        fn new(s: &CudaStream, i: &MqaInputs, sink: f32) -> Result<Mqa, GateError> {
            let tokens = i.counts.len();
            let vis: Vec<u32> = i
                .counts
                .iter()
                .flat_map(|&n| [0, u32::try_from(n).unwrap_or(u32::MAX)])
                .collect();
            let rows = tokens * HEADS;
            let segs = attn::segments(0, ROWS);
            Ok(Mqa {
                q: DeviceBuffer::from_host(s, &i.q)?,
                cache: DeviceTensor::upload(s, &i.cache, ROWS, LATENT)?,
                vis: DeviceBuffer::from_host(s, &vis)?,
                sinks: DeviceBuffer::from_host(s, &[sink; HEADS])?,
                part_v: DeviceBuffer::from_host(
                    s,
                    &vec![f32::NAN; attn::partials_v_len(rows, segs)],
                )?,
                part_ms: DeviceBuffer::from_host(
                    s,
                    &vec![f32::NAN; attn::partials_ms_len(rows, segs)],
                )?,
                y: DeviceBuffer::from_host(s, &vec![f32::NAN; rows * LATENT])?,
                tokens,
            })
        }

        /// One eager launch on NaN partials and output, read back.
        fn run(&mut self, cx: &Ctx<'_>) -> Result<MqaOut, GateError> {
            let s = cx.stream();
            self.part_v
                .copy_from_host(s, &vec![f32::NAN; self.part_v.len()])?;
            self.part_ms
                .copy_from_host(s, &vec![f32::NAN; self.part_ms.len()])?;
            self.y.copy_from_host(s, &vec![f32::NAN; self.y.len()])?;
            // SAFETY: zero u16 at the address of the cache's own live
            // allocation, aligned for u16; the view is released below, before
            // the cache can drop.
            let window = unsafe {
                DeviceTensor::<u16>::window(
                    self.cache.buf().cu_deviceptr(),
                    0,
                    LATENT,
                    cx.gpu.context(),
                )
            };
            let r = cx.ak.enqueue(
                s,
                AttnArgs {
                    q: &self.q,
                    window: &window,
                    compressed: Some(&self.cache),
                    selected: None,
                    vis: &self.vis,
                    sinks: &self.sinks,
                    scale: SCALE,
                    tokens: self.tokens,
                    heads: HEADS,
                    part_v: &mut self.part_v,
                    part_ms: &mut self.part_ms,
                    y: &mut self.y,
                    fault: cx.gpu.unlabelled_sink(),
                },
            );
            DeviceTensor::release(window);
            r?;
            s.synchronize()?;
            Ok(MqaOut {
                y: self.y.to_host_vec(s)?,
                ms: self.part_ms.to_host_vec(s)?,
                pv: self.part_v.to_host_vec(s)?,
            })
        }
    }

    /// One launch's read-back: the output and the segment partials.
    struct MqaOut {
        y: Vec<f32>,
        ms: Vec<f32>,
        pv: Vec<f32>,
    }

    /// The cache rows as f32.
    fn widen(bits: &[u16]) -> Vec<f32> {
        bits.iter().map(|&b| half_to_f32(b)).collect()
    }

    /// Clause 3 for `c`.
    fn mqa_case(cx: &Ctx<'_>, c: MqaCase) -> Result<bool, GateError> {
        let i = MqaInputs::new(c);
        let mut l = Mqa::new(cx.stream(), &i, f32::NEG_INFINITY)?;
        let y = l.run(cx)?.y;
        let y2 = l.run(cx)?.y;
        let word = cx.gpu.take_fault()?;
        let keys = widen(&i.cache);
        let (ours, st) = rule(&i.q, &keys, &i.counts, SCALE, true);
        let y_max = ours.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let segs = attn::source_segments(*i.counts.iter().max().unwrap_or(&0));
        let band = band(&st, segs, f64::from(y_max)) as f32;
        let kernel_rel = max_rel_err(&y, &ours)?;
        let rerun = bits_equal(&y, &y2);
        let mut killed = true;
        let mut moved = Vec::new();
        if c.mutants {
            let dropped: Vec<usize> = i.counts.iter().map(|&n| n - 1).collect();
            for (name, (m, _)) in [
                (
                    "scale_1/sqrt512",
                    rule(&i.q, &keys, &i.counts, 1.0 / 512f32.sqrt(), true),
                ),
                ("own_key_dropped", rule(&i.q, &keys, &dropped, SCALE, true)),
                ("query_f32", rule(&i.q, &keys, &i.counts, SCALE, false)),
            ] {
                let rel = max_rel_err(&y, &m)?;
                killed &= rel > band;
                moved.push(format!("{name}={rel:.2e}"));
            }
        }
        let pass = kernel_rel <= band && rerun && word.is_none() && killed;
        println!(
            "mqa {} tokens={} pos={}..{} segments={segs} logit_rms={:.2} mean|v|={:.3} \
             max|y|={y_max:.3} kernel_rel={kernel_rel:.2e} band={band:.2e} ({:.2} of it) rerun={} \
             fault={} mutants(rel, each > band): [{}] {}",
            c.name,
            c.tokens,
            c.first,
            c.first + c.tokens - 1,
            st.logit_rms,
            st.mean_abs_v,
            kernel_rel / band,
            if rerun { "bit-identical" } else { "DIFFERS" },
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            moved.join(" "),
            verdict(pass)
        );
        Ok(pass)
    }

    /// Clause 4, on the one-segment case: every row's first segment holds
    /// its only live partial `(m, s, v)` and every other segment is neutral
    /// (`Σ exp = 0`). The merge folds it from `(−∞, 0, 0)` — `(m, 0·0 + s,
    /// 0·0 + v)` — and then the sink as the partial `(sink, 1, 0)`, whose
    /// weight `ex2((sink − m)·log2 e)` is +0 for a −∞ sink: `(m, 1·0 + s,
    /// 0·0 + acc)`. The replay does those FMAs on the host and `(1/s)·acc`;
    /// with no sink the output is that, bit for bit.
    fn sink_case(cx: &Ctx<'_>) -> Result<bool, GateError> {
        let c = MQA_CASES[0];
        let i = MqaInputs::new(c);
        let segs = attn::segments(0, ROWS);
        let replay = |ms: &[f32], pv: &[f32]| -> (Vec<f32>, bool) {
            let rows = c.tokens * HEADS;
            let mut y = Vec::with_capacity(rows * LATENT);
            let mut neutral = true;
            for r in 0..rows {
                for seg in 1..segs {
                    neutral &= ms[2 * (r * segs + seg) + 1] == 0.0;
                }
                let s1 = ms[2 * r * segs + 1];
                let s = 1.0f32.mul_add(0.0, 0.0f32.mul_add(0.0, s1));
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                for d in 0..LATENT {
                    let v = pv[r * segs * LATENT + d];
                    let acc = 0.0f32.mul_add(0.0, 0.0f32.mul_add(0.0, v));
                    y.push(inv * acc);
                }
            }
            (y, neutral)
        };
        let mut l = Mqa::new(cx.stream(), &i, f32::NEG_INFINITY)?;
        let MqaOut { y, ms, pv } = l.run(cx)?;
        let (want, neutral) = replay(&ms, &pv);
        let same = bits_equal(&y, &want);
        let mut lf = Mqa::new(cx.stream(), &i, 0.0)?;
        let MqaOut {
            y: yf,
            ms: msf,
            pv: pvf,
        } = lf.run(cx)?;
        let (want_f, _) = replay(&msf, &pvf);
        let finite_moved = yf
            .iter()
            .zip(&want_f)
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        let word = cx.gpu.take_fault()?;
        let pass = neutral && same && finite_moved > 0 && word.is_none();
        println!(
            "sink −inf: one live segment {neutral}, output = the no-sink fold of the partials bit \
             for bit {same}; mutant sink 0 moves {finite_moved}/{} values (> 0); fault={} {}",
            yf.len(),
            word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
            verdict(pass)
        );
        Ok(pass)
    }

    // -------------------------------------------------------------- faults

    /// Clause 5: each plant on four tokens at positions 5, 8, 11, 14 in its
    /// own layer.
    fn fault_cases(cx: &Ctx<'_>) -> Result<Vec<bool>, GateError> {
        let m = 4;
        let x = stacked(m, 0x6661_756c);
        let p = Params::new(0x6670);
        let pos = scattered(m);
        let mut out = Vec::new();
        // (layer, what, index cache?, planted token, the plant, site)
        type Plant = fn(&mut [f32], &mut [u32]);
        let plants: [(usize, &str, bool, usize, Plant, FaultSite); 7] = [
            (
                41,
                "latent NaN",
                false,
                1,
                |x, _| x[STRIDE + KV_OFF + 7] = f32::NAN,
                FaultSite::CacheValue,
            ),
            (
                42,
                "latent squares overflow from finite inputs",
                false,
                2,
                |x, _| x[2 * STRIDE + KV_OFF..2 * STRIDE + K_OFF].fill(1e20),
                FaultSite::CacheValue,
            ),
            (
                43,
                "latent position past the cache",
                false,
                3,
                |_, pos| pos[3] = ROWS as u32,
                FaultSite::CachePos,
            ),
            (
                44,
                "index key NaN",
                true,
                1,
                |x, _| x[STRIDE + K_OFF + 9] = f32::NAN,
                FaultSite::CacheValue,
            ),
            (
                45,
                "index key variance overflow from finite inputs",
                true,
                2,
                |x, _| {
                    for (j, v) in x[2 * STRIDE + K_OFF..2 * STRIDE + G_OFF]
                        .iter_mut()
                        .enumerate()
                    {
                        *v = if j % 2 == 0 { 1e20 } else { -1e20 };
                    }
                },
                FaultSite::CacheValue,
            ),
            (
                46,
                "index gate past f16's range",
                true,
                0,
                |x, _| x[G_OFF + 3] = 1e5,
                FaultSite::CacheValue,
            ),
            (
                47,
                "index position past the cache",
                true,
                3,
                |_, pos| pos[3] = ROWS as u32,
                FaultSite::CachePos,
            ),
        ];
        for (layer, what, index, tok, plant, site) in plants {
            let width = if index { INDEX_ROW } else { LATENT };
            let init = vec![NAN_F16; ROWS * width];
            let (mut bx, mut bpos) = (x.clone(), pos.clone());
            plant(&mut bx, &mut bpos);
            let sink = cx.gpu.layer_sink(layer)?;
            let before = cx.gpu.fault()?;
            let card = if index {
                cx.index(sink, &bx, &p, &bpos, &init)?
            } else {
                cx.latent(sink, &bx, &p, &bpos, &init)?
            };
            let word = cx.gpu.take_fault()?;
            let _ = if index {
                cx.index(sink, &x, &p, &pos, &init)?
            } else {
                cx.latent(sink, &x, &p, &pos, &init)?
            };
            let after = cx.gpu.take_fault()?;
            let mut host = init.clone();
            let sites = if index {
                index_key_append_host(
                    &bx, STRIDE, K_OFF, G_OFF, &p.w, &p.b, &bpos, LN_EPS, &mut host,
                )
            } else {
                latent_append_host(&bx, STRIDE, KV_OFF, &p.gain, &bpos, RMS_EPS, &mut host)
            };
            let want = Fault::at(u32::try_from(layer)?, site);
            let differ = diff_u16(&card, &host);
            // The planted token's row: non-finite for a value plant; for a
            // position plant nothing is written anywhere for it, so the cache
            // is the clean launch's minus that token's row.
            let row = &card[pos[tok] as usize * width..][..width];
            let reached = if site == FaultSite::CachePos {
                row.iter().all(|&h| h == NAN_F16)
            } else {
                row.iter().any(|&h| h & 0x7c00 == 0x7c00)
            };
            let pass = before.is_none()
                && word == Some(want)
                && after.is_none()
                && sites == 1 << site as u32
                && differ == 0
                && reached;
            println!(
                "fault {what}: word \"{}\" (want \"{want}\"), clean before {} and after {}, host \
                 sites {sites:#x}, cache bits differ from the host rule's {differ}, planted row \
                 {} {reached} {}",
                word.map_or_else(|| "none".to_owned(), |f| f.to_string()),
                before.is_none(),
                after.is_none(),
                if site == FaultSite::CachePos {
                    "untouched"
                } else {
                    "non-finite"
                },
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }
}
