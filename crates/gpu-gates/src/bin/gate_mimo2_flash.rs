//! GPU gate for MiMo-V2's attention kernels: the rope-only append
//! (`neox_append_k192`) and the scalar flash over 192-value keys, 128-value
//! values, a window of the positions and per-head sinks (`gqa_flash_seg_k192`,
//! `gqa_flash_merge_sink`, and `gqa_flash_merge` for a layer with neither).
//!
//! The flash is not bit-identical to ik by construction (another exponential,
//! other sum orders), so it is held to the first-order band of
//! `gate_qwen3moe_flash` — per output value of head `h`, with the exact
//! attention computed here in f64 on the same f16 keys and values and the f32
//! query, normalized weights `p̄_j`, `A_j = scale·Σ_d |q_d·k_jd|`, `X_j = M − s_j`:
//!
//! `Σ_j p̄_j·(ε_s·A_j + ε_e(X_j) + ε_r)·|v_jd − o_d| + γ(n_acc)·(Σ_j p̄_j·|v_jd| + |o_d|) + 2u·|o_d|`
//!
//! with the score dot `ε_s` ours `γ(51)` (48 fused multiply-adds a partial,
//! two combine levels, the scale) and ik's `γ(193)`, and the sink as one
//! more key: no score error (`A = 0`), value 0, one more rescale and one more
//! term in the sums. The keys of a window layer are the last 128 of a row's
//! live count and the exact attention is taken over them alone, by the gate's
//! own rule, not by the kernel's `window_cut` (a mutant of the cut would move
//! both). Asserted:
//!
//! - append (A): the turned query and key heads, the key plane rows (f16 of the
//!   turn) and the value plane rows (f16 of `v · v_scale`, the f32 product
//!   rounded once) equal a host transcription bit for bit, over token counts,
//!   positions around the window (127, 128, 129) and the cache's end, both rope
//!   bases and both key-head counts; every other plane row keeps its sentinel;
//!   a position at the cache height raises `cache_pos`, NaNs its token and
//!   appends nothing;
//! - flash on a full layer (F1, 64 heads over 4 key heads, two blocks a group):
//!   inside the band at counts around every tile and segment edge up to 5,121,
//!   a rerun and NaN rows past the count bit-identical, [`ROWS`] rows one launch
//!   each row's one-row launch bit for bit, the refused counts raised and NaN,
//!   the captured graph the eager bits at two cache heights with the grids the
//!   launch names;
//! - flash on a window layer (F2, F3, F4; 64 heads over 8 key heads, window
//!   128, sinks): the band over `[first, n)` and the sink, NaN rows below the
//!   window and past the count changing no bit, every output finite; the window's
//!   edge with a dominant key at `first` and another below it; a 70-row prompt
//!   chunk each row its one-row launch;
//! - the sink (F5, F6, F7): `−inf` is neutral and a window layer at most 128 keys
//!   is the full layer's launch bit for bit; a sink 200 under the running max
//!   equals `−inf`; the band for a sink from 20 under to 20 over the max, and
//!   exactly `+0.0` outputs 200 over it; a NaN or `+inf` sink NaNs its head's
//!   row; the merge folds segments then the sink, bit for bit the sink as one
//!   more segment of a hand-made set of partials;
//! - the geometry (F8): 4 key heads in blocks of two equal 8 key heads of the
//!   same rows in blocks of one, bit for bit;
//! - the refusals (F9) by name before any launch;
//! - ik's sets (G1, G2): the append on ik's `Qcur-L`/`Kcur-L`/`Vcur-L` against
//!   `Qcur_roped-L`, `Kcur_roped-L`, `cache_k` and `v_cache_view` of every layer
//!   (bits), and the flash on ik's `q-L`/`k-L`/`v-L` with the layer's window and
//!   the file's sinks against `fa-L` (both bands).

#[cfg(not(feature = "gpu"))]
fn main() {
    eprintln!(
        "gate_mimo2_flash: built without the `gpu` feature; see `just gate-gpu-mimo2-flash`."
    );
    std::process::exit(2);
}

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mimo2_flash", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::fault::{Fault, FaultSink, FaultSite};
    use bloomery_gpu::flash_gqa::check_sinks;
    use bloomery_gpu::flash_gqa::{
        FlashGqaKernels, GROUP, GqaK192Args, GqaK192MergeArgs, HEAD, HEAD_K192, SEG_KEYS, SEGMENTS,
        seg_span, window_cut, window_segments,
    };
    use bloomery_gpu::rope_neox::{HEAD_K192 as ROPE_HEAD, K192Args, ROT_K192, RopeNeoxKernels};
    use bloomery_gpu::rope_table::{Direction, RopeSpec, RopeTable};
    use bloomery_gpu::{Gpu, GpuError};
    use bloomery_gpu_gates::qwen3moe::f16_logical_bits;
    use bloomery_gpu_gates::rounding::{U, gamma};
    use bloomery_gpu_gates::{
        GateError, RefManifest, activations, bits_equal, checks_failed, mask_bits_in,
        ref_dir_named, ref_model_path, ref_tensor_logical_in, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::GgmlType;
    use gguf::Split;
    use gguf::quant::{f32_to_f16_bits, half_to_f32};
    use model::arch::mimo2::hparams::{Hparams, Kind};
    use refset::arch::mimo2::IK;

    /// An f16 NaN the unread cache rows are overwritten with.
    const NAN16: u16 = 0x7e00;
    /// An f16 pattern the append gates fill the planes with (no value written
    /// by a kernel is this).
    const SENTINEL: u16 = 0x5a5a;
    /// Query rows of the multi-row clauses.
    const ROWS: usize = 8;
    /// The window of the sliding layers (the file's `sliding_window`).
    const WINDOW: usize = 128;
    /// The value multiplier (the file's `value_scale`, as an f32).
    const V_SCALE: f32 = 0.707;
    /// Query heads of every layer, and the key heads of a full and of a window layer.
    const N_HEAD: usize = 64;
    const KV_FULL: usize = 4;
    const KV_SWA: usize = 8;
    /// The seeded query's scale over [`activations`]' [-1, 1): logits of a few
    /// units, so the weights spread over several orders of magnitude.
    const Q_SCALE: f32 = 3.0;
    /// The tall cache height of the capture clause.
    const TALL_CTX: usize = 65_536;

    /// The partials' length for `m` rows of `n_head` heads cut into `segs`
    /// segments, `per` values each.
    fn plen(m: usize, n_head: usize, segs: usize, per: usize) -> usize {
        m * n_head * segs * per
    }

    fn score_scale() -> f32 {
        1.0f32 / (HEAD_K192 as f32).sqrt()
    }

    /// The window's first key by the gate's own rule: the last `window` of a
    /// row's `count` live keys, all of them without a window.
    fn first_of(count: usize, window: usize) -> usize {
        if window == 0 {
            0
        } else {
            count.saturating_sub(window)
        }
    }

    // ------------------------------------------------------------------ data

    /// One layer's f16 planes on the host: key head `j`'s rows `j·ctx ..` of
    /// [`HEAD_K192`] and value rows of [`HEAD`].
    #[derive(Clone)]
    struct Cache {
        n_kv: usize,
        ctx: usize,
        kc: Vec<u16>,
        vc: Vec<u16>,
    }

    impl Cache {
        /// Seeded keys and values over every row.
        fn seeded(n_kv: usize, ctx: usize, seed: u32) -> Cache {
            // A value that rounds to zero is lifted to 0.001: an exact zero in a value row
            // would put a zero in an output the clauses count.
            let h = |v: Vec<f32>| -> Vec<u16> {
                v.iter()
                    .map(|&x| match f32_to_f16_bits(x) {
                        b if b & 0x7fff == 0 => f32_to_f16_bits(1.0e-3),
                        b => b,
                    })
                    .collect()
            };
            Cache {
                n_kv,
                ctx,
                kc: h(activations(HEAD_K192, n_kv * ctx, seed)),
                vc: h(activations(HEAD, n_kv * ctx, seed ^ 0x5bd1_e995)),
            }
        }

        /// The cache with every row outside `keep` NaN, in every key head.
        fn masked(&self, keep: std::ops::Range<usize>) -> Cache {
            let mut c = self.clone();
            for j in 0..self.n_kv {
                for p in (0..self.ctx).filter(|p| !keep.contains(p)) {
                    c.kc[(j * self.ctx + p) * HEAD_K192..][..HEAD_K192].fill(NAN16);
                    c.vc[(j * self.ctx + p) * HEAD..][..HEAD].fill(NAN16);
                }
            }
            c
        }

        /// The cache at height `ctx`: each key head's first `live` rows, NaN rows after.
        fn with_ctx(&self, ctx: usize, live: usize) -> Cache {
            let mut c = Cache {
                n_kv: self.n_kv,
                ctx,
                kc: vec![NAN16; self.n_kv * ctx * HEAD_K192],
                vc: vec![NAN16; self.n_kv * ctx * HEAD],
            };
            for j in 0..self.n_kv {
                c.kc[j * ctx * HEAD_K192..][..live * HEAD_K192]
                    .copy_from_slice(&self.kc[j * self.ctx * HEAD_K192..][..live * HEAD_K192]);
                c.vc[j * ctx * HEAD..][..live * HEAD]
                    .copy_from_slice(&self.vc[j * self.ctx * HEAD..][..live * HEAD]);
            }
            c
        }

        /// Every key head twice over: head `j` of the result is head `j / 2`.
        fn doubled(&self) -> Cache {
            let (kp, vp) = (self.ctx * HEAD_K192, self.ctx * HEAD);
            let mut c = Cache {
                n_kv: 2 * self.n_kv,
                ctx: self.ctx,
                kc: Vec::new(),
                vc: Vec::new(),
            };
            for j in 0..2 * self.n_kv {
                c.kc.extend_from_slice(&self.kc[(j / 2) * kp..][..kp]);
                c.vc.extend_from_slice(&self.vc[(j / 2) * vp..][..vp]);
            }
            c
        }

        fn set_k(&mut self, j: usize, pos: usize, v: &[f32]) {
            for (d, &x) in v.iter().enumerate() {
                self.kc[(j * self.ctx + pos) * HEAD_K192 + d] = f32_to_f16_bits(x);
            }
        }

        fn up(&self, s: &CudaStream) -> Result<(DeviceBuffer<u16>, DeviceBuffer<u16>), GateError> {
            Ok((
                DeviceBuffer::from_host(s, &self.kc)?,
                DeviceBuffer::from_host(s, &self.vc)?,
            ))
        }
    }

    /// `m` seeded query rows of `n_head` heads, token-major.
    fn seeded_q(m: usize, n_head: usize, seed: u32) -> Vec<f32> {
        activations(HEAD_K192, m * n_head, seed)
            .iter()
            .map(|v| v * Q_SCALE)
            .collect()
    }

    /// Row `t` of the prefill shape: the query `q` with its heads rotated by `t`.
    fn rotated_rows(q: &[f32], n_head: usize) -> Vec<f32> {
        (0..ROWS)
            .flat_map(|t| {
                (0..n_head).flat_map(move |h| {
                    q[((h + t) % n_head) * HEAD_K192..][..HEAD_K192]
                        .iter()
                        .copied()
                })
            })
            .collect()
    }

    /// `ROWS` live counts spread from one key to `n`.
    fn spread_counts(n: usize) -> Vec<u32> {
        (0..ROWS)
            .map(|t| (1 + t * (n - 1) / (ROWS - 1)) as u32)
            .collect()
    }

    /// Seeded per-head sinks over [-4, 4).
    fn seeded_sinks(n_head: usize) -> Vec<f32> {
        activations(n_head, 1, 77).iter().map(|v| 4.0 * v).collect()
    }

    // ---------------------------------------------------------------- launch

    /// The launch geometry of one layer kind.
    #[derive(Clone, Copy)]
    struct Shape {
        n_kv: usize,
        n_head: usize,
        ctx: usize,
        window: usize,
    }

    impl Shape {
        fn full(ctx: usize) -> Shape {
            Shape {
                n_kv: KV_FULL,
                n_head: N_HEAD,
                ctx,
                window: 0,
            }
        }
        fn swa(ctx: usize) -> Shape {
            Shape {
                n_kv: KV_SWA,
                n_head: N_HEAD,
                ctx,
                window: WINDOW,
            }
        }
    }

    /// One launch of `counts.len()` rows (`q` holds that many), into fresh
    /// scratch, raising on `fault`, read back.
    #[allow(clippy::too_many_arguments, reason = "a test harness's flat arguments")]
    fn flash_with(
        k: &FlashGqaKernels,
        gpu: &Gpu,
        fault: FaultSink,
        sh: Shape,
        q: &[f32],
        counts: &[u32],
        planes: (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        sinks: Option<&[f32]>,
    ) -> Result<Vec<f32>, GateError> {
        let stream = gpu.stream();
        let m = counts.len();
        let segs = window_segments(sh.window);
        let qd = DeviceBuffer::from_host(stream, q)?;
        let nk = DeviceBuffer::from_host(stream, counts)?;
        let sk = sinks
            .map(|s| DeviceBuffer::from_host(stream, s))
            .transpose()?;
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, plen(m, sh.n_head, segs, HEAD))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, plen(m, sh.n_head, segs, 2))?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, m * sh.n_head * HEAD)?;
        k.enqueue_pass_k192(
            stream,
            GqaK192Args {
                q: &qd,
                kc: planes.0,
                vc: planes.1,
                n_keys: &nk,
                scale: score_scale(),
                n_kv: sh.n_kv,
                ctx: sh.ctx,
                m,
                window: sh.window,
                sinks: sk.as_ref(),
                part_v: &mut pv,
                part_ms: &mut pms,
                fault,
                y: &mut y,
            },
            sh.n_head,
        )?;
        stream.synchronize()?;
        Ok(y.to_host_vec(stream)?)
    }

    fn flash(
        k: &FlashGqaKernels,
        gpu: &Gpu,
        sh: Shape,
        q: &[f32],
        counts: &[u32],
        planes: (&DeviceBuffer<u16>, &DeviceBuffer<u16>),
        sinks: Option<&[f32]>,
    ) -> Result<Vec<f32>, GateError> {
        flash_with(k, gpu, gpu.unlabelled_sink(), sh, q, counts, planes, sinks)
    }

    // ----------------------------------------------------------------- exact

    /// One head's exact attention in f64 and each side's bound.
    struct Ex {
        h: usize,
        o: Vec<f64>,
        bound_ours: Vec<f64>,
        bound_ik: Vec<f64>,
        /// The largest logit over the keys (not the sink).
        m: f64,
    }

    /// What the exact attention reads: the cache, the window and the sinks.
    struct Spec<'a> {
        cache: &'a Cache,
        n_head: usize,
        window: usize,
        sinks: Option<&'a [f32]>,
    }

    fn widen(b: &[u16]) -> Vec<f64> {
        b.iter().map(|&h| f64::from(half_to_f32(h))).collect()
    }

    /// The exact attention of `heads` of one row of `count` live keys, with
    /// the bands of ours and of ik.
    fn exact(sp: &Spec<'_>, q_row: &[f32], count: usize, heads: &[usize]) -> Vec<Ex> {
        let scale = f64::from(score_scale());
        let group = sp.n_head / sp.cache.n_kv;
        let first = first_of(count, sp.window);
        let n = count - first;
        let mut out = Vec::new();
        for kv in 0..sp.cache.n_kv {
            let hs: Vec<usize> = heads.iter().copied().filter(|h| h / group == kv).collect();
            if hs.is_empty() {
                continue;
            }
            let kp = (kv * sp.cache.ctx + first) * HEAD_K192;
            let vp = (kv * sp.cache.ctx + first) * HEAD;
            let kf = widen(&sp.cache.kc[kp..kp + n * HEAD_K192]);
            let vf = widen(&sp.cache.vc[vp..vp + n * HEAD]);
            for h in hs {
                let q: Vec<f64> = q_row[h * HEAD_K192..][..HEAD_K192]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect();
                out.push(exact_head(
                    &q,
                    &kf,
                    &vf,
                    n,
                    scale,
                    sp.sinks.map(|s| s[h]),
                    h,
                ));
            }
        }
        out
    }

    fn exact_head(
        q: &[f64],
        kf: &[f64],
        vf: &[f64],
        n: usize,
        scale: f64,
        sink: Option<f32>,
        h: usize,
    ) -> Ex {
        let s: Vec<f64> = (0..n)
            .map(|j| {
                scale
                    * q.iter()
                        .zip(&kf[j * HEAD_K192..(j + 1) * HEAD_K192])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
            })
            .collect();
        let a: Vec<f64> = (0..n)
            .map(|j| {
                scale.abs()
                    * q.iter()
                        .zip(&kf[j * HEAD_K192..(j + 1) * HEAD_K192])
                        .map(|(a, b)| (a * b).abs())
                        .sum::<f64>()
            })
            .collect();
        let m_keys = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        // The sink: one more logit, unscaled; `−inf` is no sink.
        let sink = sink.filter(|v| *v != f32::NEG_INFINITY).map(f64::from);
        let m = sink.map_or(m_keys, |v| m_keys.max(v));
        let p: Vec<f64> = s.iter().map(|&v| (v - m).exp()).collect();
        let ps = sink.map_or(0.0, |v| (v - m).exp());
        let z: f64 = p.iter().sum::<f64>() + ps;
        let pb: Vec<f64> = p.iter().map(|&v| v / z).collect();
        let pbs = ps / z;
        let x: Vec<f64> = s.iter().map(|&v| m - v).collect();
        let xs = sink.map_or(0.0, |v| m - v);
        let xmax = x.iter().copied().fold(xs, f64::max);
        // The cut: segments of `seg_span` keys, tiles of 32 (ours); ik's
        // blocks of 32 and its thread chunks.
        let span = seg_span(n, SEGMENTS, SEG_KEYS);
        let segs = n.div_ceil(span);
        let extra = usize::from(sink.is_some());
        let r_ours = (n.div_ceil(32) + segs + extra) as f64;
        let r_ik = (n.div_ceil(32) + 33 + extra) as f64;
        let (es_o, es_i) = (gamma(51), gamma(193));
        let (acc_o, acc_i) = (gamma(span + segs + 8 + extra), gamma(n + 40 + extra));
        let mut o = vec![0.0f64; HEAD];
        for (j, &w) in pb.iter().enumerate() {
            for d in 0..HEAD {
                o[d] += w * vf[j * HEAD + d];
            }
        }
        let (mut bo, mut bi) = (vec![0.0f64; HEAD], vec![0.0f64; HEAD]);
        for d in 0..HEAD {
            let (mut t_o, mut t_i, mut pv) = (0.0f64, 0.0f64, 0.0f64);
            for j in 0..n {
                let v = vf[j * HEAD + d];
                let dev = (v - o[d]).abs();
                let ee = 4.0 * U + 2.0 * U * x[j];
                t_o += pb[j] * (es_o * a[j] + ee + 4.0 * U * r_ours + 2.0 * U * xmax) * dev;
                t_i += pb[j] * (es_i * a[j] + ee + 4.0 * U * r_ik + 2.0 * U * xmax) * dev;
                pv += pb[j] * v.abs();
            }
            // The sink's term: value 0, so it deviates from `o` by `|o|`.
            let ee = 4.0 * U + 2.0 * U * xs;
            t_o += pbs * (ee + 4.0 * U * r_ours + 2.0 * U * xmax) * o[d].abs();
            t_i += pbs * (ee + 4.0 * U * r_ik + 2.0 * U * xmax) * o[d].abs();
            bo[d] = t_o + acc_o * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
            bi[d] = t_i + acc_i * (pv + o[d].abs()) + 2.0 * U * o[d].abs();
        }
        Ex {
            h,
            o,
            bound_ours: bo,
            bound_ik: bi,
            m: m_keys,
        }
    }

    /// The heads a count is checked on: all up to 1,025 keys, a spread above.
    fn heads_for(n_head: usize, n: usize) -> Vec<usize> {
        if n <= 1025 {
            (0..n_head).collect()
        } else {
            vec![0, n_head / 4 + 1, n_head / 2, n_head - 1]
        }
    }

    /// `y_row` (`n_head · HEAD`) within its bound of the exact values; the
    /// worst measured/bound ratio.
    fn band(y_row: &[f32], ex: &[Ex]) -> (bool, f64) {
        let mut worst = 0.0f64;
        let mut ok = true;
        for e in ex {
            for d in 0..HEAD {
                let o = f64::from(y_row[e.h * HEAD + d]);
                let dev = (o - e.o[d]).abs();
                let ratio = dev / e.bound_ours[d];
                // NaN fails every comparison.
                if dev.is_nan() || dev > e.bound_ours[d] {
                    ok = false;
                }
                worst = worst.max(ratio);
            }
        }
        (ok, worst)
    }

    fn finite(y: &[f32]) -> bool {
        y.iter().all(|v| v.is_finite())
    }

    /// Whether `r` is an error naming `needle`.
    fn refused<T>(r: Result<T, GpuError>, needle: &str) -> bool {
        match r {
            Err(e) => e.to_string().contains(needle),
            Ok(_) => false,
        }
    }

    // -------------------------------------------------------------------- F1

    /// Counts around every tile and segment edge of a full layer.
    const F1_COUNTS: [usize; 15] = [
        1, 2, 31, 32, 33, 63, 64, 65, 128, 1024, 1025, 4096, 4097, 5120, 5121,
    ];

    /// F1: a full layer (64 heads over 4 key heads, two blocks a group, no
    /// window, no sinks, `gqa_flash_merge`).
    fn f1(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sh = Shape::full(5248);
        let cache = Cache::seeded(sh.n_kv, sh.ctx, 11);
        let dev = cache.up(stream)?;
        let q = seeded_q(1, sh.n_head, 3);
        let mut ok = true;
        let mut worst = 0.0f64;
        for &n in &F1_COUNTS {
            let counts = [n as u32];
            let y = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), None)?;
            let y2 = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), None)?;
            let padded = cache.masked(0..n).up(stream)?;
            let y_nan = flash(k, gpu, sh, &q, &counts, (&padded.0, &padded.1), None)?;
            let sp = Spec {
                cache: &cache,
                n_head: sh.n_head,
                window: 0,
                sinks: None,
            };
            let (in_band, w) = band(&y, &exact(&sp, &q, n, &heads_for(sh.n_head, n)));
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &y_nan));
            let pass = in_band && rerun && nan_same && finite(&y);
            worst = worst.max(w);
            if !pass {
                println!(
                    "F1 n={n}: band={in_band} (measured/bound {w:.3e}) rerun={rerun} \
                     nan_rows_same={nan_same} FAIL"
                );
            }
            ok &= pass;
        }
        println!(
            "F1 full layer 64/4 packs 2, {} counts up to 5121: band (measured/bound {worst:.3e}), \
             rerun, NaN rows past the count {}",
            F1_COUNTS.len(),
            verdict(ok)
        );

        // ROWS rows in one launch = each row's one-row launch.
        let mut rows_ok = true;
        for &n in &[65usize, 1025, 5121] {
            let counts = spread_counts(n);
            let qs = rotated_rows(&q, sh.n_head);
            let all = flash(k, gpu, sh, &qs, &counts, (&dev.0, &dev.1), None)?;
            let w = sh.n_head * HEAD;
            for t in 0..ROWS {
                let one = flash(
                    k,
                    gpu,
                    sh,
                    &qs[t * sh.n_head * HEAD_K192..(t + 1) * sh.n_head * HEAD_K192],
                    &counts[t..=t],
                    (&dev.0, &dev.1),
                    None,
                )?;
                let same = bits_equal(&all[t * w..(t + 1) * w], &one);
                if !same {
                    println!("F1 rows n={n} row {t}: not its one-row launch FAIL");
                }
                rows_ok &= same;
            }
        }
        println!(
            "F1 {ROWS} rows in one launch = each row's one-row launch, counts spread to 65, 1025, 5121 {}",
            verdict(rows_ok)
        );
        ok &= rows_ok;

        // The refused counts.
        let layer = 13usize;
        let want = Some(Fault::at(u32::try_from(layer)?, FaultSite::KeyCount));
        let qs = rotated_rows(&q, sh.n_head);
        let clean = spread_counts(1025);
        let mut bad = clean.clone();
        bad[3] = u32::try_from(sh.ctx + 1)?;
        bad[5] = 0;
        let sink = gpu.layer_sink(layer)?;
        let before = gpu.fault()?;
        let y_clean = flash_with(k, gpu, sink, sh, &qs, &clean, (&dev.0, &dev.1), None)?;
        let after_clean = gpu.fault()?;
        let y_bad = flash_with(k, gpu, sink, sh, &qs, &bad, (&dev.0, &dev.1), None)?;
        let raised = gpu.take_fault()?;
        let w = sh.n_head * HEAD;
        let (mut others, mut nan_rows) = (true, true);
        for r in 0..ROWS {
            let (a, b) = (&y_clean[r * w..(r + 1) * w], &y_bad[r * w..(r + 1) * w]);
            if r == 3 || r == 5 {
                nan_rows &= b.iter().all(|v| v.is_nan());
            } else {
                others &= bits_equal(a, b);
            }
        }
        let fault_ok =
            before.is_none() && after_clean.is_none() && raised == want && nan_rows && others;
        println!(
            "F1 refused counts (past ctx {}, and 0) at rows 3, 5: word {:?} (want {want:?}), those rows NaN \
             {nan_rows}, other rows bit-identical {others} {}",
            sh.ctx,
            raised,
            verdict(fault_ok)
        );
        ok &= fault_ok;

        // The captured launch: two kernel nodes at the launch's grids, the
        // eager bits, at the cache's height and at a tall one.
        let n_live = 4097usize;
        let counts = [n_live as u32];
        let eager = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), None)?;
        let tall = cache.with_ctx(TALL_CTX, n_live);
        let tall_dev = tall.up(stream)?;
        let want_seg = [
            u32::try_from(sh.n_kv * (sh.n_head / sh.n_kv / GROUP) * SEGMENTS)?,
            1,
            1,
        ];
        let want_merge = [u32::try_from(sh.n_head)?, 1, 1];
        let qd = DeviceBuffer::from_host(stream, &q)?;
        let nk = DeviceBuffer::from_host(stream, &counts)?;
        let mut cap_ok = true;
        for (ctx, planes) in [(sh.ctx, &dev), (TALL_CTX, &tall_dev)] {
            let mut pv = DeviceBuffer::<f32>::zeroed(stream, plen(1, sh.n_head, SEGMENTS, HEAD))?;
            let mut pms = DeviceBuffer::<f32>::zeroed(stream, plen(1, sh.n_head, SEGMENTS, 2))?;
            let mut y = DeviceBuffer::<f32>::zeroed(stream, sh.n_head * HEAD)?;
            let graph = gpu.capture(|s| {
                k.enqueue_pass_k192(
                    s,
                    GqaK192Args {
                        q: &qd,
                        kc: &planes.0,
                        vc: &planes.1,
                        n_keys: &nk,
                        scale: score_scale(),
                        n_kv: sh.n_kv,
                        ctx,
                        m: 1,
                        window: 0,
                        sinks: None,
                        part_v: &mut pv,
                        part_ms: &mut pms,
                        fault: gpu.unlabelled_sink(),
                        y: &mut y,
                    },
                    sh.n_head,
                )
            })?;
            graph.launch(stream)?;
            stream.synchronize()?;
            let nodes = graph.nodes()?;
            let grid = |prefix: &str| {
                nodes.iter().find_map(|n| {
                    n.kernel
                        .as_ref()
                        .filter(|kn| kn.name.starts_with(prefix))
                        .map(|kn| kn.grid)
                })
            };
            let two = nodes.len() == 2 && nodes.iter().all(|n| n.kernel.is_some());
            let (seg, merge) = (grid("gqa_flash_seg_k192"), grid("gqa_flash_merge"));
            let same = bits_equal(&y.to_host_vec(stream)?, &eager);
            let this = two && seg == Some(want_seg) && merge == Some(want_merge) && same;
            println!(
                "F1 graph ctx={ctx}: nodes two={two} seg_grid={seg:?} (want {want_seg:?}) merge_grid={merge:?} \
                 (want {want_merge:?}) = eager bits {same} {}",
                verdict(this)
            );
            cap_ok &= this;
        }
        ok &= cap_ok;
        Ok(ok)
    }

    // ---------------------------------------------------------- F2 F3 F4 F5 F6

    /// Counts of a window layer around the window, a tile, a segment and the
    /// cache's end.
    const F2_COUNTS: [usize; 14] = [
        1, 64, 127, 128, 129, 130, 160, 161, 191, 192, 193, 1000, 4097, 65_536,
    ];

    /// The cache rows `[first, n)` kept, the rest NaN.
    fn window_rows(cache: &Cache, n: usize) -> Cache {
        cache.masked(first_of(n, WINDOW)..n)
    }

    /// F2: a window layer with finite sinks: the band over `[first, n)` and the
    /// sink, every other row NaN changing no bit.
    fn f2(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let small = Cache::seeded(KV_SWA, 4352, 21);
        let small_dev = small.up(stream)?;
        let big = Cache::seeded(KV_SWA, TALL_CTX, 22);
        let big_dev = big.up(stream)?;
        let q = seeded_q(1, N_HEAD, 4);
        let sinks = seeded_sinks(N_HEAD);
        let mut ok = true;
        let mut worst = 0.0f64;
        for &n in &F2_COUNTS {
            let (cache, dev, ctx) = if n > 4352 {
                (&big, &big_dev, TALL_CTX)
            } else {
                (&small, &small_dev, 4352)
            };
            let sh = Shape::swa(ctx);
            let counts = [n as u32];
            let y = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&sinks))?;
            let y2 = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&sinks))?;
            let nan = window_rows(cache, n).up(stream)?;
            let y_nan = flash(k, gpu, sh, &q, &counts, (&nan.0, &nan.1), Some(&sinks))?;
            let sp = Spec {
                cache,
                n_head: N_HEAD,
                window: WINDOW,
                sinks: Some(&sinks),
            };
            let all: Vec<usize> = (0..N_HEAD).collect();
            let (in_band, w) = band(&y, &exact(&sp, &q, n, &all));
            let (rerun, nan_same) = (bits_equal(&y, &y2), bits_equal(&y, &y_nan));
            let pass = in_band && rerun && nan_same && finite(&y);
            worst = worst.max(w);
            if !pass {
                println!(
                    "F2 n={n}: band={in_band} (measured/bound {w:.3e}) rerun={rerun} \
                     nan_outside_window_same={nan_same} finite={} FAIL",
                    finite(&y)
                );
            }
            ok &= pass;
        }
        println!(
            "F2 window layer 64/8 window {WINDOW} with sinks, {} counts up to 65536: band over [first, n) and \
             the sink (measured/bound {worst:.3e}), rerun, NaN rows outside the window, finite {}",
            F2_COUNTS.len(),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The key a query dominates: `q` scaled to a logit of `target` over the
    /// score scale.
    fn boost(q_head: &[f32], target: f32) -> Vec<f32> {
        let qq: f32 = q_head.iter().map(|v| v * v).sum();
        let a = target / (score_scale() * qq);
        q_head.iter().map(|v| a * v).collect()
    }

    /// F3: the window's edge. Every head attends one query; the key at `first`
    /// dominates (logit 30) and the key below it more (30.3) and holds NaN in
    /// the second run: a kernel that reads one key too few misses `first`, one
    /// too many reads the NaN and changes bits.
    fn f3(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sinks = seeded_sinks(N_HEAD);
        let one = seeded_q(1, 1, 5);
        let q: Vec<f32> = (0..N_HEAD).flat_map(|_| one.iter().copied()).collect();
        let (hi, lo) = (boost(&one, 30.3), boost(&one, 30.0));
        let mut ok = true;
        for &n in &[129usize, 161, 1000, 4097] {
            let first = n - WINDOW;
            let mut cache = Cache::seeded(KV_SWA, 4352, 31);
            for j in 0..KV_SWA {
                cache.set_k(j, first - 1, &hi);
                cache.set_k(j, first, &lo);
            }
            let sh = Shape::swa(cache.ctx);
            let dev = cache.up(stream)?;
            let y = flash(k, gpu, sh, &q, &[n as u32], (&dev.0, &dev.1), Some(&sinks))?;
            let mut poisoned = cache.clone();
            for j in 0..KV_SWA {
                poisoned.kc[(j * cache.ctx + first - 1) * HEAD_K192..][..HEAD_K192].fill(NAN16);
                poisoned.vc[(j * cache.ctx + first - 1) * HEAD..][..HEAD].fill(NAN16);
            }
            let pd = poisoned.up(stream)?;
            let y_nan = flash(k, gpu, sh, &q, &[n as u32], (&pd.0, &pd.1), Some(&sinks))?;
            let sp = Spec {
                cache: &cache,
                n_head: N_HEAD,
                window: WINDOW,
                sinks: Some(&sinks),
            };
            let all: Vec<usize> = (0..N_HEAD).collect();
            let (in_band, w) = band(&y, &exact(&sp, &q, n, &all));
            // The key at `first` carries the row: the output is its value row
            // to within the other keys' weight.
            let near = (0..N_HEAD).all(|h| {
                let kv = h / (N_HEAD / KV_SWA);
                let v_row = &cache.vc[(kv * cache.ctx + first) * HEAD..][..HEAD];
                (0..HEAD).all(|d| {
                    (f64::from(y[h * HEAD + d]) - f64::from(half_to_f32(v_row[d]))).abs() < 0.01
                })
            });
            let same = bits_equal(&y, &y_nan);
            let pass = in_band && same && near && finite(&y);
            if !pass {
                println!(
                    "F3 n={n} first={first}: band={in_band} ({w:.3e}) key-below-first NaN same={same} \
                     output is value[first]={near} FAIL"
                );
            }
            ok &= pass;
        }
        println!(
            "F3 window edge, n = 129, 161, 1000, 4097 (first 1, 33, 872, 3969): the key at first carries the \
             row, the key below it (more dominant) is never read {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// F4: a prompt chunk of 70 rows on a window layer: row `t` attends its own
    /// window and equals its one-row launch.
    fn f4(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sh = Shape::swa(4352);
        let cache = Cache::seeded(KV_SWA, sh.ctx, 41);
        let dev = cache.up(stream)?;
        let sinks = seeded_sinks(N_HEAD);
        let m = 70usize;
        let q = seeded_q(m, N_HEAD, 6);
        let mut ok = true;
        for &b in &[90usize, 4030] {
            let counts: Vec<u32> = (0..m).map(|t| (b + t + 1) as u32).collect();
            let all = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&sinks))?;
            let w = N_HEAD * HEAD;
            let (mut same, mut in_band) = (true, true);
            for t in 0..m {
                let one = flash(
                    k,
                    gpu,
                    sh,
                    &q[t * N_HEAD * HEAD_K192..(t + 1) * N_HEAD * HEAD_K192],
                    &counts[t..=t],
                    (&dev.0, &dev.1),
                    Some(&sinks),
                )?;
                same &= bits_equal(&all[t * w..(t + 1) * w], &one);
                if [0usize, 1, 37, 38, 69].contains(&t) {
                    let sp = Spec {
                        cache: &cache,
                        n_head: N_HEAD,
                        window: WINDOW,
                        sinks: Some(&sinks),
                    };
                    let heads = [0usize, 9, 33, 63];
                    let qrow = &q[t * N_HEAD * HEAD_K192..(t + 1) * N_HEAD * HEAD_K192];
                    in_band &= band(
                        &all[t * w..(t + 1) * w],
                        &exact(&sp, qrow, counts[t] as usize, &heads),
                    )
                    .0;
                }
            }
            println!(
                "F4 chunk m={m} b={b} counts {}..={}: each row = its one-row launch {same}, rows 0, 1, 37, 38, 69 \
                 in the band {in_band} {}",
                counts[0],
                counts[m - 1],
                verdict(same && in_band)
            );
            ok &= same && in_band;
        }
        Ok(ok)
    }

    /// F5: the sink's neutral value. (a) `−inf` sinks and a window layer of at
    /// most 128 keys equal the full launch (no window, the other merge) bit for
    /// bit, with no ±0 in the output; (b) a sink 200 under the running max
    /// equals `−inf`.
    fn f5(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sh = Shape::swa(4352);
        let cache = Cache::seeded(KV_SWA, sh.ctx, 51);
        let dev = cache.up(stream)?;
        let q = seeded_q(1, N_HEAD, 7);
        let neutral = vec![f32::NEG_INFINITY; N_HEAD];
        let (mut ok_a, mut zeros) = (true, 0usize);
        for &n in &[1usize, 2, 63, 64, 65, 100, 127, 128] {
            let counts = [n as u32];
            let swa = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&neutral))?;
            let full = Shape { window: 0, ..sh };
            let plain = flash(k, gpu, full, &q, &counts, (&dev.0, &dev.1), None)?;
            let same = bits_equal(&swa, &plain);
            let z = swa.iter().filter(|v| **v == 0.0).count();
            if z > 0 {
                let at = swa.iter().position(|v| *v == 0.0).unwrap_or(0);
                println!(
                    "F5a n={n}: {z} outputs equal to zero, the first at head {} dim {} ({:?})",
                    at / HEAD,
                    at % HEAD,
                    swa[at]
                );
            }
            zeros += z;
            if !same {
                println!("F5a n={n}: window layer with -inf sinks is not the full launch FAIL");
            }
            ok_a &= same;
        }
        ok_a &= zeros == 0;
        println!(
            "F5a -inf sinks, window layer n <= 128 = the full launch (window 0, no sinks) bit for bit, outputs \
             equal to zero {zeros} {}",
            verdict(ok_a)
        );
        let mut ok_b = true;
        for &n in &[100usize, 4097] {
            let sp = Spec {
                cache: &cache,
                n_head: N_HEAD,
                window: WINDOW,
                sinks: None,
            };
            let all: Vec<usize> = (0..N_HEAD).collect();
            let m: Vec<f32> = exact(&sp, &q, n, &all)
                .iter()
                .map(|e| (e.m - 200.0) as f32)
                .collect();
            let counts = [n as u32];
            let low = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&m))?;
            let none = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&neutral))?;
            let same = bits_equal(&low, &none);
            if !same {
                println!("F5b n={n}: a sink 200 under the max is not -inf FAIL");
            }
            ok_b &= same;
        }
        println!(
            "F5b a sink 200 under each head's max equals -inf bit for bit, n = 100, 4097 {}",
            verdict(ok_b)
        );
        Ok(ok_a && ok_b)
    }

    /// F6: the sink against the running max.
    fn f6(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let sh = Shape::swa(4352);
        let cache = Cache::seeded(KV_SWA, sh.ctx, 61);
        let dev = cache.up(stream)?;
        let q = seeded_q(1, N_HEAD, 8);
        let all: Vec<usize> = (0..N_HEAD).collect();
        let mut ok = true;
        let mut worst = 0.0f64;
        for &n in &[100usize, 4097] {
            let counts = [n as u32];
            let sp = Spec {
                cache: &cache,
                n_head: N_HEAD,
                window: WINDOW,
                sinks: None,
            };
            let maxes: Vec<f64> = exact(&sp, &q, n, &all).iter().map(|e| e.m).collect();
            for delta in [-20.0f64, -10.0, -3.0, 0.0, 3.0, 10.0, 20.0] {
                let sinks: Vec<f32> = maxes.iter().map(|m| (m + delta) as f32).collect();
                let y = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&sinks))?;
                let sp = Spec {
                    cache: &cache,
                    n_head: N_HEAD,
                    window: WINDOW,
                    sinks: Some(&sinks),
                };
                let (in_band, w) = band(&y, &exact(&sp, &q, n, &all));
                worst = worst.max(w);
                if !in_band {
                    println!("F6 n={n} sink = max{delta:+}: out of the band ({w:.3e}) FAIL");
                }
                ok &= in_band;
            }
            // 200 over the max: the row is exactly +0.0.
            let sinks: Vec<f32> = maxes.iter().map(|m| (m + 200.0) as f32).collect();
            let y = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&sinks))?;
            let zero = y.iter().all(|v| v.to_bits() == 0);
            if !zero {
                println!("F6 n={n} sink = max+200: not exactly +0.0 FAIL");
            }
            ok &= zero;
            // A NaN or +inf sink on one head: that head's row is NaN, the others unchanged.
            let clean = flash(
                k,
                gpu,
                sh,
                &q,
                &counts,
                (&dev.0, &dev.1),
                Some(&seeded_sinks(N_HEAD)),
            )?;
            for bad in [f32::NAN, f32::INFINITY] {
                let mut sk = seeded_sinks(N_HEAD);
                sk[5] = bad;
                let y = flash(k, gpu, sh, &q, &counts, (&dev.0, &dev.1), Some(&sk))?;
                let head_nan = y[5 * HEAD..6 * HEAD].iter().all(|v| v.is_nan());
                let rest = bits_equal(&y[..5 * HEAD], &clean[..5 * HEAD])
                    && bits_equal(&y[6 * HEAD..], &clean[6 * HEAD..]);
                if !(head_nan && rest) {
                    println!(
                        "F6 n={n} sink {bad}: head row NaN {head_nan}, other heads same {rest} FAIL"
                    );
                }
                ok &= head_nan && rest;
            }
        }
        println!(
            "F6 sink from 20 under to 20 over each head's max: band (measured/bound {worst:.3e}); 200 over the \
             max: every output +0.0; a NaN or +inf sink NaNs its head's row alone {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// F7: the merge's order: segments ascending, then the sink — the sink as
    /// one more segment of hand-made partials gives the same bits.
    fn f7(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let (n_head, ctx) = (2usize, 4352usize);
        let segs = window_segments(0);
        let sink = 2.2f32;
        let v = activations(HEAD, 2 * 2, 91);
        let seg = [(1.5f32, 3.2f32), (2.7, 5.1)];
        let mut part_ms = vec![0.0f32; n_head * segs * 2];
        let mut part_v = vec![0.0f32; n_head * segs * HEAD];
        let mut part_ms3 = part_ms.clone();
        let mut part_v3 = part_v.clone();
        for h in 0..n_head {
            for (s, &(m, e)) in seg.iter().enumerate() {
                let at = h * segs + s;
                part_ms[2 * at] = m;
                part_ms[2 * at + 1] = e;
                part_ms3[2 * at] = m;
                part_ms3[2 * at + 1] = e;
                let row = &v[(h * 2 + s) * HEAD..][..HEAD];
                part_v[at * HEAD..][..HEAD].copy_from_slice(row);
                part_v3[at * HEAD..][..HEAD].copy_from_slice(row);
            }
            // The sink as the third segment `(sink, 1, 0)` of the second run.
            let at = h * segs + 2;
            part_ms3[2 * at] = sink;
            part_ms3[2 * at + 1] = 1.0;
        }
        let run =
            |ms: &[f32], pv: &[f32], count: u32, sinks: &[f32]| -> Result<Vec<f32>, GateError> {
                let (pvd, pmsd) = (
                    DeviceBuffer::from_host(stream, pv)?,
                    DeviceBuffer::from_host(stream, ms)?,
                );
                let nk = DeviceBuffer::from_host(stream, &[count])?;
                let sk = DeviceBuffer::from_host(stream, sinks)?;
                let mut y = DeviceBuffer::<f32>::zeroed(stream, n_head * HEAD)?;
                k.enqueue_merge_k192(
                    stream,
                    GqaK192MergeArgs {
                        part_v: &pvd,
                        part_ms: &pmsd,
                        n_keys: &nk,
                        sinks: Some(&sk),
                        ctx,
                        m: 1,
                        window: 0,
                        fault: gpu.unlabelled_sink(),
                        y: &mut y,
                    },
                    n_head,
                )?;
                stream.synchronize()?;
                Ok(y.to_host_vec(stream)?)
            };
        // 128 keys cut into two 64-key segments, the sink folded last; 129
        // keys cut into three, the third being the sink's partial, the sink
        // itself neutral.
        let a = run(&part_ms, &part_v, 128, &[sink; 2])?;
        let b = run(&part_ms3, &part_v3, 129, &[f32::NEG_INFINITY; 2])?;
        let same = bits_equal(&a, &b) && finite(&a);
        println!(
            "F7 merge(sink s, [a, b]) = merge(-inf, [a, b, (s, 1, 0)]) bit for bit {}",
            verdict(same)
        );
        Ok(same)
    }

    /// F8: four key heads in blocks of two equal the same rows over eight key
    /// heads (each doubled) in blocks of one, bit for bit.
    fn f8(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let four = Cache::seeded(KV_FULL, 4352, 81);
        let eight = four.doubled();
        let (d4, d8) = (four.up(stream)?, eight.up(stream)?);
        let (s4, s8) = (
            Shape::full(4352),
            Shape {
                n_kv: KV_SWA,
                ..Shape::full(4352)
            },
        );
        let q = seeded_q(3, N_HEAD, 9);
        let mut ok = true;
        for counts in [[100u32, 1025, 4097], [1, 64, 129]] {
            let a = flash(k, gpu, s4, &q, &counts, (&d4.0, &d4.1), None)?;
            let b = flash(k, gpu, s8, &q, &counts, (&d8.0, &d8.1), None)?;
            let same = bits_equal(&a, &b);
            if !same {
                println!("F8 counts {counts:?}: 4 kv x packs 2 differs from 8 kv x packs 1 FAIL");
            }
            ok &= same;
        }
        println!(
            "F8 4 key heads x2 packs = the same rows over 8 key heads x1 pack, 3 rows each, bit for bit {}",
            verdict(ok)
        );
        Ok(ok)
    }

    /// F9: the host refuses, by name, before any launch.
    fn f9(k: &FlashGqaKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let cache = Cache::seeded(KV_SWA, 256, 99);
        let dev = cache.up(stream)?;
        let q = DeviceBuffer::from_host(stream, &seeded_q(1, N_HEAD, 1))?;
        let nk = DeviceBuffer::from_host(stream, &[100u32])?;
        let sinks = DeviceBuffer::from_host(stream, &seeded_sinks(N_HEAD))?;
        let short = DeviceBuffer::from_host(stream, &seeded_sinks(N_HEAD - 1))?;
        let segs = window_segments(WINDOW);
        let mut pv = DeviceBuffer::<f32>::zeroed(stream, plen(1, N_HEAD, segs, HEAD))?;
        let mut pms = DeviceBuffer::<f32>::zeroed(stream, plen(1, N_HEAD, segs, 2))?;
        let mut pv_short = DeviceBuffer::<f32>::zeroed(stream, 128)?;
        let mut y = DeviceBuffer::<f32>::zeroed(stream, N_HEAD * HEAD)?;
        let mut call = |n_head: usize,
                        n_kv: usize,
                        m: usize,
                        ctx: usize,
                        scale: f32,
                        window: usize,
                        sinks: Option<&DeviceBuffer<f32>>,
                        short_v: bool|
         -> Result<(), GpuError> {
            let part_v = if short_v { &mut pv_short } else { &mut pv };
            k.enqueue_pass_k192(
                stream,
                GqaK192Args {
                    q: &q,
                    kc: &dev.0,
                    vc: &dev.1,
                    n_keys: &nk,
                    scale,
                    n_kv,
                    ctx,
                    m,
                    window,
                    sinks,
                    part_v,
                    part_ms: &mut pms,
                    fault: gpu.unlabelled_sink(),
                    y: &mut y,
                },
                n_head,
            )
        };
        let sc = score_scale();
        let mut ok = true;
        let mut check = |name: &str, got: bool| {
            if !got {
                println!("F9 {name}: not refused by name FAIL");
            }
            ok &= got;
        };
        check(
            "group 12 (48 heads over 4 key heads)",
            refused(call(48, 4, 1, 256, sc, 0, None, false), "multiple of 8"),
        );
        check(
            "short sinks",
            refused(
                call(N_HEAD, KV_SWA, 1, 256, sc, WINDOW, Some(&short), false),
                "sinks.len()",
            ),
        );
        check(
            "a window without sinks",
            refused(
                call(N_HEAD, KV_SWA, 1, 256, sc, WINDOW, None, false),
                "needs the sink merge",
            ),
        );
        check(
            "short partials",
            refused(
                call(N_HEAD, KV_SWA, 1, 256, sc, WINDOW, Some(&sinks), true),
                "part_v.len()",
            ),
        );
        check(
            "m = 0",
            refused(
                call(N_HEAD, KV_SWA, 0, 256, sc, WINDOW, Some(&sinks), false),
                "m >= 1",
            ),
        );
        check(
            "ctx = 0",
            refused(
                call(N_HEAD, KV_SWA, 1, 0, sc, WINDOW, Some(&sinks), false),
                "ctx",
            ),
        );
        check(
            "n_kv = 0",
            refused(
                call(N_HEAD, 0, 1, 256, sc, WINDOW, Some(&sinks), false),
                "n_kv",
            ),
        );
        check(
            "a NaN scale",
            refused(
                call(
                    N_HEAD,
                    KV_SWA,
                    1,
                    256,
                    f32::NAN,
                    WINDOW,
                    Some(&sinks),
                    false,
                ),
                "finite",
            ),
        );
        check(
            "an infinite scale",
            refused(
                call(N_HEAD, KV_SWA, 1, 256, f32::INFINITY, 0, None, false),
                "finite",
            ),
        );
        check(
            "a NaN sink on the host",
            refused(check_sinks(&[1.0, f32::NAN]), "sink of head 1"),
        );
        check(
            "an infinite sink on the host",
            refused(check_sinks(&[f32::NEG_INFINITY]), "sink of head 0"),
        );
        check(
            "finite sinks on the host",
            check_sinks(&seeded_sinks(N_HEAD)).is_ok(),
        );
        // The merge alone: the window without sinks.
        let (pvd, pmsd) = (
            DeviceBuffer::<f32>::zeroed(stream, plen(1, 2, segs, HEAD))?,
            DeviceBuffer::<f32>::zeroed(stream, plen(1, 2, segs, 2))?,
        );
        let mut y2 = DeviceBuffer::<f32>::zeroed(stream, 2 * HEAD)?;
        let merge = k.enqueue_merge_k192(
            stream,
            GqaK192MergeArgs {
                part_v: &pvd,
                part_ms: &pmsd,
                n_keys: &nk,
                sinks: None,
                ctx: 256,
                m: 1,
                window: WINDOW,
                fault: gpu.unlabelled_sink(),
                y: &mut y2,
            },
            2,
        );
        check(
            "the merge over a window without sinks",
            refused(merge, "needs the sink merge"),
        );
        // A good launch after the refusals: nothing was launched, nothing is poisoned.
        let after = call(N_HEAD, KV_SWA, 1, 256, sc, WINDOW, Some(&sinks), false);
        stream.synchronize()?;
        let clean = after.is_ok() && gpu.fault()?.is_none();
        println!(
            "F9 refusals before any launch (group 12, short sinks, a window without sinks, short partials, zero \
             counts, a non-finite scale, the merge over a window without sinks); a clean launch after {}",
            verdict(ok && clean)
        );
        Ok(ok && clean)
    }

    // ---------------------------------------------------------------- append

    /// The host transcription of the turn of one head's first 64 values.
    fn host_turn(head: &[f32], row: &[f32]) -> Vec<f32> {
        let mut out = head.to_vec();
        for i in 0..ROT_K192 / 2 {
            let (x0, x1) = (head[i], head[i + ROT_K192 / 2]);
            let (c, s) = (row[2 * i], row[2 * i + 1]);
            out[i] = x0.mul_add(c, -(x1 * s));
            out[i + ROT_K192 / 2] = x0.mul_add(s, x1 * c);
        }
        out
    }

    fn table_for(base: f32, ctx: usize) -> Result<Vec<f32>, GateError> {
        let t = RopeTable::new(&RopeSpec::window(base, ROT_K192))?;
        let mut v = Vec::with_capacity(ctx * ROT_K192);
        for p in 0..ctx {
            t.push(p as u32, Direction::Forward, &mut v);
        }
        Ok(v)
    }

    /// The fused rows of `m` tokens: `[query heads | key heads | value heads]`.
    fn fused(
        m: usize,
        n_head: usize,
        n_kv: usize,
        seed: u32,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let scale = |v: Vec<f32>, a: f32| -> Vec<f32> { v.iter().map(|x| a * x).collect() };
        let q = scale(activations(HEAD_K192, m * n_head, seed), 8.0);
        let kk = scale(activations(HEAD_K192, m * n_kv, seed + 1), 8.0);
        let v = scale(activations(HEAD, m * n_kv, seed + 2), 8.0);
        let mut qkv = Vec::new();
        for t in 0..m {
            qkv.extend_from_slice(&q[t * n_head * HEAD_K192..(t + 1) * n_head * HEAD_K192]);
            qkv.extend_from_slice(&kk[t * n_kv * HEAD_K192..(t + 1) * n_kv * HEAD_K192]);
            qkv.extend_from_slice(&v[t * n_kv * HEAD..(t + 1) * n_kv * HEAD]);
        }
        (qkv, q, kk, v)
    }

    /// What one append launch left on the card.
    struct Appended {
        q: Vec<f32>,
        qkv: Vec<f32>,
        kc: Vec<u16>,
        vc: Vec<u16>,
    }

    #[allow(clippy::too_many_arguments, reason = "a test harness's flat arguments")]
    fn append_once(
        a: &RopeNeoxKernels,
        gpu: &Gpu,
        fault: FaultSink,
        (n_head, n_kv, ctx): (usize, usize, usize),
        qkv: &[f32],
        table: &[f32],
        pos: &[u32],
        v_scale: f32,
    ) -> Result<Appended, GateError> {
        let stream = gpu.stream();
        let m = pos.len();
        let mut qkv_d = DeviceBuffer::from_host(stream, qkv)?;
        let mut q_d = DeviceBuffer::<f32>::zeroed(stream, m * n_head * HEAD_K192)?;
        let table_d = DeviceBuffer::from_host(stream, table)?;
        let pos_d = DeviceBuffer::from_host(stream, pos)?;
        let mut kc = DeviceBuffer::from_host(stream, &vec![SENTINEL; n_kv * ctx * HEAD_K192])?;
        let mut vc = DeviceBuffer::from_host(stream, &vec![SENTINEL; n_kv * ctx * HEAD])?;
        a.enqueue_neox_append_k192(
            stream,
            K192Args {
                qkv: &mut qkv_d,
                q: &mut q_d,
                table: &table_d,
                pos: &pos_d,
                v_scale,
                n_head,
                n_kv,
                ctx,
                m,
                fault,
                cache_k: &mut kc,
                cache_v: &mut vc,
            },
        )?;
        stream.synchronize()?;
        Ok(Appended {
            q: q_d.to_host_vec(stream)?,
            qkv: qkv_d.to_host_vec(stream)?,
            kc: kc.to_host_vec(stream)?,
            vc: vc.to_host_vec(stream)?,
        })
    }

    /// What the append writes by the host's arithmetic: the turned query rows,
    /// the fused rows with the key heads turned in place, and the two planes
    /// (sentinel except the appended rows). Tokens at `skip` append nothing and
    /// are NaN.
    fn host_append(
        (n_head, n_kv, ctx): (usize, usize, usize),
        qkv: &[f32],
        table: &[f32],
        pos: &[u32],
        v_scale: f32,
        skip: &[usize],
    ) -> Appended {
        let row = n_head * HEAD_K192 + n_kv * (HEAD_K192 + HEAD);
        let m = pos.len();
        let mut out = Appended {
            q: vec![0.0; m * n_head * HEAD_K192],
            qkv: qkv.to_vec(),
            kc: vec![SENTINEL; n_kv * ctx * HEAD_K192],
            vc: vec![SENTINEL; n_kv * ctx * HEAD],
        };
        for t in 0..m {
            let p = pos[t] as usize;
            if skip.contains(&t) {
                out.q[t * n_head * HEAD_K192..(t + 1) * n_head * HEAD_K192].fill(f32::NAN);
                out.qkv[t * row + n_head * HEAD_K192..t * row + (n_head + n_kv) * HEAD_K192]
                    .fill(f32::NAN);
                continue;
            }
            let trow = &table[p * ROT_K192..(p + 1) * ROT_K192];
            for h in 0..n_head {
                let src = &qkv[t * row + h * HEAD_K192..][..HEAD_K192];
                out.q[(t * n_head + h) * HEAD_K192..][..HEAD_K192]
                    .copy_from_slice(&host_turn(src, trow));
            }
            for j in 0..n_kv {
                let at = t * row + (n_head + j) * HEAD_K192;
                let turned = host_turn(&qkv[at..at + HEAD_K192], trow);
                out.qkv[at..at + HEAD_K192].copy_from_slice(&turned);
                for (d, &x) in turned.iter().enumerate() {
                    out.kc[(j * ctx + p) * HEAD_K192 + d] = f32_to_f16_bits(x);
                }
                let vat = t * row + (n_head + n_kv) * HEAD_K192 + j * HEAD;
                for d in 0..HEAD {
                    out.vc[(j * ctx + p) * HEAD + d] = f32_to_f16_bits(qkv[vat + d] * v_scale);
                }
            }
        }
        out
    }

    fn appended_equal(a: &Appended, b: &Appended) -> bool {
        bits_equal(&a.q, &b.q) && bits_equal(&a.qkv, &b.qkv) && a.kc == b.kc && a.vc == b.vc
    }

    /// A1 and A2: the append against the host transcription, and its refusal.
    fn append_clauses(a: &RopeNeoxKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let ctx = 4352usize;
        let tables = [
            (1.0e7f32, table_for(1.0e7, ctx)?),
            (1.0e4, table_for(1.0e4, ctx)?),
        ];
        let mut ok = true;
        let mut launches = 0usize;
        for (base, table) in &tables {
            for n_kv in [KV_FULL, KV_SWA] {
                for m in [1usize, 5, 64] {
                    for p0 in [0u32, 127, 128, 129, 4095, 4096] {
                        let pos: Vec<u32> = (0..m as u32).map(|t| p0 + t).collect();
                        let g = (N_HEAD, n_kv, ctx);
                        let (qkv, ..) = fused(m, N_HEAD, n_kv, 17 + p0);
                        let want = host_append(g, &qkv, table, &pos, V_SCALE, &[]);
                        let got = append_once(
                            a,
                            gpu,
                            gpu.unlabelled_sink(),
                            g,
                            &qkv,
                            table,
                            &pos,
                            V_SCALE,
                        )?;
                        let again = append_once(
                            a,
                            gpu,
                            gpu.unlabelled_sink(),
                            g,
                            &qkv,
                            table,
                            &pos,
                            V_SCALE,
                        )?;
                        let (same, rerun) =
                            (appended_equal(&got, &want), appended_equal(&got, &again));
                        launches += 1;
                        if !(same && rerun) {
                            println!(
                                "A1 base={base:e} n_kv={n_kv} m={m} p0={p0}: host transcription {same} (q {} qkv {} \
                                 k {} v {}) rerun {rerun} FAIL",
                                bits_equal(&got.q, &want.q),
                                bits_equal(&got.qkv, &want.qkv),
                                got.kc == want.kc,
                                got.vc == want.vc
                            );
                        }
                        ok &= same && rerun;
                    }
                }
            }
        }
        println!(
            "A1 append: {launches} launches (m = 1, 5, 64; positions 0, 127, 128, 129, 4095, 4096; bases 1e7, 1e4; \
             4 and 8 key heads) = the host transcription bit for bit (turned query, key in place, key and value \
             planes, every other row the sentinel), reruns identical {}",
            verdict(ok)
        );

        // A2: a position at the cache height.
        let layer = 13usize;
        let g = (N_HEAD, KV_FULL, ctx);
        let table = &tables[0].1;
        let pos = [5u32, ctx as u32, 7];
        let (qkv, ..) = fused(3, N_HEAD, KV_FULL, 23);
        let want = host_append(g, &qkv, table, &pos, V_SCALE, &[1]);
        let got = append_once(
            a,
            gpu,
            gpu.layer_sink(layer)?,
            g,
            &qkv,
            table,
            &pos,
            V_SCALE,
        )?;
        let raised = gpu.take_fault()?;
        let fault = Some(Fault::at(u32::try_from(layer)?, FaultSite::CachePos));
        let row = N_HEAD * HEAD_K192 + KV_FULL * (HEAD_K192 + HEAD);
        let nan_token = got.q[N_HEAD * HEAD_K192..2 * N_HEAD * HEAD_K192]
            .iter()
            .all(|v| v.is_nan())
            && got.qkv[row + N_HEAD * HEAD_K192..row + (N_HEAD + KV_FULL) * HEAD_K192]
                .iter()
                .all(|v| v.is_nan());
        let planes = got.kc == want.kc && got.vc == want.vc;
        let others = bits_equal(&got.q[..N_HEAD * HEAD_K192], &want.q[..N_HEAD * HEAD_K192])
            && bits_equal(
                &got.q[2 * N_HEAD * HEAD_K192..],
                &want.q[2 * N_HEAD * HEAD_K192..],
            );
        let a2 = raised == fault && nan_token && planes && others;
        println!(
            "A2 position {ctx} = the cache height: word {raised:?} (want {fault:?}), that token NaN {nan_token}, \
             no row appended and tokens 0, 2 as the host {planes}/{others} {}",
            verdict(a2)
        );
        ok &= a2;

        // Refusals by name.
        let table_d = DeviceBuffer::from_host(gpu.stream(), table)?;
        let stream = gpu.stream();
        let pos_d = DeviceBuffer::from_host(stream, &[1u32])?;
        let (qkv, ..) = fused(1, N_HEAD, KV_FULL, 5);
        let mut qkv_d = DeviceBuffer::from_host(stream, &qkv)?;
        let mut q_d = DeviceBuffer::<f32>::zeroed(stream, N_HEAD * HEAD_K192)?;
        let mut kc = DeviceBuffer::<u16>::zeroed(stream, KV_FULL * ctx * HEAD_K192)?;
        let mut vc = DeviceBuffer::<u16>::zeroed(stream, KV_FULL * ctx * HEAD)?;
        let mut refuse = |v_scale: f32, n_kv: usize, m: usize| {
            a.enqueue_neox_append_k192(
                stream,
                K192Args {
                    qkv: &mut qkv_d,
                    q: &mut q_d,
                    table: &table_d,
                    pos: &pos_d,
                    v_scale,
                    n_head: N_HEAD,
                    n_kv,
                    ctx,
                    m,
                    fault: gpu.unlabelled_sink(),
                    cache_k: &mut kc,
                    cache_v: &mut vc,
                },
            )
        };
        let r_nan = refused(refuse(f32::NAN, KV_FULL, 1), "finite");
        let r_inf = refused(refuse(f32::INFINITY, KV_FULL, 1), "finite");
        let r_zero = refused(refuse(V_SCALE, KV_FULL, 0), "need n_head");
        let r_short = refused(refuse(V_SCALE, KV_SWA, 1), "qkv.len()");
        let r_ok = r_nan && r_inf && r_zero && r_short;
        println!(
            "A2 refusals by name: NaN and infinite value scale, m = 0, a fused row too short for the key heads {}",
            verdict(r_ok)
        );
        ok &= r_ok;
        Ok(ok)
    }

    // -------------------------------------------------------------- ik's sets

    /// The sinks of layer `l`: `blk.L.attn_sinks.weight`, F32 `[n_head]`.
    fn sinks_of(split: &Split, l: usize, n_head: usize) -> Result<Vec<f32>, GateError> {
        let name = format!("blk.{l}.attn_sinks.weight");
        let (shard, info) = split
            .find(&name)
            .ok_or_else(|| format!("{name} is not in the model"))?;
        if info.ty != GgmlType::F32 || info.dims != [n_head as u64] {
            return Err(format!(
                "{name} is {:?} {:?}, want F32 [{n_head}]",
                info.ty, info.dims
            )
            .into());
        }
        let bytes = split.shard(shard).ok_or("shard out of range")?.data(info)?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    }

    /// The sets G1 and G2 read: a batch of five tokens, the decode steps at
    /// 4, 1,024 and 4,096 positions.
    const SETS: [&str; 4] = [
        "ref_mimo2",
        "ref_mimo2_step4",
        "ref_mimo2_d1k",
        "ref_mimo2_d4096",
    ];

    /// The visible keys of mask row `t`: `(first, live)` — the leading −inf run
    /// and the end of the zeros.
    fn mask_span(bits: &[u16], width: usize, t: usize) -> Result<(usize, usize), GateError> {
        let row = &bits[t * width..(t + 1) * width];
        let first = row.iter().take_while(|&&b| b == 0xfc00).count();
        let live = first + row[first..].iter().take_while(|&&b| b == 0).count();
        if row[live..].iter().any(|&b| b != 0xfc00) {
            return Err(format!("mask row {t} is not one visible run").into());
        }
        Ok((first, live))
    }

    /// G1 and G2 over ik's sets.
    fn ik_sets(k: &FlashGqaKernels, a: &RopeNeoxKernels, gpu: &Gpu) -> Result<bool, GateError> {
        let stream = gpu.stream();
        let mut ok = true;
        let mut split: Option<(Split, Hparams)> = None;
        let mut windows_bite = 0usize;
        for name in SETS {
            let dir = ref_dir_named(name);
            let man = RefManifest::open(&dir, &IK)?;
            if split.is_none() {
                let path = ref_model_path()?;
                let s = Split::open(&path).map_err(|e| format!("open {}: {e}", path.display()))?;
                if s.architecture() != Some("mimo2") {
                    return Err(format!(
                        "{} is {:?}, want mimo2",
                        path.display(),
                        s.architecture()
                    )
                    .into());
                }
                let hp = Hparams::read(&s).map_err(|e| e.to_string())?;
                if hp.value_scale.to_bits() != V_SCALE.to_bits()
                    || hp.window != WINDOW
                    || hp.head_k != HEAD_K192
                    || hp.head_v != HEAD
                    || hp.n_head != N_HEAD
                    || hp.rope_dims != ROT_K192
                {
                    return Err(format!("the file's attention constants moved: {hp:?}").into());
                }
                split = Some((s, hp));
            }
            let (sp_file, hp) = split.as_ref().ok_or("no model")?;
            let (pos0, toks, _) = man.step()?;
            let m = toks.len();
            let pos: Vec<u32> = (0..m as u32).map(|t| pos0 + t).collect();
            let mut layers = 0usize;
            let (mut bits_total, mut bits_same) = (0usize, 0usize);
            let (mut g1_ok, mut set_ok, mut worst_o, mut worst_i, mut worst_p) =
                (true, true, 0.0f64, 0.0f64, 0.0f64);
            while man.tensor(&format!("fa-{layers}"), 0).is_ok() {
                let l = layers;
                layers += 1;
                let swa = hp.kinds[l] == Kind::Swa;
                let n_kv = hp.kv_heads[l];
                if (n_kv == KV_SWA) != swa {
                    return Err(
                        format!("layer {l}: {n_kv} key heads but kind {:?}", hp.kinds[l]).into(),
                    );
                }
                let base = if swa { hp.rope_base_swa } else { hp.rope_base };
                let row = |n: &str| man.tensor(&format!("{n}-{l}"), 0);
                let rd = |n: &str| -> Result<Vec<f32>, GateError> {
                    Ok(ref_tensor_logical_in(&man.dir, row(n)?)?)
                };
                let (q_raw, k_raw, v_raw) = (rd("Qcur")?, rd("Kcur")?, rd("Vcur")?);
                let (q_roped, k_roped) = (rd("Qcur_roped")?, rd("Kcur_roped")?);
                // The fused rows, token-major.
                let mut qkv = Vec::new();
                for t in 0..m {
                    qkv.extend_from_slice(
                        &q_raw[t * N_HEAD * HEAD_K192..(t + 1) * N_HEAD * HEAD_K192],
                    );
                    qkv.extend_from_slice(&k_raw[t * n_kv * HEAD_K192..(t + 1) * n_kv * HEAD_K192]);
                    qkv.extend_from_slice(&v_raw[t * n_kv * HEAD..(t + 1) * n_kv * HEAD]);
                }
                let ctx = row("k")?.ne[1] as usize;
                let table = table_for(base, ctx)?;
                let got = append_once(
                    a,
                    gpu,
                    gpu.unlabelled_sink(),
                    (N_HEAD, n_kv, ctx),
                    &qkv,
                    &table,
                    &pos,
                    V_SCALE,
                )?;
                // G1: ik's roped rows, and the planes' new rows against ik's cache copies.
                let rw = N_HEAD * HEAD_K192 + n_kv * (HEAD_K192 + HEAD);
                let mut k_ours = Vec::new();
                for t in 0..m {
                    k_ours.extend_from_slice(
                        &got.qkv[t * rw + N_HEAD * HEAD_K192..t * rw + (N_HEAD + n_kv) * HEAD_K192],
                    );
                }
                let kcopy = f16_logical_bits(
                    &man.dir,
                    man.tensor(&format!("cache_k_l{l} (view) (copy of Kcur_roped-{l})"), 0)?,
                )?;
                let vcopy = f16_logical_bits(
                    &man.dir,
                    man.tensor(&format!("v_cache_view-{l} (copy of Vcur_scales-{l})"), 0)?,
                )?;
                let mut kc_same = true;
                let mut vc_same = true;
                for t in 0..m {
                    for j in 0..n_kv {
                        let at = (j * ctx + pos[t] as usize) * HEAD_K192;
                        kc_same &= got.kc[at..at + HEAD_K192]
                            == kcopy[(t * n_kv + j) * HEAD_K192..][..HEAD_K192];
                        let vat = (j * ctx + pos[t] as usize) * HEAD;
                        vc_same &=
                            got.vc[vat..vat + HEAD] == vcopy[(t * n_kv + j) * HEAD..][..HEAD];
                    }
                }
                let eq = |x: &[f32], y: &[f32]| -> usize {
                    x.iter()
                        .zip(y)
                        .filter(|(a, b)| a.to_bits() == b.to_bits())
                        .count()
                };
                let (qs, ks) = (eq(&got.q, &q_roped), eq(&k_ours, &k_roped));
                bits_total += q_roped.len() + k_roped.len();
                bits_same += qs + ks;
                let g1 = qs == q_roped.len() && ks == k_roped.len() && kc_same && vc_same;
                if !g1 {
                    println!(
                        "G1 {name} layer {l}: Qcur_roped {qs}/{} Kcur_roped {ks}/{} cache_k {kc_same} v_cache {vc_same} FAIL",
                        q_roped.len(),
                        k_roped.len()
                    );
                }
                g1_ok &= g1;

                // G2: the flash on ik's q/k/v with the layer's window and the file's sinks.
                let qrow = row("q")?;
                let krow = row("k")?;
                let vrow = row("v")?;
                let farow = row("fa")?;
                if qrow.ne != [HEAD_K192 as u64, m as u64, N_HEAD as u64, 1]
                    || krow.ne != [HEAD_K192 as u64, ctx as u64, n_kv as u64, 1]
                    || vrow.ne != [HEAD as u64, ctx as u64, n_kv as u64, 1]
                    || farow.ne != [HEAD as u64, N_HEAD as u64, m as u64, 1]
                {
                    return Err(format!(
                        "{name} layer {l}: q {:?} k {:?} v {:?} fa {:?} (m {m}, ctx {ctx})",
                        qrow.ne, krow.ne, vrow.ne, farow.ne
                    )
                    .into());
                }
                let q_flat = ref_tensor_logical_in(&man.dir, qrow)?;
                // [head][token][dim] -> token-major.
                let mut q_tok = vec![0.0f32; m * N_HEAD * HEAD_K192];
                for h in 0..N_HEAD {
                    for t in 0..m {
                        q_tok[(t * N_HEAD + h) * HEAD_K192..][..HEAD_K192]
                            .copy_from_slice(&q_flat[(h * m + t) * HEAD_K192..][..HEAD_K192]);
                    }
                }
                let want = ref_tensor_logical_in(&man.dir, farow)?;
                let cache = Cache {
                    n_kv,
                    ctx,
                    kc: f16_logical_bits(&man.dir, krow)?,
                    vc: f16_logical_bits(&man.dir, vrow)?,
                };
                let dev = cache.up(stream)?;
                let mask = man.input(if swa { "KQ_mask_swa" } else { "KQ_mask" }, 0)?;
                let width = mask.ne[0] as usize;
                let bits = mask_bits_in(&man.dir, mask)?;
                let mut counts = Vec::new();
                for (t, &p) in pos.iter().enumerate() {
                    let (first, live) = mask_span(&bits, width, t)?;
                    let want_first = if swa { first_of(live, WINDOW) } else { 0 };
                    if first != want_first || live != p as usize + 1 {
                        return Err(format!(
                            "{name} layer {l} row {t}: ik's mask shows keys {first}..{live}, the window rule says \
                             {want_first}..{}",
                            p + 1
                        )
                        .into());
                    }
                    counts.push(live as u32);
                }
                let sinks = if swa {
                    Some(sinks_of(sp_file, l, N_HEAD)?)
                } else {
                    None
                };
                if swa != sinks.is_some() {
                    return Err(format!("layer {l}: window kind and sinks disagree").into());
                }
                let sh = Shape {
                    n_kv,
                    n_head: N_HEAD,
                    ctx,
                    window: if swa { WINDOW } else { 0 },
                };
                let y = flash(
                    k,
                    gpu,
                    sh,
                    &q_tok,
                    &counts,
                    (&dev.0, &dev.1),
                    sinks.as_deref(),
                )?;
                let spec = Spec {
                    cache: &cache,
                    n_head: N_HEAD,
                    window: sh.window,
                    sinks: sinks.as_deref(),
                };
                let mut layer_ok = finite(&y);
                for t in 0..m {
                    let n = counts[t] as usize;
                    let qrow_t = &q_tok[t * N_HEAD * HEAD_K192..(t + 1) * N_HEAD * HEAD_K192];
                    let ex = exact(&spec, qrow_t, n, &heads_for(N_HEAD, n));
                    for e in &ex {
                        for d in 0..HEAD {
                            let o = f64::from(y[(t * N_HEAD + e.h) * HEAD + d]);
                            let iv = f64::from(want[(t * N_HEAD + e.h) * HEAD + d]);
                            let (d_o, d_i, d_p) =
                                ((o - e.o[d]).abs(), (iv - e.o[d]).abs(), (o - iv).abs());
                            worst_o = worst_o.max(d_o / e.bound_ours[d]);
                            worst_i = worst_i.max(d_i / e.bound_ik[d]);
                            worst_p = worst_p.max(d_p / (e.bound_ours[d] + e.bound_ik[d]));
                            layer_ok &= d_o <= e.bound_ours[d]
                                && d_i <= e.bound_ik[d]
                                && d_p <= e.bound_ours[d] + e.bound_ik[d];
                        }
                    }
                    // The window bites where the cache is deeper than it: the
                    // unwindowed attention (with the sinks) is far from the windowed.
                    if swa && n > WINDOW + 64 && t == 0 {
                        let full = Spec {
                            cache: &cache,
                            n_head: N_HEAD,
                            window: 0,
                            sinks: sinks.as_deref(),
                        };
                        let ex_full = exact(&full, qrow_t, n, &heads_for(N_HEAD, n));
                        let bites = ex.iter().zip(&ex_full).any(|(w, f)| {
                            (0..HEAD).any(|d| (w.o[d] - f.o[d]).abs() > 4.0 * w.bound_ours[d])
                        });
                        windows_bite += usize::from(bites);
                    }
                }
                if !layer_ok {
                    println!("G2 {name} layer {l} swa={swa} keys={counts:?}: FAIL");
                }
                set_ok &= layer_ok;
            }
            if layers == 0 {
                return Err(format!("{name}: no fa-0 row").into());
            }
            println!(
                "G1 {name}: {layers} layers, m={m} at position {pos0}: roped values bit-equal {bits_same}/{bits_total}; \
                 cache_k, v_cache rows bit for bit {}",
                verdict(g1_ok)
            );
            println!(
                "G2 {name}: measured / bound over {layers} layers: ours-exact {worst_o:.3e}, ik-exact {worst_i:.3e}, \
                 ours-ik {worst_p:.3e} {}",
                verdict(set_ok)
            );
            ok &= set_ok;
        }
        println!(
            "G2 window layers where the unwindowed attention is outside the band: {windows_bite}"
        );
        let bites = windows_bite > 0;
        if !bites {
            println!(
                "G2 no window layer shows the window in its output: the clause cannot see a missing window FAIL"
            );
        }
        Ok(ok && bites)
    }

    /// The window's owner against the gate's rule: `window_cut` and
    /// `window_segments` on the points that matter.
    fn owner() -> bool {
        let mut ok = true;
        for limit in [0usize, 1, 64, 127, 128, 129, 130, 4096, 4097, 65_536] {
            let (first, n_w) = window_cut(limit, WINDOW);
            ok &= first == first_of(limit, WINDOW) && first + n_w == limit;
            ok &= window_cut(limit, 0) == (0, limit);
        }
        ok &= window_segments(WINDOW) == 2 && window_segments(0) == SEGMENTS;
        println!(
            "F0 window_cut and window_segments = the gate's window rule {}",
            verdict(ok)
        );
        ok
    }

    pub fn run() -> Result<(), GateError> {
        bloomery_levers::at_main(&[])?;
        let gpu = Gpu::new()?;
        let k = FlashGqaKernels::load(gpu.context())?;
        let a = RopeNeoxKernels::load(gpu.context())?;
        println!(
            "gate_mimo2_flash: device {} — keys {HEAD_K192}, values {HEAD}, rope {ROPE_HEAD}/{ROT_K192}, window \
             {WINDOW}, group {GROUP}, scale {:e}",
            gpu.device_name()?,
            score_scale()
        );
        // `--only A,F2,G`: a partial run for a mutant's red line; the full gate takes no argument.
        let args: Vec<String> = std::env::args().skip(1).collect();
        let only: Option<Vec<String>> = match args.as_slice() {
            [] => None,
            [flag, list] if flag == "--only" => Some(list.split(',').map(str::to_owned).collect()),
            _ => {
                return Err(format!(
                    "usage: gate_mimo2_flash [--only F0,A,F1..F9,G]; got {args:?}"
                )
                .into());
            }
        };
        let want = |c: &str| only.as_ref().is_none_or(|o| o.iter().any(|x| x == c));
        let mut ok = true;
        if want("F0") {
            ok &= owner();
        }
        if want("A") {
            ok &= append_clauses(&a, &gpu)?;
        }
        if want("F1") {
            ok &= f1(&k, &gpu)?;
        }
        if want("F2") {
            ok &= f2(&k, &gpu)?;
        }
        if want("F3") {
            ok &= f3(&k, &gpu)?;
        }
        if want("F4") {
            ok &= f4(&k, &gpu)?;
        }
        if want("F5") {
            ok &= f5(&k, &gpu)?;
        }
        if want("F6") {
            ok &= f6(&k, &gpu)?;
        }
        if want("F7") {
            ok &= f7(&k, &gpu)?;
        }
        if want("F8") {
            ok &= f8(&k, &gpu)?;
        }
        if want("F9") {
            ok &= f9(&k, &gpu)?;
        }
        if want("G") {
            ok &= ik_sets(&k, &a, &gpu)?;
        }
        if let Some(o) = &only {
            println!("gate_mimo2_flash: PARTIAL run, clauses {o:?} only — not the gate");
        }
        println!("gate_mimo2_flash: {}", verdict(ok));
        if !ok {
            return Err(checks_failed());
        }
        Ok(())
    }
}
