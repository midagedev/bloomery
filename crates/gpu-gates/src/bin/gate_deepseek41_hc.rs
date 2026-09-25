//! GPU gate for the DeepSeek-V4.1 hyper-connections
//! (`bloomery_gpu_deepseek41::hc`): the chain RMS + split-K q3_K gemv +
//! HC_PRE, HC_POST with the next input fold, and the fold alone, at every
//! sub-layer the V4.1 oracle sets hold — the 5-token prefill set (T = 5) and
//! the decode-step sets (T = 1).
//!
//! PIN(2026-09-24): removed — the engine never runs `ds41_hc_pre_mix` (HC_PRE alone) or `ds41_hc_post_streams` (HC_POST alone); the kernels are gone with their checks.
//!
//! Three comparisons per op, the `gate_p4` form:
//! 1. **Kernel vs the host transcription of our rule** — bit-identical, and
//!    a rerun bit-identical. The rule is the module doc of `hc.rs`; the
//!    transcription walks the kernel's lanes, butterflies and piece order.
//!    HC_PRE's `exp` is f64 rounded to f32 on both sides.
//! 2. **ik's rule on the host vs the dump** — bit-identical. This proves the
//!    semantics: which streams feed the norm, which `pre` folds which
//!    streams (the lag), the output layout, where the multiply-adds fuse.
//!    ik's `exp` is glibc's `expf` (the host's `f32::exp`).
//! 3. **Kernel vs the dump** — HC_POST and the folds bit-identical (our rule
//!    is ik's there). The chain differs from ik's by rule: our gemv reads the raw streams
//!    quantized to q8_1 per 128 values and scales its result, ik's reads the
//!    normalized streams quantized to q8_K per 256. That difference is
//!    predicted per (token, row) in exact arithmetic from the dump's own
//!    inputs — `pred = s_ik · W·Q(x) - W·Q_K(xn)`, `s_ik` ik's RMS scale,
//!    `Q` our quantizer (pinned to `bloomery_gpu`'s below), `Q_K` ik's — and
//!    the measured difference minus `pred` must lie inside the rounding bound
//!    of the two float paths: our scale's distance from ik's, `gamma_n ·
//!    sum|terms|` for our sum order, and the same for ik's (iqk's
//!    `DequantizerQ3K`: one integer sum per super-block, the code offset `-4`
//!    as its own float term).
//!    HC_PRE's prediction is the f64 rule at the predicted mixes; the bound
//!    around it is the mix bound carried through HC_PRE to first order (the
//!    larger one-sided secant per mix, summed) plus each side's f32 rounding
//!    against the f64 rule at its own mixes. A ratio `|gap| / bound` above 1
//!    fails.
//!
//! Cross-checks with the crate's own kernels: the streams are quantized by
//! `bloomery_gpu`'s `q3k_quantize_q8_1` into a `Q8Act` at K = 20480, whose
//! q3 words and block scales must equal this transcription's, and its
//! unsplit `q3k_gemv` must agree with the split-K sum within `KERNEL_BAND`
//! (a different sum order).
//!
//! The fold alone is also checked on engram's own streams (`engram_out`)
//! against the dump: before an engram layer the chain's MoE sub-layer ends
//! with HC_POST alone and the glue folds the gated streams.
//!
//! The fault word (`bloomery_gpu::fault`): the chain's in-register q8_1 of
//! the streams has no form for a non-finite value, so a planted NaN in one
//! stream value must raise `FaultSite::HcQuant` rather than round to code 0.
//!
//! Holes, named: the T = 1 paths (ik's own HC_POST branch for one token)
//! come only from the decode-step sets, read when they are on the box and
//! named when they are not; `ds41_hc_pre` takes at most 8 tokens a launch,
//! so a longer prefill is split by its caller and no set covers that split.

#[cfg(not(feature = "deepseek41"))]
fn main() {
    eprintln!(
        "gate_deepseek41_hc: built without the `deepseek41` feature; see `just gate-gpu-ds41-hc`."
    );
    std::process::exit(2);
}

#[cfg(feature = "deepseek41")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_deepseek41_hc", gate::run())
}

#[cfg(feature = "deepseek41")]
mod gate {
    use bloomery_gpu::fused::readback_q8act;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act};
    use bloomery_gpu_deepseek41::hc::{
        HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcParams, HcPostArgs, HcPreArgs, HcPreScratch,
    };
    use bloomery_gpu_gates::hc_host::{self, exp_ik, exp_ours, hc_pre_f32};
    use bloomery_gpu_gates::ik_norm;
    use bloomery_gpu_gates::oracle::deepseek41::{D1, D1_UNFUSED, D2, D2_UNFUSED, STEP4};
    use bloomery_gpu_gates::oracle::{self, Set};
    use bloomery_gpu_gates::rounding::{U64, butterfly, gamma};
    use bloomery_gpu_gates::{
        GateError, KERNEL_BAND, RefManifest, RefRow, bits_equal, bytes_to_words, checks_failed,
        max_rel_err, ref_dir_named, ref_model_path, ref_tensor_of_in, tensor_bytes_as, verdict,
    };
    use cuda_core::{CudaStream, DeviceBuffer};
    use gguf::Split;
    use gguf::quant::{GgmlType, dequant_row, half_to_f32};
    use model::arch::Arch;

    /// The decode-step sets this gate reads, each when present (T = 1): the
    /// table's (`step_sets`) but d1n.
    const STEP_SETS: [&str; 5] = [STEP4, D1, D1_UNFUSED, D2, D2_UNFUSED];
    /// Bytes of one q3_K super-block.
    const Q3K_SB: usize = 110;
    // The host rule's layout is the kernels'.
    const _: () = assert!(hc_host::HC_STREAMS == HC_STREAMS && hc_host::HC_MIX == HC_MIX);

    /// The file's hyper-connection hyperparameters.
    struct Hp {
        n_embd: usize,
        n_layers: usize,
        iters: u32,
        eps: f32,
        rms_eps: f32,
    }

    impl Hp {
        fn read(split: &Split) -> Result<Hp, GateError> {
            let u = |k: &str| -> Result<u64, GateError> {
                split
                    .arch_get_u64(k)
                    .ok_or_else(|| format!("metadata {} missing", split.arch_key(k)).into())
            };
            let f = |k: &str| -> Result<f32, GateError> {
                split
                    .arch_get_f32(k)
                    .ok_or_else(|| format!("metadata {} missing", split.arch_key(k)).into())
            };
            let streams = u("hyper_connection.count")?;
            if streams != HC_STREAMS as u64 {
                return Err(
                    format!("hyper_connection.count = {streams}, the kernels take 4").into(),
                );
            }
            Ok(Hp {
                n_embd: usize::try_from(u("embedding_length")?)?,
                n_layers: usize::try_from(u("block_count")?)?,
                iters: u32::try_from(u("hyper_connection.sinkhorn_iterations")?)?,
                eps: f("hyper_connection.epsilon")?,
                rms_eps: f("attention.layer_norm_rms_epsilon")?,
            })
        }

        fn k(&self) -> usize {
            HC_STREAMS * self.n_embd
        }
    }

    // ------------------------------------------------------------ ik's rules

    /// ik's q8_K of `x` (`iqk_quantize_row_q8_K`, the AVX2 path the box
    /// runs): per 256 values `d = amax/127` and the codes
    /// `round_ties_even(v * (127/amax))`.
    fn q8k_ik(x: &[f32]) -> (Vec<i32>, Vec<f32>) {
        let mut q = vec![0i32; x.len()];
        let mut d = Vec::with_capacity(x.len() / 256);
        for (xb, qb) in x
            .as_chunks::<256>()
            .0
            .iter()
            .zip(q.as_chunks_mut::<256>().0)
        {
            let amax = xb.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
            let id = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            for (qi, &v) in qb.iter_mut().zip(xb) {
                *qi = (id * v).round_ties_even() as i32;
            }
            d.push(amax / 127.0);
        }
        (q, d)
    }

    /// HC_PRE in f64 throughout: the reference the chain's bound is carried
    /// through.
    fn hc_pre_f64(mix: &[f64], sc: [f64; 3], base: &[f64], eps: f64, iters: u32) -> [f64; HC_MIX] {
        let sigmoid = |i: usize, s: f64| 1.0 / (1.0 + (-(mix[i] * s + base[i])).exp());
        let mut out = [0.0f64; HC_MIX];
        let (head, comb) = out.split_at_mut(2 * HC_STREAMS);
        let (pre, post) = head.split_at_mut(HC_STREAMS);
        for (i, (p, q)) in pre.iter_mut().zip(post.iter_mut()).enumerate() {
            *p = sigmoid(i, sc[0]) + eps;
            *q = 2.0 * sigmoid(HC_STREAMS + i, sc[1]);
        }
        let mut m = [0.0f64; 16];
        for (k, mk) in m.iter_mut().enumerate() {
            *mk = mix[8 + k] * sc[2] + base[8 + k];
        }
        for row in m.as_chunks_mut::<4>().0 {
            let mx = row.iter().copied().fold(f64::MIN, f64::max);
            let mut sum = 0.0;
            for v in row.iter_mut() {
                *v = (*v - mx).exp();
                sum += *v;
            }
            for v in row.iter_mut() {
                *v = *v / sum + eps;
            }
        }
        let col = |m: &mut [f64; 16]| {
            let mut s = [eps; 4];
            for (c, sc) in s.iter_mut().enumerate() {
                *sc += m[c] + m[4 + c] + m[8 + c] + m[12 + c];
            }
            for (k, v) in m.iter_mut().enumerate() {
                *v /= s[k % 4];
            }
        };
        col(&mut m);
        for _ in 1..iters {
            for row in m.as_chunks_mut::<4>().0 {
                let s = eps + row[0] + row[1] + row[2] + row[3];
                for v in row.iter_mut() {
                    *v /= s;
                }
            }
            col(&mut m);
        }
        comb.copy_from_slice(&m);
        out
    }

    /// HC_POST of one value, ours and ik's: `fma(x, post_i, comb_0i r_0)`,
    /// then `fma(comb_ji, r_j, .)` for j = 1..3.
    fn post_elem(x: f32, r: [f32; 4], hc: &[f32]) -> [f32; 4] {
        let mut o = [0.0f32; 4];
        for (i, oi) in o.iter_mut().enumerate() {
            let mut s = x.mul_add(hc[4 + i], hc[8 + i] * r[0]);
            for (j, &rj) in r.iter().enumerate().skip(1) {
                s = hc[8 + 4 * j + i].mul_add(rj, s);
            }
            *oi = s;
        }
        o
    }

    /// The fold of four stream values by `pre` (MUL_MULTI_ADD): `o_0 pre_0`,
    /// then one fused multiply-add per stream.
    fn fold_elem(o: [f32; 4], pre: &[f32]) -> f32 {
        let mut y = o[0] * pre[0];
        for (&oj, &pj) in o.iter().zip(pre).skip(1) {
            y = oj.mul_add(pj, y);
        }
        y
    }

    /// Token `tt`'s four stream values at `d` (`[t][4][n]` streams).
    fn streams_at(s: &[f32], n: usize, tt: usize, d: usize) -> [f32; 4] {
        let b = 4 * tt * n + d;
        [s[b], s[b + n], s[b + 2 * n], s[b + 3 * n]]
    }

    /// HC_POST + fold over `t` tokens of `n` values: new streams `[t][4][n]`
    /// and the fold `[t][n]`, `hc` in the kernel layout.
    fn post_host(x: &[f32], res: &[f32], hc: &[f32], n: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let mut out = vec![0.0f32; 4 * n * t];
        let mut fold = vec![0.0f32; n * t];
        for tt in 0..t {
            let h = &hc[tt * HC_MIX..(tt + 1) * HC_MIX];
            for d in 0..n {
                let o = post_elem(x[tt * n + d], streams_at(res, n, tt, d), h);
                for (i, &oi) in o.iter().enumerate() {
                    out[4 * tt * n + i * n + d] = oi;
                }
                fold[tt * n + d] = fold_elem(o, &h[..4]);
            }
        }
        (out, fold)
    }

    /// The fold alone over `t` tokens.
    fn fold_host(s: &[f32], hc: &[f32], n: usize, t: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n * t];
        for (i, yi) in y.iter_mut().enumerate() {
            let (tt, d) = (i / n, i % n);
            *yi = fold_elem(streams_at(s, n, tt, d), &hc[tt * HC_MIX..tt * HC_MIX + 4]);
        }
        y
    }

    /// An HC_PRE node's output (`[pre S T][post S T][comb S S T]`, ik's
    /// layout) in the kernel's layout, `[t][pre, post, comb]`.
    fn node_to_hc(node: &[f32], t: usize) -> Vec<f32> {
        let mut hc = vec![0.0f32; HC_MIX * t];
        for (tt, h) in hc.as_chunks_mut::<HC_MIX>().0.iter_mut().enumerate() {
            h[..4].copy_from_slice(&node[4 * tt..4 * tt + 4]);
            h[4..8].copy_from_slice(&node[4 * t + 4 * tt..4 * t + 4 * tt + 4]);
            h[8..].copy_from_slice(&node[8 * t + 16 * tt..8 * t + 16 * tt + 16]);
        }
        hc
    }

    // ---------------------------------------------- our chain, transcribed

    /// A q3_K super-block in integers: each weight's code (-4..3), the
    /// sixteen sub-block scales (-32..31), the scale `d` — ggml's
    /// `dequantize_row_q3_K` before its two multiplies.
    struct Q3Sb {
        q: [i32; 256],
        sc: [i32; 16],
        d: f32,
    }

    fn q3k_sb(b: &[u8; Q3K_SB]) -> Q3Sb {
        let (hm, qs, sb) = (&b[0..32], &b[32..96], &b[96..108]);
        let d = half_to_f32(u16::from_le_bytes([b[108], b[109]]));
        let word =
            |i: usize| u32::from_le_bytes([sb[4 * i], sb[4 * i + 1], sb[4 * i + 2], sb[4 * i + 3]]);
        let (a0, a1, tmp) = (word(0), word(1), word(2));
        let (km1, km2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
        let aux = [
            (a0 & km2) | ((tmp & km1) << 4),
            (a1 & km2) | (((tmp >> 2) & km1) << 4),
            ((a0 >> 4) & km2) | (((tmp >> 4) & km1) << 4),
            ((a1 >> 4) & km2) | (((tmp >> 6) & km1) << 4),
        ];
        let mut sc = [0i32; 16];
        for (i, s) in sc.iter_mut().enumerate() {
            *s = i32::from(aux[i / 4].to_le_bytes()[i % 4] as i8) - 32;
        }
        let mut q = [0i32; 256];
        for (k, qk) in q.iter_mut().enumerate() {
            let (n, j, sub, l) = (k / 128, (k % 128) / 32, (k % 32) / 16, k % 16);
            let low = i32::from((qs[32 * n + 16 * sub + l] >> (2 * j)) & 3);
            let high = (hm[16 * sub + l] >> (4 * n + j)) & 1;
            *qk = low - if high == 0 { 4 } else { 0 };
        }
        Q3Sb { q, sc, d }
    }

    /// Sub-block scale index of weight `k` of a super-block.
    fn q3k_scale_index(k: usize) -> usize {
        8 * (k / 128) + 2 * ((k % 128) / 32) + (k % 32) / 16
    }

    /// q8_1 of one 128-value block by `q3k_quantize_q8_1`'s rule: the codes
    /// and `d = amax/127` (1 for an all-zero block).
    fn q8_block(x: &[f32]) -> ([i32; 128], f32) {
        let amax = x.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let d = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        let mut q = [0i32; 128];
        for (qi, &v) in q.iter_mut().zip(x) {
            *qi = (v / d).round().clamp(-127.0, 127.0) as i32;
        }
        (q, d)
    }

    /// One site's chain by our rule for `t` tokens of raw streams: the
    /// unscaled split-K sums `[t][24]`, the scaled mixes `[t][24]`, and the
    /// q8_1 codes and block scales it quantized the streams to.
    struct Chain {
        raw: Vec<f32>,
        mixes: Vec<f32>,
        q8: Vec<i32>,
        d8: Vec<f32>,
    }

    fn chain_host(sbs: &[Q3Sb], x: &[f32], k: usize, t: usize, rms_eps: f32) -> Chain {
        let (n_sb, np) = (k / 256, k / HC_PIECE);
        let mut q8 = vec![0i32; k * t];
        let mut d8 = vec![0.0f32; k / 128 * t];
        for ((xb, qb), db) in x
            .as_chunks::<128>()
            .0
            .iter()
            .zip(q8.as_chunks_mut::<128>().0)
            .zip(d8.iter_mut())
        {
            let (q, d) = q8_block(xb);
            qb.copy_from_slice(&q);
            *db = d;
        }
        // Per (piece, token): the butterfly of the lanes' squares, and each
        // row's `piece_sum` — lane group `gi` (lanes 8gi..8gi + 7) holds the
        // 128-value block `gi & 1` of super-block `2p + (gi >> 1)`.
        let mut part = vec![0.0f32; np * t * HC_MIX];
        let mut sq = vec![0.0f32; np * t];
        for p in 0..np {
            for tt in 0..t {
                let mut sq_lane = [0.0f32; 32];
                for (lane, sql) in sq_lane.iter_mut().enumerate() {
                    let (w16, sb) = (lane & 15, 2 * p + (lane >> 4));
                    let base = tt * k + 256 * sb + 128 * (w16 >> 3) + 4 * (w16 & 7);
                    let mut acc = 0.0f32;
                    for j in 0..4 {
                        for b in 0..4 {
                            let v = x[base + 32 * j + b];
                            acc = v.mul_add(v, acc);
                        }
                    }
                    *sql = acc;
                }
                sq[p * t + tt] = butterfly(sq_lane);
                for r in 0..HC_MIX {
                    let group = |gi: usize| -> (f32, f32) {
                        let (sb, g) = (2 * p + (gi >> 1), gi & 1);
                        let s = &sbs[r * n_sb + sb];
                        let blk = tt * k + 256 * sb + 128 * g;
                        let dot: i32 = (128 * g..128 * g + 128)
                            .map(|v| s.q[v] * s.sc[q3k_scale_index(v)] * q8[blk - 128 * g + v])
                            .sum();
                        (dot as f32, d8[blk / 128] * s.d)
                    };
                    let [(g0, dd0), (g1, dd1), (g2, dd2), (g3, dd3)] =
                        [group(0), group(1), group(2), group(3)];
                    let lane0 = g0.mul_add(dd0, g1 * dd1);
                    let lane16 = g2.mul_add(dd2, g3 * dd3);
                    part[(p * t + tt) * HC_MIX + r] = lane0 + lane16;
                }
            }
        }
        let mut raw = vec![0.0f32; t * HC_MIX];
        let mut mixes = vec![0.0f32; t * HC_MIX];
        for tt in 0..t {
            let mut s = sq[tt];
            for p in 1..np {
                s += sq[p * t + tt];
            }
            let scale = 1.0 / (s / k as f32 + rms_eps).sqrt();
            for r in 0..HC_MIX {
                let mut a = part[tt * HC_MIX + r];
                for p in 1..np {
                    a += part[(p * t + tt) * HC_MIX + r];
                }
                raw[tt * HC_MIX + r] = a;
                mixes[tt * HC_MIX + r] = a * scale;
            }
        }
        Chain { raw, mixes, q8, d8 }
    }

    /// `cores::q3_slot`: the u64 pair slot of value-order word `v4` in a
    /// column of the q3 buffer; bit 3 of `v4` picks the half.
    fn q3_slot(v4: usize) -> usize {
        64 * (v4 >> 7)
            + 32 * ((v4 >> 4) & 1)
            + 16 * ((v4 >> 6) & 1)
            + 8 * ((v4 >> 5) & 1)
            + (v4 & 7)
    }

    /// The crate quantizer's bytes equal the transcription's: every q3 word
    /// in its permuted slot and every block scale, bit for bit.
    fn q8act_matches(q3: &[u64], d8: &[f32], ch: &Chain, k: usize, t: usize) -> bool {
        let col_slots = 64 * (k / 512);
        let d8_ok = d8.len() >= ch.d8.len() && bits_equal(&d8[..ch.d8.len()], &ch.d8);
        let q3_ok = (0..t * k / 4).all(|w| {
            let (tt, v4) = (w / (k / 4), w % (k / 4));
            let word = (0..4).fold(0u32, |acc, b| {
                acc | (((ch.q8[tt * k + 4 * v4 + b] as u32) & 0xff) << (8 * b))
            });
            let pair = q3[tt * col_slots + q3_slot(v4)];
            let got = if v4 & 8 == 0 {
                pair as u32
            } else {
                (pair >> 32) as u32
            };
            got == word
        });
        d8_ok && q3_ok
    }

    // ------------------------------------------------ the chain's prediction

    /// Per (token, row) of the chain: the predicted difference of our rule
    /// from ik's in exact arithmetic, and the bound on the float rounding of
    /// both paths around it.
    struct MixPrediction {
        pred: Vec<f64>,
        bound: Vec<f64>,
    }

    /// The inputs of [`predict_mix`] that come from the dump: the streams and
    /// ik's norm output, `k` values per token.
    struct DumpIn<'a> {
        x: &'a [f32],
        xn: &'a [f32],
        k: usize,
        rms_eps: f32,
    }

    /// The module doc's prediction. Every product below is exact in f64 (a
    /// weight has at most 20 significant bits, a code times its scale 32);
    /// the f64 sums add at most `4 k u64` of their magnitudes, kept in the
    /// bound.
    fn predict_mix(p: &SiteParams, ch: &Chain, dump: &DumpIn<'_>) -> MixPrediction {
        let k = dump.k;
        let (np, n_sb, t) = (k / HC_PIECE, k / 256, dump.x.len() / k);
        // Our scale against ik's: our f32 sum of squares runs 16 fused
        // multiply-adds per lane, 5 butterfly levels and np - 1 piece sums;
        // ik rounds each square once. Both round the mean and the `+ eps`
        // once each (5 more), then the square root and the reciprocal.
        let delta_s = gamma(16 + 5 + (np - 1) + 5) / 2.0 + gamma(4);
        // Our mix: a block's integer dot is exact; then `dd`, the product,
        // the fused multiply-add, the add, np - 1 piece sums and the scale.
        // ik's: one product per term, one fused accumulation per super-block
        // into each of two accumulators, their sum and a 3-level horizontal
        // sum.
        let (g_ours, g_ik) = (gamma(np + 4), gamma(n_sb + 6));
        let slack = 4.0 * k as f64 * U64;
        let mut pred = vec![0.0f64; t * HC_MIX];
        let mut bound = vec![0.0f64; t * HC_MIX];
        for tt in 0..t {
            let s = f64::from(ik_norm::scale(&dump.x[tt * k..(tt + 1) * k], dump.rms_eps));
            let (qk, dk) = q8k_ik(&dump.xn[tt * k..(tt + 1) * k]);
            let ours: Vec<f64> = (tt * k..(tt + 1) * k)
                .map(|j| f64::from(ch.q8[j]) * f64::from(ch.d8[j / 128]))
                .collect();
            let theirs: Vec<f64> = qk
                .iter()
                .enumerate()
                .map(|(i, &q)| f64::from(q) * f64::from(dk[i / 256]))
                .collect();
            for r in 0..HC_MIX {
                let (w, w_ik) = (&p.w[r * k..(r + 1) * k], &p.w_ik[r * k..(r + 1) * k]);
                let (mut a, mut abs_a, mut b, mut mag_b) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
                for (((&wi, &wki), &oi), &ti) in w.iter().zip(w_ik).zip(&ours).zip(&theirs) {
                    a += wi * oi;
                    abs_a += (wi * oi).abs();
                    b += wi * ti;
                    mag_b += wki * ti.abs();
                }
                let o = tt * HC_MIX + r;
                pred[o] = s * a - b;
                bound[o] = delta_s * s * a.abs()
                    + g_ours * s * (1.0 + delta_s) * abs_a
                    + g_ik * mag_b
                    + slack * (s * abs_a + mag_b);
            }
        }
        MixPrediction { pred, bound }
    }

    /// Per (token, output) of HC_PRE: the prediction `f(m_d + pred) - f(m_d)`
    /// of the f64 rule `f`, and the bound around it (module doc).
    fn predict_hc(
        m_d: &[f32],
        m_k: &[f32],
        mp: &MixPrediction,
        p: &SiteParams,
        hp: &Hp,
    ) -> (Vec<f64>, Vec<f64>) {
        let sc = p.scale.map(f64::from);
        let base: Vec<f64> = p.base.iter().map(|&v| f64::from(v)).collect();
        let f = |m: &[f64]| hc_pre_f64(m, sc, &base, f64::from(hp.eps), hp.iters);
        let mut pred = Vec::with_capacity(m_d.len());
        let mut bound = Vec::with_capacity(m_d.len());
        for (tt, (md32, mk32)) in m_d
            .as_chunks::<HC_MIX>()
            .0
            .iter()
            .zip(m_k.as_chunks::<HC_MIX>().0)
            .enumerate()
        {
            let at = tt * HC_MIX..(tt + 1) * HC_MIX;
            let md: Vec<f64> = md32.iter().map(|&v| f64::from(v)).collect();
            let mp_at: Vec<f64> = md
                .iter()
                .zip(&mp.pred[at.clone()])
                .map(|(a, b)| a + b)
                .collect();
            let (f_d, f_p) = (f(&md), f(&mp_at));
            let mut sec = [0.0f64; HC_MIX];
            for (r, &b) in mp.bound[at].iter().enumerate() {
                let (mut up, mut dn) = (mp_at.clone(), mp_at.clone());
                up[r] += b;
                dn[r] -= b;
                let (fu, fd) = (f(&up), f(&dn));
                for (o, s) in sec.iter_mut().enumerate() {
                    *s += (fu[o] - f_p[o]).abs().max((fd[o] - f_p[o]).abs());
                }
            }
            let mk: Vec<f64> = mk32.iter().map(|&v| f64::from(v)).collect();
            let f_k = f(&mk);
            let k32 = hc_pre_f32(mk32, p.scale, &p.base, hp.eps, hp.iters, exp_ours);
            let d32 = hc_pre_f32(md32, p.scale, &p.base, hp.eps, hp.iters, exp_ik);
            for (o, s) in sec.iter().enumerate() {
                pred.push(f_p[o] - f_d[o]);
                bound.push(
                    s + (f64::from(k32[o]) - f_k[o]).abs() + (f64::from(d32[o]) - f_d[o]).abs(),
                );
            }
        }
        (pred, bound)
    }

    /// The largest `|(a - b) - pred| / bound`.
    fn gap_ratio(a: &[f32], b: &[f32], pred: &[f64], bound: &[f64]) -> f64 {
        a.iter()
            .zip(b)
            .zip(pred.iter().zip(bound))
            .map(|((&x, &y), (&p, &bd))| {
                let gap = ((f64::from(x) - f64::from(y)) - p).abs();
                if gap == 0.0 { 0.0 } else { gap / bd }
            })
            .fold(0.0f64, f64::max)
    }

    /// `max|v| / max|of|`.
    fn rel_to(v: impl Iterator<Item = f64>, of: &[f32]) -> f64 {
        let num = v.fold(0.0f64, |m, x| m.max(x.abs()));
        let den = of.iter().fold(0.0f64, |m, &x| m.max(f64::from(x).abs()));
        num / den
    }

    // ------------------------------------------------------------ the sites

    /// One sub-layer's parameters from the file, host and device.
    struct SiteParams {
        scale: [f32; 3],
        base: Vec<f32>,
        sbs: Vec<Q3Sb>,
        /// Every weight, `[row][k]`, exactly (ggml's dequantized value).
        w: Vec<f64>,
        /// Per weight, the magnitude ik's dot sums it at: `|d sc| (q + 8)`,
        /// the code offset by 4 in integers plus the `-4` term apart.
        w_ik: Vec<f64>,
        w_dev: DeviceTensor<u32>,
        scale_dev: DeviceBuffer<f32>,
        base_dev: DeviceBuffer<f32>,
    }

    fn site_params(
        split: &Split,
        stream: &CudaStream,
        layer: usize,
        sub: &str,
        k: usize,
    ) -> Result<SiteParams, GateError> {
        let bytes = |name: &str, ty: GgmlType, dims: &[u64]| -> Result<&[u8], GateError> {
            let (shard, _) = split
                .find(name)
                .ok_or_else(|| format!("tensor {name} not in the model"))?;
            let g = split.shard(shard).ok_or("shard index out of range")?;
            Ok(tensor_bytes_as(g, name, ty, Some(dims))?.1)
        };
        let f32s = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect()
        };
        let fn_name = format!("blk.{layer}.hc_{sub}_fn.weight");
        let wb = bytes(&fn_name, GgmlType::Q3_K, &[k as u64, HC_MIX as u64])?;
        let scale_v = f32s(bytes(
            &format!("blk.{layer}.hc_{sub}_scale.weight"),
            GgmlType::F32,
            &[3],
        )?);
        let base = f32s(bytes(
            &format!("blk.{layer}.hc_{sub}_base.weight"),
            GgmlType::F32,
            &[HC_MIX as u64],
        )?);
        let (n_sb, row_bytes) = (k / 256, k / 256 * Q3K_SB);
        let sbs: Vec<Q3Sb> = wb.as_chunks::<Q3K_SB>().0.iter().map(q3k_sb).collect();
        let mut w = Vec::with_capacity(HC_MIX * k);
        let mut w_ik = Vec::with_capacity(HC_MIX * k);
        let mut row = vec![0.0f32; k];
        for (r, rb) in wb.chunks_exact(row_bytes).enumerate() {
            dequant_row(GgmlType::Q3_K, rb, &mut row)?;
            // The integer decode the transcription uses is ggml's: `(d sc) q`
            // as `dequant_row` rounds it, and exact.
            for (i, &v) in row.iter().enumerate() {
                let s = &sbs[r * n_sb + i / 256];
                let (q, sc) = (s.q[i % 256], s.sc[q3k_scale_index(i % 256)]);
                let exact = f64::from(s.d) * f64::from(sc) * f64::from(q);
                if f64::from(v) != exact {
                    return Err(format!(
                        "{fn_name}: row {r} value {i}: dequant_row {v}, integer decode {exact}"
                    )
                    .into());
                }
                w.push(exact);
                w_ik.push((f64::from(s.d) * f64::from(sc)).abs() * f64::from(q + 8));
            }
        }
        Ok(SiteParams {
            scale: [scale_v[0], scale_v[1], scale_v[2]],
            w_dev: DeviceTensor::upload(stream, &bytes_to_words(wb), HC_MIX, row_bytes / 4)?,
            scale_dev: DeviceBuffer::from_host(stream, &scale_v)?,
            base_dev: DeviceBuffer::from_host(stream, &base)?,
            base,
            sbs,
            w,
            w_ik,
        })
    }

    /// The manifest rows of one sub-layer's hyper-connection.
    struct Site<'m> {
        layer: usize,
        ffn: bool,
        node: &'m RefRow,
        mixes: &'m RefRow,
        rms: &'m RefRow,
        streams: &'m RefRow,
        x: &'m RefRow,
        post: &'m RefRow,
        /// The fold of `post` by the next sub-layer's `pre`, when that fold
        /// reads `post` itself. `None` before an engram layer (it rewrites
        /// `post` first) and after the last layer (the head folds): there the
        /// boundary takes HC_POST alone and the fold is a lone launch.
        fold: Option<&'m RefRow>,
    }

    impl Site<'_> {
        fn sub(&self) -> &'static str {
            if self.ffn { "ffn" } else { "attn" }
        }
    }

    fn src0(r: &RefRow) -> &str {
        r.src0.as_deref().unwrap_or("")
    }

    /// Every sub-layer's rows, found by what each node reads — the HC_PRE
    /// node by its mixes and scale, the norm by the mixes' name, the streams
    /// by the norm's input — and proven by type, dims and op.
    fn sites<'m>(man: &'m RefManifest, hp: &Hp, t: u64) -> Result<Vec<Site<'m>>, GateError> {
        let (n, k) = (hp.n_embd as u64, hp.k() as u64);
        let mut out = Vec::with_capacity(2 * hp.n_layers);
        for layer in 0..hp.n_layers {
            for (occ, sub) in [(0u32, "attn"), (1u32, "ffn")] {
                let mixes_name = format!("hc_pre_mixes-{layer}");
                let scale_name = format!("blk.{layer}.hc_{sub}_scale.weight");
                let node = man
                    .tensors
                    .iter()
                    .find(|r| {
                        r.op == "HC_PRE"
                            && src0(r) == mixes_name
                            && r.src1.as_deref() == Some(scale_name.as_str())
                    })
                    .ok_or_else(|| {
                        format!("no HC_PRE node reads {mixes_name} with {scale_name}")
                    })?;
                node.expect(&node.name, "f32", [24 * t, 1, 1, 1], "HC_PRE")?;
                let mixes = man.tensor(&mixes_name, occ)?;
                mixes.expect(&mixes_name, "f32", [24, t, 1, 1], "MUL_MAT")?;
                if src0(mixes) != format!("blk.{layer}.hc_{sub}_fn.weight") {
                    return Err(format!(
                        "{mixes_name}/{occ} reads {}, not hc_{sub}_fn",
                        src0(mixes)
                    )
                    .into());
                }
                let rms = man.tensor(&format!("hc_pre-{layer}"), occ)?;
                rms.expect(&rms.name, "f32", [k, t, 1, 1], "RMS_NORM")?;
                let s_name = src0(rms)
                    .strip_suffix(" (reshaped)")
                    .ok_or_else(|| format!("hc_pre-{layer}/{occ} reads {}", src0(rms)))?;
                let streams = man.tensor(s_name, 0)?;
                streams.expect(s_name, "f32", [n, 4, t, 1], "in")?;
                let (x_name, post_name) = if occ == 0 {
                    (format!("attn_out-{layer}"), format!("hc_attn_post-{layer}"))
                } else {
                    (format!("ffn_out-{layer}"), format!("l_out-{layer}"))
                };
                let x = man.tensor(&x_name, 0)?;
                x.expect(&x_name, "f32", [n, t, 1, 1], "in")?;
                let post = man.tensor(&post_name, 0)?;
                post.expect(&post_name, "f32", [n, 4, t, 1], "HC_POST")?;
                if src0(post) != x_name {
                    return Err(format!("{post_name} reads {}, not {x_name}", src0(post)).into());
                }
                let fold_name = if occ == 0 {
                    Some(format!("hc_ffn_pre-{layer}"))
                } else if layer + 1 < hp.n_layers {
                    Some(format!("hc_attn_pre-{}", layer + 1))
                } else {
                    None
                };
                let fold = match fold_name {
                    Some(f) => {
                        let r = man.tensor(&f, 0)?;
                        r.expect(&f, "f32", [n, t, 1, 1], "MUL_MULTI_ADD")?;
                        let engram = format!("engram_out-{}", layer + 1);
                        if src0(r) == post_name {
                            Some(r)
                        } else if src0(r) == engram {
                            None
                        } else {
                            return Err(format!(
                                "{f} reads {}, neither {post_name} nor {engram}",
                                src0(r)
                            )
                            .into());
                        }
                    }
                    None => None,
                };
                out.push(Site {
                    layer,
                    ffn: occ == 1,
                    node,
                    mixes,
                    rms,
                    streams,
                    x,
                    post,
                    fold,
                });
            }
        }
        Ok(out)
    }

    /// Tokens of a set: the third dim of `hc_init` (`[n, 4, T]`).
    fn tokens_of(man: &RefManifest) -> Result<usize, GateError> {
        Ok(usize::try_from(man.tensor("hc_init", 0)?.ne[2])?)
    }

    // --------------------------------------------------------------- driver

    struct Ctx<'a> {
        split: &'a Split,
        hp: &'a Hp,
        gpu: &'a Gpu,
        hck: &'a HcKernels,
        scratch: HcPreScratch,
        ok: bool,
    }

    /// What `check_site` hands on: the kernel's fold of HC_POST's streams and
    /// the dump's HC_PRE output, kernel layout.
    struct SiteOut {
        fold: Vec<f32>,
        hc: Vec<f32>,
    }

    /// Every check of one site; prints its two lines.
    fn check_site(
        cx: &mut Ctx<'_>,
        man: &RefManifest,
        set: &str,
        site: &Site<'_>,
        t: usize,
    ) -> Result<SiteOut, GateError> {
        let (hp, stream) = (cx.hp, cx.gpu.stream());
        let (n, k) = (hp.n_embd, hp.k());
        let params = site_params(cx.split, stream, site.layer, site.sub(), k)?;
        let load = |r: &RefRow| ref_tensor_of_in(&man.dir, r);
        let streams = load(site.streams)?;
        let xn = load(site.rms)?;
        let mix_d = load(site.mixes)?;
        let hc_d = node_to_hc(&load(site.node)?, t);

        // 2. ik's rule vs the dump: the norm and HC_PRE.
        let rms_h: Vec<f32> = streams
            .chunks_exact(k)
            .flat_map(|c| {
                let s = ik_norm::scale(c, hp.rms_eps);
                c.iter().map(move |&v| v * s)
            })
            .collect();
        let rms_bit = bits_equal(&rms_h, &xn);
        let pre_of = |m: &[f32], exp: fn(f32) -> f32| -> Vec<f32> {
            m.as_chunks::<HC_MIX>()
                .0
                .iter()
                .flat_map(|mt| hc_pre_f32(mt, params.scale, &params.base, hp.eps, hp.iters, exp))
                .collect()
        };
        let pre_ik_bit = bits_equal(&pre_of(&mix_d, exp_ik), &hc_d);

        // 1. The chain kernel vs our rule, and a rerun.
        let x_dev = DeviceBuffer::from_host(stream, &streams)?;
        let p = HcParams {
            w: &params.w_dev,
            scale: &params.scale_dev,
            base: &params.base_dev,
            eps: hp.eps,
            iters: hp.iters,
        };
        let a = HcPreArgs {
            params: &p,
            x: &x_dev,
            tokens: t,
            rms_eps: hp.rms_eps,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut mixes_dev = DeviceBuffer::<f32>::zeroed(stream, HC_MIX * t)?;
        let mut hc_dev = DeviceBuffer::<f32>::zeroed(stream, HC_MIX * t)?;
        cx.hck
            .enqueue_pre(stream, &a, &mut cx.scratch, &mut mixes_dev, &mut hc_dev)?;
        stream.synchronize()?;
        let (mix_k, hc_k) = (mixes_dev.to_host_vec(stream)?, hc_dev.to_host_vec(stream)?);
        cx.hck
            .enqueue_pre(stream, &a, &mut cx.scratch, &mut mixes_dev, &mut hc_dev)?;
        stream.synchronize()?;
        let rerun = bits_equal(&mix_k, &mixes_dev.to_host_vec(stream)?)
            && bits_equal(&hc_k, &hc_dev.to_host_vec(stream)?);
        let ch = chain_host(&params.sbs, &streams, k, t, hp.rms_eps);
        let mix_bit = bits_equal(&mix_k, &ch.mixes);
        let hc_bit = bits_equal(&hc_k, &pre_of(&ch.mixes, exp_ours));

        // The crate's quantizer and unsplit gemv at K = 20480.
        let mut act = Q8Act::with_k(stream, t, k)?;
        cx.gpu.enqueue_quantize_q8_1(&x_dev, &mut act)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, HC_MIX * t)?;
        cx.gpu.enqueue_gemv_q3k(&params.w_dev, &act, &mut y_dev)?;
        stream.synchronize()?;
        let host_act = readback_q8act(stream, &act)?;
        let q8_bit = q8act_matches(&host_act.q3, &host_act.d8, &ch, k, t);
        let y = y_dev.to_host_vec(stream)?;
        let unsplit: Vec<f32> = (0..t * HC_MIX)
            .map(|i| y[(i % HC_MIX) * t + i / HC_MIX])
            .collect();
        let unsplit_rel = max_rel_err(&unsplit, &ch.raw)?;

        // 3. The chain vs the dump, against the prediction.
        let dump = DumpIn {
            x: &streams,
            xn: &xn,
            k,
            rms_eps: hp.rms_eps,
        };
        let mp = predict_mix(&params, &ch, &dump);
        let mix_ratio = gap_ratio(&mix_k, &mix_d, &mp.pred, &mp.bound);
        let (hc_pred, hc_bound) = predict_hc(&mix_d, &mix_k, &mp, &params, hp);
        let hc_ratio = gap_ratio(&hc_k, &hc_d, &hc_pred, &hc_bound);
        let mix_rel = rel_to(
            mix_k.iter().zip(&mix_d).map(|(a, b)| f64::from(a - b)),
            &mix_d,
        );
        let rule_rel = rel_to(mp.pred.iter().copied(), &mix_d);
        let logit_err = mix_k
            .iter()
            .zip(&mix_d)
            .enumerate()
            .filter(|(i, _)| i % HC_MIX >= 8)
            .fold(0.0f32, |m, (_, (a, b))| {
                m.max((a - b).abs() * params.scale[2].abs())
            });

        let pass_pre = rms_bit
            && pre_ik_bit
            && mix_bit
            && hc_bit
            && rerun
            && q8_bit
            && unsplit_rel <= KERNEL_BAND
            && mix_ratio <= 1.0
            && hc_ratio <= 1.0;
        println!(
            "hc_pre set={set} L={} sub={} T={t} ik_rms_bit={rms_bit} ik_hc_pre_bit={pre_ik_bit} \
             kernel_mix_bit={mix_bit} kernel_hc_bit={hc_bit} rerun_bit={rerun} \
             q8act_bit={q8_bit} unsplit_rel={unsplit_rel:.3e} \
             mix_rel={mix_rel:.3e} rule_rel={rule_rel:.3e} mix_gap_ratio={mix_ratio:.3e} \
             logit_err={logit_err:.3e} hc_gap_ratio={hc_ratio:.3e} {}",
            site.layer,
            site.sub(),
            verdict(pass_pre)
        );
        cx.ok &= pass_pre;

        // HC_POST and the fold: our rule vs the dump, the kernel vs both.
        let x = load(site.x)?;
        let post_d = load(site.post)?;
        let (out_h, fold_h) = post_host(&x, &streams, &hc_d, n, t);
        let post_ik_bit = bits_equal(&out_h, &post_d);
        let fold_d = site.fold.map(load).transpose()?;
        let fold_ik_bit = fold_d.as_ref().is_none_or(|f| bits_equal(&fold_h, f));
        let x_dev2 = DeviceBuffer::from_host(stream, &x)?;
        let hc_d_dev = DeviceBuffer::from_host(stream, &hc_d)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, 4 * n * t)?;
        let mut fold_dev = DeviceBuffer::<f32>::zeroed(stream, n * t)?;
        let pa = HcPostArgs {
            x: &x_dev2,
            res: &x_dev,
            hc: &hc_d_dev,
            n_embd: n,
            tokens: t,
        };
        cx.hck
            .enqueue_post(stream, &pa, &mut out_dev, &mut fold_dev)?;
        stream.synchronize()?;
        let (out_k, fold_k) = (out_dev.to_host_vec(stream)?, fold_dev.to_host_vec(stream)?);
        let post_bit = bits_equal(&out_k, &out_h) && bits_equal(&fold_k, &fold_h);
        let post_dump_bit = bits_equal(&out_k, &post_d);
        let fold_dump_bit = fold_d.as_ref().is_none_or(|f| bits_equal(&fold_k, f));

        let form = if site.fold.is_some() {
            "fused"
        } else {
            "post-alone"
        };
        let fold_seen = site.fold.map_or("-", |f| f.name.as_str());
        let pass_post = post_ik_bit && fold_ik_bit && post_bit && post_dump_bit && fold_dump_bit;
        println!(
            "hc_post set={set} L={} sub={} T={t} form={form} fold_node={fold_seen} \
             ik_post_bit={post_ik_bit} ik_fold_bit={fold_ik_bit} kernel_bit={post_bit} \
             post_vs_dump_bit={post_dump_bit} fold_vs_dump_bit={fold_dump_bit} {}",
            site.layer,
            site.sub(),
            verdict(pass_post)
        );
        cx.ok &= pass_post;
        Ok(SiteOut {
            fold: fold_k,
            hc: hc_d,
        })
    }

    /// One `ds41_hc_fold` launch against our rule and the dump; prints its
    /// line.
    fn check_fold(
        cx: &Ctx<'_>,
        set: &str,
        what: &str,
        s: &[f32],
        hc: &[f32],
        want: &[f32],
    ) -> Result<bool, GateError> {
        let (n, stream) = (cx.hp.n_embd, cx.gpu.stream());
        let t = want.len() / n;
        let host = fold_host(s, hc, n, t);
        let s_dev = DeviceBuffer::from_host(stream, s)?;
        let hc_dev = DeviceBuffer::from_host(stream, hc)?;
        let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n * t)?;
        cx.hck
            .enqueue_fold(stream, &s_dev, &hc_dev, n, t, &mut y_dev)?;
        stream.synchronize()?;
        let y = y_dev.to_host_vec(stream)?;
        let ik_bit = bits_equal(&host, want);
        let k_bit = bits_equal(&y, &host);
        let dump_bit = bits_equal(&y, want);
        let pass = ik_bit && k_bit && dump_bit;
        println!(
            "hc_fold set={set} site={what} T={t} ik_fold_bit={ik_bit} kernel_bit={k_bit} \
             kernel_vs_dump_bit={dump_bit} {}",
            verdict(pass)
        );
        Ok(pass)
    }

    /// The folds no HC_POST produces: the first layer's input (the one-hot
    /// `hc_pre_init`), each engram layer's (its streams are rewritten after
    /// the previous HC_POST), and the head's `hc_out` (the last token only).
    /// `ffn_hc` holds each layer's FFN HC_PRE output, `last_fold` the fold
    /// the last HC_POST launch wrote.
    fn check_lone_folds(
        cx: &mut Ctx<'_>,
        man: &RefManifest,
        set: &str,
        t: usize,
        ffn_hc: &[Vec<f32>],
        last_fold: &[f32],
    ) -> Result<(), GateError> {
        let n = cx.hp.n_embd;
        let load = |r: &RefRow| ref_tensor_of_in(&man.dir, r);
        let mut ok = true;
        for layer in 0..cx.hp.n_layers {
            let name = format!("hc_attn_pre-{layer}");
            let r = man.tensor(&name, 0)?;
            let s_name = src0(r);
            if !s_name.starts_with("engram_out-") && s_name != "hc_init" {
                continue;
            }
            let s = load(man.tensor(s_name, 0)?)?;
            let hc = if layer == 0 {
                let init = load(man.tensor("hc_pre_init", 0)?)?;
                let mut hc = vec![0.0f32; HC_MIX * t];
                for (h, pre) in hc
                    .as_chunks_mut::<HC_MIX>()
                    .0
                    .iter_mut()
                    .zip(init.as_chunks::<4>().0)
                {
                    h[..4].copy_from_slice(pre);
                }
                hc
            } else {
                ffn_hc[layer - 1].clone()
            };
            ok &= check_fold(cx, set, &format!("{name}<-{s_name}"), &s, &hc, &load(r)?)?;
        }
        let head = man.tensor("hc_out", 0)?;
        head.expect("hc_out", "f32", [n as u64, 1, 1, 1], "MUL_MULTI_ADD")?;
        let want = load(head)?;
        let last = cx.hp.n_layers - 1;
        let l_out = load(man.tensor(&format!("l_out-{last}"), 0)?)?;
        let hc_last = &ffn_hc[last][(t - 1) * HC_MIX..t * HC_MIX];
        let last_streams = &l_out[4 * n * (t - 1)..4 * n * t];
        ok &= check_fold(cx, set, "hc_out<-last_token", last_streams, hc_last, &want)?;
        let chained = bits_equal(&last_fold[n * (t - 1)..n * t], &want);
        println!(
            "hc_fold set={set} site=hc_out<-ds41_hc_post_of_the_last_ffn T=1 \
             kernel_vs_dump_bit={chained} {}",
            verdict(chained)
        );
        cx.ok &= ok && chained;
        Ok(())
    }

    /// One set: every sub-layer, then the lone folds.
    fn check_set(cx: &mut Ctx<'_>, man: &RefManifest, set: &str) -> Result<(), GateError> {
        let t = tokens_of(man)?;
        let all = sites(man, cx.hp, t as u64)?;
        println!(
            "set {set} dir={} build={} T={t} sites={}",
            man.dir.display(),
            man.build.as_deref().unwrap_or("-"),
            all.len()
        );
        // This module's launches one token of this graph takes: each
        // boundary fused except the ones before an engram layer and after
        // the last layer, whose HC_POST alone is the MoE piece's and whose
        // fold is a lone fold (the head's, after the last layer). Layer 0's
        // input fold (by the one-hot init, a copy of stream 0) is counted
        // apart.
        let lone = all.iter().filter(|s| s.fold.is_none()).count();
        let fused = all.len() - lone;
        println!(
            "nodes set={set} ds41_hc_pre={} ds41_hc_post={fused} ds41_hc_fold={lone} total={} \
             (+1 layer-0 fold)",
            all.len(),
            all.len() + fused + lone
        );
        let mut ffn_hc = vec![Vec::new(); cx.hp.n_layers];
        let mut last_fold = Vec::new();
        for site in &all {
            let out = check_site(cx, man, set, site, t)?;
            if site.ffn {
                ffn_hc[site.layer] = out.hc;
                last_fold = out.fold;
            }
        }
        check_lone_folds(cx, man, set, t, &ffn_hc, &last_fold)
    }

    /// The outputs a captured step writes.
    struct StepBufs {
        mixes: DeviceBuffer<f32>,
        hc: DeviceBuffer<f32>,
        out: DeviceBuffer<f32>,
        fold: DeviceBuffer<f32>,
    }

    impl StepBufs {
        fn read(&self, stream: &CudaStream) -> Result<[Vec<f32>; 4], GateError> {
            Ok([
                self.mixes.to_host_vec(stream)?,
                self.hc.to_host_vec(stream)?,
                self.out.to_host_vec(stream)?,
                self.fold.to_host_vec(stream)?,
            ])
        }

        fn clear(&mut self, stream: &CudaStream) -> Result<(), GateError> {
            for b in [&mut self.mixes, &mut self.hc, &mut self.out, &mut self.fold] {
                let zeros = vec![0.0f32; b.len()];
                b.copy_from_host(stream, &zeros)?;
            }
            Ok(())
        }
    }

    /// One captured graph of the chain and HC_POST, replayed twice with its
    /// outputs cleared before each, is bit-identical to the eager launches:
    /// the ticket counter is back at zero after every launch.
    fn check_graph(cx: &mut Ctx<'_>, man: &RefManifest) -> Result<(), GateError> {
        let (hp, stream) = (cx.hp, cx.gpu.stream());
        let (n, t) = (hp.n_embd, tokens_of(man)?);
        let all = sites(man, hp, t as u64)?;
        let site = all.get(4).ok_or("fewer than five sites")?;
        let params = site_params(cx.split, stream, site.layer, site.sub(), hp.k())?;
        let load = |r: &RefRow| ref_tensor_of_in(&man.dir, r);
        let x_dev = DeviceBuffer::from_host(stream, &load(site.streams)?)?;
        let sub_out = DeviceBuffer::from_host(stream, &load(site.x)?)?;
        let p = HcParams {
            w: &params.w_dev,
            scale: &params.scale_dev,
            base: &params.base_dev,
            eps: hp.eps,
            iters: hp.iters,
        };
        let a = HcPreArgs {
            params: &p,
            x: &x_dev,
            tokens: t,
            rms_eps: hp.rms_eps,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut bufs = StepBufs {
            mixes: DeviceBuffer::zeroed(stream, HC_MIX * t)?,
            hc: DeviceBuffer::zeroed(stream, HC_MIX * t)?,
            out: DeviceBuffer::zeroed(stream, 4 * n * t)?,
            fold: DeviceBuffer::zeroed(stream, n * t)?,
        };
        let hck = cx.hck;
        let step = |stream: &CudaStream,
                    scratch: &mut HcPreScratch,
                    b: &mut StepBufs|
         -> Result<(), GpuError> {
            hck.enqueue_pre(stream, &a, scratch, &mut b.mixes, &mut b.hc)?;
            let pa = HcPostArgs {
                x: &sub_out,
                res: &x_dev,
                hc: &b.hc,
                n_embd: n,
                tokens: t,
            };
            hck.enqueue_post(stream, &pa, &mut b.out, &mut b.fold)
        };
        step(stream, &mut cx.scratch, &mut bufs)?;
        stream.synchronize()?;
        let eager = bufs.read(stream)?;
        let g = cx.gpu.capture(|s| step(s, &mut cx.scratch, &mut bufs))?;
        let mut replays_bit = true;
        for _ in 0..2 {
            bufs.clear(stream)?;
            g.launch(stream)?;
            stream.synchronize()?;
            let got = bufs.read(stream)?;
            replays_bit &= got.iter().zip(&eager).all(|(a, b)| bits_equal(a, b));
        }
        let nodes = g.node_count();
        let pass = replays_bit && nodes == 2;
        println!(
            "graph site=L{}{} T={t} nodes={nodes} replay_x2_bit_identical_to_eager={replays_bit} {}",
            site.layer,
            site.sub(),
            verdict(pass)
        );
        cx.ok &= pass;
        Ok(())
    }

    /// A site's streams with one value NaN through `ds41_hc_pre`: the word,
    /// read back and cleared, must name [`FaultSite::HcQuant`] with no
    /// layer; the clean streams before it must leave the word clean.
    fn check_nan(cx: &mut Ctx<'_>, man: &RefManifest) -> Result<(), GateError> {
        let (hp, stream) = (cx.hp, cx.gpu.stream());
        let t = tokens_of(man)?;
        let all = sites(man, hp, t as u64)?;
        let site = all.get(4).ok_or("fewer than five sites")?;
        let params = site_params(cx.split, stream, site.layer, site.sub(), hp.k())?;
        let clean = cx.gpu.take_fault()?;
        let mut streams = ref_tensor_of_in(&man.dir, site.streams)?;
        streams[777] = f32::NAN;
        let x_dev = DeviceBuffer::from_host(stream, &streams)?;
        let p = HcParams {
            w: &params.w_dev,
            scale: &params.scale_dev,
            base: &params.base_dev,
            eps: hp.eps,
            iters: hp.iters,
        };
        let a = HcPreArgs {
            params: &p,
            x: &x_dev,
            tokens: t,
            rms_eps: hp.rms_eps,
            fault: cx.gpu.unlabelled_sink(),
        };
        let mut mixes = DeviceBuffer::<f32>::zeroed(stream, HC_MIX * t)?;
        let mut hc = DeviceBuffer::<f32>::zeroed(stream, HC_MIX * t)?;
        cx.hck
            .enqueue_pre(stream, &a, &mut cx.scratch, &mut mixes, &mut hc)?;
        let got = cx.gpu.take_fault()?;
        let want = Fault::at(LAYER_NONE, FaultSite::HcQuant);
        let pass = clean.is_none() && got == Some(want);
        println!(
            "fault site=L{}{} T={t} value 777 NaN: before={} want=hc_quant got={} {}",
            site.layer,
            site.sub(),
            clean.map_or("none".to_string(), |f| f.to_string()),
            got.map_or("none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        cx.ok &= pass;
        Ok(())
    }

    pub fn run() -> Result<(), GateError> {
        let split = Split::open(ref_model_path()?)?;
        let hp = Hp::read(&split)?;
        println!(
            "model n_embd={} layers={} K={} iters={} eps={:e} rms_eps={:e}",
            hp.n_embd,
            hp.n_layers,
            hp.k(),
            hp.iters,
            hp.eps,
            hp.rms_eps
        );
        let gpu = Gpu::new()?;
        let hck = HcKernels::load(gpu.context())?;
        let scratch = HcPreScratch::new(gpu.stream(), hp.k())?;
        let mut cx = Ctx {
            split: &split,
            hp: &hp,
            gpu: &gpu,
            hck: &hck,
            scratch,
            ok: true,
        };
        let table = oracle::for_arch(Arch::Deepseek41)?;
        let prefill = table.open(Set::Cpu)?;
        check_set(&mut cx, &prefill, "prefill")?;
        check_graph(&mut cx, &prefill)?;
        check_nan(&mut cx, &prefill)?;
        for name in STEP_SETS {
            if !ref_dir_named(name).join("MANIFEST.tsv").is_file() {
                println!("hole set={name} not on the box: its T = 1 checks did not run");
                continue;
            }
            let man = table.open_named(name)?;
            check_set(&mut cx, &man, name)?;
        }
        if cx.ok {
            println!(
                "PASSED: gate_deepseek41_hc — chain, HC_PRE, HC_POST and folds bit-identical to \
                 our rule; HC_POST and folds bit-identical to ik's dump; the chain inside its \
                 predicted gap"
            );
            Ok(())
        } else {
            Err(checks_failed())
        }
    }
}
