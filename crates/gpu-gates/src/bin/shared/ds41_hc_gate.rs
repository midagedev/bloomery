//! GPU gate for the DeepSeek-V4.1 hyper-connections
//! (`bloomery_gpu_deepseek41::hc`): the chain RMS + split-K q3_K gemv +
//! HC_PRE, HC_POST with the next input fold, and the fold alone, on every
//! sub-layer's own q3_K hyper-connection weights from the model file over
//! seeded streams (`hc_host::lcg`) of 5 tokens and of 1 token.
//!
//! PIN(2026-09-24): removed — the engine never runs `ds41_hc_pre_mix` (HC_PRE alone) or `ds41_hc_post_streams` (HC_POST alone); the kernels are gone with their checks.
//!
//! The kernel against the host transcription of our rule, the `gate_p4`
//! form: the chain, HC_PRE, HC_POST and the fold alone are bit-identical to
//! it, and a rerun is bit-identical. The rule is the module doc of `hc.rs`;
//! the transcription walks the kernel's lanes, butterflies and piece order.
//! HC_PRE's `exp` is f64 rounded to f32 on both sides.
//!
//! Cross-checks with the crate's own kernels: the streams are quantized by
//! `bloomery_gpu`'s `q3k_quantize_q8_1` into a `Q8Act` at K = 20480, whose
//! q3 words and block scales must equal this transcription's, and its
//! unsplit `q3k_gemv` must agree with the split-K sum within `KERNEL_BAND`
//! (a different sum order).
//!
//! The fault word (`bloomery_gpu::fault`): the chain's in-register q8_1 of
//! the streams has no form for a non-finite value, so a planted NaN in one
//! stream value must raise `FaultSite::HcQuant` rather than round to code 0,
//! with `FaultSite::F32Product` beside it (the NaN's squares leave the token
//! no scale). A finite stream value of `1e20`, whose square passes f32, raises
//! `F32Product` alone, through `ds41_hc_pre` and `ds41_hc_pre_groups`, and the
//! token's outputs are NaN, never zeros.
//!
//! The Q8_0 chain (`hc_pre_q8_0`, GLM-5.3-Flash's `hc_{attn,ffn}_fn`), model
//! free: seeded Q8_0 weights and streams (`hc_host::q8_0_fixture`, `lcg`) at
//! K = 512 (one piece) and 16384 (the GLM width), token groups of 1, 8,
//! 5 + 8 and 4 + 8 + 8, at the GLM header's epsilon, iterations and RMS
//! epsilon. The mixes are bit-identical to the host transcription
//! (`hc_host::hc_chain_q8_0`), the HC_PRE rows to `hc_pre_f32` over them, a
//! rerun into NaN-filled outputs to the first launch (the tickets are back
//! at 0), a captured graph of one node replayed twice to the eager launch,
//! and a NaN, or a finite `1e20`, in one token's streams makes all 48 of that
//! token's outputs NaN, raises `F32Product`, and leaves every other token's
//! bits. It runs before the model file opens.
//!
//! Hole, named: `ds41_hc_pre` takes at most 8 tokens a launch, so a longer
//! prefill is split by its caller and nothing here covers that split.

use bloomery_gpu::fused::readback_q8act;
use bloomery_gpu::hc::{
    HC_MAX_TOKENS, HC_MIX, HC_PIECE, HC_STREAMS, HcKernels, HcParams, HcPostArgs, HcPreArgs,
    HcPreScratch, HcQ8Params, HcQ8PreArgs,
};
use bloomery_gpu::{
    DeviceTensor, Fault, FaultSite, Gpu, GpuError, LAYER_NONE, Q8Act, col_group_count,
};
use bloomery_gpu_gates::act_rule;
use bloomery_gpu_gates::hc_host::{self, exp_ours, hc_chain_q8_0, hc_pre_f32, lcg, q8_0_fixture};
use bloomery_gpu_gates::rounding::butterfly;
use bloomery_gpu_gates::{
    GateError, KERNEL_BAND, bits_equal, bytes_to_words, checks_failed, max_rel_err, ref_model_path,
    tensor_bytes_as, verdict,
};
use cuda_core::{CudaStream, DeviceBuffer};
use gguf::Split;
use gguf::quant::{GgmlType, dequant_row, half_to_f32};

/// The token counts every sub-layer runs at: a prefill batch and a decode
/// step.
const SWEEP_TOKENS: [usize; 2] = [5, 1];
/// The sub-layer the graph and fault clauses run on, layer 2's attention,
/// and their token count.
const PROBE_LAYER: usize = 2;
const PROBE_TOKENS: usize = 5;
/// Bytes of one q3_K super-block.
const Q3K_SB: usize = 110;
// The host rule's layout is the kernels'.
const _: () = assert!(
    hc_host::HC_STREAMS == HC_STREAMS && hc_host::HC_MIX == HC_MIX && hc_host::HC_PIECE == HC_PIECE
);

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
            return Err(format!("hyper_connection.count = {streams}, the kernels take 4").into());
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

// ----------------------------------------------------------- the host rule

/// HC_POST of one value: `fma(x, post_i, comb_0i r_0)`,
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
    let (q8, d8) = act_rule::q8_1_codes(&x[..k * t]);
    let q8: Vec<i32> = q8.into_iter().map(i32::from).collect();
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
    64 * (v4 >> 7) + 32 * ((v4 >> 4) & 1) + 16 * ((v4 >> 6) & 1) + 8 * ((v4 >> 5) & 1) + (v4 & 7)
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

// ------------------------------------------------------------ the sites

/// One sub-layer's parameters from the file, host and device.
struct SiteParams {
    scale: [f32; 3],
    base: Vec<f32>,
    sbs: Vec<Q3Sb>,
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
        }
    }
    Ok(SiteParams {
        scale: [scale_v[0], scale_v[1], scale_v[2]],
        w_dev: DeviceTensor::upload(stream, &bytes_to_words(wb), HC_MIX, row_bytes / 4)?,
        scale_dev: DeviceBuffer::from_host(stream, &scale_v)?,
        base_dev: DeviceBuffer::from_host(stream, &base)?,
        base,
        sbs,
    })
}

fn sub_name(ffn: bool) -> &'static str {
    if ffn { "ffn" } else { "attn" }
}

/// The seed of a site's streams at `t` tokens: its sub-layer's index and
/// `t` apart, so no two sites share an input.
fn site_seed(layer: usize, ffn: bool, t: usize) -> Result<u32, GateError> {
    Ok(1000 + 100 * u32::try_from(layer)? + 10 * u32::from(ffn) + u32::try_from(t)?)
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

/// Every check of one site at `t` tokens of seeded streams; prints its
/// lines.
fn check_site(
    cx: &mut Ctx<'_>,
    params: &SiteParams,
    layer: usize,
    ffn: bool,
    t: usize,
) -> Result<(), GateError> {
    let (hp, stream) = (cx.hp, cx.gpu.stream());
    let (n, k) = (hp.n_embd, hp.k());
    let seed = site_seed(layer, ffn, t)?;
    let streams = lcg(k * t, seed);
    let x = lcg(n * t, seed + 3);
    let pre_of = |m: &[f32]| -> Vec<f32> {
        m.as_chunks::<HC_MIX>()
            .0
            .iter()
            .flat_map(|mt| hc_pre_f32(mt, params.scale, &params.base, hp.eps, hp.iters, exp_ours))
            .collect()
    };

    // The chain kernel vs our rule, and a rerun.
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
    let hc_h = pre_of(&ch.mixes);
    let mix_bit = bits_equal(&mix_k, &ch.mixes);
    let hc_bit = bits_equal(&hc_k, &hc_h);

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

    // PIN(2026-11-18): the real-file ik set retired; ik cannot read bq3km's engram schema; no fixture set carries this clause
    let pass_pre = mix_bit && hc_bit && rerun && q8_bit && unsplit_rel <= KERNEL_BAND;
    println!(
        "hc_pre L={layer} sub={} T={t} kernel_mix_bit={mix_bit} kernel_hc_bit={hc_bit} \
             rerun_bit={rerun} q8act_bit={q8_bit} unsplit_rel={unsplit_rel:.3e} {}",
        sub_name(ffn),
        verdict(pass_pre)
    );
    cx.ok &= pass_pre;

    // HC_POST and the fold: the kernel against our rule on the host's HC_PRE
    // row.
    let (out_h, fold_h) = post_host(&x, &streams, &hc_h, n, t);
    let x_dev2 = DeviceBuffer::from_host(stream, &x)?;
    let hc_h_dev = DeviceBuffer::from_host(stream, &hc_h)?;
    let mut out_dev = DeviceBuffer::<f32>::zeroed(stream, 4 * n * t)?;
    let mut fold_dev = DeviceBuffer::<f32>::zeroed(stream, n * t)?;
    let pa = HcPostArgs {
        x: &x_dev2,
        res: &x_dev,
        hc: &hc_h_dev,
        n_embd: n,
        tokens: t,
    };
    cx.hck
        .enqueue_post(stream, &pa, &mut out_dev, &mut fold_dev)?;
    stream.synchronize()?;
    let (out_k, fold_k) = (out_dev.to_host_vec(stream)?, fold_dev.to_host_vec(stream)?);
    let pass_post = bits_equal(&out_k, &out_h) && bits_equal(&fold_k, &fold_h);
    println!(
        "hc_post L={layer} sub={} T={t} kernel_bit={pass_post} {}",
        sub_name(ffn),
        verdict(pass_post)
    );
    cx.ok &= pass_post;

    // The fold alone, over the streams the post left.
    let fold_ok = check_fold(cx, &format!("L{layer}{}", sub_name(ffn)), &out_h, &hc_h)?;
    cx.ok &= fold_ok;
    Ok(())
}

/// One `ds41_hc_fold` launch against our rule; prints its line.
fn check_fold(cx: &Ctx<'_>, what: &str, s: &[f32], hc: &[f32]) -> Result<bool, GateError> {
    let (n, stream) = (cx.hp.n_embd, cx.gpu.stream());
    let t = hc.len() / HC_MIX;
    let host = fold_host(s, hc, n, t);
    let s_dev = DeviceBuffer::from_host(stream, s)?;
    let hc_dev = DeviceBuffer::from_host(stream, hc)?;
    let mut y_dev = DeviceBuffer::<f32>::zeroed(stream, n * t)?;
    cx.hck
        .enqueue_fold(stream, &s_dev, &hc_dev, n, t, &mut y_dev)?;
    stream.synchronize()?;
    let k_bit = bits_equal(&y_dev.to_host_vec(stream)?, &host);
    println!(
        "hc_fold site={what} T={t} kernel_bit={k_bit} {}",
        verdict(k_bit)
    );
    Ok(k_bit)
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

/// The probe sub-layer's parameters and its seeded streams (`[t][4][n]`).
fn probe(cx: &Ctx<'_>) -> Result<(SiteParams, Vec<f32>), GateError> {
    let params = site_params(
        cx.split,
        cx.gpu.stream(),
        PROBE_LAYER,
        sub_name(false),
        cx.hp.k(),
    )?;
    let seed = site_seed(PROBE_LAYER, false, PROBE_TOKENS)?;
    Ok((params, lcg(cx.hp.k() * PROBE_TOKENS, seed)))
}

/// One captured graph of the chain and HC_POST, replayed twice with its
/// outputs cleared before each, is bit-identical to the eager launches:
/// the ticket counter is back at zero after every launch.
fn check_graph(cx: &mut Ctx<'_>) -> Result<(), GateError> {
    let (hp, stream) = (cx.hp, cx.gpu.stream());
    let (n, t) = (hp.n_embd, PROBE_TOKENS);
    let (params, streams) = probe(cx)?;
    let x_dev = DeviceBuffer::from_host(stream, &streams)?;
    let sub_out = DeviceBuffer::from_host(stream, &lcg(n * t, 7))?;
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
        "graph site=L{PROBE_LAYER}attn T={t} nodes={nodes} replay_x2_bit_identical_to_eager={replays_bit} {}",
        verdict(pass)
    );
    cx.ok &= pass;
    Ok(())
}

/// The probe site's streams with one value NaN through `ds41_hc_pre`: the
/// word, read back and cleared, must name [`FaultSite::HcQuant`] and
/// [`FaultSite::F32Product`] with no layer; the clean streams before it
/// must leave the word clean.
///
/// PIN(2026-10-18): the mask gained `F32Product`. The planted NaN squares
/// to a NaN sum, so the token's scale has no norm behind it
/// (`elem::rms_scale_checked`), as it does for `rms_norm`'s NaN row; the
/// quantizer's `HcQuant` is still the smallest code, so the word is the
/// old one with a second bit in the mask. The clause fails on a kernel
/// that scales by `1/sqrt` unchecked, whose mask is `HcQuant` alone.
fn check_nan(cx: &mut Ctx<'_>) -> Result<(), GateError> {
    let (hp, stream) = (cx.hp, cx.gpu.stream());
    let t = PROBE_TOKENS;
    let (params, mut streams) = probe(cx)?;
    let clean = cx.gpu.take_fault()?;
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
    let want = Fault::of_sites(LAYER_NONE, &[FaultSite::HcQuant, FaultSite::F32Product]);
    let pass = clean.is_none() && got == Some(want);
    println!(
        "fault site=L{PROBE_LAYER}attn T={t} value 777 NaN: before={} want=hc_quant+f32_product got={} {}",
        clean.map_or("none".to_string(), |f| f.to_string()),
        got.map_or("none".to_string(), |f| f.to_string()),
        verdict(pass)
    );
    cx.ok &= pass;
    Ok(())
}

/// The probe site's streams with one value `1e20` through `ds41_hc_pre` and
/// through `ds41_hc_pre_groups`: the value is finite, so the quantizer
/// raises nothing, but its square passes f32 and the token's scale has no
/// norm behind it. The word must name [`FaultSite::F32Product`] alone,
/// with no layer; that token's 48 outputs are NaN, never the zeros `1/∞`
/// scales them to, and every other token's bits are the clean run's.
fn check_overflow(cx: &mut Ctx<'_>) -> Result<(), GateError> {
    const BIG: f32 = 1.0e20;
    let (hp, stream) = (cx.hp, cx.gpu.stream());
    let t = PROBE_TOKENS;
    let (params, streams) = probe(cx)?;
    let mut big = streams.clone();
    big[777] = BIG;
    let k = hp.k();
    let lead = t.min(HC_MAX_TOKENS);
    let mut groups_scratch = HcPreScratch::with_groups(stream, k, col_group_count(lead, t))?;
    let want = Some(Fault::at(LAYER_NONE, FaultSite::F32Product));
    for grouped in [false, true] {
        let mut run = |x: &[f32]| -> Result<(Vec<f32>, Vec<f32>), GateError> {
            let x_dev = DeviceBuffer::from_host(stream, x)?;
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
            if grouped {
                cx.hck.enqueue_pre_groups(
                    stream,
                    &a,
                    lead,
                    &mut groups_scratch,
                    &mut mixes,
                    &mut hc,
                )?;
            } else {
                cx.hck
                    .enqueue_pre(stream, &a, &mut cx.scratch, &mut mixes, &mut hc)?;
            }
            stream.synchronize()?;
            Ok((mixes.to_host_vec(stream)?, hc.to_host_vec(stream)?))
        };
        let before = cx.gpu.take_fault()?;
        let (mix_c, hc_c) = run(&streams)?;
        let clean_word = cx.gpu.take_fault()?;
        let (mix_b, hc_b) = run(&big)?;
        let got = cx.gpu.take_fault()?;
        let row = |v: &[f32], tt: usize| v[tt * HC_MIX..(tt + 1) * HC_MIX].to_vec();
        let bad_nan = row(&mix_b, 0)
            .iter()
            .chain(&row(&hc_b, 0))
            .all(|v| v.is_nan());
        let others = (1..t).all(|tt| {
            bits_equal(&row(&mix_b, tt), &row(&mix_c, tt))
                && bits_equal(&row(&hc_b, tt), &row(&hc_c, tt))
        });
        let pass = before.is_none() && clean_word.is_none() && got == want && bad_nan && others;
        println!(
            "overflow {} site=L{PROBE_LAYER}attn T={t} value 777 = {BIG:e}: word={} want=f32_product \
                 token_0_outputs_nan={bad_nan} other_tokens_bit={others} {}",
            if grouped { "groups" } else { "single" },
            got.map_or("none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        cx.ok &= pass;
    }
    Ok(())
}

// ------------------------------------------------------ the Q8_0 chain

/// The GLM-5.3-Flash header's values the Q8_0 clause runs at:
/// `hyper_connection.epsilon`, `.sinkhorn_iterations` and
/// `attention.layer_norm_rms_epsilon`.
const Q8_EPS: f32 = 1e-6;
const Q8_ITERS: u32 = 20;
const Q8_RMS_EPS: f32 = 1e-5;
/// The GLM width: four streams of 4096.
const Q8_K: usize = 16_384;
/// `(K, tokens, lead)` of the Q8_0 clause.
const Q8_CASES: [(usize, usize, usize); 7] = [
    (HC_PIECE, 1, 1),
    (HC_PIECE, 8, 8),
    (HC_PIECE, 13, 5),
    (Q8_K, 1, 1),
    (Q8_K, 8, 8),
    (Q8_K, 13, 5),
    (Q8_K, 20, 4),
];

/// One seeded Q8_0 site on the device and its host copy.
struct Q8Site {
    qs: DeviceTensor<u32>,
    d: DeviceTensor<u16>,
    scale: DeviceBuffer<f32>,
    base: DeviceBuffer<f32>,
    qs_h: Vec<u32>,
    d_h: Vec<u16>,
    sc: [f32; 3],
    base_h: Vec<f32>,
}

impl Q8Site {
    fn new(stream: &CudaStream, k: usize, seed: u32) -> Result<Q8Site, GateError> {
        let (qs_h, d_h) = q8_0_fixture(k, seed);
        let affine = lcg(3 + HC_MIX, seed ^ 0x5a5a);
        let (sc, base_h) = ([affine[0], affine[1], affine[2]], affine[3..].to_vec());
        Ok(Q8Site {
            qs: DeviceTensor::upload(stream, &qs_h, HC_MIX, k / 4)?,
            d: DeviceTensor::upload(stream, &d_h, HC_MIX, k / 32)?,
            scale: DeviceBuffer::from_host(stream, &sc)?,
            base: DeviceBuffer::from_host(stream, &base_h)?,
            qs_h,
            d_h,
            sc,
            base_h,
        })
    }

    fn params(&self) -> HcQ8Params<'_> {
        HcQ8Params {
            qs: &self.qs,
            d: &self.d,
            scale: &self.scale,
            base: &self.base,
            eps: Q8_EPS,
            iters: Q8_ITERS,
        }
    }

    /// The host rule's mixes and HC_PRE rows for `t` tokens of `x`.
    fn host(&self, x: &[f32], k: usize, t: usize) -> (Vec<f32>, Vec<f32>) {
        let mixes = hc_chain_q8_0(&self.qs_h, &self.d_h, x, k, t, Q8_RMS_EPS);
        let hc = mixes
            .as_chunks::<HC_MIX>()
            .0
            .iter()
            .flat_map(|m| hc_pre_f32(m, self.sc, &self.base_h, Q8_EPS, Q8_ITERS, exp_ours))
            .collect();
        (mixes, hc)
    }
}

/// The outputs of a Q8_0 launch.
struct Q8Out {
    mixes: DeviceBuffer<f32>,
    hc: DeviceBuffer<f32>,
}

impl Q8Out {
    /// Both filled with NaN, so a slot no launch writes cannot read as
    /// a result.
    fn clear(&mut self, stream: &CudaStream) -> Result<(), GateError> {
        for b in [&mut self.mixes, &mut self.hc] {
            let nan = vec![f32::NAN; b.len()];
            b.copy_from_host(stream, &nan)?;
        }
        Ok(())
    }

    fn read(&self, stream: &CudaStream) -> Result<(Vec<f32>, Vec<f32>), GateError> {
        Ok((
            self.mixes.to_host_vec(stream)?,
            self.hc.to_host_vec(stream)?,
        ))
    }
}

/// One eager launch into cleared outputs, read back.
fn q8_launch(
    hck: &HcKernels,
    stream: &CudaStream,
    a: &HcQ8PreArgs<'_>,
    lead: usize,
    scratch: &mut HcPreScratch,
    out: &mut Q8Out,
) -> Result<(Vec<f32>, Vec<f32>), GateError> {
    out.clear(stream)?;
    hck.enqueue_pre_q8_0(stream, a, lead, scratch, &mut out.mixes, &mut out.hc)?;
    stream.synchronize()?;
    out.read(stream)
}

/// Every check of the Q8_0 chain (the module doc's Q8_0 paragraph);
/// prints one line per case, one for the graph and one for the NaN.
fn check_q8_0(gpu: &Gpu, hck: &HcKernels) -> Result<bool, GateError> {
    let stream = gpu.stream();
    let mut ok = true;
    for (i, &(k, t, lead)) in Q8_CASES.iter().enumerate() {
        let seed = 1000 + u32::try_from(i)?;
        let site = Q8Site::new(stream, k, seed)?;
        let xs = lcg(k * t, seed + 7);
        let x = DeviceBuffer::from_host(stream, &xs)?;
        let groups = col_group_count(lead, t);
        let mut scratch = HcPreScratch::with_groups(stream, k, groups)?;
        let p = site.params();
        let a = HcQ8PreArgs {
            params: &p,
            x: &x,
            tokens: t,
            rms_eps: Q8_RMS_EPS,
            fault: gpu.unlabelled_sink(),
        };
        let mut out = Q8Out {
            mixes: DeviceBuffer::zeroed(stream, HC_MIX * t)?,
            hc: DeviceBuffer::zeroed(stream, HC_MIX * t)?,
        };
        let (mix_k, hc_k) = q8_launch(hck, stream, &a, lead, &mut scratch, &mut out)?;
        let (mix_r, hc_r) = q8_launch(hck, stream, &a, lead, &mut scratch, &mut out)?;
        let (mix_h, hc_h) = site.host(&xs, k, t);
        let mix_bit = bits_equal(&mix_k, &mix_h);
        let hc_bit = bits_equal(&hc_k, &hc_h);
        let rerun = bits_equal(&mix_r, &mix_k) && bits_equal(&hc_r, &hc_k);
        let finite = mix_h.iter().chain(&hc_h).all(|v| v.is_finite());
        let pass = mix_bit && hc_bit && rerun && finite;
        println!(
            "hc_pre_q8_0 K={k} T={t} lead={lead} groups={groups} kernel_mix_bit={mix_bit} \
                 kernel_hc_bit={hc_bit} rerun_bit={rerun} host_finite={finite} {}",
            verdict(pass)
        );
        ok &= pass;
    }

    // The GLM width at 13 tokens (lead 5): a captured graph, and a NaN.
    let (k, t, lead) = (Q8_K, 13, 5);
    let site = Q8Site::new(stream, k, 2000)?;
    let xs = lcg(k * t, 2007);
    let x = DeviceBuffer::from_host(stream, &xs)?;
    let mut scratch = HcPreScratch::with_groups(stream, k, col_group_count(lead, t))?;
    let p = site.params();
    let a = HcQ8PreArgs {
        params: &p,
        x: &x,
        tokens: t,
        rms_eps: Q8_RMS_EPS,
        fault: gpu.unlabelled_sink(),
    };
    let mut out = Q8Out {
        mixes: DeviceBuffer::zeroed(stream, HC_MIX * t)?,
        hc: DeviceBuffer::zeroed(stream, HC_MIX * t)?,
    };
    let eager = q8_launch(hck, stream, &a, lead, &mut scratch, &mut out)?;
    let g = gpu.capture(|s| {
        hck.enqueue_pre_q8_0(s, &a, lead, &mut scratch, &mut out.mixes, &mut out.hc)
    })?;
    let mut replays_bit = true;
    for _ in 0..2 {
        out.clear(stream)?;
        g.launch(stream)?;
        stream.synchronize()?;
        let got = out.read(stream)?;
        replays_bit &= bits_equal(&got.0, &eager.0) && bits_equal(&got.1, &eager.1);
    }
    let nodes = g.node_count();
    let pass = replays_bit && nodes == 1;
    println!(
        "hc_pre_q8_0 graph K={k} T={t} lead={lead} nodes={nodes} \
             replay_x2_bit_identical_to_eager={replays_bit} {}",
        verdict(pass)
    );
    ok &= pass;

    // One value of token 6 NaN, and one `1e20` (finite, its square past
    // f32): either way the token's scale has no norm behind it, so its 48
    // outputs are NaN — never the zeros `1/∞` scales them to — and the word
    // names `F32Product` with no layer; every other token's bits are the
    // clean launch's and the word is clean before.
    let bad = 6usize;
    let want = Some(Fault::at(LAYER_NONE, FaultSite::F32Product));
    for (what, value) in [("nan", f32::NAN), ("overflow", 1.0e20f32)] {
        let before = gpu.take_fault()?;
        let mut xn = xs.clone();
        xn[bad * k + 777] = value;
        let x_bad = DeviceBuffer::from_host(stream, &xn)?;
        let a_bad = HcQ8PreArgs { x: &x_bad, ..a };
        let (mix_n, hc_n) = q8_launch(hck, stream, &a_bad, lead, &mut scratch, &mut out)?;
        let word = gpu.take_fault()?;
        let row = |v: &[f32], tt: usize| v[tt * HC_MIX..(tt + 1) * HC_MIX].to_vec();
        let bad_nan = row(&mix_n, bad)
            .iter()
            .chain(&row(&hc_n, bad))
            .all(|v| v.is_nan());
        let others_bit = (0..t).filter(|&tt| tt != bad).all(|tt| {
            bits_equal(&row(&mix_n, tt), &row(&eager.0, tt))
                && bits_equal(&row(&hc_n, tt), &row(&eager.1, tt))
        });
        let pass = before.is_none() && word == want && bad_nan && others_bit;
        println!(
            "hc_pre_q8_0 {what} K={k} T={t} token {bad} value 777 = {value:e}: \
                 word={} want=f32_product its_48_outputs_nan={bad_nan} \
                 other_tokens_bit={others_bit} {}",
            word.map_or("none".to_string(), |f| f.to_string()),
            verdict(pass)
        );
        ok &= pass;
    }
    Ok(ok)
}

pub fn run() -> Result<(), GateError> {
    let gpu = Gpu::new()?;
    let hck = HcKernels::load(gpu.context())?;
    let q8_ok = check_q8_0(&gpu, &hck)?;
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
    let scratch = HcPreScratch::new(gpu.stream(), hp.k())?;
    let mut cx = Ctx {
        split: &split,
        hp: &hp,
        gpu: &gpu,
        hck: &hck,
        scratch,
        ok: q8_ok,
    };
    for layer in 0..hp.n_layers {
        for ffn in [false, true] {
            let params = site_params(&split, gpu.stream(), layer, sub_name(ffn), hp.k())?;
            for t in SWEEP_TOKENS {
                check_site(&mut cx, &params, layer, ffn, t)?;
            }
        }
    }
    check_graph(&mut cx)?;
    check_nan(&mut cx)?;
    check_overflow(&mut cx)?;
    if cx.ok {
        println!(
            "PASSED: gate_deepseek41_hc — chain, HC_PRE, HC_POST and folds bit-identical to \
                 our rule on every sub-layer's weights; the fault words; the Q8_0 chain \
                 bit-identical to our rule"
        );
        Ok(())
    } else {
        Err(checks_failed())
    }
}
