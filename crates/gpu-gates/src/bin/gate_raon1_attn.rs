//! GPU gate for Raon-1's two attention layers, model-less at the model's
//! shapes (40 heads of 128, the latent 512 wide beside 64 unrotated key
//! values), each against an f64 reference written from the HF model
//! (`modeling_raon1_mini.py`: `torch_recurrent_rda` and `Raon1MiniRDA`,
//! `Raon1MiniMLA`), with an in-gate separation that a reference or a card
//! run missing the new clause fails. Inputs are seeded; nothing is read from
//! a model file.
//!
//! 1. `write_gate`: `kda_write_gate` against `write_gate_host` bit for bit at
//!    64 tokens and at 1. A NaN β row (token 3, head 7) raises `linear_conv`
//!    with the launch's layer and writes NaN on that head's 128 values of
//!    that token alone, every other value the clean run's bits.
//! 2. `rda`: the RDA step's launches as `mixer_kda` makes them —
//!    `kda_conv_prep` (bounded sigmoid decay, bound −5, β = σ(b)),
//!    `kda_write_gate`, `kda_delta` — over 64 tokens from position 0, as one
//!    launch each and as 64 one-token steps, against the HF layer in f64 from
//!    the raw projections (the causal conv and silu, the l2 norms and the
//!    query scale, `log α = −5·σ(e^A_log·(f + dt_bias))`, `β = σ(b)`, the
//!    values times `σ(b + w)/σ(b)`, then `torch_recurrent_rda`): the largest
//!    per-token relative L2 of the delta output within [`RDA_BAND`]. The card
//!    run without the write gate lies past [`SEP`] bands.
//! 3. `mla`: `latent_gated::attend` one row a step from position 0, over a
//!    cache of 64 rows (the single-launch pass) and one of 1024 (the segment
//!    pass), against the HF attention in f64 on the dequantized `attn_k_b`
//!    (Q8_0) and `attn_v_b` (Q6_K): each head's key the nope keys `k_b`
//!    makes of the normed latent then the 64 raw extra values every head
//!    shares, its value `v_b` of the latent, the score scale `1/√192`, and
//!    the heads times `σ(gate)`: the relative L2 of the gated heads within
//!    [`MLA_BAND`] at the checked positions. At each run's last position,
//!    three references each missing one clause (the extra part, the scale at
//!    `1/√128`, the gate) lie past [`SEP`] bands.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_raon1_attn", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::latent::LATENT;
    use bloomery_gpu::latent_gated::{
        AttendIn, AttnBufs, MlaDims, MlaKernels, ValueW, attend, partials_lens,
    };
    use bloomery_gpu::linear::conv::{KdaBetaScale, KdaConvArgs, KdaGateRule};
    use bloomery_gpu::linear::delta::DeltaArgs;
    use bloomery_gpu::linear::write_gate::{WriteGateArgs, write_gate_host};
    use bloomery_gpu::linear::{CONV_TAPS, HEAD, KHeadMap, LinearKernels, LinearShape};
    use bloomery_gpu::weights::q8_0_planes;
    use bloomery_gpu::{DeviceTensor, Fault, FaultSink, FaultSite, Gpu};
    use bloomery_gpu_gates::{GateError, bits_equal, checks_failed, verdict};
    use cuda_core::DeviceBuffer;
    use gguf::quant::{GgmlType, Q8Block, dequant_row, half_to_f32};

    /// Raon-1's RDA layers: 40 query/key and 40 value heads.
    const SHAPE: LinearShape = LinearShape {
        n_k: 40,
        n_v: 40,
        map: KHeadMap::Tiled,
    };
    /// `layer_norm_rms_epsilon`, the conv prep's l2 floor in the engine.
    const EPS: f32 = 1e-5;
    /// `kda.gate_lower_bound`.
    const LB: f32 = -5.0;
    /// The RDA clause's tokens.
    const T: usize = 64;
    /// The layer the fault clause labels its launch with.
    const LAYER: usize = 11;
    /// The f32 chain against f64: the write scale sits within 4e-6 relative
    /// (`write_scale_is_the_sigmoid_ratio`), the decay's `expf` within 2 ulps
    /// a step over 64 steps, each 128-term dot within 128 ulps; 1e-4 is ten
    /// times their sum.
    const RDA_BAND: f64 = 1e-4;
    /// The f16 cache (2^-11 a value) and the attention's f16 operands each
    /// move a score or a value by under 5e-4 relative; 4e-3 is eight times.
    const MLA_BAND: f64 = 4e-3;
    /// The bands a run missing a clause must lie past.
    const SEP: f64 = 10.0;

    /// The latent layer's widths.
    const MLA: MlaDims = MlaDims {
        embd: 2560,
        heads: 40,
        q_lora: 576,
        nope: 128,
        extra: 64,
        v_head: 128,
        rms_eps: 1e-5,
        latent_eps: 1e-6,
    };
    /// The two caches of the MLA clause and the positions each run checks.
    const MLA_RUNS: [(usize, usize, &[usize]); 2] =
        [(64, 64, &[0, 1, 63]), (1024, 200, &[0, 63, 64, 199])];

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
        fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
            lo + (hi - lo) * self.unit()
        }
        fn fill(&mut self, n: usize, lo: f32, hi: f32) -> Vec<f32> {
            (0..n).map(|_| self.uniform(lo, hi)).collect()
        }
    }

    /// `‖a − b‖₂ / ‖b‖₂` in f64.
    fn rel_l2(a: &[f32], b: &[f64]) -> f64 {
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (&x, &y) in a.iter().zip(b) {
            num += (f64::from(x) - y).powi(2);
            den += y * y;
        }
        if a.len() != b.len() || !num.is_finite() {
            return f64::INFINITY;
        }
        (num / den.max(f64::MIN_POSITIVE)).sqrt()
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `log σ(x)`, stable both ways.
    fn log_sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            -(-x).exp().ln_1p()
        } else {
            x - x.exp().ln_1p()
        }
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = LinearKernels::load(gpu.context())?;
        println!(
            "gate_raon1_attn: device {} — RDA {} heads of {HEAD}, MLA {} heads, row {} (extra {} + \
             latent {LATENT}), synthetic",
            gpu.device_name()?,
            SHAPE.n_v,
            MLA.heads,
            MLA.row(),
            MLA.extra
        );
        let mut results = write_gate_clauses(&gpu, &k)?;
        results.extend(rda_clauses(&gpu, &k)?);
        results.extend(mla_clauses(&gpu)?);
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_raon1_attn: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    // ------------------------------------------------------------ write gate

    /// One write-gate launch over `m` tokens of `y` in place.
    fn write_gate(
        gpu: &Gpu,
        k: &LinearKernels,
        fault: FaultSink,
        (b, w): (&[f32], &[f32]),
        y: &[f32],
        m: usize,
    ) -> Result<Vec<f32>, GateError> {
        let s = gpu.stream();
        let (bd, wd) = (
            DeviceBuffer::from_host(s, b)?,
            DeviceBuffer::from_host(s, w)?,
        );
        let mut yd = DeviceBuffer::from_host(s, y)?;
        k.write_gate.enqueue_write_gate(
            s,
            WriteGateArgs {
                b: &bd,
                w: &wd,
                shape: SHAPE,
                m,
                fault,
                y: &mut yd,
            },
        )?;
        s.synchronize()?;
        Ok(yd.to_host_vec(s)?)
    }

    fn write_gate_clauses(gpu: &Gpu, k: &LinearKernels) -> Result<Vec<bool>, GateError> {
        let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
        let mut out = Vec::new();
        let mut r = Lcg(0x7277_6774);
        let unl = gpu.unlabelled_sink();
        for m in [T, 1] {
            let y = r.fill(m * ch, -2.0, 2.0);
            let (b, w) = (r.fill(m * nv, -6.0, 6.0), r.fill(m * nv, -6.0, 6.0));
            let got = write_gate(gpu, k, unl, (&b, &w), &y, m)?;
            let mut want = y.clone();
            write_gate_host(&b, &w, SHAPE, m, &mut want);
            let pass = bits_equal(&got, &want) && gpu.fault()?.is_none();
            println!(
                "  write_gate m={m}: card vs host bit for bit: {}",
                verdict(pass)
            );
            out.push(pass);
        }
        let m = 16;
        let y = r.fill(m * ch, -2.0, 2.0);
        let (mut b, w) = (r.fill(m * nv, -6.0, 6.0), r.fill(m * nv, -6.0, 6.0));
        let clean = write_gate(gpu, k, unl, (&b, &w), &y, m)?;
        let before = gpu.fault()?;
        b[3 * nv + 7] = f32::NAN;
        let bad = write_gate(gpu, k, gpu.layer_sink(LAYER)?, (&b, &w), &y, m)?;
        let word = gpu.take_fault()?;
        let hit = |i: usize| i / ch == 3 && (i % ch) / HEAD == 2 * SHAPE.n_k + 7;
        let shape_ok = bad.iter().zip(&clean).enumerate().all(|(i, (x, c))| {
            if hit(i) {
                x.is_nan()
            } else {
                x.to_bits() == c.to_bits()
            }
        });
        let want = Fault::at(u32::try_from(LAYER)?, FaultSite::LinearConv);
        let pass = before.is_none() && word == Some(want) && shape_ok && gpu.fault()?.is_none();
        println!(
            "  write_gate NaN beta row (token 3, head 7): word {word:?} (want {want:?}), NaN on that \
             head alone {shape_ok}: {}",
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    // ------------------------------------------------------------------ RDA

    /// One RDA layer's parameters and `T` tokens of its raw projections.
    struct Rda {
        /// Conv taps `[channel][CONV_TAPS]`, oldest first.
        taps: Vec<f32>,
        dt: Vec<f32>,
        /// `−e^A_log` per head (the file's `ssm_a`).
        sa: Vec<f32>,
        /// `[T][channels]`, `[T][n_v]`, `[T][n_v]`, `[T][n_v·HEAD]`.
        x: Vec<f32>,
        b: Vec<f32>,
        w: Vec<f32>,
        f: Vec<f32>,
    }

    impl Rda {
        fn new(seed: u64) -> Rda {
            let mut r = Lcg(seed);
            let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
            let (lo, hi) = (0.02f32.ln(), 20.0f32.ln());
            Rda {
                taps: r.fill(CONV_TAPS * ch, -0.5, 0.5),
                dt: r.fill(nv * HEAD, -1.0, 1.0),
                sa: (0..nv).map(|_| -r.uniform(lo, hi).exp()).collect(),
                x: r.fill(T * ch, -1.0, 1.0),
                b: r.fill(T * nv, -3.0, 3.0),
                w: r.fill(T * nv, -3.0, 3.0),
                f: r.fill(T * nv * HEAD, -3.0, 3.0),
            }
        }
    }

    /// The HF layer in f64 from the raw projections to the delta output
    /// `[T][n_v·HEAD]`; `write` false drops the write ratio (the separation).
    fn rda_ref(l: &Rda, write: bool) -> Vec<f64> {
        let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
        let p = nv * HEAD;
        let mut state = vec![0.0f64; nv * HEAD * HEAD];
        let mut out = vec![0.0f64; T * p];
        for t in 0..T {
            let mut s = vec![0.0f64; ch];
            for (c, sc) in s.iter_mut().enumerate() {
                let mut acc = 0.0f64;
                for j in 0..CONV_TAPS {
                    let d = CONV_TAPS - 1 - j;
                    if d <= t {
                        acc +=
                            f64::from(l.taps[CONV_TAPS * c + j]) * f64::from(l.x[(t - d) * ch + c]);
                    }
                }
                *sc = acc * sigmoid(acc);
            }
            for h in 0..nv {
                let unit = |at: usize| {
                    let v = &s[at..at + HEAD];
                    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-6);
                    v.iter().map(|x| x / n).collect::<Vec<f64>>()
                };
                let q: Vec<f64> = unit(h * HEAD)
                    .iter()
                    .map(|x| x / (HEAD as f64).sqrt())
                    .collect();
                let kk = unit(p + h * HEAD);
                let (b, w) = (f64::from(l.b[t * nv + h]), f64::from(l.w[t * nv + h]));
                let ratio = if write {
                    (log_sigmoid(b + w) - log_sigmoid(b)).exp()
                } else {
                    1.0
                };
                let v: Vec<f64> = s[2 * p + h * HEAD..2 * p + (h + 1) * HEAD]
                    .iter()
                    .map(|x| x * ratio)
                    .collect();
                let beta = sigmoid(b);
                let a = -f64::from(l.sa[h]);
                let st = &mut state[h * HEAD * HEAD..(h + 1) * HEAD * HEAD];
                for kc in 0..HEAD {
                    let dec =
                        f64::from(l.f[(t * nv + h) * HEAD + kc]) + f64::from(l.dt[h * HEAD + kc]);
                    let alpha = (f64::from(LB) * sigmoid(a * dec)).exp();
                    for vr in 0..HEAD {
                        st[vr * HEAD + kc] *= alpha;
                    }
                }
                for vr in 0..HEAD {
                    let row = &mut st[vr * HEAD..(vr + 1) * HEAD];
                    let pred: f64 = row.iter().zip(&kk).map(|(x, y)| x * y).sum();
                    let err = v[vr] - pred;
                    for (x, y) in row.iter_mut().zip(&kk) {
                        *x += beta * err * y;
                    }
                    out[t * p + h * HEAD + vr] = row.iter().zip(&q).map(|(x, y)| x * y).sum();
                }
            }
        }
        out
    }

    /// The RDA launches over tokens `t0 .. t0 + m` on the persistent ring and
    /// state; the delta output `[m][n_v·HEAD]`. `write` false skips the
    /// write gate (the separation).
    fn rda_card(
        gpu: &Gpu,
        k: &LinearKernels,
        l: &Rda,
        (t0, m): (usize, usize),
        (ring, state): (&mut DeviceBuffer<f32>, &mut DeviceBuffer<f32>),
        write: bool,
    ) -> Result<Vec<f32>, GateError> {
        let s = gpu.stream();
        let (ch, nv) = (SHAPE.channels(), SHAPE.n_v);
        let up = |v: &[f32]| DeviceBuffer::from_host(s, v);
        let x = up(&l.x[t0 * ch..(t0 + m) * ch])?;
        let b = up(&l.b[t0 * nv..(t0 + m) * nv])?;
        let w = up(&l.w[t0 * nv..(t0 + m) * nv])?;
        let f = up(&l.f[t0 * nv * HEAD..(t0 + m) * nv * HEAD])?;
        let (taps, dt, sa) = (up(&l.taps)?, up(&l.dt)?, up(&l.sa)?);
        let pos: Vec<u32> = (t0..t0 + m).map(|p| p as u32).collect();
        let pd = DeviceBuffer::from_host(s, &pos)?;
        let mut y = up(&vec![0.0f32; m * ch])?;
        let mut beta = up(&vec![0.0f32; m * nv])?;
        let mut decay = up(&vec![0.0f32; m * nv * HEAD])?;
        let fault = gpu.unlabelled_sink();
        k.conv.enqueue_kda_conv_prep(
            s,
            KdaConvArgs {
                x: &x,
                b_raw: &b,
                f: &f,
                w: &taps,
                dt_bias: &dt,
                ssm_a: &sa,
                pos: &pd,
                shape: SHAPE,
                gate: KdaGateRule::Sigmoid { lower_bound: LB },
                beta_scale: KdaBetaScale::X1,
                eps: EPS,
                m,
                fault,
                y: &mut y,
                beta: &mut beta,
                decay: &mut decay,
                ring,
            },
        )?;
        if write {
            k.write_gate.enqueue_write_gate(
                s,
                WriteGateArgs {
                    b: &b,
                    w: &w,
                    shape: SHAPE,
                    m,
                    fault,
                    y: &mut y,
                },
            )?;
        }
        let lw = DeviceBuffer::from_host(s, &[0xdead_beef_u32, 0])?;
        let mut o = up(&vec![0.0f32; m * nv * HEAD])?;
        k.delta.enqueue_kda_delta(
            s,
            DeltaArgs {
                qkv: &y,
                beta: &beta,
                decay: &decay,
                lane: &lw,
                lane_at: 1,
                lanes: 1,
                shape: SHAPE,
                m,
                fault,
                o: &mut o,
                state,
            },
        )?;
        s.synchronize()?;
        Ok(o.to_host_vec(s)?)
    }

    /// The card's delta output over all `T` tokens, as one launch each
    /// (`step` false) or as `T` one-token steps.
    fn rda_run(
        gpu: &Gpu,
        k: &LinearKernels,
        l: &Rda,
        step: bool,
        write: bool,
    ) -> Result<Vec<f32>, GateError> {
        let s = gpu.stream();
        let mut ring = DeviceBuffer::from_host(s, &vec![0.0f32; SHAPE.ring_len()])?;
        let mut state = DeviceBuffer::from_host(s, &vec![0.0f32; SHAPE.state_len()])?;
        if !step {
            return rda_card(gpu, k, l, (0, T), (&mut ring, &mut state), write);
        }
        let mut out = Vec::with_capacity(T * SHAPE.n_v * HEAD);
        for t in 0..T {
            out.extend(rda_card(gpu, k, l, (t, 1), (&mut ring, &mut state), write)?);
        }
        Ok(out)
    }

    /// The largest per-token relative L2 of `got` against `want`.
    fn worst_token(got: &[f32], want: &[f64]) -> f64 {
        let p = SHAPE.n_v * HEAD;
        (0..T)
            .map(|t| rel_l2(&got[t * p..(t + 1) * p], &want[t * p..(t + 1) * p]))
            .fold(0.0, f64::max)
    }

    fn rda_clauses(gpu: &Gpu, k: &LinearKernels) -> Result<Vec<bool>, GateError> {
        let l = Rda::new(0x0072_6461);
        let want = rda_ref(&l, true);
        let mut out = Vec::new();
        for (step, what) in [
            (false, "one launch of 64 tokens"),
            (true, "64 one-token steps"),
        ] {
            let got = rda_run(gpu, k, &l, step, true)?;
            let err = worst_token(&got, &want);
            let pass = err <= RDA_BAND && gpu.fault()?.is_none();
            println!(
                "  rda {what} vs HF f64: worst token rel-L2 {err:.3e} (band {RDA_BAND:.0e}): {}",
                verdict(pass)
            );
            out.push(pass);
        }
        let bare = rda_run(gpu, k, &l, true, false)?;
        let err = worst_token(&bare, &want);
        let pass = err > SEP * RDA_BAND;
        println!(
            "  rda separation: the card without the write gate vs HF f64: rel-L2 {err:.3e}, past \
             {SEP}x band: {}",
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    // ------------------------------------------------------------------ MLA

    /// One latent layer's weights (as the card holds them and dequantized)
    /// and `N` steps of its projections.
    struct Mla {
        kb_qs: Vec<u32>,
        kb_d: Vec<u16>,
        /// `[heads][LATENT][nope]`.
        kb: Vec<f64>,
        vb_words: Vec<u32>,
        /// `[heads][v_head][LATENT]`.
        vb: Vec<f64>,
        gain: Vec<f32>,
        /// `[N][row]` (latent then extra), `[N][heads·head_k]`,
        /// `[N][heads·v_head]`.
        kv: Vec<f32>,
        q: Vec<f32>,
        gate: Vec<f32>,
    }

    /// Q6_K bytes a 256-value super-block.
    const SB_BYTES: usize = 210;

    impl Mla {
        fn new(seed: u64, n: usize) -> Result<Mla, GateError> {
            let mut r = Lcg(seed);
            let d = MLA;
            let (h, nope, row, hk) = (d.heads, d.nope, d.row(), d.head_k());
            let kb_f = r.fill(h * LATENT * nope, -1.0, 1.0);
            let blocks: Vec<Q8Block> = kb_f
                .as_chunks::<32>()
                .0
                .iter()
                .map(|c| Q8Block::quantize_ref(c).ok_or("a non-finite k_b block"))
                .collect::<Result<_, _>>()?;
            let kb = blocks
                .iter()
                .flat_map(|b| {
                    let s = f64::from(half_to_f32(b.d));
                    b.q.map(|q| s * f64::from(q))
                })
                .collect();
            let (kb_qs, kb_d) = q8_0_planes(&blocks);
            let n_rows = h * d.v_head;
            let row_bytes = LATENT / 256 * SB_BYTES;
            let mut bytes = vec![0u8; n_rows * row_bytes];
            for (i, by) in bytes.iter_mut().enumerate() {
                *by = (r.unit() * 256.0) as u8;
                // Each super-block's f16 `d` small and finite.
                if i % SB_BYTES == SB_BYTES - 1 {
                    *by = 0x1c;
                }
            }
            let mut vb = Vec::with_capacity(n_rows * LATENT);
            let mut buf = vec![0.0f32; LATENT];
            for rr in 0..n_rows {
                dequant_row(
                    GgmlType::Q6_K,
                    &bytes[rr * row_bytes..(rr + 1) * row_bytes],
                    &mut buf,
                )?;
                vb.extend(buf.iter().map(|&v| f64::from(v)));
            }
            let vb_words = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect();
            let mut kv = Vec::with_capacity(n * row);
            for _ in 0..n {
                kv.extend(r.fill(LATENT, -1.0, 1.0));
                kv.extend(r.fill(d.extra, -2.0, 2.0));
            }
            let mut q = Vec::with_capacity(n * h * hk);
            for _ in 0..n * h {
                q.extend(r.fill(nope, -1.0, 1.0));
                q.extend(r.fill(d.extra, -2.0, 2.0));
            }
            Ok(Mla {
                kb_qs,
                kb_d,
                kb,
                vb_words,
                vb,
                gain: r.fill(LATENT, 0.5, 1.5),
                kv,
                q,
                gate: r.fill(n * h * d.v_head, -3.0, 3.0),
            })
        }
    }

    /// Which clause a reference leaves out.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Drop {
        None,
        Extra,
        Scale,
        Gate,
    }

    /// Each position's normed latent's nope keys `[heads][nope]` and values
    /// `[heads][v_head]` (HF's `kv_b_proj` of `kv_a_layernorm(kv_lat)`).
    struct Keys {
        knope: Vec<Vec<f64>>,
        vals: Vec<Vec<f64>>,
    }

    fn keys(m: &Mla, n: usize) -> Keys {
        let d = MLA;
        let (h, nope, row) = (d.heads, d.nope, d.row());
        let mut knope = Vec::with_capacity(n);
        let mut vals = Vec::with_capacity(n);
        for t in 0..n {
            let lat = &m.kv[t * row..t * row + LATENT];
            let ms = lat.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / LATENT as f64;
            let inv = 1.0 / (ms + f64::from(d.latent_eps)).sqrt();
            let ln: Vec<f64> = lat
                .iter()
                .zip(&m.gain)
                .map(|(&x, &g)| f64::from(x) * inv * f64::from(g))
                .collect();
            let mut kn = vec![0.0f64; h * nope];
            let mut vv = vec![0.0f64; h * d.v_head];
            for hh in 0..h {
                for (j, &lj) in ln.iter().enumerate() {
                    let wrow = &m.kb[(hh * LATENT + j) * nope..(hh * LATENT + j + 1) * nope];
                    for (i, &wv) in wrow.iter().enumerate() {
                        kn[hh * nope + i] += wv * lj;
                    }
                }
                for r in 0..d.v_head {
                    let wrow =
                        &m.vb[(hh * d.v_head + r) * LATENT..(hh * d.v_head + r + 1) * LATENT];
                    vv[hh * d.v_head + r] = wrow.iter().zip(&ln).map(|(a, b)| a * b).sum();
                }
            }
            knope.push(kn);
            vals.push(vv);
        }
        Keys { knope, vals }
    }

    /// The HF attention of position `p` over keys `0 ..= p`, gated:
    /// `[heads·v_head]`.
    fn mla_ref(m: &Mla, ks: &Keys, p: usize, drop: Drop) -> Vec<f64> {
        let d = MLA;
        let (h, nope, row, hk, vh) = (d.heads, d.nope, d.row(), d.head_k(), d.v_head);
        let width = if drop == Drop::Scale { nope } else { hk };
        let scale = 1.0 / (width as f64).sqrt();
        let mut out = vec![0.0f64; h * vh];
        for hh in 0..h {
            let q = &m.q[(p * h + hh) * hk..(p * h + hh + 1) * hk];
            let scores: Vec<f64> = (0..=p)
                .map(|t| {
                    let kn = &ks.knope[t][hh * nope..(hh + 1) * nope];
                    let mut s: f64 = q[..nope]
                        .iter()
                        .zip(kn)
                        .map(|(&a, b)| f64::from(a) * b)
                        .sum();
                    if drop != Drop::Extra {
                        let ex = &m.kv[t * row + LATENT..(t + 1) * row];
                        s += q[nope..]
                            .iter()
                            .zip(ex)
                            .map(|(&a, &b)| f64::from(a) * f64::from(b))
                            .sum::<f64>();
                    }
                    s * scale
                })
                .collect();
            let mx = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = scores.iter().map(|s| (s - mx).exp()).collect();
            let z: f64 = e.iter().sum();
            for r in 0..vh {
                let o: f64 = e
                    .iter()
                    .enumerate()
                    .map(|(t, w)| w * ks.vals[t][hh * vh + r])
                    .sum::<f64>()
                    / z;
                let g = if drop == Drop::Gate {
                    1.0
                } else {
                    sigmoid(f64::from(m.gate[(p * h + hh) * vh + r]))
                };
                out[hh * vh + r] = o * g;
            }
        }
        out
    }

    /// `attend` at positions `0 .. n` on a fresh cache of `ctx` rows; the
    /// gated heads at each of `at`.
    fn mla_card(
        gpu: &Gpu,
        mk: &MlaKernels,
        m: &Mla,
        (ctx, n): (usize, usize),
        at: &[usize],
    ) -> Result<Vec<Vec<f32>>, GateError> {
        let s = gpu.stream();
        let d = MLA;
        let (h, row, hk, vh) = (d.heads, d.row(), d.head_k(), d.v_head);
        let qs = DeviceTensor::upload(s, &m.kb_qs, h * LATENT, d.nope / 4)?;
        let dd = DeviceTensor::upload(s, &m.kb_d, h * LATENT, d.nope / 32)?;
        let vw = DeviceTensor::upload(s, &m.vb_words, h * vh, LATENT / 256 * SB_BYTES / 4)?;
        let gain = DeviceBuffer::from_host(s, &m.gain)?;
        let z = |n: usize| DeviceBuffer::from_host(s, &vec![0.0f32; n]);
        let (mut kv, mut q, mut gate) = (z(row)?, z(h * hk)?, z(h * vh)?);
        let (mut pos, mut n_keys) = (
            DeviceBuffer::from_host(s, &[0u32])?,
            DeviceBuffer::from_host(s, &[1u32])?,
        );
        let (pv, pm) = partials_lens(&d, ctx);
        let (mut qabs, mut att, mut av) = (z(h * row)?, z(h * LATENT)?, z(h * vh)?);
        let (mut part_v, mut part_ms) = (z(pv)?, z(pm)?);
        let mut cache = DeviceTensor::<u16>::zeroed(s, ctx, row)?;
        let mut y = z(h * vh)?;
        let mut out = Vec::new();
        for p in 0..n {
            kv.copy_from_host(s, &m.kv[p * row..(p + 1) * row])?;
            q.copy_from_host(s, &m.q[p * h * hk..(p + 1) * h * hk])?;
            gate.copy_from_host(s, &m.gate[p * h * vh..(p + 1) * h * vh])?;
            pos.copy_from_host(s, &[p as u32])?;
            n_keys.copy_from_host(s, &[p as u32 + 1])?;
            attend(
                gpu,
                mk,
                &d,
                AttendIn {
                    kv: &kv,
                    q: &q,
                    gate: &gate,
                    kv_norm: &gain,
                    k_b: (&qs, &dd),
                    v_b: ValueW::Q6K(&vw),
                    pos: &pos,
                    n_keys: &n_keys,
                    fault: gpu.unlabelled_sink(),
                },
                AttnBufs {
                    qabs: &mut qabs,
                    att: &mut att,
                    av: &mut av,
                    part_v: &mut part_v,
                    part_ms: &mut part_ms,
                },
                &mut cache,
                &mut y,
            )?;
            if at.contains(&p) {
                s.synchronize()?;
                out.push(y.to_host_vec(s)?);
            }
        }
        s.synchronize()?;
        Ok(out)
    }

    fn mla_clauses(gpu: &Gpu) -> Result<Vec<bool>, GateError> {
        let mk = MlaKernels::load(gpu, &MLA, 1)?;
        let n = MLA_RUNS.iter().map(|r| r.1).max().unwrap_or(0);
        let m = Mla::new(0x006d_6c61, n)?;
        let ks = keys(&m, n);
        let mut out = Vec::new();
        let mut seps = [f64::INFINITY; 3];
        for (ctx, steps, at) in MLA_RUNS {
            let got = mla_card(gpu, &mk, &m, (ctx, steps), at)?;
            for (&p, g) in at.iter().zip(&got) {
                let err = rel_l2(g, &mla_ref(&m, &ks, p, Drop::None));
                let pass = err <= MLA_BAND && gpu.fault()?.is_none();
                println!(
                    "  mla cache {ctx} rows, position {p} vs HF f64: rel-L2 {err:.3e} (band \
                     {MLA_BAND:.0e}): {}",
                    verdict(pass)
                );
                out.push(pass);
                if Some(&p) == at.last() {
                    for (sep, drop) in seps.iter_mut().zip([Drop::Extra, Drop::Scale, Drop::Gate]) {
                        *sep = sep.min(rel_l2(g, &mla_ref(&m, &ks, p, drop)));
                    }
                }
            }
        }
        for (sep, what) in seps.iter().zip([
            "without the extra part",
            "at scale 1/sqrt(128)",
            "without the gate",
        ]) {
            let pass = *sep > SEP * MLA_BAND;
            println!(
                "  mla separation: the card vs the HF reference {what}: least rel-L2 {sep:.3e}, past \
                 {SEP}x band: {}",
                verdict(pass)
            );
            out.push(pass);
        }
        Ok(out)
    }
}
