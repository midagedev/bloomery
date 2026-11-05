//! GPU gate for Kimi K3's KDA layer, model-less at the model's geometry (heads of 128, 2 and 96 of them:
//! the fixture's and the real file's), each against an f64 reference written from llama.cpp's
//! `build_kda_layer` (`src/models/kimi-k3.cpp`, and `build_delta_net_autoregressive` of
//! `src/models/delta-net-base.cpp`) in the graph's order. Inputs are seeded; nothing is read from a model
//! file.
//!
//! K3's layer is `mixer_kda::enqueue` with the rule `Sigmoid { lower_bound: -5 }` at `beta_scale` 1; its
//! three conv tap sets are joined in q, k, v order, and its gate is one full-rank projection that this
//! gate's [`KdaProj`] writes straight into `z`. The projections run on the card over f32 weights
//! (`f32_gemv` at one row, the F32 tile past it). The layer starts from a warm state: a random conv
//! history at positions `P0 - 3 ..`, a random recurrent state, its stamp `P0`.
//!
//! Per geometry:
//! 1. `decode`: one step (`m` = 1) against the reference: the layer's output row and the state after the
//!    step, both within [`BAND`] relative L2.
//! 2. `steps`: eight steps in a row against the reference: the worst token's output and the state
//!    after the eighth, within [`BAND`].
//! 3. `prefill`: the eight tokens as one call (`m` = 8): the worst token's output and the final state
//!    within [`BAND`], and the output, state, conv ring and stamp bit for bit the eight steps'.
//! 4. `gate`: the card run with the gate's projection replaced by +40 (σ = 1, the gate mutated to 1)
//!    lies past [`SEP`] bands of the reference, so the clauses above see the gate.
//!
//! Once: `refused`: the dims a file's numbers make (`KdaDims::of_file`) refuse by name a head width
//! other than 128, a conv of other than 4 taps, no head, a hidden width that is not a multiple of 32, an
//! epsilon that is not a positive finite number and a decay rule no conv-prep entry runs, and take K3's.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_kimi_k3_kda", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::gemm::Gemm32Kernels;
    use bloomery_gpu::linear::mixer_kda::{
        self, KdaBufs, KdaDecay, KdaDims, KdaFile, KdaIn, KdaProj, KdaRule, KdaState, KdaVecs,
    };
    use bloomery_gpu::linear::{CONV_TAPS, HEAD, LinearKernels, RING_ROWS};
    use bloomery_gpu::{DeviceTensor, Gpu, GpuError};
    use bloomery_gpu_gates::{GateError, GateRng, bits_equal, checks_failed, par_map, verdict};
    use cuda_core::DeviceBuffer;

    /// The name the refusals carry.
    const WHAT: &str = "kimi_k3 kda";
    /// `kda.gate_lower_bound`.
    const LB: f32 = -5.0;
    /// `layer_norm_rms_epsilon`: the norms' and the l2 floor.
    const EPS: f32 = 1e-5;
    /// Tokens of the prompt clause and of the steps clause.
    const T: usize = 8;
    /// The first token's position: the three conv inputs before it are the history.
    const P0: usize = 5;
    /// The layer the launches label their faults with.
    const LAYER: usize = 3;
    /// The f32 chain against f64, relative L2 of a layer's output row or state. The terms at their
    /// worst: a projection's f32 dot of up to 12,288 terms (a lane's 384 terms, then the 32-lane
    /// butterfly) 2.3e-5, the delta step's 128-term dots 1.3e-6 a step over eight steps 1e-5, the
    /// decay's and the gate's `expf` at 2 ulps 1.2e-7, the norms 1e-6: 4e-5 added up. The terms add as
    /// a random walk, and the largest clean error on the A6000 was 8.2e-7; the band is 12 times that.
    const BAND: f64 = 1e-5;
    /// The bands a run missing a clause must lie past.
    const SEP: f64 = 10.0;
    /// The value the mutated gate projection takes: σ(40) is 1 in f32.
    const GATE_ONE: f32 = 40.0;

    /// A layer's two sizes: heads (query/key and value alike) and hidden width.
    #[derive(Clone, Copy)]
    struct Geo {
        heads: usize,
        embd: usize,
    }

    impl Geo {
        /// `d_inner`: the width of q, k, v, the gate and the output's input.
        fn inner(self) -> usize {
            self.heads * HEAD
        }

        /// The joined q·k·v channels.
        fn channels(self) -> usize {
            3 * self.inner()
        }
    }

    /// The fixture's heads and the real file's, each at a hidden width its projections tile (a
    /// multiple of 64): the fixture's is the gate's own, the real file's is K3's 7,168.
    const GEOS: [Geo; 2] = [
        Geo {
            heads: 2,
            embd: 512,
        },
        Geo {
            heads: 96,
            embd: 7168,
        },
    ];

    /// One KDA layer's tensors as the file holds them, f32.
    struct Layer {
        geo: Geo,
        /// `attn_norm`, `[embd]`.
        gain: Vec<f32>,
        /// `attn_q`, `attn_k`, `attn_v` one after another, `[3·d][embd]` row-major.
        qkv: Vec<f32>,
        /// `ssm_conv1d_q`, `_k`, `_v`: each `[d][CONV_TAPS]`, oldest tap first.
        taps: [Vec<f32>; 3],
        /// `ssm_f_a`, `[HEAD][embd]`, and `ssm_f_b`, `[d][HEAD]`.
        f_a: Vec<f32>,
        f_b: Vec<f32>,
        /// `ssm_dt.bias`, `[d]`.
        dt: Vec<f32>,
        /// `ssm_a` = −exp(A_log), `[heads]`.
        sa: Vec<f32>,
        /// `ssm_beta`, `[heads][embd]`.
        beta: Vec<f32>,
        /// `ssm_g`, the full-rank gate, `[d][embd]`.
        g: Vec<f32>,
        /// `ssm_norm`, `[HEAD]`.
        onorm: Vec<f32>,
        /// `attn_output`, `[embd][d]`.
        wo: Vec<f32>,
    }

    fn fill(r: &mut GateRng, n: usize, lim: f32) -> Vec<f32> {
        (0..n).map(|_| r.uni(-lim, lim)).collect()
    }

    /// A weight of `n` values over `k` inputs: uniform with variance `1/k`, so a projection of a unit
    /// row is of unit scale.
    fn weight(r: &mut GateRng, n: usize, k: usize) -> Vec<f32> {
        fill(r, n, (3.0 / k as f32).sqrt())
    }

    impl Layer {
        fn new(geo: Geo, seed: u64) -> Layer {
            let mut r = GateRng(seed);
            let (d, embd, heads) = (geo.inner(), geo.embd, geo.heads);
            let (lo, hi) = (0.5f32.ln(), 8.0f32.ln());
            Layer {
                geo,
                gain: (0..embd).map(|_| r.uni(0.5, 1.5)).collect(),
                qkv: weight(&mut r, 3 * d * embd, embd),
                taps: [
                    fill(&mut r, d * CONV_TAPS, 0.5),
                    fill(&mut r, d * CONV_TAPS, 0.5),
                    fill(&mut r, d * CONV_TAPS, 0.5),
                ],
                f_a: weight(&mut r, HEAD * embd, embd),
                f_b: weight(&mut r, d * HEAD, HEAD),
                dt: fill(&mut r, d, 1.0),
                sa: (0..heads).map(|_| -r.uni(lo, hi).exp()).collect(),
                beta: weight(&mut r, heads * embd, embd),
                g: weight(&mut r, d * embd, embd),
                onorm: (0..HEAD).map(|_| r.uni(0.5, 1.5)).collect(),
                wo: weight(&mut r, embd * d, d),
            }
        }
    }

    /// The layer's state before the first token: the conv ring holding the three inputs before it, the
    /// recurrent state `[head][value][key]` and the tokens' hidden rows.
    struct Warm {
        /// The conv inputs of positions `P0 - 3 .. P0`, `[3][channels]`.
        hist: Vec<f32>,
        /// The conv ring, `[RING_ROWS][channels]`: the history at its slots, the rest random.
        ring: Vec<f32>,
        state: Vec<f32>,
        /// `[T][embd]`.
        x: Vec<f32>,
    }

    impl Warm {
        fn new(geo: Geo, seed: u64) -> Warm {
            let mut r = GateRng(seed);
            let c = geo.channels();
            let hist = fill(&mut r, (CONV_TAPS - 1) * c, 2.0);
            let mut ring = fill(&mut r, RING_ROWS * c, 2.0);
            for (i, row) in hist.chunks_exact(c).enumerate() {
                let slot = (P0 - (CONV_TAPS - 1) + i) % RING_ROWS;
                ring[slot * c..(slot + 1) * c].copy_from_slice(row);
            }
            Warm {
                hist,
                ring,
                state: fill(&mut r, geo.heads * HEAD * HEAD, 0.5),
                x: fill(&mut r, T * geo.embd, 2.0),
            }
        }
    }

    // ------------------------------------------------------------ the f64 reference

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// One dot in f64, four partial sums.
    fn dot(w: &[f32], x: &[f64]) -> f64 {
        let mut a = [0.0f64; 4];
        for (wc, xc) in w.as_chunks::<4>().0.iter().zip(x.as_chunks::<4>().0) {
            for ((acc, &wv), &xv) in a.iter_mut().zip(wc).zip(xc) {
                *acc += f64::from(wv) * xv;
            }
        }
        (a[0] + a[1]) + (a[2] + a[3])
    }

    /// `W · x` for `rows` rows of `k`, in f64, the rows shared among threads.
    fn matvec(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        par_map(rows, |r| dot(&w[r * k..(r + 1) * k], x))
    }

    /// What the reference leaves: the output rows `[T][embd]`, the state after the first token and after
    /// the last, `[head][value][key]`.
    struct Ref {
        out: Vec<f64>,
        first: Vec<f64>,
        last: Vec<f64>,
    }

    /// `build_kda_layer` in the graph's order over the warm layer's `T` tokens, one token at a time (the
    /// autoregressive recurrence): norm, the three projections and their convs with SiLU, the decay from
    /// `f_b(f_a x) + dt` times `ssm_a`, β, the l2 norms, the delta step, the full-rank gate, the output norm
    /// times σ(gate), the output projection.
    fn reference(l: &Layer, w: &Warm) -> Ref {
        let Geo { heads, embd } = l.geo;
        let (d, c) = (l.geo.inner(), l.geo.channels());
        // The conv's inputs by position index: the history, then each token's raw projection.
        let mut inputs: Vec<Vec<f64>> = w
            .hist
            .chunks_exact(c)
            .map(|row| row.iter().map(|&v| f64::from(v)).collect())
            .collect();
        let mut st: Vec<f64> = w.state.iter().map(|&v| f64::from(v)).collect();
        let (mut out, mut first) = (Vec::with_capacity(T * embd), Vec::new());
        for t in 0..T {
            let xt = &w.x[t * embd..(t + 1) * embd];
            let ms = xt.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / embd as f64;
            let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
            let xn: Vec<f64> = xt
                .iter()
                .zip(&l.gain)
                .map(|(&v, &g)| f64::from(v) * inv * f64::from(g))
                .collect();
            // q, k and v are three matrices one after another: the rows of the joined weight.
            inputs.push(matvec(&l.qkv, c, embd, &xn));
            // Each channel's conv over the last four inputs, oldest tap first, then SiLU.
            let span = &inputs[t..t + CONV_TAPS];
            let conv: Vec<f64> = (0..c)
                .map(|ch| {
                    let taps = &l.taps[ch / d][(ch % d) * CONV_TAPS..][..CONV_TAPS];
                    let acc: f64 = (0..CONV_TAPS)
                        .map(|j| f64::from(taps[j]) * span[j][ch])
                        .sum();
                    acc * sigmoid(acc)
                })
                .collect();
            // g1 = (f_b(f_a x) + dt) · ssm_a; the decay exponent is lb · σ(−g1).
            let fa = matvec(&l.f_a, HEAD, embd, &xn);
            let f = matvec(&l.f_b, d, HEAD, &fa);
            let g: Vec<f64> = (0..d)
                .map(|i| {
                    let a = (f[i] + f64::from(l.dt[i])) * f64::from(l.sa[i / HEAD]);
                    f64::from(LB) * sigmoid(-a)
                })
                .collect();
            let beta: Vec<f64> = matvec(&l.beta, heads, embd, &xn)
                .into_iter()
                .map(sigmoid)
                .collect();
            let g2 = matvec(&l.g, d, embd, &xn);
            let mut o = vec![0.0f64; d];
            for h in 0..heads {
                let unit = |at: usize| -> Vec<f64> {
                    let v = &conv[at..at + HEAD];
                    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                    let n = n.max(f64::from(EPS));
                    v.iter().map(|x| x / n).collect()
                };
                let q: Vec<f64> = unit(h * HEAD)
                    .iter()
                    .map(|x| x / (HEAD as f64).sqrt())
                    .collect();
                let k = unit(d + h * HEAD);
                let v = &conv[2 * d + h * HEAD..2 * d + (h + 1) * HEAD];
                let s = &mut st[h * HEAD * HEAD..(h + 1) * HEAD * HEAD];
                // The decay is per key channel: it scales column `kc` of every value row.
                for kc in 0..HEAD {
                    let a = g[h * HEAD + kc].exp();
                    for vr in 0..HEAD {
                        s[vr * HEAD + kc] *= a;
                    }
                }
                for vr in 0..HEAD {
                    let row = &mut s[vr * HEAD..(vr + 1) * HEAD];
                    let sk: f64 = row.iter().zip(&k).map(|(a, b)| a * b).sum();
                    let delta = (v[vr] - sk) * beta[h];
                    for (a, b) in row.iter_mut().zip(&k) {
                        *a += b * delta;
                    }
                    o[h * HEAD + vr] = row.iter().zip(&q).map(|(a, b)| a * b).sum();
                }
            }
            // RMS norm over each head's 128 values with `ssm_norm`, times σ of the gate projection.
            let mut gated = vec![0.0f64; d];
            for h in 0..heads {
                let oh = &o[h * HEAD..(h + 1) * HEAD];
                let ms = oh.iter().map(|x| x * x).sum::<f64>() / HEAD as f64;
                let inv = 1.0 / (ms + f64::from(EPS)).sqrt();
                for (j, (&ov, &gain)) in oh.iter().zip(&l.onorm).enumerate() {
                    gated[h * HEAD + j] = ov * inv * f64::from(gain) * sigmoid(g2[h * HEAD + j]);
                }
            }
            out.extend(matvec(&l.wo, embd, d, &gated));
            if t == 0 {
                first = st.clone();
            }
        }
        Ref {
            out,
            first,
            last: st,
        }
    }

    // ------------------------------------------------------------------ the card

    /// The layer's weights on the card: the projections as f32 tensors and the f32 vectors the mixer
    /// reads, the three conv tap sets joined in q, k, v order.
    struct Dev {
        qkv: DeviceTensor<f32>,
        f_a: DeviceTensor<f32>,
        f_b: DeviceTensor<f32>,
        beta: DeviceTensor<f32>,
        g: DeviceTensor<f32>,
        wo: DeviceTensor<f32>,
        gain: DeviceBuffer<f32>,
        conv: DeviceBuffer<f32>,
        dt: DeviceBuffer<f32>,
        sa: DeviceBuffer<f32>,
        onorm: DeviceBuffer<f32>,
    }

    impl Dev {
        fn upload(gpu: &Gpu, l: &Layer) -> Result<Dev, GateError> {
            let s = gpu.stream();
            let Geo { heads, embd } = l.geo;
            let (d, c) = (l.geo.inner(), l.geo.channels());
            let joined: Vec<f32> = l.taps.iter().flatten().copied().collect();
            Ok(Dev {
                qkv: DeviceTensor::upload(s, &l.qkv, c, embd)?,
                f_a: DeviceTensor::upload(s, &l.f_a, HEAD, embd)?,
                f_b: DeviceTensor::upload(s, &l.f_b, d, HEAD)?,
                beta: DeviceTensor::upload(s, &l.beta, heads, embd)?,
                g: DeviceTensor::upload(s, &l.g, d, embd)?,
                wo: DeviceTensor::upload(s, &l.wo, embd, d)?,
                gain: DeviceBuffer::from_host(s, &l.gain)?,
                conv: DeviceBuffer::from_host(s, &joined)?,
                dt: DeviceBuffer::from_host(s, &l.dt)?,
                sa: DeviceBuffer::from_host(s, &l.sa)?,
                onorm: DeviceBuffer::from_host(s, &l.onorm)?,
            })
        }
    }

    /// What the gate's projection writes into `z`.
    #[derive(Clone, Copy, PartialEq)]
    enum Gate {
        /// `ssm_g · xn`, the layer's own.
        Live,
        /// [`GATE_ONE`] everywhere: σ(z) = 1.
        One,
    }

    /// K3's projections over f32 weights: one `f32_gemv` at a row, the F32 tile past it, both token-major
    /// and bit for bit each other's on a column.
    struct K3<'a> {
        dev: &'a Dev,
        g32: &'a Gemm32Kernels,
        gate: Gate,
    }

    impl K3<'_> {
        fn project(
            &self,
            gpu: &Gpu,
            w: &DeviceTensor<f32>,
            x: &DeviceBuffer<f32>,
            m: usize,
            y: &mut DeviceBuffer<f32>,
        ) -> Result<(), GpuError> {
            let s = gpu.stream();
            if m == 1 {
                gpu.q8f32().enqueue_f32_gemv(s, w, x, 1, y)
            } else {
                self.g32.enqueue_f32_tile(s, w, x, m, y)
            }
        }
    }

    impl KdaProj for K3<'_> {
        fn input(
            &mut self,
            gpu: &Gpu,
            _l: usize,
            m: usize,
            xn: &DeviceBuffer<f32>,
            y: KdaIn<'_>,
        ) -> Result<(), GpuError> {
            let dev = self.dev;
            self.project(gpu, &dev.qkv, xn, m, y.qkv)?;
            self.project(gpu, &dev.f_a, xn, m, &mut *y.fa)?;
            self.project(gpu, &dev.beta, xn, m, y.beta_raw)?;
            self.project(gpu, &dev.f_b, &*y.fa, m, y.f)?;
            match self.gate {
                Gate::Live => self.project(gpu, &dev.g, xn, m, y.z),
                Gate::One => {
                    Ok(y.z.copy_from_host(gpu.stream(), &vec![GATE_ONE; m * dev.g.rows()])?)
                }
            }
        }

        fn output(
            &mut self,
            gpu: &Gpu,
            _l: usize,
            m: usize,
            gated: &DeviceBuffer<f32>,
            out: &mut DeviceBuffer<f32>,
        ) -> Result<(), GpuError> {
            self.project(gpu, &self.dev.wo, gated, m, out)
        }
    }

    /// K3's rule: the bounded sigmoid decay at the file's bound, β = σ(b), no write gate.
    fn k3_rule() -> KdaRule {
        KdaRule {
            decay: KdaDecay::Sigmoid { lower_bound: LB },
            beta_scale: 1.0,
            write_gate: false,
        }
    }

    fn k3_file(geo: Geo) -> KdaFile {
        KdaFile {
            embd: geo.embd,
            heads: geo.heads,
            head_dim: HEAD,
            conv_taps: CONV_TAPS,
            rms_eps: EPS,
        }
    }

    /// What a card pass leaves.
    struct Pass {
        /// `[T][embd]`.
        out: Vec<f32>,
        /// The state after the first step (a stepped pass only).
        first: Vec<f32>,
        state: Vec<f32>,
        ring: Vec<f32>,
        stamp: Vec<u32>,
    }

    /// The layer over the warm tokens on the card: as `T` one-row steps (`steps`) or as one call of `T`
    /// rows, from the warm ring, state and stamp.
    fn card(
        gpu: &Gpu,
        k: &LinearKernels,
        g32: &Gemm32Kernels,
        (l, dev, w): (&Layer, &Dev, &Warm),
        (steps, gate): (bool, Gate),
    ) -> Result<Pass, GateError> {
        let s = gpu.stream();
        let Geo { heads, embd } = l.geo;
        let (d, c) = (l.geo.inner(), l.geo.channels());
        let dims = KdaDims::of_file(WHAT, k3_file(l.geo), k3_rule())?;
        let mut state = DeviceBuffer::from_host(s, &w.state)?;
        let mut ring = DeviceBuffer::from_host(s, &w.ring)?;
        let mut stamp = DeviceBuffer::from_host(s, &[P0 as u32])?;
        let lane = DeviceBuffer::from_host(s, &[0u32])?;
        let calls: Vec<(usize, usize)> = if steps {
            (0..T).map(|t| (t, 1)).collect()
        } else {
            vec![(0, T)]
        };
        let (mut out_all, mut first) = (Vec::with_capacity(T * embd), Vec::new());
        for (t0, m) in calls {
            let x = DeviceBuffer::from_host(s, &w.x[t0 * embd..(t0 + m) * embd])?;
            let pos: Vec<u32> = (t0..t0 + m).map(|t| (P0 + t) as u32).collect();
            let pos = DeviceBuffer::from_host(s, &pos)?;
            let z = |n: usize| DeviceBuffer::<f32>::zeroed(s, n);
            let (mut xn, mut qkv, mut fa, mut ga) = (z(m * embd)?, z(m * c)?, z(m * HEAD)?, z(1)?);
            let (mut beta_raw, mut f, mut zz) = (z(m * heads)?, z(m * d)?, z(m * d)?);
            let (mut conv, mut beta, mut decay) = (z(m * c)?, z(m * heads)?, z(m * d)?);
            let (mut o, mut gated, mut out) = (z(m * d)?, z(m * d)?, z(m * embd)?);
            mixer_kda::enqueue(
                gpu,
                s,
                k,
                &mut K3 { dev, g32, gate },
                (LAYER, m, &dims),
                &KdaVecs {
                    norm: &dev.gain,
                    conv: &dev.conv,
                    dt_bias: &dev.dt,
                    a: &dev.sa,
                    gate_norm: &dev.onorm,
                },
                KdaState {
                    state: &mut state,
                    stamp: &mut stamp,
                    ring: &mut ring,
                    lane: &lane,
                    lanes: 1,
                    each: false,
                    row: 0,
                },
                KdaBufs {
                    x: &x,
                    pos: &pos,
                    xn: &mut xn,
                    qkv: &mut qkv,
                    fa: &mut fa,
                    ga: &mut ga,
                    beta_raw: &mut beta_raw,
                    write_raw: None,
                    f: &mut f,
                    z: &mut zz,
                    conv: &mut conv,
                    beta: &mut beta,
                    decay: &mut decay,
                    o: &mut o,
                    gated: &mut gated,
                    out: &mut out,
                },
            )?;
            s.synchronize()?;
            out_all.extend(out.to_host_vec(s)?);
            if steps && t0 == 0 {
                first = state.to_host_vec(s)?;
            }
        }
        Ok(Pass {
            out: out_all,
            first,
            state: state.to_host_vec(s)?,
            ring: ring.to_host_vec(s)?,
            stamp: stamp.to_host_vec(s)?,
        })
    }

    // ----------------------------------------------------------------- clauses

    /// `‖a − b‖₂ / ‖b‖₂` in f64; infinite for another length or a non-finite value.
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

    /// The largest per-token relative L2 of `got` against `want`, rows of `row`.
    fn worst_token(got: &[f32], want: &[f64], row: usize) -> f64 {
        (0..T)
            .map(|t| {
                let at = t * row..(t + 1) * row;
                match (got.get(at.clone()), want.get(at)) {
                    (Some(g), Some(w)) => rel_l2(g, w),
                    _ => f64::INFINITY,
                }
            })
            .fold(0.0, f64::max)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = LinearKernels::load(gpu.context())?;
        let g32 = Gemm32Kernels::load(gpu.context())?;
        println!(
            "gate_kimi_k3_kda: device {} — KDA heads of {HEAD}, geometries {}, synthetic",
            gpu.device_name()?,
            GEOS.iter()
                .map(|g| format!("{}x{HEAD} embd {}", g.heads, g.embd))
                .collect::<Vec<_>>()
                .join(" and ")
        );
        let mut results = refused_clauses();
        for (i, geo) in GEOS.into_iter().enumerate() {
            let seed = 0x4b33_0001 + i as u64;
            let l = Layer::new(geo, seed);
            let w = Warm::new(geo, seed ^ 0x9e37_79b9);
            let dev = Dev::upload(&gpu, &l)?;
            let want = reference(&l, &w);
            results.extend(geo_clauses(&gpu, &k, &g32, (&l, &dev, &w), &want)?);
        }
        let failed = results.iter().filter(|p| !**p).count();
        println!(
            "gate_kimi_k3_kda: {} — {} clauses, {failed} failed",
            verdict(failed == 0),
            results.len()
        );
        if failed == 0 {
            Ok(())
        } else {
            Err(checks_failed())
        }
    }

    fn geo_clauses(
        gpu: &Gpu,
        k: &LinearKernels,
        g32: &Gemm32Kernels,
        (l, dev, w): (&Layer, &Dev, &Warm),
        want: &Ref,
    ) -> Result<Vec<bool>, GateError> {
        let (heads, embd) = (l.geo.heads, l.geo.embd);
        let tag = format!("{heads}x{HEAD}");
        let mut out = Vec::new();
        let steps = card(gpu, k, g32, (l, dev, w), (true, Gate::Live))?;
        let batch = card(gpu, k, g32, (l, dev, w), (false, Gate::Live))?;
        let clean = gpu.fault()?.is_none();

        let e_out = rel_l2(&steps.out[..embd], &want.out[..embd]);
        let e_state = rel_l2(&steps.first, &want.first);
        let pass = e_out <= BAND && e_state <= BAND && clean;
        println!(
            "  {tag} decode (n=1) vs f64: output row rel-L2 {e_out:.3e}, state rel-L2 {e_state:.3e} \
             (band {BAND:.0e}): {}",
            verdict(pass)
        );
        out.push(pass);

        let e_out = worst_token(&steps.out, &want.out, embd);
        let e_state = rel_l2(&steps.state, &want.last);
        let pass = e_out <= BAND && e_state <= BAND && clean;
        println!(
            "  {tag} {T} steps vs f64: worst token rel-L2 {e_out:.3e}, final state rel-L2 \
             {e_state:.3e} (band {BAND:.0e}): {}",
            verdict(pass)
        );
        out.push(pass);

        let e_out = worst_token(&batch.out, &want.out, embd);
        let e_state = rel_l2(&batch.state, &want.last);
        let same = bits_equal(&batch.out, &steps.out)
            && bits_equal(&batch.state, &steps.state)
            && bits_equal(&batch.ring, &steps.ring)
            && batch.stamp == steps.stamp
            && batch.stamp == [(P0 + T) as u32];
        let pass = e_out <= BAND && e_state <= BAND && same && clean;
        println!(
            "  {tag} prefill ({T} rows in one call) vs f64: worst token rel-L2 {e_out:.3e}, final \
             state rel-L2 {e_state:.3e} (band {BAND:.0e}); output, state, ring and stamp bit for bit \
             the {T} steps' {same}: {}",
            verdict(pass)
        );
        out.push(pass);

        let one = card(gpu, k, g32, (l, dev, w), (false, Gate::One))?;
        let e = worst_token(&one.out, &want.out, embd);
        let pass = e > SEP * BAND && gpu.fault()?.is_none();
        println!(
            "  {tag} gate: the card with the gate projection at {GATE_ONE} vs f64: rel-L2 {e:.3e}, past \
             {SEP}x band: {}",
            verdict(pass)
        );
        out.push(pass);
        Ok(out)
    }

    /// Whether `r` is the refusal of `what`'s file number named by `needle`.
    fn refuses(r: Result<KdaDims, GpuError>, needle: &str) -> bool {
        r.err()
            .is_some_and(|e| e.to_string().starts_with(&format!("{WHAT}: {needle}")))
    }

    fn refused_clauses() -> Vec<bool> {
        let real = GEOS[1];
        let ok = k3_file(real);
        let taken = KdaDims::of_file(WHAT, ok, k3_rule()).is_ok_and(|d| {
            d.embd == real.embd
                && d.shape.n_k == real.heads
                && d.shape.n_v == real.heads
                && d.shape.channels() == real.channels()
                && d.rms_eps == EPS
                && d.rule == k3_rule()
        });
        let bounded = |lower_bound| KdaRule {
            decay: KdaDecay::Sigmoid { lower_bound },
            ..k3_rule()
        };
        let cases = [
            (
                "head_dim 64",
                KdaFile { head_dim: 64, ..ok },
                k3_rule(),
                "kda.head_dim 64",
            ),
            (
                "head_dim 256",
                KdaFile {
                    head_dim: 256,
                    ..ok
                },
                k3_rule(),
                "kda.head_dim 256",
            ),
            (
                "conv 3 taps",
                KdaFile { conv_taps: 3, ..ok },
                k3_rule(),
                "ssm.conv_kernel 3",
            ),
            (
                "no heads",
                KdaFile { heads: 0, ..ok },
                k3_rule(),
                "no KDA head",
            ),
            (
                "embd 7170",
                KdaFile { embd: 7170, ..ok },
                k3_rule(),
                "embedding length 7170",
            ),
            (
                "eps 0",
                KdaFile { rms_eps: 0.0, ..ok },
                k3_rule(),
                "rms epsilon 0",
            ),
            (
                "eps NaN",
                KdaFile {
                    rms_eps: f32::NAN,
                    ..ok
                },
                k3_rule(),
                "rms epsilon NaN",
            ),
            ("bound 0", ok, bounded(0.0), "no conv-prep entry runs"),
        ];
        let mut out = vec![taken];
        println!(
            "  refused: K3's 96x{HEAD} dims taken (n_k = n_v = 96, {} channels, the file's rule): {}",
            real.channels(),
            verdict(taken)
        );
        for (what, file, rule, needle) in cases {
            let pass = refuses(KdaDims::of_file(WHAT, file, rule), needle);
            println!(
                "  refused: {what} named \"{WHAT}: {needle}\": {}",
                verdict(pass)
            );
            out.push(pass);
        }
        out
    }
}
