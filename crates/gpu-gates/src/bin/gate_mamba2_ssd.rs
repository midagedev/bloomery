//! GPU gate for the chunked SSD prompt scan (`bloomery_gpu::mamba2_ssd`): one
//! launch per pass over the conv rows and `σ` the pass's pre-pass writes,
//! against an f64 twin of the scan rule run token by token, and beside the
//! engine's f32 pass (`Mamba2Kernels::enqueue_steps`, no norm) on the same
//! raw rows. Synthetic inputs at the Nemotron 3.5 Bolt widths (`d_inner`
//! 8,192 as 128 heads of 64, 8 groups, state 128, 4 conv taps), seeded, no
//! model file. Every pass starts from a nonzero scan state.
//!
//! 1. band: passes of `U` in {1, 31, 32, 33, 97, 2048} tokens (a partial
//!    chunk, one chunk, one past, many), and two passes of 97 and 64 tokens
//!    in a row (the second starts from the first's state, as a prompt's
//!    passes do). The host makes the conv rows (`conv_tokens`, SiLU) and `σ`
//!    from the raw rows and uploads them; the twin reads the same f32 values.
//!    Every `y` and the final state of the SSD are held to the derived bound
//!    below, element by element, and so are the pass's (its own f32
//!    roundings and the device conv's ulps sit orders under the f16 terms):
//!    the pass held to the bound says the twin computes what the engine's
//!    pass computes, the SSD held to it says the SSD does. The gate prints
//!    the diagnostics: the worst `|Δ|/bound`, the RMS of `Δ/magnitude` (σ)
//!    of the SSD, of the pass and of the SSD against the pass, and the mean
//!    bound over magnitude (κ). Heads mix a strong decay (`dt + bias > 20`,
//!    `σ = dt`), a weak one (`|A| <= e^-4`) and the model's range.
//! 2. determinism: the 2048-token pass run again from the same state gives
//!    the same bits (`y` and the state).
//! 3. fault: a NaN conv `x`, a NaN `B`, a `+∞` in `z` and a NaN `σ` in a
//!    97-token pass each raise `Mamba2Scan` at the launch's layer; the rows
//!    of the chunks before the plant's are the clean run's bits, every value
//!    outside what the plant reaches is too (another head, another group, or
//!    for `z` every other value), and what it reaches is non-finite from the
//!    plant's row on. Clean launches raise nothing.
//! 4. refuse: a head width other than 64 (the 4B's 80), a conv of 3 taps (a
//!    shape the host takes), a pass of 0 tokens, a `z` window past the row stride, and a
//!    short `conv`, `sd`, `y` and state are refused by name; nothing launched
//!    (`y` keeps its poison, the fault word stays clean).
//! 5. shape: `mamba2_ssd` compiles with no local depot.
//!
//! The bound of clause 1. Write `u = 2^-24` (f32), `e = 2^-11` (f16's unit
//! roundoff) and `η = 2^-25` (half f16's smallest subnormal step): an f16
//! rounding moves `v` by at most `e·|v| + η`. A chunk of `L` tokens of head
//! `h` has the decay prefix `c_t = Σ_{τ<=t} σ_τ·A` (natural log), the card's
//! in log2 units summed by a five-level shuffle scan of `fl(σ·fl(A·log2 e))`:
//! each prefix is off by at most `7u·A_c` (`A_c = Σ_chunk |σA|·log2 e`), a
//! difference of two by `14u·A_c` plus its own rounding, and `ex2.approx`
//! adds `2^-21` relative (the assumption: PTX documents `2^-22`), so every
//! decay factor the kernel forms is within `δc = ln 2·16u·A_c + 2^-21`
//! relative. `S̃` is the card's f32 state at the chunk start, within `E`
//! (per element) of the exact `S`; `m = |S| + E` bounds `|S̃|`. Per row `t`
//! and channel `p`, with `|·|` sums over the 128 states:
//! - the state term `e^{c_t}·Σ C·S`: `C` and `S̃` rounded to f16 and summed in
//!   f32, then scaled: off by `e^{c_t}·(Σ|C|·E + (2e + δc)·Σ|C|·m + η·(Σ|C| +
//!   Σ m))`;
//! - the chunk term `Σ_{τ<=t} w_tτ·G_tτ·xσ_τ` (`w = e^{c_t − c_τ}`, `G = Σ
//!   C_t·B_τ`, `g = Σ|C_t||B_τ|`): `G` from f16 `C` and `B` summed in f32 is
//!   off by `off(G) = (2e + 256u)·g + η·(Σ|C_t| + Σ|B_τ|)`; `M = fl16(G·w)`
//!   and `fl16(xσ)` add `e + δc + 2u` and `e` of `w·g·|xσ|` and an `η` each:
//!   off by `Σ_τ [w·(off(G) + g·(δc + 2e + 2u))·|xσ| + η·(|xσ| + w·g)]`;
//! - the f32 accumulation of both (161 adds and a scale): `2·162·u` of their
//!   magnitude `Y = e^{c_t}·Σ|C|·m + Σ_τ w·g·|xσ|`;
//! - the output `SiLU(z)·(y + x·D)` with `SiLU` by `ex2.approx` and
//!   `rcp.approx` (`2^-20 + 2u·|z|` relative) and three roundings: off by
//!   `|SiLU(z)|·(δy + (2^-20 + 2u·|z| + 4u)·(Y + δy + |x·D|))`.
//!
//! The state at the chunk end, `e^{c_L}·S + Σ_τ w_Lτ·xσ_τ·B_τ`: the card
//! scales `S̃` (off by `e^{c_L}·(E + δc·m)`) and adds the f16 products of
//! `fl16(xσ·w)` and `B` (off by `(2e + δc + 2u)·N + η·Σ_τ(|B_τ| + w·|xσ_τ|)`,
//! `N = Σ_τ w_Lτ·|xσ_τ|·|B_τ|`) in f32 (`2·34·u` of `e^{c_L}·m + N`): that
//! sum is the next chunk's `E`. Second-order terms are dropped (each is
//! under `e` of a first-order one), an absolute `(2·128 + 2·32)·η²` floor is
//! added, and every bound is multiplied by `1 + 2^-8`. A pass's first `E` is 0 (the
//! twin starts from the state the card reads); a second pass carries the
//! first's. The twin's own f64 roundings are below every term.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mamba2_ssd", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::linear::RING_ROWS;
    use bloomery_gpu::mamba2::{Mamba2Kernels, Mamba2PassScratch, STATE, StepsArgs};
    use bloomery_gpu::mamba2_ssd::{CHUNK, HEAD_DIM, Mamba2Ssd, SsdArgs};
    use bloomery_gpu::{Fault, FaultSite, Gpu, GpuError};
    use bloomery_gpu_gates::rounding::U;
    use bloomery_gpu_gates::{
        GateError, GateRng as Rng, bits_equal, checks_failed, no_local_depot, par_map, verdict,
    };
    use cuda_core::DeviceBuffer;
    use model::mamba2::{self, Checked, Geom};
    use std::f64::consts::{LN_2, LOG2_E};

    const GEOM: Geom = Geom {
        d_inner: 8192,
        n_heads: 128,
        n_groups: 8,
        d_state: STATE,
        d_conv: 4,
    };
    const HEADS: usize = 128;
    const GROUPS: usize = 8;
    const PER: usize = HEADS / GROUPS;
    const INNER: usize = HEADS * HEAD_DIM;
    const CONV_DIM: usize = INNER + 2 * GROUPS * STATE;
    /// A pass row: the Nemotron in-projection's `z`, `xbc` and `dt` columns,
    /// in that order (`z` at column 0).
    const X0: usize = INNER;
    const DT0: usize = INNER + CONV_DIM;
    const LD: usize = INNER + CONV_DIM + HEADS;
    /// The layer the launches raise on.
    const LAYER: usize = 7;
    /// What the ring holds where nothing was put: the pass at position 0 must
    /// not read it.
    const GARBAGE: f32 = 7.0;
    /// What `y` holds before a launch.
    const POISON: u32 = 0x7fa5_a5a5;
    /// The single passes of clause 1, and the longest pass any clause runs.
    const US: [usize; 6] = [1, 31, 32, 33, 97, 2048];
    const MAX_U: usize = 2048;
    /// f16's unit roundoff and half its smallest subnormal step (module doc).
    const E16: f64 = 1.0 / 2048.0;
    const ETA: f64 = 1.0 / 33_554_432.0;
    /// `ex2.approx`'s relative error, assumed (module doc).
    const EX2: f64 = 1.0 / 2_097_152.0;
    /// `SiLU`'s by `ex2.approx` and `rcp.approx`, beyond its `2u·|z|`.
    const SILU: f64 = 1.0 / 1_048_576.0;
    /// The dropped second-order terms (module doc).
    const SLACK: f64 = 1.0 + 1.0 / 256.0;
    const FLOOR: f64 = (2 * STATE + 2 * CHUNK) as f64 * ETA * ETA;

    #[derive(Clone)]
    struct Weights {
        dt_bias: Vec<f32>,
        a: Vec<f32>,
        d: Vec<f32>,
        conv_w: Vec<f32>,
        conv_b: Vec<f32>,
    }

    /// Every sixth head on the identity arm (`dt + bias` in (20.2, 22.8), so
    /// `σ = dt` and the decay is strong), every seventh other one with
    /// `|A| <= e^-4` (weak), the rest `A` in `−e^[−1, 2)` and `σ` about
    /// 0.005 to 0.3.
    fn weights(seed: u64) -> Weights {
        let mut r = Rng(seed);
        let (mut dt_bias, mut a) = (Vec::new(), Vec::new());
        for h in 0..HEADS {
            let (bias, log_a) = if h % 6 == 5 {
                (r.uni(21.0, 22.0), r.uni(-1.0, 2.0))
            } else if h % 7 == 3 {
                (r.uni(-4.5, -2.0), r.uni(-6.0, -4.0))
            } else {
                (r.uni(-4.5, -2.0), r.uni(-1.0, 2.0))
            };
            dt_bias.push(bias);
            a.push(-log_a.exp());
        }
        Weights {
            dt_bias,
            a,
            d: (0..HEADS).map(|_| r.uni(0.5, 1.5)).collect(),
            conv_w: (0..4 * CONV_DIM).map(|_| r.uni(-0.6, 0.6)).collect(),
            conv_b: (0..CONV_DIM).map(|_| r.uni(-0.2, 0.2)).collect(),
        }
    }

    /// One pass's inputs: the raw rows the engine's pass reads, and the
    /// conv rows and `(σ, dA)` pairs the SSD reads, as the host made them.
    #[derive(Clone)]
    struct PassIn {
        u: usize,
        rows: Vec<f32>,
        conv: Vec<f32>,
        sd: Vec<f32>,
    }

    fn silu32(v: f32) -> f32 {
        v / (1.0 + (-v).exp())
    }

    /// `u` raw rows and the host pre-pass over them. `window` is the conv's
    /// history (`conv_tokens`'s), carried from pass to pass.
    fn make_pass(
        w: &Weights,
        u: usize,
        r: &mut Rng,
        window: &mut [f32],
    ) -> Result<PassIn, GateError> {
        let mut rows = vec![0.0f32; u * LD];
        for row in rows.as_chunks_mut::<LD>().0 {
            for v in &mut row[..X0] {
                *v = r.uni(-3.0, 3.0);
            }
            for v in &mut row[X0..DT0] {
                *v = r.uni(-2.0, 2.0);
            }
            for v in &mut row[DT0..] {
                *v = r.uni(-0.8, 0.8);
            }
        }
        let x_new: Vec<f32> = rows
            .as_chunks::<LD>()
            .0
            .iter()
            .flat_map(|row| row[X0..DT0].iter().copied())
            .collect();
        let mut conv = vec![0.0f32; u * CONV_DIM];
        mamba2::conv_tokens(4, window, &x_new, &w.conv_w, &w.conv_b, &mut conv)?;
        for v in &mut conv {
            *v = silu32(*v);
        }
        let mut sd = vec![0.0f32; 2 * u * HEADS];
        let pairs_of = sd.as_chunks_mut::<{ 2 * HEADS }>().0;
        for (row, pairs) in rows.as_chunks::<LD>().0.iter().zip(pairs_of) {
            let heads = row[DT0..].iter().zip(&w.dt_bias).zip(&w.a);
            for (pair, ((&dt, &bias), &a)) in pairs.as_chunks_mut::<2>().0.iter_mut().zip(heads) {
                let v = dt + bias;
                let sigma = if v <= 20.0 { v.exp().ln_1p() } else { v };
                pair[0] = sigma;
                pair[1] = (sigma * a).exp();
            }
        }
        Ok(PassIn { u, rows, conv, sd })
    }

    struct Ctx {
        gpu: Gpu,
        steps: Mamba2Kernels,
        ssd: Mamba2Ssd,
        shape: Checked,
    }

    /// The card's copy of a layer's weights, and the engine pass's ring,
    /// position word and scratch.
    struct Layer {
        a: DeviceBuffer<f32>,
        d: DeviceBuffer<f32>,
        dt_bias: DeviceBuffer<f32>,
        conv_w: DeviceBuffer<f32>,
        conv_b: DeviceBuffer<f32>,
        ring: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        scratch: Mamba2PassScratch,
    }

    impl Layer {
        fn new(cx: &Ctx, w: &Weights) -> Result<Layer, GateError> {
            let st = cx.gpu.stream();
            Ok(Layer {
                a: DeviceBuffer::from_host(st, &w.a)?,
                d: DeviceBuffer::from_host(st, &w.d)?,
                dt_bias: DeviceBuffer::from_host(st, &w.dt_bias)?,
                conv_w: DeviceBuffer::from_host(st, &w.conv_w)?,
                conv_b: DeviceBuffer::from_host(st, &w.conv_b)?,
                ring: DeviceBuffer::from_host(st, &vec![GARBAGE; RING_ROWS * CONV_DIM])?,
                pos: DeviceBuffer::from_host(st, &[0u32])?,
                scratch: Mamba2PassScratch::new(st, cx.shape, MAX_U)?,
            })
        }
    }

    fn poisoned(cx: &Ctx, n: usize) -> Result<DeviceBuffer<f32>, GateError> {
        Ok(DeviceBuffer::from_host(
            cx.gpu.stream(),
            &vec![f32::from_bits(POISON); n],
        )?)
    }

    /// The SSD's inputs on the card.
    struct SsdIn {
        u: usize,
        rows: DeviceBuffer<f32>,
        conv: DeviceBuffer<f32>,
        sd: DeviceBuffer<f32>,
    }

    fn ssd_in(cx: &Ctx, pass: &PassIn) -> Result<SsdIn, GateError> {
        let st = cx.gpu.stream();
        Ok(SsdIn {
            u: pass.u,
            rows: DeviceBuffer::from_host(st, &pass.rows)?,
            conv: DeviceBuffer::from_host(st, &pass.conv)?,
            sd: DeviceBuffer::from_host(st, &pass.sd)?,
        })
    }

    fn ssd_args<'a>(
        cx: &Ctx,
        layer: &'a Layer,
        inp: &'a SsdIn,
        y: &'a mut DeviceBuffer<f32>,
        state: &'a mut DeviceBuffer<f32>,
    ) -> Result<SsdArgs<'a>, GpuError> {
        Ok(SsdArgs {
            shape: cx.shape,
            u: inp.u,
            conv: &inp.conv,
            sd: &inp.sd,
            rows: &inp.rows,
            ld: LD,
            z0: 0,
            a: &layer.a,
            d: &layer.d,
            fault: cx.gpu.layer_sink(LAYER)?,
            y,
            state,
        })
    }

    /// The SSD over `pass` from `state` (written in place): `y`.
    fn ssd_run(
        cx: &Ctx,
        layer: &Layer,
        pass: &PassIn,
        state: &mut DeviceBuffer<f32>,
    ) -> Result<Vec<f32>, GateError> {
        let inp = ssd_in(cx, pass)?;
        let mut y = poisoned(cx, pass.u * INNER)?;
        cx.ssd
            .enqueue(cx.gpu.stream(), ssd_args(cx, layer, &inp, &mut y, state)?)?;
        Ok(y.to_host_vec(cx.gpu.stream())?)
    }

    /// The engine's pass over `pass`'s raw rows at position `pos` from
    /// `state` (written in place), without the norm: `y`.
    fn steps_run(
        cx: &Ctx,
        layer: &mut Layer,
        pass: &PassIn,
        pos: u32,
        state: &mut DeviceBuffer<f32>,
    ) -> Result<Vec<f32>, GateError> {
        let st = cx.gpu.stream();
        let rows = DeviceBuffer::from_host(st, &pass.rows)?;
        let mut y = poisoned(cx, pass.u * INNER)?;
        layer.pos.copy_from_host(st, &[pos])?;
        cx.steps.enqueue_steps(
            st,
            StepsArgs {
                shape: cx.shape,
                u: pass.u,
                rows: &rows,
                ld: LD,
                z0: 0,
                x0: X0,
                dt0: DT0,
                dt_bias: &layer.dt_bias,
                a: &layer.a,
                d: &layer.d,
                conv_w: &layer.conv_w,
                conv_b: &layer.conv_b,
                norm: None,
                eps: 1e-5,
                pos: &layer.pos,
                pos_at: 0,
                fault: cx.gpu.layer_sink(LAYER)?,
                y: &mut y,
                ring: &mut layer.ring,
                state,
                scratch: &mut layer.scratch,
            },
        )?;
        Ok(y.to_host_vec(st)?)
    }

    fn clean(cx: &Ctx, what: &str) -> Result<bool, GateError> {
        let f = cx.gpu.take_fault()?;
        if let Some(f) = f {
            println!("{what}: the fault word held {f} FAIL");
        }
        Ok(f.is_none())
    }

    /// One head's twin state, channel-major (64 × 128): the exact state, its
    /// magnitude (every term's absolute value, decayed alike) and the bound
    /// `E` on the card's distance from the exact state.
    #[derive(Clone)]
    struct HeadState {
        s: Vec<f64>,
        abs: Vec<f64>,
        err: Vec<f64>,
    }

    /// One side's distance from the twin over a set of values.
    #[derive(Default, Clone)]
    struct Side {
        n: usize,
        over: usize,
        worst: f64,
        first: Option<String>,
        sq: f64,
        kappa: f64,
        mags: usize,
    }

    impl Side {
        /// `got` against `want` within `bound`; `mag` the value's magnitude
        /// (the diagnostics' denominator).
        fn add(&mut self, at: &dyn Fn() -> String, got: f32, want: f64, bound: f64, mag: f64) {
            let delta = (f64::from(got) - want).abs();
            self.n += 1;
            if !got.is_finite() || delta > bound {
                self.over += 1;
                if self.first.is_none() {
                    self.first = Some(format!(
                        "{} got {got:e} want {want:e} bound {bound:.3e}",
                        at()
                    ));
                }
            }
            if delta > 0.0 {
                self.worst = self.worst.max(if bound > 0.0 {
                    delta / bound
                } else {
                    f64::INFINITY
                });
            }
            if mag > 0.0 && delta.is_finite() {
                self.sq += (delta / mag).powi(2);
                self.kappa += bound / mag;
                self.mags += 1;
            }
        }

        fn merge(&mut self, o: &Side) {
            self.n += o.n;
            self.over += o.over;
            self.worst = self.worst.max(o.worst);
            if self.first.is_none() {
                self.first.clone_from(&o.first);
            }
            self.sq += o.sq;
            self.kappa += o.kappa;
            self.mags += o.mags;
        }

        fn line(&self, label: &str, judged: bool) -> String {
            let m = self.mags.max(1) as f64;
            let head = format!(
                "{label}: {} values, worst |Δ|/bound {:.3e}, σ {:.3e}, κ {:.3e}",
                self.n,
                self.worst,
                (self.sq / m).sqrt(),
                self.kappa / m
            );
            if !judged {
                return head;
            }
            match &self.first {
                Some(f) => format!("{head}, {} over, first {f} FAIL", self.over),
                None => format!("{head} {}", verdict(self.over == 0)),
            }
        }
    }

    /// What the twin compares against: each side's `y` and final state.
    struct Got<'g> {
        y_ssd: &'g [f32],
        y_steps: &'g [f32],
        s_ssd: &'g [f32],
        s_steps: &'g [f32],
    }

    /// One head's results: the SSD and the pass against the twin, `y` and
    /// the state; the SSD against the pass (diagnostic).
    #[derive(Default, Clone)]
    struct HeadStats {
        y_ssd: Side,
        y_steps: Side,
        s_ssd: Side,
        s_steps: Side,
        pair: Side,
    }

    fn silu64(z: f64) -> f64 {
        z / (1.0 + (-z).exp())
    }

    /// The twin of head `h` over `pass` from `st` (module doc): the exact
    /// scan token by token, its magnitude and the card's bound chunk by
    /// chunk, each `y` and the final state compared as they come.
    #[allow(
        clippy::needless_range_loop,
        reason = "the twin indexes a token, a key and a channel in one expression"
    )]
    fn twin_head(
        h: usize,
        w: &Weights,
        pass: &PassIn,
        mut st: HeadState,
        got: &Got<'_>,
    ) -> (HeadState, HeadStats) {
        let g = h / PER;
        let (a, d) = (f64::from(w.a[h]), f64::from(w.d[h]));
        let mut out = HeadStats::default();
        let mut k0 = 0;
        while k0 < pass.u {
            let l = CHUNK.min(pass.u - k0);
            let mut lc = [0.0f64; CHUNK];
            let (mut acc, mut a_c) = (0.0f64, 0.0f64);
            let mut xv = vec![[0.0f64; HEAD_DIM]; l];
            let mut xs = vec![[0.0f64; HEAD_DIM]; l];
            let mut b = vec![[0.0f64; STATE]; l];
            let mut c = vec![[0.0f64; STATE]; l];
            for i in 0..l {
                let sig = f64::from(pass.sd[2 * ((k0 + i) * HEADS + h)]);
                acc += sig * a;
                lc[i] = acc;
                a_c += (sig * a * LOG2_E).abs();
                let row = &pass.conv[(k0 + i) * CONV_DIM..(k0 + i + 1) * CONV_DIM];
                for p in 0..HEAD_DIM {
                    xv[i][p] = f64::from(row[h * HEAD_DIM + p]);
                    xs[i][p] = xv[i][p] * sig;
                }
                for n in 0..STATE {
                    b[i][n] = f64::from(row[INNER + g * STATE + n]);
                    c[i][n] = f64::from(row[INNER + GROUPS * STATE + g * STATE + n]);
                }
            }
            let dc = LN_2 * 16.0 * U * a_c + EX2;
            let sb: Vec<f64> = b.iter().map(|r| r.iter().map(|v| v.abs()).sum()).collect();
            let sc: Vec<f64> = c.iter().map(|r| r.iter().map(|v| v.abs()).sum()).collect();
            // Per (row, key) at or before it: the decay, the |C|·|B| sum,
            // and the chunk term's bound coefficients.
            let mut wa = vec![[0.0f64; CHUNK]; l];
            let mut coef = vec![[0.0f64; CHUNK]; l];
            let mut cons = vec![[0.0f64; CHUNK]; l];
            for i in 0..l {
                for j in 0..=i {
                    let gab: f64 = (0..STATE).map(|n| (c[i][n] * b[j][n]).abs()).sum();
                    let wij = (lc[i] - lc[j]).exp();
                    let off_g = (2.0 * E16 + 256.0 * U) * gab + ETA * (sc[i] + sb[j]);
                    wa[i][j] = wij * gab;
                    coef[i][j] = wij * (off_g + gab * (dc + 2.0 * E16 + 2.0 * U)) + ETA;
                    cons[i][j] = ETA * wij * gab;
                }
            }
            // The chunk start's magnitude bound `M = |S| + E`, and its sum
            // per channel.
            let m0: Vec<f64> = st.abs.iter().zip(&st.err).map(|(x, e)| x + e).collect();
            let rm0: Vec<f64> = m0
                .as_chunks::<STATE>()
                .0
                .iter()
                .map(|r| r.iter().sum())
                .collect();
            for i in 0..l {
                let t = k0 + i;
                let da = (f64::from(pass.sd[2 * (t * HEADS + h)]) * a).exp();
                let el = lc[i].exp();
                for p in 0..HEAD_DIM {
                    let (mut y, mut t1, mut t2) = (0.0f64, 0.0f64, 0.0f64);
                    let at = p * STATE;
                    for n in 0..STATE {
                        let s = da * st.s[at + n] + b[i][n] * xs[i][p];
                        st.s[at + n] = s;
                        y += c[i][n] * s;
                        let cn = c[i][n].abs();
                        t1 += cn * m0[at + n];
                        t2 += cn * st.err[at + n];
                    }
                    let st_abs = el * t1;
                    let st_off = el * (t2 + (2.0 * E16 + dc) * t1 + ETA * (sc[i] + rm0[p]));
                    let (mut in_abs, mut in_off) = (0.0f64, 0.0f64);
                    for j in 0..=i {
                        let x = xs[j][p].abs();
                        in_abs += wa[i][j] * x;
                        in_off += coef[i][j] * x + cons[i][j];
                    }
                    let mag_y = st_abs + in_abs;
                    let off_y = st_off + in_off + 2.0 * 162.0 * U * mag_y + FLOOR;
                    let z = f64::from(pass.rows[t * LD + h * HEAD_DIM + p]);
                    let sz = silu64(z);
                    let xd = (xv[i][p] * d).abs();
                    let want = sz * (y + xv[i][p] * d);
                    let mag = sz.abs() * (mag_y + xd);
                    let bound = SLACK
                        * sz.abs()
                        * (off_y + (SILU + 2.0 * U * z.abs() + 4.0 * U) * (mag_y + off_y + xd));
                    let col = h * HEAD_DIM + p;
                    let at_y = || format!("y[{t}][{col}]");
                    let (ys, yp) = (got.y_ssd[t * INNER + col], got.y_steps[t * INNER + col]);
                    out.y_ssd.add(&at_y, ys, want, bound, mag);
                    out.y_steps.add(&at_y, yp, want, bound, mag);
                    out.pair.add(&at_y, ys, f64::from(yp), 2.0 * bound, mag);
                }
            }
            // The state's magnitude and bound to the chunk end.
            let e_l = lc[l - 1].exp();
            let w_l: Vec<f64> = (0..l).map(|j| (lc[l - 1] - lc[j]).exp()).collect();
            let sbn: Vec<f64> = (0..STATE)
                .map(|n| (0..l).map(|j| b[j][n].abs()).sum())
                .collect();
            for p in 0..HEAD_DIM {
                let mut nn = [0.0f64; STATE];
                let mut sxp = 0.0;
                for j in 0..l {
                    let k = w_l[j] * xs[j][p].abs();
                    sxp += k;
                    for n in 0..STATE {
                        nn[n] += k * b[j][n].abs();
                    }
                }
                let at = p * STATE;
                for n in 0..STATE {
                    let m = m0[at + n];
                    st.err[at + n] = e_l * st.err[at + n]
                        + dc * e_l * m
                        + (2.0 * E16 + dc + 2.0 * U) * nn[n]
                        + ETA * (sbn[n] + sxp)
                        + 2.0 * 34.0 * U * (e_l * m + nn[n])
                        + FLOOR;
                    st.abs[at + n] = e_l * st.abs[at + n] + nn[n];
                }
            }
            k0 += CHUNK;
        }
        for p in 0..HEAD_DIM {
            for n in 0..STATE {
                let at = p * STATE + n;
                let col = (h * HEAD_DIM + p) * STATE + n;
                let at_s = || format!("state[{col}]");
                let bound = SLACK * st.err[at];
                out.s_ssd
                    .add(&at_s, got.s_ssd[col], st.s[at], bound, st.abs[at]);
                out.s_steps
                    .add(&at_s, got.s_steps[col], st.s[at], bound, st.abs[at]);
            }
        }
        (st, out)
    }

    /// Passes of `us` tokens in a row from one nonzero state: each held to
    /// the bound, the SSD and the pass each on their own state. Returns the
    /// verdict and, for a single pass, its SSD outputs (for clause 2).
    fn band_case(
        cx: &Ctx,
        label: &str,
        us: &[usize],
        seed: u64,
    ) -> Result<(bool, BandRun), GateError> {
        let w = weights(seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let s0: Vec<f32> = (0..INNER * STATE).map(|_| r.uni(-0.5, 0.5)).collect();
        let st = cx.gpu.stream();
        let mut layer = Layer::new(cx, &w)?;
        let mut st_ssd = DeviceBuffer::from_host(st, &s0)?;
        let mut st_steps = DeviceBuffer::from_host(st, &s0)?;
        let mut twin: Vec<HeadState> = s0
            .as_chunks::<{ HEAD_DIM * STATE }>()
            .0
            .iter()
            .map(|v| HeadState {
                s: v.iter().map(|&x| f64::from(x)).collect(),
                abs: v.iter().map(|&x| f64::from(x).abs()).collect(),
                err: vec![0.0; HEAD_DIM * STATE],
            })
            .collect();
        let mut window = vec![0.0f32; 3 * CONV_DIM];
        let (mut pos, mut ok) = (0u32, true);
        let mut last = BandRun::default();
        for (i, &u) in us.iter().enumerate() {
            let pass = make_pass(&w, u, &mut r, &mut window)?;
            let y_ssd = ssd_run(cx, &layer, &pass, &mut st_ssd)?;
            ok &= clean(cx, &format!("{label} pass {i} ssd"))?;
            let y_steps = steps_run(cx, &mut layer, &pass, pos, &mut st_steps)?;
            ok &= clean(cx, &format!("{label} pass {i} steps"))?;
            let s_ssd = st_ssd.to_host_vec(st)?;
            let s_steps = st_steps.to_host_vec(st)?;
            let got = Got {
                y_ssd: &y_ssd,
                y_steps: &y_steps,
                s_ssd: &s_ssd,
                s_steps: &s_steps,
            };
            let per_head = par_map(HEADS, |h| twin_head(h, &w, &pass, twin[h].clone(), &got));
            let mut all = HeadStats::default();
            twin = per_head
                .into_iter()
                .map(|(s, o)| {
                    all.y_ssd.merge(&o.y_ssd);
                    all.y_steps.merge(&o.y_steps);
                    all.s_ssd.merge(&o.s_ssd);
                    all.s_steps.merge(&o.s_steps);
                    all.pair.merge(&o.pair);
                    s
                })
                .collect();
            let pass_ok = all.y_ssd.over == 0
                && all.y_steps.over == 0
                && all.s_ssd.over == 0
                && all.s_steps.over == 0;
            let tag = format!("band {label} pass {i} (U = {u})");
            println!("{}", all.y_ssd.line(&format!("{tag} ssd y"), true));
            println!("{}", all.s_ssd.line(&format!("{tag} ssd state"), true));
            println!("{}", all.y_steps.line(&format!("{tag} pass y"), true));
            println!("{}", all.s_steps.line(&format!("{tag} pass state"), true));
            println!(
                "{}",
                all.pair.line(&format!("{tag} ssd against the pass"), false)
            );
            ok &= pass_ok;
            pos += u as u32;
            last = BandRun {
                pass: Some(pass),
                s0: s0.clone(),
                y: y_ssd,
                state: s_ssd,
                w: Some(w.clone()),
            };
        }
        Ok((ok, last))
    }

    /// The last pass of a band case and the SSD's outputs, for clause 2.
    #[derive(Default)]
    struct BandRun {
        pass: Option<PassIn>,
        s0: Vec<f32>,
        y: Vec<f32>,
        state: Vec<f32>,
        w: Option<Weights>,
    }

    fn check_band(cx: &Ctx) -> Result<bool, GateError> {
        let mut ok = true;
        let mut long = None;
        for (i, &u) in US.iter().enumerate() {
            let (pass, run) = band_case(cx, &format!("single {u}"), &[u], 0x55d0_0001 + i as u64)?;
            ok &= pass;
            if u == MAX_U {
                long = Some(run);
            }
        }
        let (pass, _) = band_case(cx, "two passes", &[97, 64], 0x55d0_0100)?;
        ok &= pass;
        println!("band: {}", verdict(ok));
        // Clause 2: the long pass again, from the same state.
        let det = match long {
            Some(BandRun {
                pass: Some(pass),
                s0,
                y,
                state,
                w: Some(w),
            }) => {
                let layer = Layer::new(cx, &w)?;
                let mut st = DeviceBuffer::from_host(cx.gpu.stream(), &s0)?;
                let y2 = ssd_run(cx, &layer, &pass, &mut st)?;
                let s2 = st.to_host_vec(cx.gpu.stream())?;
                clean(cx, "determinism")? && bits_equal(&y, &y2) && bits_equal(&state, &s2)
            }
            _ => false,
        };
        println!(
            "determinism: the {MAX_U}-token pass from the same state, y and state bit for bit {}",
            verdict(det)
        );
        Ok(ok && det)
    }

    /// Clause 3: one plant in a 97-token pass. `reach(t, col)` says whether
    /// the plant reaches `y[t][col]`; `from` is the plant's row.
    struct Plant {
        label: &'static str,
        pass: PassIn,
        from: usize,
        reach: fn(usize, usize) -> bool,
    }

    fn check_fault(cx: &Ctx) -> Result<bool, GateError> {
        let u = 97;
        let w = weights(0x0fa1_7001);
        let mut r = Rng(0x0fa1_7002);
        let s0: Vec<f32> = (0..INNER * STATE).map(|_| r.uni(-0.5, 0.5)).collect();
        let base = make_pass(&w, u, &mut r, &mut vec![0.0f32; 3 * CONV_DIM])?;
        let layer = Layer::new(cx, &w)?;
        let run = |pass: &PassIn| -> Result<Vec<f32>, GateError> {
            let mut st = DeviceBuffer::from_host(cx.gpu.stream(), &s0)?;
            ssd_run(cx, &layer, pass, &mut st)
        };
        let clean_y = run(&base)?;
        let mut ok = clean(cx, "fault base")?;
        println!(
            "fault base: the clean launch raised nothing {}",
            verdict(ok)
        );
        let want = Some(Fault::at(LAYER as u32, FaultSite::Mamba2Scan));
        let mut plants = Vec::new();
        // x of head 37, channel 5, at token 50: that channel.
        let mut p = base.clone();
        p.conv[50 * CONV_DIM + 37 * HEAD_DIM + 5] = f32::NAN;
        plants.push(Plant {
            label: "NaN in x",
            pass: p,
            from: 50,
            reach: |_, col| col == 37 * HEAD_DIM + 5,
        });
        // B of group 3, state 7, at token 70: every channel of its heads.
        let mut p = base.clone();
        p.conv[70 * CONV_DIM + INNER + 3 * STATE + 7] = f32::NAN;
        plants.push(Plant {
            label: "NaN in B",
            pass: p,
            from: 70,
            reach: |_, col| col / HEAD_DIM / PER == 3,
        });
        // z of head 15, channel 40, at token 80: that value alone.
        let mut p = base.clone();
        p.rows[80 * LD + 15 * HEAD_DIM + 40] = f32::INFINITY;
        plants.push(Plant {
            label: "+inf in z",
            pass: p,
            from: 80,
            reach: |t, col| t == 80 && col == 15 * HEAD_DIM + 40,
        });
        // σ of head 0 at token 40: every channel of head 0.
        let mut p = base;
        p.sd[2 * (40 * HEADS)] = f32::NAN;
        plants.push(Plant {
            label: "NaN sigma",
            pass: p,
            from: 40,
            reach: |_, col| col < HEAD_DIM,
        });
        for plant in &plants {
            let y = run(&plant.pass)?;
            let f = cx.gpu.take_fault()?;
            let chunk0 = plant.from / CHUNK * CHUNK;
            let (mut kept, mut lost) = (true, true);
            for t in 0..u {
                for col in 0..INNER {
                    let (v, c) = (y[t * INNER + col], clean_y[t * INNER + col]);
                    if t < chunk0 || !(plant.reach)(t, col) {
                        kept &= v.to_bits() == c.to_bits();
                    } else if t >= plant.from {
                        lost &= !v.is_finite();
                    }
                }
            }
            let pass = f == want && kept && lost;
            let shown = f.map_or("none".to_owned(), |f| f.to_string());
            println!(
                "fault {}: raised {shown}, earlier chunks and the unreached values the clean \
                 bits {kept}, the reached values non-finite from row {} {lost} {}",
                plant.label,
                plant.from,
                verdict(pass)
            );
            ok &= pass;
        }
        println!("fault: {}", verdict(ok));
        Ok(ok)
    }

    fn check_refuse(cx: &Ctx) -> Result<bool, GateError> {
        let u = 33;
        let w = weights(0x0ef0_0001);
        let mut r = Rng(0x0ef0_0002);
        let pass = make_pass(&w, u, &mut r, &mut vec![0.0f32; 3 * CONV_DIM])?;
        let layer = Layer::new(cx, &w)?;
        let st = cx.gpu.stream();
        let inp = ssd_in(cx, &pass)?;
        let mut y = poisoned(cx, u * INNER)?;
        let mut state = DeviceBuffer::from_host(st, &vec![0.0f32; INNER * STATE])?;
        let refused = |e: Result<(), GpuError>, want: &str| -> bool {
            matches!(&e, Err(GpuError::Shape { .. }))
                && e.err().is_some_and(|e| e.to_string().contains(want))
        };
        let mut named = Vec::new();
        let mut ok = true;
        // Shapes the host's check takes and the SSD refuses: another head
        // width (the 4B's), and a conv the card's check refuses.
        let hd80 = Geom {
            d_inner: 7680,
            n_heads: 96,
            ..GEOM
        };
        let shapes = [
            ("head_dim 80", hd80, "head_dim 80"),
            ("conv 3", Geom { d_conv: 3, ..GEOM }, "d_conv 3"),
        ];
        for (label, g, want) in shapes {
            let args = SsdArgs {
                shape: mamba2::check(g)?,
                ..ssd_args(cx, &layer, &inp, &mut y, &mut state)?
            };
            let pass = refused(cx.ssd.enqueue(st, args), want);
            named.push(format!("{label}={}", verdict(pass)));
            ok &= pass;
        }
        // A pass of no tokens, and a z window past the row stride.
        let e = cx.ssd.enqueue(
            st,
            SsdArgs {
                u: 0,
                ..ssd_args(cx, &layer, &inp, &mut y, &mut state)?
            },
        );
        let pass = refused(e, "u 0");
        named.push(format!("u 0={}", verdict(pass)));
        ok &= pass;
        let e = cx.ssd.enqueue(
            st,
            SsdArgs {
                z0: LD - INNER + 1,
                ..ssd_args(cx, &layer, &inp, &mut y, &mut state)?
            },
        );
        let pass = refused(e, "z window");
        named.push(format!("z window={}", verdict(pass)));
        ok &= pass;
        // Buffers one element short.
        let short = |n: usize| DeviceBuffer::from_host(st, &vec![0.0f32; n]);
        let conv_s = short(u * CONV_DIM - 1)?;
        let e = cx.ssd.enqueue(
            st,
            SsdArgs {
                conv: &conv_s,
                ..ssd_args(cx, &layer, &inp, &mut y, &mut state)?
            },
        );
        let pass = refused(e, "conv.len()");
        named.push(format!("short conv={}", verdict(pass)));
        ok &= pass;
        let sd_s = short(2 * u * HEADS - 1)?;
        let e = cx.ssd.enqueue(
            st,
            SsdArgs {
                sd: &sd_s,
                ..ssd_args(cx, &layer, &inp, &mut y, &mut state)?
            },
        );
        let pass = refused(e, "sd.len()");
        named.push(format!("short sd={}", verdict(pass)));
        ok &= pass;
        let mut y_s = short(u * INNER - 1)?;
        let e = cx
            .ssd
            .enqueue(st, ssd_args(cx, &layer, &inp, &mut y_s, &mut state)?);
        let pass = refused(e, "y.len()");
        named.push(format!("short y={}", verdict(pass)));
        ok &= pass;
        let mut state_s = short(INNER * STATE - 1)?;
        let e = cx
            .ssd
            .enqueue(st, ssd_args(cx, &layer, &inp, &mut y, &mut state_s)?);
        let pass = refused(e, "state.len()");
        named.push(format!("short state={}", verdict(pass)));
        ok &= pass;
        // Nothing launched.
        let untouched = y.to_host_vec(st)?.iter().all(|v| v.to_bits() == POISON);
        let quiet = clean(cx, "refuse")?;
        named.push(format!("nothing launched={}", verdict(untouched && quiet)));
        ok &= untouched && quiet;
        println!("refuse: {} {}", named.join(" "), verdict(ok));
        Ok(ok)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let steps = Mamba2Kernels::load(gpu.context())?;
        let ssd = Mamba2Ssd::load(gpu.context())?;
        let shape = bloomery_gpu::mamba2::check("gate_mamba2_ssd", GEOM)?;
        let cx = Ctx {
            gpu,
            steps,
            ssd,
            shape,
        };
        if let Some(f) = cx.gpu.take_fault()? {
            return Err(
                format!("gate_mamba2_ssd: the fault word held {f} before any launch").into(),
            );
        }
        let mut ok = true;
        ok &= check_band(&cx)?;
        ok &= check_fault(&cx)?;
        ok &= check_refuse(&cx)?;
        ok &= no_local_depot(&["mamba2_ssd"])?;
        if !ok {
            return Err(checks_failed());
        }
        println!(
            "PASSED: gate_mamba2_ssd the chunked SSD scan is within the derived f16 bound of the \
             f64 twin for every y and the final state (passes of 1, 31, 32, 33, 97 and 2048 \
             tokens, and 97 then 64 from the first's state; strong, weak and model-range decays), \
             and so is the engine's f32 pass on the same rows; the 2048-token pass repeats bit for \
             bit; a NaN x, a NaN B, an infinite z and a NaN sigma raise mamba2_scan, leave earlier \
             chunks and every unreached value the clean bits and the reached values non-finite; \
             head_dim 80, a conv of 3 taps, u 0, a z window past ld and a short conv, \
             sd, y and state refused by name with nothing launched; no local depot"
        );
        Ok(())
    }
}
