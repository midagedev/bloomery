//! GPU gate for the Mamba-2 decode step on the card (`bloomery_gpu::mamba2`):
//! one fused launch of the conv over the position-keyed ring, bias and SiLU,
//! the scan at state 128, `y + x·D`, the `z` gate and the grouped RMS norm,
//! against the host twin `model::mamba2::step_token`. Synthetic inputs at the
//! 4B widths (`d_inner` 7,680 as 96 heads of 80, 8 groups, state 128, 4 conv
//! taps, 9,728 conv channels), seeded, no model file. Every token starts from
//! a nonzero conv ring and scan state.
//!
//! 1. exact: heads whose `dt > 20` (so `σ = dt`, the identity arm) and whose
//!    decay is `exp(σ·A)` of `A = 0` (1) or of `σ·A < −104` (0): the device
//!    library's `exp` and `log1p` are never asked for a value that differs
//!    from libm's, so `y` of each of four tokens and the final scan state are
//!    the host twin's bit for bit, and so is the ring (the written slots hold
//!    the tokens' inputs, the others are untouched). This is the order's
//!    gate: the conv taps, bias, SiLU, the rounded products of the update,
//!    the four products per lane and the xor butterfly, `D`, the gate and the
//!    norm. Run with the gain, without it, and from the sequence start
//!    (positions 0, 1, 2, where the ring holds garbage that must not be read).
//! 2. band: realistic `dt`, `A` and `dt_bias` (some heads on each side of
//!    `dt = 20`), where `σ` and the decay come from the card's `log1pf` and
//!    `expf` and the host's libm. They are not the same function: CUDA's
//!    `expf` is within 2 ulp and `log1pf` within 1 (CUDA C++ Programming
//!    Guide, standard functions), glibc's within about 1 and 2 — so no
//!    equality holds, and `y` and the final state are held to the band below.
//!    The ring stays exact.
//! 3. refuse: state 96 and 256 and a group count that does not divide the
//!    heads are refused by name by `mamba2::check` (the host's text; enqueue
//!    calls it before a launch). A conv of 3 taps is a shape the host
//!    accepts and enqueue still refuses. A short buffer and a non-finite
//!    norm epsilon are refused by name at enqueue. Nothing launched (the
//!    output keeps its poison, the fault word stays clean).
//! 4. fault: a NaN in `x`, in `B`, an infinity in `z`, a non-finite `dt` and
//!    a non-finite `A` each raise `Mamba2Scan` at the launch's layer and
//!    leave the norm group they sit in entirely non-finite, never a
//!    plausible `y`; every other group is the clean launch's bits. Clean
//!    launches raise nothing.
//! 5. replay: the capture is one node. Replayed over three tokens whose
//!    inputs and position word are rewritten between replays, it equals the
//!    eager run bit for bit (`y` of every token, the ring, the state).
//! 6. shape: the entry compiles with no local depot.
//!
//! The band of clause 2. Write `u = 2^-24` and `ε = 2u = 2^-23`, one ulp at
//! most. Both sides are the same f32 program on inputs that differ only in
//! `σ = softplus(dt)` and `dA = exp(σ·A)`: the conv, SiLU, `x`, `B`, `C`,
//! `D`, `z` and `dt + dt_bias` are bit-equal (clause 1 holds them), so every
//! difference below is propagated from `σ` and `dA`, plus the roundings that
//! differ when an input moves.
//! - `σ = log1p(exp(dt))` through the chain: the inner `exp` (relative error
//!   `η`) moves `log1p` by `η·e/(1+e) ≤ η·σ`, so the relative error of `σ` is
//!   at most `η + ε_log1p`. Device `3ε`, host `3ε`: the sides differ by at
//!   most `κ_σ = 6ε` for `dt <= 20`, and by 0 for `dt > 20` (the identity).
//! - `dA = exp(fl(σ·A))`: the argument differs by `|σA|·(κ_σ + ε)` (the
//!   difference of the two sides' product roundings is at most `2u`), the
//!   exponentials add `2ε + ε`, so `κ_a = (1 + 3ε)·exp(|σA|·(κ_σ + ε)) − 1`.
//! - the update `s' = fl(fl(s·dA) + fl(B·fl(x·σ)))`: with `e` the absolute
//!   difference of the two sides' `s` and `P = |s|·dA + |B·x·σ|` (host values),
//!   `e' ≤ dA·(1 + κ_a + ε)·e + P·(κ_a + κ_σ + 4ε)`: the first term is the
//!   decayed difference (`dA ≤ 1`: `A ≤ 0`), then `s·dA`'s relative `κ_a` and
//!   `B·x·σ`'s `κ_σ`, and the four roundings of the update, each side's two
//!   at `u` (a difference of `2u` each). `e` starts at 0: both sides start
//!   from the same state. The final state of every element is held to `e`.
//! - `y = Σ s'·C` through four products, three adds and five butterfly adds
//!   per lane: the nine roundings of a path differ by `2·9u = 9ε` of `S =
//!   Σ|s'·C|`, and the inputs by `Σ|C|·e'`, so `δy = Σ|C|e'·(1 + 9ε) + 9ε·S`.
//! - `y + x·D` rounds once more on each side, `δ₁ = δy + ε·|y₁|`; `silu(z)·y₁`
//!   likewise with `silu(z)` equal on both sides, `δ₂ = |silu(z)|·δ₁ +
//!   ε·|y₂|`.
//! - the group norm over `width` values: `mean = Σ fl(y₂²)/width` moves by at
//!   most `Δm = 2·mean(|y₂|·δ₂) + mean(δ₂²) + 2ε·mean(y₂²)` (the last term
//!   the squares' and the narrowing's roundings), `scale = 1/√(m + eps)` by
//!   `Δm/(2(m + eps))` relative plus its three roundings per side `3ε`, and
//!   each output `o = fl(fl(scale·gain)·y₂)` by `δo = |o|·(Δscale/scale + 2ε)
//!   + scale·|gain|·δ₂`.
//!
//! Terms of order `ε²` are dropped, below 10⁻⁴ of every band (`ε ≈ 1.2·10⁻⁷`,
//! the largest `κ_a` is under 10⁻⁴); each band is multiplied by 1 + 10⁻⁴.
//! Without a gain the band is `δ₂`. The libraries' ulp bounds are the
//! assumptions; the gate prints the largest `|Δ|/band` it saw.

#[cfg(feature = "gpu")]
fn main() -> std::process::ExitCode {
    bloomery_gpu_gates::exit_with("gate_mamba2", gate::run())
}

#[cfg(feature = "gpu")]
mod gate {
    use bloomery_gpu::linear::RING_ROWS;
    use bloomery_gpu::mamba2::{Mamba2Kernels, Mamba2Scratch, STATE, StepArgs};
    use bloomery_gpu::{Fault, FaultSite, Gpu, GpuError};
    use bloomery_gpu_gates::rounding::U;
    use bloomery_gpu_gates::{GateError, bits_equal, no_local_depot, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};
    use model::mamba2::{self, Checked, Geom, Token};

    const INNER: usize = 7680;
    const HEADS: usize = 96;
    const GROUPS: usize = 8;
    const HEAD_DIM: usize = INNER / HEADS;
    const PER: usize = HEADS / GROUPS;
    const WIDTH: usize = INNER / GROUPS;
    const CONV_DIM: usize = INNER + 2 * GROUPS * STATE;
    const GEOM: Geom = Geom {
        d_inner: INNER,
        n_heads: HEADS,
        n_groups: GROUPS,
        d_state: STATE,
        d_conv: 4,
    };
    const EPS: f32 = 1e-5;
    /// The layer the launches raise on.
    const LAYER: usize = 7;
    /// The first position of every run: the ring wraps inside four tokens
    /// (slots 9, 10, 0, 1) and the conv reads slots 6, 7, 8 first.
    const P0: u32 = 9;
    /// Tokens of a run.
    const TOKENS: usize = 4;
    /// What the ring holds where nothing was put: a value the conv would
    /// show if it read it.
    const GARBAGE: f32 = 7.0;
    /// What the output holds before a launch.
    const POISON: u32 = 0x7fa5_a5a5;

    /// One ulp at most, relative: 2^-23 (module doc).
    const EPS_F: f64 = 2.0 * U;
    /// The device library's `expf` and `log1pf` ulps, and the host's libm's.
    const DEV_EXP: f64 = 2.0;
    const DEV_LOG1P: f64 = 1.0;
    const HOST_EXP: f64 = 1.0;
    const HOST_LOG1P: f64 = 2.0;
    /// The dropped second-order terms (module doc).
    const SECOND_ORDER: f64 = 1.0 + 1e-4;

    /// A fixed xorshift64 stream.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        /// Uniform in `[lo, hi)`.
        fn uni(&mut self, lo: f32, hi: f32) -> f32 {
            let u = (self.next() >> 40) as f32 / (1u64 << 24) as f32;
            lo + (hi - lo) * u
        }
    }

    /// Which inputs the weights and tokens come from (module doc, clauses 1
    /// and 2).
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Regime {
        Exact,
        Band,
    }

    struct Weights {
        dt_bias: Vec<f32>,
        a: Vec<f32>,
        d: Vec<f32>,
        conv_w: Vec<f32>,
        conv_b: Vec<f32>,
        norm: Vec<f32>,
    }

    fn weights(regime: Regime, seed: u64) -> Weights {
        let mut r = Rng(seed);
        let mut dt_bias = vec![0.0; HEADS];
        let mut a = vec![0.0; HEADS];
        for h in 0..HEADS {
            match regime {
                // dt = raw + bias is in (21, 27): the identity arm. Even
                // heads decay by 1, odd ones by 0 (σ·A < −168).
                Regime::Exact => {
                    dt_bias[h] = r.uni(22.0, 26.0);
                    a[h] = if h % 2 == 0 { 0.0 } else { -8.0 };
                }
                // A in [−2.7, −0.22]; every sixth head sits around the
                // 20 where the two arms meet.
                Regime::Band => {
                    dt_bias[h] = if h % 6 == 5 {
                        r.uni(21.0, 22.0)
                    } else {
                        r.uni(-1.5, 0.5)
                    };
                    a[h] = -r.uni(-1.5, 1.0).exp();
                }
            }
        }
        Weights {
            dt_bias,
            a,
            d: (0..HEADS).map(|_| r.uni(0.5, 1.5)).collect(),
            conv_w: (0..4 * CONV_DIM).map(|_| r.uni(-0.6, 0.6)).collect(),
            conv_b: (0..CONV_DIM).map(|_| r.uni(-0.2, 0.2)).collect(),
            norm: (0..INNER).map(|_| r.uni(0.5, 1.5)).collect(),
        }
    }

    /// One token's inputs.
    #[derive(Clone)]
    struct TokenIn {
        xbc: Vec<f32>,
        dt: Vec<f32>,
        z: Vec<f32>,
    }

    fn token(regime: Regime, r: &mut Rng) -> TokenIn {
        let dt_range = match regime {
            Regime::Exact => (-1.0, 1.0),
            Regime::Band => (-2.0, 2.0),
        };
        TokenIn {
            xbc: (0..CONV_DIM).map(|_| r.uni(-2.0, 2.0)).collect(),
            dt: (0..HEADS).map(|_| r.uni(dt_range.0, dt_range.1)).collect(),
            z: (0..INNER).map(|_| r.uni(-3.0, 3.0)).collect(),
        }
    }

    /// The state a run starts from: the host's conv window (`age + 3·channel`,
    /// age 0 the oldest) and the scan state.
    #[derive(Clone)]
    struct Init {
        conv: Vec<f32>,
        scan: Vec<f32>,
    }

    fn init(r: &mut Rng, start: bool) -> Init {
        Init {
            conv: if start {
                vec![0.0; 3 * CONV_DIM]
            } else {
                (0..3 * CONV_DIM).map(|_| r.uni(-2.0, 2.0)).collect()
            },
            scan: (0..INNER * STATE).map(|_| r.uni(-0.5, 0.5)).collect(),
        }
    }

    /// The ring a run starts from: garbage everywhere, then the three
    /// predecessors of position `p0` where the conv reads them.
    fn ring_of(init: &Init, p0: u32, start: bool) -> Vec<f32> {
        let mut ring = vec![GARBAGE; RING_ROWS * CONV_DIM];
        if !start {
            for age in 0..3 {
                let slot = (p0 as usize + RING_ROWS - 3 + age) % RING_ROWS;
                for c in 0..CONV_DIM {
                    ring[slot * CONV_DIM + c] = init.conv[age + 3 * c];
                }
            }
        }
        ring
    }

    /// The card's checked widths. The host twin uses the same geometry.
    fn device_shape() -> Result<Checked, GateError> {
        Ok(bloomery_gpu::mamba2::check("gate_mamba2", GEOM)?)
    }

    fn geom() -> Result<Checked, GateError> {
        Ok(mamba2::check(GEOM)?)
    }

    /// The host twin's run of `toks` from `init`: each token's `y`, the
    /// final window and scan state.
    struct HostRun {
        y: Vec<Vec<f32>>,
        conv: Vec<f32>,
        scan: Vec<f32>,
    }

    fn host_run(
        w: &Weights,
        init: &Init,
        toks: &[TokenIn],
        norm: bool,
    ) -> Result<HostRun, GateError> {
        let g = geom()?;
        let (mut conv, mut scan) = (init.conv.clone(), init.scan.clone());
        let mut ys = Vec::new();
        for t in toks {
            let mut y = vec![0.0; INNER];
            mamba2::step_token(
                &g,
                &mut conv,
                &mut scan,
                &Token {
                    xbc: &t.xbc,
                    dt: &t.dt,
                    dt_bias: &w.dt_bias,
                    z: &t.z,
                    a: &w.a,
                    d: &w.d,
                    conv_w: &w.conv_w,
                    conv_bias: &w.conv_b,
                    norm: norm.then_some(&w.norm[..]),
                    eps: EPS,
                },
                &mut y,
            )?;
            ys.push(y);
        }
        Ok(HostRun { y: ys, conv, scan })
    }

    /// The ring after `toks` from positions `p0`: the start's ring with each
    /// token's input in its slot.
    fn ring_after(start_ring: &[f32], toks: &[TokenIn], p0: u32) -> Vec<f32> {
        let mut ring = start_ring.to_vec();
        for (t, tok) in toks.iter().enumerate() {
            let slot = (p0 as usize + t) % RING_ROWS;
            ring[slot * CONV_DIM..(slot + 1) * CONV_DIM].copy_from_slice(&tok.xbc);
        }
        ring
    }

    struct Ctx {
        gpu: Gpu,
        k: Mamba2Kernels,
    }

    /// The card's buffers of one run.
    struct Dev {
        xbc: DeviceBuffer<f32>,
        dt: DeviceBuffer<f32>,
        z: DeviceBuffer<f32>,
        dt_bias: DeviceBuffer<f32>,
        a: DeviceBuffer<f32>,
        d: DeviceBuffer<f32>,
        conv_w: DeviceBuffer<f32>,
        conv_b: DeviceBuffer<f32>,
        norm: DeviceBuffer<f32>,
        pos: DeviceBuffer<u32>,
        y: DeviceBuffer<f32>,
        ring: DeviceBuffer<f32>,
        state: DeviceBuffer<f32>,
        scratch: Mamba2Scratch,
    }

    impl Dev {
        fn new(cx: &Ctx, w: &Weights) -> Result<Dev, GateError> {
            let st = cx.gpu.stream();
            Ok(Dev {
                xbc: DeviceBuffer::from_host(st, &vec![0.0; CONV_DIM])?,
                dt: DeviceBuffer::from_host(st, &vec![0.0; HEADS])?,
                z: DeviceBuffer::from_host(st, &vec![0.0; INNER])?,
                dt_bias: DeviceBuffer::from_host(st, &w.dt_bias)?,
                a: DeviceBuffer::from_host(st, &w.a)?,
                d: DeviceBuffer::from_host(st, &w.d)?,
                conv_w: DeviceBuffer::from_host(st, &w.conv_w)?,
                conv_b: DeviceBuffer::from_host(st, &w.conv_b)?,
                norm: DeviceBuffer::from_host(st, &w.norm)?,
                pos: DeviceBuffer::from_host(st, &[0u32])?,
                y: DeviceBuffer::from_host(st, &vec![f32::from_bits(POISON); INNER])?,
                ring: DeviceBuffer::from_host(st, &vec![0.0; RING_ROWS * CONV_DIM])?,
                state: DeviceBuffer::from_host(st, &vec![0.0; INNER * STATE])?,
                scratch: Mamba2Scratch::new(st, device_shape()?)?,
            })
        }

        /// The run's starting state, and the output poisoned.
        fn load(&mut self, cx: &Ctx, ring: &[f32], scan: &[f32]) -> Result<(), GateError> {
            let st = cx.gpu.stream();
            self.ring.copy_from_host(st, ring)?;
            self.state.copy_from_host(st, scan)?;
            self.y
                .copy_from_host(st, &vec![f32::from_bits(POISON); INNER])?;
            Ok(())
        }

        /// One token's inputs and its position word.
        fn put(&mut self, cx: &Ctx, t: &TokenIn, p: u32) -> Result<(), GateError> {
            let st = cx.gpu.stream();
            self.xbc.copy_from_host(st, &t.xbc)?;
            self.dt.copy_from_host(st, &t.dt)?;
            self.z.copy_from_host(st, &t.z)?;
            self.pos.copy_from_host(st, &[p])?;
            Ok(())
        }

        /// The launch, on `stream` (the engine's, or a capture's).
        fn step(
            &mut self,
            cx: &Ctx,
            stream: &CudaStream,
            shape: Checked,
            norm: bool,
        ) -> Result<(), GpuError> {
            cx.k.enqueue_step(
                stream,
                StepArgs {
                    shape,
                    xbc: &self.xbc,
                    dt: &self.dt,
                    dt_bias: &self.dt_bias,
                    z: &self.z,
                    a: &self.a,
                    d: &self.d,
                    conv_w: &self.conv_w,
                    conv_b: &self.conv_b,
                    norm: norm.then_some(&self.norm),
                    eps: EPS,
                    pos: &self.pos,
                    pos_at: 0,
                    fault: cx.gpu.layer_sink(LAYER)?,
                    y: &mut self.y,
                    ring: &mut self.ring,
                    state: &mut self.state,
                    scratch: &mut self.scratch,
                },
            )
        }

        fn read_y(&self, cx: &Ctx) -> Result<Vec<f32>, GateError> {
            Ok(self.y.to_host_vec(cx.gpu.stream())?)
        }
    }

    /// The card's run of `toks` from `init` at positions `P0 ..`: each
    /// token's `y`, the final ring and scan state.
    struct DevRun {
        y: Vec<Vec<f32>>,
        ring: Vec<f32>,
        scan: Vec<f32>,
    }

    fn dev_run(
        cx: &Ctx,
        dev: &mut Dev,
        ring: &[f32],
        init: &Init,
        toks: &[TokenIn],
        p0: u32,
        norm: bool,
    ) -> Result<DevRun, GateError> {
        dev.load(cx, ring, &init.scan)?;
        let shape = device_shape()?;
        let mut ys = Vec::new();
        for (t, tok) in toks.iter().enumerate() {
            dev.put(cx, tok, p0 + t as u32)?;
            dev.step(cx, cx.gpu.stream(), shape, norm)?;
            ys.push(dev.read_y(cx)?);
        }
        Ok(DevRun {
            y: ys,
            ring: dev.ring.to_host_vec(cx.gpu.stream())?,
            scan: dev.state.to_host_vec(cx.gpu.stream())?,
        })
    }

    fn clean(cx: &Ctx, what: &str) -> Result<bool, GateError> {
        let f = cx.gpu.take_fault()?;
        if let Some(f) = f {
            println!("{what}: the fault word held {f} FAIL");
        }
        Ok(f.is_none())
    }

    /// The bits that differ between two vectors: how many, and the first.
    fn diff(a: &[f32], b: &[f32]) -> (usize, Option<(usize, f32, f32)>) {
        let mut n = 0;
        let mut first = None;
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if x.to_bits() != y.to_bits() {
                n += 1;
                first.get_or_insert((i, *x, *y));
            }
        }
        (n, first)
    }

    // ---------------------------------------------------------------- clause 1

    /// One exact case (module doc, clause 1): bit equality of every token's
    /// `y`, the final state and the ring.
    fn exact_case(
        cx: &Ctx,
        label: &str,
        seed: u64,
        norm: bool,
        start: bool,
    ) -> Result<bool, GateError> {
        let w = weights(Regime::Exact, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(&mut r, start);
        let toks: Vec<TokenIn> = (0..TOKENS).map(|_| token(Regime::Exact, &mut r)).collect();
        let p0 = if start { 0 } else { P0 };
        let ring0 = ring_of(&st, p0, start);
        let want = host_run(&w, &st, &toks, norm)?;
        let mut dev = Dev::new(cx, &w)?;
        let got = dev_run(cx, &mut dev, &ring0, &st, &toks, p0, norm)?;
        let mut ok = true;
        let mut ys = Vec::new();
        for (t, (a, b)) in got.y.iter().zip(&want.y).enumerate() {
            let (n, first) = diff(a, b);
            ys.push(format!("{n}"));
            if n != 0 {
                println!("exact {label}: token {t} y first difference {first:?}");
            }
            ok &= n == 0;
        }
        let (ns, first) = diff(&got.scan, &want.scan);
        if ns != 0 {
            println!("exact {label}: state first difference {first:?}");
        }
        let ring_want = ring_after(&ring0, &toks, p0);
        let ring_ok = bits_equal(&got.ring, &ring_want);
        // The host's final window is the last three inputs: the ring's
        // slots of the last three positions.
        let mut window_ok = true;
        for age in 0..3 {
            let slot = (p0 as usize + TOKENS - 3 + age) % RING_ROWS;
            for c in 0..CONV_DIM {
                window_ok &=
                    got.ring[slot * CONV_DIM + c].to_bits() == want.conv[age + 3 * c].to_bits();
            }
        }
        ok &= ns == 0 && ring_ok && window_ok && clean(cx, label)?;
        println!(
            "exact {label}: y differing values per token [{}], state differing {ns}, ring exact \
             {ring_ok}, window exact {window_ok} {}",
            ys.join(","),
            verdict(ok)
        );
        Ok(ok)
    }

    fn check_exact(cx: &Ctx) -> Result<bool, GateError> {
        let mut ok = true;
        ok &= exact_case(cx, "gain", 0x0a11_ce01, true, false)?;
        ok &= exact_case(cx, "no gain", 0x0a11_ce02, false, false)?;
        ok &= exact_case(cx, "sequence start", 0x0a11_ce03, true, true)?;
        Ok(ok)
    }

    // ---------------------------------------------------------------- clause 2

    /// What the band needs of one host token: the values the bound is a
    /// function of (module doc).
    struct Trace {
        s_before: Vec<f32>,
        s_after: Vec<f32>,
        x: Vec<f32>,
        bc: Vec<f32>,
        dt: Vec<f32>,
        sigma: Vec<f32>,
        decay: Vec<f32>,
        y1: Vec<f32>,
        y2: Vec<f32>,
        sz: Vec<f32>,
        out: Vec<f32>,
    }

    /// The host token stage by stage, the composition `step_token` is, with
    /// the intermediates kept. Its `y` must be `step_token`'s.
    fn trace_run(
        w: &Weights,
        init: &Init,
        toks: &[TokenIn],
        norm: bool,
    ) -> Result<Vec<Trace>, GateError> {
        let g = geom()?;
        let (mut conv, mut scan) = (init.conv.clone(), init.scan.clone());
        let mut out = Vec::new();
        for t in toks {
            let mut conv_y = vec![0.0; CONV_DIM];
            mamba2::conv_tokens(4, &mut conv, &t.xbc, &w.conv_w, &w.conv_b, &mut conv_y)?;
            let ones = vec![1.0; CONV_DIM];
            let mut mixed = vec![0.0; CONV_DIM];
            qdot::swiglu(&conv_y, &ones, &mut mixed);
            let x = mixed[..INNER].to_vec();
            let bc = mixed[INNER..].to_vec();
            let dt: Vec<f32> = t.dt.iter().zip(&w.dt_bias).map(|(a, b)| a + b).collect();
            let sigma: Vec<f32> = dt
                .iter()
                .map(|&v| if v <= 20.0 { v.exp().ln_1p() } else { v })
                .collect();
            let decay: Vec<f32> = sigma.iter().zip(&w.a).map(|(s, a)| (s * a).exp()).collect();
            let s_before = scan.clone();
            let mut y = vec![0.0; INNER];
            mamba2::scan_token(
                &g,
                &mut scan,
                &x,
                &dt,
                &w.a,
                &bc[..GROUPS * STATE],
                &bc[GROUPS * STATE..],
                &mut y,
            )?;
            mamba2::skip_d(&mut y, &x, &w.d, HEAD_DIM)?;
            let y1 = y.clone();
            mamba2::gate_z(&mut y, &t.z)?;
            let y2 = y.clone();
            let mut sz = vec![0.0; INNER];
            qdot::swiglu(&t.z, &vec![1.0; INNER], &mut sz);
            if norm {
                mamba2::grouped_rms(&mut y, &w.norm, GROUPS, EPS)?;
            }
            out.push(Trace {
                s_before,
                s_after: scan.clone(),
                x,
                bc,
                dt,
                sigma,
                decay,
                y1,
                y2,
                sz,
                out: y,
            });
        }
        Ok(out)
    }

    /// The bands (module doc): per token the bound on each output value, and
    /// the bound on each final state element.
    fn bands(w: &Weights, trace: &[Trace], norm: bool) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut e = vec![0.0f64; INNER * STATE];
        let mut per_token = Vec::new();
        for tr in trace {
            let mut dy = vec![0.0f64; INNER];
            for h in 0..HEADS {
                let dec = f64::from(tr.decay[h]);
                let sig = f64::from(tr.sigma[h]);
                let k_sigma = if tr.dt[h] <= 20.0 {
                    (DEV_EXP + DEV_LOG1P + HOST_EXP + HOST_LOG1P) * EPS_F
                } else {
                    0.0
                };
                let arg = (sig * f64::from(w.a[h])).abs() * (k_sigma + EPS_F);
                let k_a = (1.0 + (DEV_EXP + HOST_EXP) * EPS_F) * arg.exp() - 1.0;
                let step = k_a + k_sigma + 4.0 * EPS_F;
                let g = h / PER;
                let bn = &tr.bc[g * STATE..(g + 1) * STATE];
                let cn = &tr.bc[GROUPS * STATE + g * STATE..GROUPS * STATE + (g + 1) * STATE];
                for ch in 0..HEAD_DIM {
                    let i = h * HEAD_DIM + ch;
                    let xs = f64::from(tr.x[i]).abs() * sig;
                    let (mut sum_ce, mut s_abs) = (0.0f64, 0.0f64);
                    for n in 0..STATE {
                        let at = i * STATE + n;
                        let p =
                            f64::from(tr.s_before[at]).abs() * dec + f64::from(bn[n]).abs() * xs;
                        e[at] = dec * (1.0 + k_a + EPS_F) * e[at] + p * step;
                        sum_ce += f64::from(cn[n]).abs() * e[at];
                        s_abs += (f64::from(tr.s_after[at]) * f64::from(cn[n])).abs();
                    }
                    dy[i] = sum_ce * (1.0 + 9.0 * EPS_F) + 9.0 * EPS_F * s_abs;
                }
            }
            let mut band = vec![0.0f64; INNER];
            let mut d2 = vec![0.0f64; INNER];
            for i in 0..INNER {
                let d1 = dy[i] + EPS_F * f64::from(tr.y1[i]).abs();
                d2[i] = f64::from(tr.sz[i]).abs() * d1 + EPS_F * f64::from(tr.y2[i]).abs();
            }
            if norm {
                for g in 0..GROUPS {
                    let r = g * WIDTH..(g + 1) * WIDTH;
                    let m = r
                        .clone()
                        .map(|i| f64::from(tr.y2[i] * tr.y2[i]))
                        .sum::<f64>()
                        / WIDTH as f64;
                    let cross = r
                        .clone()
                        .map(|i| f64::from(tr.y2[i]).abs() * d2[i])
                        .sum::<f64>()
                        / WIDTH as f64;
                    let sq = r.clone().map(|i| d2[i] * d2[i]).sum::<f64>() / WIDTH as f64;
                    let dm = 2.0 * cross + sq + 2.0 * EPS_F * m;
                    let scale = 1.0 / (m + f64::from(EPS)).sqrt();
                    let ds = dm / (2.0 * (m + f64::from(EPS))) + 3.0 * EPS_F;
                    for i in r {
                        let o = f64::from(tr.out[i]).abs();
                        band[i] =
                            o * (ds + 2.0 * EPS_F) + scale * f64::from(w.norm[i]).abs() * d2[i];
                    }
                }
            } else {
                band.copy_from_slice(&d2);
            }
            for b in &mut band {
                *b *= SECOND_ORDER;
            }
            per_token.push(band);
        }
        for v in &mut e {
            *v *= SECOND_ORDER;
        }
        (per_token, e)
    }

    /// The largest `|got − want| / band` over `got`, and how many values
    /// passed the band (a non-finite value never does).
    fn within(got: &[f32], want: &[f32], band: &[f64]) -> (usize, f64) {
        let mut bad = 0;
        let mut worst = 0.0f64;
        for ((g, w), b) in got.iter().zip(want).zip(band) {
            let d = (f64::from(*g) - f64::from(*w)).abs();
            if !d.is_finite() || d > *b {
                bad += 1;
            }
            if *b > 0.0 && d.is_finite() {
                worst = worst.max(d / b);
            }
        }
        (bad, worst)
    }

    fn check_band(cx: &Ctx) -> Result<bool, GateError> {
        let seed = 0x0ba2_d001;
        let w = weights(Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(&mut r, false);
        let toks: Vec<TokenIn> = (0..TOKENS).map(|_| token(Regime::Band, &mut r)).collect();
        let ring0 = ring_of(&st, P0, false);
        let want = host_run(&w, &st, &toks, true)?;
        let trace = trace_run(&w, &st, &toks, true)?;
        // The referee's composition is step_token.
        let mut referee = true;
        for (tr, y) in trace.iter().zip(&want.y) {
            referee &= bits_equal(&tr.out, y);
        }
        referee &= bits_equal(&trace[TOKENS - 1].s_after, &want.scan);
        let (y_bands, s_band) = bands(&w, &trace, true);
        let mut dev = Dev::new(cx, &w)?;
        let got = dev_run(cx, &mut dev, &ring0, &st, &toks, P0, true)?;
        let mut ok = referee;
        let mut y_bad = 0;
        let mut y_worst = 0.0f64;
        for t in 0..TOKENS {
            let (bad, worst) = within(&got.y[t], &want.y[t], &y_bands[t]);
            y_bad += bad;
            y_worst = y_worst.max(worst);
        }
        let (s_bad, s_worst) = within(&got.scan, &want.scan, &s_band);
        let ring_ok = bits_equal(&got.ring, &ring_after(&ring0, &toks, P0));
        ok &= y_bad == 0 && s_bad == 0 && ring_ok && clean(cx, "band")?;
        println!(
            "band: referee is step_token {referee}; y outside the band {y_bad} (largest |Δ|/band \
             {y_worst:.3}); state outside the band {s_bad} (largest |Δ|/band {s_worst:.3}); ring \
             exact {ring_ok} {}",
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------------------------- clause 3

    fn check_refuse(cx: &Ctx) -> Result<bool, GateError> {
        let w = weights(Regime::Band, 0x0ef0_0001);
        let mut dev = Dev::new(cx, &w)?;
        let mut r = Rng(0x0ef0_0002);
        let st = init(&mut r, false);
        dev.load(cx, &ring_of(&st, P0, false), &st.scan)?;
        let mut ok = true;
        let cases = [
            (
                "state 96",
                Geom {
                    d_state: 96,
                    ..GEOM
                },
                "state 96",
            ),
            (
                "state 256",
                Geom {
                    d_state: 256,
                    ..GEOM
                },
                "state 256",
            ),
            (
                "groups 7",
                Geom {
                    n_groups: 7,
                    ..GEOM
                },
                "n_group 7",
            ),
        ];
        let mut named = Vec::new();
        for (label, g, want) in cases {
            let e = bloomery_gpu::mamba2::check("mamba2::enqueue_step", g).err();
            let text = e.as_ref().map(ToString::to_string).unwrap_or_default();
            let pass = matches!(e, Some(GpuError::Shape { .. })) && text.contains(want);
            named.push(format!("{label}={}", verdict(pass)));
            ok &= pass;
        }
        // The host accepts three taps. Enqueue still refuses them, and
        // launches nothing.
        let conv3 = match mamba2::check(Geom { d_conv: 3, ..GEOM }) {
            Ok(host) => {
                let e = dev.step(cx, cx.gpu.stream(), host, true).err();
                let text = e.as_ref().map(ToString::to_string).unwrap_or_default();
                matches!(e, Some(GpuError::Shape { .. })) && text.contains("d_conv 3")
            }
            Err(_) => false,
        };
        named.push(format!("conv 3={}", verdict(conv3)));
        ok &= conv3;
        // A buffer shorter than the shape's: the state of half the channels.
        let short = {
            let mut half =
                DeviceBuffer::from_host(cx.gpu.stream(), &vec![0.0f32; INNER * STATE / 2])?;
            let e =
                cx.k.enqueue_step(
                    cx.gpu.stream(),
                    StepArgs {
                        shape: device_shape()?,
                        xbc: &dev.xbc,
                        dt: &dev.dt,
                        dt_bias: &dev.dt_bias,
                        z: &dev.z,
                        a: &dev.a,
                        d: &dev.d,
                        conv_w: &dev.conv_w,
                        conv_b: &dev.conv_b,
                        norm: Some(&dev.norm),
                        eps: EPS,
                        pos: &dev.pos,
                        pos_at: 0,
                        fault: cx.gpu.layer_sink(LAYER)?,
                        y: &mut dev.y,
                        ring: &mut dev.ring,
                        state: &mut half,
                        scratch: &mut dev.scratch,
                    },
                )
                .err();
            matches!(&e, Some(GpuError::Shape { detail, .. }) if detail.contains("state.len()"))
        };
        named.push(format!("short state={}", verdict(short)));
        ok &= short;
        // +∞ epsilon would make the norm scale 0. Refused before a launch.
        let eps_inf = {
            let e =
                cx.k.enqueue_step(
                    cx.gpu.stream(),
                    StepArgs {
                        shape: device_shape()?,
                        xbc: &dev.xbc,
                        dt: &dev.dt,
                        dt_bias: &dev.dt_bias,
                        z: &dev.z,
                        a: &dev.a,
                        d: &dev.d,
                        conv_w: &dev.conv_w,
                        conv_b: &dev.conv_b,
                        norm: Some(&dev.norm),
                        eps: f32::INFINITY,
                        pos: &dev.pos,
                        pos_at: 0,
                        fault: cx.gpu.layer_sink(LAYER)?,
                        y: &mut dev.y,
                        ring: &mut dev.ring,
                        state: &mut dev.state,
                        scratch: &mut dev.scratch,
                    },
                )
                .err();
            let text = e.as_ref().map(ToString::to_string).unwrap_or_default();
            matches!(e, Some(GpuError::Shape { .. })) && text.contains("norm eps must be finite")
        };
        named.push(format!("eps inf={}", verdict(eps_inf)));
        ok &= eps_inf;
        let y = dev.read_y(cx)?;
        let untouched = y.iter().all(|v| v.to_bits() == POISON);
        ok &= untouched && clean(cx, "refuse")?;
        println!(
            "refuse: {}; nothing launched (output keeps its poison) {untouched} {}",
            named.join(" "),
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------------------------- clause 4

    /// Every value of the norm group `g` of `y`.
    fn group(y: &[f32], g: usize) -> &[f32] {
        &y[g * WIDTH..(g + 1) * WIDTH]
    }

    fn check_fault(cx: &Ctx) -> Result<bool, GateError> {
        let seed = 0x0fa1_7001;
        let w = weights(Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(&mut r, false);
        let tok = token(Regime::Band, &mut r);
        let ring0 = ring_of(&st, P0, false);
        let mut dev = Dev::new(cx, &w)?;
        let run = |dev: &mut Dev, t: &TokenIn| {
            dev_run(cx, dev, &ring0, &st, std::slice::from_ref(t), P0, true)
        };
        let base = run(&mut dev, &tok)?;
        let mut ok = clean(cx, "fault base")?;
        println!(
            "fault base: the clean launch raised nothing {}",
            verdict(ok)
        );
        let want = Some(Fault::at(LAYER as u32, FaultSite::Mamba2Scan));

        // (label, token, the norm group the plant sits in)
        let mut plants: Vec<(&str, TokenIn, usize)> = Vec::new();
        let c_x = 37 * HEAD_DIM + 5;
        let mut t = tok.clone();
        t.xbc[c_x] = f32::NAN;
        plants.push(("NaN in x", t, c_x / WIDTH));
        let mut t = tok.clone();
        t.xbc[INNER + 3 * STATE + 7] = f32::NAN;
        plants.push(("NaN in B", t, 3));
        let mut t = tok.clone();
        t.z[5 * WIDTH + 1] = f32::INFINITY;
        plants.push(("inf in z", t, 5));
        // −∞ dt: softplus would be 0 and the scan would stay finite.
        let mut t = tok.clone();
        t.dt[0] = f32::NEG_INFINITY;
        plants.push(("-inf dt", t, 0));
        let check = |label: &str, y: &[f32], g: usize| -> Result<bool, GateError> {
            let f = cx.gpu.take_fault()?;
            let all_bad = group(y, g).iter().all(|v| !v.is_finite());
            let others = (0..GROUPS)
                .filter(|&o| o != g)
                .all(|o| bits_equal(group(y, o), group(&base.y[0], o)));
            let pass = f == want && all_bad && others;
            let shown = f.map_or("none".to_owned(), |f| f.to_string());
            println!(
                "fault {label}: raised {shown}, group {g} non-finite {all_bad}, other groups \
                 unchanged {others} {}",
                verdict(pass)
            );
            Ok(pass)
        };
        for (label, t, g) in plants {
            let got = run(&mut dev, &t)?;
            ok &= check(label, &got.y[0], g)?;
        }
        // −∞ A on head 0, the weight buffer. Restored after the plant.
        let mut a = w.a.clone();
        a[0] = f32::NEG_INFINITY;
        dev.a.copy_from_host(cx.gpu.stream(), &a)?;
        let got = run(&mut dev, &tok)?;
        dev.a.copy_from_host(cx.gpu.stream(), &w.a)?;
        ok &= check("-inf A", &got.y[0], 0)?;
        Ok(ok)
    }

    // ---------------------------------------------------------------- clause 5

    fn check_replay(cx: &Ctx) -> Result<bool, GateError> {
        let seed = 0x0e91_a701;
        let w = weights(Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(&mut r, false);
        let toks: Vec<TokenIn> = (0..3).map(|_| token(Regime::Band, &mut r)).collect();
        let ring0 = ring_of(&st, P0, false);
        let mut dev = Dev::new(cx, &w)?;
        let eager = dev_run(cx, &mut dev, &ring0, &st, &toks, P0, true)?;

        dev.load(cx, &ring0, &st.scan)?;
        let shape = device_shape()?;
        let graph = {
            let dev = &mut dev;
            cx.gpu.capture(|s| dev.step(cx, s, shape, true))?
        };
        let mut ys = Vec::new();
        for (t, tok) in toks.iter().enumerate() {
            dev.put(cx, tok, P0 + t as u32)?;
            graph.launch(cx.gpu.stream())?;
            ys.push(dev.read_y(cx)?);
        }
        let ring = dev.ring.to_host_vec(cx.gpu.stream())?;
        let scan = dev.state.to_host_vec(cx.gpu.stream())?;
        let same_y = ys
            .iter()
            .zip(&eager.y)
            .filter(|(a, b)| bits_equal(a, b))
            .count();
        let same_ring = bits_equal(&ring, &eager.ring);
        let same_state = bits_equal(&scan, &eager.scan);
        let pass = graph.node_count() == 1
            && same_y == toks.len()
            && same_ring
            && same_state
            && clean(cx, "replay")?;
        println!(
            "replay: nodes={} y bit-identical to eager {same_y}/{} ring {same_ring} state \
             {same_state} {}",
            graph.node_count(),
            toks.len(),
            verdict(pass)
        );
        Ok(pass)
    }

    pub fn run() -> Result<(), GateError> {
        let gpu = Gpu::new()?;
        let k = Mamba2Kernels::load(gpu.context())?;
        let cx = Ctx { gpu, k };
        if let Some(f) = cx.gpu.take_fault()? {
            return Err(format!("gate_mamba2: the fault word held {f} before any launch").into());
        }
        let mut ok = true;
        ok &= check_exact(&cx)?;
        ok &= check_band(&cx)?;
        ok &= check_refuse(&cx)?;
        ok &= check_fault(&cx)?;
        ok &= check_replay(&cx)?;
        ok &= no_local_depot(&["mamba2_step"])?;
        if !ok {
            return Err(bloomery_gpu_gates::checks_failed());
        }
        println!(
            "PASSED: gate_mamba2 the fused Mamba-2 decode step equals the host twin bit for bit \
             where the device library is not asked for a value libm gives differently (four \
             tokens, with and without the gain, and from the sequence start; the ring exact) and \
             within the derived band where it is; state 96 and 256, a conv of 3 taps, a group \
             count that does not divide, a short buffer and a non-finite norm eps refused by name; \
             a NaN in x or B, an infinity in z, a non-finite dt and a non-finite A raise \
             mamba2_scan and leave their norm group non-finite and the other groups the clean \
             bits; a captured graph is one node and replays three tokens bit for bit as eager; \
             no local depot"
        );
        Ok(())
    }
}
