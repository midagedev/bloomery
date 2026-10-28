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
//!    norm epsilon are refused by name at enqueue, and so are a pass of 0
//!    tokens, its `xbc` window past the row stride, rows shorter than `U`
//!    strides and a `y` shorter than `U` rows. Nothing launched (the
//!    outputs keep their poison, the fault word stays clean).
//! 4. fault: a NaN in `x`, in `B`, an infinity in `z`, a non-finite `dt` and
//!    a non-finite `A` each raise `Mamba2Scan` at the launch's layer and
//!    leave the norm group they sit in entirely non-finite, never a
//!    plausible `y`; every other group is the clean launch's bits. Clean
//!    launches raise nothing. The prompt pass with a `−∞` dt in its middle
//!    token raises it too, with and without the norm: its first row is
//!    finite and the middle row's plant (the norm group, or head 0's
//!    channels) is non-finite.
//! 5. replay: the capture is one node. Replayed over three tokens whose
//!    inputs and position word are rewritten between replays, it equals the
//!    eager run bit for bit (`y` of every token, the ring, the state).
//! 6. steps_are_steps: the prompt pass (`enqueue_steps`, four entries: the
//!    conv pre-pass into the scratch, the scan, the ring tail, the norm)
//!    over `U` rows of an in-projection output (the `z`, `xbc`, `dt` columns
//!    at the Nemotron offsets) against `U` `enqueue_step` launches from the
//!    same ring, state and position: every `y` row, the final state and the
//!    whole ring bit for bit. `U` in {1, 2, 3, 4, 5, T−1, T, T+1, 2T+3, 37,
//!    256} (`T` the scan's stage) from a ring
//!    and state that steps warmed, with the gain and without; `U` = 5 from
//!    the sequence start; a norm tie: inputs built so a group's mean square
//!    is an f32 rounding tie that only the step's chunk order resolves the
//!    step's way (the f64 sums absorb every other order difference, so a
//!    random input cannot tell the order); and a captured pass (four
//!    nodes with a gain) replayed against the eager pass. At head_dim 72
//!    (576 as 8 heads of one group) — a head whose channels are not a
//!    multiple of the staging block, so every masked-channel rule is live —
//!    the steps and a pass past two stage boundaries each equal the host
//!    twin bit for bit, a defect one kernel alone carries unable to hide
//!    behind a matching one in the other. The pass's
//!    refusals (`u` 0, a column window past `ld`, short rows, a short `y`, a
//!    `u` past the scratch's `max_u`, a scratch of another shape) are
//!    clause 3's.
//! 7. shape: the entries compile with no local depot but the pass's
//!    transpose-reduce — `mamba2_steps` alone, held to its pinned columns
//!    (PIN(2026-10-17): `ld_local` 1, `st_local` 8; the decode step keeps
//!    its per-channel butterfly and 0).
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
    use bloomery_gpu::mamba2::{
        Mamba2Kernels, Mamba2PassScratch, Mamba2Scratch, STAGE_T, STATE, StepsArgs,
    };
    use bloomery_gpu::{Fault, FaultSite, Gpu, GpuError};
    use bloomery_gpu_gates::rounding::U;
    use bloomery_gpu_gates::{GateError, bits_equal, no_local_depot, ptx_shapes, verdict};
    use cuda_core::{CudaStream, DeviceBuffer};
    use model::mamba2::{self, Checked, Geom, Token};

    const GEOM: Geom = Geom {
        d_inner: 7680,
        n_heads: 96,
        n_groups: 8,
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
    /// One geometry the cases run at: the model [`Geom`] with the widths the
    /// gate derives from it, and a pass row's column layout — the Nemotron
    /// in-projection's `z`, `xbc` and `dt` columns, in that order (`z` at
    /// column 0), and the row's stride.
    #[derive(Clone, Copy)]
    struct Shape {
        g: Geom,
        heads: usize,
        groups: usize,
        inner: usize,
        hd: usize,
        per: usize,
        width: usize,
        conv_dim: usize,
        x0: usize,
        dt0: usize,
        ld: usize,
    }

    impl Shape {
        const fn new(g: Geom) -> Shape {
            let (inner, heads, groups) = (g.d_inner, g.n_heads, g.n_groups);
            let conv_dim = inner + 2 * groups * STATE;
            Shape {
                g,
                heads,
                groups,
                inner,
                hd: inner / heads,
                per: heads / groups,
                width: inner / groups,
                conv_dim,
                x0: inner,
                dt0: inner + conv_dim,
                ld: inner + conv_dim + heads,
            }
        }
    }

    const MAIN: Shape = Shape::new(GEOM);
    /// head_dim 72: a head whose channels are not a multiple of the staging
    /// block, so the pass's staged `z` and `x` hold masked channels the
    /// staging zeroes (clause 6).
    const HD72: Shape = Shape::new(Geom {
        d_inner: 576,
        n_heads: 8,
        n_groups: 1,
        d_state: STATE,
        d_conv: 4,
    });
    /// The pass lengths of clause 6, and the steps that warm its start: the
    /// stage boundary's neighbours `STAGE_T ± 1`, `2·STAGE_T + 3` past two
    /// stages, and the odd ones around them.
    const STEPS_US: [usize; 11] = [
        1,
        2,
        3,
        4,
        5,
        STAGE_T - 1,
        STAGE_T,
        STAGE_T + 1,
        2 * STAGE_T + 3,
        37,
        256,
    ];
    const WARM: usize = 5;

    /// One ulp at most, relative: 2^-23 (module doc).
    const EPS_F: f64 = 2.0 * U;
    /// The device library's `expf` and `log1pf` ulps, and the host's libm's.
    const DEV_EXP: f64 = 2.0;
    const DEV_LOG1P: f64 = 1.0;
    const HOST_EXP: f64 = 1.0;
    const HOST_LOG1P: f64 = 2.0;
    /// The dropped second-order terms (module doc).
    const SECOND_ORDER: f64 = 1.0 + 1e-4;

    use bloomery_gpu_gates::GateRng as Rng;

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

    fn weights(s: &Shape, regime: Regime, seed: u64) -> Weights {
        let mut r = Rng(seed);
        let mut dt_bias = vec![0.0; s.heads];
        let mut a = vec![0.0; s.heads];
        for h in 0..s.heads {
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
            d: (0..s.heads).map(|_| r.uni(0.5, 1.5)).collect(),
            conv_w: (0..s.g.d_conv * s.conv_dim)
                .map(|_| r.uni(-0.6, 0.6))
                .collect(),
            conv_b: (0..s.conv_dim).map(|_| r.uni(-0.2, 0.2)).collect(),
            norm: (0..s.inner).map(|_| r.uni(0.5, 1.5)).collect(),
        }
    }

    /// One token's inputs.
    #[derive(Clone)]
    struct TokenIn {
        xbc: Vec<f32>,
        dt: Vec<f32>,
        z: Vec<f32>,
    }

    fn token(s: &Shape, regime: Regime, r: &mut Rng) -> TokenIn {
        let dt_range = match regime {
            Regime::Exact => (-1.0, 1.0),
            Regime::Band => (-2.0, 2.0),
        };
        TokenIn {
            xbc: (0..s.conv_dim).map(|_| r.uni(-2.0, 2.0)).collect(),
            dt: (0..s.heads)
                .map(|_| r.uni(dt_range.0, dt_range.1))
                .collect(),
            z: (0..s.inner).map(|_| r.uni(-3.0, 3.0)).collect(),
        }
    }

    /// The state a run starts from: the host's conv window (`age + 3·channel`,
    /// age 0 the oldest) and the scan state.
    #[derive(Clone)]
    struct Init {
        conv: Vec<f32>,
        scan: Vec<f32>,
    }

    fn init(s: &Shape, r: &mut Rng, start: bool) -> Init {
        Init {
            conv: if start {
                vec![0.0; 3 * s.conv_dim]
            } else {
                (0..3 * s.conv_dim).map(|_| r.uni(-2.0, 2.0)).collect()
            },
            scan: (0..s.inner * STATE).map(|_| r.uni(-0.5, 0.5)).collect(),
        }
    }

    /// The ring a run starts from: garbage everywhere, then the three
    /// predecessors of position `p0` where the conv reads them.
    fn ring_of(s: &Shape, init: &Init, p0: u32, start: bool) -> Vec<f32> {
        let mut ring = vec![GARBAGE; RING_ROWS * s.conv_dim];
        if !start {
            for age in 0..3 {
                let slot = (p0 as usize + RING_ROWS - 3 + age) % RING_ROWS;
                for c in 0..s.conv_dim {
                    ring[slot * s.conv_dim + c] = init.conv[age + 3 * c];
                }
            }
        }
        ring
    }

    /// The card's checked widths. The host twin uses the same geometry.
    fn device_shape(s: &Shape) -> Result<Checked, GateError> {
        Ok(bloomery_gpu::mamba2::check("gate_mamba2", s.g)?)
    }

    fn geom(s: &Shape) -> Result<Checked, GateError> {
        Ok(mamba2::check(s.g)?)
    }

    /// The host twin's run of `toks` from `init`: each token's `y`, the
    /// final window and scan state.
    struct HostRun {
        y: Vec<Vec<f32>>,
        conv: Vec<f32>,
        scan: Vec<f32>,
    }

    fn host_run(
        s: &Shape,
        w: &Weights,
        init: &Init,
        toks: &[TokenIn],
        norm: bool,
    ) -> Result<HostRun, GateError> {
        let g = geom(s)?;
        let (mut conv, mut scan) = (init.conv.clone(), init.scan.clone());
        let mut ys = Vec::new();
        for t in toks {
            let mut y = vec![0.0; s.inner];
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
    fn ring_after(s: &Shape, start_ring: &[f32], toks: &[TokenIn], p0: u32) -> Vec<f32> {
        let mut ring = start_ring.to_vec();
        for (t, tok) in toks.iter().enumerate() {
            let slot = (p0 as usize + t) % RING_ROWS;
            ring[slot * s.conv_dim..(slot + 1) * s.conv_dim].copy_from_slice(&tok.xbc);
        }
        ring
    }

    struct Ctx {
        gpu: Gpu,
        k: Mamba2Kernels,
    }

    /// The card's buffers of one run.
    struct Dev {
        s: Shape,
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
        pass_scratch: Mamba2PassScratch,
    }

    impl Dev {
        fn new(cx: &Ctx, s: &Shape, w: &Weights) -> Result<Dev, GateError> {
            let st = cx.gpu.stream();
            Ok(Dev {
                s: *s,
                xbc: DeviceBuffer::from_host(st, &vec![0.0; s.conv_dim])?,
                dt: DeviceBuffer::from_host(st, &vec![0.0; s.heads])?,
                z: DeviceBuffer::from_host(st, &vec![0.0; s.inner])?,
                dt_bias: DeviceBuffer::from_host(st, &w.dt_bias)?,
                a: DeviceBuffer::from_host(st, &w.a)?,
                d: DeviceBuffer::from_host(st, &w.d)?,
                conv_w: DeviceBuffer::from_host(st, &w.conv_w)?,
                conv_b: DeviceBuffer::from_host(st, &w.conv_b)?,
                norm: DeviceBuffer::from_host(st, &w.norm)?,
                pos: DeviceBuffer::from_host(st, &[0u32])?,
                y: DeviceBuffer::from_host(st, &vec![f32::from_bits(POISON); s.inner])?,
                ring: DeviceBuffer::from_host(st, &vec![0.0; RING_ROWS * s.conv_dim])?,
                state: DeviceBuffer::from_host(st, &vec![0.0; s.inner * STATE])?,
                scratch: Mamba2Scratch::new(st, device_shape(s)?)?,
                pass_scratch: Mamba2PassScratch::new(
                    st,
                    device_shape(s)?,
                    STEPS_US.into_iter().max().unwrap(),
                )?,
            })
        }

        /// The run's starting state, and the output poisoned.
        fn load(&mut self, cx: &Ctx, ring: &[f32], scan: &[f32]) -> Result<(), GateError> {
            let st = cx.gpu.stream();
            self.ring.copy_from_host(st, ring)?;
            self.state.copy_from_host(st, scan)?;
            self.y
                .copy_from_host(st, &vec![f32::from_bits(POISON); self.s.inner])?;
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

        /// The launch, on `stream` (the engine's, or a capture's). `eps` is
        /// the norm's epsilon.
        fn step(
            &mut self,
            cx: &Ctx,
            stream: &CudaStream,
            shape: Checked,
            norm: bool,
            eps: f32,
        ) -> Result<(), GpuError> {
            cx.k.enqueue_step(
                stream,
                bloomery_gpu::mamba2::StepArgs {
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
                    eps,
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

        /// The pass of `pass.u` rows on `stream`, from the position word,
        /// into `y`.
        fn steps(
            &mut self,
            cx: &Ctx,
            stream: &CudaStream,
            pass: Pass<'_>,
            y: &mut DeviceBuffer<f32>,
            norm: bool,
        ) -> Result<(), GpuError> {
            cx.k.enqueue_steps(
                stream,
                StepsArgs {
                    shape: pass.shape,
                    u: pass.u,
                    rows: pass.rows,
                    ld: pass.ld,
                    z0: 0,
                    x0: pass.x0,
                    dt0: pass.dt0,
                    dt_bias: &self.dt_bias,
                    a: &self.a,
                    d: &self.d,
                    conv_w: &self.conv_w,
                    conv_b: &self.conv_b,
                    norm: norm.then_some(&self.norm),
                    eps: EPS,
                    pos: &self.pos,
                    pos_at: 0,
                    fault: cx.gpu.layer_sink(LAYER)?,
                    y,
                    ring: &mut self.ring,
                    state: &mut self.state,
                    scratch: &mut self.pass_scratch,
                },
            )
        }

        fn read_y(&self, cx: &Ctx) -> Result<Vec<f32>, GateError> {
            Ok(self.y.to_host_vec(cx.gpu.stream())?)
        }
    }

    /// A pass's input rows and their layout.
    #[derive(Clone, Copy)]
    struct Pass<'b> {
        shape: Checked,
        u: usize,
        ld: usize,
        x0: usize,
        dt0: usize,
        rows: &'b DeviceBuffer<f32>,
    }

    /// The card's run of `toks` step by step from `ring` and `scan` at
    /// positions `p0 ..`: each token's `y`, the final ring and scan state.
    struct DevRun {
        y: Vec<Vec<f32>>,
        ring: Vec<f32>,
        scan: Vec<f32>,
    }

    fn dev_run(
        cx: &Ctx,
        dev: &mut Dev,
        ring: &[f32],
        scan: &[f32],
        toks: &[TokenIn],
        p0: u32,
        norm: bool,
    ) -> Result<DevRun, GateError> {
        dev.load(cx, ring, scan)?;
        let shape = device_shape(&dev.s)?;
        let mut ys = Vec::new();
        for (t, tok) in toks.iter().enumerate() {
            dev.put(cx, tok, p0 + t as u32)?;
            dev.step(cx, cx.gpu.stream(), shape, norm, EPS)?;
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
        let s = &MAIN;
        let w = weights(s, Regime::Exact, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, start);
        let toks: Vec<TokenIn> = (0..TOKENS)
            .map(|_| token(s, Regime::Exact, &mut r))
            .collect();
        let p0 = if start { 0 } else { P0 };
        let ring0 = ring_of(s, &st, p0, start);
        let want = host_run(s, &w, &st, &toks, norm)?;
        let mut dev = Dev::new(cx, s, &w)?;
        let got = dev_run(cx, &mut dev, &ring0, &st.scan, &toks, p0, norm)?;
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
        let ring_want = ring_after(s, &ring0, &toks, p0);
        let ring_ok = bits_equal(&got.ring, &ring_want);
        // The host's final window is the last three inputs: the ring's
        // slots of the last three positions.
        let mut window_ok = true;
        for age in 0..3 {
            let slot = (p0 as usize + TOKENS - 3 + age) % RING_ROWS;
            for c in 0..s.conv_dim {
                window_ok &=
                    got.ring[slot * s.conv_dim + c].to_bits() == want.conv[age + 3 * c].to_bits();
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
        s: &Shape,
        w: &Weights,
        init: &Init,
        toks: &[TokenIn],
    ) -> Result<Vec<Trace>, GateError> {
        let g = geom(s)?;
        let (mut conv, mut scan) = (init.conv.clone(), init.scan.clone());
        let mut out = Vec::new();
        for t in toks {
            let mut conv_y = vec![0.0; s.conv_dim];
            mamba2::conv_tokens(4, &mut conv, &t.xbc, &w.conv_w, &w.conv_b, &mut conv_y)?;
            let ones = vec![1.0; s.conv_dim];
            let mut mixed = vec![0.0; s.conv_dim];
            qdot::swiglu(&conv_y, &ones, &mut mixed);
            let x = mixed[..s.inner].to_vec();
            let bc = mixed[s.inner..].to_vec();
            let dt: Vec<f32> = t.dt.iter().zip(&w.dt_bias).map(|(a, b)| a + b).collect();
            let sigma: Vec<f32> = dt
                .iter()
                .map(|&v| if v <= 20.0 { v.exp().ln_1p() } else { v })
                .collect();
            let decay: Vec<f32> = sigma.iter().zip(&w.a).map(|(s, a)| (s * a).exp()).collect();
            let s_before = scan.clone();
            let mut y = vec![0.0; s.inner];
            mamba2::scan_token(
                &g,
                &mut scan,
                &x,
                &dt,
                &w.a,
                &bc[..s.groups * STATE],
                &bc[s.groups * STATE..],
                &mut y,
            )?;
            mamba2::skip_d(&mut y, &x, &w.d, s.hd)?;
            let y1 = y.clone();
            mamba2::gate_z(&mut y, &t.z)?;
            let y2 = y.clone();
            let mut sz = vec![0.0; s.inner];
            qdot::swiglu(&t.z, &vec![1.0; s.inner], &mut sz);
            mamba2::grouped_rms(&mut y, &w.norm, s.groups, EPS)?;
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
    fn bands(s: &Shape, w: &Weights, trace: &[Trace]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let mut e = vec![0.0f64; s.inner * STATE];
        let mut per_token = Vec::new();
        for tr in trace {
            let mut dy = vec![0.0f64; s.inner];
            for h in 0..s.heads {
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
                let g = h / s.per;
                let bn = &tr.bc[g * STATE..(g + 1) * STATE];
                let cn = &tr.bc[s.groups * STATE + g * STATE..s.groups * STATE + (g + 1) * STATE];
                for ch in 0..s.hd {
                    let i = h * s.hd + ch;
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
            let mut band = vec![0.0f64; s.inner];
            let mut d2 = vec![0.0f64; s.inner];
            for i in 0..s.inner {
                let d1 = dy[i] + EPS_F * f64::from(tr.y1[i]).abs();
                d2[i] = f64::from(tr.sz[i]).abs() * d1 + EPS_F * f64::from(tr.y2[i]).abs();
            }
            for g in 0..s.groups {
                let r = g * s.width..(g + 1) * s.width;
                let m = r
                    .clone()
                    .map(|i| f64::from(tr.y2[i] * tr.y2[i]))
                    .sum::<f64>()
                    / s.width as f64;
                let cross = r
                    .clone()
                    .map(|i| f64::from(tr.y2[i]).abs() * d2[i])
                    .sum::<f64>()
                    / s.width as f64;
                let sq = r.clone().map(|i| d2[i] * d2[i]).sum::<f64>() / s.width as f64;
                let dm = 2.0 * cross + sq + 2.0 * EPS_F * m;
                let scale = 1.0 / (m + f64::from(EPS)).sqrt();
                let ds = dm / (2.0 * (m + f64::from(EPS))) + 3.0 * EPS_F;
                for i in r {
                    let o = f64::from(tr.out[i]).abs();
                    band[i] = o * (ds + 2.0 * EPS_F) + scale * f64::from(w.norm[i]).abs() * d2[i];
                }
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
        let s = &MAIN;
        let seed = 0x0ba2_d001;
        let w = weights(s, Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, false);
        let toks: Vec<TokenIn> = (0..TOKENS)
            .map(|_| token(s, Regime::Band, &mut r))
            .collect();
        let ring0 = ring_of(s, &st, P0, false);
        let want = host_run(s, &w, &st, &toks, true)?;
        let trace = trace_run(s, &w, &st, &toks)?;
        // The referee's composition is step_token.
        let mut referee = true;
        for (tr, y) in trace.iter().zip(&want.y) {
            referee &= bits_equal(&tr.out, y);
        }
        referee &= bits_equal(&trace[TOKENS - 1].s_after, &want.scan);
        let (y_bands, s_band) = bands(s, &w, &trace);
        let mut dev = Dev::new(cx, s, &w)?;
        let got = dev_run(cx, &mut dev, &ring0, &st.scan, &toks, P0, true)?;
        let mut ok = referee;
        let mut y_bad = 0;
        let mut y_worst = 0.0f64;
        for (t, band) in y_bands.iter().enumerate().take(TOKENS) {
            let (bad, worst) = within(&got.y[t], &want.y[t], band);
            y_bad += bad;
            y_worst = y_worst.max(worst);
        }
        let (s_bad, s_worst) = within(&got.scan, &want.scan, &s_band);
        let ring_ok = bits_equal(&got.ring, &ring_after(s, &ring0, &toks, P0));
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
        let s = &MAIN;
        let w = weights(s, Regime::Band, 0x0ef0_0001);
        let mut dev = Dev::new(cx, s, &w)?;
        let mut r = Rng(0x0ef0_0002);
        let st = init(s, &mut r, false);
        dev.load(cx, &ring_of(s, &st, P0, false), &st.scan)?;
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
                let e = dev.step(cx, cx.gpu.stream(), host, true, EPS).err();
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
                DeviceBuffer::from_host(cx.gpu.stream(), &vec![0.0f32; s.inner * STATE / 2])?;
            // The short buffer is the state for this call only.
            std::mem::swap(&mut dev.state, &mut half);
            let e = dev
                .step(cx, cx.gpu.stream(), device_shape(s)?, true, EPS)
                .err();
            std::mem::swap(&mut dev.state, &mut half);
            matches!(&e, Some(GpuError::Shape { detail, .. }) if detail.contains("state.len()"))
        };
        named.push(format!("short state={}", verdict(short)));
        ok &= short;
        // +∞ epsilon would make the norm scale 0. Refused before a launch.
        let eps_inf = {
            let e = dev
                .step(cx, cx.gpu.stream(), device_shape(s)?, true, f32::INFINITY)
                .err();
            let text = e.as_ref().map(ToString::to_string).unwrap_or_default();
            matches!(e, Some(GpuError::Shape { .. })) && text.contains("norm eps must be finite")
        };
        named.push(format!("eps inf={}", verdict(eps_inf)));
        ok &= eps_inf;
        // The pass's own refusals: (label, u, x0, y rows, the text).
        let st = cx.gpu.stream();
        let rows = DeviceBuffer::from_host(st, &vec![0.0f32; 2 * s.ld])?;
        let mut ys = [
            DeviceBuffer::from_host(st, &vec![f32::from_bits(POISON); s.inner])?,
            DeviceBuffer::from_host(st, &vec![f32::from_bits(POISON); 3 * s.inner])?,
        ];
        let pass_cases = [
            ("pass u 0", 0, s.x0, 1, "u 0 is refused"),
            (
                "pass xbc window",
                2,
                s.ld - s.conv_dim + 1,
                1,
                "the xbc window",
            ),
            ("pass rows short", 3, s.x0, 1, "rows.len()"),
            ("pass y short", 2, s.x0, 0, "y.len()"),
        ];
        for (label, u, x0, yi, want) in pass_cases {
            let pass = Pass {
                shape: device_shape(s)?,
                u,
                ld: s.ld,
                x0,
                dt0: s.dt0,
                rows: &rows,
            };
            let e = dev.steps(cx, st, pass, &mut ys[yi], true).err();
            let text = e.as_ref().map(ToString::to_string).unwrap_or_default();
            let pass = matches!(e, Some(GpuError::Shape { .. })) && text.contains(want);
            named.push(format!("{label}={}", verdict(pass)));
            ok &= pass;
        }
        // The pass scratch's own refusals: a `u` past its `max_u` (read
        // before the buffer lengths, so a short `rows` cannot mask it) and a
        // scratch of another shape.
        for (label, u, want) in [
            ("pass u past max_u", 257, "is past the pass scratch's max_u"),
            (
                "pass scratch other shape",
                2,
                "the pass scratch holds conv_dim",
            ),
        ] {
            let pass = Pass {
                shape: device_shape(s)?,
                u,
                ld: s.ld,
                x0: s.x0,
                dt0: s.dt0,
                rows: &rows,
            };
            let scratch = if u == 257 {
                &mut dev.pass_scratch
            } else {
                let other = Geom {
                    d_inner: (s.heads + s.groups) * s.hd,
                    n_heads: s.heads + s.groups,
                    n_groups: s.groups,
                    d_state: STATE,
                    d_conv: 4,
                };
                &mut Mamba2PassScratch::new(
                    st,
                    bloomery_gpu::mamba2::check("gate_mamba2 other", other)?,
                    256,
                )?
            };
            let e =
                cx.k.enqueue_steps(
                    st,
                    StepsArgs {
                        shape: pass.shape,
                        u,
                        rows: pass.rows,
                        ld: s.ld,
                        z0: 0,
                        x0: s.x0,
                        dt0: s.dt0,
                        dt_bias: &dev.dt_bias,
                        a: &dev.a,
                        d: &dev.d,
                        conv_w: &dev.conv_w,
                        conv_b: &dev.conv_b,
                        norm: None,
                        eps: EPS,
                        pos: &dev.pos,
                        pos_at: 0,
                        fault: cx.gpu.layer_sink(LAYER)?,
                        y: &mut ys[0],
                        ring: &mut dev.ring,
                        state: &mut dev.state,
                        scratch,
                    },
                )
                .err();
            let text = e.as_ref().map(ToString::to_string).unwrap_or_default();
            let hit = matches!(e, Some(GpuError::Shape { .. })) && text.contains(want);
            named.push(format!("{label}={}", verdict(hit)));
            ok &= hit;
        }
        let mut untouched = true;
        for y in &ys {
            untouched &= y.to_host_vec(st)?.iter().all(|v| v.to_bits() == POISON);
        }
        let y = dev.read_y(cx)?;
        untouched &= y.iter().all(|v| v.to_bits() == POISON);
        ok &= untouched && clean(cx, "refuse")?;
        println!(
            "refuse: {}; nothing launched (the outputs keep their poison) {untouched} {}",
            named.join(" "),
            verdict(ok)
        );
        Ok(ok)
    }

    // ---------------------------------------------------------------- clause 4

    /// Every value of the norm group `g` of `y`.
    fn group<'a>(s: &Shape, y: &'a [f32], g: usize) -> &'a [f32] {
        &y[g * s.width..(g + 1) * s.width]
    }

    fn check_fault(cx: &Ctx) -> Result<bool, GateError> {
        let s = &MAIN;
        let seed = 0x0fa1_7001;
        let w = weights(s, Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, false);
        let tok = token(s, Regime::Band, &mut r);
        let ring0 = ring_of(s, &st, P0, false);
        let mut dev = Dev::new(cx, s, &w)?;
        let run = |dev: &mut Dev, t: &TokenIn| {
            dev_run(cx, dev, &ring0, &st.scan, std::slice::from_ref(t), P0, true)
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
        let c_x = 37 * s.hd + 5;
        let mut t = tok.clone();
        t.xbc[c_x] = f32::NAN;
        plants.push(("NaN in x", t, c_x / s.width));
        let mut t = tok.clone();
        t.xbc[s.inner + 3 * STATE + 7] = f32::NAN;
        plants.push(("NaN in B", t, 3));
        let mut t = tok.clone();
        t.z[5 * s.width + 1] = f32::INFINITY;
        plants.push(("inf in z", t, 5));
        // −∞ dt: softplus would be 0 and the scan would stay finite.
        let mut t = tok.clone();
        t.dt[0] = f32::NEG_INFINITY;
        plants.push(("-inf dt", t, 0));
        let check = |label: &str, y: &[f32], g: usize| -> Result<bool, GateError> {
            let f = cx.gpu.take_fault()?;
            let all_bad = group(s, y, g).iter().all(|v| !v.is_finite());
            let others = (0..s.groups)
                .filter(|&o| o != g)
                .all(|o| bits_equal(group(s, y, o), group(s, &base.y[0], o)));
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
        // The prompt pass with a −∞ dt in its middle token, with and without
        // the norm: the named fault, a finite first row, and the middle row's
        // plant non-finite (the norm group of head 0, or head 0's channels).
        let mut bad = tok.clone();
        bad.dt[0] = f32::NEG_INFINITY;
        let toks = [tok.clone(), bad, tok];
        for norm in [true, false] {
            let (got, _) = pass_run(cx, &mut dev, (&ring0, &st.scan, P0), &toks, norm, false)?;
            let f = cx.gpu.take_fault()?;
            let first_finite = got.y[0].iter().all(|v| v.is_finite());
            let plant = if norm {
                group(s, &got.y[1], 0)
            } else {
                &got.y[1][..s.hd]
            };
            let plant_bad = plant.iter().all(|v| !v.is_finite());
            let pass = f == want && first_finite && plant_bad;
            let shown = f.map_or("none".to_owned(), |f| f.to_string());
            println!(
                "fault -inf dt in a pass of 3 (norm {norm}): raised {shown}, row 0 finite \
                 {first_finite}, row 1 plant non-finite {plant_bad} {}",
                verdict(pass)
            );
            ok &= pass;
        }
        Ok(ok)
    }

    // ---------------------------------------------------------------- clause 5

    fn check_replay(cx: &Ctx) -> Result<bool, GateError> {
        let s = &MAIN;
        let seed = 0x0e91_a701;
        let w = weights(s, Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, false);
        let toks: Vec<TokenIn> = (0..3).map(|_| token(s, Regime::Band, &mut r)).collect();
        let ring0 = ring_of(s, &st, P0, false);
        let mut dev = Dev::new(cx, s, &w)?;
        let eager = dev_run(cx, &mut dev, &ring0, &st.scan, &toks, P0, true)?;

        dev.load(cx, &ring0, &st.scan)?;
        let shape = device_shape(s)?;
        let graph = {
            let dev = &mut dev;
            cx.gpu.capture(|s| dev.step(cx, s, shape, true, EPS))?
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

    // ---------------------------------------------------------------- clause 6

    /// The pass's input rows of `toks`: each token's `z`, `xbc` and `dt` at
    /// their columns of a row of `s.ld`.
    fn rows_of(s: &Shape, toks: &[TokenIn]) -> Vec<f32> {
        let mut rows = vec![0.0; toks.len() * s.ld];
        for (row, tok) in rows.chunks_mut(s.ld).zip(toks) {
            row[..s.inner].copy_from_slice(&tok.z);
            row[s.x0..s.x0 + s.conv_dim].copy_from_slice(&tok.xbc);
            row[s.dt0..s.dt0 + s.heads].copy_from_slice(&tok.dt);
        }
        rows
    }

    /// The card's pass of `toks` from `ring` and `scan` at position `p`,
    /// eager or captured and replayed once: each token's `y` row, the final
    /// ring and state, and the captured graph's node count.
    fn pass_run(
        cx: &Ctx,
        dev: &mut Dev,
        start: (&[f32], &[f32], u32),
        toks: &[TokenIn],
        norm: bool,
        capture: bool,
    ) -> Result<(DevRun, Option<usize>), GateError> {
        let (ring, scan, p) = start;
        let st = cx.gpu.stream();
        dev.load(cx, ring, scan)?;
        dev.pos.copy_from_host(st, &[p])?;
        let rows = DeviceBuffer::from_host(st, &rows_of(&dev.s, toks))?;
        let mut y =
            DeviceBuffer::from_host(st, &vec![f32::from_bits(POISON); toks.len() * dev.s.inner])?;
        let pass = Pass {
            shape: device_shape(&dev.s)?,
            u: toks.len(),
            ld: dev.s.ld,
            x0: dev.s.x0,
            dt0: dev.s.dt0,
            rows: &rows,
        };
        // The graph lives until the reads below are done.
        let graph = if capture {
            let (dev, y) = (&mut *dev, &mut y);
            let g = cx.gpu.capture(|s| dev.steps(cx, s, pass, y, norm))?;
            g.launch(st)?;
            Some(g)
        } else {
            dev.steps(cx, st, pass, &mut y, norm)?;
            None
        };
        let run = DevRun {
            y: y.to_host_vec(st)?
                .chunks(dev.s.inner)
                .map(<[f32]>::to_vec)
                .collect(),
            ring: dev.ring.to_host_vec(st)?,
            scan: dev.state.to_host_vec(st)?,
        };
        Ok((run, graph.map(|g| g.node_count())))
    }

    /// `got` against `want` bit for bit: the rows that differ (and the
    /// first value), the state and ring values that differ.
    fn same_run(label: &str, got: &DevRun, want: &DevRun) -> bool {
        let mut rows_same = 0;
        let mut shown = false;
        for (t, (a, b)) in got.y.iter().zip(&want.y).enumerate() {
            let (n, first) = diff(a, b);
            if n == 0 {
                rows_same += 1;
            } else if !shown {
                shown = true;
                println!("steps {label}: row {t} first difference {first:?} ({n} values)");
            }
        }
        let (ns, s_first) = diff(&got.scan, &want.scan);
        let (nr, r_first) = diff(&got.ring, &want.ring);
        if ns != 0 {
            println!("steps {label}: state first difference {s_first:?}");
        }
        if nr != 0 {
            println!("steps {label}: ring first difference {r_first:?}");
        }
        let ok = rows_same == want.y.len() && got.y.len() == want.y.len() && ns == 0 && nr == 0;
        println!(
            "steps {label}: y rows bit-identical {rows_same}/{}, state differing {ns}, ring \
             differing {nr}",
            want.y.len()
        );
        ok
    }

    /// The pass against as many steps from the same start (module doc,
    /// clause 6).
    fn steps_case(
        cx: &Ctx,
        label: &str,
        dev: &mut Dev,
        start: (&[f32], &[f32], u32),
        toks: &[TokenIn],
        norm: bool,
    ) -> Result<bool, GateError> {
        let (ring, scan, p) = start;
        let want = dev_run(cx, dev, ring, scan, toks, p, norm)?;
        let (got, _) = pass_run(cx, dev, start, toks, norm, false)?;
        let ok = same_run(label, &got, &want) & clean(cx, label)?;
        println!("steps {label}: {}", verdict(ok));
        Ok(ok)
    }

    /// `u` tokens from a ring and state that `WARM` steps left at `P0 +
    /// WARM`.
    fn steps_warm(cx: &Ctx, u: usize, norm: bool, seed: u64) -> Result<bool, GateError> {
        let s = &MAIN;
        let w = weights(s, Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, false);
        let warm: Vec<TokenIn> = (0..WARM).map(|_| token(s, Regime::Band, &mut r)).collect();
        let toks: Vec<TokenIn> = (0..u).map(|_| token(s, Regime::Band, &mut r)).collect();
        let mut dev = Dev::new(cx, s, &w)?;
        let warmed = dev_run(
            cx,
            &mut dev,
            &ring_of(s, &st, P0, false),
            &st.scan,
            &warm,
            P0,
            norm,
        )?;
        let p = P0 + WARM as u32;
        let label = format!("u={u} norm={norm} p={p}");
        steps_case(
            cx,
            &label,
            &mut dev,
            (&warmed.ring, &warmed.scan, p),
            &toks,
            norm,
        )
    }

    /// Five tokens from position 0 over a ring of garbage.
    fn steps_start(cx: &Ctx) -> Result<bool, GateError> {
        let s = &MAIN;
        let seed = 0x05e9_5701;
        let w = weights(s, Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, true);
        let toks: Vec<TokenIn> = (0..5).map(|_| token(s, Regime::Band, &mut r)).collect();
        let mut dev = Dev::new(cx, s, &w)?;
        let ring0 = ring_of(s, &st, 0, true);
        steps_case(
            cx,
            "sequence start u=5 p=0",
            &mut dev,
            (&ring0, &st.scan, 0),
            &toks,
            true,
        )
    }

    /// The role of head `h` in the norm tie: its head within its group.
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Role {
        /// `y` = 2^12·z: the four squares of 15·2^36.
        Big,
        /// `y` = z: the squares of 15·(2j + 1)·2^12, the tie's offset.
        Offset,
        /// `y` = 31·2^-14 on every channel: chunk sums under half an f64
        /// ulp of the group's total, which the step's order absorbs one by
        /// one and the reverse order adds up first.
        Small,
    }

    fn role(s: &Shape, h: usize) -> Role {
        match h % s.per {
            0 => Role::Big,
            1 => Role::Offset,
            _ => Role::Small,
        }
    }

    /// The step's norm of one group of host values `y`: the chunk sums in
    /// the step's order or reversed, the mean and the scale.
    fn norm_scale(y: &[f32], reverse: bool) -> (f32, f32) {
        let chunks: Vec<f64> = y
            .chunks(16)
            .map(|c| c.iter().fold(0.0f64, |a, &v| a + f64::from(v * v)))
            .collect();
        let mut total = 0.0f64;
        if reverse {
            chunks.iter().rev().for_each(|c| total += c);
        } else {
            for c in chunks.iter() {
                total += c;
            }
        }
        let mean = (total / y.len() as f64) as f32;
        (mean, 1.0 / (mean + EPS).sqrt())
    }

    /// The host's silu, bit for bit the card's (clause 1 holds it).
    fn silu(v: f32) -> f32 {
        let mut out = [0.0f32];
        qdot::swiglu(&[v], &[1.0], &mut out);
        out[0]
    }

    /// The norm tie (module doc, clause 6). The conv passes `x` through
    /// (taps 0, 0, 0, 1, no bias; `x` = 32 so SiLU is the identity) and
    /// makes `B` and `C` zero, so `y = silu(z)·fl(32·D)` whatever the scan
    /// holds. Per group, the mean square is `L + h` with `L = 2^30 + 128j`
    /// (even `j`, the first whose scale differs from `L + 128`'s) and `h`
    /// half its f32 ulp: the step's chunk order sums to exactly `960·(L +
    /// h)` (the Small chunks absorbed), which narrows to `L`; any order that
    /// adds the Small chunks first lands above the tie, at `L + 128`.
    fn steps_tie(cx: &Ctx) -> Result<bool, GateError> {
        let s = &MAIN;
        let seed = 0x05e9_7e01;
        let scale_of = |m: f32| 1.0 / (m + EPS).sqrt();
        let j = (0u32..1 << 16)
            .step_by(2)
            .find(|&j| {
                let l = (1u32 << 30) as f32 + 128.0 * j as f32;
                scale_of(l).to_bits() != scale_of(l + 128.0).to_bits()
            })
            .ok_or_else(|| "norm tie: no even j whose two scales differ".to_owned())?;
        // 15·(2j + 1) as a sum of squares, greedily.
        let mut offset = Vec::new();
        let mut rest = 15 * (2 * u64::from(j) + 1);
        while rest > 0 {
            let k = rest.isqrt();
            offset.push(k);
            rest -= k * k;
        }
        let z_head = |role: Role| -> Vec<f32> {
            let mut z = vec![0.0f32; s.hd];
            match role {
                Role::Big => z[..4].copy_from_slice(&[192.0, 128.0, 64.0, 64.0]),
                Role::Offset => {
                    for (zi, &k) in z.iter_mut().zip(&offset) {
                        *zi = 64.0 * k as f32;
                    }
                }
                Role::Small => z.iter_mut().for_each(|v| *v = 31.0),
            }
            // A channel left at z = 0 carries y = 0.
            z
        };
        let z: Vec<f32> = (0..s.heads).flat_map(|h| z_head(role(s, h))).collect();
        let d: Vec<f32> = (0..s.heads)
            .map(|h| match role(s, h) {
                Role::Big => 128.0,
                Role::Offset => 1.0 / 32.0,
                Role::Small => 1.0 / 524_288.0,
            })
            .collect();
        let mut conv_w = vec![0.0f32; s.g.d_conv * s.conv_dim];
        for c in 0..s.inner {
            conv_w[s.g.d_conv * c + 3] = 1.0;
        }
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let w = Weights {
            dt_bias: (0..s.heads).map(|_| r.uni(22.0, 26.0)).collect(),
            a: (0..s.heads)
                .map(|h| if h % 2 == 0 { 0.0 } else { -8.0 })
                .collect(),
            d,
            conv_w,
            conv_b: vec![0.0; s.conv_dim],
            norm: (0..s.inner).map(|_| r.uni(0.5, 1.5)).collect(),
        };
        let toks: Vec<TokenIn> = (0..5)
            .map(|_| {
                let mut xbc: Vec<f32> = (0..s.conv_dim).map(|_| r.uni(-2.0, 2.0)).collect();
                xbc[..s.inner].iter_mut().for_each(|v| *v = 32.0);
                TokenIn {
                    xbc,
                    dt: (0..s.heads).map(|_| r.uni(-1.0, 1.0)).collect(),
                    z: z.clone(),
                }
            })
            .collect();
        // The design on the host: y of group 0, its two orders' means.
        let x = silu(32.0);
        let y0: Vec<f32> = (0..s.width)
            .map(|i| silu(z[i]) * (x * w.d[i / s.hd]))
            .collect();
        let (m_step, s_step) = norm_scale(&y0, false);
        let (m_rev, s_rev) = norm_scale(&y0, true);
        let design = s_step.to_bits() != s_rev.to_bits();
        println!(
            "steps norm tie: j={j}, offset squares {offset:?}; the step's order mean {m_step} \
             scale {s_step:e}, the reverse mean {m_rev} scale {s_rev:e}; the orders' scales \
             differ {design}"
        );
        let st = init(s, &mut r, false);
        let mut dev = Dev::new(cx, s, &w)?;
        let ring0 = ring_of(s, &st, P0, false);
        let ok = steps_case(
            cx,
            "norm tie u=5",
            &mut dev,
            (&ring0, &st.scan, P0),
            &toks,
            true,
        )?;
        Ok(ok && design)
    }

    /// The pass captured (three nodes with the gain) and replayed once
    /// equals the eager pass bit for bit.
    fn steps_capture(cx: &Ctx) -> Result<bool, GateError> {
        let s = &MAIN;
        let seed = 0x05e9_ca01;
        let w = weights(s, Regime::Band, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, false);
        let toks: Vec<TokenIn> = (0..37).map(|_| token(s, Regime::Band, &mut r)).collect();
        let mut dev = Dev::new(cx, s, &w)?;
        let ring0 = ring_of(s, &st, P0, false);
        let start = (&ring0[..], &st.scan[..], P0);
        let (eager, _) = pass_run(cx, &mut dev, start, &toks, true, false)?;
        let (replay, nodes) = pass_run(cx, &mut dev, start, &toks, true, true)?;
        let same = same_run("captured u=37", &replay, &eager);
        // Four nodes with a gain: the conv pre-pass, the scan, the ring tail
        // and the norm rows.
        let ok = nodes == Some(4) && same && clean(cx, "steps captured")?;
        println!(
            "steps captured u=37: nodes={} replay bit-identical to eager {same} {}",
            nodes.unwrap_or(0),
            verdict(ok)
        );
        Ok(ok)
    }

    /// The head_dim 72 geometry (module doc, clause 6) against the host
    /// twin: a head whose channels are not a multiple of the staging
    /// block, so every masked-channel rule — the staged `z` and `x` slots,
    /// the reads the channel liveness guard holds in bounds — is live at
    /// last. Both the steps and a pass of `2·STAGE_T + 3` rows — past two
    /// stage boundaries — equal the host bit for bit, so a defect one
    /// kernel alone carries cannot hide behind a matching one in the other.
    fn hd72_case(cx: &Ctx) -> Result<bool, GateError> {
        let s = &HD72;
        let seed = 0x05e9_7201;
        let w = weights(s, Regime::Exact, seed);
        let mut r = Rng(seed ^ 0x9e37_79b9);
        let st = init(s, &mut r, false);
        let u = 2 * STAGE_T + 3;
        let toks: Vec<TokenIn> = (0..u).map(|_| token(s, Regime::Exact, &mut r)).collect();
        let ring0 = ring_of(s, &st, P0, false);
        let want = host_run(s, &w, &st, &toks, true)?;
        let mut dev = Dev::new(cx, s, &w)?;
        let steps = dev_run(cx, &mut dev, &ring0, &st.scan, &toks, P0, true)?;
        let (pass, _) = pass_run(
            cx,
            &mut dev,
            (&ring0[..], &st.scan[..], P0),
            &toks,
            true,
            false,
        )?;
        let ring_want = ring_after(s, &ring0, &toks, P0);
        let mut ok = true;
        for (label, got) in [("steps", &steps), ("pass", &pass)] {
            let mut rows_same = 0;
            let mut shown = 0;
            for (t, (a, b)) in got.y.iter().zip(&want.y).enumerate() {
                let (n, first) = diff(a, b);
                if n == 0 {
                    rows_same += 1;
                } else if shown < 3 {
                    shown += 1;
                    println!("hd72 {label} u={u}: row {t} first difference {first:?} ({n} values)");
                }
            }
            let (ns, s_first) = diff(&got.scan, &want.scan);
            let ring_ok = bits_equal(&got.ring, &ring_want);
            if ns != 0 {
                println!("hd72 {label} u={u}: state first difference {s_first:?}");
            }
            let hit = rows_same == want.y.len() && ns == 0 && ring_ok;
            println!(
                "hd72 {label} u={u}: y rows bit-identical {rows_same}/{}, state differing {ns}, \
                 ring exact {ring_ok} {}",
                want.y.len(),
                verdict(hit)
            );
            ok &= hit;
        }
        ok &= clean(cx, "hd72")?;
        Ok(ok)
    }

    fn check_steps(cx: &Ctx) -> Result<bool, GateError> {
        let mut ok = true;
        for norm in [true, false] {
            for (i, &u) in STEPS_US.iter().enumerate() {
                let seed = 0x05e9_0001 + i as u64 + if norm { 0x100 } else { 0 };
                ok &= steps_warm(cx, u, norm, seed)?;
            }
        }
        ok &= steps_start(cx)?;
        ok &= steps_tie(cx)?;
        ok &= steps_capture(cx)?;
        ok &= hd72_case(cx)?;
        println!("steps_are_steps: {}", verdict(ok));
        Ok(ok)
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
        ok &= check_steps(&cx)?;
        // The shape contract: no kernel holds local memory but the pass's
        // transpose-reduce — mamba2_steps alone, at exactly the pinned
        // columns (PIN(2026-10-17) in tools/ref/ptx-shapes.tsv; the decode
        // step keeps its per-channel butterfly and stays at 0).
        ok &= no_local_depot(&["mamba2_step", "mamba2_ring_tail", "mamba2_norm_rows"])?;
        ok &= ptx_shapes(&["mamba2_steps"], |c, _| {
            Ok((
                c.depot && c.ld_local == 1 && c.st_local == 8,
                format!(
                    "shape kernel={} local_depot={} ld_local={} st_local={} (PIN(2026-10-17): \
                     the pass's transpose-reduce, 1/8)",
                    c.name, c.depot, c.ld_local, c.st_local
                ),
            ))
        })?;
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
             the prompt pass over 1 to 256 rows equals as many steps bit for bit (every y row, \
             the state, the ring; with and without the gain, from the sequence start, on a norm \
             tie, captured; at head_dim 72, a head whose channels are not a multiple of the \
             staging block, the steps and a pass past two stage boundaries each the host twin \
             bit for bit); no local depot but the pass's transpose-reduce, mamba2_steps alone \
             at its pinned 1/8"
        );
        Ok(())
    }
}
